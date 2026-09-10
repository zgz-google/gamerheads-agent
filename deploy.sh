#!/bin/bash
set -e

export UV_INDEX_URL="https://pypi.org/simple"
export UV_DEFAULT_INDEX="https://pypi.org/simple"

GE_APP_ID=""
REGION=""
POSITIONAL=()

while [ $# -gt 0 ]; do
  case $1 in
    --ge=*)
      GE_APP_ID="${1#*=}"
      shift
      ;;
    --ge)
      GE_APP_ID="$2"
      shift 2
      ;;
    --region=*)
      REGION="${1#*=}"
      shift
      ;;
    --region|-r)
      REGION="$2"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

PROJECT_ID="${POSITIONAL[0]:?Usage: bash deploy.sh <PROJECT_ID> [REGION] [--ge APP_ID]}"
if [ -z "$REGION" ]; then
  REGION="${POSITIONAL[1]:-us-central1}"
fi
SA_NAME="gamerheads-runtime"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET_NAME="${PROJECT_ID}-gamerheads-artifacts"
DEPLOYER=$(gcloud config get-value account 2>/dev/null)

echo "Deploying GamerHeads Gameplay Reaction Director to project: $PROJECT_ID (region: $REGION)"
[ -n "$GE_APP_ID" ] && echo "  + Gemini Enterprise registration (APP_ID: $GE_APP_ID)"

gcloud config set project "$PROJECT_ID" --quiet

# ── Custom Service Account & IAM ──────────────────────────────────────
echo "===> Setting up runtime Service Account: ${SA_EMAIL}..."
gcloud services enable iam.googleapis.com --project="$PROJECT_ID" --quiet

if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="GamerHeads Director runtime" --project="$PROJECT_ID" --quiet
fi

for role in roles/aiplatform.user roles/serviceusage.serviceUsageConsumer roles/logging.logWriter roles/cloudtrace.agent; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" --role="$role" --condition=None --quiet 2>/dev/null || true
done

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
    --member="serviceAccount:${SA_EMAIL}" --role=roles/iam.serviceAccountTokenCreator \
    --project="$PROJECT_ID" --quiet 2>/dev/null || true

if [ -n "$DEPLOYER" ]; then
    gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
        --member="user:${DEPLOYER}" --role=roles/iam.serviceAccountUser \
        --project="$PROJECT_ID" --quiet 2>/dev/null || true
fi

# ── Setup GCP resources (APIs, GCS bucket) ──────────────────────────
if [ -f setup.sh ]; then
    bash setup.sh "$PROJECT_ID" "$REGION"
fi

# Grant bucket-scoped storage.objectAdmin to custom SA
echo "===> Granting bucket access to ${SA_EMAIL} on gs://${BUCKET_NAME}..."
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA_EMAIL}" --role=roles/storage.objectAdmin --quiet 2>/dev/null || true

# ── Install & Deploy via agents-cli ───────────────────────────────────
echo "===> Syncing dependencies and preparing agents-cli..."
uv sync --default-index https://pypi.org/simple --quiet
if [ ! -f .venv/bin/agents-cli ]; then
    uv pip install google-agents-cli --default-index https://pypi.org/simple --python .venv/bin/python --quiet || true
fi

CLI_BIN=".venv/bin/agents-cli"
if [ ! -f "$CLI_BIN" ]; then
    CLI_BIN="agents-cli"
fi

echo "===> Deploying agent to Vertex AI Agent Runtime..."
DEPLOY_OUTPUT=$(GOOGLE_CLOUD_PROJECT="$PROJECT_ID" GOOGLE_CLOUD_LOCATION="global" \
  ARTIFACT_BUCKET_NAME="$BUCKET_NAME" \
  "$CLI_BIN" deploy --project "$PROJECT_ID" --region "$REGION" \
  --service-account "$SA_EMAIL" \
  --update-env-vars "ARTIFACT_BUCKET_NAME=${BUCKET_NAME},GOOGLE_CLOUD_LOCATION=global" \
  --no-confirm-project \
  2>&1 | tee /dev/stderr) || true

REASONING_ENGINE_ID=$(echo "$DEPLOY_OUTPUT" | sed -n 's/.*reasoningEngines\/\([0-9]*\).*/\1/p' | tail -1)
if [ -z "$REASONING_ENGINE_ID" ] && [ -f deployment_metadata.json ]; then
    REASONING_ENGINE_ID=$(python3 -c "import json; print(json.load(open('deployment_metadata.json')).get('remote_agent_runtime_id', '').split('/')[-1])" 2>/dev/null || true)
fi

echo "===> Verifying Reasoning Engine status on Vertex AI..."
ACCESS_TOKEN=$(gcloud auth print-access-token)
ENGINE_CHECK=$(curl -s -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/reasoningEngines/${REASONING_ENGINE_ID}")

if echo "$ENGINE_CHECK" | grep -q '"name"'; then
    echo "Agent Engine deployment verified and ACTIVE!"
    echo "  Reasoning Engine ID: $REASONING_ENGINE_ID"
else
    echo "ERROR: Reasoning Engine deployment failed or is not active on Vertex AI!"
    echo "API response: $ENGINE_CHECK"
    echo "Please check Cloud Build / Cloud Logging for build errors."
    exit 1
fi

# ── Gemini Enterprise Registration ───────────────────────────────────
if [ -n "$GE_APP_ID" ] && [ -n "$REASONING_ENGINE_ID" ]; then
    CLEAN_GE_APP_ID="${GE_APP_ID##*/}"
    echo "===> Registering agent to Gemini Enterprise (Engine ID: ${CLEAN_GE_APP_ID})..."
    PROJECT_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
    ACCESS_TOKEN=$(gcloud auth print-access-token)

    # Grant Discovery Engine SA invocation permission on Vertex AI
    DISCOVERY_SA="service-${PROJECT_NUM}@gcp-sa-discoveryengine.iam.gserviceaccount.com"
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${DISCOVERY_SA}" \
        --role="roles/aiplatform.user" --condition=None --quiet 2>/dev/null || true

    AGENT_NAME="$(basename $(pwd))"
    if command -v python3 &>/dev/null && [ -f agent.yaml ]; then
        DISPLAY_NAME=$(python3 -c "
import yaml
d = yaml.safe_load(open('agent.yaml'))
dn = d.get('displayName', {})
print(dn.get('en', dn) if isinstance(dn, dict) else dn)
" 2>/dev/null || echo "$AGENT_NAME")
        AGENT_DESC=$(python3 -c "
import yaml
d = yaml.safe_load(open('agent.yaml'))
desc = d.get('description', {})
print(desc.get('en', desc) if isinstance(desc, dict) else desc)
" 2>/dev/null || echo "$DISPLAY_NAME")
    else
        DISPLAY_NAME="$AGENT_NAME"
        AGENT_DESC="$AGENT_NAME"
    fi

    REGISTER_RESPONSE=$(curl -s -X POST \
      "https://discoveryengine.googleapis.com/v1alpha/projects/${PROJECT_NUM}/locations/global/collections/default_collection/engines/${CLEAN_GE_APP_ID}/assistants/default_assistant/agents" \
      -H "Authorization: Bearer ${ACCESS_TOKEN}" \
      -H "Content-Type: application/json" \
      -H "X-Goog-User-Project: ${PROJECT_NUM}" \
      -d "{
        \"displayName\": \"${DISPLAY_NAME}\",
        \"description\": \"${AGENT_DESC}\",
        \"adk_agent_definition\": {
          \"tool_settings\": { \"tool_description\": \"${AGENT_DESC}\" },
          \"provisioned_reasoning_engine\": {
            \"reasoning_engine\": \"projects/${PROJECT_NUM}/locations/${REGION}/reasoningEngines/${REASONING_ENGINE_ID}\"
          }
        }
      }")

    if echo "$REGISTER_RESPONSE" | grep -q '"name"'; then
        echo "Gemini Enterprise registration successful!"
    else
        echo "Gemini Enterprise registration failed:"
        echo "$REGISTER_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$REGISTER_RESPONSE"
    fi
elif [ -n "$GE_APP_ID" ] && [ -z "$REASONING_ENGINE_ID" ]; then
    echo "Skipping GE registration (Agent Engine deploy failed — no Reasoning Engine ID)"
fi

echo "Deployment complete!"
