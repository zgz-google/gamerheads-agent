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

echo "Deploying AI Gaming Streamer Creative Director to project: $PROJECT_ID (region: $REGION)"
[ -n "$GE_APP_ID" ] && echo "  + Gemini Enterprise registration (APP_ID: $GE_APP_ID)"

gcloud config set project "$PROJECT_ID" --quiet

# ── Custom Service Account & IAM ──────────────────────────────────────
echo "===> Setting up runtime Service Account: ${SA_EMAIL}..."
gcloud services enable iam.googleapis.com --project="$PROJECT_ID" --quiet

if ! gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" &>/dev/null; then
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="AI Gaming Streamer Creative Director runtime" --project="$PROJECT_ID" --quiet
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
GE_RESULT_STATUS="NOT_REQUESTED"
GE_LOCATION="global"

if [ -n "$GE_APP_ID" ] && [ -n "$REASONING_ENGINE_ID" ]; then
    if [[ "$GE_APP_ID" =~ locations/([^/]+) ]]; then
        GE_LOCATION="${BASH_REMATCH[1]}"
    fi
    CLEAN_GE_APP_ID="${GE_APP_ID##*/}"
    PROJECT_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
    ACCESS_TOKEN=$(gcloud auth print-access-token)
    TARGET_RE="projects/${PROJECT_NUM}/locations/${REGION}/reasoningEngines/${REASONING_ENGINE_ID}"

    if [ "$GE_LOCATION" = "global" ]; then
        DISCOVERY_HOST="https://discoveryengine.googleapis.com"
    else
        DISCOVERY_HOST="https://${GE_LOCATION}-discoveryengine.googleapis.com"
    fi

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

    echo "===> Checking Gemini Enterprise registration status (Engine ID: ${CLEAN_GE_APP_ID})..."

    CHECK_RESULT=$(python3 -c '
import json, sys, urllib.request, urllib.error

token = sys.argv[1]
project_num = sys.argv[2]
discovery_host = sys.argv[3]
ge_location = sys.argv[4]
ge_app_id = sys.argv[5]
target_re = sys.argv[6]
target_display = sys.argv[7]
target_name = sys.argv[8]

url = f"{discovery_host}/v1alpha/projects/{project_num}/locations/{ge_location}/collections/default_collection/engines/{ge_app_id}/assistants/default_assistant/agents"
headers = {
    "Authorization": f"Bearer {token}",
    "x-goog-user-project": project_num
}

page_token = ""
matched = None
match_reason = ""

try:
    while True:
        req_url = url + (f"?pageToken={page_token}" if page_token else "")
        req = urllib.request.Request(req_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            agents = data.get("agents", [])
            for agent in agents:
                adk_def = agent.get("adkAgentDefinition") or agent.get("adk_agent_definition") or {}
                prov_re = adk_def.get("provisionedReasoningEngine") or adk_def.get("provisioned_reasoning_engine") or {}
                re_name = prov_re.get("reasoningEngine") or prov_re.get("reasoning_engine") or ""
                disp_name = agent.get("displayName", "")

                if target_re and re_name == target_re:
                    matched = agent
                    match_reason = f"Reasoning Engine matches ({re_name})"
                    break
                if target_display and disp_name.strip().lower() == target_display.strip().lower():
                    matched = agent
                    match_reason = f"Display Name matches (\"{disp_name}\")"
                    break
                if target_name and disp_name.strip().lower() == target_name.strip().lower():
                    matched = agent
                    match_reason = f"Agent Name matches (\"{disp_name}\")"
                    break

            if matched or not data.get("nextPageToken"):
                break
            page_token = data["nextPageToken"]

    if matched:
        adk_def = matched.get("adkAgentDefinition") or matched.get("adk_agent_definition") or {}
        prov_re = adk_def.get("provisionedReasoningEngine") or adk_def.get("provisioned_reasoning_engine") or {}
        re_name = prov_re.get("reasoningEngine") or prov_re.get("reasoning_engine") or ""
        res = {
            "status": "EXISTS",
            "name": matched.get("name", ""),
            "displayName": matched.get("displayName", ""),
            "reasoningEngine": re_name,
            "reason": match_reason
        }
    else:
        res = {"status": "NOT_FOUND"}
    print(json.dumps(res))
except urllib.error.HTTPError as e:
    err_body = e.read().decode(errors="ignore") if hasattr(e, "read") else str(e)
    print(json.dumps({"status": "ERROR", "code": e.code, "message": err_body}))
except Exception as e:
    print(json.dumps({"status": "ERROR", "message": str(e)}))
' "$ACCESS_TOKEN" "$PROJECT_NUM" "$DISCOVERY_HOST" "$GE_LOCATION" "$CLEAN_GE_APP_ID" "$TARGET_RE" "$DISPLAY_NAME" "$AGENT_NAME" 2>/dev/null || echo '{"status":"ERROR","message":"python execution failed"}')

    CHECK_STATUS=$(echo "$CHECK_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'ERROR'))" 2>/dev/null || echo "ERROR")

    if [ "$CHECK_STATUS" = "EXISTS" ]; then
        EXISTING_AGENT_NAME=$(echo "$CHECK_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('name', ''))")
        EXISTING_DISPLAY_NAME=$(echo "$CHECK_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('displayName', ''))")
        EXISTING_RE=$(echo "$CHECK_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('reasoningEngine', ''))")
        EXISTING_REASON=$(echo "$CHECK_RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('reason', ''))")

        echo "===> Gemini Enterprise: Agent is already registered! Skipping registration."
        echo "     Existing Agent: $EXISTING_AGENT_NAME"
        echo "     Display Name:   $EXISTING_DISPLAY_NAME"
        echo "     Match Reason:   $EXISTING_REASON"
        GE_RESULT_STATUS="SKIPPED_ALREADY_EXISTS"
    else
        echo "===> Registering agent to Gemini Enterprise (Engine ID: ${CLEAN_GE_APP_ID})..."
        REGISTER_RESPONSE=$(curl -s -X POST \
          "${DISCOVERY_HOST}/v1alpha/projects/${PROJECT_NUM}/locations/${GE_LOCATION}/collections/default_collection/engines/${CLEAN_GE_APP_ID}/assistants/default_assistant/agents" \
          -H "Authorization: Bearer ${ACCESS_TOKEN}" \
          -H "Content-Type: application/json" \
          -H "X-Goog-User-Project: ${PROJECT_NUM}" \
          -d "{
            \"displayName\": \"${DISPLAY_NAME}\",
            \"description\": \"${AGENT_DESC}\",
            \"adk_agent_definition\": {
              \"tool_settings\": { \"tool_description\": \"${AGENT_DESC}\" },
              \"provisioned_reasoning_engine\": {
                \"reasoning_engine\": \"${TARGET_RE}\"
              }
            }
          }")

        if echo "$REGISTER_RESPONSE" | grep -q '"name"'; then
            echo "Gemini Enterprise registration successful!"
            NEW_AGENT_NAME=$(echo "$REGISTER_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('name', ''))" 2>/dev/null || echo "")
            GE_RESULT_STATUS="REGISTERED"
        else
            echo "Gemini Enterprise registration failed:"
            echo "$REGISTER_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$REGISTER_RESPONSE"
            GE_RESULT_STATUS="FAILED"
            GE_ERROR_DETAILS=$(echo "$REGISTER_RESPONSE" | tr '\n' ' ')
        fi
    fi
elif [ -n "$GE_APP_ID" ] && [ -z "$REASONING_ENGINE_ID" ]; then
    echo "Skipping GE registration (Agent Engine deploy failed — no Reasoning Engine ID)"
    GE_RESULT_STATUS="FAILED_NO_RE_ID"
fi

# ── Deployment Summary ───────────────────────────────────────────────
echo ""
echo "========================================================================="
echo "                         DEPLOYMENT SUMMARY                              "
echo "========================================================================="
echo "Project ID:        $PROJECT_ID"
echo "Region:            $REGION"
echo "Service Account:   $SA_EMAIL"
echo "Artifact Bucket:   gs://$BUCKET_NAME"
if [ -n "$REASONING_ENGINE_ID" ]; then
    echo "Reasoning Engine:  projects/${PROJECT_NUM:-$PROJECT_ID}/locations/${REGION}/reasoningEngines/${REASONING_ENGINE_ID}"
    echo "Engine Console:    https://console.cloud.google.com/vertex-ai/agents/agent-engines/locations/${REGION}/agent-engines/${REASONING_ENGINE_ID}?project=${PROJECT_ID}"
fi
echo "-------------------------------------------------------------------------"

if [ "$GE_RESULT_STATUS" = "SKIPPED_ALREADY_EXISTS" ]; then
    echo "Gemini Enterprise: ⏭️  SKIPPED (Already Registered)"
    echo "  Status:          Agent is already registered in Gemini Enterprise. Skipped to prevent duplicates."
    echo "  Engine ID:       $CLEAN_GE_APP_ID"
    echo "  Existing Agent:  $EXISTING_AGENT_NAME"
    echo "  Display Name:    $EXISTING_DISPLAY_NAME"
    [ -n "$EXISTING_RE" ] && echo "  Reasoning Engine: $EXISTING_RE"
    echo "  Matched By:      $EXISTING_REASON"
    echo "  GE Console URL:  https://console.cloud.google.com/gemini-enterprise/locations/${GE_LOCATION:-global}/engines/${CLEAN_GE_APP_ID}/overview/dashboard?project=${PROJECT_ID}"
elif [ "$GE_RESULT_STATUS" = "REGISTERED" ]; then
    echo "Gemini Enterprise: ✅ SUCCESS (Newly Registered)"
    echo "  Engine ID:       $CLEAN_GE_APP_ID"
    echo "  Agent Resource:  $NEW_AGENT_NAME"
    echo "  Display Name:    $DISPLAY_NAME"
    echo "  GE Console URL:  https://console.cloud.google.com/gemini-enterprise/locations/${GE_LOCATION:-global}/engines/${CLEAN_GE_APP_ID}/overview/dashboard?project=${PROJECT_ID}"
elif [ "$GE_RESULT_STATUS" = "FAILED" ]; then
    echo "Gemini Enterprise: ❌ FAILED"
    echo "  Engine ID:       $CLEAN_GE_APP_ID"
    echo "  Error:           $GE_ERROR_DETAILS"
elif [ "$GE_RESULT_STATUS" = "FAILED_NO_RE_ID" ]; then
    echo "Gemini Enterprise: ⚠️  SKIPPED (Reasoning Engine deployment failed)"
elif [ "$GE_RESULT_STATUS" = "NOT_REQUESTED" ]; then
    echo "Gemini Enterprise: ℹ️  NOT CONFIGURED"
    echo "  Note:            Pass --ge <APP_ID> to register with Gemini Enterprise."
fi
echo "========================================================================="
echo "Deployment complete!"
