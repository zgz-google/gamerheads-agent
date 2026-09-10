#!/bin/bash
set -e

# ==============================================================================
# Setup script for GamerHeads Agent
# Provisions required GCP resources (APIs, Cloud Storage Bucket).
# Supports --cleanup flag to tear down provisioned resources.
# ==============================================================================

CLEANUP=false
POSITIONAL=()

for arg in "$@"; do
  case $arg in
    --cleanup)
      CLEANUP=true
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done

PROJECT_ID="${POSITIONAL[0]:?Usage: bash setup.sh <PROJECT_ID> [REGION] [--cleanup]}"
REGION="${POSITIONAL[1]:-us-central1}"
BUCKET_NAME="${PROJECT_ID}-gamerheads-artifacts"

# Cost warning: Mentions paid resources
echo "[INFO] Target GCP Project: ${PROJECT_ID}"
echo "[INFO] Target Region: ${REGION}"
echo "[NOTICE] This setup manages resources that may incur Google Cloud charges (Vertex AI, Cloud Storage)."

gcloud config set project "${PROJECT_ID}" --quiet

if [ "$CLEANUP" = true ]; then
  echo "===> Cleaning up provisioned resources for ${PROJECT_ID}..."
  if gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Deleting GCS Bucket gs://${BUCKET_NAME}..."
    gcloud storage rm --recursive "gs://${BUCKET_NAME}" --quiet || true
  else
    echo "GCS Bucket gs://${BUCKET_NAME} does not exist, skipping."
  fi
  echo "Cleanup complete."
  exit 0
fi

echo "===> 1. Enabling required Google Cloud APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  serviceusage.googleapis.com \
  logging.googleapis.com \
  cloudtrace.googleapis.com \
  iam.googleapis.com \
  --project="${PROJECT_ID}" --quiet

echo "===> 2. Setting up Cloud Storage Artifact Bucket..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
  echo "Creating GCS Bucket gs://${BUCKET_NAME} in location ${REGION}..."
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --uniform-bucket-level-access --quiet
  echo "Created GCS Bucket gs://${BUCKET_NAME}."
else
  echo "GCS Bucket gs://${BUCKET_NAME} already exists."
fi

echo "===> GCP Resource Setup Complete!"
echo "ARTIFACT_BUCKET_NAME=${BUCKET_NAME}"
