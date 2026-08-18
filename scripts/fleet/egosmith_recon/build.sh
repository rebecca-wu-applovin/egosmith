#!/usr/bin/env bash
# Build + push the EgoSmith reconstruction L4 image via Cloud Build (no local docker).
# Run from the repo root. --ignore-file keeps big checkpoints + stale DPVO build/ out
# of the uploaded context.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-gcr.io/strategic-atom-700/egosmith-recon-l4:v1}"
echo "[build] submitting Cloud Build -> $IMAGE"
gcloud builds submit \
  --config scripts/fleet/egosmith_recon/cloudbuild.yaml \
  --substitutions "_IMAGE=${IMAGE}" \
  --ignore-file scripts/fleet/egosmith_recon/.dockerignore \
  .
echo "[build] done: $IMAGE"
