#!/usr/bin/env bash
# Render + apply the EgoSmith reconstruction Indexed Job for one mode.
#
#   MODE=recon|use_gt  NSHARDS=..  [PARALLELISM=..] \
#   NAMESPACE=..  K8S_SA=..  IMAGE=..  bash scripts/fleet/egosmith_recon/launch.sh
#
# Smoke a single node first:  MODE=recon NSHARDS=1 PARALLELISM=1 ... bash launch.sh
# (pod_entry stride-shards, so NSHARDS=1 runs ALL clips on one node — for a true
#  ~20-clip smoke use a trimmed manifest; see SMOKE.md.)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

: "${MODE:?set MODE=recon|use_gt}"
: "${NSHARDS:?set NSHARDS}"
: "${NAMESPACE:?set NAMESPACE}"
: "${K8S_SA:?set K8S_SA (WI-bound KSA with GCS access)}"
: "${IMAGE:?set IMAGE}"
# DNS-safe job name (K8s labels forbid '_'); MODE keeps its underscore for pod logic.
export JOB_SUFFIX="${JOB_SUFFIX:-$(printf '%s' "$MODE" | tr '_' '-')}"
export MODE NSHARDS NAMESPACE K8S_SA IMAGE
export PARALLELISM="${PARALLELISM:-$NSHARDS}"
# ".smoke" targets the trimmed smoke manifests; empty = full run.
export MANIFEST_SUFFIX="${MANIFEST_SUFFIX:-}"
# Any4D depth batch (pipeline default 32; reduce only if it OOMs on L4's 22GB).
export ANY4D_BATCH_SIZE="${ANY4D_BATCH_SIZE:-32}"
# Clips per batch_infer invocation; workers respawn between chunks to bound the per-clip
# GPU/host-memory accumulation that OOM-kills long-lived slam workers.
export CHUNK_CLIPS="${CHUNK_CLIPS:-16}"
# 1 = estimate focal with AnyCalib in detect_track (needs a v2 image with anycalib installed).
export USE_ANYCALIB="${USE_ANYCALIB:-0}"

TMPL="$REPO_ROOT/scripts/fleet/egosmith_recon/job.template.yaml"
# envsubst-equivalent (envsubst may be absent): expand ${VAR} from the environment.
rendered="$(python3 -c 'import os,sys; sys.stdout.write(os.path.expandvars(sys.stdin.read()))' < "$TMPL")"
echo "$rendered"
echo "----- applying (mode=$MODE nshards=$NSHARDS parallelism=$PARALLELISM ns=$NAMESPACE sa=$K8S_SA) -----"
echo "$rendered" | kubectl apply -f -
