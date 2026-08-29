#!/usr/bin/env bash
# Phase D fleet pod: runs the incremental filter daemon on an L4 node (CPU only).
set -o pipefail
export PYTHONUNBUFFERED=1
echo "[phase_d_pod] START idx=${JOB_COMPLETION_INDEX:-?} @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
source /opt/conda/etc/profile.d/conda.sh && conda activate egosmith
python -c "import gcsfs" 2>/dev/null || python -m pip install -q gcsfs 2>&1 | tail -1
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
gcloud storage cp "$FLEET/egosmith_code.tar.gz" /tmp/code.tar.gz && tar xzf /tmp/code.tar.gz -C "$REPO_ROOT"
gcloud storage cp "$FLEET/phase_d_incremental.py" /tmp/phase_d_incremental.py
export EGOSMITH_ROOT="$REPO_ROOT" PHASED_WORK=/scratch/phased_work D_WORKER_ID="pod-${JOB_COMPLETION_INDEX:-x}"
mkdir -p "$PHASED_WORK"
# loop until D fully complete (3000 filter reports)
while :; do
  N=$(gcloud storage ls 'gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/filter_run/_shards/*.report.json' 2>/dev/null | wc -l)
  [ "$N" -ge 3000 ] && { echo "[phase_d_pod] all 3000 filtered — done"; break; }
  python /tmp/phase_d_incremental.py --workers 64 --shard_parallel 4 --once 2>&1 | tail -4
  sleep 120
done
