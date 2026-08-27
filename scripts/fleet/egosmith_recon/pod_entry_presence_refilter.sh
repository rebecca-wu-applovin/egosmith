#!/usr/bin/env bash
# pod_entry_presence_refilter.sh — per-pod entrypoint for the presence re-filter fleet
# (CPU-only). Each Indexed-Job pod owns SHARDS_PER_POD consecutive filter shards of one
# dataset and runs scripts/build/presence_refilter.py over them: ranged pred_valid reads
# from result.npz -> drop empty_pose / single_valid_hand keeps, backup-first manifest +
# annotation rewrite. Idempotent: shards with a .presence2.done marker are skipped, so
# pod restarts and job re-applies only touch unfinished shards.
set -o pipefail   # NOT -u: conda activation references unbound vars
export PYTHONUNBUFFERED=1
echo "[presence2] pod_entry START shard=${SHARD:-${JOB_COMPLETION_INDEX:-?}} @ $(date +%F_%H:%M:%S)"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; fi
conda activate egosmith 2>/dev/null || true
python -c "import gcsfs" 2>/dev/null || { echo "[presence2] installing gcsfs..."; python -m pip install -q gcsfs 2>&1 | tail -1; }

SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
case "$SHARD" in ''|*[!0-9]*) echo "[presence2] FATAL: SHARD not numeric ('$SHARD')"; exit 1;; esac
DS="${DS:-egocentric100k}"
SHARDS_PER_POD="${SHARDS_PER_POD:-10}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
WORKERS="${WORKERS:-64}"
SHARD_PARALLEL="${SHARD_PARALLEL:-1}"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet

gcloud storage cp "$FLEET/presence_refilter.py" /tmp/presence_refilter.py || { echo "[presence2] FATAL: script pull failed"; exit 1; }
A=$(( (SHARD + SHARD_OFFSET) * SHARDS_PER_POD ))
B=$(( A + SHARDS_PER_POD - 1 ))
echo "[presence2] dataset=$DS shards=$A-$B workers=$WORKERS"
python /tmp/presence_refilter.py --dataset "$DS" --shards "$A-$B" \
  --workers "$WORKERS" --shard_parallel "$SHARD_PARALLEL"
EC=$?
echo "[presence2] DONE ec=$EC @ $(date +%F_%H:%M:%S)"
exit $EC
