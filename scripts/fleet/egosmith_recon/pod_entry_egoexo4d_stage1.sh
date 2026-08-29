#!/usr/bin/env bash
# pod_entry_egoexo4d_stage1.sh — per-pod entrypoint for Stage-1 over the Ego-Exo4D
# EGO stream (Aria RGB 214-1, 1408x1408 square fisheye). Identical contract to
# pod_entry_egocentric_stage1_video.sh with ONE difference: the clip config is the
# egoexo4d variant (decode 448x448 square — the production 448x256 squash halves
# hand-box areas on square sources and collapses Gate A; measured 2026-08-25).
# Gates/thresholds are otherwise production-identical. Idempotent via report marker.
set -o pipefail   # NOT -u: conda activation references unbound vars
export PYTHONUNBUFFERED=1 NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
echo "[s1v-ee4d] pod_entry START shard=${SHARD:-${JOB_COMPLETION_INDEX:-?}} @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; fi
conda activate egosmith 2>/dev/null || true
python -c "import gcsfs" 2>/dev/null || { echo "[s1v-ee4d] installing gcsfs..."; python -m pip install -q gcsfs 2>&1 | tail -1; }

SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
case "$SHARD" in ''|*[!0-9]*) echo "[s1v-ee4d] FATAL: SHARD not numeric ('$SHARD')"; exit 1;; esac
NSHARDS="${NSHARDS:-500}"
SMOKE_LIMIT="${SMOKE_LIMIT:-0}"
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/ego_s1v}"; mkdir -p "$LOCAL_ROOT"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
VIDEOS_INDEX_GCS="${VIDEOS_INDEX_GCS:?FATAL: VIDEOS_INDEX_GCS unset}"
OUTGCS="${OUTGCS:?FATAL: OUTGCS unset}"
sfx=$(printf '%05d' "$SHARD")
log(){ echo "[s1v-ee4d][shard $SHARD/$NSHARDS] $*"; }

# overlay current code onto /repo (image may be an older commit), then pull the
# video driver + egoexo4d clip config explicitly (never touch the shared tarball).
gcloud storage cp "$FLEET/egosmith_code.tar.gz" /tmp/code.tar.gz 2>/dev/null && tar xzf /tmp/code.tar.gz -C "$REPO_ROOT" && log "overlaid current scripts+src"
gcloud storage cp "$FLEET/egocentric_stage1_video.py" scripts/build/egocentric_stage1_video.py || { log "FATAL: driver pull failed"; exit 1; }
gcloud storage cp "$FLEET/heuristic_clip_config.egoexo4d.yaml" /tmp/clip_config.egoexo4d.yaml || { log "FATAL: config pull failed"; exit 1; }

# skip-if-done (report is the done-marker, uploaded last)
if gcloud storage ls "$OUTGCS/shard_$sfx.report.json" >/dev/null 2>&1; then log "shard already done, skip"; exit 0; fi

W="$LOCAL_ROOT/$sfx"; rm -rf "$W"; mkdir -p "$W/tmp"
gcloud storage cp "$VIDEOS_INDEX_GCS" "$W/videos_index.jsonl" || { log "VIDEO INDEX PULL FAIL"; exit 1; }
# stride slice: lines where (line-1) % NSHARDS == SHARD
awk -v n="$NSHARDS" -v s="$SHARD" '((NR-1)%n)==s' "$W/videos_index.jsonl" > "$W/videos_shard.jsonl"
log "videos in shard: $(wc -l < "$W/videos_shard.jsonl") of $(wc -l < "$W/videos_index.jsonl")"

LIM=""; [ "$SMOKE_LIMIT" -gt 0 ] && LIM="--limit $SMOKE_LIMIT"
if [ -s "$W/videos_shard.jsonl" ]; then
  PYTHONPATH=src python scripts/build/egocentric_stage1_video.py \
    --videos_list "$W/videos_shard.jsonl" \
    --config /tmp/clip_config.egoexo4d.yaml \
    --out_manifest "$W/stage1.kept.jsonl" \
    --report_out "$W/stage1.report.json" \
    --work_dir "$W/tmp" $LIM 2>&1 | grep -vE "FutureWarning|warnings.warn" | tail -60
else
  echo '{"videos":0,"kept_videos":0,"raw_hours":0,"analyzed_hours":0,"kept_hours":0,"hours_fraction":0}' > "$W/stage1.report.json"
  : > "$W/stage1.kept.jsonl"
fi
[ -f "$W/stage1.report.json" ] || { log "NO REPORT PRODUCED — abort"; exit 1; }

log "upload shard survivor manifest + report..."
gcloud storage cp "$W/stage1.kept.jsonl"  "$OUTGCS/shard_$sfx.kept.jsonl"   >/dev/null 2>&1 || { log "kept upload FAIL"; exit 1; }
gcloud storage cp "$W/stage1.report.json" "$OUTGCS/shard_$sfx.report.json"  >/dev/null 2>&1 || { log "report upload FAIL"; exit 1; }  # last = done-marker
rm -rf "$W"
log "SHARD DONE @ $(date +%F_%H:%M:%S)"
