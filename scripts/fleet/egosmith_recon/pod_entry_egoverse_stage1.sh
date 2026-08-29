#!/usr/bin/env bash
# pod_entry_egoverse_stage1.sh — per-pod entrypoint for Stage-1 over EgoVerse
# subsets (eva / lightwheel / abc; aria pattern). Identical contract to
# pod_entry_egocentric_stage1_video.sh with TWO additions:
#   * optional CLIP_CONFIG env (yaml filename under $FLEET,
#     e.g. heuristic_clip_config.lightwheel.yaml; empty = production default)
#   * optional DRIVER env (stage-1 driver filename under $FLEET; default
#     egocentric_stage1_video.py; abc uses egoverse_zarr_stage1.py which reads
#     zarr-v3 JPEG episodes via ranged GCS reads).
# Idempotent via report marker.
set -o pipefail   # NOT -u: conda activation references unbound vars
export PYTHONUNBUFFERED=1 NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
echo "[s1v-egov] pod_entry START shard=${SHARD:-${JOB_COMPLETION_INDEX:-?}} @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; fi
conda activate egosmith 2>/dev/null || true
python -c "import gcsfs" 2>/dev/null || { echo "[s1v-egov] installing gcsfs..."; python -m pip install -q gcsfs 2>&1 | tail -1; }

SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
case "$SHARD" in ''|*[!0-9]*) echo "[s1v-egov] FATAL: SHARD not numeric ('$SHARD')"; exit 1;; esac
NSHARDS="${NSHARDS:-100}"
SMOKE_LIMIT="${SMOKE_LIMIT:-0}"
CLIP_CONFIG="${CLIP_CONFIG:-}"
DRIVER="${DRIVER:-egocentric_stage1_video.py}"
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/ego_s1v}"; mkdir -p "$LOCAL_ROOT"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
VIDEOS_INDEX_GCS="${VIDEOS_INDEX_GCS:?FATAL: VIDEOS_INDEX_GCS unset}"
OUTGCS="${OUTGCS:?FATAL: OUTGCS unset}"
sfx=$(printf '%05d' "$SHARD")
log(){ echo "[s1v-egov][shard $SHARD/$NSHARDS] $*"; }

# overlay current code onto /repo (image may be an older commit), then pull the
# video driver + optional clip-config variant explicitly (never touch the shared tar).
gcloud storage cp "$FLEET/egosmith_code.tar.gz" /tmp/code.tar.gz 2>/dev/null && tar xzf /tmp/code.tar.gz -C "$REPO_ROOT" && log "overlaid current scripts+src"
gcloud storage cp "$FLEET/$DRIVER" "scripts/build/$DRIVER" || { log "FATAL: driver pull failed"; exit 1; }
python -c "import zstandard" 2>/dev/null || { echo "[s1v-egov] installing zstandard..."; python -m pip install -q zstandard 2>&1 | tail -1; }
CFGARG=""
if [ -n "$CLIP_CONFIG" ]; then
  gcloud storage cp "$FLEET/$CLIP_CONFIG" "/tmp/$CLIP_CONFIG" || { log "FATAL: clip config pull failed"; exit 1; }
  CFGARG="--config /tmp/$CLIP_CONFIG"
fi

# skip-if-done (report is the done-marker, uploaded last)
if gcloud storage ls "$OUTGCS/shard_$sfx.report.json" >/dev/null 2>&1; then log "shard already done, skip"; exit 0; fi

W="$LOCAL_ROOT/$sfx"; rm -rf "$W"; mkdir -p "$W/tmp"
gcloud storage cp "$VIDEOS_INDEX_GCS" "$W/videos_index.jsonl" || { log "VIDEO INDEX PULL FAIL"; exit 1; }
# stride slice: lines where (line-1) % NSHARDS == SHARD
awk -v n="$NSHARDS" -v s="$SHARD" '((NR-1)%n)==s' "$W/videos_index.jsonl" > "$W/videos_shard.jsonl"
log "videos in shard: $(wc -l < "$W/videos_shard.jsonl") of $(wc -l < "$W/videos_index.jsonl")"

LIM=""; [ "$SMOKE_LIMIT" -gt 0 ] && LIM="--limit $SMOKE_LIMIT"
WORKARG="--work_dir $W/tmp"; [ "$DRIVER" = "egoverse_zarr_stage1.py" ] && WORKARG=""
if [ -s "$W/videos_shard.jsonl" ]; then
  PYTHONPATH=src python "scripts/build/$DRIVER" \
    --videos_list "$W/videos_shard.jsonl" \
    --out_manifest "$W/stage1.kept.jsonl" \
    --report_out "$W/stage1.report.json" \
    $WORKARG $CFGARG $LIM 2>&1 | grep -vE "FutureWarning|warnings.warn" | tail -60
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
