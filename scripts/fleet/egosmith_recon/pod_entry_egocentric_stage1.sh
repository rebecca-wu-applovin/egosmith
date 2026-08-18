#!/usr/bin/env bash
# pod_entry_egocentric_stage1.sh — per-pod entrypoint for the Egocentric-100K Stage-1-over-all fleet.
# Each pod owns a stride-slice of the global part.tar index: stream its part.tars from GCS, decode +
# fisheye-undistort each clip's mp4, run the Stage-1 gates, and upload a per-shard survivor manifest.
# NO reconstruction, NO persistent frame tars — the cheap cut over all ~2M clips. Idempotent
# (shard skipped if its report is already on GCS).
set -o pipefail   # NOT -u: conda activation references unbound vars
export PYTHONUNBUFFERED=1 NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
echo "[ego_s1] pod_entry START shard=${SHARD:-${JOB_COMPLETION_INDEX:-?}} @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; fi
conda activate egosmith 2>/dev/null || true
python -c "import gcsfs" 2>/dev/null || { echo "[ego_s1] installing gcsfs..."; python -m pip install -q gcsfs 2>&1 | tail -1; }

SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
case "$SHARD" in ''|*[!0-9]*) echo "[ego_s1] FATAL: SHARD not numeric ('$SHARD')"; exit 1;; esac
NSHARDS="${NSHARDS:-90}"
BALANCE="${BALANCE:-0.0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-0}"
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/ego_s1}"; mkdir -p "$LOCAL_ROOT"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
PARTS_INDEX_GCS="${PARTS_INDEX_GCS:-$FLEET/egocentric100k_parts_index.txt}"
OUTGCS="${OUTGCS:-gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/stage1/_shards}"
sfx=$(printf '%05d' "$SHARD")
log(){ echo "[ego_s1][shard $SHARD/$NSHARDS] $*"; }

# overlay current code onto /repo (image may be an older commit)
gcloud storage cp "$FLEET/egosmith_code.tar.gz" /tmp/code.tar.gz 2>/dev/null && tar xzf /tmp/code.tar.gz -C "$REPO_ROOT" && log "overlaid current scripts+src"
ls scripts/build/egocentric_stage1_mp4.py >/dev/null 2>&1 || { log "FATAL: egocentric_stage1_mp4.py missing after overlay"; exit 1; }

# skip-if-done (report is the done-marker, uploaded last)
if gcloud storage ls "$OUTGCS/shard_$sfx.report.json" >/dev/null 2>&1; then log "shard already done, skip"; exit 0; fi

W="$LOCAL_ROOT/$sfx"; rm -rf "$W"; mkdir -p "$W"
gcloud storage cp "$PARTS_INDEX_GCS" "$W/parts_index.txt" || { log "PARTS INDEX PULL FAIL"; exit 1; }
# stride slice: parts where (line-1) % NSHARDS == SHARD
awk -v n="$NSHARDS" -v s="$SHARD" 'NR>0 && ((NR-1)%n)==s' "$W/parts_index.txt" > "$W/parts_shard.txt"
log "parts in shard: $(wc -l < "$W/parts_shard.txt") of $(wc -l < "$W/parts_index.txt")"
[ -s "$W/parts_shard.txt" ] || { log "no parts for this shard (NSHARDS>parts?) -> done"; : > "$W/empty"; }

LIM=""; [ "$SMOKE_LIMIT" -gt 0 ] && LIM="--limit $SMOKE_LIMIT"
if [ -s "$W/parts_shard.txt" ]; then
  PYTHONPATH=src python scripts/build/egocentric_stage1_mp4.py \
    --parts_list "$W/parts_shard.txt" \
    --out_manifest "$W/stage1.kept.jsonl" \
    --report_out "$W/stage1.report.json" \
    --balance "$BALANCE" $LIM 2>&1 | grep -vE "FutureWarning|warnings.warn" | tail -40
else
  echo '{"parts":0,"total_clips":0,"kept_clips":0,"kept_hours":0}' > "$W/stage1.report.json"
  : > "$W/stage1.kept.jsonl"
fi
[ -f "$W/stage1.report.json" ] || { log "NO REPORT PRODUCED — abort"; exit 1; }

log "upload shard survivor manifest + report..."
gcloud storage cp "$W/stage1.kept.jsonl"  "$OUTGCS/shard_$sfx.kept.jsonl"   >/dev/null 2>&1 || { log "kept upload FAIL"; exit 1; }
gcloud storage cp "$W/stage1.report.json" "$OUTGCS/shard_$sfx.report.json"  >/dev/null 2>&1 || { log "report upload FAIL"; exit 1; }  # last = done-marker
rm -rf "$W"
log "SHARD DONE @ $(date +%F_%H:%M:%S)"
