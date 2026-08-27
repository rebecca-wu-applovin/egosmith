#!/usr/bin/env bash
# pod_entry_egoverse_native.sh — per-pod entrypoint for the EgoVerse NATIVE-keypoints
# builds (mecka-flagship / microagi / scale / lightwheel). egodex-filter pattern:
# convert (generate_keypoints_wds.py, GT-derived lowdim, no recon) -> optional Stage-1
# prefilter on the emitted frame tars (YOLO+flow; only when the subset never had a
# video Stage-1 — microagi/scale) -> Stage-4 native filter -> upload.
#
# Env contract:
#   DS               dataset name == spec basename (egoverse_microagi, ...)
#   NSHARDS / SHARD  episode stride-sharding over EP_LIST (i % NSHARDS == SHARD)
#   EP_LIST          filename under $FLEET: one episode ref per line (gs:// or id)
#   EP_PREFIX        optional gs prefix to prepend to non-gs lines (id -> gs://.../id.zarr)
#   INTERVALS        optional intervals JSON filename under $FLEET (--intervals_json)
#   RUN_STAGE1       1 = run stage1_prefilter between convert and stage4 (microagi/scale)
#   SEGMENT_SEC      sub-clip length (default 10)
#   SOURCE_FPS       filter fps (default 30; microagi 29)
#   MIN_PRESENCE     --min_presence_ratio for stage4 (default 0.5)
#   SMOKE_LIMIT      >0 = cap episodes per shard
#   WORKERS          converter workers (default 6)
# Idempotent: shard skipped if its stage4.kept.jsonl (done-marker) exists.
set -o pipefail
export PYTHONUNBUFFERED=1 NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
echo "[egov_native] START shard=${SHARD:-${JOB_COMPLETION_INDEX:-?}} @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; fi
conda activate egosmith 2>/dev/null || true
python -c "import gcsfs, zstandard, yaml" 2>/dev/null || python -m pip install -q gcsfs zstandard pyyaml 2>&1 | tail -1

SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
case "$SHARD" in ''|*[!0-9]*) echo "[egov_native] FATAL: SHARD not numeric ('$SHARD')"; exit 1;; esac
NSHARDS="${NSHARDS:?FATAL: NSHARDS unset}"
DS="${DS:?FATAL: DS unset}"
EP_LIST="${EP_LIST:?FATAL: EP_LIST unset}"
EP_PREFIX="${EP_PREFIX:-}"
INTERVALS="${INTERVALS:-}"
RUN_STAGE1="${RUN_STAGE1:-0}"
SEGMENT_SEC="${SEGMENT_SEC:-10}"
SOURCE_FPS="${SOURCE_FPS:-30}"
MIN_PRESENCE="${MIN_PRESENCE:-0.5}"
SMOKE_LIMIT="${SMOKE_LIMIT:-0}"
WORKERS="${WORKERS:-6}"
# RUN_TAG namespaces the per-shard manifests/markers (native/_shards/shard_X.<tag>.*)
# so a smoke run's shard-0 done-marker can never make the full run skip shard 0
# (bucket deletions are not available to the operator; markers are append-only).
RUN_TAG="${RUN_TAG:-}"
TAG=""; [ -n "$RUN_TAG" ] && TAG=".$RUN_TAG"
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/egov_native}"; mkdir -p "$LOCAL_ROOT"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
FILT=gs://foundational-research/hoi-dataset/egosmith_filtered/$DS
sfx=$(printf '%05d' "$SHARD")
log(){ echo "[egov_native][$DS shard $SHARD/$NSHARDS] $*"; }

# code overlay: the NATIVE tarball (built from git HEAD; carries the egoverse
# extractors + specs). Never reuses egosmith_code.tar.gz (frozen for the video conveyor).
gcloud storage cp "$FLEET/egosmith_code_native.tar.gz" /tmp/code.tar.gz || { log "FATAL: code tarball pull failed"; exit 1; }
tar xzf /tmp/code.tar.gz -C "$REPO_ROOT" && log "overlaid native code"
ls scripts/build/generate_keypoints_wds.py configs/keypoint_specs/$DS.yaml >/dev/null 2>&1 || { log "FATAL: converter/spec missing after overlay"; exit 1; }

# skip-if-done
if gcloud storage ls "$FILT/native/_shards/shard_$sfx$TAG.stage4.kept.jsonl" >/dev/null 2>&1; then log "shard already done, skip"; exit 0; fi

W="$LOCAL_ROOT/$sfx"; rm -rf "$W"; mkdir -p "$W/frames" "$W/outputs"
trap 'echo "[egov_native] EXIT disk:"; df -h /scratch 2>/dev/null | tail -1' EXIT

# episode slice
gcloud storage cp "$FLEET/$EP_LIST" "$W/eps_all.txt" || { log "FATAL: ep list pull failed"; exit 1; }
python3 - "$W" "$SHARD" "$NSHARDS" "$EP_PREFIX" "$SMOKE_LIMIT" <<'PY'
import sys
W, sh, n, pre, lim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
eps = [l.strip() for l in open(f"{W}/eps_all.txt") if l.strip()]
mine = [e for i, e in enumerate(eps) if i % n == sh]
if lim: mine = mine[:lim]
if pre: mine = [e if e.startswith("gs://") else f"{pre.rstrip('/')}/{e}" for e in mine]
open(f"{W}/eps.txt", "w").write(",".join(mine))
print(f"  shard episodes: {len(mine)} / {len(eps)}")
PY
EPS=$(cat "$W/eps.txt")
[ -z "$EPS" ] && { log "empty shard -> writing empty marker"; : > "$W/empty"; \
  echo -n "" | gcloud storage cp - "$FILT/native/_shards/shard_$sfx$TAG.stage4.kept.jsonl"; exit 0; }

IVARG=""
if [ -n "$INTERVALS" ]; then
  gcloud storage cp "$FLEET/$INTERVALS" "$W/intervals.json" || { log "FATAL: intervals pull failed"; exit 1; }
  IVARG="--intervals_json $W/intervals.json"
fi

log "convert..."
python scripts/build/generate_keypoints_wds.py --spec "configs/keypoint_specs/$DS.yaml" \
  --frames_root "$W/frames" --outputs_root "$W/outputs" \
  --manifest_out "$W/manifest.jsonl" --report_out "$W/convert.json" \
  --workers "$WORKERS" --segment_sec "$SEGMENT_SEC" $IVARG \
  --episodes "$EPS" 2>&1 | tail -3
ntar=$(ls "$W/frames"/*.tar 2>/dev/null | wc -l)
nrec=$(grep -c . "$W/manifest.jsonl" 2>/dev/null || echo 0)
log "tars=$ntar manifest=$nrec"
# 0-for-N guard: nothing converted AND real failures = systemic -> retry
nfail=$(python3 -c "import json;print(json.load(open('$W/convert.json'))['failed'])" 2>/dev/null || echo 0)
if [ "$ntar" -eq 0 ] && [ "$nfail" -gt 0 ]; then log "FATAL: 0 tars with $nfail failures"; exit 1; fi

S4IN="$W/manifest.jsonl"
if [ "$RUN_STAGE1" = "1" ] && [ "$ntar" -gt 0 ]; then
  log "Stage-1 prefilter..."
  python scripts/build/stage1_prefilter.py --input_manifest "$W/manifest.jsonl" \
    --output_manifest "$W/stage1.kept.jsonl" --report_out "$W/s1.json" \
    --fps "$SOURCE_FPS" 2>&1 | tail -2
  S4IN="$W/stage1.kept.jsonl"
fi

if [ "$ntar" -gt 0 ]; then
  log "Stage-4 native filter..."
  python scripts/build/filter_manifest_by_quality.py --input_manifest "$S4IN" \
    --output_manifest "$W/stage4.kept.jsonl" --report_out "$W/s4.json" \
    --stages native_features --source_fps "$SOURCE_FPS" --target_fps "$SOURCE_FPS" \
    --min_presence_ratio "$MIN_PRESENCE" --workers 12 2>&1 | grep -vE "WARNING|it/s" | tail -1
else
  : > "$W/stage4.kept.jsonl"
fi
python3 - "$W" "$SHARD" <<'PY'
import sys
w, sh = sys.argv[1:3]
def n(f):
    try: return sum(1 for _ in open(f))
    except FileNotFoundError: return -1
print(f"[egov_native][shard {sh}] FUNNEL: converted {n(w+'/manifest.jsonl')} -> s1 {n(w+'/stage1.kept.jsonl')} -> s4 {n(w+'/stage4.kept.jsonl')}", flush=True)
PY

# upload: KEPT tars only (stage4 survivors; dropped sub-clips never ship), then manifests,
# stage4.kept last (done marker).
if [ -s "$W/stage4.kept.jsonl" ]; then
  python3 - "$W" <<'PY'
import json, sys
W = sys.argv[1]
import os
keep = set()
for l in open(f"{W}/stage4.kept.jsonl"):
    if l.strip():
        keep.add(json.loads(l)["clip_id"])
with open(f"{W}/upload.txt", "w") as f:
    for c in sorted(keep):
        p = f"{W}/frames/{c}.tar"
        if os.path.isfile(p):
            f.write(p + "\n")
print(f"  kept tars to upload: {len(keep)}")
PY
  gcloud storage cp -I "$FILT/frames/shard_$sfx/" < "$W/upload.txt" >/dev/null 2>&1 || { log "TAR UPLOAD FAIL"; exit 1; }
  nrem=$(gcloud storage ls "$FILT/frames/shard_$sfx/*.tar" 2>/dev/null | wc -l)
  nwant=$(grep -c . "$W/upload.txt")
  [ "$nrem" -ge "$nwant" ] || { log "FATAL: remote tars $nrem < $nwant"; exit 1; }
fi
ok=1
gcloud storage cp "$W/manifest.jsonl" "$FILT/native/_shards/shard_$sfx$TAG.manifest.jsonl" || ok=0
[ -f "$W/stage1.kept.jsonl" ] && { gcloud storage cp "$W/stage1.kept.jsonl" "$FILT/native/_shards/shard_$sfx$TAG.stage1.kept.jsonl" || ok=0; }
gcloud storage cp "$W"/*.json "$FILT/native/_shards/reports/shard_$sfx/" 2>/dev/null || true
[ "$ok" -ne 1 ] && { log "PRE-MARKER UPLOAD FAILED"; exit 1; }
gcloud storage cp "$W/stage4.kept.jsonl" "$FILT/native/_shards/shard_$sfx$TAG.stage4.kept.jsonl" || { log "MARKER UPLOAD FAIL"; exit 1; }
rm -rf "$W"
log "SHARD DONE @ $(date +%F_%H:%M:%S)"
