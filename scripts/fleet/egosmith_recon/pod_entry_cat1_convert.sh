#!/usr/bin/env bash
# pod_entry_cat1_convert.sh — Phase B for Cat-1 VIDEO-source datasets
# (assembly101 / hd_epic / epic_kitchens_100 / ego4d / holoassist).
#
# Shard i consumes  <VIDEOS_GCS>/shard_XXXXX.videos.jsonl  (reshard_stage1_survivors.py
# output: Stage-1 survivor rows + tar-member offset/size where needed) and produces
#   <FRAMES_GCS>/shard_XXXXX/<clip>_ivNN.tar
#   <MANIFESTS_GCS>/shard_XXXXX.manifest.jsonl
#   <MANIFESTS_GCS>/shard_XXXXX.report.json      (uploaded LAST = done marker)
#
# Converter is scripts/build/generate_video_wds.py (fleet standalone overlay — the
# shared egosmith_code.tar.gz is NOT touched): per-video subprocess isolation,
# video-only remux (GoPro telemetry pitfall), time-sampled 15fps, optional
# cv2.fisheye undistort from CAT1 intrinsics JSON (assembly101 HMC + HD-EPIC Aria).
# CPU work; NPROC converter processes split the shard round-robin by video.
set -o pipefail   # NOT -u: conda activation references unbound vars
export PYTHONUNBUFFERED=1 NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
echo "[cat1_convert] START shard=${SHARD:-${JOB_COMPLETION_INDEX:-?}} ds=${DS:-?} @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; fi
conda activate egosmith 2>/dev/null || true
python -c "import gcsfs" 2>/dev/null || { echo "[cat1_convert] installing gcsfs..."; python -m pip install -q gcsfs 2>&1 | tail -1; }

SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
case "$SHARD" in ''|*[!0-9]*) echo "[cat1_convert] FATAL: SHARD not numeric ('$SHARD')"; exit 1;; esac
DS="${DS:?set DS (dataset id, e.g. hd_epic)}"
NSHARDS="${NSHARDS:-1}"
NPROC="${NPROC:-4}"
MAX_CLIPS="${MAX_CLIPS:-0}"           # >0 = smoke cap (videos per shard)
TARGET_FPS="${TARGET_FPS:-15}"
OUT_WIDTH="${OUT_WIDTH:-456}"
ID_MODE="${ID_MODE:-basename}"        # assembly101: group_basename; holoassist: session
PER_VIDEO_TIMEOUT="${PER_VIDEO_TIMEOUT:-10800}"
REMUX_MAX_GB="${REMUX_MAX_GB:-12}"
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/cat1_convert}"; mkdir -p "$LOCAL_ROOT"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
FILT=gs://foundational-research/hoi-dataset/egosmith_filtered
VIDEOS_GCS="${VIDEOS_GCS:-$FILT/$DS/phaseB_input/_shards}"
FRAMES_GCS="${FRAMES_GCS:-$FILT/$DS/frames}"
MANIFESTS_GCS="${MANIFESTS_GCS:-$FILT/$DS/phaseB/_shards}"
sfx=$(printf '%05d' "$SHARD")
log(){ echo "[cat1_convert][$DS shard $SHARD/$NSHARDS] $*"; }

# overlay current code onto /repo, then the standalone converter + intrinsics (win over tarball)
gcloud storage cp "$FLEET/egosmith_code.tar.gz" /tmp/code.tar.gz 2>/dev/null && tar xzf /tmp/code.tar.gz -C "$REPO_ROOT" && log "overlaid current scripts+src"
gcloud storage cp "$FLEET/generate_video_wds.py" scripts/build/generate_video_wds.py || { log "FATAL: converter pull failed"; exit 1; }
gcloud storage cp "$FLEET/cat1_fisheye_intrinsics.kb4.json" /tmp/cat1_fe.json || { log "FATAL: intrinsics pull failed"; exit 1; }

# skip-if-done (report is the done marker, uploaded last)
if gcloud storage ls "$MANIFESTS_GCS/shard_$sfx.report.json" >/dev/null 2>&1; then log "shard already done, skip"; exit 0; fi

W="$LOCAL_ROOT/$sfx"; rm -rf "$W"; mkdir -p "$W/frames" "$W/outputs"
trap 'echo "[cat1_convert] EXIT disk state:"; df -h /scratch 2>/dev/null | tail -1' EXIT
gcloud storage cp "$VIDEOS_GCS/shard_$sfx.videos.jsonl" "$W/videos.jsonl" || { log "VIDEOS PULL FAIL"; exit 1; }
N_VID=$(grep -c . "$W/videos.jsonl" || true)
log "videos in shard: $N_VID"

# split round-robin (by bytes-descending so procs balance) into NPROC proc files
python3 - "$W" "$NPROC" "$MAX_CLIPS" <<'PY'
import json, sys
W, nproc, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(f"{W}/videos.jsonl") if l.strip()]
if cap: rows = rows[:cap]
rows.sort(key=lambda r: -(r.get("source_bytes") or r.get("size") or 0))
outs = [open(f"{W}/videos.p{i}.jsonl", "w") for i in range(nproc)]
for i, r in enumerate(rows):
    outs[i % nproc].write(json.dumps(r) + "\n")
for o in outs: o.close()
print(f"  split {len(rows)} videos into {nproc} proc files")
PY

pids=(); rc=0
for i in $(seq 0 $((NPROC-1))); do
  if [ -s "$W/videos.p$i.jsonl" ]; then
    PYTHONPATH=src python scripts/build/generate_video_wds.py \
      --survivors "$W/videos.p$i.jsonl" \
      --frames_root "$W/frames" --outputs_root "$W/outputs" \
      --manifest_out "$W/manifest.p$i.jsonl" --report_out "$W/report.p$i.json" \
      --source_id "$DS" --id_mode "$ID_MODE" \
      --target_fps "$TARGET_FPS" --out_width "$OUT_WIDTH" \
      --fisheye_intrinsics /tmp/cat1_fe.json \
      --per_video_timeout "$PER_VIDEO_TIMEOUT" \
      --remux_max_gb "$REMUX_MAX_GB" \
      --work_dir "$W/work.p$i" > "$W/convert.p$i.log" 2>&1 &
    pids+=($!)
  fi
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
tail -n 3 "$W"/convert.p*.log
[ $rc -ne 0 ] && { log "CONVERTER PROC FAILED"; exit 1; }
cat "$W"/manifest.p*.jsonl > "$W/manifest.jsonl" 2>/dev/null || : > "$W/manifest.jsonl"
python3 - "$W" <<'PY'
import json, glob, sys
W = sys.argv[1]
tot = {}
for f in glob.glob(f"{W}/report.p*.json"):
    for k, v in json.load(open(f)).items():
        if isinstance(v, (int, float)): tot[k] = round(tot.get(k, 0) + v, 2)
json.dump(tot, open(f"{W}/report.json", "w"), indent=1)
print("merged report:", tot)
PY

# ── verify local consistency: every manifest record has a tar on disk ──
N_SUB=$(grep -c . "$W/manifest.jsonl" || true)
N_TAR=$(ls "$W/frames" 2>/dev/null | wc -l)
log "subclips=$N_SUB local_tars=$N_TAR"
[ "$N_SUB" -eq "$N_TAR" ] || { log "FATAL: manifest/tar count mismatch"; exit 1; }
# 0-for-N guard: a shard of N videos yielding nothing AND >0 errors is a systemic
# failure -> retry. (0 subclips with 0 errors = legitimately all-short intervals.)
N_ERR=$(python3 -c "import json;print(json.load(open('$W/report.json')).get('errors',0))" 2>/dev/null || echo 0)
if [ "$N_SUB" -eq 0 ] && [ "${N_ERR%.*}" -gt 0 ]; then log "FATAL: 0 subclips with $N_ERR errors"; exit 1; fi

# ── upload: tars (bulk) -> verify remote count -> manifest -> report last ──
if [ "$N_SUB" -gt 0 ]; then
  log "uploading $N_TAR tars..."
  gcloud storage cp "$W/frames/*.tar" "$FRAMES_GCS/shard_$sfx/" >/dev/null 2>&1 || { log "TAR UPLOAD FAIL"; exit 1; }
  N_REMOTE=$(gcloud storage ls "$FRAMES_GCS/shard_$sfx/*.tar" 2>/dev/null | wc -l)
  log "remote tars: $N_REMOTE"
  [ "$N_REMOTE" -ge "$N_TAR" ] || { log "FATAL: remote tar count $N_REMOTE < $N_TAR"; exit 1; }
fi
gcloud storage cp "$W/manifest.jsonl" "$MANIFESTS_GCS/shard_$sfx.manifest.jsonl" >/dev/null 2>&1 || { log "MANIFEST UPLOAD FAIL"; exit 1; }
gcloud storage cp "$W/report.json"    "$MANIFESTS_GCS/shard_$sfx.report.json"    >/dev/null 2>&1 || { log "REPORT UPLOAD FAIL"; exit 1; }  # last = done marker
rm -rf "$W"
log "SHARD DONE @ $(date +%F_%H:%M:%S)"
