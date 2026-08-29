#!/usr/bin/env bash
# pod_entry_egocentric_convert_v2.sh — Phase B: Layer-1 survivors -> 15fps undistorted
# sub-clip frame tars + per-shard clip manifest. v2 = v1 + OUT_WIDTH support for
# high-res sources (Egocentric-10K 1920x1080 -> validated 456x256 recon regime) via a
# standalone fleet copy of generate_egocentric_wds.py (the shared egosmith_code.tar.gz
# is NOT touched — the running 100K recon still pulls the original tarball).
#
# Sharding is 1:1 with the Stage-1 shards (3000): shard i consumes
#   <KEPT_GCS>/shard_XXXXX.kept.jsonl
# and produces
#   <FRAMES_GCS>/shard_XXXXX/<clip>_ivNN.tar          (one tar per valid interval)
#   <MANIFESTS_GCS>/shard_XXXXX.manifest.jsonl        (ClipManifestRecords, local paths;
#                                                      Phase C rewrites + reads extra.pinhole_focal)
#   <MANIFESTS_GCS>/shard_XXXXX.report.json           (uploaded LAST = done marker)
#
# CPU-only work (decode + fisheye remap + jpeg encode); NPROC converter processes split
# the shard by part_tar so each part streams exactly once.
set -o pipefail   # NOT -u: conda activation references unbound vars
export PYTHONUNBUFFERED=1 NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
echo "[ego_convert] START shard=${SHARD:-${JOB_COMPLETION_INDEX:-?}} @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; fi
conda activate egosmith 2>/dev/null || true
python -c "import gcsfs" 2>/dev/null || { echo "[ego_convert] installing gcsfs..."; python -m pip install -q gcsfs 2>&1 | tail -1; }

SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
case "$SHARD" in ''|*[!0-9]*) echo "[ego_convert] FATAL: SHARD not numeric ('$SHARD')"; exit 1;; esac
NSHARDS="${NSHARDS:-3000}"
NPROC="${NPROC:-4}"
MAX_CLIPS="${MAX_CLIPS:-0}"           # >0 = smoke cap
TARGET_FPS="${TARGET_FPS:-15}"
OUT_WIDTH="${OUT_WIDTH:-0}"           # >0 = downscale undistorted output to this width (10K: 456)
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/ego_convert}"; mkdir -p "$LOCAL_ROOT"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
KEPT_GCS="${KEPT_GCS:-gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/stage1/_shards}"
FRAMES_GCS="${FRAMES_GCS:-gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/frames}"
MANIFESTS_GCS="${MANIFESTS_GCS:-gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/phaseB/_shards}"
sfx=$(printf '%05d' "$SHARD")
log(){ echo "[ego_convert][shard $SHARD/$NSHARDS] $*"; }

# overlay current code onto /repo (image may be an older commit)
gcloud storage cp "$FLEET/egosmith_code.tar.gz" /tmp/code.tar.gz 2>/dev/null && tar xzf /tmp/code.tar.gz -C "$REPO_ROOT" && log "overlaid current scripts+src"
# v2: overlay the standalone out_width-capable converter LAST (wins over the tarball copy)
gcloud storage cp "$FLEET/generate_egocentric_wds.py" scripts/build/generate_egocentric_wds.py && log "overlaid standalone generate_egocentric_wds.py (out_width)" || { log "FATAL: standalone converter pull failed"; exit 1; }
ls scripts/build/generate_egocentric_wds.py >/dev/null 2>&1 || { log "FATAL: generate_egocentric_wds.py missing after overlay"; exit 1; }
python -c "import re,sys; s=open('scripts/build/generate_egocentric_wds.py').read(); sys.exit(0 if 'out_width' in s else 1)" || { log "FATAL: overlaid converter lacks out_width"; exit 1; }

# skip-if-done (report is the done marker, uploaded last)
if gcloud storage ls "$MANIFESTS_GCS/shard_$sfx.report.json" >/dev/null 2>&1; then log "shard already done, skip"; exit 0; fi

W="$LOCAL_ROOT/$sfx"; rm -rf "$W"; mkdir -p "$W/frames" "$W/outputs"
gcloud storage cp "$KEPT_GCS/shard_$sfx.kept.jsonl" "$W/kept.jsonl" || { log "KEPT PULL FAIL"; exit 1; }
N_KEPT=$(grep -c . "$W/kept.jsonl" || true)
log "kept clips in shard: $N_KEPT"
if [ "$N_KEPT" -eq 0 ]; then
  : > "$W/manifest.jsonl"
  echo '{"clips":0,"subclips":0,"frames":0,"skipped_short":0,"errors":0}' > "$W/report.json"
else
  # split by part_tar round-robin so each part streams exactly once, across NPROC procs
  python3 - "$W" "$NPROC" "$MAX_CLIPS" <<'PY'
import json, sys
W, nproc, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(f"{W}/kept.jsonl") if l.strip()]
if cap: rows = rows[:cap]
parts = {}
for r in rows: parts.setdefault(r["part_tar"], []).append(r)
outs = [open(f"{W}/kept.p{i}.jsonl", "w") for i in range(nproc)]
for i, (part, rs) in enumerate(sorted(parts.items())):
    for r in rs: outs[i % nproc].write(json.dumps(r) + "\n")
for o in outs: o.close()
print(f"  split {len(rows)} clips / {len(parts)} parts into {nproc} proc files")
PY
  pids=(); rc=0
  for i in $(seq 0 $((NPROC-1))); do
    if [ -s "$W/kept.p$i.jsonl" ]; then
      PYTHONPATH=src python scripts/build/generate_egocentric_wds.py \
        --survivors "$W/kept.p$i.jsonl" \
        --frames_root "$W/frames" --outputs_root "$W/outputs" \
        --manifest_out "$W/manifest.p$i.jsonl" --report_out "$W/report.p$i.json" \
        --target_fps "$TARGET_FPS" --out_width "$OUT_WIDTH" > "$W/convert.p$i.log" 2>&1 &
      pids+=($!)
    fi
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  tail -3 "$W"/convert.p*.log
  [ $rc -ne 0 ] && { log "CONVERTER PROC FAILED"; exit 1; }
  cat "$W"/manifest.p*.jsonl > "$W/manifest.jsonl"
  python3 - "$W" <<'PY'
import json, glob, sys
W = sys.argv[1]
tot = {}
for f in glob.glob(f"{W}/report.p*.json"):
    for k, v in json.load(open(f)).items():
        if isinstance(v, (int, float)): tot[k] = tot.get(k, 0) + v
json.dump(tot, open(f"{W}/report.json", "w"), indent=1)
print("merged report:", tot)
PY
fi

# ── verify local consistency: every manifest record has a tar on disk ──
N_SUB=$(grep -c . "$W/manifest.jsonl" || true)
N_TAR=$(ls "$W/frames" 2>/dev/null | wc -l)
log "subclips=$N_SUB local_tars=$N_TAR"
[ "$N_SUB" -eq "$N_TAR" ] || { log "FATAL: manifest/tar count mismatch"; exit 1; }

# ── upload: tars (bulk, parallel) -> verify remote count -> manifest -> report last ──
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
