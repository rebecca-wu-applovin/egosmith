#!/usr/bin/env bash
# pod_entry_egocentric_recon.sh — Phase C: full reconstruction over Egocentric-100K sub-clips.
#
# Sharding is 1:1 with Phase B (3000): shard i consumes
#   <MANIFESTS_GCS>/shard_XXXXX.manifest.jsonl   (+ frame tars under <FRAMES_GCS>/shard_XXXXX/)
# and pushes each successful seq_folder to <OUT_GCS>/shard_XXXXX/<sub_clip>/.
#
# Differences vs the generic pod_entry.sh, from the fleet-utilization audit:
#   * CHUNK_CLIPS=64 (was 16/24): stage-major waves amortize the p90 straggler over
#     8 whole GPU rounds instead of 2 ragged ones.
#   * NO per-clip `gcloud ls` / `cp` loops: one wildcard listing builds the done-set,
#     one bulk `cp -I` pulls all frame tars, results upload with xargs -P.
#   * AMP left at the code default (ON), matching every prior fleet run and the measured
#     86.8s/interval baseline; override with HAWOR_ANY4D_USE_AMP=0 if ever needed.
#   * per-clip est_focal.txt written from extra.pinhole_focal (per-worker fisheye focal,
#     138-213 — a 30% focal error broke SLAM on 3/5 TACO clips, so this is load-bearing).
# Per-clip reconstruction failures are expected attrition (Layer-C filters them); they
# never fail the shard. The shard done-marker means "every to-run clip was attempted".
set -o pipefail
export PYTHONUNBUFFERED=1 NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
echo "[ego_recon] START shard=${SHARD:-${JOB_COMPLETION_INDEX:-?}} @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then source /opt/conda/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then source /root/miniconda3/etc/profile.d/conda.sh; fi
conda activate egosmith 2>/dev/null || true
# offline-proof Any4D's DINOv2 load: torch.hub probes github.com even when cached;
# a transient egress blip otherwise kills every slam worker in the wave.
gcloud storage cp gs://foundational-research/hoi-dataset/egosmith_recon/fleet/patch_dinov2.py /tmp/patch_dinov2.py 2>/dev/null \
  && python /tmp/patch_dinov2.py "$(python -c 'import uniception.models.encoders.dinov2 as m; print(m.__file__)')" || true

SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
case "$SHARD" in ''|*[!0-9]*) echo "[ego_recon] FATAL: SHARD not numeric ('$SHARD')"; exit 1;; esac
NSHARDS="${NSHARDS:-3000}"
CHUNK_CLIPS="${CHUNK_CLIPS:-64}"
ANY4D_BATCH_SIZE="${ANY4D_BATCH_SIZE:-8}"
MAX_CLIPS="${MAX_CLIPS:-0}"          # >0 = smoke cap
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/ego_recon}"; mkdir -p "$LOCAL_ROOT"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
GCS_WEIGHTS="${GCS_WEIGHTS:-gs://foundational-research/hoi-dataset/egosmith_filtered/weights}"
FRAMES_GCS="${FRAMES_GCS:-gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/frames}"
MANIFESTS_GCS="${MANIFESTS_GCS:-gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/phaseB/_shards}"
OUT_GCS="${OUT_GCS:-gs://foundational-research/hoi-dataset/egosmith_recon/egocentric100k/recon/outputs}"
LOGS_GCS="${LOGS_GCS:-gs://foundational-research/hoi-dataset/egosmith_recon/_logs/egocentric100k}"
sfx=$(printf '%05d' "$SHARD")
log(){ echo "[ego_recon][shard $SHARD/$NSHARDS] $*"; }

export HAWOR_STAGE3_TMP_ROOT="${HAWOR_STAGE3_TMP_ROOT:-$LOCAL_ROOT/hawor_tmp}"
export HAWOR_BATCH_TMPDIR="${HAWOR_BATCH_TMPDIR:-$LOCAL_ROOT/hawor_batch_tmp}"
mkdir -p "$HAWOR_STAGE3_TMP_ROOT" "$HAWOR_BATCH_TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# AMP: code default (ON) — same as all prior fleet runs and the timing baseline.
[ -n "${HAWOR_ANY4D_USE_AMP:-}" ] && export HAWOR_ANY4D_USE_AMP

# overlay current code (image may be an older commit)
gcloud storage cp "$FLEET/egosmith_code.tar.gz" /tmp/code.tar.gz 2>/dev/null && tar xzf /tmp/code.tar.gz -C "$REPO_ROOT" && log "overlaid current scripts+src"

# skip-if-done (whole shard attempted)
if gcloud storage ls "$OUT_GCS/_done/shard_$sfx.json" >/dev/null 2>&1; then log "shard already done, skip"; exit 0; fi

# ── 1. checkpoints once (skip if present) ──
pull_ckpt(){ local name="$1" dest="$2"
  local want; want=$(gcloud storage ls -l "$GCS_WEIGHTS/$name" 2>/dev/null | awk 'NR==1{print $1}')
  local have=0; [ -f "$dest" ] && have=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  if [ -n "$want" ] && [ "$have" = "$want" ]; then log "ckpt present+verified: $dest"; return; fi
  mkdir -p "$(dirname "$dest")"; rm -f "$dest"
  gcloud storage cp "$GCS_WEIGHTS/$name" "$dest" || { log "CKPT PULL FAIL $name"; exit 1; }
  have=$(stat -c%s "$dest" 2>/dev/null || echo 0)
  # a truncated ckpt makes EVERY slam worker die silently — fail loud here instead
  [ -z "$want" ] || [ "$have" = "$want" ] || { log "FATAL: ckpt $name size $have != remote $want"; exit 1; }; }
pull_ckpt hawor.ckpt             "$REPO_ROOT/weights/hawor/checkpoints/hawor.ckpt"
pull_ckpt infiller.pt            "$REPO_ROOT/weights/hawor/checkpoints/infiller.pt"
pull_ckpt any4d_4v_combined.pth  "$REPO_ROOT/thirdparty/Any4D/checkpoints/any4d_4v_combined.pth"

# ── 2. shard manifest + done-set (ONE listing, not per-clip ls) ──
W="$LOCAL_ROOT/$sfx"; rm -rf "$W"; mkdir -p "$W/frames" "$W/outputs"
trap 'echo "[ego_recon] EXIT disk state:"; df -h /scratch 2>/dev/null | tail -1; du -sh "$W"/* 2>/dev/null | sort -rh | head -5' EXIT
if ! gcloud storage cp "$MANIFESTS_GCS/shard_$sfx.manifest.jsonl" "$W/manifest.raw.jsonl" 2>/dev/null; then
  # distinguish "empty B shard" from "B not finished"
  gcloud storage ls "$MANIFESTS_GCS/shard_$sfx.report.json" >/dev/null 2>&1 \
    && { log "B shard empty -> nothing to do"; echo '{"shard":'"$SHARD"',"torun":0,"succeeded":0}' | gcloud storage cp - "$OUT_GCS/_done/shard_$sfx.json"; exit 0; } \
    || { log "FATAL: Phase B manifest missing (B incomplete?)"; exit 1; }
fi
gcloud storage ls "$OUT_GCS/shard_$sfx/*/world_space_res.pth" 2>/dev/null | awk -F/ '{print $(NF-1)}' > "$W/done.txt" || true
log "already done on GCS: $(grep -c . "$W/done.txt" || true)"

# rewrite descriptors to local paths, write per-clip est_focal.txt, emit tar pull-list
python3 - "$W" "$FRAMES_GCS/shard_$sfx" "$MAX_CLIPS" <<'PY'
import json, os, sys
W, tars_gcs, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
done = {l.strip() for l in open(f"{W}/done.txt") if l.strip()}
out = open(f"{W}/torun.jsonl", "w"); pull = open(f"{W}/pull.txt", "w")
n = skip = 0
for line in open(f"{W}/manifest.raw.jsonl"):
    if not line.strip(): continue
    r = json.loads(line); d = r["descriptor"]
    if r["clip_id"] in done: skip += 1; continue
    if cap and n >= cap: break
    base = os.path.basename(d["shard_path"])
    d["root_dir"] = f"{W}/frames"; d["shard_path"] = f"{W}/frames/{base}"
    seq = f"{W}/outputs/{r['clip_id']}"; os.makedirs(seq, exist_ok=True); d["seq_folder"] = seq
    focal = float(d["extra"]["pinhole_focal"])
    open(f"{seq}/est_focal.txt", "w").write(f"{focal:.6f}\n")
    out.write(json.dumps(r) + "\n"); pull.write(f"{tars_gcs}/{base}\n"); n += 1
print(f"  torun={n} skipped_done={skip}")
open(f"{W}/skipped_count", "w").write(str(skip))
PY
N_RUN=$(grep -c . "$W/torun.jsonl" || true)
if [ "$N_RUN" -eq 0 ]; then
  log "all clips already done"
  # never overwrite an existing marker (pod restarts land here and would clobber real counts)
  gcloud storage ls "$OUT_GCS/_done/shard_$sfx.json" >/dev/null 2>&1 \
    || echo '{"shard":'"$SHARD"',"torun":0,"succeeded":0,"note":"all done prior"}' | gcloud storage cp - "$OUT_GCS/_done/shard_$sfx.json"
  exit 0
fi

# ── 3. bulk parallel tar pull (ONE cp -I, gcloud parallelizes internally) ──
log "pulling $N_RUN frame tars..."
gcloud storage cp -I "$W/frames/" < "$W/pull.txt" >/dev/null 2>&1 || true
N_TARS=$(ls "$W/frames" | wc -l)
log "local tars: $N_TARS / $N_RUN"
[ "$N_TARS" -gt 0 ] || { log "FATAL: no frame tars pulled"; exit 1; }
# drop manifest rows whose tar failed to pull (rare; K8s retry re-attempts them)
python3 - "$W" <<'PY'
import json, os, sys
W = sys.argv[1]
rows = [json.loads(l) for l in open(f"{W}/torun.jsonl") if l.strip()]
keep = [r for r in rows if os.path.exists(r["descriptor"]["shard_path"])]
with open(f"{W}/torun.jsonl", "w") as f:
    for r in keep: f.write(json.dumps(r) + "\n")
print(f"  runnable={len(keep)} missing_tar={len(rows)-len(keep)}")
PY

# ── 4. per-GPU independent batch_infer (NO cross-GPU wave barriers) ──
# Measured: 1 GPU alone hits 94% util; 8 GPUs under one wave scheduler only 42-48%,
# because every stage-wave waits for its slowest clip. Running 8 single-GPU
# batch_infer processes removes the barrier class entirely. PER_GPU_CHUNK bounds
# per-worker memory accumulation (workers respawn each chunk), replacing CHUNK_CLIPS.
NGPU=$(nvidia-smi -L | wc -l)
PER_GPU_CHUNK="${PER_GPU_CHUNK:-16}"
DEPTH_FLAG="--no-depth_predict_all_frames"   # A/B: accuracy-neutral (dATE -0.04mm), 39% faster
[ "${DEPTH_ALL:-0}" = "1" ] && DEPTH_FLAG=""
for g in $(seq 0 $((NGPU-1))); do awk -v n="$NGPU" -v g="$g" '((NR-1)%n)==g' "$W/torun.jsonl" > "$W/torun_g$g.jsonl"; done

run_gpu(){  # one GPU's slice: chunked batch_infer + per-chunk upload + disk cleanup
  local g=$1 ci=0 rc_g=0
  split -l "$PER_GPU_CHUNK" -d -a 4 "$W/torun_g$g.jsonl" "$W/chunk_g${g}_"
  for ch in "$W"/chunk_g${g}_*; do
    [ -s "$ch" ] || continue
    ci=$((ci+1))
    local rd="$W/runs/g${g}_c${ci}"
    HAWOR_STAGE3_TMP_ROOT="$W/hawor_tmp_g$g" HAWOR_BATCH_TMPDIR="$W/hawor_btmp_g$g" \
    python scripts/batch_infer.py \
      --descriptor_manifest "$ch" \
      --gpus "$g" \
      --run_dir "$rd" \
      --stages detect_track,motion,slam,infiller \
      --keep_intermediates all \
      --any4d_batch_size "$ANY4D_BATCH_SIZE" \
      $DEPTH_FLAG \
      --resume
    local rc=$?; [ $rc -ne 0 ] && rc_g=$rc
    [ -s "$rd/events.jsonl" ] && gcloud storage cp "$rd/events.jsonl" "$LOGS_GCS/shard_${sfx}_g${g}_c${ci}.events.jsonl" >/dev/null 2>&1
    python3 - "$W" "$ch" "$g" <<'PY'
import json, os, sys
W, ch, g = sys.argv[1], sys.argv[2], sys.argv[3]
ok, tars = [], []
for line in open(ch):
    if not line.strip(): continue
    r = json.loads(line)
    tars.append(r["descriptor"]["shard_path"])
    res = f"{W}/outputs/{r['clip_id']}/world_space_res.pth"
    if os.path.exists(res) and os.path.getsize(res) > 0:
        ok.append(r["clip_id"])
open(f"{W}/chunk_ok_g{g}.txt", "w").write("".join(c + "\n" for c in ok))
open(f"{W}/chunk_tars_g{g}.txt", "w").write("".join(t + "\n" for t in tars))
PY
    xargs -r -P 4 -I{} -a "$W/chunk_ok_g$g.txt" bash -c \
      'gcloud storage rsync -r "'"$W"'/outputs/{}" "'"$OUT_GCS"'/shard_'"$sfx"'/{}" >/dev/null 2>&1 && rm -rf "'"$W"'/outputs/{}"'
    xargs -r -a "$W/chunk_tars_g$g.txt" rm -f
    # failed clips keep nothing: their partial outputs are pure disk leak (eviction killer)
    python3 -c "import json,sys,shutil
for l in open(sys.argv[1]):
    if l.strip(): shutil.rmtree(json.loads(l)['descriptor']['seq_folder'], ignore_errors=True)" "$ch" 2>/dev/null
    rm -rf "$W/hawor_tmp_g$g" "$W/hawor_btmp_g$g" "$rd"
    cat "$W/chunk_ok_g$g.txt" >> "$W/ok_all.txt"
    echo "[gpu$g] chunk $ci: $(grep -c . "$W/chunk_ok_g$g.txt" || true) uploaded"
  done
  return $rc_g
}

BI_RC=0
set +e
pids=()
for g in $(seq 0 $((NGPU-1))); do
  run_gpu "$g" > "$W/gpu$g.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || BI_RC=1; done
set -e
N_OK=$(sort -u "$W/ok_all.txt" 2>/dev/null | grep -c . || true)
tail -n 2 "$W"/gpu*.log 2>/dev/null || true

# ── 5. shard done marker (LAST; means every to-run clip was attempted) ──
N_TRY=$(grep -c . "$W/torun.jsonl" || true)
log "shard attempted=$N_TRY succeeded=$N_OK (rc=$BI_RC; failures are expected attrition)"
# 0-for-N on a FRESH shard is a bad node / corrupt ckpt -> retry. But if thousands of
# this shard's clips already succeeded (skipped_done>0), the few remaining are genuine
# hard-failure attrition — write the marker or the pod retries the same dead clips forever.
N_SKIPPED=$(cat "$W/skipped_count" 2>/dev/null || echo 0)
if [ "$N_OK" -eq 0 ] && [ "$N_TRY" -gt 0 ] && [ "$N_SKIPPED" -eq 0 ]; then
  log "FATAL: 0/$N_TRY succeeded on fresh shard — systematic failure, retrying (no marker)"
  exit 1
fi
echo '{"shard":'"$SHARD"',"torun":'"$N_TRY"',"succeeded":'"$N_OK"',"batch_rc":'"$BI_RC"'}' \
  | gcloud storage cp - "$OUT_GCS/_done/shard_$sfx.json"
rm -rf "$W"
log "SHARD DONE @ $(date +%F_%H:%M:%S)"
exit 0
