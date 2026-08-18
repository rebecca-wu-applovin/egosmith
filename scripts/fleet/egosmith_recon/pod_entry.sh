#!/usr/bin/env bash
# pod_entry.sh — per-pod entrypoint for the EgoSmith reconstruction L4 fleet.
#
# Each pod takes a stride-slice of the global manifest, pulls its clips' frame tars
# (and, for --use_gt, their GT) from GCS, then runs ONE batch_infer over the slice so
# each stage's model loads once per GPU worker and streams many clips. Produced
# seq_folders are pushed to GCS. Idempotent: clips already on GCS_OUT are skipped
# (cross-node resume), so a job re-apply resumes cleanly.
#
# Required env:
#   MODE           recon | use_gt
#   NSHARDS        total pod count (== K8s completions)
#   MANIFEST_GCS   gs://.../<mode>.manifest.jsonl
#   FLEETMAP_GCS   gs://.../<mode>.fleetmap.tsv
#   GCS_WEIGHTS    gs://.../egosmith_filtered/weights   (hawor.ckpt/infiller.pt/any4d_*)
# Optional:
#   SHARD          default $JOB_COMPLETION_INDEX (K8s Indexed Job), else 0
#   LOCAL_ROOT     default /scratch/egosmith_recon   (node-local frames + fresh outputs)
#   GPUS           default all visible from nvidia-smi
set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/repo}"
cd "$REPO_ROOT"
source /opt/conda/etc/profile.d/conda.sh
conda activate egosmith

: "${MODE:?set MODE=recon|use_gt}"
: "${NSHARDS:?set NSHARDS}"
: "${MANIFEST_GCS:?set MANIFEST_GCS}"
: "${FLEETMAP_GCS:?set FLEETMAP_GCS}"
: "${GCS_WEIGHTS:?set GCS_WEIGHTS}"
SHARD="${SHARD:-${JOB_COMPLETION_INDEX:-0}}"
LOCAL_ROOT="${LOCAL_ROOT:-/scratch/egosmith_recon}"
USE_GT_FLAG=""; [[ "$MODE" == "use_gt" ]] && USE_GT_FLAG="--use_gt"

WORK="$LOCAL_ROOT/_work"; mkdir -p "$WORK"
export HAWOR_STAGE3_TMP_ROOT="${HAWOR_STAGE3_TMP_ROOT:-$LOCAL_ROOT/hawor_tmp}"
export HAWOR_BATCH_TMPDIR="${HAWOR_BATCH_TMPDIR:-$LOCAL_ROOT/hawor_batch_tmp}"
mkdir -p "$HAWOR_STAGE3_TMP_ROOT" "$HAWOR_BATCH_TMPDIR"
# L4 has 22 GB (vs H100 80 GB); reduce Any4D depth peak memory + fragmentation so
# larger/square clips (e.g. hot3d 1408x1408) don't OOM at slam. Env-tunable.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ANY4D_BATCH_SIZE="${ANY4D_BATCH_SIZE:-8}"
# AMP (fp16 depth) fits memory but degrades depth/scale (MPJPE 49->144mm on TACO), so
# it's OFF by default — the smaller Any4D batch + expandable segments (both accuracy-
# neutral) are relied on instead. Set ANY4D_AMP=1 only as a last-resort OOM fallback.
AMP_FLAG=""; [[ "${ANY4D_AMP:-0}" == "1" ]] && AMP_FLAG="--any4d_use_amp"
ANYCALIB_FLAG=""; [[ "${USE_ANYCALIB:-0}" == "1" ]] && ANYCALIB_FLAG="--use_anycalib"

log() { echo "[pod_entry][shard $SHARD/$NSHARDS][$MODE] $*"; }

# ── 1. pull the large checkpoints once (skip if present) into resolver paths ──
pull_ckpt() {  # <gcs_name> <dest>
  local dest="$2"
  if [[ -s "$dest" ]]; then log "ckpt present: $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  log "pull ckpt: $GCS_WEIGHTS/$1 -> $dest"
  gcloud storage cp "$GCS_WEIGHTS/$1" "$dest"
}
pull_ckpt hawor.ckpt              "$REPO_ROOT/weights/hawor/checkpoints/hawor.ckpt"
pull_ckpt infiller.pt            "$REPO_ROOT/weights/hawor/checkpoints/infiller.pt"
pull_ckpt any4d_4v_combined.pth  "$REPO_ROOT/thirdparty/Any4D/checkpoints/any4d_4v_combined.pth"
# AnyCalib weights into the torch-hub cache (pods may lack torch-hub internet).
if [[ "${USE_ANYCALIB:-0}" == "1" ]]; then
  pull_ckpt anycalib_gen.pt "/root/.cache/torch/hub/anycalib/anycalib_gen.pt"
fi

# ── 2. obtain manifest + fleetmap; align line-for-line, take this shard's stride ──
log "pull manifest + fleetmap"
gcloud storage cp "$MANIFEST_GCS" "$WORK/manifest.jsonl"
gcloud storage cp "$FLEETMAP_GCS" "$WORK/fleetmap.tsv"
# manifest line i (0-based) == fleetmap data line i (header stripped) — same gen order.
# paste them, keep stride rows, split back out.
paste <(tail -n +2 "$WORK/fleetmap.tsv") "$WORK/manifest.jsonl" \
  | awk -v s="$SHARD" -v n="$NSHARDS" '((NR-1)%n)==s' > "$WORK/shard.joined.tsv"
N_SHARD=$(grep -c . "$WORK/shard.joined.tsv" || true)
log "shard has $N_SHARD clips"
[[ "$N_SHARD" -gt 0 ]] || { log "nothing for this shard"; exit 0; }

# fleetmap cols: 1 dataset 2 mode 3 clip_id 4 gcs_frames_tar 5 local_frames_tar
#                6 gcs_gt_dir 7 local_seq_folder 8 gcs_out_dir ; col 9 = manifest json
: > "$WORK/shard.torun.jsonl"
N_RUN=0; N_SKIP=0
while IFS=$'\t' read -r ds mode clip gcs_tar local_tar gcs_gt local_seq gcs_out manifest_json; do
  # skip-if-done: final artifact already on GCS_OUT
  if gcloud storage ls "$gcs_out/world_space_res.pth" >/dev/null 2>&1 \
     || gcloud storage ls "$gcs_out/result.npz" >/dev/null 2>&1; then
    N_SKIP=$((N_SKIP+1)); continue
  fi
  # pull frame tar
  mkdir -p "$(dirname "$local_tar")" "$local_seq"
  [[ -s "$local_tar" ]] || gcloud storage cp "$gcs_tar" "$local_tar"
  # use_gt: stage GT into the fresh seq_folder so slam/infiller adopt it
  if [[ "$MODE" == "use_gt" && "$gcs_gt" != "-" ]]; then
    mkdir -p "$local_seq/SLAM"
    gcloud storage cp "$gcs_gt/SLAM/*.npz" "$local_seq/SLAM/" 2>/dev/null || \
      log "WARN: no GT SLAM npz for $clip"
    gcloud storage cp "$gcs_gt/world_space_res.pth" "$local_seq/" 2>/dev/null || \
      log "WARN: no GT world_space_res for $clip"
  fi
  printf '%s\n' "$manifest_json" >> "$WORK/shard.torun.jsonl"
  N_RUN=$((N_RUN+1))
done < "$WORK/shard.joined.tsv"
log "to-run=$N_RUN  already-done=$N_SKIP"
[[ "$N_RUN" -gt 0 ]] || { log "all clips already done"; exit 0; }

# ── 3. run ONE batch_infer over the shard (models load once per GPU worker) ──
# Per-clip reconstruction failures are EXPECTED attrition (hard clips) and are handled
# downstream by the Layer-C filter — they must NOT fail the whole shard/job. So we
# tolerate a non-zero batch_infer exit, upload whatever succeeded, and exit 0. Genuine
# infra failures (OOM, node death) kill the pod before this point → K8s still retries.
GPUS="${GPUS:-$(nvidia-smi -L | wc -l | awk '{for(i=0;i<$1;i++)printf (i? ","i : i)}')}"
# batch_infer workers load models once and stream ALL clips, so GPU/host memory
# accumulates per clip and the long-lived slam workers get OOM-killed after ~a dozen
# clips (silent SIGKILL, no traceback). Fix: run in CHUNK_CLIPS-sized batches so workers
# respawn fresh each chunk (bounded accumulation, matching the working small-smoke).
CHUNK_CLIPS="${CHUNK_CLIPS:-24}"
LOGS_GCS="gs://foundational-research/hoi-dataset/egosmith_recon/_logs/${MODE}"
split -l "$CHUNK_CLIPS" -d -a 4 "$WORK/shard.torun.jsonl" "$WORK/chunk_"
BI_RC=0; ci=0
set +e
for ch in "$WORK"/chunk_*; do
  ci=$((ci+1)); nc=$(grep -c . "$ch")
  log "batch_infer chunk $ci ($nc clips) on GPUs [$GPUS] $USE_GT_FLAG"
  python scripts/batch_infer.py \
    --descriptor_manifest "$ch" \
    --gpus "$GPUS" \
    --stages detect_track,motion,slam,infiller \
    --keep_intermediates all \
    $AMP_FLAG \
    --any4d_batch_size "$ANY4D_BATCH_SIZE" \
    $ANYCALIB_FLAG \
    --resume $USE_GT_FLAG
  rc=$?; [[ $rc -ne 0 ]] && BI_RC=$rc
  # upload this chunk's events.jsonl (append-safe per-chunk name) for diagnosis
  EV="$(ls -t /repo/batch_runs/*/events.jsonl 2>/dev/null | head -1 || true)"
  [[ -n "$EV" ]] && gcloud storage cp "$EV" "$LOGS_GCS/shard_${SHARD}_chunk_${ci}.events.jsonl" 2>/dev/null || true
done
set -e
log "batch_infer done (rc=$BI_RC over $ci chunks; per-clip failures expected; uploading successes)"

# ── 4. upload produced seq_folders to GCS ──
log "uploading results"
n_up=0; n_no=0
while IFS=$'\t' read -r ds mode clip gcs_tar local_tar gcs_gt local_seq gcs_out manifest_json; do
  if [[ -s "$local_seq/world_space_res.pth" || -s "$local_seq/result.npz" ]]; then
    gcloud storage rsync -r "$local_seq" "$gcs_out" && n_up=$((n_up+1))
  else
    n_no=$((n_no+1))
  fi
done < "$WORK/shard.joined.tsv"
log "uploaded=$n_up no_output=$n_no  DONE"
exit 0
