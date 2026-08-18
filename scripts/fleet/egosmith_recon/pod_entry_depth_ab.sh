#!/usr/bin/env bash
# pod_entry_depth_ab.sh — depth_predict_all_frames A/B on ONE L4 node (fleet conditions).
# 14 GT clips (5 taco + 9 hot3d), both arms on 8 GPUs with fleet flags; uploads
# world_space_res + SLAM npz per arm (for GT scoring off-node), events.jsonl, and the
# full batch_runs worker logs (so any worker crash carries its traceback out).
set -o pipefail
export PYTHONUNBUFFERED=1
echo "[depth_ab] START @ $(date +%F_%H:%M:%S)"
REPO_ROOT="${REPO_ROOT:-/repo}"; cd "$REPO_ROOT"
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source /root/miniconda3/etc/profile.d/conda.sh
conda activate egosmith 2>/dev/null || true
# offline-proof Any4D's DINOv2 load: torch.hub probes github.com even when cached;
# a transient egress blip otherwise kills every slam worker in the wave.
gcloud storage cp gs://foundational-research/hoi-dataset/egosmith_recon/fleet/patch_dinov2.py /tmp/patch_dinov2.py 2>/dev/null \
  && python /tmp/patch_dinov2.py "$(python -c 'import uniception.models.encoders.dinov2 as m; print(m.__file__)')" || true
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet
AB="$FLEET/depth_ab"
OUT=gs://foundational-research/hoi-dataset/egosmith_recon/egocentric100k/depth_ab
GCS_WEIGHTS=gs://foundational-research/hoi-dataset/egosmith_filtered/weights
W=/scratch/depth_ab; mkdir -p $W/frames
log(){ echo "[depth_ab] $*"; }

export HAWOR_STAGE3_TMP_ROOT=$W/hawor_tmp HAWOR_BATCH_TMPDIR=$W/hawor_batch_tmp
mkdir -p "$HAWOR_STAGE3_TMP_ROOT" "$HAWOR_BATCH_TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

gcloud storage cp "$FLEET/egosmith_code.tar.gz" /tmp/code.tar.gz && tar xzf /tmp/code.tar.gz -C "$REPO_ROOT" && log "code overlaid"
pull_ckpt(){ [ -s "$2" ] && return; mkdir -p "$(dirname "$2")"; gcloud storage cp "$GCS_WEIGHTS/$1" "$2" || exit 1; }
pull_ckpt hawor.ckpt             "$REPO_ROOT/weights/hawor/checkpoints/hawor.ckpt"
pull_ckpt infiller.pt            "$REPO_ROOT/weights/hawor/checkpoints/infiller.pt"
pull_ckpt any4d_4v_combined.pth  "$REPO_ROOT/thirdparty/Any4D/checkpoints/any4d_4v_combined.pth"

gcloud storage cp "$AB/ab_clips.json" $W/ab_clips.json || { log "SPEC PULL FAIL"; exit 1; }
python3 - "$W" <<'PY'
import json, subprocess, os
W = __import__("sys").argv[1]
spec = json.load(open(f"{W}/ab_clips.json"))
urls = [v["tar"] for v in spec.values()]
subprocess.run(["gcloud","storage","cp",*urls,f"{W}/frames/"],capture_output=True)
for arm in ("allframes","keyframes"):
    with open(f"{W}/manifest_{arm}.jsonl","w") as out:
        for c,v in spec.items():
            r=json.loads(json.dumps(v["rec"])); d=r["descriptor"]
            d["root_dir"]=f"{W}/frames"; d["shard_path"]=f"{W}/frames/{c}.tar"
            seq=f"{W}/{arm}/{v['ds']}/{c}"; os.makedirs(seq,exist_ok=True)
            open(f"{seq}/est_focal.txt","w").write(f"{v['focal']:.6f}\n")
            d["seq_folder"]=seq
            out.write(json.dumps(r)+"\n")
print("prepared", len(spec), "clips x 2 arms; tars:", len(os.listdir(f"{W}/frames")))
PY

GPUS="$(nvidia-smi -L | wc -l | awk '{for(i=0;i<$1;i++)printf (i? ","i : i)}')"
for arm in allframes keyframes; do
  EXTRA=""; [ "$arm" = "keyframes" ] && EXTRA="--no-depth_predict_all_frames"
  log "=== ARM $arm on GPUs [$GPUS] $EXTRA ==="
  rm -rf "$REPO_ROOT/batch_runs"
  t0=$SECONDS
  python scripts/batch_infer.py --descriptor_manifest $W/manifest_$arm.jsonl --gpus "$GPUS" \
    --stages detect_track,motion,slam,infiller --keep_intermediates all \
    --any4d_batch_size 8 --no-resume $EXTRA > $W/run_$arm.log 2>&1
  wall=$((SECONDS-t0))
  done_n=$(ls $W/$arm/*/*/world_space_res.pth 2>/dev/null | wc -l)
  log "ARM $arm wall=${wall}s completed=${done_n}/14"
  RUN=$(ls -td "$REPO_ROOT"/batch_runs/* 2>/dev/null | head -1)
  [ -n "$RUN" ] && gcloud storage cp "$RUN/events.jsonl" "$OUT/$arm/events.jsonl" 2>/dev/null
  [ -n "$RUN" ] && [ -d "$RUN/logs" ] && tar czf /tmp/logs_$arm.tgz -C "$RUN" logs && gcloud storage cp /tmp/logs_$arm.tgz "$OUT/$arm/worker_logs.tgz"
  gcloud storage cp $W/run_$arm.log "$OUT/$arm/" 2>/dev/null
  # upload scoring artifacts: world_space_res + SLAM npz + est_focal per clip
  (cd $W/$arm && find . -name world_space_res.pth -o -name "hawor_slam_w_scale_*.npz" -o -name est_focal.txt \
    | tar czf /tmp/res_$arm.tgz -T -) && gcloud storage cp /tmp/res_$arm.tgz "$OUT/$arm/results.tgz"
  echo "{\"arm\":\"$arm\",\"wall_sec\":$wall,\"completed\":$done_n,\"total\":14}" | gcloud storage cp - "$OUT/$arm/summary.json"
done
log "DONE @ $(date +%F_%H:%M:%S)"
