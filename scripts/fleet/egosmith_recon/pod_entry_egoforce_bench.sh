#!/usr/bin/env bash
# pod_entry_egoforce_bench.sh — EgoForce over the 331 Phase-C-smoke sub-clips on ONE L4 node,
# 8 processes (one per GPU), for a node-throughput number directly comparable to the Phase C
# smoke (503 sub-clips/h/node). Uploads per-clip npz + timing JSONLs + summary.
set -o pipefail
export PYTHONUNBUFFERED=1
echo "[ef_bench] START @ $(date +%F_%H:%M:%S)"
FLEET=gs://foundational-research/hoi-dataset/egosmith_recon/fleet/egoforce_bench
SB=gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/phaseB_smoke
OUT=gs://foundational-research/hoi-dataset/egosmith_recon/egocentric100k/egoforce_smoke
W=/scratch/ef; mkdir -p $W/frames $W/out $W/timing
log(){ echo "[ef_bench] $*"; }

# 1. env (plain tar of /root/miniconda3/envs/egoforce; extract to the SAME absolute path)
log "pulling env tarball..."
gcloud storage cp "$FLEET/egoforce_env.tar.gz" /scratch/env.tar.gz || { log "ENV PULL FAIL"; exit 1; }
mkdir -p /root/miniconda3/envs
tar -xzf /scratch/env.tar.gz -C /root/miniconda3/envs || { log "ENV EXTRACT FAIL"; exit 1; }
rm /scratch/env.tar.gz
EPY=/root/miniconda3/envs/egoforce/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/egoforce/lib:${LD_LIBRARY_PATH:-}
export PATH=/root/miniconda3/envs/egoforce/bin:$PATH
$EPY -c "import torch; print('[ef_bench] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || { log "ENV BROKEN"; exit 1; }

# 2. repo + weights + runner
log "pulling repo tarball..."
gcloud storage cp "$FLEET/egoforce_repo.tar.gz" /scratch/repo.tar.gz || { log "REPO PULL FAIL"; exit 1; }
tar -xzf /scratch/repo.tar.gz -C /root && rm /scratch/repo.tar.gz
gcloud storage cp "$FLEET/egoforce_infer_smoke.py" /root/egoforce/demo/ || { log "RUNNER PULL FAIL"; exit 1; }

# 3. the 331 smoke sub-clip tars (flattened)
log "pulling smoke frame tars..."
gcloud storage cp "$SB/frames/shard_00000/*.tar" "$SB/frames/shard_00001/*.tar" $W/frames/ >/dev/null 2>&1
N=$(ls $W/frames | wc -l); log "frame tars: $N"
[ "$N" -ge 300 ] || { log "FATAL: expected ~331 tars, got $N"; exit 1; }
ls $W/frames | sed 's/\.tar$//' > $W/clips_all.txt
for g in 0 1 2 3 4 5 6 7; do awk -v g=$g '((NR-1)%8)==g' $W/clips_all.txt > $W/clips_p$g.txt; done

# 4. run 8 procs, one per GPU (exact undistorted-pinhole intrinsics of factory001/worker001)
cd /root/egoforce/demo
T0=$SECONDS
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $EPY egoforce_infer_smoke.py \
    --clips-file $W/clips_p$g.txt --frames-dir $W/frames --out-dir $W/out \
    --timing-out $W/timing/gpu$g.jsonl \
    --fx 137.972 --fy 138.221 --cx 232.721 --cy 124.888 \
    > $W/timing/gpu$g.log 2>&1 &
done
wait
WALL=$((SECONDS-T0))
N_OK=$(ls $W/out/*.npz 2>/dev/null | wc -l)
log "8-GPU wall=${WALL}s  npz=${N_OK}/331  => $(( N_OK*3600/WALL )) clips/h/node"
tail -2 $W/timing/gpu0.log

# 5. upload
gcloud storage cp "$W/out/*.npz" "$OUT/outputs/" >/dev/null 2>&1
gcloud storage cp "$W/timing/*.jsonl" "$W/timing/*.log" "$OUT/timing/" >/dev/null 2>&1
echo "{\"wall_sec\":$WALL,\"ok\":$N_OK,\"total\":331,\"gpus\":8,\"clips_per_h_node\":$(( N_OK*3600/WALL ))}" \
  | gcloud storage cp - "$OUT/summary.json"
log "DONE @ $(date +%F_%H:%M:%S)"
