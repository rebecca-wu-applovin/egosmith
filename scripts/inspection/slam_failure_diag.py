#!/usr/bin/env python
"""Per-clip SLAM (DPVO) failure signature from dpvo_raw_*.npz — the ACTUAL reason each
degenerate recon failed. Read-only; no rerun.

Signals (see slam.py:255-263, dpvo_slam.py, est_scale_batch.py):
- n_kf                : DPVO keyframe count (traj.shape[0]); <8-15 => under-initialized
- transl_baseline     : ||span(traj[:,:3])|| in DPVO units (parallax proxy)
- scene_depth         : median(1/disps) over finite cells (DPVO units)
- baseline_over_depth : the decisive parallax number (baseline / scene_depth)
- disp_floor_frac     : fraction of disp cells at the 0.01 floor / <=1e-6 (empty-keyframe fill)
- nf_traj/nf_disps    : fraction non-finite in traj / disps
- scale, nf_scale     : est_scale value + whether it is NaN (from hawor_slam_w_scale)
- bucket              : the categorized reason
"""
import sys, glob, os, json, numpy as np

def diag(clip_dir):
    raw=glob.glob(f"{clip_dir}/SLAM/dpvo_raw_*.npz")
    if not raw: return None
    d=np.load(raw[0])
    traj=np.asarray(d["traj"]); disps=np.asarray(d["disps"]) if "disps" in d.files else None
    n_kf=int(traj.shape[0])
    t=traj[:,:3]
    fin_t=np.isfinite(t)
    span=(np.nanmax(np.where(fin_t,t,np.nan),0)-np.nanmin(np.where(fin_t,t,np.nan),0)) if fin_t.any() else np.array([np.nan]*3)
    baseline=float(np.linalg.norm(span)) if np.isfinite(span).all() else float('nan')
    nf_traj=float((~np.isfinite(traj)).mean())
    if disps is not None:
        fd=disps[np.isfinite(disps)]
        scene_depth=float(np.median(1.0/fd[fd>1e-6])) if (fd>1e-6).any() else float('nan')
        disp_floor_frac=float(((disps<=0.0101)&(disps>=0)).mean())
        nf_disps=float((~np.isfinite(disps)).mean())
    else: scene_depth=disp_floor_frac=nf_disps=float('nan')
    bod=baseline/scene_depth if (np.isfinite(baseline) and np.isfinite(scene_depth) and scene_depth>0) else float('nan')
    # scale from hawor_slam_w_scale
    sc=glob.glob(f"{clip_dir}/SLAM/hawor_slam_w_scale_*.npz"); scale=float('nan'); nf_scale=None
    if sc:
        sd=np.load(sc[0],allow_pickle=True)
        if "scale" in sd.files:
            scale=float(np.asarray(sd["scale"]).reshape(-1)[0]); nf_scale=not np.isfinite(scale)
    # bucket the reason
    if nf_traj>0:
        bucket="low-parallax BA divergence (traj non-finite)"
    elif nf_scale:
        bucket="scale-fit collapse (traj finite, scale NaN)"
    elif n_kf<15:
        bucket="under-initialized (few keyframes)"
    elif np.isfinite(bod) and bod<0.02:
        bucket="near-zero baseline (finite but ~static camera)"
    else:
        bucket="finite recon (check downstream)"
    return dict(clip=os.path.basename(clip_dir),n_kf=n_kf,baseline=round(baseline,4),
        scene_depth=round(scene_depth,3),baseline_over_depth=round(bod,4) if np.isfinite(bod) else None,
        disp_floor_frac=round(disp_floor_frac,3),nf_traj=round(nf_traj,3),nf_disps=round(nf_disps,3),
        scale=round(scale,3) if np.isfinite(scale) else None,nf_scale=nf_scale,bucket=bucket)

if __name__=="__main__":
    out={}
    for ds in ("taco","oakink_grasp","hot3d"):
        rows=[]
        for cd in sorted(glob.glob(f"/root/{ds}_faildiag/*")):
            r=diag(cd)
            if r: rows.append(r)
        out[ds]=rows
        print(f"\n===== {ds} ({len(rows)} degenerate clips) =====")
        for r in rows:
            print(f"  kf={r['n_kf']:3d}  baseline/depth={r['baseline_over_depth']}  disp_floor={r['disp_floor_frac']}  "
                  f"nf_traj={r['nf_traj']}  scale={r['scale']}  -> {r['bucket']}")
        from collections import Counter
        print("  buckets:",dict(Counter(r['bucket'] for r in rows)))
    # healthy contrast (local success clip with dpvo_raw)
    for hc in ["/tmp/claude-0/-root-egosmith/6b57c32e-2fab-4780-bf21-8479371681d9/scratchpad/l4recon/TACO_brush_brush_bowl_20231005_188"]:
        r=diag(hc)
        if r: print("\nHEALTHY contrast:",json.dumps(r))
    json.dump(out,open("/tmp/claude-0/-root-egosmith/6b57c32e-2fab-4780-bf21-8479371681d9/scratchpad/slam_faildiag.json","w"),indent=1)
    print("\nwrote slam_faildiag.json")
