#!/usr/bin/env python
"""EgoDex 4-way keypoint comparison: GT vs recon(W/2) vs recon+AnyCalib vs EgoForce.
Metrics = 6-pt (wrist+fingertips) PA-MPJPE vs GT, per method, over the clip set.
Overlay = the 6 points per hand for each method projected onto the GT video."""
import sys, io, os, json, glob, base64, tarfile, numpy as np, cv2, pickle
from pathlib import Path
from PIL import Image
sys.path.insert(0,"/root/egosmith/src"); sys.path.insert(0,"/root/egosmith/scripts/inspection")
import recon_vs_gt_accuracy as A
from recon_vs_gt_accuracy import (_read_egodex_lowdim, LD_LWRIST, LD_RWRIST, LD_LTIPS, LD_RTIPS,
    LD_EXTR, LD_INTR, FINGERTIP_INDICES, _world_joints, umeyama, _tocam_points)
from lib.pipeline.exporters.mano_features import build_mano_models
SC="/tmp/claude-0/-root-egosmith/6b57c32e-2fab-4780-bf21-8479371681d9/scratchpad"
dev="cuda"; FR="/root/egodex/frames"
IDX=[0]+list(FINGERTIP_INDICES)  # wrist + 5 fingertips
COL={"GT":(60,200,90),"recon":(230,70,60),"anycalib":(70,120,240),"egoforce":(240,160,40)}
clips=[c.strip() for c in open(f"{SC}/egodex_4way_pass_clips.txt") if c.strip()]
mano_r,mano_l=build_mano_models(dev)

def gt_6pt(tar,clip):
    ld=_read_egodex_lowdim(tar,clip); T=ld.shape[0]
    Lw=ld[:,LD_LWRIST][:,None,:]; Lt=ld[:,LD_LTIPS].reshape(T,5,3); L=np.concatenate([Lw,Lt],1)  # (T,6,3)
    Rw=ld[:,LD_RWRIST][:,None,:]; Rt=ld[:,LD_RTIPS].reshape(T,5,3); R=np.concatenate([Rw,Rt],1)
    extr=ld[:,LD_EXTR].reshape(T,4,4); intr=ld[:,LD_INTR]  # (T,4),(fx,fy,cx,cy)
    return L,R,extr,intr,T

def recon_6pt(seq,T):
    L,R=_world_joints(seq,mano_l,mano_r,dev)  # (T,21,3) world
    return L[:T,IDX,:], R[:T,IDX,:]

def pa_mpjpe(src_LR, dst_LR):
    # src,dst: tuples (L(T,6,3),R). Procrustes over both hands all frames -> mm
    s=np.concatenate([src_LR[0].reshape(-1,3),src_LR[1].reshape(-1,3)],0)
    d=np.concatenate([dst_LR[0].reshape(-1,3),dst_LR[1].reshape(-1,3)],0)
    m=np.isfinite(s).all(1)&np.isfinite(d).all(1)
    if m.sum()<10: return None
    try: sc,sR,st=umeyama(s[m],d[m])
    except Exception: return None
    sp=(sc*(s@sR.T))+st
    return float(np.sqrt(((sp[m]-d[m])**2).sum(1)).mean()*1000)

rows=[]; overlay_clips=clips[:3]; ovcards=[]
for clip in clips:
    tar=f"{FR}/{clip}.tar"
    if not os.path.exists(tar): continue
    try: gL,gR,extr,intr,T=gt_6pt(tar,clip)
    except Exception as e: print("gt fail",clip,e); continue
    # GT in camera space (for egoforce comparison)
    gLc=np.stack([_tocam_points(gL[t],extr[t]) for t in range(T)]); gRc=np.stack([_tocam_points(gR[t],extr[t]) for t in range(T)])
    res={"clip":clip,"T":T}
    # recon(W/2) + anycalib: world 6-pt, PA vs GT world
    for m,root in (("recon","/root/egodex_4way_pass/recon"),("anycalib","/root/egodex_4way_pass/recon_anycalib")):
        seq=f"{root}/{clip}"
        if os.path.exists(f"{seq}/world_space_res.pth"):
            try:
                rL,rR=recon_6pt(seq,T)
                res[m]=pa_mpjpe((rL,rR),(gL[:T],gR[:T]))
            except Exception: res[m]=None
        else: res[m]=None
    # egoforce: cam-space 6-pt, PA vs GT-cam
    ef=f"/root/egodex_4way_pass/egoforce/{clip}.npz"
    if os.path.exists(ef):
        d=np.load(ef); j3=np.asarray(d["j3d"]); Te=min(T,j3.shape[0])
        eL=j3[:Te,0][:,IDX,:]; eR=j3[:Te,1][:,IDX,:]
        res["egoforce"]=pa_mpjpe((eL,eR),(gLc[:Te],gRc[:Te]))
    else: res["egoforce"]=None
    rows.append(res); print(clip[:40],{k:(round(v,1) if isinstance(v,float) else v) for k,v in res.items() if k in ("recon","anycalib","egoforce")})

# aggregate medians
def med(key):
    v=[r[key] for r in rows if isinstance(r.get(key),float)]
    return round(float(np.median(v)),1) if v else None
agg={m:med(m) for m in ("recon","anycalib","egoforce")}
agg["n"]=len(rows)
print("\n=== MEDIAN PA-MPJPE vs GT (mm) over",len(rows),"clips ===",agg)
pickle.dump({"rows":rows,"agg":agg},open(f"{SC}/fourway_metrics.pkl","wb"))
print("saved fourway_metrics.pkl")
