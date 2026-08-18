#!/usr/bin/env python
"""4-way SIDE-BY-SIDE overlay MP4 on the GT video: 4 panels (GT | recon | recon+AnyCalib | EgoForce),
one method per panel, 6 points/hand (wrist+fingertips) + wrist->tip skeleton, per-clip PA-MPJPE label."""
import sys, io, os, glob, base64, tarfile, numpy as np, cv2, pickle
from pathlib import Path
from PIL import Image
sys.path.insert(0,"/root/egosmith/src"); sys.path.insert(0,"/root/egosmith/scripts/inspection")
from recon_vs_gt_accuracy import (_read_egodex_lowdim, LD_LWRIST, LD_RWRIST, LD_LTIPS, LD_RTIPS,
    LD_EXTR, LD_INTR, FINGERTIP_INDICES, _world_joints, umeyama, _tocam_points)
from lib.pipeline.exporters.mano_features import build_mano_models
SC="/tmp/claude-0/-root-egosmith/6b57c32e-2fab-4780-bf21-8479371681d9/scratchpad"; FR="/root/egodex/frames"; dev="cuda"
IDX=[0]+list(FINGERTIP_INDICES)
mano_r,mano_l=build_mano_models(dev)
M=pickle.load(open(f"{SC}/fourway_metrics.pkl","rb"))["rows"]
byclip={r["clip"]:r for r in M}
clips=[c.strip() for c in open(f"{SC}/egodex_4way_clips.txt") if c.strip()][:3]
PAN=[("GT","gt",(60,200,90)),("recon (W/2)","recon",(230,70,60)),("recon+AnyCalib","anycalib",(70,120,240)),("EgoForce","egoforce",(245,165,40))]
PH=300  # panel height

def projpts(P,extr,fx,fy,cx,cy):
    Pc=_tocam_points(P,extr); o=[]
    for x,y,z in Pc: o.append((fx*x/z+cx,fy*y/z+cy) if z>1e-3 else (np.nan,np.nan))
    return o
def draw_hand(im,pts,col):
    ip={i:(int(u),int(v)) for i,(u,v) in enumerate(pts) if np.isfinite(u) and 0<=u<im.shape[1] and 0<=v<im.shape[0]}
    for t in range(1,6):
        if 0 in ip and t in ip: cv2.line(im,ip[0],ip[t],col,2,cv2.LINE_AA)
    for p in ip.values(): cv2.circle(im,p,5,col,-1,cv2.LINE_AA)

def save_mp4(frames_bgr,path,fps=3):
    h,w=frames_bgr[0].shape[:2]; w-=w%2; h-=h%2
    vw=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    for f in frames_bgr: vw.write(f[:h,:w])
    vw.release()
    import subprocess as _sp
    _h=path.replace(".mp4","_h264.mp4")
    _sp.run(["ffmpeg","-y","-loglevel","error","-i",path,"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",_h],check=True)
    return "data:video/mp4;base64,"+base64.b64encode(open(_h,"rb").read()).decode()

cards=[]
for clip in clips:
    tar=f"{FR}/{clip}.tar"; ld=_read_egodex_lowdim(tar,clip); T=ld.shape[0]
    gL=np.concatenate([ld[:,LD_LWRIST][:,None],ld[:,LD_LTIPS].reshape(T,5,3)],1)
    gR=np.concatenate([ld[:,LD_RWRIST][:,None],ld[:,LD_RTIPS].reshape(T,5,3)],1)
    extr=ld[:,LD_EXTR].reshape(T,4,4); intr=ld[:,LD_INTR]
    def rmap(root):
        seq=f"{root}/{clip}"
        if not os.path.exists(f"{seq}/world_space_res.pth"): return None
        try:
            L,R=_world_joints(seq,mano_l,mano_r,dev); L=L[:T,IDX];R=R[:T,IDX]
            s=np.concatenate([L.reshape(-1,3),R.reshape(-1,3)]);d=np.concatenate([gL.reshape(-1,3),gR.reshape(-1,3)])
            m=np.isfinite(s).all(1)&np.isfinite(d).all(1); sc,rR,rt=umeyama(s[m],d[m])
            return (sc*(L@rR.T)+rt, sc*(R@rR.T)+rt)
        except Exception: return None
    RC=rmap("/root/egodex_4way/recon"); AC=rmap("/root/egodex_4way/recon_anycalib")
    ef=f"/root/egodex_4way/egoforce/{clip}.npz"; EF=np.load(ef) if os.path.exists(ef) else None
    mrow=byclip.get(clip,{});
    with tarfile.open(tar) as tf:
        names=sorted(x for x in tf.getnames() if x.endswith(".image.jpg"))
        idxs=np.linspace(0,len(names)-1,min(12,len(names))).astype(int); outf=[]
        for fi in idxs:
            base=np.array(Image.open(io.BytesIO(tf.extractfile(names[fi]).read())).convert("RGB"))
            fx,fy,cx,cy=intr[fi]; panels=[]
            for label,key,col in PAN:
                im=base.copy()
                if key=="gt":
                    for J in (gL[fi],gR[fi]): draw_hand(im,projpts(J,extr[fi],fx,fy,cx,cy),col)
                elif key in ("recon","anycalib"):
                    pk=RC if key=="recon" else AC
                    if pk is not None:
                        for J in (pk[0][fi],pk[1][fi]): draw_hand(im,projpts(J,extr[fi],fx,fy,cx,cy),col)
                elif key=="egoforce" and EF is not None and fi<EF["j3d"].shape[0]:
                    vis=EF["visible"][fi]; j2=EF["j2d"][fi]
                    for hnd in range(2):
                        if vis[hnd]: draw_hand(im,[(j2[hnd][i][0],j2[hnd][i][1]) for i in IDX],col)
                pw=int(PH*im.shape[1]/im.shape[0]); im=cv2.resize(im,(pw,PH))
                bar=np.full((30,pw,3),22,np.uint8)
                mv=mrow.get(key); mtxt=f"  {mv:.0f}mm" if isinstance(mv,float) else ("  (reference)" if key=="gt" else "")
                cv2.putText(bar,label+mtxt,(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,col[::-1] if False else (col[0],col[1],col[2]),1,cv2.LINE_AA)
                panels.append(cv2.cvtColor(np.vstack([bar,im]),cv2.COLOR_RGB2BGR))
            sep=np.full((panels[0].shape[0],3,3),40,np.uint8)
            row=panels[0]
            for p in panels[1:]: row=np.hstack([row,sep,p])
            outf.append(row)
    p=f"{SC}/fourway_sbs_{clip}.mp4"; uri=save_mp4(outf,p)
    cards.append(dict(clip=clip,uri=uri)); print("4way-sbs",clip[:40],os.path.getsize(p)//1024,"KB")
pickle.dump(cards,open(f"{SC}/fourway_overlay.pkl","wb")); print("DONE")
