#!/usr/bin/env python
"""4-way SIDE-BY-SIDE overlay MP4, DIRECT 3D projection (NO alignment). Each method's own 3D hand
keypoints projected with its own camera. recon/AnyCalib/EgoForce draw the FULL 21-joint hand
skeleton (high-contrast, white-outlined); GT draws its wrist+5 fingertips. Panel label = clip PA-MPJPE."""
import sys, io, os, glob, base64, tarfile, subprocess, numpy as np, cv2, pickle
from pathlib import Path
from PIL import Image
sys.path.insert(0,"/root/egosmith/src"); sys.path.insert(0,"/root/egosmith/scripts/inspection")
from recon_vs_gt_accuracy import (_read_egodex_lowdim, LD_LWRIST, LD_RWRIST, LD_LTIPS, LD_RTIPS,
    LD_EXTR, LD_INTR, FINGERTIP_INDICES, _world_joints, _tocam_points)
from taco_overlay_sheets import load_camera
from lib.pipeline.exporters.mano_features import build_mano_models
SC="/tmp/claude-0/-root-egosmith/6b57c32e-2fab-4780-bf21-8479371681d9/scratchpad"; FR="/root/egodex/frames"; dev="cuda"
GT6=[0]+list(FINGERTIP_INDICES)
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
mano_r,mano_l=build_mano_models(dev)
byclip={r["clip"]:r for r in pickle.load(open(f"{SC}/fourway_metrics.pkl","rb"))["rows"]}
clips=[c.strip() for c in open(f"{SC}/egodex_4way_pass_clips.txt") if c.strip()]
PAN=[("GT","gt",(60,205,90)),("recon (W/2)","recon",(235,70,60)),("recon+AnyCalib","anycalib",(70,130,245)),("EgoForce","egoforce",(248,170,45))]
PH=360

RAW="/root/datasets/egodex_dexprior_raw"
_TASKS=sorted(os.listdir(RAW)) if os.path.isdir(RAW) else []
def _gt_names(side):
    j=[f"{side}Hand"]
    for fg in ["Thumb","IndexFinger","MiddleFinger","RingFinger","LittleFinger"]:
        for suf in ["Knuckle","IntermediateBase","IntermediateTip","Tip"]: j.append(f"{side}{fg}{suf}")
    return j
def gt_full(clip):
    """Real EgoDex GT 21-joint skeleton (Vision Pro) from the raw HDF5, MANO order. None if unavailable."""
    import h5py
    rest=clip[len("egodex_ep"):] if clip.startswith("egodex_ep") else clip
    h5=None
    for t in sorted(_TASKS,key=len,reverse=True):
        if rest.startswith(t+"_"): h5=f"{RAW}/{t}/{rest[len(t)+1:]}.hdf5"; break
    if not h5 or not os.path.exists(h5): return None
    try:
        with h5py.File(h5,"r") as f:
            L=np.stack([f["transforms/"+n][:, :3,3] for n in _gt_names("left")],1).astype(np.float64)
            R=np.stack([f["transforms/"+n][:, :3,3] for n in _gt_names("right")],1).astype(np.float64)
        return L,R
    except Exception: return None

def projpts(P,extr,fx,fy,cx,cy):
    Pc=_tocam_points(P,extr); return [(fx*x/z+cx,fy*y/z+cy) if z>1e-3 else (np.nan,np.nan) for x,y,z in Pc]
def draw_skel(im,pts,col):
    ip={i:(int(u),int(v)) for i,(u,v) in enumerate(pts) if np.isfinite(u) and -30<=u<im.shape[1]+30 and -30<=v<im.shape[0]+30}
    for a,b in EDGES:
        if a in ip and b in ip:
            cv2.line(im,ip[a],ip[b],(255,255,255),4,cv2.LINE_AA); cv2.line(im,ip[a],ip[b],col,2,cv2.LINE_AA)
    for i,p in ip.items():
        r=6 if i==0 else 4
        cv2.circle(im,p,r+1,(255,255,255),-1,cv2.LINE_AA); cv2.circle(im,p,r,col,-1,cv2.LINE_AA)
def draw_gt(im,pts,col):
    # GT provides only wrist + 5 fingertips -> draw a hand outline: wrist->each tip, + big dots
    ip={i:(int(u),int(v)) for i,(u,v) in enumerate(pts) if np.isfinite(u) and 0<=u<im.shape[1] and 0<=v<im.shape[0]}
    for t in range(1,6):
        if 0 in ip and t in ip:
            cv2.line(im,ip[0],ip[t],(255,255,255),4,cv2.LINE_AA); cv2.line(im,ip[0],ip[t],col,2,cv2.LINE_AA)
    for i,p in ip.items():
        r=8 if i==0 else 6
        cv2.circle(im,p,r+1,(255,255,255),-1,cv2.LINE_AA); cv2.circle(im,p,r,col,-1,cv2.LINE_AA)

def save_mp4(frames_bgr,path,fps=3):
    h,w=frames_bgr[0].shape[:2]; w-=w%2; h-=h%2
    vw=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    for f in frames_bgr: vw.write(f[:h,:w])
    vw.release()
    _h=path.replace(".mp4","_h264.mp4")
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",path,"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",_h],check=True)
    return "data:video/mp4;base64,"+base64.b64encode(open(_h,"rb").read()).decode()

def load_method(root,clip):
    seq=f"{root}/{clip}"
    if not os.path.exists(f"{seq}/world_space_res.pth"): return None
    try:
        L,R=_world_joints(seq,mano_l,mano_r,dev)  # (Tr,21,3) recon world, FULL skeleton
        extr,intr=load_camera(Path(seq),L.shape[0]); extr=np.asarray(extr); intr=np.asarray(intr).reshape(-1)[:4]
        if not (np.isfinite(L).all() and np.isfinite(extr).all()): return None
        return (L,R,extr,intr)
    except Exception: return None

cards=[]
for clip in clips:
    tar=f"{FR}/{clip}.tar"; ld=_read_egodex_lowdim(tar,clip); T=ld.shape[0]
    gL=np.concatenate([ld[:,LD_LWRIST][:,None],ld[:,LD_LTIPS].reshape(T,5,3)],1)
    gR=np.concatenate([ld[:,LD_RWRIST][:,None],ld[:,LD_RTIPS].reshape(T,5,3)],1)
    gextr=ld[:,LD_EXTR].reshape(T,4,4); gintr=ld[:,LD_INTR]
    RC=load_method("/root/egodex_4way_pass/recon",clip); AC=load_method("/root/egodex_4way_pass/recon_anycalib",clip)
    from slam_failure_diag import diag as _diag
    def degen_reason(key):
        root="/root/egodex_4way_pass/"+("recon" if key=="recon" else "recon_anycalib")
        try:
            dd=_diag(f"{root}/{clip}")
            if dd is None: return "SLAM camera non-finite"
            if dd["nf_traj"]>0: return f"near-zero baseline -> BA diverged"
            if dd.get("nf_scale"): return "metric scale fit collapsed"
        except Exception: pass
        return "SLAM camera non-finite"
    REASON={"recon":(degen_reason("recon") if RC is None else None),"anycalib":(degen_reason("anycalib") if AC is None else None)}
    ef=f"/root/egodex_4way_pass/egoforce/{clip}.npz"; EF=np.load(ef) if os.path.exists(ef) else None
    mrow=byclip.get(clip,{}); GT21=gt_full(clip)
    with tarfile.open(tar) as tf:
        names=sorted(x for x in tf.getnames() if x.endswith(".image.jpg"))
        idxs=np.linspace(0,len(names)-1,min(12,len(names))).astype(int); outf=[]
        for fi in idxs:
            base=np.array(Image.open(io.BytesIO(tf.extractfile(names[fi]).read())).convert("RGB")); panels=[]
            for label,key,col in PAN:
                im=base.copy()
                if key=="gt":
                    fx,fy,cx,cy=gintr[fi]
                    if GT21 is not None:
                        Tg=GT21[0].shape[0]; gi=min(fi,Tg-1)
                        for J in (GT21[0][gi],GT21[1][gi]): draw_skel(im,projpts(J,gextr[fi],fx,fy,cx,cy),col)
                    else:
                        for J in (gL[fi],gR[fi]): draw_gt(im,projpts(J,gextr[fi],fx,fy,cx,cy),col)
                elif key in ("recon","anycalib"):
                    pk=RC if key=="recon" else AC
                    if pk is not None:
                        L,R,extr,intr=pk; Tr=L.shape[0]; ji=min(int(round(fi*Tr/max(1,T))),Tr-1); fx,fy,cx,cy=intr
                        for J in (L[ji],R[ji]): draw_skel(im,projpts(J,extr[ji],fx,fy,cx,cy),col)
                    else:  # degenerate SLAM camera -> can't direct-project; label it + the reason
                        oy=im.shape[0]//2; rtx=REASON.get(key) or "SLAM camera non-finite"
                        def _t(txt,y,sz):
                            cv2.putText(im,txt,(int(im.shape[1]*0.06),y),cv2.FONT_HERSHEY_SIMPLEX,sz,(0,0,0),5,cv2.LINE_AA)
                            cv2.putText(im,txt,(int(im.shape[1]*0.06),y),cv2.FONT_HERSHEY_SIMPLEX,sz,(245,90,90),2,cv2.LINE_AA)
                        _t("SLAM camera degenerate",oy-14,0.95); _t(rtx,oy+26,0.72)
                elif key=="egoforce" and EF is not None and fi<EF["j3d"].shape[0]:
                    vis=EF["visible"][fi]; j2=EF["j2d"][fi]
                    for hnd in range(2):
                        if vis[hnd]: draw_skel(im,[(j2[hnd][i][0],j2[hnd][i][1]) for i in range(21)],col)
                pw=int(PH*im.shape[1]/im.shape[0]); im=cv2.resize(im,(pw,PH))
                bar=np.full((32,pw,3),20,np.uint8)
                mv=mrow.get(key); mtxt=f"  {mv:.0f}mm" if isinstance(mv,float) else ("  (reference)" if key=="gt" else "")
                cv2.putText(bar,label+mtxt,(7,21),cv2.FONT_HERSHEY_SIMPLEX,0.52,(col[0],col[1],col[2]),1,cv2.LINE_AA)
                panels.append(cv2.cvtColor(np.vstack([bar,im]),cv2.COLOR_RGB2BGR))
            sep=np.full((panels[0].shape[0],3,3),40,np.uint8); row=panels[0]
            for p in panels[1:]: row=np.hstack([row,sep,p])
            outf.append(row)
    p=f"{SC}/fourway_sbs_{clip}.mp4"; uri=save_mp4(outf,p)
    cards.append(dict(clip=clip,uri=uri)); print("4way-sbs",clip[:40],"ok")
pickle.dump(cards,open(f"{SC}/fourway_overlay.pkl","wb")); print("DONE")
