#!/usr/bin/env python
"""use_gt (GT) vs recon hand-keypoints overlaid on the SAME video, per dataset — shows which
locks onto the hand. GT = green, recon = red; 21-joint MANO skeleton drawn for visibility."""
import sys, io, os, json, base64, tarfile, numpy as np, cv2, pickle
from pathlib import Path
sys.path.insert(0,"/root/egosmith/src"); sys.path.insert(0,"/root/egosmith"); sys.path.insert(0,"/root/egosmith/scripts/inspection")
from taco_overlay_sheets import compute_world_joints, project, load_camera
SC="/tmp/claude-0/-root-egosmith/6b57c32e-2fab-4780-bf21-8479371681d9/scratchpad"
FR={"taco":"/root/taco/frames","oakink_grasp":"/root/oakink/grasp/frames","hot3d":"/root/hot3d/frames"}
GT_C=(60,200,90); RE_C=(230,70,60)  # RGB: GT green, recon red
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
ROT={"hot3d":cv2.ROTATE_90_CLOCKWISE}

def frames_from_tar(tar, names, idxs):
    want={names[i]:i for i in idxs if i<len(names)}; out={}
    with tarfile.open(tar) as tf:
        for m in tf:
            if m.name in want: out[want[m.name]]=np.array(__import__("PIL.Image",fromlist=["Image"]).open(io.BytesIO(tf.extractfile(m).read())).convert("RGB"))
    return out

def draw(im,uvz,col):
    pts={}
    for j,(u,v,z) in enumerate(uvz):
        if z>1e-3 and -50<=u<im.shape[1]+50 and -50<=v<im.shape[0]+50: pts[j]=(int(u),int(v))
    for a,b in EDGES:
        if a in pts and b in pts: cv2.line(im,pts[a],pts[b],col,3,cv2.LINE_AA)
    for j,p in pts.items(): cv2.circle(im,p,5,col,-1,cv2.LINE_AA)

def overlay_clip(ds, clip):
    rec=json.loads(next(l for l in open(f"{SC}/{ds}_usegt.layer4_input.jsonl") if json.loads(l)["descriptor"]["clip_id"]==clip))
    d=rec["descriptor"]; names=d["frame_names"]; nfr=len(names)
    tar=f"{FR[ds]}/{clip}.tar"
    if not os.path.exists(tar): return None
    def load(seq):
        try:
            L,R=compute_world_joints(Path(seq),"cuda"); extr,intr=load_camera(Path(seq),L.shape[0])
            extr=np.asarray(extr); intr=np.asarray(intr).reshape(-1)[:4]
            if not (np.isfinite(L).all() and np.isfinite(R).all() and np.isfinite(extr).all()): return None
            return (L,R,extr,intr)
        except Exception: return None
    G=load(f"/root/{ds}_usegt/recon_outputs/{clip}"); Rn=load(f"/root/{ds}/recon_outputs/{clip}")
    if G is None: return None
    step=max(1,nfr//12); vidx=list(range(0,nfr,step))[:12]
    imgs=frames_from_tar(tar,names,vidx); ims=[]
    for vi in vidx:
        if vi not in imgs: continue
        im=imgs[vi].copy()
        for pack,col in ((G,GT_C),(Rn,RE_C)):
            if pack is None: continue
            L,R,extr,intr=pack; T=L.shape[0]; ji=min(int(round(vi*T/max(1,nfr))),T-1)
            for J in (L[ji],R[ji]): draw(im,project(J,extr[ji],intr),col)
        if ds in ROT: im=cv2.rotate(im,ROT[ds])
        h=300; im=cv2.resize(im,(int(h*im.shape[1]/im.shape[0]),h))
        ims.append(__import__("PIL.Image",fromlist=["Image"]).fromarray(im))
    if not ims: return None
    p=f"{SC}/ovl_{ds}_{clip}.gif"; ims[0].save(p,save_all=True,append_images=ims[1:],duration=480,loop=0,optimize=True)
    return dict(ds=ds,clip=clip,recon_ok=Rn is not None,uri="data:image/gif;base64,"+base64.b64encode(open(p,"rb").read()).decode())

out={}
for ds in ("taco","oakink_grasp","hot3d"):
    clips=[c.strip() for c in open(f"{SC}/{ds}_overlay_clips.txt") if c.strip()][:2]
    out[ds]=[]
    for c in clips:
        r=overlay_clip(ds,c)
        if r: out[ds].append(r); print("overlay",ds,c,"recon_ok",r["recon_ok"])
        else: print("skip",ds,c)
pickle.dump(out,open(f"{SC}/overlay_cards.pkl","wb")); print("DONE")
