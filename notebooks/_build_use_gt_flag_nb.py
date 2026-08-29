#!/usr/bin/env python3
"""Builder for notebooks/use_gt_flag.ipynb — the first-class `--use_gt` pipeline flag.

Part 2 counterpart to taco_recon_viz.ipynb. Documents + verifies the `--use_gt` flag that makes
the pipeline run the WHOLE flow (prepare -> detect_track -> motion -> slam -> infiller -> filter)
but **loads ground truth at the slam + infiller stages when present**, else reconstructs — replacing
the old bolted-on pre-write-and-skip hack (`generate_*_world_res.py` + `--stages infiller`).

Branches added (guarded by `getattr(args,"use_gt",False)`, default off -> normal path unchanged):
  - stages/slam.py::hawor_slam        -> adopt a staged GT SLAM npz (skip DPVO+Any4D+scale)
  - stages/hawor_infiller_stage.py    -> adopt staged GT world_space_res (skip infiller model+pass)
Wired through batch/{cli,config,worker_pool}, proc/{runtime,stage_api}, orchestrator/constants.

Compares one TACO clip processed three ways — **GT-ingestion**, **--use_gt** (whole pipeline),
and **full reconstruction** (video-only) — proving --use_gt is byte-equivalent to GT-ingestion while
running the extra Layer-A/B gates, and that reconstruction is the graceful fallback.

Build + execute:
  python notebooks/_build_use_gt_flag_nb.py
  PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=900 notebooks/use_gt_flag.ipynb
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "use_gt_flag.ipynb"
CLIP = "TACO_brush_brush_bowl_20231005_188"


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


C = []

C.append(md(
    "# The `--use_gt` flag — load GT instead of reconstructing, inside the whole pipeline\n"
    "\n"
    "Every dataset we filtered used **GT-ingestion**: `generate_*_world_res.py` pre-writes "
    "`world_space_res.pth` + a GT `SLAM/*.npz`, then `filter --stages infiller` runs — which "
    "**skips Layers A & B entirely** (only Layer C executes). `--use_gt` fixes that: it runs the "
    "**whole pipeline** (so the Layer-A hand-size/stability gates and Layer-B track gates execute), "
    "but at the **slam** and **infiller** stages it *loads* the staged GT instead of running "
    "DPVO/Any4D/infiller — and falls back to reconstruction when no GT is present.\n"
    "\n"
    "Branches (guarded by `--use_gt`, default off so the normal path is untouched):\n"
    "- `stages/slam.py::hawor_slam` → adopt a staged GT SLAM npz (skip DPVO+Any4D+scale)\n"
    "- `stages/hawor_infiller_stage.py::run_infiller_for_video` → adopt GT `world_space_res` (skip infiller)\n"
    "\n"
    "Wired through `batch/{cli,config,worker_pool}`, `proc/{runtime,stage_api}`, `orchestrator/constants`.\n"
    "\n"
    "Below: one TACO clip processed **three ways** and compared."
))

C.append(code(
    "import sys, os, glob, json, subprocess\n"
    "from pathlib import Path\n"
    "import numpy as np, torch\n"
    "import matplotlib.pyplot as plt\n"
    "sys.path.insert(0, os.path.abspath('../src')); sys.path.insert(0, os.path.abspath('../scripts/inspection'))\n"
    f"CLIP = {CLIP!r}\n"
    "GT     = f'/root/taco/outputs/{CLIP}'                 # (1) ground truth (TACO MANO + GT ego camera)\n"
    "USE_GT = f'/root/taco_recon/outputs_gt2/{CLIP}'       # (2) --use_gt whole-pipeline run (keep-all)\n"
    "RECON  = f'/root/taco_recon/outputs2/{CLIP}'          # (3) full reconstruction, video-only\n"
    "EV_GT  = '/root/taco_recon/run_gt2/events.jsonl'      # per-stage timing for --use_gt\n"
    "EV_REC = '/root/taco_recon/run2/events.jsonl'         # per-stage timing for reconstruction\n"
    "dev = 'cuda:0'\n"
    "print('variants on disk:', {k: os.path.isdir(v) for k,v in [('GT',GT),('USE_GT',USE_GT),('RECON',RECON)]})"
))

# commands
C.append(md(
    "## The three ways to process the clip\n"
    "```bash\n"
    "# (1) GT-ingestion (today's bolted-on path): pre-write GT, validate only infiller — skips A & B\n"
    "filter_manifest_by_quality.py --stages infiller ...\n"
    "\n"
    "# (2) --use_gt: run the WHOLE pipeline, load GT at slam+infiller (A & B gates run)\n"
    "batch_infer.py --stages detect_track,motion,slam,infiller --use_gt --keep_intermediates all ...\n"
    "\n"
    "# (3) full reconstruction (fallback / no-GT sources like EgoDex-sharpa)\n"
    "batch_infer.py --stages detect_track,motion,slam,infiller ...   # (no --use_gt)\n"
    "```"
))

# stage timing
C.append(md("## ① Per-stage wall-clock — `--use_gt` skips the heavy stages\n\nFrom each run's `events.jsonl`. Under `--use_gt`, `slam` + `infiller` collapse to ~0 (they load GT); `detect_track` + `motion` still run (Layer-B gates)."))
C.append(code(
    "def stage_times(ev):\n"
    "    t = {}\n"
    "    if not os.path.exists(ev): return t\n"
    "    for ln in open(ev):\n"
    "        d = json.loads(ln)\n"
    "        if d.get('event')=='stage_success': t[d['stage']] = d.get('wall_sec', 0.0)\n"
    "    return t\n"
    "tg, tr = stage_times(EV_GT), stage_times(EV_REC)\n"
    "stages = ['detect_track','motion','slam','infiller']\n"
    "x = np.arange(len(stages)); w = 0.38\n"
    "plt.figure(figsize=(8,3))\n"
    "plt.bar(x-w/2, [tr.get(s,0) for s in stages], w, label='full reconstruction', color='#db4437')\n"
    "plt.bar(x+w/2, [tg.get(s,0) for s in stages], w, label='--use_gt', color='#14895a')\n"
    "plt.xticks(x, stages); plt.ylabel('wall_sec'); plt.title('per-stage time: reconstruction vs --use_gt'); plt.legend()\n"
    "for xi,s in zip(x,stages):\n"
    "    plt.text(xi-w/2, tr.get(s,0), f\"{tr.get(s,0):.0f}\", ha='center', va='bottom', fontsize=8)\n"
    "    plt.text(xi+w/2, tg.get(s,0), f\"{tg.get(s,0):.0f}\", ha='center', va='bottom', fontsize=8)\n"
    "plt.tight_layout(); plt.show()\n"
    "print(f\"total: reconstruction={sum(tr.values()):.0f}s  vs  --use_gt={sum(tg.values()):.0f}s\")\n"
    "print(f\"stages that ran under --use_gt (Layer-B applied): {sorted(tg)}\")"
))

# GT adoption
C.append(md("## ② `--use_gt` adopts the GT camera (not a reconstructed one)\n\nThe slam GT branch copies the staged GT npz to the name the loaders expect. So `--use_gt`'s SLAM is byte-identical to GT; reconstruction's differs (solved focal/scale)."))
C.append(code(
    "def slam_of(seq):\n"
    "    f = sorted(glob.glob(f'{seq}/SLAM/hawor_slam_w_scale_*.npz'))[-1]; d = np.load(f)\n"
    "    return float(d['img_focal']), float(d['scale']), np.asarray(d['traj'])\n"
    "fg, sg, jg = slam_of(GT); fu, su, ju = slam_of(USE_GT); fr, sr, jr = slam_of(RECON)\n"
    "print(f\"{'variant':10} {'focal':>8} {'scale':>7}   traj==GT?\")\n"
    "print(f\"{'GT':10} {fg:8.1f} {sg:7.3f}   -\")\n"
    "print(f\"{'--use_gt':10} {fu:8.1f} {su:7.3f}   {np.allclose(ju, jg)}\")\n"
    "print(f\"{'recon':10} {fr:8.1f} {sr:7.3f}   {np.allclose(jr, jg)}\")\n"
    "import filecmp\n"
    "same = filecmp.cmp(f'{USE_GT}/world_space_res.pth', f'{GT}/world_space_res.pth', shallow=False)\n"
    "print(f\"\\n--use_gt world_space_res.pth byte-identical to GT: {same}\")\n"
    "print(f\"--use_gt created a fresh result.npz (infiller recompute)? {os.path.exists(f'{USE_GT}/result.npz')}  (False = adopted GT)\")"
))

# hand equivalence
C.append(md("## ③ Hand equivalence — `--use_gt` = GT exactly; reconstruction ≈ few-cm\n\nCamera-frame hand MPJPE vs GT. `--use_gt` is 0 (it *is* the GT); full reconstruction is the accuracy from Part 1 (this is what a no-GT source like EgoDex would get)."))
C.append(code(
    "from lib.pipeline.io.result_io import load_pose_arrays\n"
    "from lib.pipeline.slam.slam_cam import load_slam_cam\n"
    "from lib.pipeline.exporters.mano_features import _compute_hand_joints, build_mano_models\n"
    "mano_r, mano_l = build_mano_models(dev)\n"
    "def wj(seq):\n"
    "    tr,ro,hp,be,va = load_pose_arrays(seq); tr,ro,hp,be=[torch.from_numpy(np.asarray(x)) for x in (tr,ro,hp,be)]\n"
    "    L=_compute_hand_joints(mano_l,tr,ro,hp,be,0,dev).cpu().numpy(); R=_compute_hand_joints(mano_r,tr,ro,hp,be,1,dev).cpu().numpy()\n"
    "    return L,R\n"
    "def cams(seq):\n"
    "    f=sorted(glob.glob(f'{seq}/SLAM/hawor_slam_w_scale_*.npz'))[-1]; r,t,_,_=load_slam_cam(f); return np.asarray(r),np.asarray(t)\n"
    "Lg,Rg=wj(GT); Rwc_g,twc_g=cams(GT)\n"
    "def to_cam(J,Rwc,twc,T): return np.einsum('tij,tnj->tni',Rwc[:T],J[:T])+twc[:T,None,:]\n"
    "def cam_mpjpe(seq):\n"
    "    Ls,Rs=wj(seq); Rwc,twc=cams(seq); T=min(len(Ls),len(Lg)); errs=[]\n"
    "    for Js,Jg in ((Ls,Lg),(Rs,Rg)):\n"
    "        errs.append(np.sqrt(((to_cam(Js,Rwc,twc,T)-to_cam(Jg,Rwc_g,twc_g,T))**2).sum(-1)).mean()*1000)\n"
    "    return np.mean(errs)\n"
    "e_use, e_rec = cam_mpjpe(USE_GT), cam_mpjpe(RECON)\n"
    "plt.figure(figsize=(5,3)); b=plt.bar(['--use_gt','full recon'],[e_use,e_rec],color=['#14895a','#db4437'])\n"
    "plt.ylabel('cam-frame hand MPJPE vs GT (mm)'); plt.title('hand equivalence vs GT')\n"
    "for r,v in zip(b,[e_use,e_rec]): plt.text(r.get_x()+r.get_width()/2,v,f'{v:.1f}',ha='center',va='bottom')\n"
    "plt.tight_layout(); plt.show()\n"
    "print(f'--use_gt vs GT: {e_use:.2f} mm (0 = identical inputs)  |  reconstruction vs GT: {e_rec:.1f} mm')"
))

# filter equivalence
C.append(md("## ④ Filter equivalence — `--use_gt` output filters identically to GT-ingestion\n\nSame clip through the quality filter (single-clip). `--use_gt` (with `--keep_intermediates all`, so the track range survives for the filter) yields the identical keep/drop decision as the GT seq_folder."))
C.append(code(
    "def run_filter(seq, tag):\n"
    "    import tarfile\n"
    "    tp=f'/root/taco/frames/{CLIP}.tar'; fn=[];fo=[]\n"
    "    with tarfile.open(tp) as t:\n"
    "        for m in sorted((m for m in t if m.isfile() and m.name.endswith('.image.jpg')),key=lambda m:m.name):\n"
    "            fn.append(m.name);fo.append([int(m.offset_data),int(m.size)])\n"
    "    rec={'clip_id':CLIP,'source_id':'x','split':'train','group_id':'g','descriptor':{'clip_id':CLIP,'clip_name':CLIP,\n"
    "        'storage_kind':'tar_shard','root_dir':'/root/taco/frames','seq_folder':seq,'frame_names':fn,'frame_offsets':fo,\n"
    "        'shard_path':tp,'extra':{'adapter':'taco_tar'}},'metadata':{}}\n"
    "    mf=f'/tmp/_f_{tag}.jsonl'; open(mf,'w').write(json.dumps(rec)+'\\n'); rep=f'/tmp/_r_{tag}.json'\n"
    "    subprocess.run([sys.executable,'../scripts/build/filter_manifest_by_quality.py','--input_manifest',mf,\n"
    "        '--output_manifest',f'/tmp/_k_{tag}.jsonl','--report_out',rep,'--stages','infiller','--source_fps','30',\n"
    "        '--target_fps','30','--mano_gpus','0','--workers','1'],env={**os.environ,'PYTHONPATH':os.path.abspath('../src')},\n"
    "        capture_output=True)\n"
    "    r=json.load(open(rep)); return r['kept_clips'], r['dropped_clips'], [d['reasons'] for d in r['dropped']]\n"
    "for seq,tag in [(GT,'gt'),(USE_GT,'usegt')]:\n"
    "    k,d,reasons = run_filter(seq, tag)\n"
    "    print(f'{tag:6}: kept={k} dropped={d}  reasons={reasons or \"[] (KEPT)\"}')"
))

C.append(md(
    "## Verdict\n"
    "\n"
    "`--use_gt` is a **first-class** GT-load path: the whole pipeline runs (so the Layer-A hand-size / "
    "camera-stability gates and Layer-B track gates — the ones GT-ingestion skipped — now execute), while "
    "`slam` and `infiller` **adopt the staged GT** (byte-identical camera + hand poses, no recompute) and "
    "collapse to ~0 s. The filter decision is **identical to GT-ingestion**.\n"
    "\n"
    "When no GT is staged (e.g. EgoDex-sharpa videos), the same run **reconstructs** — the `full recon` "
    "column above shows what that yields: a good camera and camera-frame hands accurate to a few cm. So one "
    "flag covers both worlds: load GT where it exists, reconstruct where it doesn't, always through the full "
    "quality gauntlet.\n"
    "\n"
    "_Note: pass `--keep_intermediates all` for the whole-pipeline-then-filter workflow so the `tracks_0_N` "
    "track range survives for the filter (a general cleanup interaction, not `--use_gt`-specific)._"
))


def main():
    nb = nbf.v4.new_notebook(); nb["cells"] = C
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    OUT.write_text(nbf.writes(nb)); print("wrote", OUT)


if __name__ == "__main__":
    main()
