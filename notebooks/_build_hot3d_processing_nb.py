#!/usr/bin/env python3
"""Builder for notebooks/hot3d_processing.ipynb — how HOT3D is processed, end to end.

Documents every technique applied to turn raw HOT3D into filter-ready clips:
segmentation via official HOT3D-Clips, smplx-PCA MANO decode, fisheye->pinhole
undistortion (with a real before/after), camera + SLAM assembly, overlay validation,
and the quality-filter result.

Build + execute:
  python notebooks/_build_hot3d_processing_nb.py
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 \
      notebooks/hot3d_processing.ipynb
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "hot3d_processing.ipynb"
PROC = "/root/hot3d/viz/proc"


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


C = []

C.append(md(
    "# How HOT3D is processed\n"
    "\n"
    "End-to-end record of every technique used to turn **raw HOT3D** into filter-ready egocentric\n"
    "clips for the EgoSmith quality gate. HOT3D is head-mounted egocentric capture (Project Aria /\n"
    "Quest 3) of long, multi-activity sessions; we ingest the **Aria** view (it carries RGB) via its\n"
    "ground-truth hand + camera annotations. Converter: `scripts/build/generate_hot3d_world_res.py`.\n"
    "\n"
    "**Pipeline:** ① official-clip segmentation → ② smplx-PCA MANO decode → ③ camera → SLAM →\n"
    "④ **fisheye→pinhole undistortion** → ⑤ assemble `world_space_res` + frame tar → ⑥ overlay\n"
    "validation → ⑦ quality filter."
))

# 1. segmentation
C.append(md(
    "## ① Segmentation — reuse the official HOT3D-Clips (no heuristic segmenter)\n"
    "\n"
    "A raw HOT3D recording is ~2 min of *different* activities, so whole-sequence use is meaningless\n"
    "(hands 'swing' between unrelated tasks). Instead of tuning a motion segmenter, we reuse Meta's\n"
    "curated **HOT3D-Clips** (HF `bop-benchmark/hot3d`, `clip_definitions.json`): **3,832** hand-verified\n"
    "**150-frame (5 s) single-interaction** clips. We take the **1,983 Aria** clips (stream `214-1` = RGB;\n"
    "Quest3 clips are mono-only). Of those, **1,516 `train_aria`** carry GT hands; the 467 `test_aria`\n"
    "have GT withheld. Clips are **streamed** from HF (download tar → process → delete); no 690 GB VRS."
))

# 2. MANO decode
C.append(md(
    "## ② Ground-truth hand decode — smplx MANO, PCA-15\n"
    "\n"
    "Per frame, `<frame>.hands.json` gives each hand's `mano_pose = {thetas (15), wrist_xform (6)}` and\n"
    "the clip's `__hand_shapes.json__` gives `betas (10)`. HOT3D uses **smplx MANO with `use_pca=True,\n"
    "num_pca_comps=15, flat_hand_mean=False`** (toolkit `data_loaders/mano_layer.py`), so:\n"
    "\n"
    "- **articulation (45-d axis-angle)** = `hands_mean + thetas @ hands_components[:15]`\n"
    "- **global orient** = `wrist_xform[:3]` (axis-angle)\n"
    "- **translation** = `wrist_xform[3:6]` — the smplx `transl`, passed through unchanged (same\n"
    "  convention as EgoSmith's own smplx MANO, so no wrist-root offset like TACO/OakInk needed).\n"
    "\n"
    "This maps straight into `world_space_res` `[trans(2,T,3), rot(2,T,3), hand_pose(2,T,45),\n"
    "betas(2,T,10), valid(2,T)]` (index 0=left, 1=right)."
))
C.append(code(
    "import numpy as np\n"
    "# PCA basis exported once from the MANO pkls (avoids chumpy in the toolkit env)\n"
    "pca = {s: np.load(f'/root/hot3d/mano_pca_{s}.npz') for s in ('left','right')}\n"
    "def decode_articulation(thetas15, side):\n"
    "    d = pca[side]; return d['mean'] + np.asarray(thetas15) @ d['components'][:15]   # (45,)\n"
    "print('hands_components', pca['right']['components'].shape, '· hands_mean', pca['right']['mean'].shape)\n"
    "print('example thetas(15) -> aa45 norm:', round(float(np.linalg.norm(decode_articulation(np.zeros(15),'right'))),3))"
))

# 3. camera
C.append(md(
    "## ③ Camera → SLAM sidecar\n"
    "\n"
    "`<frame>.cameras.json['214-1']['T_world_from_camera']` is the per-frame camera pose (camera→world,\n"
    "quaternion `wxyz` + translation). EgoSmith's camera reader expects a SLAM npz of c2w trajectory\n"
    "rows `[tx,ty,tz, qx,qy,qz,qw]` (+ intrinsics), which it inverts to world→camera. So we write the\n"
    "c2w rows directly and the pinhole intrinsics from the undistortion step (below)."
))

# 4. fisheye undistortion — the headline technique
C.append(md(
    "## ④ Fisheye → pinhole undistortion (the key image step)\n"
    "\n"
    "HOT3D's Aria RGB is a **FISHEYE624** camera (Kannala-Brandt: 6 radial + 2 tangential + 4\n"
    "thin-prism), 1408×1408, ~110°+ FOV with a circular vignette. EgoSmith's off-screen / in-frame /\n"
    "camera-space rules all assume a **pinhole** projection — feeding raw fisheye would misalign the\n"
    "projected GT joints against the pixels and corrupt the off-screen statistics. So every frame is\n"
    "resampled to a pinhole camera.\n"
    "\n"
    "**Method** (`generate_hot3d_world_res.py::_build_undistort_map`, using `hand_tracking_toolkit.camera`):\n"
    "\n"
    "1. Build the source model: `fish = camera.from_json(cameras['214-1'])` → `Fisheye624CameraModel`.\n"
    "2. Build the target: `pin = PinholePlaneCameraModel(W, H, f=(f·scale, f·scale), c=(cx,cy),\n"
    "   distort_coeffs=[])` — same focal & principal point, zero distortion.\n"
    "3. Build a **remap once per clip** (calibration is pose-independent, constant across the 150 frames):\n"
    "   for each output pinhole pixel `(u,v)` → `pin.window_to_eye` gives a 3-D ray → `fish.eye_to_window`\n"
    "   gives the source fisheye pixel → `(map_x, map_y)`.\n"
    "4. `cv2.remap(frame, map_x, map_y, INTER_LINEAR)` per frame.\n"
    "5. Write the matching pinhole intrinsics `[fx,fy,cx,cy]` into the SLAM npz so image + intrinsics +\n"
    "   projection all agree."
))
C.append(code(
    "import base64\n"
    "from pathlib import Path\n"
    "from IPython.display import display, HTML, Markdown\n"
    "# base64-encode at runtime so the code cell stays short; data lands only in the output\n"
    f"PROC = {PROC!r}\n"
    "def _uri(name):\n"
    "    ext = 'jpeg' if name.endswith(('.jpg', '.jpeg')) else 'png'\n"
    "    return f'data:image/{ext};base64,' + base64.b64encode(Path(f'{PROC}/{name}').read_bytes()).decode()\n"
    "raw = _uri('fisheye_raw.png')\n"
    "s10 = _uri('pinhole_s10.png')\n"
    "display(HTML(\n"
    "  '<div style=\"display:flex;gap:14px;flex-wrap:wrap\">'\n"
    "  f'<figure style=\"margin:0\"><img src=\"{raw}\" width=340><figcaption style=\"font:12px monospace;color:#888\">RAW FISHEYE624 — barrel distortion + vignette</figcaption></figure>'\n"
    "  f'<figure style=\"margin:0\"><img src=\"{s10}\" width=340><figcaption style=\"font:12px monospace;color:#888\">UNDISTORTED pinhole (focal_scale=1.0) — straight edges</figcaption></figure>'\n"
    "  '</div>'))"
))
C.append(md(
    "**FOV trade-off (`focal_scale`).** A pinhole at `focal_scale=1.0` keeps ~98° of the fisheye's\n"
    "~110°+ — the extreme periphery is cropped (a legitimate 'off-screen' for a pinhole, and part of\n"
    "why HOT3D's off-screen drop rate is higher than flat-camera datasets). Lowering `focal_scale`\n"
    "shrinks the focal length to retain more periphery, at the cost of more edge warping:"
))
C.append(code(
    "s07 = _uri('pinhole_s07.png')\n"
    "display(HTML(\n"
    "  '<div style=\"display:flex;gap:14px;flex-wrap:wrap\">'\n"
    "  f'<figure style=\"margin:0\"><img src=\"{s10}\" width=340><figcaption style=\"font:12px monospace;color:#888\">focal_scale=1.0 (used) — ~98° FOV</figcaption></figure>'\n"
    "  f'<figure style=\"margin:0\"><img src=\"{s07}\" width=340><figcaption style=\"font:12px monospace;color:#888\">focal_scale=0.7 — more periphery, more warp</figcaption></figure>'\n"
    "  '</div>'))"
))

# 5. assemble
C.append(md(
    "## ⑤ Assemble the clip\n"
    "\n"
    "Per clip we write: the undistorted RGB as `<clip>_f%05d.image.jpg` (one tar), `world_space_res.pth`\n"
    "(the decoded MANO above), `SLAM/hawor_slam_w_scale_0.npz` (c2w traj + pinhole intrinsics), a\n"
    "`tracks_0_T` dir + `infiller` done-marker (so the filter runs with `--stages infiller`). Both hands\n"
    "are ingested when present, so the filter judges both. Isolated `hot3d` conda env for the toolkit\n"
    "(numpy 2), separate from the filter env."
))

# 6. overlay validation
C.append(md(
    "## ⑥ Overlay validation (the gate)\n"
    "\n"
    "Before the full run, project the decoded GT MANO joints (via EgoSmith's own MANO) through the\n"
    "undistorted pinhole camera onto the frames. They must lock onto the hands — confirming the PCA\n"
    "decode, transl convention, camera direction, and undistortion are all mutually consistent\n"
    "(blue=left, red=right; wrist ringed)."
))
C.append(code(
    "ov = _uri('overlay_kept.jpg')\n"
    "display(HTML(f'<img src=\"{ov}\" style=\"width:100%;max-width:1000px;border-radius:6px\">'))"
))

# 7. filter result
C.append(md("## ⑦ Quality filter — result\n\nSame gate as TACO/OakInk (`--stages infiller`). Kept vs dropped and why:"))
C.append(code(
    "import json\n"
    "from collections import Counter\n"
    "import matplotlib.pyplot as plt\n"
    "r = json.load(open('/root/hot3d/filter_run/filter_report.json'))\n"
    "conv = json.load(open('/root/hot3d/filter_run/convert_report.json'))\n"
    "print(f\"converted {conv.get('converted_ok','?')} train_aria clips (467 test_aria have GT withheld)\")\n"
    "print(f\"kept {r['kept_clips']} / {r['total_clips']} = {100*r['kept_clips']/r['total_clips']:.1f}%  |  dropped {r['dropped_clips']}\")\n"
    "rc = Counter(r['quality_reason_counts'])\n"
    "labels, vals = zip(*sorted(rc.items(), key=lambda kv:-kv[1]))\n"
    "fig, ax = plt.subplots(figsize=(9, max(2,0.4*len(labels))))\n"
    "ax.barh(range(len(labels)), vals, color='#db4437'); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)\n"
    "ax.invert_yaxis(); ax.set_xlabel('clips triggering reason'); ax.set_title('HOT3D drop reasons')\n"
    "for i,v in enumerate(vals): ax.text(v,i,f' {v}',va='center')\n"
    "plt.tight_layout(); plt.show()"
))
C.append(md(
    "The dominant drops are **off-screen** — a hand leaving the head-camera FOV within the 5 s window\n"
    "(the pinhole FOV crop from ④ contributes), with left-hand rules firing ~2× the right (right-handed\n"
    "manipulation leaves the left hand off-frame). Full analysis + failure galleries:\n"
    "`notebooks/hot3d_filter_report.ipynb`; segmented-clip videos: `scripts/inspection/hot3d_clip_videos.py`.\n"
    "\n"
    "### Techniques applied, in one line each\n"
    "| # | technique | why |\n"
    "|---|---|---|\n"
    "| ① | reuse official HOT3D-Clips segmentation | long multi-activity recordings → single-interaction 5 s clips, no heuristic segmenter |\n"
    "| ② | smplx PCA-15 MANO decode (`mean + thetas@comp`) | HOT3D stores 15 PCA coeffs, not 45 axis-angle |\n"
    "| ③ | `T_world_from_camera` (c2w) → SLAM npz | feed EgoSmith's camera reader |\n"
    "| ④ | **fisheye624 → pinhole remap** | EgoSmith projects pinhole; raw fisheye would misalign |\n"
    "| ⑤ | streaming download + isolated toolkit env | 206 GB over HF without 690 GB VRS; numpy-2 toolkit kept off the filter env |\n"
    "| ⑥ | overlay gate before full run | verify conventions end-to-end |\n"
    "| ⑦ | standard quality filter | same rules as every other dataset |"
))


def main():
    nb = nbf.v4.new_notebook(); nb["cells"] = C
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    OUT.write_text(nbf.writes(nb)); print("wrote", OUT)


if __name__ == "__main__":
    main()
