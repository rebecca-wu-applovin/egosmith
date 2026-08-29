#!/usr/bin/env python3
"""Builder for notebooks/egocentric100k_undistort_smoke.ipynb — Phase 0 validation.

Validates the ONE genuinely new component for filtering Egocentric-100K: fisheye→pinhole
undistortion of the raw clips using the per-worker Kannala-Brandt intrinsics, and confirms the
real Stage-1 gates (heuristic_video_clipper) run on the undistorted frames. Pins the output
pinhole focal F (from cv2.fisheye) that Phase B/C will pass as a global --img_focal.

Build + execute:
  python notebooks/_build_egocentric100k_undistort_smoke_nb.py
  PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 notebooks/egocentric100k_undistort_smoke.ipynb
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "egocentric100k_undistort_smoke.ipynb"
BUCKET = "foundational-research/hoi-dataset/Egocentric-100K"


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


C = []

C.append(md(
    "# Egocentric-100K — Phase 0: fisheye→pinhole undistort + Stage-1 gate smoke\n"
    "\n"
    "The reconstruction pipeline is **pinhole-only** (DPVO/HaWoR take `[fx,fy,cx,cy]`, no distortion). "
    "Egocentric-100K is **fisheye** (Kannala-Brandt `k1..k4`, per-worker `intrinsics.json`). So filtering "
    "hinges on undistorting fisheye→pinhole up front. This notebook validates that on real clips and "
    "runs the **real** Stage-1 gates (`heuristic_video_clipper`) on the undistorted frames, and pins the "
    "output pinhole focal **F** (→ global `--img_focal`)."
))

C.append(code(
    "import sys; sys.path.insert(0, '/root/egosmith/src')\n"
    "import gcsfs, json, io, tarfile, tempfile, os\n"
    "import numpy as np, cv2\n"
    "import matplotlib.pyplot as plt\n"
    "BUCKET = %r\n"
    "fs = gcsfs.GCSFileSystem()\n"
    "def gp(*p): return '/'.join([BUCKET, *map(str, p)])\n"
    "print('cv2', cv2.__version__, '| has fisheye:', hasattr(cv2, 'fisheye'))\n" % BUCKET
))

# ---- pull one worker ----
C.append(md(
    "## 1. Pull one worker: intrinsics + a few clips\n"
    "Stream `factory001/worker001/part000.tar` and pull the first few clips' mp4 (ranged reads)."
))
C.append(code(
    "worker = gp('factory001', 'worker001')\n"
    "intr = json.load(fs.open(worker + '/intrinsics.json'))\n"
    "print('intrinsics:', json.dumps(intr))\n"
    "\n"
    "clips = []   # (key, mp4_bytes, meta)\n"
    "with fs.open(worker + '/part000.tar', 'rb') as f:\n"
    "    tf = tarfile.open(fileobj=f, mode='r|')\n"
    "    pend = {}\n"
    "    for m in tf:\n"
    "        key = m.name.rsplit('.', 1)[0]\n"
    "        data = tf.extractfile(m).read()\n"
    "        pend.setdefault(key, {})[m.name.rsplit('.',1)[1]] = data\n"
    "        if 'mp4' in pend[key] and 'json' in pend[key]:\n"
    "            clips.append((key, pend[key]['mp4'], json.loads(pend[key]['json'])))\n"
    "            del pend[key]\n"
    "        if len(clips) >= 4: break\n"
    "print('pulled clips:', [c[0] for c in clips])\n"
))

# ---- undistort ----
C.append(md(
    "## 2. Fisheye → pinhole undistort (cv2.fisheye)\n"
    "`K=[[fx,0,cx],[0,fy,cy],[0,0,1]]`, `D=[k1,k2,k3,k4]`. `cv2.fisheye."
    "estimateNewCameraMatrixForUndistortRectify(..., balance=b)` picks the output pinhole matrix; "
    "`balance=0` crops to only-valid pixels (tighter, higher focal), `balance=1` keeps full FOV "
    "(more black border, lower focal). The resulting **F = new_K[0,0]** is what reconstruction uses."
))
C.append(code(
    "W, H = intr['image_width'], intr['image_height']\n"
    "K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]], np.float64)\n"
    "D = np.array([intr['k1'],intr['k2'],intr['k3'],intr['k4']], np.float64)\n"
    "\n"
    "def make_map(balance):\n"
    "    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W,H), np.eye(3), balance=balance)\n"
    "    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (W,H), cv2.CV_16SC2)\n"
    "    return newK, m1, m2\n"
    "\n"
    "def decode_frame(mp4_bytes, frac=0.5):\n"
    "    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as t:\n"
    "        t.write(mp4_bytes); p = t.name\n"
    "    cap = cv2.VideoCapture(p); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1\n"
    "    cap.set(cv2.CAP_PROP_POS_FRAMES, int((n-1)*frac)); ok, fr = cap.read()\n"
    "    cap.release(); os.unlink(p)\n"
    "    return fr if ok else None\n"
    "\n"
    "raw = decode_frame(clips[0][1], 0.5)\n"
    "fig, ax = plt.subplots(1, 3, figsize=(15, 3))\n"
    "ax[0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)); ax[0].set_title(f'raw fisheye {raw.shape[1]}x{raw.shape[0]}')\n"
    "for a, bal in zip(ax[1:], (0.0, 1.0)):\n"
    "    newK, m1, m2 = make_map(bal)\n"
    "    und = cv2.remap(raw, m1, m2, cv2.INTER_LINEAR)\n"
    "    a.imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))\n"
    "    a.set_title(f'undistort balance={bal}  F={newK[0,0]:.1f}')\n"
    "for a in ax: a.axis('off')\n"
    "plt.tight_layout(); plt.show()\n"
    "\n"
    "BALANCE = 0.0\n"
    "newK, MAP1, MAP2 = make_map(BALANCE)\n"
    "F = float(newK[0,0])\n"
    "print(f'chosen balance={BALANCE} -> pinhole F={F:.1f} (cx,cy={newK[0,2]:.1f},{newK[1,2]:.1f}); '\n"
    "      f'compare raw fisheye fx={intr[\"fx\"]:.1f}, naive pinhole W/2={W/2:.1f}')\n"
))

# ---- stage-1 gates ----
C.append(md(
    "## 3. Stage-1 gates on undistorted frames (real code)\n"
    "Wrap the undistorted frames in a minimal frame-source and call the **actual** "
    "`analyze_frame_source_intervals` (Gate A YOLO hands + Gate B optical-flow RANSAC camera + Gate C "
    "span-merge). Keep = ≥1 valid `ClipInterval`. This is exactly what Phase A will run per clip."
))
C.append(code(
    "from lib.clip.heuristic_video_clipper import load_clip_config, analyze_frame_source_intervals, _load_yolo\n"
    "cfg = load_clip_config('/root/egosmith/src/lib/clip/heuristic_clip_config.yaml')\n"
    "model = _load_yolo('/root/egosmith/weights/external/detector.pt')\n"
    "print('YOLO loaded:', model is not None)\n"
    "\n"
    "class FrameList:\n"
    "    def __init__(self, frames): self.frames = frames\n"
    "    def __len__(self): return len(self.frames)\n"
    "    def get_frame(self, i, rgb=False):\n"
    "        fr = self.frames[i]\n"
    "        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if rgb else fr\n"
    "\n"
    "def undistort_all(mp4_bytes):\n"
    "    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as t:\n"
    "        t.write(mp4_bytes); p = t.name\n"
    "    cap = cv2.VideoCapture(p); out = []\n"
    "    while True:\n"
    "        ok, fr = cap.read()\n"
    "        if not ok: break\n"
    "        out.append(cv2.remap(fr, MAP1, MAP2, cv2.INTER_LINEAR))\n"
    "    cap.release(); os.unlink(p)\n"
    "    return out\n"
    "\n"
    "print(f\"{'clip':30} {'frames':>7} {'valid%':>7} {'intervals':>10} {'kept'}\")\n"
    "for key, mp4b, meta in clips:\n"
    "    frames = undistort_all(mp4b)\n"
    "    ivs, info = analyze_frame_source_intervals(FrameList(frames), cfg, model=model, fps=meta['fps'])\n"
    "    vf = 100*info['valid_sample_count']/max(1, info['sample_count'])\n"
    "    print(f\"{key:30} {info['total_frames']:7d} {vf:6.1f}% {len(ivs):10d} {'KEEP' if ivs else 'drop'}\")\n"
    "    for iv in ivs[:3]:\n"
    "        print(f\"      interval {iv.start_sec:.1f}-{iv.end_sec:.1f}s score={iv.score:.2f}\")\n"
))

C.append(md(
    "## 4. Takeaways for the filtering run\n"
    "- **Undistortion works** with the per-worker Kannala-Brandt intrinsics via `cv2.fisheye`; the "
    "output pinhole focal **F** (≠ naive `W/2`) is fixed across the dataset and passed as global `--img_focal`.\n"
    "- **Stage-1 gates run** on the undistorted frames and produce sane keep/drop + valid intervals — "
    "this is the cheap cut Phase A applies to all ~2M clips (streamed from GCS, no persistent frames).\n"
    "- **Phase A** = per-shard: stream `partNNN.tar` → decode+undistort each mp4 → gates → per-shard "
    "`stage1.kept.jsonl` + funnel. Survivors then go to Phase B (frame tars) → C (reconstruction) → D (Stage-4).\n"
    "- Revisit `BALANCE`/`F` if edge black-borders hurt tracking; `balance=0` (tighter crop) is the default."
))

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}}
OUT.write_text(nbf.writes(nb))
print("wrote", OUT)
