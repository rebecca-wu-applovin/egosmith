#!/usr/bin/env python3
"""Builder for notebooks/egocentric100k_explore.ipynb — walk the Egocentric-100K data structure.

Goal: understand the raw layout end-to-end BEFORE filtering it. Covers the bucket tree
(factory/worker/part.tar), the WebDataset per-clip payload (mp4 + json), the per-worker fisheye
intrinsics, a decoded sample frame, dataset scale, and what all of that implies for filtering
(video-only / no GT → reconstruction track; fisheye → undistortion).

Build + execute:
  python notebooks/_build_egocentric100k_explore_nb.py
  PYTHONPATH=src jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1200 notebooks/egocentric100k_explore.ipynb
"""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "egocentric100k_explore.ipynb"
BUCKET = "foundational-research/hoi-dataset/Egocentric-100K"


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


C = []

C.append(md(
    "# Egocentric-100K — data structure walkthrough\n"
    "\n"
    "Purpose: understand the **raw layout** before filtering it through the EgoSmith pipeline.\n"
    "\n"
    "Source: `gs://foundational-research/hoi-dataset/Egocentric-100K/` (mirror of the Build AI "
    "HuggingFace dataset). Headline stats from its README:\n"
    "\n"
    "| attribute | value |\n"
    "|---|---|\n"
    "| total hours | 100,405 |\n"
    "| total frames | 10.8 billion |\n"
    "| video clips | 2,010,759 |\n"
    "| median clip length | 180 s |\n"
    "| resolution | 456×256 (256p) |\n"
    "| frame rate | 30 fps |\n"
    "| codec | H.265 / MP4 |\n"
    "| camera | monocular head-mounted **fisheye** |\n"
    "| GT labels | **none** (video + minimal metadata only) |\n"
    "| storage | 24.79 TB |\n"
))

C.append(code(
    "import gcsfs, json, io, tarfile, tempfile, os\n"
    "import numpy as np\n"
    "import cv2\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "BUCKET = %r\n"
    "fs = gcsfs.GCSFileSystem()\n"
    "def gpath(*p): return '/'.join([BUCKET, *[str(x) for x in p]])\n"
    "print('bucket:', gpath())\n" % BUCKET
))

# ---- top-level ----
C.append(md(
    "## 1. Top-level layout\n"
    "Root holds `README.md`, `.gitattributes`, and one directory per **factory**."
))
C.append(code(
    "top = fs.ls(gpath(), detail=False)\n"
    "factories = sorted(p for p in top if p.rsplit('/',1)[-1].startswith('factory'))\n"
    "files = [p.rsplit('/',1)[-1] for p in top if not p.rsplit('/',1)[-1].startswith('factory')]\n"
    "print('root files   :', files)\n"
    "print('num factories:', len(factories))\n"
    "print('first / last :', factories[0].rsplit('/',1)[-1], '...', factories[-1].rsplit('/',1)[-1])\n"
))

# ---- factory / worker ----
C.append(md(
    "## 2. factory → worker → files\n"
    "Each **factory** contains **worker** directories; each worker holds one `intrinsics.json` "
    "(its device calibration) plus a set of `partNNN.tar` WebDataset shards."
))
C.append(code(
    "fac = gpath('factory001')\n"
    "workers = sorted(p for p in fs.ls(fac, detail=False) if '/worker' in p)\n"
    "print('factory001 workers:', len(workers))\n"
    "w = workers[0]\n"
    "print('\\nworker001 contents:')\n"
    "for e in fs.ls(w, detail=True):\n"
    "    print(f\"  {e['name'].rsplit('/',1)[-1]:16} {e.get('size',0)/1e6:8.1f} MB\")\n"
))

# ---- intrinsics ----
C.append(md(
    "## 3. Camera intrinsics — fisheye (per worker/device)\n"
    "`intrinsics.json` is a **Kannala-Brandt fisheye** model (`k1..k4`), *not* a pinhole. "
    "Note the focal (`fx≈fy≈138`) is far from a pinhole `W/2 = 228` — treating this as pinhole "
    "would be wrong; the pipeline must undistort / use the fisheye model. Intrinsics vary per worker."
))
C.append(code(
    "def load_intr(worker_path):\n"
    "    with fs.open(worker_path.rstrip('/') + '/intrinsics.json') as f:\n"
    "        return json.load(f)\n"
    "\n"
    "intr0 = load_intr(workers[0])\n"
    "print(json.dumps(intr0, indent=2))\n"
    "\n"
    "print('\\nper-worker variation (first 5 workers of factory001):')\n"
    "print(f\"{'worker':10} {'model':8} {'fx':>7} {'fy':>7} {'cx':>7} {'cy':>7} {'k1':>7}\")\n"
    "for wp in workers[:5]:\n"
    "    d = load_intr(wp)\n"
    "    print(f\"{wp.rstrip('/').rsplit('/',1)[-1]:10} {d['model']:8} \"\n"
    "          f\"{d['fx']:7.1f} {d['fy']:7.1f} {d['cx']:7.1f} {d['cy']:7.1f} {d['k1']:7.3f}\")\n"
))

# ---- part.tar layout ----
C.append(md(
    "## 4. WebDataset shard (`partNNN.tar`)\n"
    "Each shard is a tar of per-clip pairs: `<key>.mp4` (H.265 video) + `<key>.json` (metadata). "
    "Key = `factory_XXX_worker_YYY_NNNN`. We stream the shard over GCS and pull just the first "
    "clip (mp4 + json) via a ranged read — no full-shard download."
))
C.append(code(
    "part = w.rstrip('/') + '/part000.tar'\n"
    "print('shard:', part, f\"({fs.info(part)['size']/1e9:.2f} GB)\")\n"
    "\n"
    "members, first_mp4, first_json = [], None, None\n"
    "with fs.open(part, 'rb') as f:\n"
    "    tf = tarfile.open(fileobj=f, mode='r|')   # streaming\n"
    "    for m in tf:\n"
    "        members.append((m.name, m.size))\n"
    "        if m.name.endswith('.mp4') and first_mp4 is None:\n"
    "            first_mp4 = (m.name, tf.extractfile(m).read())\n"
    "        elif m.name.endswith('.json') and first_json is None:\n"
    "            first_json = (m.name, tf.extractfile(m).read())\n"
    "        if len(members) >= 12 and first_mp4 and first_json:\n"
    "            break\n"
    "print('\\nfirst members:')\n"
    "for name, size in members[:8]:\n"
    "    print(f'  {name:34} {size/1e6:7.2f} MB' if size>1e5 else f'  {name:34} {size:6d} B')\n"
))

# ---- clip metadata ----
C.append(md(
    "## 5. Per-clip metadata (`<key>.json`)\n"
    "Minimal: identity + video properties. **No hand pose, no keypoints, no GT** — this is the "
    "crucial fact for filtering (see §8)."
))
C.append(code(
    "meta = json.loads(first_json[1])\n"
    "print(first_json[0])\n"
    "print(json.dumps(meta, indent=2))\n"
))

# ---- decode a frame ----
C.append(md(
    "## 6. Decoded sample frames (fisheye)\n"
    "Decode the first clip's mp4 and show a few frames. The **barrel distortion** of the fisheye "
    "lens is visible toward the edges."
))
C.append(code(
    "frames = []\n"
    "try:\n"
    "    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:\n"
    "        tmp.write(first_mp4[1]); tmp_path = tmp.name\n"
    "    cap = cv2.VideoCapture(tmp_path)\n"
    "    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0\n"
    "    fps = cap.get(cv2.CAP_PROP_FPS)\n"
    "    print(f'{first_mp4[0]}: {n} frames @ {fps:.1f} fps')\n"
    "    for frac in (0.05, 0.5, 0.95):\n"
    "        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, n-1)*frac))\n"
    "        ok, fr = cap.read()\n"
    "        if ok: frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))\n"
    "    cap.release(); os.unlink(tmp_path)\n"
    "except Exception as e:\n"
    "    print('decode failed:', e)\n"
    "\n"
    "if frames:\n"
    "    fig, ax = plt.subplots(1, len(frames), figsize=(4*len(frames), 3))\n"
    "    ax = np.atleast_1d(ax)\n"
    "    for a, fr, t in zip(ax, frames, ['~start','~middle','~end']):\n"
    "        a.imshow(fr); a.set_title(f'{t}  {fr.shape[1]}x{fr.shape[0]}'); a.axis('off')\n"
    "    plt.tight_layout(); plt.show()\n"
    "else:\n"
    "    print('no frames decoded (cv2 HEVC backend?) — mp4 bytes still verified above')\n"
))

# ---- scale ----
C.append(md("## 7. Scale (measured + README)"))
C.append(code(
    "n_fac = len(factories)\n"
    "n_w_f1 = len(workers)\n"
    "clip_mp4_mb = first_mp4 and len(first_mp4[1])/1e6\n"
    "print(f'factories                 : {n_fac}')\n"
    "print(f'workers in factory001     : {n_w_f1}')\n"
    "print(f'part000.tar size          : {fs.info(part)[\"size\"]/1e9:.2f} GB')\n"
    "print(f'sample clip mp4           : {clip_mp4_mb:.1f} MB, {meta[\"duration_sec\"]}s, '\n"
    "      f'{meta[\"width\"]}x{meta[\"height\"]} @ {meta[\"fps\"]}fps, {meta[\"codec\"]}')\n"
    "print( 'README totals             : 100,405 h | 10.8B frames | 2,010,759 clips | 24.79 TB')\n"
))

# ---- filtering implications ----
C.append(md(
    "## 8. What this means for filtering\n"
    "\n"
    "**Video-only, no ground truth.** Each clip is just an mp4 + tiny metadata json — there is no "
    "hand pose / MANO / keypoints. So, unlike:\n"
    "- **EgoDex** (native in-tar GT) and\n"
    "- **taco / hot3d / oakink** (dataset GT → `use_gt` world_space_res),\n"
    "\n"
    "Egocentric-100K must go through the **reconstruction track**: Layer-1 pre-filter on the decoded "
    "frames → detect_track → motion → SLAM (DPVO) → infiller → `world_space_res.pth` → Layer-4 quality. "
    "There is no `use_gt` option here.\n"
    "\n"
    "**Fisheye camera.** Intrinsics are Kannala-Brandt (`k1..k4`), focal ≈138 (≠ pinhole W/2=228). "
    "The per-worker `intrinsics.json` must feed the reconstruction (undistort / fisheye-aware SLAM); "
    "treating frames as pinhole would corrupt camera + hand estimates.\n"
    "\n"
    "**Long clips.** Median 180 s → Layer-1's span/merge logic (Gate C) will sub-segment each clip "
    "into shorter valid intervals before reconstruction.\n"
    "\n"
    "**Scale.** ~2M clips / 24.79 TB across 238 factories → shard by factory/worker/part and run on "
    "the L4 fleet (same pattern as the EgoDex run), streaming shards from GCS.\n"
    "\n"
    "**Identity scheme.** clip key = `factory_XXX_worker_YYY_NNNN`; calibration is per-worker.\n"
    "\n"
    "→ Next step: a conversion path (`partNNN.tar` mp4 → frame tars + per-worker fisheye intrinsics) "
    "feeding Stage-1 → reconstruction → Stage-4, sharded on the fleet."
))

nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
               "language_info": {"name": "python"}}
OUT.write_text(nbf.writes(nb))
print("wrote", OUT)
