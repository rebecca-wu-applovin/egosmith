#!/usr/bin/env python3
"""Builder for notebooks/oakink_clipping_explainer.ipynb.

A visual walkthrough of HOW OakInk-v2 is clipped into grasp sub-clips, on one concrete
example sequence:
  1. the long raw sequence + why whole-sequence ingestion drops 100%,
  2. the program-primitive timeline (Gantt) that defines the cut,
  3. the per-primitive / acting-hand-only clipping rule,
  4. the resulting sub-clips with projected-hand overlays, kept vs dropped.

Render:
    python notebooks/_build_oakink_clipping_explainer_nb.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/oakink_clipping_explainer.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "oakink_clipping_explainer.ipynb"

EXAMPLE_TOKEN = "scene_01__A001++seq__07bb164dc3d3873d6389__2023-04-27-20-45-29"
EXAMPLE_CLIP = "OAKINK_scene_01_A001_seq_07bb164dc3d3873d6389_2023_04_27_20_45_29"


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


CELLS = []

CELLS.append(md(
    "# How OakInk-v2 is clipped into grasp clips\n"
    "\n"
    "A step-by-step visual walkthrough on one example sequence:\n"
    f"`{EXAMPLE_TOKEN}`.\n"
    "\n"
    "**The problem.** OakInk-v2 is a *long-horizon complex-task* dataset — each sequence is a\n"
    "full multi-step bimanual task recorded from a head-mounted **egocentric** camera. Feeding a\n"
    "whole sequence to the quality filter as one clip, judging **both** hands, drops 100% of the\n"
    "dataset: over tens of seconds each hand inevitably leaves the head-camera view for >30\n"
    "frames (`visible_*_out_of_frame_streak_exceeded`).\n"
    "\n"
    "**The fix (this notebook).** Cut each sequence into its **primitive segments** (which OakInk\n"
    "already annotates), keep one sub-clip per *(primitive, acting hand)*, and let the filter\n"
    "judge **only the acting hand**. Each sub-clip is a short reach-grasp-manipulate action —\n"
    "exactly what robotics grasping wants."
))

CELLS.append(code(
    "import json, ast, tarfile, io\n"
    "from pathlib import Path\n"
    "from collections import Counter\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "from matplotlib.patches import Patch\n"
    "from IPython.display import display, Image, Markdown\n"
    "\n"
    f"TOKEN = {EXAMPLE_TOKEN!r}\n"
    f"CLIP = {EXAMPLE_CLIP!r}\n"
    "OAK = Path('/root/oakink')\n"
    "PROG = json.loads((OAK / 'program_info' / f'{TOKEN}.json').read_text())\n"
    "GRASP = OAK / 'grasp' / 'filter_run'\n"
    "COLOR_FPS = 30.0; MOCAP_FPS = 120.0\n"
    "print('primitive segments in this sequence:', len(PROG))"
))

# ---------- Step 1: the raw sequence ----------
CELLS.append(md(
    "## Step 1 — the raw sequence is long\n"
    "\n"
    "The whole-sequence egocentric clip and its length. Frames are sampled evenly to show the\n"
    "task spanning many sub-actions. Note how, at any instant, often only one hand (or neither)\n"
    "is in the head-camera's view — the root cause of the 100% whole-sequence drop."
))

CELLS.append(code(
    "full_tar = OAK / 'frames' / f'{CLIP}.tar'\n"
    "with tarfile.open(full_tar, 'r') as tr:\n"
    "    members = sorted([m for m in tr if m.isfile() and m.name.endswith('.image.jpg')], key=lambda m: m.name)\n"
    "    T_full = len(members)\n"
    "    pick = np.linspace(0, T_full-1, 6, dtype=int)\n"
    "    from PIL import Image as PILImage\n"
    "    imgs = []\n"
    "    want = {members[i].name: i for i in pick}\n"
    "    cache = {}\n"
    "    with tarfile.open(full_tar, 'r') as tr2:\n"
    "        for m in tr2:\n"
    "            if m.name in want:\n"
    "                cache[want[m.name]] = PILImage.open(io.BytesIO(tr2.extractfile(m).read())).convert('RGB')\n"
    "print(f'whole sequence: {T_full} egocentric frames = {T_full/COLOR_FPS:.0f}s @ {COLOR_FPS:.0f}fps')\n"
    "fig, axes = plt.subplots(1, 6, figsize=(18, 3))\n"
    "for ax, i in zip(axes, pick):\n"
    "    ax.imshow(cache[i]); ax.set_title(f'f{i} ({i/COLOR_FPS:.0f}s)', fontsize=9); ax.axis('off')\n"
    "plt.suptitle(f'{CLIP}  —  one long task, whole-sequence filter verdict: DROPPED (both hands leave view)')\n"
    "plt.tight_layout(); plt.show()"
))

# ---------- Step 2: primitive timeline ----------
CELLS.append(md(
    "## Step 2 — OakInk already segments the task into primitives\n"
    "\n"
    "`program/program_info/<seq>.json` maps `\"(lh_interval, rh_interval)\" -> {primitive,\n"
    "interaction_mode, primitive_lh, primitive_rh, ...}`. Intervals are in **mocap frames**\n"
    "(120 fps). Below is the timeline: each bar is a primitive, on the row of the hand that acts\n"
    "(`primitive_lh`/`primitive_rh` non-null). This is the cut we use."
))

CELLS.append(code(
    "segs = []\n"
    "for key, seg in PROG.items():\n"
    "    lh_iv, rh_iv = ast.literal_eval(key)\n"
    "    for hand, iv in (('lh', lh_iv), ('rh', rh_iv)):\n"
    "        prim = seg.get(f'primitive_{hand}')\n"
    "        if prim and iv is not None:\n"
    "            segs.append({'hand': hand, 'primitive': prim, 'start_s': iv[0]/MOCAP_FPS, 'end_s': iv[1]/MOCAP_FPS})\n"
    "segs.sort(key=lambda s: s['start_s'])\n"
    "fig, ax = plt.subplots(figsize=(13, 2.6))\n"
    "rows = {'lh': 1, 'rh': 0}\n"
    "colors = {'lh': '#4285f4', 'rh': '#db4437'}\n"
    "for s in segs:\n"
    "    ax.barh(rows[s['hand']], s['end_s']-s['start_s'], left=s['start_s'], height=0.6,\n"
    "            color=colors[s['hand']], alpha=0.75, edgecolor='white')\n"
    "    ax.text(s['start_s'], rows[s['hand']], f\"  {s['primitive']}\", va='center', ha='left', fontsize=8, color='white', weight='bold')\n"
    "ax.set_yticks([0,1]); ax.set_yticklabels(['right hand', 'left hand'])\n"
    "ax.set_xlabel('time (s)'); ax.set_title(f'Primitive timeline — {len(segs)} acting-hand segments')\n"
    "ax.legend(handles=[Patch(color=colors['rh'], label='rh acts'), Patch(color=colors['lh'], label='lh acts')], loc='upper right')\n"
    "plt.tight_layout(); plt.show()\n"
    "for s in segs:\n"
    "    print(f\"  {s['primitive']:16s} {s['hand']}  {s['start_s']:6.1f}s – {s['end_s']:6.1f}s  ({s['end_s']-s['start_s']:.1f}s)\")"
))

# ---------- Step 3: the clipping rule ----------
CELLS.append(md(
    "## Step 3 — the clipping rule\n"
    "\n"
    "For each acting-hand primitive segment we emit **one sub-clip** "
    "(`scripts/build/generate_oakink_grasp_clips.py`):\n"
    "\n"
    "1. **Frames** = the egocentric frames inside the acting hand's interval, remapped to a fresh\n"
    "   contiguous tar `…__<primitive>_<hand>_<k>.tar` (reuses the on-disk whole-sequence tar —\n"
    "   no image re-download).\n"
    "2. **GT** = `world_space_res.pth` sliced to the interval.\n"
    "3. **Presence = acting hand only.** The pipeline's presence is a per-hand bitmask; every\n"
    "   off-screen metric only accumulates for a hand whose bit is set\n"
    "   (`quality/accumulator.py:239-277`). We set `valid[other_hand]=0`, so the filter judges\n"
    "   **only the grasping hand** — the resting / off-view hand can no longer force a drop.\n"
    "4. Non-prehensile actuation (`press_button`, …) and sub-clips <1 s are excluded.\n"
    "\n"
    "That single change — per-primitive granularity + acting-hand-only presence — turns the\n"
    "**0%** whole-sequence keep rate into **87.4%** across the dataset."
))

CELLS.append(code(
    "# the sub-clips this sequence produced, with kept/dropped from the filter report\n"
    "man = {}\n"
    "for line in (GRASP / 'clip_manifest.jsonl').read_text().splitlines():\n"
    "    if line.strip():\n"
    "        r = json.loads(line)\n"
    "        if r['clip_id'].startswith(CLIP):\n"
    "            man[r['clip_id']] = r['descriptor']['extra']\n"
    "rep = json.loads((GRASP / 'filter_report.json').read_text())\n"
    "dropped = {x['clip_id']: x['reasons'] for x in rep['dropped']}\n"
    "print(f'{len(man)} grasp sub-clips from this one sequence:')\n"
    "for cid, ex in sorted(man.items()):\n"
    "    verdict = 'DROP: ' + ', '.join(dropped[cid]) if cid in dropped else 'KEPT'\n"
    "    print(f\"  {ex['primitive']:14s} {ex['hand']}  ->  {verdict}\")"
))

# ---------- Step 4: overlays ----------
CELLS.append(md(
    "## Step 4 — the resulting sub-clips (projected GT hand overlaid)\n"
    "\n"
    "Each sub-clip below is one primitive. Dots are the **acting hand's** GT MANO joints\n"
    "projected into the egocentric view (blue = left, red = right); the wrist has a ring. This is\n"
    "exactly the signal the quality filter sees. Sheets are titled with the filter verdict.\n"
    "\n"
    "- The `take_outside` / `rearrange` clips are **kept**: short, the acting hand reaches in,\n"
    "  grasps, and stays in view — a clean grasp clip.\n"
    "- The `grip` clip is **dropped**: it's the 41 s grip-and-use span; even the acting hand\n"
    "  leaves the head-cam view for long stretches (yellow-outlined frames), tripping the\n"
    "  out-of-frame-streak rule. (Use `--grasp_onset_sec N` to keep only the grasp onset.)"
))

CELLS.append(code(
    "ov = OAK / 'grasp' / 'example_overlays'\n"
    "order = sorted(man.items(), key=lambda kv: (kv[0] in dropped, kv[0]))  # kept first\n"
    "for cid, ex in order:\n"
    "    verdict = 'DROPPED — ' + ', '.join(dropped[cid]) if cid in dropped else 'KEPT'\n"
    "    display(Markdown(f\"### `{ex['primitive']}` ({ex['hand']}) — **{verdict}**\"))\n"
    "    p = ov / f'{cid}.jpg'\n"
    "    if p.exists():\n"
    "        display(Image(filename=str(p), width=1150))\n"
    "    else:\n"
    "        display(Markdown('_overlay sheet not found_'))"
))

# ---------- Step 5: dataset-wide outcome ----------
CELLS.append(md(
    "## Step 5 — the same cut, applied to all 627 sequences\n"
    "\n"
    "Applying this per-primitive / acting-hand clipping across the whole dataset."
))

CELLS.append(code(
    "conv = json.loads((GRASP / 'convert_report.json').read_text())\n"
    "man_all = {}\n"
    "for line in (GRASP / 'clip_manifest.jsonl').read_text().splitlines():\n"
    "    if line.strip():\n"
    "        r = json.loads(line); man_all[r['clip_id']] = r['descriptor']['extra'].get('primitive','?')\n"
    "drop_ids = {x['clip_id'] for x in rep['dropped']}\n"
    "kept_p, tot_p = Counter(), Counter()\n"
    "for cid, prim in man_all.items():\n"
    "    tot_p[prim]+=1; kept_p[prim]+= (cid not in drop_ids)\n"
    "print(f\"627 sequences  ->  {conv['subclips_total']} grasp sub-clips\")\n"
    "print(f\"whole-sequence keep rate : 0 / 627   (0.0%)\")\n"
    "print(f\"grasp-clip keep rate     : {rep['kept_clips']} / {rep['total_clips']}   ({100*rep['kept_clips']/rep['total_clips']:.1f}%)\")\n"
    "rows = sorted(tot_p.items(), key=lambda kv: -kv[1])[:20]\n"
    "labels = [p for p,_ in rows]; keeps=[kept_p[p] for p in labels]; drops=[tot_p[p]-kept_p[p] for p in labels]\n"
    "fig, ax = plt.subplots(figsize=(11, max(3, 0.34*len(labels))))\n"
    "y=range(len(labels))\n"
    "ax.barh(y, keeps, color='#0f9d58', label='kept'); ax.barh(y, drops, left=keeps, color='#db4437', label='dropped')\n"
    "ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlabel('grasp sub-clips')\n"
    "ax.legend(loc='lower right'); ax.set_title('Grasp-clip yield per primitive (top 20)')\n"
    "for i,p in enumerate(labels):\n"
    "    ax.text(tot_p[p], i, f'  {kept_p[p]}/{tot_p[p]}', va='center', fontsize=8)\n"
    "plt.tight_layout(); plt.show()"
))

CELLS.append(md(
    "## Summary\n"
    "\n"
    "| | Whole sequence | Per-primitive grasp clips |\n"
    "|---|---|---|\n"
    "| Unit | 1 clip / task (~39 s) | 1 clip / (primitive, acting hand) (~1–7 s) |\n"
    "| Hands judged | both | acting hand only |\n"
    "| Clips | 627 | 3,999 |\n"
    "| **Kept** | **0 (0%)** | **3,494 (87.4%)** |\n"
    "\n"
    "The clipping is entirely driven by OakInk's own program annotations — no heuristics — and\n"
    "the acting-hand-only presence is what lets the standard quality filter accept them. Full\n"
    "per-primitive breakdown and failure galleries: `notebooks/oakink_grasp_filter_report.ipynb`."
))


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    OUT.write_text(nbf.writes(nb))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
