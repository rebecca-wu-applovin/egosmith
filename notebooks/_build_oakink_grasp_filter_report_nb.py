#!/usr/bin/env python3
"""Builder for notebooks/oakink_grasp_filter_report.ipynb.

Constructs the OakInk-v2 quality-filter analysis notebook programmatically (nbformat),
then renders it in place:

    python notebooks/_build_oakink_filter_report_nb.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/oakink_grasp_filter_report.ipynb

The notebook reads the artifacts produced by the full run under FILTER_RUN:
  - filter_report.json      (kept/dropped, per-clip reasons+metrics, resolved thresholds)
  - convert_report.json     (conversion + pre-filter failures)
  - clip_manifest.jsonl     (scene metadata per clip)
  - failures/<reason>/*.jpg  (contact sheets, if already generated)

Point ROOT at a different run to A/B compare (e.g. GT vs reconstruction).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "oakink_grasp_filter_report.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


CELLS = []

CELLS.append(md(
    "# OakInk-v2 GRASP-clip quality-filter report\n"
    "\n"
    "OakInk-v2 is a **long-horizon complex-task** dataset: each of the 627 sequences is a full\n"
    "multi-step bimanual task (median ~39 s), annotated as a program of ~4.5 **primitives**\n"
    "(`grip`, `rearrange`, `take_outside`, `place_inside`, `pour`, `scoop`, ...). Ingesting a\n"
    "whole sequence as one clip and judging **both** hands makes the EgoSmith quality filter\n"
    "drop **100%** on off-screen rules — over 39 s the resting / off-view hand always leaves the\n"
    "head-camera FOV for >30 frames.\n"
    "\n"
    "For **robotics grasping** we only want the manipulation primitives (each begins with a hand\n"
    "acquiring an object). So we cut every sequence into **per-primitive sub-clips** using\n"
    "OakInk's own program intervals (`generate_oakink_grasp_clips.py`), and set the presence\n"
    "bitmask so the filter judges **only the acting hand** (`interaction_mode` / `primitive_lh/rh`).\n"
    "Non-prehensile actuation (`press_button`, `use_mouse`, ...) and sub-clips <1 s are excluded.\n"
    "\n"
    "This notebook reports the yield of that grasp-clip run: which primitives survive the quality\n"
    "gate and why the rest drop. GT poses (quaternion MANO), egocentric camera, `--stages infiller`.\n"
    "\n"
    "**Note on language rules:** OakInk-v2 GT carries no per-frame language instructions, so the\n"
    "instruction/annotation rules are intentionally **off** for this run. Every other rule is active."
))

CELLS.append(code(
    "import json, math\n"
    "from pathlib import Path\n"
    "from collections import Counter, defaultdict\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "from IPython.display import display, Image, Markdown\n"
    "\n"
    "ROOT = Path('/root/oakink/grasp')\n"
    "FILTER_RUN = ROOT / 'filter_run'\n"
    "FAILURES_DIR = FILTER_RUN / 'failures'\n"
    "\n"
    "report = json.loads((FILTER_RUN / 'filter_report.json').read_text())\n"
    "convert = json.loads((FILTER_RUN / 'convert_report.json').read_text())\n"
    "criteria = report['criteria']\n"
    "dropped = report['dropped']\n"
    "print('total_clips        :', report['total_clips'])\n"
    "print('build_ready_clips  :', report['build_ready_clips'])\n"
    "print('kept_clips         :', report['kept_clips'])\n"
    "print('dropped_clips      :', report['dropped_clips'])\n"
    "print('  dropped_quality  :', report['dropped_quality_clips'])\n"
    "print('  build_invalid    :', report['build_invalid_clips'])\n"
    "print('failed_sequences   :', convert.get('failed_sequences', 0), ' skipped_short:', convert.get('skipped_short', 0))"
))

CELLS.append(md(
    "## The rules\n"
    "\n"
    "A clip is **kept** iff it triggers zero reasons (`decide_clip_quality`,\n"
    "`src/lib/pipeline/quality/decision.py`). Any single frame that trips a hard rule drops the\n"
    "whole sequence. Rules fall into groups:\n"
    "\n"
    "| Group | Reason string(s) | What it catches | Threshold |\n"
    "|---|---|---|---|\n"
    "| Numeric sanity (hard) | `invalid_lowdim`, `invalid_rot6d`, `invalid_extrinsic`, `invalid_intrinsic`, `nonfinite_lowdim` | NaN/Inf, non-unit rot6d, singular/non-homogeneous camera matrices, non-positive focal | any bad frame |\n"
    "| Off-screen (fatal) | `fatal_visible_left/right_severe_offscreen` | a visible hand projects entirely far outside the image | beyond `[-0.4W,1.4W]x[-0.4H,1.4H]` (`fatal_offscreen_scale=1.4`) |\n"
    "| In-frame ratio | `visible_left/right_inframe_ratio_below_min` | visible hand rarely has any joint inside the frame | ratio `< 0.2` |\n"
    "| Off-screen streak | `visible_left/right_out_of_frame_streak_exceeded` | long run of frames with the whole hand off-screen | streak `> 30` frames |\n"
    "| Per-frame motion (glitch) | `hand_translation_step_exceeded` | wrist teleports between frames | `> 0.30 m` |\n"
    "| | `finger_translation_step_exceeded` | fingertip teleports (max over 5 tips) | `> 0.30 m` |\n"
    "| | `camera_translation_step_exceeded` | camera teleports | `> 0.20 m` |\n"
    "| | `camera_rotation_step_exceeded` | camera spins between frames (‖R[t+1]-R[t]‖_F) | `> 0.70` (~28°) |\n"
    "| | `wrist_rotation_step_exceeded` | wrist root spins between frames | `> 0.99` (~41°) |\n"
    "| Camera-space extent | `camera_space_wrist/hand_iqr_bounds_exceeded` | hand/wrist lands implausibly far from camera vs the dataset | per-axis IQR fence (k=2.5) |\n"
    "| | `camera_space_wrist/hand_axis_abs_cap_exceeded` | fallback hard cap when no IQR fence | `|x/y/z| > 1.5 m` |\n"
    "| Episode motion | `episode_camera_translation/rotation_iqr_exceeded` | whole clip's mean camera motion is a dataset outlier | episode IQR fence (k=2.5) |\n"
    "| Presence | `presence_ratio_below_min` | too few frames with a hand present | off unless `--min_presence_ratio` set |\n"
    "| Instruction (OFF here) | `missing_instruction_frame`, `instruction_num_below_min`, ... | missing/empty language | off for OakInk-v2 GT |\n"
))

CELLS.append(code(
    "# Resolved thresholds actually applied this run (includes auto IQR bounds).\n"
    "keys = ['max_hand_translation_step','max_finger_translation_step','max_camera_translation_step',\n"
    "        'max_camera_rotation_step','max_wrist_rotation_step','fatal_offscreen_scale',\n"
    "        'min_visible_hand_any_point_inframe_ratio','max_visible_hand_all_points_out_of_frame_streak',\n"
    "        'camera_space_axis_abs_cap','camera_space_iqr_multiplier','episode_camera_iqr_multiplier',\n"
    "        'min_presence_ratio','min_instruction_num']\n"
    "for k in keys:\n"
    "    print(f'{k:48s} {criteria.get(k)}')\n"
    "print()\n"
    "print('camera_space_wrist_bounds :', json.dumps(criteria.get('camera_space_wrist_bounds'))[:400])\n"
    "print('episode_camera_translation_bounds :', criteria.get('episode_camera_translation_bounds'))\n"
    "print('episode_camera_rotation_bounds    :', criteria.get('episode_camera_rotation_bounds'))"
))

CELLS.append(md(
    "## Grasp-clip yield per primitive\n"
    "\n"
    "How many sub-clips of each primitive type survive the quality gate — the robotics-grasping\n"
    "usable subset. Joined from the manifest (primitive tag) and the filter report."
))

CELLS.append(code(
    "convert = json.loads((FILTER_RUN / 'convert_report.json').read_text())\n"
    "print('sequences            :', convert['sequences'])\n"
    "print('grasp sub-clips      :', convert['subclips_total'], f\"(from {convert['sequences']} sequences)\")\n"
    "print('skipped (<1s)        :', convert['skipped_short'])\n"
    "print('kept after filter    :', report['kept_clips'], f\"({100*report['kept_clips']/max(1,report['total_clips']):.1f}%)\")\n"
    "print()\n"
    "man_prim = {}\n"
    "for line in (FILTER_RUN / 'clip_manifest.jsonl').read_text().splitlines():\n"
    "    if line.strip():\n"
    "        rec = json.loads(line)\n"
    "        man_prim[rec['clip_id']] = rec['descriptor']['extra'].get('primitive', '?')\n"
    "dropped_ids = {x['clip_id'] for x in dropped}\n"
    "kept_p, total_p = Counter(), Counter()\n"
    "for cid, prim in man_prim.items():\n"
    "    total_p[prim] += 1\n"
    "    if cid not in dropped_ids:\n"
    "        kept_p[prim] += 1\n"
    "rows = sorted(total_p.items(), key=lambda kv: -kv[1])\n"
    "labels = [p for p, _ in rows]\n"
    "keeps = [kept_p[p] for p in labels]\n"
    "drops = [total_p[p]-kept_p[p] for p in labels]\n"
    "fig, ax = plt.subplots(figsize=(11, max(3, 0.32*len(labels))))\n"
    "y = range(len(labels))\n"
    "ax.barh(y, keeps, color='#0f9d58', label='kept')\n"
    "ax.barh(y, drops, left=keeps, color='#db4437', label='dropped')\n"
    "ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis()\n"
    "ax.set_xlabel('grasp sub-clips'); ax.legend(loc='lower right')\n"
    "ax.set_title('OakInk-v2 grasp-clip yield per primitive (kept vs dropped)')\n"
    "for i, p in enumerate(labels):\n"
    "    ax.text(total_p[p], i, f'  {kept_p[p]}/{total_p[p]} ({100*kept_p[p]/total_p[p]:.0f}%)', va='center', fontsize=8)\n"
    "plt.tight_layout(); plt.show()"
))

CELLS.append(md("## Drop reasons — histogram\n\nEach dropped clip may list several reasons; counts below are reason-occurrences."))

CELLS.append(code(
    "reason_counts = Counter()\n"
    "for item in dropped:\n"
    "    for r in item['reasons']:\n"
    "        reason_counts[r] += 1\n"
    "# clips per reason (a clip counted once per distinct reason)\n"
    "print('kept :', report['kept_clips'], ' dropped :', report['dropped_clips'],\n"
    "      f\"({100*report['dropped_clips']/max(1,report['total_clips']):.1f}% dropped)\")\n"
    "print()\n"
    "if reason_counts:\n"
    "    labels, values = zip(*sorted(reason_counts.items(), key=lambda kv: -kv[1]))\n"
    "    fig, ax = plt.subplots(figsize=(10, max(2, 0.4*len(labels))))\n"
    "    ax.barh(range(len(labels)), values, color='#db4437')\n"
    "    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)\n"
    "    ax.invert_yaxis(); ax.set_xlabel('clips triggering reason')\n"
    "    for i, v in enumerate(values):\n"
    "        ax.text(v, i, f' {v}', va='center')\n"
    "    ax.set_title('OakInk-v2 drop reasons'); plt.tight_layout(); plt.show()\n"
    "else:\n"
    "    display(Markdown('**No clips were dropped.**'))"
))

CELLS.append(md("## Kept vs dropped — pie + per-triplet breakdown"))

CELLS.append(code(
    "# clip_id -> scene from the manifest\n"
    "scene_by_clip = {}\n"
    "for line in (FILTER_RUN / 'clip_manifest.jsonl').read_text().splitlines():\n"
    "    if not line.strip():\n"
    "        continue\n"
    "    rec = json.loads(line)\n"
    "    extra = rec['descriptor'].get('extra', {})\n"
    "    scene_by_clip[rec['clip_id']] = extra.get('scene', '?')\n"
    "\n"
    "dropped_ids = {item['clip_id'] for item in dropped}\n"
    "per_scene = defaultdict(lambda: [0, 0])  # triplet -> [kept, dropped]\n"
    "for clip_id, scene in scene_by_clip.items():\n"
    "    per_scene[scene][1 if clip_id in dropped_ids else 0] += 1\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(4,4))\n"
    "ax.pie([report['kept_clips'], report['dropped_clips']], labels=['kept','dropped'],\n"
    "       autopct='%1.1f%%', colors=['#0f9d58','#db4437']); ax.set_title('OakInk-v2 clips'); plt.show()\n"
    "\n"
    "worst = sorted(per_scene.items(), key=lambda kv: -(kv[1][1]/max(1,sum(kv[1]))))[:25]\n"
    "print('Scenes with highest drop rate (kept, dropped):')\n"
    "for scene, (kept, drop) in worst:\n"
    "    if drop:\n"
    "        print(f'  {drop/(kept+drop):5.0%}  ({kept:2d},{drop:2d})  {scene}')"
))

CELLS.append(md("## Metric distributions vs thresholds\n\nWhere each rule's metric sits across the dataset; the red line is the drop threshold."))

CELLS.append(code(
    "# Collect per-clip metrics from build-ready clips (kept + quality-dropped carry metrics).\n"
    "metric_specs = [\n"
    "    ('max_hand_translation_step', criteria.get('max_hand_translation_step'), 'wrist step (m)'),\n"
    "    ('max_finger_translation_step', criteria.get('max_finger_translation_step'), 'fingertip step (m)'),\n"
    "    ('max_camera_translation_step', criteria.get('max_camera_translation_step'), 'camera step (m)'),\n"
    "    ('max_camera_rotation_step', criteria.get('max_camera_rotation_step'), 'camera rot step'),\n"
    "    ('max_wrist_rotation_step', criteria.get('max_wrist_rotation_step'), 'wrist rot step'),\n"
    "]\n"
    "# metrics for dropped clips are in the report; kept-clip metrics are not persisted per-clip,\n"
    "# so distributions below are over DROPPED clips (the tail we care about) plus any available.\n"
    "vals = defaultdict(list)\n"
    "for item in dropped:\n"
    "    m = item.get('metrics', {})\n"
    "    for key, _thr, _lab in metric_specs:\n"
    "        if key in m and isinstance(m[key], (int, float)):\n"
    "            vals[key].append(m[key])\n"
    "\n"
    "fig, axes = plt.subplots(1, len(metric_specs), figsize=(4*len(metric_specs), 3))\n"
    "for ax, (key, thr, lab) in zip(np.atleast_1d(axes), metric_specs):\n"
    "    data = vals.get(key, [])\n"
    "    if data:\n"
    "        ax.hist(data, bins=30, color='#4285f4')\n"
    "    if thr is not None:\n"
    "        ax.axvline(thr, color='#db4437', lw=2)\n"
    "    ax.set_title(lab, fontsize=9); ax.set_ylabel('dropped clips')\n"
    "plt.suptitle('OakInk-v2: per-frame motion metrics on DROPPED clips (red = threshold)')\n"
    "plt.tight_layout(); plt.show()"
))

CELLS.append(md(
    "## Example galleries — contact sheets per reason\n"
    "\n"
    "Each dropped clip has a contact sheet under `failures/<primary_reason>/<clip_id>.jpg`:\n"
    "sampled frames with projected GT hand joints (blue=left, red=right), offending frames\n"
    "outlined in yellow, reasons + metrics in the caption. A few per reason are shown below."
))

CELLS.append(code(
    "PER_REASON = 3\n"
    "if FAILURES_DIR.is_dir():\n"
    "    for reason_dir in sorted(FAILURES_DIR.iterdir()):\n"
    "        if not reason_dir.is_dir():\n"
    "            continue\n"
    "        sheets = sorted(reason_dir.glob('*.jpg'))\n"
    "        display(Markdown(f'### {reason_dir.name}  ({len(sheets)} clips)'))\n"
    "        for sheet in sheets[:PER_REASON]:\n"
    "            display(Image(filename=str(sheet), width=1100))\n"
    "else:\n"
    "    display(Markdown('_Contact sheets not generated yet — run `scripts/inspection/taco_overlay_sheets.py --filter_report ...`_'))"
))

CELLS.append(md(
    "## Pre-filter failures — conversion errors\n"
    "\n"
    "OakInk-v2 sequences that failed to convert (bad/unreadable annotation, no egocentric\n"
    "frames, decode error) never reach the filter. Recorded in `convert_report.json`."
))

CELLS.append(code(
    "conv_failures = convert.get('failures', [])\n"
    "print(f'{convert.get(\"subclips_total\", 0)} sub-clips built, {convert.get(\"failed_sequences\",0)} sequences failed, {convert.get(\"skipped_short\",0)} sub-clips skipped (<1s)')\n"
    "for f in conv_failures[:25]:\n"
    "    print(' ', f.get('clip_id') or f.get('seq_token'), '::', f.get('error','')[:160])"
))

CELLS.append(md(
    "## Verdict\n"
    "\n"
    "See `filter_run/RULES.md` for the prose distillation. For OakInk-v2 egocentric, the\n"
    "dominant drop cause is the hand leaving the head-camera's field of view (off-screen /\n"
    "out-of-frame-streak), a direct consequence of ingesting long bimanual task sequences from\n"
    "a first-person view. Remaining drops fall into **numeric-sanity** (bad GT frames),\n"
    "**glitch** (per-frame teleports/spins), and **extent outliers** (implausible camera-space\n"
    "depth vs the dataset)."
))


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    OUT.write_text(nbf.writes(nb))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
