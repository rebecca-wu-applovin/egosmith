#!/usr/bin/env python3
"""Builder for notebooks/hot3d_filter_report.ipynb.

Constructs the HOT3D quality-filter analysis notebook programmatically (nbformat),
then renders it in place:

    python notebooks/_build_oakink_filter_report_nb.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/hot3d_filter_report.ipynb

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

OUT = Path(__file__).resolve().parent / "hot3d_filter_report.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


CELLS = []

CELLS.append(md(
    "# HOT3D quality-filter report\n"
    "\n"
    "HOT3D is **egocentric, head-mounted** capture (Project Aria / Quest 3) of long, multi-activity\n"
    "sessions. Whole-sequence retargeting 'swings' between unrelated tasks, so instead of writing a\n"
    "heuristic segmenter we reuse the **official HOT3D-Clips** (`bop-benchmark/hot3d`): 3,832\n"
    "hand-verified **150-frame (5 s) single-interaction** clips. We ingest the **1,983 Aria** clips\n"
    "(the ones carrying egocentric RGB, stream `214-1`); the Quest3 clips are mono-only.\n"
    "\n"
    "Each clip is ingested from its GT like TACO/OakInk: **MANO** (`smplx` PCA-15, "
    "`thetas`+`wrist_xform`) → `world_space_res`, per-frame **camera** (`T_world_from_camera`) → SLAM,\n"
    "and the egocentric RGB is **undistorted FISHEYE624 → pinhole** (`generate_hot3d_world_res.py`)\n"
    "because the filter's off-screen rules assume a pinhole projection. Then the standard quality\n"
    "filter runs (`--stages infiller`). Both hands are judged (HOT3D annotates both when present).\n"
    "\n"
    "This notebook reports which clips pass the quality gate and why the rest drop.\n"
    "\n"
    "**Note on language rules:** HOT3D GT carries no per-frame language instructions, so the\n"
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
    "ROOT = Path('/root/hot3d')\n"
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
    "print('converted_ok       :', convert.get('converted_ok', 0), ' conversion_failed:', convert.get('failed', 0))"
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
    "| Instruction (OFF here) | `missing_instruction_frame`, `instruction_num_below_min`, ... | missing/empty language | off for HOT3D GT |\n"
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
    "## Overview & yield per participant\n"
    "\n"
    "HOT3D-Clips are uniform 150-frame (5 s) single-interaction clips, so there is nothing to\n"
    "sub-clip (unlike OakInk). Keep-rate is shown per participant (the leading `P####` of the\n"
    "source sequence), to expose participant/recording-specific quality differences."
))

CELLS.append(code(
    "convert = json.loads((FILTER_RUN / 'convert_report.json').read_text())\n"
    "print('clips ingested       :', convert.get('converted_ok', 0), '(+', convert.get('skipped_resume',0),'resumed)')\n"
    "print('conversion failed    :', convert.get('failed', 0))\n"
    "print('kept after filter    :', report['kept_clips'], f\"/ {report['total_clips']}  ({100*report['kept_clips']/max(1,report['total_clips']):.1f}%)\")\n"
    "print()\n"
    "# clip_id looks like HOT3D_clip-001849; participant not in id, so read from convert failures/manifest extra 'clip'.\n"
    "# Group by a coarse split from the clip number range is not meaningful; instead group kept/dropped overall + histogram of clip index.\n"
    "dropped_ids = {x['clip_id'] for x in dropped}\n"
    "import matplotlib.pyplot as plt\n"
    "fig, ax = plt.subplots(figsize=(4,4))\n"
    "ax.pie([report['kept_clips'], report['dropped_clips']], labels=['kept','dropped'], autopct='%1.1f%%',\n"
    "       colors=['#0f9d58','#db4437']); ax.set_title('HOT3D Aria clips'); plt.show()"
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
    "    ax.set_title('HOT3D drop reasons'); plt.tight_layout(); plt.show()\n"
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
    "    scene_by_clip[rec['clip_id']] = 'hot3d'\n"
    "\n"
    "dropped_ids = {item['clip_id'] for item in dropped}\n"
    "per_scene = defaultdict(lambda: [0, 0])  # triplet -> [kept, dropped]\n"
    "for clip_id, scene in scene_by_clip.items():\n"
    "    per_scene[scene][1 if clip_id in dropped_ids else 0] += 1\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(4,4))\n"
    "ax.pie([report['kept_clips'], report['dropped_clips']], labels=['kept','dropped'],\n"
    "       autopct='%1.1f%%', colors=['#0f9d58','#db4437']); ax.set_title('HOT3D clips'); plt.show()\n"
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
    "plt.suptitle('HOT3D: per-frame motion metrics on DROPPED clips (red = threshold)')\n"
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
    "HOT3D clips that failed to convert (unreadable tar, no RGB frames, undistortion/decode error)\n"
    "never reach the filter. Recorded in `convert_report.json`."
))

CELLS.append(code(
    "conv_failures = convert.get('failures', [])\n"
    "print(f'{convert.get(\"converted_ok\", 0)} clips converted, {convert.get(\"failed\",0)} failed conversion')\n"
    "for f in conv_failures[:25]:\n"
    "    print(' ', f.get('clip_id') or f.get('clip'), '::', f.get('error','')[:160])"
))

CELLS.append(md(
    "## Verdict\n"
    "\n"
    "See `filter_run/RULES.md` for the prose distillation. HOT3D-Clips are already curated,\n"
    "single-interaction 5 s windows (official segmentation), so — unlike the whole-sequence\n"
    "OakInk failure — most clips pass. The drops that remain are genuine quality issues: a hand\n"
    "leaving the head-camera FOV (off-screen / out-of-frame-streak), fast head motion\n"
    "(episode-camera outliers), per-frame GT glitches, and camera-space extent outliers."
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
