#!/usr/bin/env python3
"""Builder for notebooks/taco_filter_report.ipynb.

Constructs the TACO quality-filter analysis notebook programmatically (nbformat),
then renders it in place:

    python notebooks/_build_taco_filter_report_nb.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/taco_filter_report.ipynb

The notebook reads the artifacts produced by the full run under FILTER_RUN:
  - filter_report.json      (kept/dropped, per-clip reasons+metrics, resolved thresholds)
  - convert_report.json     (conversion + missing_modality pre-filter failures)
  - clip_manifest.jsonl     (triplet metadata per clip)
  - failures/<reason>/*.jpg  (contact sheets, if already generated)

Point ROOT at a different run to A/B compare (e.g. GT vs reconstruction).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "taco_filter_report.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


CELLS = []

CELLS.append(md(
    "# TACO quality-filter report\n"
    "\n"
    "The whole TACO dataset ingested with its **ground-truth** MANO hand poses and egocentric\n"
    "camera parameters (no pose inference), then run through the EgoSmith quality filter\n"
    "(`scripts/build/filter_manifest_by_quality.py`, `--stages infiller`). This notebook\n"
    "documents every filter rule and shows which sequences are dropped and why.\n"
    "\n"
    "**Note on language rules:** TACO ships no per-frame language instructions, so the\n"
    "instruction/annotation rules (`missing_instruction_frame`, `instruction_num_below_min`, ...)\n"
    "are intentionally **off** for this run. Every other rule is active."
))

CELLS.append(code(
    "import json, math\n"
    "from pathlib import Path\n"
    "from collections import Counter, defaultdict\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "from IPython.display import display, Image, Markdown\n"
    "\n"
    "ROOT = Path('/root/taco')\n"
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
    "print('missing_modality   :', len(convert.get('missing_modality', [])), '(never reach the filter)')"
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
    "| Instruction (OFF here) | `missing_instruction_frame`, `instruction_num_below_min`, ... | missing/empty language | off for TACO GT |\n"
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
    "    ax.set_title('TACO drop reasons'); plt.tight_layout(); plt.show()\n"
    "else:\n"
    "    display(Markdown('**No clips were dropped.**'))"
))

CELLS.append(md("## Kept vs dropped — pie + per-triplet breakdown"))

CELLS.append(code(
    "# clip_id -> triplet from the manifest\n"
    "triplet_by_clip = {}\n"
    "for line in (FILTER_RUN / 'clip_manifest.jsonl').read_text().splitlines():\n"
    "    if not line.strip():\n"
    "        continue\n"
    "    rec = json.loads(line)\n"
    "    extra = rec['descriptor'].get('extra', {})\n"
    "    triplet_by_clip[rec['clip_id']] = extra.get('triplet', '?')\n"
    "\n"
    "dropped_ids = {item['clip_id'] for item in dropped}\n"
    "per_triplet = defaultdict(lambda: [0, 0])  # triplet -> [kept, dropped]\n"
    "for clip_id, triplet in triplet_by_clip.items():\n"
    "    per_triplet[triplet][1 if clip_id in dropped_ids else 0] += 1\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(4,4))\n"
    "ax.pie([report['kept_clips'], report['dropped_clips']], labels=['kept','dropped'],\n"
    "       autopct='%1.1f%%', colors=['#0f9d58','#db4437']); ax.set_title('TACO clips'); plt.show()\n"
    "\n"
    "worst = sorted(per_triplet.items(), key=lambda kv: -(kv[1][1]/max(1,sum(kv[1]))))[:25]\n"
    "print('Triplets with highest drop rate (kept, dropped):')\n"
    "for triplet, (kept, drop) in worst:\n"
    "    if drop:\n"
    "        print(f'  {drop/(kept+drop):5.0%}  ({kept:2d},{drop:2d})  {triplet}')"
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
    "plt.suptitle('Per-frame motion metrics on DROPPED clips (red = threshold)')\n"
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
    "## Pre-filter failures — missing modality\n"
    "\n"
    "Sequences missing one of hand poses / camera params / RGB video never reach the filter.\n"
    "These are recorded in `convert_report.json` as their own class of \"filtered\" examples."
))

CELLS.append(code(
    "missing = convert.get('missing_modality', [])\n"
    "print(f'{len(missing)} sequences dropped before the filter (missing modality)')\n"
    "miss_counter = Counter()\n"
    "for m in missing:\n"
    "    for item in m['missing']:\n"
    "        miss_counter[item.split('/')[0]] += 1\n"
    "print('by missing component:', dict(miss_counter))\n"
    "for m in missing[:20]:\n"
    "    print(' ', m['triplet'], m['seq_name'], '->', m['missing'])\n"
    "conv_failures = convert.get('failures', [])\n"
    "if conv_failures:\n"
    "    print(f'\\n{len(conv_failures)} sequences failed conversion:')\n"
    "    for f in conv_failures[:20]:\n"
    "        print(' ', f['clip_id'], '::', f.get('error','')[:160])"
))

CELLS.append(md(
    "## Verdict\n"
    "\n"
    "See `filter_run/RULES.md` for the prose distillation. The dropped set falls into:\n"
    "**pre-filter** (missing modality), **numeric-sanity** (bad GT camera/pose frames),\n"
    "**glitch** (per-frame teleports/spins in the GT), **off-screen** (hand leaves the\n"
    "egocentric frame), and **extent outliers** (implausible camera-space depth vs the dataset)."
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
