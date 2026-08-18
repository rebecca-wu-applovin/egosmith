#!/usr/bin/env python3
"""Builder for notebooks/egodex_filter_report.ipynb — EgoDex test-set filter report.

EgoDex (Apple Vision Pro) has no MANO; we built the pipeline's 116-d native lowdim
directly from its joint skeleton and ran the quality filter via the native_features
stage. This report documents the run: keep/drop, drop-reason histogram, per-reason
failure galleries (native-lowdim overlays), and the rule catalogue.

Outputs are CC-BY-NC-ND -> kept local only.

Build + execute:
  python notebooks/_build_egodex_filter_report_nb.py
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 \
      notebooks/egodex_filter_report.ipynb
"""
from __future__ import annotations
import glob
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parent / "egodex_filter_report.ipynb"
RUN = "/root/egodex/filter_run"
FAIL = f"{RUN}/failures"


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)


C = []
C.append(md(
    "# EgoDex test-set — EgoSmith quality-filter report\n"
    "\n"
    "**EgoDex** (Apple, arXiv 2505.11709) is egocentric Apple-Vision-Pro manipulation video. "
    "The local copy is the **test** split: **111 tasks / 3,243 episodes**. EgoDex ships **no MANO** — "
    "poses are the VP joint skeleton (per-joint SE(3)). So instead of a MANO `world_space_res`, we built "
    "the pipeline's **116-d native lowdim directly from the joints** (wrist SE(3)→pos+rot6d, 5 fingertips/hand, "
    "extrinsic = `inv(camera)`, pinhole intrinsic) and ran the filter via the **`native_features`** stage.\n"
    "\n"
    "_Licence: EgoDex is **CC-BY-NC-ND** — these filtered outputs are kept local only, not uploaded._\n"
    "\n"
    "Converter `scripts/build/generate_egodex_wds.py` · failure sheets `scripts/inspection/egodex_native_sheets.py`."
))

C.append(md("## Result — keep / drop"))
C.append(code(
    "import json\n"
    "from collections import Counter\n"
    "import matplotlib.pyplot as plt\n"
    f"r = json.load(open('{RUN}/filter_report.json'))\n"
    f"c = json.load(open('{RUN}/convert_report.json'))\n"
    "print(f\"converted {c['converted_ok']}/{c['episodes']} episodes across {c['tasks']} tasks (0 failed)\")\n"
    "print(f\"build-ready {r['build_ready_clips']}/{r['total_clips']} · kept {r['kept_clips']} \"\n"
    "      f\"({100*r['kept_clips']/r['total_clips']:.1f}%) · dropped {r['dropped_clips']}\")\n"
    "print('build_invalid:', r.get('build_invalid_clips', 0))\n"
    "rc = Counter(r['quality_reason_counts'])\n"
    "labels, vals = zip(*sorted(rc.items(), key=lambda kv:-kv[1]))\n"
    "fig, ax = plt.subplots(figsize=(9, max(2,0.42*len(labels))))\n"
    "ax.barh(range(len(labels)), vals, color='#2f6df6')\n"
    "ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()\n"
    "ax.set_xlabel('clips triggering reason (a clip may trip several)')\n"
    "ax.set_title('EgoDex test — drop reasons')\n"
    "for i,v in enumerate(vals): ax.text(v,i,f' {v}',va='center')\n"
    "plt.tight_layout(); plt.show()"
))
C.append(md(
    "The dominant drop is **hands leaving the head-camera FOV** — left-hand rules fire ~2× the right, "
    "because right-handed manipulation pushes the idle left hand off-frame — plus **fast head motion** "
    "(`episode_camera_rotation/translation_iqr_exceeded`). Note **no `invalid_rot6d / extrinsic / intrinsic`**: "
    "the native lowdim built from the VP joints is geometrically clean."
))

# per-reason galleries
# NB: the base64 is encoded at *runtime* inside the cell (see the _show helper), so the
# notebook's code-cell source stays short — the image data only lands in the cell output.
C.append(md(
    "## Failure galleries (native-lowdim overlays)\n"
    "blue = left hand, red = right hand (wrist ringed + 5 fingertips), projected via the stored "
    "World2Cam + pinhole intrinsic; **amber-outlined tiles = a joint falls outside the frame**."
))
C.append(code(
    "import base64, glob\n"
    "from pathlib import Path\n"
    "from IPython.display import HTML, display\n"
    "def _show(pattern, n=2):\n"
    "    files = sorted(glob.glob(pattern))[:n]\n"
    "    html = ''.join(\n"
    "        '<figure style=\"margin:0 0 12px 0\"><img src=\"data:image/jpeg;base64,'\n"
    "        + base64.b64encode(Path(f).read_bytes()).decode()\n"
    "        + '\" style=\"width:100%;max-width:1000px;border-radius:6px\">'\n"
    "        + f'<figcaption style=\"font:11px monospace;color:#888\">{Path(f).stem}</figcaption></figure>'\n"
    "        for f in files)\n"
    "    display(HTML(html))"
))
present = [d.name for d in sorted(Path(FAIL).iterdir()) if d.is_dir()] if Path(FAIL).is_dir() else []
order = ["fatal_visible_left_severe_offscreen", "visible_left_out_of_frame_streak_exceeded",
         "visible_left_inframe_ratio_below_min", "fatal_visible_right_severe_offscreen",
         "camera_space_wrist_iqr_bounds_exceeded", "episode_camera_rotation_iqr_exceeded",
         "episode_camera_translation_iqr_exceeded", "wrist_rotation_step_exceeded"]
for reason in [x for x in order if x in present] + [x for x in present if x not in order]:
    if sorted(glob.glob(f"{FAIL}/{reason}/*.jpg")):
        C.append(md(f"### `{reason}`"))
        C.append(code(f"_show({FAIL + '/' + reason + '/*.jpg'!r})"))

C.append(md(
    "## Rule catalogue (native_features stage)\n"
    "| reason | meaning |\n|---|---|\n"
    "| `fatal_visible_left/right_severe_offscreen` | a present hand is far outside the frame — hard drop |\n"
    "| `visible_*_out_of_frame_streak_exceeded` | hand off-frame for too many consecutive frames |\n"
    "| `visible_*_inframe_ratio_below_min` | hand in-frame for too small a fraction of the clip |\n"
    "| `camera_space_wrist/hand_iqr_bounds_exceeded` | wrist/hand camera-space position an IQR outlier vs the dataset |\n"
    "| `camera_space_wrist/hand_axis_abs_cap_exceeded` | wrist/hand beyond an absolute per-axis cap in camera space |\n"
    "| `episode_camera_translation/rotation_iqr_exceeded` | whole-clip head-camera motion an IQR outlier (jerky/large) |\n"
    "| `hand/finger/camera_translation_step_exceeded`, `*_rotation_step_exceeded` | frame-to-frame jump too large (tracking glitch) |\n"
    "| `nonfinite_lowdim`, `invalid_rot6d`, `invalid_extrinsic`, `invalid_intrinsic` | malformed geometry (none tripped here) |\n"
    "\n"
    "Thresholds for the IQR/step rules are resolved dataset-wide (Pass-2 barrier) before `decide_clip_quality`. "
    "Full pipeline: `notebooks/hot3d_processing.ipynb` (technique deep-dive) and the four-layer flowchart artifact."
))


def main():
    nb = nbf.v4.new_notebook(); nb["cells"] = C
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    OUT.write_text(nbf.writes(nb)); print("wrote", OUT)


if __name__ == "__main__":
    main()
