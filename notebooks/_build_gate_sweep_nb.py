#!/usr/bin/env python
"""Builder: gate_sweep_audit.ipynb — decomposition of our 24.9K kept-h vs EgoSteer's ~8K
on Egocentric-100K. Renders the threshold sweep (camera gate x min_hands) measured on a
60-part random raw sample and the funnel waterfall attributing the gap.

Usage: python notebooks/_build_gate_sweep_nb.py && \
       jupyter nbconvert --to notebook --execute --inplace notebooks/gate_sweep_audit.ipynb
Signals dir: produced by the session's gate_sweep/collect_signals.py (60 parts, 1,407
clips, 70.3 h); a copy of sweep_results.json is stored next to this builder.
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
nb = nbf.v4.new_notebook()
C = nb.cells

C.append(nbf.v4.new_markdown_cell(
    "# Egocentric-100K gate-threshold audit\n"
    "**Question:** we keep 24,960 h; EgoSteer reports ~8K h from the same dataset — is our "
    "filtering too lenient?\n\n"
    "**Method:** one decode+YOLO+flow pass at deployed Stage-1 settings over a random raw "
    "sample (60 part-tars across 60 factories, 1,407 clips, 70.3 h), recording per-frame "
    "signals; thresholds swept offline through the repo's own `_merge_valid_samples`.\n\n"
    "**Answer:** gates explain ~1% of the gap. Deployed (0.20, 2-hands) keeps 36.7% — "
    "matching the shipped funnel's 36.3% almost exactly — while the paper's stricter 0.10 "
    "camera gate keeps 36.3% (x1.01). The 3x gap is EgoSteer's undisclosed repetitive-video "
    "subsampling (implied keep ~1-in-3), stated in their paper without rate or method and "
    "absent from their released code."))

C.append(nbf.v4.new_code_cell(
    "import json, matplotlib.pyplot as plt\n"
    "from pathlib import Path\n"
    "res = json.loads((Path().resolve() / 'gate_sweep_results.json').read_text())\n"
    "CAM = [0.05, 0.10, 0.15, 0.20, 0.30]\n"
    "for mh in (1, 2):\n"
    "    print(f'min_hands={mh}: ' + '  '.join(f'{c:.2f}->{res[f\"{c}_{mh}\"]:.1%}' for c in CAM))"))

C.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "for mh, col in ((2, '#d62728'), (1, '#1f77b4')):\n"
    "    ys = [res[f'{c}_{mh}'] * 100 for c in CAM]\n"
    "    ax.plot(CAM, ys, 'o-', color=col, label=f'min_hands={mh}')\n"
    "ax.axvline(0.20, ls='--', c='gray', lw=1); ax.text(0.202, 40, 'deployed', fontsize=8)\n"
    "ax.axvline(0.10, ls=':', c='gray', lw=1); ax.text(0.102, 44, 'paper', fontsize=8)\n"
    "ax.axhline(36.3, ls='-', c='#2ca02c', lw=1, alpha=.5)\n"
    "ax.text(0.052, 34.4, 'shipped funnel 36.3%', fontsize=8, color='#2ca02c')\n"
    "ax.set_xlabel('camera_motion_thresh (fraction of long dim)')\n"
    "ax.set_ylabel('kept % of raw hours')\n"
    "ax.set_title('Stage-1 kept fraction vs gate thresholds (70.3 h raw sample)')\n"
    "ax.legend(); ax.grid(alpha=.3); plt.tight_layout()"))

C.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots(figsize=(8, 3.6))\n"
    "stages = ['raw', 'Stage-1/B spans\\n(gates ~= paper)', 'Phase-D quality\\n(x0.68)',\n"
    "          'EgoSteer subsample\\n(undisclosed, ~1-in-3)']\n"
    "ours =   [100405, 36486, 24960, None]\n"
    "theirs = [100405, 36486 * .363 / .367, 24960 * .363 / .367, 8000]\n"
    "x = range(len(stages))\n"
    "ax.bar([i - .2 for i in x], [v or 0 for v in ours], .38, label='ours (shipped)', color='#1f77b4')\n"
    "ax.bar([i + .2 for i in x], theirs, .38, label='EgoSteer (implied)', color='#ff7f0e')\n"
    "for i, v in enumerate(ours):\n"
    "    if v: ax.text(i - .2, v, f'{v/1000:.1f}K', ha='center', va='bottom', fontsize=8)\n"
    "for i, v in enumerate(theirs):\n"
    "    ax.text(i + .2, v, f'{v/1000:.1f}K', ha='center', va='bottom', fontsize=8)\n"
    "ax.set_xticks(list(x)); ax.set_xticklabels(stages, fontsize=8)\n"
    "ax.set_ylabel('hours'); ax.set_title('Where the 3x gap comes from')\n"
    "ax.legend(); plt.tight_layout()"))

C.append(nbf.v4.new_markdown_cell(
    "## Corroborating evidence\n"
    "- **Independent quality audit:** the gpt-5-mini annotator (sees actual frames) judged "
    "90.8% of a 165,324-clip random sample of our kept set `is_good_quality=true` — even "
    "dropping every flagged clip leaves ~22.6K h; no quality criterion reaches 8K.\n"
    "- **Paper:** \"To filter out highly repetitive videos, we subsample Egocentric-10K and "
    "Egocentric-100K\" — no rate/method given; Ego4D/EPIC-K use only VITRA subsets.\n"
    "- **Released code:** contains the gate stack we ran; no video-subsampling step.\n"
    "- **min_hands note:** a 1-hand variant would keep 54% of raw — our deployed 2-hand "
    "requirement is the *stricter* choice on this axis."))

json_src = Path("/tmp/claude-0/-root-egosmith/6b57c32e-2fab-4780-bf21-8479371681d9/scratchpad/gate_sweep/sweep_results.json")
(HERE / "gate_sweep_results.json").write_text(json_src.read_text())
nbf.write(nb, HERE / "gate_sweep_audit.ipynb")
print("built notebooks/gate_sweep_audit.ipynb")
