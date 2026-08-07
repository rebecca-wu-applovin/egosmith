#!/usr/bin/env python
"""Stage-1 pre-filter (EgoSmith pipeline step 1) over a clip manifest of FRAME TARS.

Runs the heuristic_video_clipper gates — Gate A (YOLO hand presence + size + ROI) and
Gate B (optical-flow RANSAC ego-motion stability), merged by Gate C (valid-span) — on
each clip's extracted frames (no raw video needed). A clip is KEPT if it contains at
least one valid span (hands adequately sized + camera stable for >= min_keep_sec);
otherwise it is dropped (locomotion / hands-too-small / no-hands / unstable camera).

Usage (egosmith env, GPU):
  python scripts/build/stage1_prefilter.py \
      --input_manifest  clip_manifest.filtered.jsonl \
      --output_manifest stage1.kept.jsonl \
      --report_out      stage1.report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from lib.clip.heuristic_video_clipper import (  # noqa: E402
    load_clip_config, analyze_frame_source_intervals, _load_yolo,
)
from lib.pipeline.clips.clip_manifest import load_clip_manifest, write_clip_manifest  # noqa: E402
from lib.pipeline.io.frame_sources import build_frame_source_from_descriptor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_manifest", required=True)
    ap.add_argument("--output_manifest", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--config", default=str(_REPO / "src/lib/clip/heuristic_clip_config.yaml"))
    ap.add_argument("--detector", default=str(_REPO / "weights/external/detector.pt"))
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    cfg = load_clip_config(args.config)
    model = _load_yolo(args.detector)
    if model is None:
        raise SystemExit(f"YOLO detector not found at {args.detector}")

    records = load_clip_manifest(args.input_manifest)
    kept, dropped = [], []
    for rec in records:
        clip = rec.descriptor.clip_id
        try:
            fs = build_frame_source_from_descriptor(rec.descriptor)
            intervals, info = analyze_frame_source_intervals(fs, cfg, model=model, fps=args.fps)
            valid_frac = (info["valid_sample_count"] / info["sample_count"]) if info["sample_count"] else 0.0
            if intervals:
                kept.append(rec)
            else:
                dropped.append({"clip_id": clip, "reason": "no_valid_span",
                                "valid_frac": round(valid_frac, 3), "samples": info["sample_count"]})
        except Exception as e:  # noqa: BLE001
            dropped.append({"clip_id": clip, "reason": f"error:{str(e)[:80]}"})

    write_clip_manifest(kept, args.output_manifest)
    total = len(records)
    report = {
        "input_manifest": args.input_manifest,
        "total_clips": total,
        "kept_clips": len(kept),
        "dropped_clips": len(dropped),
        "kept_pct": round(100.0 * len(kept) / max(1, total), 1),
        "dropped": dropped,
    }
    Path(args.report_out).write_text(json.dumps(report, indent=1))
    print(f"[stage1] {len(kept)}/{total} kept ({report['kept_pct']}%)  -> {args.output_manifest}")
    from collections import Counter
    c = Counter(d["reason"].split(":")[0] for d in dropped)
    for k, v in c.most_common():
        print(f"    dropped[{k}]: {v}")


if __name__ == "__main__":
    main()
