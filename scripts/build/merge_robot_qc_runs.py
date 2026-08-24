#!/usr/bin/env python3
"""Merge per-split robot_episode_qc.py runs into one combined filter_run.

robot_episode_qc.py runs one split-tar / data_root at a time, so datasets whose robot
episodes are sharded across several GCS prefixes (e.g. DexWild clothes_data/robot +
florist_data/robot) produce one manifest.jsonl + report.json per split. This tool:

- concatenates the kept manifests into a combined manifest.jsonl;
- re-applies the dedup gate ACROSS splits: the per-run content hash lives in each kept
  record's metadata.qc_metrics.content_hash (seen_hashes never crosses process
  boundaries), so cross-split duplicates are dropped here and reported;
- sums funnels / reason_counts / kept_hours into a combined report.json that keeps each
  split's full report under per_split{};
- passes through an optional notes JSON (e.g. cat4_notes) into the combined report.

Usage:
  merge_robot_qc_runs.py --dataset dexwild \
      --split clothes=/root/cat4_qc/dexwild/qc/clothes \
      --split florist=/root/cat4_qc/dexwild/qc/florist \
      --output_dir /root/cat4_qc/dexwild/qc/combined [--notes_json notes.json]

New file (CAT4 category-filtering sweep); does not modify robot_episode_qc.py gates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", action="append", required=True, metavar="NAME=DIR",
                    help="split name = dir containing manifest.jsonl + report.json (repeatable, order = merge order)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--notes_json", default=None, help="optional JSON file merged into the combined report as 'cat4_notes'")
    args = ap.parse_args(argv)

    splits = []
    for s in args.split:
        name, d = s.split("=", 1)
        splits.append((name, Path(d)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: dict[str, str] = {}
    kept_lines: list[str] = []
    cross_split_dupes: list[dict] = []
    per_split_reports: dict[str, dict] = {}
    funnel: "OrderedDict[str, dict]" = OrderedDict()
    reason_counts: Counter = Counter()
    totals = {"total_episodes": 0, "kept_episodes": 0, "dropped_episodes": 0, "kept_hours": 0.0}

    for name, d in splits:
        report = json.loads((d / "report.json").read_text())
        per_split_reports[name] = report
        totals["total_episodes"] += report["total_episodes"]
        totals["dropped_episodes"] += report["dropped_episodes"]
        totals["kept_hours"] += float(report.get("kept_hours") or 0.0)
        for gate, row in report["funnel"].items():
            agg = funnel.setdefault(gate, {"entered": 0, "dropped": 0, "passed": 0})
            for k in agg:
                agg[k] += row[k]
        reason_counts.update(report.get("reason_counts") or {})

        for line in (d / "manifest.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            digest = (((rec.get("metadata") or {}).get("qc_metrics")) or {}).get("content_hash")
            if digest and digest in seen_hashes:
                cross_split_dupes.append({
                    "episode_id": rec.get("clip_id"),
                    "split": name,
                    "duplicate_of": seen_hashes[digest],
                })
                dur = (((rec.get("metadata") or {}).get("qc_metrics")) or {}).get("duration_s") or 0.0
                totals["kept_hours"] -= float(dur) / 3600.0
                totals["dropped_episodes"] += 1
                reason_counts[f"dedup:cross_split:{name}"] += 1
                continue
            if digest:
                seen_hashes[digest] = rec.get("clip_id")
            kept_lines.append(line)

    totals["kept_episodes"] = len(kept_lines)
    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))

    combined = {
        "dataset": args.dataset,
        "merged_from_splits": [n for n, _ in splits],
        "output_manifest": str(manifest_path.resolve()),
        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in totals.items()},
        "funnel_combined": funnel,
        "reason_counts_combined": dict(sorted(reason_counts.items())),
        "cross_split_duplicates": cross_split_dupes,
        "per_split": per_split_reports,
    }
    if args.notes_json:
        combined["cat4_notes"] = json.loads(Path(args.notes_json).read_text())
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(combined, indent=2, default=str))

    print(f"=== {args.dataset} combined ({'+'.join(n for n, _ in splits)}) ===")
    print(f"  total {totals['total_episodes']}  kept {totals['kept_episodes']}  "
          f"kept_hours {totals['kept_hours']:.3f}  cross-split dupes {len(cross_split_dupes)}")
    print(f"manifest: {manifest_path}\nreport:   {report_path}")
    return combined


if __name__ == "__main__":
    main()
