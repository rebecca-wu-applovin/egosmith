#!/usr/bin/env python
"""Close out a recon-path dataset into the standard bucket layout.

Aggregates per-shard artifacts under gs://.../egosmith_filtered/<ds>/ into:
  filter_run/clip_manifest.jsonl           all converted sub-clips (phaseB concat)
  filter_run/clip_manifest.filtered.jsonl  Phase-D keeps (filter_run/_shards concat)
  filter_run/filter_report.json            summed drop-reason counts across shards
  filter_run/funnel.json                   episodes -> subclips -> recon -> kept (+hours)
  filter_run/BUCKET_AUDIT.json             one audit row: kept count/hours + annotation coverage

Usage: PYTHONPATH=src python scripts/build/gather_filter_run.py --dataset egotouch \
           [--source_hours 17.99] [--recon_out egosmith_recon/<ds>/recon/outputs]
"""
from __future__ import annotations

import argparse
import datetime
import json

import gcsfs

BASE = "foundational-research/hoi-dataset"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source_hours", type=float, default=None)
    ap.add_argument("--recon_out", default=None)
    args = ap.parse_args()
    ds = args.dataset
    fs = gcsfs.GCSFileSystem()
    filt_base = f"{BASE}/egosmith_filtered/{ds}"
    recon_out = args.recon_out or f"{BASE}/egosmith_recon/{ds}/recon/outputs"

    # phaseB manifests -> clip_manifest.jsonl
    b_shards = sorted(fs.glob(f"{filt_base}/phaseB/_shards/*.manifest.jsonl"))
    conv_rows, conv_h = 0, 0.0
    with fs.open(f"{filt_base}/filter_run/clip_manifest.jsonl", "w") as out:
        for p in b_shards:
            txt = fs.cat(p).decode()
            for l in txt.splitlines():
                if not l.strip():
                    continue
                r = json.loads(l)
                d = r["descriptor"]
                fps = float(d.get("fps") or d.get("extra", {}).get("recon_fps") or 15)
                n = len(d.get("frame_names") or []) or int(d.get("frame_count") or 0)
                conv_h += n / fps / 3600
                conv_rows += 1
                out.write(l + "\n")

    # recon done markers
    dones = fs.glob(f"{recon_out}/_done/*.json")
    recon_ok = 0
    for p in dones:
        try:
            recon_ok += int(json.loads(fs.cat(p)).get("succeeded", 0))
        except Exception:  # noqa: BLE001
            pass

    # filtered keeps
    f_shards = sorted(fs.glob(f"{filt_base}/filter_run/_shards/*.filtered.jsonl"))
    kept_rows, kept_h = 0, 0.0
    with fs.open(f"{filt_base}/filter_run/clip_manifest.filtered.jsonl", "w") as out:
        for p in f_shards:
            txt = fs.cat(p).decode()
            for l in txt.splitlines():
                if not l.strip():
                    continue
                r = json.loads(l)
                d = r["descriptor"]
                fps = float(d.get("fps") or d.get("extra", {}).get("recon_fps") or 15)
                n = len(d.get("frame_names") or []) or int(d.get("frame_count") or 0)
                kept_h += n / fps / 3600
                kept_rows += 1
                out.write(l + "\n")

    # aggregate drop reasons + filtered totals
    agg: dict = {"kept_clips": 0, "dropped_clips": 0, "build_invalid_clips": 0,
                 "quality_reason_counts": {}}
    for p in fs.glob(f"{filt_base}/filter_run/_shards/*.report.json"):
        try:
            r = json.loads(fs.cat(p))
        except Exception:  # noqa: BLE001
            continue
        for k in ("kept_clips", "dropped_clips", "build_invalid_clips"):
            agg[k] += int(r.get(k) or 0)
        for k, v in (r.get("quality_reason_counts") or {}).items():
            agg["quality_reason_counts"][k] = agg["quality_reason_counts"].get(k, 0) + v
    fs.pipe(f"{filt_base}/filter_run/filter_report.json", json.dumps(agg, indent=1).encode())

    # annotation coverage: merge per-shard files, dedupe by clip_id (concurrent
    # labelers can race a shard), write the merged annotations.v4.jsonl
    seen_ann = set()
    with fs.open(f"{filt_base}/filter_run/annotations.v4.jsonl", "w") as out:
        for p in sorted(fs.glob(f"{filt_base}/filter_run/annotations_v4/_shards/*.annotations.jsonl")):
            for l in fs.cat(p).decode().splitlines():
                if not l.strip():
                    continue
                cid = json.loads(l).get("clip_id")
                if cid in seen_ann:
                    continue
                seen_ann.add(cid)
                out.write(l + "\n")
    ann = len(seen_ann)

    funnel = {
        "dataset": ds,
        "source_hours": args.source_hours,
        "phaseB_subclips": conv_rows,
        "phaseB_hours": round(conv_h, 2),
        "recon_shards_done": len(dones),
        "recon_succeeded_clips": recon_ok,
        "filter_shards_done": len(f_shards),
        "final_kept_clips": kept_rows,
        "final_kept_hours": round(kept_h, 2),
        "keep_rate_of_filtered": round(kept_rows / max(1, agg["kept_clips"] + agg["dropped_clips"] + agg["build_invalid_clips"]), 4),
        "annotations_v4": ann,
    }
    fs.pipe(f"{filt_base}/filter_run/funnel.json", json.dumps(funnel, indent=1).encode())
    audit = {
        "dataset": ds,
        "audited_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bucket_prefix": f"gs://{filt_base}/filter_run/",
        "kept_clips": kept_rows,
        "kept_hours": round(kept_h, 2),
        "annotations_v4": ann,
        "annotation_coverage": round(ann / max(1, kept_rows), 4),
    }
    fs.pipe(f"{filt_base}/filter_run/BUCKET_AUDIT.json", json.dumps(audit, indent=1).encode())
    print(json.dumps({"funnel": funnel, "audit": audit}, indent=1))


if __name__ == "__main__":
    main()
