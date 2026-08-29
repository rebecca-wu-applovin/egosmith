#!/usr/bin/env python
"""Close out (or incrementally mirror) a NATIVE-keypoints dataset into the standard
bucket layout consumed by the labeler and the viewer.

The native fleet (pod_entry_egoverse_native.sh) leaves per-shard artifacts under
  <ds>/native/_shards/shard_XXXXX[.<tag>].{manifest,stage1.kept,stage4.kept}.jsonl
  <ds>/frames/shard_XXXXX/<clip>.tar        (stage-4 survivors only)

This tool:
  * per shard: rewrites stage4.kept descriptors to bucket paths (root_dir/shard_path ->
    gs://.../frames/shard_XXXXX/, seq_folder cleared) and uploads
    filter_run/_shards/shard_XXXXX.filtered.jsonl  — the labeler's chase format.
    Already-mirrored shards are skipped (idempotent; run in a loop to chase the fleet).
  * with --finalize: concatenates everything into filter_run/clip_manifest{,.filtered}.jsonl,
    sums per-shard stage-4 reports into filter_report.json, writes funnel.json and
    BUCKET_AUDIT.json (kept count/hours, tar coverage sample, annotation coverage).

Usage:
  PYTHONPATH=src python scripts/build/gather_native_run.py --dataset egoverse_scale \
      --tag full [--loop 300] [--finalize] [--source_hours 213]
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import time

import gcsfs

BASE = "foundational-research/hoi-dataset"


def mirror_pending(fs, ds: str, tag: str) -> int:
    tag_sfx = f".{tag}" if tag else ""
    kept = fs.glob(f"{BASE}/egosmith_filtered/{ds}/native/_shards/shard_*{tag_sfx}.stage4.kept.jsonl")
    done = {p.rsplit("/", 1)[-1].split(".")[0]
            for p in fs.glob(f"{BASE}/egosmith_filtered/{ds}/filter_run/_shards/shard_*.filtered.jsonl")}
    n_new = 0
    for p in sorted(kept):
        sfx = p.rsplit("/", 1)[-1].split(".")[0]          # shard_XXXXX
        if sfx in done:
            continue
        rows = []
        txt = fs.cat(p).decode()
        for l in txt.splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            d = r["descriptor"]
            base = d["shard_path"].rsplit("/", 1)[-1]
            d["root_dir"] = f"gs://{BASE}/egosmith_filtered/{ds}/frames/{sfx}"
            d["shard_path"] = f"{d['root_dir']}/{base}"
            d["seq_folder"] = ""
            rows.append(json.dumps(r))
        with fs.open(f"{BASE}/egosmith_filtered/{ds}/filter_run/_shards/{sfx}.filtered.jsonl", "w") as f:
            f.write("\n".join(rows) + ("\n" if rows else ""))
        n_new += 1
        print(f"[mirror] {sfx}: {len(rows)} kept clips", flush=True)
    return n_new


def finalize(fs, ds: str, tag: str, source_hours: float | None, fps_default: float = 30.0):
    tag_sfx = f".{tag}" if tag else ""
    pre = f"{BASE}/egosmith_filtered/{ds}"
    # concat converted + kept manifests, sum reports
    conv_rows = kept_rows = 0
    conv_h = kept_h = 0.0
    s1_rows = 0
    with fs.open(f"{pre}/filter_run/clip_manifest.jsonl", "w") as out:
        for p in sorted(fs.glob(f"{pre}/native/_shards/shard_*{tag_sfx}.manifest.jsonl")):
            for l in fs.cat(p).decode().splitlines():
                if not l.strip():
                    continue
                r = json.loads(l)
                n = len(r["descriptor"].get("frame_names") or [])
                fps = float(r["descriptor"].get("fps") or fps_default)
                conv_h += n / fps / 3600
                conv_rows += 1
                out.write(l + "\n")
    for p in sorted(fs.glob(f"{pre}/native/_shards/shard_*{tag_sfx}.stage1.kept.jsonl")):
        s1_rows += sum(1 for l in fs.cat(p).decode().splitlines() if l.strip())
    rep_tot: dict = {}
    kept_lines = []
    for p in sorted(fs.glob(f"{pre}/filter_run/_shards/shard_*.filtered.jsonl")):
        for l in fs.cat(p).decode().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            n = len(r["descriptor"].get("frame_names") or [])
            fps = float(r["descriptor"].get("fps") or fps_default)
            kept_h += n / fps / 3600
            kept_rows += 1
            kept_lines.append(l)
    with fs.open(f"{pre}/filter_run/clip_manifest.filtered.jsonl", "w") as out:
        out.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))
    for p in fs.glob(f"{pre}/native/_shards/reports/shard_*/s4.json"):
        try:
            r = json.loads(fs.cat(p))
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    rep_tot[k] = rep_tot.get(k, 0) + v
        except Exception:  # noqa: BLE001
            pass
    with fs.open(f"{pre}/filter_run/filter_report.json", "w") as f:
        f.write(json.dumps(rep_tot, indent=1))
    # tar coverage sample
    sample = random.sample(kept_lines, min(200, len(kept_lines))) if kept_lines else []
    ok = 0
    for l in sample:
        r = json.loads(l)
        try:
            fs.info(r["descriptor"]["shard_path"][5:])
            ok += 1
        except FileNotFoundError:
            pass
    ann = fs.glob(f"{pre}/filter_run/annotations_v4/_shards/*.annotations.jsonl")
    n_ann = 0
    for p in ann:
        n_ann += sum(1 for l in fs.cat(p).decode().splitlines() if l.strip())
    funnel = {"dataset": ds, "path": "native_keypoints",
              "source_hours": source_hours, "converted_subclips": conv_rows,
              "converted_hours": round(conv_h, 2), "stage1_kept": s1_rows or None,
              "stage4_kept": kept_rows, "kept_hours": round(kept_h, 2),
              "generated_utc": datetime.datetime.utcnow().isoformat()}
    with fs.open(f"{pre}/filter_run/funnel.json", "w") as f:
        f.write(json.dumps(funnel, indent=1))
    audit = {**funnel, "tar_coverage_sample": f"{ok}/{len(sample)}",
             "annotated_clips": n_ann,
             "annotation_coverage": round(n_ann / max(1, kept_rows), 4)}
    with fs.open(f"{pre}/filter_run/BUCKET_AUDIT.json", "w") as f:
        f.write(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--loop", type=int, default=0, help=">0: chase mode, poll every N sec")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--source_hours", type=float, default=None)
    ap.add_argument("--fps_default", type=float, default=30.0)
    args = ap.parse_args()
    fs = gcsfs.GCSFileSystem()
    while True:
        n = mirror_pending(fs, args.dataset, args.tag)
        if not args.loop:
            break
        print(f"[mirror] pass done (+{n}); sleeping {args.loop}s", flush=True)
        time.sleep(args.loop)
    if args.finalize:
        finalize(fs, args.dataset, args.tag, args.source_hours, args.fps_default)


if __name__ == "__main__":
    main()
