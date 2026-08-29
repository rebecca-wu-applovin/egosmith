#!/usr/bin/env python
"""Repack Cat-1 Stage-1 video survivors into Phase-B input shards sized for the fleet.

Stage-1 shard spaces were sized for scan throughput (e.g. ego4d 2000 shards at
0.33 kept-h each); Phase B/C want ~TARGET_KEPT_H per shard so each whole-node recon
pod amortizes its ~12GB checkpoint pull. This reads every
  <FILT>/<ds>/stage1/_shards/shard_*.kept.jsonl
row, optionally joins tar-member offset/size/session from a ranged-read index
(HoloAssist), verifies clip_id-base uniqueness (same id_mode logic the converter
uses), greedy-packs whole videos into shards by kept_sec (bytes-capped so convert
pods fit ephemeral storage; oversize videos get a singleton shard), and writes
  <FILT>/<ds>/phaseB_input/_shards/shard_XXXXX.videos.jsonl   (+ _index.json summary)

Usage:
  PYTHONPATH=src python scripts/build/reshard_stage1_survivors.py \
      --dataset hd_epic --target_kept_h 0.5 [--tar_index holoassist.index.jsonl] \
      [--id_mode basename] [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts/build"))

import gcsfs

from generate_video_wds import clip_base  # same id logic as the converter

FILT = "foundational-research/hoi-dataset/egosmith_filtered"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target_kept_h", type=float, default=0.75)
    ap.add_argument("--max_shard_source_gb", type=float, default=25.0,
                    help="also break shards when accumulated source bytes exceed this")
    ap.add_argument("--id_mode", choices=["basename", "group_basename", "session"], default="basename")
    ap.add_argument("--tar_index", default="", help="JSONL uri->offset/size/session (HoloAssist)")
    ap.add_argument("--exclude_ids", default="",
                    help="JSON list of episode ids (basename sans extension/_video) to drop "
                         "before packing — e.g. mecka freeform ids shipping natively via "
                         "flagship GT (dedup by construction)")
    ap.add_argument("--out_prefix", default="", help="override gs output prefix (default per-dataset)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    fs = gcsfs.GCSFileSystem()
    ds = args.dataset
    shard_paths = sorted(fs.glob(f"{FILT}/{ds}/stage1/_shards/shard_*.kept.jsonl"))
    print(f"[{ds}] {len(shard_paths)} stage1 kept shards")
    rows = []
    for p in shard_paths:
        with fs.open(p) as f:
            for ln in f:
                if ln.strip():
                    rows.append(json.loads(ln))
    print(f"[{ds}] {len(rows)} survivor videos, "
          f"{sum(r['kept_sec'] for r in rows) / 3600:.2f} kept-h")

    if args.exclude_ids:
        excl = set(json.load(open(args.exclude_ids)))
        def _eid(uri: str) -> str:
            b = uri.rsplit("/", 1)[-1]
            for sfx in ("_video.mp4", ".mp4"):
                if b.endswith(sfx):
                    return b[: -len(sfx)]
            return b
        before_n, before_h = len(rows), sum(r["kept_sec"] for r in rows) / 3600
        rows = [r for r in rows if _eid(r["uri"]) not in excl]
        print(f"[{ds}] exclude_ids: {before_n} -> {len(rows)} videos, "
              f"{before_h:.2f} -> {sum(r['kept_sec'] for r in rows) / 3600:.2f} kept-h")

    # join tar-member fetch info (offset/size/session)
    if args.tar_index:
        idx = {}
        for ln in open(args.tar_index):
            if ln.strip():
                r = json.loads(ln)
                idx[r["uri"]] = r
        miss = 0
        for r in rows:
            j = idx.get(r["uri"])
            if j:
                r["offset"], r["size"] = j["offset"], j["size"]
                if j.get("session"):
                    r["session"] = j["session"]
            elif "::" in r["uri"]:
                miss += 1
        if miss:
            sys.exit(f"FATAL: {miss} tar-member uris missing from --tar_index")

    # clip-id-base uniqueness (collisions would silently overwrite tars in a shard dir)
    seen: dict[str, str] = {}
    for r in rows:
        b = clip_base(r, args.id_mode)
        if b in seen and seen[b] != r["uri"]:
            sys.exit(f"FATAL: clip base collision '{b}':\n  {seen[b]}\n  {r['uri']}\n"
                     f"-> use a stronger --id_mode")
        seen[b] = r["uri"]

    # source object sizes (bytes-cap packing + fleet ephemeral sizing).
    # Bulk-list each parent directory once (per-object fs.info was ~30 rows/s on
    # the 62K-video mecka dir) and fall back to fs.info only for cache misses.
    size_map: dict[str, int] = {}
    parents = sorted({r["uri"][5:].rsplit("/", 1)[0] for r in rows if "size" not in r})
    for d in parents:
        try:
            for e in fs.ls(d, detail=True):
                size_map[e["name"]] = int(e.get("size") or 0)
        except Exception:  # noqa: BLE001
            pass
    for r in rows:
        if "size" in r:
            r["_bytes"] = int(r["size"])
        elif r["uri"][5:] in size_map:
            r["_bytes"] = size_map[r["uri"][5:]]
        else:
            try:
                r["_bytes"] = int(fs.info(r["uri"][5:])["size"])
            except Exception:  # noqa: BLE001
                r["_bytes"] = 0

    rows.sort(key=lambda r: r["uri"])  # deterministic
    target_sec = args.target_kept_h * 3600
    max_bytes = args.max_shard_source_gb * 1e9
    shards: list[list[dict]] = [[]]
    acc_sec = acc_b = 0.0
    for r in rows:
        vb = r["_bytes"]
        if shards[-1] and (acc_sec + r["kept_sec"] > target_sec or acc_b + vb > max_bytes):
            shards.append([])
            acc_sec = acc_b = 0.0
        shards[-1].append(r)
        acc_sec += r["kept_sec"]
        acc_b += vb
    if not shards[-1]:
        shards.pop()

    kept_h = [sum(x["kept_sec"] for x in s) / 3600 for s in shards]
    gbs = [sum(x.get("_bytes", 0) for x in s) / 1e9 for s in shards]
    print(f"[{ds}] -> {len(shards)} phaseB shards | kept-h/shard min/med/max = "
          f"{min(kept_h):.2f}/{sorted(kept_h)[len(kept_h)//2]:.2f}/{max(kept_h):.2f} | "
          f"source-GB/shard max = {max(gbs):.1f}")
    out_prefix = args.out_prefix or f"{FILT}/{ds}/phaseB_input/_shards"
    if args.dry_run:
        print("[dry-run] not writing")
        return
    for i, s in enumerate(shards):
        for r in s:
            r["source_bytes"] = r.pop("_bytes")  # kept: convert pods balance procs by size
        body = "".join(json.dumps(r) + "\n" for r in s)
        with fs.open(f"{out_prefix}/shard_{i:05d}.videos.jsonl", "w") as f:
            f.write(body)
    summary = {"dataset": ds, "n_shards": len(shards), "n_videos": len(rows),
               "kept_hours": round(sum(kept_h), 2), "target_kept_h": args.target_kept_h,
               "id_mode": args.id_mode, "kept_h_per_shard_max": round(max(kept_h), 3),
               "source_gb_per_shard_max": round(max(gbs), 2)}
    with fs.open(f"{out_prefix}/_index.json", "w") as f:
        f.write(json.dumps(summary, indent=1))
    print(f"[{ds}] wrote {len(shards)} shards -> gs://{out_prefix}/")


if __name__ == "__main__":
    main()
