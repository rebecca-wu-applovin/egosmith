#!/usr/bin/env python3
"""Build the Ego-Exo4D EGO-stream video index for Stage-1 (cat1 video driver format).

Ego-Exo4D takes contain both ego (Aria) and exo (GoPro) frame-aligned streams.
The ego stream is the Aria RGB camera: frame_aligned_videos[<aria_cam>]["rgb"],
stream_id 214-1 (1408x1408 @30fps, fisheye). Verified visually 2026-08-25:
aria01_214-1.mp4 = first-person hands view; camNN.mp4 = wall-mounted exo.

Reads takes.json from the mirrored bucket, cross-checks each ego mp4 exists and
is non-zero on GCS, and writes one index row per take:
  {"uri": gs://.../takes/<take>/frame_aligned_videos/<cam>_214-1.mp4,
   "group": <take_name>, "session": <take_name>, "bytes": N}

group = take_name so Phase B runs with ID_MODE=group_basename (basenames like
aria01_214-1.mp4 collide across takes).

Usage:
  python scripts/build/index_egoexo4d_ego.py --out egoexo4d.index.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

V2 = "gs://foundational-research/hoi-dataset/Ego-Exo4D/egoexo-public/v2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--takes_json", default="", help="local takes.json (else pulled)")
    args = ap.parse_args()

    if args.takes_json:
        takes = json.load(open(args.takes_json))
    else:
        raw = subprocess.run(["gcloud", "storage", "cat", f"{V2}/takes.json"],
                             check=True, capture_output=True).stdout
        takes = json.loads(raw)

    # authoritative presence + size check straight from the bucket
    ls = subprocess.run(
        ["gcloud", "storage", "ls", "-l",
         f"{V2}/takes/*/frame_aligned_videos/*_214-1.mp4"],
        check=True, capture_output=True, text=True).stdout
    present: dict[str, tuple[str, int]] = {}
    for line in ls.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        m = re.search(r"/v2/(takes/[^/]+)/frame_aligned_videos/[^/]+$", parts[-1])
        if m:
            present[m.group(1)] = (parts[-1], int(parts[0]))

    rows, missing, zero = [], [], []
    for t in takes:
        fav = t.get("frame_aligned_videos") or {}
        ego = None
        for cam, streams in fav.items():
            if isinstance(streams, dict) and isinstance(streams.get("rgb"), dict):
                s = streams["rgb"]
                if s.get("stream_id") == "214-1":
                    ego = s["relative_path"]
                    break
        if ego is None:
            missing.append(t["take_name"] + " (no rgb stream in metadata)")
            continue
        got = present.get(t["root_dir"])
        if not got:
            missing.append(t["take_name"])
            continue
        url, size = got
        if size == 0:
            zero.append(t["take_name"])
            continue
        rows.append({"uri": url, "group": t["take_name"],
                     "session": t["take_name"], "bytes": size,
                     "duration_sec": t["duration_sec"]})

    rows.sort(key=lambda r: r["group"])
    with open(args.out, "w") as f:
        for r in rows:
            dur = r.pop("duration_sec")
            f.write(json.dumps(r) + "\n")
            r["duration_sec"] = dur
    hours = sum(r["duration_sec"] for r in rows) / 3600
    tb = sum(r["bytes"] for r in rows) / 1e12
    print(f"[egoexo4d-index] takes={len(takes)} indexed={len(rows)} "
          f"missing={len(missing)} zero_byte={len(zero)} "
          f"ego_hours={hours:.1f} ego_tb={tb:.3f} -> {args.out}")
    for name in (missing + zero)[:20]:
        print("  PROBLEM:", name, file=sys.stderr)
    if missing or zero:
        sys.exit(2)


if __name__ == "__main__":
    main()
