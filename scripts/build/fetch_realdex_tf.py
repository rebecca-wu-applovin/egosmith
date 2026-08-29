#!/usr/bin/env python3
"""Fetch RealDex low-dim QC inputs from the GCS zip mirror via ranged reads.

RealDex object zips (gs://foundational-research/hoi-dataset/RealDex/<object>.zip)
hold, per sequence, ~1800 frames x 4 Azure Kinect cams plus small time-series dirs:

  storage/group/4dvlab/youzhuo/bags/<object>/<seq>/
      TF/*.txt                      (~100 small TF pair time series -- the QC signal)
      global_name_position/*.txt    (per-frame global joint name/position dumps)
      rgbimage_timestamp.txt
      cam{0..3}/rgb/image_raw/*.jpg (frames -- NOT downloaded here)
      cam{0..3}/depth_to_rgb/...    (depth -- NOT downloaded here)

robot_episode_qc.py's realdex adapter (iter_realdex) reconstructs joint streams from
``<data_root>/<object>/<seq>/TF/*.txt`` and reads a synthesized ``frame_counts.json``
({"rgb_frame_counts": {cam: n}, "global_name_position_frames": n}) for the video_sync
gate. This tool produces exactly that layout WITHOUT downloading the multi-GB zips:
the zip central directory is read over gcsfs, only TF/*.txt + rgbimage_timestamp.txt
members are decompressed (a few MB per zip), and the frame counts come from the
namelist (cam*/rgb/image_raw/* and global_name_position/* entry counts).

Rebuilt 2026-08-24: the original Cat-4 fetch step (Aug 19-20) lived in /tmp and was
lost to a container restart; this reimplementation matches its verified on-disk
output (diffed against the surviving air_duster/bathroom_cleaner/beer extracts).

Usage:
  fetch_realdex_tf.py --objects blue_cup,box --out_root /root/cat4_qc/realdex
  fetch_realdex_tf.py --all --skip_existing --out_root /root/cat4_qc/realdex
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

GCS_PREFIX = "foundational-research/hoi-dataset/RealDex"
_SEQ_RE = re.compile(r"^storage/group/4dvlab/youzhuo/bags/([^/]+)/([^/]+)/(.+)$")


def list_mirror_zips(fs) -> list[str]:
    return sorted(
        Path(p).stem for p in fs.ls(GCS_PREFIX) if p.endswith(".zip")
    )


def fetch_object(fs, obj: str, out_root: Path, skip_existing: bool) -> dict:
    obj_dir = out_root / obj
    result = {"object": obj, "sequences": {}, "status": "ok"}
    if skip_existing and obj_dir.is_dir() and any(
        (d / "frame_counts.json").is_file() for d in obj_dir.iterdir() if d.is_dir()
    ):
        result["status"] = "skipped_existing"
        return result
    with fs.open(f"{GCS_PREFIX}/{obj}.zip", "rb") as f:
        zf = zipfile.ZipFile(f)
        names = zf.namelist()
        seq_files: dict[str, list[str]] = {}
        rgb_counts: dict[str, Counter] = {}
        gnp_counts: Counter = Counter()
        for n in names:
            m = _SEQ_RE.match(n)
            if not m or n.endswith("/"):
                continue
            _, seq, rel = m.groups()
            if rel.startswith("TF/") and rel.endswith(".txt") or rel == "rgbimage_timestamp.txt":
                seq_files.setdefault(seq, []).append(n)
            parts = rel.split("/")
            if len(parts) == 4 and parts[1] == "rgb" and parts[2] == "image_raw":
                rgb_counts.setdefault(seq, Counter())[parts[0]] += 1
            elif len(parts) == 2 and parts[0] == "global_name_position":
                gnp_counts[seq] += 1
        for seq, members in sorted(seq_files.items()):
            seq_dir = obj_dir / seq
            (seq_dir / "TF").mkdir(parents=True, exist_ok=True)
            for member in members:
                rel = _SEQ_RE.match(member).group(3)
                target = seq_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
            fc = {
                "rgb_frame_counts": dict(sorted(rgb_counts.get(seq, {}).items())),
                "global_name_position_frames": int(gnp_counts.get(seq, 0)),
            }
            (seq_dir / "frame_counts.json").write_text(json.dumps(fc))
            result["sequences"][seq] = {
                "tf_files": sum(1 for m in members if "/TF/" in m),
                **fc,
            }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objects", default=None, help="Comma-separated object names (zip stems)")
    ap.add_argument("--all", action="store_true", help="Process every zip currently in the mirror")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip objects that already have an extracted frame_counts.json")
    args = ap.parse_args()

    import gcsfs

    fs = gcsfs.GCSFileSystem()
    if args.all:
        objects = list_mirror_zips(fs)
    elif args.objects:
        objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    else:
        ap.error("pass --objects or --all")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for obj in objects:
        try:
            res = fetch_object(fs, obj, out_root, args.skip_existing)
        except Exception as error:  # noqa: BLE001
            res = {"object": obj, "status": f"failed: {type(error).__name__}: {error}"}
            failures += 1
        print(json.dumps(res), flush=True)
    print("FETCH_REALDEX_DONE" if not failures else "FETCH_REALDEX_DONE_WITH_FAILURES", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
