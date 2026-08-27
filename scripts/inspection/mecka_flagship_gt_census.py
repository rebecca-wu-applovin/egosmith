#!/usr/bin/env python
"""Census GT-keypoint presence for every mecka/flagship zarr episode.

An episode counts as GT-BEARING when both hands' obs_keypoints arrays have
data chunks covering (approximately) the full video length: the kpt audit
(2026-08-27) found ~44% of episodes carry zarr.json only / truncated arrays
(e.g. shape [37,63] vs 3,046 frames).

Output: JSONL rows {ep_id, total_frames, kpt_frames_l, kpt_frames_r, gt}
plus a summary line. Presence test is metadata-only (zarr.json shape +
existence of chunk objects), no chunk decode — cheap enough for all 35,732.

Usage:
  PYTHONPATH=src python scripts/inspection/mecka_flagship_gt_census.py \
      --out /path/census.jsonl [--limit N] [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = "foundational-research/hoi-dataset/EgoVerse/processed_v3/mecka/flagship"


def census_episode(fs, ep_id: str) -> dict:
    base = f"{ROOT}/{ep_id}.zarr"
    zj = json.loads(fs.cat(f"{base}/zarr.json"))
    total = int(zj.get("attributes", {}).get("total_frames") or 0)
    row = {"ep_id": ep_id, "total_frames": total}
    for side in ("left", "right"):
        try:
            aj = json.loads(fs.cat(f"{base}/{side}.obs_keypoints/zarr.json"))
            n = int(aj["shape"][0])
            # shape alone lies when chunks are missing; verify chunk objects exist
            if n > 0:
                chunks = fs.ls(f"{base}/{side}.obs_keypoints/c", detail=False)
                if not chunks:
                    n = 0
        except FileNotFoundError:
            n = 0
        row[f"kpt_frames_{side[0]}"] = n
    # GT-bearing = both hands' arrays cover >=90% of the video frames
    thr = max(1, int(0.9 * total))
    row["gt"] = bool(total > 0 and row["kpt_frames_l"] >= thr and row["kpt_frames_r"] >= thr)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    import gcsfs
    fs = gcsfs.GCSFileSystem()
    eps = sorted(p.rsplit("/", 1)[-1][:-5] for p in fs.ls(ROOT, detail=False)
                 if p.endswith(".zarr"))
    if args.limit:
        eps = eps[: args.limit]
    print(f"{len(eps)} flagship zarr episodes", flush=True)

    t0 = time.time()
    n_gt = 0
    with open(args.out, "w") as f, ThreadPoolExecutor(args.workers) as ex:
        def one(e):
            try:
                return census_episode(gcsfs.GCSFileSystem(), e)
            except Exception as err:  # noqa: BLE001
                return {"ep_id": e, "error": f"{type(err).__name__}: {str(err)[:80]}"}
        for i, row in enumerate(ex.map(one, eps)):
            n_gt += int(bool(row.get("gt")))
            f.write(json.dumps(row) + "\n")
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"[{i+1}/{len(eps)}] gt={n_gt} ({n_gt/(i+1):.1%}) "
                      f"{(i+1)/el:.0f} eps/s eta {(len(eps)-i-1)/((i+1)/el)/60:.0f}m", flush=True)
    print(f"DONE {len(eps)} episodes, GT-bearing {n_gt} ({n_gt/max(1,len(eps)):.1%})", flush=True)


if __name__ == "__main__":
    main()
