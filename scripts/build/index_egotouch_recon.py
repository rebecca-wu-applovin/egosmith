#!/usr/bin/env python
"""Build Stage-1-style survivor rows for EgoTouch chest.mp4 episodes (RECON path).

EgoTouch's native 3D labels are monocular HaMeR/WiLoR pseudo-labels whose
weak-perspective depth is metric-unstable (Cat-2.5 smoke: L4 kept 0/5), so the
dataset rides the RECON path: chest.mp4 -> generate_video_wds.py -> Phase C recon.
This indexer probes each episode's chest.mp4 and emits one survivor row per episode
in the exact format generate_video_wds.py consumes, with synthesized fixed-length
intervals (default 10s) covering the whole episode:

  {uri, group, session, fps, n_frames, width, height,
   intervals: [{start_frame, end_frame, start_sec, end_sec}]}

Zero-byte episodes (~7.5% of the mirror) are excluded up front. Use
`--id_mode session` in generate_video_wds (every episode video is named chest.mp4).

Usage:
  python scripts/build/index_egotouch_recon.py \
      --gcs_prefix gs://foundational-research/hoi-dataset/EgoTouch \
      --out index.jsonl [--sample 20 --seed 0] [--shard 0 --num_shards 50]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EgoTouch episodes -> video survivor rows")
    p.add_argument("--gcs_prefix", default="gs://foundational-research/hoi-dataset/EgoTouch")
    p.add_argument("--listing", default="", help="optional pre-made `gsutil ls -l **` output file")
    p.add_argument("--out", required=True)
    p.add_argument("--segment_sec", type=float, default=10.0)
    p.add_argument("--min_segment_sec", type=float, default=2.0)
    p.add_argument("--sample", type=int, default=0, help=">0: stratified sample of N episodes (smoke)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p


def list_episodes(args) -> list[tuple[str, int]]:
    """[(chest.mp4 uri, bytes)] for non-zero episodes."""
    if args.listing:
        lines = Path(args.listing).read_text().splitlines()
    else:
        r = subprocess.run(["gsutil", "ls", "-l", f"{args.gcs_prefix.rstrip('/')}/**"],
                           capture_output=True, text=True, check=True)
        lines = r.stdout.splitlines()
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 3 and parts[-1].endswith("/chest.mp4"):
            size = int(parts[0])
            if size > 0:
                out.append((parts[-1], size))
    return sorted(out)


def probe(uri: str, tmp_dir: str) -> dict | None:
    local = Path(tmp_dir) / "chest.mp4"
    subprocess.run(["gcloud", "storage", "cp", uri, str(local)], check=True, capture_output=True)
    cap = cv2.VideoCapture(str(local))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    local.unlink(missing_ok=True)
    if n <= 0 or fps <= 0:
        return None
    return {"n_frames": n, "fps": fps, "width": w, "height": h}


def main() -> None:
    args = build_parser().parse_args()
    eps = list_episodes(args)
    print(f"[egotouch-index] {len(eps)} non-zero chest.mp4 episodes", flush=True)

    if args.sample:
        # stratify by scene so the smoke covers all 5 environments
        by_scene: dict[str, list] = {}
        for uri, size in eps:
            scene = uri.split("/EgoTouch/", 1)[1].split("/", 1)[0]
            by_scene.setdefault(scene, []).append((uri, size))
        rng = random.Random(args.seed)
        take, quota = [], max(1, args.sample // max(1, len(by_scene)))
        for scene in sorted(by_scene):
            take += rng.sample(by_scene[scene], min(quota, len(by_scene[scene])))
        eps = sorted(take)[: args.sample]
    eps = [e for i, e in enumerate(eps) if i % args.num_shards == args.shard]

    t0 = time.time()
    rows, errors = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (uri, size) in enumerate(eps):
            rel = uri.split("/EgoTouch/", 1)[1]
            scene, task, ep = rel.split("/")[0], rel.split("/")[1], rel.split("/")[2]
            try:
                meta = probe(uri, tmp)
            except Exception as e:  # noqa: BLE001
                meta = None
                print(f"  ERR {rel[:80]}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            if meta is None:
                errors += 1
                continue
            fps, n = meta["fps"], meta["n_frames"]
            seg = max(1, int(round(args.segment_sec * fps)))
            ivs = []
            for s in range(0, n, seg):
                e = min(s + seg - 1, n - 1)
                if (e - s + 1) / fps < args.min_segment_sec:
                    continue
                ivs.append({"start_frame": s, "end_frame": e,
                            "start_sec": round(s / fps, 3), "end_sec": round((e + 1) / fps, 3)})
            rows.append({"uri": uri, "group": f"{scene}/{task}", "session": f"{scene}_{task}_{ep}",
                         "fps": fps, "n_frames": n, "width": meta["width"], "height": meta["height"],
                         "video_bytes": size, "duration_sec": round(n / fps, 3), "intervals": ivs})
            if (i + 1) % 25 == 0 or i + 1 == len(eps):
                print(f"[{i+1}/{len(eps)}] probed ({time.time()-t0:.0f}s)", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total_h = sum(r["duration_sec"] for r in rows) / 3600
    print(f"[egotouch-index] wrote {len(rows)} rows ({errors} errors), "
          f"{total_h:.2f} video-hours -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
