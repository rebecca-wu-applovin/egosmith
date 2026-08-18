#!/usr/bin/env python3
"""Render segmented HOT3D clips as short real-time videos (RGB + GT hand overlay).

For each clip: decode the undistorted egocentric frames from the frame tar, project the
GT MANO joints (blue=left, red=right) with the clip's camera, and encode to a 30 fps mp4
(150 frames = 5 s real-time). Reuses the projection helpers from taco_overlay_sheets.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taco_overlay_sheets import (  # noqa: E402
    LEFT_COLOR, RIGHT_COLOR, compute_world_joints, load_camera, project, draw_joints,
)


def render_clip(record, device, out_path: Path, size: int, fps: int, overlay: bool, crf: int):
    from PIL import Image

    descriptor = record.descriptor
    seq_folder = Path(descriptor.seq_folder)
    left_j, right_j = compute_world_joints(seq_folder, device)
    T = min(len(descriptor.frame_names), left_j.shape[0])
    extr, intr = load_camera(seq_folder, T)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with tarfile.open(descriptor.shard_path, "r") as tar:
            members = {m.name: m for m in tar if m.isfile() and m.name.endswith(".image.jpg")}
            for t, name in enumerate(descriptor.frame_names[:T]):
                img = Image.open(io.BytesIO(tar.extractfile(members[name]).read())).convert("RGB")
                if overlay:
                    scale = size / img.width
                    tile = img.resize((size, int(img.height * scale)))
                    for joints, color in ((left_j, LEFT_COLOR), (right_j, RIGHT_COLOR)):
                        uvz = project(joints[t], extr[t], intr) * np.array([scale, scale, 1.0])
                        draw_joints(tile, uvz, color, radius=max(2, size // 160))
                    tile.save(tmp / f"f{t:05d}.jpg", quality=90)
                else:
                    img.resize((size, int(img.height * size / img.width))).save(tmp / f"f{t:05d}.jpg", quality=90)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", str(fps),
             "-i", str(tmp / "f%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", str(crf), "-movflags", "+faststart", str(out_path)],
            check=True, capture_output=True,
        )
    return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--clip_ids", required=True, help="file with one clip_id per line, or comma list")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=30)
    p.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    import torch
    from lib.pipeline.clips.clip_manifest import load_clip_manifest

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    by_id = {r.clip_id: r for r in load_clip_manifest(args.manifest)}
    if Path(args.clip_ids).is_file():
        ids = [x.strip() for x in Path(args.clip_ids).read_text().splitlines() if x.strip()]
    else:
        ids = [x.strip() for x in args.clip_ids.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    for cid in ids:
        rec = by_id.get(cid)
        if rec is None:
            print(f"skip {cid}: not in manifest", flush=True); continue
        suffix = "overlay" if args.overlay else "rgb"
        T = render_clip(rec, device, out_dir / f"{cid}.{suffix}.mp4", args.size, args.fps, args.overlay, args.crf)
        print(f"rendered {cid} ({T} frames)", flush=True)
    print("HOT3D_VIDEOS_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
