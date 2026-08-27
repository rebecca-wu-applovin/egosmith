#!/usr/bin/env python3
"""Overlay smoke-check for native-keypoint WDS clips (generate_keypoints_wds outputs).

For each sampled clip tar: read .image.jpg + .lowdim.npy, project the stored world
wrist + 5 fingertips per hand through the stored World2Cam extrinsic + pinhole
intrinsic, draw onto ~6 tiles, save one sheet per clip. Presence bits gate drawing
(dim marker when the hand's bit is off). This is the standard visual gate before a
native build ships (egodex_native_sheets pattern, dataset-agnostic).

Usage:
  python scripts/inspection/native_wds_overlay.py \
      --frames_root <dir of clip tars> --out_dir <dir> [--n_clips 6] [--seed 0]
"""
from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LD = {"lw": slice(0, 3), "rw": slice(3, 6), "lt": slice(18, 33), "rt": slice(33, 48),
      "extr": slice(96, 112), "intr": slice(112, 116)}
COL = {"l": (66, 133, 244), "r": (219, 68, 55)}


def sheet(tar_path: Path, out_path: Path, n_tiles=6, tile_w=420):
    with tarfile.open(tar_path) as tr:
        names = sorted(n for n in tr.getnames() if n.endswith(".image.jpg"))
        if not names:
            return None
        picks = [names[int(i * (len(names) - 1) / max(1, n_tiles - 1))] for i in range(min(n_tiles, len(names)))]
        tiles, stats = [], []
        for n in picks:
            key = n[: -len(".image.jpg")]
            img = Image.open(io.BytesIO(tr.extractfile(n).read())).convert("RGB")
            ld = np.load(io.BytesIO(tr.extractfile(key + ".lowdim.npy").read()))
            pres = json.loads(tr.extractfile(key + ".meta.json").read())["presence"]
            W, H = img.size
            s = tile_w / W
            img = img.resize((tile_w, int(H * s)))
            dr = ImageDraw.Draw(img)
            extr = ld[LD["extr"]].reshape(4, 4)
            fx, fy, cx, cy = ld[LD["intr"]]
            n_in = 0
            for side, wsl, tsl, bit in (("l", "lw", "lt", 1), ("r", "rw", "rt", 2)):
                pts = np.concatenate([ld[LD[wsl]][None], ld[LD[tsl]].reshape(5, 3)], 0)
                Xc = (np.concatenate([pts, np.ones((6, 1))], 1) @ extr.T)[:, :3]
                on = bool(pres & bit)
                for x, y, z in Xc:
                    if z <= 1e-6:
                        continue
                    u, v = (fx * x / z + cx) * s, (fy * y / z + cy) * s
                    if 0 <= u < tile_w and 0 <= v < img.size[1]:
                        r = 5 if on else 2
                        dr.ellipse([u - r, v - r, u + r, v + r],
                                   fill=COL[side] if on else None, outline=COL[side])
                        n_in += 1
            stats.append(n_in)
            tiles.append(img)
        th = max(t.size[1] for t in tiles)
        sheet_img = Image.new("RGB", (tile_w * len(tiles), th), (20, 20, 20))
        for i, t in enumerate(tiles):
            sheet_img.paste(t, (i * tile_w, 0))
        sheet_img.save(out_path, quality=88)
        return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_clips", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tars = sorted(Path(args.frames_root).glob("*.tar"))
    random.seed(args.seed)
    picks = random.sample(tars, min(args.n_clips, len(tars)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for t in picks:
        st = sheet(t, out / f"{t.stem}.jpg")
        print(f"{t.stem}: inframe-projected pts per tile = {st}")


if __name__ == "__main__":
    main()
