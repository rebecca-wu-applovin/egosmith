#!/usr/bin/env python3
"""Failure contact sheets for EgoDex dropped clips (native-lowdim, no MANO).

EgoDex clips carry the pipeline's 116-d native lowdim (built directly from Vision Pro
joints) inside each WDS tar. For each dropped clip we project the stored world wrist +
5 fingertips (per hand) through the stored World2Cam extrinsic + pinhole intrinsic onto
the undistorted frames, tile ~12 frames, outline off-screen frames, and caption with the
drop reasons. Grouped by primary reason:  failures/<primary_reason>/<clip_id>.jpg
"""
from __future__ import annotations
import argparse, io, json, tarfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

LEFT = (66, 133, 244)     # blue
RIGHT = (219, 68, 55)     # red
FLAG = (244, 208, 63)     # amber outline for off-screen frames


def _proj(P, extr, fx, fy, cx, cy):
    Xc = (extr @ np.append(P, 1.0))[:3]
    if Xc[2] <= 1e-6:
        return None
    return np.array([fx * Xc[0] / Xc[2] + cx, fy * Xc[1] / Xc[2] + cy])


def sheet(tar_path, reasons, out_path, n_tiles=12, tile_w=360):
    tr = tarfile.open(tar_path, "r")
    names = sorted(n for n in tr.getnames() if n.endswith(".image.jpg"))
    if not names:
        return False
    T = len(names)
    pick = [int(i) for i in np.linspace(0, T - 1, min(n_tiles, T))]
    tiles = []
    for idx in pick:
        key = names[idx][: -len(".image.jpg")]
        ld = np.load(io.BytesIO(tr.extractfile(key + ".lowdim.npy").read()))
        img = Image.open(io.BytesIO(tr.extractfile(names[idx]).read())).convert("RGB")
        W, H = img.size
        sc = tile_w / W
        tile = img.resize((tile_w, int(H * sc)))
        d = ImageDraw.Draw(tile)
        extr = ld[96:112].reshape(4, 4)
        fx, fy, cx, cy = ld[112:116]
        off = False
        for pts, col in ((([ld[0:3]] + list(ld[18:33].reshape(5, 3))), LEFT),
                         (([ld[3:6]] + list(ld[33:48].reshape(5, 3))), RIGHT)):
            for j, P in enumerate(pts):
                uv = _proj(P, extr, fx, fy, cx, cy)
                if uv is None:
                    continue
                if not (0 <= uv[0] < W and 0 <= uv[1] < H):
                    off = True
                r = 5 if j == 0 else 3
                x, y = uv[0] * sc, uv[1] * sc
                d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        if off:
            d.rectangle([1, 1, tile_w - 2, tile.height - 2], outline=FLAG, width=4)
        tiles.append(tile)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    th = tiles[0].height
    cap_h = 46
    sheet = Image.new("RGB", (cols * tile_w, rows * th + cap_h), (16, 16, 16))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tile_w, (i // cols) * th))
    cap = f"{out_path.stem}  T={T}  reasons: {', '.join(reasons)}"
    ImageDraw.Draw(sheet).text((8, rows * th + 12), cap[:200], fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=86)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--frames_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--per_reason", type=int, default=8, help="max sheets per primary reason")
    args = p.parse_args()
    rep = json.load(open(args.report))
    out = Path(args.out_dir)
    by_reason = {}
    for d in rep["dropped"]:
        primary = (d.get("reasons") or ["unknown"])[0]
        by_reason.setdefault(primary, []).append(d)
    made = 0
    for primary, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        for d in items[: args.per_reason]:
            tar = Path(args.frames_root) / f"{d['clip_id']}.tar"
            if not tar.is_file():
                continue
            if sheet(tar, d.get("reasons", []), out / primary / f"{d['clip_id']}.jpg"):
                made += 1
    print(f"EGODEX_SHEETS_DONE made={made} reasons={len(by_reason)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
