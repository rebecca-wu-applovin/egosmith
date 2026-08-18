#!/usr/bin/env python
"""Phase 1 validation: AnyCalib predicted focal vs GT focal (and vs the W/2 guess).

For each clip with a frame tar + a GT SLAM npz (`img_focal`), run AnyCalib on a
representative frame and compare the predicted focal to ground truth. Answers "is
AnyCalib accurate enough on our data to replace the W/2 guess" before any pipeline
integration. hot3d (fisheye) also gets a Kannala-Brandt prediction.

Usage (egosmith env):
  python scripts/inspection/anycalib_vs_gt.py \
      --dataset taco  --frames_root /root/taco/frames --gt_root /root/taco/outputs
  python scripts/inspection/anycalib_vs_gt.py \
      --dataset hot3d --frames_root /root/hot3d/frames --gt_root /root/hot3d/outputs --fisheye
"""
from __future__ import annotations

import argparse
import glob
import io
import statistics as st
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anycalib import AnyCalib


def _mid_frame(tar_path: str) -> np.ndarray:
    with tarfile.open(tar_path, "r") as tf:
        names = sorted(n for n in tf.getnames() if n.endswith(".image.jpg"))
        n = names[len(names) // 2]
        return np.array(Image.open(io.BytesIO(tf.extractfile(n).read())).convert("RGB"))


def _gt_focal(gt_seq: str):
    f = sorted(glob.glob(f"{gt_seq}/SLAM/hawor_slam_w_scale_*.npz"))
    if not f:
        return None
    d = np.load(f[0])
    return float(np.asarray(d["img_focal"]).reshape(-1)[0]) if "img_focal" in d else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--model_id", default="anycalib_gen")
    ap.add_argument("--fisheye", action="store_true", help="also predict Kannala-Brandt")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AnyCalib(model_id=args.model_id).to(dev)

    clips = sorted(p.name for p in Path(args.gt_root).iterdir() if p.is_dir())
    rows = []
    for clip in clips[: args.limit]:
        tar = Path(args.frames_root) / f"{clip}.tar"
        gt_seq = Path(args.gt_root) / clip
        if not tar.is_file():
            continue
        gtf = _gt_focal(str(gt_seq))
        if gtf is None:
            continue
        img_np = _mid_frame(str(tar))
        H, W = img_np.shape[:2]
        image = torch.tensor(img_np, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255
        with torch.no_grad():
            out = model.predict(image, cam_id="pinhole")
            fx, fy = float(out["intrinsics"][0]), float(out["intrinsics"][1])
            pin_focal = 0.5 * (fx + fy)
            kb_focal = None
            if args.fisheye:
                outk = model.predict(image, cam_id="kb:4")
                kb_focal = 0.5 * (float(outk["intrinsics"][0]) + float(outk["intrinsics"][1]))
        w2 = W / 2.0
        rows.append({
            "clip": clip, "W": W, "H": H, "gt_focal": gtf,
            "w2_guess": w2, "anycalib_pinhole": pin_focal, "anycalib_kb": kb_focal,
            "w2_err_pct": 100 * abs(w2 - gtf) / gtf,
            "anycalib_err_pct": 100 * abs(pin_focal - gtf) / gtf,
        })

    if not rows:
        print(f"[{args.dataset}] no clips with both frames+GT found"); return
    print(f"\n[{args.dataset}] {len(rows)} clips  (model={args.model_id})")
    print(f"  {'clip':45s} {'GT_f':>8s} {'W/2':>8s} {'AnyCalib':>9s} {'W/2 err%':>9s} {'AC err%':>8s}")
    for r in rows[:12]:
        kb = f" kb={r['anycalib_kb']:.0f}" if r["anycalib_kb"] else ""
        print(f"  {r['clip'][:45]:45s} {r['gt_focal']:8.1f} {r['w2_guess']:8.1f} "
              f"{r['anycalib_pinhole']:9.1f} {r['w2_err_pct']:8.1f}% {r['anycalib_err_pct']:7.1f}%{kb}")
    med_w2 = st.median(r["w2_err_pct"] for r in rows)
    med_ac = st.median(r["anycalib_err_pct"] for r in rows)
    print(f"  --- median focal error:  W/2 guess = {med_w2:.1f}%   AnyCalib = {med_ac:.1f}%  "
          f"(n={len(rows)}) ---")


if __name__ == "__main__":
    main()
