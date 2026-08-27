#!/usr/bin/env python
"""Post-ship per-session tip verification for the WIYH native tier.

Independent audit of the anchor solves on the SHIPPED artifacts: detects teal
fingertip pads in the rectified tar frames (456x256) and measures the median
distance to the nearest projected lowdim fingertip (per presence-on hand),
per session (up to 3 clips x every 10th frame).

Interpretation: median px @456w is ~3x smaller than fisheye px at the hands'
off-axis positions. Shipping gate used for v1: sessions >= 40 px dropped
(8/40); kept sessions measure 9-38 px. Values are stamped on every manifest
record as metadata.tip_verification_med_px_456w.

Usage:
  python scripts/build/wiyh_verify_shipped_tips.py \
      --filtered_manifest /root/w7_native/build/clip_manifest.filtered.jsonl \
      --frames_root /root/w7_native/build/frames \
      --out session_tip_verification.json
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def tip_px(ld):
    w2c = ld[96:112].reshape(4, 4)
    intr = ld[112:116]
    tips = np.vstack([ld[18:33].reshape(5, 3), ld[33:48].reshape(5, 3)])
    Xc = tips @ w2c[:3, :3].T + w2c[:3, 3]
    ok = Xc[:, 2] > 0.02
    u = intr[0] * Xc[:, 0] / Xc[:, 2] + intr[2]
    v = intr[1] * Xc[:, 1] / Xc[:, 2] + intr[3]
    return np.stack([u, v], 1), ok


def detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 65) & (h <= 110) & (s >= 15) & (v >= 35)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    return [cents[k] for k in range(1, n) if 15 < stats[k, 4] < 1500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered_manifest", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips_per_session", type=int, default=3)
    ap.add_argument("--frame_stride", type=int, default=10)
    a = ap.parse_args()

    by_sess = defaultdict(list)
    for l in open(a.filtered_manifest):
        r = json.loads(l)
        by_sess[r["descriptor"]["extra"]["session"]].append(r)

    results = {}
    for sess, recs in sorted(by_sess.items()):
        dists = []
        for r in recs[:a.clips_per_session]:
            tar = Path(a.frames_root) / Path(r["descriptor"]["shard_path"]).name
            if not tar.exists():
                continue
            with tarfile.open(tar) as tf:
                mem = {m.name: m for m in tf.getmembers()}
                keys = sorted({n.rsplit(".", 2)[0] for n in mem if n.endswith(".image.jpg")})
                for k in keys[::a.frame_stride]:
                    meta = json.loads(tf.extractfile(mem[k + ".meta.json"]).read())
                    if meta["presence"] == 0:
                        continue
                    ld = np.load(io.BytesIO(tf.extractfile(mem[k + ".lowdim.npy"]).read()))
                    img = cv2.imdecode(np.frombuffer(
                        tf.extractfile(mem[k + ".image.jpg"]).read(), np.uint8), cv2.IMREAD_COLOR)
                    dets = detect(img)
                    if not dets:
                        continue
                    uv, ok = tip_px(ld)
                    sides = []
                    if meta["presence"] & 1:
                        sides += list(range(0, 5))
                    if meta["presence"] & 2:
                        sides += list(range(5, 10))
                    for j in sides:
                        if ok[j]:
                            d = min(np.linalg.norm(np.array(c) - uv[j]) for c in dets)
                            dists.append(min(d, 200.0))
        med = float(np.median(dists)) if len(dists) >= 20 else None
        results[sess] = {"med_px_456w": round(med, 1) if med else None, "n": len(dists)}
        print(sess, results[sess])
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
