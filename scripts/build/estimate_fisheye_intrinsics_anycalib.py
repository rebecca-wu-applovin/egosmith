#!/usr/bin/env python
"""Estimate per-camera Kannala-Brandt (cv2.fisheye) intrinsics for a video dataset
with AnyCalib (kb:4), median over sampled frames x videos per camera key.

Used to produce scripts/build/intrinsics/cat1_fisheye_intrinsics.kb4.json:
  * Assembly101 HMC (8 serials, 636x480 mono): fx_med 221-295; keyed "HMC_<serial>"
  * HD-EPIC (Aria RGB, 9 participants, 1408x1408): fx_med 580-646 — matches the
    known Aria nominal (~610), keyed "/Videos/PXX/" (substring-safe)
AnyCalib kb:4 was GT-validated on hot3d (Aria/Quest3) at 3.6% median focal error
(scripts/inspection/anycalib_vs_gt.py). generate_video_wds.py consumes the JSON:
a video whose uri contains a key is cv2.fisheye-undistorted with those intrinsics.

Usage (env with anycalib + CUDA):
  python scripts/build/estimate_fisheye_intrinsics_anycalib.py \
      --index_jsonl idx/assembly101.index.jsonl --key_mode hmc_serial --out asm.json
  python scripts/build/estimate_fisheye_intrinsics_anycalib.py \
      --index_jsonl idx/hd_epic.index.jsonl --key_mode group_dir --out hde.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_jsonl", required=True, help="rows with at least {uri, group}")
    ap.add_argument("--key_mode", choices=["hmc_serial", "group_dir"], required=True,
                    help="hmc_serial: key 'HMC_<serial>' from basename; "
                         "group_dir: key '/Videos/<group>/'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vids_per_key", type=int, default=3)
    ap.add_argument("--frames_per_vid", type=int, default=5)
    ap.add_argument("--model_id", default="anycalib_gen")
    args = ap.parse_args()

    from anycalib import AnyCalib

    rows = [json.loads(l) for l in open(args.index_jsonl) if l.strip()]
    by_key: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if args.key_mode == "hmc_serial":
            key = "HMC_" + r["uri"].rsplit("/", 1)[-1].split("_")[1]
        else:
            key = f'/Videos/{r["group"]}/'
        by_key[key].append(r["uri"])

    model = AnyCalib(model_id=args.model_id).to("cuda")
    result = {}
    for key, uris in sorted(by_key.items()):
        picks = uris[:: max(1, len(uris) // args.vids_per_key)][: args.vids_per_key]
        params, W, H = [], None, None
        for uri in picks:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as t:
                subprocess.run(["gcloud", "storage", "cp", uri, t.name],
                               check=True, capture_output=True)
                cap = cv2.VideoCapture(t.name)
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                for f in np.linspace(n * 0.1, n * 0.9, args.frames_per_vid).astype(int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                    ok, fr = cap.read()
                    if not ok:
                        continue
                    if fr.ndim == 2:
                        fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
                    H, W = fr.shape[:2]
                    img = torch.tensor(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB),
                                       dtype=torch.float32, device="cuda").permute(2, 0, 1) / 255
                    with torch.no_grad():
                        out = model.predict(img, cam_id="kb:4")
                    intr = [float(x) for x in out["intrinsics"]]  # fx fy cx cy k1..k4
                    if all(np.isfinite(intr)) and intr[0] > 0:
                        params.append(intr)
                cap.release()
        if not params:
            print(f"[{key}] NO ESTIMATES", flush=True)
            continue
        med = np.median(np.array(params), axis=0)
        result[key] = {
            "fx": round(float(med[0]), 3), "fy": round(float(med[1]), 3),
            "cx": round(float(med[2]), 3), "cy": round(float(med[3]), 3),
            "k1": round(float(med[4]), 6), "k2": round(float(med[5]), 6),
            "k3": round(float(med[6]), 6), "k4": round(float(med[7]), 6),
            "image_width": int(W), "image_height": int(H),
            "n_samples": len(params), "model": f"{args.model_id} kb:4 median",
        }
        p10, p90 = np.percentile(np.array(params)[:, 0], [10, 90])
        print(f"[{key}] n={len(params)} fx_med={med[0]:.1f} fx_p10-90=({p10:.1f},{p90:.1f})", flush=True)

    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out} ({len(result)} keys)")


if __name__ == "__main__":
    main()
