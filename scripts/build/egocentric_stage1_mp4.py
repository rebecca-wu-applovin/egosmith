#!/usr/bin/env python
"""Stage-1 pre-filter for Egocentric-100K, run directly on raw fisheye MP4 clips.

Egocentric-100K is video-only (no GT) fisheye MP4 in WebDataset shards
(factory/worker/partNNN.tar of <key>.mp4 + <key>.json). This runs the SAME Stage-1 gates as
scripts/build/stage1_prefilter.py (Gate A YOLO hands + Gate B optical-flow RANSAC ego-motion +
Gate C valid-span merge, from lib.clip.heuristic_video_clipper) but:
  - streams each part.tar from GCS (gcsfs, no full download),
  - decodes each clip's mp4 and undistorts fisheye->pinhole (cv2.fisheye) with the worker's
    Kannala-Brandt intrinsics.json (only the sampled frames are undistorted),
  - KEEPS a clip if it has >=1 valid span, recording the valid interval(s) so Phase B can trim.

No frame tars, no reconstruction — this is the cheap cut over ALL clips. Survivors feed Phase B.

Output: a survivor JSONL (one kept clip per line: clip_id, part_tar, worker, focal, fps,
duration_sec, kept_sec, intervals[]) + a funnel report JSON.

Usage (egosmith env, GPU):
  python scripts/build/egocentric_stage1_mp4.py \
      --parts_list parts_shard.txt --out_manifest stage1.kept.jsonl --report_out s1.json
"""
from __future__ import annotations

import argparse, json, os, sys, tarfile, tempfile, time
from pathlib import Path
from collections import Counter

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import gcsfs  # noqa: E402

from lib.clip.heuristic_video_clipper import (  # noqa: E402
    load_clip_config, analyze_frame_source_intervals, _load_yolo,
)


class _UndistortedSampled:
    """Frame source exposing len()==n_frames but only serving undistorted SAMPLED frames.

    analyze_frame_source_intervals only calls get_frame() on indices where idx%skip==0, so we
    pre-decode sequentially and undistort just those (remap is skipped on the rest)."""

    def __init__(self, n, sampled):
        self._n = n
        self._sampled = sampled  # {idx: bgr uint8}

    def __len__(self):
        return self._n

    def get_frame(self, i, rgb=False):
        fr = self._sampled[i]
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if rgb else fr


def _fisheye_map(intr, balance):
    W, H = int(intr["image_width"]), int(intr["image_height"])
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    D = np.array([intr["k1"], intr["k2"], intr["k3"], intr["k4"]], np.float64)
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), np.eye(3), balance=balance)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (W, H), cv2.CV_16SC2)
    return m1, m2, float(newK[0, 0])


def _decode_sampled(mp4_bytes, m1, m2, skip):
    """Decode sequentially; undistort+keep only every `skip`-th frame. Returns (n, {idx: frame})."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
        t.write(mp4_bytes); p = t.name
    try:
        cap = cv2.VideoCapture(p)
        sampled, idx = {}, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % skip == 0:
                sampled[idx] = cv2.remap(fr, m1, m2, cv2.INTER_LINEAR)
            idx += 1
        cap.release()
        return idx, sampled
    finally:
        os.unlink(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts_list", required=True, help="file of gs:// part.tar paths (this shard)")
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--config", default=str(_REPO / "src/lib/clip/heuristic_clip_config.yaml"))
    ap.add_argument("--detector", default=str(_REPO / "weights/external/detector.pt"))
    ap.add_argument("--balance", type=float, default=0.0, help="cv2.fisheye undistort balance (FOV knob)")
    ap.add_argument("--limit", type=int, default=0, help=">0: cap clips per part (smoke)")
    args = ap.parse_args()

    cfg = load_clip_config(args.config)
    skip = max(1, int(((cfg.get("heuristic") or {}).get("skip_frames", 15))))
    model = _load_yolo(args.detector)
    if model is None:
        raise SystemExit(f"YOLO detector not found at {args.detector}")

    fs = gcsfs.GCSFileSystem()
    parts = [ln.strip() for ln in open(args.parts_list) if ln.strip()]
    kept, dropped = [], []
    total = 0
    t0 = time.time()

    for pi, part in enumerate(parts):
        gspath = part[5:] if part.startswith("gs://") else part
        worker = gspath.rsplit("/", 1)[0]
        try:
            intr = json.load(fs.open(worker + "/intrinsics.json"))
            m1, m2, focal = _fisheye_map(intr, args.balance)
        except Exception as e:  # noqa: BLE001
            print(f"[part {pi}] SKIP {part}: intrinsics err {str(e)[:80]}", flush=True)
            continue

        pend, nclip = {}, 0
        try:
            with fs.open(gspath, "rb") as f:
                tf = tarfile.open(fileobj=f, mode="r|")
                for m in tf:
                    if "." not in m.name:
                        continue
                    key, ext = m.name.rsplit(".", 1)
                    data = tf.extractfile(m).read()
                    pend.setdefault(key, {})[ext] = data
                    if "mp4" not in pend[key] or "json" not in pend[key]:
                        continue
                    rec = pend.pop(key)
                    total += 1; nclip += 1
                    meta = json.loads(rec["json"])
                    fps = float(meta.get("fps", 30.0))
                    try:
                        n, sampled = _decode_sampled(rec["mp4"], m1, m2, skip)
                        if n == 0:
                            dropped.append({"clip_id": key, "reason": "decode_empty"}); continue
                        ivs, info = analyze_frame_source_intervals(
                            _UndistortedSampled(n, sampled), cfg, model=model, fps=fps)
                        if ivs:
                            kept_sec = sum(iv.end_sec - iv.start_sec for iv in ivs)
                            kept.append({
                                "clip_id": key, "part_tar": part, "worker": "gs://" + worker,
                                "focal": round(focal, 3), "fps": fps,
                                "duration_sec": meta.get("duration_sec"),
                                "kept_sec": round(kept_sec, 2),
                                "intervals": [iv.to_dict() for iv in ivs],
                            })
                        else:
                            vf = info["valid_sample_count"] / max(1, info["sample_count"])
                            dropped.append({"clip_id": key, "reason": "no_valid_span",
                                            "valid_frac": round(vf, 3)})
                    except Exception as e:  # noqa: BLE001
                        dropped.append({"clip_id": key, "reason": f"error:{str(e)[:60]}"})
                    if args.limit and nclip >= args.limit:
                        break
        except Exception as e:  # noqa: BLE001
            print(f"[part {pi}] PART ERROR {part}: {str(e)[:100]}", flush=True)
            continue
        print(f"[part {pi+1}/{len(parts)}] {part.rsplit('/',3)[-3:]} clips={nclip} "
              f"kept={len(kept)} dropped={len(dropped)} ({time.time()-t0:.0f}s)", flush=True)

    Path(args.out_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w") as w:
        for k in kept:
            w.write(json.dumps(k) + "\n")

    kept_hours = sum(k["kept_sec"] for k in kept) / 3600.0
    report = {
        "parts": len(parts), "total_clips": total,
        "kept_clips": len(kept), "dropped_clips": len(dropped),
        "kept_pct": round(100.0 * len(kept) / max(1, total), 1),
        "kept_hours": round(kept_hours, 2),
        "balance": args.balance,
        "drop_reasons": dict(Counter(d["reason"].split(":")[0] for d in dropped)),
    }
    Path(args.report_out).write_text(json.dumps(report, indent=1))
    print(f"[stage1-mp4] {len(kept)}/{total} kept ({report['kept_pct']}%), "
          f"{kept_hours:.2f} kept-hours -> {args.out_manifest}", flush=True)


if __name__ == "__main__":
    main()
