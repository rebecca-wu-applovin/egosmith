#!/usr/bin/env python3
"""CAT-2.5 pose-accuracy audit for OpenTouch (MIT, arXiv:2512.16842).

Each session .hdf5 ships per clip: rgb_images_jpeg (T,), camera_poses (T,4,4),
right_hand_landmarks (T,21,3) in a metric SLAM/world frame, right_palm_pos (T,3),
right_pressure (T,16,16); plus /calibration/rgb (pinhole f, pp, size,
T_device_camera) and /transform_slam_to_rgb (4,4).

The landmark->pixel chain is not documented, so the numeric probe enumerates the
plausible chains and scores each by in-frame ratio + plausible camera-space depth,
then renders an overlay sheet under the winner for the eyes-on gate.

Also emits tactile sanity stats (baseline / polarity / dropout candidates) that
feed the tactile-validator spec.

Usage:
  python scripts/inspection/cat25_opentouch_audit.py --hdf5 <session.hdf5> \
      --out_dir /root/cat25_audits/opentouch --clips demo_00,demo_01,demo_02
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

RIGHT_COLOR = (219, 68, 55)
TIP_IDS = [4, 8, 12, 16, 20]  # MediaPipe-style fingertips
DEPTH_RANGE = (0.15, 2.0)     # plausible egocentric hand depth (m)


def build_chains(T_dev_cam: np.ndarray, T_slam_rgb: np.ndarray):
    """Candidate functions: (pose_t (4,4), pts_world (N,3)) -> pts_cam (N,3)."""
    inv = np.linalg.inv

    def apply(T, p):
        return (T[:3, :3] @ p.T).T + T[:3, 3]

    return {
        "inv(pose@T_dev_cam)": lambda P, p: apply(inv(P @ T_dev_cam), p),
        "inv(pose)": lambda P, p: apply(inv(P), p),
        "inv(T_dev_cam)@inv(pose)": lambda P, p: apply(inv(T_dev_cam) @ inv(P), p),
        "pose_as_w2c": lambda P, p: apply(P, p),
        "T_dev_cam@inv(pose)": lambda P, p: apply(T_dev_cam @ inv(P), p),
        "inv(pose)@T_slam_rgb": lambda P, p: apply(inv(P) @ T_slam_rgb, p),
        "inv(pose@T_dev_cam)@T_slam_rgb": lambda P, p: apply(inv(P @ T_dev_cam) @ T_slam_rgb, p),
    }


def project(pts_cam: np.ndarray, f: float, pp: np.ndarray) -> np.ndarray:
    z = np.clip(pts_cam[:, 2:3], 1e-6, None)
    u = f * pts_cam[:, 0:1] / z + pp[0]
    v = f * pts_cam[:, 1:2] / z + pp[1]
    return np.concatenate([u, v, pts_cam[:, 2:3]], axis=1)


def score_chain(fn, poses, lms, f, pp, wh):
    W, H = wh
    inframe, total, depth_ok = 0, 0, 0
    for t in range(len(poses)):
        cam = fn(poses[t], lms[t])
        uvz = project(cam, f, pp)
        ok = (uvz[:, 2] > 0) & (uvz[:, 0] >= 0) & (uvz[:, 0] < W) & (uvz[:, 1] >= 0) & (uvz[:, 1] < H)
        inframe += int(ok.sum())
        total += len(ok)
        depth_ok += int(((uvz[:, 2] > DEPTH_RANGE[0]) & (uvz[:, 2] < DEPTH_RANGE[1])).sum())
    return inframe / max(total, 1), depth_ok / max(total, 1)


def pressure_stats(pressure: np.ndarray) -> dict:
    flat = pressure.reshape(pressure.shape[0], -1)  # (T, 256)
    ch_std = flat.std(axis=0)
    return {
        "min": float(np.nanmin(flat)), "max": float(np.nanmax(flat)),
        "mean": float(np.nanmean(flat)), "median": float(np.nanmedian(flat)),
        "frozen_channels_std0": int((ch_std == 0).sum()),
        "p05_over_time_of_framewise_min": float(np.percentile(flat.min(axis=1), 5)),
        "note": "raw ADC; rest value appears HIGH (~3072) -> pressure likely decreases on contact (inverted polarity)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clips", default=None, help="comma-separated clip names; default first 3")
    ap.add_argument("--num_tiles", type=int, default=12)
    ap.add_argument("--tile_width", type=int, default=480)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with h5py.File(args.hdf5, "r") as fh:
        calib = fh["calibration/rgb"]
        f = float(calib["focal_length"][()])
        pp = np.array(calib["principal_point"])
        wh = tuple(np.array(calib["image_size"]))
        T_dev_cam = np.array(calib["T_device_camera"])
        T_slam_rgb = np.array(fh["transform_slam_to_rgb"])
        chains = build_chains(T_dev_cam, T_slam_rgb)

        clip_names = args.clips.split(",") if args.clips else list(fh["data"].keys())[:3]
        for name in clip_names:
            clip = fh["data"][name]
            poses = np.array(clip["camera_poses"])
            lms = np.array(clip["right_hand_landmarks"])
            palm = np.array(clip["right_palm_pos"])
            pressure = np.array(clip["right_pressure"])
            T = len(poses)

            scores = {k: score_chain(fn, poses, lms, f, pp, wh) for k, fn in chains.items()}
            best_name = max(scores, key=lambda k: (scores[k][0], scores[k][1]))
            best_fn = chains[best_name]

            # palm-vs-landmark-centroid consistency (both claimed same frame)
            centroid = lms.mean(axis=1)
            palm_dist = np.linalg.norm(centroid - palm, axis=1)

            # overlay sheet under winner
            sel = np.linspace(0, T - 1, args.num_tiles, dtype=int)
            tiles = []
            for t in sel:
                img = Image.open(io.BytesIO(clip["rgb_images_jpeg"][t])).convert("RGB")
                scale = args.tile_width / img.width
                tile = img.resize((args.tile_width, int(img.height * scale)))
                draw = ImageDraw.Draw(tile)
                uvz = project(best_fn(poses[t], lms[t]), f, pp) * np.array([scale, scale, 1.0])
                for j, (u, v, z) in enumerate(uvz):
                    if z <= 0 or not np.isfinite([u, v]).all():
                        continue
                    r = 4 if j in TIP_IDS else 2
                    draw.ellipse([u - r, v - r, u + r, v + r], fill=RIGHT_COLOR)
                draw.rectangle([0, 0, 60, 18], fill=(0, 0, 0))
                draw.text((4, 2), f"f{t}", fill=(255, 255, 255))
                tiles.append(tile)
            cols = 4
            rows = (len(tiles) + cols - 1) // cols
            th = tiles[0].height
            sheet = Image.new("RGB", (cols * args.tile_width, rows * th + 40), (16, 16, 16))
            for k, tile in enumerate(tiles):
                sheet.paste(tile, ((k % cols) * args.tile_width, (k // cols) * th))
            d = ImageDraw.Draw(sheet)
            d.text((8, rows * th + 4), f"OpenTouch {Path(args.hdf5).stem}/{name} T={T} chain={best_name} "
                                       f"inframe={scores[best_name][0]:.2f}", fill=(255, 255, 255))
            d.text((8, rows * th + 22),
                   f"palm-centroid dist median={np.median(palm_dist)*100:.1f}cm", fill=(255, 220, 150))
            sheet_path = out_dir / f"{Path(args.hdf5).stem}_{name}_overlay.jpg"
            sheet.save(sheet_path, quality=88)

            results.append({
                "clip": name, "frames": T,
                "chain_scores": {k: {"inframe": round(s[0], 4), "depth_ok": round(s[1], 4)} for k, s in scores.items()},
                "best_chain": best_name,
                "palm_vs_centroid_cm": {"median": float(np.median(palm_dist) * 100),
                                        "p90": float(np.percentile(palm_dist, 90) * 100)},
                "pressure": pressure_stats(pressure),
                "sheet": str(sheet_path),
            })

    summary = {"hdf5": args.hdf5, "clips": results}
    (out_dir / f"{Path(args.hdf5).stem}_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("OPENTOUCH_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
