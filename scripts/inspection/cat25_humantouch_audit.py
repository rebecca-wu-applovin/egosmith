#!/usr/bin/env python3
"""CAT-2.5 pose-accuracy audit for Xspark-HumanTouch (LeRobot v2.1).

Per episode: parquet at 60Hz with world-frame (quat xyzw) head/wrist 7D poses,
25-joint MANUS hand skeletons (pos+quat+valid), 460ch/hand raw tactile + 12-patch
calibrated pressure (newton); HEVC 1080p head cam; per-episode meta JSON carries
pinhole intrinsics + camera_from_head_tracker extrinsic (rpy deg + translation,
unity_to_opencv_y_flip flag).

Probes:
  1. internal GT consistency: skeleton wrist joint vs 6-DoF wrist-tracker position;
  2. projection convention: enumerate rpy-order x extrinsic-direction x y-flip
     variants of the head_tracker->camera chain, score by in-frame ratio of all
     valid hand joints, render overlay sheet under the winner;
  3. tactile sanity: raw-channel frozen/saturation stats, patch pressure range,
     valid-flag rates, session_time_ns cadence.

Usage:
  python scripts/inspection/cat25_humantouch_audit.py --parquet <ep.parquet> \
      --video <ep.mp4> --meta <ep.json> --out_dir /root/cat25_audits/humantouch
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

LEFT_COLOR = (66, 133, 244)
RIGHT_COLOR = (219, 68, 55)
HEAD_COLOR = (255, 200, 0)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def rpy_to_R(rpy_deg, order: str) -> np.ndarray:
    r, p, y = np.deg2rad(rpy_deg)
    def rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    def ry(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    def rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    if order == "xyz":
        return rz(y) @ ry(p) @ rx(r)
    return rx(r) @ ry(p) @ rz(y)  # "zyx" application


FLIP_Y = np.diag([1.0, -1.0, 1.0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_tiles", type=int, default=12)
    ap.add_argument("--tile_width", type=int, default=480)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(args.meta).read_text())
    calib = meta["camera_calibration"]
    ext = calib["Calibration/camera/calibration_extrinsics.json"]
    intr = calib["Calibration/camera/calibration_intrinsics_A3.json"]
    K = np.array(intr["camera_matrix"]["matrix"])
    dist = np.array(intr["distortion_coefficients"]["values"])
    W, H = intr["image_width"], intr["image_height"]

    t = pq.read_table(args.parquet)
    n = t.num_rows
    def col(name):
        return np.asarray(t[name].to_pylist(), dtype=float)
    head = col("observation.human.pose.head")            # (T,7) pos+quat(xyzw)
    wrist = {s: col(f"observation.human.pose.wrist_{s}") for s in ("left", "right")}
    pose_valid = np.asarray(t["observation.human.pose.valid"].to_pylist())  # (T,3)
    skel = {}
    for s in ("left", "right"):
        pos = t[f"observation.human.hand_skeleton.{s}.position"].to_pylist()
        val = np.asarray(t[f"observation.human.hand_skeleton.{s}.valid"].to_pylist())
        skel[s] = {"pos": np.array([np.asarray(p, dtype=float) if p and len(p) == 25 else np.full((25, 3), np.nan)
                                    for p in pos]),
                   "valid": val}
    ts_ns = np.asarray(t["observation.session_time_ns"].to_pylist(), dtype=np.int64).ravel()

    # 1. skeleton wrist vs tracker
    wrist_consistency = {}
    for s in ("left", "right"):
        d = np.linalg.norm(skel[s]["pos"][:, 0] - wrist[s][:, :3], axis=1)
        d = d[np.isfinite(d)]
        wrist_consistency[s] = {"median_cm": float(np.median(d) * 100) if d.size else None,
                                "p90_cm": float(np.percentile(d, 90) * 100) if d.size else None,
                                "n": int(d.size)}

    # 2. projection variants
    t_off = np.array(ext["translation_m"])
    variants = {}
    for order, direction, flip in itertools.product(("xyz", "zyx"), ("cam_from_ht", "ht_from_cam"), ("none", "world_y", "cam_y")):
        R_off = rpy_to_R(ext["rotation_rpy_deg"], order)
        T_off = np.eye(4); T_off[:3, :3] = R_off; T_off[:3, 3] = t_off
        if direction == "ht_from_cam":
            T_off = np.linalg.inv(T_off)
        variants[f"{order}|{direction}|{flip}"] = (T_off, flip)

    def cam_points(T_off, flip, i, pts_world):
        p = pts_world.copy()
        if flip == "world_y":
            p = p @ FLIP_Y
            hp = FLIP_Y @ head[i, :3]
            R_h = FLIP_Y @ quat_to_R(head[i, 3:]) @ FLIP_Y
        else:
            hp = head[i, :3]
            R_h = quat_to_R(head[i, 3:])
        T_w_ht = np.eye(4); T_w_ht[:3, :3] = R_h; T_w_ht[:3, 3] = hp
        T_cam_w = T_off @ np.linalg.inv(T_w_ht)
        cam = (T_cam_w[:3, :3] @ p.T).T + T_cam_w[:3, 3]
        if flip == "cam_y":
            cam = cam @ FLIP_Y
        return cam

    sample_ids = np.linspace(0, n - 1, 60, dtype=int)
    scores = {}
    for name, (T_off, flip) in variants.items():
        inframe, total = 0, 0
        for i in sample_ids:
            pts = np.concatenate([skel["left"]["pos"][i], skel["right"]["pos"][i]])
            pts = pts[np.isfinite(pts).all(axis=1)]
            if not len(pts):
                continue
            cam = cam_points(T_off, flip, i, pts)
            front = cam[:, 2] > 0.05
            total += len(cam)
            if front.any():
                uv, _ = cv2.projectPoints(cam[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
                uv = uv.reshape(-1, 2)
                ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                inframe += int(ok.sum())
        scores[name] = inframe / max(total, 1)
    best_name = max(scores, key=scores.get)
    T_off_best, flip_best = variants[best_name]

    # 3. tactile sanity
    tactile = {}
    for s in ("left", "right"):
        raw = col(f"observation.tactile.raw_{s}")
        ch_std = raw.std(axis=0)
        tactile[f"raw_{s}"] = {"channels": raw.shape[1], "min": float(raw.min()), "max": float(raw.max()),
                               "median": float(np.median(raw)), "frozen_std0": int((ch_std == 0).sum()),
                               "frac_gt0": float((raw > 0).mean())}
    patch = col("observation.tactile.patch_pressure")
    tactile["patch_pressure_newton"] = {"shape": list(patch.shape), "min": float(patch.min()),
                                        "max": float(patch.max()), "median": float(np.median(patch))}
    tactile["tactile_valid_rate"] = np.asarray(t["observation.tactile.valid"].to_pylist()).mean(axis=0).tolist()
    dt = np.diff(ts_ns) / 1e6
    timing = {"n": n, "median_dt_ms": float(np.median(dt)), "max_dt_ms": float(dt.max()),
              "monotonic": bool((dt > 0).all())}

    # overlay sheet
    cap = cv2.VideoCapture(args.video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sel = np.linspace(0, min(n, n_video) - 1, args.num_tiles, dtype=int)
    tiles = []
    for i in sel:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scale = args.tile_width / img.width
        tile = img.resize((args.tile_width, int(img.height * scale)))
        draw = ImageDraw.Draw(tile)
        for s, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
            pts = skel[s]["pos"][i]
            finite = np.isfinite(pts).all(axis=1)
            if not finite.any():
                continue
            cam = cam_points(T_off_best, flip_best, i, pts[finite])
            front = cam[:, 2] > 0.05
            if front.any():
                uv, _ = cv2.projectPoints(cam[front].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
                for u, v in uv.reshape(-1, 2) * scale:
                    if np.isfinite([u, v]).all():
                        draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=color)
        draw.rectangle([0, 0, 60, 18], fill=(0, 0, 0))
        draw.text((4, 2), f"f{int(i)}", fill=(255, 255, 255))
        tiles.append(tile)
    sheet_path = None
    if tiles:
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        th = tiles[0].height
        sheet = Image.new("RGB", (cols * args.tile_width, rows * th + 40), (16, 16, 16))
        for k, tl in enumerate(tiles):
            sheet.paste(tl, ((k % cols) * args.tile_width, (k // cols) * th))
        d = ImageDraw.Draw(sheet)
        ep_name = Path(args.parquet).stem
        d.text((8, rows * th + 4), f"HumanTouch {ep_name} variant={best_name} inframe={scores[best_name]:.2f} "
                                   f"(L=blue R=red)", fill=(255, 255, 255))
        sheet_path = out_dir / f"{ep_name}_skeleton_overlay.jpg"
        sheet.save(sheet_path, quality=88)

    summary = {
        "episode": Path(args.parquet).stem,
        "frames": n, "video_frames": n_video,
        "wrist_tracker_vs_skeleton": wrist_consistency,
        "pose_valid_rate": pose_valid.mean(axis=0).tolist(),
        "skeleton_valid_rate": {s: float(skel[s]["valid"].mean()) for s in ("left", "right")},
        "projection_variants_inframe": {k: round(v, 4) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])},
        "best_variant": best_name,
        "tactile": tactile, "timing_60hz": timing,
        "sheet": str(sheet_path),
    }
    (out_dir / f"{Path(args.parquet).stem}_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("HUMANTOUCH_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
