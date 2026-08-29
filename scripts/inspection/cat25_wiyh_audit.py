#!/usr/bin/env python3
"""CAT-2.5 pose-accuracy audit for WIYH (World In Your Hands).

WIYH ships per-sample dataset.hdf5 with:
  pose/{left_eef,right_eef}/feedback/pose_in_chest  (7D pos+quat, chest frame)
  meta/calibration/lf_chest_fisheye                 (KB4 fisheye K/D, extrinsic==I)
  hand_masks/lf_chest_fisheye/*.png                 (per-frame hand segmentation)

There are no shipped 2D keypoints, so the numeric convention probe is:
project the left/right wrist (eef position) into the chest fisheye under all 24
proper signed-permutation rotations (chest->OpenCV-camera candidates), and score
each candidate by (a) in-frame ratio and (b) mean pixel distance from the
projected wrist to the nearest hand-mask pixel. The winning candidate + its
lock-on stats are the audit verdict input; a contact sheet is rendered for eyes.

Usage:
  python scripts/inspection/cat25_wiyh_audit.py --sample_dir <extracted worldcode dir> \
      --out_dir /root/cat25_audits/wiyh
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw

CHEST_CAM = "lf_chest_fisheye"
LEFT_COLOR = (66, 133, 244)   # blue
RIGHT_COLOR = (219, 68, 55)   # red
MASK_COLOR = (0, 255, 120)


def signed_permutations() -> list[np.ndarray]:
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats  # 24 proper rotations


def project_fisheye(points_cam: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """(N,3) camera-frame -> (N,2) pixels via cv2.fisheye (KB4). Points behind cam -> nan."""
    out = np.full((points_cam.shape[0], 2), np.nan)
    front = points_cam[:, 2] > 1e-6
    if front.any():
        pts = points_cam[front].reshape(-1, 1, 3)
        uv, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), K, D.reshape(4, 1))
        out[front] = uv.reshape(-1, 2)
    return out


def mask_distance(mask_dt: np.ndarray, uv: np.ndarray) -> float:
    """Distance-transform lookup of a projected point (nan if off-frame)."""
    h, w = mask_dt.shape
    u, v = uv
    if not np.isfinite([u, v]).all() or not (0 <= u < w and 0 <= v < h):
        return np.nan
    return float(mask_dt[int(v), int(u)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_dir", required=True, help="Extracted worldcode_* sample dir (contains dataset.hdf5)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_tiles", type=int, default=12)
    ap.add_argument("--tile_width", type=int, default=480)
    args = ap.parse_args()

    sample = Path(args.sample_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(sample / "dataset.hdf5", "r") as f:
        calib = f[f"meta/calibration/{CHEST_CAM}"]
        K = np.array(calib["intrinsic"])
        D = np.array(calib["distortion"])
        extrinsic = np.array(calib["extrinsic"])
        cam_table = f[f"observation/camera/{CHEST_CAM}"][:]
        eef = {}
        for side in ("left", "right"):
            tbl = f[f"pose/{side}_eef/feedback/pose_in_chest"][:]
            eef[side] = {"pos": tbl["value"][:, :3], "ts": tbl["timestamp"], "conf": tbl["confidence"]}

    cam_ts = cam_table["timestamp"]
    frame_paths = [sample / p.decode() for p in cam_table["file_path"]]

    # align eef samples to camera frames by nearest timestamp
    aligned = {}
    for side in ("left", "right"):
        idx = np.abs(eef[side]["ts"][None, :] - cam_ts[:, None]).argmin(axis=1)
        dt_ms = np.abs(eef[side]["ts"][idx] - cam_ts)
        aligned[side] = {"pos": eef[side]["pos"][idx], "dt_ms": dt_ms, "conf": eef[side]["conf"][idx]}

    # hand-mask distance transforms (mask stem timestamps differ from jpg names -> match by order)
    mask_dir = sample / "hand_masks" / CHEST_CAM
    mask_files = sorted(mask_dir.glob("*.png")) if mask_dir.exists() else []
    mask_dts = {}
    for i in range(len(frame_paths)):
        if i < len(mask_files):
            m = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
            if m is not None and (m > 0).any():
                mask_dts[i] = cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3)

    R_ext, t_ext = extrinsic[:3, :3], extrinsic[:3, 3]
    candidates = signed_permutations()
    scores = []
    n = len(frame_paths)
    for ci, R in enumerate(candidates):
        dists, inframe, total = [], 0, 0
        for i in range(n):
            for side in ("left", "right"):
                p_chest = aligned[side]["pos"][i]
                p_cam = R @ (R_ext.T @ (p_chest - t_ext))
                uv = project_fisheye(p_cam[None], K, D)[0]
                total += 1
                if np.isfinite(uv).all() and 0 <= uv[0] < 1920 and 0 <= uv[1] < 1536:
                    inframe += 1
                    if i in mask_dts:
                        dists.append(mask_distance(mask_dts[i], uv))
        med = float(np.nanmedian(dists)) if dists else np.inf
        scores.append({"cand": ci, "R": R.tolist(), "inframe_ratio": inframe / max(total, 1),
                       "median_px_to_hand_mask": med, "n_mask_frames": len(dists)})

    ranked = sorted(scores, key=lambda s: (s["median_px_to_hand_mask"], -s["inframe_ratio"]))
    best = ranked[0]
    R_best = np.array(best["R"])

    # per-frame stats under the winning convention
    per_frame = {"left": [], "right": []}
    for i in range(n):
        for side in ("left", "right"):
            p_cam = R_best @ (R_ext.T @ (aligned[side]["pos"][i] - t_ext))
            uv = project_fisheye(p_cam[None], K, D)[0]
            d = mask_distance(mask_dts[i], uv) if i in mask_dts else np.nan
            per_frame[side].append({"frame": i, "uv": [float(x) for x in uv], "px_to_mask": None if np.isnan(d) else d,
                                    "dt_ms": float(aligned[side]["dt_ms"][i])})

    # contact sheet
    tile_w = args.tile_width
    sel = np.linspace(0, n - 1, args.num_tiles, dtype=int)
    tiles = []
    for i in sel:
        img = Image.open(frame_paths[i]).convert("RGB")
        scale = tile_w / img.width
        tile = img.resize((tile_w, int(img.height * scale)))
        if i in mask_dts and i < len(mask_files):
            m = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
            cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            draw = ImageDraw.Draw(tile)
            for c in cnts:
                pts = [(float(x) * scale, float(y) * scale) for x, y in c.reshape(-1, 2)[::4]]
                if len(pts) > 1:
                    draw.line(pts, fill=MASK_COLOR, width=1)
        draw = ImageDraw.Draw(tile)
        for side, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
            u, v = per_frame[side][i]["uv"]
            if np.isfinite([u, v]).all():
                u, v = u * scale, v * scale
                r = 6
                draw.ellipse([u - r, v - r, u + r, v + r], outline=color, width=3)
                draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=color)
        draw.rectangle([0, 0, 60, 20], fill=(0, 0, 0))
        draw.text((4, 3), f"f{i}", fill=(255, 255, 255))
        tiles.append(tile)

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    th = tiles[0].height
    sheet = Image.new("RGB", (cols * tile_w, rows * th + 40), (16, 16, 16))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * tile_w, (k // cols) * th))
    d = ImageDraw.Draw(sheet)
    d.text((8, rows * th + 4), f"WIYH {sample.name[:80]}  wrist proj (L=blue R=red, mask=green)  "
                               f"median px->mask: {best['median_px_to_hand_mask']:.1f}", fill=(255, 255, 255))
    d.text((8, rows * th + 22), f"best R (chest->cam): {best['R']}  inframe: {best['inframe_ratio']:.2f}", fill=(255, 220, 150))
    sheet_path = out_dir / f"{sample.name[:100]}_wrist_overlay.jpg"
    sheet.save(sheet_path, quality=88)

    left_d = [p["px_to_mask"] for p in per_frame["left"] if p["px_to_mask"] is not None]
    right_d = [p["px_to_mask"] for p in per_frame["right"] if p["px_to_mask"] is not None]
    summary = {
        "sample": sample.name,
        "num_frames": n,
        "best_candidate": best,
        "runner_up": ranked[1],
        "left_px_to_mask": {"median": float(np.median(left_d)) if left_d else None,
                            "p90": float(np.percentile(left_d, 90)) if left_d else None, "n": len(left_d)},
        "right_px_to_mask": {"median": float(np.median(right_d)) if right_d else None,
                             "p90": float(np.percentile(right_d, 90)) if right_d else None, "n": len(right_d)},
        "eef_to_frame_dt_ms": {s: {"median": float(np.median(aligned[s]["dt_ms"])),
                                   "max": float(np.max(aligned[s]["dt_ms"]))} for s in ("left", "right")},
        "sheet": str(sheet_path),
        "note": "wrist-point-level audit only: WIYH ships 75-dim glove joint angles without a skeleton "
                "topology; finger-level 3D cannot be verified from shipped data alone",
    }
    (out_dir / f"{sample.name[:100]}_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("WIYH_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
