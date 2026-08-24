#!/usr/bin/env python3
"""CAT-2.5 pose-accuracy audit for EgoTouch (TouchAnything).

Per-episode dir ships: chest/left/right.mp4 (640x480@30), hamer_hands.json +
wilor_hands.json (per-frame 21x3 camera-frame monocular pseudo-labels, JSONL),
rokoko_hands.json (21x3 world-frame EMF glove mocap, epoch ts), vive_poses.json,
jq_pressure.json (256ch/hand raw + quats), pressure_grids.npz (T,21,21 processed),
masks.npz (T,2,480,640 hand masks), manual_contact_annotation.json.

Probes:
  1. focal probe for HaMeR/WiLoR camera-frame joints: candidate pinhole focals
     scored by projected-joint distance to shipped hand masks;
  2. glove-vs-pseudo-label consistency: per-frame Procrustes-aligned MPJPE
     rokoko vs hamer/wilor + rokoko bone-length stability;
  3. stream alignment: rokoko/jq epoch-ts cadence vs 30fps frame clock, record
     counts vs video frames;
  4. tactile sanity numbers for the validator spec.

Renders an overlay sheet (hamer under best focal + mask contours).

Usage:
  python scripts/inspection/cat25_egotouch_audit.py --episode_dir <dir> --out_dir /root/cat25_audits/egotouch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

LEFT_COLOR = (66, 133, 244)
RIGHT_COLOR = (219, 68, 55)
MASK_COLOR = (0, 255, 120)
W, H = 640, 480
FOCAL_CANDIDATES = [5000.0, 5000.0 * 640 / 256, 5000.0 * 480 / 256, 2500.0, 1250.0, 600.0]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
         (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def joints_series(records: list[dict]) -> dict[str, np.ndarray]:
    """-> {'left': (T,21,3) with nan rows, 'right': ...}"""
    T = len(records)
    out = {s: np.full((T, 21, 3), np.nan) for s in ("left", "right")}
    for i, r in enumerate(records):
        for s in ("left", "right"):
            p = r.get(f"{s}_pos")
            if p is not None:
                arr = np.asarray(p, dtype=float)
                if arr.shape == (21, 3):
                    out[s][i] = arr
    return out


def project(pts: np.ndarray, f: float) -> np.ndarray:
    z = np.where(np.abs(pts[..., 2:3]) < 1e-6, np.nan, pts[..., 2:3])
    u = f * pts[..., 0:1] / z + W / 2
    v = f * pts[..., 1:2] / z + H / 2
    return np.concatenate([u, v, z], axis=-1)


def procrustes_mpjpe(A: np.ndarray, B: np.ndarray) -> float:
    """Rigid-align A->B (both (21,3)), return mean joint error in meters."""
    a, b = A - A.mean(0), B - B.mean(0)
    sa = np.linalg.norm(a)
    if sa < 1e-9 or np.linalg.norm(b) < 1e-9:
        return np.nan
    a = a * (np.linalg.norm(b) / sa)  # allow scale (HaMeR depth is scale-ambiguous)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return float(np.linalg.norm((R @ a.T).T - b, axis=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_tiles", type=int, default=12)
    ap.add_argument("--tile_width", type=int, default=480)
    args = ap.parse_args()

    ep = Path(args.episode_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = ep.name

    hamer = joints_series(read_jsonl(ep / "hamer_hands.json"))
    wilor = joints_series(read_jsonl(ep / "wilor_hands.json"))
    rokoko_rec = read_jsonl(ep / "rokoko_hands.json")
    rokoko = joints_series(rokoko_rec)
    jq = read_jsonl(ep / "jq_pressure.json")

    cap = cv2.VideoCapture(str(ep / "chest.mp4"))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    masks = None
    if (ep / "masks.npz").exists():
        mz = np.load(ep / "masks.npz")
        masks = mz["masks"]  # (T,2,H,W) uint8

    mono = hamer if len(hamer["left"]) else wilor
    mono_name = "hamer" if len(hamer["left"]) else "wilor"
    T = len(mono["left"])

    # 1. focal probe against masks
    focal_result = {}
    if masks is not None and T:
        sel = np.linspace(0, min(T, masks.shape[0]) - 1, 40, dtype=int)
        dts = {}
        for i in sel:
            m = (masks[i].max(axis=0) > 0).astype(np.uint8)
            if m.any():
                dts[i] = cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3)
        for f in FOCAL_CANDIDATES:
            d_all = []
            for i in dts:
                for s in ("left", "right"):
                    uvz = project(mono[s][i], f)
                    for u, v, z in uvz:
                        if np.isfinite([u, v]).all() and 0 <= u < W and 0 <= v < H:
                            d_all.append(dts[i][int(v), int(u)])
            focal_result[f] = {"median_px_to_mask": float(np.median(d_all)) if d_all else None,
                               "n_pts": len(d_all)}
        best_f = min((f for f in focal_result if focal_result[f]["median_px_to_mask"] is not None),
                     key=lambda f: focal_result[f]["median_px_to_mask"], default=FOCAL_CANDIDATES[0])
    else:
        best_f = FOCAL_CANDIDATES[0]

    # 2. glove vs pseudo-label consistency
    consist = {}
    for s in ("left", "right"):
        errs = []
        n = min(len(rokoko[s]), len(mono[s]))
        for i in range(n):
            if np.isfinite(rokoko[s][i]).all() and np.isfinite(mono[s][i]).all():
                errs.append(procrustes_mpjpe(rokoko[s][i], mono[s][i]))
        errs = [e for e in errs if np.isfinite(e)]
        # rokoko bone-length stability
        bl_cv = None
        valid = rokoko[s][np.isfinite(rokoko[s]).all(axis=(1, 2))]
        if len(valid) > 5:
            bl = np.stack([np.linalg.norm(valid[:, a] - valid[:, b], axis=1) for a, b in BONES], axis=1)
            mean = bl.mean(axis=0)
            bl_cv = float((bl.std(axis=0) / np.where(mean < 1e-9, np.nan, mean)).mean())
        consist[s] = {"procrustes_mpjpe_cm": {"median": float(np.median(errs) * 100) if errs else None,
                                              "p90": float(np.percentile(errs, 90) * 100) if errs else None,
                                              "n": len(errs)},
                      "rokoko_bone_len_cv": bl_cv,
                      "rokoko_valid_frames": int(len(valid))}

    # 3. stream alignment
    def cadence(records):
        ts = np.array([r["ts"] for r in records], dtype=float)
        if len(ts) < 3:
            return None
        dt = np.diff(ts)
        return {"n": len(ts), "median_dt_ms": float(np.median(dt) * 1000),
                "max_dt_ms": float(dt.max() * 1000), "monotonic": bool((dt > 0).all())}

    align = {"video_frames": n_video, "fps": fps,
             "mono_records": T, "rokoko": cadence(rokoko_rec), "jq": cadence(jq),
             "frame_count_match": bool(T == n_video == len(rokoko_rec))}

    # 4. tactile sanity
    tact = {}
    if jq:
        for s in ("left", "right"):
            arr = np.array([r[f"sensor_{s}"] for r in jq if r.get(f"sensor_{s}") is not None], dtype=float)
            if arr.size:
                ch_std = arr.std(axis=0)
                tact[f"jq_{s}"] = {"channels": arr.shape[1], "min": float(arr.min()), "max": float(arr.max()),
                                   "median": float(np.median(arr)), "frozen_std0": int((ch_std == 0).sum()),
                                   "active_frac_gt0": float((arr > 0).mean())}
    if (ep / "pressure_grids.npz").exists():
        gz = np.load(ep / "pressure_grids.npz")
        for s in ("left", "right"):
            g = gz[f"{s}_pressure_grid"]
            flat = g.reshape(len(g), -1)
            tact[f"grid_{s}"] = {"shape": list(g.shape),
                                 "nan_frac": float(np.isnan(g).mean()),
                                 "min": float(np.nanmin(g)), "max": float(np.nanmax(g)),
                                 "frac_frames_any_pressure": float((np.nanmax(flat, axis=1) > 0).mean())}
        tact["tactile_max_attr"] = float(gz["tactile_max"]) if "tactile_max" in gz else None
    if (ep / "manual_contact_annotation.json").exists():
        tact["manual_contact"] = json.loads((ep / "manual_contact_annotation.json").read_text())

    # overlay sheet
    sel = np.linspace(0, T - 1, args.num_tiles, dtype=int) if T else []
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
        if masks is not None and i < masks.shape[0]:
            m = (masks[i].max(axis=0) > 0).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                pts = [(float(x) * scale, float(y) * scale) for x, y in c.reshape(-1, 2)[::4]]
                if len(pts) > 1:
                    draw.line(pts, fill=MASK_COLOR, width=1)
        for s, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
            uvz = project(mono[s][i], best_f)
            for u, v, z in uvz * np.array([scale, scale, 1.0]):
                if np.isfinite([u, v]).all():
                    draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=color)
        draw.rectangle([0, 0, 60, 18], fill=(0, 0, 0))
        draw.text((4, 2), f"f{int(i)}", fill=(255, 255, 255))
        tiles.append(tile)
    if tiles:
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        th = tiles[0].height
        sheet = Image.new("RGB", (cols * args.tile_width, rows * th + 40), (16, 16, 16))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * args.tile_width, (k // cols) * th))
        d = ImageDraw.Draw(sheet)
        d.text((8, rows * th + 4), f"EgoTouch {name} {mono_name} f={best_f:.0f} (L=blue R=red, mask=green)",
               fill=(255, 255, 255))
        sheet_path = out_dir / f"{name}_{mono_name}_overlay.jpg"
        sheet.save(sheet_path, quality=88)
    else:
        sheet_path = None

    summary = {"episode": name, "mono_source": mono_name, "focal_probe": focal_result,
               "best_focal": best_f, "glove_vs_mono": consist, "alignment": align,
               "tactile": tact, "sheet": str(sheet_path)}
    (out_dir / f"{name}_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("EGOTOUCH_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
