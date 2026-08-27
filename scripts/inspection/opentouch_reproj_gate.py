#!/usr/bin/env python3
"""OpenTouch reprojection gate (W9 re-examination, 2026-08-27).

The W9 close-out dismissed the 10 depth-plausible sessions on GT-vs-detector
agreement 0/11 — invalid evidence, since detector.pt is glove-blind on this
domain (any@0.3 = 0.175). This gate re-tests viability the way the HumanTouch
Stage-1 gate did: project the shipped GT landmarks onto the video frames with
the committed chain (w2c = T_device_camera @ inv(camera_pose_t), stale-hold
validity) and judge lock-on against the VISIBLE GLOVE, visually (overlay
sheets) and numerically (manual wrist annotations on raw frames).

Subcommands:
  depths            per-session median valid wrist depth -> plausible list
  render            overlay contact sheet per plausible session (skeleton on frames)
  grid  S C F1,F2   full-res raw frames + 50px grid for manual annotation
  eval              wrist px error at the ANN manual annotations

Usage:
  python scripts/inspection/opentouch_reproj_gate.py depths
  python scripts/inspection/opentouch_reproj_gate.py render
  python scripts/inspection/opentouch_reproj_gate.py grid fablab_ml_p1 demo_27 10,40,80
  python scripts/inspection/opentouch_reproj_gate.py eval
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

DATA_DIR = Path("/root/w9/opentouch")
OUT_DIR = Path("/root/w9_reopen/gate")
MAX_WRIST_Z = 1.0
STALE_EPS = 1e-12
TIP_IDS = [4, 8, 12, 16, 20]
# MediaPipe hand topology
FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
EDGES = [e for f in FINGERS for e in ([(0, f[0])] + list(zip(f[:-1], f[1:])))]

# Manual glove-wrist annotations (u, v at full 640x480 res), read off the `grid`
# renders (raw frames, no GT drawn). Annotated 2026-08-27 (W9 re-examination).
# home_bedroom demo_103 f28 was too dark to annotate and was skipped.
ANN: dict[tuple[str, str], list[tuple[int, float, float]]] = {
    ("fablab_ml_p1", "demo_49"): [(141, 437, 331), (256, 430, 370)],
    ("office_csail_p1", "demo_091"): [(21, 405, 260), (64, 465, 240)],
    ("home_bedroom", "demo_103"): [(62, 415, 360)],
}


def load_session(session: str):
    return h5py.File(DATA_DIR / f"{session}.hdf5", "r")


def clip_arrays(f, demo: str):
    calib = f["calibration/rgb"]
    focal = float(calib["focal_length"][()])
    pp = np.asarray(calib["principal_point"], np.float64)
    T_dev_cam = np.asarray(calib["T_device_camera"], np.float64)
    clip = f["data"][demo]
    poses = np.asarray(clip["camera_poses"], np.float64)
    lms = np.asarray(clip["right_hand_landmarks"], np.float64)
    T = min(len(poses), len(lms))
    poses, lms = poses[:T], lms[:T]
    w2c = np.einsum("ij,tjk->tik", T_dev_cam, np.linalg.inv(poses))
    finite = np.isfinite(lms).all(axis=(1, 2))
    stale = np.zeros(T, bool)
    if T > 1:
        stale[1:] = np.abs(np.diff(lms, axis=0)).max(axis=(1, 2)) < STALE_EPS
    valid = finite & ~stale
    cam = np.einsum("tij,tkj->tki", w2c[:, :3, :3], lms) + w2c[:, None, :3, 3]  # (T,21,3)
    z = np.clip(cam[..., 2:3], 1e-6, None)
    uv = focal * cam[..., :2] / z + pp  # (T,21,2)
    labels = clip["labels"][0] if "labels" in clip else None
    return dict(uv=uv, z=cam[..., 2], valid=valid, labels=labels, clip=clip, T=T)


def cmd_depths():
    rows = []
    for p in sorted(DATA_DIR.glob("*.hdf5")):
        with h5py.File(p, "r") as f:
            meds = []
            for demo in list(f["data"].keys()):
                a = clip_arrays(f, demo)
                if a["valid"].any():
                    meds.append(float(np.median(a["z"][a["valid"], 0])))
            m = float(np.median(meds)) if meds else np.nan
            rows.append((p.stem, m, len(meds)))
    rows.sort(key=lambda r: r[1])
    plausible = []
    for s, m, n in rows:
        tag = "PLAUSIBLE" if m <= MAX_WRIST_Z else "corrupt-depth"
        if m <= MAX_WRIST_Z:
            plausible.append(s)
        print(f"{s:26s} median_wrist_z={m:5.2f} m  clips={n:4d}  {tag}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plausible_sessions.json").write_text(json.dumps(plausible, indent=1))
    print(f"plausible: {len(plausible)}/{len(rows)}")


def pick_clips(f, k=3):
    """Clips with the most valid frames, skipping hand_out_of_frame ones."""
    scored = []
    for demo in f["data"].keys():
        a = clip_arrays(f, demo)
        if a["labels"] is not None and int(a["labels"]["hand_out_of_frame"]):
            continue
        W, H = 640, 480
        infr = (a["uv"][:, 0, 0] >= 0) & (a["uv"][:, 0, 0] < W) & \
               (a["uv"][:, 0, 1] >= 0) & (a["uv"][:, 0, 1] < H) & a["valid"]
        scored.append((int(infr.sum()), demo))
    scored.sort(reverse=True)
    return [d for _, d in scored[:k]]


def draw_skel(draw: ImageDraw.ImageDraw, uv: np.ndarray, z: np.ndarray, scale: float):
    pts = uv * scale
    ok = (z > 0.05)
    for a, b in EDGES:
        if ok[a] and ok[b]:
            draw.line([tuple(pts[a]), tuple(pts[b])], fill=(0, 255, 90), width=2)
    for j in range(21):
        if not ok[j]:
            continue
        r = 4 if j in TIP_IDS else 2
        col = (255, 60, 60) if j == 0 else (0, 255, 90)
        if j == 0:
            r = 6
        draw.ellipse([pts[j, 0] - r, pts[j, 1] - r, pts[j, 0] + r, pts[j, 1] + r], fill=col)


def cmd_render(sessions, tiles_per_clip=6, clips_per_session=2, tile_w=480):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        with load_session(session) as f:
            demos = pick_clips(f, clips_per_session)
            tiles = []
            for demo in demos:
                a = clip_arrays(f, demo)
                vidx = np.flatnonzero(a["valid"])
                if len(vidx) == 0:
                    continue
                sel = vidx[np.linspace(0, len(vidx) - 1, tiles_per_clip, dtype=int)]
                for t in sel:
                    img = Image.open(io.BytesIO(a["clip"]["rgb_images_jpeg"][t])).convert("RGB")
                    scale = tile_w / img.width
                    tile = img.resize((tile_w, int(img.height * scale)))
                    d = ImageDraw.Draw(tile)
                    draw_skel(d, a["uv"][t], a["z"][t], scale)
                    d.rectangle([0, 0, 190, 18], fill=(0, 0, 0))
                    d.text((4, 2), f"{demo} f{t} z={a['z'][t, 0]:.2f}m", fill=(255, 255, 255))
                    tiles.append(tile)
            if not tiles:
                continue
            cols = 3
            rows = (len(tiles) + cols - 1) // cols
            th = tiles[0].height
            sheet = Image.new("RGB", (cols * tile_w, rows * th + 26), (16, 16, 16))
            for k, t in enumerate(tiles):
                sheet.paste(t, ((k % cols) * tile_w, (k // cols) * th))
            d = ImageDraw.Draw(sheet)
            d.text((8, rows * th + 5), f"OpenTouch {session}: GT skeleton via w2c=T_dev_cam@inv(pose), "
                                       "valid frames only. wrist=RED", fill=(255, 255, 255))
            out = OUT_DIR / f"gate_{session}_overlay.jpg"
            sheet.save(out, quality=88)
            print(out)


def cmd_grid(session, demo, frames, up=2):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with load_session(session) as f:
        a = clip_arrays(f, demo)
        for t in frames:
            img = Image.open(io.BytesIO(a["clip"]["rgb_images_jpeg"][t])).convert("RGB")
            W, H = img.width, img.height
            img = img.resize((W * up, H * up))
            d = ImageDraw.Draw(img)
            for x in range(0, W + 1, 50):
                d.line([(x * up, 0), (x * up, H * up)], fill=(255, 255, 0), width=1)
                d.text((x * up + 2, 2), str(x), fill=(255, 255, 0))
            for y in range(0, H + 1, 50):
                d.line([(0, y * up), (W * up, y * up)], fill=(255, 255, 0), width=1)
                d.text((2, y * up + 2), str(y), fill=(255, 255, 0))
            out = OUT_DIR / f"grid_{session}_{demo}_f{t}.jpg"
            img.save(out, quality=90)
            print(out, "gt_wrist_uv=", np.round(a["uv"][t, 0], 1), "valid=", bool(a["valid"][t]))


def cmd_eval():
    errs_all = {}
    for (session, demo), pts in ANN.items():
        with load_session(session) as f:
            a = clip_arrays(f, demo)
            errs = []
            for t, u, v in pts:
                e = float(np.linalg.norm(a["uv"][t, 0] - [u, v]))
                errs.append(e)
                print(f"{session}/{demo} f{t}: gt={np.round(a['uv'][t, 0], 1)} ann=({u},{v}) err={e:.1f}px valid={bool(a['valid'][t])}")
            errs_all[f"{session}/{demo}"] = errs
    flat = [e for v in errs_all.values() for e in v]
    if flat:
        print(f"\nOVERALL wrist error: n={len(flat)} median={np.median(flat):.1f}px "
              f"p90={np.percentile(flat, 90):.1f}px max={max(flat):.1f}px")
    per_sess = {}
    for k, v in errs_all.items():
        per_sess.setdefault(k.split("/")[0], []).extend(v)
    for s, v in sorted(per_sess.items()):
        print(f"  {s:24s} n={len(v):2d} median={np.median(v):6.1f}px")
    (OUT_DIR / "wrist_eval.json").write_text(json.dumps(
        {"per_clip": errs_all,
         "overall": {"n": len(flat), "median_px": float(np.median(flat)) if flat else None,
                     "p90_px": float(np.percentile(flat, 90)) if flat else None}}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["depths", "render", "grid", "eval"])
    ap.add_argument("session", nargs="?")
    ap.add_argument("demo", nargs="?")
    ap.add_argument("frames", nargs="?")
    a = ap.parse_args()
    if a.cmd == "depths":
        cmd_depths()
    elif a.cmd == "render":
        sessions = json.loads((OUT_DIR / "plausible_sessions.json").read_text()) \
            if not a.session else [a.session]
        cmd_render(sessions)
    elif a.cmd == "grid":
        cmd_grid(a.session, a.demo, [int(x) for x in a.frames.split(",")])
    else:
        cmd_eval()


if __name__ == "__main__":
    main()
