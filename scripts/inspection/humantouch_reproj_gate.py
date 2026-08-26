#!/usr/bin/env python3
"""HumanTouch Stage-1 reprojection gate check (W10, 2026-08-26).

Projects MANUS 25-joint world-frame GT through head_tracker pose ->
camera_from_head_tracker -> per-device ChArUco intrinsics onto cam_head frames,
and solves/evaluates per-session extrinsics.

Verified chain (all confirmed empirically on 10 episodes / 5 tasks / 5 devices):
  p_tracker = R(q_head)^T (p_world - t_head)          # pose.head = tracker->world, quat xyzw
  p_cam     = M @ p_tracker + t                        # M IMPROPER (det=-1): one y-flip
                                                       # (unity_to_opencv_y_flip), base
                                                       # composition Rz(180)Ry(178)Rx(50) @ diag(1,-1,1)
  px        = distort(K, dist, p_cam)                  # OpenCV 5-coeff pinhole

Findings the gate rests on:
- Dataset left/right labels are CORRECT (swap hypothesis rejected: anchored fit
  median 18px 'same' vs 67-78px 'swap').
- The SHIPPED extrinsic is a hand-tuned viz offset: 450-800px median wrist error
  under the best of 96+ composition variants. Unusable without refinement.
- A per-session rigid extrinsic solved from 8 manual landmark annotations
  (2 frames x [wrist crease + middle-MCP] x 2 hands) locks skeletons onto the
  gloves episode-wide: 18px median at fit points, ~36px at held-out points
  (annotation noise +-25px), fingertips track visible glove fingers with correct
  chirality. Camera-from-tracker |t| = 0.235 m (plausible rig geometry).
- Mount varies BETWEEN sessions (~9-14 deg => ~120px if uncorrected) but is
  stable across ADJACENT episodes (block-stable): calibration effort scales with
  mount blocks, not episodes.
- Automatic refiners tried (darkness, symmetric chamfer, glove-blob centroids,
  MANUS dorsal-unit PnP) all converge only to 36-120px and can fail silently;
  none validated to the <=25px bar yet. Full conversion needs a per-block
  anchoring pass (manual or hardened auto-cal) first.

Usage:
  python scripts/inspection/humantouch_reproj_gate.py solve   # solve anchor from ANN
  python scripts/inspection/humantouch_reproj_gate.py eval    # error table at ANN points
  python scripts/inspection/humantouch_reproj_gate.py render X009 005346 300,1000
Requires sample episodes + sidecars staged as in the Stage-1 scratchpad layout
(see --root; default the W10 gate scratchpad).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rt

DEFAULT_ROOT = ('/tmp/claude-0/-root-egosmith/6b57c32e-2fab-4780-bf21-8479371681d9/'
                'scratchpad/ht_calib')
W, H = 1920, 1080
F = np.diag([1.0, -1.0, 1.0])
T0 = np.array([-0.034, 0.137, -0.028])          # shipped viz translation
RPY = np.array([50.0, 178.0, 180.0])            # shipped viz rpy (deg)

# Manual landmark annotations (frame, dataset side, joint idx, u, v), joint 0 =
# wrist root, 11 = middle-MCP knuckle. Sides verified physically correct.
ANN = {
    ('X009', '005346'): [
        (654, 'left', 0, 870, 870), (654, 'left', 11, 930, 720),
        (654, 'right', 0, 1415, 805), (654, 'right', 11, 1280, 670),
        (1248, 'left', 0, 880, 880), (1248, 'left', 11, 820, 750),
        (1248, 'right', 0, 1290, 750), (1248, 'right', 11, 1190, 640)],
    ('X007', '000800'): [
        (400, 'right', 0, 1140, 610), (400, 'right', 11, 1030, 550),
        (400, 'left', 0, 450, 870)],
    ('X001', '000005'): [
        (300, 'right', 0, 1480, 670), (300, 'right', 11, 1390, 560),
        (300, 'left', 0, 660, 820), (300, 'left', 11, 780, 660)],
}

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14),
           (15, 16, 17, 18, 19), (20, 21, 22, 23, 24)]
EDGES = [e for f in FINGERS for e in ([(0, f[0])] + list(zip(f[:-1], f[1:])))]


def base_M() -> np.ndarray:
    """Best-scoring shipped-rpy composition (improper: includes the y-flip)."""
    R = Rt.from_euler('ZYX', [RPY[2], RPY[1], RPY[0]], degrees=True).as_matrix()
    return R @ F


def load_calib(root: str, task: str, ep: str):
    for c in (f'{root}/{task}_ep{ep}.json', f'{root}/{task}_ep1.json'):
        if Path(c).exists():
            cc = json.loads(Path(c).read_text())['camera_calibration']
            intr = [v for k, v in cc.items() if 'intrinsics' in k][0]
            return (np.array(intr['camera_matrix']['matrix']),
                    np.array(intr['distortion_coefficients']['values']))
    raise FileNotFoundError(f'no sidecar for {task} {ep} under {root}')


def project_px(p: np.ndarray, K: np.ndarray, dist: np.ndarray):
    z = p[:, 2]
    ok = z > 0.05
    xn = np.zeros((len(p), 2))
    xn[ok] = p[ok, :2] / z[ok, None]
    r2 = (xn ** 2).sum(1)
    ok = ok & (r2 < 2.0)
    k1, k2, p1, p2, k3 = dist
    x, y = xn[:, 0], xn[:, 1]
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    yd = y * rad + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1), ok


def tracker_pts(row, side: str, joints=None) -> np.ndarray:
    hp = np.array(row['observation.human.pose.head'])
    Rwh = Rt.from_quat(hp[3:]).as_matrix()
    pts = np.stack(row[f'observation.human.hand_skeleton.{side}.position'])
    if joints is not None:
        pts = pts[joints]
    return (pts - hp[:3]) @ Rwh


def solve_anchor(root: str):
    task, ep = 'X009', '005346'
    df = pd.read_parquet(f'{root}/samples/{task}_{ep}.parquet')
    K, dist = load_calib(root, task, ep)
    M0 = base_M()
    obs = [(tracker_pts(df.iloc[fi], side, [j])[0], np.array([u, v], float))
           for fi, side, j, u, v in ANN[(task, ep)]]

    def resid(x):
        M = Rt.from_rotvec(x[:3]).as_matrix() @ M0
        t = T0 + x[3:]
        r = []
        for ph, uv in obs:
            px, ok = project_px((ph @ M.T + t).reshape(1, 3), K, dist)
            r += list(px[0] - uv if ok[0] else [999.0, 999.0])
        return np.array(r)

    sol = least_squares(resid, np.zeros(6), method='lm', max_nfev=8000)
    M = Rt.from_rotvec(sol.x[:3]).as_matrix() @ M0
    t = T0 + sol.x[3:]
    e = np.linalg.norm(resid(sol.x).reshape(-1, 2), axis=1)
    print(f'anchor solved on {task}_{ep}: median {np.median(e):.1f}px '
          f'max {e.max():.1f}px |t| {np.linalg.norm(t):.3f}m')
    out = {'M': M.tolist(), 't': t.tolist(), 'fit_median_px': float(np.median(e))}
    Path(f'{root}/anchor_X009_005346.json').write_text(json.dumps(out, indent=1))
    return M, t


def eval_all(root: str):
    anchor = json.loads(Path(f'{root}/anchor_X009_005346.json').read_text())
    variants = {'shipped': (base_M(), T0),
                'anchor': (np.array(anchor['M']), np.array(anchor['t']))}
    for (task, ep), pts in ANN.items():
        df = pd.read_parquet(f'{root}/samples/{task}_{ep}.parquet')
        K, dist = load_calib(root, task, ep)
        line = f'{task}_{ep}:'
        for name, (M, t) in variants.items():
            errs = []
            for fi, side, j, u, v in pts:
                ph = tracker_pts(df.iloc[fi], side, [j])[0]
                px, ok = project_px((ph @ M.T + t).reshape(1, 3), K, dist)
                errs.append(np.linalg.norm(px[0] - [u, v]) if ok[0] else np.inf)
            line += f'  {name} med {np.median(errs):7.1f}px'
        print(line)


def render(root: str, task: str, ep: str, frames):
    anchor = json.loads(Path(f'{root}/anchor_X009_005346.json').read_text())
    M, t = np.array(anchor['M']), np.array(anchor['t'])
    df = pd.read_parquet(f'{root}/samples/{task}_{ep}.parquet')
    K, dist = load_calib(root, task, ep)
    cap = cv2.VideoCapture(f'{root}/samples/{task}_{ep}_head.mp4')
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        for side, color in (('left', (0, 0, 255)), ('right', (255, 128, 0))):
            px, okz = project_px(tracker_pts(df.iloc[fi], side) @ M.T + t, K, dist)
            for a, b in EDGES:
                if okz[a] and okz[b]:
                    cv2.line(img, tuple(px[a].astype(int)), tuple(px[b].astype(int)), color, 2)
            for j in range(25):
                if okz[j]:
                    cv2.circle(img, tuple(px[j].astype(int)),
                               8 if j == 0 else 4, color, -1)
        out = f'{root}/overlays_ref/gate_{task}_{ep}_f{fi}.jpg'
        Path(out).parent.mkdir(exist_ok=True)
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=['solve', 'eval', 'render'])
    ap.add_argument('task', nargs='?')
    ap.add_argument('ep', nargs='?')
    ap.add_argument('frames', nargs='?')
    ap.add_argument('--root', default=DEFAULT_ROOT)
    a = ap.parse_args()
    if a.cmd == 'solve':
        solve_anchor(a.root)
    elif a.cmd == 'eval':
        eval_all(a.root)
    else:
        render(a.root, a.task, a.ep, [int(x) for x in a.frames.split(',')])
