#!/usr/bin/env python3
"""HumanTouch per-block anchor pass (W10 Stage-2, step 2).

For each mount block from the census (scripts/build/humantouch_block_census.py),
solve a per-block camera_from_head_tracker extrinsic from 8 manually-annotated
glove landmarks (2 frames x [wrist root j0 + middle-MCP j11] x 2 hands) on a
representative episode, seeded by the X009 Stage-1 anchor (+ the block's census
chamfer fingerprint rotation).

Workflow per block (the agent IS the annotator, via vision):
  prep    download full-res head mp4 + GT, auto-pick frames with both hands
          in view, write annotation sheets (full frame + per-hand zoom crops
          with predicted j0/j11 markers)
  solve   LM solve from an annotation JSON; report fit residuals
  render  full-skeleton overlays on held-out frames for visual verification

Annotation JSON format ({WORK}/ann/<block>.json):
  {"task": "X009", "ep": "005346",
   "points": [[frame, side, joint, u, v], ...]}   # full-res px

Outputs: {WORK}/extrinsics/<block>.json  {M, t, fit_median_px, fit_max_px, ...}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rt

WORK = Path('/root/w7_full/humantouch/anchors')
CENSUS = Path('/root/w7_full/humantouch/census')
CALIB = '/root/w7_full/humantouch/humantouch_calibration.json'
ANCHOR_JSON = '/root/w7_full/humantouch/stage1_gate/anchor_X009_005346.json'
SRC = 'gs://foundational-research/hoi-dataset/Xspark-HumanTouch'
W, H = 1920, 1080

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14),
           (15, 16, 17, 18, 19), (20, 21, 22, 23, 24)]
EDGES = [e for f in FINGERS for e in ([(0, f[0])] + list(zip(f[:-1], f[1:])))]


def load_calib_full(dev):
    cal = json.load(open(CALIB))
    intr = cal['devices'][dev]['intrinsics']
    return (np.array(intr['camera_matrix']['matrix'], float),
            np.array(intr['distortion_coefficients']['values'], float))


def anchor_Mt():
    a = json.load(open(ANCHOR_JSON))
    return np.array(a['M']), np.array(a['t'])


def project_px(p, K, dist):
    z = p[:, 2]
    okz = z > 0.05
    xn = np.zeros((len(p), 2))
    xn[okz] = p[okz, :2] / z[okz, None]
    r2 = (xn ** 2).sum(1)
    okz = okz & (r2 < 2.0)
    k1, k2, p1, p2, k3 = dist
    x, y = xn[:, 0], xn[:, 1]
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    yd = y * rad + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1), okz


def tracker_pts(row, side, joints=None):
    hp = np.array(row['observation.human.pose.head'])
    Rwh = Rt.from_quat(hp[3:]).as_matrix()
    pts = np.stack(row[f'observation.human.hand_skeleton.{side}.position'])
    if joints is not None:
        pts = pts[joints]
    return (pts - hp[:3]) @ Rwh


def blocks():
    return {b['block_id']: b for b in json.load(open(CENSUS / 'blocks.json'))}


def block_seed(block):
    """Anchor extrinsic rotated by the block's census fingerprint median."""
    Ma, ta = anchor_Mt()
    rv = block.get('fp_median_rotvec_deg')
    if rv is None:
        return Ma, ta
    return Rt.from_rotvec(np.deg2rad(rv)).as_matrix() @ Ma, ta


def stage_episode(task, ep_key):
    d = WORK / 'samples'
    d.mkdir(parents=True, exist_ok=True)
    mp4 = d / f'{task}_{ep_key}.mp4'
    pq = d / f'{task}_{ep_key}.parquet'
    if not mp4.exists():
        subprocess.run(['gsutil', '-q', 'cp',
                        f'{SRC}/{task}/videos/chunk-000/observation.images.cam_head/episode_{ep_key}.mp4',
                        str(mp4)], check=True)
    if not pq.exists():
        subprocess.run(['gsutil', '-q', 'cp',
                        f'{SRC}/{task}/data/chunk-000/episode_{ep_key}.parquet',
                        str(pq)], check=True)
    return mp4, pq


def pick_frames(df, M, t, K, dist, n=4, min_wrist_sep=250.0):
    """Frames where both wrists project comfortably in-frame, well separated."""
    cands = []
    for fi in range(60, len(df) - 30, 12):
        row = df.iloc[fi]
        if not row['observation.human.pose.valid'][0]:
            continue
        try:
            pl = tracker_pts(row, 'left', [0, 11])
            pr = tracker_pts(row, 'right', [0, 11])
        except Exception:  # noqa: BLE001
            continue
        px, ok = project_px(np.concatenate([pl, pr]) @ M.T + t, K, dist)
        if not ok.all():
            continue
        m = 180
        if not ((px[:, 0] > m) & (px[:, 0] < W - m) & (px[:, 1] > m) & (px[:, 1] < H - m)).all():
            continue
        sep = np.linalg.norm(px[0] - px[2])
        if sep < min_wrist_sep:
            continue
        cands.append((fi, px, sep))
    if not cands:
        return []
    # spread picks over time
    picks = []
    idxs = np.linspace(0, len(cands) - 1, min(n, len(cands))).astype(int)
    for i in sorted(set(idxs)):
        picks.append(cands[i])
    return picks


def cmd_prep(args):
    bl = blocks()[args.block]
    task = bl['task']
    ep_key = args.ep or bl.get('anchor_ep') or bl['ep_mid']
    M, t = block_seed(bl)
    K, dist = load_calib_full(bl['dev'])
    mp4, pq = stage_episode(task, ep_key)
    df = pd.read_parquet(pq)
    picks = pick_frames(df, M, t, K, dist, n=args.n_frames)
    if not picks:
        print(f'NO_FRAMES for {args.block} {task}_{ep_key}')
        return
    cap = cv2.VideoCapture(str(mp4))
    outdir = WORK / 'sheets' / args.block
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dict(block=args.block, task=task, ep=ep_key, dev=bl['dev'], frames=[])
    for fi, px, sep in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        row = df.iloc[fi]
        vis = img.copy()
        # draw predicted skeleton faintly + j0/j11 markers
        for side, color in (('left', (0, 0, 255)), ('right', (255, 128, 0))):
            pts = tracker_pts(row, side)
            pxs, okz = project_px(pts @ M.T + t, K, dist)
            for a, b in EDGES:
                if okz[a] and okz[b]:
                    cv2.line(vis, tuple(pxs[a].astype(int)), tuple(pxs[b].astype(int)), color, 1)
            for j, r in ((0, 10), (11, 7)):
                if okz[j]:
                    cv2.circle(vis, tuple(pxs[j].astype(int)), r, color, 2)
        cv2.imwrite(str(outdir / f'f{fi}_full.jpg'), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        fmeta = dict(frame=fi, crops={})
        # per-hand zoom crops around predicted wrist, with grid ticks
        for side in ('left', 'right'):
            pts = tracker_pts(row, side, [0, 11])
            pxs, okz = project_px(pts @ M.T + t, K, dist)
            cx, cy = pxs.mean(0).astype(int)
            half = args.crop_half
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(W, cx + half), min(H, cy + half)
            crop = img[y0:y1, x0:x1].copy()
            crop = cv2.resize(crop, None, fx=args.zoom, fy=args.zoom,
                              interpolation=cv2.INTER_CUBIC)
            for j, col, r in ((0, (0, 255, 255), 12), (1, (255, 0, 255), 9)):
                if okz[j]:
                    p = ((pxs[j] - [x0, y0]) * args.zoom).astype(int)
                    cv2.drawMarker(crop, tuple(p), col, cv2.MARKER_CROSS, 2 * r, 2)
            cv2.imwrite(str(outdir / f'f{fi}_{side}.jpg'), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            fmeta['crops'][side] = dict(x0=int(x0), y0=int(y0), zoom=args.zoom,
                                        pred_j0=list(map(float, pxs[0])),
                                        pred_j11=list(map(float, pxs[1])))
        meta['frames'].append(fmeta)
    (outdir / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def solve_block(task, ep_key, dev, points, M0, t0):
    df = pd.read_parquet(WORK / 'samples' / f'{task}_{ep_key}.parquet')
    K, dist = load_calib_full(dev)
    obs = [(tracker_pts(df.iloc[fi], side, [j])[0], np.array([u, v], float))
           for fi, side, j, u, v in points]

    def resid(x):
        M = Rt.from_rotvec(x[:3]).as_matrix() @ M0
        t = t0 + x[3:]
        r = []
        for ph, uv in obs:
            px, ok = project_px((ph @ M.T + t).reshape(1, 3), K, dist)
            r += list(px[0] - uv if ok[0] else [999.0, 999.0])
        return np.array(r)

    sol = least_squares(resid, np.zeros(6), method='lm', max_nfev=8000)
    M = Rt.from_rotvec(sol.x[:3]).as_matrix() @ M0
    t = t0 + sol.x[3:]
    e = np.linalg.norm(resid(sol.x).reshape(-1, 2), axis=1)
    return M, t, e


def cmd_solve(args):
    bl = blocks()[args.block]
    ann = json.load(open(WORK / 'ann' / f'{args.block}.json'))
    task, ep_key = ann['task'], ann['ep']
    M0, t0 = block_seed(bl)
    M, t, e = solve_block(task, ep_key, bl['dev'], ann['points'], M0, t0)
    out = dict(block=args.block, task=task, ep=ep_key, dev=bl['dev'],
               M=M.tolist(), t=t.tolist(),
               fit_median_px=float(np.median(e)), fit_max_px=float(e.max()),
               n_points=len(ann['points']),
               t_norm_m=float(np.linalg.norm(t)))
    d = WORK / 'extrinsics'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{args.block}.json').write_text(json.dumps(out, indent=1))
    print(f'{args.block} {task}_{ep_key}: fit median {np.median(e):.1f}px '
          f'max {e.max():.1f}px |t| {np.linalg.norm(t):.3f}m n={len(e)}')


def cmd_render(args):
    bl = blocks()[args.block]
    ex = json.load(open(WORK / 'extrinsics' / f'{args.block}.json'))
    M, t = np.array(ex['M']), np.array(ex['t'])
    task = args.task or ex['task']
    ep_key = args.ep or ex['ep']
    K, dist = load_calib_full(bl['dev'])
    mp4, pq = stage_episode(task, ep_key)
    df = pd.read_parquet(pq)
    if args.frames:
        frames = [int(x) for x in args.frames.split(',')]
    else:
        picks = pick_frames(df, M, t, K, dist, n=3)
        frames = [fi for fi, _, _ in picks]
    cap = cv2.VideoCapture(str(mp4))
    outdir = WORK / 'renders' / args.block
    outdir.mkdir(parents=True, exist_ok=True)
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            continue
        row = df.iloc[fi]
        for side, color in (('left', (0, 0, 255)), ('right', (255, 128, 0))):
            pts = tracker_pts(row, side)
            pxs, okz = project_px(pts @ M.T + t, K, dist)
            for a, b in EDGES:
                if okz[a] and okz[b]:
                    cv2.line(img, tuple(pxs[a].astype(int)), tuple(pxs[b].astype(int)), color, 2)
            for j in range(25):
                if okz[j]:
                    cv2.circle(img, tuple(pxs[j].astype(int)), 8 if j == 0 else 4, color, -1)
        p = outdir / f'{task}_{ep_key}_f{fi}.jpg'
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep')
    p.add_argument('--block', required=True)
    p.add_argument('--ep', default=None)
    p.add_argument('--n-frames', type=int, default=3, dest='n_frames')
    p.add_argument('--crop-half', type=int, default=220, dest='crop_half')
    p.add_argument('--zoom', type=float, default=2.0)
    p = sub.add_parser('solve')
    p.add_argument('--block', required=True)
    p = sub.add_parser('render')
    p.add_argument('--block', required=True)
    p.add_argument('--task', default=None)
    p.add_argument('--ep', default=None)
    p.add_argument('--frames', default=None)
    args = ap.parse_args()
    {'prep': cmd_prep, 'solve': cmd_solve, 'render': cmd_render}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
