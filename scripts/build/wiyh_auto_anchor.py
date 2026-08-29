#!/usr/bin/env python
"""WIYH automatic per-session glove->eef extrinsic solver ("auto-anchor").

Replaces the manual vision-anchor grind: the HS-2 gloves carry mint/teal
fingertip pads that are machine-detectable; with the census wrist gate
guaranteeing eef lock-on, the extrinsic is solved from scratch per session:

  observations = teal-pad centroids on wrist-gated frames (mount markers
                 excluded by an eef-proximity veto)
  model        = 25-joint glove-local skeleton -> eef SE3 -> KB4 projection
  solve        = 24 proper-permutation restarts x (Hungarian tip assignment
                 <-> 6-DoF LM refine) with a wrist->eef deadband anchor
                 (|t_gl| <= ~3 cm means joint 24 projects within ~45 px of eef)
  gate         = fit_med < 30 px over >= 24 associations on >= 12 frames AND
                 split-half holdout med < 60 px

Per-SESSION solving also absorbs strap re-donning (observed to break rotation
transfer WITHIN a device-day on 27094 2025-10-25).

Validated against the manual anchor solves (see wiyh_anchor_run.py registry):
run with --dir <staged sample> [--compare ref.json].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation as Rt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiyh_gate_census import CHEST, SampleStreams, gate_dists, match_masks  # noqa: E402

TIPS = [3, 8, 13, 18, 23]
GATE_FIT_MED = 30.0
GATE_MIN_OBS = 24
GATE_MIN_FRAMES = 12
GATE_HOLDOUT_MED = 60.0


def proper_perms():
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats


def detect_teal(img_bgr, eef_px=None, veto_px=110.0, mask_coords=None, mask_gate_px=25.0):
    """Teal blob centroids, mount markers vetoed by eef proximity + size.

    mask_coords: hand-mask nonzero coords (Nx2 full-res px). When given, HSV
    thresholds run LOOSE (Supermarket pads are desaturated) and blobs must sit
    on the hand mask — spatial gating replaces color strictness."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if mask_coords is not None and len(mask_coords):
        m = ((h >= 65) & (h <= 110) & (s >= 15) & (s <= 200) & (v >= 35)).astype(np.uint8)
    else:
        m = ((h >= 70) & (h <= 105) & (s >= 40) & (s <= 180) & (v >= 70)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for k in range(1, n):
        if not (70 < stats[k, 4] < 12000):
            continue
        c = cents[k]
        if eef_px is not None and np.linalg.norm(c - eef_px) < veto_px:
            continue
        if mask_coords is not None and len(mask_coords):
            if np.linalg.norm(mask_coords - c, axis=1).min() > mask_gate_px:
                continue
        out.append((c, stats[k, 4]))
    out.sort(key=lambda d: -d[1])
    return [c for c, _ in out[:6]]


class SideSolver:
    """Vectorized projection of the 25-joint model for a fixed frame subset."""

    def __init__(self, ss: SampleStreams, side: str, frames: list[int]):
        self.ss, self.side, self.frames = ss, side, frames
        eef = ss.eef[side][frames]
        self.Re = Rt.from_quat(eef[:, 3:]).as_matrix()          # (F,3,3)
        self.te = eef[:, :3]
        self.loc = ss.pts[side][frames]                          # (F,25,3)
        self.fidx = {f: k for k, f in enumerate(frames)}

    def project(self, R, t):
        """(F,25,2) pixel projections under glove->eef (R,t)."""
        ss = self.ss
        pc = np.einsum("fij,fkj->fki", self.Re, self.loc @ R.T + t) + self.te[:, None]
        cam = np.einsum("ij,fkj->fki", ss.R_ext.T, pc - ss.t_ext)
        F = cam.shape[0]
        flat = cam.reshape(-1, 3)
        out = np.full((flat.shape[0], 2), np.nan)
        front = flat[:, 2] > 1e-6
        if front.any():
            uv, _ = cv2.fisheye.projectPoints(
                np.ascontiguousarray(flat[front]).reshape(-1, 1, 3),
                np.zeros(3), np.zeros(3), ss.K, ss.D.reshape(4, 1))
            out[front] = uv.reshape(-1, 2)
        return out.reshape(F, 25, 2)


def eef_px_map(ss, side, frames):
    out = {}
    for i in frames:
        v = ss.eef[side][i]
        cam = ss.R_ext.T @ (v[:3] - ss.t_ext)
        if cam[2] <= 1e-6:
            continue
        uv, _ = cv2.fisheye.projectPoints(cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                          ss.K, ss.D.reshape(4, 1))
        p = uv.reshape(2)
        if np.isfinite(p).all():
            out[i] = p
    return out


def solve_side(ss, side, dets, iters=3):
    """dets: {frame: [centroid,...]} on gated frames. Returns (R, t, report)."""
    frames = sorted(dets)
    if len(frames) < GATE_MIN_FRAMES:
        return None, {"error": f"frames_with_dets={len(frames)}"}
    sv = SideSolver(ss, side, frames)
    eefuv = eef_px_map(ss, side, frames)

    def assign(R, t, gate):
        uv = sv.project(R, t)
        obs = []  # (frame_k, joint, u, v)
        for f in frames:
            k = sv.fidx[f]
            cs = dets[f]
            cost = np.full((len(cs), 5), 1e5)
            for a, c in enumerate(cs):
                for b, j in enumerate(TIPS):
                    if np.isfinite(uv[k, j]).all():
                        cost[a, b] = np.linalg.norm(uv[k, j] - c)
            ra, cb = linear_sum_assignment(cost)
            for a, b in zip(ra, cb):
                if cost[a, b] < gate:
                    obs.append((k, TIPS[b], cs[a][0], cs[a][1]))
        return obs

    def refine(R, t, obs):
        if len(obs) < 8:
            return R, t
        ks = np.array([o[0] for o in obs])
        js = np.array([o[1] for o in obs])
        tgt = np.array([[o[2], o[3]] for o in obs])
        wk = np.array([sv.fidx[f] for f in frames if f in eefuv])
        wt = np.array([eefuv[f] for f in frames if f in eefuv])

        def resid(x):
            Rx = Rt.from_rotvec(x[:3]).as_matrix() @ R
            tx = t + x[3:]
            uv = sv.project(Rx, tx)
            r = uv[ks, js] - tgt
            r = np.where(np.isfinite(r), r, 300.0).ravel()
            w = uv[wk, 24] - wt
            wd = np.linalg.norm(np.where(np.isfinite(w), w, 300.0), axis=1)
            excess = np.maximum(0.0, wd - 45.0)
            return np.concatenate([r, excess, [np.linalg.norm(x[3:]) * 500.0]])

        sol = least_squares(resid, np.zeros(6), method="trf", loss="soft_l1",
                            f_scale=20.0, max_nfev=120)
        return Rt.from_rotvec(sol.x[:3]).as_matrix() @ R, t + sol.x[3:]

    best = None
    for R0 in proper_perms():
        R, t = R0, np.zeros(3)
        for it, gate in zip(range(iters), (250.0, 120.0, 70.0)):
            obs = assign(R, t, gate)
            if len(obs) < 8:
                break
            R, t = refine(R, t, obs)
        obs = assign(R, t, 70.0)
        if len(obs) < 8:
            continue
        uv = sv.project(R, t)
        errs = [np.linalg.norm(uv[k, j] - [u, v]) for k, j, u, v in obs]
        med = float(np.median(errs))
        score = med + max(0, GATE_MIN_OBS - len(obs)) * 5.0
        if best is None or score < best[0]:
            best = (score, med, len(obs), R, t, obs)
    if best is None:
        return None, {"error": "no perm converged"}
    _, med, n_obs, R, t, obs = best
    n_frames = len({k for k, *_ in obs})

    # split-half holdout: refit on even frames, evaluate on odd-frame associations
    fr_used = sorted({frames[k] for k, *_ in obs})
    even = {f for i, f in enumerate(fr_used) if i % 2 == 0}
    obs_even = [o for o in obs if frames[o[0]] in even]
    obs_odd = [o for o in obs if frames[o[0]] not in even]
    ho_med = None
    if len(obs_even) >= 8 and len(obs_odd) >= 6:
        Rh, th = refine(R, t, obs_even)
        uvh = sv.project(Rh, th)
        ho = [np.linalg.norm(uvh[k, j] - [u, v]) for k, j, u, v in obs_odd]
        ho_med = float(np.median(ho))
    ok = (med < GATE_FIT_MED and n_obs >= GATE_MIN_OBS and n_frames >= GATE_MIN_FRAMES
          and (ho_med is None or ho_med < GATE_HOLDOUT_MED))
    rep = {"fit_med_px": round(med, 1), "n_obs": n_obs, "n_frames": n_frames,
           "holdout_med_px": round(ho_med, 1) if ho_med is not None else None,
           "pass": bool(ok)}
    return (R, t), rep


def auto_anchor_sample(root: Path, jpg_reader=None, max_frames=70):
    """Solve both hands for one staged sample dir. Returns {side: {R,t,report}}."""
    ss = SampleStreams((root / "dataset.hdf5").read_bytes())
    masks = {}
    for p in sorted((root / "hand_masks" / CHEST).glob("*.png")):
        mm = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        small = mm[::4, ::4]
        ys, xs = np.nonzero(small)
        masks[p.name] = np.stack([xs, ys], 1).astype(np.int16)
    masks = match_masks(ss, masks)
    dists = gate_dists(ss, masks)
    out = {}
    img_cache = {}

    def read_img(i):
        if i not in img_cache:
            img_cache[i] = cv2.imread(str(root / "camera" / CHEST / ss.frame_names[i]))
        return img_cache[i]

    for side in ("left", "right"):
        d = dists[side]
        ok = [i for i in range(ss.n) if 0 <= d[i] < 30]
        if len(ok) > max_frames:
            ok = [ok[k] for k in np.linspace(0, len(ok) - 1, max_frames, dtype=int)]
        eefuv = eef_px_map(ss, side, ok)
        dets = {}
        for i in ok:
            img = read_img(i)
            if img is None:
                continue
            nm = ss.frame_names[i].replace(".jpg", ".png")
            mc = masks.get(nm)
            mc4 = mc.astype(np.float32) * 4 if mc is not None and len(mc) else None
            cs = detect_teal(img, eef_px=eefuv.get(i), mask_coords=mc4)
            if cs:
                dets[i] = cs
        sol, rep = solve_side(ss, side, dets)
        rep["gated_frames"] = len(ok)
        if sol is not None:
            R, t = sol
            out[side] = {"R": R.tolist(), "t": t.tolist(), **rep}
        else:
            out[side] = rep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None)
    a = ap.parse_args()
    root = Path(a.dir)
    res = auto_anchor_sample(root)
    for side, r in res.items():
        print(f"[{side}] " + json.dumps({k: v for k, v in r.items() if k not in ("R", "t")}))
    if a.compare:
        ref = json.loads(Path(a.compare).read_text())
        ss = SampleStreams((root / "dataset.hdf5").read_bytes())
        for side in ("left", "right"):
            if side not in ref or "R" not in res.get(side, {}):
                continue
            fr = list(range(0, ss.n, 7))
            sv = SideSolver(ss, side, fr)
            ua = sv.project(np.array(res[side]["R"]), np.array(res[side]["t"]))
            ub = sv.project(np.array(ref[side]["R"]), np.array(ref[side]["t"]))
            m = np.isfinite(ua).all(2) & np.isfinite(ub).all(2)
            dv = np.linalg.norm(ua - ub, axis=2)
            med = float(np.median(dv[m]))
            tipm = m[:, TIPS]
            tips_med = float(np.median(dv[:, TIPS][tipm]))
            print(f"[{side}] vs-reference: med={med:.1f}px tips_med={tips_med:.1f}px")
    if a.out:
        Path(a.out).write_text(json.dumps({"sample_dir": str(root), **res}, indent=1))
        print(a.out)


if __name__ == "__main__":
    main()
