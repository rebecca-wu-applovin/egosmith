#!/usr/bin/env python
"""WIYH per-device-day extrinsic auto-refinement against shipped hand masks.

Given a same-device INIT extrinsic (solved by the manual vision-anchor pass on
one day), refine (R, t) per hand for another day by minimizing the hand-mask
distance-transform at the 25 projected joints over many wrist-gate-passing
frames, with a soft prior pulling toward the init. Local refinement only — the
pilot showed mask-only FROM-SCRATCH fitting is unreliable, but day-to-day drift
is 50-150 px which is well inside the refinement basin.

Validation mode: --compare against a manually-solved extrinsic reports the
joint-projection deviation between refined and reference across frames.

Usage:
  python scripts/build/wiyh_mask_refine.py --dir <staged sample> \
      --init extrinsic.json [--out refined.json] [--compare ref.json] \
      [--sides left,right] [--max_frames 120] [--prior_px 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiyh_gate_census import CHEST, SampleStreams, gate_dists, match_masks  # noqa: E402

MASK_DT_SCALE = 2  # distance transform at 1/2 res (px error ~2 << gate scale)


def load_sample(root: Path):
    ss = SampleStreams((root / "dataset.hdf5").read_bytes())
    masks = {}
    for p in sorted((root / "hand_masks" / CHEST).glob("*.png")):
        mm = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        small = mm[::4, ::4]
        ys, xs = np.nonzero(small)
        masks[p.name] = np.stack([xs, ys], 1).astype(np.int16)
    return ss, match_masks(ss, masks)


def mask_dts(root: Path, ss, frames):
    """Distance transforms (at 1/MASK_DT_SCALE res) for selected frames."""
    files = {p.name: p for p in (root / "hand_masks" / CHEST).glob("*.png")}
    names = sorted(files)
    out = {}
    for i in frames:
        n = ss.frame_names[i].replace(".jpg", ".png")
        p = files.get(n)
        if p is None:  # sorted-order fallback
            if i < len(names):
                p = files[names[i]]
            else:
                continue
        mm = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mm is None or not (mm > 0).any():
            continue
        small = (mm[::MASK_DT_SCALE, ::MASK_DT_SCALE] == 0).astype(np.uint8)
        out[i] = cv2.distanceTransform(small, cv2.DIST_L2, 3) * MASK_DT_SCALE
    return out


def project_joints(ss, i, side, R, t):
    v = ss.eef[side][i]
    Re, te = Rt.from_quat(v[3:]).as_matrix(), v[:3]
    pc = (Re @ (R @ ss.pts[side][i].T + t.reshape(3, 1))).T + te
    cam = (ss.R_ext.T @ (pc - ss.t_ext).T).T
    out = np.full((25, 2), np.nan)
    front = cam[:, 2] > 1e-6
    if front.any():
        uv, _ = cv2.fisheye.projectPoints(
            np.ascontiguousarray(cam[front]).reshape(-1, 1, 3),
            np.zeros(3), np.zeros(3), ss.K, ss.D.reshape(4, 1))
        out[front] = uv.reshape(-1, 2)
    return out


def refine_side(ss, dists, dts, side, R0, t0, prior_px=30.0, max_frames=120):
    d = dists[side]
    ok = [i for i in range(ss.n) if 0 <= d[i] < 30 and i in dts]
    if len(ok) < 15:
        return None, {"error": f"only {len(ok)} gated frames"}
    if len(ok) > max_frames:
        ok = [ok[k] for k in np.linspace(0, len(ok) - 1, max_frames, dtype=int)]

    def resid(x):
        R = Rt.from_rotvec(x[:3]).as_matrix() @ R0
        t = t0 + x[3:]
        r = []
        for i in ok:
            uv = project_joints(ss, i, side, R, t)
            dt = dts[i]
            H, W = dt.shape
            for j in range(25):
                u, v = uv[j]
                if not np.isfinite(u):
                    r.append(120.0)
                    continue
                xi = int(min(max(u / MASK_DT_SCALE, 0), W - 1))
                yi = int(min(max(v / MASK_DT_SCALE, 0), H - 1))
                r.append(float(dt[yi, xi]))
        # soft prior: rotation (rad -> ~px via 500px focal * angle) + translation (m -> px)
        r.append(prior_px * float(np.linalg.norm(x[:3])) * 500.0 / 30.0 * 0.1)
        r.append(prior_px * float(np.linalg.norm(x[3:])) * 1500.0 / 30.0 * 0.1)
        return np.array(r)

    sol = least_squares(resid, np.zeros(6), method="trf", loss="soft_l1",
                        f_scale=15.0, max_nfev=200)
    R = Rt.from_rotvec(sol.x[:3]).as_matrix() @ R0
    t = t0 + sol.x[3:]
    res = resid(sol.x)[:-2]
    per_frame = np.array(res).reshape(len(ok), 25)
    info = {"n_frames": len(ok),
            "init_med_px_to_mask": float(np.median(resid(np.zeros(6))[:-2])),
            "refined_med_px_to_mask": float(np.median(per_frame)),
            "rot_delta_deg": float(np.degrees(np.linalg.norm(sol.x[:3]))),
            "t_delta_cm": float(np.linalg.norm(sol.x[3:]) * 100)}
    return (R, t), info


def compare(ss, dists, side, Ra, ta, Rb, tb):
    """Median px deviation between two extrinsics' joint projections on gated frames."""
    d = dists[side]
    ok = [i for i in range(ss.n) if 0 <= d[i] < 30]
    devs, tip_devs = [], []
    for i in ok[:150]:
        ua = project_joints(ss, i, side, Ra, ta)
        ub = project_joints(ss, i, side, Rb, tb)
        m = np.isfinite(ua).all(1) & np.isfinite(ub).all(1)
        if m.any():
            dv = np.linalg.norm(ua[m] - ub[m], axis=1)
            devs.append(np.median(dv))
            tm = [k for k, j in enumerate(np.where(m)[0]) if j in (3, 8, 13, 18, 23)]
            if tm:
                tip_devs.append(np.median(dv[tm]))
    return {"frames": len(devs),
            "med_px": float(np.median(devs)) if devs else None,
            "p90_px": float(np.percentile(devs, 90)) if devs else None,
            "tips_med_px": float(np.median(tip_devs)) if tip_devs else None}


TIPS = [3, 8, 13, 18, 23]


def detect_teal(img_bgr):
    """Mint fingertip-pad blob centroids (the HS-2 glove's distinctive markers)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 70) & (h <= 105) & (s >= 40) & (s <= 180) & (v >= 70)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    return [cents[k] for k in range(1, n) if 150 < stats[k, 4] < 25000]


def pad_refine_side(root, ss, dists, side, R0, t0, max_frames=150,
                    gates=(150.0, 80.0, 50.0)):
    """ICP-style: associate teal-pad detections to projected fingertips, re-solve."""
    d = dists[side]
    ok = [i for i in range(ss.n) if 0 <= d[i] < 30]
    if len(ok) < 10:
        return None, {"error": f"only {len(ok)} gated frames"}
    if len(ok) > max_frames:
        ok = [ok[k] for k in np.linspace(0, len(ok) - 1, max_frames, dtype=int)]
    fdir = root / "camera" / CHEST
    dets = {}
    for i in ok:
        img = cv2.imread(str(fdir / ss.frame_names[i]))
        if img is None:
            continue
        c = detect_teal(img)
        if c:
            dets[i] = np.array(c)
    R, t = R0.copy(), np.array(t0, np.float64)
    info = {"n_gated_frames": len(ok), "n_frames_with_dets": len(dets)}
    # wrist anchor: joint 24 sits |t_gl| (~1-3 cm, <=45 px) from the raw eef point,
    # which the census verified locks onto the hand — kills flipped ICP basins.
    from scipy.spatial.transform import Rotation as _Rt
    eef_uv = {}
    for i in ok:
        v = ss.eef[side][i]
        cam = ss.R_ext.T @ (v[:3] - ss.t_ext)
        if cam[2] > 1e-6:
            uv, _ = cv2.fisheye.projectPoints(cam.reshape(1, 1, 3), np.zeros(3),
                                              np.zeros(3), ss.K, ss.D.reshape(4, 1))
            p = uv.reshape(2)
            if np.isfinite(p).all():
                eef_uv[i] = p
    for rnd, gate in enumerate(gates):
        obs = []
        for i, cs in dets.items():
            uv = project_joints(ss, i, side, R, t)
            for j in TIPS:
                if not np.isfinite(uv[j]).all():
                    continue
                dd = np.linalg.norm(cs - uv[j], axis=1)
                k = int(dd.argmin())
                if dd[k] < gate:
                    obs.append((i, j, float(cs[k][0]), float(cs[k][1])))
        if len(obs) < 12:
            info["error"] = f"round{rnd}: only {len(obs)} associations"
            return None, info

        def resid(x):
            Rx = Rt.from_rotvec(x[:3]).as_matrix() @ R
            tx = t + x[3:]
            r = []
            for i, j, u, v in obs:
                uv = project_joints(ss, i, side, Rx, tx)[j]
                r += list(uv - [u, v]) if np.isfinite(uv).all() else [200.0, 200.0]
            # wrist-to-eef anchors (deadband 45 px: only penalize excess)
            for i, p in eef_uv.items():
                uv = project_joints(ss, i, side, Rx, tx)[24]
                if np.isfinite(uv).all():
                    excess = max(0.0, float(np.linalg.norm(uv - p)) - 45.0)
                    r.append(excess)
                else:
                    r.append(200.0)
            r.append(np.linalg.norm(x[:3]) * 300.0)
            r.append(np.linalg.norm(x[3:]) * 800.0)
            return np.array(r)

        sol = least_squares(resid, np.zeros(6), method="trf", loss="soft_l1",
                            f_scale=15.0, max_nfev=300)
        R = Rt.from_rotvec(sol.x[:3]).as_matrix() @ R
        t = t + sol.x[3:]
        errs = []
        for i, j, u, v in obs:
            uv = project_joints(ss, i, side, R, t)[j]
            if np.isfinite(uv).all():
                errs.append(float(np.linalg.norm(uv - [u, v])))
        info[f"round{rnd}"] = {"gate": gate, "n_obs": len(obs),
                               "med_px": float(np.median(errs)),
                               "n_frames": len({o[0] for o in obs})}
    last = info[f"round{len(gates)-1}"]
    info["final_med_px"] = last["med_px"]
    info["final_n_obs"] = last["n_obs"]
    info["final_n_frames"] = last["n_frames"]
    return (R, t), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--init", required=True, help="init extrinsic json ({side: {R,t}})")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", default=None, help="reference extrinsic json to compare against")
    ap.add_argument("--sides", default="left,right")
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--prior_px", type=float, default=30.0)
    ap.add_argument("--mode", choices=["mask", "pads"], default="pads")
    a = ap.parse_args()

    root = Path(a.dir)
    ss, masks = load_sample(root)
    dists = gate_dists(ss, masks)
    init = json.loads(Path(a.init).read_text())
    ref = json.loads(Path(a.compare).read_text()) if a.compare else None
    out = {"sample_dir": str(root), "init": a.init}
    frames_all = sorted({i for side in ("left", "right") for i in range(ss.n)
                         if 0 <= dists[side][i] < 30})
    dts = mask_dts(root, ss, frames_all)
    for side in a.sides.split(","):
        if side not in init:
            continue
        R0, t0 = np.array(init[side]["R"]), np.array(init[side]["t"])
        if a.mode == "pads":
            sol, info = pad_refine_side(root, ss, dists, side, R0, t0,
                                        max_frames=a.max_frames)
        else:
            sol, info = refine_side(ss, dists, dts, side, R0, t0,
                                    prior_px=a.prior_px, max_frames=a.max_frames)
        print(f"[{side}] {info}")
        if sol is None:
            out[side] = {"error": info.get("error")}
            continue
        R, t = sol
        out[side] = {"R": R.tolist(), "t": t.tolist(), **info}
        if ref and side in ref:
            cmpres = compare(ss, dists, side, R, t,
                             np.array(ref[side]["R"]), np.array(ref[side]["t"]))
            cmp0 = compare(ss, dists, side, R0, t0,
                           np.array(ref[side]["R"]), np.array(ref[side]["t"]))
            out[side]["vs_reference"] = cmpres
            out[side]["init_vs_reference"] = cmp0
            print(f"[{side}] refined-vs-ref: {cmpres}")
            print(f"[{side}]    init-vs-ref: {cmp0}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(a.out)


if __name__ == "__main__":
    main()
