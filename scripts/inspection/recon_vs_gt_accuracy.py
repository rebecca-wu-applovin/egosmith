#!/usr/bin/env python
"""Batch reconstruction-vs-GT accuracy for the EgoSmith recon fleet (Job 2).

Compares each clip's *reconstructed* output (from the no-GT fleet run) against GT and
writes a per-clip CSV + per-dataset summary. The math is IDENTICAL to the validated
walkthrough notebook (camera ATE via Umeyama on camera centres; camera-frame hand
MPJPE + wrist-relative articulation), reusing the real pipeline helpers.

Two dataset branches:
  taco / oakink_grasp / hot3d : GT = world_space_res.pth + SLAM/hawor_slam_w_scale_*.npz
                                (full 21-joint MPJPE + camera ATE).
  egodex                      : GT = native 116-d lowdim from the WDS frame tar
                                (wrist + 5 fingertips, camera frame via the lowdim
                                 World2Cam extrinsic; NO camera ATE).

Usage (egosmith env):
  # MANO datasets — recon_root/<clip>/{world_space_res.pth,SLAM/*.npz}, gt_root likewise
  python scripts/inspection/recon_vs_gt_accuracy.py --dataset taco \
      --recon_root /root/egosmith_recon/taco/recon/outputs \
      --gt_root    gs-mirror-or-local/taco/outputs \
      --out_csv    /root/egosmith_recon/accuracy/taco.csv
  # EgoDex — gt is the WDS frame tar dir (per-clip <clip>.tar with .lowdim.npy)
  python scripts/inspection/recon_vs_gt_accuracy.py --dataset egodex \
      --recon_root /root/egosmith_recon/egodex/recon/outputs \
      --egodex_frames_root /root/egodex/frames \
      --out_csv /root/egosmith_recon/accuracy/egodex.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import statistics as st
import sys
import tarfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402
from lib.pipeline.io.result_io import load_pose_arrays  # noqa: E402
from lib.pipeline.exporters.mano_features import (  # noqa: E402
    _compute_hand_joints,
    build_mano_models,
    FINGERTIP_INDICES,
)
from lib.pipeline.slam.slam_cam import load_slam_cam  # noqa: E402

# EgoDex 116-d lowdim slices (see generate_egodex_wds / quality/constants).
LD_LWRIST = slice(0, 3)
LD_RWRIST = slice(3, 6)
LD_LTIPS = slice(18, 33)
LD_RTIPS = slice(33, 48)
LD_EXTR = slice(96, 112)   # World2Cam 4x4 (row-major)
LD_INTR = slice(112, 116)  # fx,fy,cx,cy


# ---------- shared math (verbatim from the walkthrough) ----------
def umeyama(s, d):
    ms, md_ = s.mean(0), d.mean(0)
    U, D, Vt = np.linalg.svd((d - md_).T @ (s - ms) / len(s))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    sc = np.trace(np.diag(D)) / (((s - ms) ** 2).sum() / len(s))
    return sc, R, md_ - sc * R @ ms


def _cam(seq: str):
    f = sorted(glob.glob(f"{seq}/SLAM/hawor_slam_w_scale_*.npz"))
    if not f:
        return None  # slam did not complete -> no camera
    r, t, rc, tc = load_slam_cam(f[-1])
    return np.asarray(r), np.asarray(t), np.asarray(tc)  # R_w2c, t_w2c, t_c2w


def _world_joints(seq: str, mano_l, mano_r, dev):
    tr, ro, hp, be, _ = load_pose_arrays(seq)
    tr, ro, hp, be = [torch.from_numpy(np.asarray(x)) for x in (tr, ro, hp, be)]
    L = _compute_hand_joints(mano_l, tr, ro, hp, be, 0, dev).cpu().numpy()
    R = _compute_hand_joints(mano_r, tr, ro, hp, be, 1, dev).cpu().numpy()
    return L, R  # each (T,21,3) world


# ---------- MANO-dataset branch (taco/oakink/hot3d) ----------
def eval_mano_clip(recon_seq, gt_seq, mano_l, mano_r, dev):
    Lr, Rr = _world_joints(recon_seq, mano_l, mano_r, dev)
    Lg, Rg = _world_joints(gt_seq, mano_l, mano_r, dev)
    cr = _cam(recon_seq)
    cg = _cam(gt_seq)
    # Classify reconstruction validity BEFORE metrics, so a failed recon is counted as a
    # failure (usable-rate) rather than crashing the batch or hiding behind an "error".
    if cr is None:
        return {"status": "no_recon_camera"}       # slam never wrote hawor_slam_w_scale
    if cg is None:
        return {"status": "no_gt_camera"}
    Rwc_r, twc_r, Cr = cr
    Rwc_g, twc_g, Cg = cg
    if not (np.isfinite(Lr).all() and np.isfinite(Rr).all()
            and np.isfinite(Cr).all() and np.isfinite(Rwc_r).all()):
        return {"status": "degenerate_recon"}       # DPVO diverged -> NaN/inf (see stage_validators)
    T = min(len(Lr), len(Lg), len(Cr), len(Cg))
    if T < 2:
        return {"status": "too_few_frames"}

    # Best-fit similarity (scale+rot+trans) mapping recon-world -> gt-world, solved
    # from the CAMERA centres. This is the 7-DoF monocular gauge that is fundamentally
    # unrecoverable from a single view; removing it is required for a fair global
    # comparison (standard PA-MPJPE practice).
    sc, Rm, tm = umeyama(Cr[:T], Cg[:T])
    ate = np.sqrt(((sc * (Rm @ Cr[:T].T).T + tm - Cg[:T]) ** 2).sum(1)).mean()

    def tocam(J, Rwc, twc):
        return np.einsum("tij,tnj->tni", Rwc[:T], J[:T]) + twc[:T, None, :]

    # PA-global gauge: one similarity fit on the HANDS (both hands, all frames jointly).
    # The hands sweep a real volume over the clip, so this is well-conditioned — unlike
    # the camera centres, which barely move in an egocentric clip (that camera-based
    # Sim3 is degenerate and blows the hand error up, so it is NOT used for the hands).
    src = np.concatenate([Lr[:T].reshape(-1, 3), Rr[:T].reshape(-1, 3)], 0)
    dst = np.concatenate([Lg[:T].reshape(-1, 3), Rg[:T].reshape(-1, 3)], 0)
    try:
        ps, pR, pt = umeyama(src, dst)
    except np.linalg.LinAlgError:
        return {"status": "degenerate_recon"}  # SVD non-convergence = degenerate geometry

    def pa(J):
        return (ps * (pR @ J[:T].reshape(-1, 3).T).T + pt).reshape(T, -1, 3)

    out = {"status": "ok", "frames": T, "camera_ate_mm": float(ate * 1000), "pa_scale": float(ps)}
    for nm, Jr, Jg in [("left", Lr, Lg), ("right", Rr, Rg)]:
        # (1) PA-MPJPE — honest global error after removing the unrecoverable world
        #     similarity (gauge+scale), aligned on the hands. Residual = true recon error.
        out[f"{nm}_pa_mpjpe_mm"] = float(np.sqrt(((pa(Jr) - Jg[:T]) ** 2).sum(-1)).mean() * 1000)
        # (2) Raw camera-frame MPJPE — hand position relative to the camera; gauge-free
        #     (rot+trans cancel physically) but still carries the monocular SCALE error.
        rc, gc = tocam(Jr, Rwc_r, twc_r), tocam(Jg, Rwc_g, twc_g)
        out[f"{nm}_camframe_mpjpe_mm"] = float(np.sqrt(((rc - gc) ** 2).sum(-1)).mean() * 1000)
        # (3) Articulation — wrist-relative shape, gauge/scale-free by construction.
        out[f"{nm}_artic_mm"] = float(
            np.sqrt((((rc - rc[:, :1]) - (gc - gc[:, :1])) ** 2).sum(-1)).mean() * 1000
        )
    return out


# ---------- EgoDex branch (recon MANO vs native lowdim, camera frame, 6-pt) ----------
def _read_egodex_lowdim(tar_path: str, clip_id: str):
    """Return (T,116) lowdim stacked over frames in the clip tar."""
    rows = []
    with tarfile.open(tar_path, "r") as tf:
        names = sorted(n for n in tf.getnames() if n.endswith(".lowdim.npy"))
        for n in names:
            rows.append(np.load(io.BytesIO(tf.extractfile(n).read())))
    if not rows:
        raise FileNotFoundError(f"no .lowdim.npy in {tar_path}")
    return np.stack(rows).astype(np.float64)


def _tocam_points(P_world, extr4x4):
    """P_world (...,3) -> camera frame via World2Cam 4x4."""
    P = np.concatenate([P_world, np.ones(P_world.shape[:-1] + (1,))], axis=-1)
    return (P @ extr4x4.T)[..., :3]


def eval_egodex_clip(recon_seq, tar_path, clip_id, mano_l, mano_r, dev):
    # Same PA method as the MANO branch, but GT is the native lowdim wrist+5 fingertips
    # (ARKit world) and recon is reduced to the matching 6 MANO joints. Both are in world
    # frames that differ by a Sim3 (monocular recon vs ARKit), so joint Procrustes removes
    # the gauge — no per-frame extrinsic needed.
    Lr, Rr = _world_joints(recon_seq, mano_l, mano_r, dev)  # (T,21,3) recon world
    if not (np.isfinite(Lr).all() and np.isfinite(Rr).all()):
        return {"status": "degenerate_recon"}
    ld = _read_egodex_lowdim(tar_path, clip_id)             # (T,116) native GT
    T = min(len(Lr), len(ld))
    if T < 2:
        return {"status": "too_few_frames"}
    idx = [0] + list(FINGERTIP_INDICES)                    # wrist + 5 fingertips
    Lr6, Rr6 = Lr[:T, idx, :], Rr[:T, idx, :]              # (T,6,3) recon world
    Lg = np.concatenate([ld[:T, LD_LWRIST][:, None, :], ld[:T, LD_LTIPS].reshape(T, 5, 3)], axis=1)
    Rg = np.concatenate([ld[:T, LD_RWRIST][:, None, :], ld[:T, LD_RTIPS].reshape(T, 5, 3)], axis=1)
    if not (np.isfinite(Lg).all() and np.isfinite(Rg).all()):
        return {"status": "no_gt_camera"}                  # bad GT lowdim

    src = np.concatenate([Lr6.reshape(-1, 3), Rr6.reshape(-1, 3)], 0)
    dst = np.concatenate([Lg.reshape(-1, 3), Rg.reshape(-1, 3)], 0)
    try:
        ps, pR, pt = umeyama(src, dst)
    except np.linalg.LinAlgError:
        return {"status": "degenerate_recon"}

    def pa(J):
        return (ps * (pR @ J.reshape(-1, 3).T).T + pt).reshape(J.shape)

    out = {"status": "ok", "frames": T, "pa_scale": float(ps)}
    for nm, Jr6, Jg in [("left", Lr6, Lg), ("right", Rr6, Rg)]:
        out[f"{nm}_pa_mpjpe_mm"] = float(np.sqrt(((pa(Jr6) - Jg) ** 2).sum(-1)).mean() * 1000)
        rc_rel, gc_rel = pa(Jr6) - pa(Jr6)[:, :1], Jg - Jg[:, :1]
        out[f"{nm}_artic_mm"] = float(np.sqrt(((rc_rel - gc_rel) ** 2).sum(-1)).mean() * 1000)
    return out


def _summ(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {}
    vals.sort()
    return {
        "n": len(vals),
        "median": round(st.median(vals), 1),
        "p90": round(vals[min(len(vals) - 1, int(0.9 * len(vals)))], 1),
        "mean": round(st.mean(vals), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["taco", "oakink_grasp", "hot3d", "egodex"])
    ap.add_argument("--recon_root", required=True, help="dir of <clip>/ recon seq_folders")
    ap.add_argument("--gt_root", help="MANO datasets: dir of <clip>/ GT seq_folders")
    ap.add_argument("--egodex_frames_root", help="egodex: dir of <clip>.tar WDS frame tars")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    mano_r, mano_l = build_mano_models(dev)

    clips = sorted(p.name for p in Path(args.recon_root).iterdir() if p.is_dir())
    rows = []
    for clip in clips:
        recon_seq = str(Path(args.recon_root) / clip)
        try:
            if args.dataset == "egodex":
                tar = str(Path(args.egodex_frames_root) / f"{clip}.tar")
                m = eval_egodex_clip(recon_seq, tar, clip, mano_l, mano_r, dev)
            else:
                m = eval_mano_clip(recon_seq, str(Path(args.gt_root) / clip), mano_l, mano_r, dev)
        except Exception as e:  # noqa: BLE001 — unexpected: a real harness/GT problem, not attrition
            m = {"status": "error", "detail": str(e)[:160]}
        m["clip_id"] = clip
        rows.append(m)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- three buckets: ok (metrics) | failed recon (degenerate/no-camera) | error (harness) ----
    by = {}
    for r in rows:
        by.setdefault(r.get("status", "error"), []).append(r["clip_id"])
    ok = [r for r in rows if r.get("status") == "ok"]
    n_ok = len(ok)
    n_failrecon = sum(len(by.get(s, [])) for s in ("degenerate_recon", "no_recon_camera", "too_few_frames"))
    n_err = len(by.get("error", [])) + len(by.get("no_gt_camera", []))
    usable = 100.0 * n_ok / max(1, n_ok + n_failrecon)

    # record the failed clips (so they can be inspected / re-run)
    fail_path = out.with_suffix(".failed.txt")
    with fail_path.open("w") as f:
        for r in rows:
            if r.get("status") != "ok":
                f.write(f"{r['clip_id']}\t{r.get('status','error')}\t{r.get('detail','')}\n")

    print(f"\n[{args.dataset}] {len(rows)} clips -> {out}")
    print(f"  usable-rate = {usable:.1f}%  (ok={n_ok}  failed_recon={n_failrecon}  harness_err={n_err})")
    for s, ids in sorted(by.items()):
        if s != "ok":
            print(f"    {s}: {len(ids)}  (recorded in {fail_path.name})")
    if args.dataset != "egodex":
        print("  camera_ate_mm            :", _summ([r.get("camera_ate_mm") for r in ok]))
    for nm in ("left", "right"):
        print(f"  {nm}_pa_mpjpe_mm (PA, gauge-free):", _summ([r.get(f"{nm}_pa_mpjpe_mm") for r in ok]))
    print("  pa_scale                 :", _summ([r.get("pa_scale") for r in ok]))
    for nm in ("left", "right"):
        print(f"  {nm}_artic_mm             :", _summ([r.get(f"{nm}_artic_mm") for r in ok]))


if __name__ == "__main__":
    main()
