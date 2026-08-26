#!/usr/bin/env python
"""Per-clip card renderers for the filtered-data viewer.

Three card kinds:
  - recon overlay: MANO-forward world joints (world_space_res.pth / result.npz) projected
    through the recon SLAM camera onto the sub-clip frames (taco_overlay_sheets math).
  - native overlay: wrist + 5 fingertips per hand straight from per-frame .lowdim.npy
    (EgoDex-style WDS tars; no MANO, no seq folder).
  - plain video: frames-tar or source-mp4 interval transcode (robot / Stage-1-only cards).

All mp4s: libx264, yuv420p, crf 26, +faststart (GCS range-request streaming), true fps,
upscaled to >= MIN_W wide with the intrinsic scaled identically.
"""
import io
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import cv2

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO / "src"), str(_REPO), str(_REPO / "scripts" / "inspection")):
    if p not in sys.path:
        sys.path.insert(0, p)

import imageio_ffmpeg  # noqa: E402

MIN_W = 512
L_C = (66, 133, 244)   # RGB left=blue
R_C = (219, 68, 55)    # RGB right=red
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
# 116-d lowdim slices (lowdim_assembly.py layout)
LD_LWRIST, LD_RWRIST = slice(0, 3), slice(3, 6)
LD_LTIPS, LD_RTIPS = slice(18, 33), slice(33, 48)
LD_EXTR, LD_INTR = slice(96, 112), slice(112, 116)


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def load_tar_frames(tar_path, max_frames=0):
    """Sorted .image.jpg members -> list of RGB arrays (+ optional matching lowdim)."""
    frames, lowdims = [], {}
    with tarfile.open(tar_path) as tf:
        names = sorted(m.name for m in tf.getmembers())
        jpgs = [n for n in names if n.endswith(".image.jpg") or n.endswith(".jpg")]
        if max_frames:
            jpgs = jpgs[:max_frames]
        want_ld = {n[: -len(".image.jpg")] + ".lowdim.npy" for n in jpgs if n.endswith(".image.jpg")}
        for n in jpgs:
            buf = tf.extractfile(n).read()
            im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        for n in names:
            if n in want_ld:
                lowdims[n] = np.load(io.BytesIO(tf.extractfile(n).read()))
        ld = [lowdims.get(j[: -len(".image.jpg")] + ".lowdim.npy") for j in jpgs] if lowdims else None
    return frames, ld


def draw_skel(im, uvz, col):
    """White-outlined 21-joint skeleton; returns joints drawn."""
    h, w = im.shape[:2]
    pts = {j: (int(u), int(v)) for j, (u, v, z) in enumerate(uvz)
           if z > 1e-3 and np.isfinite(u) and -60 <= u < w + 60 and -60 <= v < h + 60}
    for a, b in EDGES:
        if a in pts and b in pts:
            cv2.line(im, pts[a], pts[b], (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(im, pts[a], pts[b], col, 2, cv2.LINE_AA)
    for j, p in pts.items():
        r = 5 if j == 0 else 3
        cv2.circle(im, p, r + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(im, p, r, col, -1, cv2.LINE_AA)
    return len(pts)


def draw_hand6(im, uvz, col):
    """Wrist + 5 fingertips outline (native lowdim cards)."""
    h, w = im.shape[:2]
    pts = {j: (int(u), int(v)) for j, (u, v, z) in enumerate(uvz)
           if z > 1e-3 and np.isfinite(u) and -30 <= u < w + 30 and -30 <= v < h + 30}
    for t in range(1, 6):
        if 0 in pts and t in pts:
            cv2.line(im, pts[0], pts[t], (255, 255, 255), 4, cv2.LINE_AA)
            cv2.line(im, pts[0], pts[t], col, 2, cv2.LINE_AA)
    for j, p in pts.items():
        r = 7 if j == 0 else 5
        cv2.circle(im, p, r + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(im, p, r, col, -1, cv2.LINE_AA)
    return len(pts)


def _writer(out_path, w, h, fps):
    wr = imageio_ffmpeg.write_frames(
        str(out_path), (w, h), fps=fps, codec="libx264",
        output_params=["-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart"])
    wr.send(None)
    return wr


def _upscale(w, h):
    s = max(1, math.ceil(MIN_W / w))
    return s, (w * s) // 2 * 2, (h * s) // 2 * 2


def render_recon_overlay(tar_path, seq_folder, out_mp4, fps=15.0):
    """MANO recon overlay card. Falls back to RGB-only with a diagnosis caption."""
    from taco_overlay_sheets import compute_world_joints, project, load_camera
    frames, _ = load_tar_frames(tar_path)
    if not frames:
        raise RuntimeError(f"no frames in {tar_path}")
    status, note, L = "overlay", None, None
    absent = {"L": False, "R": False}
    try:
        L, R = compute_world_joints(Path(seq_folder), _device())
        L, R = np.asarray(L), np.asarray(R)
        extr, intr = load_camera(Path(seq_folder), L.shape[0])
        extr = np.asarray(extr)
        intr = np.asarray(intr).reshape(-1)[:4]
        for key, Jseq in (("L", L), ("R", R)):
            # absent hand in single-hand GT = dummy MANO params -> either a frozen
            # identity-pose hand (zero motion across the whole clip) or a degenerate
            # per-frame joint cluster far smaller than a real hand
            static = float(np.ptp(Jseq, axis=0).max()) < 1e-4 if Jseq.shape[0] > 1 else False
            tiny = float(np.median(np.ptp(Jseq, axis=1).max(axis=1))) < 0.02
            absent[key] = static or tiny
    except Exception as e:  # noqa: BLE001
        status = "rgb_only"
        try:
            from slam_failure_diag import diag
            note = f"camera degenerate: {diag(Path(seq_folder))}"
        except Exception:  # noqa: BLE001
            note = f"overlay unavailable: {type(e).__name__}: {str(e)[:120]}"
    T_img = len(frames)
    h, w = frames[0].shape[:2]
    scale, W, H = _upscale(w, h)
    wr = _writer(out_mp4, W, H, fps)
    inframe = 0
    for i, im in enumerate(frames):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
        if status == "overlay":
            T = L.shape[0]
            ji = min(int(round(i * T / max(1, T_img))), T - 1)
            n = 0
            for key, Jseq, col in (("L", L, L_C), ("R", R, R_C)):
                J = Jseq[ji]
                # skip absent hands (clip-level frozen-dummy mask + per-frame
                # degenerate-cluster check)
                if absent[key] or float(np.ptp(J, axis=0).max()) < 0.02:
                    continue
                n += draw_skel(im, project(J, extr[ji], intr * scale), col)
            inframe += n > 0
        wr.send(np.ascontiguousarray(im))
    wr.close()
    return {"status": status, "note": note, "frames": T_img, "fps": fps, "w": w, "h": h,
            "dur_s": round(T_img / fps, 2),
            "hand_pct": round(100 * inframe / T_img, 1) if status == "overlay" else None}


def _load_tar_manos(tar_path, jpg_names):
    """Per-frame (2,55) MANO samples matching the jpg order, or None if absent."""
    manos = {}
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if m.name.endswith(".mano.npy"):
                manos[m.name] = np.load(io.BytesIO(tf.extractfile(m).read()))
    if not manos:
        return None
    out = [manos.get(j[: -len(".image.jpg")] + ".mano.npy") for j in jpg_names]
    return out if any(v is not None for v in out) else None


def _rot6d_to_aa(r6):
    """rot6d (Gram-Schmidt columns) -> axis-angle (3,)."""
    a1, a2 = r6[:3], r6[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    aa, _ = cv2.Rodrigues(R.astype(np.float64))
    return aa.reshape(3).astype(np.float32)


def _native_mano_joints(lds, manos):
    """Full 21-joint world skeletons from .mano.npy (PCA45+betas) + lowdim wrist/rot6d.

    Reuses the pipeline's own MANO forward (mano_features._compute_hand_joints) so the
    viewer renders exactly the joints training would see. Returns (L, R), each (T,21,3)
    world, anchored so joint0 == lowdim wrist (mano_joint_0_world semantics)."""
    import torch
    from lib.pipeline.exporters.mano_features import build_mano_models, _compute_hand_joints
    from lib.pipeline.exporters.mano_codec import hand_pose_pca_to_axis_angle
    T = len(lds)
    trans = np.zeros((2, T, 3), np.float32)
    rot = np.zeros((2, T, 3), np.float32)
    pose = np.zeros((2, T, 45), np.float32)
    betas = np.zeros((2, T, 10), np.float32)
    wrists = np.zeros((2, T, 3), np.float32)
    for t, (ld, mn) in enumerate(zip(lds, manos)):
        ld = ld.reshape(-1)
        mn = np.asarray(mn).reshape(2, 55)
        wrists[0, t], wrists[1, t] = ld[0:3], ld[3:6]
        rot[0, t], rot[1, t] = _rot6d_to_aa(ld[6:12]), _rot6d_to_aa(ld[12:18])
        for hi, side in ((0, "left"), (1, "right")):
            pose[hi, t] = hand_pose_pca_to_axis_angle(mn[hi, :45], side=side)
            betas[hi, t] = mn[hi, 45:]
    dev = _device()
    mano_r, mano_l = build_mano_models(dev)
    tt = lambda a: torch.from_numpy(a)
    out = []
    for hi, model in ((0, mano_l), (1, mano_r)):
        J = _compute_hand_joints(model, tt(trans), tt(rot), tt(pose), tt(betas),
                                 hand_index=hi, device=dev)
        J = J.detach().cpu().numpy()
        J = J - J[:, :1, :] + wrists[hi][:, None, :]  # anchor joint0 at lowdim wrist
        out.append(J)
    return out[0], out[1]


def render_native_overlay(tar_path, out_mp4, fps=30.0):
    """Native-lowdim overlay. Full 21-joint skeletons when .mano.npy ships in the tar
    (EgoDex); wrist+fingertips fallback otherwise (keypoints-only datasets)."""
    frames, lds = load_tar_frames(tar_path)
    if not frames:
        raise RuntimeError(f"no frames in {tar_path}")
    if not lds or all(v is None for v in lds):
        raise RuntimeError(f"no .lowdim.npy members in {tar_path}")
    with tarfile.open(tar_path) as tf:
        jpg_names = sorted(n for n in tf.getnames() if n.endswith(".image.jpg"))
    manos = _load_tar_manos(tar_path, jpg_names)
    joints = None
    # zeros_2x55 schema = placeholder, not real MANO (e.g. EgoDex) -> 6-pt fallback
    if manos is not None and all(m is not None for m in manos) \
            and all(l is not None for l in lds) \
            and max(float(np.abs(np.asarray(m)).max()) for m in manos) > 1e-8:
        try:
            joints = _native_mano_joints(lds, manos)
        except Exception:  # noqa: BLE001  (fall back to 6-pt on any decode issue)
            joints = None
    h, w = frames[0].shape[:2]
    scale, W, H = _upscale(w, h)
    wr = _writer(out_mp4, W, H, fps)
    inframe = 0
    for t, (im, ld) in enumerate(zip(frames, lds)):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
        n = 0
        if ld is not None and ld.shape[-1] >= 116:
            ld = ld.reshape(-1)
            w2c = ld[LD_EXTR].reshape(4, 4)
            fx, fy, cx, cy = ld[LD_INTR] * scale

            def _proj(pts):
                homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
                cam = (w2c @ homo.T).T[:, :3]
                z = np.clip(cam[:, 2:3], 1e-6, None)
                return np.concatenate(
                    [fx * cam[:, :1] / z + cx, fy * cam[:, 1:2] / z + cy, z], axis=1)

            for hi, (sl_w, sl_t, col) in enumerate(
                    ((LD_LWRIST, LD_LTIPS, L_C), (LD_RWRIST, LD_RTIPS, R_C))):
                wrist = ld[sl_w].reshape(1, 3)
                if not np.isfinite(wrist).all() or np.abs(wrist).max() < 1e-8:
                    continue  # absent hand (presence bitmask semantics)
                if joints is not None:
                    n += draw_skel(im, _proj(joints[hi][t]), col)
                else:
                    pts = np.concatenate([wrist, ld[sl_t].reshape(5, 3)])
                    if not np.isfinite(pts).all():
                        continue
                    n += draw_hand6(im, _proj(pts), col)
        inframe += n > 0
        wr.send(np.ascontiguousarray(im))
    wr.close()
    return {"status": "overlay", "note": None, "frames": len(frames), "fps": fps,
            "w": w, "h": h, "dur_s": round(len(frames) / fps, 2),
            "hand_pct": round(100 * inframe / len(frames), 1)}


def render_plain_tar(tar_path, out_mp4, fps=15.0):
    frames, _ = load_tar_frames(tar_path)
    if not frames:
        raise RuntimeError(f"no frames in {tar_path}")
    h, w = frames[0].shape[:2]
    _, W, H = _upscale(w, h)
    wr = _writer(out_mp4, W, H, fps)
    for im in frames:
        wr.send(np.ascontiguousarray(cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)))
    wr.close()
    return {"status": "video_only", "note": None, "frames": len(frames), "fps": fps,
            "w": w, "h": h, "dur_s": round(len(frames) / fps, 2), "hand_pct": None}


def render_mp4_interval(src_mp4, out_mp4, start_sec, dur_sec, max_h=480):
    """Cut + transcode an interval from a source mp4 (Stage-1 / robot episode cards)."""
    vf = f"scale=-2:'min({max_h},ih)'"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(),
           "-y", "-loglevel", "error", "-ss", f"{start_sec:.2f}", "-t", f"{dur_sec:.2f}",
           "-i", str(src_mp4), "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "26", "-movflags", "+faststart", "-an", str(out_mp4)]
    subprocess.run(cmd, check=True, capture_output=True)
    return {"status": "video_only", "note": None, "frames": None, "fps": None,
            "w": None, "h": None, "dur_s": round(dur_sec, 2), "hand_pct": None}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["recon", "native", "plain"])
    ap.add_argument("--tar", required=True)
    ap.add_argument("--seq", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.kind == "recon":
        st = render_recon_overlay(a.tar, a.seq, a.out, a.fps)
    elif a.kind == "native":
        st = render_native_overlay(a.tar, a.out, a.fps)
    else:
        st = render_plain_tar(a.tar, a.out, a.fps)
    print(st)
