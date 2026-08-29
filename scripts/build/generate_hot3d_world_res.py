#!/usr/bin/env python3
"""Convert HOT3D-Clips (official 150-frame Aria clips) into filter-ready artifacts.

HOT3D sequences are long multi-activity egocentric recordings; the official
**HOT3D-Clips** (HF `bop-benchmark/hot3d`) are 3,832 hand-verified 5-second (150-frame)
single-interaction clips. We ingest the **Aria** clips (which carry egocentric RGB,
stream `214-1`) — the other datasets' analog of a per-interaction clip, so no heuristic
segmenter is needed.

Per clip tar this writes the same artifacts as the TACO/OakInk GT ingestion, so
`scripts/build/filter_manifest_by_quality.py --stages infiller` runs unchanged:
- `frames_root/<clip_id>.tar` — egocentric RGB, **undistorted fisheye->pinhole**
  (egosmith's off-screen rules assume pinhole), `<clip_id>_f%05d.image.jpg`.
- `outputs_root/<clip_id>/world_space_res.pth` — [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)]; index 0=left, 1=right.
- `outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0.npz` — c2w traj + pinhole intrinsics.
- tracks_0_<T> + infiller marker.

MANO convention (HOT3D toolkit data_loaders/mano_layer.py): smplx MANO, use_pca=True,
num_pca_comps=15, flat_hand_mean=False. So the full 45-d axis-angle articulation is
`hands_mean + thetas @ hands_components[:15]`; `wrist_xform[:3]` is global orient,
`wrist_xform[3:6]` is the smplx transl (pass-through — same as egosmith's smplx MANO).
Camera `T_world_from_camera` is c2w; RGB is FISHEYE624 (undistorted here).

Runs in the `hot3d` conda env (hand_tracking_toolkit for the fisheye camera model);
does NOT import torch/egosmith. MANO PCA basis is loaded from pre-extracted npz
(`mano_pca_{left,right}.npz`) to avoid the chumpy dependency of the raw MANO pkls.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image

RGB_STREAM = "214-1"


def build_parser():
    p = argparse.ArgumentParser(description="Convert HOT3D-Clips (Aria) to filter-ready artifacts")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--outputs_root", required=True)
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--mano_pca_dir", default="/root/hot3d", help="dir with mano_pca_{left,right}.npz")
    p.add_argument("--hf_repo", default="bop-benchmark/hot3d")
    p.add_argument("--splits", default="train_aria,test_aria", help="HF folders to ingest")
    p.add_argument("--local_tar_dir", default=None, help="Use pre-downloaded clip tars here instead of HF (smoke)")
    p.add_argument("--work_dir", default="/root/hot3d/_work")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--focal_scale", type=float, default=1.0, help="Pinhole focal scale (smaller keeps more fisheye FOV)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include", default=None, help="Regex on clip filename")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--source_id", default="hot3d")
    p.add_argument("--split_label", default="train")
    return p


def _load_pca(mano_pca_dir):
    out = {}
    for hand_index, side in ((0, "left"), (1, "right")):
        d = np.load(Path(mano_pca_dir) / f"mano_pca_{side}.npz")
        out[hand_index] = (d["components"][:15].astype(np.float64), d["mean"].astype(np.float64))
    return out


def _list_hf_clips(hf_repo, splits):
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(hf_repo, repo_type="dataset")
    clips = []
    for sp in splits:
        for f in files:
            if f.startswith(sp + "/") and f.endswith(".tar"):
                clips.append(f)
    return sorted(clips)


def _frame_keys(tar):
    keys = set()
    for n in tar.getnames():
        m = re.match(r"^(\d{6})\.image_214-1\.jpg$", os.path.basename(n))
        if m:
            keys.add(m.group(1))
    return sorted(keys)


def _build_undistort_map(cam_json_214, focal_scale):
    """Return (map_x, map_y, fx, fy, cx, cy, W, H) mapping pinhole pixels -> fisheye source."""
    from hand_tracking_toolkit import camera
    import cv2

    fish = camera.from_json(cam_json_214)
    W, H = int(fish.width), int(fish.height)
    f = np.atleast_1d(np.asarray(fish.f, dtype=np.float64))
    fx = fy = float(f[0]) * focal_scale
    cx, cy = float(fish.c[0]), float(fish.c[1])
    pin = camera.PinholePlaneCameraModel(width=W, height=H, f=(fx, fy), c=(cx, cy),
                                         distort_coeffs=[], T_world_from_eye=fish.T_world_from_eye)
    ys, xs = np.mgrid[0:H, 0:W]
    win = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    eye = pin.window_to_eye(win)            # (N,3) rays in eye space (pose-independent)
    src = fish.eye_to_window(eye)           # (N,2) source fisheye pixels
    map_x = src[:, 0].reshape(H, W).astype(np.float32)
    map_y = src[:, 1].reshape(H, W).astype(np.float32)
    return map_x, map_y, fx, fy, cx, cy, W, H


def _quat_trans_to_w2c(q_wxyz, t_xyz):
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()  # c2w rot
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = np.asarray(t_xyz)
    return c2w


def convert_clip(clip_ref, args, pca):
    import cv2
    from scipy.spatial.transform import Rotation

    clip_name = Path(clip_ref).stem  # clip-001849
    clip_id = "HOT3D_" + re.sub(r"[^A-Za-z0-9]+", "_", clip_name).strip("_")
    frames_root = Path(args.frames_root); outputs_root = Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_out = frames_root / f"{clip_id}.tar"
    done_marker = seq_folder / ".stage_done_infiller"
    result = {"clip": clip_name, "clip_id": clip_id, "status": "ok"}
    if args.resume and tar_out.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        return {**result, "status": "skipped"}

    work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=args.work_dir))
    local_tar = None
    try:
        if args.local_tar_dir:
            local_tar = Path(args.local_tar_dir) / f"{clip_name}.tar"
            if not local_tar.is_file():
                raise FileNotFoundError(f"local tar missing: {local_tar}")
        else:
            from huggingface_hub import hf_hub_download
            local_tar = Path(hf_hub_download(args.hf_repo, clip_ref, repo_type="dataset", local_dir=str(work)))

        with tarfile.open(local_tar, "r") as tar:
            keys = _frame_keys(tar)
            if len(keys) < 2:
                raise ValueError(f"too few RGB frames: {len(keys)}")
            # hand shape (per clip, per hand)
            betas = {0: np.zeros(10), 1: np.zeros(10)}
            if "__hand_shapes.json__" in tar.getnames():
                sh = json.load(tar.extractfile("__hand_shapes.json__"))
                mb = np.asarray(sh["mano"], dtype=np.float64).reshape(-1)
                betas[0] = betas[1] = mb[:10]  # HOT3D stores one mano beta per clip

            T = len(keys)
            trans = np.zeros((2, T, 3), np.float32); rot = np.zeros((2, T, 3), np.float32)
            hand_pose = np.zeros((2, T, 45), np.float32); betas_arr = np.zeros((2, T, 10), np.float32)
            valid = np.zeros((2, T), np.float32)
            traj = np.zeros((T, 7), np.float32)
            umap = None; intr = None
            frames_jpg = []

            for t, k in enumerate(keys):
                cams = json.load(tar.extractfile(f"{k}.cameras.json"))[RGB_STREAM]
                if umap is None:
                    mx, my, fx, fy, cx, cy, W, H = _build_undistort_map(cams, args.focal_scale)
                    umap = (mx, my); intr = (fx, fy, cx, cy, W, H)
                # camera pose (per frame): c2w
                twc = cams["T_world_from_camera"]
                c2w = _quat_trans_to_w2c(twc["quaternion_wxyz"], twc["translation_xyz"])
                q_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                traj[t] = np.concatenate([c2w[:3, 3], q_xyzw]).astype(np.float32)
                # hands
                hands = json.load(tar.extractfile(f"{k}.hands.json"))
                for hand_index, side in ((0, "left"), (1, "right")):
                    hd = hands.get(side)
                    betas_arr[hand_index, t] = betas[hand_index]
                    if not hd or "mano_pose" not in hd:
                        continue
                    mp = hd["mano_pose"]
                    thetas = np.asarray(mp["thetas"], dtype=np.float64).reshape(-1)[:15]
                    wx = np.asarray(mp["wrist_xform"], dtype=np.float64).reshape(-1)
                    comp, mean = pca[hand_index]
                    aa45 = mean + thetas @ comp                      # (45,)
                    rot[hand_index, t] = wx[:3].astype(np.float32)
                    trans[hand_index, t] = wx[3:6].astype(np.float32)
                    hand_pose[hand_index, t] = aa45.astype(np.float32)
                    valid[hand_index, t] = 1.0
                # undistort RGB
                img = np.array(Image.open(io.BytesIO(tar.extractfile(f"{k}.image_{RGB_STREAM}.jpg").read())).convert("RGB"))
                und = cv2.remap(img, umap[0], umap[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                buf = io.BytesIO(); Image.fromarray(und).save(buf, format="JPEG", quality=args.jpeg_quality)
                frames_jpg.append(buf.getvalue())

        # write outputs
        seq_folder.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump([trans, rot, hand_pose, betas_arr, valid], seq_folder / "world_space_res.pth")
        fx, fy, cx, cy, W, H = intr
        slam_dir = seq_folder / "SLAM"; slam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(slam_dir / "hawor_slam_w_scale_0.npz", tstamp=np.arange(T, dtype=np.int64), traj=traj,
                 scale=np.float64(1.0), img_focal=np.float64(0.5 * (fx + fy)),
                 img_center=np.asarray([cx, cy], dtype=np.float64))
        tmp_tar = tar_out.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as tw:
            for t, payload in enumerate(frames_jpg):
                info = tarfile.TarInfo(name=f"{clip_id}_f{t:05d}.image.jpg"); info.size = len(payload)
                tw.addfile(info, io.BytesIO(payload))
        tmp_tar.replace(tar_out)
        tracks = seq_folder / f"tracks_0_{T}"; tracks.mkdir(parents=True, exist_ok=True)
        (tracks / ".hot3d_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
        done_marker.touch()
        result["frames"] = int(T)
        result["participant"] = clip_name
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def _convert_star(task):
    return convert_clip(*task)


def write_manifest(results, args):
    # Use the canonical helpers (like generate_taco_world_res.py) so the descriptor is
    # serialized via ClipDescriptor.to_dict / ClipManifestRecord — no hand-built dicts to drift.
    from lib.pipeline.clips.clip_manifest import write_clip_manifest, ClipManifestRecord
    from lib.pipeline.datasets.descriptors import ClipDescriptor

    records = []
    for res in results:
        if res["status"] not in ("ok", "skipped"):
            continue
        clip_id = res["clip_id"]
        tar_path = Path(args.frames_root) / f"{clip_id}.tar"
        if not tar_path.is_file():
            continue
        frame_names, frame_offsets = [], []
        with tarfile.open(tar_path, "r") as reader:
            members = sorted([m for m in reader if m.isfile() and m.name.endswith(".image.jpg")], key=lambda m: m.name)
            for m in members:
                frame_names.append(m.name); frame_offsets.append([int(m.offset_data), int(m.size)])
        descriptor = ClipDescriptor.from_tar_shard(
            clip_id=clip_id,
            clip_name=clip_id,
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / clip_id).resolve()),
            shard_path=str(tar_path.resolve()),
            frame_names=frame_names,
            frame_offsets=frame_offsets,
            extra={"adapter": "hot3d_clip", "dataset_name": args.source_id, "clip": res["clip"]},
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split_label,
                descriptor=descriptor,
                group_id=args.source_id,
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main():
    args = build_parser().parse_args()
    for d in (args.frames_root, args.outputs_root, args.work_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    pca = _load_pca(args.mano_pca_dir)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.local_tar_dir:
        clips = sorted(str(p.name) for p in Path(args.local_tar_dir).glob("clip-*.tar"))
    else:
        clips = _list_hf_clips(args.hf_repo, splits)
    if args.include:
        pat = re.compile(args.include); clips = [c for c in clips if pat.search(c)]
    if args.limit:
        clips = clips[: args.limit]
    print(f"HOT3D clips to convert: {len(clips)}", flush=True)

    started = time.perf_counter()
    if args.workers <= 1:
        results = [convert_clip(c, args, pca) for c in clips]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_convert_star, [(c, args, pca) for c in clips], chunksize=1)):
                results.append(r)
                if (i + 1) % 20 == 0 or (i + 1) == len(clips):
                    print(f"[{i+1}/{len(clips)}] {r['clip_id']} {r['status']}", flush=True)

    n = write_manifest(results, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {"hf_repo": args.hf_repo, "splits": splits, "total": len(clips),
              "converted_ok": sum(1 for r in results if r["status"] == "ok"),
              "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
              "failed": len(failed), "manifest_records": n,
              "conventions": {"mano": "smplx use_pca=15 flat_hand_mean=False", "trans": "smplx transl passthrough",
                              "camera": "T_world_from_camera c2w", "rgb": "fisheye624->pinhole undistorted",
                              "focal_scale": args.focal_scale},
              "failures": failed[:20], "elapsed_sec": time.perf_counter() - started}
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("HOT3D_CONVERT_DONE" if not failed else "HOT3D_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
