#!/usr/bin/env python3
"""Convert ARCTIC egocentric sequences into filter-ready artifacts (GT ingestion).

ARCTIC (Fan et al., CVPR'23) is dexterous bimanual object manipulation captured with an
allocentric rig **and** an egocentric (Aria) head camera (view ``0``). It ships
ground-truth **bimanual MANO** (`raw_seqs/<subj>/<seq>.mano.npy`) and per-frame ego
**world->cam** extrinsics + intrinsics + 8-param distortion
(`raw_seqs/<subj>/<seq>.egocam.dist.npy`). We ingest the **ego view** so the same quality
filter used for TACO / OakInk / HOT3D runs unchanged.

Per sequence this writes the standard artifacts (`--stages infiller`):
- ``frames_root/<clip_id>.tar`` — ego RGB, **undistorted** (cv2 rational model, K kept),
  ``<clip_id>_f%05d.image.jpg``.
- ``outputs_root/<clip_id>/world_space_res.pth`` — [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)]; index 0=left, 1=right.
- ``outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0.npz`` — c2w traj + pinhole K.
- ``tracks_0_<T>`` + ``.stage_done_infiller`` marker.

Conventions (verified against the ARCTIC repo):
- MANO: full 45-d axis-angle articulation (`pose`), global orient `rot`, `trans`, `shape`.
  ARCTIC kp2d/kp3d live in **undistorted** image space (`arctic_dataset.py`), so we
  undistort the frame with (K, dist8) keeping K, and project with K.
- Camera: `egocam.dist.npy` gives world->cam (`R_k_cam_np`, `T_k_cam_np`); we invert to
  c2w for the SLAM sidecar (the camera reader re-inverts to w2c). Ego K used **as-is** on
  the 2800x2000 image (repo: "no scaling for egocam to make intrinsics consistent").
- Sync: image file number = gt_index + `ioi_offset[subject]` (from `meta/misc.json`).

Runs in the `egosmith` conda env (numpy/cv2/scipy/joblib/PIL); no torch, no HF.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

EGO_VIEW = "0"


def build_parser():
    p = argparse.ArgumentParser(description="Convert ARCTIC ego sequences to filter-ready artifacts")
    p.add_argument("--arctic_root", default="/root/arctic/data/arctic_data/data")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--outputs_root", required=True)
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--subjects", default=None, help="comma list e.g. s01,s02 (default: all with ego images)")
    p.add_argument("--work_dir", default="/root/arctic_run/_work")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--focal_scale", type=float, default=1.0,
                   help="Pinhole focal scale for undistortion. Keep at 1.0 (K-as-is, ~60deg HFOV): this "
                        "is ARCTIC's own convention and the only region the 8-param rational distortion "
                        "represents cleanly. Values <1 try to widen FOV but the rational model extrapolates "
                        "to garbage (black lobes) beyond ~60deg, so do not lower for ARCTIC.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include", default=None, help="substring filter on <subj>/<seq>")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--source_id", default="arctic")
    p.add_argument("--split_label", default="train")
    return p


def _load_misc(arctic_root):
    misc = json.load(open(Path(arctic_root) / "meta" / "misc.json"))
    return {s: int(misc[s]["ioi_offset"]) for s in misc}


def _discover_seqs(arctic_root, subjects, include):
    root = Path(arctic_root)
    img_root = root / "images"
    raw_root = root / "raw_seqs"
    subs = subjects or sorted(d.name for d in img_root.iterdir() if d.is_dir() and d.name.startswith("s"))
    seqs = []
    for subj in subs:
        subj_img = img_root / subj
        if not subj_img.is_dir():
            continue
        for seq_dir in sorted(subj_img.iterdir()):
            if not (seq_dir / EGO_VIEW).is_dir():
                continue
            seq = seq_dir.name
            mano = raw_root / subj / f"{seq}.mano.npy"
            ego = raw_root / subj / f"{seq}.egocam.dist.npy"
            if mano.is_file() and ego.is_file():
                if include and include not in f"{subj}/{seq}":
                    continue
                seqs.append((subj, seq))
    return seqs


def _build_undistort_map(K, dist8, W, H, focal_scale):
    """Undistort to a pinhole with focal scaled by ``focal_scale`` (keeps principal point).

    Smaller focal_scale widens the retained field of view: at 1.0 only ~60 deg HFOV of the
    Aria fisheye survives; ~0.5 retains ~98 deg. The scaled ``newK`` is what we project with
    and write to the SLAM sidecar, so image + intrinsics + projection stay mutually consistent.
    """
    newK = K.copy()
    newK[0, 0] = K[0, 0] * focal_scale
    newK[1, 1] = K[1, 1] * focal_scale
    map1, map2 = cv2.initUndistortRectifyMap(
        K, np.asarray(dist8, dtype=np.float64), None, newK, (W, H), cv2.CV_32FC1
    )
    return map1, map2, newK


def convert_seq(subj, seq, args, ioi_offset):
    root = Path(args.arctic_root)
    clip_id = f"ARCTIC_{subj}_{seq}"
    frames_root = Path(args.frames_root)
    outputs_root = Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_out = frames_root / f"{clip_id}.tar"
    done_marker = seq_folder / ".stage_done_infiller"
    result = {"subj": subj, "seq": seq, "clip_id": clip_id, "status": "ok"}
    if args.resume and tar_out.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        return {**result, "status": "skipped"}

    work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=args.work_dir))
    try:
        mano = np.load(root / "raw_seqs" / subj / f"{seq}.mano.npy", allow_pickle=True).item()
        ego = np.load(root / "raw_seqs" / subj / f"{seq}.egocam.dist.npy", allow_pickle=True).item()
        K = np.asarray(ego["intrinsics"], dtype=np.float64)
        dist8 = np.asarray(ego["dist8"], dtype=np.float64).reshape(-1)
        R_w2c = np.asarray(ego["R_k_cam_np"], dtype=np.float64)   # (T,3,3)
        t_w2c = np.asarray(ego["T_k_cam_np"], dtype=np.float64).reshape(-1, 3)  # (T,3)

        img_dir = root / "images" / subj / seq / EGO_VIEW
        off = ioi_offset.get(subj, 1)
        t_mano = int(mano["right"]["pose"].shape[0])
        t_cam = int(R_w2c.shape[0])
        # gt index t -> image file (t + off) (1-indexed files)
        max_t_img = 0
        while (img_dir / f"{max_t_img + off:05d}.jpg").is_file():
            max_t_img += 1
        T = min(t_mano, t_cam, max_t_img)
        if T < 2:
            raise ValueError(f"too few synced frames: mano={t_mano} cam={t_cam} img={max_t_img}")

        W, H = 2800, 2000
        map1, map2, newK = _build_undistort_map(K, dist8, W, H, args.focal_scale)

        trans = np.zeros((2, T, 3), np.float32)
        rot = np.zeros((2, T, 3), np.float32)
        hand_pose = np.zeros((2, T, 45), np.float32)
        betas_arr = np.zeros((2, T, 10), np.float32)
        valid = np.zeros((2, T), np.float32)
        for hand_index, side in ((0, "left"), (1, "right")):
            hd = mano.get(side)
            if hd is None:
                continue
            betas = np.asarray(hd["shape"], dtype=np.float32).reshape(-1)[:10]
            r = np.asarray(hd["rot"], dtype=np.float32)[:T]
            p = np.asarray(hd["pose"], dtype=np.float32)[:T]
            tr = np.asarray(hd["trans"], dtype=np.float32)[:T]
            rot[hand_index, :T] = r
            hand_pose[hand_index, :T] = p
            trans[hand_index, :T] = tr
            betas_arr[hand_index, :T] = betas
            valid[hand_index, :T] = 1.0

        traj = np.zeros((T, 7), np.float32)
        frames_jpg = []
        for t in range(T):
            Rc2w = R_w2c[t].T
            tc2w = -Rc2w @ t_w2c[t]
            q_xyzw = Rotation.from_matrix(Rc2w).as_quat()
            traj[t] = np.concatenate([tc2w, q_xyzw]).astype(np.float32)
            img = np.array(Image.open(img_dir / f"{t + off:05d}.jpg").convert("RGB"))
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            buf = io.BytesIO()
            Image.fromarray(und).save(buf, format="JPEG", quality=args.jpeg_quality)
            frames_jpg.append(buf.getvalue())

        seq_folder.mkdir(parents=True, exist_ok=True)
        joblib.dump([trans, rot, hand_pose, betas_arr, valid], seq_folder / "world_space_res.pth")
        fx, fy, cx, cy = float(newK[0, 0]), float(newK[1, 1]), float(newK[0, 2]), float(newK[1, 2])
        slam_dir = seq_folder / "SLAM"
        slam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            slam_dir / "hawor_slam_w_scale_0.npz",
            tstamp=np.arange(T, dtype=np.int64), traj=traj, scale=np.float64(1.0),
            img_focal=np.float64(0.5 * (fx + fy)), img_center=np.asarray([cx, cy], dtype=np.float64),
        )
        tmp_tar = tar_out.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as tw:
            for t, payload in enumerate(frames_jpg):
                info = tarfile.TarInfo(name=f"{clip_id}_f{t:05d}.image.jpg")
                info.size = len(payload)
                tw.addfile(info, io.BytesIO(payload))
        tmp_tar.replace(tar_out)
        tracks = seq_folder / f"tracks_0_{T}"
        tracks.mkdir(parents=True, exist_ok=True)
        (tracks / ".arctic_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
        done_marker.touch()
        result["frames"] = int(T)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def _convert_star(task):
    return convert_seq(*task)


def write_manifest(results, args):
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
            members = sorted(
                [m for m in reader if m.isfile() and m.name.endswith(".image.jpg")],
                key=lambda m: m.name,
            )
            for m in members:
                frame_names.append(m.name)
                frame_offsets.append([int(m.offset_data), int(m.size)])
        descriptor = {
            "clip_id": clip_id, "clip_name": clip_id, "storage_kind": "tar_shard",
            "root_dir": str(Path(args.frames_root).resolve()),
            "seq_folder": str((Path(args.outputs_root) / clip_id).resolve()),
            "frame_names": frame_names, "frame_offsets": frame_offsets,
            "shard_path": str(tar_path.resolve()),
            "extra": {"adapter": "arctic_tar", "dataset_name": args.source_id,
                      "subject": res["subj"], "arctic_seq_name": res["seq"]},
        }
        records.append({"clip_id": clip_id, "source_id": args.source_id, "split": args.split_label,
                        "group_id": res["subj"], "descriptor": descriptor, "metadata": {}})
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def main():
    args = build_parser().parse_args()
    for d in (args.frames_root, args.outputs_root, args.work_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    ioi_offset = _load_misc(args.arctic_root)
    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else None
    seqs = _discover_seqs(args.arctic_root, subjects, args.include)
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"ARCTIC sequences to convert: {len(seqs)}", flush=True)

    started = time.perf_counter()
    tasks = [(subj, seq, args, ioi_offset) for subj, seq in seqs]
    if args.workers <= 1:
        results = [convert_seq(*t) for t in tasks]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_convert_star, tasks, chunksize=1)):
                results.append(r)
                if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                    print(f"[{i+1}/{len(tasks)}] {r['clip_id']} {r['status']}", flush=True)

    n = write_manifest(results, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "arctic_root": args.arctic_root, "total": len(seqs),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed), "manifest_records": n,
        "conventions": {"mano": "full 45d aa (rot,pose,trans,shape); undistorted-space",
                        "camera": "egocam world->cam inverted to c2w; K as-is (no scale)",
                        "rgb": f"cv2 rational undistort, focal_scale={args.focal_scale} (~98deg HFOV at 0.5)",
                        "sync": "img = gt + ioi_offset"},
        "failures": failed[:20], "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("ARCTIC_CONVERT_DONE" if not failed else "ARCTIC_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
