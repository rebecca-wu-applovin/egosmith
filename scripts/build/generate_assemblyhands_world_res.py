#!/usr/bin/env python3
"""Convert AssemblyHands ego GT into filter-ready seq_folders + frame tars + manifest.

AssemblyHands (CVPR 2023, ut-vision / Meta) layers triangulated 3D hand keypoints on
Assembly101 monochrome egocentric video. The GCS mirror ships the annotations
(``gs://foundational-research/hoi-dataset/AssemblyHands/annotations/<split>/``):
- ``assemblyhands_<split>_ego_data_v1-1.json``   COCO images + 2D kps + joint_valid
- ``assemblyhands_<split>_joint_3d_v1-1.json``   per (seq, frame) 42x3 world-mm joints
- ``assemblyhands_<split>_ego_calib_v1-1.json``  per-seq pinhole K per ego cam +
  per-frame 3x4 world->cam extrinsics (camera keys carry the ``_mono10bit`` suffix)
The RECTIFIED ego images (636x480 pinhole) are NOT on the GCS mirror; they are fetched
from the official Google Drive release (one tar.gz per sequence segment) and extracted
under ``--images_root`` as ``ego_images_rectified/<split>/<seq>/<cam>/%06d.jpg``.

Convention (probe-verified vs the dataset's own 2D keypoints, mean 2.2e-5 px):
``uv = K @ (R @ X_world_mm + t)``. World units mm. Annotations are at 30 Hz on a
60 fps frame-index grid (frame ids step by 2).

Skeleton (annotations/skeleton.txt): 42 joints, right 0-20 / left 21-41, each finger
ordered TIP->proximal (e.g. r_thumb4 = idx 0 is the tip, r_thumb1 = idx 3 the CMC),
wrist last (right 20, left 41). All 21 MANO-order targets exist natively (real
thumb-CMC), so no synthetic joints are needed. MANO is torch-fit to the world targets
with the SHOW3D centroid-translation fitter (the triangulated wrist does not coincide
with MANO's); the left hand is fitted via x-mirroring.

Clips: one per (sequence, ego camera, contiguous 30 Hz annotated run); runs break
where the annotated frame index steps by more than 2 (images without hands were
removed from the release) or where the image file is missing on disk.

NOTE: AssemblyHands is EVAL/CALIBRATION ONLY downstream -- do not ship to train.

Outputs per clip (identical contract to generate_show3d_world_res.py):
- frames_root/<clip_id>.tar (<clip_id>_f%05d.image.jpg)
- outputs_root/<clip_id>/world_space_res.pth [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)] (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz (per-frame c2w, meters)
- est_focal.txt, tracks_0_<T-1>/.assemblyhands_gt, infiller done marker
- manifest JSONL + conversion report JSON
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402
from generate_show3d_world_res import _fit_right_mano_centroid, _mirror_aa, _MIRROR  # noqa: E402

# MANO 21 target slot [wrist, thumb1..tip, index1..tip, middle.., ring.., pinky..]
# -> AssemblyHands joint index (right hand block; left = same order in 21..41 block).
_ASSY_RIGHT_TO_MANO = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]
_ASSY_LEFT_TO_MANO = [41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert AssemblyHands ego GT to filter-ready artifacts")
    parser.add_argument("--annotations_root", required=True, help="Local dir with <split>/assemblyhands_<split>_{ego_data,joint_3d,ego_calib}_v1-1.json")
    parser.add_argument("--images_root", required=True, help="Local dir containing ego_images_rectified/<split>/...")
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--splits", default="val", help="Comma-separated annotation splits (train,val)")
    parser.add_argument("--min_frames", type=int, default=60, help="Min 30Hz frames per clip (60 = 2s)")
    parser.add_argument("--min_valid_joints", type=int, default=21, help="3D joint_valid count needed for a hand-frame to be a fit target")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit_iters", type=int, default=400)
    parser.add_argument("--fit_chunk", type=int, default=256, help="Fitter frame chunk (keep small on a shared GPU)")
    parser.add_argument("--include", default=None, help="Regex on clip_id")
    parser.add_argument("--limit", type=int, default=None, help="Cap on clips")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_id", default="assemblyhands")
    return parser


def _seq_short(seq_name: str) -> str:
    # nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345 -> 9012_c07c_164345
    m = re.match(r"nusar-2021_action_both_(\d+)-(\w+)_\d+_user_id_[\d-]+_(\d+)", seq_name)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return re.sub(r"[^A-Za-z0-9]+", "_", seq_name)[-24:]


def assy_to_mano_targets(world_mm: np.ndarray, side: str) -> np.ndarray:
    """(42,3) AssemblyHands world joints (mm) -> (21,3) MANO-ordered targets (m)."""
    idx = _ASSY_LEFT_TO_MANO if side == "left" else _ASSY_RIGHT_TO_MANO
    return np.asarray(world_mm, dtype=np.float64)[idx] * 1e-3


def fit_side(targets_world: np.ndarray, side: str, device: str, num_iters: int, chunk: int):
    targets = np.asarray(targets_world, dtype=np.float64)
    if side == "left":
        targets = targets @ _MIRROR.T
    trans, rot, pose, betas = _fit_right_mano_centroid(targets.astype(np.float32), device, num_iters, chunk=chunk)
    if side == "left":
        trans = (trans @ _MIRROR.T).astype(np.float32)
        rot = _mirror_aa(rot)
        pose = _mirror_aa(pose.reshape(-1, 15, 3)).reshape(-1, 45)
    return trans.astype(np.float32), rot.astype(np.float32), pose.astype(np.float32), betas.reshape(10).astype(np.float32)


def discover_clips(split: str, args) -> tuple[list[dict], dict]:
    """Load split annotations, group images per (seq, cam), split into contiguous runs."""
    ann_dir = Path(args.annotations_root) / split
    ego_data = json.loads((ann_dir / f"assemblyhands_{split}_ego_data_v1-1.json").read_text())
    joint3d = json.loads((ann_dir / f"assemblyhands_{split}_joint_3d_v1-1.json").read_text())["annotations"]
    calib = json.loads((ann_dir / f"assemblyhands_{split}_ego_calib_v1-1.json").read_text())["calibration"]

    by_seq_cam: dict[tuple[str, str], list[dict]] = {}
    for im in ego_data["images"]:
        by_seq_cam.setdefault((im["seq_name"], im["camera"]), []).append(im)

    clips = []
    stats = {"images": len(ego_data["images"]), "seq_cams": len(by_seq_cam), "runs_too_short": 0,
             "missing_image_file": 0, "missing_j3d_or_extr": 0}
    for (seq, cam), ims in sorted(by_seq_cam.items()):
        cam_full = cam + "_mono10bit"
        seq_j3d = joint3d.get(seq, {})
        seq_cal = calib.get(seq, {})
        extr = seq_cal.get("extrinsics", {})
        intr = seq_cal.get("intrinsics", {}).get(cam_full)
        if intr is None:
            continue
        usable = []
        for im in sorted(ims, key=lambda r: r["frame_idx"]):
            fid = "%06d" % im["frame_idx"]
            img_path = Path(args.images_root) / im["file_name"]
            if fid not in seq_j3d or fid not in extr or cam_full not in extr.get(fid, {}):
                stats["missing_j3d_or_extr"] += 1
                continue
            if not img_path.is_file():
                stats["missing_image_file"] += 1
                continue
            usable.append((im["frame_idx"], img_path, fid))
        # contiguous 30Hz runs: frame ids step by 2
        run: list = []
        runs = []
        for item in usable:
            if run and item[0] - run[-1][0] != 2:
                runs.append(run)
                run = []
            run.append(item)
        if run:
            runs.append(run)
        for run_idx, run in enumerate(runs):
            if len(run) < args.min_frames:
                stats["runs_too_short"] += 1
                continue
            clip_id = f"ASSYHANDS_{split}_{_seq_short(seq)}_{cam.replace('HMC_', '')}_r{run_idx:02d}"
            clips.append({
                "clip_id": clip_id, "split": split, "seq": seq, "camera": cam,
                "camera_full": cam_full, "run": run,
                "K": np.asarray(intr, dtype=np.float64),
                "j3d": seq_j3d, "extr": extr,
            })
    return clips, stats


def build_world_res(clip: dict, args) -> tuple[list[np.ndarray], dict]:
    run = clip["run"]
    T = len(run)
    trans = np.zeros((2, T, 3), dtype=np.float32)
    rot = np.zeros((2, T, 3), dtype=np.float32)
    pose45 = np.zeros((2, T, 45), dtype=np.float32)
    betas = np.zeros((2, T, 10), dtype=np.float32)
    valid = np.zeros((2, T), dtype=np.float32)
    fit_stats = {}
    for side, hand_index, sl in (("left", 0, slice(21, 42)), ("right", 1, slice(0, 21))):
        target_list, frame_pos = [], []
        for t, (_fidx, _path, fid) in enumerate(run):
            entry = clip["j3d"][fid]
            jv = np.asarray(entry["joint_valid"], dtype=np.float64).reshape(-1)[sl]
            if int((jv > 0).sum()) < args.min_valid_joints:
                continue
            target_list.append(assy_to_mano_targets(np.asarray(entry["world_coord"]), side))
            frame_pos.append(t)
        fit_stats[side] = {"valid_frames": len(frame_pos), "total_frames": T}
        if len(frame_pos) < 2:
            continue
        targets = np.stack(target_list, axis=0)
        f_trans, f_rot, f_pose, f_betas = fit_side(targets, side, args.device, args.fit_iters, args.fit_chunk)
        idx = np.asarray(frame_pos, dtype=np.int64)
        trans[hand_index, idx] = f_trans
        rot[hand_index, idx] = f_rot
        pose45[hand_index, idx] = f_pose
        betas[hand_index] = f_betas[None, :]
        valid[hand_index, idx] = 1.0
    return [trans, rot, pose45, betas, valid], fit_stats


def convert_clip(clip: dict, args) -> dict:
    clip_id = clip["clip_id"]
    frames_root, outputs_root = Path(args.frames_root), Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_path = frames_root / f"{clip_id}.tar"
    result = {"clip_id": clip_id, "split": clip["split"], "seq": clip["seq"],
              "camera": clip["camera"], "status": "ok"}

    done_marker = get_stage_done_marker(seq_folder, "infiller")
    if args.resume and tar_path.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        result["status"] = "skipped"
        return result
    try:
        run = clip["run"]
        T = len(run)
        payload, fit_stats = build_world_res(clip, args)
        result["fit_stats"] = fit_stats

        seq_folder.mkdir(parents=True, exist_ok=True)
        tmp_tar = tar_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as writer:
            for t, (_fidx, img_path, _fid) in enumerate(run):
                writer.add(img_path, arcname=f"{clip_id}_f{t:05d}.image.jpg")
        tmp_tar.replace(tar_path)

        joblib.dump(payload, seq_folder / "world_space_res.pth")

        cam_full = clip["camera_full"]
        c2w = np.zeros((T, 4, 4), dtype=np.float64)
        for t, (_fidx, _path, fid) in enumerate(run):
            rt = np.asarray(clip["extr"][fid][cam_full], dtype=np.float64)  # 3x4 w2c, mm
            m = np.eye(4)
            m[:3, :3] = rt[:, :3]
            m[:3, 3] = rt[:, 3] * 1e-3
            c2w[t] = np.linalg.inv(m)
        quat_xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
        traj = np.concatenate([c2w[:, :3, 3], quat_xyzw], axis=1).astype(np.float32)
        K = clip["K"]
        focal = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
        slam_dir = seq_folder / "SLAM"
        slam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
            tstamp=np.arange(T, dtype=np.int64),
            traj=traj,
            scale=np.float64(1.0),
            img_focal=np.float64(focal),
            img_center=np.asarray([float(K[0, 2]), float(K[1, 2])], dtype=np.float64),
        )
        (seq_folder / "est_focal.txt").write_text(f"{focal:.6f}\n")
        tracks_dir = seq_folder / f"tracks_0_{T - 1}"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        (tracks_dir / ".assemblyhands_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        result["frames"] = int(T)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def write_manifest(results: list[dict], clips: list[dict], args) -> int:
    from lib.pipeline.clips.clip_manifest import write_clip_manifest, ClipManifestRecord
    from lib.pipeline.datasets.descriptors import ClipDescriptor

    by_id = {c["clip_id"]: c for c in clips}
    records = []
    for result in results:
        if result["status"] not in ("ok", "skipped"):
            continue
        clip_id = result["clip_id"]
        tar_path = Path(args.frames_root) / f"{clip_id}.tar"
        if not tar_path.is_file():
            continue
        frame_names, frame_offsets = [], []
        with tarfile.open(tar_path, "r") as reader:
            members = sorted([m for m in reader if m.isfile() and m.name.endswith(".image.jpg")],
                             key=lambda m: m.name)
        for member in members:
            frame_names.append(member.name)
            frame_offsets.append([int(member.offset_data), int(member.size)])
        clip = by_id[clip_id]
        descriptor = ClipDescriptor.from_tar_shard(
            clip_id=clip_id,
            clip_name=clip_id,
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / clip_id).resolve()),
            shard_path=str(tar_path.resolve()),
            frame_names=frame_names,
            frame_offsets=frame_offsets,
            extra={
                "adapter": "assemblyhands_tar",
                "dataset_name": args.source_id,
                "seq_name": clip["seq"],
                "camera": clip["camera"],
                "annotation_split": clip["split"],
                "usage": "eval_calibration_only",
            },
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=clip["split"],
                descriptor=descriptor,
                group_id=clip["seq"],
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)

    clips, disc_stats = [], {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_clips, stats = discover_clips(split, args)
        clips.extend(split_clips)
        disc_stats[split] = stats | {"clips": len(split_clips)}
    if args.include:
        pattern = re.compile(args.include)
        clips = [c for c in clips if pattern.search(c["clip_id"])]
    if args.limit:
        clips = clips[: args.limit]
    print(f"AssemblyHands: {len(clips)} clips selected; discovery={json.dumps(disc_stats)}", flush=True)

    started = time.perf_counter()
    results = []
    for idx, clip in enumerate(clips):
        results.append(convert_clip(clip, args))
        print(f"[{idx + 1}/{len(clips)}] {results[-1]['clip_id']} {results[-1]['status']}"
              + (f" :: {results[-1].get('error', '')[:200]}" if results[-1]["status"] == "failed" else ""), flush=True)

    manifest_count = write_manifest(results, clips, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "splits": args.splits,
        "discovery": disc_stats,
        "clips_selected": len(clips),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "usage": "EVAL/CALIBRATION ONLY -- do not ship to train",
        "conventions": {
            "gt": "42-joint triangulated skeleton (mm, world) -> MANO torch fit (SHOW3D centroid fitter)",
            "joint_order": "per finger TIP->proximal, wrist last (right 0-20, left 21-41)",
            "left_hand": "x-mirror fit (diag(-1,1,1) conjugation on aa params)",
            "camera": "per-frame 3x4 w2c extrinsics (mm) inverted to c2w (m); rectified pinhole K",
            "projection_probe": "uv = K @ (R@X+t) vs shipped 2D kps: mean 2.2e-5 px (demo+val)",
            "fps": "annotations at 30 Hz (frame ids step 2 on the 60fps grid)",
        },
        "failures": failed[:20],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("ASSEMBLYHANDS_CONVERT_DONE" if not failed else "ASSEMBLYHANDS_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
