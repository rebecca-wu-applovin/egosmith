#!/usr/bin/env python3
"""Convert DexCap raw mocap recordings into filter-ready seq_folders + frame tars + manifest.

DexCap (Wang et al. 2024, CC-BY-4.0, HF chenwangj/DexCap-Data) is chest-camera
egocentric bimanual hand mocap: RealSense L515 RGB-D (1280x720 @ 30 Hz -- confirmed:
wipe raws total 54,786 frames ~= the stated 30 minutes) + per-frame EMF-glove hand
joints (occlusion-free in-contact fingers) + T265 6-DoF trackers for the chest camera
and both wrists. Raw layout per scenario dir (``save_data_*``):
``frame_i/{color_image.jpg, depth_image.png, pose.txt (chest T265 c2w),
pose_2.txt/pose_3.txt (left/right wrist-tracker c2w), left/right_hand_joint.txt
(21x3 m, glove frame), left/right_hand_joint_ori.txt (21x4 quat xyzw)}`` plus
per-scenario ``calib_offset*.txt`` / ``calib_ori_offset*.txt`` (manual visual
alignment of glove joints onto the RGB-D cloud) and ``clip_marks.json`` (demo
start/end frame ids -- outside the marks hands leave the view / reset).

Joint order is wrist + 4 joints/finger thumb->pinky == the MANO 21-target order of
the pipeline's FPHA fitter (identity mapping; bone-length probe confirms).

World-frame joints reproduce the official ``STEP2_build_dataset/dataset_utils.py::
read_pose_data`` chain (minus robot-table shift + robot retarget): translate wrist to
origin -> rotate by wrist-quat^T -> flip z -> axis rotations y(-90) x(+90) z(-90) ->
per-scenario calib euler + translation -> apply ``pose_{2,3} @ between_cam_{2,3}``.
Camera c2w = ``pose.txt @ between_cam`` (R=diag(1,-1,-1), t=(0,0.076,0)); intrinsics
are the repo's L515 constants (fx=898.2010, fy=897.8667, cx=657.4981, cy=364.3095).
Convention probe: projected joints lock onto the gloved hands; depth cross-check
median |depth - z| ~ 14-25 mm (right) / 38-73 mm (left; glove thickness + native
session drift of the wrist trackers).

MANO is torch-fit to the world joints with the SHOW3D centroid-translation fitter
(EMF wrist sits inside the glove mount, not at MANO's wrist); left hand mirror-fit.

Outputs per clip (identical contract to generate_show3d_world_res.py):
- frames_root/<clip_id>.tar (<clip_id>_f%05d.image.jpg)
- outputs_root/<clip_id>/world_space_res.pth [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)] (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz (per-frame c2w, meters)
- est_focal.txt, tracks_0_<T-1>/.dexcap_gt, infiller done marker
- manifest JSONL + conversion report JSON
"""

from __future__ import annotations

import argparse
import json
import re
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

# RealSense L515 color/depth intrinsics (DexCap STEP2_build_dataset/hyperparameters.py)
FX, FY, CX, CY = 898.2010498046875, 897.86669921875, 657.4981079101562, 364.30950927734375
IMG_W, IMG_H = 1280, 720
NATIVE_FPS = 30.0

_BETWEEN_CAM = np.eye(4)
_BETWEEN_CAM[:3, :3] = np.diag([1.0, -1.0, -1.0])
_BETWEEN_CAM[:3, 3] = [0.0, 0.076, 0.0]
_BETWEEN_CAM_2 = np.eye(4)
_BETWEEN_CAM_2[:3, 3] = [0.0, -0.032, 0.0]  # left wrist tracker
_BETWEEN_CAM_3 = np.eye(4)
_BETWEEN_CAM_3[:3, 3] = [0.0, -0.064, 0.0]  # right wrist tracker

_AX_Y = Rotation.from_rotvec(np.array([0.0, 1.0, 0.0]) * (-np.pi / 2)).as_matrix()
_AX_X = Rotation.from_rotvec(np.array([1.0, 0.0, 0.0]) * (np.pi / 2)).as_matrix()
_AX_Z = Rotation.from_rotvec(np.array([0.0, 0.0, 1.0]) * (-np.pi / 2)).as_matrix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert DexCap raw recordings to filter-ready artifacts")
    parser.add_argument("--raw_roots", required=True,
                        help="Comma-separated dirs containing save_data_* scenario folders "
                             "(e.g. .../wipe/save_wipe_1-14,.../packaging/save_packaging_wild_1-20)")
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--min_frames", type=int, default=60, help="Min frames per clip (2s @ 30fps)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit_iters", type=int, default=400)
    parser.add_argument("--fit_chunk", type=int, default=256, help="Fitter frame chunk (keep small on a shared GPU)")
    parser.add_argument("--include", default=None, help="Regex on clip_id")
    parser.add_argument("--limit", type=int, default=None, help="Cap on clips")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_id", default="dexcap")
    parser.add_argument("--split", default="train")
    return parser


def _scenario_task(scenario_dir: Path) -> str:
    # save_data_wipe_1-14_01 -> wipe ; save_data_packaging_05 -> packaging
    m = re.match(r"save_data_([A-Za-z]+)", scenario_dir.name)
    return m.group(1) if m else "task"


def _scenario_tag(scenario_dir: Path) -> str:
    m = re.search(r"(\d+)$", scenario_dir.name)
    return m.group(1) if m else re.sub(r"[^A-Za-z0-9]+", "", scenario_dir.name)[-6:]


def discover_clips(raw_roots: list[Path], args) -> list[dict]:
    clips = []
    for root in raw_roots:
        for scenario in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("save_data_")):
            marks_path = scenario / "clip_marks.json"
            if not marks_path.is_file():
                continue
            marks = json.loads(marks_path.read_text())
            task, tag = _scenario_task(scenario), _scenario_tag(scenario)
            for demo_idx, mark in enumerate(marks):
                start = int(str(mark["start"]).split("_")[-1])
                end = int(str(mark["end"]).split("_")[-1])
                if end - start + 1 < args.min_frames:
                    continue
                clips.append({
                    "clip_id": f"DEXCAP_{task}_{tag}_d{demo_idx:02d}",
                    "scenario": str(scenario),
                    "scenario_name": scenario.name,
                    "task": task,
                    "start": start,
                    "end": end,
                })
    return clips


def _load_calib(scenario: Path) -> dict:
    return {
        "left": (np.loadtxt(scenario / "calib_ori_offset_left.txt"),
                 np.loadtxt(scenario / "calib_offset_left.txt")),
        "right": (np.loadtxt(scenario / "calib_ori_offset.txt"),
                  np.loadtxt(scenario / "calib_offset.txt")),
    }


def world_joints_for_frame(frame_dir: Path, side: str, calib: dict) -> np.ndarray | None:
    """(21,3) world-frame hand joints (m) via the official read_pose_data chain."""
    pose_path = frame_dir / ("pose_2.txt" if side == "left" else "pose_3.txt")
    joint_path = frame_dir / f"{side}_hand_joint.txt"
    ori_path = frame_dir / f"{side}_hand_joint_ori.txt"
    if not (pose_path.is_file() and joint_path.is_file() and ori_path.is_file()):
        return None
    pose_h = np.loadtxt(pose_path)
    joints = np.loadtxt(joint_path)
    quat = np.loadtxt(ori_path)[0]
    if pose_h.shape != (4, 4) or joints.shape != (21, 3) or not (
            np.isfinite(pose_h).all() and np.isfinite(joints).all() and np.isfinite(quat).all()):
        return None
    pose_h = pose_h @ (_BETWEEN_CAM_2 if side == "left" else _BETWEEN_CAM_3)
    j = joints - joints[0]
    j = (Rotation.from_quat(quat).as_matrix().T @ j.T).T
    j[:, -1] *= -1.0
    j = j @ _AX_Y.T @ _AX_X.T @ _AX_Z.T
    ori_off, trans_off = calib[side]
    j = j @ Rotation.from_euler("xyz", ori_off).as_matrix().T
    j = j + trans_off
    jh = np.hstack([j, np.ones((21, 1))])
    return (jh @ pose_h.T)[:, :3]


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


def convert_clip(clip: dict, args) -> dict:
    clip_id = clip["clip_id"]
    frames_root, outputs_root = Path(args.frames_root), Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_path = frames_root / f"{clip_id}.tar"
    result = {"clip_id": clip_id, "scenario": clip["scenario_name"], "task": clip["task"], "status": "ok"}

    done_marker = get_stage_done_marker(seq_folder, "infiller")
    if args.resume and tar_path.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        result["status"] = "skipped"
        return result
    try:
        scenario = Path(clip["scenario"])
        calib = _load_calib(scenario)
        frame_dirs = []
        for fid in range(clip["start"], clip["end"] + 1):
            fdir = scenario / f"frame_{fid}"
            if (fdir / "color_image.jpg").is_file() and (fdir / "pose.txt").is_file():
                frame_dirs.append(fdir)
        T = len(frame_dirs)
        result["frames"] = int(T)
        if T < args.min_frames:
            raise ValueError(f"too few frames with color+pose: {T}")

        trans = np.zeros((2, T, 3), dtype=np.float32)
        rot = np.zeros((2, T, 3), dtype=np.float32)
        pose45 = np.zeros((2, T, 45), dtype=np.float32)
        betas = np.zeros((2, T, 10), dtype=np.float32)
        valid = np.zeros((2, T), dtype=np.float32)
        fit_stats = {}
        for side, hand_index in (("left", 0), ("right", 1)):
            target_list, frame_pos = [], []
            for t, fdir in enumerate(frame_dirs):
                wj = world_joints_for_frame(fdir, side, calib)
                if wj is None:
                    continue
                target_list.append(wj)
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
        result["fit_stats"] = fit_stats

        seq_folder.mkdir(parents=True, exist_ok=True)
        tmp_tar = tar_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as writer:
            for t, fdir in enumerate(frame_dirs):
                writer.add(fdir / "color_image.jpg", arcname=f"{clip_id}_f{t:05d}.image.jpg")
        tmp_tar.replace(tar_path)

        joblib.dump([trans, rot, pose45, betas, valid], seq_folder / "world_space_res.pth")

        c2w = np.stack([np.loadtxt(fdir / "pose.txt") @ _BETWEEN_CAM for fdir in frame_dirs])
        if not np.isfinite(c2w).all():
            raise ValueError("non-finite camera pose")
        quat_xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
        traj = np.concatenate([c2w[:, :3, 3], quat_xyzw], axis=1).astype(np.float32)
        focal = 0.5 * (FX + FY)
        slam_dir = seq_folder / "SLAM"
        slam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
            tstamp=np.arange(T, dtype=np.int64),
            traj=traj,
            scale=np.float64(1.0),
            img_focal=np.float64(focal),
            img_center=np.asarray([CX, CY], dtype=np.float64),
        )
        (seq_folder / "est_focal.txt").write_text(f"{focal:.6f}\n")
        tracks_dir = seq_folder / f"tracks_0_{T - 1}"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        (tracks_dir / ".dexcap_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
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
                "adapter": "dexcap_tar",
                "dataset_name": args.source_id,
                "task": clip["task"],
                "scenario": clip["scenario_name"],
                "demo_range": [clip["start"], clip["end"]],
                "native_fps": NATIVE_FPS,
                "camera": "chest_l515",
            },
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                group_id=clip["scenario_name"],
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)
    raw_roots = [Path(p.strip()) for p in args.raw_roots.split(",") if p.strip()]

    clips = discover_clips(raw_roots, args)
    if args.include:
        pattern = re.compile(args.include)
        clips = [c for c in clips if pattern.search(c["clip_id"])]
    if args.limit:
        clips = clips[: args.limit]
    print(f"DexCap: {len(clips)} demo clips selected from {len(raw_roots)} raw roots", flush=True)

    started = time.perf_counter()
    results = []
    for idx, clip in enumerate(clips):
        results.append(convert_clip(clip, args))
        print(f"[{idx + 1}/{len(clips)}] {results[-1]['clip_id']} {results[-1]['status']}"
              + (f" :: {results[-1].get('error', '')[:200]}" if results[-1]["status"] == "failed" else ""), flush=True)

    manifest_count = write_manifest(results, clips, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "raw_roots": [str(p) for p in raw_roots],
        "clips_selected": len(clips),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "native_fps": NATIVE_FPS,
        "conventions": {
            "gt": "EMF-glove 21-joint skeleton (m, world via wrist T265 + per-scenario visual calib) -> MANO torch fit (SHOW3D centroid fitter)",
            "joint_order": "wrist + 4/finger thumb->pinky == MANO target order (identity mapping)",
            "left_hand": "x-mirror fit (diag(-1,1,1) conjugation on aa params)",
            "camera": "c2w = pose.txt @ between_cam (diag(1,-1,-1), +7.6cm); L515 intrinsics fx/fy/cx/cy = 898.20/897.87/657.50/364.31",
            "probe": "projection locks on gloved hands; depth cross-check median 14-25mm (R) / 38-73mm (L)",
            "clips": "clip_marks.json demo segments (hands leave view between demos)",
        },
        "failures": failed[:20],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("DEXCAP_CONVERT_DONE" if not failed else "DEXCAP_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
