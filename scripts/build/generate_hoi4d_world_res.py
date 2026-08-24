#!/usr/bin/env python3
"""Convert HOI4D GT annotations into filter-ready seq_folders + frame tars + manifest.

HOI4D (CVPR 2022) is egocentric (head-mounted RGBD) human-object interaction:
~2,970 task sequences laid out ``ZY<cam>/<H>/<C>/<N>/<S>/<s>/<T>`` across 4 camera
rigs. On GCS the pieces are separate zips (all read in-place, no extraction):
- ``HOI4D_release.zip``      — ``.../align_rgb/image.mp4`` (1920x1080 @15fps RGB)
- ``HOI4D_Hand_pose.zip``    — ``Hand_pose/handpose_{right,left}_hand/.../<frame>.pickle``
  with {poseCoeff(48), beta(10), trans(3), kps2D(21,2)} in the CAMERA frame
- ``HOI4D_annotations.zip``  — ``.../3Dseg/output.log`` per-frame camera poses
  (Redwood .log format, camera-to-world)
- ``camera_params.zip``      — per-rig ``intrin.npy`` 3x3 K

Conventions (probe-verified vs the dataset's own kps2D: 1.4 px chamfer):
- poseCoeff is absolute axis-angle: rot = poseCoeff[:3], hand_pose45 = poseCoeff[3:]
  passthrough; ``trans`` is the smplx transl (passthrough) — all in the camera frame.
- World frame := the 3Dseg trajectory frame. Camera-frame MANO params are lifted per
  frame with c2w = (Rc,tc):  rot' = Rc @ R,  transl' = Rc @ (j0 + transl) + tc - j0
  (j0 = rest-pose MANO root joint of beta), which is the exact smplx composition.
- Frames without a hand pickle get valid=0/zeros for that hand. HOI4D hand GT is
  known to be noisy: spot-check overlays before large runs.

Outputs per clip (identical contract to generate_taco_world_res.py):
- frames_root/<clip_id>.tar (<clip_id>_f%05d.image.jpg)
- outputs_root/<clip_id>/world_space_res.pth [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)] (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz + est_focal.txt
- tracks_0_<T-1>/.hoi4d_gt + infiller done marker; manifest JSONL + report JSON
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402

_TASK_RE = re.compile(r"(ZY\d+)/(H\d+)/(C\d+)/(N\d+)/(S\d+)/(s\d+)/(T\d+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HOI4D GT to filter-ready artifacts")
    parser.add_argument("--release_zip", required=True)
    parser.add_argument("--handpose_zip", required=True)
    parser.add_argument("--annotations_zip", required=True)
    parser.add_argument("--camera_params_zip", required=True)
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--work_dir", default=None)
    parser.add_argument("--jpeg_quality", type=int, default=3, help="ffmpeg -q:v")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--include", default=None, help="Regex on clip_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_id", default="hoi4d")
    parser.add_argument("--split", default="train")
    return parser


def _ffmpeg_exe() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _task_key(name: str):
    match = _TASK_RE.search(name)
    return "/".join(match.groups()) if match else None


def discover_tasks(args) -> tuple[list[dict], dict]:
    """Intersect release videos, right-hand pose dirs, and 3Dseg output.log tasks."""
    videos, logs = {}, {}
    hands: dict[str, dict[str, dict[int, str]]] = {}
    with zipfile.ZipFile(args.release_zip) as zf:
        for name in zf.namelist():
            if name.endswith("align_rgb/image.mp4"):
                key = _task_key(name)
                if key:
                    videos[key] = name
    with zipfile.ZipFile(args.annotations_zip) as zf:
        for name in zf.namelist():
            if name.endswith("3Dseg/output.log"):
                key = _task_key(name)
                if key:
                    logs[key] = name
    with zipfile.ZipFile(args.handpose_zip) as zf:
        for name in zf.namelist():
            if not name.endswith(".pickle"):
                continue
            side = "right" if "handpose_right_hand" in name else ("left" if "handpose_left_hand" in name else None)
            key = _task_key(name)
            if side is None or key is None:
                continue
            stem = Path(name).stem
            if not stem.isdigit():
                continue
            hands.setdefault(key, {}).setdefault(side, {})[int(stem)] = name

    tasks, funnel = [], {"videos": len(videos), "logs": len(logs), "hand_tasks": len(hands)}
    for key in sorted(videos):
        if key not in logs or key not in hands:
            continue
        tasks.append({"key": key, "video": videos[key], "log": logs[key], "hands": hands[key]})
    funnel["complete"] = len(tasks)
    return tasks, funnel


def parse_redwood_log(text: str) -> np.ndarray:
    """Redwood .log trajectory -> (T,4,4) camera-to-world."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    mats = []
    for i in range(0, len(lines) - 4, 5):
        rows = [[float(v) for v in lines[i + 1 + r].split()] for r in range(4)]
        mats.append(np.asarray(rows, dtype=np.float64))
    return np.stack(mats, axis=0)


def _mano_root_joint(betas: np.ndarray, side: str) -> np.ndarray:
    from lib.pipeline.hands.mano_runtime import resolve_mano_model_dir

    mano_dir = resolve_mano_model_dir(is_right=(side == "right"))
    pkl = mano_dir / ("MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl")
    with open(pkl, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    v_template = np.asarray(model["v_template"], dtype=np.float64)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
    if side == "left":
        shapedirs = shapedirs.copy()
        shapedirs[:, 0, :] *= -1
    j_regressor = model["J_regressor"]
    j_regressor = np.asarray(j_regressor.todense() if hasattr(j_regressor, "todense") else j_regressor,
                             dtype=np.float64)
    return (j_regressor @ (v_template + shapedirs[:, :, : betas.shape[-1]] @ betas.reshape(-1)))[0]


def build_world_res(hand_pickles: dict, c2w: np.ndarray, T: int, hz: zipfile.ZipFile) -> tuple[list[np.ndarray], dict]:
    trans = np.zeros((2, T, 3), dtype=np.float32)
    rot = np.zeros((2, T, 3), dtype=np.float32)
    pose45 = np.zeros((2, T, 45), dtype=np.float32)
    betas = np.zeros((2, T, 10), dtype=np.float32)
    valid = np.zeros((2, T), dtype=np.float32)
    stats = {}
    j0_cache: dict[tuple, np.ndarray] = {}
    for side, hand_index in (("left", 0), ("right", 1)):
        frames = hand_pickles.get(side) or {}
        used = 0
        for t in range(T):
            name = frames.get(t)
            if name is None:
                continue
            d = pickle.loads(hz.read(name))
            pose = np.asarray(d["poseCoeff"], dtype=np.float64).reshape(48)
            beta = np.asarray(d["beta"], dtype=np.float64).reshape(10)
            tr_cam = np.asarray(d["trans"], dtype=np.float64).reshape(3)
            key = (side, round(float(beta.sum()), 5))
            j0 = j0_cache.get(key)
            if j0 is None:
                j0 = _mano_root_joint(beta, side)
                j0_cache[key] = j0
            Rc, tc = c2w[t, :3, :3], c2w[t, :3, 3]
            R_world = Rc @ Rotation.from_rotvec(pose[:3]).as_matrix()
            rot[hand_index, t] = Rotation.from_matrix(R_world).as_rotvec().astype(np.float32)
            trans[hand_index, t] = (Rc @ (j0 + tr_cam) + tc - j0).astype(np.float32)
            pose45[hand_index, t] = pose[3:].astype(np.float32)
            betas[hand_index, t] = beta.astype(np.float32)
            valid[hand_index, t] = 1.0
            used += 1
        stats[side] = used
    return [trans, rot, pose45, betas, valid], stats


def convert_task(task: dict, args) -> dict:
    key = task["key"]
    clip_id = "HOI4D_" + re.sub(r"[^A-Za-z0-9]+", "_", key)
    frames_root, outputs_root = Path(args.frames_root), Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_path = frames_root / f"{clip_id}.tar"
    result = {"key": key, "clip_id": clip_id, "status": "ok"}
    done_marker = get_stage_done_marker(seq_folder, "infiller")
    if args.resume and tar_path.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        result["status"] = "skipped"
        return result

    work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=args.work_dir))
    try:
        cam_rig = key.split("/")[0]
        with zipfile.ZipFile(args.camera_params_zip) as zf:
            K = np.load(io.BytesIO(zf.read(f"camera_params/{cam_rig}/intrin.npy"))).astype(np.float64)
        with zipfile.ZipFile(args.annotations_zip) as zf:
            c2w = parse_redwood_log(zf.read(task["log"]).decode("utf8"))
        with zipfile.ZipFile(args.release_zip) as zf:
            video_path = work / "video.mp4"
            video_path.write_bytes(zf.read(task["video"]))

        frames_dir = work / "frames"
        frames_dir.mkdir()
        subprocess.run(
            [_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(video_path),
             "-vsync", "0", "-start_number", "0", "-q:v", str(args.jpeg_quality),
             str(frames_dir / f"{clip_id}_f%05d.image.jpg")],
            check=True, capture_output=True)
        frame_files = sorted(frames_dir.glob("*.image.jpg"))
        T = min(len(frame_files), len(c2w))
        result["frame_counts"] = {"video": len(frame_files), "camera_log": len(c2w), "used": T}
        if T < 2:
            raise ValueError(f"too few frames: {result['frame_counts']}")

        with zipfile.ZipFile(args.handpose_zip) as hz:
            payload, hand_stats = build_world_res(task["hands"], c2w, T, hz)
        result["hand_frames"] = hand_stats
        if hand_stats.get("left", 0) + hand_stats.get("right", 0) == 0:
            raise ValueError("no hand pose frames within video/camera range")

        seq_folder.mkdir(parents=True, exist_ok=True)
        tmp_tar = tar_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as writer:
            for frame in frame_files[:T]:
                writer.add(frame, arcname=frame.name)
        tmp_tar.replace(tar_path)

        joblib.dump(payload, seq_folder / "world_space_res.pth")
        quat_xyzw = Rotation.from_matrix(c2w[:T, :3, :3]).as_quat()
        traj = np.concatenate([c2w[:T, :3, 3], quat_xyzw], axis=1).astype(np.float32)
        slam_dir = seq_folder / "SLAM"
        slam_dir.mkdir(parents=True, exist_ok=True)
        focal = 0.5 * (K[0, 0] + K[1, 1])
        np.savez(
            slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
            tstamp=np.arange(T, dtype=np.int64),
            traj=traj,
            scale=np.float64(1.0),
            img_focal=np.float64(focal),
            img_center=np.asarray([K[0, 2], K[1, 2]], dtype=np.float64),
        )
        (seq_folder / "est_focal.txt").write_text(f"{focal:.6f}\n")
        tracks_dir = seq_folder / f"tracks_0_{T - 1}"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        (tracks_dir / ".hoi4d_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        result["frames"] = int(T)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            result["error"] += " :: " + error.stderr.decode("utf8", "replace")[-300:]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def _convert_star(task_args):
    return convert_task(*task_args)


def write_manifest(results: list[dict], args) -> int:
    from lib.pipeline.clips.clip_manifest import write_clip_manifest, ClipManifestRecord
    from lib.pipeline.datasets.descriptors import ClipDescriptor

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
        key = result["key"]
        parts = key.split("/")
        descriptor = ClipDescriptor.from_tar_shard(
            clip_id=clip_id,
            clip_name=clip_id,
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / clip_id).resolve()),
            shard_path=str(tar_path.resolve()),
            frame_names=frame_names,
            frame_offsets=frame_offsets,
            extra={"adapter": "hoi4d_tar", "dataset_name": args.source_id, "hoi4d_task": key,
                   "camera_rig": parts[0]},
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                group_id="/".join(parts[:5]),  # ZY/H/C/N/S = same scene+object
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    tasks, funnel = discover_tasks(args)
    if args.include:
        pattern = re.compile(args.include)
        tasks = [t for t in tasks if pattern.search("HOI4D_" + re.sub(r"[^A-Za-z0-9]+", "_", t["key"]))]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"HOI4D funnel: {funnel}; selected {len(tasks)}", flush=True)

    started = time.perf_counter()
    if args.workers <= 1:
        results = [convert_task(t, args) for t in tasks]
    else:
        with get_context("spawn").Pool(min(args.workers, max(1, len(tasks)))) as pool:
            results = []
            for idx, result in enumerate(pool.imap_unordered(_convert_star, [(t, args) for t in tasks], chunksize=1)):
                results.append(result)
                print(f"[{idx + 1}/{len(tasks)}] {result['clip_id']} {result['status']}", flush=True)

    manifest_count = write_manifest(results, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "funnel": funnel,
        "selected": len(tasks),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "conventions": {
            "mano": "poseCoeff absolute aa48 passthrough; trans == smplx transl (camera frame)",
            "world": "3Dseg output.log c2w; params lifted per frame via smplx composition",
            "camera": "per-rig intrin.npy",
            "quality_note": "HOI4D hand GT is noisy -- spot-check overlays",
        },
        "failures": failed[:20],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("HOI4D_CONVERT_DONE" if not failed else "HOI4D_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
