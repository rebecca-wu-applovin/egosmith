#!/usr/bin/env python3
"""Convert TACO GT annotations into filter-ready seq_folders + frame tars + manifest.

TACO (CVPR 2024) ships ground-truth bimanual MANO hand poses, per-frame egocentric
camera extrinsics/intrinsics, and egocentric RGB videos, laid out as
``<component>/(tool, action, object)/<date>_<id>/...``.

Per sequence this script produces the same artifacts the FPHA GT ingestion produces
(`generate_fpha_world_res.py`), so `scripts/build/filter_manifest_by_quality.py`
runs on TACO with ``--stages infiller``:

- ``frames_root/<clip_id>.tar`` — fpha_tar-style frame tar (`<clip_id>_f%05d.image.jpg`)
- ``outputs_root/<clip_id>/world_space_res.pth`` — joblib [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)]; index 0 = left, 1 = right
- ``outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0.npz`` — c2w traj rows
  [tx,ty,tz,qx,qy,qz,qw] + scale/img_focal/img_center (camera_features.py contract)
- ``outputs_root/<clip_id>/tracks_0_<T>/.taco_gt`` + infiller done marker

Also emits a clip manifest JSONL (one ClipManifestRecord per converted sequence) and
a conversion report JSON including `missing_modality` sequences the filter never sees.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
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

_SEQ_DIR_RE = re.compile(r"^\d{8}_\d+$")
_TRIPLET_RE = re.compile(r"^\((?P<tool>[^,]+),\s*(?P<action>[^,]+),\s*(?P<object>[^)]+)\)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert TACO GT to filter-ready seq_folders + frame tars")
    parser.add_argument("--taco_root", required=True, help="Root containing Hand_Poses/, Egocentric_Camera_Parameters/, Egocentric_RGB_Videos/")
    parser.add_argument("--frames_root", required=True, help="Output dir for per-sequence frame tars")
    parser.add_argument("--outputs_root", required=True, help="Output dir for per-sequence seq_folders")
    parser.add_argument("--manifest_out", required=True, help="Clip manifest JSONL output path")
    parser.add_argument("--report_out", default=None, help="Conversion report JSON output path")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--jpeg_quality", type=int, default=3, help="ffmpeg mjpeg -q:v qscale (2=best, 31=worst); 3 ~ visually lossless")
    parser.add_argument("--include", default=None, help="Optional regex on clip_id to select a subset (smoke tests)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of sequences")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Skip sequences whose tar + done marker already exist")
    parser.add_argument("--source_id", default="taco")
    parser.add_argument("--split", default="train")
    # MANO convention knobs -- resolved by the smoke-test overlay gate before the full run.
    parser.add_argument(
        "--hand_mean_offset",
        choices=("none", "add", "subtract"),
        default="none",
        help="Adjust the 45-d articulation by the MANO hand mean: none = pass poseCoeff[3:] through; add/subtract = convert between flat_hand_mean conventions",
    )
    parser.add_argument(
        "--trans_convention",
        choices=("wrist_root", "transl", "manopth_origin"),
        default="wrist_root",
        help="wrist_root (TACO: manopth center_idx=0) = trans is the world wrist-joint position, converted via transl = trans - j0(beta); transl = pass through unchanged; manopth_origin = origin-rotated manopth convention",
    )
    parser.add_argument(
        "--extrinsic_direction",
        choices=("w2c", "c2w"),
        default="w2c",
        help="Interpretation of egocentric_frame_extrinsic.npy 4x4s",
    )
    return parser


def sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "-", text.strip()).strip("-")
    return token or "x"


def discover_sequences(taco_root: Path) -> tuple[list[dict], list[dict]]:
    """Return (complete sequences, missing_modality records)."""
    hand_root = taco_root / "Hand_Poses"
    cam_root = taco_root / "Egocentric_Camera_Parameters"
    video_root = taco_root / "Egocentric_RGB_Videos"

    seqs: dict[tuple[str, str], dict] = {}
    for component, root, key in (
        ("hand_poses", hand_root, "hand_dir"),
        ("camera", cam_root, "cam_dir"),
        ("video", video_root, "video_dir"),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"TACO component root not found: {root}")
        for triplet_dir in sorted(root.iterdir()):
            if not triplet_dir.is_dir():
                continue
            for seq_dir in sorted(triplet_dir.iterdir()):
                if not seq_dir.is_dir() or not _SEQ_DIR_RE.match(seq_dir.name):
                    continue
                entry = seqs.setdefault(
                    (triplet_dir.name, seq_dir.name),
                    {"triplet": triplet_dir.name, "seq_name": seq_dir.name},
                )
                entry[key] = str(seq_dir)

    complete, missing = [], []
    for (triplet, seq_name), entry in sorted(seqs.items()):
        required = {
            "hand_dir": ("left_hand.pkl", "right_hand.pkl", "left_hand_shape.pkl", "right_hand_shape.pkl"),
            "cam_dir": ("egocentric_frame_extrinsic.npy", "egocentric_intrinsic.txt"),
            "video_dir": ("color.mp4",),
        }
        missing_items = []
        for key, files in required.items():
            base = entry.get(key)
            if base is None:
                missing_items.append(key)
                continue
            for file_name in files:
                if not (Path(base) / file_name).is_file():
                    missing_items.append(f"{key}/{file_name}")
        match = _TRIPLET_RE.match(triplet)
        if match is None:
            missing_items.append("unparseable_triplet")
        if missing_items:
            missing.append({"triplet": triplet, "seq_name": seq_name, "missing": missing_items})
            continue
        tool, action, obj = (sanitize_token(match.group(g)) for g in ("tool", "action", "object"))
        entry["clip_id"] = f"TACO_{tool}_{action}_{obj}_{seq_name}"
        entry["tool"], entry["action"], entry["object"] = tool, action, obj
        complete.append(entry)
    return complete, missing


def load_hand(hand_dir: Path, side: str) -> dict:
    """Load TACO hand pkls (torch-pickled).

    ``<side>_hand.pkl`` is a dict keyed by frame id string ('00001'...) with per-frame
    ``{"hand_pose": (48,), "hand_trans": (3,)}``; sorted keys are positionally aligned
    with video frames (TACO-Instructions hand_pose_loader.py). ``<side>_hand_shape.pkl``
    holds one per-sequence ``{"hand_shape": (10,)}``. hand_pose is absolute axis-angle
    (manopth flat_hand_mean=True); hand_trans is the WORLD position of MANO joint 0
    (manopth center_idx=0).
    """
    with open(hand_dir / f"{side}_hand.pkl", "rb") as f:
        pose = pickle.load(f)
    with open(hand_dir / f"{side}_hand_shape.pkl", "rb") as f:
        shape = pickle.load(f)

    def _np(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    keys = sorted(pose.keys())
    pose_coeff = np.stack([_np(pose[key]["hand_pose"]).reshape(48) for key in keys], axis=0)
    trans = np.stack([_np(pose[key]["hand_trans"]).reshape(3) for key in keys], axis=0)
    beta = _np(shape["hand_shape"]).reshape(10)
    return {"pose_coeff": pose_coeff, "trans": trans, "beta": beta}


def load_camera(cam_dir: Path, extrinsic_direction: str) -> dict:
    extr = np.load(cam_dir / "egocentric_frame_extrinsic.npy").astype(np.float64)
    if extr.ndim != 3 or extr.shape[1:] != (4, 4):
        raise ValueError(f"extrinsic expected (T,4,4), got {extr.shape}")
    intr_text = (cam_dir / "egocentric_intrinsic.txt").read_text().split()
    intr = np.asarray([float(v) for v in intr_text], dtype=np.float64)
    if intr.size != 9:
        raise ValueError(f"intrinsic expected 9 values, got {intr.size}")
    K = intr.reshape(3, 3)
    w2c = extr if extrinsic_direction == "w2c" else np.linalg.inv(extr)
    return {"w2c": w2c, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}


def _mano_root_joint(betas: np.ndarray, side: str) -> np.ndarray:
    """Rest-pose root joint j0(beta) of the SAME MANO model the pipeline uses.

    Mirrors build_mano_models: neutral model per side, and the smplx left-hand
    shapedirs x-sign fix (`mano_left.shapedirs[:, 0, :] *= -1`).
    """
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
    j_regressor = np.asarray(j_regressor.todense() if hasattr(j_regressor, "todense") else j_regressor, dtype=np.float64)
    v_shaped = v_template + shapedirs[:, :, : betas.shape[-1]] @ betas.reshape(-1)
    return (j_regressor @ v_shaped)[0]


def _mano_hand_mean(side: str) -> np.ndarray:
    from lib.pipeline.hands.mano_runtime import resolve_mano_model_dir

    mano_dir = resolve_mano_model_dir(is_right=(side == "right"))
    pkl = mano_dir / ("MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl")
    with open(pkl, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    return np.asarray(model["hands_mean"], dtype=np.float64).reshape(45)


def build_world_res_payload(left: dict, right: dict, T: int, args) -> list[np.ndarray]:
    trans = np.zeros((2, T, 3), dtype=np.float32)
    rot = np.zeros((2, T, 3), dtype=np.float32)
    hand_pose = np.zeros((2, T, 45), dtype=np.float32)
    betas = np.zeros((2, T, 10), dtype=np.float32)
    valid = np.ones((2, T), dtype=np.float32)

    for hand_index, (side, hand) in enumerate((("left", left), ("right", right))):
        pose_coeff = hand["pose_coeff"][:T]
        rot[hand_index] = pose_coeff[:, :3].astype(np.float32)
        articulation = pose_coeff[:, 3:48]
        if args.hand_mean_offset != "none":
            mean = _mano_hand_mean(side)
            articulation = articulation + mean if args.hand_mean_offset == "add" else articulation - mean
        hand_pose[hand_index] = articulation.astype(np.float32)
        beta = hand["beta"]
        betas[hand_index] = (np.tile(beta.reshape(1, 10), (T, 1)) if beta.ndim == 1 else beta[:T]).astype(np.float32)

        hand_trans = hand["trans"][:T].astype(np.float64)
        if args.trans_convention == "wrist_root":
            # TACO (manopth center_idx=0): hand_trans is the WORLD wrist-joint position.
            # smplx-style MANO: root joint lands at j0(beta) + transl  =>  transl = hand_trans - j0
            j0 = _mano_root_joint(np.asarray(beta.reshape(-1)[:10]), side)
            hand_trans = hand_trans - j0[None, :]
        elif args.trans_convention == "manopth_origin":
            # manopth (no center_idx): world = R_g @ x + trans, x in canonical space.
            # smplx: world = R_g @ (x - j0) + j0 + transl  =>  transl = trans + R_g @ j0 - j0
            j0 = _mano_root_joint(np.asarray(beta.reshape(-1)[:10]), side)
            R = Rotation.from_rotvec(pose_coeff[:, :3]).as_matrix()
            hand_trans = hand_trans + (R @ j0) - j0
        trans[hand_index] = hand_trans.astype(np.float32)

    return [trans, rot, hand_pose, betas, valid]


def build_slam_npz(seq_folder: Path, w2c: np.ndarray, cam: dict, T: int) -> None:
    c2w = np.linalg.inv(w2c[:T])
    quat_xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
    traj = np.concatenate([c2w[:, :3, 3], quat_xyzw], axis=1).astype(np.float32)
    slam_dir = seq_folder / "SLAM"
    slam_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        slam_dir / "hawor_slam_w_scale_0.npz",
        tstamp=np.arange(T, dtype=np.int64),
        traj=traj,
        scale=np.float64(1.0),
        img_focal=np.float64(0.5 * (cam["fx"] + cam["fy"])),
        img_center=np.asarray([cam["cx"], cam["cy"]], dtype=np.float64),
    )


def extract_frames_to_tar(video_path: Path, tar_path: Path, clip_id: str, T: int, jpeg_quality: int) -> int:
    """Extract exactly the first T frames into an fpha_tar-style tar; returns frame count written."""
    with tempfile.TemporaryDirectory(dir=tar_path.parent) as tmp_dir:
        out_pattern = str(Path(tmp_dir) / f"{clip_id}_f%05d.image.jpg")
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", str(video_path),
            "-vsync", "0",
            "-frames:v", str(T),
            "-start_number", "0",
            "-q:v", str(jpeg_quality),
            out_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        frames = sorted(Path(tmp_dir).glob("*.image.jpg"))
        if not frames:
            raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
        tmp_tar = tar_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as tar_writer:
            for frame in frames:
                tar_writer.add(frame, arcname=frame.name)
        tmp_tar.replace(tar_path)
        return len(frames)


def count_video_frames(video_path: Path) -> int:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    return int(probe.stdout.strip())


def convert_sequence(entry: dict, args) -> dict:
    clip_id = entry["clip_id"]
    frames_root = Path(args.frames_root)
    outputs_root = Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_path = frames_root / f"{clip_id}.tar"
    done_marker = None

    result = {"clip_id": clip_id, "triplet": entry["triplet"], "seq_name": entry["seq_name"], "status": "ok"}
    try:
        left = load_hand(Path(entry["hand_dir"]), "left")
        right = load_hand(Path(entry["hand_dir"]), "right")
        cam = load_camera(Path(entry["cam_dir"]), args.extrinsic_direction)
        video_path = Path(entry["video_dir"]) / "color.mp4"
        video_frames = count_video_frames(video_path)

        T = min(left["pose_coeff"].shape[0], right["pose_coeff"].shape[0], cam["w2c"].shape[0], video_frames)
        result["frame_counts"] = {
            "left_pose": int(left["pose_coeff"].shape[0]),
            "right_pose": int(right["pose_coeff"].shape[0]),
            "extrinsic": int(cam["w2c"].shape[0]),
            "video": int(video_frames),
            "used": int(T),
        }
        if T < 2:
            raise ValueError(f"too few frames: {result['frame_counts']}")

        done_marker = get_stage_done_marker(seq_folder, "infiller")
        if args.resume and tar_path.is_file() and done_marker.exists():
            result["status"] = "skipped"
            return result

        seq_folder.mkdir(parents=True, exist_ok=True)
        written = extract_frames_to_tar(video_path, tar_path, clip_id, T, args.jpeg_quality)
        if written < T:
            T = written
        payload = build_world_res_payload(left, right, T, args)
        joblib.dump(payload, seq_folder / "world_space_res.pth")
        build_slam_npz(seq_folder, cam["w2c"], cam, T)

        tracks_dir = seq_folder / f"tracks_0_{T}"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        (tracks_dir / ".taco_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        result["frames"] = int(T)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            result["error"] += " :: " + error.stderr.decode("utf8", "replace")[-300:]
    return result


def _convert_star(task):
    entry, args = task
    return convert_sequence(entry, args)


def write_manifest(results: list[dict], entries: list[dict], args) -> int:
    from lib.pipeline.clips.clip_manifest import write_clip_manifest
    from lib.pipeline.clips.clip_manifest import ClipManifestRecord
    from lib.pipeline.datasets.descriptors import ClipDescriptor

    by_id = {entry["clip_id"]: entry for entry in entries}
    records = []
    for result in results:
        if result["status"] not in ("ok", "skipped"):
            continue
        entry = by_id[result["clip_id"]]
        tar_path = Path(args.frames_root) / f"{result['clip_id']}.tar"
        frame_names, frame_offsets = [], []
        with tarfile.open(tar_path, "r") as tar_reader:
            members = [m for m in tar_reader if m.isfile() and m.name.endswith(".image.jpg")]
        members.sort(key=lambda m: m.name)
        for member in members:
            frame_names.append(member.name)
            frame_offsets.append([int(member.offset_data), int(member.size)])
        descriptor = ClipDescriptor.from_tar_shard(
            clip_id=result["clip_id"],
            clip_name=result["clip_id"],
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / result["clip_id"]).resolve()),
            shard_path=str(tar_path.resolve()),
            frame_names=frame_names,
            frame_offsets=frame_offsets,
            extra={
                "adapter": "taco_tar",
                "dataset_name": args.source_id,
                "triplet": entry["triplet"],
                "tool": entry["tool"],
                "action": entry["action"],
                "object": entry["object"],
                "taco_seq_name": entry["seq_name"],
            },
        )
        records.append(
            ClipManifestRecord(
                clip_id=result["clip_id"],
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                group_id=entry["triplet"],
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    taco_root = Path(args.taco_root)
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)

    entries, missing = discover_sequences(taco_root)
    if args.include:
        pattern = re.compile(args.include)
        entries = [entry for entry in entries if pattern.search(entry["clip_id"])]
    if args.limit:
        entries = entries[: args.limit]
    print(f"Sequences: {len(entries)} complete, {len(missing)} missing modality", flush=True)

    started = time.perf_counter()
    if args.workers <= 1:
        results = [convert_sequence(entry, args) for entry in entries]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = []
            for idx, result in enumerate(pool.imap_unordered(_convert_star, [(entry, args) for entry in entries], chunksize=1)):
                results.append(result)
                if (idx + 1) % 25 == 0 or (idx + 1) == len(entries):
                    print(f"[{idx + 1}/{len(entries)}] converted (last: {result['clip_id']} {result['status']})", flush=True)

    manifest_count = write_manifest(results, entries, args)

    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "taco_root": str(taco_root.resolve()),
        "total_discovered": len(entries) + len(missing),
        "complete": len(entries),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "missing_modality": missing,
        "failures": failed,
        "conventions": {
            "hand_mean_offset": args.hand_mean_offset,
            "trans_convention": args.trans_convention,
            "extrinsic_direction": args.extrinsic_direction,
        },
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    summary = {k: v for k, v in report.items() if k not in ("missing_modality", "failures")}
    summary["failures_preview"] = failed[:8]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("TACO_CONVERT_DONE" if not failed else "TACO_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
