#!/usr/bin/env python3
"""Convert OakInk-v2 GT annotations into filter-ready seq_folders + frame tars + manifest.

OakInk-v2 (CVPR 2024) ships per-sequence pickles with ground-truth bimanual MANO hand
poses (quaternion form), multi-view camera intrinsics/extrinsics (3 allocentric + 1
egocentric), and multi-view PNG image streams. This script ingests the **egocentric**
camera view (the direct analog of TACO's egocentric ingestion) and produces the same
artifacts the FPHA/TACO GT ingestion produces, so
`scripts/build/filter_manifest_by_quality.py --stages infiller` runs unchanged.

Because the full image set is ~2 TB, this converter **streams**: each worker downloads
one sequence's anno pkl + 5 GB image tar from GCS to a temp dir, extracts only the
egocentric frames into a compact JPEG frame tar, writes the GT sidecars, then deletes
the source tar. Durable footprint is just the JPEG frame tars + small sidecars.

Per sequence it writes:
- ``frames_root/<clip_id>.tar`` — fpha_tar-style tar (`<clip_id>_f%05d.image.jpg`),
  frames remapped to contiguous 0..T-1 (OakInk frame ids are non-contiguous).
- ``outputs_root/<clip_id>/world_space_res.pth`` — joblib [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)]; index 0 = left, 1 = right.
- ``outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0.npz`` — c2w traj rows + intrinsics.
- ``outputs_root/<clip_id>/tracks_0_<T>/.oakink_gt`` + infiller done marker.

MANO conventions (OakInk2 toolkit seg_3d.py): pose_coeffs are per-joint unit quaternions
(w,x,y,z), 16 joints (joint0 = global); manotorch center_idx=0 so tsl is the world
wrist-joint position; cam_extr is world->camera.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402

GCS_ROOT = "gs://foundational-research/hoi-dataset/OakInk-v2"
EGOCENTRIC = "egocentric"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream-convert OakInk-v2 GT (egocentric) to filter-ready artifacts")
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--work_dir", default="/root/oakink/_work", help="Scratch dir for per-seq downloads (deleted after each seq)")
    parser.add_argument("--gcs_root", default=GCS_ROOT)
    parser.add_argument("--camera", default=EGOCENTRIC, help="Camera layout name to ingest")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--jpeg_quality", type=int, default=90, help="PIL JPEG quality (1-95) for extracted frames")
    parser.add_argument("--seq_list", default=None, help="Optional file with one seq_token per line (default: list all from GCS)")
    parser.add_argument("--include", default=None, help="Regex on seq_token to select a subset (smoke tests)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_id", default="oakink_v2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--trans_convention", choices=("wrist_root", "transl"), default="wrist_root",
                        help="wrist_root (manotorch center_idx=0): transl = tsl - j0(beta)")
    parser.add_argument("--quat_order", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument("--extrinsic_direction", choices=("w2c", "c2w"), default="w2c")
    parser.add_argument("--keep_local_anno_dir", default=None, help="If set, read anno pkls from here instead of GCS")
    parser.add_argument("--keep_local_data_dir", default=None, help="If set, read data tars from here instead of GCS")
    return parser


def list_seq_tokens(gcs_root: str) -> list[str]:
    out = subprocess.run(["gsutil", "ls", f"{gcs_root}/anno_preview/"], check=True, capture_output=True, text=True)
    tokens = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.endswith(".pkl"):
            tokens.append(Path(line).name[: -len(".pkl")])
    return sorted(tokens)


def clip_id_from_token(seq_token: str) -> str:
    return "OAKINK_" + re.sub(r"[^A-Za-z0-9]+", "_", seq_token).strip("_")


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_to_axis_angle(quat: np.ndarray, quat_order: str) -> np.ndarray:
    """(...,4) unit quaternions -> (...,3) axis-angle. Input order per quat_order."""
    q = quat.reshape(-1, 4).astype(np.float64)
    if quat_order == "wxyz":
        q = q[:, [1, 2, 3, 0]]  # -> scipy xyzw
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    return Rotation.from_quat(q).as_rotvec().astype(np.float32)


def _mano_root_joint(betas: np.ndarray, side: str):
    """Rest-pose root joint j0(beta) of the pipeline's MANO model (with left x-sign fix)."""
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


def build_world_res(anno: dict, frame_ids: list[int], args) -> list[np.ndarray]:
    T = len(frame_ids)
    trans = np.zeros((2, T, 3), dtype=np.float32)
    rot = np.zeros((2, T, 3), dtype=np.float32)
    hand_pose = np.zeros((2, T, 45), dtype=np.float32)
    betas = np.zeros((2, T, 10), dtype=np.float32)
    valid = np.ones((2, T), dtype=np.float32)

    raw_mano = anno["raw_mano"]
    j0_cache: dict[tuple, np.ndarray] = {}
    for hand_index, prefix in ((0, "lh"), (1, "rh")):
        side = "left" if prefix == "lh" else "right"
        for t, fid in enumerate(frame_ids):
            entry = raw_mano[fid]
            pose_q = _np(entry[f"{prefix}__pose_coeffs"]).reshape(16, 4)
            aa = _quat_to_axis_angle(pose_q, args.quat_order)  # (16,3)
            rot[hand_index, t] = aa[0]
            hand_pose[hand_index, t] = aa[1:16].reshape(45)
            beta = _np(entry[f"{prefix}__betas"]).reshape(-1)[:10].astype(np.float64)
            betas[hand_index, t] = beta.astype(np.float32)
            tsl = _np(entry[f"{prefix}__tsl"]).reshape(3).astype(np.float64)
            if args.trans_convention == "wrist_root":
                key = (side, round(float(beta.sum()), 4))
                j0 = j0_cache.get(key)
                if j0 is None:
                    j0 = _mano_root_joint(beta, side)
                    j0_cache[key] = j0
                tsl = tsl - j0
            trans[hand_index, t] = tsl.astype(np.float32)
    return [trans, rot, hand_pose, betas, valid]


def build_slam_npz(seq_folder: Path, anno: dict, frame_ids: list[int], camera: str, args) -> None:
    cam_extr = anno["cam_extr"][camera]
    cam_intr = anno["cam_intr"][camera]
    T = len(frame_ids)
    w2c = np.stack([_np(cam_extr[fid]).astype(np.float64).reshape(4, 4) for fid in frame_ids], axis=0)
    if args.extrinsic_direction == "c2w":
        w2c = np.linalg.inv(w2c)
    c2w = np.linalg.inv(w2c)
    quat_xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
    traj = np.concatenate([c2w[:, :3, 3], quat_xyzw], axis=1).astype(np.float32)
    K = _np(cam_intr[frame_ids[0]]).astype(np.float64).reshape(3, 3)
    slam_dir = seq_folder / "SLAM"
    slam_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        slam_dir / "hawor_slam_w_scale_0.npz",
        tstamp=np.arange(T, dtype=np.int64),
        traj=traj,
        scale=np.float64(1.0),
        img_focal=np.float64(0.5 * (K[0, 0] + K[1, 1])),
        img_center=np.asarray([K[0, 2], K[1, 2]], dtype=np.float64),
    )


def extract_frames_to_tar(src_tar: Path, seq_token: str, serial: str, frame_ids: list[int],
                          tar_path: Path, clip_id: str, jpeg_quality: int) -> int:
    """Re-encode the egocentric PNGs (named <serial>/<fid:06d>.png) into a contiguous JPEG tar."""
    wanted = {f"{seq_token}/{serial}/{fid:06d}.png": t for t, fid in enumerate(frame_ids)}
    encoded: dict[int, bytes] = {}
    with tarfile.open(src_tar, "r") as reader:
        for member in reader:
            if not member.isfile():
                continue
            t = wanted.get(member.name) or wanted.get(Path(*Path(member.name).parts[-3:]).as_posix())
            if t is None:
                continue
            payload = reader.extractfile(member).read()
            with Image.open(io.BytesIO(payload)) as image:
                image = image.convert("RGB")
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=jpeg_quality)
                encoded[t] = buf.getvalue()

    if not encoded:
        raise RuntimeError(f"No egocentric frames matched in {src_tar} for serial {serial}")
    contiguous = sorted(encoded)
    tmp_tar = tar_path.with_suffix(".tar.tmp")
    with tarfile.open(tmp_tar, "w") as writer:
        for t in contiguous:
            name = f"{clip_id}_f{t:05d}.image.jpg"
            info = tarfile.TarInfo(name=name)
            info.size = len(encoded[t])
            writer.addfile(info, io.BytesIO(encoded[t]))
    tmp_tar.replace(tar_path)
    return len(contiguous)


def _gsutil_cp(src: str, dst: Path) -> None:
    subprocess.run(["gsutil", "-q", "cp", src, str(dst)], check=True, capture_output=True)


def convert_sequence(seq_token: str, args) -> dict:
    clip_id = clip_id_from_token(seq_token)
    frames_root = Path(args.frames_root)
    outputs_root = Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_path = frames_root / f"{clip_id}.tar"
    result = {"seq_token": seq_token, "clip_id": clip_id, "status": "ok"}

    done_marker = get_stage_done_marker(seq_folder, "infiller")
    if args.resume and tar_path.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        result["status"] = "skipped"
        return result

    work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=args.work_dir))
    local_tar = work / "data.tar"
    local_anno = work / "anno.pkl"
    try:
        if args.keep_local_anno_dir:
            local_anno = Path(args.keep_local_anno_dir) / f"{seq_token}.pkl"
        else:
            _gsutil_cp(f"{args.gcs_root}/anno_preview/{seq_token}.pkl", local_anno)
        with open(local_anno, "rb") as f:
            anno = pickle.load(f)

        cam_def = anno["cam_def"]
        serial = next((s for s, name in cam_def.items() if name == args.camera), None)
        if serial is None:
            raise ValueError(f"camera {args.camera} not in cam_def {cam_def}")
        if args.camera not in anno.get("cam_selection", []):
            result["note"] = f"{args.camera} not in cam_selection {anno.get('cam_selection')}"

        # egocentric annotated frames: intersection of image frame ids and pose availability
        cam_frame_ids = sorted(int(f) for f in anno["cam_extr"][args.camera].keys())
        raw_mano = anno["raw_mano"]
        frame_ids = [f for f in cam_frame_ids if f in raw_mano and f in anno["cam_intr"][args.camera]]
        result["frame_counts"] = {"cam_frames": len(cam_frame_ids), "usable": len(frame_ids)}
        if len(frame_ids) < 2:
            raise ValueError(f"too few usable frames: {result['frame_counts']}")

        if args.keep_local_data_dir:
            local_tar = Path(args.keep_local_data_dir) / f"{seq_token}.tar"
        else:
            _gsutil_cp(f"{args.gcs_root}/data/{seq_token}.tar", local_tar)

        seq_folder.mkdir(parents=True, exist_ok=True)
        written = extract_frames_to_tar(local_tar, seq_token, serial, frame_ids, tar_path, clip_id, args.jpeg_quality)
        frame_ids = frame_ids[:written]
        T = len(frame_ids)

        payload = build_world_res(anno, frame_ids, args)
        joblib.dump(payload, seq_folder / "world_space_res.pth")
        build_slam_npz(seq_folder, anno, frame_ids, args.camera, args)

        tracks_dir = seq_folder / f"tracks_0_{T}"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        (tracks_dir / ".oakink_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        result["frames"] = int(T)
        result["scene"] = seq_token.split("__", 1)[0]
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            result["error"] += " :: " + error.stderr.decode("utf8", "replace")[-300:]
    finally:
        import shutil

        if not args.keep_local_data_dir and local_tar.exists():
            local_tar.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    return result


def _convert_star(task):
    seq_token, args = task
    return convert_sequence(seq_token, args)


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
            members = [m for m in reader if m.isfile() and m.name.endswith(".image.jpg")]
        members.sort(key=lambda m: m.name)
        for member in members:
            frame_names.append(member.name)
            frame_offsets.append([int(member.offset_data), int(member.size)])
        seq_token = result["seq_token"]
        descriptor = ClipDescriptor.from_tar_shard(
            clip_id=clip_id,
            clip_name=clip_id,
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / clip_id).resolve()),
            shard_path=str(tar_path.resolve()),
            frame_names=frame_names,
            frame_offsets=frame_offsets,
            extra={
                "adapter": "oakink_tar",
                "dataset_name": args.source_id,
                "camera": args.camera,
                "seq_token": seq_token,
                "scene": seq_token.split("__", 1)[0],
            },
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                group_id=seq_token.split("__", 1)[0],
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    if args.seq_list:
        tokens = [ln.strip() for ln in Path(args.seq_list).read_text().splitlines() if ln.strip()]
    else:
        tokens = list_seq_tokens(args.gcs_root)
    if args.include:
        pattern = re.compile(args.include)
        tokens = [tok for tok in tokens if pattern.search(tok)]
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"OakInk sequences to convert: {len(tokens)}", flush=True)

    started = time.perf_counter()
    if args.workers <= 1:
        results = []
        for idx, tok in enumerate(tokens):
            results.append(convert_sequence(tok, args))
            if (idx + 1) % 10 == 0 or (idx + 1) == len(tokens):
                print(f"[{idx + 1}/{len(tokens)}] {results[-1]['clip_id']} {results[-1]['status']}", flush=True)
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = []
            for idx, result in enumerate(pool.imap_unordered(_convert_star, [(tok, args) for tok in tokens], chunksize=1)):
                results.append(result)
                if (idx + 1) % 10 == 0 or (idx + 1) == len(tokens):
                    print(f"[{idx + 1}/{len(tokens)}] {result['clip_id']} {result['status']}", flush=True)

    manifest_count = write_manifest(results, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "gcs_root": args.gcs_root,
        "camera": args.camera,
        "total": len(tokens),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "failures": failed,
        "conventions": {
            "trans_convention": args.trans_convention,
            "quat_order": args.quat_order,
            "extrinsic_direction": args.extrinsic_direction,
        },
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("OAKINK_CONVERT_DONE" if not failed else "OAKINK_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
