#!/usr/bin/env python3
"""Convert HO3D_v3 GT annotations into filter-ready seq_folders + frame tars + manifest.

HO3D_v3 (HOnnotate, CVPR 2020) ships 55 train sequences (right hand + YCB object,
static RGBD cameras, 640x480) with per-frame MANO GT under
``train/<seq>/{rgb/%04d.jpg, meta/%04d.pkl}``. The ``evaluation`` split withholds hand
pose and is excluded. This converter reads straight out of the official
``HO3D_v3.zip`` (random access; no extraction).

Conventions (README + probe-verified vs the dataset's own handJoints3D: 0.7 mm
chamfer / exact wrist):
- ``handPose`` (48,) is absolute axis-angle (mean folded in) fed directly to MANO =>
  rot = handPose[:3], hand_pose45 = handPose[3:] passthrough.
- ``handTrans`` is the smplx transl (passthrough); ``handBeta`` per frame.
- All annotations are in the CAMERA frame using the **OpenGL** convention (hand along
  negative z). world := that frame; the (static) camera extrinsic is the constant flip
  w2c = c2w = diag(1,-1,-1), stored in the SLAM npz as a 180-degree rotation about x.
- Frames whose meta fields are None (HOnnotate optimization failed) get valid=0/zeros;
  frames stay contiguous 0..T-1 so tar/GT indices align.
- Right hand only; left hand valid=0/zeros.

Outputs per clip (identical contract to generate_taco_world_res.py):
- frames_root/<clip_id>.tar (raw jpgs re-tarred as <clip_id>_f%05d.image.jpg)
- outputs_root/<clip_id>/world_space_res.pth [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)] (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz + est_focal.txt
- tracks_0_<T-1>/.ho3d_gt + infiller done marker; manifest JSONL + report JSON
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import sys
import tarfile
import time
import zipfile
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402

# w2c == c2w == diag(1,-1,-1): 180-degree rotation about x (OpenGL cam -> CV pinhole cam)
_FLIP_QUAT_XYZW = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HO3D_v3 GT (train split) to filter-ready artifacts")
    parser.add_argument("--zip_path", required=True, help="Local HO3D_v3.zip")
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--split_dir", default="train", help="Zip folder to ingest (evaluation has no hand GT)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include", default=None, help="Regex on clip_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_id", default="ho3d")
    parser.add_argument("--split", default="train")
    return parser


def discover_sequences(zip_path: str, split_dir: str) -> dict[str, dict]:
    """{seq: {"rgb": {fid: name}, "meta": {fid: name}}} from the zip namelist."""
    seqs: dict[str, dict] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) != 4 or parts[0] != split_dir or not parts[3]:
                continue
            seq, kind, fname = parts[1], parts[2], parts[3]
            entry = seqs.setdefault(seq, {"rgb": {}, "meta": {}})
            stem = fname.split(".")[0]
            if kind == "rgb" and fname.endswith(".jpg") and stem.isdigit():
                entry["rgb"][int(stem)] = name
            elif kind == "meta" and fname.endswith(".pkl") and stem.isdigit():
                entry["meta"][int(stem)] = name
    return seqs


def build_world_res(metas: list[dict | None]) -> tuple[list[np.ndarray], dict | None]:
    """metas: per-frame meta dicts (None-annotated frames tolerated). Returns (payload, camera)."""
    T = len(metas)
    trans = np.zeros((2, T, 3), dtype=np.float32)
    rot = np.zeros((2, T, 3), dtype=np.float32)
    hand_pose = np.zeros((2, T, 45), dtype=np.float32)
    betas = np.zeros((2, T, 10), dtype=np.float32)
    valid = np.zeros((2, T), dtype=np.float32)
    camera = None
    for t, meta in enumerate(metas):
        if meta is None:
            continue
        cam_mat = meta.get("camMat")
        if camera is None and cam_mat is not None:
            K = np.asarray(cam_mat, dtype=np.float64)
            camera = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
        pose = meta.get("handPose")
        if pose is None or meta.get("handTrans") is None or meta.get("handBeta") is None:
            continue
        pose = np.asarray(pose, dtype=np.float64).reshape(48)
        rot[1, t] = pose[:3].astype(np.float32)
        hand_pose[1, t] = pose[3:].astype(np.float32)      # absolute aa45, passthrough
        trans[1, t] = np.asarray(meta["handTrans"], dtype=np.float64).reshape(3).astype(np.float32)
        betas[1, t] = np.asarray(meta["handBeta"], dtype=np.float64).reshape(10).astype(np.float32)
        valid[1, t] = 1.0
    return [trans, rot, hand_pose, betas, valid], camera


def convert_sequence(seq: str, entry: dict, args) -> dict:
    clip_id = f"HO3D_{re.sub(r'[^A-Za-z0-9]+', '_', seq)}"
    frames_root, outputs_root = Path(args.frames_root), Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_path = frames_root / f"{clip_id}.tar"
    result = {"seq": seq, "clip_id": clip_id, "status": "ok"}
    try:
        done_marker = get_stage_done_marker(seq_folder, "infiller")
        if args.resume and tar_path.is_file() and done_marker.exists() \
                and (seq_folder / "world_space_res.pth").is_file():
            result["status"] = "skipped"
            return result

        rgb_ids = sorted(entry["rgb"])
        # contiguous prefix 0..T-1
        T = 0
        while T < len(rgb_ids) and rgb_ids[T] == T:
            T += 1
        if T < 2:
            raise ValueError(f"too few contiguous rgb frames: {len(rgb_ids)}")
        result["frame_counts"] = {"rgb": len(rgb_ids), "meta": len(entry["meta"]), "used": T}

        with zipfile.ZipFile(args.zip_path) as zf:
            metas = []
            for t in range(T):
                name = entry["meta"].get(t)
                metas.append(pickle.loads(zf.read(name)) if name else None)
            payload, camera = build_world_res(metas)
            if camera is None:
                raise ValueError("no annotated frame with camMat")

            seq_folder.mkdir(parents=True, exist_ok=True)
            tmp_tar = tar_path.with_suffix(".tar.tmp")
            with tarfile.open(tmp_tar, "w") as writer:
                for t in range(T):
                    data = zf.read(entry["rgb"][t])
                    info = tarfile.TarInfo(name=f"{clip_id}_f{t:05d}.image.jpg")
                    info.size = len(data)
                    writer.addfile(info, io.BytesIO(data))
            tmp_tar.replace(tar_path)

        joblib.dump(payload, seq_folder / "world_space_res.pth")
        traj = np.zeros((T, 7), dtype=np.float32)
        traj[:, 3:7] = _FLIP_QUAT_XYZW[None, :]  # [tx,ty,tz,qx,qy,qz,qw]; c2w = diag(1,-1,-1)
        slam_dir = seq_folder / "SLAM"
        slam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
            tstamp=np.arange(T, dtype=np.int64),
            traj=traj,
            scale=np.float64(1.0),
            img_focal=np.float64(0.5 * (camera["fx"] + camera["fy"])),
            img_center=np.asarray([camera["cx"], camera["cy"]], dtype=np.float64),
        )
        (seq_folder / "est_focal.txt").write_text(f"{0.5 * (camera['fx'] + camera['fy']):.6f}\n")
        tracks_dir = seq_folder / f"tracks_0_{T - 1}"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        (tracks_dir / ".ho3d_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        result["frames"] = int(T)
        result["valid_frames"] = int(np.asarray(payload[4])[1].sum())
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _convert_star(task):
    seq, entry, args = task
    return convert_sequence(seq, entry, args)


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
        seq = result["seq"]
        descriptor = ClipDescriptor.from_tar_shard(
            clip_id=clip_id,
            clip_name=clip_id,
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / clip_id).resolve()),
            shard_path=str(tar_path.resolve()),
            frame_names=frame_names,
            frame_offsets=frame_offsets,
            extra={"adapter": "ho3d_tar", "dataset_name": args.source_id, "ho3d_seq": seq},
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                # multi-camera captures share the stem (ABF10..ABF14 -> ABF1)
                group_id=seq[:-1] if seq[-1].isdigit() and len(seq) > 1 else seq,
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)

    seqs = discover_sequences(args.zip_path, args.split_dir)
    items = sorted(seqs.items())
    if args.include:
        pattern = re.compile(args.include)
        items = [(s, e) for s, e in items if pattern.search(f"HO3D_{s}")]
    if args.limit:
        items = items[: args.limit]
    print(f"HO3D sequences: {len(seqs)} discovered, {len(items)} selected", flush=True)

    started = time.perf_counter()
    if args.workers <= 1:
        results = [convert_sequence(s, e, args) for s, e in items]
    else:
        with get_context("spawn").Pool(min(args.workers, max(1, len(items)))) as pool:
            results = []
            for idx, result in enumerate(pool.imap_unordered(_convert_star, [(s, e, args) for s, e in items], chunksize=1)):
                results.append(result)
                print(f"[{idx + 1}/{len(items)}] {result['clip_id']} {result['status']}", flush=True)

    manifest_count = write_manifest(results, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "zip_path": str(Path(args.zip_path).resolve()),
        "split_dir": args.split_dir,
        "discovered": len(seqs),
        "selected": len(items),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "conventions": {
            "mano": "handPose absolute aa48 passthrough (rot=aa[:3], hand_pose=aa[3:])",
            "trans": "handTrans == smplx transl passthrough",
            "camera": "static; world := OpenGL camera frame; w2c = c2w = diag(1,-1,-1)",
        },
        "failures": failed[:20],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("HO3D_CONVERT_DONE" if not failed else "HO3D_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
