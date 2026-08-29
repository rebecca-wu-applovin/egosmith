#!/usr/bin/env python3
"""Convert DexYCB GT annotations into filter-ready seq_folders + frame tars + manifest.

DexYCB (CVPR 2021) ships 1,000 sequences (10 subjects x 100) of single-hand grasping,
captured by 8 static, calibrated RealSense cameras at 640x480/30fps, with per-frame
MANO GT. On GCS the data lives as one tar.gz per subject
(``gs://foundational-research/hoi-dataset/DexYCB/<capture>-subject-NN.tar.gz``) laid out
``<subject>/<seq>/<serial>/{color_%06d.jpg, labels_%06d.npz, aligned_depth...}`` +
``<subject>/<seq>/meta.yml``; ``calibration.tar.gz`` holds per-camera intrinsics and
per-subject MANO betas.

This converter **streams** each subject tar.gz (no full extraction): it buffers one
sequence at a time (color jpgs + labels for the selected cameras only, depth skipped)
and emits ONE CLIP PER (sequence, camera). Cameras are static, so world := that
camera's frame and the SLAM trajectory is identity.

MANO conventions (dex-ycb-toolkit): ``pose_m`` (1,51) per labels npz is
[global_orient aa(3), pca(45), trans(3)] in the CAMERA frame, manopth
``ManoLayer(flat_hand_mean=False, ncomps=45, use_pca=True)`` => full articulation is
``hands_mean + pca @ hands_components``; trans is the smplx transl (passthrough).
Probe-verified vs the dataset's own joint_3d: 0.7 mm chamfer / exact wrist.
Frames where pose_m is all-zero (hand untracked) get valid=0. Only one hand per
sequence (``mano_sides``); the missing hand is valid=0/zeros.

Outputs per clip (identical contract to generate_taco_world_res.py):
- frames_root/<clip_id>.tar        (<clip_id>_f%05d.image.jpg, raw jpg bytes re-tarred)
- outputs_root/<clip_id>/world_space_res.pth  [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)]  (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz  (identity c2w traj)
- outputs_root/<clip_id>/est_focal.txt, tracks_0_<T-1>/.dexycb_gt, infiller done marker
- manifest JSONL (ClipManifestRecord) + conversion report JSON
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
import time
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402

GCS_ROOT = "gs://foundational-research/hoi-dataset/DexYCB"
SUBJECTS = [
    "20200709-subject-01", "20200813-subject-02", "20200820-subject-03",
    "20200903-subject-04", "20200908-subject-05", "20200918-subject-06",
    "20200928-subject-07", "20201002-subject-08", "20201015-subject-09",
    "20201022-subject-10",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream-convert DexYCB GT to filter-ready artifacts")
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--calibration_dir", required=True, help="Extracted calibration/ dir (intrinsics + mano ymls)")
    parser.add_argument("--gcs_root", default=GCS_ROOT)
    parser.add_argument("--local_tar_dir", default=None, help="Read <subject>.tar.gz from here instead of GCS")
    parser.add_argument("--subjects", default=",".join(SUBJECTS), help="Comma-separated subject archive stems")
    parser.add_argument("--cameras", default=None, help="Comma-separated camera serials (default: all 8 in meta.yml)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel subject streams (max = #subjects)")
    parser.add_argument("--include", default=None, help="Regex on clip_id")
    parser.add_argument("--limit", type=int, default=None, help="Cap on clips per subject stream (smoke)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_id", default="dexycb")
    parser.add_argument("--split", default="train")
    return parser


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that reads the `!!python/tuple` tags DexYCB calibration ymls contain."""


_TolerantLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: list(loader.construct_sequence(node)),
)


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_TolerantLoader)


def _load_pca(side: str) -> tuple[np.ndarray, np.ndarray]:
    """(hands_components(45,45), hands_mean(45)) of the pipeline's MANO model."""
    from lib.pipeline.hands.mano_runtime import resolve_mano_model_dir

    mano_dir = resolve_mano_model_dir(is_right=(side == "right"))
    pkl = mano_dir / ("MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl")
    with open(pkl, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    return (
        np.asarray(model["hands_components"], dtype=np.float64),
        np.asarray(model["hands_mean"], dtype=np.float64).reshape(45),
    )


class Calibration:
    def __init__(self, calibration_dir: Path):
        self.root = Path(calibration_dir)
        self._intr_cache: dict[str, dict] = {}
        self._betas_cache: dict[str, np.ndarray] = {}

    def intrinsics(self, serial: str) -> dict:
        if serial not in self._intr_cache:
            data = _load_yaml(self.root / "intrinsics" / f"{serial}_640x480.yml")["color"]
            self._intr_cache[serial] = {"fx": float(data["fx"]), "fy": float(data["fy"]),
                                        "cx": float(data["ppx"]), "cy": float(data["ppy"])}
        return self._intr_cache[serial]

    def betas(self, mano_calib_id: str) -> np.ndarray:
        if mano_calib_id not in self._betas_cache:
            data = _load_yaml(self.root / f"mano_{mano_calib_id}" / "mano.yml")
            self._betas_cache[mano_calib_id] = np.asarray(data["betas"], dtype=np.float64).reshape(10)
        return self._betas_cache[mano_calib_id]


def build_world_res(pose_m: np.ndarray, side: str, betas10: np.ndarray, pca: dict) -> list[np.ndarray]:
    """pose_m (T,51) camera-frame -> world_space_res payload (world := camera frame)."""
    T = pose_m.shape[0]
    trans = np.zeros((2, T, 3), dtype=np.float32)
    rot = np.zeros((2, T, 3), dtype=np.float32)
    hand_pose = np.zeros((2, T, 45), dtype=np.float32)
    betas = np.zeros((2, T, 10), dtype=np.float32)
    valid = np.zeros((2, T), dtype=np.float32)

    hand_index = 1 if side == "right" else 0
    comps, mean = pca[side]
    present = np.abs(pose_m).sum(axis=1) > 0
    aa45 = mean[None, :] + pose_m[:, 3:48] @ comps  # (T,45) manopth use_pca=True, flat_hand_mean=False
    rot[hand_index] = pose_m[:, :3].astype(np.float32)
    hand_pose[hand_index] = aa45.astype(np.float32)
    trans[hand_index] = pose_m[:, 48:51].astype(np.float32)  # manopth trans == smplx transl
    betas[hand_index] = np.tile(betas10.reshape(1, 10), (T, 1)).astype(np.float32)
    valid[hand_index] = present.astype(np.float32)
    # zero out params on untracked frames so downstream never consumes garbage
    for arr in (trans, rot, hand_pose):
        arr[hand_index][~present] = 0.0
    return [trans, rot, hand_pose, betas, valid]


def write_clip(clip_id: str, jpgs: list[bytes], pose_m: np.ndarray, side: str, betas10: np.ndarray,
               intr: dict, pca: dict, frames_root: Path, outputs_root: Path) -> int:
    from scipy.spatial.transform import Rotation  # noqa: F401  (parity with templates)

    T = len(jpgs)
    seq_folder = outputs_root / clip_id
    seq_folder.mkdir(parents=True, exist_ok=True)
    tar_path = frames_root / f"{clip_id}.tar"

    tmp_tar = tar_path.with_suffix(".tar.tmp")
    with tarfile.open(tmp_tar, "w") as writer:
        for t, payload in enumerate(jpgs):
            info = tarfile.TarInfo(name=f"{clip_id}_f{t:05d}.image.jpg")
            info.size = len(payload)
            writer.addfile(info, io.BytesIO(payload))
    tmp_tar.replace(tar_path)

    joblib.dump(build_world_res(pose_m[:T], side, betas10, pca), seq_folder / "world_space_res.pth")

    # static camera; world := camera frame => identity c2w for every frame
    traj = np.zeros((T, 7), dtype=np.float32)
    traj[:, 6] = 1.0  # qw (xyzw order: [tx,ty,tz,qx,qy,qz,qw])
    slam_dir = seq_folder / "SLAM"
    slam_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
        tstamp=np.arange(T, dtype=np.int64),
        traj=traj,
        scale=np.float64(1.0),
        img_focal=np.float64(0.5 * (intr["fx"] + intr["fy"])),
        img_center=np.asarray([intr["cx"], intr["cy"]], dtype=np.float64),
    )
    (seq_folder / "est_focal.txt").write_text(f"{0.5 * (intr['fx'] + intr['fy']):.6f}\n")

    tracks_dir = seq_folder / f"tracks_0_{T - 1}"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    (tracks_dir / ".dexycb_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
    marker = get_stage_done_marker(seq_folder, "infiller")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return T


def process_sequence(subject: str, seq_name: str, buf: dict, meta_bytes: bytes, args, calib: Calibration,
                     pca: dict, emitted: list[dict]) -> list[dict]:
    """buf: {serial: {frame_idx: {"jpg": bytes, "label": bytes}}}"""
    meta = yaml.safe_load(meta_bytes.decode("utf8"))
    serials = [s for s in meta["serials"] if (not args.cameras) or s in args.cameras]
    side = meta["mano_sides"][0]
    betas10 = calib.betas(meta["mano_calib"][0])
    num_frames = int(meta["num_frames"])
    include = re.compile(args.include) if args.include else None

    results = []
    for serial in serials:
        clip_id = f"DEXYCB_{subject.replace('-', '_')}_{seq_name}_{serial}"
        if include and not include.search(clip_id):
            continue
        result = {"clip_id": clip_id, "subject": subject, "seq": seq_name, "serial": serial, "status": "ok"}
        try:
            seq_folder = Path(args.outputs_root) / clip_id
            tar_path = Path(args.frames_root) / f"{clip_id}.tar"
            if args.resume and tar_path.is_file() and get_stage_done_marker(seq_folder, "infiller").exists() \
                    and (seq_folder / "world_space_res.pth").is_file():
                result["status"] = "skipped"
                results.append(result)
                continue
            cam = buf.get(serial, {})
            frame_ids = sorted(t for t in cam if "jpg" in cam[t] and "label" in cam[t])
            # frames must be contiguous 0..T-1
            T = 0
            while T < len(frame_ids) and frame_ids[T] == T:
                T += 1
            if T < 2:
                raise ValueError(f"too few contiguous frames: {len(frame_ids)} present, meta says {num_frames}")
            pose_m = np.stack([
                np.load(io.BytesIO(cam[t]["label"]))["pose_m"][0].astype(np.float64) for t in range(T)
            ], axis=0)
            jpgs = [cam[t]["jpg"] for t in range(T)]
            frames = write_clip(clip_id, jpgs, pose_m, side, betas10, calib.intrinsics(serial),
                                pca, Path(args.frames_root), Path(args.outputs_root))
            result["frames"] = int(frames)
            result["side"] = side
            result["valid_frames"] = int((np.abs(pose_m[:frames]).sum(axis=1) > 0).sum())
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
        results.append(result)
        emitted.append(result)
        if args.limit and len(emitted) >= args.limit:
            break
    return results


class _LimitReached(Exception):
    pass


def stream_subject(subject: str, args) -> list[dict]:
    calib = Calibration(args.calibration_dir)
    pca = {"left": _load_pca("left"), "right": _load_pca("right")}
    cameras = set(args.cameras) if args.cameras else None

    if args.local_tar_dir:
        src = Path(args.local_tar_dir) / f"{subject}.tar.gz"
        proc = None
        stream = tarfile.open(src, mode="r|gz")
    else:
        proc = subprocess.Popen(["gcloud", "storage", "cat", f"{args.gcs_root}/{subject}.tar.gz"],
                                stdout=subprocess.PIPE)
        stream = tarfile.open(fileobj=proc.stdout, mode="r|gz")

    results: list[dict] = []
    emitted: list[dict] = []
    current_seq = None
    buf: dict = {}
    meta_bytes = None
    file_re = re.compile(r"^(color|labels)_(\d{6})\.(jpg|npz)$")
    try:
        for member in stream:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 3:
                continue
            seq_name = parts[1]
            if current_seq is not None and seq_name != current_seq:
                if meta_bytes is not None:
                    results.extend(process_sequence(subject, current_seq, buf, meta_bytes, args, calib, pca, emitted))
                    if args.limit and len(emitted) >= args.limit:
                        raise _LimitReached
                buf, meta_bytes = {}, None
            current_seq = seq_name
            if parts[2] == "meta.yml":
                meta_bytes = stream.extractfile(member).read()
                continue
            if len(parts) != 4:
                continue
            serial = parts[2]
            if cameras is not None and serial not in cameras:
                continue
            match = file_re.match(parts[3])
            if match is None:
                continue  # depth etc.
            kind, frame_idx = match.group(1), int(match.group(2))
            slot = buf.setdefault(serial, {}).setdefault(frame_idx, {})
            slot["jpg" if kind == "color" else "label"] = stream.extractfile(member).read()
        if current_seq is not None and meta_bytes is not None:
            results.extend(process_sequence(subject, current_seq, buf, meta_bytes, args, calib, pca, emitted))
    except _LimitReached:
        pass
    finally:
        stream.close()
        if proc is not None:
            proc.kill()
            proc.wait()
    return results


def _stream_star(task):
    subject, args = task
    try:
        return stream_subject(subject, args)
    except Exception as error:  # a whole-subject stream failure
        return [{"clip_id": f"DEXYCB_{subject}", "subject": subject, "status": "failed",
                 "error": f"{type(error).__name__}: {error}"}]


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
        descriptor = ClipDescriptor.from_tar_shard(
            clip_id=clip_id,
            clip_name=clip_id,
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / clip_id).resolve()),
            shard_path=str(tar_path.resolve()),
            frame_names=frame_names,
            frame_offsets=frame_offsets,
            extra={
                "adapter": "dexycb_tar",
                "dataset_name": args.source_id,
                "subject": result.get("subject"),
                "dexycb_seq": result.get("seq"),
                "serial": result.get("serial"),
                "mano_side": result.get("side"),
            },
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                group_id=f"{result.get('subject')}/{result.get('seq')}",
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)
    if args.cameras:
        args.cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    started = time.perf_counter()
    if args.workers <= 1 or len(subjects) == 1:
        all_results = []
        for subject in subjects:
            all_results.extend(_stream_star((subject, args)))
            print(f"[{subject}] done ({len(all_results)} clips so far)", flush=True)
    else:
        with get_context("spawn").Pool(min(args.workers, len(subjects))) as pool:
            all_results = []
            for chunk in pool.imap_unordered(_stream_star, [(s, args) for s in subjects], chunksize=1):
                all_results.extend(chunk)
                print(f"[+{len(chunk)}] clips (total {len(all_results)})", flush=True)

    manifest_count = write_manifest(all_results, args)
    failed = [r for r in all_results if r["status"] == "failed"]
    sequences = {(r.get("subject"), r.get("seq")) for r in all_results if r.get("seq")}
    report = {
        "gcs_root": args.gcs_root,
        "subjects": subjects,
        "cameras": args.cameras or "all",
        "sequences_seen": len(sequences),
        "clips_total": len(all_results),
        "converted_ok": sum(1 for r in all_results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in all_results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "conventions": {
            "mano": "manopth use_pca ncomps=45 flat_hand_mean=False; aa45 = hands_mean + pca @ hands_components",
            "trans": "pose_m[48:51] passthrough (== smplx transl); camera frame == world",
            "camera": "static per clip; identity c2w traj; intrinsics from calibration ymls",
        },
        "failures": failed[:20],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("DEXYCB_CONVERT_DONE" if not failed else "DEXYCB_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
