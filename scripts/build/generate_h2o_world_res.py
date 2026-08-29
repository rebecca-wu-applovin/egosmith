#!/usr/bin/env python3
"""Convert H2O (ETH, Kwon et al. ICCV 2021) GT annotations into filter-ready artifacts.

H2O ships 4 subjects x {h1,h2,k1,k2,o1,o2} scenes x numbered sequences, each recorded by
5 synchronized cameras (cam0..cam3 static, cam4 = head-mounted egocentric). On GCS the
data lives as one tar.gz per subject (``gs://foundational-research/hoi-dataset/H2O/
subject{N}_v1_1.tar.gz``) laid out ``subject/scene/seq/camX/{rgb/%06d.png,
hand_pose/%06d.txt, hand_pose_mano/%06d.txt, cam_pose/%06d.txt, cam_intrinsics.txt, ...}``.

This converter **streams** each subject tar.gz (no full extraction), buffers ONE sequence's
cam4 (egocentric) members at a time, and emits ONE CLIP PER (subject, scene, seq).

Conventions (probe-verified on subject1/h1/0/cam4, scripts smoke 2026-08; see W9 notes):
- ``hand_pose_mano/%06d.txt`` = 124 floats: per hand [valid(1), trans(3), rot aa(3),
  pose aa(45), betas(10)], LEFT first then RIGHT, in the **containing camera's frame**,
  **smplx convention** (trans == smplx transl passthrough; probe: 2.0 mm mean / 0.4 mm wrist
  joint error vs the dataset's own ``hand_pose`` 21x3 GT joints; the manopth alternatives
  score ~100 mm).
- ``cam_pose/%06d.txt`` = 4x4 **cam-to-world** (egocentric world anchored near the first
  frame's head pose).
- ``cam_intrinsics.txt`` = fx fy cx cy W H (1280x720 pinhole RGB).
- World re-expression (world_space_res is world-frame): R_w = R_c2w @ R_g and
  transl_w = R_c2w @ (transl_c + j0) + t_c2w - j0, with j0 = MANO joint-0 of (betas,
  zero pose) — smplx rotates about j0. Verified in-converter (--verify_world) by
  comparing MANO joints under world params vs c2w-transformed camera-frame joints.

Outputs per clip (identical contract to generate_taco_world_res.py / dexycb):
- frames_root/<clip_id>.tar          (<clip_id>_f%05d.image.jpg, png re-encoded jpg q90)
- outputs_root/<clip_id>/world_space_res.pth  [trans(2,T,3), rot(2,T,3), hand_pose(2,T,45),
  betas(2,T,10), valid(2,T)] (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz  (GT c2w traj, scale 1)
- outputs_root/<clip_id>/est_focal.txt, tracks_0_<T-1>/.h2o_gt, infiller done marker
- manifest JSONL (ClipManifestRecord) + conversion report JSON

H2O is 30 fps -> filter with ``--stages infiller --source_fps 30 --target_fps 30``.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
import time
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402

GCS_ROOT = "gs://foundational-research/hoi-dataset/H2O"
SUBJECTS = ["subject1_v1_1", "subject2_v1_1", "subject3_v1_1", "subject4_v1_1"]
EGO_CAM = "cam4"
FPS = 30.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stream-convert H2O GT (egocentric cam4) to filter-ready artifacts")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--outputs_root", required=True)
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--gcs_root", default=GCS_ROOT)
    p.add_argument("--local_tar_dir", default=None, help="Read <subject>.tar.gz from here instead of GCS")
    p.add_argument("--subjects", default=",".join(SUBJECTS))
    p.add_argument("--workers", type=int, default=4, help="Parallel subject streams (max = #subjects)")
    p.add_argument("--include", default=None, help="Regex on clip_id")
    p.add_argument("--limit", type=int, default=None, help="Cap on clips per subject stream (smoke)")
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verify_world", action=argparse.BooleanOptionalAction, default=True,
                   help="Per-clip numeric check: world params reproduce c2w @ camera-frame joints")
    p.add_argument("--source_id", default="h2o")
    p.add_argument("--split", default="train")
    return p


def _parse_floats(raw: bytes) -> np.ndarray:
    return np.array([float(x) for x in raw.split()], dtype=np.float64)


def _aa_to_R(aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(aa).as_matrix()


def _R_to_aa(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def _mano_joints(trans, rot, pose, betas, use_cuda=False):
    """(2,T,...) numpy -> (2,T,21,3) numpy via the pipeline's smplx MANO."""
    import torch
    from lib.pipeline.hands.mano_runtime import run_mano_twohands
    with torch.inference_mode():
        out = run_mano_twohands(
            torch.from_numpy(np.ascontiguousarray(trans)).float(),
            torch.from_numpy(np.ascontiguousarray(rot)).float(),
            torch.from_numpy(np.ascontiguousarray(pose)).float(),
            None,
            torch.from_numpy(np.ascontiguousarray(betas)).float(),
            use_cuda=use_cuda,
        )
    return out["joints"].cpu().numpy()


def build_world_res(mano124: np.ndarray, c2w: np.ndarray, verify: bool) -> tuple[list[np.ndarray], float]:
    """mano124 (T,124) camera-frame + c2w (T,4,4) -> world_space_res payload.

    Returns (payload, verify_err_m). Invalid frames get zeroed params + valid=0."""
    T = mano124.shape[0]
    trans_c = np.stack([mano124[:, 1:4], mano124[:, 63:66]])       # (2,T,3)
    rot_c = np.stack([mano124[:, 4:7], mano124[:, 66:69]])         # (2,T,3)
    pose = np.stack([mano124[:, 7:52], mano124[:, 69:114]])        # (2,T,45)
    betas = np.stack([mano124[:, 52:62], mano124[:, 114:124]])     # (2,T,10)
    valid = np.stack([mano124[:, 0], mano124[:, 62]]) > 0.5        # (2,T)

    # j0 = MANO joint-0 for (betas, zero pose/rot/trans); smplx global_orient pivots about it
    j0 = _mano_joints(np.zeros_like(trans_c), np.zeros_like(rot_c), np.zeros_like(pose), betas)[:, :, 0]

    Rw = c2w[:, :3, :3]
    tw = c2w[:, :3, 3]
    trans_w = np.empty_like(trans_c)
    rot_w = np.empty_like(rot_c)
    for h in range(2):
        Rg = _aa_to_R(rot_c[h])                                    # (T,3,3)
        rot_w[h] = _R_to_aa(Rw @ Rg)
        trans_w[h] = np.einsum("tij,tj->ti", Rw, trans_c[h] + j0[h]) + tw - j0[h]

    err = 0.0
    if verify:
        j_cam = _mano_joints(trans_c, rot_c, pose, betas)          # (2,T,21,3)
        j_w = _mano_joints(trans_w, rot_w, pose, betas)
        j_cam_w = np.einsum("tij,htkj->htki", Rw, j_cam) + tw[None, :, None]
        err = float(np.abs(j_w - j_cam_w).max())

    payload = [np.zeros((2, T, 3), np.float32), np.zeros((2, T, 3), np.float32),
               np.zeros((2, T, 45), np.float32), np.zeros((2, T, 10), np.float32),
               valid.astype(np.float32)]
    for h in range(2):
        v = valid[h]
        payload[0][h][v] = trans_w[h][v].astype(np.float32)
        payload[1][h][v] = rot_w[h][v].astype(np.float32)
        payload[2][h][v] = pose[h][v].astype(np.float32)
        payload[3][h][v] = betas[h][v].astype(np.float32)
    return payload, err


def write_clip(clip_id: str, jpgs: list[bytes], mano124: np.ndarray, c2w: np.ndarray,
               intr: np.ndarray, args) -> dict:
    from scipy.spatial.transform import Rotation

    T = len(jpgs)
    frames_root, outputs_root = Path(args.frames_root), Path(args.outputs_root)
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

    payload, verr = build_world_res(mano124[:T], c2w[:T], args.verify_world)
    joblib.dump(payload, seq_folder / "world_space_res.pth")

    quat_xyzw = Rotation.from_matrix(c2w[:T, :3, :3]).as_quat()
    traj = np.concatenate([c2w[:T, :3, 3], quat_xyzw], axis=1).astype(np.float32)
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    slam_dir = seq_folder / "SLAM"
    slam_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
        tstamp=np.arange(T, dtype=np.int64),
        traj=traj,
        scale=np.float64(1.0),
        img_focal=np.float64(0.5 * (fx + fy)),
        img_center=np.asarray([cx, cy], dtype=np.float64),
    )
    (seq_folder / "est_focal.txt").write_text(f"{0.5 * (fx + fy):.6f}\n")

    tracks_dir = seq_folder / f"tracks_0_{T - 1}"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    (tracks_dir / ".h2o_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
    marker = get_stage_done_marker(seq_folder, "infiller")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return {"frames": T, "verify_world_err_m": verr}


def _recode_png_to_jpg(png_bytes: bytes, quality: int) -> bytes:
    import cv2
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("png decode failed")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpg encode failed")
    return bytes(buf)


def process_sequence(subject: str, seq_key: str, buf: dict, args, emitted: list) -> dict | None:
    """buf: {"rgb": {idx: png bytes}, "mano": {idx: (124,)}, "pose": {idx: (4,4)},
    "intr": (6,), "action": {idx: int}}"""
    clip_id = f"H2O_{seq_key.replace('/', '_')}_{EGO_CAM}"
    include = re.compile(args.include) if args.include else None
    if include and not include.search(clip_id):
        return None
    result = {"clip_id": clip_id, "subject": subject, "seq": seq_key, "status": "ok"}
    try:
        seq_folder = Path(args.outputs_root) / clip_id
        tar_path = Path(args.frames_root) / f"{clip_id}.tar"
        if args.resume and tar_path.is_file() and get_stage_done_marker(seq_folder, "infiller").exists() \
                and (seq_folder / "world_space_res.pth").is_file():
            result["status"] = "skipped"
            return result
        if buf.get("intr") is None:
            raise ValueError("missing cam_intrinsics.txt")
        idxs = sorted(set(buf["rgb"]) & set(buf["mano"]) & set(buf["pose"]))
        T = 0
        while T < len(idxs) and idxs[T] == T:
            T += 1
        if T < 3:
            raise ValueError(f"too few contiguous frames: {len(idxs)} complete")
        jpgs = [_recode_png_to_jpg(buf["rgb"][t], args.jpeg_quality) for t in range(T)]
        mano124 = np.stack([buf["mano"][t] for t in range(T)])
        c2w = np.stack([buf["pose"][t] for t in range(T)])
        if mano124.shape[1] != 124:
            raise ValueError(f"bad hand_pose_mano width {mano124.shape}")
        stats = write_clip(clip_id, jpgs, mano124, c2w, buf["intr"], args)
        result.update(stats)
        result["valid_left"] = int((mano124[:T, 0] > 0.5).sum())
        result["valid_right"] = int((mano124[:T, 62] > 0.5).sum())
        acts = [buf["action"][t] for t in range(T) if t in buf.get("action", {})]
        if acts:
            result["action_labels"] = sorted({int(a) for a in acts})
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        emitted.append(result)
    return result


class _LimitReached(Exception):
    pass


_MEMBER_RE = re.compile(
    rf"^(subject\d+)/([^/]+)/(\d+)/{EGO_CAM}/(rgb/(\d{{6}})\.png|hand_pose_mano/(\d{{6}})\.txt|"
    rf"cam_pose/(\d{{6}})\.txt|action_label/(\d{{6}})\.txt|cam_intrinsics\.txt)$")


def stream_subject(subject: str, args) -> list[dict]:
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

    def _flush():
        nonlocal buf
        if current_seq is not None and buf.get("rgb"):
            process_sequence(subject, current_seq, buf, args, emitted)
            done = [r for r in emitted if r["status"] in ("ok", "skipped", "failed")]
            print(f"[{subject}] {emitted[-1]['clip_id']} {emitted[-1]['status']}"
                  + (f" :: {emitted[-1].get('error')}" if emitted[-1]["status"] == "failed" else "")
                  + f" ({len(done)} clips)", flush=True)
            if args.limit and len(emitted) >= args.limit:
                raise _LimitReached
        buf = {"rgb": {}, "mano": {}, "pose": {}, "action": {}, "intr": None}

    try:
        buf = {"rgb": {}, "mano": {}, "pose": {}, "action": {}, "intr": None}
        for member in stream:
            if not member.isfile():
                continue
            m = _MEMBER_RE.match(member.name)
            if m is None:
                continue
            seq_key = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
            if current_seq is not None and seq_key != current_seq:
                _flush()
            current_seq = seq_key
            payload = stream.extractfile(member).read()
            rest = m.group(4)
            if rest == "cam_intrinsics.txt":
                buf["intr"] = _parse_floats(payload)
            elif rest.startswith("rgb/"):
                buf["rgb"][int(m.group(5))] = payload
            elif rest.startswith("hand_pose_mano/"):
                buf["mano"][int(m.group(6))] = _parse_floats(payload)
            elif rest.startswith("cam_pose/"):
                buf["pose"][int(m.group(7))] = _parse_floats(payload).reshape(4, 4)
            elif rest.startswith("action_label/"):
                buf["action"][int(m.group(8))] = int(payload.split()[0])
        _flush()
    except _LimitReached:
        pass
    finally:
        stream.close()
        if proc is not None:
            proc.kill()
            proc.wait()
    return emitted


def _stream_star(task):
    subject, args = task
    try:
        return stream_subject(subject, args)
    except Exception as error:
        return [{"clip_id": f"H2O_{subject}", "subject": subject, "status": "failed",
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
        parts = result.get("seq", "").split("/")
        descriptor = ClipDescriptor.from_tar_shard(
            clip_id=clip_id,
            clip_name=clip_id,
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / clip_id).resolve()),
            shard_path=str(tar_path.resolve()),
            frame_names=frame_names,
            frame_offsets=frame_offsets,
            extra={
                "adapter": "h2o_tar",
                "dataset_name": args.source_id,
                "subject": result.get("subject"),
                "scene": parts[1] if len(parts) > 2 else None,
                "h2o_seq": result.get("seq"),
                "camera": EGO_CAM,
                "fps": FPS,
                "action_labels": result.get("action_labels"),
            },
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                group_id=result.get("seq", ""),
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)
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
    verrs = [r.get("verify_world_err_m") for r in all_results if r.get("verify_world_err_m") is not None]
    report = {
        "gcs_root": args.gcs_root,
        "subjects": subjects,
        "camera": EGO_CAM,
        "fps": FPS,
        "clips_total": len(all_results),
        "converted_ok": sum(1 for r in all_results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in all_results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "verify_world_err_m_max": max(verrs) if verrs else None,
        "conventions": {
            "mano": "smplx passthrough; hand_pose_mano = [valid,trans3,rot3,pose45,betas10] x {L,R}, camera frame",
            "world": "R_w = R_c2w @ R_g; transl_w = R_c2w @ (transl_c + j0) + t_c2w - j0",
            "camera": "cam_pose = c2w per frame (GT headset pose); intrinsics fx fy cx cy W H",
        },
        "failures": failed[:20],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("H2O_CONVERT_DONE" if not failed else "H2O_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
