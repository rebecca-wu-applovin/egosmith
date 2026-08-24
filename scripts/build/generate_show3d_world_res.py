#!/usr/bin/env python3
"""Convert SHOW3D GT annotations into filter-ready seq_folders + frame tars + manifest.

SHOW3D (CVPR 2026, facebook/show3d-dataset) is in-the-wild egocentric HOI: per
recording two Quest-3 headset cameras (1024x1280 monochrome, 60 fps, released
UNDISTORTED pinhole) + per-frame 21-landmark UmeTrack hand poses
(``hand_pose/v2/scenes/<subject>/<scene>/hand_pose.json``) and per-frame headset
extrinsics (``scenes/<subject>/<scene>/camera_calibration/headset{0,1}.json``,
``T_WorldFromCamera`` = c2w in the RIG frame, translations in mm).

GT is a 21-joint skeleton (not MANO), so this converter torch-fits MANO to the
world-frame landmarks with the pipeline's FPHA fitter
(``lib.pipeline.hands.fpha_skeleton.fit_right_hand_mano_sequence``); the left hand is
fitted via x-mirroring (reflect targets with diag(-1,1,1), fit right-MANO, mirror the
axis-angle/translation params back), which is exact for the pipeline's left MANO
(x-mirrored template incl. the shapedirs sign fix).

Landmark order (HOT3D/UmeTrack): 0-4 fingertips (thumb..pinky), 5 wrist, 6-7 thumb
(intermediate, distal), 8-10 index, 11-13 middle, 14-16 ring, 17-19 pinky, 20 palm.
MANO target order is [wrist, thumb1..4, index1..4, middle1..4, ring1..4, pinky1..4];
the missing thumb-CMC is synthesized at wrist + 0.5*(thumb_intermediate - wrist).
Projection convention probe-verified vs the dataset's own landmarks_2d: <1e-4 px.

Frames with hand ``confidence <= conf_thresh`` (default 0.5, the dataset's
recommended threshold) get valid=0/zeros. One clip per (scene, headset camera).

Outputs per clip (identical contract to generate_taco_world_res.py):
- frames_root/<clip_id>.tar (<clip_id>_f%05d.image.jpg)
- outputs_root/<clip_id>/world_space_res.pth [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)] (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz (per-frame c2w, meters)
- est_focal.txt, tracks_0_<T-1>/.show3d_gt, infiller done marker
- manifest JSONL + conversion report JSON
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402

GCS_ROOT = "gs://foundational-research/hoi-dataset/SHOW3D"
_MIRROR = np.diag([-1.0, 1.0, 1.0])


def _ffmpeg_exe() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()

# MANO 21 target slot -> UmeTrack landmark index (see module docstring). -1 = synthetic thumb1.
_MANO_TO_UME = [5, -1, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert SHOW3D GT (headset views) to filter-ready artifacts")
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--gcs_root", default=GCS_ROOT)
    parser.add_argument("--hand_pose_version", default="v2", help="hand_pose tree version (v2 fixes v1 landmark scale)")
    parser.add_argument("--work_dir", default=None, help="Scratch dir for per-scene downloads")
    parser.add_argument("--cameras", default="headset0,headset1")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--jpeg_quality", type=int, default=3, help="ffmpeg -q:v")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit_iters", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1, help="Parallel clip workers (share the fit GPU)")
    parser.add_argument("--include", default=None, help="Regex on clip_id")
    parser.add_argument("--limit", type=int, default=None, help="Cap on clips")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_id", default="show3d")
    parser.add_argument("--split", default="train")
    return parser


def discover_scenes(gcs_root: str, version: str) -> list[tuple[str, str]]:
    """[(subject_id, scene_id)] from the hand_pose index parquet (train release)."""
    import io
    import pyarrow.parquet as pq

    raw = subprocess.run(["gcloud", "storage", "cat", f"{gcs_root}/hand_pose/{version}/index.parquet"],
                         check=True, capture_output=True).stdout
    table = pq.read_table(io.BytesIO(raw)).to_pandas()
    return [(str(r.subject_id), str(r.scene_id)) for r in table.itertuples()]


def _gcs_cp(src: str, dst: Path) -> None:
    subprocess.run(["gcloud", "storage", "cp", src, str(dst)], check=True, capture_output=True)


def ume_to_mano_targets(landmarks_mm: np.ndarray) -> np.ndarray:
    """(21,3) UmeTrack landmarks (mm, world) -> (21,3) MANO-ordered targets (meters)."""
    lm = np.asarray(landmarks_mm, dtype=np.float64) * 1e-3
    out = np.zeros((21, 3), dtype=np.float64)
    for mano_idx, ume_idx in enumerate(_MANO_TO_UME):
        if ume_idx >= 0:
            out[mano_idx] = lm[ume_idx]
    out[1] = lm[5] + 0.5 * (lm[6] - lm[5])  # synthetic thumb-CMC (ume6 = thumb intermediate)
    return out


def _mirror_aa(aa: np.ndarray) -> np.ndarray:
    """Axis-angle conjugated by diag(-1,1,1): (x,y,z) -> (x,-y,-z)."""
    out = np.asarray(aa, dtype=np.float32).copy()
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def _fit_right_mano_centroid(targets: "np.ndarray", device: str, num_iters: int, chunk: int = 1024):
    """Fit right MANO to (T,21,3) targets with CLOSED-FORM weighted translation per frame.

    Unlike fpha_skeleton.fit_right_hand_mano_sequence (which pins the MANO wrist exactly
    onto target joint 0), translation is solved as the weighted centroid offset each
    iteration. The UmeTrack wrist sits ~1 cm into the forearm relative to MANO's, so the
    exact-wrist alignment displaces the whole hand (43 mm MPJPE); centroid alignment
    absorbs it (5 mm MPJPE on the smoke scene). The synthetic thumb-CMC gets weight 0.
    """
    import torch
    from lib.pipeline.hands.fpha_skeleton import (
        _build_right_mano, _forward_right_mano, _estimate_initial_global_rot,
    )

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mano = _build_right_mano(dev)
    weights = torch.ones(1, 21, 1, device=dev)
    weights[0, 1, 0] = 0.0                      # synthetic thumb-CMC
    weights[0, [4, 8, 12, 16, 20], 0] = 2.0     # fingertips
    weight_sum = weights.sum()
    with torch.no_grad():
        canon = _forward_right_mano(
            mano, torch.zeros(1, 3, device=dev), torch.zeros(1, 45, device=dev),
            torch.zeros(1, 10, device=dev))[0].cpu().numpy()

    T_all = np.asarray(targets, dtype=np.float32)
    init_rot_all = _estimate_initial_global_rot(T_all, canon)
    shared_betas = None
    out_trans, out_rot, out_pose = [], [], []
    for start in range(0, len(T_all), chunk):
        tgt = torch.tensor(T_all[start:start + chunk], device=dev)
        n = len(tgt)
        go = torch.nn.Parameter(torch.tensor(init_rot_all[start:start + chunk], device=dev))
        hp = torch.nn.Parameter(torch.zeros(n, 45, device=dev))
        params = [go, hp]
        if shared_betas is None:
            betas = torch.nn.Parameter(torch.zeros(1, 10, device=dev))
            params.append(betas)
        else:
            betas = shared_betas
        opt = torch.optim.Adam(params, lr=5e-2)
        for it in range(int(num_iters)):
            if it in (int(num_iters * 0.5), int(num_iters * 0.8)):
                for g in opt.param_groups:
                    g["lr"] *= 0.3
            opt.zero_grad(set_to_none=True)
            local = _forward_right_mano(mano, go, hp, betas.expand(n, -1))
            transl = ((tgt - local) * weights).sum(dim=1) / weight_sum
            res = local + transl[:, None, :] - tgt
            loss = (res.abs() * weights).mean()
            loss = loss + 2.0 * res[:, [4, 8, 12, 16, 20]].norm(dim=-1).mean()
            loss = loss + 1e-4 * hp.square().mean() + 1e-3 * betas.square().mean()
            if n > 1:
                loss = loss + 1e-3 * (hp[1:] - hp[:-1]).square().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            local = _forward_right_mano(mano, go, hp, betas.expand(n, -1))
            transl = ((tgt - local) * weights).sum(dim=1) / weight_sum
        if shared_betas is None:
            shared_betas = betas.detach()
        out_trans.append(transl.detach().cpu().numpy())
        out_rot.append(go.detach().cpu().numpy())
        out_pose.append(hp.detach().cpu().numpy())
    return (np.concatenate(out_trans, axis=0), np.concatenate(out_rot, axis=0),
            np.concatenate(out_pose, axis=0), shared_betas.cpu().numpy().reshape(10))


def fit_side(targets_world: np.ndarray, side: str, device: str, num_iters: int):
    """Fit MANO to (T,21,3) world targets for one side. Returns (trans, rot, pose45, betas10)."""
    targets = np.asarray(targets_world, dtype=np.float64)
    if side == "left":
        targets = targets @ _MIRROR.T  # reflect into right-hand chirality
    trans, rot, pose, betas = _fit_right_mano_centroid(targets.astype(np.float32), device, num_iters)
    if side == "left":
        trans = (trans @ _MIRROR.T).astype(np.float32)
        rot = _mirror_aa(rot)
        pose = _mirror_aa(pose.reshape(-1, 15, 3)).reshape(-1, 45)
    return trans.astype(np.float32), rot.astype(np.float32), pose.astype(np.float32), betas.reshape(10).astype(np.float32)


def build_world_res(hand_pose: dict, T: int, args) -> tuple[list[np.ndarray], dict]:
    trans = np.zeros((2, T, 3), dtype=np.float32)
    rot = np.zeros((2, T, 3), dtype=np.float32)
    pose45 = np.zeros((2, T, 45), dtype=np.float32)
    betas = np.zeros((2, T, 10), dtype=np.float32)
    valid = np.zeros((2, T), dtype=np.float32)
    fit_stats = {}
    for hand_key, side, hand_index in (("0", "left", 0), ("1", "right", 1)):
        target_list, frame_idx = [], []
        for t in range(T):
            entry = hand_pose.get(str(t))
            if entry is None:
                continue
            h = (entry.get("hand_poses") or {}).get(hand_key)
            if not h or (h.get("confidence") or 0.0) <= args.conf_thresh or h.get("landmarks_3d_mm") is None:
                continue
            target_list.append(ume_to_mano_targets(np.asarray(h["landmarks_3d_mm"])))
            frame_idx.append(t)
        fit_stats[side] = {"valid_frames": len(frame_idx)}
        if len(frame_idx) < 2:
            continue
        targets = np.stack(target_list, axis=0)
        f_trans, f_rot, f_pose, f_betas = fit_side(targets, side, args.device, args.fit_iters)
        idx = np.asarray(frame_idx, dtype=np.int64)
        trans[hand_index, idx] = f_trans
        rot[hand_index, idx] = f_rot
        pose45[hand_index, idx] = f_pose
        betas[hand_index] = f_betas[None, :]
        valid[hand_index, idx] = 1.0
    return [trans, rot, pose45, betas, valid], fit_stats


def convert_clip(subject: str, scene: str, camera: str, args) -> dict:
    clip_id = "SHOW3D_" + re.sub(r"[^A-Za-z0-9]+", "_", f"{subject}_{scene}_{camera[-2:]}")
    frames_root, outputs_root = Path(args.frames_root), Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_path = frames_root / f"{clip_id}.tar"
    result = {"subject": subject, "scene": scene, "camera": camera, "clip_id": clip_id, "status": "ok"}

    done_marker = get_stage_done_marker(seq_folder, "infiller")
    if args.resume and tar_path.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        result["status"] = "skipped"
        return result

    work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=args.work_dir))
    try:
        scene_uri = f"{args.gcs_root}/scenes/{subject}/{scene}"

        def _fetch(gcs_uri: str, hf_relpath: str, dst) -> None:
            """GCS first; fall back to the HF source of truth when the mirror object is
            0-byte (two bad-copy batches exist: 68 hand_pose.json + a set of
            camera_calibration jsons, all Aug-18 timestamps)."""
            _gcs_cp(gcs_uri, dst)
            if dst.stat().st_size == 0:
                from huggingface_hub import hf_hub_download

                src = hf_hub_download("facebook/show3d-dataset", hf_relpath,
                                      repo_type="dataset", local_dir=str(work / "_hf"))
                shutil.copyfile(src, dst)

        _fetch(f"{args.gcs_root}/hand_pose/{args.hand_pose_version}/scenes/{subject}/{scene}/hand_pose.json",
               f"hand_pose/{args.hand_pose_version}/scenes/{subject}/{scene}/hand_pose.json",
               work / "hand_pose.json")
        _fetch(f"{scene_uri}/camera_calibration/{camera}.json",
               f"scenes/{subject}/{scene}/camera_calibration/{camera}.json",
               work / "cal.json")
        _fetch(f"{scene_uri}/{camera}.mp4",
               f"scenes/{subject}/{scene}/{camera}.mp4",
               work / "video.mp4")

        cal = json.loads((work / "cal.json").read_text())
        hand_pose = json.loads((work / "hand_pose.json").read_text())
        tw_by_idx = cal["T_WorldFromCamera_by_index"]

        frames_dir = work / "frames"
        frames_dir.mkdir()
        subprocess.run(
            [_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(work / "video.mp4"),
             "-vsync", "0", "-start_number", "0", "-q:v", str(args.jpeg_quality),
             str(frames_dir / f"{clip_id}_f%05d.image.jpg")],
            check=True, capture_output=True)
        frame_files = sorted(frames_dir.glob("*.image.jpg"))
        T = min(len(frame_files), len(hand_pose), len(tw_by_idx))
        result["frame_counts"] = {"video": len(frame_files), "hand_pose": len(hand_pose),
                                  "calibration": len(tw_by_idx), "used": T}
        if T < 2:
            raise ValueError(f"too few frames: {result['frame_counts']}")

        payload, fit_stats = build_world_res(hand_pose, T, args)
        result["fit_stats"] = fit_stats

        seq_folder.mkdir(parents=True, exist_ok=True)
        tmp_tar = tar_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as writer:
            for frame in frame_files[:T]:
                writer.add(frame, arcname=frame.name)
        tmp_tar.replace(tar_path)

        joblib.dump(payload, seq_folder / "world_space_res.pth")

        c2w = np.stack([np.asarray(tw_by_idx[str(t)]["T_WorldFromCamera"], dtype=np.float64) for t in range(T)])
        c2w[:, :3, 3] *= 1e-3  # mm -> meters
        quat_xyzw = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()
        traj = np.concatenate([c2w[:, :3, 3], quat_xyzw], axis=1).astype(np.float32)
        slam_dir = seq_folder / "SLAM"
        slam_dir.mkdir(parents=True, exist_ok=True)
        focal = 0.5 * (float(cal["fx"]) + float(cal["fy"]))
        np.savez(
            slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
            tstamp=np.arange(T, dtype=np.int64),
            traj=traj,
            scale=np.float64(1.0),
            img_focal=np.float64(focal),
            img_center=np.asarray([float(cal["cx"]), float(cal["cy"])], dtype=np.float64),
        )
        (seq_folder / "est_focal.txt").write_text(f"{focal:.6f}\n")
        tracks_dir = seq_folder / f"tracks_0_{T - 1}"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        (tracks_dir / ".show3d_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
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


def _convert_star(task):
    subject, scene, camera, args = task
    return convert_clip(subject, scene, camera, args)


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
                "adapter": "show3d_tar",
                "dataset_name": args.source_id,
                "subject": result["subject"],
                "scene": result["scene"],
                "camera": result["camera"],
            },
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                group_id=f"{result['subject']}/{result['scene']}",
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
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    scenes = discover_scenes(args.gcs_root, args.hand_pose_version)
    tasks = [(subject, scene, camera) for subject, scene in scenes for camera in cameras]
    if args.include:
        pattern = re.compile(args.include)
        tasks = [t for t in tasks if pattern.search(f"SHOW3D_{t[0]}_{t[1]}_{t[2][-2:]}")]
    if args.limit:
        tasks = [t for t in tasks if True][: args.limit]
    print(f"SHOW3D: {len(scenes)} recordings with hand_pose {args.hand_pose_version}; {len(tasks)} clips selected", flush=True)

    started = time.perf_counter()
    results = []
    if args.workers <= 1:
        for idx, (subject, scene, camera) in enumerate(tasks):
            results.append(convert_clip(subject, scene, camera, args))
            print(f"[{idx + 1}/{len(tasks)}] {results[-1]['clip_id']} {results[-1]['status']}", flush=True)
    else:
        from multiprocessing import get_context

        with get_context("spawn").Pool(args.workers) as pool:
            for idx, result in enumerate(pool.imap_unordered(_convert_star, [(s, sc, c, args) for s, sc, c in tasks], chunksize=1)):
                results.append(result)
                if (idx + 1) % 10 == 0 or (idx + 1) == len(tasks) or result["status"] == "failed":
                    print(f"[{idx + 1}/{len(tasks)}] {result['clip_id']} {result['status']}"
                          + (f" :: {result.get('error', '')[:160]}" if result["status"] == "failed" else ""), flush=True)

    manifest_count = write_manifest(results, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "gcs_root": args.gcs_root,
        "hand_pose_version": args.hand_pose_version,
        "recordings_discovered": len(scenes),
        "clips_selected": len(tasks),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed),
        "manifest_records": manifest_count,
        "conventions": {
            "gt": "21-landmark UmeTrack skeleton (mm, rig-world) -> MANO torch fit (FPHA fitter)",
            "left_hand": "x-mirror fit (diag(-1,1,1) conjugation on aa params)",
            "camera": "T_WorldFromCamera per frame (c2w, mm->m); released views already pinhole-undistorted",
            "conf_thresh": args.conf_thresh,
        },
        "failures": failed[:20],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("SHOW3D_CONVERT_DONE" if not failed else "SHOW3D_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
