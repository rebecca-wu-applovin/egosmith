#!/usr/bin/env python3
"""Convert EgoTouch (TouchAnything) episodes into filter-ready seq_folders (SMOKE).

Category-2.5 / Category-2 mechanism note: EgoTouch's per-episode pose labels are
(a) HaMeR/WiLoR monocular pseudo-labels in the chest-camera frame (21x3, MANO/OpenPose
joint order, JSONL per frame with nulls on miss) and (b) Rokoko EMF glove mocap in a
world frame WITHOUT usable camera extrinsics (vive_poses is empty on audited
episodes). Per the Cat-2 decision tree, the actionable mechanism today is the
monocular pseudo-label track (recon-grade, NOT GT): fit MANO to the HaMeR camera-frame
joints and treat the camera frame as world (identity c2w) -- the same posture as
EgoDex native-lowdim. The glove->MANO GT track is blocked on camera extrinsics.

Convention (probe-verified in scripts/inspection/cat25_egotouch_audit.py):
full-image pinhole f = 5000/256*640 = 12500.0, pp = (320, 240), 640x480 @ 30fps;
median projected-joint -> shipped-hand-mask distance 0.0-9.2 px on audited episodes.

Outputs per clip (contract identical to generate_show3d_world_res.py):
- frames_root/<clip_id>.tar          (<clip_id>_f%05d.image.jpg from chest.mp4)
- outputs_root/<clip_id>/world_space_res.pth [trans(2,T,3), rot(2,T,3),
  hand_pose(2,T,45), betas(2,T,10), valid(2,T)] (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz (identity c2w)
- est_focal.txt, tracks_0_<T-1>/.egotouch_pseudo, infiller done marker
- manifest JSONL + conversion report JSON

Usage (5-clip smoke):
  python scripts/build/generate_egotouch_world_res.py \
      --episodes Home/fold_t_shirt/20260320_151616_503,... \
      --frames_root /root/cat25_smoke/egotouch/frames \
      --outputs_root /root/cat25_smoke/egotouch/outputs \
      --manifest_out /root/cat25_smoke/egotouch/manifest.jsonl
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
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402

GCS_ROOT = "gs://foundational-research/hoi-dataset/EgoTouch"
_MIRROR = np.diag([-1.0, 1.0, 1.0])
FOCAL = 5000.0 / 256.0 * 640.0  # probe-verified full-image HaMeR focal
CENTER = (320.0, 240.0)


def _ffmpeg_exe() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert EgoTouch episodes to filter-ready artifacts (pseudo-label track)")
    parser.add_argument("--episodes", required=True, help="Comma-separated Scene/task/timestamp episode paths")
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--report_out", default=None)
    parser.add_argument("--gcs_root", default=GCS_ROOT)
    parser.add_argument("--local_root", default=None, help="Dir with pre-downloaded episode dirs named Scene_task_ts (skips GCS)")
    parser.add_argument("--work_dir", default=None)
    parser.add_argument("--jpeg_quality", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit_iters", type=int, default=180)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source_id", default="egotouch")
    parser.add_argument("--split", default="train")
    return parser


def _mirror_aa(aa: np.ndarray) -> np.ndarray:
    """Conjugate axis-angle stacks by diag(-1,1,1) (exact left-MANO mirror)."""
    out = aa.copy().reshape(-1, 3)
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out.reshape(aa.shape)


def fit_side(targets: np.ndarray, side: str, device: str, num_iters: int):
    from lib.pipeline.hands.fpha_skeleton import fit_right_hand_mano_sequence

    work_targets = targets @ _MIRROR.T if side == "left" else targets
    trans, rot, pose, betas = fit_right_hand_mano_sequence(
        work_targets.astype(np.float32), device=device, num_iters=num_iters)
    if side == "left":
        trans = trans @ _MIRROR.T
        rot = _mirror_aa(rot)
        pose = _mirror_aa(pose.reshape(-1, 15, 3)).reshape(-1, 45)
    return trans, rot, pose, betas.reshape(-1)


def load_joints(path: Path) -> dict[str, np.ndarray]:
    """hamer/wilor JSONL -> {'left': (T,21,3) nan-padded, 'right': ...}"""
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    T = len(records)
    out = {s: np.full((T, 21, 3), np.nan, dtype=np.float64) for s in ("left", "right")}
    for i, r in enumerate(records):
        for s in ("left", "right"):
            p = r.get(f"{s}_pos")
            if p is not None:
                arr = np.asarray(p, dtype=np.float64)
                if arr.shape == (21, 3):
                    out[s][i] = arr
    return out


def build_world_res(joints: dict[str, np.ndarray], T: int, args) -> tuple[list[np.ndarray], dict]:
    trans = np.zeros((2, T, 3), dtype=np.float32)
    rot = np.zeros((2, T, 3), dtype=np.float32)
    pose45 = np.zeros((2, T, 45), dtype=np.float32)
    betas = np.zeros((2, T, 10), dtype=np.float32)
    valid = np.zeros((2, T), dtype=np.float32)
    fit_stats = {}
    for side, hand_index in (("left", 0), ("right", 1)):
        j = joints[side][:T]
        finite = np.isfinite(j).all(axis=(1, 2))
        # degenerate frames (all-zero / collapsed joints) break the fitter's palm-frame init
        spread = np.nan_to_num(j.std(axis=1)).max(axis=1)
        idx = np.nonzero(finite & (spread > 1e-4))[0]
        fit_stats[side] = {"valid_frames": int(len(idx))}
        if len(idx) < 2:
            continue
        targets = joints[side][idx]
        f_trans, f_rot, f_pose, f_betas = fit_side(targets, side, args.device, args.fit_iters)
        trans[hand_index, idx] = f_trans
        rot[hand_index, idx] = f_rot
        pose45[hand_index, idx] = f_pose
        betas[hand_index] = f_betas[None, :]
        valid[hand_index, idx] = 1.0
    return [trans, rot, pose45, betas, valid], fit_stats


def convert_episode(episode: str, args) -> dict:
    clip_id = "EGOTOUCH_" + re.sub(r"[^A-Za-z0-9]+", "_", episode)
    frames_root, outputs_root = Path(args.frames_root), Path(args.outputs_root)
    seq_folder = outputs_root / clip_id
    tar_path = frames_root / f"{clip_id}.tar"
    result = {"episode": episode, "clip_id": clip_id, "status": "ok"}

    done_marker = get_stage_done_marker(seq_folder, "infiller")
    if args.resume and tar_path.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        result["status"] = "skipped"
        return result

    work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=args.work_dir))
    try:
        if args.local_root:
            src = Path(args.local_root) / episode.replace("/", "_")
            for name in ("chest.mp4", "hamer_hands.json", "wilor_hands.json"):
                if (src / name).is_file():
                    shutil.copyfile(src / name, work / name)
        else:
            for name in ("chest.mp4", "hamer_hands.json", "wilor_hands.json"):
                subprocess.run(["gsutil", "-q", "cp", f"{args.gcs_root}/{episode}/{name}", str(work / name)],
                               check=False, capture_output=True)
        pose_file = work / "hamer_hands.json"
        if not pose_file.is_file() or pose_file.stat().st_size == 0:
            pose_file = work / "wilor_hands.json"
            result["pose_source"] = "wilor"
        else:
            result["pose_source"] = "hamer"
        if not pose_file.is_file() or pose_file.stat().st_size == 0:
            raise ValueError("no non-empty hamer/wilor pose file (zero-byte mirror gap?)")
        if not (work / "chest.mp4").is_file() or (work / "chest.mp4").stat().st_size == 0:
            raise ValueError("chest.mp4 missing or zero-byte")

        frames_dir = work / "frames"
        frames_dir.mkdir()
        subprocess.run(
            [_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(work / "chest.mp4"),
             "-vsync", "0", "-start_number", "0", "-q:v", str(args.jpeg_quality),
             str(frames_dir / f"{clip_id}_f%05d.image.jpg")],
            check=True, capture_output=True)
        frame_files = sorted(frames_dir.glob("*.image.jpg"))
        joints = load_joints(pose_file)
        T = min(len(frame_files), len(joints["left"]))
        result["frame_counts"] = {"video": len(frame_files), "pose": int(len(joints["left"])), "used": T}
        if T < 2:
            raise ValueError(f"too few frames: {result['frame_counts']}")

        payload, fit_stats = build_world_res(joints, T, args)
        result["fit_stats"] = fit_stats

        seq_folder.mkdir(parents=True, exist_ok=True)
        tmp_tar = tar_path.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as writer:
            for frame in frame_files[:T]:
                writer.add(frame, arcname=frame.name)
        tmp_tar.replace(tar_path)

        joblib.dump(payload, seq_folder / "world_space_res.pth")

        # identity c2w: camera frame == world frame (monocular pseudo-label track)
        traj = np.zeros((T, 7), dtype=np.float32)
        traj[:, 6] = 1.0  # unit quaternion xyzw
        slam_dir = seq_folder / "SLAM"
        slam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
            tstamp=np.arange(T, dtype=np.int64),
            traj=traj,
            scale=np.float64(1.0),
            img_focal=np.float64(FOCAL),
            img_center=np.asarray(CENTER, dtype=np.float64),
        )
        (seq_folder / "est_focal.txt").write_text(f"{FOCAL:.6f}\n")
        tracks_dir = seq_folder / f"tracks_0_{T - 1}"
        tracks_dir.mkdir(parents=True, exist_ok=True)
        (tracks_dir / ".egotouch_pseudo").write_text(json.dumps({"clip_id": clip_id, "frames": T}))
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
                "adapter": "egotouch_tar",
                "dataset_name": args.source_id,
                "episode": result["episode"],
                "pose_source": result.get("pose_source", "hamer"),
                "pose_grade": "monocular_pseudo_label",
            },
        )
        records.append(
            ClipManifestRecord(
                clip_id=clip_id,
                source_id=args.source_id,
                split=args.split,
                descriptor=descriptor,
                group_id="/".join(result["episode"].split("/")[:2]),
            )
        )
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)
    episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]
    results = [convert_episode(ep, args) for ep in episodes]
    kept = write_manifest(results, args)
    report = {
        "dataset": args.source_id,
        "episodes": len(episodes),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "manifest_records": kept,
        "pose_grade": "monocular_pseudo_label (HaMeR/WiLoR camera frame, identity c2w)",
        "glove_gt_track": "BLOCKED: rokoko world frame lacks camera extrinsics (vive_poses empty)",
        "results": results,
    }
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for r in results:
        print(f"  {r['clip_id']}: {r['status']}" + (f" ({r.get('error','')})" if r["status"] == "failed" else
                                                    f" T={r.get('frames')} fit={r.get('fit_stats')}"))
    print("EGOTOUCH_CONVERT_DONE" if all(r["status"] != "failed" for r in results) else "EGOTOUCH_CONVERT_FAILURES")
    return 0 if all(r["status"] != "failed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
