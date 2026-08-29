#!/usr/bin/env python3
"""Convert GigaHands (Fu et al., CVPR'25) sequences into filter-ready artifacts (GT ingestion).

GigaHands is a large bimanual hand-activity corpus captured in a BRICS multi-camera pod:
14k+ short sequences (~30 fps, 1280x720), each with EasyMocap-style **bimanual MANO** GT
(`hand_poses/<scene>/params/<seq>.json`: per-hand poses(T,48 global-zero)/shapes(1,10)/
Rh(T,3)/Th(T,3)), triangulated 21-joint world keypoints (`keypoints_3d/<seq>/{left,right}.jsonl`,
meters, x,y,z,conf), and per-scene camera calibration (`optim_params.txt`: K + k1 k2 p1 p2 +
COLMAP-style **w2c** qvec/tvec — probe-verified by projecting GT keypoints onto raw frames).

The GCS mirror ships an extracted subset of camera views per scene under
`multiview_rgb_vids/<scene>/<cam>/<cam>_<ts>.mp4` (map: `multiview_camera_video_map.csv`).
Per sequence we pick the available camera with the highest GT-keypoint in-frame ratio and
ship that single view, undistorted to a pinhole (cv2, K kept — same as the ARCTIC converter).

EasyMocap -> pipeline MANO convention (probe-verified with --probe against keypoints_3d):
EasyMocap poses vertices as ``v = Rh @ v_local + Th`` (rotation about the model ORIGIN,
global orient inside `poses` is zero), while the pipeline's smplx MANOLayer poses them as
``v = Rh @ (v_local - j0) + j0 + trans`` (rotation about the shaped root joint j0). Hence
``trans = Th + Rh @ j0 - j0`` with ``j0 = j0(betas)`` from the pipeline's own MANO models.
`hand_pose` passes through as-is (EasyMocap stores absolute 45-d pose, no mean offset) —
--probe reports MPJPE for both the as-is and +hands_mean variants to re-verify.

Outputs per sequence (identical contract to generate_arctic_world_res.py):
- frames_root/<clip_id>.tar                         (<clip_id>_f%05d.image.jpg, undistorted)
- outputs_root/<clip_id>/world_space_res.pth        [trans(2,T,3), rot(2,T,3), hand_pose(2,T,45),
                                                     betas(2,T,10), valid(2,T)] (0=left, 1=right)
- outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0.npz  (constant c2w — static rig camera)
- est_focal.txt, tracks_0_<T>/.gigahands_gt, .stage_done_infiller
- manifest JSONL + conversion report JSON
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GCS_PREFIX = "gs://foundational-research/hoi-dataset/GigaHands"


def build_parser():
    p = argparse.ArgumentParser(description="Convert GigaHands sequences to filter-ready artifacts")
    p.add_argument("--gcs_prefix", default=GCS_PREFIX)
    p.add_argument("--frames_root", required=True)
    p.add_argument("--outputs_root", required=True)
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--cache_dir", default="/w6/gigahands/_cache", help="per-scene calib/listing cache")
    p.add_argument("--work_dir", default="/w6/gigahands/_work")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--min_frames", type=int, default=45)
    p.add_argument("--min_conf", type=float, default=0.3, help="keypoint conf floor for a frame to count as valid")
    p.add_argument("--include", default=None, help="substring filter on <scene>/<seq>")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--probe", action="store_true", help="convention probe only: report MPJPE vs keypoints_3d")
    p.add_argument("--mano_pkl_dir", default="/root/arctic/unpack/body_models/mano",
                   help="dir with MANO_{LEFT,RIGHT}.pkl (hands_mean source; EasyMocap poses exclude the mean)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--source_id", default="gigahands")
    p.add_argument("--split_label", default="train")
    return p


# ----------------------------------------------------------------------------------------
# GCS helpers (gsutil; per-scene artifacts cached on disk)
# ----------------------------------------------------------------------------------------

def _gsutil_cp(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp{Path(tempfile.mktemp(dir='.')).name[-8:]}")
    subprocess.run(["gsutil", "-q", "cp", url, str(tmp)], check=True, capture_output=True)
    tmp.replace(dest)


def _cached(url: str, cache_path: Path) -> Path:
    if not cache_path.is_file():
        _gsutil_cp(url, cache_path)
    return cache_path


def load_scene_calib(prefix: str, scene: str, cache_dir: Path) -> dict:
    path = _cached(f"{prefix}/hand_poses/{scene}/optim_params.txt", cache_dir / scene / "optim_params.txt")
    cams = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) < 19:
            continue
        name = p[11]
        q = np.array([float(p[12]), float(p[13]), float(p[14]), float(p[15])])  # w x y z
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()            # w2c
        cams[name] = {
            "w": int(p[1]), "h": int(p[2]),
            "K": np.array([[float(p[3]), 0, float(p[5])], [0, float(p[4]), float(p[6])], [0, 0, 1]]),
            "dist": np.array([float(p[7]), float(p[8]), float(p[9]), float(p[10])]),
            "R": R, "t": np.array([float(p[16]), float(p[17]), float(p[18])]),
        }
    return cams


def list_scene_cam_dirs(prefix: str, scene: str, cache_dir: Path) -> list[str]:
    cache = cache_dir / scene / "cam_dirs.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    out = subprocess.run(["gsutil", "ls", f"{prefix}/multiview_rgb_vids/{scene}/"],
                         capture_output=True, text=True)
    cams = [u.rstrip("/").split("/")[-1] for u in out.stdout.split() if u.endswith("/")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cams))
    return cams


# ----------------------------------------------------------------------------------------
# MANO conversion (EasyMocap -> pipeline convention)
# ----------------------------------------------------------------------------------------

_MANO_CACHE = {}


def _mano_root_j0(betas: np.ndarray, is_right: bool) -> np.ndarray:
    """Shaped-template root joint j0(betas) from the pipeline's own MANO models (CPU)."""
    import torch
    key = "right" if is_right else "left"
    if key not in _MANO_CACHE:
        from lib.pipeline.exporters.mano_features import build_mano_models
        if "pair" not in _MANO_CACHE:
            _MANO_CACHE["pair"] = build_mano_models("cpu")
        _MANO_CACHE[key] = _MANO_CACHE["pair"][0 if is_right else 1]
    model = _MANO_CACHE[key]
    with torch.no_grad():
        eye = torch.eye(3)[None]
        out = model(
            global_orient=eye[:, None].expand(1, 1, 3, 3),
            hand_pose=eye[None].expand(1, 15, 3, 3),
            betas=torch.from_numpy(betas.reshape(1, 10)).float(),
            transl=torch.zeros(1, 3),
            pose2rot=False,
        )
    return out.joints[0, 0].numpy().astype(np.float64)  # openpose joint 0 == wrist/root


_HANDS_MEAN = {}


def _hands_mean(side: str, mano_pkl_dir: str) -> np.ndarray:
    if side not in _HANDS_MEAN:
        import pickle
        mdl = pickle.load(open(Path(mano_pkl_dir) / f"MANO_{side.upper()}.pkl", "rb"), encoding="latin1")
        _HANDS_MEAN[side] = np.asarray(mdl["hands_mean"], np.float64).reshape(45)
    return _HANDS_MEAN[side]


def easymocap_to_world_res(params: dict, T: int, min_conf_valid: dict,
                           mano_pkl_dir: str | None = None) -> list[np.ndarray]:
    trans = np.zeros((2, T, 3), np.float32)
    rot = np.zeros((2, T, 3), np.float32)
    hand_pose = np.zeros((2, T, 45), np.float32)
    betas_arr = np.zeros((2, T, 10), np.float32)
    valid = np.zeros((2, T), np.float32)
    for hand_index, side in ((0, "left"), (1, "right")):
        hp = params.get(side)
        if hp is None:
            continue
        poses = np.asarray(hp["poses"], np.float64)[:T]
        if mano_pkl_dir:  # EasyMocap poses exclude the MANO mean; pipeline treats pose as absolute
            poses = poses.copy()
            poses[:, 3:48] = poses[:, 3:48] + _hands_mean(side, mano_pkl_dir)[None]
        Rh = np.asarray(hp["Rh"], np.float64)[:T]
        Th = np.asarray(hp["Th"], np.float64)[:T]
        betas = np.asarray(hp["shapes"], np.float64).reshape(-1)[:10]
        j0 = _mano_root_j0(betas, is_right=(side == "right"))
        Rm = Rotation.from_rotvec(Rh).as_matrix()                     # (T,3,3)
        tr = Th + np.einsum("tij,j->ti", Rm, j0) - j0[None]
        rot[hand_index] = Rh.astype(np.float32)
        hand_pose[hand_index] = poses[:, 3:48].astype(np.float32)
        trans[hand_index] = tr.astype(np.float32)
        betas_arr[hand_index] = betas.astype(np.float32)[None]
        valid[hand_index] = min_conf_valid[side][:T]
    return [trans, rot, hand_pose, betas_arr, valid]


def pipeline_mano_joints(world_res: list[np.ndarray], hand_index: int, T: int) -> np.ndarray:
    """(T,21,3) world joints via the pipeline's MANO forward (for probe/verification)."""
    import torch
    from lib.pipeline.exporters.mano_features import run_mano_forward
    side_right = hand_index == 1
    _mano_root_j0(np.zeros(10), is_right=side_right)  # ensure model cached
    model = _MANO_CACHE["right" if side_right else "left"]
    trans, rot, hand_pose, betas, _ = world_res
    joints = run_mano_forward(
        model,
        torch.from_numpy(trans[hand_index][None]),
        torch.from_numpy(rot[hand_index][None]),
        torch.from_numpy(hand_pose[hand_index][None]),
        torch.from_numpy(betas[hand_index][None]),
        "cpu",
    )
    return joints[0, :, :21].numpy()


# ----------------------------------------------------------------------------------------
# per-sequence conversion
# ----------------------------------------------------------------------------------------

def _load_keypoints(work: Path, prefix: str, scene: str, seq: str) -> dict:
    out = {}
    for side in ("left", "right"):
        p = work / f"kp_{side}.jsonl"
        _gsutil_cp(f"{prefix}/hand_poses/{scene}/keypoints_3d/{seq}/{side}.jsonl", p)
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        out[side] = np.asarray(rows, np.float64)  # (T,21,4)
    return out


def pick_camera(kps: dict, cams: dict, candidates: list[str]) -> tuple[str, float]:
    best, best_ratio = None, -1.0
    pts = np.concatenate([kps["left"][::5, :, :3], kps["right"][::5, :, :3]], axis=1).reshape(-1, 3)
    for cam in candidates:
        c = cams.get(cam)
        if c is None:
            continue
        Xc = (c["R"] @ pts.T).T + c["t"]
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = c["K"][0, 0] * Xc[:, 0] / z + c["K"][0, 2]
            v = c["K"][1, 1] * Xc[:, 1] / z + c["K"][1, 2]
        ok = (z > 0.05) & (u >= 0) & (u < c["w"]) & (v >= 0) & (v < c["h"])
        ratio = float(ok.mean())
        if ratio > best_ratio:
            best, best_ratio = cam, ratio
    return best, best_ratio


def convert_seq(task):
    scene, seq, video_rel, args = task
    clip_id = f"GIGAHANDS_{scene}_{seq}"
    frames_root, outputs_root = Path(args.frames_root), Path(args.outputs_root)
    cache_dir = Path(args.cache_dir)
    seq_folder = outputs_root / clip_id
    tar_out = frames_root / f"{clip_id}.tar"
    done_marker = seq_folder / ".stage_done_infiller"
    result = {"scene": scene, "seq": seq, "clip_id": clip_id, "status": "ok"}
    if args.resume and tar_out.is_file() and done_marker.exists() and (seq_folder / "world_space_res.pth").is_file():
        return {**result, "status": "skipped"}

    work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=args.work_dir))
    try:
        cams = load_scene_calib(args.gcs_prefix, scene, cache_dir)
        kps = _load_keypoints(work, args.gcs_prefix, scene, seq)
        _gsutil_cp(f"{args.gcs_prefix}/hand_poses/{scene}/params/{seq}.json", work / "params.json")
        params = json.loads((work / "params.json").read_text())

        candidates = sorted(set(video_rel.keys()) & set(list_scene_cam_dirs(args.gcs_prefix, scene, cache_dir)))
        if not candidates:
            raise ValueError("no extracted camera dir matches the video map for this sequence")
        cam, inframe = pick_camera(kps, cams, candidates)
        if cam is None:
            raise ValueError("no candidate camera has calibration")
        result["camera"], result["kp_inframe_ratio"] = cam, round(inframe, 4)
        c = cams[cam]

        _gsutil_cp(f"{args.gcs_prefix}/multiview_rgb_vids/{scene}/{video_rel[cam]}", work / "video.mp4")
        # decode + undistort + re-encode on the fly (cv2; no ffmpeg binary dependency)
        map1, map2 = cv2.initUndistortRectifyMap(c["K"], c["dist"], None, c["K"], (c["w"], c["h"]), cv2.CV_32FC1)
        jpgs = []
        cap = cv2.VideoCapture(str(work / "video.mp4"))
        while True:
            ok, img = cap.read()
            if not ok:
                break
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            ok, buf = cv2.imencode(".jpg", und, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not ok:
                raise ValueError(f"jpeg encode failed at frame {len(jpgs)}")
            jpgs.append(buf.tobytes())
        cap.release()

        n_pose = min(len(params[s]["poses"]) for s in ("left", "right") if params.get(s)) if params else 0
        T = min(len(jpgs), n_pose, len(kps["left"]), len(kps["right"]))
        if T < args.min_frames:
            raise ValueError(f"too few synced frames: video={len(jpgs)} pose={n_pose}")

        conf_valid = {
            side: ((np.nan_to_num(kps[side][:T, :, 3]) > args.min_conf).mean(axis=1) > 0.5).astype(np.float32)
            for side in ("left", "right")
        }
        world_res = easymocap_to_world_res(params, T, conf_valid, mano_pkl_dir=args.mano_pkl_dir)
        if not np.isfinite(np.concatenate([a.reshape(-1) for a in world_res])).all():
            raise ValueError("non-finite values in converted world_space_res")

        seq_folder.mkdir(parents=True, exist_ok=True)
        tmp_tar = tar_out.with_suffix(".tar.tmp")
        with tarfile.open(tmp_tar, "w") as tw:
            for t in range(T):
                info = tarfile.TarInfo(name=f"{clip_id}_f{t:05d}.image.jpg")
                info.size = len(jpgs[t])
                tw.addfile(info, io.BytesIO(jpgs[t]))
        tmp_tar.replace(tar_out)

        joblib.dump(world_res, seq_folder / "world_space_res.pth")

        c2w_R = c["R"].T
        c2w_t = -c["R"].T @ c["t"]
        q_xyzw = Rotation.from_matrix(c2w_R).as_quat()
        traj = np.tile(np.concatenate([c2w_t, q_xyzw]).astype(np.float32), (T, 1))
        slam_dir = seq_folder / "SLAM"
        slam_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            slam_dir / "hawor_slam_w_scale_0.npz",
            tstamp=np.arange(T, dtype=np.int64), traj=traj, scale=np.float64(1.0),
            img_focal=np.float64(0.5 * (c["K"][0, 0] + c["K"][1, 1])),
            img_center=np.asarray([c["K"][0, 2], c["K"][1, 2]], dtype=np.float64),
        )
        (seq_folder / "est_focal.txt").write_text(f"{0.5 * (c['K'][0, 0] + c['K'][1, 1]):.6f}\n")
        tracks = seq_folder / f"tracks_0_{T}"
        tracks.mkdir(parents=True, exist_ok=True)
        (tracks / ".gigahands_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T, "camera": cam}))
        done_marker.touch()
        result["frames"] = int(T)
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


def probe_seq(task):
    """Convention probe: MPJPE of pipeline-MANO joints vs triangulated keypoints_3d."""
    scene, seq, _video_rel, args = task
    work = Path(tempfile.mkdtemp(prefix="probe_", dir=args.work_dir))
    try:
        kps = _load_keypoints(work, args.gcs_prefix, scene, seq)
        _gsutil_cp(f"{args.gcs_prefix}/hand_poses/{scene}/params/{seq}.json", work / "params.json")
        params = json.loads((work / "params.json").read_text())
        T = min(len(kps["left"]), len(kps["right"]), len(params["left"]["poses"]))
        conf_valid = {s: np.ones(T, np.float32) for s in ("left", "right")}
        report = {"scene": scene, "seq": seq, "T": T}
        for variant in ("as_is", "plus_mean"):
            p2 = {}
            for side in ("left", "right"):
                hp = {k: np.asarray(v, np.float64) for k, v in params[side].items()}
                if variant == "plus_mean":
                    import pickle
                    mdl = pickle.load(open(f"/root/arctic/unpack/body_models/mano/MANO_{side.upper()}.pkl", "rb"),
                                      encoding="latin1")
                    hp["poses"] = hp["poses"].copy()
                    hp["poses"][:, 3:] = hp["poses"][:, 3:] + np.asarray(mdl["hands_mean"]).reshape(1, 45)
                p2[side] = hp
            wr = easymocap_to_world_res(p2, T, conf_valid)
            errs = {}
            for hi, side in ((0, "left"), (1, "right")):
                j = pipeline_mano_joints(wr, hi, T)
                gt = kps[side][:T, :, :3]
                conf = np.nan_to_num(kps[side][:T, :, 3]) > 0.3
                e = np.linalg.norm(j - gt, axis=2)[conf]
                errs[side] = float(np.mean(e)) if e.size else None
            report[variant] = errs
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------------------
# discovery + manifest
# ----------------------------------------------------------------------------------------

def discover(args, cache_dir: Path) -> list[tuple]:
    """Return [(scene, seq, {cam: video_rel_path}), ...] for every (scene, seq) with params."""
    map_csv = _cached(f"{args.gcs_prefix}/multiview_camera_video_map.csv", cache_dir / "camera_video_map.csv")
    vid_map: dict[tuple, dict] = {}
    with open(map_csv) as f:
        for row in csv.DictReader(f):
            key = (row["scene"], row["sequence"])
            vid_map[key] = {cam: rel for cam, rel in row.items()
                            if cam not in ("scene", "sequence") and rel}
    out = subprocess.run(["gsutil", "ls", f"{args.gcs_prefix}/hand_poses/*/params/*.json"],
                         capture_output=True, text=True, check=True)
    tasks = []
    for url in sorted(out.stdout.split()):
        parts = url.rstrip("/").split("/")
        scene, seq = parts[-3], parts[-1][:-5]
        key = (scene, str(int(seq)))
        rel = vid_map.get(key) or vid_map.get((scene, seq))
        if not rel:
            continue
        if args.include and args.include not in f"{scene}/{seq}":
            continue
        tasks.append((scene, seq, rel, args))
    return tasks


def write_manifest(results, args):
    records = []
    for res in results:
        if res["status"] not in ("ok", "skipped"):
            continue
        clip_id = res["clip_id"]
        tar_path = Path(args.frames_root) / f"{clip_id}.tar"
        if not tar_path.is_file():
            continue
        frame_names, frame_offsets = [], []
        with tarfile.open(tar_path, "r") as reader:
            for m in reader:
                if m.isfile() and m.name.endswith(".image.jpg"):
                    frame_names.append(m.name)
                    frame_offsets.append([int(m.offset_data), int(m.size)])
        order = sorted(range(len(frame_names)), key=lambda i: frame_names[i])
        descriptor = {
            "clip_id": clip_id, "clip_name": clip_id, "storage_kind": "tar_shard",
            "root_dir": str(Path(args.frames_root).resolve()),
            "seq_folder": str((Path(args.outputs_root) / clip_id).resolve()),
            "frame_names": [frame_names[i] for i in order],
            "frame_offsets": [frame_offsets[i] for i in order],
            "shard_path": str(tar_path.resolve()),
            "extra": {"adapter": "gigahands_tar", "dataset_name": args.source_id,
                      "scene": res["scene"], "gigahands_seq": res["seq"],
                      "camera": res.get("camera"),
                      "source_text": res.get("source_text", "")},
        }
        records.append({"clip_id": clip_id, "source_id": args.source_id, "split": args.split_label,
                        "group_id": res["scene"], "descriptor": descriptor, "metadata": {}})
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def main():
    args = build_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    for d in (args.frames_root, args.outputs_root, args.work_dir, cache_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    tasks = discover(args, cache_dir)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"GigaHands sequences to {'probe' if args.probe else 'convert'}: {len(tasks)}", flush=True)

    # source-text lookup (annotations_v2)
    ann_path = _cached(f"{args.gcs_prefix}/annotations_v2.jsonl", cache_dir / "annotations_v2.jsonl")
    text = {}
    for line in ann_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            text[(r["scene"], str(r["sequence"]))] = r.get("clarify_annotation") or r.get("description") or ""

    started = time.perf_counter()
    fn = probe_seq if args.probe else convert_seq
    if args.workers <= 1:
        results = [fn(t) for t in tasks]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1)):
                results.append(r)
                if args.probe or (i + 1) % 25 == 0 or (i + 1) == len(tasks):
                    print(f"[{i+1}/{len(tasks)}] {json.dumps(r)[:240]}", flush=True)

    if args.probe:
        print(json.dumps(results, indent=2))
        return 0

    for r in results:
        r["source_text"] = text.get((r["scene"], r["seq"]), text.get((r["scene"], str(int(r["seq"]))), ""))
    n = write_manifest(results, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "total": len(tasks),
        "converted_ok": sum(1 for r in results if r["status"] == "ok"),
        "skipped_resume": sum(1 for r in results if r["status"] == "skipped"),
        "failed": len(failed), "manifest_records": n,
        "conventions": {
            "mano": "EasyMocap params: rot=Rh, pose=poses[3:48]+hands_mean, trans=Th+Rh@j0-j0 "
                    "(probe: 1.6-2.1mm MPJPE vs keypoints_3d_mano_align)",
            "camera": "optim_params qvec/tvec = w2c (COLMAP); undistorted to pinhole, K kept; static rig",
            "keypoints": "keypoints_3d jsonl (T,21,4) world meters used for camera choice + validity",
            "fps": "30",
        },
        "failures": failed[:20], "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("GIGAHANDS_CONVERT_DONE" if not failed else "GIGAHANDS_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
