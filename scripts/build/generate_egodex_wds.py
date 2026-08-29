#!/usr/bin/env python3
"""Convert EgoDex (Apple Vision Pro) episodes into native-feature WDS clips for the filter.

EgoDex has NO MANO — poses are the Vision Pro hand skeleton as per-joint SE(3) 4x4
transforms in the ARKit world frame. But it provides exactly what the EgoSmith quality
filter needs, so we build the pipeline's 116-d **native lowdim** directly from the joints
(no MANO fit) and emit native-feature WDS shards (the same path HOT3D-WDS uses):

  <clip>_f%05d.image.jpg / .lowdim.npy / .mano.npy / .meta.json  per frame, one tar/episode.

lowdim (116-d) schema (quality/constants.py), built per frame t:
  0:3   left wrist world xyz         (leftHand[:3,3])
  3:6   right wrist world xyz        (rightHand[:3,3])
  6:12  left root rot6d              (leftHand[:3,0], leftHand[:3,1])
  12:18 right root rot6d
  18:33 left 5 fingertips world xyz  (thumb,index,middle,ring,little Tip)
  33:48 right 5 fingertips world xyz
  48:66 wrist action  = wrist_state[t+1]
  66:96 hand action   = hand_state[t+1]
  96:112 extrinsic World2Cam (4x4)   = inv(transforms/camera[t])
  112:116 intrinsic [fx,fy,cx,cy]    (camera/intrinsic; EgoDex RGB is pinhole)
mano is zeros (2,55) — unused by the quality filter; native path only validates lowdim.
presence bitmask from per-joint confidence (leftHand bit0, rightHand bit1).

Run the filter with --stages native_features. EgoDex is CC-BY-NC-ND: keep outputs local.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LTIPS = ["leftThumbTip", "leftIndexFingerTip", "leftMiddleFingerTip", "leftRingFingerTip", "leftLittleFingerTip"]
RTIPS = ["rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip", "rightRingFingerTip", "rightLittleFingerTip"]


def build_parser():
    p = argparse.ArgumentParser(description="EgoDex -> native-feature WDS clips")
    p.add_argument("--egodex_root", required=True, help="dir with <task>/<i>.hdf5 + <i>.mp4 (e.g. .../dataset/test)")
    p.add_argument("--frames_root", required=True, help="output dir for per-episode WDS tars")
    p.add_argument("--outputs_root", required=True, help="seq_folder root (native path needs no stage outputs)")
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--jpeg_quality", type=int, default=4, help="ffmpeg mjpeg qscale (2 best..31)")
    p.add_argument("--conf_thresh", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include", default=None, help="regex on task name")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--source_id", default="egodex")
    p.add_argument("--split", default="test")
    p.add_argument("--part", default=None, help="part tag inserted into clip_id to keep tars unique across parts (e.g. part1)")
    return p


def _rot6d(R):  # first two columns -> 6
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def _read_episode(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as f:
        cam = f["transforms/camera"][:].astype(np.float64)          # (N,4,4) cam->world
        K = f["camera/intrinsic"][:].astype(np.float64)
        lh = f["transforms/leftHand"][:].astype(np.float64)
        rh = f["transforms/rightHand"][:].astype(np.float64)
        ltip = np.stack([f[f"transforms/{n}"][:, :3, 3] for n in LTIPS], axis=1).astype(np.float64)  # (N,5,3)
        rtip = np.stack([f[f"transforms/{n}"][:, :3, 3] for n in RTIPS], axis=1).astype(np.float64)
        def conf(name):
            k = f"confidences/{name}"
            return f[k][:].astype(np.float64) if k in f else np.ones(cam.shape[0])
        cl, cr = conf("leftHand"), conf("rightHand")
        desc = str(f.attrs.get("llm_description") or f.attrs.get("description") or "")
    fx = float(K[0, 0]); fy = float(K[1, 1]); cx = float(K[0, 2]); cy = float(K[1, 2])
    return dict(cam=cam, lh=lh, rh=rh, ltip=ltip, rtip=rtip, cl=cl, cr=cr,
                intr=np.array([fx, fy, cx, cy], np.float32), desc=desc)


def _build_lowdim(ep, T):
    lowdim = np.zeros((T, 116), np.float32)
    presence = np.zeros((T,), np.uint8)
    wrist_state = np.zeros((T, 18), np.float32)
    hand_state = np.zeros((T, 30), np.float32)
    for t in range(T):
        wrist_state[t, 0:3] = ep["lh"][t, :3, 3]
        wrist_state[t, 3:6] = ep["rh"][t, :3, 3]
        wrist_state[t, 6:12] = _rot6d(ep["lh"][t, :3, :3])
        wrist_state[t, 12:18] = _rot6d(ep["rh"][t, :3, :3])
        hand_state[t, 0:15] = ep["ltip"][t].reshape(15)
        hand_state[t, 15:30] = ep["rtip"][t].reshape(15)
        presence[t] = (int(ep["cl"][t] > 0) ) | (int(ep["cr"][t] > 0) << 1)
    for t in range(T):
        nt = min(t + 1, T - 1)
        lowdim[t, 0:18] = wrist_state[t]
        lowdim[t, 18:48] = hand_state[t]
        lowdim[t, 48:66] = wrist_state[nt]
        lowdim[t, 66:96] = hand_state[nt]
        w2c = np.linalg.inv(ep["cam"][t])
        w2c[3, :] = [0, 0, 0, 1]
        lowdim[t, 96:112] = w2c.reshape(16).astype(np.float32)
        lowdim[t, 112:116] = ep["intr"]
    return lowdim, presence


def convert_episode(task, h5_path, args):
    _pt = f"{args.part}_" if getattr(args, "part", None) else ""
    clip_id = f"egodex_{_pt}ep{task}_{Path(h5_path).stem}"
    clip_id = clip_id.replace(" ", "-")
    tar_path = Path(args.frames_root) / f"{clip_id}.tar"
    seq_folder = Path(args.outputs_root) / clip_id
    result = {"clip_id": clip_id, "task": task, "status": "ok"}
    if args.resume and tar_path.is_file():
        return {**result, "status": "skipped"}
    mp4 = Path(h5_path).with_suffix(".mp4")
    if not mp4.is_file():
        return {**result, "status": "failed", "error": "missing mp4"}
    work = Path(tempfile.mkdtemp(prefix=f"{clip_id}_", dir=args.frames_root))
    try:
        ep = _read_episode(h5_path)
        N = ep["cam"].shape[0]
        # extract frames
        outp = str(work / "f%05d.jpg")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(mp4),
                        "-vsync", "0", "-q:v", str(args.jpeg_quality), "-start_number", "0", outp],
                       check=True, capture_output=True)
        frames = sorted(work.glob("f*.jpg"))
        T = min(N, len(frames))
        if T < 3:
            raise ValueError(f"too few frames N={N} mp4={len(frames)}")
        lowdim, presence = _build_lowdim(ep, T)
        mano = np.zeros((2, 55), np.float32)

        frame_names, frame_offsets = [], []
        tmp_tar = tar_path.with_suffix(".tar.tmp")
        seq_folder.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tmp_tar, "w") as tw:
            for t in range(T):
                key = f"{clip_id}_f{t:05d}"
                # image
                img = frames[t].read_bytes()
                info = tarfile.TarInfo(f"{key}.image.jpg"); info.size = len(img)
                tw.addfile(info, io.BytesIO(img))
                # record offset for the image member (heavyweight tar reader uses these)
                # lowdim
                for suffix, arr in ((".lowdim.npy", lowdim[t]), (".mano.npy", mano)):
                    buf = io.BytesIO(); np.save(buf, arr); b = buf.getvalue()
                    ti = tarfile.TarInfo(f"{key}{suffix}"); ti.size = len(b); tw.addfile(ti, io.BytesIO(b))
                meta = json.dumps({"presence": int(presence[t])}).encode()
                tm = tarfile.TarInfo(f"{key}.meta.json"); tm.size = len(meta); tw.addfile(tm, io.BytesIO(meta))
        tmp_tar.replace(tar_path)
        # recompute image offsets from the finished tar
        with tarfile.open(tar_path, "r") as tr:
            for m in tr:
                if m.isfile() and m.name.endswith(".image.jpg"):
                    frame_names.append(m.name); frame_offsets.append([int(m.offset_data), int(m.size)])
        frame_names.sort(); result["frames"] = T; result["desc"] = ep["desc"]
        result["_frame_names"] = frame_names; result["_frame_offsets"] = frame_offsets
    except Exception as error:
        result["status"] = "failed"; result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            result["error"] += " :: " + error.stderr.decode("utf8", "replace")[-200:]
    finally:
        import shutil; shutil.rmtree(work, ignore_errors=True)
    return result


def _star(a):
    return convert_episode(*a)


def write_manifest(results, args):
    from lib.pipeline.clips.clip_manifest import write_clip_manifest, ClipManifestRecord
    from lib.pipeline.datasets.descriptors import ClipDescriptor
    recs = []
    for r in results:
        if r["status"] != "ok":
            continue
        tar_path = Path(args.frames_root) / f"{r['clip_id']}.tar"
        # if resumed we may not have frame names cached; read from tar
        fn = r.get("_frame_names"); fo = r.get("_frame_offsets")
        if fn is None:
            fn, fo = [], []
            with tarfile.open(tar_path, "r") as tr:
                for m in tr:
                    if m.isfile() and m.name.endswith(".image.jpg"):
                        fn.append(m.name); fo.append([int(m.offset_data), int(m.size)])
        order = sorted(range(len(fn)), key=lambda i: fn[i]); fn=[fn[i] for i in order]; fo=[fo[i] for i in order]
        desc = ClipDescriptor.from_tar_shard(
            clip_id=r["clip_id"], clip_name=r["clip_id"],
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / r["clip_id"]).resolve()),
            shard_path=str(tar_path.resolve()), frame_names=fn, frame_offsets=fo,
            extra={"adapter": "egodex_wds", "native_feature_source": "wds_lowdim_mano_v1",
                   "lowdim_schema": "egodex_vp_world_v1", "mano_schema": "zeros_2x55",
                   "dataset_name": args.source_id, "task": r["task"]})
        recs.append(ClipManifestRecord(clip_id=r["clip_id"], source_id=args.source_id,
                                       split=args.split, descriptor=desc, group_id=r["task"]))
    write_clip_manifest(recs, args.manifest_out)
    return len(recs)


def main():
    import re
    args = build_parser().parse_args()
    for d in (args.frames_root, args.outputs_root):
        Path(d).mkdir(parents=True, exist_ok=True)
    root = Path(args.egodex_root)
    tasks = sorted([t for t in root.iterdir() if t.is_dir()])
    if args.include:
        pat = re.compile(args.include); tasks = [t for t in tasks if pat.search(t.name)]
    jobs = []
    for t in tasks:
        for h5 in sorted(t.glob("*.hdf5")):
            jobs.append((t.name, str(h5), args))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"EgoDex episodes to convert: {len(jobs)} across {len(tasks)} tasks", flush=True)

    started = time.perf_counter()
    if args.workers <= 1:
        results = [convert_episode(*j) for j in jobs]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_star, jobs, chunksize=1)):
                results.append(r)
                if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                    print(f"[{i+1}/{len(jobs)}] {r['clip_id']} {r['status']}", flush=True)
    n = write_manifest(results, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {"egodex_root": str(root), "episodes": len(jobs), "tasks": len(tasks),
              "converted_ok": sum(1 for r in results if r["status"] == "ok"),
              "skipped": sum(1 for r in results if r["status"] == "skipped"),
              "failed": len(failed), "manifest_records": n,
              "failures": [{k: v for k, v in f.items() if not k.startswith("_")} for f in failed[:20]],
              "elapsed_sec": time.perf_counter() - started}
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("EGODEX_CONVERT_DONE" if not failed else "EGODEX_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
