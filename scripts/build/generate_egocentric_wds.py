#!/usr/bin/env python
"""Phase B: Egocentric-100K Layer-1 survivors -> undistorted pinhole frame tars + clip manifest.

Input is the Layer-1 survivor JSONL (one record per kept clip: clip_id, part_tar, worker, focal,
fps, intervals[]). For each VALID INTERVAL this emits one sub-clip:

  - frames_root/<clip_id>_ivNN.tar        `<sub_clip>_fNNNNN.image.jpg`, fisheye->pinhole undistorted
  - outputs_root/<clip_id>_ivNN/est_focal.txt   the undistorted pinhole focal for that worker's device
                                                (resolve_calibration reads this per clip, so the
                                                 per-worker calibration is honoured without a global flag)
  - a ClipManifestRecord per sub-clip

Reconstruction runs at --target_fps (default 15): hand motion is well sampled at 15fps and it halves
frames vs the 30fps source, which halves both recon GPU time and frame-tar storage.

Usage:
  python scripts/build/generate_egocentric_wds.py \
      --survivors stage1.kept.jsonl --frames_root F --outputs_root O --manifest_out m.jsonl
"""
from __future__ import annotations

import argparse, io, json, os, sys, tarfile, tempfile, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import numpy as np
import cv2
import gcsfs

from lib.pipeline.clips.clip_manifest import ClipManifestRecord, write_clip_manifest  # noqa: E402
from lib.pipeline.datasets.descriptors import ClipDescriptor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Egocentric-100K survivors -> undistorted frame tars")
    p.add_argument("--survivors", required=True, help="Layer-1 kept JSONL (or a shard of it)")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--outputs_root", required=True)
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--target_fps", type=float, default=15.0, help="Recon fps (source is 30)")
    p.add_argument("--balance", type=float, default=0.0, help="cv2.fisheye undistort balance (FOV knob)")
    p.add_argument("--jpeg_quality", type=int, default=88)
    p.add_argument("--max_clips", type=int, default=0, help=">0: cap clips (smoke)")
    p.add_argument("--max_intervals_per_clip", type=int, default=0, help=">0: cap intervals per clip")
    p.add_argument("--min_interval_sec", type=float, default=2.0)
    p.add_argument("--source_id", default="egocentric100k")
    p.add_argument("--split", default="train")
    return p


_MAPS: dict[str, tuple] = {}


def worker_map(fs, worker_gs: str, balance: float):
    """cv2.fisheye undistort maps + resulting pinhole focal for one worker's device."""
    if worker_gs in _MAPS:
        return _MAPS[worker_gs]
    path = worker_gs.replace("gs://", "") + "/intrinsics.json"
    intr = json.load(fs.open(path))
    W, H = int(intr["image_width"]), int(intr["image_height"])
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    D = np.array([intr["k1"], intr["k2"], intr["k3"], intr["k4"]], np.float64)
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), np.eye(3), balance=balance)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (W, H), cv2.CV_16SC2)
    out = (m1, m2, float(newK[0, 0]), W, H)
    _MAPS[worker_gs] = out
    return out


def main() -> None:
    args = build_parser().parse_args()
    fs = gcsfs.GCSFileSystem()
    frames_root = Path(args.frames_root); frames_root.mkdir(parents=True, exist_ok=True)
    outputs_root = Path(args.outputs_root); outputs_root.mkdir(parents=True, exist_ok=True)

    records, stats = [], {"clips": 0, "subclips": 0, "frames": 0, "skipped_short": 0, "errors": 0}
    t0 = time.time()
    survivors = [json.loads(l) for l in open(args.survivors) if l.strip()]
    if args.max_clips:
        survivors = survivors[:args.max_clips]

    # group by part tar so each shard is streamed once
    by_part: dict[str, list] = {}
    for r in survivors:
        by_part.setdefault(r["part_tar"], []).append(r)

    for part, rows in by_part.items():
        want = {r["clip_id"]: r for r in rows}
        gspath = part.replace("gs://", "")
        try:
            with fs.open(gspath, "rb") as fh:
                tf = tarfile.open(fileobj=fh, mode="r|")
                for m in tf:
                    if not m.name.endswith(".mp4"):
                        continue
                    key = m.name[:-4]
                    if key not in want:
                        continue
                    rec = want.pop(key)
                    mp4_bytes = tf.extractfile(m).read()
                    try:
                        _convert_clip(rec, mp4_bytes, fs, args, frames_root, outputs_root, records, stats)
                    except Exception as e:  # noqa: BLE001
                        stats["errors"] += 1
                        print(f"  ERR {key}: {str(e)[:90]}", flush=True)
                    if not want:
                        break
        except Exception as e:  # noqa: BLE001
            print(f"  PART ERR {part}: {str(e)[:100]}", flush=True)
        print(f"[{len(records)} subclips] part done ({time.time()-t0:.0f}s)", flush=True)

    write_clip_manifest(records, args.manifest_out)
    stats["wall_sec"] = round(time.time() - t0, 1)
    stats["target_fps"] = args.target_fps
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(stats, indent=1))
    print(f"[egocentric-wds] {stats['subclips']} subclips from {stats['clips']} clips, "
          f"{stats['frames']} frames -> {args.manifest_out}  ({stats['wall_sec']}s)", flush=True)


def _convert_clip(rec, mp4_bytes, fs, args, frames_root, outputs_root, records, stats):
    m1, m2, focal, W, H = worker_map(fs, rec["worker"], args.balance)
    src_fps = float(rec.get("fps") or 30.0)
    step = max(1, int(round(src_fps / args.target_fps)))          # 30 -> 15fps = every 2nd frame
    ivs = rec.get("intervals", [])
    if args.max_intervals_per_clip:
        ivs = sorted(ivs, key=lambda s: s["start_sec"] - s["end_sec"])[:args.max_intervals_per_clip]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
        t.write(mp4_bytes); mp4_path = t.name
    try:
        cap = cv2.VideoCapture(mp4_path)
        # decode once, keep only the frames any interval needs
        wanted = {}
        for k, s in enumerate(ivs):
            if (s["end_sec"] - s["start_sec"]) < args.min_interval_sec:
                stats["skipped_short"] += 1
                continue
            for f in range(int(s["start_frame"]), int(s["end_frame"]) + 1, step):
                wanted.setdefault(f, []).append(k)
        if not wanted:
            return
        got: dict[int, bytes] = {}
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx in wanted:
                und = cv2.remap(fr, m1, m2, cv2.INTER_LINEAR)
                ok2, buf = cv2.imencode(".jpg", und, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if ok2:
                    got[idx] = buf.tobytes()
            idx += 1
        cap.release()
    finally:
        os.unlink(mp4_path)

    stats["clips"] += 1
    for k, s in enumerate(ivs):
        if (s["end_sec"] - s["start_sec"]) < args.min_interval_sec:
            continue
        fr_idx = [f for f in range(int(s["start_frame"]), int(s["end_frame"]) + 1, step) if f in got]
        if len(fr_idx) < 8:
            continue
        sub = f'{rec["clip_id"]}_iv{k:02d}'
        tar_path = frames_root / f"{sub}.tar"
        names, offsets = [], []
        with tarfile.open(tar_path, "w") as tar:
            for j, f in enumerate(fr_idx):
                name = f"{sub}_f{j:05d}.image.jpg"
                data = got[f]
                ti = tarfile.TarInfo(name); ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
                names.append(name)
        with tarfile.open(tar_path, "r") as tr:      # byte offsets for direct pread
            for mem in tr:
                if mem.isfile():
                    offsets.append([int(mem.offset_data), int(mem.size)])
        seq = outputs_root / sub
        seq.mkdir(parents=True, exist_ok=True)
        (seq / "est_focal.txt").write_text(f"{focal:.6f}\n")   # per-worker pinhole focal
        desc = ClipDescriptor.from_tar_shard(
            clip_id=sub, clip_name=sub, root_dir=str(frames_root.resolve()),
            seq_folder=str(seq.resolve()), shard_path=str(tar_path.resolve()),
            frame_names=names, frame_offsets=offsets,
            extra={"adapter": "egocentric_tar", "dataset_name": args.source_id,
                   "worker": rec["worker"], "part_tar": rec["part_tar"],
                   "pinhole_focal": round(focal, 4), "undistort": "cv2.fisheye",
                   "source_fps": src_fps, "recon_fps": args.target_fps,
                   "interval_sec": [s["start_sec"], s["end_sec"]]})
        records.append(ClipManifestRecord(clip_id=sub, source_id=args.source_id, split=args.split,
                                          descriptor=desc, group_id=rec["worker"].rsplit("/", 2)[-2]))
        stats["subclips"] += 1
        stats["frames"] += len(names)


if __name__ == "__main__":
    main()
