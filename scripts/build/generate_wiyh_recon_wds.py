#!/usr/bin/env python
"""Phase B for WIYH (World In Your Hands): chest-fisheye JPEGs -> recon sub-clip frame tars.

WIYH ships as outer PLAIN-ustar part tars (misnamed *.tar.gz.~NNN) whose members are
per-sample worldcode_*.tar.gz archives (~400-500MB). Each sample holds ~22s of
1920x1536 chest-mounted KB4 fisheye JPEGs at 10fps plus dataset.hdf5 (camera table +
per-sample fisheye calibration). There are no usable native hand labels for our
contract (75-dim glove angles without a skeleton topology, wrist-only eef poses), so
WIYH rides the RECON path: this converter emits the exact egocentric Phase-B format
consumed by phase C recon / phase_d_incremental / the labeler:

  - frames_root/<clip_id>_sNN.tar       members <sub>_fNNNNN.image.jpg
  - outputs_root/<sub>/est_focal.txt    synthetic pinhole focal (Phase C honours it)
  - ClipManifestRecord per sub-clip     (descriptor.fps + extra.recon_fps = EFFECTIVE
                                         output fps; WIYH is 10fps native -> 10, i.e.
                                         BELOW the usual 15fps target, never upsampled)

The fisheye -> pinhole undistort uses a CHOSEN focal (--pinhole_focal, expressed at
--out_width scale) into a fixed --out_width x --out_height target instead of
cv2.fisheye.estimateNewCameraMatrixForUndistortRectify: the full ~160deg fisheye FOV
maps to a pinhole focal of ~43px @456w, far outside the recon-validated 138-213
regime, so we rectify a center crop at a validated focal instead. Samples are chopped
into --segment_sec sub-clips (10s default).

Usage (smoke, local sample archives):
  python scripts/build/generate_wiyh_recon_wds.py --sample_tars a.tar.gz b.tar.gz \
      --frames_root F --outputs_root O --manifest_out m.jsonl --report_out r.json

Usage (fleet, stream a part object):
  python scripts/build/generate_wiyh_recon_wds.py \
      --part gs://.../WIYH/Candlelight/Candlelight.tar.gz.~000 --max_samples 0 ...
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from lib.pipeline.clips.clip_manifest import ClipManifestRecord, write_clip_manifest  # noqa: E402
from lib.pipeline.datasets.descriptors import ClipDescriptor  # noqa: E402

CHEST_CAM = "lf_chest_fisheye"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WIYH worldcode samples -> recon frame tars")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--part", default="", help="gs:// outer part tar to stream")
    src.add_argument("--sample_tars", nargs="+", default=None, help="local worldcode_*.tar.gz files")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--outputs_root", required=True)
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--target_fps", type=float, default=15.0,
                   help="output fps ceiling; WIYH is 10fps native so the effective fps is "
                        "min(native, target) — frames are never upsampled")
    p.add_argument("--segment_sec", type=float, default=10.0)
    p.add_argument("--min_segment_sec", type=float, default=2.0)
    p.add_argument("--min_frames", type=int, default=8)
    p.add_argument("--out_width", type=int, default=456)
    p.add_argument("--out_height", type=int, default=256)
    p.add_argument("--pinhole_focal", type=float, default=150.0,
                   help="synthetic pinhole focal at --out_width scale (recon-validated "
                        "regime is 138-213 @456w; smaller = wider rectified FOV)")
    p.add_argument("--jpeg_quality", type=int, default=88)
    p.add_argument("--max_samples", type=int, default=0, help=">0: stop after N samples (smoke)")
    p.add_argument("--skip_samples", type=int, default=0, help="skip the first N members (part mode)")
    p.add_argument("--source_id", default="wiyh")
    p.add_argument("--split", default="train")
    p.add_argument("--work_dir", default="/tmp/wiyh_wds")
    return p


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "-", s)


def clip_base(sample_name: str) -> str:
    base = sample_name[len("worldcode_"):] if sample_name.startswith("worldcode_") else sample_name
    return _sanitize(base.replace("_s0_vlta_reorg_sample", ""))


def _extract_needed(gz_path: Path, dest: Path) -> Path | None:
    """Selective extract: dataset.hdf5 + chest-camera jpgs only. Returns sample root."""
    root = None
    with tarfile.open(gz_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            keep = name.endswith("dataset.hdf5") or (f"camera/{CHEST_CAM}/" in name and name.endswith(".jpg"))
            if not keep:
                continue
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(out, "wb") as w:
                shutil.copyfileobj(src, w, 16 * 1024 * 1024)
            if name.endswith("dataset.hdf5") and root is None:
                root = out.parent
    return root


def _undistort_map(K: np.ndarray, D: np.ndarray, focal_at_out: float, Wo: int, Ho: int):
    newK = np.array([[focal_at_out, 0, Wo / 2.0], [0, focal_at_out, Ho / 2.0], [0, 0, 1]], np.float64)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (Wo, Ho), cv2.CV_16SC2)
    return m1, m2


def convert_sample(sample_root: Path, sample_name: str, args, records: list, stats: dict) -> None:
    with h5py.File(sample_root / "dataset.hdf5", "r") as f:
        cal = f[f"meta/calibration/{CHEST_CAM}"]
        K = np.array(cal["intrinsic"], np.float64)
        D = np.array(cal["distortion"], np.float64).reshape(-1)[:4]
        cam = f[f"observation/camera/{CHEST_CAM}"][:]
    ts = cam["timestamp"].astype(np.float64)  # ms
    paths = [sample_root / p.decode() for p in cam["file_path"]]
    if len(ts) < args.min_frames:
        stats["skipped_short_samples"] += 1
        return
    dt = np.median(np.diff(ts)) / 1000.0
    src_fps = 1.0 / dt if dt > 0 else 10.0
    step = max(1, int(round(src_fps / args.target_fps)))  # never upsample
    eff_fps = src_fps / step
    duration = (ts[-1] - ts[0]) / 1000.0
    stats["source_sec"] += float(duration)
    stats["samples"] += 1

    m1, m2 = _undistort_map(K, D, args.pinhole_focal, args.out_width, args.out_height)
    base = clip_base(sample_name)
    frames_root = Path(args.frames_root)
    outputs_root = Path(args.outputs_root)
    seg_frames = max(args.min_frames, int(round(args.segment_sec * eff_fps)))
    sampled = list(range(0, len(paths), step))

    for k in range(0, len(sampled), seg_frames):
        seg_idx = sampled[k:k + seg_frames]
        if len(seg_idx) < max(args.min_frames, int(args.min_segment_sec * eff_fps)):
            continue
        sub = f"{base}_s{k // seg_frames:02d}"
        tar_path = frames_root / f"{sub}.tar"
        names, offsets = [], []
        with tarfile.open(tar_path, "w") as tar:
            for j, fi in enumerate(seg_idx):
                img = cv2.imread(str(paths[fi]), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
                ok, buf = cv2.imencode(".jpg", und, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    continue
                data = buf.tobytes()
                name = f"{sub}_f{len(names):05d}.image.jpg"
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
                names.append(name)
        if len(names) < args.min_frames:
            tar_path.unlink(missing_ok=True)
            continue
        with tarfile.open(tar_path, "r") as tr:  # byte offsets for direct pread
            for mem in tr:
                if mem.isfile():
                    offsets.append([int(mem.offset_data), int(mem.size)])
        seq = outputs_root / sub
        seq.mkdir(parents=True, exist_ok=True)
        (seq / "est_focal.txt").write_text(f"{args.pinhole_focal:.6f}\n")
        desc = ClipDescriptor.from_tar_shard(
            clip_id=sub, clip_name=sub, root_dir=str(frames_root.resolve()),
            seq_folder=str(seq.resolve()), shard_path=str(tar_path.resolve()),
            frame_names=names, frame_offsets=offsets,
            extra={"adapter": "wiyh_tar", "dataset_name": args.source_id,
                   "sample": sample_name, "pinhole_focal": round(args.pinhole_focal, 4),
                   "undistort": "cv2.fisheye/fixed-focal-center-crop",
                   "source_fps": round(src_fps, 3), "recon_fps": round(eff_fps, 3),
                   "segment_sec": [round(k / eff_fps, 2), round((k + len(seg_idx)) / eff_fps, 2)]})
        desc.fps = float(round(eff_fps, 3))
        desc.width, desc.height = int(args.out_width), int(args.out_height)
        # group = capture session (sample name minus the trailing sample counter)
        group = re.sub(r"_sample_\d+-\d+$", "", sample_name)
        records.append(ClipManifestRecord(clip_id=sub, source_id=args.source_id,
                                          split=args.split, descriptor=desc, group_id=group))
        stats["subclips"] += 1
        stats["frames"] += len(names)
        stats["kept_sec_out"] += len(names) / eff_fps


def _iter_sample_tars(args):
    """Yield (sample_name, local_gz_path, gz_bytes, cleanup_fn)."""
    if args.sample_tars:
        for p in args.sample_tars:
            p = Path(p)
            yield p.name.replace(".tar.gz", ""), p, p.stat().st_size, lambda: None
        return
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    seen = 0
    with fs.open(args.part.replace("gs://", ""), "rb", block_size=32 * 1024 * 1024) as fh:
        tf = tarfile.open(fileobj=fh, mode="r|")
        for m in tf:
            if not (m.isfile() and m.name.endswith(".tar.gz")):
                continue
            seen += 1
            if seen <= args.skip_samples:
                continue
            local = work / Path(m.name).name
            with tf.extractfile(m) as src, open(local, "wb") as w:
                shutil.copyfileobj(src, w, 16 * 1024 * 1024)
            yield local.name.replace(".tar.gz", ""), local, m.size, lambda l=local: l.unlink(missing_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    Path(args.frames_root).mkdir(parents=True, exist_ok=True)
    Path(args.outputs_root).mkdir(parents=True, exist_ok=True)
    records: list = []
    stats = {"samples": 0, "subclips": 0, "frames": 0, "errors": 0,
             "skipped_short_samples": 0, "source_sec": 0.0, "kept_sec_out": 0.0,
             "sample_gz_bytes": 0}
    t0 = time.time()
    n_done = 0
    for sample_name, gz_path, gz_bytes, cleanup in _iter_sample_tars(args):
        tmp = Path(tempfile.mkdtemp(dir=args.work_dir if not args.sample_tars else None,
                                    prefix="wiyh_x_"))
        try:
            root = _extract_needed(gz_path, tmp)
            if root is None:
                raise RuntimeError("no dataset.hdf5 in sample")
            convert_sample(root, sample_name, args, records, stats)
            stats["sample_gz_bytes"] += int(gz_bytes)
        except Exception as e:  # noqa: BLE001
            stats["errors"] += 1
            print(f"  ERR {sample_name[:70]}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            cleanup()
        n_done += 1
        print(f"[{n_done}] {sample_name[:70]} subclips_total={stats['subclips']} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if args.max_samples and n_done >= args.max_samples:
            break

    write_clip_manifest(records, args.manifest_out)
    stats["wall_sec"] = round(time.time() - t0, 1)
    stats["source_sec"] = round(stats["source_sec"], 1)
    stats["kept_sec_out"] = round(stats["kept_sec_out"], 1)
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(stats, indent=1))
    print(f"[wiyh-wds] {stats['subclips']} subclips from {stats['samples']} samples, "
          f"{stats['frames']} frames, {stats['errors']} errors -> {args.manifest_out} "
          f"({stats['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
