#!/usr/bin/env python
"""Phase B for VIDEO-source datasets: Stage-1 survivors -> 15fps sub-clip frame tars.

Sibling of generate_egocentric_wds.py (which consumes tar-of-jpg part tars from the
Egocentric-100K/10K layout). This one consumes the Cat-1 Stage-1 video survivor rows
produced by egocentric_stage1_video.py:

  {uri, group, fps, n_frames, width, height, intervals: [{start_frame, end_frame,
   start_sec, end_sec, score}], (offset, size, session for tar-member uris)}

For each valid interval it emits one sub-clip tar `<clip_id>_ivNN.tar` with members
`<sub>_fNNNNN.image.jpg` plus a ClipManifestRecord (storage_kind tar_shard, fps 15,
extra.recon_fps/source_fps, width/height) — the exact format phase C
(pod_entry_*recon), phase_d_incremental and the labeler already consume.

Differences vs the 100K converter, driven by the sources:
  * sources are plain video objects (or tar members via ranged read), not part tars;
  * frames are TIME-sampled to --target_fps (source fps varies: 24.46/30/59.94/60),
    index_j = start_frame + round(j * src_fps / target_fps) — exact 15fps everywhere,
    not the integer-step approximation (24.46fps sources would land at 12.2fps);
  * pinhole sources are only RESIZED to --out_width (no undistort, no focal written:
    Phase C estimates the focal with --use_anycalib). Fisheye sources (Assembly101
    HMC) undistort via cv2.fisheye with per-camera estimated intrinsics
    (--fisheye_intrinsics JSON keyed by a substring of the uri, e.g. "HMC_21110305");
    those DO get extra.pinhole_focal, which Phase C honours via est_focal.txt;
  * GoPro-family sources carry telemetry data streams that stall cv2 -> every video
    is remuxed video-only (ffmpeg -map 0:v:0 -c copy) before decode, and
    OPENCV_FFMPEG_READ_ATTEMPTS is raised pre-import (Stage-1 driver v3 lessons);
  * every video is fetched+decoded in an isolated SUBPROCESS: a native segfault or
    hang costs one video, not the shard (Stage-1 driver v2 lesson).

Usage:
  PYTHONPATH=src python scripts/build/generate_video_wds.py \
      --survivors shard.videos.jsonl --frames_root F --outputs_root O \
      --manifest_out m.jsonl --report_out r.json --source_id hd_epic \
      --out_width 456 --id_mode basename
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "1000000")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lib.pipeline.clips.clip_manifest import ClipManifestRecord, write_clip_manifest  # noqa: E402
from lib.pipeline.datasets.descriptors import ClipDescriptor  # noqa: E402

SEEK_GAP_FRAMES = 240  # jump via CAP_PROP_POS_FRAMES when the next wanted frame is further


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Video-source Stage-1 survivors -> 15fps frame tars")
    p.add_argument("--survivors", required=True, help="Stage-1 video survivor JSONL (this shard)")
    p.add_argument("--frames_root", required=True)
    p.add_argument("--outputs_root", required=True)
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--target_fps", type=float, default=15.0)
    p.add_argument("--out_width", type=int, default=456, help="downscale (never upscale) to this width")
    p.add_argument("--jpeg_quality", type=int, default=88)
    p.add_argument("--min_interval_sec", type=float, default=2.0)
    p.add_argument("--min_frames", type=int, default=8)
    p.add_argument("--max_clips", type=int, default=0, help=">0: cap videos (smoke)")
    p.add_argument("--max_intervals_per_clip", type=int, default=0)
    p.add_argument("--source_id", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--id_mode", choices=["basename", "group_basename", "session"], default="basename",
                   help="clip_id base: video stem / <group>__<stem> (Assembly101: HMC stems repeat "
                        "across sessions) / tar-member session (HoloAssist)")
    p.add_argument("--fisheye_intrinsics", default="",
                   help="JSON {key: {fx,fy,cx,cy,k1..k4,image_width,image_height}}; a video whose uri "
                        "contains a key is fisheye-undistorted with those intrinsics (cv2.fisheye)")
    p.add_argument("--fisheye_balance", type=float, default=0.0)
    p.add_argument("--per_video_timeout", type=int, default=10800)
    p.add_argument("--remux_max_gb", type=float, default=12.0,
                   help="skip the video-only remux above this source size (remux writes a "
                        "full copy -> 2x disk peak; OPENCV_FFMPEG_READ_ATTEMPTS covers "
                        "correctness, remux is a decode-speed optimization)")
    p.add_argument("--work_dir", default="/tmp/vwds")
    # internal isolation-worker mode
    p.add_argument("--one_video_json", default="")
    p.add_argument("--result_prefix", default="")
    return p


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "-", s)


def clip_base(rec: dict, id_mode: str) -> str:
    uri = rec["uri"]
    if id_mode == "session":
        return _sanitize(rec.get("session") or uri.split("::", 1)[-1].rsplit(".", 1)[0].replace("/", "__"))
    stem = uri.split("::", 1)[-1].rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    if id_mode == "group_basename":
        return _sanitize(f'{rec.get("group", "")}__{stem}')
    return _sanitize(stem)


def _find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    for cand in ("/opt/conda/envs/egosmith/bin/ffmpeg", "/opt/conda/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if Path(cand).is_file():
            return cand
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _remux_video_only(path: Path, ffmpeg: str) -> Path:
    out = path.with_name(path.stem + ".vonly.mp4")
    try:
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(path),
                            "-map", "0:v:0", "-c", "copy", str(out)],
                           capture_output=True, timeout=3600)
        if r.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            path.unlink(missing_ok=True)
            return out
    except Exception:  # noqa: BLE001
        pass
    out.unlink(missing_ok=True)
    return path


def _fetch(rec: dict, dest_dir: Path) -> Path:
    """Plain object via gcloud cp; tar member via gcsfs ranged read (needs offset+size)."""
    uri = rec["uri"]
    if "::" in uri:
        import gcsfs
        tar_uri, member = uri.split("::", 1)
        fs = gcsfs.GCSFileSystem()
        dest = dest_dir / ((rec.get("session") or member.replace("/", "_")) + ".mp4")
        off, size = int(rec["offset"]), int(rec["size"])
        with fs.open(tar_uri[5:], "rb") as f, open(dest, "wb") as w:
            f.seek(off)
            remaining = size
            while remaining > 0:
                chunk = f.read(min(32 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError("short read from tar member")
                w.write(chunk)
                remaining -= len(chunk)
        return dest
    name = uri.rsplit("/", 1)[-1]
    if "." not in name:
        name += ".mp4"
    dest = dest_dir / name
    subprocess.run(["gcloud", "storage", "cp", uri, str(dest)], check=True, capture_output=True)
    return dest


_FE_MAPS: dict[str, tuple] = {}


def _fisheye_map(key: str, intr: dict, balance: float, out_width: int):
    """cv2.fisheye undistort maps + pinhole focal (same construction as the validated
    100K worker_map: undistort straight into the downscaled pinhole target)."""
    if key in _FE_MAPS:
        return _FE_MAPS[key]
    W, H = int(intr["image_width"]), int(intr["image_height"])
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    D = np.array([intr["k1"], intr["k2"], intr["k3"], intr["k4"]], np.float64)
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), np.eye(3), balance=balance)
    Wo, Ho = W, H
    if out_width and out_width < W:
        sc = out_width / W
        Wo, Ho = out_width, int(round(H * sc / 2) * 2)
        newK = newK.copy()
        newK[0, :] *= sc
        newK[1, :] *= Ho / H
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (Wo, Ho), cv2.CV_16SC2)
    _FE_MAPS[key] = (m1, m2, float(newK[0, 0]), Wo, Ho)
    return _FE_MAPS[key]


def _wanted_frames(ivs: list[dict], src_fps: float, target_fps: float,
                   min_sec: float, max_ivs: int) -> tuple[list[dict], dict[int, list[tuple[int, int]]]]:
    """Time-sampled frame indices per valid interval. Returns (valid intervals,
    {frame_idx: [(iv_idx, j_within_iv), ...]})."""
    if max_ivs:
        ivs = sorted(ivs, key=lambda s: s["start_sec"] - s["end_sec"])[:max_ivs]
    step = src_fps / target_fps  # >= 1 for all Cat-1 sources (24.46..60 fps)
    wanted: dict[int, list[tuple[int, int]]] = {}
    valid = []
    for s in ivs:
        if (s["end_sec"] - s["start_sec"]) < min_sec:
            continue
        k = len(valid)
        valid.append(s)
        start, end = int(s["start_frame"]), int(s["end_frame"])
        j = 0
        while True:
            f = start + int(round(j * step))
            if f > end:
                break
            wanted.setdefault(f, []).append((k, j))
            j += 1
    return valid, wanted


def _decode_wanted(path: str, wanted: dict[int, list], jpeg_quality: int, transform) -> dict[int, bytes]:
    """Sequential decode with seek-ahead over gaps; returns {frame_idx: jpg_bytes}."""
    cap = cv2.VideoCapture(path)
    got: dict[int, bytes] = {}
    pos = 0  # index of the next frame the decoder will return
    try:
        for f in sorted(wanted):
            if f < pos:
                continue
            if f - pos > SEEK_GAP_FRAMES:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                pos = f
            while pos < f:
                if not cap.grab():
                    return got
                pos += 1
            ok, fr = cap.read()
            if not ok:
                return got
            pos += 1
            if fr.ndim == 2:
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
            fr = transform(fr)
            ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok2:
                got[f] = buf.tobytes()
    finally:
        cap.release()
    return got


def _convert_one(rec: dict, args, fisheye_cfg: dict) -> dict:
    """Fetch + convert ONE video; writes sub-clip tars + returns manifest rows/stats.
    Runs inside the isolation subprocess."""
    frames_root = Path(args.frames_root)
    outputs_root = Path(args.outputs_root)
    work = Path(args.work_dir)
    base = clip_base(rec, args.id_mode)
    src_fps = float(rec["fps"])
    stats = {"subclips": 0, "frames": 0, "skipped_short": 0, "kept_sec_out": 0.0}
    rows_out = []

    fe_key = next((k for k in fisheye_cfg if k in rec["uri"]), None)
    if fe_key:
        m1, m2, focal, Wo, Ho = _fisheye_map(fe_key, fisheye_cfg[fe_key], args.fisheye_balance, args.out_width)
        transform = lambda fr: cv2.remap(fr, m1, m2, cv2.INTER_LINEAR)  # noqa: E731
    else:
        focal = None
        sw = int(rec.get("width") or 0)
        if args.out_width and sw > args.out_width:
            sc = args.out_width / sw
            Wo, Ho = args.out_width, int(round(int(rec["height"]) * sc / 2) * 2)
            transform = lambda fr: cv2.resize(fr, (Wo, Ho), interpolation=cv2.INTER_AREA)  # noqa: E731
        else:
            Wo, Ho = sw, int(rec.get("height") or 0)
            transform = lambda fr: fr  # noqa: E731

    ivs, wanted = _wanted_frames(rec.get("intervals", []), src_fps, args.target_fps,
                                 args.min_interval_sec, args.max_intervals_per_clip)
    stats["skipped_short"] = sum(1 for s in rec.get("intervals", [])
                                 if (s["end_sec"] - s["start_sec"]) < args.min_interval_sec)
    if not wanted:
        return {"rows": [], "stats": stats}

    local = _fetch(rec, work)
    try:
        ffmpeg = _find_ffmpeg()
        if ffmpeg and local.stat().st_size <= args.remux_max_gb * 1e9:
            local = _remux_video_only(local, ffmpeg)
        got = _decode_wanted(str(local), wanted, args.jpeg_quality, transform)
    finally:
        local.unlink(missing_ok=True)

    step = src_fps / args.target_fps
    for k, s in enumerate(ivs):
        fr_idx = []
        j = 0
        start, end = int(s["start_frame"]), int(s["end_frame"])
        while True:
            f = start + int(round(j * step))
            if f > end:
                break
            if f in got:
                fr_idx.append(f)
            j += 1
        if len(fr_idx) < args.min_frames:
            continue
        sub = f"{base}_iv{k:02d}"
        tar_path = frames_root / f"{sub}.tar"
        names, offsets = [], []
        with tarfile.open(tar_path, "w") as tar:
            for j2, f in enumerate(fr_idx):
                name = f"{sub}_f{j2:05d}.image.jpg"
                data = got[f]
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
                names.append(name)
        with tarfile.open(tar_path, "r") as tr:  # byte offsets for direct pread
            for mem in tr:
                if mem.isfile():
                    offsets.append([int(mem.offset_data), int(mem.size)])
        seq = outputs_root / sub
        seq.mkdir(parents=True, exist_ok=True)
        extra = {"adapter": "video_wds", "dataset_name": args.source_id,
                 "source_uri": rec["uri"],
                 "source_fps": src_fps, "recon_fps": args.target_fps,
                 "interval_sec": [s["start_sec"], s["end_sec"]],
                 "interval_score": s.get("score")}
        if focal is not None:
            extra["pinhole_focal"] = round(focal, 4)
            extra["undistort"] = "cv2.fisheye/anycalib-kb4"
            (seq / "est_focal.txt").write_text(f"{focal:.6f}\n")
        desc = ClipDescriptor.from_tar_shard(
            clip_id=sub, clip_name=sub, root_dir=str(frames_root.resolve()),
            seq_folder=str(seq.resolve()), shard_path=str(tar_path.resolve()),
            frame_names=names, frame_offsets=offsets, extra=extra)
        desc.fps = float(args.target_fps)
        desc.width, desc.height = int(Wo), int(Ho)
        rows_out.append(ClipManifestRecord(clip_id=sub, source_id=args.source_id, split=args.split,
                                           descriptor=desc, group_id=str(rec.get("group", ""))))
        stats["subclips"] += 1
        stats["frames"] += len(names)
        stats["kept_sec_out"] += len(names) / args.target_fps
    return {"rows": rows_out, "stats": stats}


def main() -> None:
    args = build_parser().parse_args()
    fisheye_cfg = json.load(open(args.fisheye_intrinsics)) if args.fisheye_intrinsics else {}

    if args.one_video_json:  # ---- isolation worker: one video, then exit ----
        rec = json.loads(Path(args.one_video_json).read_text())
        res = _convert_one(rec, args, fisheye_cfg)
        write_clip_manifest(res["rows"], args.result_prefix + ".manifest.jsonl")
        Path(args.result_prefix + ".stats.json").write_text(json.dumps(res["stats"]))
        return

    frames_root = Path(args.frames_root); frames_root.mkdir(parents=True, exist_ok=True)
    outputs_root = Path(args.outputs_root); outputs_root.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)

    videos = [json.loads(l) for l in open(args.survivors) if l.strip()]
    if args.max_clips:
        videos = videos[: args.max_clips]

    stats = {"clips": 0, "subclips": 0, "frames": 0, "skipped_short": 0,
             "errors": 0, "kept_sec_out": 0.0}
    t0 = time.time()
    manifest_parts = []
    for i, rec in enumerate(videos):
        vj = work / f"_v{i}.json"
        rp = str(work / f"_v{i}.res")
        vj.write_text(json.dumps(rec))
        err = None
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--one_video_json", str(vj), "--result_prefix", rp,
                 "--survivors", "-", "--manifest_out", "-",
                 "--frames_root", args.frames_root, "--outputs_root", args.outputs_root,
                 "--work_dir", args.work_dir, "--source_id", args.source_id,
                 "--split", args.split, "--id_mode", args.id_mode,
                 "--target_fps", str(args.target_fps), "--out_width", str(args.out_width),
                 "--jpeg_quality", str(args.jpeg_quality),
                 "--min_interval_sec", str(args.min_interval_sec),
                 "--min_frames", str(args.min_frames),
                 "--max_intervals_per_clip", str(args.max_intervals_per_clip)]
                + (["--fisheye_intrinsics", args.fisheye_intrinsics] if args.fisheye_intrinsics else [])
                + (["--fisheye_balance", str(args.fisheye_balance)] if args.fisheye_intrinsics else []),
                timeout=args.per_video_timeout)
            if os.path.exists(rp + ".stats.json"):
                st = json.loads(open(rp + ".stats.json").read())
                for k, v in st.items():
                    stats[k] = stats.get(k, 0) + v
                stats["clips"] += 1
                manifest_parts.append(rp + ".manifest.jsonl")
            else:
                err = f"worker_crash rc={proc.returncode}"
        except subprocess.TimeoutExpired:
            err = f"worker_timeout >{args.per_video_timeout}s"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:120]}"
        finally:
            vj.unlink(missing_ok=True)
            for leftover in work.iterdir():  # crashed worker's fetched video
                if leftover.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                    leftover.unlink(missing_ok=True)
        if err:
            stats["errors"] += 1
            print(f"  ERR {rec['uri'].rsplit('/', 1)[-1][:70]}: {err}", flush=True)
        print(f"[{i+1}/{len(videos)}] {rec['uri'].rsplit('/', 1)[-1][:60]} "
              f"subclips_total={stats['subclips']} ({time.time()-t0:.0f}s)", flush=True)

    with open(args.manifest_out, "w") as out:
        for part in manifest_parts:
            with open(part) as f:
                out.write(f.read())
            os.unlink(part)
            sj = part.replace(".manifest.jsonl", ".stats.json")
            if os.path.exists(sj):
                os.unlink(sj)
    stats["wall_sec"] = round(time.time() - t0, 1)
    stats["target_fps"] = args.target_fps
    stats["kept_sec_out"] = round(stats["kept_sec_out"], 1)
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(stats, indent=1))
    print(f"[video-wds] {stats['subclips']} subclips from {stats['clips']} videos, "
          f"{stats['frames']} frames, {stats['errors']} errors -> {args.manifest_out} "
          f"({stats['wall_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
