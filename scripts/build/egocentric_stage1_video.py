#!/usr/bin/env python
"""Stage-1 pre-filter over RAW VIDEO datasets on GCS (full pass, no window sampling).

Generalizes egocentric_stage1_mp4.py (tar-of-mp4) to datasets that are plain video
objects (Assembly101 / HD-EPIC / EPIC-KITCHENS-100 / Ego4D) or tar-packed members
fetched by ranged read (HoloAssist). Runs the SAME Stage-1 gates as production
(lib.clip.heuristic_video_clipper: Gate A YOLO hands + Gate B optical-flow RANSAC
ego-motion + Gate C valid-span merge) over the ENTIRE video via a sequential
streaming decoder (memory-safe: no frame accumulation), and records the kept
interval(s) so Phase B can trim.

Input: --videos_list JSONL, one video per line:
  {"uri": "gs://.../video.mp4", "group": "P01"}                     # plain object
  {"uri": "gs://.../pack.tar::member/path.mp4", "group": "ATV",
   "offset": 123, "size": 456, "session": "R090-..."}               # tar member (ranged read)

Output: survivor JSONL (uri, group, fps, duration_sec, kept_sec, intervals[]) +
funnel report JSON (raw_hours, kept_hours, hours_fraction, byte totals).

Usage (egosmith env, GPU):
  PYTHONPATH=src python scripts/build/egocentric_stage1_video.py \
      --videos_list videos_shard.jsonl --out_manifest stage1.kept.jsonl \
      --report_out stage1.report.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# Some GoPro/Aria MP4s interleave large audio/telemetry streams; cv2's default
# packet-read attempt cap (4096) makes grabFrame bail after seconds of video
# (silent partial analysis) — and older FFmpeg builds spin instead. Raise the cap
# BEFORE cv2 import, and additionally remux to video-only when ffmpeg is present.
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", "1000000")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import cv2  # noqa: E402


def _find_ffmpeg() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    for cand in ("/opt/conda/envs/egosmith/bin/ffmpeg", "/opt/conda/bin/ffmpeg",
                 "/root/miniconda3/envs/egoforce/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if Path(cand).is_file():
            return cand
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _remux_video_only(path: Path, ffmpeg: str) -> Path:
    """Stream-copy the first video stream to a clean container (no re-encode).
    Returns the remuxed path, or the original on any failure."""
    out = path.with_name(path.stem + ".vonly.mp4")
    try:
        r = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(path),
                            "-map", "0:v:0", "-c", "copy", str(out)],
                           capture_output=True, timeout=1800)
        if r.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            path.unlink(missing_ok=True)
            return out
    except Exception:  # noqa: BLE001
        pass
    out.unlink(missing_ok=True)
    return path

from lib.clip.heuristic_video_clipper import (  # noqa: E402
    load_clip_config, analyze_frame_source_intervals, _load_yolo,
)


class _SeqSource:
    """Sequential streaming frame source over [0, n) of a video file.

    analyze_frame_source_intervals asks only for indices i where i%skip==0, strictly
    increasing; skipped frames are grab()'d (decoded, not converted). Monochrome
    frames are expanded to BGR. Raises EOFError on container frame-count overshoot;
    caller retries with n clamped to .last_decoded."""

    def __init__(self, path: str, n_frames: int):
        self._cap = cv2.VideoCapture(path)
        self._n = n_frames
        self._next = 0
        self.last_decoded = 0

    def __len__(self):
        return self._n

    def get_frame(self, i, rgb=False):
        while self._next < i:
            if not self._cap.grab():
                raise EOFError("decode ended early")
            self._next += 1
            self.last_decoded = self._next
        ok, fr = self._cap.read()
        self._next += 1
        if not ok:
            raise EOFError("decode ended early")
        self.last_decoded = self._next
        if fr.ndim == 2:
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if rgb else fr

    def release(self):
        self._cap.release()


def _video_meta(path: str):
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, n, w, h


def _fetch(rec: dict, dest_dir: Path) -> Path:
    uri = rec["uri"]
    if "::" in uri:  # tar member, ranged read via gcsfs
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
        name += ".mp4"  # extensionless (Ego4D uids)
    dest = dest_dir / name
    subprocess.run(["gcloud", "storage", "cp", uri, str(dest)],
                   check=True, capture_output=True)
    return dest


def _analyze_full(path: str, cfg: dict, model) -> dict:
    fps, n_frames, w, h = _video_meta(path)
    if fps <= 1 or n_frames <= 0:
        return {"error": "bad_metadata", "fps": fps, "n_frames": n_frames}
    n = max(1, n_frames - 5)  # container counts routinely overshoot by a few frames
    for _attempt in range(2):
        src = _SeqSource(path, n)
        try:
            ivs, info = analyze_frame_source_intervals(src, cfg, model=model, fps=fps)
            break
        except EOFError:
            n = max(1, src.last_decoded - 1)  # clamp to what actually decoded; retry once
            ivs, info = [], {"sample_count": 0, "valid_sample_count": 0}
        finally:
            src.release()
    kept_sec = sum(iv.end_sec - iv.start_sec for iv in ivs)
    analyzed_sec = n / fps
    return {"fps": round(fps, 3), "n_frames": n_frames, "analyzed_frames": n,
            "width": w, "height": h,
            "duration_sec": round(n_frames / fps, 1),
            "analyzed_sec": round(analyzed_sec, 1),
            "kept_sec": round(kept_sec, 2),
            "valid_frac": round(info["valid_sample_count"] / max(1, info["sample_count"]), 3),
            "intervals": [iv.to_dict() for iv in ivs]}


def _run_one(rec: dict, work: Path, cfg_path: str, det_path: str) -> dict:
    """Fetch + analyze ONE video (runs inside the isolation subprocess)."""
    cfg = load_clip_config(cfg_path)
    model = _load_yolo(det_path)
    if model is None:
        raise SystemExit(f"YOLO detector not found at {det_path}")
    try:
        model.to("cuda:0")
    except Exception:  # noqa: BLE001
        pass
    local = None
    out = {"uri": rec["uri"], "group": rec.get("group", "")}
    try:
        tf = time.time()
        local = _fetch(rec, work)
        out["fetched_bytes"] = local.stat().st_size
        out["fetch_sec"] = round(time.time() - tf, 1)
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            local = _remux_video_only(local, ffmpeg)
            out["remuxed"] = local.name.endswith(".vonly.mp4")
        ta = time.time()
        out.update(_analyze_full(str(local), cfg, model))
        out["analyze_sec"] = round(time.time() - ta, 1)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        if local is not None:
            local.unlink(missing_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_list", required=True, help="JSONL of videos (this shard)")
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--config", default=str(_REPO / "src/lib/clip/heuristic_clip_config.yaml"))
    ap.add_argument("--detector", default=str(_REPO / "weights/external/detector.pt"))
    ap.add_argument("--work_dir", default="/tmp/s1v")
    ap.add_argument("--limit", type=int, default=0, help=">0: cap videos (smoke)")
    ap.add_argument("--per_video_timeout", type=int, default=10800,
                    help="seconds before a single video's worker is killed")
    ap.add_argument("--one_video_json", default="", help="internal: single-video worker mode")
    ap.add_argument("--result_out", default="", help="internal: single-video result path")
    args = ap.parse_args()

    if args.one_video_json:  # ---- isolation worker mode: one video, then exit ----
        rec = json.loads(Path(args.one_video_json).read_text())
        out = _run_one(rec, Path(args.work_dir), args.config, args.detector)
        Path(args.result_out).write_text(json.dumps(out))
        return

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    videos = [json.loads(ln) for ln in open(args.videos_list) if ln.strip()]
    if args.limit:
        videos = videos[: args.limit]

    kept, dropped = [], []
    raw_sec = analyzed_sec = kept_sec = 0.0
    fetched_bytes = 0
    t0 = time.time()
    with open(args.out_manifest, "w") as w:
        for i, rec in enumerate(videos):
            # Each video runs in its OWN subprocess: native segfaults in decode
            # (cv2/ffmpeg on odd containers) cost one video, not the shard.
            vj, vr = work / f"_v{i}.json", work / f"_v{i}.result.json"
            vj.write_text(json.dumps(rec))
            vr.unlink(missing_ok=True)
            out = {"uri": rec["uri"], "group": rec.get("group", "")}
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "--one_video_json", str(vj),
                     "--result_out", str(vr), "--work_dir", str(work),
                     "--config", args.config, "--detector", args.detector,
                     "--videos_list", "-", "--out_manifest", "-", "--report_out", "-"],
                    timeout=args.per_video_timeout)
                if vr.is_file():
                    out = json.loads(vr.read_text())
                else:
                    out["error"] = f"worker_crash: rc={proc.returncode}, no result"
            except subprocess.TimeoutExpired:
                out["error"] = f"worker_timeout: >{args.per_video_timeout}s"
            except Exception as e:  # noqa: BLE001
                out["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            finally:
                vj.unlink(missing_ok=True)
                vr.unlink(missing_ok=True)
                for leftover in work.iterdir():  # crashed worker's fetched video
                    if leftover.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                        leftover.unlink(missing_ok=True)
            fetched_bytes += out.pop("fetched_bytes", 0)
            if "error" in out:
                dropped.append({"uri": out["uri"],
                                "reason": "error_" + out["error"].split(":")[0].strip()})
            else:
                raw_sec += out["duration_sec"]
                analyzed_sec += out["analyzed_sec"]
                kept_sec += out["kept_sec"]
                if out["intervals"]:
                    w.write(json.dumps(out) + "\n")
                    w.flush()
                    kept.append(out)
                else:
                    dropped.append({"uri": out["uri"], "reason": "no_valid_span",
                                    "valid_frac": out["valid_frac"]})
            print(f"[{i+1}/{len(videos)}] {rec['uri'].rsplit('/', 1)[-1][:60]} "
                  f"kept={out.get('kept_sec', 'ERR')}s of {out.get('analyzed_sec', '-')}s "
                  f"({time.time()-t0:.0f}s)", flush=True)

    report = {
        "videos": len(videos), "kept_videos": len(kept), "dropped_videos": len(dropped),
        "raw_hours": round(raw_sec / 3600.0, 3),
        "analyzed_hours": round(analyzed_sec / 3600.0, 3),
        "kept_hours": round(kept_sec / 3600.0, 4),
        "hours_fraction": round(kept_sec / analyzed_sec, 4) if analyzed_sec else 0.0,
        "fetched_gb": round(fetched_bytes / 1e9, 2),
        "wallclock_sec": round(time.time() - t0, 1),
        "drop_reasons": dict(Counter(d["reason"].split(":")[0] for d in dropped)),
    }
    Path(args.report_out).write_text(json.dumps(report, indent=1))
    print(f"[stage1-video] {len(kept)}/{len(videos)} videos kept, "
          f"{report['kept_hours']}h of {report['analyzed_hours']}h "
          f"(hf={report['hours_fraction']}) -> {args.out_manifest}", flush=True)


if __name__ == "__main__":
    main()
