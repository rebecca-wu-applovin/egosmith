#!/usr/bin/env python
"""Generic Stage-1 probe runner for video-only HOI datasets on GCS.

For a given dataset it:
  1. lists the videos from the dataset's GCS layout (per-dataset shim below),
  2. draws a stratified ~1% sample (capped by --max_videos),
  3. fetches each sampled video locally (gsutil cp, or ranged tar-member read),
  4. runs the SAME Stage-1 gates as production (lib.clip.heuristic_video_clipper:
     Gate A YOLO hands + Gate B optical-flow RANSAC ego-motion + Gate C span merge)
     on up to --windows windows of --window_sec seconds each (long videos are
     window-sampled instead of fully decoded; the kept/sampled ratio is an
     unbiased estimator of the full-video valid-span fraction),
  5. writes out_dir/videos.jsonl (per-video detail) and out_dir/probe_report.json:
       {raw_hours_sampled, kept_hours, hours_fraction, clip_keep_fraction,
        resolution, fps, camera_notes, est_full_cost: {stage1_gpu_h,
        kept_hours_projected, recon_gpu_h, label_usd}}

hours_fraction is THE probe number: fraction of raw wallclock that survives Stage-1.

Usage (egosmith env, GPU box):
  PYTHONPATH=src python scripts/build/probe_dataset.py --dataset hd_epic \
      --out_dir /root/cat1_probes/hd_epic --max_videos 8
Datasets: epic_kitchens_100 | hd_epic | assembly101 | ego4d | holoassist
holoassist additionally needs --tar_index (see /root/cat1_probes/holoassist/index_tar.py).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

warnings.filterwarnings("ignore")

import cv2  # noqa: E402

from lib.clip.heuristic_video_clipper import (  # noqa: E402
    analyze_frame_source_intervals,
    load_clip_config,
    _load_yolo,
)

BUCKET = "gs://foundational-research/hoi-dataset"

# ---- cost model (agreed constants) -------------------------------------------------
STAGE1_GPUH_PER_RAW_H = 0.084
RECON_GPUH_PER_KEPT_H = 2.96
LABEL_USD_PER_7S_SUBCLIP = 0.002

FFPROBE = "/root/miniconda3/envs/egoforce/bin/ffprobe"  # fallback metadata reader


# ====================================================================================
# Source shims: each yields VideoRef(uri, group, meta) and knows how to fetch one.
# ====================================================================================
@dataclass
class VideoRef:
    uri: str            # gs:// path, or "<tar>::<member>" for tar-packed sources
    group: str          # stratification key (participant / session / uid-prefix)
    meta: dict = field(default_factory=dict)


class GsutilShim:
    """Base shim for datasets that are plain video objects on GCS."""

    camera_notes = ""

    def list_videos(self) -> list[VideoRef]:
        raise NotImplementedError

    def fetch(self, ref: VideoRef, dest_dir: Path) -> Path:
        name = ref.uri.rsplit("/", 1)[-1]
        if "." not in name:
            name += ".mp4"
        dest = dest_dir / name
        subprocess.run(["gsutil", "-q", "cp", ref.uri, str(dest)], check=True)
        return dest

    @staticmethod
    def _ls(pattern: str) -> list[str]:
        out = subprocess.run(["gsutil", "ls", pattern], check=True,
                             capture_output=True, text=True).stdout
        return [ln.strip() for ln in out.splitlines() if ln.strip()]


class EpicKitchens100Shim(GsutilShim):
    """EPIC-KITCHENS-100: flat videos/PXX_YY.MP4 (700 videos). GoPro head-mounted."""

    camera_notes = ("GoPro head-mounted, 1080p wide-FOV; no calibration files in bucket "
                    "(annotations/ = narration CSVs; rgb_frames/ + flow_frames/ also present).")

    def list_videos(self) -> list[VideoRef]:
        return [VideoRef(u, group=u.rsplit("/", 1)[-1].split("_")[0])
                for u in self._ls(f"{BUCKET}/EPIC-KITCHENS-100/videos/*.MP4")]


class HdEpicShim(GsutilShim):
    """HD-EPIC: Videos/PXX/*.mp4 (156 videos). Project Aria RGB exports."""

    camera_notes = ("Project Aria glasses RGB (fisheye); bucket has VRS/ originals, "
                    "SLAM-and-Gaze/PXX/{SLAM,GAZE_HAND}, and per-video mp4<->vrs "
                    "timestamp CSVs next to each mp4 -> calibrated intrinsics + GT "
                    "trajectories recoverable later.")

    def list_videos(self) -> list[VideoRef]:
        return [VideoRef(u, group=u.rsplit("/", 2)[-2])
                for u in self._ls(f"{BUCKET}/HD-EPIC/Videos/P*/*.mp4")]


class Assembly101Shim(GsutilShim):
    """Assembly101 egocentric views: recordings/<session>/HMC_*_mono10bit.mp4
    (362 sessions x 4 headset-mounted mono cams). One HMC view per sampled session."""

    camera_notes = ("Ego views are 4x headset-mounted MONOCHROME (mono10bit) cams per "
                    "session; 8 fixed RGB cams (C1xxxx_rgb.mp4) also present but not "
                    "egocentric. Camera calibration published with poses (poses@60fps/, "
                    "AssemblyPoses.zip); annotations/ = action labels.")

    def list_videos(self) -> list[VideoRef]:
        refs = [VideoRef(u, group=u.rsplit("/", 2)[-2])
                for u in self._ls(f"{BUCKET}/Assembly101/recordings/*/HMC_*_mono10bit.mp4")]
        # keep one deterministic HMC view per session (rotate which serial)
        by_sess: dict[str, list[VideoRef]] = defaultdict(list)
        for r in refs:
            by_sess[r.group].append(r)
        out = []
        for i, sess in enumerate(sorted(by_sess)):
            views = sorted(by_sess[sess], key=lambda r: r.uri)
            out.append(views[i % len(views)])
        return out


class Ego4dShim(GsutilShim):
    """Ego4D v1: public/v1/full_scale/<uid> (no extension; 9645 canonical videos)."""

    camera_notes = ("Mixed devices (GoPro / Vuzix / WeeView / Pupil ...), mixed FOV and "
                    "resolution; no per-video calibration in bucket (ego4d.json has "
                    "device metadata). Expect low Stage-1 yield (much non-manipulation "
                    "footage).")

    def list_videos(self) -> list[VideoRef]:
        uris = self._ls(f"{BUCKET}/Ego4D/public/v1/full_scale/*")
        uris = [u for u in uris if not u.endswith("/")]
        return [VideoRef(u, group=u.rsplit("/", 1)[-1][0]) for u in uris]  # stratify by uid hex prefix


class HoloAssistShim(GsutilShim):
    """HoloAssist: single 155 GB video_compress.tar of <session>/Export_py/Video_compress.mp4
    (+ Pose_sync.txt / VideoMp4Timing.txt). Members are fetched with ranged reads using a
    pre-built header index (offset_data,size) -- no full-tar download."""

    camera_notes = ("HoloLens2 RGB (video_compress.tar); cam_info.tar has per-session "
                    "camera info, plus head/hand/eye pose tars (head.tar, hands.tar, "
                    "eyes.tar, imu.tar) and depth (ahat_depth.tar) -> calibration + GT "
                    "head pose recoverable later.")

    TAR = f"{BUCKET}/HoloAssist/video_compress.tar"

    def __init__(self, tar_index: str):
        self.index_path = tar_index

    def list_videos(self) -> list[VideoRef]:
        refs = []
        with open(self.index_path) as f:
            for ln in f:
                rec = json.loads(ln)
                if not rec["name"].lower().endswith(".mp4"):
                    continue
                session = rec["name"].split("/", 1)[0]
                # stratify by capture device suffix in the session name (GoPro/Switch/...)
                device = session.rsplit("-", 1)[-1]
                refs.append(VideoRef(f"{self.TAR}::{rec['name']}", group=device,
                                     meta={"offset": rec["offset_data"], "size": rec["size"],
                                           "session": session}))
        return refs

    def fetch(self, ref: VideoRef, dest_dir: Path) -> Path:
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        dest = dest_dir / (ref.meta["session"] + ".mp4")
        off, size = int(ref.meta["offset"]), int(ref.meta["size"])
        with fs.open(self.TAR[5:], "rb") as f, open(dest, "wb") as w:
            f.seek(off)
            remaining = size
            while remaining > 0:
                chunk = f.read(min(32 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError("short read from tar member")
                w.write(chunk)
                remaining -= len(chunk)
        return dest


def make_shim(dataset: str, args) -> GsutilShim:
    if dataset == "epic_kitchens_100":
        return EpicKitchens100Shim()
    if dataset == "hd_epic":
        return HdEpicShim()
    if dataset == "assembly101":
        return Assembly101Shim()
    if dataset == "ego4d":
        return Ego4dShim()
    if dataset == "holoassist":
        if not args.tar_index or not Path(args.tar_index).is_file():
            raise SystemExit("holoassist needs --tar_index (build with "
                             "/root/cat1_probes/holoassist/index_tar.py)")
        return HoloAssistShim(args.tar_index)
    raise SystemExit(f"unknown dataset {dataset}")


# ====================================================================================
# Stratified sampling
# ====================================================================================
def stratified_sample(refs: list[VideoRef], n: int, seed: int = 0) -> list[VideoRef]:
    rng = random.Random(seed)
    groups: dict[str, list[VideoRef]] = defaultdict(list)
    for r in refs:
        groups[r.group].append(r)
    for g in groups.values():
        rng.shuffle(g)
    picked, gi = [], 0
    keys = sorted(groups)
    while len(picked) < min(n, len(refs)):
        g = groups[keys[gi % len(keys)]]
        if g:
            picked.append(g.pop())
        gi += 1
        if all(not groups[k] for k in keys):
            break
    return picked


# ====================================================================================
# Window-sampled Stage-1 analysis
# ====================================================================================
class _WindowSource:
    """Frame source over [start_frame, start_frame+n) of an open VideoCapture.

    analyze_frame_source_intervals asks for window-local indices i where i%skip==0,
    strictly increasing; skipped frames are grab()'d (decoded but not converted)."""

    def __init__(self, cap, start_frame: int, n_frames: int):
        self._cap = cap
        self._n = n_frames
        self._next = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def __len__(self):
        return self._n

    def get_frame(self, i, rgb=False):
        while self._next < i:
            if not self._cap.grab():
                raise EOFError("decode ended early")
            self._next += 1
        ok, fr = self._cap.read()
        self._next += 1
        if not ok:
            raise EOFError("decode ended early")
        if fr.ndim == 2:
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if rgb else fr


def _video_meta(path: Path) -> tuple[float, int, int, int]:
    """(fps, n_frames, width, height) via cv2, ffprobe fallback."""
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 1 or n <= 0:
        try:
            out = subprocess.run(
                [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=r_frame_rate,nb_frames,width,height,duration", "-of", "json", str(path)],
                capture_output=True, text=True, check=True).stdout
            st = json.loads(out)["streams"][0]
            num, den = st["r_frame_rate"].split("/")
            fps = float(num) / max(1.0, float(den))
            n = int(st.get("nb_frames") or float(st.get("duration", 0)) * fps)
            w, h = int(st["width"]), int(st["height"])
        except Exception:
            pass
    return fps, n, w, h


def probe_video(path: Path, cfg: dict, model, *, windows: int, window_sec: float) -> dict:
    fps, n_frames, w, h = _video_meta(path)
    if fps <= 1 or n_frames <= 0:
        return {"error": "bad_metadata", "fps": fps, "n_frames": n_frames}
    duration = n_frames / fps
    win_frames = int(window_sec * fps)

    # short video -> analyze everything as one window; else evenly-placed windows
    if duration <= windows * window_sec * 1.2:
        plan = [(0, max(1, n_frames - 5))]
    else:
        fracs = [0.08, 0.45, 0.80, 0.20, 0.60, 0.92][:windows]
        plan = []
        for fr in sorted(fracs):
            s = int(fr * n_frames)
            plan.append((s, min(win_frames, n_frames - 5 - s)))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "open_failed"}
    sampled_sec = kept_sec = 0.0
    win_out = []
    try:
        for start, count in plan:
            if count <= 0:
                continue
            src = _WindowSource(cap, start, count)
            try:
                ivs, info = analyze_frame_source_intervals(src, cfg, model=model, fps=fps)
            except EOFError:
                # container frame-count overshoot: count what we actually walked
                info = {"sample_count": src._next // 15, "valid_sample_count": 0}
                ivs = []
                count = src._next
            k = sum(iv.end_sec - iv.start_sec for iv in ivs)
            sampled_sec += count / fps
            kept_sec += k
            win_out.append({"start_frame": start, "frames": count, "kept_sec": round(k, 2),
                            "valid_frac": round(info["valid_sample_count"] / max(1, info["sample_count"]), 3)})
    finally:
        cap.release()

    return {"fps": round(fps, 2), "n_frames": n_frames, "width": w, "height": h,
            "duration_sec": round(duration, 1), "sampled_sec": round(sampled_sec, 1),
            "kept_sec": round(kept_sec, 2),
            "hours_fraction": round(kept_sec / sampled_sec, 4) if sampled_sec else 0.0,
            "windows": win_out}


# ====================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["epic_kitchens_100", "hd_epic", "assembly101", "ego4d", "holoassist"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frac", type=float, default=0.01, help="target fraction of videos to probe")
    ap.add_argument("--max_videos", type=int, default=30)
    ap.add_argument("--min_videos", type=int, default=6)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--window_sec", type=float, default=90.0)
    ap.add_argument("--config", default=str(_REPO / "src/lib/clip/heuristic_clip_config.yaml"))
    ap.add_argument("--detector", default=str(_REPO / "weights/external/detector.pt"))
    ap.add_argument("--tar_index", default="/root/cat1_probes/holoassist/tar_index.jsonl")
    ap.add_argument("--keep_downloads", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "tmp").mkdir(parents=True, exist_ok=True)

    cfg = load_clip_config(args.config)
    model = _load_yolo(args.detector)
    if model is None:
        raise SystemExit(f"YOLO detector not found at {args.detector}")
    try:
        model.to("cuda:0")
    except Exception:
        pass

    shim = make_shim(args.dataset, args)
    refs = shim.list_videos()
    n_total = len(refs)
    n_probe = max(args.min_videos, min(args.max_videos, int(round(args.frac * n_total))))
    picked = stratified_sample(refs, n_probe)
    print(f"[{args.dataset}] {n_total} videos total -> probing {len(picked)} "
          f"(stratified over {len(set(r.group for r in refs))} groups)", flush=True)

    per_video = []
    t0 = time.time()
    with open(out_dir / "videos.jsonl", "w") as w:
        for i, ref in enumerate(picked):
            rec = {"uri": ref.uri, "group": ref.group}
            local = None
            try:
                tf = time.time()
                local = shim.fetch(ref, out_dir / "tmp")
                rec["fetch_sec"] = round(time.time() - tf, 1)
                rec.update(probe_video(local, cfg, model,
                                       windows=args.windows, window_sec=args.window_sec))
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
            finally:
                if local is not None and not args.keep_downloads:
                    local.unlink(missing_ok=True)
            per_video.append(rec)
            w.write(json.dumps(rec) + "\n")
            w.flush()
            print(f"  [{i+1}/{len(picked)}] {ref.uri.rsplit('/', 1)[-1]} "
                  f"hf={rec.get('hours_fraction', 'ERR')} kept={rec.get('kept_sec', '-')}s "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ok = [r for r in per_video if "error" not in r]
    raw_hours_sampled = sum(r["duration_sec"] for r in ok) / 3600.0
    sampled_h = sum(r["sampled_sec"] for r in ok) / 3600.0
    kept_h = sum(r["kept_sec"] for r in ok) / 3600.0
    hours_fraction = kept_h / sampled_h if sampled_h else 0.0
    clip_keep_fraction = (sum(1 for r in ok if r["kept_sec"] > 0) / len(ok)) if ok else 0.0

    est_total_raw_h = raw_hours_sampled / max(1, len(ok)) * n_total
    kept_proj = hours_fraction * est_total_raw_h
    report = {
        "dataset": args.dataset,
        "n_videos_total": n_total,
        "n_videos_probed": len(picked),
        "n_probe_errors": len(per_video) - len(ok),
        "raw_hours_sampled": round(raw_hours_sampled, 2),
        "analyzed_hours_sampled": round(sampled_h, 2),
        "kept_hours": round(kept_h, 3),
        "hours_fraction": round(hours_fraction, 4),
        "clip_keep_fraction": round(clip_keep_fraction, 3),
        "resolution": Counter(f"{r['width']}x{r['height']}" for r in ok).most_common(3),
        "fps": Counter(r["fps"] for r in ok).most_common(3),
        "camera_notes": shim.camera_notes,
        "est_dataset_raw_hours": round(est_total_raw_h, 1),
        "est_full_cost": {
            "stage1_gpu_h": round(STAGE1_GPUH_PER_RAW_H * est_total_raw_h, 1),
            "kept_hours_projected": round(kept_proj, 1),
            "recon_gpu_h": round(RECON_GPUH_PER_KEPT_H * kept_proj, 1),
            "label_usd": round(kept_proj * 3600.0 / 7.0 * LABEL_USD_PER_7S_SUBCLIP, 2),
        },
        "probe_wallclock_sec": round(time.time() - t0, 1),
    }
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
