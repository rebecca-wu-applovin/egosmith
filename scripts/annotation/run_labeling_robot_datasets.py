#!/usr/bin/env python
"""Production LLM labeling over the Cat-4 ROBOT datasets (QC-kept episodes).

Same frozen config as run_labeling_gt_datasets.py (gpt-5-mini, effort=medium, v4
prompt, P2 sampling: 1024px, detail=high, ~3fps, 12-40 frames) but robot episodes
are native_episode records, not frame tars, so each dataset gets a frame fetcher:

- dexora           per-episode LeRobot v2.1 mp4 (observation.images.top, front fallback)
- trex             LeRobot v3.0 concatenated mp4s; episodes grouped per video file and
                   sliced by meta/episodes from/to timestamps (head_left camera).
                   Episodes with an EMPTY head video slice are skipped + logged.
- dexwild          jpg frames inside the robot HDF5 split tars on GCS (ranged reads;
                   preferred camera order: right_pinky_cam, right_thumb_cam, left_*)
- hrdexdb_allegro  <gcs_prefix>/vid/<serial>.mp4 (fixed first-sorted serial per episode)
- realdex          cam0/rgb/image_raw jpgs ranged-read straight out of the object zips

Output: <out_dir>/<ds>.robot.annotations.jsonl, one row per episode, schema identical
to the GT runner. Resume-safe on clip_id. NEVER commit the API key; env only.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/root/egosmith/src")
sys.path.insert(0, "/root/egosmith/scripts/build")

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(f"{_HERE}/annotation_harness_reference.py").read().split("clips = pick_clips(10)")[0]
ns = {}
exec(compile(_src, "ablation_harness", "exec"), ns)
ns["PROMPT"] = open("/root/egosmith/src/lib/annotation/prompts/with_clip/annotation_general_clip_v4.txt").read()
MODEL, EFFORT, PRICE, PROMPT = ns["MODEL"], ns["EFFORT"], ns["PRICE"], ns["PROMPT"]
client = ns["client"]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

GCS = "gs://foundational-research/hoi-dataset/egosmith_filtered"
OUT_DIR = "/root/egosmith_annotations"
PX, DETAIL, TARGET_FPS, FMIN, FMAX_DEFAULT = 1024, "high", 3.0, 12, 40
PROMPT_VERSION = "annotation_general_clip_v4"

_write_lock = threading.Lock()
_stats_lock = threading.Lock()
_stats = {"done": 0, "err": 0, "skip": 0, "cost": 0.0, "t0": time.time()}


def _enc(frame_bgr):
    h, w = frame_bgr.shape[:2]
    s = PX / max(h, w)
    if s < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _sample_idx(total, nframes):
    if total <= nframes:
        return list(range(total))
    return sorted({round(i * (total - 1) / (nframes - 1)) for i in range(nframes)})


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _frames_from_video(path, nframes, t0=0.0, t1=None):
    """Sample nframes uniformly from [t0, t1) of a video file -> [(t_rel, jpg_bytes)].

    cv2 first; AV1 files (e.g. every Dexora mp4) fall back to imageio-ffmpeg's
    dav1d decoder via a select-filter jpg dump.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(round(t0 * fps))
    f1 = int(round(t1 * fps)) if t1 is not None else total
    f1 = min(f1, total) if total > 0 else f1
    span = max(f1 - f0, 1)
    idx = _sample_idx(span, nframes)
    out = []
    for k in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0 + k)
        ok, fr = cap.read()
        if not ok:
            continue
        out.append((k / fps, _enc(fr)))
    cap.release()
    if not out:  # cv2 could not decode (AV1 etc.) -> ffmpeg/dav1d fallback
        with tempfile.TemporaryDirectory(prefix="ffdec_") as td:
            sel = "+".join(f"eq(n\\,{f0 + k})" for k in idx)
            r = subprocess.run(
                [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path),
                 "-vf", f"select='{sel}'", "-vsync", "vfr", "-q:v", "2",
                 os.path.join(td, "f%04d.jpg")],
                capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg decode failed: {r.stderr.decode()[-200:]}")
            jpgs = sorted(Path(td).glob("f*.jpg"))
            for k, p in zip(idx, jpgs):
                img = cv2.imread(str(p))
                if img is not None:
                    out.append((k / fps, _enc(img)))
    return out, span / fps


def _gcs_cp(url, dest):
    r = subprocess.run(["gcloud", "storage", "cp", url, dest], capture_output=True)
    if r.returncode != 0 or not os.path.exists(dest):
        raise RuntimeError(f"gcs cp failed: {url}: {r.stderr.decode()[-200:]}")


# ---------------------------------------------------------------- dexora
def fetch_dexora(rec, nframes, scratch):
    ex = rec["descriptor"]["extra"]
    idx = ex["episode_index"]
    dest = os.path.join(scratch, f"dexora_{os.getpid()}_{threading.get_ident()}.mp4")
    last = None
    for cam in ("observation.images.top", "observation.images.front"):
        url = f'{ex["gcs_prefix"]}/videos/chunk-000/{cam}/episode_{idx:06d}.mp4'
        try:
            _gcs_cp(url, dest)
            break
        except RuntimeError as e:
            last = e
    else:
        raise last
    try:
        return _frames_from_video(dest, nframes)
    finally:
        if os.path.exists(dest):
            os.unlink(dest)


# ---------------------------------------------------------------- trex
_TREX_META = {}
_trex_lock = threading.Lock()
_TREX_CAM = "observation.images.head_left"


def _trex_meta():
    with _trex_lock:
        if _TREX_META:
            return _TREX_META
        import gcsfs
        import pyarrow.parquet as pq
        fs = gcsfs.GCSFileSystem()
        cols = ["episode_index", f"videos/{_TREX_CAM}/chunk_index", f"videos/{_TREX_CAM}/file_index",
                f"videos/{_TREX_CAM}/from_timestamp", f"videos/{_TREX_CAM}/to_timestamp"]
        for p in sorted(fs.glob("foundational-research/hoi-dataset/T-Rex/meta/episodes/chunk-*/file-*.parquet")):
            tb = pq.ParquetFile(fs.open(p, "rb")).read(columns=cols).to_pylist()
            for r in tb:
                _TREX_META[int(r["episode_index"])] = (
                    int(r[f"videos/{_TREX_CAM}/chunk_index"]), int(r[f"videos/{_TREX_CAM}/file_index"]),
                    float(r[f"videos/{_TREX_CAM}/from_timestamp"]), float(r[f"videos/{_TREX_CAM}/to_timestamp"]))
        return _TREX_META


_trex_vid_cache_lock = threading.Lock()
_trex_vid_locks = {}


def fetch_trex(rec, nframes, scratch):
    ex = rec["descriptor"]["extra"]
    idx = ex["episode_index"]
    meta = _trex_meta()
    if idx not in meta:
        raise RuntimeError("no head_left metadata")
    chunk, fidx, t0, t1 = meta[idx]
    if t1 - t0 < 0.5:
        raise RuntimeError(f"empty_head_video ({t1 - t0:.2f}s)")
    dest = os.path.join(scratch, f"trex_head_c{chunk:03d}_f{fidx:03d}.mp4")
    with _trex_vid_cache_lock:
        lock = _trex_vid_locks.setdefault(dest, threading.Lock())
    with lock:  # one download per shared video file; keep for reuse (LRU by scratch size below)
        if not os.path.exists(dest):
            _gcs_cp(f"gs://foundational-research/hoi-dataset/T-Rex/videos/{_TREX_CAM}/"
                    f"chunk-{chunk:03d}/file-{fidx:03d}.mp4", dest)
    return _frames_from_video(dest, nframes, t0=t0, t1=t1)


# ---------------------------------------------------------------- dexwild
_dxw_lock = threading.Lock()
_dxw_files = {}


def _dexwild_h5(prefix):
    """One shared read handle per split; h5py isn't thread-safe -> guard with _dxw_lock."""
    from robot_episode_qc import _MultiPartGCSFile
    import gcsfs
    import h5py
    if prefix not in _dxw_files:
        fs = gcsfs.GCSFileSystem()
        parts = sorted(p for p in fs.ls(prefix[len("gs://"):]) if ".part_" in p)
        _dxw_files[prefix] = h5py.File(_MultiPartGCSFile(fs, parts), "r")
    return _dxw_files[prefix]


_DXW_CAM_ORDER = ("right_pinky_cam", "right_thumb_cam", "left_pinky_cam", "left_thumb_cam")


def fetch_dexwild(rec, nframes, scratch):
    ex = rec["descriptor"]["extra"]  # native{} is flattened into extra by the QC writer
    src, key = ex["source"], ex["hdf5_group"]
    fps = float(rec["descriptor"].get("fps") or 30.0)
    with _dxw_lock:
        h5 = _dexwild_h5(src)
        g = h5[key]
        cam = next((c for c in _DXW_CAM_ORDER if c in g), None)
        if cam is None:
            raise RuntimeError(f"no camera group in {key}: {sorted(g.keys())}")
        names = sorted(n for n in g[cam] if str(n).endswith(".jpg"))
        # names are ns timestamps; datasets are decoded (H,W,3) uint8 arrays (RGB)
        t_ns0 = int(Path(names[0]).stem)
        dur = (int(Path(names[-1]).stem) - t_ns0) / 1e9 if len(names) > 1 else len(names) / fps
        out = []
        for i in _sample_idx(len(names), nframes):
            arr = np.asarray(g[cam][names[i]])
            if arr.ndim != 3 or arr.dtype != np.uint8:
                continue
            t_rel = (int(Path(names[i]).stem) - t_ns0) / 1e9
            out.append((t_rel, _enc(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))))
    return out, dur


# ---------------------------------------------------------------- hrdexdb
def fetch_hrdexdb(rec, nframes, scratch):
    ex = rec["descriptor"]["extra"]
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    vids = sorted(p for p in fs.ls(ex["gcs_prefix"][len("gs://"):] + "/vid") if p.endswith(".mp4"))
    if not vids:
        raise RuntimeError("no vid/*.mp4")
    dest = os.path.join(scratch, f"hrdexdb_{os.getpid()}_{threading.get_ident()}.mp4")
    _gcs_cp("gs://" + vids[0], dest)
    try:
        return _frames_from_video(dest, nframes)
    finally:
        if os.path.exists(dest):
            os.unlink(dest)


# ---------------------------------------------------------------- realdex
_rdx_lock = threading.Lock()
_rdx_namelists = {}


def fetch_realdex(rec, nframes, scratch):
    ex = rec["descriptor"]["extra"]  # native{} is flattened into extra by the QC writer
    zip_url, seq = ex["gcs_zip"], ex["sequence"]
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    with _rdx_lock:
        if zip_url not in _rdx_namelists:
            f = fs.open(zip_url[len("gs://"):], "rb")
            zf = zipfile.ZipFile(f)
            _rdx_namelists[zip_url] = zf
        zf = _rdx_namelists[zip_url]
    frames = sorted(
        (n for n in zf.namelist() if f"/{seq}/cam0/rgb/image_raw/" in n and n.endswith(".jpg")),
        key=lambda n: int(Path(n).stem))
    if not frames:
        raise RuntimeError(f"no cam0 rgb frames for {seq}")
    picks = [frames[i] for i in _sample_idx(len(frames), nframes)]
    out = []
    with _rdx_lock:
        for i, n in zip(_sample_idx(len(frames), nframes), picks):
            raw = zf.read(n)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            out.append((i / 30.0, _enc(img)))
    return out, len(frames) / 30.0


FETCHERS = {
    "dexora": fetch_dexora,
    "trex": fetch_trex,
    "dexwild": fetch_dexwild,
    "hrdexdb_allegro": fetch_hrdexdb,
    "realdex": fetch_realdex,
}


def build_content(frames, dur):
    content = [{"type": "text", "text": PROMPT +
                f"\n\n## This video\nDuration: {dur:.2f} seconds. {len(frames)} frames sampled uniformly; each "
                f"frame is preceded by its timestamp. Use them to set accurate start/end within [0, {dur:.2f}]."}]
    for t, jpg in frames:
        content.append({"type": "text", "text": f"t = {t:.2f}s"})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                                      "detail": DETAIL}})
    return content


def annotate(ds, rec_json, out_f, err_f, scratch, fmax, dry):
    rec = json.loads(rec_json)
    clip_id = rec["clip_id"]
    try:
        fps = float(rec["descriptor"].get("fps") or 30.0)
        n_over = rec["descriptor"].get("frame_count_override") or 0
        dur_est = n_over / fps if n_over else 15.0
        nf = max(FMIN, min(fmax, round(dur_est * TARGET_FPS)))
        frames, dur = FETCHERS[ds](rec, nf, scratch)
        if not frames:
            raise RuntimeError("no frames fetched")
        content = build_content(frames, dur)
        if dry:
            approx_tok = sum(len(j) for _, j in frames) // 3
            with _write_lock:
                out_f.write(json.dumps({"clip_id": clip_id, "dry": True, "frames": len(frames),
                                        "jpeg_bytes": sum(len(j) for _, j in frames),
                                        "approx_prompt_tokens": approx_tok}) + "\n")
                out_f.flush()
            with _stats_lock:
                _stats["done"] += 1
            return

        last_err = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": content}],
                    reasoning_effort=EFFORT)
                txt = (resp.choices[0].message.content or "").strip()
                if txt.startswith("```"):
                    txt = txt.split("```", 2)[1]
                    txt = txt[4:] if txt.startswith("json") else txt
                parsed = json.loads(txt)
                segs = parsed.get("segments", parsed) if isinstance(parsed, dict) else parsed
                assert isinstance(segs, list) and segs, "no segments"
                for s in segs:
                    li = s.get("language_instructions") or {}
                    assert li.get("level1"), "missing level1"
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(60, 2 ** attempt * 3))
        else:
            raise RuntimeError(f"5 attempts failed: {last_err}")

        u = resp.usage
        cost = u.prompt_tokens * PRICE[0] / 1e6 + u.completion_tokens * PRICE[1] / 1e6
        row = {"clip_id": clip_id, "dataset": ds, "model": MODEL,
               "prompt": PROMPT_VERSION, "effort": EFFORT,
               "config": {"px": PX, "detail": DETAIL, "fps": TARGET_FPS, "frames_sent": len(frames)},
               "duration_sec": round(dur, 2),
               "annotation": parsed if isinstance(parsed, dict) else {"segments": segs},
               "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens},
               "cost_usd": round(cost, 6)}
        with _write_lock:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
        with _stats_lock:
            _stats["done"] += 1
            _stats["cost"] += cost
            if _stats["done"] % 200 == 0:
                el = time.time() - _stats["t0"]
                print(f'[{time.strftime("%H:%M:%S")}] done={_stats["done"]} err={_stats["err"]} '
                      f'skip={_stats["skip"]} ${_stats["cost"]:.2f} rate={_stats["done"]/el*3600:.0f}/h', flush=True)
    except Exception as e:  # noqa: BLE001
        is_skip = "empty_head_video" in str(e)
        with _write_lock:
            err_f.write(json.dumps({"clip_id": clip_id, "dataset": ds,
                                    "error": f"{type(e).__name__}: {e}"[:400],
                                    "skip": is_skip}) + "\n")
            err_f.flush()
        with _stats_lock:
            _stats["skip" if is_skip else "err"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--datasets", default="trex,dexora,dexwild,hrdexdb_allegro,realdex")
    ap.add_argument("--limit", type=int, default=0, help="per-dataset cap (pilot)")
    ap.add_argument("--fmax", type=int, default=FMAX_DEFAULT)
    ap.add_argument("--manifest", action="append", default=[],
                    metavar="DS=PATH", help="override manifest source (local path) for a dataset")
    ap.add_argument("--dry_run", action="store_true", help="fetch frames + report sizes, no API calls")
    ap.add_argument("--out_suffix", default="robot")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.manifest)
    os.makedirs(OUT_DIR, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="robot_label_")
    tasks = []
    for ds in args.datasets.split(","):
        out_path = f"{OUT_DIR}/{ds}.{args.out_suffix}.annotations.jsonl"
        done = set()
        if os.path.exists(out_path):
            for l in open(out_path):
                try:
                    done.add(json.loads(l)["clip_id"])
                except Exception:  # noqa: BLE001
                    pass
        skip_ids = set()
        err_path = f"{OUT_DIR}/{ds}.{args.out_suffix}.errors.jsonl"
        if os.path.exists(err_path):
            for l in open(err_path):
                try:
                    r = json.loads(l)
                    if r.get("skip"):
                        skip_ids.add(r["clip_id"])
                except Exception:  # noqa: BLE001
                    pass
        if ds in overrides:
            man = open(overrides[ds]).read()
        else:
            man = subprocess.run(["gcloud", "storage", "cat",
                                  f"{GCS}/{ds}/filter_run/clip_manifest.filtered.jsonl"],
                                 capture_output=True, text=True).stdout
        out_f = open(out_path, "a")
        err_f = open(err_path, "a")
        n = 0
        for line in man.splitlines():
            if not line.strip():
                continue
            cid = json.loads(line)["clip_id"]
            if cid in done or cid in skip_ids:
                continue
            tasks.append((ds, line, out_f, err_f, scratch, args.fmax, args.dry_run))
            n += 1
            if args.limit and n >= args.limit:
                break
        print(f"{ds}: {n} to annotate ({len(done)} done, {len(skip_ids)} known-skips)", flush=True)

    print(f"TOTAL {len(tasks)} episodes, {args.workers} workers, model={MODEL} effort={EFFORT} "
          f"prompt={PROMPT_VERSION} fmax={args.fmax} dry={args.dry_run}", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda t: annotate(*t), tasks))
    el = time.time() - _stats["t0"]
    print(f'FINISHED done={_stats["done"]} err={_stats["err"]} skip={_stats["skip"]} '
          f'cost=${_stats["cost"]:.2f} wall={el/3600:.2f}h', flush=True)


if __name__ == "__main__":
    main()
