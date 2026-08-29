#!/usr/bin/env python
"""Segment-level Phase-D re-filter over an already-reconstructed dataset.

Motivation (measured on egoverse_aria): Phase-D decides at CLIP granularity, and
the aria clips were full ~94s episodes — a single glitch event (one wrist-rotation
step, one offscreen streak) anywhere in the episode dropped the entire clip.
Shard 00000: 25/27 dropped, many with presence_ratio 1.0. The recon outputs for
every dropped clip are already on GCS, so segment recovery costs no new GPU recon.

Method (production-equivalent decisions):
  For each shard: for every clip that has recon output but was NOT kept by the
  original Phase-D run, materialize the lightweight seq_folder (same ranged-npz
  trick as phase_d_incremental.py), slice it into non-overlapping fixed windows
  (default 150 frames = 10s @15fps; a short tail merges into the previous window),
  and run the CANONICAL filter (scripts/build/filter_manifest_by_quality.py
  --stages infiller, same flags/gates as the production conveyor) over the window
  population. Windows that pass become new kept sub-clips (clip_id <parent>_sNN);
  the original run's kept rows are carried through UNCHANGED.

Outputs land in a SEPARATE GCS dir (…/filter_run_refilter/_shards) — production
manifests are only replaced later, backup-first, by the closeout step.

Env (same contract as phase_d_incremental.py):
  PHASED_GCS_OUT   recon outputs prefix     (…/egosmith_recon/<ds>/recon/outputs)
  PHASED_GCS_B     phaseB manifests prefix  (…/egosmith_filtered/<ds>/phaseB/_shards)
  PHASED_GCS_FILT  original filter shards   (…/egosmith_filtered/<ds>/filter_run/_shards)
  PHASED_GCS_REFILT output prefix           (…/egosmith_filtered/<ds>/filter_run_refilter/_shards)
  PHASED_WORK      local work dir
  PHASED_SOURCE_FPS / PHASED_MAX_* gates    same defaults as phase_d_incremental
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EGOSMITH_ROOT = os.environ.get("EGOSMITH_ROOT", "/root/egosmith")
sys.path.insert(0, f"{EGOSMITH_ROOT}/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gcsfs  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402

from phase_d_incremental import ranged_npz_extract  # noqa: E402  (same conveyor dir)

GCS_OUT = os.environ["PHASED_GCS_OUT"]
GCS_B = os.environ["PHASED_GCS_B"]
GCS_FILT = os.environ["PHASED_GCS_FILT"]
GCS_REFILT = os.environ["PHASED_GCS_REFILT"]
WORK = Path(os.environ.get("PHASED_WORK", "/root/egosmith_annotations/_refilter_work"))

SOURCE_FPS = os.environ.get("PHASED_SOURCE_FPS", "15")
GATE_WRIST = os.environ.get("PHASED_MAX_WRIST_ROT_STEP", "1.98")
GATE_HAND = os.environ.get("PHASED_MAX_HAND_TRANS_STEP", "0.6")
GATE_FINGER = os.environ.get("PHASED_MAX_FINGER_TRANS_STEP", "0.6")
GATE_CAM_T = os.environ.get("PHASED_MAX_CAM_TRANS_STEP", "0.4")
GATE_CAM_R = os.environ.get("PHASED_MAX_CAM_ROT_STEP", "1.4")

fs = gcsfs.GCSFileSystem()


def _slice_arr(a, s: int, e: int):
    """Slice numpy/tensor along its time axis (axis 1 for (2,T,…) pose arrays,
    axis 0 for (T,…) SLAM arrays), clamping to the actual length."""
    if a.ndim >= 2 and a.shape[0] == 2:      # (2, T, ...) MANO param stacks
        e2 = min(e, a.shape[1])
        return a[:, s:e2]
    e2 = min(e, a.shape[0])                   # (T, ...) traj / tstamp
    return a[s:e2]


def materialize_windows(shard_sfx: str, clip_id: str, rec: dict, base_dir: Path,
                        win: int, min_tail: int) -> list[dict]:
    """Fetch one clip's recon artifacts, write per-window seq_folders, and return
    the corresponding sub-clip manifest rows."""
    pre = f"{GCS_OUT}/shard_{shard_sfx}/{clip_id}"
    npzs = [p for p in fs.ls(f"{pre}/SLAM") if "hawor_slam_w_scale_" in p]
    if not npzs:
        raise FileNotFoundError("no slam npz")
    npz_name = os.path.basename(npzs[0])
    small = ranged_npz_extract(npzs[0])
    tmp_pth = base_dir / f".{clip_id}.world.pth"
    fs.get(f"{pre}/world_space_res.pth", str(tmp_pth))
    world = joblib.load(tmp_pth)
    tmp_pth.unlink()
    T = int(world[0].shape[1])
    n_names = len(rec["descriptor"]["frame_names"])
    T = min(T, n_names)

    # windows: [0,win), [win,2win), … ; a tail shorter than min_tail merges into
    # the previous window (so the last window may be up to win+min_tail-1 long).
    bounds = list(range(0, T, win)) + [T]
    if len(bounds) >= 3 and bounds[-1] - bounds[-2] < min_tail:
        bounds.pop(-2)
    windows = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    cx, cy = np.asarray(small["img_center"]).reshape(-1)[:2].tolist()
    rows = []
    for ii, (s, e) in enumerate(windows):
        sub_id = f"{clip_id}_s{ii:02d}"
        seq = base_dir / sub_id
        (seq / "SLAM").mkdir(parents=True, exist_ok=True)
        L = e - s
        sub_small = dict(small)
        for k in ("traj", "tstamp"):
            if k in sub_small:
                sub_small[k] = _slice_arr(np.asarray(sub_small[k]), s, e)
        np.savez(seq / "SLAM" / f"hawor_slam_w_scale_0_{L}.npz", **sub_small)
        sub_world = [_slice_arr(w if isinstance(w, np.ndarray) else w, s, e) for w in world]
        joblib.dump(sub_world, seq / "world_space_res.pth")
        tdir = seq / f"tracks_0_{L}"
        tdir.mkdir(exist_ok=True)
        (tdir / ".materialized").write_text("segment re-filter stub; parent tracks on GCS\n")

        sub = copy.deepcopy(rec)
        d = sub["descriptor"]
        sub["clip_id"] = sub_id
        d["clip_id"] = sub_id
        d["clip_name"] = sub_id
        d["frame_names"] = d["frame_names"][s:e]
        if d.get("frame_offsets"):
            d["frame_offsets"] = d["frame_offsets"][s:e]
        if d.get("frame_count_override") is not None:
            d["frame_count_override"] = len(d["frame_names"])
        d["seq_folder"] = str(seq)
        d["width"] = int(round(cx * 2))
        d["height"] = int(round(cy * 2))
        d["fps"] = float(d.get("extra", {}).get("recon_fps") or float(SOURCE_FPS))
        d.setdefault("extra", {})["refilter_window"] = [s, e]
        d["extra"]["parent_clip_id"] = clip_id
        rows.append(sub)
    return rows


def process_shard(sfx: str, workers: int, win: int, min_tail: int, keep_work: bool) -> dict:
    t0 = time.time()
    man_raw = fs.cat(f"{GCS_B}/shard_{sfx}.manifest.jsonl").decode()
    recs = {json.loads(l)["clip_id"]: json.loads(l) for l in man_raw.splitlines() if l.strip()}
    prev_rows = [json.loads(l) for l in
                 fs.cat(f"{GCS_FILT}/shard_{sfx}.filtered.jsonl").decode().splitlines() if l.strip()]
    prev_kept = {r["clip_id"] for r in prev_rows}
    have = {os.path.basename(os.path.dirname(p))
            for p in fs.glob(f"{GCS_OUT}/shard_{sfx}/*/world_space_res.pth")}
    todo = sorted((have & set(recs)) - prev_kept)

    base = WORK / sfx
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    all_rows, n_err = [], 0

    def mat(cid):
        try:
            return materialize_windows(sfx, cid, recs[cid], base, win, min_tail)
        except Exception as e:  # noqa: BLE001
            return {"__err__": f"{cid}: {str(e)[:150]}"}

    with ThreadPoolExecutor(workers) as ex:
        for out in ex.map(mat, todo):
            if isinstance(out, dict) and "__err__" in out:
                n_err += 1
            else:
                all_rows.extend(out)

    in_man = base / "input.jsonl"
    with open(in_man, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    out_man = base / "segments.filtered.jsonl"
    rep = base / "segments.report.json"
    if all_rows:
        r = subprocess.run(
            [sys.executable, f"{EGOSMITH_ROOT}/scripts/build/filter_manifest_by_quality.py",
             "--input_manifest", str(in_man), "--output_manifest", str(out_man),
             "--report_out", str(rep), "--stages", "infiller", "--workers", "16",
             "--source_fps", SOURCE_FPS, "--target_fps", "30",
             "--min_presence_ratio", "0.5",
             # Stage-1 certified two hands; segment recovery must not reintroduce
             # single-hand clips (same gate as phase_d_incremental).
             "--min_presence_ratio_per_hand", "0.5",
             "--max_wrist_rotation_step", GATE_WRIST,
             "--max_hand_translation_step", GATE_HAND,
             "--max_finger_translation_step", GATE_FINGER,
             "--max_camera_translation_step", GATE_CAM_T,
             "--max_camera_rotation_step", GATE_CAM_R],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": f"{EGOSMITH_ROOT}/src"})
        if r.returncode != 0 or not out_man.exists():
            raise RuntimeError(f"filter failed: {r.stderr[-400:]}")
    else:
        out_man.write_text("")
        rep.write_text(json.dumps({"total_clips": 0, "kept_clips": 0}))

    kept_rows = [json.loads(l) for l in open(out_man) if l.strip()]
    # combined manifest: original keeps unchanged + newly kept segments
    comb = base / "filtered.jsonl"
    with open(comb, "w") as f:
        for r in prev_rows + kept_rows:
            f.write(json.dumps(r) + "\n")
    fs.put(str(out_man), f"{GCS_REFILT}/shard_{sfx}.segments.filtered.jsonl")
    fs.put(str(rep), f"{GCS_REFILT}/shard_{sfx}.segments.report.json")
    fs.put(str(comb), f"{GCS_REFILT}/shard_{sfx}.filtered.jsonl")

    fps = float(SOURCE_FPS)
    seg_h = sum(len(r["descriptor"]["frame_names"]) for r in kept_rows) / fps / 3600
    stats = {"shard": sfx, "clips_refiltered": len(todo), "windows": len(all_rows),
             "windows_kept": len(kept_rows), "recovered_hours": round(seg_h, 4),
             "prev_kept": len(prev_rows), "mat_errors": n_err,
             "wall_sec": round(time.time() - t0, 1)}
    fs.pipe(f"{GCS_REFILT}/shard_{sfx}.refilter.stats.json", json.dumps(stats).encode())
    if not keep_work:
        shutil.rmtree(base)
    print(f"[{time.strftime('%H:%M:%S')}] shard_{sfx}: clips={len(todo)} windows={len(all_rows)} "
          f"KEPT={len(kept_rows)} (+{seg_h:.2f}h) prev={len(prev_rows)} err={n_err} "
          f"{time.time()-t0:.0f}s", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="", help="comma list / ranges e.g. 0-152; empty = all found")
    ap.add_argument("--workers", type=int, default=24, help="materialization threads")
    ap.add_argument("--window_frames", type=int, default=150, help="segment window length (frames @source fps)")
    ap.add_argument("--min_tail_frames", type=int, default=75, help="tail shorter than this merges into previous window")
    ap.add_argument("--keep_work", action="store_true")
    ap.add_argument("--limit_shards", type=int, default=0)
    args = ap.parse_args()

    if args.shards:
        sfxs = []
        for part in args.shards.split(","):
            if "-" in part:
                a, b = part.split("-")
                sfxs += [f"{i:05d}" for i in range(int(a), int(b) + 1)]
            else:
                sfxs.append(f"{int(part):05d}")
    else:
        sfxs = sorted({os.path.basename(p).split(".")[0].replace("shard_", "")
                       for p in fs.ls(GCS_B) if p.endswith(".manifest.jsonl")})
    if args.limit_shards:
        sfxs = sfxs[:args.limit_shards]

    done = {os.path.basename(p).split(".")[0].replace("shard_", "")
            for p in (fs.ls(GCS_REFILT) if fs.exists(GCS_REFILT) else [])
            if p.endswith(".refilter.stats.json")}
    todo = [s for s in sfxs if s not in done]
    print(f"shards: {len(sfxs)} requested, {len(done)} already done, {len(todo)} to run", flush=True)
    agg = {"windows": 0, "windows_kept": 0, "recovered_hours": 0.0, "clips": 0}
    for s in todo:
        st = process_shard(s, args.workers, args.window_frames, args.min_tail_frames, args.keep_work)
        agg["windows"] += st["windows"]; agg["windows_kept"] += st["windows_kept"]
        agg["recovered_hours"] += st["recovered_hours"]; agg["clips"] += st["clips_refiltered"]
    print("TOTAL:", json.dumps(agg), flush=True)


if __name__ == "__main__":
    main()
