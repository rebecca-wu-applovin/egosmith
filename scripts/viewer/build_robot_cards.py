#!/usr/bin/env python
"""Video-only viewer cards for Cat-4 robot datasets (trex/dexora/dexwild/hrdexdb_allegro/realdex).

Per dataset: sample up to N kept episodes from
gs://.../egosmith_filtered/<ds>/filter_run/clip_manifest.filtered.jsonl, fetch the episode's
RGB media in its native layout, write spec-compliant mp4s (libx264/yuv420p/crf26/faststart/-an,
h<=480) to /root/viewer_work/<ds>/clips/ + cards.json.

LLM annotations (filter_run/annotations.v4.jsonl, keyed by clip_id) are joined into every
card's `annotation` field — build_viewer's standard L1-L4 segment rendering takes over from
extra_rows when present. `--join-only` re-joins annotations into an existing cards.json
without re-rendering any clips (dexora/hrdexdb_allegro/realdex refresh path).

Sampling biases to annotated episodes. trex additionally includes telemetry-repaired
episodes (metadata.telemetry_repair, badge via note); dexwild stratifies across its five
task splits (clothes/florist/pour/spray/toy).
"""
import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
import zipfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "build"))

import cv2  # noqa: E402
import gcsfs  # noqa: E402
import numpy as np  # noqa: E402

from render_clip_card import render_mp4_interval, _writer  # noqa: E402
from robot_episode_qc import _MultiPartGCSFile  # noqa: E402

B = "foundational-research/hoi-dataset"
fs = gcsfs.GCSFileSystem()

ROBOT_FACTS = {
    "trex": [["robot", "Dexmate Vega-1 + 2x SharpaWave hands"],
             ["dof", "58 (2x7-DoF arm + 2x22-DoF hand)"]],
    "dexora": [["robot", "AIRBOT Play bimanual"],
               ["dof", "39 (2x6 arm + 2x12 hand + 2 head + 1 spine)"]],
    "dexwild": [["robot", "LEAP Hand V2 bimanual on xArm"],
                ["dof", "17 per hand (LEAP V2)"]],
    "hrdexdb_allegro": [["robot", "Allegro Hand 16-DoF + 6-DoF arm"],
                        ["dof", "16 hand + 6 arm"]],
    "realdex": [["robot", "Shadow Dexterous Hand + UR arm"],
                ["dof", "24 hand + 6 arm"]],
}


def read_manifest(ds):
    raw = fs.open(f"{B}/egosmith_filtered/{ds}/filter_run/clip_manifest.filtered.jsonl").read()
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]


def read_annotations(ds):
    """clip_id -> annotation dict ({segments:[...]}) from filter_run/annotations.v4.jsonl."""
    path = f"{B}/egosmith_filtered/{ds}/filter_run/annotations.v4.jsonl"
    if not fs.exists(path):
        return {}
    raw = fs.open(path).read().decode()
    out = {}
    for l in raw.splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("annotation"):
                out[r["clip_id"]] = r["annotation"]
    return out


def join_annotations(cards, anns):
    """Attach LLM annotations to kept cards by clip_id; returns #joined."""
    n = 0
    for c in cards:
        if c.get("section") == "dropped":
            continue
        a = anns.get(c["clip_id"])
        if a:
            c["annotation"] = a
            n += 1
    return n


def gate_count(ds):
    try:
        rep = json.loads(fs.open(f"{B}/egosmith_filtered/{ds}/filter_run/report.json").read())
        fn = rep.get("funnel") or rep.get("funnel_combined") or {}
        return len(fn) or None
    except Exception:  # noqa: BLE001
        return None


def qc_row(ds, ngates):
    return ["QC", f"passed {ngates}/{ngates} gates"] if ngates else ["QC", "passed all gates"]


def mk_card(cid, dur, fps, extra_rows, note=None):
    return dict(clip_id=cid, video=f"clips/{cid}.mp4", status="video_only", note=note,
                meta={"dur_s": round(float(dur), 2), "fps": round(float(fps), 2) if fps else None},
                annotation=None, extra_rows=extra_rows)


def encode_frames(frames_rgb, fps, out_mp4, max_h=480):
    """RGB frame list -> spec mp4 (h capped at 480, dims multiple of 16 so imageio_ffmpeg
    never inserts its own -vf scale). One retry on transient BrokenPipe."""
    h0, w0 = frames_rgb[0].shape[:2]
    if h0 > max_h:
        w0 = int(round(w0 * max_h / h0))
        h0 = max_h
    w0, h0 = max(16, w0 // 16 * 16), max(16, h0 // 16 * 16)
    for attempt in (0, 1):
        try:
            wr = _writer(out_mp4, w0, h0, fps)
            for f in frames_rgb:
                if f.shape[0] != h0 or f.shape[1] != w0:
                    f = cv2.resize(f, (w0, h0), interpolation=cv2.INTER_AREA)
                wr.send(np.ascontiguousarray(f))
            wr.close()
            return
        except OSError:
            if attempt:
                raise
            time.sleep(2)


# ---------------------------------------------------------------- trex
def build_trex(recs, clips_dir, n, seed, ngates, ann_ids=frozenset()):
    import pandas as pd
    rng = random.Random(seed)
    ep = pd.read_parquet(fs.open(f"{B}/T-Rex/meta/episodes/chunk-000/file-000.parquet"))
    cam = "observation.images.head_left"
    ep = ep.set_index("episode_index")
    # A large share of the mirrored head_left video objects are 0-byte: restrict
    # sampling to episodes whose video file actually has bytes.
    valid_files = set()
    for chunk_dir in fs.ls(f"{B}/T-Rex/videos/{cam}"):
        for o in fs.ls(chunk_dir, detail=True):
            if o["name"].endswith(".mp4") and o["size"] > 0:
                ci = int(chunk_dir.rsplit("-", 1)[-1])
                fi = int(os.path.basename(o["name"]).split("-")[-1].split(".")[0])
                valid_files.add((ci, fi))
    print(f"[trex] non-empty video files: {len(valid_files)}", flush=True)

    def file_key(r):
        row = ep.loc[r["descriptor"]["extra"]["episode_index"]]
        return (int(row[f"videos/{cam}/chunk_index"]), int(row[f"videos/{cam}/file_index"]))

    def is_repaired(r):
        return bool((r.get("metadata") or {}).get("telemetry_repair"))

    pool = [r for r in recs if file_key(r) in valid_files]
    print(f"[trex] episodes on non-empty files: {len(pool)}/{len(recs)}", flush=True)
    # bias to annotated episodes; guarantee a slice of telemetry-repaired ones
    ann_pool = [r for r in pool if r["clip_id"] in ann_ids] or pool
    repaired = [r for r in ann_pool if is_repaired(r)] or [r for r in pool if is_repaired(r)]
    n_rep = min(max(5, n // 5), len(repaired), n)
    samples = rng.sample(repaired, n_rep)
    chosen = {r["clip_id"] for r in samples}
    rest = [r for r in ann_pool if r["clip_id"] not in chosen and not is_repaired(r)]
    samples += rng.sample(rest, min(n - len(samples), len(rest)))
    print(f"[trex] sampled {len(samples)} ({n_rep} telemetry-repaired, "
          f"{sum(1 for r in samples if r['clip_id'] in ann_ids)} annotated)", flush=True)
    by_file = {}
    for r in samples:
        idx = r["descriptor"]["extra"]["episode_index"]
        row = ep.loc[idx]
        key = (int(row[f"videos/{cam}/chunk_index"]), int(row[f"videos/{cam}/file_index"]))
        by_file.setdefault(key, []).append(
            (r, float(row[f"videos/{cam}/from_timestamp"]), float(row[f"videos/{cam}/to_timestamp"])))
    cards = []
    for (ci, fi), grp in sorted(by_file.items()):
        if len(cards) >= n:
            break
        src = f"{B}/T-Rex/videos/{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4"
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=clips_dir.parent) as tmp:
            fs.get(src, tmp.name)
            for r, t0, t1 in grp:
                if len(cards) >= n:
                    break
                cid = r["clip_id"]
                dur = r["metadata"]["qc_metrics"]["duration_s"]
                try:
                    render_mp4_interval(tmp.name, clips_dir / f"{cid}.mp4", t0, t1 - t0)
                except Exception as e:  # noqa: BLE001
                    print(f"[trex] SKIP {cid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    continue
                task = (r["metadata"].get("tasks") or [""])[0]
                rows = [["episode", cid], ["duration", f"{dur:.1f}s"], qc_row("trex", ngates)]
                rows += ROBOT_FACTS["trex"]
                if task:
                    rows.append(["task", task[:160]])
                note = None
                tr = (r.get("metadata") or {}).get("telemetry_repair")
                if tr:
                    note = (f"telemetry-repaired: {tr.get('n_windows', '?')} stall windows "
                            f"interpolated ({tr.get('method', '?')})")
                    rows.append(["telemetry", "repaired"])
                cards.append(mk_card(cid, dur, r["descriptor"]["fps"], rows, note=note))
                print(f"[trex] {len(cards)} {cid}", flush=True)
    return cards


# ---------------------------------------------------------------- dexora
def build_dexora(recs, clips_dir, n, seed, ngates, ann_ids=frozenset()):
    rng = random.Random(seed)
    samples = rng.sample(recs, min(n, len(recs)))
    cam_cache = {}
    cards = []
    for r in samples:
        cid = r["clip_id"]
        ex = r["descriptor"]["extra"]
        prefix = ex["gcs_prefix"][5:]  # strip gs://
        idx = int(ex["episode_index"])
        chunk = idx // 1000
        if prefix not in cam_cache:
            cams = [os.path.basename(p.rstrip("/"))
                    for p in fs.ls(f"{prefix}/videos/chunk-{chunk:03d}")]
            cam_cache[prefix] = ("observation.images.front" if "observation.images.front" in cams
                                 else sorted(cams)[0])
        cam = cam_cache[prefix]
        src = f"{prefix}/videos/chunk-{chunk:03d}/{cam}/episode_{idx:06d}.mp4"
        dur = r["metadata"]["qc_metrics"]["duration_s"]
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", dir=clips_dir.parent) as tmp:
                fs.get(src, tmp.name)
                render_mp4_interval(tmp.name, clips_dir / f"{cid}.mp4", 0.0, dur + 1.0)
        except Exception as e:  # noqa: BLE001
            print(f"[dexora] SKIP {cid}: {e}", flush=True)
            continue
        task = (r["metadata"].get("tasks") or [""])[0]
        rows = [["episode", cid], ["duration", f"{dur:.1f}s"], qc_row("dexora", ngates)]
        rows += ROBOT_FACTS["dexora"]
        if task:
            rows.append(["task", task[:160]])
        cards.append(mk_card(cid, dur, r["descriptor"]["fps"], rows))
        print(f"[dexora] {len(cards)} {cid}", flush=True)
    return cards


# ---------------------------------------------------------------- hrdexdb
def build_hrdexdb(recs, clips_dir, n, seed, ngates, ann_ids=frozenset()):
    rng = random.Random(seed)
    samples = rng.sample(recs, min(n, len(recs)))
    cards = []
    for r in samples:
        cid = r["clip_id"]
        prefix = r["descriptor"]["extra"]["gcs_prefix"][5:]
        vids = sorted(p for p in fs.ls(f"{prefix}/vid") if p.endswith(".mp4"))
        if not vids:
            print(f"[hrdexdb] SKIP {cid}: no vids", flush=True)
            continue
        dur = r["metadata"]["qc_metrics"]["duration_s"]
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", dir=clips_dir.parent) as tmp:
                fs.get(vids[0], tmp.name)
                render_mp4_interval(tmp.name, clips_dir / f"{cid}.mp4", 0.0, max(dur + 2.0, 30.0))
        except Exception as e:  # noqa: BLE001
            print(f"[hrdexdb] SKIP {cid}: {e}", flush=True)
            continue
        gr = r["metadata"].get("grasp_result") or {}
        rows = [["episode", cid], ["duration", f"{dur:.1f}s"], qc_row("hrdexdb_allegro", ngates)]
        rows += ROBOT_FACTS["hrdexdb_allegro"]
        rows.append(["object", r["metadata"].get("object", "?")])
        if "grasp_success" in gr:
            rows.append(["grasp", "success" if gr["grasp_success"] else "failure"])
        cards.append(mk_card(cid, dur, 30.0, rows))
        print(f"[hrdexdb_allegro] {len(cards)} {cid}", flush=True)
    return cards


# ---------------------------------------------------------------- realdex
def build_realdex(recs, clips_dir, n, seed, ngates, ann_ids=frozenset()):
    cards = []
    for r in recs[:n]:
        cid = r["clip_id"]
        ex = r["descriptor"]["extra"]
        seq = ex["sequence"]
        dur = r["metadata"]["qc_metrics"]["duration_s"]
        t0 = time.time()
        z = zipfile.ZipFile(fs.open(ex["gcs_zip"][5:]))
        jpgs = [nm for nm in z.namelist()
                if f"/{seq}/" in nm and nm.endswith(".jpg") and "depth" not in nm]
        by_dir = {}
        for nm in jpgs:
            by_dir.setdefault(os.path.dirname(nm), []).append(nm)
        if not by_dir:
            print(f"[realdex] SKIP {cid}: no rgb jpgs", flush=True)
            continue
        cam_dir = sorted(by_dir.items(), key=lambda kv: -len(kv[1]))[0]
        names = sorted(cam_dir[1], key=lambda nm: int(Path(nm).stem))
        native_fps = len(names) / max(dur, 1e-3)
        step = max(1, math.ceil(len(names) / 600))
        names = names[::step]
        frames = []
        for nm in names:
            im = cv2.imdecode(np.frombuffer(z.read(nm), np.uint8), cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        out_fps = max(native_fps / step, 2.0)
        encode_frames(frames, out_fps, clips_dir / f"{cid}.mp4")
        rows = [["episode", cid], ["duration", f"{dur:.1f}s"], qc_row("realdex", ngates)]
        rows += ROBOT_FACTS["realdex"]
        rows.append(["object", r["metadata"].get("object", "?")])
        cards.append(mk_card(cid, dur, out_fps, rows))
        print(f"[realdex] {len(cards)} {cid} frames={len(frames)} {time.time()-t0:.0f}s", flush=True)
    return cards


# ---------------------------------------------------------------- dexwild
def build_dexwild(recs, clips_dir, n, seed, ngates, ann_ids=frozenset(), max_frames=240):
    import h5py
    rng = random.Random(seed)
    # stratify across the task splits (group_id: clothes/florist/pour/spray/toy _data),
    # biased to annotated episodes within each split
    by_split = {}
    for r in recs:
        by_split.setdefault(r["group_id"], []).append(r)
    per = max(1, n // max(1, len(by_split)))
    samples = []
    for g in sorted(by_split):
        grp = [r for r in by_split[g] if r["clip_id"] in ann_ids] or by_split[g]
        samples += rng.sample(grp, min(per, len(grp)))
    if len(samples) < n:  # top up from remaining annotated episodes
        chosen = {r["clip_id"] for r in samples}
        rest = [r for r in recs if r["clip_id"] not in chosen and r["clip_id"] in ann_ids]
        samples += rng.sample(rest, min(n - len(samples), len(rest)))
    import collections
    print(f"[dexwild] sampled per split: "
          f"{dict(collections.Counter(r['group_id'] for r in samples))}", flush=True)
    h5_cache = {}
    cards = []
    for r in samples:
        cid = r["clip_id"]
        ex = r["descriptor"]["extra"]
        src = ex["source"][5:]
        key = ex["hdf5_group"]
        if src not in h5_cache:
            parts = sorted(p for p in fs.ls(src) if ".part_" in p)
            h5_cache[src] = h5py.File(_MultiPartGCSFile(fs, parts), "r")
        h5 = h5_cache[src]
        t0 = time.time()
        try:
            g = h5[key]
            cam_grp, cam_names = None, []
            for sub in sorted(g.keys()):
                child = g[sub]
                if not hasattr(child, "keys"):
                    continue
                try:
                    kids = list(child.keys())
                except Exception:  # noqa: BLE001
                    continue
                if kids and str(kids[0]).endswith(".jpg"):
                    cam_grp, cam_names = child, sorted(kids)
                    break
            if cam_grp is None:
                print(f"[dexwild] SKIP {cid}: no jpg camera group", flush=True)
                continue
            dur = r["descriptor"]["frame_count_override"] / (r["descriptor"]["fps"] or 30.0)
            native_fps = len(cam_names) / max(dur, 1e-3)
            step = max(1, math.ceil(len(cam_names) / max_frames))
            frames = []
            for nm in cam_names[::step]:
                v = cam_grp[nm][()]
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.dtype == np.uint8:
                    im = v  # stored as decoded HWC uint8 (BGR, cv2-written)
                else:
                    buf = np.frombuffer(v if isinstance(v, bytes) else v.tobytes(), np.uint8)
                    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if im is None:
                    continue
                frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if len(frames) < 8:
                print(f"[dexwild] SKIP {cid}: only {len(frames)} frames decoded", flush=True)
                continue
            out_fps = max(native_fps / step, 2.0)
            encode_frames(frames, out_fps, clips_dir / f"{cid}.mp4")
        except Exception as e:  # noqa: BLE001
            print(f"[dexwild] SKIP {cid}: {type(e).__name__}: {e}", flush=True)
            continue
        rows = [["episode", cid], ["duration", f"{dur:.1f}s"], qc_row("dexwild", ngates)]
        rows += ROBOT_FACTS["dexwild"]
        rows.append(["task group", r["group_id"]])
        cards.append(mk_card(cid, dur, out_fps, rows))
        print(f"[dexwild] {len(cards)} {cid} frames={len(frames)} {time.time()-t0:.0f}s", flush=True)
    return cards


BUILDERS = {"trex": build_trex, "dexora": build_dexora, "dexwild": build_dexwild,
            "hrdexdb_allegro": build_hrdexdb, "realdex": build_realdex}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--work", default="/root/viewer_work")
    ap.add_argument("--join-only", action="store_true",
                    help="join annotations into existing cards.json; no rendering")
    args = ap.parse_args()
    for ds in args.datasets:
        wdir = Path(args.work) / ds
        clips = wdir / "clips"
        clips.mkdir(parents=True, exist_ok=True)
        anns = read_annotations(ds)
        if args.join_only:
            cards = json.loads((wdir / "cards.json").read_text())
            nj = join_annotations(cards, anns)
            (wdir / "cards.json").write_text(json.dumps(cards, indent=1))
            print(f"[{ds}] JOIN-ONLY annotated {nj}/{len(cards)} cards "
                  f"(ann rows: {len(anns)})", flush=True)
            continue
        recs = read_manifest(ds)
        ngates = gate_count(ds)
        print(f"[{ds}] kept={len(recs)} gates={ngates} ann_rows={len(anns)}", flush=True)
        cards = BUILDERS[ds](recs, clips, args.n, args.seed, ngates, set(anns))
        nj = join_annotations(cards, anns)
        (wdir / "cards.json").write_text(json.dumps(cards, indent=1))
        print(f"[{ds}] DONE cards={len(cards)} annotated={nj}", flush=True)


if __name__ == "__main__":
    main()
