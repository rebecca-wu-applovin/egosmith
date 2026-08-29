#!/usr/bin/env python
"""Add "Dropped examples" cards to the Cat-4 robot galleries.

Failure lists + gate reasons come from filter_run/report.json (robot_episode_qc.py
funnel report; dexwild uses the per-split report.<split>.json files across all five
task splits because the merged report's dropped list is empty). Episodes are rendered
exactly like kept cards, section="dropped", reasons=[failed gate names].

Selection: seed 7, round-robin across first_failed_gate groups for reason diversity.
trex: restricted to non-empty head_left video objects (known ~40% zero-byte mirrors).
Idempotent: existing dropped cards are removed before re-adding; kept cards get
section="kept".
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
import numpy as np  # noqa: E402

from render_clip_card import render_mp4_interval  # noqa: E402
from robot_episode_qc import _MultiPartGCSFile  # noqa: E402
from build_robot_cards import B, fs, ROBOT_FACTS, encode_frames  # noqa: E402


def load_dropped(ds):
    if ds == "dexwild":
        out = []
        for sub in ("clothes", "florist", "pour", "spray", "toy"):
            rep = json.loads(fs.open(
                f"{B}/egosmith_filtered/dexwild/filter_run/report.{sub}.json").read())
            out += rep.get("dropped") or []
        return out
    rep = json.loads(fs.open(f"{B}/egosmith_filtered/{ds}/filter_run/report.json").read())
    return rep.get("dropped") or []


def gate_names(d):
    seen, out = set(), []
    for r in d.get("reasons") or []:
        g = r.split(":", 1)[0]
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out or [d.get("first_failed_gate") or "?"]


def diversify(dropped, k, seed):
    """Round-robin across first_failed_gate groups, shuffled within each."""
    rng = random.Random(seed)
    by_gate = {}
    for d in dropped:
        by_gate.setdefault(d.get("first_failed_gate") or "?", []).append(d)
    for v in by_gate.values():
        rng.shuffle(v)
    gates = sorted(by_gate)
    out, i = [], 0
    while len(out) < k and any(by_gate.values()):
        g = gates[i % len(gates)]
        if by_gate[g]:
            out.append(by_gate[g].pop())
        i += 1
    return out


def drop_card(ds, d, note_extra=None):
    cid = d["episode_id"]
    gates = gate_names(d)
    dur = float(d["metrics"].get("duration_s") or 0.0)
    rows = [["episode", cid], ["duration", f"{dur:.1f}s"],
            ["QC", f"failed: {', '.join(gates)}"]] + ROBOT_FACTS[ds]
    if note_extra:
        rows.append(note_extra)
    return dict(clip_id=cid, video=f"clips/{cid}.mp4", status="video_only",
                section="dropped", reasons=gates,
                note=f"QC-rejected episode — first failing gate: {d.get('first_failed_gate')}",
                meta={"dur_s": round(dur, 2), "fps": None},
                annotation=None, extra_rows=rows)


# ------------------------------------------------------------------ renderers
def render_trex(cands, clips, n):
    import pandas as pd
    ep = pd.read_parquet(fs.open(f"{B}/T-Rex/meta/episodes/chunk-000/file-000.parquet"))
    cam = "observation.images.head_left"
    ep = ep.set_index("episode_index")
    valid = set()
    for chunk_dir in fs.ls(f"{B}/T-Rex/videos/{cam}"):
        for o in fs.ls(chunk_dir, detail=True):
            if o["name"].endswith(".mp4") and o["size"] > 0:
                ci = int(chunk_dir.rsplit("-", 1)[-1])
                fi = int(os.path.basename(o["name"]).split("-")[-1].split(".")[0])
                valid.add((ci, fi))
    print(f"[trex] non-empty video files: {len(valid)}", flush=True)

    def key(d):
        row = ep.loc[int(d["episode_id"].replace("trex_ep", ""))]
        return (int(row[f"videos/{cam}/chunk_index"]), int(row[f"videos/{cam}/file_index"]))

    usable = [(d, key(d)) for d in cands
              if int(d["episode_id"].replace("trex_ep", "")) in ep.index]
    usable = [(d, k) for d, k in usable if k in valid]
    print(f"[trex] dropped candidates on non-empty files: {len(usable)}/{len(cands)}",
          flush=True)
    by_file = {}
    for d, k in usable:
        by_file.setdefault(k, []).append(d)
    cards = []
    for (ci, fi), grp in sorted(by_file.items()):
        if len(cards) >= n:
            break
        src = f"{B}/T-Rex/videos/{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4"
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=clips.parent) as tmp:
            fs.get(src, tmp.name)
            for d in grp:
                if len(cards) >= n:
                    break
                cid = d["episode_id"]
                row = ep.loc[int(cid.replace("trex_ep", ""))]
                t0 = float(row[f"videos/{cam}/from_timestamp"])
                t1 = float(row[f"videos/{cam}/to_timestamp"])
                try:
                    render_mp4_interval(tmp.name, clips / f"{cid}.mp4", t0, t1 - t0)
                except Exception as e:  # noqa: BLE001
                    print(f"[trex] SKIP {cid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    continue
                cards.append(drop_card("trex", d))
                print(f"[trex] dropped {len(cards)} {cid}", flush=True)
    return cards


def render_dexora(cands, clips, n):
    import re
    cam_cache, prefix_cache, cards = {}, {}, []
    for d in cands:
        if len(cards) >= n:
            break
        cid = d["episode_id"]
        group = d["group_id"]
        idx = int(re.search(r"(\d+)$", cid).group(1))
        chunk = idx // 1000
        try:
            if group not in prefix_cache:
                # airbot_* repos sit at Dexora/<group>, the rest at Dexora/dexora/<group>
                for cand in (f"{B}/Dexora/dexora/{group}", f"{B}/Dexora/{group}"):
                    if fs.exists(f"{cand}/meta/info.json") or fs.exists(f"{cand}/videos"):
                        prefix_cache[group] = cand
                        break
                else:
                    print(f"[dexora] SKIP {cid}: no GCS prefix for group {group}", flush=True)
                    prefix_cache[group] = None
            prefix = prefix_cache[group]
            if prefix is None:
                continue
            if prefix not in cam_cache:
                cams = [os.path.basename(p.rstrip("/"))
                        for p in fs.ls(f"{prefix}/videos/chunk-{chunk:03d}")]
                cam_cache[prefix] = ("observation.images.front"
                                     if "observation.images.front" in cams else sorted(cams)[0])
            cam = cam_cache[prefix]
            src = f"{prefix}/videos/chunk-{chunk:03d}/{cam}/episode_{idx:06d}.mp4"
            dur = float(d["metrics"].get("duration_s") or 30.0)
            with tempfile.NamedTemporaryFile(suffix=".mp4", dir=clips.parent) as tmp:
                fs.get(src, tmp.name)
                render_mp4_interval(tmp.name, clips / f"{cid}.mp4", 0.0, dur + 1.0)
        except Exception as e:  # noqa: BLE001
            print(f"[dexora] SKIP {cid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        cards.append(drop_card("dexora", d, note_extra=["task group", group]))
        print(f"[dexora] dropped {len(cards)} {cid}", flush=True)
    return cards


def render_hrdexdb(cands, clips, n):
    cards = []
    for d in cands:
        if len(cards) >= n:
            break
        cid = d["episode_id"]
        group = d["group_id"]
        idx = cid[len(f"hrdexdb_allegro_{group}_"):]
        prefix = f"{B}/HRDexDB/allegro_v5/{group}/{idx}"
        dur = float(d["metrics"].get("duration_s") or 30.0)
        try:
            vids = sorted(p for p in fs.ls(f"{prefix}/vid") if p.endswith(".mp4"))
            if not vids:
                print(f"[hrdexdb] SKIP {cid}: no vids", flush=True)
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp4", dir=clips.parent) as tmp:
                fs.get(vids[0], tmp.name)
                render_mp4_interval(tmp.name, clips / f"{cid}.mp4", 0.0, max(dur + 2.0, 30.0))
        except Exception as e:  # noqa: BLE001
            print(f"[hrdexdb] SKIP {cid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        cards.append(drop_card("hrdexdb_allegro", d, note_extra=["object", group]))
        print(f"[hrdexdb_allegro] dropped {len(cards)} {cid}", flush=True)
    return cards


def render_realdex(cands, clips, n):
    cards = []
    for d in cands:
        if len(cards) >= n:
            break
        cid = d["episode_id"]
        group = d["group_id"]
        seq = cid[len(f"realdex_{group}_"):]
        dur = float(d["metrics"].get("duration_s") or 30.0)
        t0 = time.time()
        try:
            z = zipfile.ZipFile(fs.open(f"{B}/RealDex/{group}.zip"))
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
            frames = [cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(nm), np.uint8),
                                                cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                      for nm in names]
            out_fps = max(native_fps / step, 2.0)
            encode_frames(frames, out_fps, clips / f"{cid}.mp4")
        except Exception as e:  # noqa: BLE001
            print(f"[realdex] SKIP {cid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        cards.append(drop_card("realdex", d, note_extra=["object", group]))
        print(f"[realdex] dropped {len(cards)} {cid} frames={len(frames)} "
              f"{time.time()-t0:.0f}s", flush=True)
    return cards


def render_dexwild(cands, clips, n, max_frames=240):
    import h5py
    h5_cache, cards = {}, []
    for d in cands:
        if len(cards) >= n:
            break
        cid = d["episode_id"]
        group = d["group_id"]
        key = cid[len(f"dexwild_{group}_"):]
        src = f"{B}/DexWild/{group}/robot"
        t0 = time.time()
        try:
            if src not in h5_cache:
                parts = sorted(p for p in fs.ls(src) if ".part_" in p)
                h5_cache[src] = h5py.File(_MultiPartGCSFile(fs, parts), "r")
            g = h5_cache[src][key]
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
            dur = float(d["metrics"].get("duration_s") or 20.0)
            native_fps = len(cam_names) / max(dur, 1e-3)
            step = max(1, math.ceil(len(cam_names) / max_frames))
            frames = []
            for nm in cam_names[::step]:
                v = cam_grp[nm][()]
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.dtype == np.uint8:
                    im = v
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
            encode_frames(frames, out_fps, clips / f"{cid}.mp4")
        except Exception as e:  # noqa: BLE001
            print(f"[dexwild] SKIP {cid}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        cards.append(drop_card("dexwild", d, note_extra=["task group", group]))
        print(f"[dexwild] dropped {len(cards)} {cid} frames={len(frames)} "
              f"{time.time()-t0:.0f}s", flush=True)
    return cards


RENDERERS = {"trex": render_trex, "dexora": render_dexora, "dexwild": render_dexwild,
             "hrdexdb_allegro": render_hrdexdb, "realdex": render_realdex}


def build(ds, n, seed, work):
    wdir = Path(work) / ds
    clips = wdir / "clips"
    cards = json.loads((wdir / "cards.json").read_text())
    for c in cards:
        if c.get("section") == "dropped":
            (clips / Path(c["video"]).name).unlink(missing_ok=True)
    cards = [c for c in cards if c.get("section") != "dropped"]
    for c in cards:
        c["section"] = "kept"
    dropped = load_dropped(ds)
    print(f"[{ds}] dropped episodes in report: {len(dropped)}", flush=True)
    if not dropped:
        print(f"[{ds}] NO dropped episodes — skipping (kept cards tagged only)", flush=True)
        (wdir / "cards.json").write_text(json.dumps(cards, indent=1))
        return
    cands = diversify(dropped, 4 * n, seed)  # extra candidates for render failures
    new = RENDERERS[ds](cands, clips, n)
    cards += new
    (wdir / "cards.json").write_text(json.dumps(cards, indent=1))
    kept_n = sum(1 for c in cards if c["section"] == "kept")
    print(f"[{ds}] DONE kept={kept_n} dropped={len(new)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--work", default="/root/viewer_work")
    args = ap.parse_args()
    for ds in args.datasets:
        build(ds, args.n, args.seed, args.work)


if __name__ == "__main__":
    main()
