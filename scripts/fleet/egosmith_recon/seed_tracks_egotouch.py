#!/usr/bin/env python
"""EgoTouch HaMeR/WiLoR-box seeding for Phase C (fleet generalization of the W7
pilot's seed_tracks_from_hamer.py — same projection/box conventions, validated
10/19 presence-gated keeps vs 0/19 unseeded).

For every clip in a (local-path-rewritten) recon manifest, pull the episode's own
hamer_hands.json (fallback wilor_hands.json) camera-frame joints from the mirror
and synthesize the exact detect_track outputs tracks_0_N/{model_boxes,model_tracks}
.npy in the clip's seq_folder BEFORE batch_infer: detect_track --resume skips on
model_boxes.npy existence, so the (glove-blind) YOLO detector never runs and
motion/slam/infiller consume these boxes instead.

Projection: full-image weak-perspective pinhole f=5000/256*W (audit-validated
0-9px on the instrumented gloves). Because that focal convention scales linearly
with frame width and EgoTouch sources share the 4:3 aspect (640x480 AND 320x240 —
the pilot's source-scale projection + resize is algebraically identical to
projecting straight at OUTPUT scale), boxes are projected directly in subclip
coordinates with f=5000/256*out_width. Boxes = joint bbox +15% margin;
orig_frame = round(start_sec*src_fps) + round(j*src_fps/recon_fps). Episodes
without either json are left unseeded (YOLO runs; Phase D drops them) — counted,
never fatal.

Usage: python seed_tracks_egotouch.py --manifest torun.jsonl [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MARGIN = 0.15
CONF = 0.9

_cache: dict[str, list | None] = {}
_cache_lk = threading.Lock()


def project_bbox(j3d, wo, ho):
    j3d = np.asarray(j3d, float)
    if j3d.shape != (21, 3) or not np.isfinite(j3d).all():
        return None
    z = j3d[:, 2]
    if (np.abs(z) < 1e-6).any() or (z <= 0).any():
        return None
    f_out = 5000.0 / 256 * wo   # width-scaled weak-perspective focal, at output scale
    u = f_out * j3d[:, 0] / z + wo / 2
    v = f_out * j3d[:, 1] / z + ho / 2
    x1, x2 = u.min(), u.max()
    y1, y2 = v.min(), v.max()
    mw, mh = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    x1, x2, y1, y2 = x1 - mw, x2 + mw, y1 - mh, y2 + mh
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(wo - 1.0, x2), min(ho - 1.0, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return [x1, y1, x2, y2]


def episode_recs(ep_uri: str, fs) -> list | None:
    with _cache_lk:
        if ep_uri in _cache:
            return _cache[ep_uri]
    recs = None
    for name in ("hamer_hands.json", "wilor_hands.json"):
        try:
            raw = fs.cat_file(f"{ep_uri}/{name}".replace("gs://", ""))
            if raw:
                recs = [json.loads(ln) for ln in raw.decode().splitlines() if ln.strip()]
                break
        except Exception:  # noqa: BLE001
            continue
    with _cache_lk:
        _cache[ep_uri] = recs
    return recs


def seed_clip(r: dict, fs) -> str:
    d = r["descriptor"]
    seq = Path(d["seq_folder"])
    n = len(d["frame_names"])
    wo, ho = d["width"], d["height"]
    ex = d["extra"]
    ep_uri = ex["source_uri"].rsplit("/", 1)[0]
    src_fps = float(ex["source_fps"])
    recon_fps = float(ex.get("recon_fps") or d.get("fps") or 15.0)
    start_frame = int(round(float(ex["interval_sec"][0]) * src_fps))

    recs = episode_recs(ep_uri, fs)
    if recs is None:
        return "no_json"

    boxes_raw = [np.array([]).reshape(0, 5)] * n
    tracks: dict[int, list] = {}
    n_det = 0
    for j in range(n):
        of = start_frame + int(round(j * src_fps / recon_fps))
        if of >= len(recs):
            break
        frame_boxes = []
        for side, cls in (("left", 0), ("right", 1)):
            b = project_bbox(recs[of].get(f"{side}_pos"), wo, ho)
            if b is None:
                continue
            det5 = np.array([[b[0], b[1], b[2], b[3], CONF]])
            frame_boxes.append(det5[0])
            tid = 1 + cls
            tracks.setdefault(tid, []).append(
                {"frame": j, "det": True, "det_box": det5,
                 "det_handedness": np.array([float(cls)]), "is_near_edge": False})
            n_det += 1
        if frame_boxes:
            boxes_raw[j] = np.stack(frame_boxes)
    tdir = seq / f"tracks_0_{n}"
    tdir.mkdir(parents=True, exist_ok=True)
    np.save(tdir / "model_boxes.npy", np.array(boxes_raw, dtype=object))
    np.save(tdir / "model_tracks.npy", np.array(tracks, dtype=object))
    return "ok" if n_det else "ok_empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]

    counts: dict[str, int] = {}
    lk = threading.Lock()

    def one(r):
        try:
            st = seed_clip(r, fs)
        except Exception as e:  # noqa: BLE001
            print(f"  SEED_ERR {r['clip_id']}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            st = "error"
        with lk:
            counts[st] = counts.get(st, 0) + 1

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, rows))
    print(f"SEED_DONE total={len(rows)} {counts}", flush=True)
    # never fail the shard: unseeded clips fall back to YOLO and die in Phase D
    sys.exit(0)


if __name__ == "__main__":
    main()
