#!/usr/bin/env python
"""STRICT per-session tip verification for the WIYH native tier (v2, 2026-08-28).

Replaces the v1 shipped-tar verifier, which was UNRELIABLE: it detected teal in the
rectified 456x256 tar frames with NO hand-mask gating and NO eef veto, so teal-range
false positives (shelf products, background) fed a one-way nearest-tip metric that
read 9-38 px on sessions whose anchor ROTATION was visually wrong (strict re-audit
2026-08-28: 20/32 shipped sessions failed and were dropped, reason
anchor_rotation_invalid; see filter_run/FILTER_MODE.txt).

v2 = the strict audit's approach (/root/w7_native/strict_audit/run_strict_audit.py),
run per session on ONE source sample member (census member preferred):
  - recompute the world-frame 25-joint skeleton exactly as
    wiyh_native_extractor.load() does (p_world = R_eef @ (R_gl @ p_local + t_gl)
    + t_eef with the accepted per-session extrinsic),
  - project the 5 fingertips per hand through the ORIGINAL 1920x1536 KB4 fisheye
    (SideSolver.project),
  - detect teal pads STRICTLY (detect_teal: eef veto + hand-mask gating),
  - measure a TWO-WAY metric on wrist-gated frames:
      med_a_px : each detection -> its Hungarian-matched projected tip
      med_b_px : each projected tip -> nearest detection (misplaced tips cannot
                 hide — a tip far from every detection scores large)
  - optionally render one annotated frame per session for the mandatory VISUAL read.

The med_b floor is SCENE-DEPENDENT (pad detectability varies with lighting and
background), so metric numbers alone must not drive session drops: sessions with
med_b > 60 px can track correctly. Ship/drop decisions require the visual read of
the rendered frames; stamped numbers are reported as-is.

Output JSON shape matches the shipped filter_run/session_tip_verification.json:
  {"method": ..., "sessions": {session: {med_a_px, med_b_px, p90_b_px,
                                          n_detections, n_frames_sampled}}}

Usage:
  python scripts/build/wiyh_verify_shipped_tips.py \
      --manifest gs://.../wiyh_native/filter_run/clip_manifest.filtered.jsonl \
      --index_dir /root/w7_full/wiyh/index \
      --census /root/w7_native/census/census.jsonl \
      --extrinsics /root/w7_native/anchors/accepted_extrinsics.json \
      --out session_tip_verification.json --render_dir renders/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from wiyh_gate_census import (  # noqa: E402
    SampleStreams, gate_dists, load_sessions, match_masks, stream_sample)
from wiyh_auto_anchor import SideSolver, detect_teal, eef_px_map  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

TIPS = [3, 8, 13, 18, 23]
GATE_PX = 30.0
METHOD = "strict_source_audit_v1_20260828"

_FS = None
_CFG = {}


def _init(cfg):
    global _FS, _CFG
    import gcsfs
    _FS = gcsfs.GCSFileSystem()
    _CFG = cfg


def dedup(dets, tol=3.0):
    out = []
    for c in dets:
        c = np.asarray(c, np.float64)
        if all(np.linalg.norm(c - o) > tol for o in out):
            out.append(c)
    return out


def audit_one(job):
    session, member, parts = job
    row = {"session": session, "member": member["base"]}
    import cv2
    t0 = time.time()
    err = None
    for attempt in range(3):
        try:
            h5, masks, jpgs = stream_sample(_FS, parts, member, want_jpgs=True)
            if h5 is None:
                row["error"] = "no dataset.hdf5"
                return row
            break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(10 * (attempt + 1))
    else:
        row["error"] = err
        return row
    try:
        max_frames = int(_CFG.get("max_frames", 60))
        ss = SampleStreams(h5)
        masks = match_masks(ss, masks)
        dists = gate_dists(ss, masks)
        extr = json.loads(Path(_CFG["extrinsics"]).read_text())[session]

        gated = {s: {i for i in range(ss.n) if 0 <= dists[s][i] < GATE_PX}
                 for s in ("left", "right")}
        union = sorted(gated["left"] | gated["right"])
        if len(union) > max_frames:
            union = [union[k] for k in np.linspace(0, len(union) - 1, max_frames, dtype=int)]
        row["n_gated_frames_total"] = len(gated["left"] | gated["right"])
        row["n_frames_sampled"] = len(union)
        if not union:
            row["error"] = "no wrist-gated frames"
            return row

        # fingertip projections per side on the sampled frames (extractor chain +
        # original KB4 fisheye projection)
        uv = {}
        for s in ("left", "right"):
            sv = SideSolver(ss, s, union)
            R = np.array(extr[s]["R"], np.float64)
            t = np.array(extr[s]["t"], np.float64)
            uv[s] = sv.project(R, t)  # (F,25,2)
        eefuv = {s: eef_px_map(ss, s, union) for s in ("left", "right")}

        # strict detection per frame (eef veto + mask gate), pooled + deduped
        d_a, d_b = [], []
        n_dets = 0
        frame_info = []
        for k, f in enumerate(union):
            img = None
            nm = ss.frame_names[f]
            raw = jpgs.get(nm)
            if raw is not None:
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            mc = masks.get(nm.replace(".jpg", ".png"))
            mc4 = mc.astype(np.float32) * 4 if mc is not None and len(mc) else None
            pool = []
            for s in ("left", "right"):
                if f in gated[s]:
                    pool += detect_teal(img, eef_px=eefuv[s].get(f), mask_coords=mc4)
            dets = dedup(pool)
            cols = []  # (side, tip_slot, uv) of gated sides
            for s in ("left", "right"):
                if f not in gated[s]:
                    continue
                for b, j in enumerate(TIPS):
                    p = uv[s][k, j]
                    if np.isfinite(p).all():
                        cols.append((s, b, p))
            n_dets += len(dets)
            frame_info.append((f, k, dets, cols))
            if not dets or not cols:
                continue
            cost = np.array([[np.linalg.norm(c - p) for _, _, p in cols] for c in dets])
            ra, cb = linear_sum_assignment(cost)
            d_a += [float(cost[a, b]) for a, b in zip(ra, cb)]
            d_b += [float(min(np.linalg.norm(c - p) for c in dets)) for _, _, p in cols]

        row["n_detections"] = n_dets
        row["n_assoc_a"] = len(d_a)
        row["n_tips_b"] = len(d_b)
        row["med_a_px"] = round(float(np.median(d_a)), 1) if d_a else None
        row["med_b_px"] = round(float(np.median(d_b)), 1) if d_b else None
        row["p90_b_px"] = round(float(np.percentile(d_b, 90)), 1) if d_b else None

        # render the frame with most detections for the mandatory visual read
        rdir = _CFG.get("render_dir")
        frame_info.sort(key=lambda z: (len(z[2]), len(z[3])), reverse=True)
        if rdir and frame_info:
            f, k, dets, cols = frame_info[0]
            img = cv2.imdecode(np.frombuffer(jpgs[ss.frame_names[f]], np.uint8),
                               cv2.IMREAD_COLOR)
            colors = {"left": (0, 0, 255), "right": (255, 0, 255)}  # BGR
            pts_all = []
            for s in ("left", "right"):
                if f not in gated[s]:
                    continue
                col = colors[s]
                for j in range(25):
                    p = uv[s][k, j]
                    if np.isfinite(p).all():
                        pt = tuple(np.round(p).astype(int))
                        pts_all.append(p)
                        if j in TIPS:
                            cv2.circle(img, pt, 14, col, 3)
                        elif j == 24:
                            cv2.circle(img, pt, 10, (0, 255, 255), -1)  # model wrist
                        else:
                            cv2.circle(img, pt, 4, col, -1)
                w = eefuv[s].get(f)
                if w is not None:
                    cv2.drawMarker(img, tuple(np.round(w).astype(int)), (0, 255, 0),
                                   cv2.MARKER_TILTED_CROSS, 30, 4)  # eef wrist
                    pts_all.append(w)
            for c in dets:
                cv2.drawMarker(img, tuple(np.round(c).astype(int)), (255, 255, 0),
                               cv2.MARKER_CROSS, 26, 3)  # detection
                pts_all.append(np.asarray(c))
            cv2.putText(img, f"{session} f{f} med_b={row['med_b_px']}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            H, W = img.shape[:2]
            full = cv2.resize(img, (W // 2, H // 2))
            if pts_all:
                P = np.array(pts_all)
                x0 = int(max(0, P[:, 0].min() - 120)); x1 = int(min(W, P[:, 0].max() + 120))
                y0 = int(max(0, P[:, 1].min() - 120)); y1 = int(min(H, P[:, 1].max() + 120))
                crop = img[y0:y1, x0:x1]
                sc = (H // 2) / max(1, crop.shape[0])
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), H // 2))
                comp = np.concatenate([full, crop], axis=1)
            else:
                comp = full
            cv2.imwrite(str(Path(rdir) / f"{session}.jpg"), comp,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            row["render_frame"] = int(f)
        row["elapsed_s"] = round(time.time() - t0, 1)
        return row
    except Exception:  # noqa: BLE001
        row["error"] = traceback.format_exc()[-400:]
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="shipped clip_manifest.filtered.jsonl (local or gs://)")
    ap.add_argument("--index_dir", default="/root/w7_full/wiyh/index")
    ap.add_argument("--census", default="/root/w7_native/census/census.jsonl")
    ap.add_argument("--extrinsics", default="/root/w7_native/anchors/accepted_extrinsics.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render_dir", default=None,
                    help="write one annotated frame per session (mandatory visual read)")
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    if a.render_dir:
        Path(a.render_dir).mkdir(parents=True, exist_ok=True)

    # sessions from the shipped manifest
    sessions = set()
    if a.manifest.startswith("gs://"):
        import gcsfs
        mf = gcsfs.GCSFileSystem().open(a.manifest.replace("gs://", ""), "rb")
    else:
        mf = open(a.manifest, "rb")
    with mf:
        for line in mf:
            if line.strip():
                sessions.add(json.loads(line)["descriptor"]["extra"]["session"])

    census = {}
    for line in open(a.census):
        r = json.loads(line)
        census[r["session"]] = r
    sess_members = load_sessions(Path(a.index_dir))
    parts_by_scene = {f.stem.split(".")[0]: json.loads(f.read_text())
                      for f in Path(a.index_dir).glob("*.parts.json")}
    jobs = []
    for s in sorted(sessions):
        members = sess_members[s]
        cm = census[s].get("member")
        member = next((m for m in members if m["base"] == cm), None)
        if member is None:  # fall back: middle big member
            cands = [m for m in members if m["size"] > 100_000_000]
            member = cands[len(cands) // 2]
        jobs.append((s, member, parts_by_scene[member["scene"]]))
    print(f"[verify] sessions={len(jobs)}", flush=True)

    cfg = {"extrinsics": a.extrinsics, "render_dir": a.render_dir,
           "max_frames": a.max_frames}
    rows = []
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, row in enumerate(pool.imap_unordered(audit_one, jobs), 1):
            rows.append(row)
            print(f"[verify] {i}/{len(jobs)} {row['session']} "
                  f"med_a={row.get('med_a_px')} med_b={row.get('med_b_px')} "
                  f"ndet={row.get('n_detections')} err={row.get('error', '')[:80]}",
                  flush=True)

    out = {
        "method": METHOD,
        "metric": ("two-way px on original 1920x1536 fisheye: med_a = Hungarian "
                   "detection->tip, med_b = projected tip -> nearest detection; "
                   "mask-gated + eef-vetoed teal detection"),
        "note": ("med_b floor is scene-dependent; ship/drop decisions require the "
                 "visual read of the rendered frames, not the metric alone"),
        "sessions": {r["session"]: {k: r[k] for k in
                     ("med_a_px", "med_b_px", "p90_b_px", "n_detections",
                      "n_frames_sampled") if k in r} |
                     ({"error": r["error"]} if "error" in r else {})
                     for r in rows},
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[verify] wrote {a.out}")


if __name__ == "__main__":
    main()
