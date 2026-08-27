#!/usr/bin/env python
"""Finalize the WIYH per-session anchor registry -> accepted extrinsics json.

Acceptance policy (documented in the tier's FILTER_MODE.txt):
  strict:        per-hand fit_med < 30 px AND holdout < 60 px AND
                 n_obs >= 24 AND n_frames >= 12
  corroborated:  per-hand fit_med < 45 px AND holdout < 55 px AND n_obs >= 18,
                 AND another session of the SAME device-day whose independent
                 solve agrees (rot delta < 15 deg, |t| delta < 4 cm) for that
                 hand — wrong-basin solves scatter, agreement implies correct.
A session ships only when BOTH hands are accepted. Manual entries (pilot +
click-solves, --manual) are merged with acceptance "manual".

Usage:
  python scripts/build/wiyh_finalize_anchors.py \
      --registry /root/w7_native/anchors/session_registry.jsonl \
      --manual /root/w7_native/anchors/manual_sessions.json \
      --out /root/w7_native/anchors/accepted_extrinsics.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rot_delta_deg(Ra, Rb):
    Rr = np.array(Ra) @ np.array(Rb).T
    c = (np.trace(Rr) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def strict_ok(h):
    return (h.get("fit_med_px") is not None and h["fit_med_px"] < 30.0
            and (h.get("holdout_med_px") is None or
                 (h["holdout_med_px"] == h["holdout_med_px"] and h["holdout_med_px"] < 60.0))
            and h.get("n_obs", 0) >= 24 and h.get("n_frames", 0) >= 12)


def loose_ok(h):
    ho = h.get("holdout_med_px")
    return (h.get("fit_med_px") is not None and h["fit_med_px"] < 45.0
            and (ho is None or ho != ho or ho < 55.0)
            and h.get("n_obs", 0) >= 18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--manual", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = {}
    for l in open(a.registry):
        r = json.loads(l)
        if "error" not in r:
            rows[r["session"]] = r  # later rows win (reruns)
    by_dd = defaultdict(list)
    for r in rows.values():
        by_dd[f"{r['dev']}_{r['date']}"].append(r)

    accepted = {}
    stats = {"strict": 0, "corroborated": 0, "manual": 0,
             "rejected": 0, "sessions": len(rows)}
    for dd, rs in by_dd.items():
        for r in rs:
            sides = {}
            for side in ("left", "right"):
                h = r.get(side, {})
                if "R" not in h:
                    continue
                if strict_ok(h):
                    sides[side] = (h, "strict")
                elif loose_ok(h):
                    # corroboration: any other session in dd agreeing on this side
                    for r2 in rs:
                        if r2 is r:
                            continue
                        h2 = r2.get(side, {})
                        if "R" not in h2 or not loose_ok(h2):
                            continue
                        if (rot_delta_deg(h["R"], h2["R"]) < 15.0
                                and np.linalg.norm(np.array(h["t"]) - np.array(h2["t"])) < 0.04):
                            sides[side] = (h, "corroborated")
                            break
            if len(sides) == 2:
                mode = ("strict" if all(m == "strict" for _, m in sides.values())
                        else "corroborated")
                stats[mode] += 1
                accepted[r["session"]] = {
                    "status": "pass", "acceptance": mode, "scene": r["scene"],
                    "device_day": dd, "solved_member": r["member"],
                    "left": {"R": sides["left"][0]["R"], "t": sides["left"][0]["t"],
                             "fit_med_px": sides["left"][0]["fit_med_px"],
                             "holdout_med_px": sides["left"][0].get("holdout_med_px")},
                    "right": {"R": sides["right"][0]["R"], "t": sides["right"][0]["t"],
                              "fit_med_px": sides["right"][0]["fit_med_px"],
                              "holdout_med_px": sides["right"][0].get("holdout_med_px")},
                }
            else:
                stats["rejected"] += 1

    if a.manual:
        man = json.loads(Path(a.manual).read_text())
        for sess, entry in man.items():
            if sess not in accepted:  # auto wins when both exist
                entry = dict(entry)
                entry.setdefault("status", "pass")
                entry["acceptance"] = "manual"
                accepted[sess] = entry
                stats["manual"] += 1

    Path(a.out).write_text(json.dumps(accepted, indent=1))
    per_scene = defaultdict(int)
    for e in accepted.values():
        per_scene[e.get("scene", "?")] += 1
    print(f"[finalize] {stats} accepted={len(accepted)} per_scene={dict(per_scene)}")


if __name__ == "__main__":
    main()
