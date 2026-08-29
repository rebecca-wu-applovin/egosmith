#!/usr/bin/env python3
"""Infill invalid gaps in GT world_space_res.pth payloads (post-conversion fix).

The quality filter's ``--stages infiller`` contract expects gap-free MANO params: in
the recon pipeline the HAWOR infiller interpolates frames where the hand was not
tracked. GT converters instead wrote zeros on invalid frames, so a
valid -> zeros -> valid transition looks like a ~0.5 m teleport and falsely trips the
hand/finger/wrist step gates (observed on HO3D: 44/55 clips tripped all three).

This script rewrites each ``<outputs_root>/<clip>/world_space_res.pth`` in place:
per hand, params on invalid frames are linearly interpolated between the surrounding
valid keyframes (rotations via slerp), and held constant before the first / after the
last valid frame. ``valid`` flags are NOT modified — presence semantics downstream
are unchanged. Hands with zero valid frames stay all-zero (FPHA precedent).
Idempotent: a hand with no invalid gaps (or no valid frames) is left untouched;
files are only re-dumped when something changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def infill_payload(payload: list) -> tuple[list, bool]:
    trans, rot, hand_pose, betas, valid = [np.asarray(item) for item in payload]
    T = trans.shape[1]
    changed = False
    for hand_index in range(2):
        v = valid[hand_index] > 0.5
        n_valid = int(v.sum())
        if n_valid == 0 or n_valid == T:
            continue
        vi = np.flatnonzero(v)
        all_t = np.arange(T, dtype=np.float64)
        # linear interp + end clamping for translation / articulation / betas
        for arr in (trans, hand_pose, betas):
            block = arr[hand_index]
            for dim in range(block.shape[1]):
                block[:, dim] = np.interp(all_t, vi.astype(np.float64), block[vi, dim].astype(np.float64))
        # slerp for global orient
        rots_valid = Rotation.from_rotvec(rot[hand_index][vi].astype(np.float64))
        if len(vi) == 1:
            rot[hand_index][:] = rot[hand_index][vi[0]]
        else:
            slerp = Slerp(vi.astype(np.float64), rots_valid)
            query = np.clip(all_t, float(vi[0]), float(vi[-1]))
            rot[hand_index] = slerp(query).as_rotvec().astype(rot.dtype)
        changed = True
    return [trans.astype(np.float32), rot.astype(np.float32), hand_pose.astype(np.float32),
            betas.astype(np.float32), valid.astype(np.float32)], changed


def fix_clip(seq_folder: Path) -> dict:
    result = {"clip_id": seq_folder.name, "status": "ok", "changed": False}
    path = seq_folder / "world_space_res.pth"
    try:
        if not path.is_file():
            result["status"] = "missing"
            return result
        payload = joblib.load(path)
        fixed, changed = infill_payload(payload)
        if changed:
            tmp = path.with_suffix(".pth.tmp")
            joblib.dump(fixed, tmp)
            tmp.replace(path)
            result["changed"] = True
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _star(p):
    return fix_clip(Path(p))


def main() -> int:
    parser = argparse.ArgumentParser(description="Infill invalid gaps in world_space_res.pth (in place)")
    parser.add_argument("--outputs_root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--report_out", default=None)
    args = parser.parse_args()

    seq_folders = sorted(p for p in Path(args.outputs_root).iterdir() if p.is_dir())
    print(f"clips: {len(seq_folders)}", flush=True)
    started = time.perf_counter()
    if args.workers <= 1:
        results = [fix_clip(p) for p in seq_folders]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_star, [str(p) for p in seq_folders], chunksize=8))
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "outputs_root": str(Path(args.outputs_root).resolve()),
        "total": len(results),
        "changed": sum(1 for r in results if r.get("changed")),
        "unchanged": sum(1 for r in results if r["status"] == "ok" and not r.get("changed")),
        "missing": sum(1 for r in results if r["status"] == "missing"),
        "failed": len(failed),
        "failures": failed[:10],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("INFILL_DONE" if not failed else "INFILL_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
