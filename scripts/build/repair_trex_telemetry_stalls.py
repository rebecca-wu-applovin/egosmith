#!/usr/bin/env python3
"""Detect + repair T-Rex telemetry stalls (per-joint sample-and-hold then snap).

Failure mode (verified on trex_ep000000, left_arm_q_4): a proprio joint's telemetry
freezes at an exactly-repeated value for several frames while the robot keeps moving
(the commanded action stream advances smoothly), then snaps to the true position in a
single frame -- e.g. 8 frozen steps then a 0.2749 rad jump. The QC velocity gate
correctly flags the snap (|dq|/dt >> max_vel) but the episode's underlying motion is
fine; 558 episodes were first-dropped at the velocity gate in the shipped T-Rex run
and most are this artifact, not real teleports.

Detector: a velocity-limit-violating step on joint j is a STALL-SNAP iff it is
immediately preceded by >= --min_freeze_steps consecutive EXACT-zero deltas on that
joint (float-identical repeats: telemetry hold, not quiet motion, which always carries
encoder noise on this rig). Violations with no preceding freeze are GENUINE spikes and
are left untouched.

Repair (episodes with stall evidence only): linear interpolation (on the data
timestamps) across EVERY exact-hold window of >= --min_freeze_steps zero deltas that
ends in a change -- not just the limit-violating ones -- because pervasively stalled
episodes (e.g. trex_ep000058: ~54% of state cells are stale holds) otherwise keep
failing the accel gate on the residual staircase. Each window's rows a+1..i of joint j
interpolate between the last live sample q[a] and the first fresh sample q[i+1]; this
never invents values outside the observed envelope. The matching action rows are only
interpolated when the action stream shows the same hold over the same window (it
usually does not -- action is the commanded stream and stays live). Episodes whose
velocity violations have NO stall attribution are left untouched (genuine spikes).
Raw source data on GCS is NEVER modified: repaired streams are materialized as
per-episode .npz files in the output dir, and provenance (``repaired_windows``) is
stamped into the manifest record metadata + descriptor extra (inline when <= 64
windows, else count + full list inside the .npz).

Re-gate: every repaired episode runs back through the identical QC gate stack
(robot_episode_qc.evaluate_episode, same robot spec); the dedup gate is seeded with the
content hashes of the already-kept manifest so a repaired episode can never duplicate a
kept one. Kept episodes are emitted as manifest.repaired_kept.jsonl ready to append to
the filtered manifest (backup-first convention).

Usage:
  repair_trex_telemetry_stalls.py \
      --data_root /root/cat4_qc/trex_full \
      --robot_spec configs/robot_specs/trex.yaml \
      --qc_report /root/cat4_qc/trex/qc_full/report.json \
      --kept_manifest /root/cat4_qc/trex/qc_full/manifest.jsonl \
      --output_dir /root/cat4_qc/trex/repair_run \
      [--episodes trex_ep000000,...] [--limit 10] [--plot_ids trex_ep000000,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from robot_episode_qc import (  # noqa: E402
    EpisodePayload,
    evaluate_episode,
    iter_trex,
    make_record,
    _limits_array,
)
from lib.pipeline.clips.clip_manifest import write_clip_manifest  # noqa: E402

REPAIR_METHOD = "linear_interp_over_stall_window_v1"

# dataviz reference palette (validated): slot1 blue / slot2 orange + neutral inks
C_BEFORE, C_AFTER, C_LIMIT, C_WIN, C_TEXT = "#2a78d6", "#eb6834", "#6b7280", "#eda100", "#374151"


def detect_stall_windows(q: np.ndarray, t: np.ndarray, max_vel: np.ndarray,
                         min_freeze_steps: int) -> tuple[list[dict], int, int]:
    """Find per-joint stall-then-snap windows behind velocity-limit violations.

    Returns (windows, n_stall_violations, n_genuine_violations). Each window is
    {joint, start_row, end_row, freeze_steps, snap_rad, snap_ratio}: rows
    start_row..end_row hold the stale value; the snap lands on end_row + 1.
    """
    dts = np.clip(np.diff(t.astype(np.float64)), 1e-4, None)
    dq = np.diff(q, axis=0)
    v = np.abs(dq) / dts[:, None]
    over = v > max_vel[None, :]
    windows, n_stall, n_genuine = [], 0, 0
    for j in np.unique(np.nonzero(over)[1]):
        steps = np.nonzero(over[:, j])[0]
        zero = dq[:, j] == 0.0  # exact repeat = telemetry hold
        claimed_until = -1
        for i in steps:
            # count consecutive exact-zero deltas immediately before the snap step
            f = 0
            k = i - 1
            while k >= 0 and zero[k]:
                f += 1
                k -= 1
            if f >= min_freeze_steps:
                n_stall += 1
                a = i - f  # last live row
                if a + 1 <= claimed_until:  # overlapping claim from a previous snap
                    continue
                claimed_until = i
                windows.append({
                    "joint": int(j),
                    "start_row": int(a + 1),
                    "end_row": int(i),
                    "freeze_steps": int(f),
                    "snap_rad": round(float(abs(dq[i, j])), 5),
                    "snap_ratio": round(float(v[i, j] / max_vel[j]), 3),
                })
            else:
                n_genuine += 1
    return windows, n_stall, n_genuine


def hold_repair_windows(q: np.ndarray, min_freeze_steps: int) -> list[tuple[int, int, int]]:
    """Enumerate ALL exact-hold windows that end in a change (de-staircase set).

    Returns (joint, start_row, end_row) triples: rows start_row..end_row repeat the
    value of row start_row - 1 exactly, and row end_row + 1 is a fresh sample. Holds
    running to the episode end (no fresh anchor after) are left alone.
    """
    wins: list[tuple[int, int, int]] = []
    for j in range(q.shape[1]):
        dq = np.diff(q[:, j])
        zero = dq == 0.0
        i = 0
        while i < len(dq):
            if zero[i]:
                k = i
                while k + 1 < len(dq) and zero[k + 1]:
                    k += 1
                if k - i + 1 >= min_freeze_steps and k + 2 < len(q):
                    wins.append((j, i + 1, k + 1))
                i = k + 1
            else:
                i += 1
    return wins


def apply_repair(q: np.ndarray, a_cmd: np.ndarray, t: np.ndarray,
                 windows: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Linear-interp state (and matching frozen action rows) across each hold window."""
    q_rep = q.copy()
    a_rep = a_cmd.copy()
    n_action_repaired = 0
    for j, r0, r1 in windows:
        lo, hi = r0 - 1, r1 + 1  # anchor rows: last live sample, first fresh sample
        if lo < 0 or hi >= len(q):
            continue
        seg_t = t[r0:r1 + 1]
        q_rep[r0:r1 + 1, j] = np.interp(seg_t, [t[lo], t[hi]], [q[lo, j], q[hi, j]])
        # action repaired ONLY if it shows the same hold across the window
        if np.all(a_cmd[lo:r1 + 1, j] == a_cmd[lo, j]):
            a_rep[r0:r1 + 1, j] = np.interp(seg_t, [t[lo], t[hi]], [a_cmd[lo, j], a_cmd[hi, j]])
            n_action_repaired += 1
    return q_rep, a_rep, n_action_repaired


def velocity_ratio(q: np.ndarray, t: np.ndarray, max_vel: np.ndarray) -> np.ndarray:
    dts = np.clip(np.diff(t.astype(np.float64)), 1e-4, None)
    return np.max(np.abs(np.diff(q, axis=0)) / dts[:, None] / max_vel[None, :], axis=1)


def plot_before_after(ep_id: str, q: np.ndarray, q_rep: np.ndarray, t: np.ndarray,
                      max_vel: np.ndarray, windows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r0, r1 = velocity_ratio(q, t, max_vel), velocity_ratio(q_rep, t, max_vel)
    x = np.arange(len(r0))
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    for w in windows:
        ax.axvspan(w["start_row"] - 1, w["end_row"] + 0.5, color=C_WIN, alpha=0.12, lw=0)
    ax.plot(x, r0, color=C_BEFORE, lw=1.6, label="before repair")
    ax.plot(x, r1, color=C_AFTER, lw=1.6, label="after repair")
    ax.axhline(1.0, color=C_LIMIT, lw=1.0, ls="--")
    ax.axhline(3.0, color=C_LIMIT, lw=1.0, ls=":")
    ax.text(len(x) - 1, 1.02, "velocity limit (1.0x)", ha="right", va="bottom", fontsize=8, color=C_LIMIT)
    ax.text(len(x) - 1, 3.02, "hard peak cap (3.0x)", ha="right", va="bottom", fontsize=8, color=C_LIMIT)
    ax.set_xlabel("frame step", color=C_TEXT)
    ax.set_ylabel("max over joints |dq|/dt / limit", color=C_TEXT)
    ax.set_title(f"{ep_id}: velocity ratio, {len(windows)} stall window(s) repaired", color=C_TEXT, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, color="#e5e7eb", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="jpg")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--robot_spec", required=True)
    ap.add_argument("--qc_report", required=True, help="shipped report.json (velocity-drop census source)")
    ap.add_argument("--kept_manifest", required=True, help="shipped kept manifest (dedup hash seed)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--episodes", default=None, help="comma list of episode_ids (smoke)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--plot_ids", default=None, help="comma list to render before/after jpgs")
    ap.add_argument("--min_freeze_steps", type=int, default=2)
    ap.add_argument("--repaired_gcs_prefix",
                    default="gs://foundational-research/hoi-dataset/egosmith_filtered/trex/repair_run/episodes")
    ap.add_argument("--source_id", default="trex_robot")
    ap.add_argument("--split", default="train")
    args = ap.parse_args(argv)

    spec = yaml.safe_load(Path(args.robot_spec).read_text())
    max_vel = _limits_array(spec["streams"]["state"]["max_vel"], int(spec["streams"]["state"]["dof"]), np.inf)
    report = json.loads(Path(args.qc_report).read_text())
    vel_dropped = [d["episode_id"] for d in report["dropped"] if d["first_failed_gate"] == "velocity"]
    drop_metrics = {d["episode_id"]: d["metrics"] for d in report["dropped"]}
    wanted = set(vel_dropped)
    if args.episodes:
        wanted &= set(args.episodes.split(","))
    plot_ids = set(args.plot_ids.split(",")) if args.plot_ids else set()

    # dedup seed: content hashes of already-kept episodes
    seen_hashes: dict[str, str] = {}
    for line in Path(args.kept_manifest).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        h = ((rec.get("metadata") or {}).get("qc_metrics") or {}).get("content_hash")
        if h:
            seen_hashes[h] = rec["clip_id"]
    print(f"velocity-dropped episodes: {len(vel_dropped)}; scanning {len(wanted)}; "
          f"dedup seeded with {len(seen_hashes)} kept hashes", file=sys.stderr)

    out_dir = Path(args.output_dir)
    (out_dir / "episodes_npz").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    census, records = [], []
    n_scanned = 0
    for ep in iter_trex(Path(args.data_root), None):
        if ep.episode_id not in wanted:
            continue
        if args.limit is not None and n_scanned >= args.limit:
            break
        n_scanned += 1
        state, action = ep.stream("state"), ep.stream("action")
        q, a_cmd, t = state.q, action.q, state.t
        windows, n_stall, n_genuine = detect_stall_windows(q, t, max_vel, args.min_freeze_steps)
        row = {
            "episode_id": ep.episode_id,
            "n_frames": int(q.shape[0]),
            "stall_windows": len(windows),
            "frozen_frames_total": int(sum(w["end_row"] - w["start_row"] + 1 for w in windows)),
            "max_snap_rad": max((w["snap_rad"] for w in windows), default=0.0),
            "vel_violations_stall": n_stall,
            "vel_violations_genuine": n_genuine,
            "orig_metrics": {k: v for k, v in drop_metrics.get(ep.episode_id, {}).items()
                             if k.startswith(("state_vel", "state_acc"))},
        }
        if not windows:
            row["regate"] = {"keep": False, "reason": "no_stall_windows_detected"}
            census.append(row)
            continue
        repair_wins = hold_repair_windows(q, args.min_freeze_steps)
        q_rep, a_rep, n_act = apply_repair(q, a_cmd, t, repair_wins)
        row["repair_windows_total"] = len(repair_wins)
        row["repaired_cell_frac"] = round(float(np.mean(q_rep != q)), 5)
        row["action_rows_repaired_windows"] = n_act
        ep_rep = EpisodePayload(
            episode_id=ep.episode_id, group_id=ep.group_id, dataset=ep.dataset,
            streams=[
                type(state)("state", q_rep, t=t, kind="measured"),
                type(action)("action", a_rep, t=t, kind="commanded"),
            ],
            fps_nominal=ep.fps_nominal,
            video_frame_counts=ep.video_frame_counts,
            video_expected_frames=ep.video_expected_frames,
            meta=ep.meta, native=ep.native, seq_folder=ep.seq_folder, root_dir=ep.root_dir,
        )
        result = evaluate_episode(ep_rep, spec, seen_hashes)
        row["regate"] = {"keep": result["keep"], "first_failed_gate": result["first_failed_gate"],
                         "reasons": result["reasons"],
                         "metrics": {k: v for k, v in result["metrics"].items()
                                     if k.startswith(("state_", "action_"))}}
        if ep.episode_id in plot_ids:
            plot_before_after(ep.episode_id, q, q_rep, t, max_vel, windows,
                              out_dir / "plots" / f"{ep.episode_id}.velocity_before_after.jpg")
        if result["keep"]:
            npz_path = out_dir / "episodes_npz" / f"{ep.episode_id}.npz"
            np.savez_compressed(
                npz_path, state=q_rep, action=a_rep, timestamp=t,
                repaired_windows=json.dumps({
                    "format": "[joint, start_row, end_row] (rows hold the pre-window value; interp anchors are start_row-1 / end_row+1)",
                    "windows": [list(w) for w in repair_wins],
                    "vel_snap_windows": windows,
                }))
            rec = make_record(ep_rep, args.source_id, args.split, result)
            rec.descriptor.extra["telemetry_repaired"] = True
            rec.descriptor.extra["repaired_npz"] = f"{args.repaired_gcs_prefix}/{ep.episode_id}.npz"
            prov = {
                "method": REPAIR_METHOD,
                "detector": f"exact_hold>={args.min_freeze_steps}_steps_then_vel_snap",
                "n_windows": len(repair_wins),
                "repaired_cell_frac": row["repaired_cell_frac"],
                "vel_snap_windows": len(windows),
                "max_snap_rad": row["max_snap_rad"],
                "source_data_untouched": True,
            }
            if len(repair_wins) <= 64:
                prov["repaired_windows"] = [list(w) for w in repair_wins]
            else:
                prov["repaired_windows"] = f"see repaired_npz ({len(repair_wins)} windows)"
            rec.metadata["telemetry_repair"] = prov
            records.append(rec)
        census.append(row)
        if n_scanned % 50 == 0:
            kept = sum(1 for c in census if c["regate"].get("keep"))
            print(f"scanned {n_scanned} ({kept} re-gate kept)", file=sys.stderr)

    manifest_path = out_dir / "manifest.repaired_kept.jsonl"
    write_clip_manifest(records, manifest_path)

    stall_eps = [c for c in census if c["stall_windows"] > 0]
    kept_eps = [c for c in census if c["regate"].get("keep")]
    summary = {
        "velocity_dropped_total": len(vel_dropped),
        "scanned": n_scanned,
        "stall_affected": len(stall_eps),
        "no_stall_detected": n_scanned - len(stall_eps),
        "regate_kept": len(kept_eps),
        "regate_dropped_after_repair": len(stall_eps) - len(kept_eps),
        "regate_fail_gates": {},
        "windows_total": int(sum(c["stall_windows"] for c in census)),
        "frozen_frames_total": int(sum(c["frozen_frames_total"] for c in census)),
        "stall_vs_genuine_violations": {
            "stall": int(sum(c["vel_violations_stall"] for c in census)),
            "genuine": int(sum(c["vel_violations_genuine"] for c in census)),
        },
        "min_freeze_steps": args.min_freeze_steps,
        "repair_method": REPAIR_METHOD,
        "robot_spec": args.robot_spec,
        "qc_report": args.qc_report,
        "output_manifest": str(manifest_path.resolve()),
    }
    from collections import Counter
    summary["regate_fail_gates"] = dict(Counter(
        c["regate"].get("first_failed_gate") or c["regate"].get("reason", "unknown")
        for c in census if not c["regate"].get("keep")))
    out = {"summary": summary, "episodes": census}
    (out_dir / "repair_report.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}\nreport:   {out_dir / 'repair_report.json'}")
    return out


if __name__ == "__main__":
    main()
