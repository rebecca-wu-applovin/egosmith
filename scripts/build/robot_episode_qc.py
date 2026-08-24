#!/usr/bin/env python3
"""Robot-episode QC harness: physical-limit gates + sync checks + funnel report.

Mirrors the video quality filter's philosophy (scripts/build/filter_manifest_by_quality.py):
every episode runs through an ordered list of config-driven gates; any failure drops the
episode; the funnel report attributes each drop to the FIRST failing gate while still
recording every triggered reason. Output is the standard manifest contract:

- ``<output_dir>/manifest.jsonl``  -- one ClipManifestRecord per KEPT episode. Descriptors
  use ``storage_kind="native_episode"`` and point at the episode's native files via
  ``extra{}`` (these datasets are not video-tar shaped).
- ``<output_dir>/report.json``     -- funnel with per-gate drop counts, reason counts,
  resolved criteria, and full per-episode drop details (reasons + metrics).

Gates (all config-driven via configs/robot_specs/<robot>.yaml):
- dof_spec            joint count per stream matches the robot spec
- nan_channels        non-finite values in any low-dim stream
- const_channels      frozen streams / too many zero-variance channels
- pos_limits          joint positions outside per-joint limits
- velocity            per-frame joint deltas scaled by the *measured* frame interval
                      (dt from data timestamps when present, else 1/fps from metadata --
                      never an assumed fps; see the "limits are velocity x frame-interval"
                      bug note in the module history)
- accel               second differences scaled by dt^2
- action_tracking     commanded vs measured joint error bounds (optional lag search)
- timestamps          monotonicity, dt jitter, dropped-frame detection
- video_sync          video frame count vs state/action count alignment
- min_length          episode duration floor
- stall               no-joint-movement window longer than max_stall_s
- dedup               downsampled + quantized joint-trajectory content hash across dataset

Datasets (adapters):
- trex            LeRobot v3.0 (T-Rex; dexmate_vega1 + SharpaWave 22-DoF hands)
- dexora          LeRobot v2.1 per-task repos (airbot_play; 12-DoF hands)
- dexwild         robot HDF5 inside split tar on GCS (LEAP V2 17-DoF); remote ranged reads
- hrdexdb_allegro HRDexDB allegro_v5 slice only (Allegro 16-DoF + 6-DoF arm npy dirs)
- realdex         rosbag-extracted TF time series (Shadow Hand + UR arm); license concern
                  is recorded in the report (CC-style academic terms; process anyway)

Usage:
  robot_episode_qc.py --dataset trex --data_root /root/cat4_qc/trex \
      --robot_spec configs/robot_specs/trex.yaml --output_dir /root/cat4_qc/trex/qc --limit 50
  robot_episode_qc.py --dataset trex --data_root ... --calibrate   # print limit suggestions
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.clips.clip_manifest import ClipManifestRecord, write_clip_manifest  # noqa: E402
from lib.pipeline.datasets.descriptors import ClipDescriptor  # noqa: E402

STORAGE_NATIVE_EPISODE = "native_episode"

GATE_ORDER = [
    "dof_spec",
    "nan_channels",
    "const_channels",
    "pos_limits",
    "velocity",
    "accel",
    "action_tracking",
    "timestamps",
    "video_sync",
    "min_length",
    "stall",
    "dedup",
]


# ---------------------------------------------------------------------------
# Episode payload model
# ---------------------------------------------------------------------------


@dataclass
class JointStream:
    """One low-dim time series: measured joints, commanded joints, eef pose, ..."""

    name: str
    q: np.ndarray  # (T, D) float64
    t: Optional[np.ndarray] = None  # (T,) seconds (absolute or episode-relative); None if untimed
    kind: str = "measured"  # measured | commanded | aux
    joint_names: Optional[list[str]] = None

    @property
    def n(self) -> int:
        return int(self.q.shape[0])

    @property
    def dof(self) -> int:
        return int(self.q.shape[1])

    def dts(self, nominal_fps: Optional[float]) -> Optional[np.ndarray]:
        """Per-frame intervals in seconds. Timestamps win; nominal fps is the fallback."""
        if self.t is not None and len(self.t) > 1:
            return np.diff(self.t.astype(np.float64))
        if nominal_fps and self.n > 1:
            return np.full(self.n - 1, 1.0 / float(nominal_fps))
        return None


@dataclass
class EpisodePayload:
    episode_id: str
    group_id: str
    dataset: str
    streams: list[JointStream]
    fps_nominal: Optional[float] = None  # from dataset metadata, only used when no timestamps
    video_frame_counts: dict = field(default_factory=dict)  # cam -> frame count
    video_expected_frames: Optional[float] = None  # what the state stream implies
    meta: dict = field(default_factory=dict)
    native: dict = field(default_factory=dict)  # descriptor extra{} payload
    seq_folder: str = ""
    root_dir: str = ""

    def stream(self, name: str) -> Optional[JointStream]:
        for s in self.streams:
            if s.name == name:
                return s
        return None


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _limits_array(spec_val, dof: int, default: float) -> np.ndarray:
    if spec_val is None:
        return np.full(dof, default)
    arr = np.asarray(spec_val, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(dof, float(arr))
    if arr.shape[0] != dof:
        raise ValueError(f"limits length {arr.shape[0]} != dof {dof}")
    return arr


def gate_dof_spec(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    fails = []
    for stream_name, scfg in (spec.get("streams") or {}).items():
        expected = scfg.get("dof")
        st = ep.stream(stream_name)
        if st is None:
            if scfg.get("required", True):
                fails.append((f"dof_spec:missing_stream:{stream_name}", "missing_stream", 1.0))
            continue
        if expected is not None and st.dof != int(expected):
            fails.append((f"dof_spec:{stream_name}:dof_{st.dof}_expected_{expected}", f"{stream_name}_dof", float(st.dof)))
    return fails


def gate_nan_channels(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    cfg = (spec.get("gates") or {}).get("health") or {}
    max_nan_frac = float(cfg.get("max_nan_frac", 0.0))
    fails = []
    for st in ep.streams:
        bad = ~np.isfinite(st.q)
        frac = float(bad.mean()) if st.q.size else 0.0
        if frac > max_nan_frac:
            fails.append((f"nan_channels:{st.name}", f"{st.name}_nonfinite_frac", frac))
    return fails


def gate_const_channels(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    cfg = (spec.get("gates") or {}).get("health") or {}
    max_const = cfg.get("max_const_channel_frac", 0.5)
    fails = []
    for st in ep.streams:
        if st.kind == "aux" or st.n < 3:
            continue
        ptp = np.nanmax(st.q, axis=0) - np.nanmin(st.q, axis=0)
        const_frac = float((ptp < 1e-9).mean())
        if const_frac >= 1.0:
            fails.append((f"const_channels:{st.name}:frozen_stream", f"{st.name}_const_channel_frac", const_frac))
        elif const_frac > float(max_const):
            fails.append((f"const_channels:{st.name}:too_many_const", f"{st.name}_const_channel_frac", const_frac))
    return fails


def gate_pos_limits(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    cfg = (spec.get("gates") or {}).get("pos_limit") or {}
    max_frac = float(cfg.get("max_violation_frac", 0.001))
    margin = float(cfg.get("margin_rad", 0.01))  # excursions below this are envelope-edge noise
    fails = []
    for stream_name, scfg in (spec.get("streams") or {}).items():
        st = ep.stream(stream_name)
        if st is None or st.kind == "aux":
            continue
        if scfg.get("pos_min") is None and scfg.get("pos_max") is None:
            continue
        lo = _limits_array(scfg.get("pos_min"), st.dof, -np.inf)
        hi = _limits_array(scfg.get("pos_max"), st.dof, np.inf)
        viol = (st.q < (lo - margin)[None, :]) | (st.q > (hi + margin)[None, :])
        frac = float(viol.any(axis=1).mean())
        if frac > max_frac:
            worst = float(np.nanmax(np.maximum(lo[None, :] - st.q, st.q - hi[None, :])))
            fails.append((f"pos_limits:{stream_name}", f"{stream_name}_pos_violation_frac", round(frac, 6)))
            fails.append((f"pos_limits:{stream_name}:worst_excess_rad", f"{stream_name}_pos_worst_excess", round(worst, 4)))
    return fails


def _rate_violation(st: JointStream, limit: np.ndarray, dts: np.ndarray, order: int) -> tuple[int, float]:
    """Count frames whose |d^order q / dt^order| exceeds limit on any joint."""
    q = st.q
    if order == 1:
        d = np.diff(q, axis=0) / dts[:, None]
    else:
        v = np.diff(q, axis=0) / dts[:, None]
        mid_dt = 0.5 * (dts[1:] + dts[:-1])
        d = np.diff(v, axis=0) / mid_dt[:, None]
    over = np.abs(d) > limit[None, :]
    n_bad = int(over.any(axis=1).sum())
    peak = float(np.nanmax(np.abs(d) / np.maximum(limit[None, :], 1e-12))) if d.size else 0.0
    return n_bad, peak


def _gate_rate(ep: EpisodePayload, spec: dict, *, gate_key: str, limit_key: str, order: int, min_n: int) -> list[tuple[str, str, float]]:
    """Shared velocity/accel gate.

    Limits derive from p99.9 x 1.5, which by construction leaves ~0.1% of frames above
    them -- so the drop rule is FRACTION-based (max_violation_frac of frames) plus a hard
    teleport cap (any single frame beyond hard_peak_ratio x limit fails immediately).
    """
    cfg = (spec.get("gates") or {}).get(gate_key) or {}
    max_frac = float(cfg.get("max_violation_frac", 0.005))
    hard_ratio = float(cfg.get("hard_peak_ratio", 3.0))
    short = "vel" if order == 1 else "acc"
    fails = []
    for stream_name, scfg in (spec.get("streams") or {}).items():
        st = ep.stream(stream_name)
        if st is None or st.kind == "aux" or scfg.get(limit_key) is None or st.n < min_n:
            continue
        dts = st.dts(ep.fps_nominal)
        if dts is None:
            continue
        dts = np.clip(dts, 1e-4, None)
        limit = _limits_array(scfg.get(limit_key), st.dof, np.inf)
        n_bad, peak = _rate_violation(st, limit, dts, order=order)
        frac = n_bad / max(st.n - order, 1)
        if frac > max_frac or peak > hard_ratio:
            fails.append((f"{gate_key}:{stream_name}", f"{stream_name}_{short}_violation_frac", round(frac, 5)))
            fails.append((f"{gate_key}:{stream_name}:peak_x_limit", f"{stream_name}_{short}_peak_ratio", round(peak, 3)))
    return fails


def gate_velocity(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    return _gate_rate(ep, spec, gate_key="velocity", limit_key="max_vel", order=1, min_n=2)


def gate_accel(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    return _gate_rate(ep, spec, gate_key="accel", limit_key="max_acc", order=2, min_n=3)


def gate_action_tracking(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    cfg = (spec.get("gates") or {}).get("tracking") or {}
    pairs = cfg.get("pairs") or []
    if not pairs:
        return []
    max_rmse = float(cfg.get("max_rmse", 0.25))
    max_p95 = float(cfg.get("max_p95", 0.6))
    lag_frames = int(cfg.get("allow_lag_frames", 3))
    fails = []
    for cmd_name, meas_name in pairs:
        cmd, meas = ep.stream(cmd_name), ep.stream(meas_name)
        if cmd is None or meas is None:
            continue
        n = min(cmd.n, meas.n)
        if n < 5 or cmd.dof != meas.dof:
            continue
        best_rmse, best_p95, best_lag = np.inf, np.inf, 0
        for lag in range(0, lag_frames + 1):
            a = cmd.q[: n - lag]
            b = meas.q[lag:n]
            err = a - b
            rmse = float(np.sqrt(np.nanmean(err**2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_p95 = float(np.nanpercentile(np.abs(err), 95))
                best_lag = lag
        if best_rmse > max_rmse:
            fails.append((f"action_tracking:{cmd_name}_vs_{meas_name}:rmse", f"{meas_name}_tracking_rmse", round(best_rmse, 4)))
        if best_p95 > max_p95:
            fails.append((f"action_tracking:{cmd_name}_vs_{meas_name}:p95", f"{meas_name}_tracking_p95", round(best_p95, 4)))
        if fails:
            fails.append((f"action_tracking:{cmd_name}_vs_{meas_name}:best_lag", f"{meas_name}_tracking_lag", float(best_lag)))
    return fails


def gate_timestamps(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    cfg = (spec.get("gates") or {}).get("timestamps") or {}
    dropped_factor = float(cfg.get("dropped_dt_factor", 1.8))
    max_dropped_ratio = float(cfg.get("max_dropped_frame_ratio", 0.02))
    max_nonmono = int(cfg.get("max_nonmonotonic", 0))
    allow = cfg.get("streams")  # optional allowlist: only these streams are strictly checked
    fails = []
    for st in ep.streams:
        if st.t is None or st.n < 3:
            continue
        if allow is not None and st.name not in allow:
            continue
        dt = np.diff(st.t.astype(np.float64))
        nonmono = int((dt <= 0).sum())
        if nonmono > max_nonmono:
            fails.append((f"timestamps:{st.name}:nonmonotonic", f"{st.name}_nonmonotonic_steps", float(nonmono)))
        med = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0
        if med > 0:
            dropped = int((dt > dropped_factor * med).sum())
            ratio = dropped / max(len(dt), 1)
            if ratio > max_dropped_ratio:
                fails.append((f"timestamps:{st.name}:dropped_frames", f"{st.name}_dropped_frame_ratio", round(ratio, 5)))
                fails.append((f"timestamps:{st.name}:worst_gap_x_median", f"{st.name}_worst_gap_ratio", round(float(dt.max() / med), 2)))
    return fails


def gate_video_sync(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    cfg = (spec.get("gates") or {}).get("video_sync") or {}
    max_rel = float(cfg.get("max_rel_mismatch", 0.03))
    max_abs = float(cfg.get("max_abs_mismatch", 5))
    expected = ep.video_expected_frames
    if expected is None or not ep.video_frame_counts:
        return []
    fails = []
    for cam, count in ep.video_frame_counts.items():
        diff = abs(float(count) - float(expected))
        if diff > max_abs and diff / max(expected, 1.0) > max_rel:
            fails.append((f"video_sync:{cam}", f"{cam}_frame_count_mismatch", round(diff, 1)))
    return fails


def gate_min_length(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    cfg = (spec.get("gates") or {}).get("health") or {}
    min_len_s = float(cfg.get("min_len_s", 1.0))
    primary = ep.streams[0] if ep.streams else None
    if primary is None:
        return [("min_length:no_streams", "duration_s", 0.0)]
    if primary.t is not None and primary.n > 1:
        dur = float(primary.t[-1] - primary.t[0])
    elif ep.fps_nominal:
        dur = primary.n / float(ep.fps_nominal)
    else:
        return []
    if dur < min_len_s:
        return [("min_length", "duration_s", round(dur, 3))]
    return []


def gate_stall(ep: EpisodePayload, spec: dict) -> list[tuple[str, str, float]]:
    cfg = (spec.get("gates") or {}).get("health") or {}
    max_stall_s = float(cfg.get("max_stall_s", 5.0))
    eps_rad_s = float(cfg.get("stall_eps_rad_s", 0.005))
    fails = []
    for st in ep.streams:
        if st.kind != "measured" or st.n < 3:
            continue
        dts = st.dts(ep.fps_nominal)
        if dts is None:
            continue
        dts = np.clip(dts, 1e-4, None)
        speed = np.nanmax(np.abs(np.diff(st.q, axis=0)), axis=1) / dts
        stalled = speed < eps_rad_s
        # longest stalled run, in seconds
        longest, cur = 0.0, 0.0
        for flag, dt in zip(stalled, dts):
            cur = cur + dt if flag else 0.0
            longest = max(longest, cur)
        if longest > max_stall_s:
            fails.append((f"stall:{st.name}", f"{st.name}_longest_stall_s", round(longest, 2)))
    return fails


def episode_content_hash(ep: EpisodePayload, spec: dict) -> str:
    cfg = (spec.get("gates") or {}).get("dedup") or {}
    target_hz = float(cfg.get("downsample_hz", 5.0))
    quant = float(cfg.get("quant", 0.01))
    h = hashlib.blake2b(digest_size=16)
    for st in ep.streams:
        if st.kind != "measured":
            continue
        if ep.fps_nominal:
            stride = max(1, int(round(float(ep.fps_nominal) / target_hz)))
        elif st.t is not None and st.n > 1:
            med_dt = float(np.median(np.diff(st.t)))
            stride = max(1, int(round((1.0 / max(med_dt, 1e-4)) / target_hz)))
        else:
            stride = 1
        ds = st.q[::stride]
        qz = np.round(ds / quant).astype(np.int64)
        h.update(st.name.encode())
        h.update(np.ascontiguousarray(qz).tobytes())
        h.update(str(qz.shape).encode())
    return h.hexdigest()


GATE_FUNCS = {
    "dof_spec": gate_dof_spec,
    "nan_channels": gate_nan_channels,
    "const_channels": gate_const_channels,
    "pos_limits": gate_pos_limits,
    "velocity": gate_velocity,
    "accel": gate_accel,
    "action_tracking": gate_action_tracking,
    "timestamps": gate_timestamps,
    "video_sync": gate_video_sync,
    "min_length": gate_min_length,
    "stall": gate_stall,
    # dedup handled inline (needs cross-episode state)
}


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def iter_trex(data_root: Path, limit: Optional[int]) -> Iterator[EpisodePayload]:
    """LeRobot v3.0 layout: chunked multi-episode parquet + meta/episodes parquet.

    Expects data_root to contain (mirroring gs://.../T-Rex/):
      info.json (or meta/info.json), meta_episodes_*.parquet or meta/episodes/...,
      data_file-*.parquet or data/chunk-*/file-*.parquet
    """
    import pandas as pd

    info_path = next(p for p in [data_root / "info.json", data_root / "meta" / "info.json"] if p.exists())
    info = json.loads(info_path.read_text())
    fps = float(info.get("fps") or 30.0)
    ep_meta_files = sorted(data_root.glob("meta_episodes_*.parquet")) or sorted((data_root / "meta" / "episodes").rglob("*.parquet"))
    ep_meta = pd.concat([pd.read_parquet(p) for p in ep_meta_files], ignore_index=True)

    def data_file(chunk_idx: int, file_idx: int) -> Optional[Path]:
        cands = [
            data_root / f"data_file-{file_idx:03d}.parquet",
            data_root / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet",
        ]
        return next((c for c in cands if c.exists()), None)

    video_keys = sorted({m.group(1) for c in ep_meta.columns for m in [re.match(r"videos/(.+)/from_timestamp", c)] if m})
    loaded: dict[tuple[int, int], "pd.DataFrame"] = {}
    emitted = 0
    for _, row in ep_meta.iterrows():
        if limit is not None and emitted >= limit:
            return
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        path = data_file(*key)
        if path is None:
            continue  # data file not mirrored locally -- sampling mode
        if key not in loaded:
            loaded.clear()  # keep memory bounded; files are visited in order
            loaded[key] = pd.read_parquet(path)
        df = loaded[key]
        sl = df[(df["index"] >= int(row["dataset_from_index"])) & (df["index"] < int(row["dataset_to_index"]))]
        if not len(sl):
            continue
        state = np.stack(sl["observation.state"].to_numpy())
        action = np.stack(sl["action"].to_numpy())
        t = sl["timestamp"].to_numpy().astype(np.float64)
        ep_idx = int(row["episode_index"])
        video_counts = {}
        for vk in video_keys:
            f0 = row.get(f"videos/{vk}/from_timestamp")
            f1 = row.get(f"videos/{vk}/to_timestamp")
            if f0 is not None and f1 is not None and np.isfinite(f0) and np.isfinite(f1):
                video_counts[vk] = round((float(f1) - float(f0)) * fps)
        yield EpisodePayload(
            episode_id=f"trex_ep{ep_idx:06d}",
            group_id="T-Rex",
            dataset="trex",
            streams=[
                JointStream("state", state, t=t, kind="measured"),
                JointStream("action", action, t=t, kind="commanded"),
            ],
            fps_nominal=fps,
            video_frame_counts=video_counts,
            video_expected_frames=float(len(sl)),
            meta={
                "tasks": list(row.get("tasks") or []),
                "length_meta": int(row["length"]),
                "robot_type": info.get("robot_type"),
            },
            native={
                "format": "lerobot_v3",
                "gcs_prefix": "gs://foundational-research/hoi-dataset/T-Rex",
                "data_file": f"data/chunk-{key[0]:03d}/file-{key[1]:03d}.parquet",
                "episode_index": ep_idx,
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
            },
            seq_folder=str(path.parent),
            root_dir=str(data_root),
        )
        emitted += 1


def iter_dexora(data_root: Path, limit: Optional[int]) -> Iterator[EpisodePayload]:
    """LeRobot v2.1 per-task repos: <task>/data/chunk-*/episode_*.parquet + meta/*.jsonl."""
    import pandas as pd

    emitted = 0
    # Task-level repos (dexora/<task>) are the canonical 12.2K-episode view; the airbot_*
    # category repos regroup a subset of the same episodes. Order task-level first so the
    # dedup gate keeps the canonical copy and attributes the airbot_* re-uploads as dupes.
    task_dirs = sorted(
        (p for p in data_root.iterdir() if (p / "meta" / "info.json").exists()),
        key=lambda p: (p.name.startswith("airbot_"), p.name),
    )
    for task_dir in task_dirs:
        info = json.loads((task_dir / "meta" / "info.json").read_text())
        fps = float(info.get("fps") or 20.0)
        lengths = {}
        ep_jsonl = task_dir / "meta" / "episodes.jsonl"
        if ep_jsonl.exists():
            for line in ep_jsonl.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    lengths[int(rec["episode_index"])] = rec
        for pq in sorted(task_dir.glob("data/chunk-*/episode_*.parquet")):
            if limit is not None and emitted >= limit:
                return
            try:
                df = pd.read_parquet(pq)
                if not len(df):
                    raise ValueError("empty parquet (0 rows)")
            except Exception as exc:  # zero-byte / corrupt episode files -> build-invalid drop
                yield EpisodePayload(
                    episode_id=f"dexora_{task_dir.name}_{pq.stem}",
                    group_id=task_dir.name,
                    dataset="dexora",
                    streams=[],
                    fps_nominal=fps,
                    meta={"read_error": str(exc)[:200]},
                    native={"format": "lerobot_v2.1", "episode_parquet": str(pq)},
                    seq_folder=str(pq.parent),
                    root_dir=str(task_dir),
                )
                emitted += 1
                continue
            ep_idx = int(df["episode_index"].iloc[0])
            state = np.stack(df["observation.state"].to_numpy())
            action = np.stack(df["action"].to_numpy())
            t = df["timestamp"].to_numpy().astype(np.float64)
            meta_rec = lengths.get(ep_idx) or {}
            yield EpisodePayload(
                episode_id=f"dexora_{task_dir.name}_ep{ep_idx:06d}",
                group_id=task_dir.name,
                dataset="dexora",
                streams=[
                    JointStream("state", state, t=t, kind="measured"),
                    JointStream("action", action, t=t, kind="commanded"),
                ],
                fps_nominal=fps,
                video_frame_counts={},
                video_expected_frames=float(meta_rec.get("length")) if meta_rec.get("length") else None,
                meta={"tasks": meta_rec.get("tasks"), "robot_type": info.get("robot_type"), "length_meta": meta_rec.get("length")},
                native={
                    "format": "lerobot_v2.1",
                    "gcs_prefix": f"gs://foundational-research/hoi-dataset/Dexora/dexora/{task_dir.name}",
                    "episode_parquet": str(pq.relative_to(task_dir)),
                    "episode_index": ep_idx,
                },
                seq_folder=str(pq.parent),
                root_dir=str(task_dir),
            )
            emitted += 1


class _MultiPartGCSFile(io.RawIOBase):
    """Read-only file-like over concatenated GCS objects with a block cache.

    Used to open the DexWild robot HDF5 that lives inside a split tar
    (single-member tar => payload starts at byte 512).
    """

    BLOCK = 4 * 1024 * 1024

    def __init__(self, fs, paths: list[str], offset: int = 512, max_cache_blocks: int = 64):
        super().__init__()
        self.fs = fs
        self.paths = paths
        self.offset = offset
        self.sizes = [fs.info(p)["size"] for p in paths]
        self.cum = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.cum[-1]) - offset
        self.pos = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._max_blocks = max_cache_blocks

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, pos, whence=0):
        self.pos = {0: pos, 1: self.pos + pos, 2: self.total + pos}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def _raw_range(self, gstart: int, n: int) -> bytes:
        out = b""
        pos, rem = gstart, n
        while rem > 0:
            i = int(np.searchsorted(self.cum, pos, side="right")) - 1
            if i >= len(self.paths):
                break
            local = pos - int(self.cum[i])
            take = min(rem, int(self.sizes[i]) - local)
            if take <= 0:
                break
            out += self.fs.cat_file(self.paths[i], start=local, end=local + take)
            pos += take
            rem -= take
        return out

    def _block(self, idx: int) -> bytes:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        data = self._raw_range(idx * self.BLOCK, self.BLOCK)
        self._cache[idx] = data
        if len(self._cache) > self._max_blocks:
            self._cache.popitem(last=False)
        return data

    def readinto(self, b):
        n = len(b)
        gstart = self.pos + self.offset
        out = b""
        while len(out) < n:
            idx, off = divmod(gstart + len(out), self.BLOCK)
            blk = self._block(idx)
            if not blk:
                break
            out += blk[off : off + (n - len(out))]
            if off + (n - len(out)) >= len(blk) and len(blk) < self.BLOCK:
                break
        b[: len(out)] = out
        self.pos += len(out)
        return len(out)


def iter_dexwild(data_root: str, limit: Optional[int]) -> Iterator[EpisodePayload]:
    """DexWild robot episodes.

    data_root is either a gs:// prefix holding <name>.part_NN split-tar pieces of a single
    HDF5 (e.g. gs://.../DexWild/clothes_data/robot) or a local .hdf5 path.
    Only low-dim streams are read; camera frames are counted from HDF5 keys (metadata only).
    """
    import h5py

    if str(data_root).startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        prefix = str(data_root)[len("gs://") :].rstrip("/")
        parts = sorted(p for p in fs.ls(prefix) if ".part_" in p)
        if not parts:
            raise FileNotFoundError(f"No .part_NN objects under {data_root}")
        h5 = h5py.File(_MultiPartGCSFile(fs, parts), "r")
        native_src = str(data_root)
    else:
        h5 = h5py.File(data_root, "r")
        native_src = str(data_root)

    group_id = Path(str(data_root).rstrip("/")).parent.name or "dexwild"
    ep_keys = sorted(h5.keys())
    if limit is not None:
        ep_keys = ep_keys[:limit]
    for key in ep_keys:
        g = h5[key]
        streams = []
        video_counts = {}
        ts_norm: dict[str, dict] = {}
        n_frames = None
        ts_ds = g.get("timesteps/timesteps")
        if ts_ds is not None:
            raw = ts_ds[()]
            if isinstance(raw, bytes):
                frame_ts = np.array([int(x) for x in raw.decode().split() if x.strip()], dtype=np.int64)
                n_frames = len(frame_ts)
        for sub in g.keys():
            ds = g.get(f"{sub}/{sub}")
            if isinstance(ds, type(None)):
                # camera group: count children cheaply (len avoids walking every link name,
                # which is prohibitively slow over remote ranged reads)
                child = g[sub]
                if hasattr(child, "keys"):
                    try:
                        first = next(iter(child), None)
                    except Exception:
                        first = None
                    if first is not None and str(first).endswith(".jpg"):
                        video_counts[sub] = len(child)
                continue
            try:
                arr = np.asarray(ds[:], dtype=np.float64)
            except Exception:
                continue
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            t = arr[:, 0] / 1e9 if arr[:, 0].max() > 1e15 else None  # col 0 = ns timestamps
            q = arr[:, 1:] if t is not None else arr
            if t is not None and len(t) > 1 and bool((np.diff(t) <= 0).any()):
                # Transport-reordered samples: DexWild HDF5 rows are ROS messages and a
                # few arrive out of order (single-sample 4-65ms reversals with tiny |dq|,
                # verified on toy_data 2026-08-24 -- NOT motion glitches). Stable-sort by
                # timestamp and drop exact-duplicate stamps so the timestamps/velocity
                # gates measure the motion, not message transport. Counts recorded in
                # meta[timestamp_normalization] for provenance.
                order = np.argsort(t, kind="stable")
                reordered = int((order != np.arange(len(order))).sum())
                t, q = t[order], q[order]
                keep = np.concatenate([[True], np.diff(t) > 0])
                ts_norm[sub] = {"reordered_rows": reordered, "duplicate_rows_dropped": int((~keep).sum())}
                t, q = t[keep], q[keep]
            kind = "measured" if "leapv2" in sub else "aux"
            streams.append(JointStream(sub, q, t=t, kind=kind))
        # put leapv2 streams first so min_length/stall use the hand joints
        streams.sort(key=lambda s: (s.kind != "measured", s.name))
        yield EpisodePayload(
            episode_id=f"dexwild_{group_id}_{key}",
            group_id=group_id,
            dataset="dexwild",
            streams=streams,
            fps_nominal=None,  # dt comes from per-sample ns timestamps in the data
            video_frame_counts=video_counts,
            video_expected_frames=float(n_frames) if n_frames else (float(streams[0].n) if streams else None),
            meta={"n_timesteps": n_frames, **({"timestamp_normalization": ts_norm} if ts_norm else {})},
            native={"format": "dexwild_hdf5_split_tar", "source": native_src, "hdf5_group": key},
            seq_folder=f"{native_src}#{key}",
            root_dir=str(data_root),
        )


def iter_hrdexdb_allegro(data_root: Path, limit: Optional[int]) -> Iterator[EpisodePayload]:
    """HRDexDB Allegro slice: <object>/<idx>/raw/{arm,hand,timestamps}/*.npy.

    Only the allegro_v5 slice is processed (16-DoF Allegro hand + 6-DoF arm).
    Streams are multi-rate; each carries its own Unix timestamps.
    """
    emitted = 0
    for obj_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for ep_dir in sorted((p for p in obj_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            raw = ep_dir / "raw"
            if not (raw / "hand" / "position.npy").exists():
                continue
            if limit is not None and emitted >= limit:
                return

            def _load(rel):
                p = raw / rel
                if not p.exists():
                    return None
                try:
                    return np.asarray(np.load(p, allow_pickle=True), dtype=np.float64)
                except (ValueError, TypeError):
                    return None  # ragged / non-numeric object arrays

            hand_q = _load("hand/position.npy")
            hand_a = _load("hand/action.npy")
            hand_t = _load("hand/time.npy")
            arm_q = _load("arm/position.npy")
            arm_t = _load("arm/time.npy")
            cam_t = _load("timestamps/timestamp.npy")
            frame_id = _load("timestamps/frame_id.npy")
            streams = []
            if hand_q is not None and hand_q.ndim == 2 and len(hand_q):
                streams.append(JointStream("hand_position", hand_q, t=hand_t, kind="measured"))
            if hand_a is not None and hand_a.ndim == 2 and len(hand_a):
                streams.append(JointStream("hand_action", hand_a, t=hand_t, kind="commanded"))
            if arm_q is not None and arm_q.ndim == 2 and len(arm_q):
                streams.append(JointStream("arm_position", arm_q, t=arm_t, kind="measured"))
            video_counts = {}
            video_expected = None
            if cam_t is not None and len(cam_t) > 1 and frame_id is not None:
                # frame_id gaps directly encode dropped camera frames
                span = float(frame_id[-1] - frame_id[0]) + 1
                video_counts["sync_cams"] = float(len(frame_id))
                video_expected = span
            grasp = {}
            gr_path = ep_dir / "grasp_result.json"
            if gr_path.exists():
                grasp = json.loads(gr_path.read_text())
            yield EpisodePayload(
                episode_id=f"hrdexdb_allegro_{obj_dir.name}_{ep_dir.name}",
                group_id=obj_dir.name,
                dataset="hrdexdb_allegro",
                streams=streams,
                fps_nominal=None,
                video_frame_counts=video_counts,
                video_expected_frames=video_expected,
                meta={"object": obj_dir.name, "grasp_result": grasp},
                native={
                    "format": "hrdexdb_raw_npy",
                    "gcs_prefix": f"gs://foundational-research/hoi-dataset/HRDexDB/allegro_v5/{obj_dir.name}/{ep_dir.name}",
                },
                seq_folder=str(ep_dir),
                root_dir=str(obj_dir),
            )
            emitted += 1


_TF_JOINT_FILES_HAND = [
    # Shadow Hand right: knuckle/proximal/middle/distal chains + thumb + wrist (revolute TF pairs)
    "rh_palm-rh_ffknuckle", "rh_ffknuckle-rh_ffproximal", "rh_ffproximal-rh_ffmiddle", "rh_ffmiddle-rh_ffdistal",
    "rh_palm-rh_mfknuckle", "rh_mfknuckle-rh_mfproximal", "rh_mfproximal-rh_mfmiddle", "rh_mfmiddle-rh_mfdistal",
    "rh_palm-rh_rfknuckle", "rh_rfknuckle-rh_rfproximal", "rh_rfproximal-rh_rfmiddle", "rh_rfmiddle-rh_rfdistal",
    "rh_palm-rh_lfmetacarpal", "rh_lfmetacarpal-rh_lfknuckle", "rh_lfknuckle-rh_lfproximal",
    "rh_lfproximal-rh_lfmiddle", "rh_lfmiddle-rh_lfdistal",
    "rh_palm-rh_thbase", "rh_thbase-rh_thproximal", "rh_thproximal-rh_thhub",
    "rh_thhub-rh_thmiddle", "rh_thmiddle-rh_thdistal",
    "rh_forearm-rh_wrist", "rh_wrist-rh_palm",
]
_TF_JOINT_FILES_ARM = [
    "ra_base_link_inertia-ra_shoulder_link", "ra_shoulder_link-ra_upper_arm_link",
    "ra_upper_arm_link-ra_forearm_link", "ra_forearm_link-ra_wrist_1_link",
    "ra_wrist_1_link-ra_wrist_2_link", "ra_wrist_2_link-ra_wrist_3_link",
]


def _parse_tf_txt(path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Parse a RealDex TF dump: lines of `t tx ty tz qx qy qz qw` (whitespace/comma splits).

    Returns (t_seconds, joint_angle) where joint_angle is the rotation magnitude of the
    relative transform -- for a revolute joint this is the joint angle up to sign.
    """
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        vals = [v for v in re.split(r"[,\s]+", line.strip()) if v]
        try:
            nums = [float(v) for v in vals]
        except ValueError:
            continue
        if len(nums) >= 8:
            rows.append(nums[:8])
    if len(rows) < 3:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    t = arr[:, 0]
    if t.max() > 1e15:  # ns
        t = t / 1e9
    quat = arr[:, 4:8]
    w = np.clip(np.abs(quat[:, 3] / np.maximum(np.linalg.norm(quat, axis=1), 1e-12)), 0.0, 1.0)
    angle = 2.0 * np.arccos(w)
    return t, angle


def iter_realdex(data_root: Path, limit: Optional[int]) -> Iterator[EpisodePayload]:
    """RealDex sequences: <object>/<seq>/TF/*.txt (+ frame_counts.json from the fetch step).

    Joint streams are reconstructed from TF revolute-pair time series (rotation magnitude
    == joint angle up to sign). Streams are resampled onto the arm clock for the composite
    hand stream; per-stream timestamps drive dt.
    """
    emitted = 0
    for obj_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for seq_dir in sorted(p for p in obj_dir.iterdir() if p.is_dir()):
            tf_dir = seq_dir / "TF"
            if not tf_dir.is_dir():
                continue
            if limit is not None and emitted >= limit:
                return
            hand_series = []
            hand_names = []
            for name in _TF_JOINT_FILES_HAND:
                parsed = _parse_tf_txt(tf_dir / f"{name}.txt")
                if parsed is not None and len(parsed[0]) > 10:
                    hand_series.append(parsed)
                    hand_names.append(name)
            arm_series = []
            arm_names = []
            for name in _TF_JOINT_FILES_ARM:
                parsed = _parse_tf_txt(tf_dir / f"{name}.txt")
                if parsed is not None and len(parsed[0]) > 10:
                    arm_series.append(parsed)
                    arm_names.append(name)
            streams = []
            for label, series, names in (("hand_angles", hand_series, hand_names), ("arm_angles", arm_series, arm_names)):
                if not series:
                    continue
                # resample every joint onto the densest joint's clock
                ref_t = max((t for t, _ in series), key=len)
                cols = [np.interp(ref_t, t, a) for t, a in series]
                streams.append(JointStream(label, np.stack(cols, axis=1), t=ref_t, kind="measured", joint_names=names))
            video_counts = {}
            video_expected = None
            fc_path = seq_dir / "frame_counts.json"
            if fc_path.exists():
                fc = json.loads(fc_path.read_text())
                video_counts = {k: float(v) for k, v in (fc.get("rgb_frame_counts") or {}).items()}
                if fc.get("global_name_position_frames"):
                    video_expected = float(fc["global_name_position_frames"])
            yield EpisodePayload(
                episode_id=f"realdex_{obj_dir.name}_{seq_dir.name}",
                group_id=obj_dir.name,
                dataset="realdex",
                streams=streams,
                fps_nominal=None,
                video_frame_counts=video_counts,
                video_expected_frames=video_expected,
                meta={"object": obj_dir.name, "n_hand_joints_found": len(hand_names), "n_arm_joints_found": len(arm_names)},
                native={
                    "format": "realdex_rosbag_extract",
                    "gcs_zip": f"gs://foundational-research/hoi-dataset/RealDex/{obj_dir.name}.zip",
                    "sequence": seq_dir.name,
                },
                seq_folder=str(seq_dir),
                root_dir=str(obj_dir),
            )
            emitted += 1


ADAPTERS = {
    "trex": iter_trex,
    "dexora": iter_dexora,
    "dexwild": iter_dexwild,
    "hrdexdb_allegro": iter_hrdexdb_allegro,
    "realdex": iter_realdex,
}


# ---------------------------------------------------------------------------
# Calibration (data-driven limits where no spec sheet exists)
# ---------------------------------------------------------------------------


def run_calibration(episodes: Iterable[EpisodePayload]) -> dict:
    """Suggest per-stream limits: pos at [p0.1 - pad, p99.9 + pad], vel/acc at p99.9 x 1.5."""
    pos: dict[str, list] = {}
    vel: dict[str, list] = {}
    acc: dict[str, list] = {}
    n = 0
    for ep in episodes:
        n += 1
        for st in ep.streams:
            if st.kind == "aux" or st.n < 3:
                continue
            dts = st.dts(ep.fps_nominal)
            pos.setdefault(st.name, []).append(st.q)
            if dts is not None:
                dts = np.clip(dts, 1e-4, None)
                v = np.diff(st.q, axis=0) / dts[:, None]
                vel.setdefault(st.name, []).append(v)
                mid = 0.5 * (dts[1:] + dts[:-1])
                acc.setdefault(st.name, []).append(np.diff(v, axis=0) / mid[:, None])
    out = {"episodes_seen": n, "streams": {}}
    for name in pos:
        q = np.concatenate(pos[name], axis=0)
        entry = {
            "dof": int(q.shape[1]),
            "pos_p001": np.nanpercentile(q, 0.1, axis=0).round(4).tolist(),
            "pos_p999": np.nanpercentile(q, 99.9, axis=0).round(4).tolist(),
        }
        if name in vel:
            v = np.abs(np.concatenate(vel[name], axis=0))
            entry["vel_p999"] = np.nanpercentile(v, 99.9, axis=0).round(3).tolist()
            entry["vel_limit_suggest_p999x1.5"] = (np.nanpercentile(v, 99.9, axis=0) * 1.5).round(3).tolist()
        if name in acc:
            a = np.abs(np.concatenate(acc[name], axis=0))
            entry["acc_p999"] = np.nanpercentile(a, 99.9, axis=0).round(1).tolist()
            entry["acc_limit_suggest_p999x1.5"] = (np.nanpercentile(a, 99.9, axis=0) * 1.5).round(1).tolist()
        out["streams"][name] = entry
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def evaluate_episode(ep: EpisodePayload, spec: dict, seen_hashes: dict) -> dict:
    reasons: list[str] = []
    metrics: dict = {}
    first_gate = None
    enabled = spec.get("enabled_gates") or GATE_ORDER
    for gate in GATE_ORDER:
        if gate not in enabled:
            continue
        if gate == "dedup":
            digest = episode_content_hash(ep, spec)
            metrics["content_hash"] = digest
            if digest in seen_hashes:
                reasons.append(f"dedup:duplicate_of:{seen_hashes[digest]}")
                metrics["duplicate_of"] = seen_hashes[digest]
                first_gate = first_gate or "dedup"
            else:
                seen_hashes[digest] = ep.episode_id
            continue
        fails = GATE_FUNCS[gate](ep, spec)
        if fails:
            first_gate = first_gate or gate
            for reason, mkey, mval in fails:
                reasons.append(reason)
                metrics[mkey] = mval
    primary = ep.streams[0] if ep.streams else None
    if primary is not None:
        if primary.t is not None and primary.n > 1:
            metrics.setdefault("duration_s", round(float(primary.t[-1] - primary.t[0]), 3))
        elif ep.fps_nominal:
            metrics.setdefault("duration_s", round(primary.n / float(ep.fps_nominal), 3))
        metrics.setdefault("n_frames", primary.n)
    return {
        "episode_id": ep.episode_id,
        "group_id": ep.group_id,
        "keep": not reasons,
        "first_failed_gate": first_gate,
        "reasons": reasons,
        "metrics": metrics,
    }


def make_record(ep: EpisodePayload, source_id: str, split: str, result: dict) -> ClipManifestRecord:
    primary = ep.streams[0] if ep.streams else None
    fps = ep.fps_nominal
    if fps is None and primary is not None and primary.t is not None and primary.n > 1:
        med_dt = float(np.median(np.diff(primary.t)))
        fps = round(1.0 / med_dt, 3) if med_dt > 0 else None
    descriptor = ClipDescriptor(
        clip_id=ep.episode_id,
        clip_name=ep.episode_id,
        storage_kind=STORAGE_NATIVE_EPISODE,
        root_dir=ep.root_dir,
        seq_folder=ep.seq_folder,
        frame_names=[],
        fps=fps,
        frame_count_override=primary.n if primary is not None else None,
        extra={
            "robot_episode": True,
            **ep.native,
            "streams": {s.name: {"shape": list(s.q.shape), "kind": s.kind} for s in ep.streams},
        },
    )
    return ClipManifestRecord(
        clip_id=ep.episode_id,
        source_id=source_id,
        split=split,
        descriptor=descriptor,
        group_id=ep.group_id,
        metadata={**ep.meta, "qc_metrics": {k: v for k, v in result["metrics"].items() if isinstance(v, (int, float, str))}},
    )


def build_report(dataset: str, spec_path: str, spec: dict, results: list[dict], output_manifest: Path) -> dict:
    kept = [r for r in results if r["keep"]]
    dropped = [r for r in results if not r["keep"]]
    funnel = OrderedDict()
    remaining = len(results)
    enabled = spec.get("enabled_gates") or GATE_ORDER
    for gate in GATE_ORDER:
        if gate not in enabled:
            continue
        n_drop = sum(1 for r in dropped if r["first_failed_gate"] == gate)
        funnel[gate] = {"entered": remaining, "dropped": n_drop, "passed": remaining - n_drop}
        remaining -= n_drop
    reason_counts = Counter()
    for r in dropped:
        reason_counts.update({re.sub(r":ep\d+.*$", "", reason) for reason in r["reasons"]})
    kept_hours = sum(float(r["metrics"].get("duration_s") or 0.0) for r in kept) / 3600.0
    report = {
        "dataset": dataset,
        "robot_spec": spec_path,
        "resolved_spec": spec,
        "output_manifest": str(output_manifest.resolve()),
        "total_episodes": len(results),
        "kept_episodes": len(kept),
        "kept_hours": round(kept_hours, 3),
        "dropped_episodes": len(dropped),
        "funnel": funnel,
        "reason_counts": dict(sorted(reason_counts.items())),
        "example_rejections": [
            {"episode_id": r["episode_id"], "reasons": r["reasons"], "metrics": r["metrics"]}
            for r in dropped[:3]
        ],
        "dropped": dropped,
    }
    if spec.get("license_note"):
        report["license_note"] = spec["license_note"]
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robot-episode QC gate framework")
    p.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--data_root", required=True, help="Local sample root (or gs:// prefix for dexwild)")
    p.add_argument("--robot_spec", required=True, help="configs/robot_specs/<robot>.yaml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit", type=int, default=None, help="Max episodes (smoke tests)")
    p.add_argument("--source_id", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--calibrate", action="store_true", help="Print data-driven limit suggestions instead of gating")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = yaml.safe_load(Path(args.robot_spec).read_text()) if Path(args.robot_spec).exists() else {}
    adapter = ADAPTERS[args.dataset]
    data_root = args.data_root if args.dataset == "dexwild" and str(args.data_root).startswith("gs://") else Path(args.data_root)
    episodes = adapter(data_root, args.limit)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.calibrate:
        calib = run_calibration(episodes)
        calib_path = out_dir / "calibration.json"
        calib_path.write_text(json.dumps(calib, indent=2))
        print(json.dumps(calib, indent=2))
        print(f"\nWrote {calib_path}", file=sys.stderr)
        return calib

    source_id = args.source_id or f"{args.dataset}_robot"
    seen_hashes: dict[str, str] = {}
    results, records = [], []
    n = 0
    for ep in episodes:
        result = evaluate_episode(ep, spec, seen_hashes)
        results.append(result)
        if result["keep"]:
            records.append(make_record(ep, source_id, args.split, result))
        n += 1
        if n % 25 == 0:
            print(f"[{args.dataset}] processed {n} episodes ({sum(r['keep'] for r in results)} kept)", file=sys.stderr)

    manifest_path = out_dir / "manifest.jsonl"
    write_clip_manifest(records, manifest_path)
    report = build_report(args.dataset, args.robot_spec, spec, results, manifest_path)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== {args.dataset} QC funnel ===")
    for gate, row in report["funnel"].items():
        print(f"  {gate:16s} entered={row['entered']:4d} dropped={row['dropped']:4d}")
    print(f"  kept {report['kept_episodes']}/{report['total_episodes']}")
    print(f"manifest: {manifest_path}\nreport:   {report_path}")
    return report


if __name__ == "__main__":
    main()
