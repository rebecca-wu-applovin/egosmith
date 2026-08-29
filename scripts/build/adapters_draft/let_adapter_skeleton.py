#!/usr/bin/env python3
"""DRAFT adapter skeleton: LET-Base-Dataset (dex_hand subset) -> robot_episode_qc payloads.

STATUS: DRAFT ONLY -- NOT wired into robot_episode_qc.ADAPTERS. Do not use in production.
Written during the CAT4 category-filtering sweep as the starting point for a real adapter.

Dataset facts (verified against the GCS mirror on 2026-08-20):
- Source: gs://foundational-research/hoi-dataset/LET-Base-Dataset/ -- MERGED mirror of
  HuggingFace LejuRobotics/LET-Base-Dataset + ModelScope lejurobot/LET-Base-Dataset
  (neither is a superset; see PROVENANCE.md in the bucket). dex_hand subset:
  HF 4,454 bags / 2.33 TB + ModelScope 22,681 bags / 23.04 TB (~2,283 shared)
  -> ~24.9K unique episodes, ~380 h (sampled-sidecar estimate: mean bag ~55 s).
- Platform: LejuRobotics Kuavo 4 Pro full-size humanoid (1.66 m / 55 kg / 40 DoF
  whole-body). License: CC BY-NC-SA 4.0 (non-commercial!).
- HAND MODEL / DoF (CAT4 category-fit caveat): the "dexterous hand" streams are
  6 DoF PER HAND -- /dexhand/state is a 12-joint JointState (first 6 = left hand,
  last 6 = right hand); commanded via /control_robot_hand_position as 6 floats per
  hand normalized to [0,100] (0 open, 100 closed). This is BELOW the Category-4
  ">12 actuated DoF" per-hand bar (12 only when summed across BOTH hands) --
  keep-list decision required before ingesting.
- Episode layout: datasets/rosbag/real/Labelled/<task>-P4-dex_hand/<episode>.bag
  (+ sibling .json sidecar with scene metadata and per-step skill marks incl.
  durations). leju_claw variants exist alongside; SKIP those for the dex-hand QC.
- Relevant rosbag topics for QC streams:
    /dexhand/state                sensor_msgs/JointState  12 pos/vel/effort (L6+R6)   measured
    /control_robot_hand_position  kuavo_msgs/robotHandPosition  2x6 in [0,100]        commanded
                                  (normalize to rad-equivalents or gate in native units;
                                   pos limits must then be expressed in the SAME units)
    /kuavo_arm_traj               sensor_msgs/JointState  14 arm joints               commanded
    /sensors_data_raw             kuavo_msgs/sensorsData  28 joints (12 leg + 14 arm
                                  + 2 head) + IMU                                    measured
    /cam_{h,l,r}/color/...        CompressedImage -> frame counts for video_sync
- Tactile/force: hand joint EFFORT (currents, A) in /dexhand/state; treat as kind="aux"
  (no joint-limit gates) but usable by a Cat-2.5-style tactile sanity validator later.

Blockers before wiring:
1. rosbag reading without a ROS install: use the `rosbags` pip package (pure-python,
   reads bag1/bag2 + custom msg defs -- kuavo_msgs/* need message definitions extracted
   from the bag connection headers, which rosbags supports).
2. Remote reads: bags average ~200-400 MB; either stream via gcsfs ranged reads
   (rosbags accepts file-like objects) or stage per-episode locally (preferred: bags are
   small enough, unlike DexWild's 100 GB monolith).
3. configs/robot_specs/let_kuavo4pro.yaml must be created via --calibrate on a
   ~50-episode sample; hand channels are in [0,100] command units vs rad state units --
   keep streams separate, do NOT mix units in one stream.
4. Keep-list decision on the 6-DoF-per-hand category fit (see above).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

# Intentionally NOT importing from robot_episode_qc at module level -- this file must not
# create an import-time dependency until the adapter is real. Signatures mirror the
# iter_<dataset>() adapters in scripts/build/robot_episode_qc.py.


def iter_let_dexhand(data_root: Path, limit: Optional[int]) -> Iterator["EpisodePayload"]:  # noqa: F821
    """Yield EpisodePayload per LET dex_hand rosbag episode.

    Planned mapping (per episode = one .bag + one .json sidecar):
      streams = [
        JointStream("dexhand_state", q=(T,12) positions from /dexhand/state,
                    t=header stamps, kind="measured"),
        JointStream("dexhand_command", q=(T,12) from /control_robot_hand_position
                    (L6+R6, native [0,100] units), t=stamps, kind="commanded"),
        JointStream("arm_traj_command", q=(T,14) from /kuavo_arm_traj, kind="commanded"),
        JointStream("body_state", q=(T,28) from /sensors_data_raw joint_data.position,
                    t=sensor_time, kind="measured"),
        JointStream("dexhand_effort", q=(T,12) currents, kind="aux"),  # tactile-ish
      ]
      fps_nominal = None (per-message header stamps drive dt)
      video_frame_counts = {cam: n_messages(/cam_x/color/...)} ; video_expected_frames
        from the densest state stream resampled to the camera rate (topic rates differ --
        video_sync gate config needs a rate-aware expected count, NOT raw state length)
      meta = sidecar JSON (scene codes, skill marks, device serial)
      native = {"format": "let_rosbag", "gcs_bag": ..., "sidecar_json": ...,
                "end_effector": "dex_hand", "license": "CC BY-NC-SA 4.0"}
    """
    raise NotImplementedError(
        "DRAFT skeleton -- see module docstring blockers (rosbags reader, kuavo_msgs "
        "definitions, let_kuavo4pro.yaml spec, category-fit decision)."
    )
