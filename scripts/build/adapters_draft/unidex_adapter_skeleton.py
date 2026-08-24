#!/usr/bin/env python3
"""DRAFT adapter skeleton: UniDex retargeted multi-embodiment episodes -> robot_episode_qc.

STATUS: DRAFT ONLY -- NOT wired into robot_episode_qc.ADAPTERS, and currently UNRUNNABLE:
the GCS mirror gs://foundational-research/hoi-dataset/UniDex/ holds ONLY
dataset_annotations/ (H2o_annotation.tar.gz, hot3d_prompts.tar.gz) and
pretrained_checkpoints/ -- NO retargeted trajectory data is mirrored (verified
2026-08-20). UniDex is PENDING-DOWNLOAD.

CRITICAL METADATA FLAG (must be stamped on every episode if this dataset is ever
ingested): UniDex trajectories are RETARGETED from human demonstrations to 8 robot
hand embodiments in a shared FAAS action space and were NEVER PHYSICALLY EXECUTED on
real hardware. They carry no guarantee of dynamic feasibility -- velocity/accel/limit
gates are the whole point of QC'ing them, applied PER TARGET EMBODIMENT.

Design decisions locked in for the real adapter:
- Per-embodiment robot specs: configs/robot_specs/unidex_<hand>.yaml, one per target
  hand (8 hands), each with that hand's published joint limits; velocity/accel limits
  from --calibrate on that embodiment's slice (retargeted data has no hardware-derived
  dynamics, so data-driven limits mainly catch retarget blowups/teleports).
- episode_id namespaced as unidex_<hand>_<source>_<idx> so cross-embodiment copies of
  the same source demo are NOT deduped against each other (same underlying motion is
  expected 8x); the dedup gate stays per-embodiment via group_id = <hand>.
- native{} extra must carry: {"format": "unidex_retargeted", "embodiment": <hand>,
  "action_space": "FAAS", "never_physically_executed": True,
  "source_demo": <human dataset provenance>}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

#: Placeholder for UniDex's 8 target embodiments; fill from the release when the
#: trajectory data lands (names TBD -- verify against the paper/repo, do not guess in
#: production code).
UNIDEX_EMBODIMENTS: list[str] = []


def iter_unidex(data_root: Path, limit: Optional[int], embodiment: Optional[str] = None) -> Iterator["EpisodePayload"]:  # noqa: F821
    """Yield EpisodePayload per retargeted UniDex episode for one embodiment.

    Planned shape once trajectories are mirrored:
      streams = [JointStream("qpos", (T, dof_of(embodiment)), t=..., kind="measured"),
                 JointStream("action", (T, faas_dim), kind="commanded")]
      fps_nominal from release metadata; dt from timestamps when present.
      meta = {"never_physically_executed": True, "embodiment": embodiment, ...}
    Gate config comes from configs/robot_specs/unidex_<embodiment>.yaml.
    """
    raise NotImplementedError(
        "DRAFT skeleton -- UniDex trajectories are not in the GCS mirror yet "
        "(PENDING-DOWNLOAD; only annotations + checkpoints present)."
    )
