#!/usr/bin/env python3
"""Extract HumanTouch (Xspark-HumanTouch) head-camera calibration into one table.

HumanTouch ships per-episode calibration sidecars the earlier probe missed, at
``<TASK>/meta/humantouch/episodes/episode_XXXXXX.json`` (NOT meta/info.json),
top-level key ``camera_calibration``:

- ``Calibration/camera/calibration_intrinsics_<DEVICE>.json``: per capture-device
  pinhole intrinsics for the 1920x1080 head cam only (wrist cams have none).
  ChArUco-calibrated (14x9, DICT_5X5_100), RMS 0.24-0.31 px, full OpenCV 5-coeff
  distortion. Device ids seen: A1-A3, C1-C4, D1, D2.
- ``Calibration/camera/calibration_extrinsics.json``: a single
  ``camera_from_head_tracker`` transform (translation_m [-0.034, 0.137, -0.028],
  rotation_rpy_deg [50, 178, 180], ``unity_to_opencv_y_flip: true``), byte-identical
  across ALL tasks/episodes/devices and described as the offset used by the
  provider's ``render_egocentric_skeleton_overlay.py``. Whole-degree rotations =>
  hand-tuned viz offset, NOT per-unit optimized; treat as an initialization only.

IMPORTANT (verified on the GCS mirror): the capture device VARIES WITHIN a task
(e.g. X002 episode_000001 is device D2 but episode_009942 is device A3), so the
"read one sidecar per task" shortcut is unsafe. This script reads EVERY episode
sidecar and emits a per-episode device map plus a deduplicated per-device
intrinsics table.

Output JSON:
{
  "extrinsic": {...},                       # the single shared block (verified unique)
  "devices": {dev: {intrinsics json, "calib_md5": ...}},
  "episode_device": {task: {episode_key: dev}},
  "stats": {...}
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

INTR_RE = re.compile(r"calibration_intrinsics_(\w+)\.json$")
EXTR_KEY = "Calibration/camera/calibration_extrinsics.json"
DEFAULT_SRC = "gs://foundational-research/hoi-dataset/Xspark-HumanTouch"


def list_tasks(src: str) -> list[str]:
    if src.startswith("gs://"):
        out = subprocess.run(["gsutil", "ls", src.rstrip("/") + "/"],
                             check=True, capture_output=True, text=True).stdout
        return sorted(p.rstrip("/").rsplit("/", 1)[-1]
                      for p in out.splitlines() if p.endswith("/"))
    return sorted(p.name for p in Path(src).iterdir() if p.is_dir())


def stage_task_sidecars(src: str, task: str, workdir: Path) -> Path:
    """Mirror <task>/meta/humantouch/episodes/*.json into workdir (cached)."""
    dst = workdir / task
    if src.startswith("gs://"):
        dst.mkdir(parents=True, exist_ok=True)
        uri = f"{src.rstrip('/')}/{task}/meta/humantouch/episodes/*"
        # -n: skip files already staged from a previous (partial) run
        subprocess.run(["gsutil", "-m", "-q", "cp", "-n", uri, str(dst) + "/"], check=True)
        return dst
    return Path(src) / task / "meta" / "humantouch" / "episodes"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="dataset root (gs:// or local mirror)")
    ap.add_argument("--workdir", default="/root/w7_full/humantouch/calib_sidecars",
                    help="staging dir for downloaded sidecars (cached across runs)")
    ap.add_argument("--out", default="/root/w7_full/humantouch/humantouch_calibration.json")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="subset of tasks (default: all)")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    tasks = args.tasks or list_tasks(args.src)
    tasks = [t for t in tasks if re.fullmatch(r"X\d+", t)]
    print(f"tasks: {tasks}")

    devices: dict[str, dict] = {}
    device_md5: dict[str, str] = {}
    extrinsics: dict[str, dict] = {}
    episode_device: dict[str, dict[str, str]] = {}
    n_eps = 0
    dev_counter: Counter[str] = Counter()

    for task in tasks:
        ep_dir = stage_task_sidecars(args.src, task, workdir)
        files = sorted(ep_dir.glob("episode_*.json"))
        if not files:
            print(f"WARNING: no sidecars for {task}", file=sys.stderr)
            continue
        episode_device[task] = {}
        for f in files:
            d = json.loads(f.read_text())
            cc = d.get("camera_calibration")
            ep_key = d.get("episode_key") or f.stem.split("_")[-1]
            if not cc:
                print(f"WARNING: {task}/{f.name}: no camera_calibration", file=sys.stderr)
                continue
            n_eps += 1
            extr = cc.get(EXTR_KEY)
            if extr is not None:
                extrinsics[hashlib.md5(
                    json.dumps(extr, sort_keys=True).encode()).hexdigest()] = extr
            devs = [(INTR_RE.search(k).group(1), v)
                    for k, v in cc.items() if INTR_RE.search(k)]
            if len(devs) != 1:
                print(f"WARNING: {task}/{f.name}: {len(devs)} intrinsics blocks",
                      file=sys.stderr)
                continue
            dev, intr = devs[0]
            md5 = hashlib.md5(json.dumps(intr, sort_keys=True).encode()).hexdigest()
            if dev in device_md5 and device_md5[dev] != md5:
                print(f"ERROR: device {dev} has conflicting intrinsics "
                      f"({task}/{f.name})", file=sys.stderr)
                return 1
            device_md5[dev] = md5
            devices[dev] = intr
            episode_device[task][ep_key] = dev
            dev_counter[dev] += 1
        print(f"{task}: {len(files)} sidecars, devices "
              f"{sorted(set(episode_device[task].values()))}")

    if len(extrinsics) != 1:
        print(f"ERROR: expected 1 unique extrinsic, got {len(extrinsics)}",
              file=sys.stderr)
        return 1

    out = {
        "source": args.src,
        "extrinsic": next(iter(extrinsics.values())),
        "devices": {d: {"intrinsics": v, "calib_md5": device_md5[d]}
                    for d, v in sorted(devices.items())},
        "episode_device": episode_device,
        "stats": {
            "n_tasks": len(episode_device),
            "n_episodes": n_eps,
            "episodes_per_device": dict(dev_counter),
            "tasks_with_multiple_devices": sorted(
                t for t, m in episode_device.items() if len(set(m.values())) > 1),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {n_eps} episodes, {len(devices)} devices, "
          f"multi-device tasks: {out['stats']['tasks_with_multiple_devices']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
