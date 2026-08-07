#!/usr/bin/env python3
"""Cut OakInk-v2 sequences into per-primitive GRASP sub-clips (acting hand only).

Why this exists: OakInk-v2 sequences are long-horizon complex bimanual TASKS (median
~39 s, up to 5.5 min), so ingesting one clip per sequence and judging BOTH hands makes
the EgoSmith quality filter drop ~100% on off-screen rules (the resting / off-view hand
always trips them over such a long span). For robotics grasping we only want the
manipulation PRIMITIVES (grip / rearrange / take_outside / place_* / pour / ...), each of
which begins with a hand acquiring an object.

OakInk-v2 already delineates these: ``program/program_info/<seq_token>.json`` maps
``"(lh_interval, rh_interval)"`` -> {primitive, interaction_mode, primitive_lh/rh, ...}.
This script emits ONE sub-clip per (primitive, involved hand):
- frames = the acting hand's interval, remapped to a contiguous JPEG tar;
- world_space_res.pth sliced to that range, with ``valid`` set for the ACTING hand only
  (presence bitmask -> the filter judges just that hand);
- SLAM npz sliced to the range; tracks marker + infiller done marker.

It reuses the already-extracted egocentric frame tars on disk (NO 2 TB re-download); it
re-reads only the small per-sequence anno pkls (~37 GB total, streamed + deleted) to
rebuild the GT precisely. Output feeds ``filter_manifest_by_quality.py --stages infiller``
via the ``oakink_tar`` adapter, exactly like the whole-sequence run.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import pickle
import sys
import tarfile
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.pipeline.proc.stage_api import get_stage_done_marker  # noqa: E402
from generate_oakink_world_res import (  # noqa: E402
    EGOCENTRIC,
    build_slam_npz,
    build_world_res,
    clip_id_from_token,
    _gsutil_cp,
)

# Primitives that are prehensile object-acquisition / transport / manipulation — every
# one contains a grasp. Non-grasp actuation (press_button, trigger_lever, use_mouse,
# use_keyboard, use_gamecontroller) is excluded by default. Override with --primitives.
_NON_GRASP = {"press_button", "trigger_lever", "use_mouse", "use_keyboard", "use_gamecontroller"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cut OakInk-v2 into per-primitive grasp sub-clips (acting hand only)")
    p.add_argument("--full_frames_root", required=True, help="Dir with whole-sequence egocentric tars (OAKINK_<seq>.tar)")
    p.add_argument("--program_dir", required=True, help="program/program_info dir (per-seq primitive JSONs)")
    p.add_argument("--frames_root", required=True, help="Output dir for per-primitive sub-clip frame tars")
    p.add_argument("--outputs_root", required=True, help="Output dir for per-primitive seq_folders")
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--work_dir", default="/root/oakink/_work_grasp")
    p.add_argument("--gcs_root", default="gs://foundational-research/hoi-dataset/OakInk-v2")
    p.add_argument("--camera", default=EGOCENTRIC)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min_keep_sec", type=float, default=1.0, help="Drop sub-clips shorter than this (color fps)")
    p.add_argument("--color_fps", type=float, default=30.0)
    p.add_argument("--grasp_onset_sec", type=float, default=None,
                   help="If set, keep only the first N sec of each primitive (approach+grasp onset)")
    p.add_argument("--primitives", default=None, help="Comma-separated primitive allowlist (default: all except non-grasp actuation)")
    p.add_argument("--include", default=None, help="Regex on seq_token")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--source_id", default="oakink_v2_grasp")
    p.add_argument("--split", default="train")
    p.add_argument("--trans_convention", choices=("wrist_root", "transl"), default="wrist_root")
    p.add_argument("--quat_order", choices=("wxyz", "xyzw"), default="wxyz")
    p.add_argument("--extrinsic_direction", choices=("w2c", "c2w"), default="w2c")
    return p


def _involved_hands(seg: dict) -> list[tuple[int, str]]:
    """Return [(hand_index, side)] for hands with a non-null primitive in this segment."""
    hands = []
    if seg.get("primitive_lh"):
        hands.append((0, "lh"))
    if seg.get("primitive_rh"):
        hands.append((1, "rh"))
    return hands


def _seg_interval(key: str, hand_side: str):
    lh_iv, rh_iv = ast.literal_eval(key)  # ((s,e)|None, (s,e)|None)
    return lh_iv if hand_side == "lh" else rh_iv


def _read_full_tar_index(full_tar: Path) -> list[tuple[str, bytes]]:
    """Return the whole-sequence egocentric tar as an ordered list of (name, jpeg_bytes)."""
    frames = []
    with tarfile.open(full_tar, "r") as reader:
        members = [m for m in reader if m.isfile() and m.name.endswith(".image.jpg")]
        members.sort(key=lambda m: m.name)
        for m in members:
            frames.append((m.name, reader.extractfile(m).read()))
    return frames


def _write_subclip_tar(frames: list[tuple[str, bytes]], indices: list[int], tar_path: Path, clip_id: str) -> int:
    tmp = tar_path.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w") as writer:
        for out_idx, src_idx in enumerate(indices):
            payload = frames[src_idx][1]
            info = tarfile.TarInfo(name=f"{clip_id}_f{out_idx:05d}.image.jpg")
            info.size = len(payload)
            writer.addfile(info, io.BytesIO(payload))
    tmp.replace(tar_path)
    return len(indices)


def convert_sequence(seq_token: str, args) -> dict:
    seq_clip = clip_id_from_token(seq_token)
    full_tar = Path(args.full_frames_root) / f"{seq_clip}.tar"
    prog_path = Path(args.program_dir) / f"{seq_token}.json"
    result = {"seq_token": seq_token, "status": "ok", "subclips": [], "skipped_short": 0}
    if not full_tar.is_file():
        return {**result, "status": "failed", "error": f"missing full tar {full_tar}"}
    if not prog_path.is_file():
        return {**result, "status": "failed", "error": f"missing program_info {prog_path}"}

    allow = set(x.strip() for x in args.primitives.split(",")) if args.primitives else None
    min_frames = max(1, int(round(args.min_keep_sec * args.color_fps)))
    onset_frames = int(round(args.grasp_onset_sec * args.color_fps)) if args.grasp_onset_sec else None

    work = Path(tempfile.mkdtemp(prefix=f"{seq_clip}_", dir=args.work_dir))
    local_anno = work / "anno.pkl"
    try:
        _gsutil_cp(f"{args.gcs_root}/anno_preview/{seq_token}.pkl", local_anno)
        with open(local_anno, "rb") as f:
            anno = pickle.load(f)
        cam_def = anno["cam_def"]
        if not any(name == args.camera for name in cam_def.values()):
            return {**result, "status": "failed", "error": f"no {args.camera} camera"}
        raw_mano = anno["raw_mano"]
        cam_extr = anno["cam_extr"][args.camera]
        cam_intr = anno["cam_intr"][args.camera]
        # full egocentric frame_ids in the SAME order the whole-seq converter used
        full_frame_ids = sorted(int(f) for f in cam_extr.keys() if f in raw_mano and f in cam_intr)
        pos_of = {fid: i for i, fid in enumerate(full_frame_ids)}

        program = json.loads(prog_path.read_text())
        full_frames = _read_full_tar_index(full_tar)
        if len(full_frames) != len(full_frame_ids):
            # frame_ids recomputed from anno must match the on-disk tar; guard against drift
            n = min(len(full_frames), len(full_frame_ids))
            full_frame_ids = full_frame_ids[:n]
            pos_of = {fid: i for i, fid in enumerate(full_frame_ids)}

        seg_counter = {}
        for key, seg in program.items():
            primitive = seg.get("primitive") or "unknown"
            if allow is not None and primitive not in allow:
                continue
            if allow is None and primitive in _NON_GRASP:
                continue
            for hand_index, side in _involved_hands(seg):
                iv = _seg_interval(key, side)
                if iv is None:
                    continue
                lo, hi = int(iv[0]), int(iv[1])
                sel_fids = [fid for fid in full_frame_ids if lo <= fid <= hi]
                if onset_frames is not None:
                    sel_fids = sel_fids[:onset_frames]
                if len(sel_fids) < min_frames:
                    result["skipped_short"] += 1
                    continue
                indices = [pos_of[fid] for fid in sel_fids]
                k = seg_counter.get(primitive, 0)
                seg_counter[primitive] = k + 1
                clip_id = f"{seq_clip}__{primitive}_{side}_{k:02d}"
                out_seq_folder = Path(args.outputs_root) / clip_id
                out_tar = Path(args.frames_root) / f"{clip_id}.tar"
                done = get_stage_done_marker(out_seq_folder, "infiller")
                if args.resume and out_tar.is_file() and done.exists():
                    result["subclips"].append({"clip_id": clip_id, "primitive": primitive, "hand": side, "frames": len(sel_fids), "status": "skipped"})
                    continue

                out_seq_folder.mkdir(parents=True, exist_ok=True)
                T = _write_subclip_tar(full_frames, indices, out_tar, clip_id)
                payload = build_world_res(anno, sel_fids, args)  # [trans,rot,hand_pose,betas,valid]
                # acting hand only: zero the other hand's validity (presence bit) so the
                # filter's off-screen rules judge just the grasping hand.
                other = 1 - hand_index
                payload[4] = payload[4].copy()
                payload[4][other, :] = 0.0
                joblib.dump(payload, out_seq_folder / "world_space_res.pth")
                build_slam_npz(out_seq_folder, anno, sel_fids, args.camera, args)
                tracks = out_seq_folder / f"tracks_0_{T}"
                tracks.mkdir(parents=True, exist_ok=True)
                (tracks / ".oakink_gt").write_text(json.dumps({"clip_id": clip_id, "frames": T, "primitive": primitive, "hand": side}))
                done.parent.mkdir(parents=True, exist_ok=True)
                done.touch()
                result["subclips"].append({
                    "clip_id": clip_id, "primitive": primitive, "hand": side,
                    "frames": T, "obj": seg.get(f"obj_list_{side}"), "status": "ok",
                    "scene": seq_token.split("__", 1)[0],
                })
    except Exception as error:
        import subprocess
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            result["error"] += " :: " + error.stderr.decode("utf8", "replace")[-200:]
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    return result


def _convert_star(task):
    return convert_sequence(*task)


def write_manifest(results: list[dict], args) -> int:
    from lib.pipeline.clips.clip_manifest import write_clip_manifest, ClipManifestRecord
    from lib.pipeline.datasets.descriptors import ClipDescriptor

    records = []
    for res in results:
        for sc in res.get("subclips", []):
            if sc["status"] not in ("ok", "skipped"):
                continue
            clip_id = sc["clip_id"]
            tar_path = Path(args.frames_root) / f"{clip_id}.tar"
            if not tar_path.is_file():
                continue
            frame_names, frame_offsets = [], []
            with tarfile.open(tar_path, "r") as reader:
                members = [m for m in reader if m.isfile() and m.name.endswith(".image.jpg")]
            members.sort(key=lambda m: m.name)
            for m in members:
                frame_names.append(m.name)
                frame_offsets.append([int(m.offset_data), int(m.size)])
            scene = res["seq_token"].split("__", 1)[0]
            descriptor = ClipDescriptor.from_tar_shard(
                clip_id=clip_id, clip_name=clip_id,
                root_dir=str(Path(args.frames_root).resolve()),
                seq_folder=str((Path(args.outputs_root) / clip_id).resolve()),
                shard_path=str(tar_path.resolve()),
                frame_names=frame_names, frame_offsets=frame_offsets,
                extra={"adapter": "oakink_tar", "dataset_name": args.source_id, "camera": args.camera,
                       "seq_token": res["seq_token"], "scene": scene,
                       "primitive": sc["primitive"], "hand": sc["hand"]},
            )
            records.append(ClipManifestRecord(clip_id=clip_id, source_id=args.source_id, split=args.split,
                                              descriptor=descriptor, group_id=sc["primitive"]))
    write_clip_manifest(records, args.manifest_out)
    return len(records)


def main() -> int:
    args = build_parser().parse_args()
    for d in (args.frames_root, args.outputs_root, args.work_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    tokens = sorted(p.stem for p in Path(args.program_dir).glob("*.json"))
    if args.include:
        import re
        pat = re.compile(args.include)
        tokens = [t for t in tokens if pat.search(t)]
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"OakInk sequences to sub-clip: {len(tokens)}", flush=True)

    started = time.perf_counter()
    if args.workers <= 1:
        results = [convert_sequence(t, args) for t in tokens]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = []
            for idx, r in enumerate(pool.imap_unordered(_convert_star, [(t, args) for t in tokens], chunksize=1)):
                results.append(r)
                if (idx + 1) % 10 == 0 or (idx + 1) == len(tokens):
                    n = sum(len(x.get("subclips", [])) for x in results)
                    print(f"[{idx + 1}/{len(tokens)}] {r['seq_token'][:40]} {r['status']} (subclips so far: {n})", flush=True)

    manifest_count = write_manifest(results, args)
    from collections import Counter
    prim_counts = Counter()
    for r in results:
        for sc in r.get("subclips", []):
            if sc["status"] in ("ok", "skipped"):
                prim_counts[sc["primitive"]] += 1
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "gcs_root": args.gcs_root, "camera": args.camera,
        "sequences": len(tokens),
        "subclips_total": sum(len(r.get("subclips", [])) for r in results),
        "manifest_records": manifest_count,
        "skipped_short": sum(r.get("skipped_short", 0) for r in results),
        "failed_sequences": len(failed),
        "per_primitive": dict(prim_counts.most_common()),
        "conventions": {"trans_convention": args.trans_convention, "quat_order": args.quat_order,
                        "extrinsic_direction": args.extrinsic_direction, "min_keep_sec": args.min_keep_sec,
                        "grasp_onset_sec": args.grasp_onset_sec},
        "failures": failed[:20],
        "elapsed_sec": time.perf_counter() - started,
    }
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False, indent=2))
    print("OAKINK_GRASP_CLIPS_DONE" if not failed else "OAKINK_GRASP_CLIPS_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
