#!/usr/bin/env python3
"""Contact sheets for TACO clips: sampled frames with projected MANO GT joints.

Used two ways:
- smoke gate: visually verify the TACO->EgoSmith convention conversion (projected
  joints must lock onto the hands in RGB) before the full filter run;
- failure package: one sheet per dropped clip, organized by primary drop reason,
  with reasons + key metrics in the caption.

Joints are computed with the pipeline's own MANO models + world_space_res.pth and
projected with the converted SLAM extrinsics/intrinsics, i.e. exactly the data the
quality filter sees.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LEFT_COLOR = (66, 133, 244)    # blue = left hand
RIGHT_COLOR = (219, 68, 55)    # red = right hand
FLAG_COLOR = (255, 200, 0)     # yellow border = offending frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TACO overlay contact sheets")
    parser.add_argument("--manifest", required=True, help="Clip manifest JSONL")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--clip_ids", default=None, help="Comma-separated clip ids (default: all in manifest)")
    parser.add_argument("--include", default=None, help="Regex on clip_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--filter_report", default=None, help="filter_report.json; when given, sheets are grouped by primary reason with captions")
    parser.add_argument("--num_tiles", type=int, default=12)
    parser.add_argument("--tile_width", type=int, default=480)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary_out", default=None, help="Optional JSON with per-clip projection sanity stats")
    return parser


def load_frames(descriptor, frame_indices: list[int]) -> dict[int, "object"]:
    from PIL import Image

    wanted = {descriptor.frame_names[i]: i for i in frame_indices}
    images = {}
    with tarfile.open(descriptor.shard_path, "r") as tar_reader:
        for member in tar_reader:
            if member.name in wanted:
                data = tar_reader.extractfile(member).read()
                images[wanted[member.name]] = Image.open(io.BytesIO(data)).convert("RGB")
    return images


def compute_world_joints(seq_folder: Path, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_joints, right_joints) as (T, 21, 3) world-space arrays."""
    import torch
    from lib.pipeline.exporters.mano_features import _compute_hand_joints, build_mano_models

    payload = joblib.load(seq_folder / "world_space_res.pth")
    pred_trans, pred_rot, pred_hand_pose, pred_betas, _ = [torch.from_numpy(np.asarray(item)) for item in payload]
    mano_right, mano_left = build_mano_models(device)
    right = _compute_hand_joints(mano_right, pred_trans, pred_rot, pred_hand_pose, pred_betas, hand_index=1, device=device)
    left = _compute_hand_joints(mano_left, pred_trans, pred_rot, pred_hand_pose, pred_betas, hand_index=0, device=device)
    return left.cpu().numpy(), right.cpu().numpy()


def load_camera(seq_folder: Path, num_frames: int) -> tuple[np.ndarray, np.ndarray]:
    from lib.pipeline.exporters.camera_features import _load_episode_camera_features

    ep = {"crop_dir": str(seq_folder), "episode_id": seq_folder.name}
    extrinsics, intrinsic = _load_episode_camera_features(ep, num_frames)
    return np.asarray(extrinsics), np.asarray(intrinsic)


def project(points_world: np.ndarray, w2c: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """(J,3) world -> (J,3) [u, v, z_cam]."""
    homo = np.concatenate([points_world, np.ones((points_world.shape[0], 1))], axis=1)
    cam = (w2c @ homo.T).T[:, :3]
    z = np.clip(cam[:, 2:3], 1e-6, None)
    fx, fy, cx, cy = intrinsic
    u = fx * cam[:, 0:1] / z + cx
    v = fy * cam[:, 1:2] / z + cy
    return np.concatenate([u, v, cam[:, 2:3]], axis=1)


def draw_joints(image, uvz: np.ndarray, color, radius: int = 4):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for u, v, z in uvz:
        if not np.isfinite([u, v]).all() or z <= 0:
            continue
        draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=color)
    # wrist ring
    if np.isfinite(uvz[0, :2]).all() and uvz[0, 2] > 0:
        u, v = uvz[0, :2]
        draw.ellipse([u - radius * 2, v - radius * 2, u + radius * 2, v + radius * 2], outline=color, width=2)


def detect_offending_frames(left_joints, right_joints, extrinsics, intrinsic, image_wh,
                            fatal_scale: float = 1.4, max_examples: int = 6) -> set[int]:
    """Frames that visibly show a drop cause, computed from the actual projection.

    Covers the dominant TACO drop reasons: a hand fully off-screen (streak / in-frame
    ratio), a hand severely off-screen (fatal), and the single largest wrist-translation
    and camera-rotation glitch frames. This is reason-agnostic and always reflects the
    real data, unlike the aggregate-count metrics in the report.
    """
    width, height = image_wh
    T = left_joints.shape[0]
    fatal = set()
    out_of_frame = set()
    for frame_idx in range(T):
        for joints in (left_joints, right_joints):
            uvz = project(joints[frame_idx], extrinsics[frame_idx], intrinsic)
            u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not inside.any():
                out_of_frame.add(frame_idx)
            beyond = (u < -0.4 * width) | (u > fatal_scale * width) | (v < -0.4 * height) | (v > fatal_scale * height)
            if (z > 0).any() and beyond.all():
                fatal.add(frame_idx)

    flagged = set(sorted(fatal)[:max_examples]) | set(sorted(out_of_frame)[:max_examples])
    # largest per-frame wrist translation + camera rotation glitch
    if T > 1:
        wrist = np.stack([left_joints[:, 0], right_joints[:, 0]], axis=0)  # (2,T,3)
        wrist_step = np.linalg.norm(np.diff(wrist, axis=1), axis=2).max(axis=0)  # (T-1,)
        flagged.add(int(np.argmax(wrist_step)) + 1)
        rot = extrinsics[:, :3, :3]
        cam_rot_step = np.linalg.norm(np.diff(rot, axis=0).reshape(T - 1, -1), axis=1)
        flagged.add(int(np.argmax(cam_rot_step)) + 1)
    return {i for i in flagged if 0 <= i < T}


def make_sheet(descriptor, seq_folder: Path, device, num_tiles: int, tile_width: int,
               caption: str = "", detect_flagged: bool = False):
    from PIL import Image, ImageDraw

    left_joints, right_joints = compute_world_joints(seq_folder, device)
    T = min(len(descriptor.frame_names), left_joints.shape[0])
    left_joints, right_joints = left_joints[:T], right_joints[:T]
    extrinsics, intrinsic = load_camera(seq_folder, T)

    image_wh = _descriptor_image_wh(descriptor)
    flagged_frames = detect_offending_frames(left_joints, right_joints, extrinsics, intrinsic, image_wh) if detect_flagged else set()
    flagged = sorted(flagged_frames)
    sample = sorted(set(np.linspace(0, T - 1, num_tiles, dtype=int).tolist()) | set(flagged[:num_tiles]))[: num_tiles + 4]
    images = load_frames(descriptor, sample)

    inframe_hits, inframe_total = 0, 0
    tiles = []
    for frame_idx in sample:
        image = images.get(frame_idx)
        if image is None:
            continue
        scale = tile_width / image.width
        tile = image.resize((tile_width, int(image.height * scale)))
        for joints, color in ((left_joints, LEFT_COLOR), (right_joints, RIGHT_COLOR)):
            uvz = project(joints[frame_idx], extrinsics[frame_idx], intrinsic) * np.array([scale, scale, 1.0])
            draw_joints(tile, uvz, color)
            wrist_u, wrist_v, wrist_z = uvz[0]
            inframe_total += 1
            if wrist_z > 0 and 0 <= wrist_u < tile_width and 0 <= wrist_v < tile.height:
                inframe_hits += 1
        draw = ImageDraw.Draw(tile)
        label = f"f{frame_idx}"
        if frame_idx in (flagged_frames or set()):
            draw.rectangle([0, 0, tile.width - 1, tile.height - 1], outline=FLAG_COLOR, width=6)
            label += " !"
        draw.rectangle([0, 0, 90, 22], fill=(0, 0, 0))
        draw.text((6, 4), label, fill=(255, 255, 255))
        tiles.append(tile)

    if not tiles:
        raise RuntimeError(f"No frames decoded for {descriptor.clip_id}")

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tile_h = tiles[0].height
    caption_h = 72 if caption else 28
    sheet = Image.new("RGB", (cols * tile_width, rows * tile_h + caption_h), (16, 16, 16))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * tile_width, (idx // cols) * tile_h))
    draw = ImageDraw.Draw(sheet)
    header = f"{descriptor.clip_id}  (T={T}, left=blue right=red)"
    draw.text((8, rows * tile_h + 6), header, fill=(255, 255, 255))
    if caption:
        for line_no, line in enumerate(caption.split("\n")[:3]):
            draw.text((8, rows * tile_h + 24 + 16 * line_no), line[:220], fill=(255, 220, 150))
    stats = {"clip_id": descriptor.clip_id, "frames": int(T),
             "wrist_inframe_ratio": (inframe_hits / inframe_total) if inframe_total else 0.0}
    return sheet, stats


def _descriptor_image_wh(descriptor) -> tuple[int, int]:
    from lib.pipeline.exporters.manifest_build.episodes import _load_descriptor_image_size

    return _load_descriptor_image_size(descriptor)


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from lib.pipeline.clips.clip_manifest import load_clip_manifest

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = load_clip_manifest(args.manifest)
    by_id = {record.clip_id: record for record in records}

    dropped_info = {}
    if args.filter_report:
        report = json.loads(Path(args.filter_report).read_text())
        for item in report.get("dropped", []):
            dropped_info[item["clip_id"]] = item

    if args.clip_ids:
        selected = [cid.strip() for cid in args.clip_ids.split(",") if cid.strip()]
    elif dropped_info:
        selected = list(dropped_info)
    else:
        selected = list(by_id)
    if args.include:
        pattern = re.compile(args.include)
        selected = [cid for cid in selected if pattern.search(cid)]
    if args.limit:
        selected = selected[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_stats, failures = [], []
    for pos, clip_id in enumerate(selected):
        record = by_id.get(clip_id)
        if record is None:
            failures.append({"clip_id": clip_id, "error": "not in manifest"})
            continue
        info = dropped_info.get(clip_id)
        caption, dest = "", out_dir
        if info is not None:
            reasons = info.get("reasons") or ["unknown"]
            caption = "DROPPED: " + "; ".join(reasons)
            dest = out_dir / re.sub(r"[^A-Za-z0-9_.-]+", "-", reasons[0])[:80]
            dest.mkdir(parents=True, exist_ok=True)
        try:
            sheet, stats = make_sheet(
                record.descriptor,
                Path(record.descriptor.seq_folder),
                device,
                args.num_tiles,
                args.tile_width,
                caption=caption,
                detect_flagged=info is not None,
            )
            sheet.save(dest / f"{clip_id}.jpg", quality=88)
            all_stats.append(stats)
        except Exception as error:
            failures.append({"clip_id": clip_id, "error": f"{type(error).__name__}: {error}"})
        if (pos + 1) % 20 == 0 or (pos + 1) == len(selected):
            print(f"[{pos + 1}/{len(selected)}] sheets written", flush=True)

    summary = {
        "sheets": len(all_stats),
        "failures": failures,
        "mean_wrist_inframe_ratio": float(np.mean([s["wrist_inframe_ratio"] for s in all_stats])) if all_stats else None,
        "per_clip": all_stats,
    }
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_clip"}, ensure_ascii=False, indent=2))
    print("TACO_SHEETS_DONE" if not failures else "TACO_SHEETS_DONE_WITH_FAILURES", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
