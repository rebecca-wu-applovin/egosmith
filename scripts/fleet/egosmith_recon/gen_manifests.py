#!/usr/bin/env python
"""Build per-mode clip manifests + fleet maps for the EgoSmith L4 reconstruction fleet.

From each dataset's Layer-C `clip_manifest.filtered.jsonl` (the kept clips), emit — per
run mode — a single global manifest whose descriptors point at *node-local* canonical
paths (frames tar + fresh output seq_folder), plus a parallel TSV "fleet map" that
`pod_entry.sh` uses to pull each clip's frame tar (and, for --use_gt, its GT) from GCS
and to push the produced seq_folder back.

Two modes:
  recon   — full reconstruction (no GT); all four datasets.
  use_gt  — GT loaded at slam+infiller; only datasets with injectable GT (taco/oakink/hot3d).

The reconstruction output NEVER touches the GT dirs: seq_folder is remapped to a fresh
per-mode root, so the original `outputs/<clip>/` (GT) is left intact.

Usage (run in the egosmith env):
  python scripts/fleet/egosmith_recon/gen_manifests.py --out_dir /root/egosmith_recon/manifests
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# repo `src/` on the path so `lib.*` imports resolve when run from the repo root.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from lib.pipeline.clips.clip_manifest import load_clip_manifest, write_clip_manifest  # noqa: E402

GCS_FILTERED = "gs://foundational-research/hoi-dataset/egosmith_filtered"
GCS_OUT_BASE = "gs://foundational-research/hoi-dataset/egosmith_recon"

# dataset -> config. `gcs_name` is the sub-prefix under egosmith_filtered/ and egosmith_recon/.
DATASETS = {
    "taco": {
        "manifest": "/root/taco/filter_run/clip_manifest.filtered.jsonl",
        "gcs_name": "taco",
        "has_gt": True,
    },
    "oakink_grasp": {
        "manifest": "/root/oakink/grasp/filter_run/clip_manifest.filtered.jsonl",
        "gcs_name": "oakink_grasp",
        "has_gt": True,
    },
    "hot3d": {
        "manifest": "/root/hot3d/filter_run/clip_manifest.filtered.jsonl",
        "gcs_name": "hot3d",
        "has_gt": True,
    },
    "egodex": {
        "manifest": "/root/egodex/filter_run/clip_manifest.filtered.jsonl",
        "gcs_name": "egodex",
        "has_gt": False,  # native lowdim GT only; reconstruction-only, no --use_gt
    },
}

MODES = {
    "recon": {"use_gt": False},   # all datasets
    "use_gt": {"use_gt": True},   # gt datasets only
}


def build_mode(mode: str, local_root: Path, out_dir: Path) -> tuple[int, Path, Path]:
    """Write the combined manifest + fleet map for one mode. Returns (n_clips, manifest, map)."""
    want_gt = MODES[mode]["use_gt"]
    all_records = []
    map_lines = ["\t".join([
        "dataset", "mode", "clip_id",
        "gcs_frames_tar", "local_frames_tar",
        "gcs_gt_dir", "local_seq_folder", "gcs_out_dir",
    ])]

    for ds, cfg in DATASETS.items():
        if want_gt and not cfg["has_gt"]:
            continue
        gcs_name = cfg["gcs_name"]
        records = load_clip_manifest(cfg["manifest"])
        frames_dir = local_root / "frames" / ds
        seq_root = local_root / mode / ds / "outputs"
        for rec in records:
            d = rec.descriptor
            clip = d.clip_id
            local_tar = frames_dir / f"{clip}.tar"
            local_seq = seq_root / clip
            # rewrite descriptor to node-local canonical paths (pods populate these)
            d.root_dir = str(frames_dir)
            d.shard_path = str(local_tar)
            d.seq_folder = str(local_seq)
            all_records.append(rec)

            gcs_frames_tar = f"{GCS_FILTERED}/{gcs_name}/frames/{clip}.tar"
            gcs_gt_dir = f"{GCS_FILTERED}/{gcs_name}/outputs/{clip}" if (want_gt and cfg["has_gt"]) else "-"
            gcs_out_dir = f"{GCS_OUT_BASE}/{gcs_name}/{mode}/outputs/{clip}"
            map_lines.append("\t".join([
                ds, mode, clip,
                gcs_frames_tar, str(local_tar),
                gcs_gt_dir, str(local_seq), gcs_out_dir,
            ]))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{mode}.manifest.jsonl"
    map_path = out_dir / f"{mode}.fleetmap.tsv"
    write_clip_manifest(all_records, manifest_path)
    map_path.write_text("\n".join(map_lines) + "\n", encoding="utf-8")
    return len(all_records), manifest_path, map_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/root/egosmith_recon/manifests")
    ap.add_argument("--local_root", default="/scratch/egosmith_recon",
                    help="node-local root pods use for frames + fresh outputs")
    ap.add_argument("--modes", default="recon,use_gt")
    args = ap.parse_args()

    local_root = Path(args.local_root)
    out_dir = Path(args.out_dir)
    for mode in args.modes.split(","):
        mode = mode.strip()
        n, mpath, mapp = build_mode(mode, local_root, out_dir)
        print(f"[{mode}] {n:>5d} clips -> {mpath}  (+ {mapp.name})")


if __name__ == "__main__":
    main()
