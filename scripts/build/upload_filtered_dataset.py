#!/usr/bin/env python3
"""Publish a locally-built filtered dataset to the canonical GCS layout.

Rewrites manifest descriptor paths (root_dir/shard_path/seq_folder) from local to gs://,
then uploads frames tars, (use_gt mode) per-clip stage outputs, and the filter_run files.

Layouts (docs/filtered_dataset.md):
  use_gt: frames -> egosmith_filtered/<ds>/frames/, outputs -> egosmith_recon/<ds>/use_gt/outputs/
  native: frames -> egosmith_filtered/<ds>/frames/, seq_folder cleared (GT lives in the tars)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BUCKET = "gs://foundational-research/hoi-dataset"


def rewrite_manifest(src: Path, dest: Path, ds: str, mode: str) -> int:
    frames_gcs = f"{BUCKET}/egosmith_filtered/{ds}/frames"
    n = 0
    with open(src) as fin, open(dest, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            d = rec["descriptor"]
            tar_name = Path(d["shard_path"]).name
            d["root_dir"] = frames_gcs
            d["shard_path"] = f"{frames_gcs}/{tar_name}"
            if mode == "use_gt":
                d["seq_folder"] = f"{BUCKET}/egosmith_recon/{ds}/use_gt/outputs/{rec['clip_id']}"
            else:
                d["seq_folder"] = ""
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", choices=["use_gt", "native"], required=True)
    p.add_argument("--frames_root", required=True)
    p.add_argument("--outputs_root", default=None, help="use_gt mode: local stage outputs to publish")
    p.add_argument("--manifest", required=True, help="local-path clip_manifest.jsonl (input population)")
    p.add_argument("--filtered_manifest", required=True)
    p.add_argument("--extra_files", nargs="*", default=[], help="reports/FILTER_MODE.txt for filter_run/")
    p.add_argument("--filtered_only_frames", action=argparse.BooleanOptionalAction, default=True,
                   help="upload only tars/outputs referenced by the FILTERED manifest (default)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    ds = args.dataset
    work = Path(args.filtered_manifest).parent
    gcs_run = f"{BUCKET}/egosmith_filtered/{ds}/filter_run"

    kept_ids = [json.loads(l)["clip_id"] for l in open(args.filtered_manifest) if l.strip()]
    out_manifest = work / "clip_manifest.gcs.jsonl"
    out_filtered = work / "clip_manifest.filtered.gcs.jsonl"
    n_all = rewrite_manifest(Path(args.manifest), out_manifest, ds, args.mode)
    n_kept = rewrite_manifest(Path(args.filtered_manifest), out_filtered, ds, args.mode)
    print(f"{ds}: manifest {n_all} clips, filtered {n_kept} clips", flush=True)

    if args.filtered_only_frames:
        tars = [str(Path(args.frames_root) / f"{cid}.tar") for cid in kept_ids]
    else:
        tars = [str(p) for p in sorted(Path(args.frames_root).glob("*.tar"))]
    missing = [t for t in tars if not Path(t).is_file()]
    if missing:
        print(f"FATAL: {len(missing)} frame tars missing, e.g. {missing[:3]}", flush=True)
        return 1

    def run(cmd, **kw):
        print("+", " ".join(map(str, cmd[:6])), f"... ({len(cmd)} args)", flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, **kw)

    # frames (stdin file list keeps the argv small)
    if not args.dry_run:
        list_file = work / "_upload_tars.txt"
        list_file.write_text("\n".join(tars) + "\n")
        subprocess.run(["gsutil", "-m", "-q", "cp", "-I", f"{BUCKET}/egosmith_filtered/{ds}/frames/"],
                       input=list_file.read_text(), text=True, check=True)
    print(f"frames uploaded: {len(tars)}", flush=True)

    if args.mode == "use_gt":
        if not args.outputs_root:
            print("FATAL: use_gt mode needs --outputs_root", flush=True)
            return 1
        src = Path(args.outputs_root)
        if not args.dry_run:
            if args.filtered_only_frames:
                inc = work / "_upload_outputs.txt"
                inc.write_text("\n".join(str(src / cid) for cid in kept_ids) + "\n")
                # rsync each kept clip dir (parallel via gsutil -m across dirs in one process each
                # would be slow; use one rsync of the whole tree when most clips are kept)
                kept_ratio = len(kept_ids) / max(1, len(list(src.iterdir())))
                if kept_ratio > 0.5:
                    run(["gsutil", "-m", "-q", "rsync", "-r", str(src),
                         f"{BUCKET}/egosmith_recon/{ds}/use_gt/outputs/"])
                else:
                    for cid in kept_ids:
                        run(["gsutil", "-m", "-q", "rsync", "-r", str(src / cid),
                             f"{BUCKET}/egosmith_recon/{ds}/use_gt/outputs/{cid}/"])
            else:
                run(["gsutil", "-m", "-q", "rsync", "-r", str(src),
                     f"{BUCKET}/egosmith_recon/{ds}/use_gt/outputs/"])
        print("outputs uploaded", flush=True)

    run(["gsutil", "-q", "cp", str(out_manifest), f"{gcs_run}/clip_manifest.jsonl"])
    run(["gsutil", "-q", "cp", str(out_filtered), f"{gcs_run}/clip_manifest.filtered.jsonl"])
    for f in args.extra_files:
        run(["gsutil", "-q", "cp", f, f"{gcs_run}/"])
    print(f"UPLOAD_DONE {ds}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
