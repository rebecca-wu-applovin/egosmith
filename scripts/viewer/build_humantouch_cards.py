#!/usr/bin/env python
"""Viewer cards for HumanTouch (MANUS glove GT + block-anchor extrinsics).

Samples 50 kept clips stratified across tasks X001-X010 (anchor-diverse within
each task) plus ~12 dropped clips spread over the drop reasons of BOTH
reconciliations: the 2026-08-28 remediation (far anchor assignment / block-QA
visual off; tars under _dropped_20260828/frames/) and the 2026-08-29 full-tier
screen (full_screen_visual_off; tars under _dropped_fullscreen_20260829/frames/).
Pulls each clip's frames tar from the sharded GCS layout, renders a MANO
overlay card from the local seq_folder, and writes
/root/viewer_work/humantouch/cards.json + clips/ for build_viewer --publish.
"""
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "inspection"))

from render_clip_card import render_recon_overlay, render_plain_tar  # noqa: E402

MANIFEST = "/root/w7_full/humantouch/final/manifest_filtered.jsonl"
GCS = "gs://foundational-research/hoi-dataset/egosmith_filtered/humantouch/frames"
# (reconciliation json, GCS base its relocated tars live under)
RECONCILIATIONS = [
    ("/root/w7_full/humantouch/remediation_20260828/remediation_20260828.json",
     "gs://foundational-research/hoi-dataset/egosmith_filtered/humantouch/"
     "_dropped_20260828/frames"),
    ("/root/w7_full/humantouch/fullscreen_drop_20260829/fullscreen_drop_20260829.json",
     "gs://foundational-research/hoi-dataset/egosmith_filtered/humantouch/"
     "_dropped_fullscreen_20260829/frames"),
    ("/root/w7_full/humantouch/fullscreen_drop_20260829/fullscreen_ext_20260829.json",
     "gs://foundational-research/hoi-dataset/egosmith_filtered/humantouch/"
     "_dropped_fullscreen_20260829/frames"),
]
WORK = Path("/root/viewer_work/humantouch")
N_PER_TASK = 5
N_DROPPED = 12


def sample_clips():
    by_task = defaultdict(list)
    for line in open(MANIFEST):
        rec = json.loads(line)
        by_task[rec["group_id"]].append(rec)
    rng = random.Random(7)
    picked = []
    for task in sorted(by_task):
        pool = by_task[task]
        rng.shuffle(pool)
        seen_blocks, first, rest = set(), [], []
        for rec in pool:
            blk = rec["descriptor"]["extra"].get("mount_block")
            (rest if blk in seen_blocks else first).append(rec)
            seen_blocks.add(blk)
        picked += (first + rest)[:N_PER_TASK]
    return picked


def sample_dropped():
    """~N_DROPPED dropped clips, spread over the drop reasons of every
    reconciliation; each pick carries the GCS base its tar was relocated to."""
    rng = random.Random(11)
    by_reason = defaultdict(list)
    n_files = 0
    for path, gcs_base in RECONCILIATIONS:
        try:
            dropped = json.load(open(path))["dropped_clips"]
        except FileNotFoundError:
            continue
        n_files += 1
        for r in dropped:
            by_reason[r["reason"]].append(dict(r, gcs_dropped=gcs_base))
    if not by_reason:
        return []
    picked = []
    per = max(1, N_DROPPED // len(by_reason))
    for reason in sorted(by_reason):
        picked += rng.sample(by_reason[reason], min(per, len(by_reason[reason])))
    return picked[:N_DROPPED]


def fetch_tar(rec, cache_dir, gcs_base=None):
    d = rec["descriptor"]
    base = os.path.basename(d["shard_path"])
    m = re.search(r"convert/(\d+)/frames", d["root_dir"])
    remote = f"{gcs_base or GCS}/shard_{m.group(1)}/{base}"
    local = os.path.join(cache_dir, base)
    r = subprocess.run(["gcloud", "storage", "cp", "-q", remote, local],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(local):
        raise RuntimeError(f"tar fetch failed: {remote}")
    return local


# dropped clips are absent from the shipped manifest; their records live in the
# pre-change backups (oldest covers everything dropped since)
PRE_MANIFESTS = [MANIFEST + ".preremediation_20260828.bak",
                 MANIFEST + ".prefullscreen_20260829.bak"]


def dropped_records(picked):
    """Manifest records for dropped clips (absent from the shipped manifest)."""
    want = {p["clip_id"]: p for p in picked}
    out, seen = [], set()
    for pre in PRE_MANIFESTS:
        if not os.path.exists(pre) or not (set(want) - seen):
            continue
        for line in open(pre):
            rec = json.loads(line)
            cid = rec["clip_id"]
            if cid in want and cid not in seen:
                seen.add(cid)
                out.append((rec, want[cid]["reason"], want[cid]["gcs_dropped"]))
    return out


def main():
    (WORK / "clips").mkdir(parents=True, exist_ok=True)
    cards, n_overlay, n_fallback = [], 0, 0
    todo = [(rec, "kept", None, None) for rec in sample_clips()]
    todo += [(rec, "dropped", reason, gcs_base)
             for rec, reason, gcs_base in dropped_records(sample_dropped())]
    with tempfile.TemporaryDirectory() as cache:
        for rec, section, reason, gcs_base in todo:
            cid = rec["clip_id"]
            d = rec["descriptor"]
            extra = d["extra"]
            out_mp4 = WORK / "clips" / f"{cid}.mp4"
            tar = None
            try:
                tar = fetch_tar(rec, cache, gcs_base=gcs_base)
                try:
                    meta = render_recon_overlay(tar, d["seq_folder"], str(out_mp4),
                                                fps=d["fps"] or 15.0)
                except Exception as e:  # noqa: BLE001
                    meta = render_plain_tar(tar, str(out_mp4), fps=d["fps"] or 15.0)
                    meta["note"] = f"overlay failed: {type(e).__name__}: {str(e)[:100]}"
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {cid}: {e}", flush=True)
                continue
            finally:
                if tar and os.path.exists(tar):
                    os.unlink(tar)
            status = meta.pop("status")
            note = meta.pop("note")
            if status == "overlay":
                n_overlay += 1
            else:
                n_fallback += 1
            cards.append(dict(
                clip_id=cid, video=f"clips/{cid}.mp4", status=status, note=note,
                meta=meta, annotation=None, section=section,
                reasons=[reason] if reason else [],
                extra_rows=[["anchor block", extra.get("mount_block")],
                            ["anchor fit (px)", round(extra.get("block_fit_median_px", 0), 1)],
                            ["gt mode", extra.get("gt_mode")],
                            ["source", extra.get("source_uri", "").split("hoi-dataset/")[-1]]]))
            print(f"{cid} {status} {section}{' (' + note + ')' if note else ''}", flush=True)
    (WORK / "cards.json").write_text(json.dumps(cards, indent=1))
    print(f"DONE cards={len(cards)} overlay={n_overlay} fallback={n_fallback}")


if __name__ == "__main__":
    main()
