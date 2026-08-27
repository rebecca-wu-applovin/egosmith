#!/usr/bin/env python
"""Presence re-filter: purge empty-pose and single-valid-hand clips from shipped shards.

Why: Stage-1 certified TWO visible hands (min_hands=2) for every clip in the recon
datasets in scope, but (a) the presence gate was off when egocentric100k/10k were
filtered, so clips with ZERO valid reconstructed poses leaked through, and (b) no
gate ever tied Stage-1's hand count to reconstruction's delivered hands, so clips
where one of two visible hands failed reconstruction were kept by design. This
script re-reads each kept clip's `pred_valid` (2,T) from its consolidated
`result.npz` on GCS (ranged zip-member read; no full download) and rewrites the
per-shard filtered manifests dropping:

  empty_pose         any-hand valid ratio  < --min_ratio (default 0.5)
  single_valid_hand  either hand's ratio   < --min_ratio

Backup-first: original filtered.jsonl + annotations + report.json are copied to
filter_run/_prepresence2_backup/ before any overwrite (a backup object is never
overwritten). Idempotent via a shard_X.presence2.done marker (holds the stats).
Annotations for removed clips are pruned from the shard annotations file, and the
shard report.json gains a `presence_refilter` key with the removal breakdown.

Self-contained on purpose (stdlib + gcsfs + numpy only) so hub CPU pods can run it
from a single fetched file. Modes:

  # measure only (no writes): 500-clip sampled empty/one-hand rates
  python presence_refilter.py --dataset egocentric10k --sample 500

  # purge a shard range (hub pod: idx=$JOB_COMPLETION_INDEX -> --shards a-b)
  python presence_refilter.py --dataset egocentric100k --shards 0-9 --workers 64

  # aggregate all done markers into filter_run/presence_refilter_reconciliation.json
  python presence_refilter.py --dataset egocentric100k --reconcile
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import gcsfs
import numpy as np

BASE = os.environ.get("PRESENCE2_GCS_BASE", "foundational-research/hoi-dataset")
MARKER_SUFFIX = ".presence2.done"
BACKUP_DIRNAME = "_prepresence2_backup"

fs = gcsfs.GCSFileSystem()


def dataset_prefixes(ds: str) -> dict:
    return {
        "filt": os.environ.get(
            "PRESENCE2_GCS_FILT", f"{BASE}/egosmith_filtered/{ds}/filter_run/_shards"),
        "ann": os.environ.get(
            "PRESENCE2_GCS_ANN", f"{BASE}/egosmith_filtered/{ds}/filter_run/annotations_v4/_shards"),
        "recon": os.environ.get(
            "PRESENCE2_GCS_RECON", f"{BASE}/egosmith_recon/{ds}/recon/outputs"),
        "backup": os.environ.get(
            "PRESENCE2_GCS_BACKUP", f"{BASE}/egosmith_filtered/{ds}/filter_run/{BACKUP_DIRNAME}"),
        "recon_json": os.environ.get(
            "PRESENCE2_GCS_RECONCILE",
            f"{BASE}/egosmith_filtered/{ds}/filter_run/presence_refilter_reconciliation.json"),
    }


def ranged_npz_extract(gcs_path: str, keys: list[str]) -> dict:
    """Extract selected members from a remote .npz (zip) without downloading it all.

    Same mechanics as phase_d_incremental.ranged_npz_extract; inlined so hub pods
    only need this one file.
    """
    info = fs.info(gcs_path)
    size = info["size"]
    tail = fs.read_block(gcs_path, max(0, size - 65536), min(65536, size))
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("no EOCD")
    cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    cd = fs.read_block(gcs_path, cd_off, cd_size)
    out = {}
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos:pos + 4] != b"PK\x01\x02":
            break
        comp = struct.unpack("<H", cd[pos + 10:pos + 12])[0]
        csize = struct.unpack("<I", cd[pos + 20:pos + 24])[0]
        nlen = struct.unpack("<H", cd[pos + 28:pos + 30])[0]
        elen = struct.unpack("<H", cd[pos + 30:pos + 32])[0]
        clen = struct.unpack("<H", cd[pos + 32:pos + 34])[0]
        lho = struct.unpack("<I", cd[pos + 42:pos + 46])[0]
        name = cd[pos + 46:pos + 46 + nlen].decode()
        key = name[:-4] if name.endswith(".npy") else name
        if key in keys:
            lh = fs.read_block(gcs_path, lho, 30)
            lnlen, lelen = struct.unpack("<HH", lh[26:30])
            data = fs.read_block(gcs_path, lho + 30 + lnlen + lelen, csize)
            if comp == 8:
                data = zlib.decompress(data, -15)
            out[key] = np.load(io.BytesIO(data), allow_pickle=False)
        pos += 46 + nlen + elen + clen
    missing = [k for k in keys if k not in out]
    if missing:
        raise ValueError(f"npz members missing: {missing}")
    return out


def read_pred_valid(recon_prefix: str, sfx: str, row: dict) -> np.ndarray:
    """(2,T) bool validity for one manifest row (segment windows sliced)."""
    extra = (row.get("descriptor") or {}).get("extra") or {}
    recon_clip = extra.get("parent_clip_id") or row["clip_id"]
    npz_path = f"{recon_prefix}/shard_{sfx}/{recon_clip}/result.npz"
    last_err = None
    for attempt in range(3):
        try:
            pv = np.asarray(ranged_npz_extract(npz_path, ["pred_valid"])["pred_valid"])
            break
        except Exception as err:  # noqa: BLE001
            last_err = err
            time.sleep(0.5 * (attempt + 1))
    else:
        raise RuntimeError(f"pred_valid read failed: {str(last_err)[:160]}")
    if pv.ndim != 2 or pv.shape[0] != 2:
        raise ValueError(f"unexpected pred_valid shape {pv.shape}")
    window = extra.get("refilter_window")
    if window:
        s, e = int(window[0]), int(window[1])
        pv = pv[:, s:e]
    return pv.astype(np.float32) > 0.5


def classify_row(recon_prefix: str, sfx: str, row: dict, min_ratio: float) -> dict:
    """Return {clip_id, reason(None=keep), ratios, hours, error}. Errors -> keep."""
    d = row.get("descriptor") or {}
    fps = float(d.get("fps") or (d.get("extra") or {}).get("recon_fps") or 15.0)
    n = len(d.get("frame_names") or []) or int(d.get("frame_count") or 0)
    out = {"clip_id": row["clip_id"], "reason": None, "error": None,
           "hours": n / fps / 3600.0 if fps > 0 else 0.0,
           "any_ratio": None, "left_ratio": None, "right_ratio": None}
    try:
        pv = read_pred_valid(recon_prefix, sfx, row)
    except Exception as err:  # noqa: BLE001
        out["error"] = str(err)[:200]
        return out
    frames = pv.shape[1]
    if frames == 0:
        out["any_ratio"] = out["left_ratio"] = out["right_ratio"] = 0.0
    else:
        out["left_ratio"] = float(pv[0].mean())
        out["right_ratio"] = float(pv[1].mean())
        out["any_ratio"] = float((pv[0] | pv[1]).mean())
    if out["any_ratio"] < min_ratio:
        out["reason"] = "empty_pose"
    elif min(out["left_ratio"], out["right_ratio"]) < min_ratio:
        out["reason"] = "single_valid_hand"
    return out


def _copy_backup_once(src: str, dst: str) -> bool:
    """Copy src->dst only if dst does not exist yet. Returns True if copied."""
    if fs.exists(dst):
        return False
    fs.copy(src, dst)
    return True


def process_shard(sfx: str, pre: dict, min_ratio: float, workers: int,
                  dry_run: bool, force: bool, max_error_frac: float) -> dict | None:
    t0 = time.time()
    man_path = f"{pre['filt']}/shard_{sfx}.filtered.jsonl"
    rep_path = f"{pre['filt']}/shard_{sfx}.report.json"
    ann_path = f"{pre['ann']}/shard_{sfx}.annotations.jsonl"
    marker = f"{pre['filt']}/shard_{sfx}{MARKER_SUFFIX}"

    if not force and fs.exists(marker):
        return None
    raw = fs.cat(man_path).decode()
    rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    stats = {
        "shard": sfx, "min_ratio": min_ratio,
        "kept_before": len(rows), "kept_after": len(rows),
        "removed_empty": 0, "removed_single_hand": 0, "errors": 0,
        "removed_hours": 0.0,
        "clip_ids": {"empty_pose": [], "single_valid_hand": []},
        "error_clip_ids": [],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if rows:
        with ThreadPoolExecutor(min(workers, max(1, len(rows)))) as ex:
            verdicts = list(ex.map(
                lambda r: classify_row(pre["recon"], sfx, r, min_ratio), rows))
        n_err = sum(1 for v in verdicts if v["error"])
        if n_err > max_error_frac * len(rows):
            raise RuntimeError(
                f"shard_{sfx}: {n_err}/{len(rows)} pred_valid reads failed; not marking done")
        removed = {v["clip_id"]: v for v in verdicts if v["reason"]}
        stats["errors"] = n_err
        stats["error_clip_ids"] = [v["clip_id"] for v in verdicts if v["error"]]
        for v in removed.values():
            stats["clip_ids"][v["reason"]].append(v["clip_id"])
            stats["removed_hours"] += v["hours"]
        stats["removed_empty"] = len(stats["clip_ids"]["empty_pose"])
        stats["removed_single_hand"] = len(stats["clip_ids"]["single_valid_hand"])
        stats["removed_hours"] = round(stats["removed_hours"], 4)
        kept_rows = [r for r in rows if r["clip_id"] not in removed]
        stats["kept_after"] = len(kept_rows)
        assert stats["kept_after"] + len(removed) == stats["kept_before"], "reconcile failed"

        if removed and not dry_run:
            # backup-first (never overwrite an existing backup)
            _copy_backup_once(man_path, f"{pre['backup']}/shard_{sfx}.filtered.jsonl")
            if fs.exists(rep_path):
                _copy_backup_once(rep_path, f"{pre['backup']}/shard_{sfx}.report.json")
            fs.pipe(man_path, ("".join(json.dumps(r) + "\n" for r in kept_rows)).encode())
            # prune annotations rows for removed clips
            if fs.exists(ann_path):
                _copy_backup_once(ann_path, f"{pre['backup']}/shard_{sfx}.annotations.jsonl")
                ann_keep, ann_removed = [], 0
                for l in fs.cat(ann_path).decode().splitlines():
                    if not l.strip():
                        continue
                    if json.loads(l).get("clip_id") in removed:
                        ann_removed += 1
                    else:
                        ann_keep.append(l)
                fs.pipe(ann_path, ("\n".join(ann_keep) + ("\n" if ann_keep else "")).encode())
                stats["annotations_pruned"] = ann_removed
            else:
                stats["annotations_pruned"] = None
            # append breakdown to the shard report
            if fs.exists(rep_path):
                report = json.loads(fs.cat(rep_path).decode())
                report["presence_refilter"] = stats
                fs.pipe(rep_path, json.dumps(report, ensure_ascii=False, indent=2).encode())
    if not dry_run:
        fs.pipe(marker, json.dumps(stats).encode())
    print(f"[{time.strftime('%H:%M:%S')}] shard_{sfx}: kept {stats['kept_before']} -> "
          f"{stats['kept_after']} (empty={stats['removed_empty']} "
          f"one_hand={stats['removed_single_hand']} err={stats['errors']}) "
          f"{time.time() - t0:.0f}s{' [DRY]' if dry_run else ''}", flush=True)
    return stats


def list_shards(pre: dict) -> list[str]:
    return sorted(os.path.basename(p).split(".")[0].replace("shard_", "")
                  for p in fs.ls(pre["filt"]) if p.endswith(".filtered.jsonl"))


def parse_shards(spec: str) -> list[str]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += [f"{i:05d}" for i in range(int(a), int(b) + 1)]
        else:
            out.append(f"{int(part):05d}")
    return out


def run_sample(pre: dict, n: int, min_ratio: float, workers: int, seed: int) -> None:
    rng = random.Random(seed)
    shards = list_shards(pre)
    rng.shuffle(shards)
    rows = []
    per_shard_cap = max(50, n // 10)  # spread the sample across shards
    for sfx in shards[:max(50, n // 40 + 1)]:
        raw = fs.cat(f"{pre['filt']}/shard_{sfx}.filtered.jsonl").decode()
        shard_rows = [(sfx, json.loads(l)) for l in raw.splitlines() if l.strip()]
        if len(shard_rows) > per_shard_cap:
            shard_rows = rng.sample(shard_rows, per_shard_cap)
        rows += shard_rows
        if len(rows) >= 20 * n:
            break
    sample = rng.sample(rows, min(n, len(rows)))
    with ThreadPoolExecutor(workers) as ex:
        verdicts = list(ex.map(
            lambda t: classify_row(pre["recon"], t[0], t[1], min_ratio), sample))
    ok = [v for v in verdicts if not v["error"]]
    n_empty = sum(1 for v in ok if v["reason"] == "empty_pose")
    n_one = sum(1 for v in ok if v["reason"] == "single_valid_hand")
    print(json.dumps({
        "sampled": len(sample), "measured": len(ok),
        "errors": len(verdicts) - len(ok),
        "empty_pose": n_empty, "empty_pose_rate": round(n_empty / max(1, len(ok)), 4),
        "single_valid_hand": n_one, "single_valid_hand_rate": round(n_one / max(1, len(ok)), 4),
    }, indent=2))


def run_reconcile(pre: dict) -> None:
    markers = sorted(p for p in fs.ls(pre["filt"]) if p.endswith(MARKER_SUFFIX))
    per_shard, tot = [], {"kept_before": 0, "kept_after": 0, "removed_empty": 0,
                          "removed_single_hand": 0, "errors": 0, "removed_hours": 0.0}
    for p in markers:
        s = json.loads(fs.cat(p).decode())
        per_shard.append(s)
        for k in tot:
            tot[k] += s.get(k) or 0
    tot["removed_hours"] = round(tot["removed_hours"], 3)
    tot["shards_done"] = len(markers)
    tot["reconciled"] = (tot["kept_after"] + tot["removed_empty"]
                         + tot["removed_single_hand"] == tot["kept_before"])
    payload = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "totals": tot, "shards": per_shard}
    fs.pipe(pre["recon_json"], json.dumps(payload, ensure_ascii=False, indent=2).encode())
    print(json.dumps(tot, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="dataset name under egosmith_filtered/")
    ap.add_argument("--shards", default="", help="e.g. 0-9,100,200-210; empty = all shards found")
    ap.add_argument("--workers", type=int, default=64, help="ranged-read threads per shard")
    ap.add_argument("--shard_parallel", type=int, default=1, help="shards processed concurrently")
    ap.add_argument("--min_ratio", type=float, default=0.5)
    ap.add_argument("--max_error_frac", type=float, default=0.05,
                    help="fail the shard (no done marker) above this read-error fraction")
    ap.add_argument("--dry_run", action="store_true", help="classify + print, write nothing")
    ap.add_argument("--force", action="store_true", help="reprocess shards with done markers")
    ap.add_argument("--sample", type=int, default=0, help="measure-only mode: N random kept clips")
    ap.add_argument("--reconcile", action="store_true",
                    help="aggregate done markers into presence_refilter_reconciliation.json")
    args = ap.parse_args()

    pre = dataset_prefixes(args.dataset)
    if args.sample:
        run_sample(pre, args.sample, args.min_ratio, args.workers, seed=7)
        return
    if args.reconcile:
        run_reconcile(pre)
        return

    shards = parse_shards(args.shards) if args.shards else list_shards(pre)
    existing = set(list_shards(pre))
    shards = [s for s in shards if s in existing]
    print(f"{args.dataset}: {len(shards)} shard(s) to process", flush=True)

    failed = []

    def safe(sfx: str):
        try:
            process_shard(sfx, pre, args.min_ratio, args.workers,
                          args.dry_run, args.force, args.max_error_frac)
        except Exception as err:  # noqa: BLE001
            failed.append(sfx)
            print(f"shard_{sfx} FAILED: {str(err)[:250]}", flush=True)

    if args.shard_parallel > 1:
        with ThreadPoolExecutor(args.shard_parallel) as ex:
            list(ex.map(safe, shards))
    else:
        for sfx in shards:
            safe(sfx)
    if failed:
        raise SystemExit(f"{len(failed)} shard(s) failed: {','.join(failed[:20])}")


if __name__ == "__main__":
    main()
