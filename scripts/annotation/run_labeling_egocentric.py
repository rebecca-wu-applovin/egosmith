#!/usr/bin/env python
"""Rolling Egocentric-100K labeling: chases Phase D's per-shard kept-manifests.

Loop: list filter_run/_shards/*.filtered.jsonl -> for each shard without an uploaded
annotations file, annotate its kept clips (frames pulled per-clip from GCS and deleted
after use), append to a durable per-shard local JSONL (intra-shard resume), and upload
the shard's annotations when complete.

Config identical to the validated four-dataset run: gpt-5-mini, effort=medium,
v4 prompt (L1-L4), P2 sampling (1024px, detail high, ~3fps) with FMIN=8 for the
short sub-clips (pilot-validated: $0.0026/clip median).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/egosmith/src")
_HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
_src = open(f"{_HERE}/annotation_harness_reference.py").read().split("clips = pick_clips(10)")[0]
ns = {}
exec(compile(_src, "ablation_harness", "exec"), ns)
ns["PROMPT"] = open("/root/egosmith/src/lib/annotation/prompts/with_clip/annotation_general_clip_v4.txt").read()
MODEL, EFFORT, PRICE = ns["MODEL"], ns["EFFORT"], ns["PRICE"]
build_content, client = ns["build_content"], ns["client"]
from lib.pipeline.clips.clip_manifest import ClipManifestRecord  # noqa: E402
import gcsfs
_fs = gcsfs.GCSFileSystem()

GCS_FILT = "gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/filter_run/_shards"
GCS_FR = "gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/frames"
GCS_ANN = "gs://foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/filter_run/annotations_v4/_shards"
OUT = "/root/egosmith_annotations/ego"
PX, DETAIL, TARGET_FPS, FMIN, FMAX = 1024, "high", 3.0, 8, 40
PROMPT_VERSION = "annotation_general_clip_v4"
MAX_SPEND = float(os.environ.get("EGO_LABEL_MAX_SPEND", "45000"))

_locks = {}
_lk = threading.Lock()
_stats = {"done": 0, "err": 0, "cost": 0.0, "t0": time.time()}


def shard_lock(sfx):
    with _lk:
        return _locks.setdefault(sfx, threading.Lock())


def annotate_clip(sfx, rec_json):
    rec = json.loads(rec_json)
    clip_id = rec["clip_id"]
    d = rec["descriptor"]
    base = os.path.basename(d["shard_path"])
    tar = f"{OUT}/_tarcache/{sfx}__{base}"
    os.makedirs(os.path.dirname(tar), exist_ok=True)
    try:
        # in-process download (a gcloud subprocess per clip costs ~1s CPU just booting)
        _fs.get(f"{GCS_FR}/shard_{sfx}/{base}".replace("gs://", ""), tar)
        d["root_dir"] = os.path.dirname(tar); d["shard_path"] = tar
        cm = ClipManifestRecord.from_json(json.dumps(rec))
        dur = cm.descriptor.frame_count / float(cm.descriptor.fps or 15)
        nf = max(FMIN, min(FMAX, round(dur * TARGET_FPS)))
        content, dur2, nsent = build_content(cm, nf, PX, DETAIL)
        last = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": content}],
                    reasoning_effort=EFFORT)
                txt = (resp.choices[0].message.content or "").strip()
                if txt.startswith("```"):
                    txt = txt.split("```", 2)[1]
                    txt = txt[4:] if txt.startswith("json") else txt
                parsed = json.loads(txt)
                segs = parsed.get("segments", parsed) if isinstance(parsed, dict) else parsed
                assert isinstance(segs, list) and segs and (segs[0].get("language_instructions") or {}).get("level1")
                break
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(60, 2 ** attempt * 3))
        else:
            raise RuntimeError(f"5 attempts: {last}")
        u = resp.usage
        cost = u.prompt_tokens * PRICE[0] / 1e6 + u.completion_tokens * PRICE[1] / 1e6
        row = {"clip_id": clip_id, "dataset": "egocentric100k", "shard": sfx,
               "model": MODEL, "prompt": PROMPT_VERSION, "effort": EFFORT,
               "config": {"px": PX, "detail": DETAIL, "fps": TARGET_FPS, "frames_sent": nsent},
               "duration_sec": round(dur2, 2),
               "annotation": parsed if isinstance(parsed, dict) else {"segments": segs},
               "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens},
               "cost_usd": round(cost, 6)}
        with shard_lock(sfx):
            with open(f"{OUT}/shard_{sfx}.annotations.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        with _lk:
            _stats["done"] += 1; _stats["cost"] += cost
            if _stats["done"] % 1000 == 0:
                el = time.time() - _stats["t0"]
                print(f'[{time.strftime("%H:%M:%S")}] done={_stats["done"]} err={_stats["err"]} '
                      f'${_stats["cost"]:.0f} rate={_stats["done"]/el*3600:.0f}/h', flush=True)
    except Exception as e:  # noqa: BLE001
        with shard_lock(sfx):
            with open(f"{OUT}/shard_{sfx}.errors.jsonl", "a") as f:
                f.write(json.dumps({"clip_id": clip_id, "error": str(e)[:200]}) + "\n")
        with _lk:
            _stats["err"] += 1
    finally:
        if os.path.exists(tar):
            os.unlink(tar)


def finalize_shard(sfx, kept_n):
    """Upload the shard's annotations when all kept clips are accounted for."""
    ann = f"{OUT}/shard_{sfx}.annotations.jsonl"
    errs = f"{OUT}/shard_{sfx}.errors.jsonl"
    n_ok = sum(1 for _ in open(ann)) if os.path.exists(ann) else 0
    n_err = len({json.loads(l)["clip_id"] for l in open(errs)}) if os.path.exists(errs) else 0
    if n_ok + n_err < kept_n:
        return False
    subprocess.run(["gcloud", "storage", "cp", ann, f"{GCS_ANN}/shard_{sfx}.annotations.jsonl"],
                   capture_output=True)
    if os.path.exists(errs):
        subprocess.run(["gcloud", "storage", "cp", errs, f"{GCS_ANN}/shard_{sfx}.errors.jsonl"],
                       capture_output=True)
    print(f"shard_{sfx} FINALIZED: {n_ok} annotated, {n_err} failed", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=150, help="threads in THIS process")
    ap.add_argument("--stride", type=int, default=1, help="total labeler processes")
    ap.add_argument("--stride_idx", type=int, default=0, help="this process's index")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--limit_shards", type=int, default=0)
    ap.add_argument("--limit_clips", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(f"{OUT}/_tarcache", exist_ok=True)
    pool = ThreadPoolExecutor(args.workers)
    tag = f"[p{args.stride_idx}]"
    finalized = set()
    for pth in subprocess.run(["gcloud", "storage", "ls", f"{GCS_ANN}/"],
                              capture_output=True, text=True).stdout.splitlines():
        if pth.endswith(".annotations.jsonl"):
            finalized.add(os.path.basename(pth).split(".")[0].replace("shard_", ""))
    print(f"{tag} already finalized on GCS: {len(finalized)}", flush=True)
    while True:
        if _stats["cost"] >= MAX_SPEND / max(1, args.stride):
            print(f"{tag} per-process spend cap reached — stopping", flush=True)
            break
        ls = subprocess.run(["gcloud", "storage", "ls", f"{GCS_FILT}/"],
                            capture_output=True, text=True).stdout.splitlines()
        shards = sorted(os.path.basename(pth).split(".")[0].replace("shard_", "")
                        for pth in ls if pth.endswith(".filtered.jsonl"))
        pend = [s for s in shards
                if s not in finalized and int(s) % args.stride == args.stride_idx]
        if args.limit_shards:
            pend = pend[:args.limit_shards]
        print(f"{tag}[scan] filtered={len(shards)} mine-pending={len(pend)}", flush=True)
        # per-shard sequential: annotate -> finalize/upload -> next (streaming uploads)
        for sfx in pend:
            man = subprocess.run(["gcloud", "storage", "cat", f"{GCS_FILT}/shard_{sfx}.filtered.jsonl"],
                                 capture_output=True, text=True).stdout
            rows = [l for l in man.splitlines() if l.strip()]
            done_ids = set()
            for pth in (f"{OUT}/shard_{sfx}.annotations.jsonl", f"{OUT}/shard_{sfx}.errors.jsonl"):
                if os.path.exists(pth):
                    for l in open(pth):
                        try: done_ids.add(json.loads(l)["clip_id"])
                        except Exception: pass
            todo = [l for l in rows if json.loads(l)["clip_id"] not in done_ids]
            if args.limit_clips:
                todo = todo[:args.limit_clips]
            if todo:
                list(pool.map(lambda l: annotate_clip(sfx, l), todo))
            if finalize_shard(sfx, len(rows)):
                finalized.add(sfx)
        if args.once:
            break
        time.sleep(300)
    el = time.time() - _stats["t0"]
    print(f'{tag} LOOP_END done={_stats["done"]} err={_stats["err"]} '
          f'cost=${_stats["cost"]:.0f} wall={el/3600:.1f}h', flush=True)


if __name__ == "__main__":
    main()
