#!/usr/bin/env python
"""Production LLM labeling over the four filtered datasets.

Config frozen by the experiments: gpt-5-mini, reasoning_effort=medium, v4 prompt
(levels 1-4, no dense L5), P2 sampling (1024px, detail=high, ~3fps, 12-40 frames).

Resume-safe: appends one JSON line per clip to <ds>.annotations.jsonl; already-done
clip_ids are skipped on restart. Frame tars resolve to local dirs when present,
otherwise pulled from GCS to a scratch cache and deleted after use.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/egosmith/src")

# reuse the exact build_content/client from the validated experiment harness
_src = open(f"{_HERE}/annotation_harness_reference.py").read().split("clips = pick_clips(10)")[0]
ns = {}
exec(compile(_src, "ablation_harness", "exec"), ns)
ns["PROMPT"] = open("/root/egosmith/src/lib/annotation/prompts/with_clip/annotation_general_clip_v4.txt").read()
MODEL, EFFORT, PRICE = ns["MODEL"], ns["EFFORT"], ns["PRICE"]
build_content, client = ns["build_content"], ns["client"]

from lib.pipeline.clips.clip_manifest import ClipManifestRecord  # noqa: E402

GCS = "gs://foundational-research/hoi-dataset/egosmith_filtered"
DATASETS = {
    "taco":           {"local": "/root/taco/frames"},
    "oakink_actions": {"local": "/root/oakink/grasp/frames"},
    "hot3d":          {"local": "/root/hot3d/frames"},
    "egodex":         {"local": "/root/egodex/frames"},
}
OUT_DIR = "/root/egosmith_annotations"
PX, DETAIL, TARGET_FPS, FMIN, FMAX = 1024, "high", 3.0, 12, 40
PROMPT_VERSION = "annotation_general_clip_v4"

_write_lock = threading.Lock()
_stats_lock = threading.Lock()
_stats = {"done": 0, "err": 0, "cost": 0.0, "t0": time.time()}


def resolve_tar(ds, rec):
    base = os.path.basename(rec["descriptor"]["shard_path"])
    local = os.path.join(DATASETS[ds]["local"], base)
    if os.path.exists(local):
        return local, False
    cache = os.path.join(OUT_DIR, "_tarcache", base)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    r = subprocess.run(["gcloud", "storage", "cp", f"{GCS}/{ds}/frames/{base}", cache],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(cache):
        raise RuntimeError(f"tar unavailable: {base}")
    return cache, True


def annotate(ds, rec_json, out_f, err_f):
    rec = json.loads(rec_json)
    clip_id = rec["clip_id"]
    tar, ephemeral = None, False
    try:
        tar, ephemeral = resolve_tar(ds, rec)
        d = rec["descriptor"]
        d["root_dir"] = os.path.dirname(tar)
        d["shard_path"] = tar
        cm = ClipManifestRecord.from_json(json.dumps(rec))
        dur = cm.descriptor.frame_count / float(cm.descriptor.fps or 30)
        nf = max(FMIN, min(FMAX, round(dur * TARGET_FPS)))
        content, dur2, nsent = build_content(cm, nf, PX, DETAIL)

        last_err = None
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
                assert isinstance(segs, list) and segs, "no segments"
                for s in segs:
                    li = s.get("language_instructions") or {}
                    assert li.get("level1"), "missing level1"
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(60, 2 ** attempt * 3))
        else:
            raise RuntimeError(f"5 attempts failed: {last_err}")

        u = resp.usage
        cost = u.prompt_tokens * PRICE[0] / 1e6 + u.completion_tokens * PRICE[1] / 1e6
        row = {"clip_id": clip_id, "dataset": ds, "model": MODEL,
               "prompt": PROMPT_VERSION, "effort": EFFORT,
               "config": {"px": PX, "detail": DETAIL, "fps": TARGET_FPS, "frames_sent": nsent},
               "duration_sec": round(dur2, 2),
               "annotation": parsed if isinstance(parsed, dict) else {"segments": segs},
               "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens},
               "cost_usd": round(cost, 6)}
        with _write_lock:
            out_f.write(json.dumps(row) + "\n"); out_f.flush()
        with _stats_lock:
            _stats["done"] += 1; _stats["cost"] += cost
            if _stats["done"] % 200 == 0:
                el = time.time() - _stats["t0"]
                print(f'[{time.strftime("%H:%M:%S")}] done={_stats["done"]} err={_stats["err"]} '
                      f'${_stats["cost"]:.2f} rate={_stats["done"]/el*3600:.0f}/h', flush=True)
    except Exception as e:  # noqa: BLE001
        with _write_lock:
            err_f.write(json.dumps({"clip_id": clip_id, "dataset": ds, "error": str(e)[:300]}) + "\n")
            err_f.flush()
        with _stats_lock:
            _stats["err"] += 1
    finally:
        if ephemeral and tar and os.path.exists(tar):
            os.unlink(tar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=320)
    ap.add_argument("--datasets", default="taco,oakink_actions,hot3d,egodex")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = []
    for ds in args.datasets.split(","):
        out_path = f"{OUT_DIR}/{ds}.annotations.jsonl"
        done = set()
        if os.path.exists(out_path):
            for l in open(out_path):
                try: done.add(json.loads(l)["clip_id"])
                except Exception: pass
        man = subprocess.run(["gcloud", "storage", "cat",
                              f"{GCS}/{ds}/filter_run/clip_manifest.filtered.jsonl"],
                             capture_output=True, text=True).stdout
        out_f = open(out_path, "a")
        err_f = open(f"{OUT_DIR}/{ds}.errors.jsonl", "a")
        n = 0
        for line in man.splitlines():
            if not line.strip(): continue
            cid = json.loads(line)["clip_id"]
            if cid in done: continue
            tasks.append((ds, line, out_f, err_f)); n += 1
            if args.limit and n >= args.limit: break
        print(f"{ds}: {n} to annotate ({len(done)} already done)", flush=True)

    print(f"TOTAL {len(tasks)} clips, {args.workers} workers, model={MODEL} effort={EFFORT} prompt={PROMPT_VERSION}", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda t: annotate(*t), tasks))
    el = time.time() - _stats["t0"]
    print(f'FINISHED done={_stats["done"]} err={_stats["err"]} cost=${_stats["cost"]:.2f} wall={el/3600:.2f}h', flush=True)


if __name__ == "__main__":
    main()
