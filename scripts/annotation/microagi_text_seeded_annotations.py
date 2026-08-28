#!/usr/bin/env python
"""microagi text-seeded v4 annotations — spend-bounded alternative to per-clip VLM labeling.

microagi ships ~640K stage-4-kept 10s sub-clips; full v4 VLM labeling would cost
$1.5-2.5K (over the $300/dataset flag). Every episode carries in-zarr GT activity text
({text,start_idx,end_idx}, extracted into descriptor.extra.annotations at conversion).
This tool:

  1. collects every kept clip's seed = ordered (rel_start, rel_end, text) spans clipped
     to its segment window;
  2. dedupes by the text tuple (~200K unique of 640K) and expands each unique tuple ONCE
     with a TEXT-ONLY gpt-5-mini call into v4-structured language_instructions
     (levels 1-4; level 4 is necessarily generic — no frames are shown — documented);
  3. broadcasts the expansion to every clip carrying that tuple, mapping times into the
     clip window, and writes standard per-shard annotations files
     (filter_run/annotations_v4/_shards/shard_X.annotations.jsonl) with
     "seeded_from": "in_zarr_text" provenance;
  4. (separate QA step) a random VLM-labeled sample compares level1 agreement.

Usage:
  # step 1+2: build seeds + run expansions (resume-safe cache JSONL)
  OPENAI_API_KEY=... python microagi_text_seeded_annotations.py --phase expand \
      --work /root/egosmith_annotations/microagi_seeded [--limit N]
  # step 3: broadcast + upload
  python microagi_text_seeded_annotations.py --phase broadcast --work ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "foundational-research/hoi-dataset/egosmith_filtered/egoverse_microagi"
MODEL = "gpt-5-mini"
PRICE = (0.25, 2.00)  # $/M prompt, completion

EXPAND_PROMPT = """You are generating language annotations for an egocentric bimanual \
manipulation video segment. You are NOT shown the video; instead you get the capture \
system's ground-truth activity label (trusted for WHAT happens).

Ground-truth activity: {seeds}

Produce ONE JSON object "language_instructions" with:
- "level1": verb + object, max 5 words, from the ground-truth text.
- "level2": concise hand-action gist, max 15 words.
- "level3": object-centric description (object, likely parts/state), max 30 words.
- "level4": hand-centric description (both hands' plausible roles/grasps for this task), max 50 words.
Rules: start every instruction with an action verb; no subjects ("I", "the person"); \
use "the" not "a/an"; do not invent brands/materials; where the ground truth does not \
determine a detail, stay generic rather than inventing specifics. \
Return ONLY the JSON object, no fences: \
{{"level1": str, "level2": str, "level3": str, "level4": str}}"""


def seed_key(spans) -> str:
    return hashlib.sha1(json.dumps([s["text"] for s in spans]).encode()).hexdigest()[:16]


def collect(work: Path):
    """Pull all kept rows -> clips.jsonl (clip_id, shard, spans, dur) + uniq.jsonl."""
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    ps = sorted(fs.glob(f"{BASE}/native/_shards/shard_*.full.stage4.kept.jsonl"))
    uniq: dict[str, list] = {}
    n = 0
    with open(work / "clips.jsonl", "w") as out:
        def rd(p):
            return p, fs.cat(p).decode()
        with ThreadPoolExecutor(24) as ex:
            for p, txt in ex.map(rd, ps):
                sfx = p.rsplit("/", 1)[-1].split(".")[0]  # shard_XXXXX
                for l in txt.splitlines():
                    if not l.strip():
                        continue
                    r = json.loads(l)
                    d = r["descriptor"]
                    ex_ = d["extra"]
                    fps = 29.0
                    rng = ex_.get("segment_frame_range") or [0, 10 ** 9]
                    anns = ex_.get("annotations") or (
                        [{"text": ex_.get("task") or "manipulate objects", "start_idx": 0, "end_idx": rng[1]}])
                    spans = []
                    for a in anns:
                        a0, a1 = int(a.get("start_idx", 0)), int(a.get("end_idx", 0))
                        if a1 < rng[0] or a0 > rng[1]:
                            continue
                        spans.append({"text": a.get("text", ""),
                                      "start": round(max(0.0, (a0 - rng[0]) / fps), 2),
                                      "end": round(max(0.1, (min(a1, rng[1]) - rng[0]) / fps), 2)})
                    if not spans:
                        spans = [{"text": ex_.get("task") or "manipulate objects", "start": 0.0,
                                  "end": round(len(d.get("frame_names") or [1]) / fps, 2)}]
                    k = seed_key(spans)
                    uniq.setdefault(k, spans)
                    out.write(json.dumps({"clip_id": r["clip_id"], "shard": sfx, "key": k,
                                          "dur": round(len(d.get("frame_names") or [1]) / fps, 2),
                                          "spans": spans}) + "\n")
                    n += 1
    with open(work / "uniq.jsonl", "w") as f:
        for k, spans in uniq.items():
            f.write(json.dumps({"key": k, "spans": spans}) + "\n")
    print(f"collected {n} clips, {len(uniq)} unique seed tuples", flush=True)


_lk = threading.Lock()
_stats = {"done": 0, "err": 0, "cost": 0.0, "t0": time.time()}


def expand(work: Path, workers: int, limit: int, max_spend: float):
    from openai import OpenAI
    client = OpenAI()
    done = set()
    cache = work / "expansions.jsonl"
    if cache.exists():
        for l in open(cache):
            try:
                done.add(json.loads(l)["key"])
            except Exception:  # noqa: BLE001
                pass
    todo = []
    for l in open(work / "uniq_texts.jsonl"):
        r = json.loads(l)
        if r["key"] not in done:
            todo.append(r)
    if limit:
        todo = todo[:limit]
    print(f"expanding {len(todo)} unique tuples ({len(done)} cached)", flush=True)
    out_f = open(cache, "a")

    def one(r):
        if _stats["cost"] >= max_spend:
            return
        prompt = EXPAND_PROMPT.format(seeds=r["text"])
        last = None
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}],
                    reasoning_effort="minimal")
                txt = (resp.choices[0].message.content or "").strip()
                if txt.startswith("```"):
                    txt = txt.split("```", 2)[1]
                    txt = txt[4:] if txt.startswith("json") else txt
                li = json.loads(txt)
                if isinstance(li, dict) and "language_instructions" in li:
                    li = li["language_instructions"]
                assert isinstance(li, dict) and li.get("level1")
                u = resp.usage
                cost = u.prompt_tokens * PRICE[0] / 1e6 + u.completion_tokens * PRICE[1] / 1e6
                with _lk:
                    out_f.write(json.dumps({"key": r["key"], "language_instructions": li,
                                            "usage": {"prompt_tokens": u.prompt_tokens,
                                                      "completion_tokens": u.completion_tokens},
                                            "cost_usd": round(cost, 6)}) + "\n")
                    _stats["done"] += 1
                    _stats["cost"] += cost
                    if _stats["done"] % 2000 == 0:
                        el = time.time() - _stats["t0"]
                        print(f"[{time.strftime('%H:%M:%S')}] done={_stats['done']} "
                              f"err={_stats['err']} ${_stats['cost']:.0f} "
                              f"rate={_stats['done'] / el * 3600:.0f}/h", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(min(30, 2 ** attempt * 2))
        with _lk:
            _stats["err"] += 1
            if _stats["err"] <= 5:
                print("ERR", r["key"], str(last)[:120], flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    out_f.close()
    print(f"EXPAND DONE done={_stats['done']} err={_stats['err']} cost=${_stats['cost']:.2f}", flush=True)


def broadcast(work: Path):
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    import hashlib as _h
    exp = {}
    for l in open(work / "expansions.jsonl"):
        r = json.loads(l)
        exp[r["key"]] = r
    shards: dict[str, list] = {}
    n_miss = 0
    for l in open(work / "clips.jsonl"):
        c = json.loads(l)
        segs = []
        miss = False
        for s in c["spans"]:
            e = exp.get(_h.sha1(s["text"].encode()).hexdigest()[:16])
            if e is None:
                miss = True
                continue
            segs.append({"start": float(min(max(0.0, s["start"]), c["dur"])),
                         "end": float(min(max(s["start"] + 0.05, s["end"]), c["dur"])),
                         "is_good_quality": True,
                         "language_instructions": e["language_instructions"]})
        if miss and not segs:
            n_miss += 1
            continue
        row = {"clip_id": c["clip_id"], "dataset": "egoverse_microagi", "shard": c["shard"].split("_")[1],
               "model": MODEL, "prompt": "microagi_text_seeded_v4", "effort": "minimal",
               "config": {"mode": "text_seeded", "frames_sent": 0},
               "duration_sec": c["dur"], "annotation": {"segments": segs},
               "seeded_from": "in_zarr_text",
               "usage": e["usage"], "cost_usd": 0.0}
        shards.setdefault(c["shard"], []).append(json.dumps(row))
    print(f"broadcast: {sum(len(v) for v in shards.values())} rows over {len(shards)} shards "
          f"(missing expansion: {n_miss})", flush=True)
    for sfx, rows in sorted(shards.items()):
        with fs.open(f"{BASE}/filter_run/annotations_v4/_shards/{sfx}.annotations.jsonl", "w") as f:
            f.write("\n".join(rows) + "\n")
    print("upload done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["collect", "expand", "broadcast"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--workers", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_spend", type=float, default=290.0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.phase == "collect":
        collect(work)
    elif args.phase == "expand":
        if not (work / "uniq.jsonl").exists():
            collect(work)
        expand(work, args.workers, args.limit, args.max_spend)
    else:
        broadcast(work)


if __name__ == "__main__":
    main()
