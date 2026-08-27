#!/usr/bin/env python
"""LLM labeling for the WIYH native tier — v4 prompt seeded with WIYH's shipped
subtask annotations.

Same frozen config as run_labeling_gt_datasets.py (gpt-5-mini, effort=medium,
v4 prompt, P2 sampling 1024/high/3fps/12-40 frames). The one addition: each
clip's prompt carries a SEED block built from the sample's shipped
annotation/task_description rows (atomic subtask text + frame-accurate time
ranges intersected with the segment) so segmentation + L1 start from the
dataset's own labels. Seeds may be Chinese (2/3 of rows lack English); the
prompt instructs the model to output English regardless, which makes a
separate zh->en pass redundant (gpt-5-mini reads zh natively).

Reads a LOCAL filtered manifest + local tars (labeling runs pre-upload).
Resume-safe by clip_id; spend-capped via --max_spend.

Usage:
  OPENAI_API_KEY=... python scripts/annotation/run_labeling_wiyh_native.py \
      --manifest /root/w7_native/build/manifest.filtered.jsonl \
      --out /root/w7_native/build/annotations/wiyh_native.annotations.jsonl \
      --workers 64 [--limit 0] [--max_spend 150]
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/egosmith/src")

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(f"{_HERE}/annotation_harness_reference.py").read().split("clips = pick_clips(10)")[0]
ns = {}
exec(compile(_src, "ablation_harness", "exec"), ns)
ns["PROMPT"] = open("/root/egosmith/src/lib/annotation/prompts/with_clip/annotation_general_clip_v4.txt").read()
MODEL, EFFORT, PRICE = ns["MODEL"], ns["EFFORT"], ns["PRICE"]
build_content, client = ns["build_content"], ns["client"]

from lib.pipeline.clips.clip_manifest import ClipManifestRecord  # noqa: E402

PX, DETAIL, TARGET_FPS, FMIN, FMAX = 1024, "high", 3.0, 12, 40
PROMPT_VERSION = "annotation_general_clip_v4+wiyh_seed"

_write_lock = threading.Lock()
_stats_lock = threading.Lock()
_stats = {"done": 0, "err": 0, "cost": 0.0, "t0": time.time()}
_stop = threading.Event()


def seed_block(rec: dict) -> str:
    """Seed text from shipped subtasks overlapping this segment (frame ranges)."""
    ex = rec["descriptor"].get("extra", {})
    subs = ex.get("subtasks") or []
    rng = ex.get("segment_frame_range")
    fps = float(rec["descriptor"].get("fps") or 10.0)
    lines = []
    for st in subs:
        s, e = int(st["s_frame"]), int(st["e_frame"])
        if rng:
            s0, e0 = int(rng[0]), int(rng[1])
            os_, oe = max(s, s0), min(e, e0)
            if oe <= os_:
                continue
            ls, le = (os_ - s0) / fps, (oe - s0) / fps
        else:
            ls, le = s / fps, e / fps
        txt = (st.get("en") or st.get("zh") or "").strip()
        if txt and txt.lower() != "all_success":
            lines.append(f"- [{ls:.1f}s - {le:.1f}s] {txt}")
    if not lines:
        task = ex.get("task", "")
        return (f"\n\n## Dataset-provided task hint (verify against the video)\n{task}\n"
                if task else "")
    return ("\n\n## Dataset-provided subtask annotations (SEED — verify against the video; "
            "they may be imperfect or in Chinese. Use them to seed your segmentation and "
            "level1 verbs, but always describe what you actually see, in English.)\n"
            + "\n".join(lines) + "\n")


def annotate(rec_json, out_f, err_f, max_spend):
    if _stop.is_set():
        return
    rec = json.loads(rec_json)
    clip_id = rec["clip_id"]
    try:
        cm = ClipManifestRecord.from_json(rec_json)
        dur = cm.descriptor.frame_count / float(cm.descriptor.fps or 10)
        nf = max(FMIN, min(FMAX, round(dur * TARGET_FPS)))
        content, dur2, nsent = build_content(cm, nf, PX, DETAIL)
        seed = seed_block(rec)
        if seed:
            content[0]["text"] += seed

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
        row = {"clip_id": clip_id, "dataset": "wiyh_native", "model": MODEL,
               "prompt": PROMPT_VERSION, "effort": EFFORT, "seeded": bool(seed),
               "config": {"px": PX, "detail": DETAIL, "fps": TARGET_FPS, "frames_sent": nsent},
               "duration_sec": round(dur2, 2),
               "annotation": parsed if isinstance(parsed, dict) else {"segments": segs},
               "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens},
               "cost_usd": round(cost, 6)}
        with _write_lock:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
        with _stats_lock:
            _stats["done"] += 1
            _stats["cost"] += cost
            if _stats["cost"] > max_spend:
                _stop.set()
                print(f"SPEND CAP ${max_spend} reached — stopping", flush=True)
            if _stats["done"] % 100 == 0:
                el = time.time() - _stats["t0"]
                print(f'[{time.strftime("%H:%M:%S")}] done={_stats["done"]} err={_stats["err"]} '
                      f'${_stats["cost"]:.2f} rate={_stats["done"]/el*3600:.0f}/h', flush=True)
    except Exception as e:  # noqa: BLE001
        with _write_lock:
            err_f.write(json.dumps({"clip_id": clip_id, "error": str(e)[:300]}) + "\n")
            err_f.flush()
        with _stats_lock:
            _stats["err"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_spend", type=float, default=150.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["clip_id"])
            except Exception:  # noqa: BLE001
                pass
    out_f = open(args.out, "a")
    err_f = open(args.out.replace(".annotations.jsonl", ".errors.jsonl"), "a")
    tasks = []
    for line in open(args.manifest):
        if not line.strip():
            continue
        if json.loads(line)["clip_id"] in done:
            continue
        tasks.append(line)
        if args.limit and len(tasks) >= args.limit:
            break
    print(f"wiyh_native: {len(tasks)} to annotate ({len(done)} done) "
          f"model={MODEL} prompt={PROMPT_VERSION} cap=${args.max_spend}", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda t: annotate(t, out_f, err_f, args.max_spend), tasks))
    el = time.time() - _stats["t0"]
    print(f'WIYH_LABELING_DONE done={_stats["done"]} err={_stats["err"]} '
          f'cost=${_stats["cost"]:.2f} wall={el/3600:.2f}h', flush=True)


if __name__ == "__main__":
    main()
