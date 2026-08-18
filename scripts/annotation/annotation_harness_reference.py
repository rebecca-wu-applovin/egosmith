#!/usr/bin/env python
"""Ablation: does denser frame sampling and/or higher image detail fix HOI captioning?

10 TACO clips (one per distinct GT action verb) x 4 arms, gpt-5-mini + with_clip segmentation prompt.
Scored automatically against TACO's ground-truth (action, tool, object) triplet.
"""
import os, sys, json, base64, time, re, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/root/egosmith/src")
import cv2
from openai import OpenAI
from lib.pipeline.clips.clip_manifest import ClipManifestRecord
from lib.pipeline.io.frame_sources import build_frame_source_from_descriptor

W = os.environ.get("ANNOT_WORK_DIR", "/tmp/annot_harness"); os.makedirs(f"{W}/frames", exist_ok=True)
GCS = "gs://foundational-research/hoi-dataset/egosmith_filtered/taco/filter_run/clip_manifest.filtered.jsonl"
client = OpenAI()  # auth via OPENAI_API_KEY env var
PROMPT = open("/root/egosmith/src/lib/annotation/prompts/with_clip/annotation_general_clip.txt").read()
MODEL, EFFORT, PRICE = "gpt-5-mini", "medium", (0.125, 1.00)

ARMS = [  # (name, n_frames, px, detail)
    ("A0 baseline",     12, 512,  "low"),
    ("A1 denser",       24, 512,  "low"),
    ("A2 high-detail",  12, 1024, "high"),
    ("A3 both",         24, 1024, "high"),
]

# GT verb -> acceptable surface forms in a caption
SYN = {
    "brush": ["brush", "scrub", "sweep"], "dust": ["dust", "brush", "sweep", "wipe"],
    "cut": ["cut", "slice", "chop"], "scrape off": ["scrape", "scrub", "scoop"],
    "skim off": ["skim", "scoop", "ladle"], "pour in some": ["pour", "tip"],
    "put in": ["put in", "insert", "place", "drop"], "put out": ["take out", "remove", "lift out", "put out"],
    "stir": ["stir", "mix", "swirl"], "smear": ["smear", "spread"],
    "measure": ["measure"], "screw": ["screw", "tighten"], "empty": ["empty", "dump", "pour out"],
    "roll": ["roll"], "clamp": ["clamp", "grip"], "shake": ["shake"],
}


def pick_clips(n=10):
    """One clip per distinct GT action verb (triplet[0])."""
    raw = subprocess.run(["gcloud", "storage", "cat", GCS], capture_output=True, text=True).stdout
    seen, out = set(), []
    for line in raw.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tri = r["descriptor"].get("extra", {}).get("triplet")
        if not tri:
            continue
        act = tri.strip("()").split(",")[0].strip()
        if act in seen:
            continue
        seen.add(act)
        parts = [p.strip() for p in tri.strip("()").split(",")]
        r["_gt"] = {"action": parts[0], "tool": parts[1] if len(parts) > 1 else "",
                    "object": parts[2] if len(parts) > 2 else ""}
        out.append(r)
        if len(out) >= n:
            break
    # localize tars
    for r in out:
        d = r["descriptor"]; loc = f"{W}/frames/{os.path.basename(d['shard_path'])}"
        if not os.path.exists(loc):
            subprocess.run(["gcloud", "storage", "cp", d["shard_path"], loc], capture_output=True)
        d["root_dir"] = f"{W}/frames"; d["shard_path"] = loc
    return out


def build_content(rec, nframes, px, detail):
    fs = build_frame_source_from_descriptor(rec.descriptor)
    total = len(fs); fps = float(rec.descriptor.fps or 30); dur = total / fps
    idx = sorted({round(i * (total - 1) / (nframes - 1)) for i in range(nframes)}) if total > nframes else list(range(total))
    content = [{"type": "text", "text": PROMPT +
                f"\n\n## This video\nDuration: {dur:.2f} seconds. {len(idx)} frames sampled uniformly; each "
                f"frame is preceded by its timestamp. Use them to set accurate start/end within [0, {dur:.2f}]."}]
    for i in idx:
        fr = fs.get_frame(i, rgb=False)
        h, w = fr.shape[:2]; s = px / max(h, w)
        if s < 1.0:
            fr = cv2.resize(fr, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        content.append({"type": "text", "text": f"t = {i/fps:.2f}s"})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                                      "detail": detail}})
    return content, dur, len(idx)


def score(segs, gt):
    """Did the captions name the GT action / tool / object?"""
    txt = " ".join(f'{s.get("language_instructions",{}).get(f"level{i}","")} ' for s in segs for i in range(1, 6)).lower()
    l1 = " ".join(s.get("language_instructions", {}).get("level1", "") for s in segs).lower()
    def hit(word, hay):
        forms = SYN.get(word, [word])
        return any(re.search(r"\b" + re.escape(f.split()[0])[:6], hay) for f in forms)
    return {
        "action_hit": hit(gt["action"], txt),
        "action_in_L1": hit(gt["action"], l1),      # stricter: is it the headline verb?
        "tool_hit": bool(gt["tool"]) and gt["tool"].lower()[:5] in txt,
        "object_hit": bool(gt["object"]) and gt["object"].lower()[:5] in txt,
    }


def run(rec_json, gt, arm):
    name, nf, px, det = arm
    rec = ClipManifestRecord.from_json(json.dumps(rec_json))
    content, dur, nsent = build_content(rec, nf, px, det)
    try:
        t0 = time.time()
        resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": content}],
                                              reasoning_effort=EFFORT)
        dt = time.time() - t0
        txt = (resp.choices[0].message.content or "").strip()
        if txt.startswith("```"):
            txt = txt.split("```", 2)[1]
            txt = txt[4:] if txt.startswith("json") else txt
        segs = json.loads(txt)
        if isinstance(segs, dict):
            segs = segs.get("segments", [])
        u = resp.usage
        cost = u.prompt_tokens * PRICE[0] / 1e6 + u.completion_tokens * PRICE[1] / 1e6
        return {"arm": name, "segments": segs, "prompt_tok": u.prompt_tokens, "frames_sent": nsent,
                "completion_tok": u.completion_tokens, "cost": cost, "sec": round(dt, 1), **score(segs, gt)}
    except Exception as e:
        return {"arm": name, "error": str(e)[:180], "frames_sent": nsent}


clips = pick_clips(10)
print("clips (one per GT action verb):")
for c in clips:
    print(f'  {c["clip_id"][:46]:46} GT={c["_gt"]}')
print(f"\n{len(clips)} clips x {len(ARMS)} arms = {len(clips)*len(ARMS)} calls\n", flush=True)

results = []
for c in clips:
    gt = c["_gt"]
    with ThreadPoolExecutor(max_workers=4) as ex:
        runs = list(ex.map(lambda a: run(c, gt, a), ARMS))
    row = {"clip_id": c["clip_id"], "gt": gt, "runs": {r["arm"]: r for r in runs}}
    results.append(row)
    json.dump(results, open(f"{W}/ablation_results.json", "w"))
    for r in runs:
        if "error" in r:
            print(f'  {c["clip_id"][:30]:30} {r["arm"]:14} ERROR {r["error"][:70]}', flush=True); continue
        l1s = " | ".join(s.get("language_instructions", {}).get("level1", "?") for s in r["segments"])
        flags = f'act={"Y" if r["action_hit"] else "n"}({"L1" if r["action_in_L1"] else "--"}) tool={"Y" if r["tool_hit"] else "n"} obj={"Y" if r["object_hit"] else "n"}'
        print(f'  {c["clip_id"][:30]:30} {r["arm"]:14} {len(r["segments"])}seg ${r["cost"]:.4f} {flags} :: {l1s[:70]}', flush=True)
    print("", flush=True)

print("=== ABLATION SUMMARY (n=%d clips) ===" % len(results))
print(f'{"arm":15} {"action":>7} {"act@L1":>7} {"tool":>6} {"object":>7} {"segs":>5} {"$/clip":>8} {"in→out tok":>13}')
for name, *_ in ARMS:
    rs = [r["runs"][name] for r in results if "cost" in r["runs"].get(name, {})]
    if not rs:
        print(f"  {name}: no successful runs"); continue
    n = len(rs); pc = lambda k: 100 * sum(1 for x in rs if x.get(k)) / n
    print(f'{name:15} {pc("action_hit"):6.0f}% {pc("action_in_L1"):6.0f}% {pc("tool_hit"):5.0f}% '
          f'{pc("object_hit"):6.0f}% {sum(len(x["segments"]) for x in rs)/n:5.1f} '
          f'${sum(x["cost"] for x in rs)/n:7.4f} {sum(x["prompt_tok"] for x in rs)//n:6d}→{sum(x["completion_tok"] for x in rs)//n:<6d}')
print("DONE", flush=True)
