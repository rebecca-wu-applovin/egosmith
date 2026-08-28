#!/usr/bin/env python
"""Prompt viewer: v4 vs v5 labeling-prompt A/B, side by side on the same clips.

Compares the shipped v4 annotation prompt (annotation_general_clip_v4) against the
final v5 prompt on the 40-clip eval set built for the v5 prompt-engineering pass
(/root/egosmith_annotations/v5_prompt_eng): 10 clips each for dexcap / egodex /
taco / epic_kitchens_100, stratified over <4s videos, same-object over-splitting,
and long-clip segmentation.

A/B pair (fresh same-day runs, 2026-08-28, gpt-5-mini effort=medium, production
P2 sampling): outputs/v4_baseline.jsonl vs outputs/v5d_full.jsonl — the v5d prompt
is byte-identical to the shipped src/lib/annotation/prompts/with_clip/
annotation_general_clip_v5.txt. Uses ONLY already-recorded LLM outputs (global
labeling hold: no new labeling happens here).

Per clip: one card with TWO columns (left v4, right v5). Each column lists that
prompt's segments in order — the segment's own video cut (correct native fps,
360p/crf28, 288p/crf30 on clips >60s) with L1-L4 beneath, plus compliance badges
derived with the SAME logic as score_annotations.py:
  red    "duration X.Xs out of 3-10s"   (except sanctioned >10s same-object
                                         merges and the whole-video <4s segment)
  orange "same-object split"            (consecutive same-object segments, per
                                         the scorer's pair_violates)
  green  "whole-video segment"          (<4s video covered by ONE segment)
  blue   "merge-wins: >10s continuous same-object" (sanctioned merge)

Publishes (Cache-Control: no-store, self-contained data-URI videos) to
  gs://foundational-research/hoi-dataset/egosmith_filtered/viewer/prompt_viewer/
Pages over ~140 MB are split (<ds>_p1.html, <ds>_p2.html, ...).

Phases: --render (tars -> per-clip mp4s), --publish (cut segments -> HTML -> GCS),
--selfcheck (download published pages, verify every segment video's duration
against its declared start/end, badge counts vs the scoring JSONs, both columns
present for all 40 clips).
"""
import argparse
import base64
import html
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
EVAL_DIR = Path("/root/egosmith_annotations/v5_prompt_eng")
for p in (str(_HERE), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_viewer as bv  # noqa: E402  (CSS, esc, VIEWER, ADAPTERS fps, upload style)
import score_annotations as sa  # noqa: E402  (VERBS/STOP_TAIL/head_object/count_l1_verbs)

V4_TAG, V5_TAG = "v4_baseline", "v5d_full"
DATASETS = ("dexcap", "egodex", "taco", "epic_kitchens_100")
PV = f"{bv.VIEWER}/prompt_viewer"
AUTH = f"https://storage.cloud.google.com/{PV}"
MAX_PAGE_BYTES = 140_000_000
LONG_CLIP_SEC = 60.0          # clips longer than this embed at 288p/crf30
DUR_TOL = 0.25                # selfcheck tolerance (s) on cut duration

# ------------------------------------------------------------- inputs
def load_inputs():
    eval_set = json.load(open(EVAL_DIR / "eval_set.json"))
    runs = {}
    for tag in (V4_TAG, V5_TAG):
        runs[tag] = {json.loads(l)["clip_id"]: json.loads(l)
                     for l in open(EVAL_DIR / "outputs" / f"{tag}.jsonl") if l.strip()}
    scores = {tag: json.load(open(EVAL_DIR / "outputs" / f"{tag}.scores.json"))
              for tag in (V4_TAG, V5_TAG)}
    manifests = {}
    for ds in DATASETS:
        manifests[ds] = {json.loads(l)["clip_id"]: json.loads(l)
                         for l in open(EVAL_DIR / "data" / f"{ds}.manifest.jsonl")
                         if l.strip()}
    return eval_set, runs, scores, manifests


# ------------------------------------------------- badge logic (mirrors score_clip)
TRANSPORT = {"pick", "grab", "grasp", "lift", "place", "put", "release", "move",
             "set", "take", "lower", "raise", "hold", "position", "drop"}


def analyze_segments(row):
    """Per-segment compliance flags with EXACTLY the score_annotations.score_clip
    logic (R1 whole-video, R2 pair_violates, R3 3-10s + merge-wins exemption)."""
    dur = row["duration_sec"]
    segs = row["annotation"]["segments"]
    n = len(segs)
    objs, verbs, verbs1, sd = [], [], [], []
    for s in segs:
        l1 = (s.get("language_instructions") or {}).get("level1", "")
        objs.append(sa.head_object(l1))
        w = re.sub(r"[^a-z ]", "", l1.lower()).split()
        verbs.append(w[0] if w else "")
        verbs1.append(sa.count_l1_verbs(l1) == 1)
        sd.append(float(s["end"]) - float(s["start"]))

    def pair_violates(i):
        a, b = objs[i], objs[i + 1]
        if not (a and a == b and a not in sa.STOP_TAIL):
            return False
        return verbs[i] == verbs[i + 1] or verbs[i] in TRANSPORT or verbs[i + 1] in TRANSPORT

    n_pairs = sum(1 for i in range(n - 1) if pair_violates(i))
    adj_same = [False] * n
    for i in range(n - 1):
        if pair_violates(i):
            adj_same[i] = adj_same[i + 1] = True

    is_under4 = dur < 4.0
    whole_video_single = (n == 1 and float(segs[0]["start"]) <= 0.5
                          and float(segs[0]["end"]) >= dur - 0.5)

    out = []
    for i, (s, d) in enumerate(zip(segs, sd)):
        badges = []
        dur_ok = True
        if 3.0 <= d <= 10.0:
            pass
        elif d > 10.0 and verbs1[i] and not adj_same[i]:
            badges.append(("b-info", f"merge-wins: {d:.1f}s continuous same-object"))
        elif is_under4 and whole_video_single:
            pass  # green whole-video badge added below
        else:
            dur_ok = False
            badges.append(("b-bad", f"duration {d:.1f}s out of 3-10s"))
        if adj_same[i]:
            badges.append(("b-warn", "same-object split"))
        if is_under4 and whole_video_single:
            badges.append(("b-ok", "whole-video segment"))
        out.append(dict(seg=s, dur=d, badges=badges, dur_ok=dur_ok))
    return out, n_pairs


def verify_vs_scores(clip_id, tag, per_seg, n_pairs, scores):
    """Build-time gate: badge computation must agree with the recorded scoring JSON."""
    pc = next(r for r in scores[tag]["per_clip"] if r["clip_id"] == clip_id)
    n_red = sum(1 for s in per_seg if not s["dur_ok"])
    if len(per_seg) != pc["n_segments"]:
        raise AssertionError(f"{tag}/{clip_id}: n_segments {len(per_seg)} != {pc['n_segments']}")
    if n_red != pc["n_segments"] - pc["r3_seg_ok"]:
        raise AssertionError(f"{tag}/{clip_id}: red badges {n_red} != "
                             f"{pc['n_segments'] - pc['r3_seg_ok']}")
    if n_pairs != pc["r2_sameobj_pairs"]:
        raise AssertionError(f"{tag}/{clip_id}: sameobj pairs {n_pairs} != "
                             f"{pc['r2_sameobj_pairs']}")
    got = [round(s["dur"], 2) for s in per_seg]
    if got != pc["seg_durations"]:
        raise AssertionError(f"{tag}/{clip_id}: seg_durations drift")


# ------------------------------------------------------------- rendering
def clip_fps(ds, rec):
    return float(rec["descriptor"].get("fps") or bv.ADAPTERS[ds]["fps"])


def render_clip_mp4(ds, rec, out_mp4, max_h=360):
    """Stream the clip's frames (descriptor.frame_names order) from the local
    tarcache tar into an mp4 at the dataset's native fps."""
    import cv2
    import numpy as np
    from render_clip_card import _writer
    d = rec["descriptor"]
    tar = EVAL_DIR / "tarcache" / ds / os.path.basename(d["shard_path"])
    if not tar.exists():
        raise FileNotFoundError(f"tar not cached: {tar}")
    fps = clip_fps(ds, rec)
    names = d.get("frame_names") or []
    wr, W, H = None, None, None
    with tarfile.open(tar) as tf:
        members = {m.name: m for m in tf.getmembers()}
        if not names:
            names = sorted(n for n in members if n.endswith(".image.jpg"))
        nfrm = 0
        for nm in names:
            m = members.get(nm) or members.get(nm + ".image.jpg")
            if m is None:
                raise KeyError(f"{tar.name}: missing frame {nm}")
            im = cv2.imdecode(np.frombuffer(tf.extractfile(m).read(), np.uint8),
                              cv2.IMREAD_COLOR)
            if wr is None:
                h, w = im.shape[:2]
                s = min(1.0, max_h / h)
                W, H = int(w * s) // 2 * 2, int(h * s) // 2 * 2
                wr = _writer(out_mp4, W, H, fps)
            if (im.shape[1], im.shape[0]) != (W, H):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            wr.send(np.ascontiguousarray(im[:, :, ::-1]))
            nfrm += 1
    wr.close()
    return nfrm, fps


def probe_duration(path):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    reader.close()
    return float(meta["duration"])


def cut_window(s0, s1, clip_dur):
    """Clamp an annotation's [s0, s1] to a cut window that always contains frames:
    a start no later than clip_dur - 0.35 (a start at/past the last frame makes
    ffmpeg emit a streamless mp4) and a minimum 0.35s span. Returns
    (cut_start, cut_end, expected_physical_duration)."""
    a = max(0.0, min(float(s0), clip_dur - 0.35))
    b = min(float(s1), clip_dur + 0.5)
    t = max(0.35, b - a)
    return a, a + t, min(t, clip_dur - a)


def segment_uri(mp4_path, start, end, max_h, crf):
    """data:video/mp4 URI for one segment — same cut/re-encode recipe as
    build_viewer._segment_uri, parameterized for the long-clip 288p tier."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
        tmp = t.name
    dur = max(0.3, end - start)
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
                    "-t", f"{dur:.2f}", "-i", str(mp4_path),
                    "-vf", f"scale=-2:'min({max_h},ih)'", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart",
                    "-an", tmp], check=True, capture_output=True)
    data = Path(tmp).read_bytes()
    os.unlink(tmp)
    if not data:
        raise RuntimeError("empty segment cut")
    return "data:video/mp4;base64," + base64.b64encode(data).decode()


# ------------------------------------------------------------- HTML
EXTRA_CSS = """
.grid1{display:grid;grid-template-columns:1fr;gap:1.4rem;max-width:1400px}
.abwrap{display:grid;grid-template-columns:1fr 1fr;gap:.9rem;align-items:start}
.col{min-width:0}
.colhead{font-weight:700;font-size:.95rem;margin:.2rem 0 .5rem;color:var(--acc);
display:flex;align-items:center;gap:.5rem}
.colhead .chip{font-size:.78rem;padding:.2rem .55rem}
.b-warn{background:rgba(255,167,38,.15);color:#ffb74d;border:1px solid rgba(255,167,38,.45)}
.b-info{background:rgba(124,179,255,.12);color:var(--acc);border:1px solid rgba(124,179,255,.4)}
.why{font-size:.82rem;color:var(--mut)}
.why b{color:var(--tx);font-weight:600}
@media(max-width:900px){.abwrap{grid-template-columns:1fr}}
"""

STRATUM_LABEL = {
    "under4": "&lt;4s video (rule-1 target)",
    "sameobj": "same-object over-splitting (rule-2 target)",
    "long": "long clip (rule-3 3-10s segmentation)",
}


def seg_block_html(clip_id, col, i, sv):
    sg, d = sv["seg"], sv["dur"]
    s0, s1 = float(sg.get("start", 0)), float(sg["end"])
    badges = "".join(f'<span class="badge {cls}">{bv.esc(txt)}</span>'
                     for cls, txt in sv["badges"])
    lv = sg.get("language_instructions") or {}
    rows = "".join(
        f'<div class="lv"><span class="lab">L{j}</span><span>{bv.esc(lv[k])}</span></div>'
        for j, k in enumerate(("level1", "level2", "level3", "level4"), 1) if lv.get(k))
    return (f'<div class="segblock"><video controls muted playsinline preload="metadata" '
            f'data-clip="{bv.esc(clip_id)}" data-col="{col}" data-i="{i}" '
            f'data-s0="{s0:g}" data-s1="{s1:g}" src="{sv["uri"]}"></video>'
            f'<div class="seg"><div class="rng">seg {i} &middot; {s0:g}&ndash;{s1:g}s '
            f'({d:.1f}s){badges}</div>{rows}</div></div>')


def card_html(e, cols):
    cid = e["clip_id"]
    stratum = STRATUM_LABEL.get(e["stratum"], bv.esc(e["stratum"]))
    colhtml = ""
    for tag, label in ((V4_TAG, "v4"), (V5_TAG, "v5")):
        c = cols[tag]
        blocks = "".join(seg_block_html(cid, label, i + 1, sv)
                         for i, sv in enumerate(c["segs"]))
        colhtml += (f'<div class="col"><div class="colhead">{label} '
                    f'<span class="chip">{len(c["segs"])} seg'
                    f'{"s" if len(c["segs"]) != 1 else ""}</span></div>{blocks}</div>')
    return (f'<figure class="card"><div class="body">'
            f'<div class="cid">{bv.esc(cid)}</div>'
            f'<div class="meta">{e["dataset"]} &middot; {e["duration"]:g}s &middot; '
            f'{cols[V4_TAG]["fps"]:g} fps</div>'
            f'<div class="why"><b>why selected:</b> {stratum} &mdash; '
            f'{bv.esc(e["reason"])}</div>'
            f'<div class="abwrap">{colhtml}</div></div></figure>')


def page_html(ds, part, nparts, cards_html, chips):
    part_note = f" &mdash; page {part}/{nparts}" if nparts > 1 else ""
    chiph = "".join(f'<span class="chip">{k} <b>{bv.esc(v)}</b></span>' for k, v in chips)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{bv.esc(ds)} — prompt viewer (v4 vs v5)</title>'
            f'<style>{bv.CSS}{EXTRA_CSS}</style></head><body>'
            f'<h1>{bv.esc(ds)} — v4 vs v5 prompt A/B{part_note}</h1>'
            f'<div class="sub"><a href="{AUTH}/index.html">&larr; prompt viewer index</a>'
            f' &middot; left column = v4 baseline, right column = v5 final; each segment '
            f'is cut at that prompt&rsquo;s own start/end at native fps</div>'
            f'<div class="stats">{chiph}</div>'
            f'<div class="grid1">{"".join(cards_html)}</div></body></html>')


METRIC_ROWS = [
    ("clips", "clips"),
    ("mean_segs_per_clip", "mean segments / clip"),
    ("pct_seg_duration_ok", "% segments duration-compliant (3-10s incl. sanctioned merges)"),
    ("under4_single_seg", "<4s clips labeled as ONE whole-video segment (of 9)"),
    ("sameobj_consecutive_pairs", "consecutive same-object split pairs (R2 violations)"),
    ("pct_single_verb_l1", "% L1 with exactly one verb"),
    ("pct_l1_verb_start", "% L1 starting with a verb"),
    ("pct_levels_present", "% segments with all of L1-L4"),
    ("pct_word_caps_ok", "% segments within word caps"),
    ("pct_l4_hand_side", "% L4 naming a hand side"),
    ("mean_object_recall_vs_v4", "object recall vs fresh v4 baseline"),
    ("clips_with_overlap", "clips with overlapping segments"),
    ("total_cost_usd", "labeling cost (USD, 40 clips)"),
]


def index_html(scores, ds_rows, page_links):
    a4, a5 = scores[V4_TAG]["aggregate"]["ALL"], scores[V5_TAG]["aggregate"]["ALL"]

    def cell(agg, key):
        v = agg.get(key)
        if key == "mean_object_recall_vs_v4" and agg is a4:
            return "1.0 (self)"
        return "—" if v is None else str(v)

    mrows = "".join(f'<tr><td>{bv.esc(lab)}</td>'
                    f'<td class="num">{cell(a4, k)}</td>'
                    f'<td class="num">{cell(a5, k)}</td></tr>'
                    for k, lab in METRIC_ROWS)
    drows = ""
    for ds, nclip, s4, s5 in ds_rows:
        links = " &middot; ".join(f'<a href="{AUTH}/{fn}">{bv.esc(fn)}</a> ({mb:.0f} MB)'
                                  for fn, mb in page_links[ds])
        drows += (f'<tr><td>{bv.esc(ds)}</td><td class="num">{nclip}</td>'
                  f'<td class="num">{s4}</td><td class="num">{s5}</td>'
                  f'<td class="num">{s5 - s4:+d}</td><td>{links}</td></tr>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prompt viewer — v4 vs v5 labeling prompt A/B</title>
<style>{bv.CSS}{EXTRA_CSS}</style></head><body>
<h1>Prompt viewer: v4 vs v5 labeling prompt A/B</h1>
<div class="sub"><a href="{bv.AUTH_BASE}/index.html">&larr; dataset viewer</a> &middot;
40-clip eval set &middot; fresh same-day runs 2026-08-28 &middot; gpt-5-mini, effort=medium,
production P2 sampling (1024px, ~3fps, 12-40 frames) &middot; recorded outputs only (labeling hold)</div>

<h2>What is compared</h2>
<div class="seg" style="max-width:960px">
<div class="lv"><span class="lab">v4</span><span><b>Shipped v4 doctrine</b> (annotation_general_clip_v4,
run tag <code>v4_baseline</code>): atomic segments; a tool-use sequence is <em>decomposed</em> into
acquire / use / release, each its own segment, with boundaries at grasp and release.</span></div>
<div class="lv"><span class="lab">v5</span><span><b>Final v5 prompt</b> (run tag <code>v5d_full</code>,
byte-identical to the shipped <code>annotation_general_clip_v5.txt</code>). Three rules +
merge-wins: <b>R1</b> a video under 4s gets exactly ONE whole-video segment; <b>R2</b> consecutive
motions on the same object (reach / grasp / use / release) merge into ONE segment with a
single-verb L1 — never a boundary at the grasp or the release; <b>R3</b> segments target 3-10s,
and a continuous same-object manipulation longer than 10s stays ONE segment
(<em>merge&nbsp;wins</em> over the 10s cap).</span></div>
</div>

<h2>A/B metrics (ALL 40 clips)</h2>
<div class="sub">Baked from the recorded scoring results
(<code>outputs/{V4_TAG}.scores.json</code> / <code>outputs/{V5_TAG}.scores.json</code>,
scored by <code>score_annotations.py</code>).</div>
<table><tr><th>metric</th><th>v4 ({V4_TAG})</th><th>v5 ({V5_TAG})</th></tr>{mrows}</table>
<div class="sub" style="margin-top:.6rem">Object recall vs v4 is expected to drop under v5:
merged segments have ONE head object per L1, so v4's per-phase objects (tool + target listed
separately) fold into fewer L1s. See the per-clip cards to judge whether the merged L1 still
names the right action.</div>

<h2>Per-dataset pages</h2>
<div class="sub">One card per eval clip: v4 segments (left) vs v5 segments (right), each cut
at its own start/end with compliance badges. Pages over ~140&nbsp;MB are split.</div>
<table><tr><th>dataset</th><th>clips</th><th>v4 segs</th><th>v5 segs</th><th>&Delta;</th>
<th>pages</th></tr>{drows}</table>
</body></html>"""


# ------------------------------------------------------------- phases
def render_phase(work, eval_set, manifests, only_ds=None):
    clips_dir = work / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for e in eval_set:
        ds, cid = e["dataset"], e["clip_id"]
        if only_ds and ds not in only_ds:
            continue
        out = clips_dir / f"{cid}.mp4"
        rec = manifests[ds][cid]
        if out.exists():
            continue
        nfrm, fps = render_clip_mp4(ds, rec, out)
        got = probe_duration(out)
        want = nfrm / fps
        if abs(got - want) > max(0.2, 2.0 / fps):
            raise RuntimeError(f"{cid}: rendered {got:.2f}s != {want:.2f}s ({nfrm}f @ {fps}g)")
        print(f"[render] {cid} {nfrm}f @ {fps:g}fps -> {got:.2f}s "
              f"{out.stat().st_size/1e6:.1f}MB", flush=True)


def publish_phase(work, eval_set, runs, scores, manifests, only_ds=None, upload=True):
    clips_dir = work / "clips"
    by_ds = {}
    for e in eval_set:
        by_ds.setdefault(e["dataset"], []).append(e)
    page_links = {}
    ds_rows = []
    pool = ThreadPoolExecutor(8)
    for ds in DATASETS:
        if only_ds and ds not in only_ds:
            continue
        cards, seg_counts = [], {V4_TAG: 0, V5_TAG: 0}
        for e in by_ds[ds]:
            cid = e["clip_id"]
            mp4 = clips_dir / f"{cid}.mp4"
            if not mp4.exists():
                raise FileNotFoundError(f"run --render first: {mp4}")
            clip_dur = e["duration"]
            max_h, crf = (288, 30) if clip_dur > LONG_CLIP_SEC else (360, 28)
            cols = {}
            for tag in (V4_TAG, V5_TAG):
                row = runs[tag][cid]
                per_seg, n_pairs = analyze_segments(row)
                verify_vs_scores(cid, tag, per_seg, n_pairs, scores)
                cuts = list(pool.map(
                    lambda sv: segment_uri(
                        mp4,
                        *cut_window(sv["seg"].get("start", 0), sv["seg"]["end"],
                                    clip_dur)[:2],
                        max_h, crf),
                    per_seg))
                for sv, uri in zip(per_seg, cuts):
                    sv["uri"] = uri
                cols[tag] = dict(segs=per_seg, fps=clip_fps(ds, manifests[ds][cid]))
                seg_counts[tag] += len(per_seg)
            cards.append(card_html(e, cols))
            print(f"[{ds}] {cid} v4={len(cols[V4_TAG]['segs'])} "
                  f"v5={len(cols[V5_TAG]['segs'])} segs", flush=True)
        # greedy split into <=MAX_PAGE_BYTES pages, preserving eval-set order
        groups, cur, cur_b = [], [], 0
        for ch in cards:
            b = len(ch.encode())
            if cur and cur_b + b > MAX_PAGE_BYTES:
                groups.append(cur)
                cur, cur_b = [], 0
            cur.append(ch)
            cur_b += b
        if cur:
            groups.append(cur)
        a4 = scores[V4_TAG]["aggregate"][ds]
        a5 = scores[V5_TAG]["aggregate"][ds]
        chips = [("clips", len(by_ds[ds])),
                 ("v4 segments", seg_counts[V4_TAG]),
                 ("v5 segments", seg_counts[V5_TAG]),
                 ("v4 % duration-ok", a4["pct_seg_duration_ok"]),
                 ("v5 % duration-ok", a5["pct_seg_duration_ok"])]
        links = []
        for i, grp in enumerate(groups, 1):
            fn = f"{ds}.html" if len(groups) == 1 else f"{ds}_p{i}.html"
            pg = work / fn
            pg.write_text(page_html(ds, i, len(groups), grp, chips))
            mb = pg.stat().st_size / 1e6
            links.append((fn, mb))
            print(f"[{ds}] wrote {fn} {mb:.1f} MB", flush=True)
        page_links[ds] = links
        ds_rows.append((ds, len(by_ds[ds]), seg_counts[V4_TAG], seg_counts[V5_TAG]))
    # index
    idx = work / "index.html"
    idx.write_text(index_html(scores, ds_rows, page_links))
    print(f"[index] {idx.stat().st_size/1e3:.0f} KB", flush=True)
    if upload:
        htmls = [str(idx)] + [str(work / fn) for links in page_links.values()
                              for fn, _ in links]
        subprocess.run(["gcloud", "storage", "cp", "--content-type=text/html",
                        "--cache-control=no-store", "-q"] + htmls + [f"gs://{PV}/"],
                       check=True)
        print(f"published -> {AUTH}/index.html", flush=True)
    return page_links


# ------------------------------------------------------------- selfcheck
VID_RE = re.compile(
    r'<video[^>]*data-clip="([^"]+)" data-col="(v4|v5)" data-i="(\d+)" '
    r'data-s0="([^"]+)" data-s1="([^"]+)" src="data:video/mp4;base64,([^"]+)"')


def selfcheck(work, eval_set, scores):
    ck = work / "_selfcheck"
    ck.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(["gcloud", "storage", "ls", f"gs://{PV}/"],
                             capture_output=True, text=True, check=True).stdout.split()
    pages = [p for p in listing if p.endswith(".html") and not p.endswith("index.html")]
    durs = {e["clip_id"]: e["duration"] for e in eval_set}
    pc = {tag: {r["clip_id"]: r for r in scores[tag]["per_clip"]}
          for tag in (V4_TAG, V5_TAG)}
    seen = {V4_TAG: {}, V5_TAG: {}}   # clip -> [seg indices found on pages]
    bad = []
    for url in sorted(pages):
        local = ck / os.path.basename(url)
        subprocess.run(["gcloud", "storage", "cp", "-q", url, str(local)], check=True)
        txt = local.read_text()
        for m in VID_RE.finditer(txt):
            cid, col, i, s0, s1, b64 = m.groups()
            tag = V4_TAG if col == "v4" else V5_TAG
            tmp = ck / "_seg.mp4"
            tmp.write_bytes(base64.b64decode(b64))
            got = probe_duration(tmp)
            _, _, want = cut_window(float(s0), float(s1), durs[cid])
            if abs(got - want) > DUR_TOL:
                bad.append(f"{os.path.basename(url)} {cid} {col} seg{i}: "
                           f"cut {got:.2f}s != expected {want:.2f}s ({s0}-{s1})")
            seen[tag].setdefault(cid, []).append(int(i))
        # per-card: both columns present; red badge counts vs the scoring JSON
        for card in txt.split('<figure class="card">')[1:]:
            mcid = re.search(r'<div class="cid">([^<]+)</div>', card)
            if not mcid:
                continue
            cid = html.unescape(mcid.group(1))
            colparts = card.split('<div class="colhead">')
            if len(colparts) < 3:
                bad.append(f"{cid}: missing column(s) ({len(colparts) - 1} found)")
                continue
            for blk, tag in ((colparts[1], V4_TAG), (colparts[2], V5_TAG)):
                nred = blk.count("out of 3-10s")
                r = pc[tag][cid]
                if nred != r["n_segments"] - r["r3_seg_ok"]:
                    bad.append(f"{cid} {tag}: red badges {nred} != "
                               f"{r['n_segments'] - r['r3_seg_ok']}")
        print(f"[check] {os.path.basename(url)} ok so far={not bad}", flush=True)
    # coverage: every clip, both columns, all segments
    for e in eval_set:
        cid = e["clip_id"]
        for tag in (V4_TAG, V5_TAG):
            want_n = pc[tag][cid]["n_segments"]
            got = sorted(seen[tag].get(cid, []))
            if got != list(range(1, want_n + 1)):
                bad.append(f"{cid} {tag}: segments on page {got} != 1..{want_n}")
    if bad:
        print("\nSELF-CHECK FAILURES:")
        for b in bad:
            print(" ", b)
        sys.exit(1)
    nseg = sum(len(v) for t in seen.values() for v in t.values())
    print(f"\nSELF-CHECK PASS: {len(pages)} pages, 40 clips x 2 columns, "
          f"{nseg} segment videos verified", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/viewer_work/prompt_viewer")
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    work = Path(args.work)
    only = set(args.datasets) or None
    eval_set, runs, scores, manifests = load_inputs()
    if args.render:
        render_phase(work, eval_set, manifests, only)
    if args.publish:
        publish_phase(work, eval_set, runs, scores, manifests, only,
                      upload=not args.no_upload)
    if args.selfcheck:
        selfcheck(work, eval_set, scores)


if __name__ == "__main__":
    main()
