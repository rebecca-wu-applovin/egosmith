#!/usr/bin/env python
"""Build the filtered-data viewer: per-dataset sample galleries hosted on GCS.

Per dataset: sample N kept clips (joined with their LLM annotations), render a card
video for each (keypoint overlay where recon/native keypoints exist), emit an
index.html card grid, and upload to
  gs://foundational-research/hoi-dataset/egosmith_filtered/viewer/<ds>/
Browsable via authenticated links:
  https://storage.cloud.google.com/foundational-research/hoi-dataset/egosmith_filtered/viewer/index.html

Phases (composable for parallel per-dataset agents):
  --render   sample + render clips + write <work>/<ds>/cards.json
  --publish  cards.json + clips -> index.html -> GCS
  --root     regenerate the root index from every published dataset

cards.json contract (list of card dicts) — external adapters (robot / Stage-1 datasets)
may produce this file themselves and only use --publish:
  {clip_id, video: "clips/<f>.mp4", status: overlay|rgb_only|video_only, note: str|null,
   meta: {dur_s, fps, w, h, hand_pct, ...}, annotation: {segments:[...]} | null,
   extra_rows: [[label, value], ...]}
"""
import argparse
import html
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO / "src"), str(_REPO), str(_REPO / "scripts" / "build"),
          str(_REPO / "scripts" / "viewer")):
    if p not in sys.path:
        sys.path.insert(0, p)

import gcsfs  # noqa: E402
import numpy as np  # noqa: E402

from render_clip_card import (render_recon_overlay, render_native_overlay,  # noqa: E402
                              render_plain_tar)
from phase_d_incremental import ranged_npz_extract  # noqa: E402

BUCKET = "foundational-research/hoi-dataset"
VIEWER = f"{BUCKET}/egosmith_filtered/viewer"
fs = gcsfs.GCSFileSystem()

# ---------------------------------------------------------------- adapters
ADAPTERS = {
    "egocentric100k": dict(kind="ego", fps=15.0,
                           filt=f"{BUCKET}/egosmith_filtered/egocentric100k/filter_run/_shards",
                           ann=f"{BUCKET}/egosmith_filtered/egocentric100k/filter_run/annotations_v4/_shards",
                           frames=f"{BUCKET}/egosmith_filtered/egocentric100k/frames",
                           recon=f"{BUCKET}/egosmith_recon/egocentric100k/recon/outputs",
                           funnel=f"{BUCKET}/egosmith_filtered/egocentric100k/filter_run/funnel.json"),
    "egocentric10k": dict(kind="ego", fps=15.0,
                          filt=f"{BUCKET}/egosmith_filtered/egocentric10k/filter_run/_shards",
                          ann=f"{BUCKET}/egosmith_filtered/egocentric10k/filter_run/annotations_v4/_shards",
                          frames=f"{BUCKET}/egosmith_filtered/egocentric10k/frames",
                          recon=f"{BUCKET}/egosmith_recon/egocentric10k/recon/outputs"),
    # Cat-3, recon colocated under egosmith_filtered/<ds>/outputs/
    # fps=30: descriptor.fps is null for these; annotations use frames/30 (native rate)
    "dexycb": dict(kind="cat3", fps=30.0,
                   outputs=f"{BUCKET}/egosmith_filtered/dexycb/outputs"),
    "ho3d_v3": dict(kind="cat3", fps=30.0,
                    outputs=f"{BUCKET}/egosmith_filtered/ho3d_v3/outputs"),
    "show3d": dict(kind="cat3", fps=30.0,
                   outputs=f"{BUCKET}/egosmith_filtered/show3d/outputs"),
    "hoi4d": dict(kind="cat3", fps=30.0,
                  outputs=f"{BUCKET}/egosmith_filtered/hoi4d/outputs"),
    # Cat-3, GT-mode recon under egosmith_recon/<ds>/use_gt/outputs/
    "taco": dict(kind="cat3", outputs=f"{BUCKET}/egosmith_recon/taco/use_gt/outputs"),
    "oakink_actions": dict(kind="cat3",
                           outputs=f"{BUCKET}/egosmith_recon/oakink_actions/use_gt/outputs"),
    # hot3d ships pre-rendered viz/*.overlay.mp4 — copy, don't re-render
    "hot3d": dict(kind="hot3d_viz"),
    # native lowdim (Vision Pro GT) — overlay straight from .lowdim.npy
    "egodex": dict(kind="native", fps=30.0),
}
CATEGORY = {"egocentric100k": "Egocentric (recon)", "egocentric10k": "Egocentric (recon)",
            "dexycb": "Cat-3 GT", "ho3d_v3": "Cat-3 GT", "show3d": "Cat-3 GT",
            "hoi4d": "Cat-3 GT", "taco": "Cat-3 GT", "oakink_actions": "Cat-3 GT",
            "hot3d": "Cat-3 GT", "egodex": "Cat-2 native GT",
            # Cat-4 robot datasets (video-only cards, external adapter)
            "trex": "Cat-4 robot", "dexora": "Cat-4 robot", "dexwild": "Cat-4 robot",
            "hrdexdb_allegro": "Cat-4 robot", "realdex": "Cat-4 robot",
            # Cat-1 Stage-1-only previews (video-only cards, external adapter)
            "assembly101": "Cat-1 Stage-1 preview", "hd_epic": "Cat-1 Stage-1 preview",
            "holoassist": "Cat-1 Stage-1 preview", "ego4d": "Cat-1 Stage-1 preview",
            "epic_kitchens_100": "Cat-1 Stage-1 preview"}


def _read_jsonl_gcs(path):
    with fs.open(path, "rb") as f:
        return [json.loads(l) for l in f.read().decode().splitlines() if l.strip()]


def _cat3_paths(ds):
    base = f"{BUCKET}/egosmith_filtered/{ds}/filter_run"
    manifest = f"{base}/clip_manifest.filtered.jsonl"
    ann = None
    for cand in ("annotations.v4.jsonl", "annotations_v4.jsonl", "annotations.jsonl"):
        if fs.exists(f"{base}/{cand}"):
            ann = f"{base}/{cand}"
            break
    return manifest, ann


def load_annotations(ds, ad):
    """clip_id -> annotation row (whole-file for cat3, per-shard for ego)."""
    if ad["kind"] == "ego":
        return None  # joined per-shard in sample_ego
    _, ann = _cat3_paths(ds)
    if not ann:
        return {}
    return {r["clip_id"]: r for r in _read_jsonl_gcs(ann)}


def sample_ego(ds, ad, n, seed, shards_k=10):
    """Sample across shards that finished filter+annotation; join by clip_id."""
    rng = random.Random(seed)
    ann_shards = {os.path.basename(p).split(".")[0] for p in fs.ls(ad["ann"])
                  if p.endswith(".annotations.jsonl")}
    filt_shards = sorted(os.path.basename(p).split(".")[0] for p in fs.ls(ad["filt"])
                         if p.endswith(".filtered.jsonl"))
    ready = [s for s in filt_shards if s in ann_shards]
    if not ready:
        raise RuntimeError(f"{ds}: no shards with both filter + annotations")
    picked = rng.sample(ready, min(shards_k, len(ready)))
    per = max(1, n // len(picked))
    out = []
    for sh in picked:
        recs = _read_jsonl_gcs(f"{ad['filt']}/{sh}.filtered.jsonl")
        anns = {r["clip_id"]: r for r in _read_jsonl_gcs(f"{ad['ann']}/{sh}.annotations.jsonl")}
        joined = [r for r in recs if r["clip_id"] in anns]
        for r in rng.sample(joined, min(per, len(joined))):
            sfx = re.search(r"(\d{5})", r["descriptor"]["root_dir"]).group(1)
            out.append(dict(rec=r, ann=anns[r["clip_id"]], shard=sfx))
    return out[:n]


def sample_dropped_ego(ds, ad, n, seed):
    """Sample DROPPED clips (with reasons) from per-shard filter reports; descriptors
    come from the phaseB manifest (dropped clips are absent from filtered.jsonl)."""
    rng = random.Random(seed + 1)
    pb = ad["filt"].replace("/filter_run/_shards", "/phaseB/_shards")
    reports = [p for p in fs.ls(ad["filt"]) if p.endswith(".report.json")]
    out, seen_reasons = [], {}
    for rp in rng.sample(reports, min(12, len(reports))):
        if len(out) >= n:
            break
        rep = json.loads(fs.open(rp, "rb").read())
        dropped = [d for d in rep.get("dropped", []) if d.get("drop_category") == "quality"]
        rng.shuffle(dropped)
        sfx = os.path.basename(rp).split(".")[0].replace("shard_", "")
        picked = []
        for d in dropped:  # spread across distinct primary reasons
            key = (d.get("reasons") or ["?"])[0]
            if seen_reasons.get(key, 0) >= max(2, n // 5):
                continue
            seen_reasons[key] = seen_reasons.get(key, 0) + 1
            picked.append(d)
            if len(picked) >= 2:
                break
        if not picked:
            continue
        by_id = {d["clip_id"]: d for d in picked}
        for line in fs.open(f"{pb}/shard_{sfx}.manifest.jsonl", "rb").read().decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["clip_id"] in by_id:
                out.append(dict(rec=r, ann=None, shard=sfx,
                                reasons=by_id[r["clip_id"]].get("reasons", [])))
    return out[:n]


def sample_dropped_cat3(ds, ad, n, seed):
    rng = random.Random(seed + 1)
    base = f"{BUCKET}/egosmith_filtered/{ds}/filter_run"
    rep_path = f"{base}/filter_report.json"
    if not fs.exists(rep_path):
        return []
    rep = json.loads(fs.open(rep_path, "rb").read())
    dropped = rep.get("dropped", [])
    if not dropped:
        return []
    pre = f"{base}/clip_manifest.jsonl"
    recs = {r["clip_id"]: r for r in _read_jsonl_gcs(pre)} if fs.exists(pre) else {}
    rng.shuffle(dropped)
    out = []
    for d in dropped:
        if len(out) >= n:
            break
        r = recs.get(d["clip_id"])
        if r:
            out.append(dict(rec=r, ann=None, shard=None, reasons=d.get("reasons", [])))
    return out


def sample_cat3(ds, ad, n, seed):
    rng = random.Random(seed)
    manifest, _ = _cat3_paths(ds)
    recs = _read_jsonl_gcs(manifest)
    anns = load_annotations(ds, ad)
    joined = [r for r in recs if r["clip_id"] in anns] or recs
    return [dict(rec=r, ann=anns.get(r["clip_id"]), shard=None)
            for r in rng.sample(joined, min(n, len(joined)))]


NPZ_KEYS = ["traj", "tstamp", "img_focal", "img_center", "scale"]


def materialize_seq(gcs_prefix, dst):
    """Minimal local seq folder: small SLAM members (ranged, skips disps) + pose file."""
    dst.mkdir(parents=True, exist_ok=True)
    npzs = [p for p in fs.ls(f"{gcs_prefix}/SLAM") if "hawor_slam_w_scale_" in p]
    if not npzs:
        raise FileNotFoundError(f"no slam npz under {gcs_prefix}/SLAM")
    small = ranged_npz_extract(npzs[0], NPZ_KEYS)
    (dst / "SLAM").mkdir(exist_ok=True)
    np.savez(dst / "SLAM" / os.path.basename(npzs[0]), **small)
    fs.get(f"{gcs_prefix}/world_space_res.pth", str(dst / "world_space_res.pth"))
    if fs.exists(f"{gcs_prefix}/est_focal.txt"):
        fs.get(f"{gcs_prefix}/est_focal.txt", str(dst / "est_focal.txt"))
    return dst


def render_dataset(ds, n, seed, work):
    ad = ADAPTERS[ds]
    wdir = work / ds
    clips_dir = wdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    cards, errors = [], []

    if ad["kind"] == "hot3d_viz":
        vizs = sorted(p for p in fs.ls(f"{BUCKET}/egosmith_filtered/hot3d/viz")
                      if p.endswith(".overlay.mp4"))
        anns = load_annotations(ds, ad)
        rng = random.Random(seed)
        for p in rng.sample(vizs, min(n, len(vizs))):
            cid = os.path.basename(p).replace(".overlay.mp4", "")
            fs.get(p, str(clips_dir / f"{cid}.mp4"))
            cards.append(dict(clip_id=cid, video=f"clips/{cid}.mp4", status="overlay",
                              note="pre-rendered GT-mode overlay", meta={},
                              annotation=(anns.get(cid) or {}).get("annotation"),
                              extra_rows=[]))
        # dropped: no pre-rendered viz — render from frames tar + use_gt recon outputs
        dropped = sample_dropped_cat3(ds, ad, max(6, n // 5), seed)
        print(f"[{ds}] sampled {len(cards)} kept + {len(dropped)} dropped", flush=True)
        for s in dropped:
            cid = s["rec"]["clip_id"]
            d = s["rec"]["descriptor"]
            tar_local = wdir / "_tars" / f"{cid}.tar"
            tar_local.parent.mkdir(parents=True, exist_ok=True)
            out = clips_dir / f"{cid}.mp4"
            try:
                fs.get(f"{BUCKET}/egosmith_filtered/{ds}/frames/"
                       f"{os.path.basename(d['shard_path'])}", str(tar_local))
                seq = materialize_seq(f"{BUCKET}/egosmith_recon/{ds}/use_gt/outputs/{cid}",
                                      wdir / "_seq" / cid)
                st = render_recon_overlay(tar_local, seq, out, float(d.get("fps") or 30.0))
            except Exception as e:  # noqa: BLE001
                errors.append({"clip_id": cid, "stage": "dropped", "error": str(e)[:200]})
                continue
            finally:
                tar_local.unlink(missing_ok=True)
            cards.append(dict(clip_id=cid, video=f"clips/{cid}.mp4", status=st["status"],
                              note=st.get("note"), section="dropped",
                              reasons=s.get("reasons", []),
                              meta={k: st.get(k) for k in ("dur_s", "fps", "w", "h", "hand_pct")},
                              annotation=None, extra_rows=[]))
            print(f"[{ds}] dropped {cid} {st['status']}", flush=True)
        _finish(ds, wdir, cards, errors, n)
        return

    n_dropped = max(6, n // 5)
    if ad["kind"] == "ego":
        samples = sample_ego(ds, ad, n, seed)
        dropped = sample_dropped_ego(ds, ad, n_dropped, seed)
    else:
        samples = sample_cat3(ds, ad, n, seed)
        dropped = sample_dropped_cat3(ds, ad, n_dropped, seed)
    for s in samples:
        s["section"] = "kept"
    for s in dropped:
        s["section"] = "dropped"
    samples = samples + dropped
    print(f"[{ds}] sampled {len(samples) - len(dropped)} kept + {len(dropped)} dropped",
          flush=True)

    def fetch(s):
        rec, cid = s["rec"], s["rec"]["clip_id"]
        d = rec["descriptor"]
        tar_local = wdir / "_tars" / f"{cid}.tar"
        tar_local.parent.mkdir(parents=True, exist_ok=True)
        if ad["kind"] == "ego":
            fs.get(f"{ad['frames']}/shard_{s['shard']}/{cid}.tar", str(tar_local))
            seq = materialize_seq(f"{ad['recon']}/shard_{s['shard']}/{cid}", wdir / "_seq" / cid)
        elif ad["kind"] == "cat3":
            fs.get(f"{BUCKET}/egosmith_filtered/{ds}/frames/{os.path.basename(d['shard_path'])}",
                   str(tar_local))
            seq = materialize_seq(f"{ad['outputs']}/{cid}", wdir / "_seq" / cid)
        else:  # native
            fs.get(f"{BUCKET}/egosmith_filtered/{ds}/frames/{os.path.basename(d['shard_path'])}",
                   str(tar_local))
            seq = None
        return tar_local, seq

    pool = ThreadPoolExecutor(6)
    fetched = list(pool.map(lambda s: _safe(fetch, s), samples))
    for s, f in zip(samples, fetched):
        cid = s["rec"]["clip_id"]
        if isinstance(f, Exception):
            errors.append({"clip_id": cid, "stage": "fetch", "error": str(f)[:200]})
            continue
        tar_local, seq = f
        d = s["rec"]["descriptor"]
        fps = float(d.get("fps") or d.get("extra", {}).get("recon_fps")
                    or ad.get("fps") or 15.0)
        out = clips_dir / f"{cid}.mp4"
        try:
            # one retry: transient ffmpeg broken-pipes under batch memory pressure
            for attempt in (0, 1):
                try:
                    if ad["kind"] == "native":
                        st = render_native_overlay(tar_local, out, fps)
                    else:
                        st = render_recon_overlay(tar_local, seq, out, fps)
                    break
                except Exception:  # noqa: BLE001
                    if attempt:
                        raise
        except Exception as e:  # noqa: BLE001
            try:
                st = render_plain_tar(tar_local, out, fps)
                st["note"] = f"overlay failed: {type(e).__name__}: {str(e)[:120]}"
            except Exception as e2:  # noqa: BLE001
                errors.append({"clip_id": cid, "stage": "render", "error": str(e2)[:200]})
                continue
        finally:
            tar_local.unlink(missing_ok=True)
        cards.append(dict(clip_id=cid, video=f"clips/{cid}.mp4", status=st["status"],
                          note=st.get("note"), section=s.get("section", "kept"),
                          reasons=s.get("reasons", []),
                          meta={k: st.get(k) for k in ("dur_s", "fps", "w", "h", "hand_pct")},
                          annotation=(s["ann"] or {}).get("annotation"),
                          extra_rows=[]))
        print(f"[{ds}] {len(cards)}/{len(samples)} {cid} {st['status']} {s.get('section')}",
              flush=True)
    _finish(ds, wdir, cards, errors, n)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        return e


def _finish(ds, wdir, cards, errors, n):
    (wdir / "cards.json").write_text(json.dumps(cards, indent=1))
    if errors:
        (wdir / "errors.json").write_text(json.dumps(errors, indent=1))
    ov = sum(1 for c in cards if c["status"] == "overlay")
    print(f"[{ds}] DONE cards={len(cards)}/{n} overlay={ov} "
          f"rgb_only={sum(1 for c in cards if c['status']=='rgb_only')} errors={len(errors)}",
          flush=True)


# ---------------------------------------------------------------- HTML
CSS = """
:root{--bg:#101318;--panel:#1a1f27;--panel2:#232a34;--tx:#e8eaed;--mut:#9aa0a6;
--acc:#7cb3ff;--good:#4caf7d;--bad:#e07a5f;--line:#2d3540}
*{box-sizing:border-box}body{background:var(--bg);color:var(--tx);margin:0;
font:15px/1.5 system-ui,-apple-system,sans-serif;padding:2rem}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:1.5rem;margin:0 0 .3rem}
h2{font-size:1.15rem;margin:2rem 0 .3rem;color:var(--bad)}
.sub{color:var(--mut);margin-bottom:1.2rem}
.legend{display:inline-flex;gap:1rem;background:var(--panel);border:1px solid var(--line);
border-radius:6px;padding:.4rem .8rem;font-size:.85rem;margin-bottom:1.4rem}
.dotl,.dotr{display:inline-block;width:.7em;height:.7em;border-radius:50%;margin-right:.35em}
.dotl{background:rgb(66,133,244)}.dotr{background:rgb(219,68,55)}
.stats{display:flex;gap:.6rem;flex-wrap:wrap;margin:0 0 1.4rem}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:6px;
padding:.35rem .7rem;font-size:.85rem}.chip b{color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:1.2rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
overflow:hidden;display:flex;flex-direction:column}
.card video{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:contain}
.body{padding:.8rem .9rem;display:flex;flex-direction:column;gap:.5rem}
.cid{font-family:ui-monospace,Menlo,monospace;font-size:.78rem;color:var(--mut);
word-break:break-all}
.meta{font-size:.8rem;color:var(--mut)}
.badge{display:inline-block;border-radius:4px;padding:.05rem .45rem;font-size:.72rem;
margin-left:.4rem;vertical-align:1px}
.b-ok{background:rgba(76,175,125,.15);color:var(--good);border:1px solid rgba(76,175,125,.4)}
.b-bad{background:rgba(224,122,95,.15);color:var(--bad);border:1px solid rgba(224,122,95,.4)}
.note{font-size:.78rem;color:var(--bad)}
.seg{background:var(--panel2);border-radius:6px;padding:.5rem .6rem}
.seg .rng{font-size:.72rem;color:var(--mut);margin-bottom:.25rem}
.lv{font-size:.8rem;margin:.15rem 0;display:flex;gap:.5rem}
.lv .lab{flex:0 0 2rem;color:var(--acc);font-weight:600;font-size:.72rem;padding-top:.1rem}
table{border-collapse:collapse;width:100%;max-width:960px}
td,th{border-bottom:1px solid var(--line);padding:.5rem .7rem;text-align:left;font-size:.9rem}
th{color:var(--mut);font-weight:600;font-size:.78rem;text-transform:uppercase;
letter-spacing:.05em}td.num{font-variant-numeric:tabular-nums}
"""


def esc(s):
    return html.escape(str(s)) if s is not None else ""


# storage.cloud.google.com serves HTML via one-time signed googleusercontent redirects,
# so RELATIVE links/subresources 404. Dataset pages therefore embed videos as data URIs
# (self-contained) and all inter-page links are absolute auth URLs.
AUTH_BASE = f"https://storage.cloud.google.com/{VIEWER}"
EMBED_MAX_BYTES = 900_000   # embed as-is under this; re-encode smaller above it


def _embed_uri(mp4_path):
    """data:video/mp4 URI, re-encoding big clips down (8s cap, 288p, crf 33)."""
    import base64
    import tempfile
    p = Path(mp4_path)
    data = p.read_bytes()
    if len(data) > EMBED_MAX_BYTES:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
            tmp = t.name
        subprocess.run([ff, "-y", "-loglevel", "error", "-t", "8", "-i", str(p),
                        "-vf", "scale=-2:'min(288,ih)'", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "33", "-movflags", "+faststart",
                        "-an", tmp], check=True, capture_output=True)
        data = Path(tmp).read_bytes()
        os.unlink(tmp)
    return "data:video/mp4;base64," + base64.b64encode(data).decode()


def _card_html(c):
    m = c.get("meta") or {}
    bits = []
    if m.get("dur_s"):
        bits.append(f"{m['dur_s']}s")
    if m.get("fps"):
        bits.append(f"{m['fps']:g} fps")
    if m.get("w"):
        bits.append(f"{m['w']}&times;{m['h']}")
    if m.get("hand_pct") is not None:
        bits.append(f"hands in frame {m['hand_pct']}%")
    badge = ""
    if c["status"] == "rgb_only":
        badge = '<span class="badge b-bad">no overlay</span>'
    elif c["status"] == "video_only":
        badge = '<span class="badge b-bad">video only</span>'
    note = f'<div class="note">{esc(c["note"])}</div>' if c.get("note") else ""
    segs_html = ""
    ann = c.get("annotation")
    if ann and ann.get("segments"):
        for sg in ann["segments"]:
            q = sg.get("is_good_quality")
            qb = ('<span class="badge b-ok">good</span>' if q
                  else '<span class="badge b-bad">flagged</span>' if q is False else "")
            rng = ""
            if sg.get("end") is not None:
                rng = f'<div class="rng">segment {sg.get("start", 0):g}&ndash;{sg["end"]:g}s{qb}</div>'
            lv = sg.get("language_instructions") or {}
            rows = "".join(f'<div class="lv"><span class="lab">L{i}</span><span>{esc(lv[k])}</span></div>'
                           for i, k in enumerate(("level1", "level2", "level3", "level4"), 1)
                           if lv.get(k))
            segs_html += f'<div class="seg">{rng}{rows}</div>'
    elif c.get("extra_rows"):
        rows = "".join(f'<div class="lv"><span class="lab">{esc(a)}</span><span>{esc(b)}</span></div>'
                       for a, b in c["extra_rows"])
        segs_html = f'<div class="seg">{rows}</div>'
    else:
        segs_html = '<div class="seg"><div class="lv"><span>&mdash; no annotation &mdash;</span></div></div>'
    if c.get("reasons"):
        note += ('<div class="note">dropped: '
                 + " ".join(f'<span class="badge b-bad">{esc(r)}</span>' for r in c["reasons"])
                 + "</div>")
    src = c.get("video_uri") or c["video"]
    raw = f' &middot; <a href="{esc(c["raw_url"])}">full quality &nearr;</a>' if c.get("raw_url") else ""
    return (f'<figure class="card"><video controls loop muted playsinline preload="metadata" '
            f'src="{src}"></video><div class="body">'
            f'<div class="cid">{esc(c["clip_id"])}{badge}</div>'
            f'<div class="meta">{" &middot; ".join(bits)}{raw}</div>{note}{segs_html}</div></figure>')


def dataset_html(ds, cards, stats):
    chips = "".join(f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in stats)
    kept = [c for c in cards if c.get("section", "kept") == "kept"]
    dropped = [c for c in cards if c.get("section") == "dropped"]
    body = "".join(_card_html(c) for c in kept)
    dropped_html = ""
    if dropped:
        dropped_html = (f'<h2>Dropped examples ({len(dropped)})</h2>'
                        f'<div class="sub">Clips the quality filter rejected, with reasons '
                        f'&mdash; overlays show <em>why</em> (off-screen wrists, kinematic '
                        f'jumps, camera drift).</div>'
                        f'<div class="grid">{"".join(_card_html(c) for c in dropped)}</div>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{esc(ds)} — filtered data viewer</title><style>{CSS}</style></head><body>'
            f'<h1>{esc(ds)}</h1><div class="sub"><a href="{AUTH_BASE}/index.html">&larr; all datasets</a>'
            f' &middot; {CATEGORY.get(ds, "")} &middot; {len(kept)} kept instances</div>'
            f'<div class="legend"><span><span class="dotl"></span>left hand</span>'
            f'<span><span class="dotr"></span>right hand</span></div>'
            f'<div class="stats">{chips}</div><div class="grid">{body}</div>'
            f'{dropped_html}</body></html>')


def root_html(rows):
    trs = "".join(
        f'<tr><td><a href="{AUTH_BASE}/{esc(ds)}/index.html">{esc(ds)}</a></td><td>{esc(cat)}</td>'
        f'<td class="num">{n}</td><td class="num">{ov}</td><td>{esc(extra)}</td></tr>'
        for ds, cat, n, ov, extra in rows)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>EgoSmith filtered data viewer</title><style>{CSS}</style></head><body>'
            f'<h1>EgoSmith filtered data viewer</h1>'
            f'<div class="sub">Sampled instances per dataset: video &middot; reconstructed '
            f'keypoints overlaid &middot; LLM annotation (L1&ndash;L4)</div>'
            f'<table><tr><th>dataset</th><th>category</th><th>cards</th><th>with overlay</th>'
            f'<th>dataset stats</th></tr>{trs}</table></body></html>')


def dataset_stats(ds, ad, cards):
    stats = [("cards", len(cards)),
             ("with overlay", sum(1 for c in cards if c["status"] == "overlay"))]
    try:
        if ad and ad.get("funnel") and fs.exists(ad["funnel"]):
            fn = json.loads(fs.open(ad["funnel"], "rb").read())
            stats += [("kept clips", f"{fn['final_kept_clips']:,}"),
                      ("kept hours", f"{fn['final_hours']:,}")]
        elif ad and ad["kind"] in ("cat3", "native", "hot3d_viz"):
            rep = f"{BUCKET}/egosmith_filtered/{ds}/filter_run/filter_report.json"
            if fs.exists(rep):
                r = json.loads(fs.open(rep, "rb").read())
                if r.get("kept_clips"):
                    stats.append(("kept clips", f"{r['kept_clips']:,}"))
    except Exception:  # noqa: BLE001
        pass
    return stats


def publish(ds, work, upload_clips=True):
    wdir = work / ds
    cards = json.loads((wdir / "cards.json").read_text())
    for c in cards:
        mp4 = wdir / c["video"]
        if mp4.exists():
            c["video_uri"] = _embed_uri(mp4)
            c["raw_url"] = f"{AUTH_BASE}/{ds}/{c['video']}"
    stats = dataset_stats(ds, ADAPTERS.get(ds), cards)
    page = wdir / "index.html"
    page.write_text(dataset_html(ds, cards, stats))
    print(f"[{ds}] page size {page.stat().st_size/1e6:.1f} MB", flush=True)
    dst = f"gs://{VIEWER}/{ds}"
    subprocess.run(["gcloud", "storage", "cp", "--content-type=text/html", "-q",
                    str(page), f"{dst}/index.html"], check=True)
    if upload_clips:
        subprocess.run(["gcloud", "storage", "cp", "--content-type=video/mp4", "-q"]
                       + [str(p) for p in sorted((wdir / "clips").glob("*.mp4"))]
                       + [f"{dst}/clips/"], check=True)
        subprocess.run(["gcloud", "storage", "cp", "-q", str(wdir / "cards.json"),
                        f"{dst}/cards.json"], check=True)
    print(f"[{ds}] published -> https://storage.cloud.google.com/{VIEWER}/{ds}/index.html",
          flush=True)


def build_root(work):
    rows = []
    for p in sorted(fs.ls(VIEWER)):
        ds = os.path.basename(p.rstrip("/"))
        if ds.startswith("_") or ds.endswith(".html"):
            continue
        try:
            cards = json.loads(fs.open(f"{VIEWER}/{ds}/cards.json", "rb").read())
        except Exception:  # noqa: BLE001
            continue
        ov = sum(1 for c in cards if c["status"] == "overlay")
        stats = dataset_stats(ds, ADAPTERS.get(ds), cards)
        extra = " · ".join(f"{k}: {v}" for k, v in stats[2:])
        rows.append((ds, CATEGORY.get(ds, "—"), len(cards), ov, extra))
    out = work / "index.html"
    out.write_text(root_html(rows))
    subprocess.run(["gcloud", "storage", "cp", "--content-type=text/html", "-q",
                    str(out), f"gs://{VIEWER}/index.html"], check=True)
    print(f"root -> https://storage.cloud.google.com/{VIEWER}/index.html", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--work", default=os.environ.get("VIEWER_WORK", "/root/viewer_work"))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--root", action="store_true")
    args = ap.parse_args()
    work = Path(args.work)
    for ds in args.datasets:
        if args.render:
            render_dataset(ds, args.n, args.seed, work)
        if args.publish:
            publish(ds, work)
    if args.root:
        build_root(work)


if __name__ == "__main__":
    main()
