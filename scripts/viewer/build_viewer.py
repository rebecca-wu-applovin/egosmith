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
def _ego_adapter(ds, **kw):
    """Standard egocentric layout (egocentric10k-style): sharded filter + v4
    annotations + frames tars + HaWoR recon outputs."""
    return dict(kind="ego", fps=15.0,
                filt=f"{BUCKET}/egosmith_filtered/{ds}/filter_run/_shards",
                ann=f"{BUCKET}/egosmith_filtered/{ds}/filter_run/annotations_v4/_shards",
                frames=f"{BUCKET}/egosmith_filtered/{ds}/frames",
                recon=f"{BUCKET}/egosmith_recon/{ds}/recon/outputs", **kw)


def _native_sharded_adapter(ds, fps=30.0, **kw):
    """Native-GT dataset in the sharded ego-style layout: per-shard filtered manifests +
    per-shard v4 annotations + frames/shard_XXXXX/<cid>.tar; no recon outputs (overlay
    comes from the in-tar lowdim). Dropped examples come from the stage-4 drop reports
    under native/_shards/reports/."""
    return dict(kind="native", fps=fps,
                filt=f"{BUCKET}/egosmith_filtered/{ds}/filter_run/_shards",
                ann=f"{BUCKET}/egosmith_filtered/{ds}/filter_run/annotations_v4/_shards",
                frames=f"{BUCKET}/egosmith_filtered/{ds}/frames",
                native=f"{BUCKET}/egosmith_filtered/{ds}/native/_shards", **kw)


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
    # Cat-1, fully processed (recon + presence-gated Layer-4 + v4 annotations);
    # exact egocentric10k layout
    "hd_epic": _ego_adapter("hd_epic"),
    "assembly101": _ego_adapter("assembly101"),
    "epic_kitchens_100": _ego_adapter("epic_kitchens_100"),
    "holoassist": _ego_adapter("holoassist"),
    "ego4d": _ego_adapter("ego4d"),
    # egoverse_aria v1 REPLACED by egoverse_aria_v2 (see egosmith_filtered/
    # egoverse_aria/REPLACED.md): full Layer-1-first rerun, Stage-1 interval trim,
    # fresh recon; 6,802 kept sub-clips / 7.71 h post presence re-filter
    "egoverse_aria_v2": _ego_adapter("egoverse_aria_v2"),
    # egoexo4d: standard ego conveyor (aria RGB, kb4 undistort), 86/86 shards;
    # 16,516 kept / 39.81 h; 16,139 annotation rows merged pre-hold (the 377
    # unannotated keeps stay unlabeled until the labeling hold lifts)
    "egoexo4d": _ego_adapter("egoexo4d"),
    # GigaHands: EasyMocap bimanual MANO GT, use_gt mode (same render path as
    # taco/h2o); native 30 fps; annotations: NONE — published under the global
    # labeling hold, cards render "annotation: null".
    # 2026-08-28 ego trim: kept = first-person (head-position rig slot) clips only,
    # per the (scene, camera-slot) audit (filter_run/ego_audit_20260828/); dropped
    # non_egocentric_view tars live under _dropped_egotrim_20260828/frames/ and the
    # dropped examples come from the explicit drop manifest below.
    "gigahands": dict(kind="cat3", fps=30.0, reconvert="gigahands",
                      outputs=f"{BUCKET}/egosmith_recon/gigahands/use_gt/outputs",
                      dropped_manifest=f"{BUCKET}/egosmith_filtered/gigahands/filter_run/"
                                       "clip_manifest.filtered.egotrimdropped.jsonl",
                      dropped_frames=f"{BUCKET}/egosmith_filtered/gigahands/"
                                     "_dropped_egotrim_20260828/frames",
                      dropped_reason="non_egocentric_view"),
    # HaMeR-seeded recon (gloved hands); sharded ego layout + funnel stats
    "egotouch": _ego_adapter(
        "egotouch",
        funnel=f"{BUCKET}/egosmith_filtered/egotouch/filter_run/funnel.json"),
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
    "taco": dict(kind="cat3", fps=30.0, outputs=f"{BUCKET}/egosmith_recon/taco/use_gt/outputs"),
    "oakink_actions": dict(kind="cat3", fps=30.0,
                           outputs=f"{BUCKET}/egosmith_recon/oakink_actions/use_gt/outputs"),
    # Cat-3, GT-mode recon under egosmith_recon/<ds>/use_gt/outputs/
    "arctic": dict(kind="cat3", fps=30.0,
                   outputs=f"{BUCKET}/egosmith_recon/arctic/use_gt/outputs"),
    # Cat-2 glove GT -> MANO fit; recon colocated under egosmith_filtered/dexcap/outputs/
    "dexcap": dict(kind="cat3", fps=30.0,
                   outputs=f"{BUCKET}/egosmith_filtered/dexcap/outputs"),
    # W9: H2O egocentric cam4, GT MANO+camera (use_gt)
    "h2o": dict(kind="cat3", fps=30.0,
                outputs=f"{BUCKET}/egosmith_recon/h2o/use_gt/outputs"),
    # WIYH: sharded ego layout but SINGLE annotations.v4.jsonl; native 10fps
    "wiyh": dict(kind="ego", fps=10.0,
                 filt=f"{BUCKET}/egosmith_filtered/wiyh/filter_run/_shards",
                 ann_file=f"{BUCKET}/egosmith_filtered/wiyh/filter_run/annotations.v4.jsonl",
                 frames=f"{BUCKET}/egosmith_filtered/wiyh/frames",
                 recon=f"{BUCKET}/egosmith_recon/wiyh/recon/outputs",
                 funnel=f"{BUCKET}/egosmith_filtered/wiyh/filter_run/funnel.json",
                 # keeps are sparse (~14% of 686 shards non-empty): scan them all so
                 # ~50 kept cards materialize (the old shards_k=60 yielded only 8)
                 shards_k=700),
    # HumanTouch: MANUS glove GT + block-anchor extrinsics; cards produced by
    # scripts/viewer/build_humantouch_cards.py (external cards.json, publish-only);
    # single annotations.v4.jsonl like wiyh
    "humantouch": dict(kind="ego", fps=15.0,
                       ann_file=f"{BUCKET}/egosmith_filtered/humantouch/filter_run/annotations.v4.jsonl",
                       frames=f"{BUCKET}/egosmith_filtered/humantouch/frames",
                       funnel=f"{BUCKET}/egosmith_filtered/humantouch/filter_run/funnel.json"),
    # hot3d ships pre-rendered viz/*.overlay.mp4 — copy, don't re-render
    "hot3d": dict(kind="hot3d_viz"),
    # native lowdim (Vision Pro GT); full 21-joint skeletons pulled per clip from the
    # raw EgoDex hdf5s (ranged zip reads, egodex_rawgt.py) — lowdim carries only 6 pts
    "egodex": dict(kind="native", fps=30.0, raw_gt="egodex"),
    # EgoVerse 2026-08-28 ships: freeform = recon conveyor; the rest = native GT.
    # Native datasets use the SHARDED layout (filter_run/_shards + annotations_v4/_shards
    # + frames/shard_XXXXX/) — _native_sharded_adapter joins annotations per shard and
    # sources dropped examples from native/_shards/reports/shard_X/s4.json.
    # fps notes: scale/flagship/lightwheel per-clip fps is uniformly 30.0 (convert.json).
    # microagi episodes are MOSTLY 30fps at the source, but the whole build (funnel
    # kept-hours AND v4 annotation segment times) used the 29fps default — render at 29
    # so segment cuts stay aligned with annotations (worst case 3.4% slow playback).
    "egoverse_mecka_freeform": _ego_adapter("egoverse_mecka_freeform"),
    "egoverse_mecka_flagship": _native_sharded_adapter("egoverse_mecka_flagship"),
    "egoverse_lightwheel": _native_sharded_adapter("egoverse_lightwheel"),
    "egoverse_microagi": _native_sharded_adapter("egoverse_microagi", fps=29.0,
                                                 shards_k=20),
    "egoverse_scale": _native_sharded_adapter("egoverse_scale"),
    # WIYH native tier: 25-joint glove GT via per-session anchor solve; tagged
    # finger_quality=approximate_35_65px (see egosmith_filtered/wiyh_native/FILTER_MODE.txt).
    # Post strict-audit remediation (2026-08-28): manifest = 440 survivors pointing at
    # frames_v2/ (per-frame .gt_joints.npy -> full 25-joint MANUS overlays); dropped
    # examples come from the remediation drop manifest (anchor_rotation_invalid), whose
    # v1 tars now live under _dropped_20260828/frames/.
    "wiyh_native": dict(
        kind="native", fps=10.0,
        dropped_manifest=f"{BUCKET}/egosmith_filtered/wiyh_native/filter_run/"
                         "clip_manifest.filtered.remediationdropped.jsonl",
        dropped_frames=f"{BUCKET}/egosmith_filtered/wiyh_native/_dropped_20260828/frames",
        dropped_reason="anchor_rotation_invalid"),
}

# Datasets dropped from the shipped set (tombstoned, data preserved on the bucket —
# see egosmith_filtered/<ds>/DROPPED.md). The root builder skips them even if stale
# viewer objects resurface.
DROPPED = {
    "assemblyhands",   # user-ordered drop 2026-08-26: data-quality concerns
    "egoverse_aria",   # v1 REPLACED by egoverse_aria_v2 2026-08-27 (REPLACED.md)
}
CATEGORY = {"egocentric100k": "Egocentric (recon)", "egocentric10k": "Egocentric (recon)",
            "dexycb": "Cat-3 GT", "ho3d_v3": "Cat-3 GT", "show3d": "Cat-3 GT",
            "hoi4d": "Cat-3 GT", "taco": "Cat-3 GT", "oakink_actions": "Cat-3 GT",
            "hot3d": "Cat-3 GT", "arctic": "Cat-3 GT", "h2o": "Cat-3 GT",
            "gigahands": "Cat-3 GT",
            "egodex": "Cat-2 native GT (Vision Pro 21-joint)",
            "dexcap": "Cat-2 GT (glove→MANO)",
            "humantouch": "Cat-2 GT (MANUS glove→MANO, anchor extrinsics)",
            # Cat-4 robot datasets (video-only cards, external adapter)
            "trex": "Cat-4 robot", "dexora": "Cat-4 robot", "dexwild": "Cat-4 robot",
            "hrdexdb_allegro": "Cat-4 robot", "realdex": "Cat-4 robot",
            # Cat-1, fully processed (recon overlays + v4 annotations)
            "assembly101": "Cat-1 (recon)", "hd_epic": "Cat-1 (recon)",
            "holoassist": "Cat-1 (recon)", "ego4d": "Cat-1 (recon)",
            "epic_kitchens_100": "Cat-1 (recon)", "egoverse_aria_v2": "Cat-1 (recon)",
            "egoexo4d": "Cat-1 (recon)",
            "egoverse_mecka_freeform": "Cat-1 (recon)",
            "egoverse_mecka_flagship": "EgoVerse native GT (21-joint)",
            "egoverse_lightwheel": "EgoVerse native GT (21-joint)",
            "egoverse_microagi": "EgoVerse native GT (21-joint)",
            "egoverse_scale": "EgoVerse native GT (21-joint)",
            "egotouch": "Cat-1 (seeded recon, gloved hands)",
            "wiyh": "Cat-2 (recon, exoskeleton hands — 0.37% keep)",
            "wiyh_native": "Cat-2 native GT (25-joint glove, approx 35-65px fingers)"}


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
    ann_all = None
    if ad.get("ann_file"):
        ann_all = {r["clip_id"]: r for r in _read_jsonl_gcs(ad["ann_file"])}
        ann_shards = None
    else:
        ann_shards = {os.path.basename(p).split(".")[0] for p in fs.ls(ad["ann"])
                      if p.endswith(".annotations.jsonl")}
    filt_shards = sorted(os.path.basename(p).split(".")[0] for p in fs.ls(ad["filt"])
                         if p.endswith(".filtered.jsonl"))
    ready = filt_shards if ann_shards is None else [s for s in filt_shards if s in ann_shards]
    if not ready:
        raise RuntimeError(f"{ds}: no shards with both filter + annotations")
    picked = rng.sample(ready, min(shards_k, len(ready)))
    per = max(1, -(-n // len(picked)))  # ceil: floor under-fills when n % shards != 0
    out = []
    for sh in picked:
        recs = _read_jsonl_gcs(f"{ad['filt']}/{sh}.filtered.jsonl")
        anns = ann_all if ann_all is not None else \
            {r["clip_id"]: r for r in _read_jsonl_gcs(f"{ad['ann']}/{sh}.annotations.jsonl")}
        joined = [r for r in recs if r["clip_id"] in anns]
        for r in rng.sample(joined, min(per, len(joined))):
            # shard suffix from root_dir when it carries one (ego conveyor layout);
            # fall back to the filter-shard name (egoverse freeform's root_dir is a
            # local scratch path like .../mkf_shard0 with no 5-digit suffix)
            m = re.search(r"(\d{5})", r["descriptor"]["root_dir"])
            sfx = m.group(1) if m else sh.replace("shard_", "")
            out.append(dict(rec=r, ann=anns[r["clip_id"]], shard=sfx))
    return out[:n]


PURGE_TAG = "presence re-filter"


def sample_dropped_ego(ds, ad, n, seed):
    """Sample DROPPED clips (with reasons) from per-shard filter reports; descriptors
    come from the phaseB manifest (dropped clips are absent from filtered.jsonl).

    Also surfaces the 2026-08-27 presence re-filter purge (empty_pose /
    single_valid_hand), recorded per shard under report['presence_refilter']
    ['clip_ids']: up to 2 examples per purge reason, on top of the n quality drops.
    Purged clips render through the normal path — their frames tars and recon
    outputs were never deleted, only the manifest/annotation rows were pruned."""
    rng = random.Random(seed + 1)
    pb = ad["filt"].replace("/filter_run/_shards", "/phaseB/_shards")
    reports = [p for p in fs.ls(ad["filt"]) if p.endswith(".report.json")]
    out, seen_reasons = [], {}
    purge_left = {"empty_pose": 2, "single_valid_hand": 2}
    n_quality = 0
    for rp in rng.sample(reports, min(12, len(reports))):
        if n_quality >= n and not any(purge_left.values()):
            break
        rep = json.loads(fs.open(rp, "rb").read())
        sfx = os.path.basename(rp).split(".")[0].replace("shard_", "")
        picked = []
        if n_quality < n:
            dropped = [d for d in rep.get("dropped", []) if d.get("drop_category") == "quality"]
            rng.shuffle(dropped)
            for d in dropped:  # spread across distinct primary reasons
                key = (d.get("reasons") or ["?"])[0]
                if seen_reasons.get(key, 0) >= max(2, n // 5):
                    continue
                seen_reasons[key] = seen_reasons.get(key, 0) + 1
                picked.append(d)
                if len(picked) >= 2:
                    break
        pr = (rep.get("presence_refilter") or {}).get("clip_ids") or {}
        for reason in ("empty_pose", "single_valid_hand"):
            ids = pr.get(reason) or []
            take = min(purge_left.get(reason, 0), len(ids), 1)  # <=1/shard: spread out
            for cid in rng.sample(ids, take):
                picked.append({"clip_id": cid, "reasons": [f"{reason} ({PURGE_TAG})"]})
                purge_left[reason] -= 1
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
        n_quality = sum(1 for o in out
                        if not any(PURGE_TAG in x for x in o["reasons"]))
    if any(purge_left.values()):
        # sparse purges (e.g. wiyh: 13 clips over 686 shards) rarely land in the 12
        # sampled reports — top up from the reconciliation file, which indexes the
        # purged clip_ids per shard
        rec_path = ad["filt"].replace("/_shards", "/presence_refilter_reconciliation.json")
        if fs.exists(rec_path):
            shards_w = [sh for sh in
                        json.loads(fs.open(rec_path, "rb").read()).get("shards", [])
                        if any((sh.get("clip_ids") or {}).get(r) for r in purge_left)]
            rng.shuffle(shards_w)
            for sh in shards_w[:6]:
                if not any(purge_left.values()):
                    break
                sfx = sh["shard"]
                by_id = {}
                for reason in ("empty_pose", "single_valid_hand"):
                    ids = (sh.get("clip_ids") or {}).get(reason) or []
                    for cid in rng.sample(ids, min(purge_left.get(reason, 0), len(ids), 1)):
                        by_id[cid] = [f"{reason} ({PURGE_TAG})"]
                        purge_left[reason] -= 1
                if not by_id:
                    continue
                for line in fs.open(f"{pb}/shard_{sfx}.manifest.jsonl",
                                    "rb").read().decode().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["clip_id"] in by_id:
                        out.append(dict(rec=r, ann=None, shard=sfx,
                                        reasons=by_id[r["clip_id"]]))
    return out[:n + 4]


def sample_dropped_native(ds, ad, n, seed):
    """Dropped examples for sharded native-GT datasets: stage-4 drop reports under
    native/_shards/reports/shard_X/s4.json carry {clip_id, reasons}; descriptors come
    from the per-shard full manifest. The bucket mirrors KEPT frames tars only, so each
    sample also carries its source-episode ref (from the shard convert.json) for
    on-demand local re-conversion at fetch time."""
    rng = random.Random(seed + 1)
    shard_dirs = [p for p in fs.ls(f"{ad['native']}/reports") if "shard_" in p]
    out, seen_reasons = [], {}
    for sd in rng.sample(shard_dirs, min(12, len(shard_dirs))):
        if len(out) >= n:
            break
        sd = sd.rstrip("/")
        sh = os.path.basename(sd)                      # shard_00000
        sfx = sh.replace("shard_", "")
        rp = f"{sd}/s4.json"
        if not fs.exists(rp):
            continue
        rep = json.loads(fs.open(rp, "rb").read())
        dropped = [d for d in rep.get("dropped", [])
                   if d.get("drop_category", "quality") == "quality"]
        rng.shuffle(dropped)
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
        conv = json.loads(fs.open(f"{sd}/convert.json", "rb").read())
        ep_by_id = {r["clip_id"]: r.get("episode") for r in conv.get("results", [])
                    if r.get("clip_id") in by_id}
        iv = f"{sd}/intervals.json"
        iv_gcs = iv if fs.exists(iv) else None         # stage-1 gated conversions
        mpath = f"{ad['native']}/{sh}.full.manifest.jsonl"
        if not fs.exists(mpath):
            mpath = f"{ad['native']}/{sh}.manifest.jsonl"
        for line in fs.open(mpath, "rb").read().decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["clip_id"] in by_id:
                out.append(dict(rec=r, ann=None, shard=sfx,
                                reasons=by_id[r["clip_id"]].get("reasons", []),
                                ep_ref=ep_by_id.get(r["clip_id"]), intervals_gcs=iv_gcs))
    return out[:n]


def _reconvert_native_tar(ds, s, tar_local, wdir):
    """Re-convert one source episode with the production converter params
    (segment_sec=10, default min_segment_sec) and pull out the target sub-clip tar.
    Validates the reproduced segmentation against the manifest's frame count."""
    import shutil
    cid = s["rec"]["clip_id"]
    if not s.get("ep_ref"):
        raise FileNotFoundError(f"{cid}: no episode ref for reconvert")
    tmp = wdir / "_reconv" / cid
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(_REPO / "scripts" / "build" / "generate_keypoints_wds.py"),
           "--spec", str(_REPO / "configs" / "keypoint_specs" / f"{ds}.yaml"),
           "--frames_root", str(tmp / "frames"), "--outputs_root", str(tmp / "outputs"),
           "--manifest_out", str(tmp / "manifest.jsonl"),
           "--report_out", str(tmp / "convert.json"),
           "--workers", "1", "--segment_sec", "10.0", "--episodes", s["ep_ref"]]
    if s.get("intervals_gcs"):
        fs.get(s["intervals_gcs"], str(tmp / "intervals.json"))
        cmd += ["--intervals_json", str(tmp / "intervals.json")]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    src = tmp / "frames" / f"{cid}.tar"
    if not src.exists():
        raise FileNotFoundError(f"reconvert produced no {cid}.tar")
    want = len(s["rec"]["descriptor"].get("frame_names") or [])
    if want:
        import tarfile
        with tarfile.open(src) as tf:
            got = sum(1 for m in tf.getnames() if m.endswith(".image.jpg"))
        if got != want:
            raise RuntimeError(f"reconvert segmentation drift: {got} frames != {want}")
    shutil.move(str(src), str(tar_local))
    shutil.rmtree(tmp, ignore_errors=True)
    return tar_local


def _reconvert_gigahands_tar(s, tar_local, wdir):
    """GigaHands frames/ mirrors KEPT tars only, so dropped-example tars are
    re-converted on demand from the raw capture on the bucket (the converter streams
    per-sequence via --include <scene>/<seq>). use_gt outputs for dropped clips still
    exist on the bucket, so the overlay renders through the normal cat3 path."""
    import shutil
    cid = s["rec"]["clip_id"]
    ex = s["rec"]["descriptor"].get("extra") or {}
    scene, seq = ex.get("scene"), ex.get("gigahands_seq")
    if not scene or seq is None:
        raise FileNotFoundError(f"{cid}: no scene/seq in descriptor.extra")
    tmp = wdir / "_reconv" / cid
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable,
           str(_REPO / "scripts" / "build" / "generate_gigahands_world_res.py"),
           "--frames_root", str(tmp / "frames"), "--outputs_root", str(tmp / "outputs"),
           "--manifest_out", str(tmp / "manifest.jsonl"),
           "--cache_dir", str(tmp / "_cache"), "--work_dir", str(tmp / "_work"),
           "--workers", "1", "--include", f"{scene}/{seq}"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
    src = tmp / "frames" / f"{cid}.tar"
    if not src.exists():
        raise FileNotFoundError(f"reconvert produced no {cid}.tar")
    want = len(s["rec"]["descriptor"].get("frame_names") or [])
    if want:
        import tarfile
        with tarfile.open(src) as tf:
            got = sum(1 for m in tf.getnames() if m.endswith(".image.jpg"))
        if got != want:
            raise RuntimeError(f"reconvert frame drift: {got} != {want}")
    shutil.move(str(src), str(tar_local))
    shutil.rmtree(tmp, ignore_errors=True)
    return tar_local


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


def sample_dropped_manifest(ds, ad, n, seed):
    """Dropped examples from an explicit drop manifest (wiyh_native strict-audit
    remediation): full records whose tars were moved to ad['dropped_frames']."""
    rng = random.Random(seed + 1)
    recs = _read_jsonl_gcs(ad["dropped_manifest"])
    out = []
    for r in rng.sample(recs, min(n, len(recs))):
        d = r["descriptor"]
        d["shard_path"] = f"gs://{ad['dropped_frames']}/{os.path.basename(d['shard_path'])}"
        out.append(dict(rec=r, ann=None, shard=None, reasons=[ad["dropped_reason"]]))
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
    if "filt" in ad:  # sharded layout (ego conveyor OR sharded native GT)
        samples = sample_ego(ds, ad, n, seed, shards_k=ad.get("shards_k", 10))
        dropped = (sample_dropped_native(ds, ad, n_dropped, seed) if "native" in ad
                   else sample_dropped_ego(ds, ad, n_dropped, seed))
    else:
        samples = sample_cat3(ds, ad, n, seed)
        dropped = (sample_dropped_manifest(ds, ad, n_dropped, seed)
                   if "dropped_manifest" in ad
                   else sample_dropped_cat3(ds, ad, n_dropped, seed))
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
            sp = d.get("shard_path") or ""
            # absolute gs:// shard_path (e.g. sample_dropped_manifest rewrites dropped
            # tars to _dropped_*/frames/) wins over the kept-frames mirror
            gcs_tar = (sp.replace("gs://", "") if sp.startswith("gs://")
                       else f"{BUCKET}/egosmith_filtered/{ds}/frames/{os.path.basename(sp)}")
            if fs.exists(gcs_tar):
                fs.get(gcs_tar, str(tar_local))
            elif ad.get("reconvert") == "gigahands":  # kept-only mirror -> re-convert
                _reconvert_gigahands_tar(s, tar_local, wdir)
            else:
                fs.get(gcs_tar, str(tar_local))  # surface the original error
            seq = materialize_seq(f"{ad['outputs']}/{cid}", wdir / "_seq" / cid)
        elif "filt" in ad:  # sharded native GT: frames[_v2]/shard_XXXXX/<cid>.tar
            # kept clips: follow the manifest's own shard_path when absolute (tracks
            # the gt_joints retrofit flip to frames_v2/); dropped clips carry local
            # converter paths -> fall back to the frames mirror, then re-convert
            sp = d.get("shard_path") or ""
            gcs_tar = (sp.replace("gs://", "") if sp.startswith("gs://")
                       else f"{ad['frames']}/shard_{s['shard']}/{cid}.tar")
            if fs.exists(gcs_tar):
                fs.get(gcs_tar, str(tar_local))
            else:  # dropped clips: kept-only mirror -> re-convert from source episode
                _reconvert_native_tar(ds, s, tar_local, wdir)
            seq = None
        else:  # flat native (egodex/wiyh_native): descriptor shard_path (gs://) when
            # absolute — follows manifest flips to frames_v2/ and the wiyh_native
            # remediation-dropped tars; frames/<basename> fallback otherwise
            sp = d["shard_path"]
            if sp.startswith("gs://"):
                fs.get(sp.replace("gs://", ""), str(tar_local))
            else:
                fs.get(f"{BUCKET}/egosmith_filtered/{ds}/frames/{os.path.basename(sp)}",
                       str(tar_local))
            seq = None
        return tar_local, seq

    rawgt = None
    if ad.get("raw_gt") == "egodex":
        from egodex_rawgt import EgoDexRawGT
        rawgt = EgoDexRawGT(fs=fs)

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
        gtj, gt_err = None, None
        if rawgt is not None:
            try:
                gtj = rawgt.joints(cid)
            except Exception as e:  # noqa: BLE001
                gt_err = f"raw GT unavailable: {type(e).__name__}: {str(e)[:120]}"
        try:
            # one retry: transient ffmpeg broken-pipes under batch memory pressure
            for attempt in (0, 1):
                try:
                    if ad["kind"] == "native":
                        st = render_native_overlay(tar_local, out, fps, gt_joints=gtj)
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
        if gt_err and not st.get("note"):
            st["note"] = gt_err + " -> 6-pt overlay"
        cards.append(dict(clip_id=cid, video=f"clips/{cid}.mp4", status=st["status"],
                          note=st.get("note"), section=s.get("section", "kept"),
                          reasons=s.get("reasons", []),
                          meta={k: st.get(k) for k in
                                ("dur_s", "fps", "w", "h", "hand_pct", "gt_align_px")},
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
.segblock{margin:.6rem 0;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.segblock video{width:100%;display:block;background:#000}
.segblock .seg{border-radius:0}
.seg.seekable{cursor:pointer;border:1px solid transparent}
.seg.seekable:hover{border-color:var(--acc)}
.seg.playing{border-color:var(--acc);background:#25314a}
.play{color:var(--acc);font-weight:600}
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


EMBED_CAP_SEC = 8


def _segment_uri(mp4_path, start, end):
    """data:video/mp4 URI for one annotation segment, cut+re-encoded from the card mp4
    (360p cap, crf 28 — segments partition the clip, so total page weight stays flat)."""
    import base64
    import tempfile
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
        tmp = t.name
    dur = max(0.3, end - start)
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
                    "-i", str(mp4_path), "-vf", "scale=-2:'min(360,ih)'", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart",
                    "-an", tmp], check=True, capture_output=True)
    data = Path(tmp).read_bytes()
    os.unlink(tmp)
    if not data:
        raise RuntimeError("empty segment cut")
    return "data:video/mp4;base64," + base64.b64encode(data).decode()


def _embed_uri(mp4_path):
    """(data:video/mp4 URI, truncated: bool) — big clips re-encode to an 8s/288p preview."""
    import base64
    import tempfile
    p = Path(mp4_path)
    data = p.read_bytes()
    truncated = False
    if len(data) > EMBED_MAX_BYTES:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as t:
            tmp = t.name
        subprocess.run([ff, "-y", "-loglevel", "error", "-t", str(EMBED_CAP_SEC),
                        "-i", str(p),
                        "-vf", "scale=-2:'min(288,ih)'", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "33", "-movflags", "+faststart",
                        "-an", tmp], check=True, capture_output=True)
        data = Path(tmp).read_bytes()
        os.unlink(tmp)
        truncated = True
    return "data:video/mp4;base64," + base64.b64encode(data).decode(), truncated


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
            rng, seek = "", ""
            if sg.get("end") is not None:
                s0, s1 = sg.get("start", 0), sg["end"]
                seek = f' data-start="{s0:g}" data-end="{s1:g}"'
                rng = (f'<div class="rng"><span class="play">&#9654; play segment</span> '
                       f'{s0:g}&ndash;{s1:g}s{qb}</div>')
            lv = sg.get("language_instructions") or {}
            rows = "".join(f'<div class="lv"><span class="lab">L{i}</span><span>{esc(lv[k])}</span></div>'
                           for i, k in enumerate(("level1", "level2", "level3", "level4"), 1)
                           if lv.get(k))
            segs_html += f'<div class="seg{" seekable" if seek else ""}"{seek}>{rng}{rows}</div>'
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
    if c.get("embed_truncated"):
        dur = (c.get("meta") or {}).get("dur_s")
        raw = (f' &middot; <span class="badge b-bad">preview: first {EMBED_CAP_SEC}s'
               + (f" of {dur:g}s" if dur else "") + "</span>"
               + f' &middot; <a href="{esc(c["raw_url"])}">full video &nearr;</a>')
    if c.get("seg_videos"):
        blocks = ""
        for i, sv in enumerate(c["seg_videos"], 1):
            sg = sv["seg"]
            q = sg.get("is_good_quality")
            qb = ('<span class="badge b-ok">good</span>' if q
                  else '<span class="badge b-bad">flagged</span>' if q is False else "")
            lv = sg.get("language_instructions") or {}
            rows = "".join(
                f'<div class="lv"><span class="lab">L{j}</span><span>{esc(lv[k])}</span></div>'
                for j, k in enumerate(("level1", "level2", "level3", "level4"), 1) if lv.get(k))
            blocks += (f'<div class="segblock"><video controls muted playsinline '
                       f'preload="metadata" src="{sv["uri"]}"></video>'
                       f'<div class="seg"><div class="rng">segment {i} &middot; '
                       f'{sg.get("start", 0):g}&ndash;{sg["end"]:g}s{qb}</div>{rows}</div></div>')
        return (f'<figure class="card"><div class="body">'
                f'<div class="cid">{esc(c["clip_id"])}{badge}</div>'
                f'<div class="meta">{" &middot; ".join(bits)}'
                f' &middot; <a href="{esc(c["raw_url"])}">full clip &nearr;</a></div>'
                f'{note}{blocks}</div></figure>')
    return (f'<figure class="card"><video controls loop muted playsinline preload="metadata" '
            f'src="{src}"></video><div class="body">'
            f'<div class="cid">{esc(c["clip_id"])}{badge}</div>'
            f'<div class="meta">{" &middot; ".join(bits)}{raw}</div>{note}{segs_html}</div></figure>')


SEG_JS = """
document.querySelectorAll('.seg.seekable').forEach(function(seg){
  seg.addEventListener('click', function(){
    var card = seg.closest('.card'); if(!card) return;
    var v = card.querySelector('video'); if(!v) return;
    var s = parseFloat(seg.dataset.start), e = parseFloat(seg.dataset.end);
    card.querySelectorAll('.seg.playing').forEach(function(x){x.classList.remove('playing')});
    seg.classList.add('playing');
    v.currentTime = s; v.loop = false; v.play();
    if (v._segHandler) v.removeEventListener('timeupdate', v._segHandler);
    v._segHandler = function(){
      if (v.currentTime >= e){ v.pause(); v.removeEventListener('timeupdate', v._segHandler);
        v._segHandler = null; seg.classList.remove('playing'); }
    };
    v.addEventListener('timeupdate', v._segHandler);
  });
});
"""


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
            f'{dropped_html}<script>{SEG_JS}</script></body></html>')


def root_html(rows, totals=None):
    trs = "".join(
        f'<tr><td><a href="{AUTH_BASE}/{esc(ds)}/index.html">{esc(ds)}</a></td><td>{esc(cat)}</td>'
        f'<td class="num"><b>{esc(hours)}</b></td>'
        f'<td class="num">{n}</td><td class="num">{ov}</td></tr>'
        for ds, cat, hours, n, ov in rows)
    if totals:
        th, tn, tov = totals
        trs += (f'<tr><td><b>total</b> ({len(rows)} datasets)</td><td></td>'
                f'<td class="num"><b>{th:,.0f}</b></td>'
                f'<td class="num"><b>{tn}</b></td><td class="num"><b>{tov}</b></td></tr>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>EgoSmith filtered data viewer</title><style>{CSS}</style></head><body>'
            f'<h1>EgoSmith filtered data viewer</h1>'
            f'<div class="sub">Sampled instances per dataset: video &middot; reconstructed '
            f'keypoints overlaid &middot; LLM annotation (L1&ndash;L4)</div>'
            f'<table><tr><th>dataset</th><th>category</th><th>kept hours</th><th>cards</th>'
            f'<th>with overlay</th></tr>{trs}</table></body></html>')


# Kept hours per dataset (bucket-audited; funnel.json overrides when present).
# Sources: funnels (100k/10k/egotouch/wiyh), the Cat-1 conveyor completion report,
# per-manifest frame/fps sums (Cat-3, robots from QC reports).
KEPT_HOURS = {
    # 2026-08-28 refresh: post presence-re-filter values. 100k/10k/egotouch/wiyh
    # come from refreshed funnel.json (funnel overrides these anyway); the Cat-1
    # conveyor datasets have no funnel.json — values below = pre-purge hours minus
    # presence_refilter_reconciliation.json removed_hours.
    "egocentric100k": 24687.3, "egocentric10k": 2590.6,
    "ego4d": 323.3, "holoassist": 49.9, "epic_kitchens_100": 26.6,
    "assembly101": 23.1, "hd_epic": 3.4, "egoverse_aria_v2": 7.71,
    "egodex": 464.6, "egotouch": 5.0, "wiyh": 2.01,
    "gigahands": 4.24, "egoexo4d": 39.81,
    # post-remediation (ab99d40): funnel hours_kept — 50,475 clips / 125.2 h
    "humantouch": 125.2,
    "dexycb": 4.36, "show3d": 24.82, "hoi4d": 3.59, "taco": 2.72,
    "ho3d_v3": 0.68, "hot3d": 0.5, "oakink_actions": 6.63,
    "arctic": 1.61, "dexcap": 0.79, "h2o": 0.85,
    "trex": 44.7, "dexora": 39.75, "dexwild": 5.22,
    "hrdexdb_allegro": 0.32, "realdex": 1.16,
    # EgoVerse 2026-08-28 ships (funnel.json values; funnel overrides live anyway)
    "egoverse_microagi": 1658.18, "egoverse_mecka_freeform": 785.91,
    "egoverse_scale": 140.83, "egoverse_mecka_flagship": 97.64,
    "egoverse_lightwheel": 43.49,
}


def dataset_kept_hours(ds, ad):
    try:
        fp = (ad or {}).get("funnel") or f"{BUCKET}/egosmith_filtered/{ds}/filter_run/funnel.json"
        if fs.exists(fp):
            fn = json.loads(fs.open(fp, "rb").read())
            # hours_kept: humantouch remediation funnel spelling (2026-08-28)
            h = fn.get("final_hours", fn.get("final_kept_hours",
                fn.get("kept_hours", fn.get("hours_kept"))))
            if h:
                return float(h)
    except Exception:  # noqa: BLE001
        pass
    return KEPT_HOURS.get(ds)


def dataset_stats(ds, ad, cards):
    stats = []
    h = dataset_kept_hours(ds, ad)
    if h is not None:
        stats.append(("kept hours", f"{h:,.2f}" if h < 100 else f"{h:,.0f}"))
    stats += [("cards", len(cards)),
              ("with overlay", sum(1 for c in cards if c["status"] == "overlay"))]
    return stats


def publish(ds, work, upload_clips=True):
    wdir = work / ds
    cards = json.loads((wdir / "cards.json").read_text())
    for c in cards:
        mp4 = wdir / c["video"]
        if mp4.exists():
            c["raw_url"] = f"{AUTH_BASE}/{ds}/{c['video']}"
            segs = ((c.get("annotation") or {}).get("segments")) or []
            timed = [sg for sg in segs if sg.get("end") is not None]
            if timed:
                # one video per annotated segment, annotation rendered beneath each
                c["seg_videos"] = []
                for sg in timed:
                    s0 = float(sg.get("start", 0)); s1 = float(sg["end"])
                    try:
                        uri = _segment_uri(mp4, s0, s1)
                    except Exception:  # noqa: BLE001  (corrupt cut -> whole-clip card)
                        c["seg_videos"] = None
                        break
                    c["seg_videos"].append({"uri": uri, "seg": sg})
            if not c.get("seg_videos"):
                c["video_uri"], c["embed_truncated"] = _embed_uri(mp4)
    stats = dataset_stats(ds, ADAPTERS.get(ds), cards)
    page = wdir / "index.html"
    page.write_text(dataset_html(ds, cards, stats))
    print(f"[{ds}] page size {page.stat().st_size/1e6:.1f} MB", flush=True)
    dst = f"gs://{VIEWER}/{ds}"
    subprocess.run(["gcloud", "storage", "cp", "--content-type=text/html", "--cache-control=no-store", "-q",
                    str(page), f"{dst}/index.html"], check=True)
    if upload_clips:
        subprocess.run(["gcloud", "storage", "cp", "--content-type=video/mp4",
                        "--cache-control=no-store", "-q"]
                       + [str(p) for p in sorted((wdir / "clips").glob("*.mp4"))]
                       + [f"{dst}/clips/"], check=True)
        subprocess.run(["gcloud", "storage", "cp", "--cache-control=no-store", "-q",
                        str(wdir / "cards.json"), f"{dst}/cards.json"], check=True)
    print(f"[{ds}] published -> https://storage.cloud.google.com/{VIEWER}/{ds}/index.html",
          flush=True)


def build_root(work):
    rows, tot_h, tot_n, tot_ov = [], 0.0, 0, 0
    for p in sorted(fs.ls(VIEWER)):
        ds = os.path.basename(p.rstrip("/"))
        if ds.startswith("_") or ds.endswith(".html") or ds in DROPPED:
            continue
        try:
            cards = json.loads(fs.open(f"{VIEWER}/{ds}/cards.json", "rb").read())
        except Exception:  # noqa: BLE001
            continue
        ov = sum(1 for c in cards if c["status"] == "overlay")
        h = dataset_kept_hours(ds, ADAPTERS.get(ds))
        hours = ("—" if h is None else f"{h:,.2f}" if h < 100 else f"{h:,.0f}")
        rows.append((ds, CATEGORY.get(ds, "—"), hours, len(cards), ov))
        tot_h, tot_n, tot_ov = tot_h + (h or 0.0), tot_n + len(cards), tot_ov + ov
    out = work / "index.html"
    out.write_text(root_html(rows, totals=(tot_h, tot_n, tot_ov)))
    subprocess.run(["gcloud", "storage", "cp", "--content-type=text/html", "--cache-control=no-store", "-q",
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
