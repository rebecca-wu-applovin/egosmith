#!/usr/bin/env python
"""Incremental Phase D: quality-filter Egocentric-100K shards as Phase C completes them.

Loop: find shards with a Phase-C done-marker but no filter report yet ->
materialize lightweight seq_folders locally (world_space_res.pth in full; SLAM npz
via RANGED zip-member extraction of only the keys the filter reads — traj/tstamp/
img_focal/img_center/scale, skipping the never-read multi-MB `disps`; an empty-but-
valid tracks_<s>_<e>/ dir whose real contents stay on GCS) -> run the CANONICAL
filter (scripts/build/filter_manifest_by_quality.py --stages infiller) -> upload the
per-shard kept manifest + report -> delete local materialization.

Descriptors get width/height injected (from 2*img_center) so the filter's
image-size fast path avoids needing frame tars.
"""
import argparse
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EGOSMITH_ROOT = os.environ.get("EGOSMITH_ROOT", "/root/egosmith")
sys.path.insert(0, f"{EGOSMITH_ROOT}/src")
import gcsfs
import numpy as np

# Dataset prefixes are env-overridable (defaults = Egocentric-100K, unchanged behavior).
# For other datasets in the same conveyor (e.g. egocentric10k), set all three envs.
GCS_OUT = os.environ.get(
    "PHASED_GCS_OUT", "foundational-research/hoi-dataset/egosmith_recon/egocentric100k/recon/outputs")
GCS_B = os.environ.get(
    "PHASED_GCS_B", "foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/phaseB/_shards")
GCS_FILT = os.environ.get(
    "PHASED_GCS_FILT", "foundational-research/hoi-dataset/egosmith_filtered/egocentric100k/filter_run/_shards")
WORK = Path(os.environ.get("PHASED_WORK", "/root/egosmith_annotations/_phased_work"))
NPZ_KEYS = ["traj", "tstamp", "img_focal", "img_center", "scale"]

# Gate config (env-overridable). Defaults = the shipped 15fps regime (2x the
# 30fps-tuned per-frame steps). For other source fps, scale steps by 30/fps
# (e.g. WIYH 10fps native: SOURCE_FPS=10, steps 3x -> 2.97/0.9/0.9/0.6/2.1).
SOURCE_FPS = os.environ.get("PHASED_SOURCE_FPS", "15")
GATE_WRIST = os.environ.get("PHASED_MAX_WRIST_ROT_STEP", "1.98")
GATE_HAND = os.environ.get("PHASED_MAX_HAND_TRANS_STEP", "0.6")
GATE_FINGER = os.environ.get("PHASED_MAX_FINGER_TRANS_STEP", "0.6")
GATE_CAM_T = os.environ.get("PHASED_MAX_CAM_TRANS_STEP", "0.4")
GATE_CAM_R = os.environ.get("PHASED_MAX_CAM_ROT_STEP", "1.4")

fs = gcsfs.GCSFileSystem()


def ranged_npz_extract(gcs_path: str, keys=NPZ_KEYS) -> dict:
    """Extract selected members from a remote .npz (zip) without downloading it all."""
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
        (comp, csize, usize, nlen, elen, clen) = struct.unpack("<H", cd[pos+10:pos+12])[0], \
            struct.unpack("<I", cd[pos+20:pos+24])[0], struct.unpack("<I", cd[pos+24:pos+28])[0], \
            struct.unpack("<H", cd[pos+28:pos+30])[0], struct.unpack("<H", cd[pos+30:pos+32])[0], \
            struct.unpack("<H", cd[pos+32:pos+34])[0]
        lho = struct.unpack("<I", cd[pos+42:pos+46])[0]
        name = cd[pos+46:pos+46+nlen].decode()
        key = name[:-4] if name.endswith(".npy") else name
        if key in keys:
            lh = fs.read_block(gcs_path, lho, 30)
            lnlen, lelen = struct.unpack("<HH", lh[26:30])
            data = fs.read_block(gcs_path, lho + 30 + lnlen + lelen, csize)
            if comp == 8:
                import zlib
                data = zlib.decompress(data, -15)
            out[key] = np.load(io.BytesIO(data), allow_pickle=False)
        pos += 46 + nlen + elen + clen
    missing = [k for k in keys if k not in out]
    if missing:
        raise ValueError(f"npz members missing: {missing}")
    return out


def materialize_clip(shard_sfx, clip_id, rec, base_dir):
    """Build a minimal local seq_folder the canonical filter accepts."""
    seq = base_dir / clip_id
    seq.mkdir(parents=True, exist_ok=True)
    pre = f"{GCS_OUT}/shard_{shard_sfx}/{clip_id}"
    # SLAM npz: find name (has start/end), extract small members, rewrite locally
    npzs = [p for p in fs.ls(f"{pre}/SLAM") if "hawor_slam_w_scale_" in p]
    if not npzs:
        raise FileNotFoundError("no slam npz")
    npz_name = os.path.basename(npzs[0])
    small = ranged_npz_extract(npzs[0])
    (seq / "SLAM").mkdir(exist_ok=True)
    np.savez(seq / "SLAM" / npz_name, **small)
    # world_space_res.pth in full (small: pose params only)
    fs.get(f"{pre}/world_space_res.pth", str(seq / "world_space_res.pth"))
    # tracks dir: correct range from npz name; contents stay on GCS (validation only
    # needs a non-empty dir with the right name)
    s_e = npz_name.replace("hawor_slam_w_scale_", "").replace(".npz", "")
    tdir = seq / f"tracks_{s_e}"
    tdir.mkdir(exist_ok=True)
    (tdir / ".materialized").write_text("contents on GCS; range from dirname\n")
    # descriptor: point at local seq, inject width/height so no frame read happens,
    # and fps (Phase B left descriptor.fps None -> filter defaulted to 5fps, wrecking
    # the time base of every velocity gate; recon ran at extra.recon_fps = 15)
    d = rec["descriptor"]
    d["seq_folder"] = str(seq)
    cx, cy = small["img_center"].tolist()
    d["width"] = int(round(cx * 2)); d["height"] = int(round(cy * 2))
    d["fps"] = float(d.get("extra", {}).get("recon_fps") or float(SOURCE_FPS))
    return rec


CLAIMS = f"{GCS_FILT}/_claims"
WORKER_ID = os.environ.get("D_WORKER_ID", "box")


def try_claim(sfx: str) -> bool:
    path = f"{CLAIMS}/shard_{sfx}.claim"
    try:
        info = fs.info(path)
        import datetime
        age = time.time() - info.get("mtime", info.get("updated")).timestamp()
        if age < 7200:
            return False   # someone else is on it
    except FileNotFoundError:
        pass
    except Exception:
        pass
    try:
        with fs.open(path, "w") as f:
            f.write(WORKER_ID)
        return True
    except Exception:
        return False


def process_shard(sfx: str, workers: int):
    t0 = time.time()
    man_raw = fs.cat(f"{GCS_B}/shard_{sfx}.manifest.jsonl").decode()
    recs = {json.loads(l)["clip_id"]: json.loads(l) for l in man_raw.splitlines() if l.strip()}
    have = {os.path.basename(os.path.dirname(p))
            for p in fs.glob(f"{GCS_OUT}/shard_{sfx}/*/world_space_res.pth")}
    base = WORK / sfx
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    todo = [(c, recs[c]) for c in have if c in recs]
    ok_rows, n_err = [], 0
    def mat(t):
        c, r = t
        try:
            return materialize_clip(sfx, c, r, base)
        except Exception as e:  # noqa: BLE001
            return {"__err__": f"{c}: {str(e)[:120]}"}
    with ThreadPoolExecutor(workers) as ex:
        for out in ex.map(mat, todo):
            if "__err__" in out:
                n_err += 1
            else:
                ok_rows.append(out)
    in_man = base / "input.jsonl"
    with open(in_man, "w") as f:
        for r in ok_rows:
            f.write(json.dumps(r) + "\n")
    out_man = base / "filtered.jsonl"
    rep = base / "report.json"
    r = subprocess.run(
        [sys.executable, f"{EGOSMITH_ROOT}/scripts/build/filter_manifest_by_quality.py",
         "--input_manifest", str(in_man), "--output_manifest", str(out_man),
         "--report_out", str(rep), "--stages", "infiller", "--workers", "16",
         "--source_fps", SOURCE_FPS, "--target_fps", "30",
         # step gates encode (max velocity x frame interval); tuned on 30fps data,
         # scaled by 30/source_fps for the same physical limit (defaults = 15fps).
         # W7 pilot finding: clips with zero valid poses pass all motion gates
         # trivially (1.2% leak measured on shipped 100K keeps) — require poses
         # present in at least half the frames.
         "--min_presence_ratio", "0.5",
         # Stage-1 certified TWO visible hands for these datasets (min_hands=2);
         # reconstruction must deliver both — drop single_valid_hand clips.
         "--min_presence_ratio_per_hand", "0.5",
         "--max_wrist_rotation_step", GATE_WRIST,
         "--max_hand_translation_step", GATE_HAND,
         "--max_finger_translation_step", GATE_FINGER,
         "--max_camera_translation_step", GATE_CAM_T,
         "--max_camera_rotation_step", GATE_CAM_R],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{EGOSMITH_ROOT}/src"})
    if r.returncode != 0 or not out_man.exists():
        raise RuntimeError(f"filter failed: {r.stderr[-300:]}")
    kept = sum(1 for l in open(out_man) if l.strip())
    fs.put(str(out_man), f"{GCS_FILT}/shard_{sfx}.filtered.jsonl")
    fs.put(str(rep), f"{GCS_FILT}/shard_{sfx}.report.json")
    shutil.rmtree(base)
    print(f"[{time.strftime('%H:%M:%S')}] shard_{sfx}: recon={len(have)} materialized={len(ok_rows)} "
          f"err={n_err} KEPT={kept} ({100*kept/max(1,len(ok_rows)):.1f}%) {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24, help="parallel clip materializers per shard")
    ap.add_argument("--shard_parallel", type=int, default=6, help="shards processed concurrently")
    ap.add_argument("--once", action="store_true", help="one pass, no daemon loop")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    while True:
        fs.invalidate_cache()
        done_c = {os.path.basename(p).split(".")[0].replace("shard_", "")
                  for p in fs.glob(f"{GCS_OUT}/_done/*.json")}
        done_d = {os.path.basename(p).split(".")[0].replace("shard_", "")
                  for p in fs.glob(f"{GCS_FILT}/*.report.json")}
        pend = sorted(done_c - done_d)
        if args.limit:
            pend = pend[:args.limit]
        print(f"pending shards: {len(pend)} (C-done={len(done_c)} D-done={len(done_d)})", flush=True)
        def safe(sfx):
            try:
                if not try_claim(sfx):
                    return
                process_shard(sfx, args.workers)
            except Exception as e:  # noqa: BLE001
                print(f"shard_{sfx} FAILED: {str(e)[:200]}", flush=True)
        with ThreadPoolExecutor(args.shard_parallel) as sex:
            list(sex.map(safe, pend))
        if args.once:
            break
        time.sleep(600)


if __name__ == "__main__":
    main()
