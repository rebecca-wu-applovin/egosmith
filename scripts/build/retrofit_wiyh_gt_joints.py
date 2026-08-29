#!/usr/bin/env python
"""Retrofit shipped wiyh_native WDS tars with the full 25-joint GT (.gt_joints.npy).

The shipped wiyh_native tars carry {frame}.image.jpg/.lowdim.npy/.mano.npy(zeros)/
.meta.json; the 116-d lowdim keeps only wrist+5 fingertips per hand. The source
carries the full MANUS 25-joint wrist-local glove skeleton — this pass re-streams
each keeper session's source sample (gcsfs, same chain as
scripts/build/wiyh_native_extractor.py) and recomputes the world(=chest)-frame
skeleton per frame with the accepted per-session extrinsic:

    p_world = R_eef @ (R_gl @ p_local + t_gl) + t_eef

then APPENDS one member per frame:

  {frame}.gt_joints.npy : (2, 25, 3) float32 world-frame joints
                          (index 0 = left hand; joint order = MANUS glove order:
                           thumb 0-3, index 4-8, middle 9-13, ring 14-18,
                           little 19-23, wrist 24; tips [3,8,13,18,23])
  schema tag: "wiyh_manus_world_25_v1"

Append-at-end keeps every original member byte-identical at its original offset, so
existing descriptor.frame_offsets stay valid (original archive = members + zero
trailer; rewritten = members + new members + zero trailer). Sequential native
loaders skip unknown member names.

Per-clip validation (always on, retrofit): the tar's own lowdim wrist (ld[0:3],
ld[3:6]) and tips (ld[18:33], ld[33:48]) must equal gt[:, :, 24] / gt[:, :, TIPS]
within --tol on every PRESENCE-ON frame+hand (lowdim applies nearest-valid ffill to
presence-off frames — generate_keypoints_wds._fill_missing_hand — so only
presence-on values are raw). Mismatch = failure, clip is NOT uploaded. Since the
lowdim was float32-cast from the identical float64 chain, max_err is ~0.

Segment mapping: descriptor.extra.segment_frame_range = [s, e) episode-frame span;
tar frame f%05d local index i maps to episode frame s+i. Episode source member is
resolved via extra.session_id == "wiyh_native_" + clip_base(member).

Outputs go to a NEW prefix (frames_v2/) — v1 tars are never touched. Modes:
  retrofit  one job per EPISODE (source streamed once): recompute skeleton,
            validate + append + upload every clip of the episode. Resumable
            (skips clips whose dst object already exists with the expected size).
  flip      rewrite clip_manifest.filtered.jsonl -> frames_v2 pointers + extra
            flags (gt_joints/gt_joints_schema), backing the old manifest up to
            filter_run/_pre25_backup/ FIRST.
  audit     verify every manifest clip has its frames_v2 tar at the exact expected
            size + deep readback of a random sample.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

GT_SCHEMA = "wiyh_manus_world_25_v1"
GT_SUFFIX = ".gt_joints.npy"
RECORDSIZE = tarfile.RECORDSIZE  # 10240
TIPS = [3, 8, 13, 18, 23]
WRIST = 24
PER_FRAME_APPEND = 1536  # 512 tar hdr + (128 npy hdr + 600 data -> 1024 padded)

# lowdim slices (quality/constants.py native schema)
LD_LWRIST, LD_RWRIST = slice(0, 3), slice(3, 6)
LD_LTIPS, LD_RTIPS = slice(18, 33), slice(33, 48)

_FS = None
_CFG = {}


def _init(cfg):
    global _FS, _CFG
    import gcsfs
    _FS = gcsfs.GCSFileSystem()
    _CFG = cfg


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _member_bytes(name: str, data: bytes) -> bytes:
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mtime = 0
    ti.mode = 0o644
    try:
        hdr = ti.tobuf(tarfile.USTAR_FORMAT)
    except ValueError:  # name too long for ustar
        hdr = ti.tobuf(tarfile.GNU_FORMAT)
    pad = (512 - len(data) % 512) % 512
    return hdr + data + b"\0" * pad


def _scan_tar(raw: bytes):
    """-> (frame keys sorted, lowdim (T,116) f32, presence (T,) uint8, end offset)."""
    lowdims, presence, jpgs = {}, {}, []
    end = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            end = max(end, m.offset_data + ((m.size + 511) // 512) * 512)
            if m.name.endswith(".image.jpg"):
                jpgs.append(m.name)
            elif m.name.endswith(".lowdim.npy"):
                lowdims[m.name[: -len(".lowdim.npy")]] = np.load(
                    io.BytesIO(tf.extractfile(m).read()), allow_pickle=False)
            elif m.name.endswith(".meta.json"):
                presence[m.name[: -len(".meta.json")]] = int(
                    json.loads(tf.extractfile(m).read())["presence"])
            elif m.name.endswith(GT_SUFFIX):
                raise ValueError(f"tar already carries {m.name}")
    keys = sorted(n[: -len(".image.jpg")] for n in jpgs)
    if sorted(lowdims) != keys or sorted(presence) != keys:
        raise ValueError("jpg/lowdim/meta member mismatch")
    if raw[end:].lstrip(b"\0"):
        raise ValueError("non-zero bytes after last member (unexpected trailer)")
    ld = np.stack([lowdims[k] for k in keys]).astype(np.float32)
    pres = np.array([presence[k] for k in keys], np.uint8)
    return keys, ld, pres, end


def expected_new_size(old_size: int, old_end: int, n_frames: int) -> int:
    body = old_end + n_frames * PER_FRAME_APPEND
    return ((body + 1024 + RECORDSIZE - 1) // RECORDSIZE) * RECORDSIZE


def episode_skeleton(session: str, member: dict, parts: dict) -> np.ndarray:
    """Stream the source sample and recompute (T, 2, 25, 3) float64 world joints
    exactly as wiyh_native_extractor.load() does (left = index 0)."""
    from scipy.spatial.transform import Rotation as Rt
    from wiyh_gate_census import SampleStreams, stream_sample

    extr = json.loads(Path(_CFG["extrinsics"]).read_text())[session]
    err = None
    for attempt in range(3):
        try:
            h5, _masks, _ = stream_sample(_FS, parts, member, want_jpgs=False)
            break
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"stream_sample failed: {err}")
    if h5 is None:
        raise ValueError("no dataset.hdf5 in sample")
    ss = SampleStreams(h5)
    pw = {}
    for side in ("left", "right"):
        Rg = np.array(extr[side]["R"], np.float64)
        tg = np.array(extr[side]["t"], np.float64)
        eef = ss.eef[side]
        Re = Rt.from_quat(eef[:, 3:]).as_matrix()
        te = eef[:, :3]
        loc = ss.pts[side]
        pw[side] = np.einsum("tij,tkj->tki", Re, loc @ Rg.T + tg) + te[:, None]
    return np.stack([pw["left"], pw["right"]], axis=1)  # (T,2,25,3) f64


def retrofit_episode(job):
    """job = (session, member, parts, clips=[{clip_id, shard, seg:[s,e]}]).
    Returns one result dict per clip."""
    session, member, parts, clips = job
    results = []
    try:
        gt_ep = episode_skeleton(session, member, parts).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        return [dict(clip_id=c["clip_id"], ok=False,
                     error=f"episode: {type(e).__name__}: {e}") for c in clips]
    tol = float(_CFG["tol"])
    dst_prefix = _CFG["dst_prefix"]
    for c in clips:
        t0 = time.time()
        out = dict(clip_id=c["clip_id"])
        try:
            raw = _FS.cat(c["shard"].replace("gs://", ""))
            keys, ld, pres, end = _scan_tar(raw)
            T = len(keys)
            s, e = c["seg"]
            if e - s != T:
                raise ValueError(f"segment_frame_range {c['seg']} != tar frames {T}")
            if e > gt_ep.shape[0]:
                raise ValueError(f"segment end {e} > episode frames {gt_ep.shape[0]}")
            gt = gt_ep[s:e]  # (T,2,25,3) f32
            if not np.isfinite(gt).all():
                raise ValueError("non-finite recomputed GT joints")
            # validation on presence-on frame+hand (lowdim is ffilled elsewhere)
            errs, n_checked = [], 0
            for hb, (wsl, tsl) in enumerate(((LD_LWRIST, LD_LTIPS),
                                             (LD_RWRIST, LD_RTIPS))):
                on = (pres & (1 << hb)) > 0
                if not on.any():
                    continue
                n_checked += int(on.sum())
                errs.append(np.abs(ld[on, wsl] - gt[on, hb, WRIST]).max())
                errs.append(np.abs(ld[on, tsl].reshape(-1, 5, 3)
                                   - gt[on, hb][:, TIPS]).max())
            if n_checked == 0:
                raise ValueError("no presence-on frames to validate against")
            err = float(max(errs))
            if err > tol:
                raise ValueError(f"lowdim/recompute mismatch: max_err={err:.3e} > {tol}")
            parts_b = [raw[:end]]
            parts_b += [_member_bytes(k + GT_SUFFIX, _npy_bytes(gt[t]))
                        for t, k in enumerate(keys)]
            body = sum(len(p) for p in parts_b)
            total = ((body + 1024 + RECORDSIZE - 1) // RECORDSIZE) * RECORDSIZE
            parts_b.append(b"\0" * (total - body))
            blob = b"".join(parts_b)
            assert len(blob) == expected_new_size(len(raw), end, T), "size-model drift"
            dst = f"{dst_prefix}/{os.path.basename(c['shard'])}"
            _FS.pipe(dst.replace("gs://", ""), blob)
            out.update(ok=True, frames=T, checked=n_checked, max_err=err,
                       old_size=len(raw), new_size=len(blob))
        except Exception as e:  # noqa: BLE001
            out.update(ok=False, error=f"{type(e).__name__}: {e}")
        out["sec"] = round(time.time() - t0, 3)
        results.append(out)
    return results


def _iter_manifest(path):
    if path.startswith("gs://"):
        import gcsfs
        f = gcsfs.GCSFileSystem().open(path.replace("gs://", ""), "rb")
    else:
        f = open(path, "rb")
    with f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _episode_jobs(a, records):
    """Group manifest records by episode and resolve source members."""
    from wiyh_gate_census import load_sessions
    from wiyh_native_extractor import clip_base

    sess_members = load_sessions(Path(a.index_dir))
    parts_by_scene = {f.stem.split(".")[0]: json.loads(f.read_text())
                      for f in Path(a.index_dir).glob("*.parts.json")}
    member_by_epid = {}
    for session in {r["descriptor"]["extra"]["session"] for r in records}:
        for m in sess_members[session]:
            member_by_epid[f"wiyh_native_{clip_base(m['base'])}"] = m
    by_ep = {}
    for r in records:
        ex = r["descriptor"]["extra"]
        by_ep.setdefault(ex["session_id"], []).append(
            dict(clip_id=r["clip_id"], shard=r["descriptor"]["shard_path"],
                 seg=ex["segment_frame_range"], session=ex["session"]))
    jobs = []
    for epid, clips in sorted(by_ep.items()):
        m = member_by_epid.get(epid)
        if m is None:
            raise KeyError(f"no source member for episode {epid}")
        jobs.append((clips[0]["session"], m, parts_by_scene[m["scene"]], clips))
    return jobs


def run_retrofit(a):
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    records = list(_iter_manifest(a.manifest))
    if a.include:
        import re
        records = [r for r in records if re.search(a.include, r["clip_id"])]
    jobs = _episode_jobs(a, records)
    done = {}
    if a.resume:
        try:
            done = {os.path.basename(p): i["size"] for p, i in
                    fs.find(a.dst_prefix.replace("gs://", ""), detail=True).items()}
        except FileNotFoundError:
            done = {}
        print(f"[resume] {len(done)} objects already at {a.dst_prefix}", flush=True)
        pruned = []
        for session, m, parts, clips in jobs:
            todo = [c for c in clips if os.path.basename(c["shard"]) not in done]
            if todo:
                pruned.append((session, m, parts, todo))
        jobs = pruned
    n_clips = sum(len(j[3]) for j in jobs)
    print(f"[retrofit] {len(jobs)} episodes / {n_clips} clips -> {a.dst_prefix} "
          f"(workers={a.workers})", flush=True)
    cfg = dict(extrinsics=a.extrinsics, dst_prefix=a.dst_prefix, tol=a.tol)
    results = []
    log = open(a.log_out, "a") if a.log_out else None
    t0 = time.time()
    # spawn: gcsfs (used in the driver for the resume listing) is not fork-safe —
    # forked workers inherit its asyncio state and hang on the first read
    ctx = __import__("multiprocessing").get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, outs in enumerate(pool.imap_unordered(retrofit_episode, jobs), 1):
            results += outs
            if log:
                for o in outs:
                    log.write(json.dumps(o) + "\n")
                log.flush()
            fails = sum(1 for r in results if not r["ok"])
            print(f"  ep {i}/{len(jobs)}  clips={len(results)}/{n_clips}  "
                  f"fails={fails}  {(time.time()-t0)/60:.1f} min", flush=True)
    if log:
        log.close()
    fails = [r for r in results if not r["ok"]]
    oks = [r for r in results if r["ok"]]
    if oks:
        print(f"[retrofit] max_err over all clips: {max(r['max_err'] for r in oks):.3e}")
    print(f"[retrofit] done: {len(oks)} ok, {len(fails)} failed")
    for f in fails[:20]:
        print("  FAIL", f["clip_id"], f["error"])
    return 1 if fails else 0


def run_flip(a):
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    src = a.manifest.replace("gs://", "")
    backup = a.backup.replace("gs://", "")
    if not fs.exists(backup):
        fs.copy(src, backup)
        print(f"[flip] backup: {a.manifest} -> {a.backup}")
    else:
        print(f"[flip] backup already present: {a.backup}")
    old_root = a.dst_prefix.rsplit("/", 1)[0] + "/frames"
    n = 0
    with open(a.flip_tmp, "w") as out:
        for rec in _iter_manifest(a.manifest):
            d = rec["descriptor"]
            if d["root_dir"].rstrip("/") != a.dst_prefix.rstrip("/"):
                assert d["root_dir"].rstrip("/") == old_root, d["root_dir"]
                d["root_dir"] = a.dst_prefix
                d["shard_path"] = f"{a.dst_prefix}/{os.path.basename(d['shard_path'])}"
                d["extra"].update({"gt_joints": True, "gt_joints_schema": GT_SCHEMA})
            out.write(json.dumps(rec) + "\n")
            n += 1
    fs.put(a.flip_tmp, src)
    print(f"[flip] rewrote {n} records -> {a.manifest}")
    return 0


def run_audit(a):
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    recs = [(r["clip_id"], os.path.basename(r["descriptor"]["shard_path"]))
            for r in _iter_manifest(a.manifest)]
    logged = {}
    for lp in (a.log_out or "").split(","):
        if lp and os.path.exists(lp):
            for line in open(lp):
                r = json.loads(line)
                if r.get("ok"):
                    logged[r["clip_id"]] = r["new_size"]
    objs = {os.path.basename(p): i["size"] for p, i in
            fs.find(a.dst_prefix.replace("gs://", ""), detail=True).items()}
    missing, badsize, unlogged = [], [], 0
    for clip_id, base in recs:
        if base not in objs:
            missing.append(base)
            continue
        want = logged.get(clip_id)
        if want is None:
            unlogged += 1
        elif objs[base] != want:
            badsize.append((base, objs[base], want))
    print(f"[audit] manifest clips={len(recs)} dst objects={len(objs)} "
          f"missing={len(missing)} badsize={len(badsize)} unlogged={unlogged}")
    for b in (missing + [x[0] for x in badsize])[:20]:
        print("  BAD", b)
    rng = random.Random(0)
    sample = rng.sample(recs, min(a.sample, len(recs)))
    bad = 0
    for _clip_id, base in sample:
        raw = fs.cat(f"{a.dst_prefix}/{base}".replace("gs://", ""))
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            names = tf.getnames()
            njpg = sum(n.endswith(".image.jpg") for n in names)
            ngt = sum(n.endswith(GT_SUFFIX) for n in names)
            arr = np.load(io.BytesIO(tf.extractfile(
                sorted(n for n in names if n.endswith(GT_SUFFIX))[0]).read()),
                allow_pickle=False)
        if njpg != ngt or arr.shape != (2, 25, 3) or not np.isfinite(arr).all():
            bad += 1
            print(f"  DEEP-BAD {base} njpg={njpg} ngt={ngt} shape={arr.shape}")
    print(f"[audit] deep sample {len(sample)}: {len(sample)-bad} ok, {bad} bad")
    return 1 if (missing or badsize or bad) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retrofit", "flip", "audit"])
    ap.add_argument("--manifest",
                    default="gs://foundational-research/hoi-dataset/egosmith_filtered/"
                            "wiyh_native/filter_run/clip_manifest.filtered.jsonl")
    ap.add_argument("--dst_prefix",
                    default="gs://foundational-research/hoi-dataset/egosmith_filtered/"
                            "wiyh_native/frames_v2")
    ap.add_argument("--backup",
                    default="gs://foundational-research/hoi-dataset/egosmith_filtered/"
                            "wiyh_native/filter_run/_pre25_backup/clip_manifest.filtered.jsonl")
    ap.add_argument("--index_dir", default="/root/w7_full/wiyh/index")
    ap.add_argument("--extrinsics",
                    default="/root/w7_native/anchors/accepted_extrinsics.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--include", default=None)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log_out", default=None)
    ap.add_argument("--flip_tmp", default="/root/w7_native/remediation/manifest.flipped.jsonl")
    ap.add_argument("--sample", type=int, default=60, help="audit deep-readback size")
    a = ap.parse_args()
    sys.exit({"retrofit": run_retrofit, "flip": run_flip, "audit": run_audit}[a.mode](a))


if __name__ == "__main__":
    main()
