#!/usr/bin/env python
"""Retrofit shipped EgoDex WDS tars with full-articulation GT joints (.gt_joints.npy).

The shipped egodex tars carry {frame}.image.jpg/.lowdim.npy/.mano.npy(zeros)/.meta.json;
the 116-d lowdim keeps only wrist+5 fingertips per hand. The raw Vision Pro GT has
per-joint SE(3) for the full hand skeleton — this pass re-reads each clip's raw hdf5
(ranged zip reads via scripts/viewer/egodex_rawgt.py) and APPENDS one member per frame:

  {frame}.gt_joints.npy : (2, 21, 3) float32 world-frame joints, MANO order
                          (index 0 = left hand; joint order egodex_rawgt.joint_names)
  schema tag: "vp_world_21_mano_order_v1"

Append-at-end keeps every original member byte-identical at its original offset, so
existing descriptor.frame_offsets stay valid (verified: the original archive is
`members + zero trailer`; we rewrite as `members + new members + zero trailer`).
The sequential native loader (episodes.py, tarfile "r|") skips unknown member names.

Outputs go to a NEW prefix (frames_v2/) — v1 tars are never touched. Modes:
  retrofit  pull tar + raw hdf5, validate joints vs lowdim (wrist+tips, per frame),
            append members, upload to --dst_prefix. Multiprocess; resumable
            (skips clips whose dst object already exists with the expected size).
  flip      rewrite clip_manifest.filtered.jsonl -> frames_v2 pointers + extra flags
            (gt_joints/gt_joints_schema/mano_note), backing the old manifest up to
            filter_run/_prev21_backup/ FIRST.
  audit     verify every manifest clip has its frames_v2 tar at the exact expected
            size (original size - trailer + per-frame append + trailer) + deep
            readback of a random sample.

Per-clip validation (always on, retrofit): hdf5 wrist+tips must match the tar's own
lowdim (same source, float32 cast) within --tol on every frame; mismatch = failure,
clip is NOT uploaded.
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO), str(_REPO / "scripts" / "viewer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from egodex_rawgt import EgoDexRawGT, TIP_IDX  # noqa: E402

GT_SCHEMA = "vp_world_21_mano_order_v1"
GT_SUFFIX = ".gt_joints.npy"
RECORDSIZE = tarfile.RECORDSIZE  # 10240
MANO_NOTE = "zeros placeholder (no MANO fit); real full-articulation GT is per-frame .gt_joints.npy"

# lowdim slices (quality/constants.py native schema)
LD_LWRIST, LD_RWRIST = slice(0, 3), slice(3, 6)
LD_LTIPS, LD_RTIPS = slice(18, 33), slice(33, 48)

_fs = None
_rawgt = None


def _get_fs():
    global _fs
    if _fs is None:
        import gcsfs
        _fs = gcsfs.GCSFileSystem()
    return _fs


def _get_rawgt():
    global _rawgt
    if _rawgt is None:
        _rawgt = EgoDexRawGT(fs=_get_fs())
    return _rawgt


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
    """-> (sample_keys ordered by jpg name, lowdim (T,116) f32, end_of_members offset)."""
    lowdims, jpgs = {}, []
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
            elif m.name.endswith(GT_SUFFIX):
                raise ValueError(f"tar already carries {m.name}")
    keys = sorted(n[: -len(".image.jpg")] for n in jpgs)
    if sorted(lowdims) != keys:
        raise ValueError("jpg/lowdim member mismatch")
    if raw[end:].lstrip(b"\0"):
        raise ValueError("non-zero bytes after last member (unexpected trailer)")
    ld = np.stack([lowdims[k] for k in keys]).astype(np.float32)
    return keys, ld, end


def expected_new_size(old_size: int, old_end: int, n_frames: int) -> int:
    """gt_joints npy payload is 632 B (128 hdr + 2*21*3*4) -> 512 hdr + 1024 padded."""
    body = old_end + n_frames * 1536
    return ((body + 1024 + RECORDSIZE - 1) // RECORDSIZE) * RECORDSIZE


def retrofit_one(clip_id: str, shard: str, dst_prefix: str, tol: float) -> dict:
    fs, rawgt = _get_fs(), _get_rawgt()
    dst = f"{dst_prefix}/{os.path.basename(shard)}"
    raw = fs.cat(shard.replace("gs://", ""))
    keys, ld, end = _scan_tar(raw)
    T = len(keys)
    L, R = rawgt.joints(clip_id)          # (Traw,21,3) f64 each
    if L.shape[0] < T:
        raise ValueError(f"hdf5 shorter than tar: {L.shape[0]} < {T}")
    gt = np.stack([L[:T], R[:T]], axis=1).astype(np.float32)   # (T,2,21,3)
    if not np.isfinite(gt).all():
        raise ValueError("non-finite raw GT joints")
    err = max(
        np.abs(ld[:, LD_LWRIST] - gt[:, 0, 0]).max(),
        np.abs(ld[:, LD_RWRIST] - gt[:, 1, 0]).max(),
        np.abs(ld[:, LD_LTIPS].reshape(T, 5, 3) - gt[:, 0, TIP_IDX]).max(),
        np.abs(ld[:, LD_RTIPS].reshape(T, 5, 3) - gt[:, 1, TIP_IDX]).max(),
    )
    if err > tol:
        raise ValueError(f"lowdim/raw-GT mismatch: max_err={err:.3e} > {tol}")
    parts = [raw[:end]]
    parts += [_member_bytes(k + GT_SUFFIX, _npy_bytes(gt[t])) for t, k in enumerate(keys)]
    body = sum(len(p) for p in parts)
    total = ((body + 1024 + RECORDSIZE - 1) // RECORDSIZE) * RECORDSIZE
    parts.append(b"\0" * (total - body))
    blob = b"".join(parts)
    assert len(blob) == expected_new_size(len(raw), end, T), "size-model drift"
    fs.pipe(dst.replace("gs://", ""), blob)
    return dict(clip_id=clip_id, ok=True, frames=T, max_err=float(err),
                old_size=len(raw), new_size=len(blob))


def _worker(args):
    clip_id, shard, dst_prefix, tol = args
    t0 = time.time()
    try:
        out = retrofit_one(clip_id, shard, dst_prefix, tol)
    except Exception as e:  # noqa: BLE001
        out = dict(clip_id=clip_id, ok=False, error=f"{type(e).__name__}: {e}")
    out["sec"] = round(time.time() - t0, 3)
    return out


def _iter_manifest(path):
    """Stream manifest records (local path or gs://) without loading 4 GB at once."""
    if path.startswith("gs://"):
        f = _get_fs().open(path.replace("gs://", ""), "rb")
    else:
        f = open(path, "rb")
    with f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_retrofit(a):
    fs = _get_fs()
    items = [(r["clip_id"], r["descriptor"]["shard_path"]) for r in _iter_manifest(a.manifest)]
    if a.include:
        import re
        items = [it for it in items if re.search(a.include, it[0])]
    if a.limit:
        items = items[: a.limit]
    done = {}
    if a.resume:
        try:
            done = {os.path.basename(p): i["size"] for p, i in
                    fs.find(a.dst_prefix.replace("gs://", ""), detail=True).items()}
        except FileNotFoundError:
            done = {}
        print(f"[resume] {len(done)} objects already at {a.dst_prefix}", flush=True)
    todo = [it for it in items if os.path.basename(it[1]) not in done]
    print(f"[retrofit] {len(todo)}/{len(items)} clips to process "
          f"-> {a.dst_prefix} (workers={a.workers})", flush=True)
    results, t0, nbytes = [], time.time(), 0
    log = open(a.log_out, "a") if a.log_out else None
    # spawn: gcsfs (used in the driver for the resume listing) is not fork-safe
    mp = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp) as ex:
        for i, out in enumerate(ex.map(_worker,
                                       ((c, s, a.dst_prefix, a.tol) for c, s in todo),
                                       chunksize=4)):
            results.append(out)
            if log:
                log.write(json.dumps(out) + "\n")
            if out["ok"]:
                nbytes += out["old_size"] + out["new_size"]
            if (i + 1) % 200 == 0 or (i + 1) == len(todo):
                dt = time.time() - t0
                fails = sum(1 for r in results if not r["ok"])
                print(f"  {i+1}/{len(todo)}  {(i+1)/dt:.1f} clips/s  "
                      f"{nbytes/dt/1e6:.0f} MB/s r+w  fails={fails}  "
                      f"eta={(len(todo)-i-1)/max((i+1)/dt,1e-9)/60:.0f} min", flush=True)
                if log:
                    log.flush()
    if log:
        log.close()
    fails = [r for r in results if not r["ok"]]
    print(f"[retrofit] done: {len(results)-len(fails)} ok, {len(fails)} failed")
    for f in fails[:20]:
        print("  FAIL", f["clip_id"], f["error"])
    return 1 if fails else 0


def run_flip(a):
    """Backup manifest, then rewrite pointers frames/ -> frames_v2/ + extra flags."""
    fs = _get_fs()
    src = a.manifest.replace("gs://", "")
    backup = a.backup.replace("gs://", "")
    if not fs.exists(backup):
        fs.copy(src, backup)
        print(f"[flip] backup: {a.manifest} -> {a.backup}")
    else:
        print(f"[flip] backup already present: {a.backup}")
    old_root = a.dst_prefix.rsplit("/", 1)[0] + "/frames"
    tmp = a.flip_tmp
    n = 0
    with open(tmp, "w") as out:
        for rec in _iter_manifest(a.manifest):
            d = rec["descriptor"]
            if d["root_dir"].rstrip("/") != a.dst_prefix.rstrip("/"):
                assert d["root_dir"].rstrip("/") == old_root, d["root_dir"]
                d["root_dir"] = a.dst_prefix
                d["shard_path"] = f"{a.dst_prefix}/{os.path.basename(d['shard_path'])}"
                d["extra"].update({"gt_joints": True, "gt_joints_schema": GT_SCHEMA,
                                   "mano_note": MANO_NOTE})
            out.write(json.dumps(rec) + "\n")
            n += 1
    fs.put(tmp, src)
    print(f"[flip] rewrote {n} records -> {a.manifest}")
    return 0


def run_audit(a):
    """Every manifest clip must exist at dst with the exact size the retrofit worker
    recorded (workers assert the size model + validate content before upload)."""
    fs = _get_fs()
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
    # deep readback sample
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
        if njpg != ngt or arr.shape != (2, 21, 3) or not np.isfinite(arr).all():
            bad += 1
            print(f"  DEEP-BAD {base} njpg={njpg} ngt={ngt} shape={arr.shape}")
    print(f"[audit] deep sample {len(sample)}: {len(sample)-bad} ok, {bad} bad")
    return 1 if (missing or badsize or bad) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retrofit", "flip", "audit"])
    ap.add_argument("--manifest",
                    default="gs://foundational-research/hoi-dataset/egosmith_filtered/"
                            "egodex/filter_run/clip_manifest.filtered.jsonl")
    ap.add_argument("--dst_prefix",
                    default="gs://foundational-research/hoi-dataset/egosmith_filtered/"
                            "egodex/frames_v2")
    ap.add_argument("--backup",
                    default="gs://foundational-research/hoi-dataset/egosmith_filtered/"
                            "egodex/filter_run/_prev21_backup/clip_manifest.filtered.jsonl")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include", default=None)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log_out", default=None)
    ap.add_argument("--flip_tmp", default="/root/egodex_retrofit/manifest.flipped.jsonl")
    ap.add_argument("--sample", type=int, default=200, help="audit deep-readback size")
    a = ap.parse_args()
    sys.exit({"retrofit": run_retrofit, "flip": run_flip, "audit": run_audit}[a.mode](a))


if __name__ == "__main__":
    main()
