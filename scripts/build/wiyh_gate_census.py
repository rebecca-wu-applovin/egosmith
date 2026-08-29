#!/usr/bin/env python
"""WIYH wrist-lock gate census over all (serial, datetime) sessions.

Methodology = anchor-pilot gate_stats: project each hand's eef wrist point
(pose/{side}_eef/feedback/pose_in_chest, index_map-aligned, conf==1, |dt|<=25ms)
through the chest KB4 fisheye and measure pixel distance to the nearest
hand-mask pixel (hand_masks/lf_chest_fisheye). A session is LOCKED when BOTH
hands have >=--min_n valid frames and frac(dist<30px) >= --lock_frac.

One member (the middle sample) is streamed per session; the inner tar.gz is
inflated once, keeping only dataset.hdf5 + hand-mask nonzero coords (masks sit
at the END of each sample stream, so the full member must be read regardless).

Output: JSONL, one row per session with per-hand stats + per-frame distance
codes (px int; -1 conf/dt fail, -2 unprojectable, -3 no mask) — the same codes
the converter later re-derives per converted sample.

Usage:
  python scripts/build/wiyh_gate_census.py --index_dir /root/w7_full/wiyh/index \
      --out /root/w7_native/census/census.jsonl --workers 32
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import tarfile
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CHEST = "lf_chest_fisheye"
NAME_RE = re.compile(
    r"worldcode_(HS-\d-\d+)_([\d]{4}-[\d]{2}-[\d]{2}-[\d-]+?)_(\d+)_s0_vlta_reorg_sample_(\d+)-(\d+)$")
MASK_SCALE = 4  # downscale factor for mask nonzero coords (30px gate >> 4px error)
MAX_DT_MS = 25.0
GATE_PX = 30.0


def parse_member(name: str):
    base = Path(name).name.replace(".tar.gz", "")
    m = NAME_RE.match(base)
    if not m:
        return None
    serial, dt, epi, si, sj = m.groups()
    return {"base": base, "serial": serial, "dt": dt, "epi": int(epi),
            "si": int(si), "session": f"{serial}_{dt}",
            "dev": serial[-5:], "date": dt[:10]}


def load_sessions(index_dir: Path, scenes=None):
    sess = defaultdict(list)
    for f in sorted(index_dir.glob("*.members.jsonl")):
        scene = f.stem.split(".")[0]
        if scenes and scene not in scenes:
            continue
        for l in open(f):
            if not l.strip():
                continue
            m = json.loads(l)
            p = parse_member(m["name"])
            if p is None:
                continue
            m.update(p)
            m["scene"] = scene
            sess[p["session"]].append(m)
    for v in sess.values():
        v.sort(key=lambda m: (m["epi"], m["si"]))
    return sess


class StreamedConcat:
    """Sequential file-like over [offset, offset+size) of the concatenated parts.

    read() serves from a fetched chunk via a position pointer (NO per-call buffer
    re-slicing: gzip pulls ~8KB at a time, and slicing a 32MB tail each call made
    the census CPU-bound at ~20x slowdown). Short reads are fine for gzip/tarfile
    stream consumers."""

    def __init__(self, fs, parts, offset, size, chunk=64 * 1024 * 1024):
        self.fs, self.parts, self.chunk = fs, parts, chunk
        self.abs = offset
        self.end = offset + size
        self.buf = memoryview(b"")
        self.pos = 0
        self.starts = []
        pos = 0
        for p in parts:
            self.starts.append(pos)
            pos += p["size"]

    def _fetch(self):
        if self.abs >= self.end:
            return b""
        want = min(self.chunk, self.end - self.abs)
        out = []
        pos = self.abs
        while want > 0:
            i = 0
            while i < len(self.parts) and self.starts[i] + self.parts[i]["size"] <= pos:
                i += 1
            p, s = self.parts[i], self.starts[i]
            lo = pos - s
            hi = min(lo + want, p["size"])
            out.append(self.fs.cat_file(p["uri"].replace("gs://", ""), start=lo, end=hi))
            got = hi - lo
            pos += got
            want -= got
        blob = b"".join(out)
        self.abs += len(blob)
        return blob

    def read(self, n=-1):
        if n is None or n < 0:
            chunks = [bytes(self.buf[self.pos:])]
            self.buf, self.pos = memoryview(b""), 0
            while True:
                b = self._fetch()
                if not b:
                    break
                chunks.append(b)
            return b"".join(chunks)
        if self.pos >= len(self.buf):
            self.buf = memoryview(self._fetch())
            self.pos = 0
            if not self.buf:
                return b""
        out = bytes(self.buf[self.pos:self.pos + n])
        self.pos += len(out)
        return out


def stream_sample(fs, parts, member, want_jpgs=False):
    """Inflate one worldcode member; return (h5_bytes, {mask_name: coords}, {jpg_name: bytes}).
    Mask coords are Nx2 int16 (x, y) at 1/MASK_SCALE resolution."""
    import cv2
    raw = StreamedConcat(fs, parts, int(member["offset"]), int(member["size"]))
    gz = gzip.GzipFile(fileobj=raw, mode="rb")
    h5 = None
    masks = {}
    jpgs = {}
    with tarfile.open(fileobj=gz, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            n = m.name
            if n.endswith("dataset.hdf5"):
                with tf.extractfile(m) as src:
                    h5 = src.read()
            elif f"hand_masks/{CHEST}/" in n and n.endswith(".png"):
                with tf.extractfile(m) as src:
                    arr = np.frombuffer(src.read(), np.uint8)
                mm = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if mm is None:
                    masks[Path(n).name] = None
                else:
                    small = mm[::MASK_SCALE, ::MASK_SCALE]
                    ys, xs = np.nonzero(small)
                    masks[Path(n).name] = np.stack([xs, ys], 1).astype(np.int16)
            elif want_jpgs and f"camera/{CHEST}/" in n and n.endswith(".jpg"):
                with tf.extractfile(m) as src:
                    jpgs[Path(n).name] = src.read()
    return h5, masks, jpgs


class SampleStreams:
    """Aligned per-frame streams parsed from a sample's dataset.hdf5 bytes."""

    def __init__(self, h5_bytes: bytes):
        import io
        import h5py
        f = h5py.File(io.BytesIO(h5_bytes), "r")
        self.f = f
        cal = f[f"meta/calibration/{CHEST}"]
        self.K = np.array(cal["intrinsic"], np.float64)
        self.D = np.array(cal["distortion"], np.float64).reshape(-1)[:4]
        self.W, self.H = int(cal["width"][0]), int(cal["height"][0])
        ext = np.array(cal["extrinsic"], np.float64)
        self.R_ext, self.t_ext = ext[:3, :3], ext[:3, 3]
        cam = f[f"observation/camera/{CHEST}"][:]
        self.ts = cam["timestamp"].astype(np.float64)
        self.frame_names = [Path(p.decode()).name for p in cam["file_path"]]
        self.n = len(self.frame_names)
        self.pts, self.eef, self.conf, self.gdt, self.edt = {}, {}, {}, {}, {}
        for side in ("left", "right"):
            g = f[f"action/{side}_hand_glove/feedback/joint_angle"][:]
            e = f[f"pose/{side}_eef/feedback/pose_in_chest"][:]
            gi = f[f"meta/index_map/action/{side}_hand_glove/feedback/joint_angle"][:]
            ei = f[f"meta/index_map/pose/{side}_eef/feedback/pose_in_chest"][:]
            self.pts[side] = g["value"].reshape(-1, 25, 3)[gi["aligned_index"]]
            self.eef[side] = e["value"][ei["aligned_index"]]
            self.conf[side] = e["confidence"][ei["aligned_index"]]
            self.gdt[side] = np.abs(gi["time_diff"])
            self.edt[side] = np.abs(ei["time_diff"])

    def frame_ok(self, i, side, max_dt_ms=MAX_DT_MS):
        return bool(self.conf[side][i] >= 1 and self.edt[side][i] <= max_dt_ms
                    and self.gdt[side][i] <= max_dt_ms)

    def eef_px(self, i, side):
        import cv2
        from scipy.spatial.transform import Rotation as Rt
        v = self.eef[side][i]
        Re, te = Rt.from_quat(v[3:]).as_matrix(), v[:3]
        cam = self.R_ext.T @ (te - self.t_ext)
        if cam[2] <= 1e-6:
            return None
        uv, _ = cv2.fisheye.projectPoints(cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                          self.K, self.D.reshape(4, 1))
        p = uv.reshape(2)
        return p if np.isfinite(p).all() else None


def gate_dists(ss: SampleStreams, masks: dict):
    """Per-frame per-hand gate distance codes for a sample (px; see module doc)."""
    mask_by_name = masks
    out = {}
    for side in ("left", "right"):
        d = np.full(ss.n, -1, np.int32)
        for i in range(ss.n):
            if not ss.frame_ok(i, side):
                continue
            p = ss.eef_px(i, side)
            if p is None or not (0 <= p[0] < ss.W and 0 <= p[1] < ss.H):
                d[i] = -2
                continue
            coords = mask_by_name.get(ss.frame_names[i].replace(".jpg", ".png"))
            if coords is None or len(coords) == 0:
                d[i] = -3
                continue
            dx = coords[:, 0].astype(np.float32) * MASK_SCALE - p[0]
            dy = coords[:, 1].astype(np.float32) * MASK_SCALE - p[1]
            d[i] = int(np.sqrt((dx * dx + dy * dy).min()))
        out[side] = d
    return out


def side_stats(d: np.ndarray):
    v = d[d >= 0].astype(np.float64)
    if len(v) == 0:
        return {"n": 0, "med": None, "p90": None, "frac_lt30": 0.0}
    return {"n": int(len(v)), "med": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "frac_lt30": float((v < GATE_PX).mean())}


def match_masks(ss: SampleStreams, masks: dict):
    """Map mask files to frames by name; fall back to sorted-order alignment."""
    hit = sum(1 for n in ss.frame_names if n.replace(".jpg", ".png") in masks)
    if hit >= 0.9 * ss.n:
        return masks
    names = sorted(masks)
    return {ss.frame_names[i].replace(".jpg", ".png"): masks[names[i]]
            for i in range(min(ss.n, len(names)))}


_FS = None


def _worker_init():
    global _FS
    import gcsfs
    _FS = gcsfs.GCSFileSystem()


def _census_one(job):
    session, members, parts_by_scene, min_n, lock_frac = job
    row = {"session": session, "scene": members[0]["scene"], "dev": members[0]["dev"],
           "date": members[0]["date"], "n_members": len(members)}
    cands = [m for m in members if m["size"] > 100_000_000]
    if not cands:
        row["error"] = "all members stub"
        return row
    member = cands[len(cands) // 2]
    row["member"] = member["base"]
    row["member_bytes"] = int(member["size"])
    parts = parts_by_scene[member["scene"]]
    for attempt in range(3):
        try:
            h5, masks, _ = stream_sample(_FS, parts, member)
            if h5 is None:
                row["error"] = "no dataset.hdf5"
                return row
            ss = SampleStreams(h5)
            masks = match_masks(ss, masks)
            dists = gate_dists(ss, masks)
            row["n_frames"] = ss.n
            locked = True
            for side in ("left", "right"):
                st = side_stats(dists[side])
                row[side] = st
                row[side]["dists"] = dists[side].tolist()
                if st["n"] < min_n or st["frac_lt30"] < lock_frac:
                    locked = False
            row["locked"] = locked
            return row
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:150]}"
            time.sleep(5 * (attempt + 1))
    row["error"] = err
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min_n", type=int, default=20)
    ap.add_argument("--lock_frac", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help=">0: only N sessions (smoke)")
    a = ap.parse_args()

    idx = Path(a.index_dir)
    sess = load_sessions(idx, a.scenes)
    parts_by_scene = {f.stem.split(".")[0]: json.loads(f.read_text())
                      for f in idx.glob("*.parts.json")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            try:
                done.add(json.loads(l)["session"])
            except Exception:  # noqa: BLE001
                pass
    todo = [(s, v, parts_by_scene, a.min_n, a.lock_frac)
            for s, v in sorted(sess.items()) if s not in done]
    import random
    random.Random(7).shuffle(todo)  # scene-interleaved progress; resume-safe (done-set)
    if a.limit:
        todo = todo[:a.limit]
    print(f"[census] sessions={len(sess)} done={len(done)} todo={len(todo)} "
          f"workers={a.workers}", flush=True)
    t0 = time.time()
    n_lock = n_err = 0
    with Pool(a.workers, initializer=_worker_init) as pool, open(out, "a") as w:
        for k, row in enumerate(pool.imap_unordered(_census_one, todo, chunksize=1), 1):
            w.write(json.dumps(row) + "\n")
            w.flush()
            n_lock += int(bool(row.get("locked")))
            n_err += int("error" in row)
            if k % 25 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[census] {k}/{len(todo)} locked={n_lock} err={n_err} "
                      f"({el:.0f}s, {k/el*3600:.0f}/h)", flush=True)
    print(f"[census] DONE locked={n_lock} err={n_err}", flush=True)
    print("WIYH_CENSUS_DONE")


if __name__ == "__main__":
    sys.exit(main())
