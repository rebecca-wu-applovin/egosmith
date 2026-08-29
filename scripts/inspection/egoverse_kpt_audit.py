#!/usr/bin/env python
"""Pixel-agreement audit for EgoVerse zarr-v3 subsets with native hand keypoints
(microagi / scale / mecka-flagship; aria pattern). Overlay smoke gate per
category_filtering_plan.md:132 — project the subset's own 21-kpt hands onto its
RGB and check lock-on, BEFORE any conversion/filter run.

Reads zarr v3 manually (vendor shard indexes carry corrupt crc32c; same ranged
parse as egoverse_zarr_stage1.py): numeric arrays are sharding_indexed with inner
chunk = 1 frame, codecs bytes+zstd; images are vlen-bytes JPEG + zstd.

Convention candidates tested per episode (best-by-score wins, reported):
  A) keypoints world-frame, w2c = inv(obs_head_pose as c2w)   [aria convention]
  B) keypoints world-frame, w2c = obs_head_pose directly
  C) keypoints already camera-frame (project directly)

Usage:
  PYTHONPATH=src python scripts/inspection/egoverse_kpt_audit.py \
    --subset microagi --n_episodes 50 --n_overlays 5 --out_dir /tmp/ma_audit
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np


def _fs():
    import gcsfs
    return gcsfs.GCSFileSystem()


class ZArr:
    """Manual reader for zarr-v3 arrays (sharding_indexed or plain bytes+zstd).

    Numeric arrays: use read_all() -> full ndarray (they are small).
    vlen-bytes JPEG image arrays (1-D grid, single shard c/0): use chunk(i).
    """

    def __init__(self, fs, array_dir: str):
        import zstandard as zstd
        self.fs = fs
        self.dir = array_dir
        zj = json.loads(fs.cat(f"{array_dir}/zarr.json"))
        self.shape = zj["shape"]
        self.dtype = zj["data_type"]
        self.vlen = self.dtype == "variable_length_bytes"
        self.grid = zj["chunk_grid"]["configuration"]["chunk_shape"]
        c0 = zj["codecs"][0]
        self.sharded = c0["name"] == "sharding_indexed"
        self.inner = c0["configuration"]["chunk_shape"] if self.sharded else self.grid
        self._dec = zstd.ZstdDecompressor()
        self._np_dtype = {"float64": "<f8", "float32": "<f4",
                          "int64": "<i8", "int32": "<i4"}.get(self.dtype)
        self._index = None

    # ---- vlen JPEG access (1-D grid; one shard object per grid chunk c/<k>) ----
    def _load_index(self, k: int):
        if self._index is None:
            self._index = {}
        if k in self._index:
            return
        gsize = min(self.grid[0], self.shape[0] - k * self.grid[0])
        n_inner = (gsize + self.inner[0] - 1) // self.inner[0] if self.sharded else 1
        # index length uses the FULL grid inner count (zarr pads shards uniformly)
        n_inner_full = (self.grid[0] + self.inner[0] - 1) // self.inner[0]
        shard = f"{self.dir}/c/{k}"
        size = self.fs.info(shard)["size"]
        idx_len = n_inner_full * 16 + 4
        raw = self.fs.read_block(shard, size - idx_len, idx_len)
        self._index[k] = np.frombuffer(raw[: n_inner_full * 16], dtype="<u8").reshape(-1, 2)

    def chunk(self, i: int):
        """i-th element of a 1-D vlen array (JPEG bytes)."""
        k, j = i // self.grid[0], (i % self.grid[0]) // self.inner[0]
        self._load_index(k)
        off, nb = int(self._index[k][j, 0]), int(self._index[k][j, 1])
        if nb == 0 or off == 2**64 - 1:
            return None
        d = self._dec.decompress(self.fs.read_block(f"{self.dir}/c/{k}", off, nb),
                                 max_output_size=100_000_000)
        if self.vlen:
            jj = d.find(b"\xff\xd8\xff")
            return d[jj:] if jj >= 0 else None
        return np.frombuffer(d, dtype=self._np_dtype)

    # ---- full numeric read ----
    def read_all(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self._np_dtype)
        n_grid = [(s + g - 1) // g for s, g in zip(self.shape, self.grid)]
        import itertools
        for gidx in itertools.product(*[range(n) for n in n_grid]):
            path = f"{self.dir}/c/" + "/".join(str(g) for g in gidx)
            data = self.fs.cat(path)
            g0 = [gi * g for gi, g in zip(gidx, self.grid)]
            gshape = [min(g, s - o) for g, s, o in zip(self.grid, self.shape, g0)]
            if self.sharded:
                n_inner = 1
                for gs, ic in zip(self.grid, self.inner):
                    n_inner *= (gs + ic - 1) // ic
                idx_len = n_inner * 16 + 4
                index = np.frombuffer(data[-idx_len:][: n_inner * 16], dtype="<u8").reshape(-1, 2)
                n_in = [(gs + ic - 1) // ic for gs, ic in zip(self.grid, self.inner)]
                for k, iidx in enumerate(itertools.product(*[range(n) for n in n_in])):
                    off, nb = int(index[k, 0]), int(index[k, 1])
                    if nb == 0 or off == 2**64 - 1:
                        continue
                    d = self._dec.decompress(data[off:off + nb], max_output_size=100_000_000)
                    i0 = [g + ii * ic for g, ii, ic in zip(g0, iidx, self.inner)]
                    ishape = [min(ic, s - o) for ic, s, o in zip(self.inner, self.shape, i0)]
                    arr = np.frombuffer(d, dtype=self._np_dtype)[:int(np.prod(self.inner))]
                    arr = arr.reshape(self.inner)[tuple(slice(0, n) for n in ishape)]
                    out[tuple(slice(o, o + n) for o, n in zip(i0, ishape))] = arr
            else:
                d = self._dec.decompress(data, max_output_size=200_000_000)
                arr = np.frombuffer(d, dtype=self._np_dtype)[:int(np.prod(self.grid))]
                arr = arr.reshape(self.grid)[tuple(slice(0, n) for n in gshape)]
                out[tuple(slice(o, o + n) for o, n in zip(g0, gshape))] = arr
        return out


def quat_to_R(qw, qx, qy, qz):
    n = np.linalg.norm([qw, qx, qy, qz]) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def pose7_to_T(p):
    """[x,y,z,qw,qx,qy,qz] -> 4x4 (spec: quat_order wxyz)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(p[3], p[4], p[5], p[6])
    T[:3, 3] = p[:3]
    return T


def project(kpts_w, T_w2c, fx, fy, cx, cy):
    Xc = (T_w2c[:3, :3] @ kpts_w.T).T + T_w2c[:3, 3]
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
    return u, v, z


def episode_audit(fs, ep_root: str, sample_stride: int):
    attrs = json.loads(fs.cat(f"{ep_root}/zarr.json")).get("attributes", {})
    intr = attrs.get("intrinsics", {}).get("front_1") or attrs.get("camera_intrinsics")
    if intr is None:
        raise ValueError("no intrinsics in attrs")
    if isinstance(intr, dict):
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr.get("cx"), intr.get("cy")
        W = intr.get("width") or intr.get("w") or (int(round(cx * 2)) if cx else None)
        H = intr.get("height") or intr.get("h") or (int(round(cy * 2)) if cy else None)
        if cx is None:
            cx, cy = W / 2, H / 2
    else:
        K = np.array(intr)
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        W, H = int(round(cx * 2)), int(round(cy * 2))
    head = ZArr(fs, f"{ep_root}/obs_head_pose").read_all()
    kl = ZArr(fs, f"{ep_root}/left.obs_keypoints").read_all()
    kr = ZArr(fs, f"{ep_root}/right.obs_keypoints").read_all()
    T = min(head.shape[0], kl.shape[0], kr.shape[0])
    idxs = list(range(0, T, max(1, sample_stride)))[:60]
    per_cand = {c: {"z": 0, "inb": 0, "tot": 0} for c in "ABC"}
    for t in idxs:
        Tc2w = pose7_to_T(head[t])
        cands = {"A": np.linalg.inv(Tc2w), "B": Tc2w, "C": np.eye(4)}
        for side, arr in (("l", kl), ("r", kr)):
            k = arr[t].reshape(21, 3)
            if not np.isfinite(k).all() or np.abs(k).sum() < 1e-6:
                continue
            for c, Tw2c in cands.items():
                u, v, z = project(k, Tw2c, fx, fy, cx, cy)
                ok_z = z > 0.02
                inb = ok_z & (u >= -0.3 * W) & (u < 1.3 * W) & (v >= -0.3 * H) & (v < 1.3 * H)
                per_cand[c]["z"] += int(ok_z.sum())
                per_cand[c]["inb"] += int(inb.sum())
                per_cand[c]["tot"] += 21
    best = max("ABC", key=lambda c: per_cand[c]["inb"])
    st = per_cand[best]
    return {"ep": ep_root.rsplit("/", 1)[-1], "T": T, "best_cand": best,
            "W": W, "H": H, "fx": fx,
            "z_frac": round(st["z"] / max(1, st["tot"]), 3),
            "inb_frac": round(st["inb"] / max(1, st["tot"]), 3),
            "all": {c: round(per_cand[c]["inb"] / max(1, per_cand[c]["tot"]), 3) for c in "ABC"}}


def render_overlay(fs, ep_root: str, cand: str, out_path: str):
    attrs = json.loads(fs.cat(f"{ep_root}/zarr.json")).get("attributes", {})
    intr = attrs.get("intrinsics", {}).get("front_1") or attrs.get("camera_intrinsics")
    if isinstance(intr, dict):
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr.get("cx"), intr.get("cy")
    else:
        K = np.array(intr)
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    img_arr = ZArr(fs, f"{ep_root}/images.front_1")
    head = ZArr(fs, f"{ep_root}/obs_head_pose").read_all()
    kl = ZArr(fs, f"{ep_root}/left.obs_keypoints").read_all()
    kr = ZArr(fs, f"{ep_root}/right.obs_keypoints").read_all()
    T = min(img_arr.shape[0], head.shape[0], kl.shape[0], kr.shape[0])
    t = T // 2
    jpg = img_arr.chunk(t)
    if jpg is None:
        return False
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    H_img, W_img = img.shape[:2]
    if cx is None:
        cx, cy = W_img / 2, H_img / 2
    Tc2w = pose7_to_T(head[t])
    Tw2c = {"A": np.linalg.inv(Tc2w), "B": Tc2w, "C": np.eye(4)}[cand]
    for arr, col in ((kl, (0, 0, 255)), (kr, (0, 255, 0))):
        k = arr[t].reshape(21, 3)
        u, v, z = project(k, Tw2c, fx, fy, cx, cy)
        for ui, vi, zi in zip(u, v, z):
            if zi > 0.02 and 0 <= ui < W_img and 0 <= vi < H_img:
                cv2.circle(img, (int(ui), int(vi)), max(3, W_img // 200), col, -1)
    cv2.imwrite(out_path, img)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="microagi | scale | mecka/flagship")
    ap.add_argument("--root", default="gs://foundational-research/hoi-dataset/EgoVerse/processed_v3")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--n_overlays", type=int, default=5)
    ap.add_argument("--sample_stride", type=int, default=30)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--episodes_list", default="", help="optional local file of .zarr paths")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fs = _fs()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.episodes_list:
        eps = [l.strip() for l in open(args.episodes_list) if l.strip().endswith(".zarr/") or l.strip().endswith(".zarr")]
    else:
        pre = f"{args.root}/{args.subset}".replace("gs://", "")
        eps = ["gs://" + p for p in fs.ls(pre) if p.endswith(".zarr")]
    random.seed(args.seed)
    picks = random.sample(eps, min(args.n_episodes, len(eps)))

    results, errs = [], 0
    for i, ep in enumerate(picks):
        ep = ep.rstrip("/")
        try:
            r = episode_audit(fs, ep.replace("gs://", ""), args.sample_stride)
            results.append(r)
            if i < args.n_overlays:
                render_overlay(fs, ep.replace("gs://", ""), r["best_cand"],
                               str(out / f"overlay_{r['ep'][:24]}.jpg"))
            print(f"[{i+1}/{len(picks)}] {r['ep'][:28]} cand={r['best_cand']} "
                  f"z={r['z_frac']} inb={r['inb_frac']} all={r['all']}", flush=True)
        except Exception as e:  # noqa: BLE001
            errs += 1
            print(f"[{i+1}/{len(picks)}] {ep.rsplit('/',1)[-1][:28]} ERR {type(e).__name__}: {str(e)[:100]}", flush=True)

    if results:
        inb = [r["inb_frac"] for r in results]
        cands = {}
        for r in results:
            cands[r["best_cand"]] = cands.get(r["best_cand"], 0) + 1
        summary = {"subset": args.subset, "episodes": len(results), "errors": errs,
                   "best_cand_hist": cands,
                   "inb_frac_median": float(np.median(inb)),
                   "inb_frac_p10": float(np.percentile(inb, 10)),
                   "z_frac_median": float(np.median([r["z_frac"] for r in results]))}
    else:
        summary = {"subset": args.subset, "episodes": 0, "errors": errs}
    json.dump({"summary": summary, "episodes": results},
              open(out / "audit_results.json", "w"), indent=1)
    print("SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
