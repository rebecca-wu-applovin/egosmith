#!/usr/bin/env python3
"""Generic provided-keypoints -> native-feature WDS converter (Category-2 ingestion track).

Converts datasets that ship per-frame hand keypoints / wrist poses (GT mocap, triangulated,
or pseudo-labels) into the same native-feature WDS clips that ``generate_egodex_wds.py``
produces: per-frame ``.image.jpg`` + ``.lowdim.npy`` (116-d) + ``.mano.npy`` (zeros 2x55)
+ ``.meta.json`` packed into one tar per clip, with
``descriptor.extra["native_feature_source"] = "wds_lowdim_mano_v1"`` so the quality filter
runs with ``--stages native_features``.

The dataset-specific part lives in a **spec** (YAML under ``configs/keypoint_specs/<ds>.yaml``
or a python dict) that declares:

  extractor:        one of the registered extractor plugins below
  extractor_args:   plugin kwargs (GCS layout, array keys, joint indices, unit scale, ...)
  fps:              real capture fps of the frames+poses (pass to the filter as --source_fps)
  source_id/split:  manifest bookkeeping

Each extractor returns a normalized ``EpisodeData`` dict (all world-frame, meters):

  lw_t (T,3) / rw_t (T,3)      wrist translations (MANO-joint-0-like point)
  lw_R (T,3,3) / rw_R (T,3,3)  wrist rotations (rot6d = first two columns)
  ltips / rtips (T,5,3)        thumb..little fingertip positions
  valid_l / valid_r (T,)       bool per-frame hand validity -> presence bitmask
  w2c (T,4,4)                  world->camera extrinsics
  intr (4,)                    fx, fy, cx, cy
  frames                       dict(mode="mp4", path=...) | dict(mode="jpeg_list", jpegs=[bytes])
                               | dict(mode="placeholder", width=, height=)
  task / desc                  strings for the manifest

Missing-hand convention (motion/camera-space gates in quality/accumulator.py are
presence-blind, so raw zeros would fake >9 m/s glitches):
  * presence bit is OFF for every invalid frame (the filter's projection gates skip
    non-visible hands via presence);
  * invalid spans are forward/backward filled from the nearest valid pose so
    per-frame steps stay zero across gaps;
  * a hand that is never valid in the episode is parked 0.4 m in front of the camera
    with identity rotation (keeps rot6d structurally valid and camera-space abs small).

Frame/pose length mismatches are truncated to the shorter of the two, like the EgoDex
template. All GCS access goes through gsutil; episodes are staged to a temp dir per worker.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # OpenPose-order MANO joints (exporters/mano_features.py)


# --------------------------------------------------------------------------------------
# small math helpers
# --------------------------------------------------------------------------------------

def _rot6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def _aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Rodrigues for a batch (T,3) -> (T,3,3)."""
    aa = np.asarray(aa, np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)  # (T,1)
    small = theta[..., 0] < 1e-8
    axis = np.where(theta > 1e-8, aa / np.maximum(theta, 1e-12), 0.0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = np.zeros_like(x)
    K = np.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1).reshape(-1, 3, 3)
    th = theta.reshape(-1, 1, 1)
    R = np.eye(3)[None] + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    R[small] = np.eye(3)
    return R


def _quat_to_rotmat(q: np.ndarray, order: str) -> np.ndarray:
    """(T,4) quaternion -> (T,3,3). order in {"wxyz", "xyzw"}."""
    q = np.asarray(q, np.float64)
    if order == "xyzw":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif order == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(f"bad quat order {order}")
    n = np.sqrt(w * w + x * x + y * y + z * z)
    n = np.maximum(n, 1e-12)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((q.shape[0], 3, 3), np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _invert_se3(T: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4), (T.shape[0], 1, 1))
    R = T[:, :3, :3]
    out[:, :3, :3] = np.transpose(R, (0, 2, 1))
    out[:, :3, 3] = -np.einsum("tji,tj->ti", R, T[:, :3, 3])
    return out


def _wrist_frame_from_keypoints(wrist: np.ndarray, mcp_middle: np.ndarray, mcp_index: np.ndarray) -> np.ndarray:
    """Deterministic wrist SO3 from keypoints (datasets without wrist rotation, e.g. AssemblyHands).

    x = wrist->middle-MCP; z = x cross (wrist->index-MCP); y = z cross x. Falls back to
    identity when the points are degenerate. Returns (T,3,3)."""
    T = wrist.shape[0]
    R = np.tile(np.eye(3), (T, 1, 1))
    ex = mcp_middle - wrist
    helper = mcp_index - wrist
    nx = np.linalg.norm(ex, axis=1)
    ok = nx > 1e-6
    ex[ok] /= nx[ok, None]
    ez = np.cross(ex, helper)
    nz = np.linalg.norm(ez, axis=1)
    ok = ok & (nz > 1e-6)
    ez[ok] /= nz[ok, None]
    ey = np.cross(ez, ex)
    R[ok, :, 0] = ex[ok]
    R[ok, :, 1] = ey[ok]
    R[ok, :, 2] = ez[ok]
    return R


def _fill_missing_hand(t, R, tips, valid, w2c):
    """Apply the missing-hand convention documented in the module docstring (in-place copies)."""
    T = t.shape[0]
    valid = np.asarray(valid, bool)
    t = t.copy(); R = R.copy(); tips = tips.copy()
    if valid.any():
        idx = np.arange(T)
        vidx = idx[valid]
        # nearest valid index per frame (ffill then bfill semantics via searchsorted midpoint)
        pos = np.searchsorted(vidx, idx).clip(0, len(vidx) - 1)
        prev = vidx[np.maximum(pos - 1, 0)]
        nxt = vidx[pos]
        take_prev = (idx - prev) <= np.abs(nxt - idx)
        nearest = np.where(take_prev & (prev <= idx), prev, nxt)
        nearest[valid] = idx[valid]
        t = t[nearest]; R = R[nearest]; tips = tips[nearest]
    else:
        c2w = np.linalg.inv(w2c)
        park_cam = np.array([0.0, 0.0, 0.4, 1.0])
        park_world = np.einsum("tij,j->ti", c2w, park_cam)[:, :3]
        t = park_world
        R = np.tile(np.eye(3), (T, 1, 1))
        tips = np.tile(park_world[:, None, :], (1, 5, 1))
    return t, R, tips


# --------------------------------------------------------------------------------------
# GCS helpers
# --------------------------------------------------------------------------------------

def _gsutil(args, **kw):
    # hard timeout: a single hung gsutil (observed on the lightwheel fleet: pod wedged
    # >2 h in convert) otherwise blocks its worker forever — the episode fails instead
    # and is retried by the next pod via skip-if-done.
    kw.setdefault("timeout", 1800)
    return subprocess.run(["gsutil", "-q"] + args, check=True, capture_output=True, **kw)


def gcs_list(prefix: str) -> list[str]:
    out = subprocess.run(["gsutil", "ls", prefix], check=True, capture_output=True, text=True)
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def gcs_download(url: str, dest: Path, recursive: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["-m", "cp"] + (["-r"] if recursive else []) + [url, str(dest)]
    _gsutil(args)


# --------------------------------------------------------------------------------------
# minimal zarr-v3 reader (sharding_indexed + bytes/zstd), enough for EgoVerse pose arrays
# --------------------------------------------------------------------------------------

def read_zarr3_array(array_dir: Path) -> np.ndarray:
    meta = json.loads((array_dir / "zarr.json").read_text())
    assert meta["zarr_format"] == 3 and meta["node_type"] == "array"
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["data_type"])
    outer = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    sep = meta["chunk_key_encoding"]["configuration"].get("separator", "/")
    fill = meta.get("fill_value", 0)
    codecs = meta["codecs"]
    if codecs[0]["name"] == "sharding_indexed":
        cfg = codecs[0]["configuration"]
        inner = tuple(cfg["chunk_shape"])
        inner_codecs = [c["name"] for c in cfg["codecs"]]
    else:
        # plain (unsharded) chunks: codecs apply directly, grid chunk == inner chunk
        cfg = None
        inner = outer
        inner_codecs = [c["name"] for c in codecs]
    assert inner_codecs[0] == "bytes", f"unsupported inner codecs {inner_codecs}"
    use_zstd = "zstd" in inner_codecs
    assert cfg is None or cfg.get("index_location", "end") == "end"
    if use_zstd:
        from numcodecs import Zstd
        zstd = Zstd()

    out = np.full(shape, fill, dtype=dtype)
    n_outer = [int(np.ceil(s / c)) for s, c in zip(shape, outer)]
    n_inner_per_outer = [int(np.ceil(o / i)) for o, i in zip(outer, inner)]
    n_inner_total = int(np.prod(n_inner_per_outer))

    for outer_idx in np.ndindex(*n_outer):
        chunk_path = array_dir / "c" / sep.join(str(i) for i in outer_idx)
        if not chunk_path.is_file():
            continue
        blob = chunk_path.read_bytes()
        if cfg is None:
            spans = [(0, len(blob))]  # whole chunk = one inner chunk
        else:
            index_size = n_inner_total * 16 + 4  # (offset,u64)+(nbytes,u64) per inner + crc32c
            index = blob[-index_size:-4]
            spans = [struct.unpack_from("<QQ", index, f * 16) for f in range(n_inner_total)]
        for flat, inner_idx in enumerate(np.ndindex(*n_inner_per_outer)):
            off, nb = spans[flat]
            if off == 0xFFFFFFFFFFFFFFFF:
                continue
            raw = blob[off: off + nb]
            if use_zstd:
                raw = zstd.decode(raw)
            arr = np.frombuffer(raw, dtype=dtype.newbyteorder("<")).reshape(inner)
            starts = [oi * o + ii * i for oi, o, ii, i in zip(outer_idx, outer, inner_idx, inner)]
            sl, asl = [], []
            for s, i_sz, dim in zip(starts, inner, shape):
                stop = min(s + i_sz, dim)
                sl.append(slice(s, stop)); asl.append(slice(0, stop - s))
            out[tuple(sl)] = arr[tuple(asl)]
    return out


def read_zarr3_vlen_bytes(array_dir: Path) -> list:
    """Local reader for 1-D variable_length_bytes zarr-v3 arrays (JPEG frames /
    annotation JSON). Returns list of bytes (None where the inner chunk is absent).
    Same shard layout as read_zarr3_array but payloads are vlen (u32 count header
    per shard concat is NOT used here — each inner chunk decompresses to one
    4-byte-length-prefixed record stream for annotations, or raw JPEG for images;
    callers split as needed). We return the decompressed inner chunk bytes."""
    import zstandard as zstd
    meta = json.loads((array_dir / "zarr.json").read_text())
    assert meta["zarr_format"] == 3 and meta["data_type"] == "variable_length_bytes"
    shape = tuple(meta["shape"])
    outer = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    sharded = meta["codecs"][0]["name"] == "sharding_indexed"
    inner = tuple(meta["codecs"][0]["configuration"]["chunk_shape"]) if sharded else outer
    dec = zstd.ZstdDecompressor()
    n_outer = int(np.ceil(shape[0] / outer[0]))
    n_inner_per_outer = int(np.ceil(outer[0] / inner[0]))
    out: list = [None] * shape[0]
    for k in range(n_outer):
        chunk_path = array_dir / "c" / str(k)
        if not chunk_path.is_file():
            continue
        blob = chunk_path.read_bytes()
        if sharded:
            index_size = n_inner_per_outer * 16 + 4
            index = blob[-index_size:-4]
            spans = [struct.unpack_from("<QQ", index, j * 16) for j in range(n_inner_per_outer)]
        else:
            spans = [(0, len(blob))]
        for j, (off, nb) in enumerate(spans):
            if off == 0xFFFFFFFFFFFFFFFF or nb == 0:
                continue
            i = k * outer[0] + j * inner[0]
            if i >= shape[0]:
                break
            out[i] = dec.decompress(blob[off: off + nb], max_output_size=100_000_000)
    return out


def parse_vlen_annotations(raw: bytes) -> list[dict]:
    """EgoVerse `annotations` inner chunk: u32 record-count, then per record a u32
    byte-length + JSON payload ({text,start_idx,end_idx})."""
    if not raw or len(raw) < 4:
        return []
    n = struct.unpack_from("<I", raw, 0)[0]
    pos, out = 4, []
    for _ in range(n):
        if pos + 4 > len(raw):
            break
        ln = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        try:
            out.append(json.loads(raw[pos: pos + ln]))
        except Exception:  # noqa: BLE001
            pass
        pos += ln
    return out


# --------------------------------------------------------------------------------------
# extractor plugins
# --------------------------------------------------------------------------------------

class HaworNpzExtractor:
    """OpenAoE-style HaWoR pseudo-labels: per-episode dir on GCS holding

        ego_process/ego_hands_reconstruction/hands.npz      (pred_trans/rot/hand_pose/betas/valid, R_w2c/t_w2c)
        ego_process/ego_undistorted_video/raw_video_undistorted.mp4 + undistorted_video_info.json

    Wrist translation = MANO joint-0 world (matches the pipeline's wrist_translation_semantics);
    fingertips from the MANO forward pass (run_mano_twohands), wrist rotation from pred_rot."""

    def __init__(self, gcs_prefix, hands_rel="ego_process/ego_hands_reconstruction/hands.npz",
                 video_rel="ego_process/ego_undistorted_video/raw_video_undistorted.mp4",
                 video_info_rel="ego_process/ego_undistorted_video/undistorted_video_info.json",
                 valid_thresh=0.5, use_cuda=True):
        self.gcs_prefix = gcs_prefix.rstrip("/")
        self.hands_rel = hands_rel
        self.video_rel = video_rel
        self.video_info_rel = video_info_rel
        self.valid_thresh = float(valid_thresh)
        self.use_cuda = bool(use_cuda)

    def list_episodes(self, limit=None):
        eps = [u.rstrip("/") for u in gcs_list(self.gcs_prefix + "/") if u.endswith("/")]
        eps.sort()
        return eps[:limit] if limit else eps

    def load(self, episode_url: str, work: Path) -> dict:
        name = episode_url.rstrip("/").split("/")[-1]
        gcs_download(f"{episode_url}/{self.hands_rel}", work / "hands.npz")
        gcs_download(f"{episode_url}/{self.video_info_rel}", work / "video_info.json")
        gcs_download(f"{episode_url}/{self.video_rel}", work / "video.mp4")
        d = np.load(work / "hands.npz")
        info = json.loads((work / "video_info.json").read_text())
        cam = info["cameraParams"]
        intr = np.array([cam["fx_pixels"], cam["fy_pixels"], cam["cx_pixels"], cam["cy_pixels"]], np.float32)
        fps = float(info.get("fps", 30))

        pred_trans = d["pred_trans"]      # (2,T,3) world  [0]=left [1]=right
        pred_rot = d["pred_rot"]          # (2,T,3) axis-angle world
        pred_pose = d["pred_hand_pose"]   # (2,T,45)
        pred_betas = d["pred_betas"]      # (2,T,10)
        valid = d["pred_valid"] > self.valid_thresh  # (2,T)
        T = pred_trans.shape[1]
        w2c = np.tile(np.eye(4), (T, 1, 1))
        w2c[:, :3, :3] = d["R_w2c"]
        w2c[:, :3, 3] = d["t_w2c"]

        import torch
        from lib.pipeline.hands.mano_runtime import run_mano_twohands
        with torch.inference_mode():
            joints = run_mano_twohands(
                torch.from_numpy(pred_trans).float(),
                torch.from_numpy(pred_rot).float(),
                torch.from_numpy(pred_pose).float(),
                None,
                torch.from_numpy(pred_betas).float(),
                use_cuda=self.use_cuda,
            )["joints"].cpu().numpy()  # (2,T,21,3) world
        lw_R = _aa_to_rotmat(pred_rot[0])
        rw_R = _aa_to_rotmat(pred_rot[1])
        return dict(
            lw_t=joints[0, :, 0], rw_t=joints[1, :, 0],
            lw_R=lw_R, rw_R=rw_R,
            ltips=joints[0][:, FINGERTIP_INDICES], rtips=joints[1][:, FINGERTIP_INDICES],
            valid_l=valid[0], valid_r=valid[1],
            w2c=w2c, intr=intr, fps=fps,
            frames=dict(mode="mp4", path=str(work / "video.mp4")),
            task=name, desc="", episode_name=name,
        )


class EgoverseZarr3Extractor:
    """EgoVerse processed_v3: per-clip ``<stamp>.zarr`` (zarr v3, sharded+zstd) + sibling
    ``<stamp>.mp4``. Uses left/right ``obs_wrist_pose`` (xyz+quat), ``obs_keypoints``
    (21x3 world, tips at 4,8,12,16,20), ``obs_head_pose`` as the camera c2w, and
    ``attributes.intrinsics[camera]``."""

    def __init__(self, gcs_prefix, sources=("aria",), quat_order="wxyz", camera="front_1",
                 keypoint_tip_indices=(4, 8, 12, 16, 20), pose_frame="world",
                 frames_source="mp4", min_kpt_coverage=0.0, read_annotations=False):
        self.gcs_prefix = gcs_prefix.rstrip("/")
        self.sources = list(sources)
        self.quat_order = quat_order
        self.camera = camera
        self.tip_idx = list(keypoint_tip_indices)
        assert pose_frame == "world"
        assert frames_source in ("mp4", "zarr_images")
        self.frames_source = frames_source
        # GT-coverage guard: mecka/flagship carries truncated obs_keypoints on ~47% of
        # episodes (e.g. 37 rows vs 3,046 frames). Below this coverage the episode is
        # rejected (native path is GT-derived; a mostly-empty GT episode is not).
        self.min_kpt_coverage = float(min_kpt_coverage)
        self.read_annotations = bool(read_annotations)

    def list_episodes(self, limit=None):
        eps = []
        for src in self.sources:
            for u in gcs_list(f"{self.gcs_prefix}/{src}/"):
                if u.endswith(".zarr/"):
                    eps.append(u.rstrip("/"))
            if limit and len(eps) >= limit:
                break
        eps.sort()
        return eps[:limit] if limit else eps

    def load(self, episode_url: str, work: Path) -> dict:
        stem = episode_url.rsplit("/", 1)[-1][: -len(".zarr")]
        src = episode_url.rstrip("/").split("/")[-2]
        zdir = work / "z"
        zdir.mkdir(parents=True, exist_ok=True)
        gcs_download(f"{episode_url}/zarr.json", zdir / "zarr.json")
        arrays = ["left.obs_wrist_pose", "right.obs_wrist_pose", "left.obs_keypoints",
                  "right.obs_keypoints", "obs_head_pose"]
        if self.frames_source == "zarr_images":
            arrays.append(f"images.{self.camera}")
        if self.read_annotations:
            arrays.append("annotations")
        for a in arrays:
            try:
                gcs_download(f"{episode_url}/{a}", zdir, recursive=True)
            except Exception:
                if a == "annotations":
                    continue  # optional
                raise
        if self.frames_source == "mp4":
            gcs_download(f"{episode_url.rsplit('/',1)[0]}/{stem}.mp4", work / "video.mp4")

        root = json.loads((zdir / "zarr.json").read_text())
        attrs = root["attributes"]
        fps = float(attrs.get("fps", 30))
        intr_raw = attrs.get("intrinsics", {}).get(self.camera) or attrs.get("camera_intrinsics")
        if isinstance(intr_raw, dict):
            intr = np.array([intr_raw["fx"], intr_raw["fy"], intr_raw["cx"], intr_raw["cy"]], np.float32)
        else:
            K = np.array(intr_raw, np.float64)
            intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float32)

        def arr(name):
            return read_zarr3_array(zdir / name)

        lw = arr("left.obs_wrist_pose"); rw = arr("right.obs_wrist_pose")
        lk = arr("left.obs_keypoints").reshape(-1, 21, 3)
        rk = arr("right.obs_keypoints").reshape(-1, 21, 3)
        head = arr("obs_head_pose")
        if self.min_kpt_coverage > 0:
            total = int(attrs.get("total_frames") or head.shape[0])
            cov = min(len(lk), len(rk)) / max(1, total)
            if cov < self.min_kpt_coverage:
                raise ValueError(f"gt_truncated: kpt coverage {cov:.2f} < {self.min_kpt_coverage}")
        T = min(len(lw), len(rw), len(lk), len(rk), len(head))
        lw, rw, lk, rk, head = lw[:T], rw[:T], lk[:T], rk[:T], head[:T]

        c2w = np.tile(np.eye(4), (T, 1, 1))
        c2w[:, :3, :3] = _quat_to_rotmat(head[:, 3:7], self.quat_order)
        c2w[:, :3, 3] = head[:, :3]
        w2c = _invert_se3(c2w)

        # validity: all-zero rows mean no track
        valid_l = np.abs(lk).sum(axis=(1, 2)) > 1e-8
        valid_r = np.abs(rk).sum(axis=(1, 2)) > 1e-8
        if self.frames_source == "zarr_images":
            raw = read_zarr3_vlen_bytes(zdir / f"images.{self.camera}")
            jpegs = []
            for r in raw[:T]:
                if r is None:
                    jpegs.append(None)
                    continue
                recs = None
                if len(r) >= 8:
                    try:
                        n0, l0 = struct.unpack_from("<II", r, 0)
                        if n0 == 1 and 8 + l0 <= len(r):
                            recs = r[8: 8 + l0]
                    except struct.error:
                        pass
                if recs is None:  # fall back: scan for JPEG SOI
                    j = r.find(b"\xff\xd8\xff")
                    recs = r[j:] if j >= 0 else None
                jpegs.append(recs)
            # drop trailing missing frames; interior gaps replicate previous frame
            last_ok = max((i for i, b in enumerate(jpegs) if b), default=-1)
            jpegs = jpegs[: last_ok + 1]
            for i in range(len(jpegs)):
                if jpegs[i] is None:
                    jpegs[i] = jpegs[i - 1] if i else b""
            frames = dict(mode="jpeg_list", jpegs=jpegs)
        else:
            frames = dict(mode="mp4", path=str(work / "video.mp4"))
        task = attrs.get("task") or attrs.get("task_name") or src
        desc = attrs.get("task_description", "")
        extra = {}
        if self.read_annotations and (zdir / "annotations").is_dir():
            try:
                ann_raw = read_zarr3_vlen_bytes(zdir / "annotations")
                anns = []
                for chunk in ann_raw:
                    if chunk:
                        anns.extend(parse_vlen_annotations(chunk))
                if anns:
                    extra["annotations"] = anns
                    if not attrs.get("task"):
                        task = anns[0].get("text") or task
            except Exception:  # noqa: BLE001 — annotations are best-effort
                pass
        return dict(
            lw_t=lw[:, :3], rw_t=rw[:, :3],
            lw_R=_quat_to_rotmat(lw[:, 3:7], self.quat_order),
            rw_R=_quat_to_rotmat(rw[:, 3:7], self.quat_order),
            ltips=lk[:, self.tip_idx], rtips=rk[:, self.tip_idx],
            valid_l=valid_l, valid_r=valid_r,
            w2c=w2c, intr=intr, fps=fps,
            frames=frames, extra=extra,
            task=task, desc=desc,
            episode_name=f"{src.replace('/', '_')}_{stem}",
        )


class LightwheelPoseJsonExtractor:
    """EgoVerse lightwheel: per-episode dir ``<uuid>/`` with ``pose.json`` (per-frame
    body/head/left_hand/right_hand — hands are 21 joints of {quat(wxyz keys), x,y,z},
    world frame, meters), ``head_left_camera/head_left_camera_params.json`` (per-frame
    R_w2c/t_w2c aligned 1:1 with video frames, 1-based 'frame' field, +
    undistorted_intrinsics fx/fy/cx/cy) and
    ``head_left_camera/head_left_camera_undistorted.mp4`` (1920x1456@30).
    Frame alignment verified on 6/6 episodes in the 2026-08-27 kpt audit
    (_audits/egoverse_lightwheel_kpt_audit_2026-08-27/)."""

    def __init__(self, gcs_prefix, camera="head_left_camera",
                 keypoint_tip_indices=(4, 8, 12, 16, 20)):
        self.gcs_prefix = gcs_prefix.rstrip("/")
        self.camera = camera
        self.tip_idx = list(keypoint_tip_indices)

    def list_episodes(self, limit=None):
        eps = [u.rstrip("/") for u in gcs_list(f"{self.gcs_prefix}/")
               if u.endswith("/") and not u.rstrip("/").rsplit("/", 1)[-1].endswith("h_0514")]
        eps.sort()
        return eps[:limit] if limit else eps

    @staticmethod
    def _hand_arrays(frames, key, tip_idx):
        T = len(frames)
        pos = np.zeros((T, 21, 3), np.float64)
        quat = np.zeros((T, 4), np.float64)  # wrist quat, wxyz
        valid = np.zeros(T, bool)
        for t, fr in enumerate(frames):
            joints = fr.get(key) or []
            if len(joints) != 21:
                continue
            ok = True
            for j, d in enumerate(joints):
                try:
                    pos[t, j] = (d["x"], d["y"], d["z"])
                except (KeyError, TypeError):
                    ok = False
                    break
            if not ok:
                continue
            q = joints[0].get("quat") or {}
            quat[t] = (q.get("w", 1.0), q.get("x", 0.0), q.get("y", 0.0), q.get("z", 0.0))
            valid[t] = np.abs(pos[t]).sum() > 1e-8
        return pos, quat, valid

    def load(self, episode_url: str, work: Path) -> dict:
        ep_id = episode_url.rstrip("/").rsplit("/", 1)[-1]
        cam = self.camera
        gcs_download(f"{episode_url}/pose.json", work / "pose.json")
        gcs_download(f"{episode_url}/{cam}/{cam}_params.json", work / "params.json")
        gcs_download(f"{episode_url}/{cam}/{cam}_undistorted.mp4", work / "video.mp4")
        pose = json.loads((work / "pose.json").read_text())
        prm = json.loads((work / "params.json").read_text())
        ki = prm["undistorted_intrinsics"]
        intr = np.array([ki["fx"], ki["fy"], ki["cx"], ki["cy"]], np.float32)
        pf = pose["frames"]
        cf = prm["frames"]
        T = min(len(pf), len(cf))
        w2c = np.tile(np.eye(4), (T, 1, 1))
        for t in range(T):
            w2c[t, :3, :3] = np.array(cf[t]["R_w2c"], np.float64)
            w2c[t, :3, 3] = np.array(cf[t]["t_w2c"], np.float64)
        lk, lq, valid_l = self._hand_arrays(pf[:T], "left_hand", self.tip_idx)
        rk, rq, valid_r = self._hand_arrays(pf[:T], "right_hand", self.tip_idx)
        task = ""
        try:
            gcs_download(f"{episode_url}/annotation.json", work / "annotation.json")
            ann = json.loads((work / "annotation.json").read_text())
            if isinstance(ann, dict):
                task = ann.get("task") or ann.get("task_name") or ann.get("description") or ""
            extra = {"annotation": ann} if ann else {}
        except Exception:  # noqa: BLE001
            extra = {}
        return dict(
            lw_t=lk[:, 0], rw_t=rk[:, 0],
            lw_R=_quat_to_rotmat(lq, "wxyz"), rw_R=_quat_to_rotmat(rq, "wxyz"),
            ltips=lk[:, self.tip_idx], rtips=rk[:, self.tip_idx],
            valid_l=valid_l, valid_r=valid_r,
            w2c=w2c, intr=intr, fps=30.0,
            frames=dict(mode="mp4", path=str(work / "video.mp4")),
            task=task, desc="", extra=extra,
            episode_name=ep_id,
        )


class AssemblyHandsExtractor:
    """AssemblyHands ego annotations (triangulated 3D keypoints, world frame, **millimeters**).

    One clip per (seq_name, ego camera). joint_3d gives 42x3 world_coord (right 0..20,
    left 21..41; tips are *4 rows: right [0,4,8,12,16], left [21,25,29,33,37]; wrists 20/41).
    ego_calib gives per-seq per-camera K (3x3) and per-frame per-camera w2c (3x4, mm).
    Wrist rotation is derived from keypoints (see _wrist_frame_from_keypoints).

    The rectified ego images (ego_images_rectified/...) are NOT present under the GCS
    prefix; the raw ego videos live in Assembly101/recordings/<seq_name>/<camera>_mono10bit.mp4
    (60 fps; annotation frame_idx indexes those video frames). Set
    ``assembly101_video_prefix`` to join them (frames are seek-extracted per frame_idx),
    else ``frames.mode`` falls back to ``placeholder``. NOTE: the raw HMC videos are
    unrectified while the calib intrinsics describe the rectified images, so joined pixels
    are approximate — fine for kinematic gates, not for pixel-accurate overlay."""

    R_TIPS = [0, 4, 8, 12, 16]; R_MCP = {"middle": 11, "index": 7}; R_WRIST = 20
    L_TIPS = [21, 25, 29, 33, 37]; L_MCP = {"middle": 32, "index": 28}; L_WRIST = 41

    def __init__(self, gcs_prefix, split="val", version="v1-1", images_root=None,
                 annotations_root=None, assembly101_video_prefix=None, min_valid_tips=3,
                 unit_scale=0.001, max_frames_per_clip=None, split_runs=False,
                 min_run_frames=60):
        self.gcs_prefix = gcs_prefix.rstrip("/")
        self.annotations_root = annotations_root  # local <root>/<split>/assemblyhands_* jsons
        self.split = split
        self.version = version
        self.images_root = images_root
        self.assembly101_video_prefix = (assembly101_video_prefix.rstrip("/")
                                         if assembly101_video_prefix else None)
        self.min_valid_tips = int(min_valid_tips)
        self.unit_scale = float(unit_scale)
        self.max_frames_per_clip = max_frames_per_clip
        # AssemblyHands annotations are 30 Hz on a 60 fps frame grid (frame_idx step 2), but
        # frames without hands were removed from the release, leaving index gaps that would
        # read as teleports to the step gates. split_runs=True enumerates one episode per
        # contiguous run (frame_idx step == 2) of >= min_run_frames frames instead of one
        # per (seq, camera). Measured on val: 93% of frames live in runs >= 60.
        self.split_runs = bool(split_runs)
        self.min_run_frames = int(min_run_frames)
        self._cache = {}

    def _annos(self, work: Path):
        if "data" in self._cache:
            return self._cache["data"]
        base = f"{self.gcs_prefix}/annotations/{self.split}"
        names = {k: f"assemblyhands_{self.split}_{k}_{self.version}.json"
                 for k in ("ego_data", "joint_3d", "ego_calib")}
        out = {}
        if self.annotations_root:  # local mirror (avoids concurrent-download races)
            for k, n in names.items():
                out[k] = json.loads((Path(self.annotations_root) / self.split / n).read_text())
            self._cache["data"] = out
            return out
        local = work.parent / "_ah_annotations"
        local.mkdir(parents=True, exist_ok=True)
        for k, n in names.items():
            p = local / n
            if not p.is_file():
                gcs_download(f"{base}/{n}", p)
            out[k] = json.loads(p.read_text())
        self._cache["data"] = out
        return out

    def _seq_cam_rows(self, seq: str, camera: str, data: dict) -> list:
        """Sorted (im, fkey) rows for (seq, cam) that have both joint_3d and extrinsics."""
        ims = [im for im in data["ego_data"]["images"]
               if im["seq_name"] == seq and im["camera"] == camera]
        ims.sort(key=lambda im: im["frame_idx"])
        joints = data["joint_3d"]["annotations"].get(seq, {})
        calib = data["ego_calib"]["calibration"].get(seq, {})
        cam_key = camera if camera in calib.get("intrinsics", {}) else f"{camera}_mono10bit"
        extr = calib.get("extrinsics", {})
        rows = []
        for im in ims:
            fkey = f"{im['frame_idx']:06d}"
            if fkey in joints and fkey in extr and cam_key in extr[fkey]:
                rows.append((im, fkey))
        return rows

    @staticmethod
    def _runs(rows: list) -> list:
        """Split rows into contiguous 30 Hz runs (frame_idx step == 2)."""
        runs, cur = [], []
        for row in rows:
            if cur and row[0]["frame_idx"] - cur[-1][0]["frame_idx"] != 2:
                runs.append(cur)
                cur = []
            cur.append(row)
        if cur:
            runs.append(cur)
        return runs

    def list_episodes(self, limit=None):
        # episode ref = "seq_name::camera" (whole sequence) or "seq_name::camera::rNNN"
        # (one contiguous annotated run) when split_runs is on.
        work = Path(tempfile.mkdtemp(prefix="ah_list_"))
        data = self._annos(work / "x")
        pairs = sorted({(im["seq_name"], im["camera"]) for im in data["ego_data"]["images"]})
        if not self.split_runs:
            refs = [f"{s}::{c}" for s, c in pairs]
            return refs[:limit] if limit else refs
        refs = []
        for s, c in pairs:
            runs = self._runs(self._seq_cam_rows(s, c, data))
            refs.extend(f"{s}::{c}::r{i:03d}" for i, run in enumerate(runs)
                        if len(run) >= self.min_run_frames)
        return refs[:limit] if limit else refs

    def load(self, episode_ref: str, work: Path) -> dict:
        parts = episode_ref.split("::")
        seq, camera = parts[0], parts[1]
        run_idx = int(parts[2][1:]) if len(parts) > 2 else None
        data = self._annos(work)
        calib = data["ego_calib"]["calibration"][seq]
        cam_key = camera if camera in calib["intrinsics"] else f"{camera}_mono10bit"
        K = np.array(calib["intrinsics"][cam_key], np.float64)
        intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float32)
        joints = data["joint_3d"]["annotations"][seq]

        rows = self._seq_cam_rows(seq, camera, data)
        if run_idx is not None:
            runs = self._runs(rows)
            if run_idx >= len(runs):
                raise ValueError(f"run index out of range for {episode_ref}: {len(runs)} runs")
            rows = runs[run_idx]
        if len(rows) < 3:
            raise ValueError(f"too few annotated frames for {episode_ref}: {len(rows)}")
        if self.max_frames_per_clip:
            rows = rows[: int(self.max_frames_per_clip)]
        T = len(rows)
        wc = np.stack([np.array(joints[fk]["world_coord"], np.float64) for _, fk in rows]) * self.unit_scale
        jv = np.stack([np.array(joints[fk]["joint_valid"], np.float64).reshape(-1) for _, fk in rows]) > 0.5
        w2c = np.tile(np.eye(4), (T, 1, 1))
        for i, (_, fk) in enumerate(rows):
            E = np.array(calib["extrinsics"][fk][cam_key], np.float64)  # (3,4) world->cam, mm
            w2c[i, :3, :3] = E[:, :3]
            w2c[i, :3, 3] = E[:, 3] * self.unit_scale

        def hand(tips_idx, wrist_idx, mcp):
            t = wc[:, wrist_idx]
            tips = wc[:, tips_idx]
            R = _wrist_frame_from_keypoints(t, wc[:, mcp["middle"]], wc[:, mcp["index"]])
            valid = jv[:, wrist_idx] & (jv[:, tips_idx].sum(axis=1) >= self.min_valid_tips)
            return t, R, tips, valid

        rt, rR, rtips, rvalid = hand(self.R_TIPS, self.R_WRIST, self.R_MCP)
        lt, lR, ltips, lvalid = hand(self.L_TIPS, self.L_WRIST, self.L_MCP)

        w0, h0 = rows[0][0]["width"], rows[0][0]["height"]
        frames = dict(mode="placeholder", width=int(w0), height=int(h0), count=T)
        extra = {"image_placeholder": True}
        if self.images_root:
            jpegs = []
            for im, _ in rows:
                p = Path(self.images_root) / im["file_name"]
                jpegs.append(p.read_bytes())
            frames = dict(mode="jpeg_list", jpegs=jpegs)
            extra = {}
        elif self.assembly101_video_prefix:
            import cv2
            vurl = f"{self.assembly101_video_prefix}/{seq}/{camera}_mono10bit.mp4"
            vpath = work / "ego.mp4"
            gcs_download(vurl, vpath)
            cap = cv2.VideoCapture(str(vpath))
            jpegs, last = [], None
            for im, _ in rows:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(im["frame_idx"]))
                ok, frame = cap.read()
                if ok:
                    frame = cv2.resize(frame, (int(w0), int(h0)))
                    ok2, buf = cv2.imencode(".jpg", frame)
                    last = bytes(buf) if ok2 else last
                if last is None:
                    raise ValueError(f"cannot read frame {im['frame_idx']} from {vurl}")
                jpegs.append(last)
            cap.release()
            frames = dict(mode="jpeg_list", jpegs=jpegs)
            extra = {"image_source": "assembly101_recordings_unrectified"}
        # effective annotated rate: frame_idx steps by 2 at 1/30 s -> 30 fps effective
        ts = [im["timestamp"] for im, _ in rows]
        fps = float(1.0 / np.median(np.diff(ts))) if len(ts) > 2 else 30.0
        return dict(
            lw_t=lt, rw_t=rt, lw_R=lR, rw_R=rR, ltips=ltips, rtips=rtips,
            valid_l=lvalid, valid_r=rvalid, w2c=w2c, intr=intr, fps=fps,
            frames=frames, task=seq, desc="",
            episode_name=f"{seq}_{camera}" + (f"_r{run_idx:03d}" if run_idx is not None else ""),
            extra=extra,
        )


class DexcapHdf5Extractor:
    """DexCap postprocessed robomimic HDF5 (the raw zips on GCS are 0-byte, so this is the
    only usable path). Streams demos straight from GCS via gcsfs; never downloads the file.

    Layout per ``data/demo_N``:
      obs/robot0_eef_pos  (T,6)  two wrist translations  [hand0 xyz, hand1 xyz]
      obs/robot0_eef_quat (T,8)  two wrist quaternions (xyzw assumed, scipy/robomimic default)
      glove_states        (T,63) 21x3 palm-frame joints for the gloved (right) hand,
                                 wrist at origin, tips at rows 4,8,12,16,20
      obs/agentview_image (T,84,84,3) chest-camera crop (only RGB available)
      states              (T,16) 4x4 chest-camera pose (c2w assumed)

    Caveats (recorded into descriptor.extra):
      * no camera intrinsics anywhere -> nominal intrinsic is fabricated (projection gates
        are not meaningful; kinematic gates are);
      * only one glove -> the non-gloved hand reuses mirrored palm offsets (x negated);
      * hand0/hand1 -> left/right assignment follows ``hand_order`` and is unverified.
    """

    def __init__(self, gcs_url, hand_order="left_right", quat_order="xyzw",
                 tip_rows=(4, 8, 12, 16, 20), nominal_intr=(60.0, 60.0, 42.0, 42.0),
                 image_key="obs/agentview_image", task="dexcap"):
        self.gcs_url = gcs_url
        self.hand_order = hand_order
        self.quat_order = quat_order
        self.tip_rows = list(tip_rows)
        self.nominal_intr = np.asarray(nominal_intr, np.float32)
        self.image_key = image_key
        self.task = task

    def _open(self):
        import gcsfs, h5py
        fs = gcsfs.GCSFileSystem()
        fobj = fs.open(self.gcs_url, "rb", block_size=4 * 1024 * 1024)
        return h5py.File(fobj, "r")

    def list_episodes(self, limit=None):
        with self._open() as h:
            demos = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
        return demos[:limit] if limit else demos

    def load(self, episode_ref: str, work: Path) -> dict:
        import cv2
        with self._open() as h:
            d = h[f"data/{episode_ref}"]
            eef_pos = d["obs/robot0_eef_pos"][:]
            eef_quat = d["obs/robot0_eef_quat"][:]
            glove = d["glove_states"][:].reshape(-1, 21, 3)
            imgs = d[self.image_key][:]
            states = d["states"][:]
        T = min(len(eef_pos), len(glove), len(imgs), len(states))
        eef_pos, eef_quat, glove, imgs, states = (a[:T] for a in (eef_pos, eef_quat, glove, imgs, states))

        h0_t, h1_t = eef_pos[:, 0:3], eef_pos[:, 3:6]
        h0_R = _quat_to_rotmat(eef_quat[:, 0:4], self.quat_order)
        h1_R = _quat_to_rotmat(eef_quat[:, 4:8], self.quat_order)
        if self.hand_order == "left_right":
            lt, lR, rt, rR = h0_t, h0_R, h1_t, h1_R
        else:
            lt, lR, rt, rR = h1_t, h1_R, h0_t, h0_R

        tips_palm = glove[:, self.tip_rows]                       # (T,5,3) gloved (right) hand
        tips_palm_mirror = tips_palm * np.array([-1.0, 1.0, 1.0])  # crude left approximation
        rtips = np.einsum("tij,tkj->tki", rR, tips_palm) + rt[:, None]
        ltips = np.einsum("tij,tkj->tki", lR, tips_palm_mirror) + lt[:, None]

        c2w = states.reshape(T, 4, 4)
        w2c = _invert_se3(c2w)
        jpgs = []
        for t in range(T):
            ok, buf = cv2.imencode(".jpg", imgs[t][..., ::-1])
            jpgs.append(bytes(buf))
        valid = np.ones(T, bool)
        return dict(
            lw_t=lt, rw_t=rt, lw_R=lR, rw_R=rR, ltips=ltips, rtips=rtips,
            valid_l=valid, valid_r=valid, w2c=w2c, intr=self.nominal_intr,
            fps=None, frames=dict(mode="jpeg_list", jpegs=jpgs),
            task=self.task, desc="", episode_name=f"{self.task}_{episode_ref}",
            extra={"intrinsic_nominal": True, "left_tips_mirrored": True,
                   "hand_order_assumed": self.hand_order},
        )


class OpenTouchHdf5Extractor:
    """OpenTouch (MIT, arXiv:2512.16842): 26 session .hdf5 files, each ``data/demo_NNN``
    holding rgb_images_jpeg (T,), camera_poses (T,4,4), right_hand_landmarks (T,21,3)
    world/SLAM frame (MediaPipe order: wrist 0, tips 4/8/12/16/20), timestamps (T,) ns,
    labels (1,) [low_light, hand_out_of_frame]; ``calibration/rgb`` gives pinhole
    focal/pp/size + T_device_camera. Right hand only.

    Conventions (Cat-2.5 audit, 2026-08-20): the landmark->pixel chain is
    ``p_cam = T_device_camera @ inv(camera_pose_t) @ p_world`` (inframe 1.00 on 3/3 probe
    clips) => ``w2c_t = T_device_camera @ inv(camera_pose_t)``. Landmarks are HELD-STALE on
    tracking loss (median 4.3%, p90 16% of frames identical to the previous frame), so
    per-frame validity is derived: a frame is invalid when its landmarks are bit-identical
    to the previous frame (the hold repeats garbage, semantically a dropout). Wrist rotation
    is derived from keypoints (wrist / middle-MCP 9 / index-MCP 5). fps from timestamps.

    Metric-integrity gates (W9 probe, 2026-08-25): sessions are BIMODAL in camera-space
    wrist depth — ~10 sessions at a plausible 0.50-0.88 m working distance, ~16 sessions
    at 1.3-1.9 m, beyond human arm reach for the wearer's own hand. Clips whose median
    valid wrist depth exceeds ``max_wrist_z`` are rejected at conversion
    (``depth_implausible``), as are clips the dataset itself labels ``hand_out_of_frame``.

    DO NOT INGEST (W9 reprojection gate, 2026-08-27 — scripts/inspection/
    opentouch_reproj_gate.py, verdict at egosmith_filtered/opentouch/probe/reproj_gate/):
    ``camera_poses`` are the IDENTITY in every frame of all 26 sessions, so the audited
    chain was vacuous (all pose-dependent candidates coincide) and the 'SLAM scale'
    reading of the depth bimodality was wrong — landmarks live in an undocumented
    device-like frame. Projected GT misses the visible glove by 102-234 px (median 157 px
    vs manual wrist annotations, n=5 over 3 depth-plausible sessions); no global or
    per-session rigid correction reaches the 25 px bar, and the tracker emits NON-stale
    garbage during tracking loss (landmarks freeze while the hand moves, bit-different
    every frame), which the stale-eps validity gate cannot detect. Kept for provenance;
    native ingestion requires GT repair (re-tracking), not this adapter.
    """

    def __init__(self, local_dir, sessions=None, stale_eps=1e-12, min_frames=3,
                 max_wrist_z=1.0, drop_hand_out_of_frame=True):
        self.local_dir = Path(local_dir)
        self.sessions = sessions  # optional explicit list of session stems
        self.stale_eps = float(stale_eps)
        self.min_frames = int(min_frames)
        self.max_wrist_z = float(max_wrist_z)
        self.drop_hand_out_of_frame = bool(drop_hand_out_of_frame)

    def _session_paths(self):
        if self.sessions:
            return [self.local_dir / f"{s}.hdf5" for s in self.sessions]
        return sorted(self.local_dir.glob("*.hdf5"))

    def list_episodes(self, limit=None):
        import h5py
        refs = []
        for p in self._session_paths():
            with h5py.File(p, "r") as f:
                for name in sorted(f["data"].keys()):
                    if f["data"][name]["timestamps"].shape[0] >= self.min_frames:
                        refs.append(f"{p.stem}::{name}")
            if limit and len(refs) >= limit:
                break
        return refs[:limit] if limit else refs

    def load(self, episode_ref: str, work: Path) -> dict:
        import h5py
        session, demo = episode_ref.split("::")
        with h5py.File(self.local_dir / f"{session}.hdf5", "r") as f:
            calib = f["calibration/rgb"]
            focal = float(calib["focal_length"][()])
            pp = np.asarray(calib["principal_point"], np.float64)
            T_dev_cam = np.asarray(calib["T_device_camera"], np.float64)
            clip = f["data"][demo]
            poses = np.asarray(clip["camera_poses"], np.float64)     # (T,4,4) device pose
            lms = np.asarray(clip["right_hand_landmarks"], np.float64)  # (T,21,3) world
            ts = np.asarray(clip["timestamps"], np.int64)
            jpegs = [bytes(b) for b in clip["rgb_images_jpeg"][:]]
            labels = clip["labels"][0] if "labels" in clip else None
        T = min(len(poses), len(lms), len(jpegs), len(ts))
        poses, lms, jpegs, ts = poses[:T], lms[:T], jpegs[:T], ts[:T]

        if self.drop_hand_out_of_frame and labels is not None and int(labels["hand_out_of_frame"]):
            raise ValueError("hand_out_of_frame_clip: dataset labels this clip's hand as out of frame")

        w2c = np.einsum("ij,tjk->tik", T_dev_cam, _invert_se3(poses))
        # derived per-frame validity: stale-held frames (identical to previous) are dropouts
        finite = np.isfinite(lms).all(axis=(1, 2))
        stale = np.zeros(T, bool)
        if T > 1:
            stale[1:] = np.abs(np.diff(lms, axis=0)).max(axis=(1, 2)) < self.stale_eps
        valid_r = finite & ~stale

        # metric-integrity gate: wrist depth must be within the wearer's physical reach
        wrist_cam = np.einsum("tij,tj->ti", w2c[:, :3, :3], lms[:, 0]) + w2c[:, :3, 3]
        z_valid = wrist_cam[valid_r, 2] if valid_r.any() else wrist_cam[:, 2]
        med_z = float(np.median(z_valid))
        if med_z > self.max_wrist_z:
            raise ValueError(f"depth_implausible: median wrist z {med_z:.2f} m > {self.max_wrist_z} m "
                             "(per-session SLAM scale inconsistency)")

        rw_t = lms[:, 0]
        rtips = lms[:, FINGERTIP_INDICES]
        rw_R = _wrist_frame_from_keypoints(rw_t, lms[:, 9], lms[:, 5])
        zeros_t = np.zeros((T, 3)); zeros_R = np.tile(np.eye(3), (T, 1, 1))
        fps = float(1e9 / np.median(np.diff(ts))) if T > 2 else 30.0
        extra = {"right_hand_only": True, "stale_frames": int(stale.sum())}
        if labels is not None:
            extra["session_labels"] = {"low_light": int(labels["low_light"]),
                                       "hand_out_of_frame": int(labels["hand_out_of_frame"])}
        return dict(
            lw_t=zeros_t, rw_t=rw_t,
            lw_R=zeros_R, rw_R=rw_R,
            ltips=np.zeros((T, 5, 3)), rtips=rtips,
            valid_l=np.zeros(T, bool), valid_r=valid_r,
            w2c=w2c, intr=np.array([focal, focal, pp[0], pp[1]], np.float32), fps=fps,
            frames=dict(mode="jpeg_list", jpegs=jpegs),
            task=session, desc="", episode_name=f"{session}_{demo}",
            extra=extra,
        )


def _wiyh_native_factory(**kw):
    """Lazy import: the WIYH extractor pulls streaming deps (gcsfs/h5py) on use only."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from wiyh_native_extractor import WiyhNativeExtractor
    return WiyhNativeExtractor(**kw)


EXTRACTORS = {
    "hawor_npz": HaworNpzExtractor,
    "egoverse_zarr3": EgoverseZarr3Extractor,
    "lightwheel_posejson": LightwheelPoseJsonExtractor,
    "assemblyhands_coco": AssemblyHandsExtractor,
    "dexcap_hdf5": DexcapHdf5Extractor,
    "opentouch_hdf5": OpenTouchHdf5Extractor,
    "wiyh_native": _wiyh_native_factory,
}


# --------------------------------------------------------------------------------------
# core: EpisodeData -> lowdim + tar + manifest record
# --------------------------------------------------------------------------------------

def _flag_tracker_glitches(t, R, tips, valid, jump_thresh_m=None, rot_spike_frob=None):
    """Mark isolated single-frame tracker glitches invalid (returns updated valid mask).

    Rationale: a tracker re-acquisition glitch is semantically a dropout that reports
    garbage instead of absence, so it gets the same invalid->ffill treatment as missing
    hands — we are NOT loosening the recon-tuned Stage-4 gates. Measured on EgoVerse aria:
    wrist-rotation Frobenius spikes >0.99 occur on 0.32% of frames and 100% of them
    co-locate with >=5 cm single-frame keypoint jumps.

    A frame t is flagged only on the true single-frame-outlier signature: it is far
    from BOTH neighbors AND its neighbors are close to EACH OTHER (the track "returns"),
    per metric — translation (max of wrist + fingertip steps, meters) or wrist rotation
    (Frobenius ||R[t]-R[t-1]||). The neighbor-return condition is what spares genuine
    fast sustained motion: there, t-1 and t+1 are also far apart, so nothing is flagged.
    (A both-sides-only rule without the return condition wholesale-invalidates fast
    reaches, and the ffill then collapses them into giant mid-gap jumps — measured on
    EgoVerse aria it made the funnel worse, 16% -> 3.6% segment keep.) Edge frames are
    never flagged (no return evidence)."""
    T = len(t)
    v = np.asarray(valid, bool).copy()
    if T < 3 or (not jump_thresh_m and not rot_spike_frob):
        return v
    bad = np.zeros(T, bool)
    if jump_thresh_m:
        thr = float(jump_thresh_m)
        pts = np.concatenate([t[:, None, :], tips], axis=1)          # (T,6,3)
        d_in = np.linalg.norm(pts[1:-1] - pts[:-2], axis=2).max(axis=1)   # t-1 -> t
        d_out = np.linalg.norm(pts[2:] - pts[1:-1], axis=2).max(axis=1)   # t -> t+1
        d_ret = np.linalg.norm(pts[2:] - pts[:-2], axis=2).max(axis=1)    # t-1 -> t+1
        bad[1:-1] |= (d_in > thr) & (d_out > thr) & (d_ret < thr)
    if rot_spike_frob:
        thr = float(rot_spike_frob)
        r_in = np.linalg.norm((R[1:-1] - R[:-2]).reshape(T - 2, 9), axis=1)
        r_out = np.linalg.norm((R[2:] - R[1:-1]).reshape(T - 2, 9), axis=1)
        r_ret = np.linalg.norm((R[2:] - R[:-2]).reshape(T - 2, 9), axis=1)
        bad[1:-1] |= (r_in > thr) & (r_out > thr) & (r_ret < thr)
    return v & ~bad


def _projects_visible(t, tips, w2c, intr, image_size, margin_scale=1.3):
    """Per-frame bool: does any of wrist+5 tips project inside margin_scale x image with z>0?

    Used to refine presence from "tracked" to "visible in THIS camera" — headset trackers
    (Aria) and multi-view triangulation keep labeling hands far outside the ego camera FOV,
    which trips the filter's visible-*-severe-offscreen / out-of-frame gates."""
    W, H = image_size
    pts = np.concatenate([t[:, None, :], tips], axis=1)  # (T,6,3)
    Xc = np.einsum("tij,tkj->tki", w2c[:, :3, :3], pts) + w2c[:, None, :3, 3]
    z = Xc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr[0] * Xc[..., 0] / z + intr[2]
        v = intr[1] * Xc[..., 1] / z + intr[3]
    mx, my = (margin_scale - 1.0) * W / 2, (margin_scale - 1.0) * H / 2
    ok = (z > 0.02) & (u >= -mx) & (u < W + mx) & (v >= -my) & (v < H + my)
    return ok.any(axis=1)


def _build_lowdim(ep: dict, T: int, image_size=None, presence_requires_projection=False) -> tuple[np.ndarray, np.ndarray]:
    lw_t, lw_R, ltips = _fill_missing_hand(ep["lw_t"][:T], ep["lw_R"][:T], ep["ltips"][:T], ep["valid_l"][:T], ep["w2c"][:T])
    rw_t, rw_R, rtips = _fill_missing_hand(ep["rw_t"][:T], ep["rw_R"][:T], ep["rtips"][:T], ep["valid_r"][:T], ep["w2c"][:T])
    vis_l = ep["valid_l"][:T].astype(bool)
    vis_r = ep["valid_r"][:T].astype(bool)
    if presence_requires_projection and image_size is not None:
        intr = np.asarray(ep["intr"], np.float64)
        vis_l = vis_l & _projects_visible(lw_t, ltips, ep["w2c"][:T], intr, image_size)
        vis_r = vis_r & _projects_visible(rw_t, rtips, ep["w2c"][:T], intr, image_size)
    presence = vis_l.astype(np.uint8) | (vis_r.astype(np.uint8) << 1)

    wrist_state = np.zeros((T, 18), np.float32)
    hand_state = np.zeros((T, 30), np.float32)
    wrist_state[:, 0:3] = lw_t
    wrist_state[:, 3:6] = rw_t
    for t in range(T):
        wrist_state[t, 6:12] = _rot6d(lw_R[t])
        wrist_state[t, 12:18] = _rot6d(rw_R[t])
    hand_state[:, 0:15] = ltips.reshape(T, 15)
    hand_state[:, 15:30] = rtips.reshape(T, 15)

    lowdim = np.zeros((T, 116), np.float32)
    nxt = np.minimum(np.arange(T) + 1, T - 1)
    lowdim[:, 0:18] = wrist_state
    lowdim[:, 18:48] = hand_state
    lowdim[:, 48:66] = wrist_state[nxt]
    lowdim[:, 66:96] = hand_state[nxt]
    w2c = ep["w2c"][:T].copy()
    w2c[:, 3, :] = [0, 0, 0, 1]
    lowdim[:, 96:112] = w2c.reshape(T, 16).astype(np.float32)
    lowdim[:, 112:116] = np.asarray(ep["intr"], np.float32)[None]
    return lowdim, presence.astype(np.uint8)


def _ffmpeg_bin() -> str:
    import os
    if os.environ.get("FFMPEG_BIN"):
        return os.environ["FFMPEG_BIN"]
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _iter_frames(frames: dict, work: Path, jpeg_quality: int):
    """Yield jpg bytes per frame index (ordered)."""
    if frames["mode"] == "mp4":
        out = work / "frames"
        out.mkdir(exist_ok=True)
        subprocess.run([_ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-i", frames["path"],
                        "-vsync", "0", "-q:v", str(jpeg_quality), "-start_number", "0",
                        str(out / "f%06d.jpg")], check=True, capture_output=True)
        return [p.read_bytes() for p in sorted(out.glob("f*.jpg"))]
    if frames["mode"] == "jpeg_list":
        return frames["jpegs"]
    if frames["mode"] == "placeholder":
        import cv2
        img = np.full((frames["height"], frames["width"], 3), 128, np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return [bytes(buf)] * frames["count"]
    raise ValueError(f"bad frames mode {frames['mode']}")


def _slice_ep(ep: dict, s: int, e: int) -> dict:
    """Slice the per-frame arrays of an EpisodeData dict to [s, e)."""
    out = dict(ep)
    for k in ("lw_t", "rw_t", "lw_R", "rw_R", "ltips", "rtips", "valid_l", "valid_r", "w2c"):
        out[k] = ep[k][s:e]
    if ep.get("frame_meta") is not None:
        out["frame_meta"] = ep["frame_meta"][s:e]
    return out


def _write_clip_tar(sub_id, jpgs, lowdim, presence, frames_root, outputs_root, frame_meta=None):
    """Write one sub-clip tar (template layout) and return sorted (frame_names, offsets).

    frame_meta: optional per-frame dicts merged into each .meta.json next to "presence"
    (e.g. WIYH per-frame wrist-gate pixel codes)."""
    tar_path = Path(frames_root) / f"{sub_id}.tar"
    (Path(outputs_root) / sub_id).mkdir(parents=True, exist_ok=True)
    mano = np.zeros((2, 55), np.float32)
    tmp_tar = tar_path.with_suffix(".tar.tmp")
    with tarfile.open(tmp_tar, "w") as tw:
        for t in range(len(jpgs)):
            key = f"{sub_id}_f{t:05d}"
            members = [(f"{key}.image.jpg", jpgs[t])]
            for suffix, arr in ((".lowdim.npy", lowdim[t]), (".mano.npy", mano)):
                buf = io.BytesIO(); np.save(buf, arr)
                members.append((f"{key}{suffix}", buf.getvalue()))
            meta = {"presence": int(presence[t])}
            if frame_meta is not None:
                meta.update(frame_meta[t])
            members.append((f"{key}.meta.json", json.dumps(meta).encode()))
            for name, payload in members:
                ti = tarfile.TarInfo(name); ti.size = len(payload)
                tw.addfile(ti, io.BytesIO(payload))
    tmp_tar.replace(tar_path)
    fn, fo = [], []
    with tarfile.open(tar_path, "r") as tr:
        for m in tr:
            if m.isfile() and m.name.endswith(".image.jpg"):
                fn.append(m.name); fo.append([int(m.offset_data), int(m.size)])
    order = sorted(range(len(fn)), key=lambda i: fn[i])
    return [fn[i] for i in order], [fo[i] for i in order]


def convert_episode(episode_ref: str, spec: dict, args) -> list[dict]:
    """Convert one source episode; returns one result per emitted sub-clip.

    With --segment_sec N > 0 the session is split into consecutive N-second segments
    (clip_id suffix _segNN, one tar per segment — the generate_egocentric_wds.py _ivNN
    pattern). lowdim is built per segment, so next-frame action copies clamp at segment
    boundaries exactly like the template clamps at episode end. Trailing segments
    shorter than --min_segment_sec are dropped."""
    ds = spec["dataset"]
    extractor = EXTRACTORS[spec["extractor"]](**spec.get("extractor_args", {}))
    result = {"episode": episode_ref, "status": "ok"}
    work = Path(tempfile.mkdtemp(prefix=f"{ds}_", dir=args.frames_root))
    try:
        ep = extractor.load(episode_ref, work)
        if spec.get("recenter_world"):
            # Re-gauge: translate the world origin to the first-frame camera centre.
            # Physically a no-op (rigid translation of every world quantity), but the
            # quality filter's camera_translation_step reads the w2c translation vector,
            # which for a far world origin (|C|~43 m on egoverse/scale) swings by
            # |dR| x |C| under pure head rotation — 22 cm/frame of fake camera motion.
            R0, t0 = ep["w2c"][0, :3, :3], ep["w2c"][0, :3, 3]
            C0 = -(R0.T @ t0)
            for k in ("lw_t", "rw_t"):
                ep[k] = ep[k] - C0
            for k in ("ltips", "rtips"):
                ep[k] = ep[k] - C0
            w2c = ep["w2c"].copy()
            w2c[:, :3, 3] = w2c[:, :3, 3] + np.einsum("tij,j->ti", w2c[:, :3, :3], C0)
            ep["w2c"] = w2c
        clip_id = f"{ds}_{re.sub(r'[^A-Za-z0-9_.-]', '-', ep['episode_name'])}"
        result["clip_id"] = clip_id
        segmented = float(args.segment_sec or 0) > 0
        first_tar = Path(args.frames_root) / (f"{clip_id}_seg00.tar" if segmented else f"{clip_id}.tar")
        if args.resume and first_tar.is_file():
            return [{**result, "status": "skipped"}]
        jpgs = _iter_frames(ep["frames"], work, args.jpeg_quality)
        T = min(len(jpgs), ep["lw_t"].shape[0], ep["w2c"].shape[0])
        if T < 3:
            raise ValueError(f"too few frames: poses={ep['lw_t'].shape[0]} imgs={len(jpgs)}")
        image_size = None
        try:
            import cv2
            img0 = cv2.imdecode(np.frombuffer(jpgs[0], np.uint8), cv2.IMREAD_COLOR)
            image_size = (int(img0.shape[1]), int(img0.shape[0]))
        except Exception:
            pass
        # Pre-clean tracker re-acquisition glitches ONCE on the full session (so segment
        # boundaries keep real neighbors), before segmentation + ffill. See
        # _flag_tracker_glitches for the rationale + measured defaults.
        if spec.get("glitch_jump_thresh_m") or spec.get("glitch_rot_spike_frob"):
            for tk, Rk, tipk, vk in (("lw_t", "lw_R", "ltips", "valid_l"),
                                     ("rw_t", "rw_R", "rtips", "valid_r")):
                ep[vk] = _flag_tracker_glitches(
                    ep[tk][:T], ep[Rk][:T], ep[tipk][:T], ep[vk][:T],
                    spec.get("glitch_jump_thresh_m"), spec.get("glitch_rot_spike_frob"))
        fps = float(ep.get("fps") or spec.get("fps") or 30.0)
        seg_len = max(3, int(round(float(args.segment_sec or 0) * fps))) if segmented else T
        min_len = max(3, int(round(float(args.min_segment_sec) * fps)))
        ivs = None
        if getattr(args, "_intervals", None) is not None:
            for key in (ep.get("episode_name", ""), episode_ref.rstrip("/").rsplit("/", 1)[-1],
                        Path(episode_ref.rstrip("/")).stem):
                if key in args._intervals:
                    ivs = args._intervals[key]
                    break
            if ivs is None:
                return [{**result, "status": "skipped", "error": "no stage1 intervals"}]
        if ivs is not None:
            # Stage-1 kept intervals as the segment gate: convert only kept frame spans,
            # splitting each span by --segment_sec. Sub-clips keep the _segNN naming.
            bounds = []
            for s0, e0 in ivs:
                s0, e0 = max(0, int(s0)), min(T, int(e0))
                step = seg_len if segmented else (e0 - s0)
                for s in range(s0, e0, max(3, step)):
                    e = min(s + step, e0)
                    if (e - s) >= min_len:
                        bounds.append((s, e))
        elif segmented:
            bounds = [(s, min(s + seg_len, T)) for s in range(0, T, seg_len)]
            bounds = [(s, e) for s, e in bounds if (e - s) >= min_len]
        else:
            bounds = [(0, T)]
        segmented = segmented or ivs is not None
        results = []
        for k, (s, e) in enumerate(bounds):
            sub_id = f"{clip_id}_seg{k:02d}" if segmented else clip_id
            sub_ep = _slice_ep(ep, s, e)
            lowdim, presence = _build_lowdim(
                sub_ep, e - s, image_size=image_size,
                presence_requires_projection=bool(spec.get("presence_requires_projection", False)))
            fn, fo = _write_clip_tar(sub_id, jpgs[s:e], lowdim, presence,
                                     args.frames_root, args.outputs_root,
                                     frame_meta=sub_ep.get("frame_meta"))
            r = {"episode": episode_ref, "status": "ok", "clip_id": sub_id,
                 "frames": e - s, "task": ep.get("task", ""), "desc": ep.get("desc", ""),
                 "fps": fps, "presence_ratio": float((presence > 0).mean()),
                 "extra": {**ep.get("extra", {}),
                           **({"session_id": clip_id, "segment_index": k,
                               "segment_frame_range": [int(s), int(e)],
                               "segment_sec": float(args.segment_sec)} if segmented else {})},
                 "_frame_names": fn, "_frame_offsets": fo}
            if not np.isfinite(lowdim).all():
                r["nonfinite_frames"] = int((~np.isfinite(lowdim).all(axis=1)).sum())
            results.append(r)
        if not results:
            raise ValueError(f"no segments >= min_segment_sec ({args.min_segment_sec}s) in {T} frames")
        return results
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            result["error"] += " :: " + error.stderr.decode("utf8", "replace")[-300:]
        return [result]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _star(a):
    return convert_episode(*a)


def write_manifest(results, spec, args) -> int:
    from lib.pipeline.clips.clip_manifest import write_clip_manifest, ClipManifestRecord
    from lib.pipeline.datasets.descriptors import ClipDescriptor
    expanded = []
    for r in results:
        if r["status"] == "ok":
            expanded.append(r)
        elif r["status"] == "skipped" and "clip_id" in r:
            # resume: recover sub-clips from existing tars (plain or _segNN)
            base = r["clip_id"]
            tars = sorted(Path(args.frames_root).glob(f"{base}_seg*.tar")) or \
                   [p for p in [Path(args.frames_root) / f"{base}.tar"] if p.is_file()]
            for p in tars:
                expanded.append({**r, "status": "ok", "clip_id": p.stem,
                                 "_frame_names": None, "_frame_offsets": None})
    recs = []
    for r in expanded:
        if "clip_id" not in r:
            continue
        tar_path = Path(args.frames_root) / f"{r['clip_id']}.tar"
        fn, fo = r.get("_frame_names"), r.get("_frame_offsets")
        if fn is None:
            fn, fo = [], []
            with tarfile.open(tar_path, "r") as tr:
                for m in tr:
                    if m.isfile() and m.name.endswith(".image.jpg"):
                        fn.append(m.name); fo.append([int(m.offset_data), int(m.size)])
            order = sorted(range(len(fn)), key=lambda i: fn[i])
            fn = [fn[i] for i in order]; fo = [fo[i] for i in order]
        extra = {"adapter": "keypoints_wds", "native_feature_source": "wds_lowdim_mano_v1",
                 "lowdim_schema": spec.get("lowdim_schema", f"{spec['dataset']}_keypoints_world_v1"),
                 "mano_schema": "zeros_2x55", "dataset_name": spec.get("source_id", spec["dataset"]),
                 "keypoint_spec": spec.get("_spec_path", ""), "task": r.get("task", "")}
        extra.update(r.get("extra", {}))
        desc = ClipDescriptor.from_tar_shard(
            clip_id=r["clip_id"], clip_name=r["clip_id"],
            root_dir=str(Path(args.frames_root).resolve()),
            seq_folder=str((Path(args.outputs_root) / r["clip_id"]).resolve()),
            shard_path=str(tar_path.resolve()), frame_names=fn, frame_offsets=fo,
            extra=extra)
        recs.append(ClipManifestRecord(clip_id=r["clip_id"], source_id=spec.get("source_id", spec["dataset"]),
                                       split=spec.get("split", args.split), descriptor=desc,
                                       group_id=r.get("task", "")))
    write_clip_manifest(recs, args.manifest_out)
    return len(recs)


def load_spec(path: str) -> dict:
    import yaml
    spec = yaml.safe_load(Path(path).read_text())
    spec["_spec_path"] = str(Path(path).resolve())
    assert spec["extractor"] in EXTRACTORS, f"unknown extractor {spec['extractor']}"
    return spec


def build_parser():
    p = argparse.ArgumentParser(description="provided-keypoints -> native-feature WDS clips")
    p.add_argument("--spec", required=True, help="YAML spec (configs/keypoint_specs/<ds>.yaml)")
    p.add_argument("--frames_root", required=True, help="output dir for per-episode WDS tars")
    p.add_argument("--outputs_root", required=True, help="seq_folder root (native path needs no stage outputs)")
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--jpeg_quality", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--episodes", default=None, help="comma-separated explicit episode refs (overrides listing)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--split", default="train")
    p.add_argument("--intervals_json", default=None,
                   help="JSON {episode_key: [[start_frame,end_frame],...]} — convert only these "
                        "source-frame spans (Stage-1 kept intervals as segment gate). Episodes "
                        "absent from the map are skipped. Keys match episode_name or the episode "
                        "ref basename/stem.")
    p.add_argument("--segment_sec", type=float, default=0.0,
                   help=">0: split each session into consecutive N-second sub-clips (_segNN tars)")
    p.add_argument("--min_segment_sec", type=float, default=2.0,
                   help="drop trailing segments shorter than this (only with --segment_sec)")
    return p


def main():
    args = build_parser().parse_args()
    spec = load_spec(args.spec)
    args._intervals = None
    if args.intervals_json:
        args._intervals = json.loads(Path(args.intervals_json).read_text())
        print(f"[intervals] {len(args._intervals)} episodes gated by stage-1 intervals", flush=True)
    for d in (args.frames_root, args.outputs_root):
        Path(d).mkdir(parents=True, exist_ok=True)
    extractor = EXTRACTORS[spec["extractor"]](**spec.get("extractor_args", {}))
    if args.episodes:
        episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]
    else:
        episodes = extractor.list_episodes(limit=args.limit)
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"[{spec['dataset']}] episodes to convert: {len(episodes)}", flush=True)

    started = time.perf_counter()
    jobs = [(e, spec, args) for e in episodes]

    def _log(i, batch):
        head = batch[0]
        print(f"[{i+1}/{len(jobs)}] {head.get('clip_id', head['episode'])} {head['status']}"
              + (f" x{len(batch)} segs" if len(batch) > 1 else "")
              + (f" :: {head['error']}" if head["status"] == "failed" else ""), flush=True)

    results = []
    if args.workers <= 1:
        for i, j in enumerate(jobs):
            batch = convert_episode(*j)
            results.extend(batch); _log(i, batch)
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            for i, batch in enumerate(pool.imap_unordered(_star, jobs, chunksize=1)):
                results.extend(batch); _log(i, batch)
    n = write_manifest(results, spec, args)
    failed = [r for r in results if r["status"] == "failed"]
    report = {"dataset": spec["dataset"], "spec": spec["_spec_path"], "episodes": len(jobs),
              "segment_sec": float(args.segment_sec or 0),
              "converted_ok": sum(1 for r in results if r["status"] == "ok"),
              "skipped": sum(1 for r in results if r["status"] == "skipped"),
              "failed": len(failed), "manifest_records": n,
              "fps": spec.get("fps"),
              "results": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
              "elapsed_sec": time.perf_counter() - started}
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    print("KEYPOINTS_CONVERT_DONE" if not failed else "KEYPOINTS_CONVERT_DONE_WITH_FAILURES", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
