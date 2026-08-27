#!/usr/bin/env python3
"""WIYH native-GT extractor for generate_keypoints_wds.py (locked-tier ingestion).

Episode = one worldcode sample member (~22 s @ 10 fps chest KB4 fisheye) of a
wrist-LOCKED session (gate census: both hands >=80% frames with eef-vs-hand-mask
< 30 px). Hand GT chain per frame (all chest-frame; WIYH has no odometry, so
world := chest frame and w2c is the constant chest-cam extrinsic):

    p_world = R_eef @ (R_gl @ p_local + t_gl) + t_eef

with the 25-joint wrist-local skeleton (action/{side}_hand_glove, 50 Hz), the
per-frame wrist SE3 (pose/{side}_eef/feedback/pose_in_chest) and the per-device-day
glove->eef extrinsic (R_gl, t_gl) solved by the vision anchor pass
(scripts/inspection/wiyh_anchor_pilot.py).

Frames: the chest fisheye's forward pinhole crop almost never contains the hands
(they sit 60-90 deg off-axis, and the mount roll varies by rig: bottom of image
on some devices, left/right edges on others). Each sample therefore gets an
AUTO-AIMED virtual pinhole: optical axis = mean gate-passing wrist direction,
roll chosen so the left/right hands span the image x-axis. The per-sample
rotation is folded into w2c (camera pose is per-frame data in the lowdim
contract), and the fisheye->pinhole remap uses the same rotation, so projection
and pixels stay consistent by construction.

Per-frame validity (-> presence bit) = eef conf==1 AND stream/frame |dt|<=25 ms
AND wrist-gate dist < 30 px against the shipped hand mask. Raw gate codes ship
in every frame's .meta.json as {"gate_px": {"l": int, "r": int}} (px; -1 conf/dt
fail, -2 unprojectable, -3 no mask).

Finger-level accuracy is anchor-solve limited (~35-65 px) -> the tier is TAGGED:
record metadata carries finger_quality = "approximate_35_65px" (patch applied by
the shard runner; see scripts/build/wiyh_native_convert.sh).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from wiyh_gate_census import (  # noqa: E402
    GATE_PX, SampleStreams, gate_dists, load_sessions, match_masks, parse_member,
    stream_sample)

TIPS = [3, 8, 13, 18, 23]  # thumb, index, middle, ring, little
MIDDLE_MCP, INDEX_MCP, WRIST = 9, 4, 24
FINGER_QUALITY = "approximate_35_65px"


def clip_base(sample_name: str) -> str:
    import re
    base = sample_name[len("worldcode_"):] if sample_name.startswith("worldcode_") else sample_name
    return re.sub(r"[^A-Za-z0-9_\-]", "-", base.replace("_s0_vlta_reorg_sample", ""))


class WiyhNativeExtractor:
    def __init__(self, index_dir: str, census_path: str, extrinsics_path: str,
                 out_width: int = 456, out_height: int = 256, pinhole_focal: float = 150.0,
                 jpeg_quality: int = 88, gate_px: float = GATE_PX):
        self.index_dir = Path(index_dir)
        self.out_w, self.out_h = int(out_width), int(out_height)
        self.focal = float(pinhole_focal)
        self.jq = int(jpeg_quality)
        self.gate_px = float(gate_px)
        self.extr = json.loads(Path(extrinsics_path).read_text())
        self.locked = {}
        for l in open(census_path):
            row = json.loads(l)
            if row.get("locked"):
                self.locked[row["session"]] = {"scene": row["scene"], "dev": row["dev"],
                                               "date": row["date"]}
        self._members = None
        self._parts = {f.stem.split(".")[0]: json.loads(f.read_text())
                       for f in self.index_dir.glob("*.parts.json")}

    # -------------------------------------------------------------- listing
    def _load_members(self):
        if self._members is not None:
            return
        self._members = {}
        sess = load_sessions(self.index_dir)
        for session, members in sess.items():
            if session not in self.locked:
                continue
            dd = f"{members[0]['dev']}_{members[0]['date']}"
            if dd not in self.extr or self.extr[dd].get("status") != "pass":
                continue
            for m in members:
                if m["size"] <= 100_000_000:  # stub exports
                    continue
                self._members[m["base"]] = m

    def list_episodes(self, limit=None):
        self._load_members()
        eps = sorted(self._members)
        return eps[:limit] if limit else eps

    # -------------------------------------------------------------- loading
    def load(self, episode_ref: str, work: Path) -> dict:
        import cv2
        import gcsfs
        from scipy.spatial.transform import Rotation as Rt
        from generate_keypoints_wds import _wrist_frame_from_keypoints

        self._load_members()
        m = self._members[episode_ref]
        dd = f"{m['dev']}_{m['date']}"
        ex = self.extr[dd]
        fs = gcsfs.GCSFileSystem()
        h5, masks, jpgs_raw = stream_sample(fs, self._parts[m["scene"]], m, want_jpgs=True)
        if h5 is None:
            raise ValueError("no dataset.hdf5 in sample")
        ss = SampleStreams(h5)
        masks = match_masks(ss, masks)
        dists = gate_dists(ss, masks)
        T = ss.n

        # ---- hands -> world (= chest frame)
        out = {}
        valid = {}
        for side in ("left", "right"):
            Rg = np.array(ex[side]["R"], np.float64)
            tg = np.array(ex[side]["t"], np.float64)
            eef = ss.eef[side]
            Re = Rt.from_quat(eef[:, 3:]).as_matrix()          # (T,3,3)
            te = eef[:, :3]
            loc = ss.pts[side]                                  # (T,25,3) wrist-local
            pw = np.einsum("tij,tkj->tki", Re, loc @ Rg.T + tg) + te[:, None]
            wrist = pw[:, WRIST]
            tips = pw[:, TIPS]
            R = _wrist_frame_from_keypoints(wrist, pw[:, MIDDLE_MCP], pw[:, INDEX_MCP])
            d = dists[side]
            valid[side] = np.array([(0 <= d[i] < self.gate_px) for i in range(T)], bool)
            out[side] = (wrist, R, tips)

        # ---- per-sample auto-aimed virtual pinhole (see module docstring)
        R_vc = self._auto_view(ss, out, valid)                  # columns: virt axes in cam
        newK = np.array([[self.focal, 0, self.out_w / 2.0],
                         [0, self.focal, self.out_h / 2.0], [0, 0, 1]], np.float64)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(
            ss.K, ss.D, R_vc.T, newK, (self.out_w, self.out_h), cv2.CV_16SC2)
        jpegs = []
        for name in ss.frame_names:
            raw = jpgs_raw.get(name)
            if raw is None:
                raise ValueError(f"missing frame jpg {name}")
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
            ok, buf = cv2.imencode(".jpg", und, [cv2.IMWRITE_JPEG_QUALITY, self.jq])
            if not ok:
                raise ValueError(f"jpeg encode failed {name}")
            jpegs.append(bytes(buf))
        del jpgs_raw

        # ---- w2c: world(=chest) -> virtual pinhole cam
        w2c1 = np.eye(4)
        w2c1[:3, :3] = R_vc.T @ ss.R_ext.T
        w2c1[:3, 3] = -w2c1[:3, :3] @ ss.t_ext
        w2c = np.tile(w2c1, (T, 1, 1))

        fps = float(1000.0 / np.median(np.diff(ss.ts))) if T > 2 else 10.0

        # ---- shipped subtask annotations (L1 seed + segmentation seed)
        task, desc, subtasks = self._annotations(ss)

        frame_meta = [{"gate_px": {"l": int(dists["left"][i]), "r": int(dists["right"][i])}}
                      for i in range(T)]
        return dict(
            lw_t=out["left"][0], rw_t=out["right"][0],
            lw_R=out["left"][1], rw_R=out["right"][1],
            ltips=out["left"][2], rtips=out["right"][2],
            valid_l=valid["left"], valid_r=valid["right"],
            w2c=w2c,
            intr=np.array([self.focal, self.focal, self.out_w / 2.0, self.out_h / 2.0],
                          np.float32),
            fps=fps,
            frames=dict(mode="jpeg_list", jpegs=jpegs),
            frame_meta=frame_meta,
            task=task, desc=desc, episode_name=clip_base(m["base"]),
            extra={"session": m["session"], "device_day": dd, "scene": m["scene"],
                   "anchor_extrinsic": dd, "gt_mode": "wiyh_native_25joint",
                   "finger_quality": FINGER_QUALITY,
                   "wrist_gate_px": self.gate_px,
                   "gate_frac": {s: float(valid[s].mean()) for s in ("left", "right")},
                   "view_R_virt2cam": np.round(R_vc, 6).tolist(),
                   "subtasks": subtasks},
        )

    def _auto_view(self, ss, out, valid):
        """Virtual-camera rotation (columns = virt axes in fisheye-cam coords):
        z aims at the mean gate-passing wrist direction; x spans left->right hand."""
        dirs = {}
        for side in ("left", "right"):
            if valid[side].sum() < 5:
                continue
            Xc = np.einsum("ij,tj->ti", ss.R_ext.T, out[side][0][valid[side]] - ss.t_ext)
            n = np.linalg.norm(Xc, axis=1)
            keep = n > 1e-6
            if keep.sum() < 5:
                continue
            dirs[side] = (Xc[keep] / n[keep, None]).mean(0)
        if not dirs:
            return np.eye(3)
        z = np.sum(list(dirs.values()), axis=0)
        z = z / np.linalg.norm(z)
        if len(dirs) == 2:
            x_raw = dirs["right"] - dirs["left"]
        else:
            x_raw = np.array([1.0, 0.0, 0.0])
        x = x_raw - np.dot(x_raw, z) * z
        nx = np.linalg.norm(x)
        if nx < 1e-6:  # degenerate: hands dead-center; keep camera x
            x = np.array([1.0, 0.0, 0.0]) - z[0] * z
            nx = np.linalg.norm(x)
        x = x / nx
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)

    def _annotations(self, ss: SampleStreams):
        """Pull annotation/task_description rows; map timestamps to frame indices."""
        task = desc = ""
        subs = []
        try:
            ann = ss.f["annotation/task_description"][:]
        except Exception:  # noqa: BLE001
            return task, desc, subs

        def dec(v):
            return v.decode("utf-8", "replace").strip() if isinstance(v, bytes) else str(v)

        ts = ss.ts
        for row in ann:
            zh = dec(row["atomic_task_description"])
            en = dec(row["atomic_task_description_en"])
            t_zh = dec(row["task_description"])
            t_en = dec(row["task_description_en"])
            s_f = int(np.searchsorted(ts, float(row["start_timestamp"])))
            e_f = int(np.searchsorted(ts, float(row["end_timestamp"])))
            s_f, e_f = max(0, min(s_f, ss.n)), max(0, min(e_f, ss.n))
            if e_f <= s_f:
                continue
            subs.append({"s_frame": s_f, "e_frame": e_f, "zh": zh, "en": en,
                         "status": dec(row["status"])})
            if not task:
                task, desc = (t_en or t_zh), (en or zh)
        return task, desc, subs
