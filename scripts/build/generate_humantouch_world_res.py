#!/usr/bin/env python3
"""Convert HumanTouch (Xspark) MANUS-glove GT into filter-ready seq_folders.

GT-mode (Cat-2.5 -> GT-path): the recon track kept 0 (tactile gloves are
detector-blind); instead we project the shipped 25-joint MANUS world-frame
skeletons through per-MOUNT-BLOCK solved camera_from_head_tracker extrinsics
(Stage-2 anchor pass) and fit MANO to the GT joints.

Coordinate handling (Stage-1-verified chain):
  p_tracker = R(q_head)^T (p_world - t_head)     # pose.head = tracker->world, xyzw
  p_cam     = M p_tracker + t                    # M IMPROPER (det=-1, unity y-flip)
MANUS world is a LEFT-handed (Unity) representation. All output artifacts are
expressed in the MIRRORED world  w'' = F_w @ w,  F_w = diag(1,-1,1), which is
right-handed and related to the OpenCV camera by a PROPER rotation:
  R_cam_from_w'' = M R(q_head)^T F_w            (det +1)
  p_cam = R_cam_from_w'' (p'' - F_w t_head) + t
In w'' the dataset's left/right hands have their physical chirality, so left
fits left-MANO (via the standard mirror trick) and right fits right-MANO.

Joint mapping MANUS-25 -> MANO-21 (drop the four finger metacarpals; thumb
metacarpal doubles as a real CMC):
  [0, 1,2,3,4, 6,7,8,9, 11,12,13,14, 16,17,18,19, 21,22,23,24]

Two phases (MANO fitting is extrinsic-INDEPENDENT, so it runs before/while the
anchor pass completes):
  fit        per-episode GPU MANO fit at the converted 15 fps rows ->
             staging npz {WORK}/gt_fit/<task>_<ep>.npz
  finalize   per-subclip contract artifacts from staging + per-block extrinsics
             (world_space_res.pth, SLAM c2w npz, est_focal, markers, manifest)
  smoke      overlay render of fitted MANO joints on converted tar frames

Output contract per clip identical to generate_show3d_world_res.py /
generate_dexcap_world_res.py:
  outputs_root/<clip_id>/world_space_res.pth [trans(2,T,3), rot(2,T,3),
    hand_pose(2,T,45), betas(2,T,10), valid(2,T)] (0=left, 1=right)
  outputs_root/<clip_id>/SLAM/hawor_slam_w_scale_0_<T-1>.npz (per-frame c2w in
    w'', scale 1, intrinsics at the 456x256 frame scale)
  est_focal.txt, tracks_0_<T-1>/.humantouch_gt, infiller done marker
Frames are NOT rewritten: the Phase-B tars on GCS are reused; the manifest is
the Phase-B manifest with seq_folder repointed (+ gt provenance extras).
"""

from __future__ import annotations

import argparse
import json
import queue as queue_mod
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts" / "build")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

WORK = Path("/root/w7_full/humantouch")
SRC = "gs://foundational-research/hoi-dataset/Xspark-HumanTouch"
CALIB = WORK / "humantouch_calibration.json"
FW, FH = 456, 256
W0, H0 = 1920, 1080
F_W = np.diag([1.0, -1.0, 1.0])
_MIRROR = np.diag([-1.0, 1.0, 1.0])
MANUS_TO_MANO21 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14,
                   16, 17, 18, 19, 21, 22, 23, 24]

PQ_COLS = ["observation.human.pose.head", "observation.human.pose.valid",
           "observation.human.hand_skeleton.left.position",
           "observation.human.hand_skeleton.left.valid",
           "observation.human.hand_skeleton.right.position",
           "observation.human.hand_skeleton.right.valid"]


# ---------------------------------------------------------------- inventory

def load_clip_index(manifest_dir: Path) -> dict[str, list[dict]]:
    """{session: [{clip_id, iv, start_frame, T, manifest_row}, ...]} from the
    Phase-B manifests (downloaded locally)."""
    imeta = {}
    for line in open(WORK / "index.full.jsonl"):
        d = json.loads(line)
        imeta[d["session"]] = d
    out: dict[str, list[dict]] = {}
    for mp in sorted(manifest_dir.glob("*.manifest.jsonl")):
        for line in open(mp):
            r = json.loads(line)
            cid = r["clip_id"]                     # X001_000001_iv00
            session, ivs = cid.rsplit("_iv", 1)
            iv = int(ivs)
            meta = imeta[session]
            out.setdefault(session, []).append(dict(
                clip_id=cid, iv=iv,
                start_frame=meta["intervals"][iv]["start_frame"],
                T=len(r["descriptor"]["frame_names"]),
                row=r))
    for session in out:
        out[session].sort(key=lambda c: c["iv"])
    return out


def episode_rows(clips: list[dict], fps_ratio: int = 4) -> np.ndarray:
    rows = []
    for c in clips:
        rows.extend(c["start_frame"] + fps_ratio * np.arange(c["T"]))
    return np.asarray(sorted(set(rows)), dtype=np.int64)


def fetch_gt(task: str, ep_key: str, rows: np.ndarray):
    """Column-projected GT read. Returns dict of arrays indexed like `rows`."""
    import gcsfs
    import pyarrow.parquet as pq
    fs = gcsfs.GCSFileSystem()
    uri = f"{SRC}/{task}/data/chunk-000/episode_{ep_key}.parquet"
    with fs.open(uri.replace("gs://", ""), "rb") as f:
        t = pq.read_table(f, columns=PQ_COLS)
    n = t.num_rows
    rows = rows[rows < n]
    d = {c: t.column(c).to_pylist() for c in PQ_COLS}
    T = len(rows)
    head = np.full((T, 7), np.nan)
    jl = np.full((T, 25, 3), np.nan)
    jr = np.full((T, 25, 3), np.nan)
    vl = np.zeros((T,), bool)
    vr = np.zeros((T,), bool)
    for i, r in enumerate(rows):
        if not d["observation.human.pose.valid"][r][0]:
            continue
        head[i] = np.asarray(d["observation.human.pose.head"][r], float)
        L = np.asarray(d["observation.human.hand_skeleton.left.position"][r], float)
        R = np.asarray(d["observation.human.hand_skeleton.right.position"][r], float)
        lv = np.asarray(d["observation.human.hand_skeleton.left.valid"][r], bool)
        rv = np.asarray(d["observation.human.hand_skeleton.right.valid"][r], bool)
        jl[i], jr[i] = L, R
        vl[i] = bool(lv[MANUS_TO_MANO21].all() and np.isfinite(L).all())
        vr[i] = bool(rv[MANUS_TO_MANO21].all() and np.isfinite(R).all())
    return dict(rows=rows, head=head, joints_left=jl, joints_right=jr,
                valid_left=vl, valid_right=vr)


# ---------------------------------------------------------------- fit phase

def fit_episode(gt: dict, device: str, num_iters: int) -> dict:
    from generate_show3d_world_res import fit_side  # centroid fitter + mirror

    T = len(gt["rows"])
    out = dict(rows=gt["rows"], head=gt["head"].astype(np.float32),
               trans=np.zeros((2, T, 3), np.float32),
               rot=np.zeros((2, T, 3), np.float32),
               pose=np.zeros((2, T, 45), np.float32),
               betas=np.zeros((2, 10), np.float32),
               valid=np.zeros((2, T), np.float32))
    for side, hidx in (("left", 0), ("right", 1)):
        v = gt[f"valid_{side}"]
        idx = np.nonzero(v)[0]
        if len(idx) < 2:
            continue
        # MANUS world -> MANO21 targets in mirrored world w''
        tgt = gt[f"joints_{side}"][idx][:, MANUS_TO_MANO21, :] @ F_W.T
        trans, rot, pose45, betas = fit_side(tgt, side, device, num_iters)
        out["trans"][hidx, idx] = trans
        out["rot"][hidx, idx] = rot
        out["pose"][hidx, idx] = pose45
        out["betas"][hidx] = betas
        out["valid"][hidx, idx] = 1.0
    return out


def cmd_fit(args):
    calib = json.load(open(CALIB))
    clip_index = load_clip_index(Path(args.manifest_dir))
    sessions = sorted(clip_index)
    if args.sessions:
        want = set(args.sessions.split(","))
        sessions = [s for s in sessions if s in want]
    if args.shard:
        i, n = (int(x) for x in args.shard.split(":"))
        sessions = sessions[i::n]
    stage_dir = Path(args.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in sessions if not (stage_dir / f"{s}.npz").exists()]
    print(f"{len(todo)}/{len(sessions)} episodes to fit", flush=True)

    q: queue_mod.Queue = queue_mod.Queue(maxsize=args.prefetch)
    stop = object()

    def producer(sub):
        for session in sub:
            task, ep_key = session.split("_")
            try:
                rows = episode_rows(clip_index[session])
                gt = fetch_gt(task, ep_key, rows)
                q.put((session, gt, None))
            except Exception as e:  # noqa: BLE001
                q.put((session, None, repr(e)))

    n_threads = 8
    subs = [todo[i::n_threads] for i in range(n_threads)]
    threads = [threading.Thread(target=producer, args=(s,), daemon=True) for s in subs]
    for t in threads:
        t.start()

    def closer():
        for t in threads:
            t.join()
        q.put((None, None, stop))

    threading.Thread(target=closer, daemon=True).start()

    n_done = n_err = 0
    errs = []
    while True:
        session, gt, err = q.get()
        if err is stop:
            break
        if err is not None:
            n_err += 1
            errs.append((session, err))
            print(f"FETCH_FAIL {session}: {err[:200]}", flush=True)
            continue
        try:
            res = fit_episode(gt, args.device, args.fit_iters)
            np.savez_compressed(stage_dir / f"{session}.npz", **res)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            errs.append((session, repr(e)))
            print(f"FIT_FAIL {session}: {e!r}", flush=True)
        if (n_done + n_err) % 50 == 0:
            print(f"  {n_done + n_err}/{len(todo)} (err {n_err})", flush=True)
    print(f"FIT_DONE ok={n_done} err={n_err}", flush=True)
    if errs:
        (stage_dir / "_fit_errors.json").write_text(json.dumps(errs, indent=1))


# ------------------------------------------------------------ finalize phase

def cam_from_world_dd(head_row: np.ndarray, M: np.ndarray, t: np.ndarray):
    """R, t of camera_from_w'' for one frame (proper)."""
    Rwh = Rt.from_quat(head_row[3:]).as_matrix()
    twh = head_row[:3]
    R = M @ Rwh.T @ F_W
    tt = t - R @ (F_W @ twh)
    return R, tt


def scaled_intrinsics(calib: dict, dev: str):
    intr = calib["devices"][dev]["intrinsics"]
    K = np.array(intr["camera_matrix"]["matrix"], float)
    fx, fy = K[0, 0] * FW / W0, K[1, 1] * FH / H0
    cx, cy = K[0, 2] * FW / W0, K[1, 2] * FH / H0
    return float((fx + fy) / 2.0), (float(cx), float(cy))


def cmd_finalize(args):
    import joblib
    from lib.pipeline.proc.stage_api import get_stage_done_marker

    calib = json.load(open(CALIB))
    clip_index = load_clip_index(Path(args.manifest_dir))
    assign = json.loads(Path(args.assignments).read_text())  # session -> block info
    extr = {p.stem: json.loads(p.read_text())
            for p in Path(args.extrinsics_dir).glob("*.json")}
    stage_dir = Path(args.stage_dir)
    outputs_root = Path(args.outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    report = dict(sessions=0, clips=0, skipped_sessions=[], failed=[])

    sessions = sorted(clip_index)
    if args.sessions:
        want = set(args.sessions.split(","))
        sessions = [s for s in sessions if s in want]
    for session in sessions:
        a = assign.get(session)
        if not a or a.get("status") != "pass":
            report["skipped_sessions"].append(session)
            continue
        npz_path = stage_dir / f"{session}.npz"
        if not npz_path.exists():
            report["failed"].append((session, "no staging npz"))
            continue
        ex = extr[a["block"]]
        M, t = np.array(ex["M"]), np.array(ex["t"])
        task = session.split("_")[0]
        dev = calib["episode_device"][task][session.split("_")[1]]
        focal, center = scaled_intrinsics(calib, dev)
        st = np.load(npz_path)
        rows_all = st["rows"]
        row_pos = {int(r): i for i, r in enumerate(rows_all)}
        report["sessions"] += 1
        for c in clip_index[session]:
            try:
                T = c["T"]
                idxs = [row_pos.get(c["start_frame"] + 4 * k) for k in range(T)]
                if any(i is None for i in idxs):
                    raise ValueError("staging rows missing for clip")
                idxs = np.asarray(idxs)
                trans = st["trans"][:, idxs].copy()
                rot = st["rot"][:, idxs].copy()
                pose = st["pose"][:, idxs].copy()
                valid = st["valid"][:, idxs].copy()
                betas = np.repeat(st["betas"][:, None, :], T, axis=1)
                head = st["head"][idxs]
                ok_head = np.isfinite(head).all(axis=1)
                if not ok_head.any():
                    raise ValueError("no valid head poses in clip")
                # hold-last for invalid head rows (rare; hands there are invalid anyway)
                last = np.where(ok_head)[0][0]
                for i in range(T):
                    if ok_head[i]:
                        last = i
                    else:
                        head[i] = head[last]
                        valid[:, i] = 0.0
                traj = np.zeros((T, 7), np.float32)
                for i in range(T):
                    R, tt = cam_from_world_dd(head[i].astype(float), M, t)
                    c2w_R = R.T
                    traj[i, :3] = -R.T @ tt
                    traj[i, 3:] = Rt.from_matrix(c2w_R).as_quat()
                seq_folder = outputs_root / c["clip_id"]
                slam_dir = seq_folder / "SLAM"
                slam_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump([trans, rot, pose, betas.astype(np.float32),
                             valid.astype(np.float32)],
                            seq_folder / "world_space_res.pth")
                np.savez(slam_dir / f"hawor_slam_w_scale_0_{T - 1}.npz",
                         tstamp=np.arange(T, dtype=np.int64),
                         traj=traj,
                         scale=np.float64(1.0),
                         img_focal=np.float64(focal),
                         img_center=np.asarray(center, dtype=np.float64))
                (seq_folder / "est_focal.txt").write_text(f"{focal:.6f}\n")
                tracks_dir = seq_folder / f"tracks_0_{T - 1}"
                tracks_dir.mkdir(exist_ok=True)
                (tracks_dir / ".humantouch_gt").write_text(json.dumps(
                    dict(clip_id=c["clip_id"], frames=T, block=a["block"])))
                dm = get_stage_done_marker(seq_folder, "infiller")
                dm.parent.mkdir(parents=True, exist_ok=True)
                dm.touch()
                r = json.loads(json.dumps(c["row"]))    # deep copy
                d = r["descriptor"]
                d["seq_folder"] = str(seq_folder.resolve())
                d["extra"].update(adapter="humantouch_gt",
                                  pose_grade="manus_glove_gt",
                                  mount_block=a["block"],
                                  block_fit_median_px=extr[a["block"]].get("fit_median_px"),
                                  gt_mode="gt_derived_extrinsic_block_anchor")
                manifest_rows.append(r)
                report["clips"] += 1
            except Exception as e:  # noqa: BLE001
                report["failed"].append((c["clip_id"], repr(e)))
    with open(args.manifest_out, "w") as f:
        for r in manifest_rows:
            f.write(json.dumps(r) + "\n")
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v)
                      for k, v in report.items()}))
    print("HUMANTOUCH_FINALIZE_DONE")


# ---------------------------------------------------------------- smoke

def cmd_smoke(args):
    """Overlay fitted MANO joints + raw GT joints on converted tar frames."""
    import cv2
    import torch
    from lib.pipeline.hands.fpha_skeleton import _build_right_mano, _forward_right_mano

    calib = json.load(open(CALIB))
    session = args.session
    task, ep_key = session.split("_")
    dev = calib["episode_device"][task][ep_key]
    st = np.load(Path(args.stage_dir) / f"{session}.npz")
    ex = json.loads(Path(args.extrinsic).read_text())
    M, t = np.array(ex["M"]), np.array(ex["t"])
    intr = calib["devices"][dev]["intrinsics"]
    K = np.array(intr["camera_matrix"]["matrix"], float)
    Ks = K.copy()
    Ks[0] *= FW / W0
    Ks[1] *= FH / H0
    dist = np.array(intr["distortion_coefficients"]["values"], float)

    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "build"))
    from humantouch_block_census import tar_index, fetch_tar_frames, interval_meta

    tidx = tar_index()[(task, ep_key)]
    imeta = interval_meta()[session]
    iv = sorted(tidx)[0]
    frames = fetch_tar_frames(tidx[iv], n=4)
    start = imeta["intervals"][iv]["start_frame"]
    row_pos = {int(r): i for i, r in enumerate(st["rows"])}

    dev_t = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mano = _build_right_mano(dev_t)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fi, img in frames:
        r = start + 4 * fi
        i = row_pos.get(r)
        if i is None:
            continue
        head = st["head"][i].astype(float)
        R, tt = cam_from_world_dd(head, M, t)
        vis = img.copy()
        for hidx, side, color in ((0, "left", (0, 0, 255)), (1, "right", (255, 128, 0))):
            if st["valid"][hidx, i] < 0.5:
                continue
            rot = st["rot"][hidx, i].copy()
            pose = st["pose"][hidx, i].copy()
            trans = st["trans"][hidx, i].copy()
            betas = st["betas"][hidx].copy()
            if side == "left":
                # mirror left params into right-MANO space for forward pass
                mrot = rot.copy()
                mrot[1] *= -1.0
                mrot[2] *= -1.0
                mpose = pose.reshape(15, 3).copy()
                mpose[:, 1] *= -1.0
                mpose[:, 2] *= -1.0
                with torch.no_grad():
                    j = _forward_right_mano(
                        mano,
                        torch.tensor(mrot[None], dtype=torch.float32, device=dev_t),
                        torch.tensor(mpose.reshape(1, 45), dtype=torch.float32, device=dev_t),
                        torch.tensor(betas[None], dtype=torch.float32, device=dev_t),
                    )[0].cpu().numpy()
                j = j @ _MIRROR.T + trans
            else:
                with torch.no_grad():
                    j = _forward_right_mano(
                        mano,
                        torch.tensor(rot[None], dtype=torch.float32, device=dev_t),
                        torch.tensor(pose[None], dtype=torch.float32, device=dev_t),
                        torch.tensor(betas[None], dtype=torch.float32, device=dev_t),
                    )[0].cpu().numpy()
                j = j + trans
            p_cam = j @ R.T + tt
            z = p_cam[:, 2]
            ok = z > 0.05
            uv = np.zeros((len(j), 2))
            uv[ok] = p_cam[ok, :2] / z[ok, None]
            px = np.stack([Ks[0, 0] * uv[:, 0] + Ks[0, 2],
                           Ks[1, 1] * uv[:, 1] + Ks[1, 2]], 1)
            for k in range(len(px)):
                if ok[k]:
                    cv2.circle(vis, tuple(px[k].astype(int)), 2, color, -1)
        p = outdir / f"{session}_iv{iv:02d}_f{fi:03d}_mano.jpg"
        cv2.imwrite(str(p), vis)
        print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--manifest_dir", default=str(WORK / "phaseB_manifests"))
    p.add_argument("--stage_dir", default=str(WORK / "gt_fit"))
    p.add_argument("--sessions", default=None)
    p.add_argument("--shard", default=None, help="i:n round-robin session shard")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fit_iters", type=int, default=400)
    p.add_argument("--prefetch", type=int, default=16)
    p = sub.add_parser("finalize")
    p.add_argument("--manifest_dir", default=str(WORK / "phaseB_manifests"))
    p.add_argument("--stage_dir", default=str(WORK / "gt_fit"))
    p.add_argument("--assignments", required=True)
    p.add_argument("--extrinsics_dir", default="/root/w7_full/humantouch/anchors/extrinsics")
    p.add_argument("--outputs_root", default=str(WORK / "gt_outputs"))
    p.add_argument("--manifest_out", required=True)
    p.add_argument("--report_out", default=None)
    p.add_argument("--sessions", default=None)
    p = sub.add_parser("smoke")
    p.add_argument("--session", required=True)
    p.add_argument("--stage_dir", default=str(WORK / "gt_fit"))
    p.add_argument("--extrinsic", required=True)
    p.add_argument("--outdir", default=str(WORK / "gt_smoke"))
    p.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    {"fit": cmd_fit, "finalize": cmd_finalize, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
