#!/usr/bin/env python3
"""HumanTouch mount-block census (W10 Stage-2, step 1).

Segments all ~17.6k episodes into MOUNT BLOCKS: contiguous (task, device)
episode ranges over which camera_from_head_tracker is stable. Mount varies
~9-14 deg between wearing sessions (Stage-1), stable within.

Method: per sampled episode, fit a 6-DoF extrinsic offset (from the shipped
base composition) by symmetric chamfer between projected MANUS skeletons and
dark-glove masks, using the ALREADY-CONVERTED 456x256 frame tars on GCS
(egosmith_filtered/humantouch/frames) + column-projected GT parquets from the
source bucket. The fitted ROTATION is the mount fingerprint; adjacent-sample
rotation deltas > threshold mark block boundaries, refined by bisection.

Chamfer fits are NOT trusted as calibration (Stage-1: 36-120px, silent
failures) -- only as a *relative* change signal within a (task, device) scene.

Subcommands:
  fingerprint --queue Q.jsonl     compute fingerprints for queued episodes
  plan                            emit next sampling queue (stride + bisection)
  segment                         write blocks.json from converged fingerprints
  verify-mapping                  overlay sanity render for X009_005346
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rt

WORK = Path('/root/w7_full/humantouch/census')
CALIB = '/root/w7_full/humantouch/humantouch_calibration.json'
TAR_LIST = '/root/w7_full/humantouch/_dwork/frames_tar_list.txt'
SRC = 'gs://foundational-research/hoi-dataset/Xspark-HumanTouch'

FW, FH = 456, 256                      # converted frame size
W0, H0 = 1920, 1080                    # native size
SX, SY = FW / W0, FH / H0
F = np.diag([1.0, -1.0, 1.0])
T0 = np.array([-0.034, 0.137, -0.028])
RPY = np.array([50.0, 178.0, 180.0])
CAP_PX = 150.0                          # chamfer cap, full-res px
NPX = 150                               # sampled glove px per frame
N_FRAMES = 10                           # frames decoded per tar
ANCHOR_JSON = '/root/w7_full/humantouch/stage1_gate/anchor_X009_005346.json'

PQ_COLS = ['observation.human.pose.head', 'observation.human.pose.valid',
           'observation.human.hand_skeleton.left.position',
           'observation.human.hand_skeleton.right.position']


def base_M() -> np.ndarray:
    return Rt.from_euler('ZYX', [RPY[2], RPY[1], RPY[0]], degrees=True).as_matrix() @ F


def load_calib_table():
    return json.load(open(CALIB))


def scaled_K(cal, dev):
    intr = cal['devices'][dev]['intrinsics']
    K = np.array(intr['camera_matrix']['matrix'], float).copy()
    K[0] *= SX
    K[1] *= SY
    dist = np.array(intr['distortion_coefficients']['values'], float)
    return K, dist


def project_px(p, K, dist):
    z = p[:, 2]
    okz = z > 0.05
    xn = np.zeros((len(p), 2))
    xn[okz] = p[okz, :2] / z[okz, None]
    r2 = (xn ** 2).sum(1)
    okz = okz & (r2 < 2.0)
    k1, k2, p1, p2, k3 = dist
    x, y = xn[:, 0], xn[:, 1]
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    yd = y * rad + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1), okz


def frame_maps(img, rng):
    """Glove-mask DT + sampled glove px on a 456x256 frame. Distances in
    FULL-RES px equivalents (divide by SX)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = (g > 110).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright)
    if n < 2:
        return None
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[big, cv2.CC_STAT_AREA] < 30000 * (FW * FH) / (960 * 540):
        return None
    comp = (lab == big).astype(np.uint8)
    hull = cv2.convexHull(np.column_stack(np.where(comp)[::-1]))
    hm = np.zeros_like(comp)
    cv2.fillConvexPoly(hm, hull, 1)
    hm = cv2.erode(hm, np.ones((3, 3), np.uint8))
    glove = ((g < 85) & (hm > 0)).astype(np.uint8)
    glove = cv2.morphologyEx(glove, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if glove.sum() < 1000 * (FW * FH) / (960 * 540):
        return None
    dt = np.minimum(cv2.distanceTransform(1 - glove, cv2.DIST_L2, 3).astype(np.float32) / SX,
                    CAP_PX)
    ys, xs = np.where(glove > 0)
    sel = rng.choice(len(xs), size=min(NPX, len(xs)), replace=False)
    gpx = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    return dt, gpx


def tar_index():
    """{(task, ep_key): {iv: gcs_uri}}"""
    idx = {}
    for line in open(TAR_LIST):
        line = line.strip()
        if not line.endswith('.tar'):
            continue
        name = line.rsplit('/', 1)[-1][:-4]          # X001_000001_iv00
        task, ep, iv = name.split('_')
        idx.setdefault((task, ep), {})[int(iv[2:])] = line
    return idx


def fetch_tar_frames(uri, n=N_FRAMES, cache_dir=None):
    """Download subclip tar, decode n frames spread across it.
    Returns list of (tar_frame_idx, bgr_img)."""
    if cache_dir:
        local = Path(cache_dir) / uri.rsplit('/', 1)[-1]
        if not local.exists():
            subprocess.run(['gsutil', '-q', 'cp', uri, str(local)], check=True)
        data = local.read_bytes()
    else:
        data = subprocess.run(['gsutil', '-q', 'cat', uri],
                              check=True, capture_output=True).stdout
    out = []
    with tarfile.open(fileobj=BytesIO(data)) as tf:
        members = sorted((m for m in tf.getmembers() if m.name.endswith('.image.jpg')),
                         key=lambda m: m.name)
        if not members:
            return out
        picks = sorted(set(np.linspace(0, len(members) - 1, n).astype(int)))
        for i in picks:
            buf = tf.extractfile(members[i]).read()
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                fidx = int(members[i].name.rsplit('_f', 1)[1].split('.')[0])
                out.append((fidx, img))
    return out


def fetch_parquet_rows(task, ep_key, row_idxs):
    """Column-projected read of the episode GT parquet; returns dict row->fields."""
    import pyarrow.parquet as pq
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    uri = f'{SRC}/{task}/data/chunk-000/episode_{ep_key}.parquet'
    with fs.open(uri.replace('gs://', ''), 'rb') as f:
        t = pq.read_table(f, columns=PQ_COLS)
    n = t.num_rows
    rows = {}
    cols = {c: t.column(c) for c in PQ_COLS}
    for r in row_idxs:
        if r >= n:
            continue
        rows[r] = {c: cols[c][r].as_py() for c in PQ_COLS}
    return rows, n


def interval_meta():
    """{session: [interval dicts]} from the convert index."""
    meta = {}
    for line in open('/root/w7_full/humantouch/index.full.jsonl'):
        d = json.loads(line)
        meta[d['session']] = d
    return meta


class EpData:
    """Chamfer-ready frames for one episode from its converted tar(s)."""

    def __init__(self, task, ep_key, dev, tar_uris, ivs, ivmeta, cache_dir=None):
        self.task, self.ep, self.dev = task, ep_key, dev
        rng = np.random.default_rng(0)
        cal = load_calib_table()
        self.K, self.dist = scaled_K(cal, dev)
        fps_ratio = round(ivmeta['fps'] / 15.0)      # 60 -> 4
        rowmap = {}
        frames = []
        for tar_uri, iv in zip(tar_uris, ivs):
            start = ivmeta['intervals'][iv]['start_frame']
            for fi, img in fetch_tar_frames(tar_uri, cache_dir=cache_dir):
                key = (iv, fi)
                rowmap[key] = start + fps_ratio * fi
                frames.append((key, img))
        rows, self.n_rows = fetch_parquet_rows(task, ep_key, sorted(set(rowmap.values())))
        self.frames = []
        for key, img in frames:
            r = rowmap[key]
            if r not in rows:
                continue
            row = rows[r]
            if not row['observation.human.pose.valid'][0]:
                continue
            fm = frame_maps(img, rng)
            if fm is None:
                continue
            hp = np.array(row['observation.human.pose.head'])
            Rwh = Rt.from_quat(hp[3:]).as_matrix()
            pts = np.concatenate([np.asarray(row['observation.human.hand_skeleton.left.position']),
                                  np.asarray(row['observation.human.hand_skeleton.right.position'])], 0)
            self.frames.append((fm[0], fm[1], (pts - hp[:3]) @ Rwh, r, img))


def cost(M, t, ep, w_bwd=1.0, trim=0.8):
    tot = []
    for dt, gpx, ph, _, _ in ep.frames:
        p = ph @ M.T + t
        px, okz = project_px(p, ep.K, ep.dist)
        u, v = px[:, 0], px[:, 1]
        inb = okz & (u >= 0) & (u < FW - 1) & (v >= 0) & (v < FH - 1)
        fwd = np.full(len(px), CAP_PX)
        if inb.any():
            fwd[inb] = dt[v[inb].astype(int), u[inb].astype(int)]
        dd = np.sqrt(((gpx[:, None, :] - px[None, :, :]) ** 2).sum(2)) / SX
        bwd = np.minimum(dd.min(1), CAP_PX)
        tot.append(fwd.mean() + w_bwd * bwd.mean())
    if not tot:
        return float('inf')
    tot = np.sort(tot)
    k = max(1, int(len(tot) * trim))
    return float(np.mean(tot[:k]))


def rot_grid(n_dirs=48, mags=(4, 8, 12, 16, 20, 24)):
    rng = np.random.default_rng(1)
    dirs = rng.normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    grid = [np.zeros(3)]
    for m in mags:
        grid += [d * np.deg2rad(m) for d in dirs]
    return grid


def anchor_Mt():
    a = json.load(open(ANCHOR_JSON))
    return np.array(a['M']), np.array(a['t'])


def fit_fingerprint(ep, seeds=None):
    """Rotation-only chamfer fit around the X009 anchor extrinsic, with the
    anchor's translation held FIXED (physically-real rig geometry, shared
    mount hardware). Fingerprint = rotvec offset from the anchor rotation.

    Rotation-only + fixed t kills the rot/trans trade-off that made the free
    6-DoF fit non-repeatable (pilot: 7-20 deg same-block scatter)."""
    Ma, ta = anchor_Mt()
    cands = list(rot_grid())
    for s in (seeds or []):
        cands.append(np.asarray(s, float)[:3])

    def unpack(rv):
        return Rt.from_rotvec(rv).as_matrix() @ Ma, ta

    best = (float('inf'), None)
    for rv in cands:
        c = cost(*unpack(rv), ep)
        if c < best[0]:
            best = (c, rv)

    res = minimize(lambda rv: cost(*unpack(rv), ep), best[1], method='Powell',
                   options=dict(maxiter=1500, xtol=1e-4, ftol=1e-4))
    rv = res.x
    return dict(rotvec_deg=list(np.rad2deg(rv)), cost=float(res.fun),
                grid_cost=float(best[0]), n_frames=len(ep.frames),
                x=list(rv))


def _fp_one(job):
    task, ep_key, dev, tar_uris, ivs, ivmeta, seeds = job
    try:
        ep = EpData(task, ep_key, dev, tar_uris, ivs, ivmeta)
        if len(ep.frames) < 6:
            return dict(task=task, ep=ep_key, dev=dev, status='too_few_frames',
                        n_frames=len(ep.frames))
        r = fit_fingerprint(ep, seeds)
        r.update(task=task, ep=ep_key, dev=dev, ivs=ivs, status='ok')
        return r
    except Exception as e:  # noqa: BLE001
        return dict(task=task, ep=ep_key, dev=dev, status=f'error: {e!r:.200}')


def pick_ivs(ivs_for_ep, k=2):
    """Spread k intervals across the episode (e.g. first + middle)."""
    ks = sorted(ivs_for_ep)
    if len(ks) <= k:
        return ks
    picks = sorted(set(int(round(i)) for i in np.linspace(0, len(ks) - 1, k)))
    return [ks[i] for i in picks]


def cmd_fingerprint(args):
    WORK.mkdir(parents=True, exist_ok=True)
    out_path = WORK / 'fingerprints.jsonl'
    done = set()
    if out_path.exists():
        for line in open(out_path):
            d = json.loads(line)
            done.add((d['task'], d['ep']))
    tidx = tar_index()
    imeta = interval_meta()
    jobs = []
    for line in open(args.queue):
        q = json.loads(line)
        task, ep_key = q['task'], q['ep']
        if (task, ep_key) in done:
            continue
        ivs = tidx.get((task, ep_key))
        sess = f'{task}_{ep_key}'
        if not ivs or sess not in imeta:
            jobs.append(None)  # placeholder to record missing
            with open(out_path, 'a') as f:
                f.write(json.dumps(dict(task=task, ep=ep_key, status='no_tar_or_meta')) + '\n')
            done.add((task, ep_key))
            continue
        sel = pick_ivs(ivs)
        seeds = [s for s in [q.get('seed')] if s]
        jobs.append((task, ep_key, q['dev'], [ivs[i] for i in sel], sel,
                     imeta[sess], seeds))
    jobs = [j for j in jobs if j]
    print(f'{len(jobs)} episodes to fingerprint', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_fp_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            with open(out_path, 'a') as f:
                f.write(json.dumps(r) + '\n')
            if (i + 1) % 20 == 0:
                print(f'  {i + 1}/{len(jobs)}', flush=True)
    print('done', flush=True)


def load_fps():
    fps = {}
    p = WORK / 'fingerprints.jsonl'
    if p.exists():
        for line in open(p):
            d = json.loads(line)
            fps[(d['task'], d['ep'])] = d
    return fps


def device_runs():
    """[(task, dev, [ep_keys sorted])]: contiguous same-device runs per task."""
    cal = load_calib_table()
    runs = []
    for task in sorted(cal['episode_device']):
        em = cal['episode_device'][task]
        keys = sorted(em)
        cur = None
        for k in keys:
            d = em[k]
            if cur and cur[1] == d:
                cur[2].append(k)
            else:
                cur = [task, d, [k]]
                runs.append(cur)
        # merge interleaved: same (task, dev) runs separated by other-dev
        # episodes get merged into per-device ordered sequences
    merged = {}
    order = []
    for task, dev, eps in runs:
        key = (task, dev)
        if key not in merged:
            merged[key] = []
            order.append(key)
        merged[key].extend(eps)
    return [(t, d, sorted(merged[(t, d)])) for t, d in order]


def rot_dist_deg(a, b):
    Ra = Rt.from_rotvec(np.deg2rad(a)).as_matrix()
    Rb = Rt.from_rotvec(np.deg2rad(b)).as_matrix()
    return float(np.rad2deg(np.abs(Rt.from_matrix(Ra.T @ Rb).magnitude())))


def cmd_plan(args):
    fps = load_fps()
    queue = []
    queued = set()

    def enqueue(task, dev, eps, i, seed=None):
        """Queue eps[i]; if a prior attempt at eps[i] failed, substitute the
        nearest untried neighbor (+-3)."""
        for off in (0, 1, -1, 2, -2, 3, -3):
            j = i + off
            if not (0 <= j < len(eps)):
                continue
            key = (task, eps[j])
            prior = fps.get(key)
            if prior is not None and prior.get('status') != 'ok':
                continue                       # tried and failed
            if prior is not None:
                return                         # already have an ok sample here
            if key in queued:
                return
            queued.add(key)
            q = dict(task=task, dev=dev, ep=eps[j])
            if seed:
                q['seed'] = seed
            queue.append(q)
            return

    for task, dev, eps in device_runs():
        idx_of = {k: i for i, k in enumerate(eps)}
        ok = [(idx_of[k], fps[(task, k)]) for k in eps
              if fps.get((task, k), {}).get('status') == 'ok']
        ok.sort()
        # coverage: no gap between ok samples (or run ends) wider than stride
        bounds = [-1] + [i for i, _ in ok] + [len(eps)]
        for a, b in zip(bounds, bounds[1:]):
            gap = b - a
            if gap > args.stride:
                for i in range(a + args.stride, b, args.stride):
                    enqueue(task, dev, eps, i)
            elif a == -1 and gap > 1:
                enqueue(task, dev, eps, 0)     # always sample run start
            elif b == len(eps) and gap > 1:
                enqueue(task, dev, eps, len(eps) - 1)
        # bisection: adjacent ok samples with big delta and idx gap > resolution
        for (i1, d1), (i2, d2) in zip(ok, ok[1:]):
            if i2 - i1 <= args.resolution:
                continue
            if rot_dist_deg(d1['rotvec_deg'], d2['rotvec_deg']) > args.split_deg:
                enqueue(task, dev, eps, (i1 + i2) // 2, seed=d1['x'])
    qp = WORK / 'queue.jsonl'
    with open(qp, 'w') as f:
        for q in queue:
            f.write(json.dumps(q) + '\n')
    print(f'{len(queue)} episodes queued -> {qp}')


def cmd_segment(args):
    fps = load_fps()
    blocks = []
    for task, dev, eps in device_runs():
        ok = [(k, fps[(task, k)]) for k in eps if
              (task, k) in fps and fps[(task, k)].get('status') == 'ok']
        if not ok:
            blocks.append(dict(task=task, dev=dev, ep_start=eps[0], ep_end=eps[-1],
                               n_eps=len(eps), status='UNSAMPLED'))
            continue
        idx_of = {k: i for i, k in enumerate(eps)}
        cuts = [0]
        for (k1, d1), (k2, d2) in zip(ok, ok[1:]):
            if rot_dist_deg(d1['rotvec_deg'], d2['rotvec_deg']) > args.split_deg:
                cuts.append(idx_of[k2])
        cuts.append(len(eps))
        for a, b in zip(cuts, cuts[1:]):
            seg_eps = eps[a:b]
            seg_fp = [d for k, d in ok if a <= idx_of[k] < b]
            rvs = np.array([d['rotvec_deg'] for d in seg_fp]) if seg_fp else None
            blocks.append(dict(
                task=task, dev=dev, ep_start=seg_eps[0], ep_end=seg_eps[-1],
                n_eps=len(seg_eps), n_sampled=len(seg_fp),
                fp_median_rotvec_deg=(list(np.median(rvs, 0)) if rvs is not None else None),
                fp_spread_deg=(float(np.max([rot_dist_deg(r, np.median(rvs, 0))
                                             for r in rvs])) if rvs is not None else None),
                median_cost=(float(np.median([d['cost'] for d in seg_fp]))
                             if seg_fp else None)))
    for i, b in enumerate(blocks):
        b['block_id'] = f'B{i:03d}'
    out = WORK / 'blocks.json'
    json.dump(blocks, open(out, 'w'), indent=1)
    n_ok = sum(1 for b in blocks if b.get('n_sampled'))
    print(f'{len(blocks)} blocks ({n_ok} sampled) -> {out}')


def cmd_segment2(args):
    """Full-coverage segmentation: greedy medoid clustering of per-episode
    fingerprints within each (task, device), followed by a medoid-merge pass.

    Contiguity is NOT required: what matters for sharing an extrinsic is mount
    similarity, which the fingerprint measures directly (within a task the
    chamfer scene-bias is common mode). Requires (near-)full fingerprint
    coverage from the full sweep."""
    fps = load_fps()
    blocks = []
    unfp = []
    for task, dev, eps in device_runs():
        clusters = []  # each: dict(eps=[], rvs=[])
        for k in eps:
            d = fps.get((task, k))
            if not d or d.get('status') != 'ok':
                unfp.append((task, dev, k, (d or {}).get('status', 'missing')))
                continue
            rv = d['rotvec_deg']
            best = (1e9, None)
            for c in clusters:
                dist = rot_dist_deg(rv, c['med'])
                if dist < best[0]:
                    best = (dist, c)
            if best[1] is not None and best[0] <= args.radius:
                c = best[1]
                c['eps'].append(k)
                c['rvs'].append(rv)
                if len(c['eps']) % 8 == 0:      # refresh medoid occasionally
                    c['med'] = list(np.median(np.array(c['rvs']), 0))
            else:
                clusters.append(dict(eps=[k], rvs=[rv], med=list(rv)))
        # merge pass
        for c in clusters:
            c['med'] = list(np.median(np.array(c['rvs']), 0))
        merged = []
        for c in sorted(clusters, key=lambda c: -len(c['eps'])):
            for m in merged:
                if rot_dist_deg(c['med'], m['med']) <= args.merge:
                    m['eps'] += c['eps']
                    m['rvs'] += c['rvs']
                    m['med'] = list(np.median(np.array(m['rvs']), 0))
                    break
            else:
                merged.append(c)
        for c in merged:
            rvs = np.array(c['rvs'])
            med = np.median(rvs, 0)
            dists = [rot_dist_deg(r, med) for r in rvs]
            medoid_i = int(np.argmin([rot_dist_deg(r, med) for r in rvs]))
            costs = [fps[(task, k)]['cost'] for k in c['eps']]
            blocks.append(dict(
                task=task, dev=dev, n_eps=len(c['eps']),
                eps=sorted(c['eps']),
                anchor_ep=c['eps'][medoid_i],
                fp_median_rotvec_deg=list(med),
                fp_spread_p90_deg=float(np.percentile(dists, 90)),
                median_cost=float(np.median(costs))))
    blocks.sort(key=lambda b: -b['n_eps'])
    for i, b in enumerate(blocks):
        b['block_id'] = f'B{i:03d}'
    out = WORK / 'blocks2.json'
    json.dump(dict(blocks=blocks,
                   unfingerprinted=[dict(task=t, dev=d, ep=k, status=s)
                                    for t, d, k, s in unfp]),
              open(out, 'w'), indent=1)
    n_eps = sum(b['n_eps'] for b in blocks)
    big = [b for b in blocks if b['n_eps'] >= args.min_block]
    print(f'{len(blocks)} clusters covering {n_eps} eps '
          f'({len(unfp)} unfingerprinted); '
          f'{len(big)} clusters >= {args.min_block} eps covering '
          f'{sum(b["n_eps"] for b in big)} eps -> {out}')


def cmd_verify_mapping(args):
    """Render anchor-extrinsic overlays on tar frames of X009_005346."""
    anchor = json.load(open('/root/w7_full/humantouch/stage1_gate/anchor_X009_005346.json'))
    M, t = np.array(anchor['M']), np.array(anchor['t'])
    tidx = tar_index()
    imeta = interval_meta()
    task, ep_key = 'X009', '005346'
    ivs = tidx[(task, ep_key)]
    sel = pick_ivs(ivs, k=1)
    ep = EpData(task, ep_key, 'C2', [ivs[i] for i in sel], sel,
                imeta[f'{task}_{ep_key}'])
    print(f'{len(ep.frames)} usable frames; anchor cost = {cost(M, t, ep):.1f} '
          f'(full-res px equiv)')
    outdir = WORK / 'verify'
    outdir.mkdir(parents=True, exist_ok=True)
    for dtm, gpx, ph, r, img in ep.frames[:4]:
        px, okz = project_px(ph @ M.T + t, ep.K, ep.dist)
        vis = img.copy()
        for j in range(len(px)):
            if okz[j]:
                cv2.circle(vis, tuple(px[j].astype(int)), 2,
                           (0, 0, 255) if j < 25 else (255, 128, 0), -1)
        p = outdir / f'{task}_{ep_key}_row{r}.jpg'
        cv2.imwrite(str(p), vis)
        print(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('fingerprint')
    p.add_argument('--queue', required=True)
    p.add_argument('--workers', type=int, default=16)
    p = sub.add_parser('plan')
    p.add_argument('--stride', type=int, default=40)
    p.add_argument('--split-deg', type=float, default=5.0, dest='split_deg')
    p.add_argument('--resolution', type=int, default=8)
    p = sub.add_parser('segment')
    p.add_argument('--split-deg', type=float, default=5.0, dest='split_deg')
    p = sub.add_parser('segment2')
    p.add_argument('--radius', type=float, default=2.5)
    p.add_argument('--merge', type=float, default=2.0)
    p.add_argument('--min-block', type=int, default=20, dest='min_block')
    sub.add_parser('verify-mapping')
    args = ap.parse_args()
    {'fingerprint': cmd_fingerprint, 'plan': cmd_plan, 'segment': cmd_segment,
     'segment2': cmd_segment2, 'verify-mapping': cmd_verify_mapping}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
