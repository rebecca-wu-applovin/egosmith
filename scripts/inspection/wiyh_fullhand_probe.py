#!/usr/bin/env python3
"""WIYH full-hand GT probe: are action/*_hand_glove joint 'angles' really 25x3
wrist-local joint POSITIONS that project onto the visible gloves?

Chain: p_chest = R_eef @ (R_g @ p_local) + t_eef  (R_g in 24 signed perms)
       p_cam   = R_ext^T (p_chest - t_ext)        (chest cam extrinsic, ~I)
       px      = KB4 fisheye projection
Score: median px distance of ALL projected joints to the hand mask.
"""
import io, json, sys, tarfile, itertools
from pathlib import Path
import numpy as np
import cv2
import h5py
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/egosmith/scripts/build")
from generate_wiyh_recon_wds import _ranged_concat_read  # noqa
import gcsfs

IDX = Path("/root/w7_full/wiyh/index")
OUT = Path("/root/w7_reopen/fullhand")
OUT.mkdir(parents=True, exist_ok=True)
CHEST = "lf_chest_fisheye"
# 25-joint topology inferred from bone-length audit: wrist=24, thumb=0-3,
# index=4-8, middle=9-13, ring=14-18, pinky=19-23
CHAINS = [[24, 0, 1, 2, 3], [24, 4, 5, 6, 7, 8], [24, 9, 10, 11, 12, 13],
          [24, 14, 15, 16, 17, 18], [24, 19, 20, 21, 22, 23]]
EDGES = [(c[i], c[i + 1]) for c in CHAINS for i in range(len(c) - 1)]
TIPS = [3, 8, 13, 18, 23]

SCENE = sys.argv[1] if len(sys.argv) > 1 else "Banquet"
SAMPLE = sys.argv[2] if len(sys.argv) > 2 else None


def signed_perms():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats


def quat_to_R(q):  # xyzw
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project_fisheye(pts_cam, K, D):
    out = np.full((pts_cam.shape[0], 2), np.nan)
    front = pts_cam[:, 2] > 1e-6
    if front.any():
        uv, _ = cv2.fisheye.projectPoints(pts_cam[front].reshape(-1, 1, 3),
                                          np.zeros(3), np.zeros(3), K, D.reshape(4, 1))
        out[front] = uv.reshape(-1, 2)
    return out


# ---- fetch sample with frames + masks ----
members = [json.loads(l) for l in open(IDX / f"{SCENE}.members.jsonl") if l.strip()]
parts = json.loads((IDX / f"{SCENE}.parts.json").read_text())
if SAMPLE:
    m = next(m for m in members if SAMPLE in m["name"])
else:
    cand = [m for m in members if m["size"] > 100_000_000]
    m = cand[len(cand) // 2]
name = Path(m["name"]).name.replace(".tar.gz", "")
print(f"fetching {name} ({m['size']/1e6:.0f} MB)", flush=True)
fs = gcsfs.GCSFileSystem()
blob = _ranged_concat_read(fs, parts, int(m["offset"]), int(m["size"]))
root = OUT / name
if not (root / "dataset.hdf5").exists():
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            n = mem.name
            keep = (n.endswith("dataset.hdf5") or f"camera/{CHEST}/" in n
                    or f"hand_masks/{CHEST}/" in n)
            if not keep:
                continue
            rel = "/".join(n.split("/")[2:]) if n.count("/") >= 2 else n
            out = root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(mem) as src, open(out, "wb") as w:
                w.write(src.read())
del blob

# ---- load streams ----
with h5py.File(root / "dataset.hdf5", "r") as f:
    cal = f[f"meta/calibration/{CHEST}"]
    K = np.array(cal["intrinsic"]); D = np.array(cal["distortion"], np.float64)
    ext = np.array(cal["extrinsic"], np.float64)
    W, H = int(cal["width"][0]), int(cal["height"][0])
    cam_tbl = f[f"observation/camera/{CHEST}"][:]
    glove, eef = {}, {}
    for side in ("left", "right"):
        g = f[f"action/{side}_hand_glove/feedback/joint_angle"][:]
        glove[side] = {"ts": g["timestamp"], "pts": g["value"].reshape(-1, 25, 3)}
        e = f[f"pose/{side}_eef/feedback/pose_in_chest"][:]
        eef[side] = {"ts": e["timestamp"], "val": e["val" "ue"], "conf": e["confidence"]}

cam_ts = cam_tbl["timestamp"]
frame_paths = [root / p.decode() for p in cam_tbl["file_path"]]
n = len(frame_paths)
R_ext, t_ext = ext[:3, :3], ext[:3, 3]

aligned = {}
for side in ("left", "right"):
    gi = np.abs(glove[side]["ts"][None, :] - cam_ts[:, None]).argmin(1)
    ei = np.abs(eef[side]["ts"][None, :] - cam_ts[:, None]).argmin(1)
    aligned[side] = {
        "pts": glove[side]["pts"][gi],
        "eef": eef[side]["val"][ei], "conf": eef[side]["conf"][ei],
        "gdt": np.abs(glove[side]["ts"][gi] - cam_ts),
        "edt": np.abs(eef[side]["ts"][ei] - cam_ts)}

mask_dir = root / "hand_masks" / CHEST
mask_files = sorted(mask_dir.glob("*.png"))
mask_dts = {}
for i in range(min(n, len(mask_files))):
    mm = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
    if mm is not None and (mm > 0).any():
        mask_dts[i] = cv2.distanceTransform((mm == 0).astype(np.uint8), cv2.DIST_L2, 3)
print(f"frames={n} masked={len(mask_dts)}", flush=True)


def joints_cam(i, side, Rg):
    a = aligned[side]
    p, q = a["eef"][i, :3], a["eef"][i, 3:]
    Re = quat_to_R(q)
    pc = (Re @ (Rg @ a["pts"][i].T)).T + p          # chest frame
    return (R_ext.T @ (pc - t_ext).T).T             # cam frame


sub = np.linspace(0, n - 1, min(n, 120), dtype=int)
scores = []
for ci, Rg in enumerate(signed_perms()):
    dists = []
    for i in sub:
        if i not in mask_dts:
            continue
        for side in ("left", "right"):
            if aligned[side]["conf"][i] < 1:
                continue
            uv = project_fisheye(joints_cam(i, side, Rg), K, D)
            ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            for u, v in uv[ok]:
                dists.append(mask_dts[i][int(v), int(u)])
    med = float(np.median(dists)) if dists else np.inf
    scores.append({"cand": ci, "R": Rg.tolist(), "median_px_to_mask": med, "n": len(dists)})

ranked = sorted(scores, key=lambda s: s["median_px_to_mask"])
print("top3:", json.dumps(ranked[:3], indent=1), flush=True)
Rg = np.array(ranked[0]["R"])

# per-frame stats under winner: median over joints, and TIP-only
per_frame_med, tip_med = [], []
for i in sub:
    if i not in mask_dts:
        continue
    for side in ("left", "right"):
        if aligned[side]["conf"][i] < 1:
            continue
        uv = project_fisheye(joints_cam(i, side, Rg), K, D)
        ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        d = [mask_dts[i][int(v), int(u)] for u, v in uv[ok]]
        if d:
            per_frame_med.append(np.median(d))
        dt = [mask_dts[i][int(v), int(u)] for j, (u, v) in enumerate(uv) if ok[j] and j in TIPS]
        if dt:
            tip_med.append(np.median(dt))

# overlay sheet
tile_w = 640
sel = np.linspace(0, n - 1, 9, dtype=int)
tiles = []
for i in sel:
    img = Image.open(frame_paths[i]).convert("RGB")
    s = tile_w / img.width
    tile = img.resize((tile_w, int(img.height * s)))
    d = ImageDraw.Draw(tile)
    if i < len(mask_files):
        mm = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
        if mm is not None:
            cnts, _ = cv2.findContours((mm > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                pts = [(float(x) * s, float(y) * s) for x, y in c.reshape(-1, 2)[::4]]
                if len(pts) > 1:
                    d.line(pts, fill=(0, 255, 120), width=1)
    for side, col in (("left", (66, 133, 244)), ("right", (255, 60, 60))):
        if aligned[side]["conf"][i] < 1:
            continue
        uv = project_fisheye(joints_cam(i, side, Rg), K, D) * s
        okj = np.isfinite(uv).all(1)
        for a, b in EDGES:
            if okj[a] and okj[b]:
                d.line([tuple(uv[a]), tuple(uv[b])], fill=col, width=2)
        for j in range(25):
            if okj[j]:
                r = 4 if j in TIPS else (6 if j == 24 else 2)
                d.ellipse([uv[j, 0] - r, uv[j, 1] - r, uv[j, 0] + r, uv[j, 1] + r], fill=col)
    d.rectangle([0, 0, 60, 20], fill=(0, 0, 0))
    d.text((4, 3), f"f{i}", fill=(255, 255, 255))
    tiles.append(tile)
cols = 3
rows = (len(tiles) + cols - 1) // cols
th = tiles[0].height
sheet = Image.new("RGB", (cols * tile_w, rows * th + 30), (16, 16, 16))
for k, t in enumerate(tiles):
    sheet.paste(t, ((k % cols) * tile_w, (k // cols) * th))
d = ImageDraw.Draw(sheet)
d.text((8, rows * th + 6), f"WIYH {name[:70]} 25-joint glove skeleton -> chest cam. "
                           f"best Rg={ranked[0]['R']} med_px_to_mask={ranked[0]['median_px_to_mask']:.1f}",
       fill=(255, 255, 255))
sheet_path = OUT / f"{SCENE}_{name[:60]}_fullhand_overlay.jpg"
sheet.save(sheet_path, quality=88)

summary = {
    "scene": SCENE, "sample": name, "frames": n,
    "topology": "wrist=24, thumb=0-3, index=4-8, middle=9-13, ring=14-18, pinky=19-23 (bone-len verified)",
    "best_Rg": ranked[0], "runner_up": ranked[1],
    "all_joints_px_to_mask": {"median": float(np.median(per_frame_med)) if per_frame_med else None,
                              "p90": float(np.percentile(per_frame_med, 90)) if per_frame_med else None,
                              "n_frames": len(per_frame_med)},
    "tips_px_to_mask": {"median": float(np.median(tip_med)) if tip_med else None,
                        "p90": float(np.percentile(tip_med, 90)) if tip_med else None},
    "sheet": str(sheet_path)}
(OUT / f"{SCENE}_fullhand_probe.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print("WIYH_FULLHAND_DONE")
