#!/usr/bin/env python3
"""WIYH sensor-GT hunt: sample N scenes' inner tar member lists + full hdf5 trees.

For each chosen scene: ranged-fetch one worldcode_*.tar.gz via the member index,
list ALL inner member names, grep for pose/joint/exo/angle/mano/keypoint/glove/
imu/tf/urdf/calib/readme hints, extract dataset.hdf5 + any small metadata files,
and dump the complete HDF5 tree (paths, shapes, dtypes, attrs).
"""
import json, sys, tarfile, io
from pathlib import Path
import numpy as np

sys.path.insert(0, "/root/egosmith/scripts/build")
from generate_wiyh_recon_wds import _ranged_concat_read  # noqa

import gcsfs

IDX = Path("/root/w7_full/wiyh/index")
OUT = Path("/root/w7_reopen")
OUT.mkdir(exist_ok=True)
HINTS = ["joint", "pose", "exo", "angle", "mano", "keypoint", "glove", "imu",
         "tf", "urdf", "calib", "readme", "meta", "skeleton", "hand", "finger",
         "action", "feedback", "state", "json", "yaml", "txt", "md"]

SCENES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Candlelight", "Apartment", "Logistics", "Banquet"]

fs = gcsfs.GCSFileSystem()
report = {}
for scene in SCENES:
    members = [json.loads(l) for l in open(IDX / f"{scene}.members.jsonl") if l.strip()]
    parts = json.loads((IDX / f"{scene}.parts.json").read_text())
    # pick a mid-list member with a real payload
    cand = [m for m in members if m["size"] > 100_000_000]
    m = cand[len(cand) // 2]
    name = Path(m["name"]).name
    print(f"[{scene}] fetching {name} ({m['size']/1e6:.0f} MB)", flush=True)
    blob = _ranged_concat_read(fs, parts, int(m["offset"]), int(m["size"]))
    scene_out = OUT / scene
    scene_out.mkdir(exist_ok=True)
    names = []
    small_files = {}
    h5_local = None
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            names.append(f"{mem.name}\t{mem.size}")
            low = mem.name.lower()
            base = low.rsplit("/", 1)[-1]
            # extract dataset.hdf5 and any small non-image metadata file
            if low.endswith("dataset.hdf5"):
                h5_local = scene_out / "dataset.hdf5"
                with tf.extractfile(mem) as src, open(h5_local, "wb") as w:
                    w.write(src.read())
            elif mem.size < 5_000_000 and not base.endswith((".jpg", ".png", ".jpeg")):
                with tf.extractfile(mem) as src:
                    small_files[mem.name] = src.read()
    (scene_out / "member_names.txt").write_text("\n".join(names))
    hint_hits = {}
    for h in HINTS:
        hits = [n for n in names if h in n.lower().rsplit("\t", 1)[0]]
        if hits:
            hint_hits[h] = hits[:20] + ([f"... +{len(hits)-20} more"] if len(hits) > 20 else [])
    for fn, data in small_files.items():
        p = scene_out / "smallfiles" / fn
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    tree = []
    if h5_local:
        import h5py
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                at = {k: str(v)[:120] for k, v in obj.attrs.items()}
                tree.append(f"{name}  shape={obj.shape} dtype={obj.dtype}" + (f" attrs={at}" if at else ""))
            else:
                at = {k: str(v)[:120] for k, v in obj.attrs.items()}
                if at:
                    tree.append(f"{name}/  attrs={at}")
        with h5py.File(h5_local, "r") as f:
            fa = {k: str(v)[:200] for k, v in f.attrs.items()}
            if fa:
                tree.append(f"ROOT attrs={fa}")
            f.visititems(visit)
    (scene_out / "h5_tree.txt").write_text("\n".join(tree))
    report[scene] = {"sample": name, "n_members": len(names),
                     "small_files": sorted(small_files), "hint_hits": {k: len(v) for k, v in hint_hits.items()},
                     "h5_datasets": len(tree)}
    (scene_out / "hint_hits.json").write_text(json.dumps(hint_hits, indent=1))
    print(f"[{scene}] members={len(names)} h5_datasets={len(tree)} smallfiles={sorted(small_files)[:8]}", flush=True)

(OUT / "hunt_report.json").write_text(json.dumps(report, indent=1))
print("WIYH_HUNT_DONE")
