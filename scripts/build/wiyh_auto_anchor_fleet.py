#!/usr/bin/env python
"""Fleet runner: auto-anchor every census-LOCKED WIYH session.

Streams the census member of each locked session (dataset.hdf5 + chest jpgs +
hand masks to a temp dir), runs wiyh_auto_anchor.auto_anchor_sample, appends a
registry row per session. Resume-safe (existing sessions skipped).

Output rows: {"session", "scene", "dev", "date", "member",
              "left": {R,t,fit stats,pass}|{error}, "right": ...,
              "anchor_pass": bool (both hands pass)}

Usage:
  python scripts/build/wiyh_auto_anchor_fleet.py \
      --census /root/w7_native/census/census.jsonl \
      --index_dir /root/w7_full/wiyh/index \
      --out /root/w7_native/anchors/session_registry.jsonl --workers 24
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tarfile
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from wiyh_gate_census import CHEST, StreamedConcat, load_sessions  # noqa: E402

_FS = None
_PARTS = None
_MEMBERS = None


def _init(index_dir):
    global _FS, _PARTS, _MEMBERS
    import gcsfs
    _FS = gcsfs.GCSFileSystem()
    idx = Path(index_dir)
    _PARTS = {f.stem.split(".")[0]: json.loads(f.read_text())
              for f in idx.glob("*.parts.json")}
    _MEMBERS = {}
    for v in load_sessions(idx).values():
        for m in v:
            _MEMBERS[m["base"]] = m


def _stage(member, dest: Path):
    m = _MEMBERS[member]
    raw = StreamedConcat(_FS, _PARTS[m["scene"]], int(m["offset"]), int(m["size"]))
    gz = gzip.GzipFile(fileobj=raw, mode="rb")
    with tarfile.open(fileobj=gz, mode="r|") as tf:
        for mem in tf:
            if not mem.isfile():
                continue
            n = mem.name
            if not (n.endswith("dataset.hdf5") or f"camera/{CHEST}/" in n
                    or f"hand_masks/{CHEST}/" in n):
                continue
            rel = "/".join(n.split("/")[2:]) if n.count("/") >= 2 else n
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(mem) as src, open(out, "wb") as w:
                w.write(src.read())


def _one(job):
    row, work_root = job
    out = {"session": row["session"], "scene": row["scene"], "dev": row["dev"],
           "date": row["date"], "member": row["member"]}
    tmp = Path(tempfile.mkdtemp(prefix="aa_", dir=work_root))
    try:
        from wiyh_auto_anchor import auto_anchor_sample
        for attempt in range(2):
            try:
                _stage(row["member"], tmp)
                break
            except Exception as e:  # noqa: BLE001
                err = f"stage: {type(e).__name__}: {str(e)[:120]}"
                time.sleep(10)
        else:
            out["error"] = err
            return out
        res = auto_anchor_sample(tmp)
        out.update(res)
        out["anchor_pass"] = bool(res.get("left", {}).get("pass")
                                  and res.get("right", {}).get("pass"))
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--work_root", default="/root/w7_native/_aawork")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    locked = []
    for l in open(a.census):
        r = json.loads(l)
        if r.get("locked"):
            locked.append({k: r[k] for k in ("session", "scene", "dev", "date", "member")})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            try:
                done.add(json.loads(l)["session"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in locked if r["session"] not in done]
    import random
    random.Random(7).shuffle(todo)
    if a.limit:
        todo = todo[:a.limit]
    Path(a.work_root).mkdir(parents=True, exist_ok=True)
    print(f"[aa-fleet] locked={len(locked)} done={len(done)} todo={len(todo)} "
          f"workers={a.workers}", flush=True)
    t0 = time.time()
    n_pass = n_err = 0
    with Pool(a.workers, initializer=_init, initargs=(a.index_dir,)) as pool, \
            open(out, "a") as w:
        jobs = [(r, a.work_root) for r in todo]
        for k, row in enumerate(pool.imap_unordered(_one, jobs, chunksize=1), 1):
            w.write(json.dumps(row) + "\n")
            w.flush()
            n_pass += int(bool(row.get("anchor_pass")))
            n_err += int("error" in row)
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[aa-fleet] {k}/{len(todo)} pass={n_pass} err={n_err} "
                      f"({el:.0f}s, {k/el*3600:.0f}/h)", flush=True)
    print(f"[aa-fleet] DONE pass={n_pass} err={n_err}", flush=True)
    print("WIYH_AA_FLEET_DONE")


if __name__ == "__main__":
    main()
