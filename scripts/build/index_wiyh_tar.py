#!/usr/bin/env python
"""Member-offset index for WIYH's byte-split scene tars (HoloAssist tar_index pattern).

Each scene ships as ONE plain ustar archive split at raw byte boundaries into
~52.6GB parts misnamed `<Scene>.tar.gz.~NNN` (parts ~001+ start mid-member, so
stream-mode tarfile cannot open them individually). This indexer walks the tar
HEADER chain over the virtual concatenation of all parts — one 512B ranged read
per member, skipping the data — and emits, per scene:

  <out>/<Scene>.parts.json      ordered [{uri, size}]  (offset mapping)
  <out>/<Scene>.members.jsonl   {scene, name, offset, size}
                                offset = ABSOLUTE data offset in the logical tar
                                (header at offset-512); ranged fetch may span parts.

Total cost: ~1 GET/member (~31.5K members over 36.8TB) — no bulk data read.

Usage:
  python scripts/build/index_wiyh_tar.py --out /root/w7_full/wiyh/index \
      [--scenes Candlelight ...] [--gcs_prefix gs://.../WIYH]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gcsfs

BLK = 512
SCENES = ["Apartment", "Banquet", "Candlelight", "Hotel",
          "Laundry", "Logistics", "Office", "Supermarket"]


def list_parts(prefix: str, scene: str) -> list[dict]:
    r = subprocess.run(["gsutil", "ls", "-l", f"{prefix}/{scene}/{scene}.tar.gz.~*"],
                       capture_output=True, text=True, check=True)
    parts = []
    for ln in r.stdout.splitlines():
        f = ln.split()
        if len(f) >= 3 and ".tar.gz.~" in f[-1]:
            parts.append({"uri": f[-1], "size": int(f[0])})
    parts.sort(key=lambda p: int(p["uri"].rsplit("~", 1)[1]))
    # byte-split invariant: every part but the last has identical size
    sizes = {p["size"] for p in parts[:-1]}
    assert len(sizes) <= 1, f"{scene}: non-uniform part sizes {sorted(sizes)[:3]}..."
    return parts


class ConcatRanged:
    """Ranged reads over the logical concatenation of the parts."""

    def __init__(self, fs: gcsfs.GCSFileSystem, parts: list[dict]):
        self.fs = fs
        self.parts = parts
        self.starts = []
        pos = 0
        for p in parts:
            self.starts.append(pos)
            pos += p["size"]
        self.total = pos

    def read(self, offset: int, size: int) -> bytes:
        out = []
        end = min(offset + size, self.total)
        i = 0
        while i < len(self.parts) and self.starts[i] + self.parts[i]["size"] <= offset:
            i += 1
        while offset < end and i < len(self.parts):
            p, s = self.parts[i], self.starts[i]
            lo = offset - s
            hi = min(end - s, p["size"])
            out.append(self.fs.cat_file(p["uri"].replace("gs://", ""), start=lo, end=hi))
            offset += hi - lo
            i += 1
        return b"".join(out)


def _hdr_str(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("utf-8", "replace")


def _hdr_num(b: bytes) -> int:
    if b and (b[0] & 0x80):  # GNU base-256
        n = 0
        for c in b:
            n = (n << 8) | c
        return n & ((1 << (8 * len(b) - 1)) - 1)
    s = _hdr_str(b).strip().strip("\0 ")
    return int(s, 8) if s else 0


def walk_scene(fs, prefix: str, scene: str, out_dir: Path) -> dict:
    parts = list_parts(prefix, scene)
    cf = ConcatRanged(fs, parts)
    (out_dir / f"{scene}.parts.json").write_text(json.dumps(parts, indent=1))
    members, pos, zeros, n_hdr = [], 0, 0, 0
    t0 = time.time()
    pending_longname = None
    while pos + BLK <= cf.total:
        hdr = cf.read(pos, BLK)
        n_hdr += 1
        if hdr == b"\0" * BLK:
            zeros += 1
            if zeros >= 2:
                break
            pos += BLK
            continue
        zeros = 0
        size = _hdr_num(hdr[124:136])
        typ = hdr[156:157]
        name = _hdr_str(hdr[0:100])
        prefix_f = _hdr_str(hdr[345:500]) if hdr[257:262] == b"ustar" else ""
        if prefix_f:
            name = f"{prefix_f}/{name}"
        data_off = pos + BLK
        pos = data_off + ((size + BLK - 1) // BLK) * BLK
        if typ == b"L":  # GNU longname: payload is the next member's name
            pending_longname = _hdr_str(cf.read(data_off, size))
            continue
        if typ in (b"x", b"g"):
            continue
        if pending_longname:
            name = pending_longname
            pending_longname = None
        if typ in (b"0", b"\x00") and name.endswith(".tar.gz"):
            members.append({"scene": scene, "name": name, "offset": data_off, "size": size})
            if len(members) % 500 == 0:
                print(f"  [{scene}] {len(members)} members, pos {pos/1e12:.2f}TB "
                      f"({time.time()-t0:.0f}s)", flush=True)
    with open(out_dir / f"{scene}.members.jsonl", "w") as f:
        for m in members:
            f.write(json.dumps(m) + "\n")
    span = sum(m["size"] for m in members)
    rep = {"scene": scene, "parts": len(parts), "total_bytes": cf.total,
           "members": len(members), "member_bytes": span,
           "coverage": round(span / cf.total, 4), "header_reads": n_hdr,
           "wall_sec": round(time.time() - t0, 1)}
    print(f"[{scene}] DONE {rep}", flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs_prefix", default="gs://foundational-research/hoi-dataset/WIYH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(scene):
        fs = gcsfs.GCSFileSystem()  # per-thread instance
        try:
            return walk_scene(fs, args.gcs_prefix, scene, out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[{scene}] FAILED: {type(e).__name__}: {e}", flush=True)
            return {"scene": scene, "error": str(e)[:200]}

    with ThreadPoolExecutor(args.parallel) as ex:
        reps = list(ex.map(one, args.scenes))
    (out_dir / "_summary.json").write_text(json.dumps(reps, indent=1))
    ok = [r for r in reps if "error" not in r]
    print(f"[wiyh-index] {sum(r['members'] for r in ok)} members / "
          f"{sum(r['total_bytes'] for r in ok)/1e12:.2f}TB across {len(ok)}/{len(reps)} scenes")


if __name__ == "__main__":
    sys.exit(main())
