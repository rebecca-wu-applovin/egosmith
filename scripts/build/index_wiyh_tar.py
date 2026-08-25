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
    # Part sizes are USUALLY uniform 52.6GB but not always (Apartment/Office/
    # Supermarket carry odd-sized parts). Order correctness is self-validated by
    # the header-chain walk: a mis-ordered part derails the chain immediately
    # (garbage size fields), so coverage ~1.0 in the report proves the order.
    sizes = {p["size"] for p in parts[:-1]}
    if len(sizes) > 1:
        print(f"[{scene}] note: non-uniform part sizes ({len(sizes)} distinct); "
              "relying on header-chain self-validation", flush=True)
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


_NAME_RE = re.compile(rb"worldcode_.*\.tar\.gz")


def _valid_hdr(hdr: bytes) -> bool:
    if hdr[257:262] != b"ustar":
        return False
    try:
        _hdr_num(hdr[124:136])
    except ValueError:
        return False
    name = hdr[0:100].split(b"\0", 1)[0]
    return bool(_NAME_RE.search(name)) or name.startswith(b"WIYH-")


def _scan_span(cf: "ConcatRanged", pos: int, limit: int) -> int | None:
    """Unaligned scan for the next valid member header within [pos, pos+limit)."""
    win = 16 * 1024 * 1024
    scanned = 0
    while scanned < limit and pos + BLK <= cf.total:
        n = min(win, cf.total - pos)
        chunk = cf.read(pos, n)
        i = 0
        while True:
            i = chunk.find(b"ustar", i)
            if i < 0 or i - 257 + BLK > len(chunk):
                break
            cand = i - 257
            if cand >= 0 and _valid_hdr(chunk[cand:cand + BLK]):
                return pos + cand
            i += 1
        # overlap windows by 512 so a header straddling the seam isn't missed
        step = max(1, n - BLK)
        pos += step
        scanned += step
    return None


def _resync(cf: "ConcatRanged", pos: int, limit: int = 2 * 10**9) -> int | None:
    """Find the next valid member header after a damaged span. The odd re-split /
    overlapping parts (Apartment/Office/Supermarket) can shift alignment by
    arbitrary byte counts, so the scan is UNALIGNED; if nothing is found near the
    break, hop to each subsequent part boundary and scan there."""
    hit = _scan_span(cf, pos, limit)
    if hit is not None:
        return hit
    for s in cf.starts:
        if s <= pos:
            continue
        hit = _scan_span(cf, s, 256 * 1024 * 1024)
        if hit is not None:
            return hit
    return None


def walk_scene(fs, prefix: str, scene: str, out_dir: Path) -> dict:
    parts = list_parts(prefix, scene)
    cf = ConcatRanged(fs, parts)
    (out_dir / f"{scene}.parts.json").write_text(json.dumps(parts, indent=1))
    members, pos, zeros, n_hdr = [], 0, 0, 0
    resyncs, skipped_bytes = 0, 0
    t0 = time.time()
    pending_longname = None

    def try_resync(from_pos, why):
        nonlocal resyncs, skipped_bytes, zeros, pending_longname
        nxt = _resync(cf, from_pos)
        if nxt is None:
            return None
        resyncs += 1
        skipped_bytes += nxt - from_pos
        zeros = 0
        pending_longname = None
        print(f"  [{scene}] RESYNC #{resyncs} ({why}) at {from_pos/1e12:.3f}TB -> "
              f"+{(nxt-from_pos)/1e9:.2f}GB skipped", flush=True)
        return nxt

    while pos + BLK <= cf.total:
        hdr = cf.read(pos, BLK)
        n_hdr += 1
        if hdr == b"\0" * BLK:
            zeros += 1
            if zeros >= 2:
                # end-of-archive marker; if plenty of bytes remain, another
                # span may follow the damaged/padded region
                if cf.total - pos > 10**9:
                    nxt = try_resync(pos + BLK, "zeros-mid-stream")
                    if nxt is not None:
                        pos = nxt
                        continue
                break
            pos += BLK
            continue
        zeros = 0
        if not _valid_hdr(hdr) and hdr[257:262] != b"ustar":
            nxt = try_resync(pos, "bad-header")
            if nxt is None:
                print(f"  [{scene}] chain lost at {pos/1e12:.3f}TB, no resync in 4GB — stop",
                      flush=True)
                break
            pos = nxt
            continue
        try:
            size = _hdr_num(hdr[124:136])
        except ValueError:
            nxt = try_resync(pos, "bad-size-field")
            if nxt is None:
                break
            pos = nxt
            continue
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
           "resyncs": resyncs, "skipped_bytes": skipped_bytes,
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
