#!/usr/bin/env python
"""Stage-1 pre-filter over EgoVerse ZARR-episode datasets (abc; eva zarr twin).

The abc subset stores episodes as zarr-v3 groups of vlen-bytes JPEG arrays
(images.front_1 + wrist cams) — no mp4 objects. This driver runs the SAME
Stage-1 gates as production (lib.clip.heuristic_video_clipper:
analyze_frame_source_intervals — Gate A YOLO hands + Gate B optical-flow RANSAC
+ Gate C span merge) over images.front_1, fetching ONLY the sampled frames
(every skip_frames-th, default 15) via ranged GCS reads of the zarr shard —
~15x less I/O than a full decode.

The vendor zarrs carry corrupt crc32c checksums in their shard indexes
(zarr-python refuses them), so the shard index + chunks are parsed manually:
index = last (n*16 + 4) bytes -> (offset,nbytes) u8 pairs; each inner chunk is
zstd-compressed vlen-bytes (8-byte prefix, JPEG SOI located explicitly).

Input: --videos_list JSONL, one episode per line:
  {"uri": "gs://.../episode.zarr", "group": "<task or date>"}
Output: survivor JSONL + funnel report, same schema as egocentric_stage1_video.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from lib.clip.heuristic_video_clipper import (  # noqa: E402
    load_clip_config, analyze_frame_source_intervals, _load_yolo,
)


def _fs():
    import gcsfs
    return gcsfs.GCSFileSystem()


class _ZarrShardReader:
    """Manual reader for one zarr-v3 sharding_indexed vlen-bytes array (1 shard file)."""

    def __init__(self, fs, array_dir: str):
        self.fs = fs
        zj = json.loads(fs.cat(f"{array_dir}/zarr.json"))
        self.n = int(zj["shape"][0])
        self.shard = f"{array_dir}/c/0"
        size = fs.info(self.shard)["size"]
        idx_len = self.n * 16 + 4
        raw = fs.read_block(self.shard, size - idx_len, idx_len)
        self.index = np.frombuffer(raw[: self.n * 16], dtype="<u8").reshape(self.n, 2)
        import zstandard as zstd
        self._dec = zstd.ZstdDecompressor()

    def frame(self, i: int) -> bytes | None:
        off, nb = int(self.index[i, 0]), int(self.index[i, 1])
        if nb == 0 or off == 2**64 - 1:
            return None
        d = self._dec.decompress(self.fs.read_block(self.shard, off, nb),
                                 max_output_size=100_000_000)
        j = d.find(b"\xff\xd8\xff")
        return d[j:] if j >= 0 else None


class _ZarrSource:
    """Frame source over a zarr JPEG array; only sampled indices are fetched."""

    def __init__(self, reader: _ZarrShardReader):
        self._r = reader
        self.last_decoded = 0

    def __len__(self):
        return self._r.n

    def get_frame(self, i, rgb=False):
        b = self._r.frame(i)
        if b is None:
            raise EOFError("empty chunk")
        fr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if fr is None:
            raise EOFError("bad jpeg")
        self.last_decoded = i + 1
        if fr.ndim == 2:
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if rgb else fr

    def release(self):
        pass


def run_one(rec: dict, fs, cfg: dict, model, image_key: str) -> dict:
    uri = rec["uri"].rstrip("/")
    out = {"uri": rec["uri"], "group": rec.get("group", "")}
    try:
        root = uri.replace("gs://", "")
        attrs = json.loads(fs.cat(f"{root}/zarr.json")).get("attributes", {})
        fps = float(attrs.get("fps") or 30.0)
        reader = _ZarrShardReader(fs, f"{root}/{image_key}")
        src = _ZarrSource(reader)
        t0 = time.time()
        ivs, info = analyze_frame_source_intervals(src, cfg, model=model, fps=fps)
        kept_sec = sum(iv.end_sec - iv.start_sec for iv in ivs)
        out.update({
            "fps": round(fps, 3), "n_frames": reader.n,
            "duration_sec": round(reader.n / fps, 1),
            "kept_sec": round(kept_sec, 2),
            "valid_frac": round(info["valid_sample_count"] / max(1, info["sample_count"]), 3),
            "intervals": [iv.to_dict() for iv in ivs],
            "embodiment": attrs.get("embodiment"),
            "task_name": attrs.get("task_name"),
            "analyze_sec": round(time.time() - t0, 1),
        })
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_list", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--config", default=str(_REPO / "src/lib/clip/heuristic_clip_config.yaml"))
    ap.add_argument("--detector", default=str(_REPO / "weights/external/detector.pt"))
    ap.add_argument("--image_key", default="images.front_1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_clip_config(args.config)
    model = _load_yolo(args.detector)
    if model is None:
        raise SystemExit(f"YOLO detector not found at {args.detector}")
    try:
        model.to("cuda:0")
    except Exception:  # noqa: BLE001
        pass
    fs = _fs()

    recs = [json.loads(l) for l in open(args.videos_list) if l.strip()]
    if args.limit:
        recs = recs[: args.limit]

    kept, raw_sec, kept_sec, drops = [], 0.0, 0.0, Counter()
    t0 = time.time()
    with open(args.out_manifest + ".tmp", "w") as w:
        for i, rec in enumerate(recs):
            out = run_one(rec, fs, cfg, model, args.image_key)
            raw_sec += out.get("duration_sec", 0) or 0
            if out.get("error"):
                drops["error_" + out["error"].split(":")[0]] += 1
            elif out.get("kept_sec", 0) > 0:
                kept.append(out)
                kept_sec += out["kept_sec"]
                w.write(json.dumps(out) + "\n")
            else:
                drops["no_valid_span"] += 1
            if (i + 1) % 10 == 0 or i + 1 == len(recs):
                print(f"[{i+1}/{len(recs)}] kept={len(kept)} kept_h={kept_sec/3600:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    os.replace(args.out_manifest + ".tmp", args.out_manifest)
    report = {
        "videos": len(recs), "kept_videos": len(kept),
        "dropped_videos": len(recs) - len(kept),
        "raw_hours": round(raw_sec / 3600, 3),
        "analyzed_hours": round(raw_sec / 3600, 3),
        "kept_hours": round(kept_sec / 3600, 4),
        "hours_fraction": round(kept_sec / max(1e-9, raw_sec), 4),
        "wallclock_sec": round(time.time() - t0, 1),
        "drop_reasons": dict(drops),
    }
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[zarr-stage1] {len(kept)}/{len(recs)} kept, {kept_sec/3600:.3f}h of "
          f"{raw_sec/3600:.3f}h -> {args.out_manifest}", flush=True)


if __name__ == "__main__":
    main()
