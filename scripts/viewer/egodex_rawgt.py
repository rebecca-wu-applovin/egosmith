#!/usr/bin/env python
"""Raw EgoDex ground-truth access: clip_id -> full 21-joint Vision Pro hand skeletons.

The shipped egodex WDS tars carry only the 116-d lowdim (wrist + 5 fingertips;
.mano.npy is a zeros_2x55 placeholder), but the raw EgoDex release ships per-joint
SE(3) 4x4 world transforms for the full Vision Pro hand skeleton per episode HDF5.

Raw layout on GCS: gs://foundational-research/egodex/{part1..part5,extra,test}.zip,
members "<part>/<task>/<num>.hdf5" (+ .mp4). We ranged-read single hdf5 members via
zipfile-over-gcsfs (central directory ~2 s once per part, ~1-3 MB per episode hdf5).

clip_id format (generate_egodex_wds.py): egodex_<part>_ep<task>_<num>, task
sanitized with " " -> "-". Frames in the tar are 1:1 with hdf5 timesteps
(t = frame index, T = min(N_hdf5, N_mp4)), so no resampling is needed — validate
per clip by matching the lowdim wrist/fingertips (same hdf5 source) anyway.

Joint order (MANO-compatible, matches render_clip_card.EDGES):
  0 wrist, then per finger [Knuckle, IntermediateBase, IntermediateTip, Tip]
  for Thumb, Index, Middle, Ring, Little  -> tips at 4, 8, 12, 16, 20.
"""
import io
import re
import zipfile

import numpy as np

RAW_PREFIX = "foundational-research/egodex"
FINGERS = ["Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger"]
SUFFIXES = ["Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]
TIP_IDX = [4, 8, 12, 16, 20]
_CID_RE = re.compile(r"^egodex_([A-Za-z0-9]+)_ep(.+)_(\d+)$")


def joint_names(side):
    """21 joint names in MANO order for side in {'left','right'}."""
    names = [f"{side}Hand"]
    for fg in FINGERS:
        names += [f"{side}{fg}{suf}" for suf in SUFFIXES]
    return names


class EgoDexRawGT:
    """Lazy per-part zip handles + sanitized-task -> member-dir index."""

    def __init__(self, fs=None, raw_prefix=RAW_PREFIX):
        if fs is None:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
        self.fs = fs
        self.raw_prefix = raw_prefix
        self._zips = {}      # part -> ZipFile
        self._taskdir = {}   # part -> {sanitized_task: zip member dir}

    def _zip(self, part):
        if part not in self._zips:
            f = self.fs.open(f"{self.raw_prefix}/{part}.zip", "rb",
                             block_size=8 * 2 ** 20, cache_type="readahead")
            zf = zipfile.ZipFile(f)
            taskdir = {}
            for n in zf.namelist():
                if n.endswith(".hdf5") and "/" in n:
                    d = n.rsplit("/", 1)[0]
                    taskdir[d.rsplit("/", 1)[-1].replace(" ", "-")] = d
            self._zips[part] = zf
            self._taskdir[part] = taskdir
        return self._zips[part], self._taskdir[part]

    def resolve(self, clip_id):
        """clip_id -> (part, zip member path). Raises KeyError if unresolvable."""
        m = _CID_RE.match(clip_id)
        if not m:
            raise KeyError(f"unparseable egodex clip_id: {clip_id}")
        part, rest = m.group(1), f"{m.group(2)}_{m.group(3)}"
        _, taskdir = self._zip(part)
        # task names may contain underscores and digits -> longest-match against
        # the actual task list, requiring a purely-numeric episode suffix
        for t in sorted(taskdir, key=len, reverse=True):
            if rest.startswith(t + "_") and rest[len(t) + 1:].isdigit():
                return part, f"{taskdir[t]}/{rest[len(t) + 1:]}.hdf5"
        raise KeyError(f"no raw task match for {clip_id} in {part}.zip")

    def read_hdf5_bytes(self, clip_id):
        part, member = self.resolve(clip_id)
        zf, _ = self._zip(part)
        return zf.open(member).read()

    def joints(self, clip_id):
        """(L, R) world-frame joint positions, each (T, 21, 3) float64."""
        import h5py
        with h5py.File(io.BytesIO(self.read_hdf5_bytes(clip_id)), "r") as f:
            out = []
            for side in ("left", "right"):
                out.append(np.stack([f[f"transforms/{n}"][:, :3, 3]
                                     for n in joint_names(side)], 1).astype(np.float64))
        return out[0], out[1]
