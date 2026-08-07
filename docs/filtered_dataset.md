# EgoSmith filtered HOI datasets

`gs://foundational-research/hoi-dataset/egosmith_filtered/`

Hand–object-interaction clips that passed the EgoSmith filter pipeline (**Layer‑1** pre-filter +
**Layer‑4** quality filter). Each dataset is self-contained: frame tars + a filtered clip manifest.
One loader consumes all of them — the GT source is auto-detected per clip.

## Layout

```
egosmith_filtered/
├── taco/            frames/*.tar   filter_run/{clip_manifest.jsonl, clip_manifest.filtered.jsonl, filter_report.json, FILTER_MODE.txt}
├── hot3d/           (same layout)
├── oakink_actions/  (same layout; + program_info/)      # OakInk-v2, per-action segments
├── egodex/          frames/*.tar   filter_run/{clip_manifest.jsonl, clip_manifest.stage1.kept.jsonl, clip_manifest.filtered.jsonl, filter_report.json, FILTER_MODE.txt, _shards/}
├── weights/         (model weights used by the pipeline — not dataset clips)
├── scripts/  notebooks/            (tooling — not dataset clips)
└── README.md        (this file)
```

- **`clip_manifest.filtered.jsonl` is the shipped dataset** for each dataset (one JSON record per kept clip).
- `clip_manifest.jsonl` = pre-filter input population; for egodex, `clip_manifest.stage1.kept.jsonl` = Layer‑1 kept.
- **Recon datasets** (taco/hot3d/oakink_actions) also have per-clip pose under
  `gs://…/egosmith_recon/<ds>/use_gt/outputs/<clip>/world_space_res.pth`.

## Kept-clip counts

| dataset | kept (filtered) | GT mode | frame-tar payload |
|---|---:|---|---|
| taco | 1,846 | `use_gt` — dataset GT MANO+camera → `world_space_res.pth` | `.image.jpg` |
| hot3d | 357 | `use_gt` — dataset GT MANO+camera → `world_space_res.pth` | `.image.jpg` |
| oakink_actions | 2,488 | `use_gt` — dataset GT MANO+camera → `world_space_res.pth` | `.image.jpg` |
| egodex | 158,564 | native — GT read straight from the tar | `.image.jpg` + `.lowdim.npy` + `.mano.npy` + `.meta.json` |

All GT here is **ground truth, not pixel-estimated**. For taco/hot3d/oakink the datasets ship GT
MANO + GT camera, which the converter packages into the pipeline's canonical world-space
`world_space_res.pth` (GT re-expressed, no SLAM/HaWoR estimation). A video-only reconstruction
track (pose estimated from pixels) existed but was dropped — it had scale artifacts.

EgoDex funnel: 338,234 converted → 177,979 Layer‑1 → **158,564** Layer‑4.

## Record / descriptor schema

Each manifest line is a `ClipManifestRecord`
(`src/lib/pipeline/clips/clip_manifest.py`):

```json
{"clip_id","source_id","split","group_id","descriptor":{...},"metadata":{...}}
```

`descriptor` is a `ClipDescriptor` (`src/lib/pipeline/datasets/descriptors.py`):
`clip_id, clip_name, storage_kind="tar_shard", root_dir, seq_folder, frame_names[],
frame_offsets[[off,size],…], shard_path, fps/width/height, extra{}`.
All paths are `gs://` URLs. `frame_offsets` allow direct ranged reads of each JPEG in the tar.

## Two GT modes (auto-detected — no per-dataset code)

The single loader `load_descriptor_episode_features`
(`src/lib/pipeline/exporters/manifest_build/episodes.py`) branches on
`descriptor.extra["native_feature_source"]`:

- **native** (`== "wds_lowdim_mano_v1"`, EgoDex): per-frame GT read **from the tar** —
  `.lowdim.npy` (116‑d Vision‑Pro world pose) + `.mano.npy`. `seq_folder` unused.
- **use_gt** (taco/hot3d/oakink_actions): the dataset's **GT** world-space MANO pose, stored in
  `seq_folder/world_space_res.pth` (`trans/rot/hand_pose/betas`, 2 hands × T), then MANO forward →
  lowdim features. This is GT-derived (GT MANO + GT camera converted to canonical world space) — not
  estimated from pixels.

So every dataset flows through the same code path; you do **not** special-case by dataset name.

## How to use it downstream

**Mental model:** each dataset *is* its `clip_manifest.filtered.jsonl`. Every line is one kept clip
and tells you where its frames and ground-truth live. That's the whole dataset.

**Three steps to consume it:**
1. **Read the manifest** → one record per clip.
2. **Get frames** — each clip is a `.tar`; read each JPEG straight from its byte offset
   (`frame_offsets`), no unpacking.
3. **Get ground-truth** — chosen automatically from `descriptor.extra["native_feature_source"]`:
   - set → **egodex**: pose is *inside* the tar (`.lowdim.npy` 116-d + `.mano.npy`).
   - absent → **taco/hot3d/oakink_actions**: pose is in `seq_folder/world_space_res.pth`.

You write the same code for all four datasets — the GT mode is chosen for you.

**Easiest — the VLA builder CLI** (manifest → training episodes):

```bash
python scripts/build/build_vla_from_manifest.py \
  --manifest gs://.../egosmith_filtered/<ds>/filter_run/clip_manifest.filtered.jsonl \
  --output <out_dir>
```
To combine datasets, concatenate their manifests.

**Or in your own loader:**

```python
from lib.pipeline.clips.clip_manifest import ClipManifestRecord
from lib.pipeline.exporters.manifest_build.episodes import load_descriptor_episode_features
for line in open("clip_manifest.filtered.jsonl"):
    rec = ClipManifestRecord.from_json(line)
    feats = load_descriptor_episode_features(rec.descriptor, ...)  # frames + GT, mode auto-handled
```

**Two things to remember:**
- **Frames** are read by byte offset from the `.tar` (`src/lib/pipeline/io/frame_sources.py`) — don't untar.
- The **`world_space_res.pth`** (use_gt datasets) is a repo-specific format — load it via the repo loader
  (`load_descriptor_episode_features` / `_load_world_space_prediction`), **not** raw `torch.load`.
