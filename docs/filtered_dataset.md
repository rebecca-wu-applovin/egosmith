# EgoSmith filtered HOI datasets

`gs://foundational-research/hoi-dataset/egosmith_filtered/`

Hand–object-interaction clips that passed the EgoSmith filter pipeline (**Layer‑1** pre-filter +
**Layer‑4** quality filter). Each dataset is self-contained: frame tars + a filtered clip manifest.
One loader consumes all of them — the GT source is auto-detected per clip.

**Browse samples** (video + keypoint overlay + LLM annotation, ~50 instances/dataset;
requires bucket read access):
<https://storage.cloud.google.com/foundational-research/hoi-dataset/egosmith_filtered/viewer/index.html>
— built by `scripts/viewer/build_viewer.py` (re-run per dataset to refresh).

## Layout

```
egosmith_filtered/
├── taco/            frames/*.tar   filter_run/{clip_manifest.jsonl, clip_manifest.filtered.jsonl, filter_report.json, FILTER_MODE.txt}
├── hot3d/           (same layout)
├── oakink_actions/  (same layout; + program_info/)      # OakInk-v2, per-action segments
├── egodex/          frames_v2/*.tar (shipped; frames/ = pre-gt_joints v1, kept)   filter_run/{clip_manifest.jsonl, clip_manifest.stage1.kept.jsonl, clip_manifest.filtered.jsonl, filter_report.json, FILTER_MODE.txt, _shards/, _prev21_backup/}
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
| egodex | 158,564 | native — GT read straight from the tar | `.image.jpg` + `.lowdim.npy` + `.mano.npy` + `.meta.json` + `.gt_joints.npy` |
| h2o | 149 | `use_gt` — dataset GT MANO+camera → `world_space_res.pth` | `.image.jpg` |
| wiyh_native | see `filter_run/BUCKET_AUDIT.json` | native (TAGGED) — 25-joint glove GT via per-session anchor solve | `.image.jpg` + `.lowdim.npy` + `.mano.npy` + `.meta.json` (with per-frame `gate_px`) |

H2O (ETH, ICCV 2021; egocentric cam4 only, 30 fps, two-hands+object tabletop manipulation;
built by `scripts/build/generate_h2o_world_res.py`, W9 2026-08-25): 184 sequences converted
(1.06 h) → 149 kept (0.85 h) under the canonical GT filter (`--stages infiller --source_fps 30
--target_fps 30 --min_presence_ratio 0.5`); drops are motion-step glitches. License: academic
use only (see `hoi-dataset/H2O/PROVENANCE.md`).

**WIYH native tier caveat (W7 locked-tier ingestion, 2026-08-28):** `wiyh_native` is a
TAGGED approximate tier — every record carries `metadata.finger_quality =
"approximate_35_65px"`. WIYH ships a real 25-joint glove skeleton (50 Hz) + per-frame wrist
SE3, but NOT the glove→eef mounting extrinsic; that extrinsic is solved per session from
vision (auto-anchor on the gloves' fingertip pads; fit 15–45 px, holdout <60 px). Wrist
translation is sensor-locked (census median 0–2 px vs hand masks); finger articulation is
approximate at the 35–65 px level — mask finger-level targets for pixel-tight work; wrist
pose, video, and language annotations are reliable. Only wrist-LOCKED sessions ship (both
hands ≥80% frames <30 px eef-to-hand-mask; 269/4,420 sessions locked, 43 anchored-and-
accepted in v1 — the census + per-session anchor registry are additive, so later anchor
passes can extend the tier without touching shipped clips). Per-frame wrist-gate pixel codes
ship in every frame's `.meta.json`. `wiyh_native` EXTENDS the `wiyh` recon tier (757 clips,
disjoint method — that tier stays as-is); see `egosmith_filtered/wiyh_native/filter_run/FILTER_MODE.txt`.

**DexCap caveat (audit 2026-08-25):** every kept clip carries
`metadata.finger_articulation_unreliable=true` (`severity: severe` for `packaging_*`,
`moderate` for `wipe_*`) — DexCap's native EMF-glove finger articulation contradicts the
video (raw-target projection proof; converter exonerated at 5.7–7.0 mm fit). Mask
finger-level targets in training; wrist pose, video, and language annotations are reliable.
Evidence: `egosmith_filtered/_audits/handedness_audit_2026-08-25/`.

## Dropped datasets

Tombstoned — bucket contents preserved as-is under `egosmith_filtered/<ds>/` with a
`DROPPED.md`, but **not part of the shipped set** (excluded from the viewer root and from
any combined-manifest training builds):

- **assemblyhands** — user-ordered drop 2026-08-26 (data-quality concerns). 477 frame tars +
  filter_run kept in place; see `egosmith_filtered/assemblyhands/DROPPED.md`.

All GT here is **ground truth, not pixel-estimated**. For taco/hot3d/oakink the datasets ship GT
MANO + GT camera, which the converter packages into the pipeline's canonical world-space
`world_space_res.pth` (GT re-expressed, no SLAM/HaWoR estimation). A video-only reconstruction
track (pose estimated from pixels) existed but was dropped — it had scale artifacts.

EgoDex funnel: 338,234 converted → 177,979 Layer‑1 → **158,564** Layer‑4.

**EgoDex full-articulation GT (`.gt_joints.npy`, retrofit 2026-08-26):** every shipped egodex
tar carries one extra member per frame — `{frame}.gt_joints.npy` = **(2, 21, 3) float32**
world-frame joint positions of the full Vision Pro hand skeleton, MANO joint order
(index 0 = left hand; per hand: wrist, then per finger Knuckle/IntermediateBase/
IntermediateTip/Tip for thumb→little; tips at 4,8,12,16,20). Schema tag
`descriptor.extra["gt_joints_schema"] = "vp_world_21_mano_order_v1"`, flag
`extra["gt_joints"] = true`. `.mano.npy` remains the zeros (2,55) placeholder for format
compatibility (`extra["mano_note"]`); the 116-d `.lowdim.npy` is unchanged (wrist+5 tips are
an exact subset of `gt_joints` — validated per clip at conversion, max err 0.0). Per-frame
`presence` in `.meta.json` still gates hand validity. Shipped tars live under `frames_v2/`
(append-only rewrite of v1; original member bytes/offsets preserved verbatim, so
`frame_offsets` stayed valid); the pre-retrofit manifest is backed up at
`filter_run/_prev21_backup/`. Built by `scripts/build/retrofit_egodex_gt_joints.py`.

### Out-of-scope prefixes (user decisions, 2026-08-27)
- `taco-brush-allegro`, `taco-brush-sharpa`, `taco-overall-sharpa(-mirror)` — retargeted
  Allegro/Sharpa robot-hand trajectories derived from TACO (sim scenes + trajectory npz,
  NO RGB video → the L1+L4+LLM pipeline cannot apply). Source demos already in the corpus
  via `taco`. Ignored.
- `taco-dataset-pre-release` — pre-release TACO depth videos, redundant with the processed
  TACO. Ignored.
- EgoVerse `test_*` splits (test_aria, test_eva, test_eva2, proc_test_aria, scale_old,
  scale_test) — HELD OUT pending user confirmation (probable eval splits/duplicates).

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
  `.lowdim.npy` (116‑d Vision‑Pro world pose) + `.mano.npy` + `.gt_joints.npy`
  (full 21-joint articulation, see above; the loader's lowdim path ignores members it
  doesn't know, so `gt_joints` is opt-in). `seq_folder` unused.
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
