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
| gigahands | 2,076 (4.24 h, first-person views only; ego trim 2026-08-28) | `use_gt` — EasyMocap bimanual MANO GT + per-scene calib → `world_space_res.pth` (probe 1.6–2.1 mm) | `.image.jpg` |
| wiyh_native | 440 (1.16 h, 12 sessions; post strict-audit remediation 2026-08-28) | native (TAGGED) — 25-joint glove GT via per-session anchor solve | `.image.jpg` + `.lowdim.npy` + `.mano.npy` + `.meta.json` (with per-frame `gate_px`) + `.gt_joints.npy` (frames_v2) |
| humantouch | 50,475 (125.2 h) | `gt_derived_extrinsic_block_anchor` — MANUS glove GT staged to MANO; camera extrinsic per mount-cluster from 118 vision-annotated anchors (4.5° assignment cap post-remediation, gated propagation) → `world_space_res.pth` | `.image.jpg` (sharded `frames/shard_XXXXX/`) |
| egoverse_mecka_freeform | 56,365 (785.9 h) | recon — HaWoR conveyor (456w/15fps, anycalib, Phase-D with BOTH presence gates) | `.image.jpg` (sharded `frames/shard_XXXXX/`) |
| egoverse_mecka_flagship | 45,224 (97.6 h) | native — in-zarr 21-kpt GT (convention A) → 116-d lowdim; frames = in-zarr 640×360 JPEGs | `.image.jpg` + `.lowdim.npy` + `.mano.npy` + `.meta.json` + `.gt_joints.npy` `egoverse_world_21_mano_order_v1` (sharded `frames_v2/shard_XXXXX/`) |
| egoverse_lightwheel | 25,462 (43.5 h) | native — pose.json 21-kpt world + wrist quat through per-frame R_w2c + undistorted K (1920×1456) | same native payload + `.gt_joints.npy` `lightwheel_world_21_mano_order_v1` (sharded `frames_v2/`) |
| egoverse_microagi | 649,264 (1,658.2 h @29 fps) | native — in-zarr 21-kpt GT; annotations text-seeded (see caveat) | same native payload + `.gt_joints.npy` `egoverse_world_21_mano_order_v1` (sharded `frames_v2/`) |
| egoverse_scale | 51,574 (140.8 h) | native — in-zarr 21-kpt GT; frames = in-zarr 640×480 JPEGs (`recenter_world` re-gauge applied) | same native payload + `.gt_joints.npy` `egoverse_world_21_mano_order_v1` (sharded `frames_v2/`) |
| egoexo4d | 16,516 (39.81 h) | recon — HaWoR conveyor (aria ego RGB, kb4 undistort, 15fps; Phase-D with BOTH presence gates from the first shard) | `.image.jpg` (sharded; manifests in `filter_run/_shards/` + aggregated `clip_manifest.filtered.jsonl`) |

H2O (ETH, ICCV 2021; egocentric cam4 only, 30 fps, two-hands+object tabletop manipulation;
built by `scripts/build/generate_h2o_world_res.py`, W9 2026-08-25): 184 sequences converted
(1.06 h) → 149 kept (0.85 h) under the canonical GT filter (`--stages infiller --source_fps 30
--target_fps 30 --min_presence_ratio 0.5`); drops are motion-step glitches. License: academic
use only (see `hoi-dataset/H2O/PROVENANCE.md`).

GigaHands (Fu et al., CVPR'25; BRICS multi-camera rig, 30 fps 1280×720, bimanual
tabletop activities; built by `scripts/build/generate_gigahands_world_res.py`, W6 redo
2026-08-28): 12,775 sequences → 12,764 converted (32.73 h; single best camera per
sequence, undistorted to pinhole) → **11,542 kept (27.47 h)** under the canonical GT
filter with BOTH presence gates (`--min_presence_ratio 0.5
--min_presence_ratio_per_hand 0.5`, 30 fps step gates). The 2026-08-25 filter pass ran
without the presence gates and was never shipped (kept as
`filter_run/filter_report.v1_no_presence_gates.json`). No annotations — shipped under
the global labeling hold; see `filter_run/BUCKET_AUDIT.json` `annotation_status`.
**Ego-only trim (2026-08-28):** 11,542 → **2,076 kept (4.24 h)**. GigaHands has no GoPro
footage — "gopro" in scene names is the manipulated object; every clip is a static
brics-odroid rig view. A per-(scene, camera-slot) frame audit of all 1,376 pairs
(`filter_run/ego_audit_20260828/`) found 175 first-person pairs (camera at the
participant's head position, e.g. `001_cam0/cam1` in every audited scene); the other
9,466 clips (23.24 h) were dropped as `non_egocentric_view` — tars moved to
`_dropped_egotrim_20260828/frames/`, reconciliation in
`filter_run/egotrim_reconciliation.json`, pre-trim state in
`filter_run/_preegotrim_backup_20260828/`. use_gt pose outputs remain untouched
(view-independent; still cover the dropped clips).

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
disjoint method — that tier stays as-is, 744 clips after the presence re-filter);
see `egosmith_filtered/wiyh_native/filter_run/FILTER_MODE.txt`.

**WIYH native strict-audit remediation (2026-08-28):** a strict per-session re-audit
(mask-gated + eef-vetoed teal-pad detection, two-way px metric on the original fisheye,
plus a visual read per session) found **20 of the 32 shipped sessions visually WRONG** —
auto-anchor ROTATION errors on washed-out Supermarket pads that the fit/holdout gates and
the v1 post-ship verifier (no mask gating → teal false positives on shelf products) failed
to catch. The 20 sessions (325 clips, 0.87 h) were dropped with reason
`anchor_rotation_invalid`; the tier now ships **440 clips / 1.16 h / 12 sessions** (hours
are frame-count hours; the earlier 1.22 h figure was a 10 s/clip approximation). The keep
decision is **visual-verdict-driven**: the strict med_b metric floor is scene-dependent
(pad detectability varies with lighting/background), so 7/12 keepers measure med_b > 60 px
fisheye while tracking correctly on visual read; measured strict numbers (med_b 41–78 px
@1920w) are stamped as-is on every record as `metadata.tip_verification` (the unreliable v1
`tip_verification_med_px_456w` stamps are removed). Annotations were pruned to the 440
survivors (row deletion only — global labeling hold, no re-labeling). Dropped tars:
`wiyh_native/_dropped_20260828/frames/`; pre-remediation state + reconciliation:
`wiyh_native/filter_run/_preremediation_backup_20260828/`. Verifier replaced by the strict
`scripts/build/wiyh_verify_shipped_tips.py`. **25-joint retrofit:** every surviving clip's
tar (shipped as `frames_v2/`, append-only rewrite, v1 tars byte-preserved) carries one
`{frame}.gt_joints.npy = (2, 25, 3) float32` world(=chest)-frame MANUS skeleton per frame
(index 0 = left; thumb 0–3, index 4–8, middle 9–13, ring 14–18, little 19–23, wrist 24;
tips [3,8,13,18,23]), recomputed from source exactly as the extractor and validated per
clip against the shipped lowdim wrist+tips;
`descriptor.extra.gt_joints_schema = "wiyh_manus_world_25_v1"`
(`scripts/build/retrofit_wiyh_gt_joints.py`).

**DexCap caveat (audit 2026-08-25):** every kept clip carries
`metadata.finger_articulation_unreliable=true` (`severity: severe` for `packaging_*`,
`moderate` for `wipe_*`) — DexCap's native EMF-glove finger articulation contradicts the
video (raw-target projection proof; converter exonerated at 5.7–7.0 mm fit). Mask
finger-level targets in training; wrist pose, video, and language annotations are reliable.
Evidence: `egosmith_filtered/_audits/handedness_audit_2026-08-25/`.

**HumanTouch caveats (ship 2026-08-28):** camera extrinsics are solved per
mount-cluster from manual glove-landmark anchors (118 anchors, held-out validated,
fit p50 ~25 px) — absolute world/camera-frame translation carries weak observability
on ~18 clusters (bias up to ~0.2 m possible; rotation and reprojection consistency
are validated; relative/articulated pose is glove-GT-grade). MANUS thumb-chain GT
reprojects 60–100 px off across the dataset (glove-internal calibration) — treat
thumb articulation as lower confidence. Per-clip provenance:
`descriptor.extra.mount_block` + `block_fit_median_px`. Full details:
`egosmith_filtered/humantouch/filter_run/FILTER_MODE.txt`; anchor evidence:
`filter_run/anchor_audit/` + `filter_run/anchor_results/`. LABELING HOLD
(user-ordered 2026-08-28): the tier ships unlabeled — no LLM annotations.

**HumanTouch remediation (2026-08-28):** a stratified 100-clip visual alignment
audit found ~0.8% of clips visually off, concentrated in far anchor assignments
and 6 mount blocks. Remediation applied to the shipped tier: (1) anchor-assignment
radius tightened to **4.5°** (436 clips dropped); (2) per-episode render QA of
blocks A029/A060/A119/A085/A097/A098 (1,329 episodes; 578 renders visually read)
dropped 40 visually-off episodes (111 clips). Post-remediation: **50,475 clips /
125.2 h**. Dropped tars preserved under `humantouch/_dropped_20260828/frames/`;
reconciliation `filter_run/remediation_20260828.json`; pre-change backup
`filter_run/_preremediation_backup_20260828/`. Standing ship gate: any re-ship
must run `scripts/build/humantouch_ship_gate_qa.py` (≥100-clip stratified
visual QA, weighted off-rate ≤ 2%). **Gate result 2026-08-28: FAIL** — 6/100
visually off (bin-weighted 5.9%), a scattered per-episode mis-assignment long
tail across 6 non-remediated blocks (inside the original audit's 0.2–9.3%
band). Treat clip-level alignment as ~94% locked / ~6% off until a follow-up
pass is approved; evidence `filter_run/ship_gate_qa_20260828/`.

**EgoVerse ships (2026-08-28, five datasets):**
- **mecka freeform/flagship dedup-by-construction:** 3,300 episode ids exist in BOTH
  mecka subsets as the *same recording* (flagship mp4 = 320×180 downscale of the
  freeform 960×540 video; frame-verified). The 3,292 GT-bearing + stage1-kept ones
  ship ONCE, natively via `egoverse_mecka_flagship`, and were excluded from the
  freeform recon reshard (2 ids fell in the gap — failed flagship Stage-1, ~0.05 h,
  dropped). Do not concatenate the two datasets expecting disjoint recordings beyond
  this rule; clip ids share the episode-id stem across both.
- **flagship is native-only by measured verdict:** recon of the 320×180 mp4s is
  degraded even with GT intrinsics injected (camera-frame hand MPJPE med ~96 mm vs
  the 26–49 mm calibrated-regime baseline; persistent 0.7 pa_scale). Evidence:
  `egosmith_filtered/_audits/mecka_flagship_320p_recon_smoke_2026-08-27/VERDICT.md`.
  The GT-less ~47% of flagship *episodes* was dropped — measured at only 11.7 raw-h /
  0.25 stage1-kept-h (avg 2.5 s episodes), see the same VERDICT + the census tool
  `scripts/inspection/mecka_flagship_gt_census.py`.
- **microagi annotations are text-seeded, not per-clip VLM:** 649,264 clips would cost
  $1.5–2.5K to VLM-label; instead each of 246,468 unique in-zarr GT activity texts was
  expanded ONCE (text-only gpt-5-mini) into level1–4 instructions and broadcast to its
  clips ($61.56). Every row carries `"seeded_from": "in_zarr_text"`; level4 hand-centric
  detail is generic (no frames were shown). QA: 256-clip independent VLM sample shows
  80% content-token agreement on level1; disagreements are segmentation granularity,
  not contradictions — evidence `_audits/microagi_text_seeded_qa_2026-08-28/`.
- **Sub-100% annotation coverage (labeling hold):** a global labeling hold froze
  top-ups; residual unannotated clips are freeform 19, flagship 19, lightwheel 11,
  scale 55 (≤0.11% each; transient labeler errors, ids in
  `filter_run/annotations_v4/_shards/*.errors.jsonl`). microagi is 100% (broadcast).
- **egoverse_scale gauge note:** scale's raw world origin sits ~43 m from the camera;
  lowdim was re-gauged per episode to the first-frame camera centre
  (`recenter_world`, rigid translation — physics unchanged). World coordinates are
  per-episode frames across ALL EgoVerse natives (as with EgoDex/ARKit).
- Stage-1 interval gating: lightwheel and flagship converted only Stage-1 kept spans;
  microagi/scale ran the on-pod Stage-1 prefilter (tuned `min_area 0.005`, `min_hands 1`
  configs — production `min_area 0.02` kept 3/21 on the microagi pilot).

## Presence re-filter (2026-08-27/28) — empty-pose + single-valid-hand purge

Two keep-classes were purged from every shipped **recon-path** dataset whose Stage-1
certified two visible hands (`min_hands=2`):

- `empty_pose` — any-hand valid-pose ratio < 0.5 (clips with zero/near-zero valid
  reconstructed poses passed the old motion gates trivially; the presence gate was off
  when egocentric100k/10k were filtered),
- `single_valid_hand` — either hand's valid ratio < 0.5 (Stage-1 promised two hands;
  reconstruction delivered one).

Mechanics: `scripts/build/presence_refilter.py` ranged-reads each kept clip's
`pred_valid (2,T)` from its `result.npz` on GCS, rewrites the per-shard filtered
manifests, and prunes annotation rows for dropped clips (row deletion only — the global
labeling hold forbids new labels). Backup-first: originals live in each dataset's
`filter_run/_prepresence2_backup/`; totals + per-clip drop lists in
`filter_run/presence_refilter_reconciliation.json`. Top-level
`clip_manifest.filtered.jsonl` / `funnel.json` aggregates were rebuilt from the purged
shards where they exist (100k via server-side GCS compose, byte-sum verified).

| dataset | kept before → after | empty_pose | single_valid_hand | removed hours |
|---|---:|---:|---:|---:|
| egocentric100k | 13,111,076 → 12,811,150 | 27,170 | 272,756 | 272.7 |
| egocentric10k | 1,337,259 → 1,305,039 | 3,757 | 28,463 | 31.4 |
| ego4d | 137,106 → 127,944 | 0 | 9,162 | 12.0 |
| holoassist | 18,403 → 17,981 | 0 | 422 | 0.4 |
| assembly101 | 14,303 → 13,910 | 0 | 393 | 0.4 |
| epic_kitchens_100 | 11,864 → 11,643 | 0 | 221 | 0.2 |
| egoverse_aria_v2 | 6,903 → 6,802 | 0 | 101 | 0.1 |
| hd_epic | 2,248 → 2,191 | 0 | 57 | 0.1 |
| egotouch | 2,245 → 2,241 | 0 | 4 | ~0 |
| wiyh (recon tier) | 757 → 744 | 0 | 13 | ~0 |
| egoverse_aria (v1) | 156 → 156 (superseded by aria_v2) | 0 | 0 | 0 |

Not in scope (validity is GT-derived or Stage-1 never required 2 hands): GT/native
datasets (taco, hot3d, oakink, egodex, h2o, arctic, gigahands, humantouch, wiyh_native,
EgoVerse natives), robot datasets. `egoverse_eva` had no Layer-4 output yet (conveyor
in-flight; it inherits both gates). `egoexo4d` and `egoverse_mecka_freeform` ran with
both gates from their first shard — verified across every shard report — so no retro
purge was needed. `egoverse_aria_v2` had 18/19 shards filtered before the per-hand flag
landed in the conveyor; it got the retro purge above.

Verification: acceptance clips confirmed dropped
(`factory013_worker005_00028_iv09` → 10k shard 01681 `single_valid_hand`;
`factory_022_worker_097_0064_iv05` + `factory_065_worker_058_0089_iv10` → 100k shard
02265 `empty_pose`); post-purge audits re-read `pred_valid` on fresh samples — 240-clip
samples for the nine large datasets and full-coverage for wiyh (744/744) and
egoverse_aria_v2 (fresh 240 after its purge) — all report **0 empty-pose and 0
one-hand keeps**.

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

**EgoVerse-native full-skeleton GT (`.gt_joints.npy`, retrofit 2026-08-28):** the four
native tiers (egoverse_lightwheel / mecka_flagship / scale / microagi) shipped the same
wrist+5-tips-only defect — the extractors read the full in-source 21-joint GT but only
the 116-d lowdim survived. Every shipped tar was rewritten append-only into
`frames_v2/shard_XXXXX/` with one `{frame}.gt_joints.npy` member per frame:
**(2, 21, 3) float32** world-frame joints, MANO/OpenPose order (index 0 = left hand;
wrist 0, tips 4/8/12/16/20). Schema tags: `egoverse_world_21_mano_order_v1` (zarr
tiers), `lightwheel_world_21_mano_order_v1`; flag `extra["gt_joints"] = true`. An
all-zero hand-frame is the untracked sentinel (`presence` in `.meta.json` stays
authoritative); `.lowdim.npy` is unchanged and its wrist+tips are an exact subset of
`gt_joints`, validated per clip on every presence-on frame (max err 0.0 across all
771,524 clips). World-frame note: the fleet builds straddled the `recenter_world` spec
change, so lightwheel ships 24,080 raw-world + 1,382 recentered clips (zarr tiers:
recentered throughout) — the retrofit auto-detected the mode per clip against the
clip's own lowdim, and `gt_joints` always matches the clip's lowdim/w2c world frame.
Pre-retrofit manifests: `filter_run/_pre21_backup/` (combined + `_shards/`). The
converter now emits `gt_joints` natively (`generate_keypoints_wds.py`), so future
native builds carry the full skeletons from birth. Built by
`scripts/build/retrofit_egoverse_gt_joints.py`.

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
