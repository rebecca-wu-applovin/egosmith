# Filtering plans by data category (Final Keep-List)

## Execution model: 5 parallel category agents + the L4-node fleet

**5 agents, one per category (1, 2, 2.5, 3, 4), running in parallel.** Categories are
independent end-to-end (different sources, converters, filter configs, upload
prefixes), so each agent owns its category's full loop: probe/convert -> filter ->
upload -> report. Shared contracts that keep them from colliding:
- common output layout `gs://foundational-research/hoi-dataset/egosmith_filtered/<ds>/`
  (frames/ outputs/ filter_run/), common funnel-report format;
- the cross-cutting rules below (fps gate scaling, infill-before-L4, overlay smoke
  gate) are binding on every agent;
- compute partitioning (see "Compute reality" below for the binding constraints):
  the single-box H100 (this devbox) is for smokes, converters, and small/medium
  Layer-4 runs (Cat-3-sized). Fleet templates exist per job type under
  `scripts/fleet/egosmith_recon/` (`job_egocentric_stage1` = Layer-1,
  `job_egocentric_convert`, `job_egocentric_recon`, `job_egodex_filter` = fleet
  Layer-4, `job_phase_d_fleet`). **Recon-class jobs (whole-node, 8-GPU) run ONLY on
  the g2-standard-96 pool — currently 100% occupied (97 nodes) by Egocentric-100K
  recon. Single-GPU jobs (L1/L4/convert/fits) run on the NEW MKS g2-standard-8 L4
  nodes.** (Naming disambiguation: "Layer-4" = the quality filter; "L4 nodes" =
  NVIDIA-L4 GPU nodes.)

### Compute reality (per user, Aug 21 — verified against fleet templates)
- **g2-standard-96 pool (8x L4/node): FULLY OCCUPIED — 97 nodes running the
  Egocentric-100K reconstruction.** No new recon-class capacity until that run
  drains; new recon jobs QUEUE BEHIND it.
  Live status (GCS done-markers, Aug 20 19:31 UTC): 2754/3000 shards recon-complete
  (91.8%), ~30 shards/h tail rate -> pool drains in ~8-12h; phase-D filter tracking
  7 shards behind (2747 filtered). Queued recon (Egocentric-10K, calibrated sets,
  EgoVerse-I re-label) can likely start on the 96-pool ~Aug 21, inside the Aug-23
  window.
- **Recon pods are whole-node by design** (`job_egocentric_recon.template.yaml`:
  requests 8x GPU / 80 vCPU / 300Gi, limits 8/90/360Gi) ->
  **g2-standard-8 (1x L4, 8 vCPU, ~32Gi) is TOO SMALL for reconstruction. Never
  schedule recon there.**
- **g2-standard-8 nodes ARE right-sized for single-GPU jobs**: Layer-1 prefilter
  (template requests 1 GPU / 10 vCPU / 24Gi — trim cpu request to <=7 and the memory
  limit from 48Gi to ~28Gi to fit), Layer-4 filter, converters, SHOW3D-style MANO
  fits. This is what the new MKS capacity is FOR.

### NEW L4 capacity in MKS (per user, Aug 21; LOCATION per user Aug 21 late: project prj-al-p-axon-max-314e, region us-east1, cluster p-axonmax-use1-ayop, node pool model-serve-gpu-9e24 — cross-region vs the us-central1 fleet/bucket, throughput to be measured in the L1 smoke; USER-APPROVED up to the FULL 1000-node autoscale ceiling. ACCESS PATH (ops, Aug 21): bidderml-airflow@strategic-atom-700 submits jobs to HUB cluster `x-deployment-usc1-hubx`, namespace `robot`, which dispatches to any of the 5 prod clusters; pods run as the `robot` KSA (workload-identity-bound to the fr-airflow GSA, already granted for GCS). Job manifests: namespace robot + serviceAccountName robot + existing nodeSelector/tolerations for the destination L4 pool + hub queue/placement metadata (discover from existing jobs in the namespace).)
Additional L4 nodes are available in the MKS cluster on **g2-standard-8** instances
(1x NVIDIA L4, 8 vCPU per node — one pod per node). **Single-GPU batch jobs only
(Layer-1 / Layer-4 / convert / fits — NOT recon, see above)** schedule onto them by
adding exactly this to the pod spec:

```yaml
nodeSelector:
  node.kubernetes.io/instance-type: g2-standard-8
tolerations:
  - key: applvn.dev/workload
    operator: Equal
    value: model-serve
    effect: NoSchedule
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

Template changes required before fleet launches: in
`scripts/fleet/egosmith_recon/job*.template.yaml`, swap the old
`node_pool: l4` selector (+ `l4-pool` toleration) for the snippet above, and shrink
per-pod resource requests to fit 1x L4 / 8 vCPU (e.g. 1 GPU, <=7 vCPU, <=28Gi;
no 3-pods-per-node packing). NOTE: these nodes are shared with model-serve
workloads (that is what the applvn.dev/workload=model-serve taint marks) — batch
jobs should set conservative resource requests and be preemption-tolerant
(idempotent shards + resume, which all our job types already are).
- the annotation daemon stays a single central service (not per-agent).

**Deadline (per user, Aug 21): Aug 23 is the WHOLE-CATEGORY deliverable — all five
categories, not just Cat-3.** Consequence: all 5 category agents spin up NOW in
parallel, each scoped to what is physically completable by Aug 23:
- Cat-3 (this sweep): finish show3d -> full category uploaded + funnel report. DONE-track.
- Cat-2: EgoDex already shipped; Aug-23 deliverable = triangulated-GT sets present on
  GCS (AssemblyHands, DexCap) converted+filtered; Open-AoE/EgoVerse funnel started on
  MKS L4s (pseudo-label volume won't finish in 2 days — deliverable is launched fleet
  + first-shard funnels).
- Cat-2.5: pose-accuracy audits + smokes for WIYH/HumanTouch/EgoTouch + tactile
  validator spec'd; full filtering follows.
- Cat-1: Egocentric-100K RECON is already running (97x g2-standard-96 nodes — the
  whole 8-GPU pool). Aug-23 deliverable = (a) that run's progress/yield report,
  (b) **L1 prefilter jobs launched on the new g2-standard-8 MKS nodes** for
  Egocentric-10K + calibrated sets (L1 is single-GPU and fits; recon for those sets
  QUEUES behind the 100K run on the 96-pool — do NOT put recon on g2-standard-8).
- Cat-4: T-Rex first — embodiment-validity filter run + upload; others behind
  license/model verification.
(Where a category cannot physically complete by Aug 23, the deliverable is: agents
running at scale on the MKS L4 capacity + per-dataset funnel reports for everything
that has flowed through.)

Grounded in the repo's two existing filter layers plus the converters/fitters built this
sweep. The two reusable filter layers are:

- **Layer-1 (L1)** — heuristic prefilter, `scripts/build/stage1_prefilter.py`:
  YOLO hand presence/size/ROI (Gate A) + optical-flow RANSAC ego-motion stability
  (Gate B) merged by valid-span (Gate C). Cheap GPU; prunes locomotion / no-hands /
  tiny-hands / unstable camera. Operates on frame tars, no pose needed.
- **Layer-4 (L4)** — quality filter, `scripts/build/filter_manifest_by_quality.py
  --stages infiller`: per-frame step gates (hand/finger/wrist/camera), visibility rules
  (off-screen streaks, severe-offscreen fatals, in-frame ratio, hand-size ratio),
  camera-space IQR auto-bounds, chunk-window extremes. Consumes MANO
  `world_space_res.pth` + SLAM npz (GT mode) or native lowdim (EgoDex mode).

Cross-cutting rules (apply to every category):
0. **MIRROR-COMPLETENESS VERIFICATION FIRST (every dataset, before any conversion).**
   Per user (Aug 21): DexWild and AssemblyHands downloads are NOT finished. Every
   agent must verify its dataset's GCS mirror is complete before spending compute:
   (a) object count + total bytes vs the official source / the folder's PROVENANCE.md
       (DexYCB precedent: byte-exact table);
   (b) zero-byte / truncated file scan (SHOW3D precedent: 68 zero-byte
       hand_pose.json needed HF backfill);
   (c) split-archive completeness (HOI4D precedent: .tar.gz0..6 must all exist);
   (d) per-clip modality coverage — every required component present (TACO
       discover_sequences missing-modality pattern);
   (e) write the verdict to <ds>/filter_run/MIRROR_VERIFICATION.json (counts, bytes,
       missing items, verdict COMPLETE / INCOMPLETE) and log it in the update log.
   INCOMPLETE -> mark the dataset **PENDING-DOWNLOAD**, skip processing, report what
   is missing so the download can be finished; never ship a filtered set from a
   partial mirror.
   Current PENDING-DOWNLOAD list (user-confirmed): **DexWild (Cat-4)**,
   **AssemblyHands (Cat-2)**; plus known gaps: ContactPose (sample-only), EgoDex raw
   (absent), TriHands (unreleased).
1. **fps scaling of step gates** (phase_d precedent, gates tuned at 30fps):
   gate x (30 / source_fps). 30fps -> defaults; 15fps -> 2x (1.98/0.6/0.6/0.4/1.4);
   60fps -> 0.5x (0.495/0.15/0.15/0.10/0.35). `--source_fps = --target_fps = native`.
2. **Gap infill before L4** for any GT/keypoint source with per-frame validity gaps
   (`scripts/build/infill_world_res_gaps.py`): interpolate params across invalid
   frames (slerp rot), hold ends, never touch `valid`. Without it, valid->zeros
   transitions read as teleports and the step gates mass-drop (HO3D: 11/55 -> 45/55
   after fix).
3. **Overlay smoke gate** (5 clips, projected joints must lock onto hands in RGB)
   before any full conversion/filter run; convention probes vs the dataset's own
   joints/kps2D (sub-mm / sub-px) before that.
4. **Funnel reporting**: discovered -> converted -> (L1 kept) -> L4 kept, top drop
   reasons, kept hours, upload path under
   `gs://foundational-research/hoi-dataset/egosmith_filtered/<ds>/`.

---

## Category 1 — Video-only egocentric (no labels) -> EgoSmith recon  [9 sets]
Sets (updated keep-list): Egocentric-100K (primary feedstock, 256p caveat),
Egocentric-10K (1080p) + Egocentric-10K-Evaluation (held-out), Ego4D (process LAST,
~2% yield), **Ego-Exo4D (NEW IN BUCKET)**, EPIC-KITCHENS-100, HD-EPIC (calibrated
Aria, fixture twins = penetration QC), HoloAssist (calibrated), Assembly101 (base
corpus; AssemblyHands is its label layer).

Ego-Exo4D being in the bucket unblocks four things (record as sub-plans):
(a) TriHands pixels — once TriHands labels release, they join Cat-2 with video here;
(b) calibrated-mono EgoSmith path using MPS trajectories (skip SLAM, GT camera);
(c) triangulation extension (multi-view exo rigs);
(d) EgoSmith-vs-TriHands calibration experiment (accuracy audit of our recon).

Pipeline: **L1 FIRST, before any GPU recon** (yield economics — don't reconstruct
clips with no usable manipulation spans), then EgoSmith reconstruction
(detect_track + anycalib est_focal -> DPVO SLAM -> HaWoR -> infiller) at the recon
fps (15), then **L4 in recon mode** with 15fps 2x gates (exact phase_d shard flow:
`--source_fps 15 --target_fps 30`, gates 1.98/0.6/0.6/0.4/1.4) plus recon-only
validity (degenerate-SLAM / NaN checks already in the filter).

Per-set notes:
- Egocentric-100K: 256p wrist-level -> hand-size gate is the dominant L1 dropper;
  budget for very low keep-rate, shard via phase_d incremental daemon.
- Calibrated sets (Assembly101, HoloAssist, HD-EPIC, Ego-Exo4D): skip anycalib —
  write known intrinsics to `est_focal.txt`; undistort to pinhole where fisheye
  (HOT3D cv2/fisheye pattern). HD-EPIC fixture twins reserved for penetration QC.
- Ego4D: run last; expect ~2% yield; L1 sharding mandatory.

## Category 2 — Video + hand keypoints (no objects)  [4 sets + 2 pipeline products]
Sets (updated keep-list): Open-AoE (OpenAoE-2000h/, largest T4), EgoVerse (SPLIT A/I
by provenance; I = re-label candidate), AssemblyHands (**PENDING-DOWNLOAD — mirror
incomplete per user Aug 21; verify + finish download before converting**;
eval/calibration only, monochrome ego), DexCap (RECLASSIFIED from Cat-4: human-side
EMF glove mocap, occlusion-free in-contact fingers). Pipeline products: egosmith_filtered/,
egosmith_recon/ (our own outputs — not sources).
STILL MISSING from bucket: EgoDex raw (829h — ND-license decision to record; our
egosmith_filtered/egodex is the processed derivative, raws absent), TriHands
(release pending; actionable when labels drop since Ego-Exo4D pixels now exist).

Split the plan by LABEL MECHANISM, not by dataset:
- **Device/native GT (EgoDex)**: no MANO conversion; L1 -> L4 native-lowdim mode at
  30fps. (Shipped: 338,234 -> 177,979 L1 -> 158,564 L4.)
- **Triangulated keypoints (TriHands, AssemblyHands, DexCap EMF)**: skeleton -> MANO
  torch-fit first (FPHA fitter; use the SHOW3D centroid-translation variant
  `_fit_right_mano_centroid` when the rig's wrist differs from MANO's, mirror-fit for
  left) -> infill -> L4 GT mode at native fps. AssemblyHands stays eval/calibration
  only — filter for sanity, do not ship to train.
- **Monocular pseudo-labels (Open-AoE, EgoVerse-I)**: treat as recon-grade, NOT GT:
  L1 -> L4 with recon-mode strictness; EgoVerse-I is the re-label candidate — route
  through EgoSmith re-labeling (manufactured Tier-4) rather than trusting source
  labels; then the Category-1 plan applies to the re-labeled output.

## Category 2.5 — Hands + tactile/contact  [4 sets]
Sets (updated keep-list): WIYH (largest; audit pose accuracy + glove visual domain),
HumanTouch (Xspark-HumanTouch/; full-palm 360-pt; gloved-RGB caveat), EgoTouch
(pressure maps + pose; clarify license), **OpenTouch (PROMOTED to KEEP)** —
wild-scene full-hand contact (169 palm-side points), complements HumanTouch's
scripted tasks; same audit-first treatment.

AUDIT RESULTS (Cat-2.5 agent, Aug 21 — sheets/JSONs in /root/cat25_audits/):
- **WIYH: pose TRUSTWORTHY at wrist level** (identity chest->cam probe: median
  wrist-vs-hand-mask 0.0 px). BUT: **NO tactile channel exists in the data** ->
  taxonomy correction: WIYH is effectively Cat-2 (video+keypoints, glove domain);
  finger-level GT unusable as-is (75-dim glove angles, topology unpublished).
  36.7 TB bulk = queued by design.
- **HumanTouch: TRUSTWORTHY pending calibration handshake** — internal GT perfect
  (exact 60 Hz, 9.1 cm rigid wrist-tracker offset), shipped rpy extrinsic wrong under
  all 128 conventions but a per-session fitted rigid extrinsic locks skeletons on
  (rms 71 px @1080p) -> auto-calib at conversion or ask authors. License unresolved.
- **EgoTouch: 2D pseudo-label only; 3D is recon-grade.** Smoke verdict decisive:
  raw HaMeR weak-perspective depth is metric-unstable (wrist z 6-975 m) -> L4 kept
  0/5. Route bulk through the recon/re-label track, never raw HaMeR fits. Plus:
  **~7% zero-byte mirror gap (1811/25049 objects) -> add to mirror-verification
  pending list**; license UNKNOWN (no license file).
- **OpenTouch: TRUSTWORTHY-conditional** — projection chain verified (in-frame 1.00),
  but landmarks HELD STALE on tracking loss -> derive per-frame validity
  (zero-velocity runs + hand_out_of_frame labels) before filtering. Pressure
  polarity inverted (rest = rail-high 3072) — normalize in validator.
- Tactile validator spec shipped: scripts/build/tactile_episode_qc_SPEC.md
  (7 gates on the robot_episode_qc contract; no-visibility-gating rule explicit).
- New audit/converter files: scripts/inspection/cat25_*_audit.py,
  scripts/build/generate_egotouch_world_res.py.

Pose channel: same decision tree as Category 2 by label mechanism, with a mandatory
**pose-accuracy audit first** (WIYH explicitly): recon_vs_gt_accuracy-style probe +
overlay sheets on a sample before any bulk conversion.
Tactile channel (new, no existing filter): lightweight validator, NOT L4 —
(a) sensor sanity: dropout/saturation/frozen-channel detection, (b) time alignment:
tactile timestamps vs frame timestamps within tolerance, (c) contact plausibility:
nonzero pressure only when fingertips near/inside object region (where hand pose
exists). Critically: do NOT gate on hand visibility — occluded in-contact frames are
the value of this category (DexCap precedent). Keep/drop at clip level = pose filter
AND tactile sanity.

## Category 3 — Video + hands + object pose/mesh  [13 folders / 12 unique — THIS SWEEP mostly executed]
Sets (updated keep-list): GigaHands (GigaHands/ + gigahand/ DUPLICATES — consolidate
first), SHOW3D (primary held-out HOI eval), HOI4D (keep w/ caution, noisy),
OakInk-v2, HOT3D (hot3d/; fisheye/mono caveat), DexYCB (estimator eval baseline),
TACO (texture-bake source), ARCTIC (gold standard; marker artifacts), ContactPose
(only measured-contact GT — thermal), HO-3D v3 (occlusion eval), OakInk-v1 (Shape
grasps are Tink-synthesized — flag in metadata), **H2O (PROMOTED to KEEP)** —
bimanual + object 6D + ego view, small supplementary -> needs a NEW converter
(generate_h2o_world_res.py; H2O ships per-frame MANO + camera, template-clone job).
EXCLUDED from mix but on disk: GazeHOI (gaze unused, unaudited).

Plan (executed): GT converter (native MANO where shipped; skeleton->MANO fit where
21-joint, e.g. SHOW3D UmeTrack) -> **infill** -> **L4 GT mode** with fps-scaled gates
-> upload standard layout. L1 is intentionally NOT in front (controlled captures,
hands guaranteed; L4's visibility rules already handle off-screen — that is what
structurally rejects full OakInk2 sequences, 0/627).

Status / remaining:
- DONE+uploaded: ho3d_v3 (45/55, 0.68h), dexycb (6475/8000, 4.36h),
  hoi4d (1291/1682, 7.17h), oakink_v2_full (provenance record; usable set =
  oakink_actions 3,999 clips), taco/hot3d/oakink_actions (prior runs).
- IN FLIGHT: show3d (3378 clips, 60fps 0.5x gates; ETA ~11h; 68 zero-byte mirror
  hand_pose.json backfilled from HF).
- TODO (this category, post-sweep): GigaHands (dedup two folders first), ARCTIC
  (converter exists, gold standard), OakInk v1, ContactPose (BLOCKED: GCS has sample
  only — needs full-image mirror).
- Eval reservations to respect at split time (not filter time): SHOW3D = primary
  held-out HOI eval; DexYCB = estimator eval; HO3D = occlusion eval; HD-EPIC twins.

## Category 4 — Robot dexterous hand (>12 actuated DoF)  [7 sets]
Sets (updated keep-list): T-Rex (#1 — SharpaWave 22-DoF exact match), Dexora (#2 —
12-DoF; verify hand model; reconcile 40.5h on HF vs paper's 177.5h), DexWild (**PENDING-DOWNLOAD — mirror
incomplete per user Aug 21; verify + finish download before QC**; LEAP V2A 17-DoF;
human-robot cotraining structure), HRDexDB (Allegro slice ONLY; Inspire
slices excluded), RealDex (ShadowHand ~20-DoF; verify license),
**LET (LET-Base-Dataset/, NEW)** — dexterous-hand version with tactile + force;
log hand model/DoF + hours in Coda, **NO robot_episode_qc adapter yet — new adapter
required**, **UniDex (NEW)** — retargeted multi-embodiment (8 hands, FAAS action
space); cross-hand generalization signal; FLAG in metadata: never physically
executed; **NO robot_episode_qc adapter yet — new adapter required** (per-embodiment
specs; velocity/limit gates apply per target hand).

**Existing code (verified): `scripts/build/robot_episode_qc.py`** — 12 config-driven
gates (dof_spec, nan_channels, const_channels, pos_limits, velocity & accel scaled by
MEASURED dt, action_tracking, timestamps, video_sync, min_length, stall, dedup) +
per-robot specs `configs/robot_specs/<robot>.yaml` + `--calibrate` limit-suggestion
mode. Adapters already cover the whole keep-list: trex (LeRobot v3, SharpaWave
22-DoF), dexora (LeRobot v2.1, 12-DoF), dexwild (HDF5-in-tar, LEAP V2 17-DoF, remote
ranged reads), hrdexdb_allegro (16-DoF npy dirs), realdex (rosbag TF; license concern
auto-recorded in report). Output = standard manifest contract
(`manifest.jsonl` of kept ClipManifestRecords, storage_kind=native_episode +
`report.json` funnel).
Shipped already: trex / hrdexdb_allegro / realdex filter_runs on GCS.

Cat-4 agent recon (Aug 21, verified):
- **Dexora: ALREADY QC'd + shipped** (prior session): 23,715 parquets -> 10,909 kept
  / 39.75 kept-h (dedup dropped 10,156 aggregate-repo re-uploads by design);
  hours reconciliation MEASURED: task metadata 12,199 eps / 45.09h; HF release
  40.5h; paper's 177.5h was never released. Mirror == full HF release. Remaining:
  top-up filter_report.json copy + notes.
- **DexWild: incomplete mirror CONFIRMED** (matches user's pending-download):
  prior partial run covered clothes_data/robot only (295 -> 197 kept);
  florist_data/robot (67.8 GiB) landed on GCS Aug 20-21; pour_data has NO robot
  split. Plan: verify florist completeness (rule 0) -> smoke + possible
  dexwild_v2.yaml recalibration -> full clothes+florist runs -> cross-split dedup
  merge -> upload.
- **UniDex mirror = annotations + checkpoints ONLY, no trajectories** -> nothing to
  QC; PENDING-DOWNLOAD (trajectories); adapter stays a draft skeleton.
- **LET** = Kuavo 4 Pro humanoid, rosbag format, cc-by-nc-sa-4.0, dex-hand manifest
  TSVs present; hand model/DoF/hours extraction scoped (log to Coda).
- Both spec YAMLs already exist (dexora.yaml fully calibrated; dexwild.yaml
  calibrated on only 12 eps — refresh-before-full-run warning honored via v2 spec
  decision gate).
License/model verification (RealDex terms) recorded in reports before shipping.

---

## EXECUTION TIMELINE (today = **Aug 21** per user — box clock is UTC-behind;
hard finish Aug 23 EOD; ~2 days -> less buffer, same critical path)

**Aug 21, T+0 (on approval)** — kickoffs, all parallel:
- Copy this plan to `docs/category_filtering_plan.md` (user monitoring).
- Spin up the 4 remaining category agents (Cat-1, 2, 2.5, 4) with their briefs.
- Cat-3: labelers start on ho3d_v3 + dexycb + hoi4d (run_labeling_gt_datasets.py,
  ~1-2h, <$50); show3d conversion continues (ETA overnight); optionally chase-start
  show3d's L4 on already-converted clips.
- Cat-4 agent: `robot_episode_qc.py --limit 50` smokes for dexora + dexwild
  (+ --calibrate pass); verify specs.
- Cat-1 agent: patch fleet templates with g2-standard-8 selector/tolerations; launch
  L1 (stage-1) jobs for Egocentric-10K + calibrated sets on the g2-standard-8 pool.
- Cat-2 agent: AssemblyHands/DexCap format probes + skeleton->MANO converter smokes.
- Cat-2.5 agent: WIYH/HumanTouch/EgoTouch pose-accuracy audit probes + overlay smokes.

**Aug 21 (same day, after kickoffs)** — main compute begins:
- 100K recon fleet drains (~8-12h from 91.8%); D+E chase closes behind it ->
  Egocentric-100K funnel report = Cat-1 centerpiece deliverable.
- Cat-3: show3d convert done (overnight into Aug 22 latest) -> infill -> L4
  (60fps 0.5x) -> upload -> labeling -> **Cat-3 COMPLETE + final funnel report.**
- Cat-1: the moment the 96-pool frees, queue Phase-B convert + recon for
  Egocentric-10K + calibrated sets (D+E chase attaches automatically). Measured
  560 raw-h/wall-h (90 nodes): Egocentric-10K ~10,000 raw h ~= 18 fleet-hours ->
  MUST start by Aug 22 ~06:00 UTC to land inside Aug 23. This is the critical path.
- Cat-4: full QC runs dexora + dexwild same day (small); upload filter_runs; trex
  refresh if the 50h open split adds episodes.
- Cat-2: triangulated sets (AssemblyHands eval-manifest, DexCap) convert + filter +
  upload; Open-AoE / EgoVerse L1 passes launch on g2-standard-8.

**Aug 22** — single drain day (no slack for new starts after ~noon):
- Cat-1: Egocentric-10K + calibrated recon runs through the day; D+E chase closes
  behind it; funnels written as shards land.
- Cat-2: EgoVerse-A L4 pass on source labels (Tier-3); Open-AoE L1->L4 on
  pseudo-labels; EgoVerse-I re-label DEFERRED past Aug 23 (recon-class, queued) —
  recorded as such in the deliverable.
- Cat-2.5: EgoTouch (20h) + HumanTouch sample converted/filtered if audits pass;
  WIYH scoped by audit outcome (1,000h won't fully process — deliverable = audit +
  smoke + launched jobs).
- Retries/failures absorbed same-day (everything resume-safe).
- Stretch items (do in this order IF the critical path is healthy by Aug 22 noon,
  else document as queued): (1) GigaHands consolidate + converter smoke,
  (2) H2O converter (template clone) + smoke, (3) LET + UniDex robot_episode_qc
  adapters + smokes, (4) Ego-Exo4D L1 launch + calibrated-mono path probe,
  (5) OpenTouch audit probe, (6) ARCTIC refresh + OakInk-v1 converter.

**Aug 23** — deliverable freeze:
- All uploads verified (object counts vs manifests); annotations running/complete for
  every kept set; whole-category report: per-dataset funnels (discovered -> converted
  -> L1 -> L4 kept), drop reasons, kept hours, upload paths, cost actuals.
- Documented leftovers (by design, not slippage): Ego4D (process-last policy),
  Ego-Exo4D recon (new in bucket; L1 + calibrated-path probe by Aug 23, recon queued),
  EgoDex-829h ND license decision, TriHands release pending, ContactPose mirror gap,
  EgoVerse-I re-label queued, WIYH/OpenTouch bulk (audit + smoke only), RealDex
  license verdict, H2O + OakInk-v1 + GigaHands converters (Aug-22 stretch, else
  documented as queued with converter specs), LET + UniDex robot_episode_qc adapters
  (new code — Aug-22 stretch, else queued), Dexora 40.5h-vs-177.5h reconciliation.

## Housekeeping (from updated inventory, 40 folders = 37 datasets + 2 pipeline + 1 dup)
- GigaHands/ vs gigahand/: duplicate — consolidate before converting (dedup gate).
- Excluded-but-on-disk: GazeHOI, HRDexDB Inspire slices (do not ingest).
- Missing vs keep-list: EgoDex raw, TriHands (external), RoboCOIN (pending
  DataManager DoF check — correctly not downloaded).
- LET: log hand model / DoF / hours in Coda when convenient (user note).

Feasibility check (2-day window): critical path = 96-pool drain (8-12h from now) ->
Egocentric-10K recon (~18 fleet-hours) MUST be queued the moment the pool frees and
start by Aug 22 ~06:00 UTC; buffer is thin but real (~12h). Mitigation if the drain
slips: pre-stage Phase-B (CPU pods, pool-independent) TONIGHT so recon starts the
instant nodes free; if still tight, split 10K recon 50/50 and declare the remainder
"launched + funnel from processed shards" (allowed by the deliverable definition).
All single-GPU work rides the new MKS g2-standard-8 nodes and the dev-box H100 in
parallel and is not on the critical path. Cat-3/4 are small and front-loaded.

---

## Chase architecture — L4 filtering + annotation CONCURRENT with recon
(retrieved from `docs/egocentric100k_pipeline.md`; live on the dev box right now)

Recon (C), Layer-4 filter (D), and LLM annotation (E) run **at the same time**,
chasing each other through GCS markers — no stage waits for the previous stage to
fully finish:

```
C: recon fleet (8xL4 whole nodes)  -> writes  egosmith_recon/<ds>/recon/outputs/_done/shard_X.json
D: phase_d_incremental.py daemons  -> polls _done markers, CLAIMS shards via
   filter_run/_shards/_claims/, materializes clips, runs canonical
   filter_manifest_by_quality.py per shard -> writes shard_X.{filtered.jsonl,report.json}
E: scripts/annotation/run_labeling_egocentric.py daemons -> polls D's filtered
   shard manifests -> gpt-5-mini v4 -> annotations_v4/_shards/shard_X.annotations.jsonl
```

Mechanics that make it safe: per-shard GCS claim files (multiple D workers — box
daemons AND optional fleet pods `job_phase_d_fleet.template.yaml`/`pod_entry_phase_d.sh`
— share the queue without duplication); everything per-shard/per-clip resume-safe;
kill/restart loses only in-flight clips. D is ~free (CPU); E costs ~$0.002/kept clip
(gpt-5-mini, ~174K clips/h at 10 procs on Tier-5).

Dev-box runbook (verbatim from the doc):
```bash
pgrep -fc 'incrementa[l].py --workers'   # Phase D daemons (expect ~8)
pgrep -fc 'labeling_eg[o].py --workers'  # E labelers (expect ~10)
bash /root/egosmith_annotations/rescale_all.sh   # relaunch all (idempotent)
# kill by PID or 'incrementa[l]' bracket-pattern — plain pkill matches your own shell
```
Live proof it works: egocentric-100k D is 7 shards behind C (2747 vs 2754 of 3000).

How each category uses the chase pattern:
- **Cat-1 (new sets: Egocentric-10K, calibrated)**: full A->B->C||D||E conveyor,
  identical to the 100K run. D daemons: dev box (spare CPU) + optionally MKS
  g2-standard-8 pods (phase_d fleet template + new selector/tolerations).
  E: needs OPENAI_API_KEY; scale procs to keep pace.
- **Cat-2/2.5**: pseudo-label & re-label tracks are recon-class -> same conveyor.
  Triangulated-GT sets skip C; D+E run back-to-back per dataset (small).
- **Cat-3 (this sweep)**: L4 already done on-box for ho3d_v3/dexycb/hoi4d; show3d's
  D can be chase-started NOW on its converted clips instead of waiting for the full
  conversion (the runner currently filters after convert completes — optional
  optimization). Annotation: use the GT-dataset path
  `scripts/annotation/run_labeling_gt_datasets.py` -> `<ds>/filter_run/annotations.v4.jsonl`
  (precedent: 163,236 clips, $975, 5.1h for taco/oakink_actions/hot3d/egodex).
  New kept sets to label: ho3d_v3 (45), dexycb (6475), hoi4d (1291), show3d (TBD)
  — est. <$50 total at measured rates; can start for the three finished sets
  IMMEDIATELY, in parallel with show3d's conversion.
- **Cat-4**: no recon; native-label mapping first, then E only for gap backfill.

Cost model for capacity planning (measured on the 100K run): C = 2.96 GPU-h per kept
hour (~93% of compute); 1 raw hour ≈ 1.18 L4-GPU-h end-to-end; 1 final dataset-hour ≈
4.7 GPU-h + $0.26 labeling; 90-node fleet ≈ 560 raw video-hours per wall-clock hour.

## Annotation layer (applies AFTER filtering, before VLA build — all categories)

Every shipped dataset also carries `filter_run/annotations.v4.jsonl`: per-clip,
per-segment natural-language instructions at 4 granularity levels (level1 short verb
phrase ... level4 finger-level grasp description) + `is_good_quality` per segment.

**EXACT ANNOTATION CONFIG (the frozen "final" config — verified in the code of BOTH
labeling scripts and in the live output stream):**
- **Prompt: `annotation_general_clip_v4`**, loaded from
  `/root/egosmith/src/lib/annotation/prompts/with_clip/annotation_general_clip_v4.txt`
  (hardcoded in `scripts/annotation/run_labeling_egocentric.py` line 27 AND
  `scripts/annotation/run_labeling_gt_datasets.py` line 27; PROMPT_VERSION =
  "annotation_general_clip_v4" is stamped into every output record).
- **Model: gpt-5-mini, reasoning_effort=medium** ("Config frozen by the experiments"
  per the script docstring).
- **Sampling: P2 — 1024px, detail=high, ~3 fps, FMIN=8 frames minimum per clip.**
- Live confirmation: shard_02733 records written minutes ago carry exactly
  `{"model":"gpt-5-mini","prompt":"annotation_general_clip_v4","effort":"medium",
  "config":{"px":1024,"detail":"high","fps":3.0,...}}`.
- The new Cat-3 sets (ho3d_v3/dexycb/hoi4d/show3d) will be labeled with the SAME
  frozen config via `run_labeling_gt_datasets.py` — no prompt drift.
- NOTE: if the "final prompt" decided elsewhere is anything NEWER than
  `annotation_general_clip_v4`, it is NOT in this repo — both scripts hardcode v4;
  flag it and the scripts get updated before the Cat-3 labeling kickoff.

Ordering per clip: convert -> [L1] -> [recon] -> infill -> L4 -> **annotate (central)**
-> VLA build (`build_vla_from_manifest`, which can enforce `--min_instruction_num`).
Annotating only L4-kept clips is what keeps API cost sane at Cat-1/2 volumes.

Sweep-specific consequence (Cat-3): the four new kept sets — ho3d_v3 (45), dexycb
(6475), hoi4d (1291), show3d (TBD) — still need enqueueing into the central annotation
flow to receive their `annotations.v4.jsonl`; taco / oakink_actions / hot3d / egodex
already have theirs. This sweep's deliverable intentionally stops at L4 + upload per
spec ("labeling handled centrally afterward").

Category nuances:
- Cat 1/2/2.5: annotate post-L4 only; volume -> cost is the governing constraint
  (level-of-detail / fps / frames_sent knobs in the v4 config).
- Cat 3: as above — enqueue the new kept sets centrally.
- Cat 4: robot demos usually ship native task labels; LLM annotation optional —
  map native labels into the instruction schema first, only backfill gaps via API.

## Immediate next actions (in priority order)
1. Let show3d runner finish (convert -> infill -> L4 60fps -> upload) — watcher on it.
2. Final Cat-3 sweep report (funnels, drop reasons, kept hours, paths).
2b. Annotation of new Cat-3 kept sets via
    `scripts/annotation/run_labeling_gt_datasets.py` (needs OPENAI_API_KEY): start
    ho3d_v3 + dexycb + hoi4d NOW (parallel with show3d conversion, chase-style);
    show3d follows its filter. Est. <$50 total at measured $0.002-0.006/clip.
3. Decision needed: chain L1 over the egocentric Cat-3 kept sets (show3d, hoi4d) for
   parity with taco's L1->L4 lineage? (Composes cleanly; prune-only.)
4. Post-sweep Cat-3 stragglers: GigaHands dedup+convert, ARCTIC refresh, OakInk v1;
   ContactPose blocked on mirror.
5. Spin up ALL FOUR remaining category agents immediately (Aug-23 whole-category
   deliverable): each agent gets its category section above as its brief, the
   cross-cutting rules as binding constraints, the MKS g2-standard-8 L4 snippet for
   any fleet job it launches, and a per-category Aug-23 deliverable definition from
   the Deadline section. Cat-3 agent = this session (finishing show3d + report).
