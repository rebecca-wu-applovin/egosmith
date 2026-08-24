# tactile_episode_qc — SPEC (Category 2.5 tactile validator)

Status: SPEC (Aug 2026 Cat-2.5 sweep deliverable). Implementation target:
`scripts/build/tactile_episode_qc.py`, modeled 1:1 on the gate/report harness of
`scripts/build/robot_episode_qc.py` (ordered config-driven gates, first-failing-gate
funnel attribution, `manifest.jsonl` of kept ClipManifestRecords with
`storage_kind="native_episode"`, `report.json` with per-gate drop counts + resolved
criteria + per-episode drop details, `--calibrate` limit-suggestion mode).

This validator is deliberately **NOT Layer-4**. L4 (or the Cat-2 pose filter chosen per
label mechanism) owns the pose channel. This validator owns the tactile channel only.
Clip-level keep decision for Cat-2.5 datasets = **pose filter AND tactile sanity**
(logical AND of the two independent verdicts, computed separately).

## Binding design rule: NO hand-visibility gating

The value of Category-2.5 is exactly the frames where the hand is occluded (inside a
bag, under a cloth, wrapped around an object) while tactile says "in contact". Neither
this validator nor the pose filter configuration used for these datasets may drop a
frame/episode *because the hand is not visible in RGB* (DexCap precedent). Concretely:

- do NOT enable L4 visibility fatals (off-screen streaks / severe-offscreen /
  in-frame-ratio) as *drop* reasons for tactile datasets when the tactile stream shows
  valid contact during those spans; where the pose filter is used, run it with
  visibility rules demoted to metrics-only for these sets;
- this validator has no gate that consumes RGB hand detection at all.

## Audited input formats (measured 2026-08, see /root/cat25_audits/*)

| dataset | tactile signal | rate | rest/polarity | structural quirks |
|---|---|---|---|---|
| EgoTouch | `jq_pressure.json`: 256 ch/hand raw ints (0..~135) + `quat_left/right`; `pressure_grids.npz`: (T,21,21)/hand normalized 0..1 (`tactile_max` attr) | 30 Hz, frame-locked to chest.mp4 (counts match exactly; epoch ts) | rest = 0, positive on contact | grids have a FIXED hand-shaped footprint: exactly 224/441 cells NaN in every frame — structural, not dropout. ~100–185/256 raw channels are never-active in short clips. ~7% of GCS objects are zero-byte (mirror gap) -> episode-level exclusion up front. Early episodes lack `hamer_hands.json`/`masks.npz` (schema drift). `manual_contact_annotation.json` was False on 3/3 clearly-in-contact episodes — treat as unreliable/unpopulated, do not gate on it. |
| HumanTouch (Xspark) | parquet `observation.tactile.raw_{left,right}` 460 ch (ints 0..~108) + `patch_pressure` 12 patches CALIBRATED newton (0..~29 N) + `tactile.valid` (2) + `patch_pressure_valid` | 60 Hz exact (`session_time_ns`, max jitter 0 observed) | rest = 0, positive on contact | `meta/humantouch/tactile_mapping.json` maps sensor_id -> region; LEFT hand has "unassigned" channels (README: 360 valid pts of 460) -> frozen-channel gate must count ASSIGNED channels only. 117–170/460 std==0 observed in one episode = mostly the unassigned set. |
| OpenTouch | per-clip hdf5 `right_pressure` (T,16,16) raw ADC | per-frame, frame-indexed with rgb (T matches) | rest ≈ 3072 (rail-HIGH), value DROPS on contact -> **inverted polarity**; normalize `p = rest_baseline - raw` first | right hand only. 76–98/256 ch std==0 in short clips (rest-railed). Clip attrs carry `hand_out_of_frame`/`low_light` labels. |
| WIYH | **NONE** — no tactile channel exists (fisheye rigs + 75-dim glove joint angles + 6DoF eef only) | — | — | WIYH is pose/glove only: it bypasses this validator entirely and is governed by the pose filter alone. Do not fabricate a tactile verdict for it. |

Normalization step (before any gate): per-dataset adapter emits a canonical
`TactileStream`: `(T, C) float32`, rest=0, positive=contact-pressure, plus
`channel_mask (C,) bool` (structural/assigned channels only), `t (T,)` seconds when
shipped, `units: raw|normalized|newton`, `hand: left|right`. Grid datasets flatten with
their structural-NaN footprint folded into `channel_mask`.

## Gate order (first failing gate attributes the drop)

```
GATE_ORDER = [
    "stream_present",        # A0
    "sensor_dropout",        # A1
    "sensor_saturation",     # A2
    "frozen_channels",       # A3
    "time_alignment",        # B
    "contact_plausibility",  # C  (only when pose available; else metrics-only)
    "min_length",
]
```

### A. Sensor sanity

- **A0 stream_present**: tactile file exists, non-zero-byte, parses, `T > 0`, channel
  count matches spec `channels:` (EgoTouch 256/hand, HumanTouch 460/hand,
  OpenTouch 256). Zero-byte GCS mirrors (EgoTouch ~7%) die here with reason
  `tactile_missing` — recorded, not silently skipped.
- **A1 sensor_dropout**: after normalization, a run of >= `max_dead_window_s`
  (default 2.0 s) where ALL masked channels are exactly at rest **while the hand is
  moving** (wrist speed from the pose stream > `move_eps`, when pose exists; without
  pose, use the stricter all-rest-for-entire-episode test only). Whole-episode all-rest
  -> `tactile_flatline`. Also: NaN fraction *inside* `channel_mask` > `max_nan_frac`
  (default 0.01) -> `tactile_nan` (NaN outside the mask is structural and free).
- **A2 sensor_saturation**: fraction of samples at the rail (per-dataset `rail_value`:
  EgoTouch/HumanTouch = observed max-code, OpenTouch = 0 after inversion i.e. raw 0)
  above `max_saturation_frac` (default 0.05 of masked samples) or any single channel
  railed for > `max_rail_window_s` (default 5 s) -> `tactile_saturated`.
- **A3 frozen_channels**: std == 0 over the episode among MASKED channels, with the
  channel-mask sourced from `tactile_mapping.json` (HumanTouch) / grid footprint
  (EgoTouch) / none (OpenTouch full 256). Threshold `max_frozen_frac` default 0.60 of
  masked channels for episodes >= 10 s (short clips legitimately touch few taxels —
  scale expectation by episode length in `--calibrate`). Reason `tactile_frozen`.

### B. Time alignment (tactile vs frames)

- Monotonic timestamps, `median_dt` within `tol_rel` (default 10%) of the nominal rate
  (EgoTouch 33.3 ms, HumanTouch 16.67 ms, OpenTouch = video dt), max gap <=
  `max_gap_frames` (default 3) nominal intervals -> else `tactile_ts_jitter`.
- Count agreement with the video/frame clock: `|T_tactile - T_frames| <=
  count_tol` (default 2 frames; measured 0 on EgoTouch and HumanTouch;
  OpenTouch arrays share T by construction) -> else `tactile_frame_count_mismatch`.
- Cross-stream offset: where both tactile and frames carry absolute timestamps
  (EgoTouch epoch ts), `median(|t_tactile - t_frame|)` per aligned index <=
  `max_offset_ms` (default 20 ms at 30 fps, i.e. sub-frame; scale by 30/fps like the
  L4 step gates) -> else `tactile_misaligned`.

### C. Contact plausibility (requires pose; else metrics-only, never a drop)

Intuition gates only — tolerant by design, meant to catch glove-death and ghost
pressure, not to arbitrate physics:

- **ghost pressure**: fraction of frames with `sum(pressure) > contact_thresh` while
  every fingertip is > `far_dist` (default 25 cm) from the nearest scene-contact proxy
  (object mask back-projection where masks exist [EgoTouch masks.npz], else wrist-speed
  + tabletop-height heuristic; per-dataset adapter decides the proxy) must be <
  `max_ghost_frac` (default 0.20) -> else `tactile_ghost_pressure`.
- **dead-on-grasp**: a contiguous >= `grasp_window_s` (default 3 s) span where the pose
  says sustained grasp posture near the contact proxy AND pressure stays at rest ->
  count spans; > `max_dead_grasp_spans` (default 2) -> `tactile_dead_on_grasp`.
- **pressure>0 only near contact** is evaluated as a RATE, not per-frame fatal, and
  it consumes 3-D pose only. RGB visibility is explicitly NOT an input (see binding
  rule) — an occluded hand with pressure and a plausible 3-D pose is a KEEP.
- OpenTouch caveat: landmarks go HELD-STALE on tracking loss (measured); the adapter
  must derive per-frame pose validity (landmark-velocity==0 runs + shipped
  `hand_out_of_frame` label) and exclude invalid-pose frames from C entirely rather
  than letting stale poses fabricate ghost-pressure violations.

### min_length

Episode duration floor after all trims, default 2.0 s (`min_length_s`), consistent
with robot_episode_qc.

## Config: `configs/tactile_specs/<dataset>.yaml`

```yaml
dataset: egotouch
hands: [left, right]
channels: 256
rate_hz: 30.0
rail_value: 135          # from --calibrate; per-hand override allowed
polarity: positive        # opentouch: inverted (rest_high: 3072)
channel_mask: grid_footprint   # or tactile_mapping.json path, or "all"
gates:                    # same enable/threshold override mechanics as robot specs
  max_dead_window_s: 2.0
  max_saturation_frac: 0.05
  max_frozen_frac: 0.60
  tol_rel: 0.10
  max_offset_ms: 20
  contact_thresh: auto    # --calibrate suggests from episode ensemble
  far_dist_m: 0.25
  max_ghost_frac: 0.20
  min_length_s: 2.0
enabled_gates: [...]      # subset of GATE_ORDER, like robot_episode_qc
license_note: "EgoTouch license unclarified as of Aug 2026 — record in report"
```

`--calibrate` mode (as in robot_episode_qc): run the ensemble, print suggested
`rail_value`, `contact_thresh`, per-length frozen-frac expectations, observed dt stats.

## Adapters

- `iter_egotouch(root)`: episode dirs `<Scene>/<task>/<ts>/`; reads jq_pressure.json
  (+ grids), chest.mp4 frame count, hamer/wilor/rokoko for pose-side inputs of gate C;
  skips-with-record zero-byte episodes.
- `iter_humantouch(root)`: LeRobot v2.1 parquet per episode; channel mask from
  `meta/humantouch/tactile_mapping.json` (left/right region != "unassigned").
- `iter_opentouch(root)`: per-clip groups inside session hdf5; invert polarity with
  per-clip rest baseline = mode of first second; right hand only.
- WIYH: no adapter (no tactile). Guard: `--dataset wiyh` exits with an explicit error.

## Output contract

Identical to robot_episode_qc: `<output_dir>/manifest.jsonl` (kept episodes,
`storage_kind="native_episode"`, tactile stream paths in `descriptor.extra`),
`<output_dir>/report.json` funnel `{gate: {checked, dropped}}` + `reason_counts` +
`per_episode` drop details + `license_note` passthrough. The Cat-2.5 shipping step
intersects this manifest with the pose-filter manifest (clip-level AND) before upload
to `gs://foundational-research/hoi-dataset/egosmith_filtered/<ds>/`.

## Verdict inputs from the Aug-2026 audits (defaults above derive from these)

- EgoTouch: raw 0..135, median 0, ~30% samples > 0; grids 0..1 with `tactile_max`
  attr 60–61; all streams frame-locked (dt exactly 33.333 ms, monotonic).
- HumanTouch: raw 0..108; patch pressure 0..29.3 N; dt exactly 16.667 ms; tactile
  valid flags all-true on the audited episode.
- OpenTouch: raw rest 3072 (rail-high, INVERTED), contact dips toward 0; short-clip
  frozen-channel fractions 30–72% -> length-scaled expectations are mandatory.
