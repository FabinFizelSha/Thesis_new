# Revisit Experiment — Part 1: Track Re-identification Accuracy

**Status: complete.** Five sessions run 2026-09-09, evaluated below. Part 2
(frame-throughput / FPS improvement across sessions, at a higher bag playback
rate) is a separate, later experiment — see the note in Section 7.

## 1. Objective

Test whether the multi-session resume mechanism (Phase 1 tracker-state
persistence + Hydra DSG/mesh resume, both documented in full in
`IMPLEMENTATION.md`) actually achieves its purpose: when the pipeline is
launched again against the same environment, does it **recognise objects it
has already mapped**, rather than silently re-discovering the same physical
scene as a set of brand-new objects each time?

## 2. Hypothesis

Across N consecutive sessions replaying the same environment:

1. Session 1 establishes the baseline map from an empty state — every object
   is necessarily `new_track`.
2. Sessions 2 onward should resolve the large majority of re-encountered
   objects as `global_revisit_association` (Phase 1's cross-session
   association mode — see `IMPLEMENTATION.md` Section 2.2 for why every
   resume is deliberately evaluated under revisit-mode thresholds), with
   `new_track` events approaching zero.
3. As a consequence, the object-labelling VLM should be called far less
   often from session 2 onward — a restored, already-labelled track is never
   re-dispatched (`IMPLEMENTATION.md` Section 2.3) — so `vlm_call_count`
   should fall sharply after session 1.

## 3. Experimental setup

**Bag**: `uHumans2_office_s1_00h_ros2`, topics `/clock /tf /tf_static
/tesse/left_cam/rgb/image_raw /tesse/depth_cam/mono/image_raw
/tesse/left_cam/camera_info /tesse/odom`, `--rate 0.1`, wrapped in
`timeout --signal=INT 300` — a fixed 300-real-second segment from the
bag's own beginning every session (not paused/resumed — each session is a
fresh `ros2 bag play` invocation, deliberately: this is Scenario B,
"next-day revisit", from `IMPLEMENTATION.md` Section 5).

**Pipeline**: `ros2 launch rsg rsg_all.launch.py start_risk_vlm:=false`.

**Config held fixed for all 5 sessions** (`rsg_pipeline.yaml` unless noted):
- `phase1.risk_vlm.enabled: false` — a second VLM server on the same GPU
  would confound any GPU-load reading; irrelevant to this Part but kept off
  for consistency with Part 2.
- `phase1.persistent_tracking.session_persistence.enabled: true`,
  `time_shift_sec: 86400.0` — the mechanism that forces every resumed
  session's first re-observation of a track through revisit-mode
  thresholds regardless of real elapsed time (`IMPLEMENTATION.md` Section 2.2).
- Tracking gates: `global_min_independent_groups: 3`,
  `global_centroid_pass_m: 0.60`, `global_containment_threshold: 0.92`,
  `global_recent_min_score` / `global_revisit_min_score: 0.30`. These are the
  same values already in production use, not tuned for this experiment —
  see the project memory `tracking-quorum-gate-decision.md` for their
  history.
- `hydra.enable_object_merging: true`, `resume_reset_trajectory: true` — both
  defaults, see `IMPLEMENTATION.md` Section 3.

**Diagnostics**: `phase1.diagnostics.enabled: true`,
`diagnostics.log_tracking: true` — produces
`debug/object_tracking_experiment_part2/tracking_quality/tracking_associations_*.jsonl`,
one row per mask-to-track association *decision* Phase 1 makes (not one row
per frame — a frame with several visible objects produces several rows).
This is the ground truth this report is built on: the `reason` field records
exactly which branch fired (`new_track`, `global_revisit_association`,
`global_recent_association`) for every single decision, so "how many objects
were recognised" is read directly rather than inferred.

**Procedure**: `python3 debug/revisit_experiment/run_session.py` once per
session (see `README_FPS_EXPERIMENT.md`) — automatic session id, automatic
archiving on pipeline shutdown, memory **not** cleared between sessions 1–5
(each resumes from the previous session's save; only session 1 followed a
`clear_memory.py` baseline).

## 4. Results

### 4.1 Association decisions, by session (ground truth)

Source: `tracking_associations_*.jsonl`, `reason` field, every row.

| Session | start (local) | `new_track` | `global_revisit_association` | `global_recent_association` | total decisions |
|---|---|---:|---:|---:|---:|
| 1 | 23:04:34 | **53** | 0 | 779 | 832 |
| 2 | 23:12:03 | **1** | 47 | 790 | 838 |
| 3 | 23:20:18 | **3** | 50 | 777 | 830 |
| 4 | 23:36:56 | **0** | 52 | 785 | 837 |
| 5 | 23:46:18 | **1** | 50 | 785 | 836 |

Sessions 2–5 combined: **5 new_track vs. 199 global_revisit_association** —
**97.5% of re-encountered objects correctly resolved as revisits**
(199 / 204). Session 4 achieved the ideal outcome outright: zero new tracks,
52 revisits, nothing left over.

### 4.2 Corroborating metrics, by session

Source: `debug/revisit_experiment/sessions/summary_all_sessions.csv`
(archived by `run_session.py`; `track_count` from Phase 1's own saved
tracker state, `vlm_call_count` from the phase1 timing CSV's
`event=="completed"` rows).

| Session | `track_count` (cumulative) | Δ from previous | `vlm_call_count` | avg GPU % | avg FPS | avg frame latency |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 53 | — (baseline) | 28 | 51.6 | 0.489 | 3328 ms |
| 2 | 54 | +1 | 11 | 25.6 | 0.491 | 3280 ms |
| 3 | 57 | +3 | 5 | 19.4 | 0.491 | 3282 ms |
| 4 | 57 | +0 | 2 | 30.3 | 0.496 | 3232 ms |
| 5 | 58 | +1 | 1 | 22.8 | 0.495 | 3254 ms |

The `track_count` deltas match Section 4.1's `new_track` counts exactly
(+1, +3, +0, +1), as they must — this is a consistency check on the
tooling, not an independent finding.

`vlm_call_count` falls monotonically: **28 → 11 → 5 → 2 → 1**. This is the
sharpest, cleanest trend in the whole dataset, and is exactly what the
resume mechanism is supposed to produce: a restored track already carries a
label and is never re-dispatched to the VLM (`IMPLEMENTATION.md` Section 2.3),
so the number of calls should track the number of *not-yet-labelled* objects
still being discovered — which Section 4.1 shows collapses to almost nothing
after session 1.

## 5. The residual misses — not random noise

Five `new_track` events occurred outside session 1. Their 3D centroids:

| Session | Centroid (m) | mask area (px) |
|---|---|---:|
| 2 | (−10.14, 28.97, 3.35) | 21202 |
| 3 | (−8.17, 21.67, 2.32) | 11198 |
| 3 | (−10.35, 28.90, **4.15**) | 4782 |
| 3 | (−11.03, 28.69, 2.42) | 19670 |
| 5 | (−11.14, 29.22, 3.65) | 12639 |

Three of these five — session 2's, session 3's second, and session 5's —
sit within **0.8–1.1 m of each other** (pairwise distances 0.83 m, 0.99 m,
1.07 m), all in the same corner of the map and all above 3.3 m in height.
This is very unlikely to be independent random error: it reads as **one
specific physical object or region that intermittently fails revisit
association**, correctly recognised in some sessions (4, and the other two
session-3 spots do not recur) and missed in others.

A tall or vertically-extended object — where the recognised centroid can
shift noticeably between viewing angles or as more of it comes into view —
is exactly the failure mode `global_centroid_pass_m` (0.60 m) was already
flagged as a possible bottleneck for in `IMPLEMENTATION.md` Section 6 and
the project memory `tracking-quorum-gate-decision.md`. This location is now
a concrete, reproducible test case for that investigation, rather than a
theoretical concern — worth revisiting the crop/mask around
(−10.5, 29.0, ~3.5–4.0) specifically before touching that gate.

**Not pursued further here** per standing direction to exhaust measurement
before touching tracking gates again (`feedback-prefer-parameter-tuning`
project memory) — this section documents the lead, it does not act on it.

## 6. Secondary observations

- **Hydra-side duplicate slots**: `hydra_duplicate_slot_count` was 3, 6, 6,
  6, 6 across the five sessions (from
  `debug/check_duplicate_objects.py` against each session's archived
  `hydra_dsg.json`). All are archived-node pairs (never an active node
  colliding with an archived one), so none represent an unresolved
  duplicate in the live view — see `IMPLEMENTATION.md` Section 4.1 for why
  that specific case is deliberately left alone. Flat after session 2;
  not investigated further as part of this experiment.
- **GPU load did not decline monotonically** (51.6 → 25.6 → 19.4 → 30.3 →
  22.8%) despite `vlm_call_count` doing so cleanly. Frame throughput and
  latency were essentially flat across all five sessions (0.489–0.496 fps;
  3232–3328 ms). Read together, this suggests Phase 1 was **input-bound**
  at this bag's `--rate 0.1` — frames simply were not arriving fast enough
  for GPU headroom freed up by fewer VLM calls to translate into more
  frames processed per second. This is the reason Part 2 (Section 7) tests
  at a higher playback rate rather than reusing this data for a
  throughput claim.
- **Decision-count consistency**: total association decisions per session
  (832, 838, 830, 837, 836) and frames processed (147, 147, 147, 148, 148)
  are stable within a narrow band across all five sessions, indicating the
  bag segment and processing conditions were comparable run to run — a
  basic sanity check that the comparison in Section 4 is measuring the
  hypothesis and not run-to-run variability in workload.

## 7. Conclusion and relationship to Part 2

**The revisit-accuracy objective (Part 1) is met.** 97.5% of re-encountered
objects across four resumed sessions were correctly identified as
already-known tracks rather than duplicated, with one session achieving a
perfect result (zero new tracks). The residual failures cluster at a single
identifiable location rather than being spread randomly, which is itself
informative: it points at a specific, boundable gap rather than a systemic
failure of the resume mechanism.

**Part 2** (not yet run) tests the second half of the original hypothesis:
that reduced VLM load from successful revisit recognition frees GPU
capacity that shows up as *increased frame throughput* on later sessions.
This dataset's flat FPS/latency despite a 28→1 call reduction suggests that
claim needs a bag played at a higher rate than 0.1x to have a chance of
showing up — at 0.1x, Phase 1 appears limited by how fast frames arrive, not
by GPU capacity. Part 2 will re-run this same session-by-session structure
at a higher `--rate` and report frame throughput as the primary metric,
using this document's Section 6 GPU-load observation as its starting
hypothesis rather than assuming the FPS half of the original hypothesis is
already disproven.

## 8. Data and reproducibility

- Raw per-session archives: `debug/revisit_experiment/sessions/session_*/`
  (not tracked in git — see `.gitignore`; regenerate by re-running
  `run_session.py` against the same bag).
- Combined table: `debug/revisit_experiment/sessions/summary_all_sessions.csv`.
- Ground-truth association log per session:
  `debug/object_tracking_experiment_part2/tracking_quality/tracking_associations_*_final.jsonl`,
  matched to sessions by timestamp (session 1 → `..._211144_final.jsonl`,
  session 2 → `..._211956_final.jsonl`, session 3 →
  `..._212858_final.jsonl`, session 4 → `..._214500_final.jsonl`, session 5
  → `..._215158_final.jsonl`). These filenames embed UTC time while
  sessions and the resource/timing CSVs use local time (CEST, UTC+2) — a
  cosmetic inconsistency between two independently-built diagnostic
  systems, not a data problem; matching was done by chronological order
  and cross-checked against each file's `track_count`/new-track total
  agreeing exactly with the corresponding session's summary row.
- Tooling: `debug/revisit_experiment/{run_session.py,snapshot_run.py,monitor_resources.py}`.
- Full implementation reference for every mechanism named above:
  `debug/revisit_experiment/IMPLEMENTATION.md`.
