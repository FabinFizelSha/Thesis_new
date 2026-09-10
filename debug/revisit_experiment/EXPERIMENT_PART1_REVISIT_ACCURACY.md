# Revisit Experiment — Part 1: Track Re-identification Accuracy

**Status: closed.** Five sessions run 2026-09-09, evaluated below. This
experiment's scope is the revisit-accuracy result alone; the
frame-throughput/FPS question is explicitly out of scope here and will be
its own separate experiment, run and reported independently — see Section 7.

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
  would confound any GPU-load reading; not central to this experiment's
  result but kept off throughout for a clean GPU baseline.
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
  22.8%) despite `vlm_call_count` doing so cleanly, and frame throughput and
  latency were essentially flat across all five sessions (0.489–0.496 fps;
  3232–3328 ms). Read in isolation this looks like Phase 1 was
  input-starved at `--rate 0.1`, with no queue backlog for reduced GPU load
  to drain faster. **That reading does not survive checking
  `frame_drop_count` against the bag's actual native rate**, corrected here
  before this became the closing record: `frame_drop_count` was
  413–416 per session against only 147–148 frames processed — a
  **73.6–73.9% drop rate, flat across all five sessions**, all logged as
  `frame_fifo_full`. The bag's native RGB rate is 16.4 Hz
  (`ros2 bag info`: 8307 messages / 506.1 s), and the preprocessor's own
  rate limiter is a no-op at the current config (capped at 25 Hz, above
  that 16.4 Hz), so at `--rate 0.1` frames reach Phase 1 at roughly 1.6 Hz
  wall-clock — already over three times faster than Phase 1's own ~0.49 Hz
  processing ceiling. The queue was constantly overflowing, not idle: **Phase
  1 was already compute-saturated at the slowest rate tested**, and the flat
  `avg_sam_inference_ms` (~1990–2020 ms) regardless of concurrent GPU load
  (19–52%) says individual SAM calls were not measurably slowed by whatever
  VLM contention was happening. Whether a much higher arrival rate — closer
  to native or beyond it — creates enough *simultaneous* GPU contention
  between SAM and VLM inference to show up in this data is the real open
  question, not "was the queue full" (it already was).
- **Decision-count consistency**: total association decisions per session
  (832, 838, 830, 837, 836) and frames processed (147, 147, 147, 148, 148)
  are stable within a narrow band across all five sessions, indicating the
  bag segment and processing conditions were comparable run to run — a
  basic sanity check that the comparison in Section 4 is measuring the
  hypothesis and not run-to-run variability in workload.

## 7. Conclusion and closing notes

**Status: this experiment is closed at Part 1.** Its objective (Section 1)
was the revisit-accuracy half of the original hypothesis, and that objective
is met: **97.5% of re-encountered objects across four resumed sessions
(199 of 204) were correctly identified as already-known tracks rather than
duplicated**, with one session (4) achieving a perfect result — zero new
tracks. The residual failures cluster at a single identifiable location
(Section 5) rather than being spread randomly, which is itself informative:
it points at a specific, boundable gap rather than a systemic failure of the
resume mechanism.

**The frame-throughput half of the original hypothesis — that reduced VLM
load should free GPU capacity that shows up as higher FPS on later
sessions — is neither confirmed nor disproven by this experiment**, and is
deliberately left open rather than force-fitted here. This dataset's flat
FPS/latency despite a 28→1 VLM-call reduction was initially read as Phase 1
being input-starved at `--rate 0.1`; Section 6 corrects that reading with
the `frame_drop_count` evidence — the pipeline was in fact already
compute-saturated at that rate (a ~74% drop rate, arrival already ~3× faster
than Phase 1's processing ceiling), so the flat result may instead mean SAM
inference simply is not measurably slowed by the VLM contention this
dataset produced, at any input rate.

**Distinguishing those two explanations needs a dedicated follow-up
experiment, run separately from this one** — a higher bag playback rate
(discussed with the user going into it, not part of this document's
results), its own session-by-session structure, and its own report,
rather than a "Part 2" appended here. This document's scope ends at the
revisit-accuracy result above; the throughput question is out of scope for
it, not merely deferred within it.

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
