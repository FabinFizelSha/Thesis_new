# Session report: optimisation3_20260806_205816

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART3_REPORT.md](../PART3_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol.

## Purpose

Part 3 Step 1 finer profiling: split `assignment_candidate_search_ms`
(82.2% of `frame_assignment_ms` in the prior session) into row
construction, 3D AABB geometry, centroid/IoU, and scoring, to determine
which specific sub-cost to target and re-check the candidate-count growth
hypothesis on a second, independent run. Profiling-only — no behavior
change under test.

## Code change under test

Purely additive: `stage_ms` sink added to `_find_match()`, threaded from
`prepare_frame_assignments()`. Full description in `../PART3_REPORT.md`
Step 1 (finer profiling addendum).

## Runtime configuration

- Session folder: `debug/optimisation/optimisation_part3/optimisation3_20260806_205816/`
- Standard test protocol per `../../OPTIMISATION_REPORT.md`.
- Pre-run verification: 34/34 tests pass, `colcon build` succeeds.

## Results (this session)

- Received/processed/dropped/failed: 2,517 / 273 / 2,244 / 0.
- Mean masks per processed frame: 7.722 — consistent with every prior
  session.
- Total latency mean/median/p95/max: 601.630 / 587.148 / 761.702 /
  1,357.068 ms. Classifier mean: 539.628 ms — within normal SAM-driven
  run-to-run variance (`sam_inference_ms` this run: 315.190 ms mean).

## assignment_candidate_search_ms four-way breakdown

| Sub-step | Mean ms | % of assignment_candidate_search_ms | Max ms |
|---|---:|---:|---:|
| `assignment_3d_geometry_ms` | 20.451 | 35.4% | 361.938 |
| `assignment_row_init_ms` | 12.531 | 21.7% | 351.020 |
| `assignment_centroid_iou_ms` | 11.176 | 19.3% | 45.418 |
| `assignment_scoring_ms` | 7.115 | 12.3% | 34.172 |
| Sum of four | 51.273 | 88.8% | — |
| `assignment_candidate_search_ms` (unchanged outer) | 57.763 | 100% | 425.805 |
| Unaccounted (likely `_candidate_track_ids()` spatial-index lookup, called once per `_find_match` call, outside all four timed sections) | ~6.49 | ~11.2% | — |

**No single sub-step dominates.** The 3D AABB geometry block (10 geometric
calls per candidate) is largest at 35%, but row construction (22%) and
centroid/IoU (19%) are close behind — a genuinely distributed cost, not one
fixable line. The ~11% gap between the four sub-steps and the outer timer is
attributed to `_candidate_track_ids()` itself (the spatial-cell-set lookup
and union), which runs once per `_find_match` call *before* the
instrumented per-candidate loop starts, so it is not currently attributed to
any of the four columns — this cost would also plausibly scale with track
density (bigger cells to union as more tracks accumulate), consistent with
the growth pattern.

## Candidate-count growth: reconfirmed on a second, independent run

| Quartile | Frames | `frame_assignment_ms` mean | candidate count (total) mean | candidate count (max) mean |
|---:|---:|---:|---:|---:|
| 1 | 68 | 37.862 ms | 155.85 | 29.72 |
| 2 | 68 | 62.753 ms | 461.57 | 119.18 |
| 3 | 68 | 66.970 ms | 656.51 | 223.28 |
| 4 | 69 | 113.236 ms | 1027.16 | 290.91 |

Candidate count grows 6.6x (total) / 9.8x (max) Q1→Q4, closely matching the
prior session's 6.4x/9.2x — the growth trend is reproducible, not a one-run
artifact.

## New finding: scattered severe tail-latency spikes, not explained by the growth hypothesis

`assignment_3d_geometry_ms` max is 361.938 ms against a 20.451 ms mean (18x)
— a single frame (index 226). `assignment_row_init_ms` max is 351.020 ms
against a 12.531 ms mean (28x) — but here **four separate frames** (indices
238, 271, 258, 148) all show similarly extreme values (351.0 / 340.9 /
340.6 / 337.8 ms), not just one outlier. `track_association_ms` — a
separate, normally near-zero-cost stage (mean 5.402 ms; it only does real
work when `associate()`'s fallback path runs, which is rare once
`prepare_frame_assignments` has already installed forced matches) — spikes
to 330.456 ms on frame 185.

These spikes do **not** co-occur on the same frame index across stages
(226, 238/271/258/148, and 185 are all different frames), which argues
against a single global stall affecting everything at once, but their
magnitude (300+ ms on stages whose mean is 5-20 ms) is far larger than
gradual candidate-count growth would produce on any single frame — the
growth hypothesis predicts a smooth increase toward the end of the run, not
isolated 15-30x spikes on scattered frames throughout. Two hypotheses, not
yet distinguished:

1. **Lock contention** with the async RAP/VLM worker threads — both
   `prepare_frame_assignments()` and `associate()` hold
   `PersistentObjectTracker._lock` for their full body, as does every
   RAP/VLM-thread call into the tracker (`prepare_active_for_labeling`,
   `apply_rap_result`, `apply_vlm_result`, `is_semantic_labeling_open`). If a
   worker thread holds the lock during expensive work, the main
   classification thread blocks, and that wait time is currently
   indistinguishable from real compute in these timers.
2. **System-level jitter** (Jetson thermal/scheduling, GC pause, or RAP
   memory disk I/O) unrelated to this code specifically.

Not distinguishable with the current instrumentation — would need explicit
lock-wait timing (time spent blocked acquiring `self._lock` specifically,
separate from time spent doing work after acquiring it) to confirm or rule
out hypothesis 1.

## Functional / accuracy evidence

- `_find_match()`'s returned match/score/evaluations and
  `prepare_frame_assignments()`'s keep-mask are unchanged — proven by
  `test_find_match_stage_ms_sink_does_not_change_result` and the existing
  `test_prepare_frame_assignments_stage_ms_sink_does_not_change_keep_mask`.
- Zero processing failures. Mask-count distribution stable vs. all prior
  sessions (7.722 vs. 7.67-7.73 range).
- 34/34 unit tests passed and `colcon build` succeeded before this run.

## Decision

**Finer profiling validated; two distinct findings, both requiring your
direction before further code changes:**

1. **Mean-cost growth** (confirmed, reproducible): candidate count grows
   6-10x over a run; cost is spread across 3D geometry / row construction /
   centroid-IoU roughly evenly (35/22/19%), plus an unattributed ~11% likely
   in the spatial-index lookup itself. No single-line fix — reducing
   candidate count directly (benefits all sub-steps at once) is the higher-
   leverage lever, but carries real risk of excluding a legitimate match if
   done imprecisely.
2. **Tail-latency spikes** (newly observed, not explained by growth):
   300+ ms spikes on normally-cheap stages, scattered across different
   frames, magnitude far exceeding what gradual candidate growth explains.
   Candidate cause: lock contention with async RAP/VLM threads on
   `PersistentObjectTracker._lock`, not yet confirmed.

See `../PART3_REPORT.md` "Step 3" for both options laid out for your
decision.
