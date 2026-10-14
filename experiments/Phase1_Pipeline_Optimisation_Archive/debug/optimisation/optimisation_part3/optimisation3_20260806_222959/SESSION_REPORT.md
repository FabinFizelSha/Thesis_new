# Session report: optimisation3_20260806_222959

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART3_REPORT.md](../PART3_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol.

## Purpose

Part 3 Path B: measure `assignment_lock_wait_ms` (in `prepare_frame_assignments`)
and `association_lock_wait_ms` (in `associate`) to confirm or rule out lock
contention with the async RAP/VLM worker threads as the cause of the
scattered 300+ ms tail spikes observed in Run 2/Run 3. Profiling-only — no
behavior change under test.

## Code change under test

Purely additive: lock-wait timing added around `self._lock.acquire()` in
`prepare_frame_assignments()` and `associate()`, behaviorally identical to
the `with self._lock:` it replaced. Full description in `../PART3_REPORT.md`
Step 5.

## Runtime configuration

- Session folder: `debug/optimisation/optimisation_part3/optimisation3_20260806_222959/`
- Standard test protocol per `../../OPTIMISATION_REPORT.md`.
- Pre-run verification: 35/35 tests pass, `colcon build` succeeds.

## Results (this session)

- Received/processed/dropped/failed: 2,383 / 277 / 2,106 / 0.
- Mean masks per processed frame: 7.646 — consistent with every prior
  session.
- Total latency mean/median/p95/max: 568.379 / 554.715 / 722.210 /
  1,293.912 ms. Classifier mean: 508.370 ms. `sam_inference_ms` 304.238 ms
  mean — within the hardware's already-characterized run-to-run range.

## Path B result: lock contention ruled out

| Stage | Mean ms | Max ms |
|---|---:|---:|
| `assignment_lock_wait_ms` | 0.010 | 1.369 |
| `association_lock_wait_ms` | 0.051 | 3.939 |

Both are negligible — nowhere near the 300+ ms spikes under investigation,
even at their own worst case across the whole 277-frame session.

**The decisive check:** this session did reproduce a severe spike
(`frame_assignment_ms` = 335.660 ms on sequence 1954, `num_masks` = 7,
`assignment_candidate_count_total` = 474 — an unremarkable mask/candidate
count, not itself an extreme outlier). On that *exact* frame:

| Field | Value |
|---|---:|
| `assignment_candidate_search_ms` | 314.699 ms |
| `assignment_row_init_ms` | 1.268 ms |
| `assignment_3d_geometry_ms` | 5.872 ms |
| `assignment_centroid_iou_ms` | 5.164 ms |
| `assignment_scoring_ms` | 11.851 ms |
| **Sum of the four instrumented sub-steps** | **24.155 ms** |
| **Unaccounted gap** (`candidate_search - sum of four`) | **290.544 ms** |
| `assignment_lock_wait_ms` (this frame) | 0.005 ms |
| `association_lock_wait_ms` (this frame) | 0.334 ms |

On this frame, every instrumented sub-step was completely ordinary (24ms
total, typical for 7 masks) and lock-wait was negligible — yet
`assignment_candidate_search_ms` still hit 314.7ms. The only code that runs
inside `assignment_candidate_search_ms` but *outside* all four instrumented
sub-steps and outside the lock-wait window is `_candidate_track_ids()`
itself, called once per observation (7 times this frame) before the
per-candidate loop begins.

**This directly and precisely implicates `_candidate_track_ids()`'s own
body** — most plausibly its final line,
`[track_id for track_id in self._tracks if track_id in candidate_ids]`,
which scans every track in `self._tracks` (the full tracker, not the
filtered candidate set) to preserve dict-insertion order. This is exactly
the finding flagged (but not yet acted on) before this session's Path B
work began. `self._tracks` grows across a run; a one-off ~290ms cost on a
single mid-to-late-run frame, with unremarkable candidate counts, is
consistent with this scan becoming expensive once the total track count
crosses some size — not with lock contention, which this session
conclusively rules out for these two call sites.

Cross-checking the other top-`frame_assignment_ms` frames this session
(126.7, 123.0, 122.2, 121.6, 117.2, 116.1 ms) shows a *different* pattern:
their instrumented sub-steps (particularly `assignment_3d_geometry_ms`,
45-60ms on several of them) are themselves elevated, consistent with the
already-confirmed candidate-count growth mechanism — ordinary tail-of-
distribution behavior, not anomalous. Sequence 1954 stands alone as the
one frame where the cost is almost entirely in the uninstrumented gap.

## Functional / accuracy evidence

- `associate()`'s returned metadata/track record unchanged — proven by
  `test_associate_stage_ms_sink_does_not_change_result`.
- Zero processing failures. Mask-count distribution stable vs. all prior
  sessions.
- 35/35 unit tests passed and `colcon build` succeeded before this run.

## Decision

**Path B closed: lock contention ruled out** for `prepare_frame_assignments`
and `associate` (the two most-implicated call sites). The tail-spike
investigation redirects to `_candidate_track_ids()`'s final-line full-track
scan, with this session providing frame-level evidence (not just
aggregate/statistical correlation) that this specific mechanism, not
locking, is responsible for at least the most severe outlier observed so
far. See `../PART3_REPORT.md` for the updated plan — the previously-flagged
order-preservation question (does `_find_match`'s tie-breaking or the
Hungarian assignment depend on `self._tracks`'s iteration order) needs to be
resolved before changing this line, same discipline as every other change
in this campaign.
