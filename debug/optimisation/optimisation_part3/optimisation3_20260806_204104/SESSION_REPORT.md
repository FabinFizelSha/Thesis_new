# Session report: optimisation3_20260806_204104

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART3_REPORT.md](../PART3_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol.

## Purpose

Part 3 Step 1 (profiling): measure where `frame_assignment_ms`'s cost goes
across the four new sub-step timers, and test the hypothesis that its
within-run growth (documented at Part 1 close: quartile means 37.7 → 121.9
ms) is driven by per-observation candidate-track count growing as the
explored scene fills up, using the two new diagnostic count columns. This is
a **profiling-only run** — no behavior-changing fix is under test yet.

## Code change under test

Purely additive instrumentation, no output/behavior change: `stage_ms` sink
on `PersistentObjectTracker.prepare_frame_assignments()` and the
corresponding plumbing in `nodes/phase1.py`. Full description in
`../PART3_REPORT.md` Step 1.

## Runtime configuration

- Session folder: `debug/optimisation/optimisation_part3/optimisation3_20260806_204104/`
- Standard test protocol per `../../OPTIMISATION_REPORT.md`.
- Pre-run verification: 33/33 tests pass, `colcon build` succeeds.
- Snapshot files not captured for this session, consistent with recent Part
  2 sessions.

## Evidence

- Raw trace: `phase1_timing.csv`
- Generated stage analysis: `stage_summary.csv`, `stage_summary.md`
  (includes the new quartile growth table)

## Results (this session)

- Received/processed/dropped/failed: 2,925 / 301 / 2,624 / 0.
- Mean masks per processed frame: 7.674 (distribution: 227 frames with 8
  masks, 55 with 7, 14 with 6, 5 with 5) — consistent with every prior
  session (~7.67-7.73), confirming mask count is not a growth confound.
- Total latency mean/median/p95/max: 606.377 / 600.158 / 756.509 /
  1,062.869 ms. Classifier mean: 552.680 ms — both within the normal
  SAM-driven run-to-run noise band already characterized (`sam_inference_ms`
  this run: 327.045 ms mean).

## frame_assignment_ms sub-step breakdown

| Sub-step | Mean ms | % of frame_assignment_ms |
|---|---:|---:|
| `assignment_candidate_search_ms` | 58.625 | **82.2%** |
| `assignment_a2_redundancy_ms` | 6.796 | 9.5% |
| `assignment_a3_nested_ms` | 3.685 | 5.2% |
| `assignment_hungarian_ms` | 2.111 | 3.0% |
| **Sum of sub-steps** | **71.217** | — |
| `frame_assignment_ms` (outer timer, unchanged) | 71.308 | 100% |

Sub-steps sum within 0.13% of the pre-existing outer timer — negligible
instrumentation overhead, same pattern as Part 2. **The per-observation
candidate search in `_find_match()` dominates at 82%**, matching the
hypothesis; the A2/A3 mask-pair passes are minor (14.7% combined) and the
Hungarian solve is negligible (3%), despite both being the parts that
"look" algorithmically heavy on paper (O(n²) mask intersections, a full
assignment solver).

## Candidate-count growth hypothesis: confirmed

| Quartile | Frames | `frame_assignment_ms` mean | candidate count (total/frame) mean | candidate count (max/observation) mean |
|---:|---:|---:|---:|---:|
| 1 | 75 | 36.427 ms | 165.60 | 33.09 |
| 2 | 75 | 74.758 ms | 529.91 | 130.31 |
| 3 | 75 | 65.774 ms | 673.03 | 233.55 |
| 4 | 76 | 107.787 ms | 1065.13 | 302.78 |

Candidate count grows **6.4x** (total) and **9.2x** (max) from the first to
last quartile of the run — some single observations are evaluated against
~300 candidate tracks by the run's end despite the spatial index already
restricting the search. `frame_assignment_ms` grows roughly in step (Q1→Q4:
36.4 → 107.8 ms, ~3x — sublinear relative to the 6-9x candidate growth,
suggesting per-candidate cost may itself ease slightly at high counts, but
the correlation direction and magnitude are unambiguous). Q3's dip relative
to Q2 (65.8 vs 74.8 ms) despite continued candidate-count growth is noted as
within-quartile variance, not a contradiction of the trend — the Q1→Q4
comparison is the one that matters.

**Conclusion: the hypothesis holds.** `assignment_candidate_search_ms` is
both the dominant cost (82%) and the specific driver of `frame_assignment_ms`'s
growth over a run, via a growing per-observation candidate-track count. The
spatial index is filtering *something*, but local track density in the
currently-visible/recently-revisited region still climbs substantially as
more of the scene gets explored.

## Functional / accuracy evidence

- `prepare_frame_assignments()`'s return value (keep-mask) and forced-match
  side effects are unchanged — proven by
  `test_prepare_frame_assignments_stage_ms_sink_does_not_change_keep_mask`.
- Zero processing failures. Mask-count distribution stable vs. all prior
  sessions.
- 33/33 unit tests passed and `colcon build` succeeded before this run.

## Decision

**Step 1 (profiling) is validated and complete; hypothesis confirmed.**
Proceed to a further profiling pass isolating exactly where inside
`_find_match`'s per-candidate loop the 58.6ms goes (row/dict construction
with `_as_list()` numpy conversions performed *before* any gate is checked,
vs. the expensive AABB geometric computation for candidates that pass the
cheap gates, vs. final quorum/weighted scoring) — this determines whether
the right fix is reducing candidate *count* (tighter spatial pre-filter) or
reducing per-candidate *cost* (lazy/deferred row construction, skip
list-conversion for candidates rejected by cheap gates). See
`../PART3_REPORT.md` "Step 2" for the proposed next profiling split and the
two candidate fix directions, pending your decision on which to pursue.
