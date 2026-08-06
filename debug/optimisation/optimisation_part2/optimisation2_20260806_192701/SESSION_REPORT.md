# Session report: optimisation2_20260806_192701

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART2_REPORT.md](../PART2_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol.

## Purpose

Part 2 Step 2: measure the effect of rewriting
`ObjectGeometryEstimator.estimate()`'s world-transform line from
`(rot_m @ points_cam.T).T + tx.reshape(1, 3)` to
`points_cam @ rot_m.T + tx.reshape(1, 3)` (algebraically identical, targeting
`geometry_projection_ms`, the largest sub-step found in Step 1's profiling —
see `optimisation2_20260806_022259/SESSION_REPORT.md`), plus a bundled
`.tolist()` swap in the stats block.

## Code change under test

`nodes/support/phase1/object_geometry.py`: the projection-formula reorder and
the `.tolist()` swap described above. New equivalence tests in
`tests/test_object_geometry.py` verified before this run: 33/33 tests pass,
`colcon build --packages-select rsg` succeeds.

## Runtime configuration

- Session folder: `debug/optimisation/optimisation_part2/optimisation2_20260806_192701/`
- Same hardware/software/bag/launch protocol as the standard test protocol in
  `../../OPTIMISATION_REPORT.md`.
- Snapshot files (`git_status.txt`, `working_tree.patch`,
  `rsg_pipeline_snapshot.yaml`, `system_setup.txt`) were not captured for
  this session, consistent with `optimisation2_20260806_022259`.

## Evidence

- Raw trace: `phase1_timing.csv`
- Generated stage analysis: `stage_summary.csv`, `stage_summary.md`
- Compared against: `../optimisation2_20260806_022259` (Run 1 — the session
  that isolated NanoSAM's run-to-run variance, so it is the correct baseline
  for reading this specific change's effect, per the campaign's measurement
  caveat).

## Results (this session)

- Received/processed/dropped/failed: 2,895 / 308 / 2,587 / 0.
- Mean masks per processed frame: 7.679 (distribution: 233 frames with 8
  masks, 56 with 7, 14 with 6, 5 with 5) — essentially identical to Run 1's
  7.680, so the two sessions processed comparable scene content.
- Total latency mean/median/p95/max: 603.371 / 596.320 / 770.743 /
  1,403.228 ms.
- Classifier latency mean/median/p95/max: 539.546 / 543.376 / 649.224 /
  932.017 ms.

## Comparison to Run 1 (`optimisation2_20260806_022259`, before this change)

| Metric | Run 1 (before) | This session (after) | Change | Verdict |
|---|---:|---:|---:|---|
| `geometry_projection_ms` (line changed) | 36.690 ms | 42.296 ms | **+5.606 ms (+15.3%)** | **worse** |
| `geometry_stats_ms` (`.tolist()` change) | 18.467 ms | 28.276 ms | **+9.809 ms (+53.1%)** | **worse** |
| `geometry_mask_extract_ms` (untouched) | 24.135 ms | 24.496 ms | +0.361 ms | noise |
| `geometry_depth_gather_ms` (untouched) | 6.860 ms | 6.975 ms | +0.115 ms | noise |
| **`geometry_metadata_ms` total** | **87.017 ms** | **102.888 ms** | **+15.871 ms (+18.2%)** | **worse** |
| `sam_inference_ms` (unrelated, context) | 337.122 ms | 302.563 ms | -34.559 ms | system was *faster*, not slower |
| `classifier_delay_ms - sam_delay_ms` (non-SAM portion) | 211.233 ms | 216.283 ms | +5.050 ms | worse, net of geometry vs. other stage variance |
| `frame_assignment_ms` (untouched) | 73.183 ms | 65.766 ms | -7.417 ms | improved, but not caused by this change |
| Mean masks/processed frame | 7.680 | 7.679 | -0.001 | scene content comparable — not a confound |

**Both sub-changes regressed, and the system was under less load (not more)
during this run** — `sam_inference_ms` dropped 34.6ms, meaning the Jetson was
faster overall, yet the exact code this change touched got slower. This
rules out "adverse system conditions masked a real gain" as an explanation.
`frame_assignment_ms` improving by -7.4ms is very likely ordinary run-to-run
track-registry-state variance (this part's code was not touched by Step 2)
and does not offset the geometry regression in the metric that matters for
this step.

## Interpretation

The projection-formula reorder (`points_cam @ rot_m.T` in place of
`(rot_m @ points_cam.T).T`) is a well-known general numpy heuristic —
operate on the already-contiguous array, avoid a transposed view — but it did
not hold on this hardware. The most likely explanation: for a 3×3 matrix,
numpy/BLAS on this Jetson's numpy build likely does not dispatch either
matmul form through a real BLAS `dgemm` call (the dimension is too small to
benefit), so the actual cost is dominated by array-construction/dispatch
overhead specific to this platform's numpy build, where the reordering did
not help and, on this data, measurably hurt. The `.tolist()` swap regressed
even more sharply (+53.1%) — also unexpected, and not yet explained; it may
interact with something platform/numpy-version-specific rather than being a
universal slowdown.

This is exactly the scenario the controlled-testing protocol exists to
catch: a plausible, textbook micro-optimization that regresses in practice on
the actual target hardware. Timing intuition from general numpy guidance is
not a substitute for a measured run on this platform.

## Functional / accuracy evidence

- `test_object_geometry.py`'s equivalence tests confirmed the change was
  numerically correct (same output) before this run — the regression is a
  pure performance finding, not a correctness bug. No accuracy concern here.
- Zero processing failures.

## Decision

**Reject and revert.** Both parts of the Step 2 change (`nodes/support/phase1/object_geometry.py`'s
projection-formula reorder and the `.tolist()` swap) have been reverted to
their pre-Step-2 form (matching the Step 1 profiling-only state). The
now-obsolete formula-equivalence test was removed from
`tests/test_object_geometry.py`; the remaining regression tests in that file
stay as a general `ObjectGeometryEstimator` correctness suite. Verified after
revert: 32/32 tests pass, `colcon build --packages-select rsg` succeeds.

Part 2 returns to its Step 1 state (profiling complete, start point
unchanged at 602.392 ms mean inherited from Part 1's close) and needs a new
Step 2 candidate. `geometry_mask_extract_ms` (28% of cost, the `np.where`
mask-coordinate extraction) is the next largest sub-step, though it was
previously assessed as "largely fundamental at the current interface" in
Step 1 — see `../PART2_REPORT.md` for the updated plan.
