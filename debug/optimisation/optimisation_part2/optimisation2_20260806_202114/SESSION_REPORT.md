# Session report: optimisation2_20260806_202114

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART2_REPORT.md](../PART2_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol.

## Purpose

Revert-verification run. Confirm that reverting the Step 2 attempt 1 change
(projection-formula reorder + `.tolist()` swap, both measured to regress
`geometry_metadata_ms` in session `optimisation2_20260806_192701`) actually
restored `object_geometry.py`'s performance to its pre-attempt state, and
that no other regression was introduced by the revert itself.

## Code change under test

None — this run is against the reverted code, i.e. the same state as Step 1
(profiling instrumentation only, no Step 2 change). `object_geometry.py`'s
`estimate()` uses `(rot_m @ points_cam.T).T + tx.reshape(1, 3)` and
`[float(v) for v in centroid]`-style list comprehensions, exactly as in
session `optimisation2_20260806_022259`.

## Runtime configuration

- Session folder: `debug/optimisation/optimisation_part2/optimisation2_20260806_202114/`
- Standard test protocol per `../../OPTIMISATION_REPORT.md`.
- Pre-run verification: `pytest -q tests -p no:anyio` → 32/32 pass;
  `colcon build --packages-select rsg` → succeeds.
- Snapshot files not captured for this session, consistent with the two
  prior Part 2 sessions.

## Evidence

- Raw trace: `phase1_timing.csv`
- Generated stage analysis: `stage_summary.csv`, `stage_summary.md`
- Compared against both `optimisation2_20260806_022259` (Run 1, pre-attempt)
  and `optimisation2_20260806_192701` (Run 2, the regressed attempt)

## Results (this session)

- Received/processed/dropped/failed: 2,906 / 304 / 2,603 / 0.
- Mean masks per processed frame: 7.671 (distribution: 232 frames with 8
  masks, 51 with 7, 14 with 6, 7 with 5) — comparable to both prior sessions.
- Total latency mean/median/p95/max: 605.903 / 595.679 / 770.376 /
  1,229.409 ms.
- Classifier latency mean/median/p95/max: 548.102 / 542.662 / 682.875 /
  983.873 ms.

## Three-way comparison

| Metric | Run 1 (pre-attempt) | Run 2 (regressed attempt) | Run 3 (this session, post-revert) | Run 3 vs. Run 1 |
|---|---:|---:|---:|---:|
| `geometry_projection_ms` | 36.690 ms | 42.296 ms | **36.293 ms** | -0.397 ms (-1.1%, noise) |
| `geometry_stats_ms` | 18.467 ms | 28.276 ms | **18.145 ms** | -0.322 ms (-1.7%, noise) |
| `geometry_mask_extract_ms` | 24.135 ms | 24.496 ms | 24.104 ms | -0.031 ms (noise) |
| `geometry_depth_gather_ms` | 6.860 ms | 6.975 ms | 6.292 ms | -0.568 ms (noise) |
| **`geometry_metadata_ms` total** | **87.017 ms** | **102.888 ms** | **85.669 ms** | **-1.348 ms (-1.5%, noise)** |
| `sam_inference_ms` (context) | 337.122 ms | 302.563 ms | 322.547 ms | -14.575 ms (run-to-run variance) |
| `total_delay_ms` | 626.940 ms | 603.371 ms | 605.903 ms | -21.037 ms (mostly SAM variance) |
| Mean masks/processed frame | 7.680 | 7.679 | 7.671 | comparable |
| Processed frames | 294 | 308 | 304 | comparable |

**Confirmed: the revert fully restored `geometry_metadata_ms` to its
pre-attempt level.** Run 3's 85.669ms sits within ~1.5% of Run 1's 87.017ms —
well inside the noise band already characterized across this campaign
(compare to attempt 1's regression, which was +18.2%, more than 10× larger
than this run-to-run gap). None of the four sub-steps show any residual
elevation from the reverted change. The `total_delay_ms`/`sam_inference_ms`
differences across all three sessions continue to reflect this hardware's
already-documented run-to-run SAM variance, not code behavior.

## Functional / accuracy evidence

- Zero processing failures.
- Mean masks/processed frame stable across all three sessions (7.680 / 7.679
  / 7.671), confirming comparable scene content and no tracking-side
  behavior change from the revert.
- 32/32 unit tests passed and `colcon build` succeeded before this run.

## Decision

**Revert confirmed clean.** Part 2 is verified back at its Step 1
(profiling-complete, zero regression) state. No further action needed on
attempt 1. Proceed per `../PART2_REPORT.md` "Step 2 (attempt 2)" — awaiting
your direction on whether to pursue `geometry_mask_extract_ms` next or close
Part 2 here and move to `frame_assignment_ms`.
