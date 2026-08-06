# Session report: optimisation2_20260806_022259

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART2_REPORT.md](../PART2_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol.

## Purpose

Part 2 Step 1 (profiling): measure where `geometry_metadata_ms`'s ~87ms mean
cost actually goes, using the four new sub-step timers added to
`ObjectGeometryEstimator.estimate()` (`geometry_mask_extract_ms`,
`geometry_depth_gather_ms`, `geometry_projection_ms`, `geometry_stats_ms`).
This is a **profiling-only run** — no behavior-changing fix is under test
yet; the goal is to attribute cost before proposing one, per the Part 2
protocol.

## Code change under test

Purely additive instrumentation, no output/behavior change:
`nodes/support/phase1/object_geometry.py` (optional `stage_ms` sink on
`estimate()`) and the corresponding plumbing in `nodes/phase1.py`
(`build_object_metadata`, `run_rap_and_metadata`,
`_publish_hydra_from_result`). Full description in `../PART2_REPORT.md`
Step 1.

## Runtime configuration

- Session folder: `debug/optimisation/optimisation_part2/optimisation2_20260806_022259/`
- Same hardware/software/bag/launch protocol as Part 1 (see
  `../../OPTIMISATION_REPORT.md` "Standard test protocol") — Jetson AGX Orin,
  ROS 2 Humble, `rsg_all.launch.py`, `office_s1_00h_v2` bag at rate 1 for
  180 s.
- **Note:** `git_status.txt`, `working_tree.patch`, `rsg_pipeline_snapshot.yaml`,
  and `system_setup.txt` were not captured for this session (snapshot capture
  remains a manual step by choice, not yet automated — see
  `../../OPTIMISATION_REPORT.md` for the standard commands to capture them on
  future runs if a full evidence set is wanted for this session).

## Evidence

- Raw trace: `phase1_timing.csv` (3,032 data rows)
- Generated stage analysis: `stage_summary.csv`, `stage_summary.md`
- Compared against: `../../optimisation_part1/optimisation1_20260806_015126`
  (Part 1 close — the current Part 2 start point)

## Results (this session)

- Received/processed/dropped/failed: 2,928 / 294 / 2,635 / 0.
- Processing ratio: 10.04%.
- Total latency mean/median/p95/max: 626.940 / 612.556 / 800.173 /
  1,203.036 ms.
- Classifier latency mean/median/p95/max: 569.159 / 563.537 / 719.875 /
  1,031.938 ms.
- Mean masks per processed frame: 7.680 (distribution: 218 frames with 8
  masks, 61 with 7, 12 with 6, 3 with 5).

## Geometry sub-step breakdown (the reason for this run)

| Sub-step | Mean ms | % of geometry_metadata_ms | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|
| `geometry_projection_ms` | 36.690 | 42.2% | 32.235 | 80.139 | 118.609 |
| `geometry_mask_extract_ms` | 24.135 | 27.7% | 21.329 | 40.803 | 63.663 |
| `geometry_stats_ms` | 18.467 | 21.2% | 16.514 | 33.465 | 52.132 |
| `geometry_depth_gather_ms` | 6.860 | 7.9% | 5.411 | 16.844 | 30.897 |
| **Sum of sub-steps** | **86.152** | — | — | — | — |
| `geometry_metadata_ms` (outer timer, unchanged) | 87.017 | 100% | 82.687 | 148.585 | 195.524 |

The four sub-steps sum to within 0.87ms (1.0%) of the pre-existing outer
timer — consistent with normal per-call/dict-accumulation overhead, not a
measurement error. **`geometry_projection_ms` (the camera back-projection and
world-transform: `(xs_valid-cx)*z_valid/fx`, `(ys_valid-cy)*z_valid/fy`, and
`rot_m @ points_cam.T`) is the single largest sub-step at 42%.**
`geometry_mask_extract_ms` (`np.where(mask)` plus bbox/centroid/area, run
once per mask before any depth/3D work) is second at 28% and is largely
unavoidable at the current interface — it is the cost of finding every `True`
pixel in a full-resolution boolean mask, up to 8 times per frame.
`geometry_depth_gather_ms`, where `projection_stride` actually applies, is
the *smallest* sub-step at 8% — confirming stride is not the right lever here
(and it stays fixed at 1 per the Part 2 constraint regardless).

## Comparison to previous session (`optimisation1_20260806_015126`, Part 1 close)

| Metric | Part 1 close | This session | Change | Attribution |
|---|---:|---:|---:|---|
| Frames received | 2,829 | 2,928 | +99 | run-to-run bag/system variance |
| Frames processed | 303 | 294 | -9 (-3.0%) | see SAM variance below |
| Processing ratio | 10.71% | 10.04% | -0.67 pp | — |
| Mean total latency | 602.392 ms | 626.940 ms | +24.548 ms (+4.08%) | **not a regression — see below** |
| Mean classifier latency | 541.846 ms | 569.159 ms | +27.313 ms (+5.04%) | mostly SAM variance |
| **Mean SAM inference** | **308.803 ms** | **337.122 ms** | **+28.319 ms (+9.17%)** | **run-to-run hardware/thermal variance — NanoSAM code and config were not touched between these two sessions** |
| Mean geometry metadata | 86.966 ms | 87.017 ms | +0.051 ms (+0.06%) | **confirms the new profiling instrumentation itself has negligible overhead** |
| Mean frame assignment | 76.349 ms | 73.183 ms | -3.167 ms | run-to-run variance |
| Mean crop maintenance | 1.666 ms | 1.558 ms | -0.108 ms | unchanged, consistent with Part 1 |
| Mean masks/processed frame | 7.726 | 7.680 | -0.046 | comparable — no tracking/mask-cap change |

**Interpretation:** the +24.5ms total-latency increase is almost entirely
explained by the +28.3ms rise in raw `sam_inference_ms` — NanoSAM's own
wall-clock cost on this Jetson varying between two separate `ros2 launch`
invocations (thermal state, background load, TensorRT/CUDA warm-up), not by
any code change in this session (only the profiling sink was added, entirely
outside NanoSAM). This is supported by `geometry_metadata_ms` itself moving
by only 0.051ms despite carrying four new internal timers — direct evidence
that the instrumentation's own overhead is immaterial. The drop in processed
frame count (-3.0%) is a secondary effect of the same SAM variance: with
`request_queue_size: 1` and drop-oldest-when-full, a slower SAM directly
reduces frames processed per unit time.

**Methodological note for future comparisons:** because `sam_inference_ms`
is not perfectly stable run-to-run on this hardware even with zero relevant
code changes, evaluate Part 2 (and later parts') fixes primarily by the
targeted stage's own delta and by `classifier_delay_ms` minus
`sam_delay_ms` (i.e. the non-SAM portion), not by raw `total_delay_ms` alone,
which will always carry some SAM-driven noise.

## Functional / accuracy evidence

- `ObjectGeometryEstimator.estimate()`'s return dict is unchanged — no new
  key, no altered value, verified by inspection of the diff (the `stage_ms`
  parameter is a side-channel dict passed by reference, never read back into
  `geometry`).
- `projection_stride` remained at 1 for this run.
- Persistent tracking, global assignment, crop maintenance, label-map,
  fuser, and Hydra publication logic were not touched.
- Zero processing failures; mean masks/frame comparable to Part 1 close
  (7.680 vs 7.726).
- Full unit test suite (29/29) passed and `colcon build --packages-select rsg`
  succeeded before this run.

## Decision

**Step 1 (profiling) is validated and complete.** The instrumentation adds
negligible overhead and successfully attributes `geometry_metadata_ms`'s
cost: projection (42%) > mask extraction (28%) > stats (21%) > depth
gathering (8%). This session is not itself an accept/reject candidate (no
behavior-changing fix was under test) — the campaign's current point remains
Part 1's close (602.392 ms mean) until Part 2 lands and validates an actual
fix. Proceed to Part 2 Step 2: propose a fix targeting
`geometry_projection_ms` first (largest, and the back-projection matmul has
a known output-identical reordering available — see `../PART2_REPORT.md`).
