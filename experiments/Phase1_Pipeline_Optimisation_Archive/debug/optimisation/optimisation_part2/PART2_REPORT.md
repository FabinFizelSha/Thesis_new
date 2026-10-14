# Phase 1 optimisation Part 2 report

Level 2 of 3 in the campaign documentation hierarchy — see
[../OPTIMISATION_REPORT.md](../OPTIMISATION_REPORT.md) for the whole-campaign
record, global non-negotiable constraints, and the standard test protocol
this part follows. Level 3 (per-session) reports will live in each
`optimisation2_<timestamp>/SESSION_REPORT.md` below as runs complete.

## Objective

Continue minimizing synchronous per-frame Phase 1 latency (callback entry
through Hydra publication) without reducing object-tracking, fuser, RAP, or
VLM accuracy, for exactly one major synchronous section: representative
object geometry estimation. Picks up from Part 1's close — see
`../optimisation_part1/PART1_REPORT.md`.

## Start point

- Inherited from: Part 1 close (`../optimisation_part1/PART1_REPORT.md`
  "End point").
- Reference session: `optimisation_part1/optimisation1_20260806_015126`
  (Part 1 Iteration 1, accepted).
- End-to-end latency mean/median/p95/max: **602.392 / 603.160 / 813.911 /
  1,257.563 ms**.
- Classifier latency mean/median/p95/max: 541.846 / 545.937 / 722.995 /
  1,022.002 ms.
- Selected target: `geometry_metadata_ms` — 86.966 ms mean, 82.380 ms
  median, 150.665 ms p95, 207.999 ms max — the largest eligible synchronous
  leaf remaining after Part 1 closed.
- Runner-up, trending upward across the run (quartile means 37.747 / 67.782 /
  77.452 / 121.910 ms, sequence correlation 0.442): `frame_assignment_ms` —
  76.349 ms mean, 472.879 ms max. Flagged as important but explicitly not the
  Part 2 target; remains the next candidate after Part 2 closes (see
  `../OPTIMISATION_REPORT.md` "Next candidates").
- NanoSAM inference: 308.803 ms mean. Frozen for the whole campaign.

## End point

**Closed — no accepted code change; net latency unchanged from start point.**

- Reference session: `optimisation2_20260806_202114` (revert verification).
- End-to-end latency mean/median/p95/max: 605.903 / 595.679 / 770.376 /
  1,229.409 ms — within normal run-to-run noise of the 602.392 ms start
  point (the difference is attributable to `sam_inference_ms` variance, not
  to Phase 1 code, per the campaign's measurement-caveat note).
- `geometry_metadata_ms`: 85.669 ms, within 1.5% of the 86.966-87.017 ms
  measured at the start point and in Run 1 — confirms zero net change to the
  targeted stage.
- **Summary of this part:** profiled `geometry_metadata_ms` down to four
  sub-steps (Step 1, kept — negligible instrumentation overhead, permanently
  useful diagnostic value). Attempted one fix targeting the largest sub-step
  (`geometry_projection_ms`, a numpy transpose-avoidance reorder); it was
  algebraically correct but measured to regress performance by +18.2% on
  this hardware and was rejected and reverted, confirmed clean by a third
  controlled run. Net effect on campaign latency: **zero** (a correctly
  rejected change, not a loss).
- Remaining sub-steps (`geometry_mask_extract_ms` 28%, `geometry_stats_ms`
  21%, `geometry_depth_gather_ms` 8%) were assessed as either requiring a
  larger-scope interface change or already tried and failed; not pursued
  further in this part.
- **Key methodological lesson carried forward:** a plausible, textbook numpy
  micro-optimization is not reliable on this hardware without a measured
  controlled run — general performance heuristics must be verified, not
  trusted, on this specific Jetson/numpy build. Applied directly to Part 3's
  planning (see `../optimisation_part3/PART3_REPORT.md`): profile before
  proposing, and prefer hypotheses grounded in this codebase's actual
  candidate-count/data-volume behavior over generic "should be faster"
  reasoning.
- Hands off to Part 3 (frame assignment) — see
  `../optimisation_part3/PART3_REPORT.md`.

## Selected Part 2 section

**Representative object geometry estimation** — `ObjectGeometryEstimator`,
`nodes/support/phase1/object_geometry.py`, invoked once per retained SAM mask
from `Phase1SemanticCoordinator.build_object_metadata()` /
`run_rap_and_metadata()` in `nodes/phase1.py`.

## Test protocol

Identical to the standard protocol in `../OPTIMISATION_REPORT.md`, pointed at
this part's own session folder and analyser:

```bash
cd ~/rsg_ros2_ws
mkdir -p ~/rsg_rap_memory
find ~/rsg_rap_memory -mindepth 1 -delete
rm -f ~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl
cd ~/rsg_ros2_ws/src/rsg && python3 -m pytest -q tests -p no:anyio
cd ~/rsg_ros2_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select rsg
source install/setup.bash
ros2 launch rsg rsg_all.launch.py
```

```bash
timeout --signal=INT --kill-after=15s 180s \
  ros2 bag play ~/datasets/uhumans2/office_s1_00h_v2 \
  --rate 1 \
  --qos-profile-overrides-path ~/.tf_overrides.yaml
```

Stop the launch cleanly with Ctrl+C after playback ends. Analyze the newest
session:

```bash
ls -dt debug/optimisation/optimisation_part2/optimisation2_* | head -n 1
python3 debug/optimisation/optimisation_part2/analyse_timing.py \
  debug/optimisation/optimisation_part2/optimisation2_SESSION_ID/phase1_timing.csv
```

`phase1.performance.timing_csv_path` in `config/rsg_pipeline.yaml` already
points new sessions at `optimisation_part2/optimisation2_<timestamp>/`.

## Step 1 — profiling instrumentation (no behavior change, implemented)

Before proposing any fix, `ObjectGeometryEstimator.estimate()` was given an
optional `stage_ms` accumulator (default `None`, so the method's return value
and every existing caller/output is unchanged when not supplied). Phase 1
passes a per-frame dict through when `performance.measure_timing` is `true`
(current config value), attributing time to four sub-steps:

| New column | What it measures |
|---|---|
| `geometry_mask_extract_ms` | `np.where(mask)` + `bbox_2d`/`centroid_2d`/`mask_area_px` |
| `geometry_depth_gather_ms` | stride sampling, depth fancy-indexing, finite/range filter |
| `geometry_projection_ms` | intrinsics back-projection + world transform (`rot_m @ points_cam.T`) |
| `geometry_stats_ms` | centroid/min/max/size/volume + dict assembly |

`geometry_metadata_ms` (the pre-existing aggregate) is still recorded
unchanged by the outer timer in `run_rap_and_metadata`; it should closely
match the sum of the four sub-step columns, modulo per-mask call/dict
overhead. This change touches:

- `nodes/support/phase1/object_geometry.py` — added `stage_ms` parameter and
  timing wraps around the four sub-steps; zero change to any returned
  geometry value or code path taken.
- `nodes/phase1.py` — `build_object_metadata()` accepts and forwards an
  optional `geometry_stage_ms` sink; `run_rap_and_metadata()` creates one
  per-frame dict (only when `timing_enabled`), passes it into every
  `build_object_metadata()` call in the per-mask geometry loop, and folds it
  into the returned stage-ms dict; `_publish_hydra_from_result()` reads the
  four new keys into the `timing_recorder.add_sample(...)` frame-trace row.
- `config/rsg_pipeline.yaml` — `phase1.performance.timing_csv_path` /
  `timing_sheet_name` repointed from `optimisation_part1` to
  `optimisation_part2` so new sessions land in this folder instead of
  appending to the closed Part 1 evidence set.
- `debug/optimisation/optimisation_part2/analyse_timing.py` — forked from
  Part 1's copy (not edited in place; Part 1's tooling/evidence is left
  untouched) with the four new columns added as `leaf` stages excluded from
  "largest bottleneck" candidate selection (they are sub-components of
  `geometry_metadata_ms`, not independent stages — same treatment as the
  existing `sam_prepare_ms`/`sam_inference_ms`/`sam_restore_ms` split).

Verified before requesting a run: `python3 -m pytest -q tests -p no:anyio`
(29/29 pass) and `colcon build --packages-select rsg` (succeeds).

**Result (session `optimisation2_20260806_022259`, full detail in that
session's `SESSION_REPORT.md`):** the four sub-steps sum to 86.152 ms,
within 1.0% of the unchanged outer `geometry_metadata_ms` timer (87.017 ms)
— confirming the instrumentation itself adds negligible overhead. Breakdown:

| Sub-step | Mean ms | % of geometry_metadata_ms |
|---|---:|---:|
| `geometry_projection_ms` (back-projection + world transform) | 36.690 | 42.2% |
| `geometry_mask_extract_ms` (`np.where` + bbox/centroid/area) | 24.135 | 27.7% |
| `geometry_stats_ms` (centroid/min/max/size/volume + dict assembly) | 18.467 | 21.2% |
| `geometry_depth_gather_ms` (stride sampling + validity filter) | 6.860 | 7.9% |

`geometry_projection_ms` dominates. `geometry_depth_gather_ms` — the only
sub-step `projection_stride` actually affects — is the *smallest* of the
four, confirming stride was never the right lever (and it stays fixed at 1
regardless, per this part's constraints). Total end-to-end latency rose
+24.5ms versus the Part 1 close session, but that is attributable to
`sam_inference_ms` (NanoSAM, untouched) varying +28.3ms run-to-run on this
hardware — not to this instrumentation. Full attribution reasoning is in the
session report's "Comparison to previous session" section.

## Step 2 (attempt 1) — projection-formula reorder: REVERTED, regressed performance

Targeting `geometry_projection_ms` first (largest sub-step, 42%):

**Finding:** `object_geometry.py`'s `estimate()` computes
`points_world = (rot_m @ points_cam.T).T + tx.reshape(1, 3)`. This
transposes the (N, 3) points array to (3, N), matrix-multiplies by the (3, 3)
rotation, then transposes the (3, N) result back to (N, 3). The mathematically
identical, algebraically equal computation `points_cam @ rot_m.T` produces
the same (N, 3) result directly, operating on the already-C-contiguous
`points_cam` array without constructing a transposed view/copy in either
direction.

**Proposed change:** replace
`points_world = (rot_m @ points_cam.T).T + tx.reshape(1, 3)`
with
`points_world = points_cam @ rot_m.T + tx.reshape(1, 3)`
in `ObjectGeometryEstimator.estimate()`. This is a pure linear-algebra
reordering — `(A @ B.T).T == B @ A.T` for any conformant matrices — so the
output should be numerically identical up to floating-point associativity
(verifiable with an exact/near-exact equivalence test comparing both forms on
the same random input, the same pattern already used for the Part 1 crop
equivalence test).

**Not proposed for Step 2:** changes to `geometry_mask_extract_ms` (28%) —
the `np.where(mask)` cost appears largely fundamental to the current
dense-boolean-mask interface and any fix would be more invasive (e.g.
requiring SAM to hand back sparse coordinates) — or to
`geometry_depth_gather_ms` (8%, already the smallest, and the only
stride-sensitive part, which is frozen). `geometry_stats_ms` (21%) has a
minor secondary candidate (`.tolist()` instead of the current per-element
`[float(v) for v in centroid]` list comprehensions) worth including in the
same iteration if authorized, since it is in the same function and equally
low-risk.

**Implemented in `nodes/support/phase1/object_geometry.py`:**

```python
# before
points_world = (rot_m @ points_cam.T).T + tx.reshape(1, 3)
# after
points_world = points_cam @ rot_m.T + tx.reshape(1, 3)
```

Bundled in the same change: `geometry_stats_ms`'s four list comprehensions
(`[float(v) for v in centroid]` etc.) replaced with `.tolist()` — same
output type (plain Python floats), one fewer Python-level loop per mask.

**Correctness verification completed before the run:** new
`test_object_geometry.py` proved the rewrite numerically identical to the
original formula and that `estimate()`'s full output was unaffected; 33/33
tests passed; `colcon build` succeeded. The change was correct.

**Controlled-run result (session `optimisation2_20260806_192701`, compared
against `optimisation2_20260806_022259` — full detail in that session's
`SESSION_REPORT.md`): the change measurably REGRESSED performance.**

| Sub-step | Before (Run 1) | After (this attempt) | Change |
|---|---:|---:|---:|
| `geometry_projection_ms` | 36.690 ms | 42.296 ms | **+5.606 ms (+15.3%) — worse** |
| `geometry_stats_ms` | 18.467 ms | 28.276 ms | **+9.809 ms (+53.1%) — worse** |
| `geometry_metadata_ms` total | 87.017 ms | 102.888 ms | **+15.871 ms (+18.2%) — worse** |

Both sub-changes regressed, and the run's `sam_inference_ms` was actually
34.6ms *faster* than Run 1 (the system was under less load, not more), which
rules out adverse system conditions as the explanation. Most likely cause:
for a 3×3 matrix, numpy on this Jetson's build probably does not route
either matmul form through a real BLAS `dgemm` call — the "avoid a
transposed view" argument is a general heuristic that assumes a BLAS
dispatch benefit which may not apply at this matrix size on this platform.
This is exactly the failure mode the controlled-testing protocol exists to
catch: a textbook-plausible micro-optimization that regresses on the actual
target hardware.

**Decision: reject and revert.** Both parts of this change (the projection
reorder and the `.tolist()` swap) were reverted in
`nodes/support/phase1/object_geometry.py`, restoring the exact Step 1
(profiling-only) code. `test_object_geometry.py`'s
`test_transpose_reorder_matches_original_formula` test (specific to this
abandoned rewrite) was removed; the remaining three tests
(`test_estimate_matches_independent_reference`,
`test_estimate_output_types_are_plain_python_floats`,
`test_stage_ms_sink_does_not_change_geometry_output`) stay as a general
`ObjectGeometryEstimator` regression suite, since they're useful
independent of this specific attempt. Post-revert verification: 32/32 tests
pass, `colcon build --packages-select rsg` succeeds.

Part 2 is back at its Step 1 (profiling-complete) state. See "Step 2
(attempt 2)" below for the next candidate.

## Step 2 (attempt 2) — next candidate, not yet started

With the projection-formula reorder ruled out, the remaining sub-steps are:

- `geometry_mask_extract_ms` (28% of cost, 24.1 ms mean) — `np.where(mask)`
  plus bbox/centroid/area. Previously assessed in Step 1 as "largely
  fundamental at the current interface" since the coordinate list it
  produces is also required by `geometry_depth_gather_ms` immediately after.
  Any real fix here likely means a more invasive interface change (e.g. SAM
  handing back sparse coordinates instead of a dense boolean mask), which is
  a bigger, riskier change than this part's remaining scope — needs explicit
  authorization before attempting, and should probably be profiled with a
  microbenchmark in isolation first (outside the full pipeline) given attempt
  1's lesson that plausible numpy micro-optimizations don't always hold on
  this hardware.
- `geometry_stats_ms` (21%, 18.5 ms mean) — the `.tolist()` idea already
  failed; no other candidate identified yet in this block (centroid/min/max
  via `np.median`/`np.min`/`np.max`, already vectorized).
- `geometry_depth_gather_ms` (8%, smallest, stride-frozen) — not worth
  pursuing further given its small share.

**Recommendation:** before attempting another code change, benchmark
candidate rewrites in isolation on this specific Jetson (a small standalone
script timing alternative numpy formulations directly, outside the full
ROS pipeline) rather than reasoning from general numpy performance
heuristics — attempt 1 showed that reasoning alone is not reliable on this
hardware. Given `geometry_mask_extract_ms` and `geometry_stats_ms` together
are only marginally more addressable than `geometry_projection_ms` was, it
may also be reasonable to treat Part 2 as at or near its practical floor at
the per-mask-Python level, and consider closing Part 2 at its Step 1 state
(instrumentation only, no regression) while moving to `frame_assignment_ms`
(the other flagged candidate) instead. This needs your decision before
proceeding.

## Constraints specific to this part

(In addition to the global constraints in `../OPTIMISATION_REPORT.md`.)

- Preserve every existing geometry/tracking output value; profiling-only
  changes must not alter any published field.
- `projection_stride` stays at `1` for this profiling run — any stride change
  is a separate, explicitly authorized step with its own accuracy check.
- Do not change NanoSAM, frame assignment, or result-message construction in
  this part.
- One attributable change per iteration; retain every session folder; never
  overwrite an earlier one.

## Run registry

### Run 1 — profiling (`optimisation2_20260806_022259`)

- Folder: `debug/optimisation/optimisation_part2/optimisation2_20260806_022259/`
- Purpose: attribute `geometry_metadata_ms`'s cost across four sub-steps
  before proposing any fix.
- Received/processed/dropped/failed: 2,928 / 294 / 2,635 / 0.
- Total latency mean/median/p95/max: 626.940 / 612.556 / 800.173 /
  1,203.036 ms. Classifier mean: 569.159 ms.
- Sub-step breakdown: `geometry_projection_ms` 36.690 ms (42.2%),
  `geometry_mask_extract_ms` 24.135 ms (27.7%), `geometry_stats_ms`
  18.467 ms (21.2%), `geometry_depth_gather_ms` 6.860 ms (7.9%). Sum
  86.152 ms vs. unchanged outer `geometry_metadata_ms` 87.017 ms (1.0% call
  overhead).
- Verification: 29/29 tests pass; `colcon build` succeeds; instrumentation
  changes no returned geometry value (side-channel dict only).
- Comparison to Part 1 close: total latency +24.5ms, attributed to
  `sam_inference_ms` (NanoSAM, untouched) varying +28.3ms run-to-run on this
  hardware, not to this session's code change — `geometry_metadata_ms`
  itself moved only +0.051ms, confirming negligible instrumentation
  overhead. Full table in this session's `SESSION_REPORT.md`.
- Decision: **Step 1 accepted as profiling evidence.** Proceed to Step 2 (see
  above) pending authorization to implement the proposed
  `geometry_projection_ms` fix.

### Run 2 — Step 2 attempt 1: projection-formula reorder (`optimisation2_20260806_192701`) — REVERTED

- Folder: `debug/optimisation/optimisation_part2/optimisation2_20260806_192701/`
- Purpose: measure the effect of the `points_cam @ rot_m.T` reorder and the
  `.tolist()` swap against Run 1.
- Received/processed/dropped/failed: 2,895 / 308 / 2,587 / 0.
- Mean masks/processed frame: 7.679 (vs. Run 1's 7.680 — comparable scene
  content, not a confound).
- Result: `geometry_projection_ms` +5.606 ms (+15.3%), `geometry_stats_ms`
  +9.809 ms (+53.1%), `geometry_metadata_ms` total +15.871 ms (+18.2%) — all
  **worse**, while `sam_inference_ms` was 34.6ms *faster* than Run 1 (system
  was under less load, ruling out an adverse-conditions explanation). Full
  table in this session's `SESSION_REPORT.md`.
- Verification: correctness confirmed by `test_object_geometry.py` before
  the run (33/33 tests, `colcon build` succeeds) — this was a genuine
  performance regression, not a correctness bug.
- Decision: **Reject and revert.** Both parts of the change reverted in
  `object_geometry.py`; the now-obsolete formula-equivalence test removed
  from `test_object_geometry.py`. Post-revert: 32/32 tests pass, `colcon
  build` succeeds. Part 2 returns to its Step 1 (profiling-complete) state —
  see "Step 2 (attempt 2)" above for the next candidate, pending your
  direction.

### Run 3 — revert verification (`optimisation2_20260806_202114`)

- Folder: `debug/optimisation/optimisation_part2/optimisation2_20260806_202114/`
- Purpose: confirm the revert actually restored pre-attempt-1 performance.
- Received/processed/dropped/failed: 2,906 / 304 / 2,603 / 0. Mean
  masks/processed frame: 7.671 (comparable to Runs 1 and 2).
- Result: `geometry_metadata_ms` 85.669 ms — within 1.5% of Run 1's
  87.017 ms (noise), and 17.2 ms below Run 2's regressed 102.888 ms. All four
  sub-steps individually confirmed back to Run-1-comparable levels. Full
  three-way table in this session's `SESSION_REPORT.md`.
- Decision: **Revert confirmed clean.** Part 2 verified back at its Step 1
  state with zero residual regression.
