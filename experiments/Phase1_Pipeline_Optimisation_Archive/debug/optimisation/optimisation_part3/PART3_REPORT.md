# Phase 1 optimisation Part 3 report

Level 2 of 3 in the campaign documentation hierarchy — see
[../OPTIMISATION_REPORT.md](../OPTIMISATION_REPORT.md) for the whole-campaign
record, global non-negotiable constraints, and the standard test protocol
this part follows. Level 3 (per-session) reports will live in each
`optimisation3_<timestamp>/SESSION_REPORT.md` below as runs complete.

## Objective

Continue minimizing synchronous per-frame Phase 1 latency (callback entry
through Hydra publication) without reducing object-tracking, fuser, RAP, or
VLM accuracy, for exactly one major synchronous section: persistent-track
frame assignment. Picks up from Part 2's close — see
`../optimisation_part2/PART2_REPORT.md`.

## Start point

- Inherited from: Part 2 close (`../optimisation_part2/PART2_REPORT.md` "End
  point") — Part 2 accepted no code change, so this is numerically identical
  to Part 1's close.
- Reference session: `optimisation_part1/optimisation1_20260806_015126`
  (Part 1 Iteration 1, the last accepted change).
- End-to-end latency mean/median/p95/max: **602.392 / 603.160 / 813.911 /
  1,257.563 ms**.
- Selected target: `frame_assignment_ms` — 76.349 ms mean at Part 1 close
  (69.2-73.5 ms across the three Part 2 sessions, consistent with normal
  run-to-run variance), but with a documented **growth pattern within a
  single run**: quartile means 37.7 / 67.8 / 77.5 / 121.9 ms, sequence
  correlation 0.442 (measured at Part 1 close). A flat whole-run mean
  understates this — the growth trend is the real signal being targeted.
- NanoSAM inference: ~310-340 ms mean (run-to-run variance already
  characterized in Part 2). Frozen for the whole campaign.

## End point

**Pending.** Profiling instrumentation implemented and verified (see "Step
1" below); no controlled run yet.

## Selected Part 3 section

**Persistent-track frame assignment** —
`PersistentObjectTracker.prepare_frame_assignments()` in
`nodes/support/phase1/persistent_object_tracker.py`, called once per frame
from `Phase1SemanticCoordinator.run_rap_and_metadata()` in `nodes/phase1.py`.
This function implements three algorithmically distinct passes, documented
in `docs/TRACK_A2_GLOBAL_ASSIGNMENT.md` as deliberate correctness features:

- **A2 — track-aware union redundancy**: for every pair of retained SAM
  masks, computes full-resolution boolean-mask intersection
  (`np.count_nonzero(child & large)`) to detect when several smaller masks
  jointly explain one enclosing mask, so the enclosing mask can be suppressed
  in favor of the decomposition. O(n²) mask-pair operations for n masks in
  the frame (n ≤ 8 today, so ≤ 28 pairs).
- **A3 — same-track nested-duplicate suppression**: a second, similar O(n²)
  mask-pair pass, resolving cases where two masks (one nested in the other)
  both prefer the same established track.
- **E — global one-to-one assignment**: for every retained observation,
  `_find_match()` scores every *candidate* track (see below) using a
  multi-cue weighted evaluation (historical/recent 3D overlap, centroid
  distance, vertical compatibility, 2D IoU fallback), then a Hungarian
  solver (`_hungarian_maximize`, a hand-written O(rows² × cols) pure-Python
  implementation) picks the maximum-utility one-to-one assignment across all
  observations and candidate tracks simultaneously, so no single mask gets
  first-come-first-served priority.

**Important prior-art correction:** before reading this code, my working
assumption (recorded in earlier campaign notes) was that `_find_match`
brute-force-scans every active track. That was wrong. The tracker already
maintains a spatial hash index (`_spatial_bbox_cells`, `_spatial_centroid_cells`,
built by `_refresh_spatial_index`) and `_candidate_track_ids()` uses it to
restrict `_find_match`'s per-observation candidate set to only tracks whose
stored bbox/centroid cells overlap the current observation's — this is
already a real, working optimization, not something to (re)build from
scratch. There is also a separate, empty, abandoned exploration of this
exact idea from before this documentation hierarchy existed:
`debug/optimization/persistent_tracker_optimisation/candidate_index_compact_evaluation/`
— every file in it (reports and CSVs alike) is 0 bytes, so it provides no
usable prior conclusion, but the folder name suggests someone had already
identified this as the area to investigate. Given the spatial index already
exists and works, the question for Part 3 is not "does a candidate index
exist" but "why does the effective candidate count still grow as a run
progresses, and where exactly does the remaining cost go."

## Hypothesis (grounded in the code, to be tested by profiling, not assumed)

Because `frame_assignment_ms` grows within a single run rather than staying
flat, and A2/A3's mask-pair cost is a function of *masks per frame*
(empirically flat at ~7.68 masks/frame across every session so far, not
growing), the A2/A3 passes are unlikely to be the source of the growth
trend. The more likely source is `_find_match`'s per-observation candidate
search: `_candidate_track_ids()` returns every track whose spatial cell
overlaps the current observation, and as more of the scene gets explored and
more tracks accumulate in previously-visited (revisited or nearby) areas,
that candidate set can grow even though the index itself is working exactly
as designed — the *local track density* in the currently-visible region
increases over a run, not just the *total* track count. Each candidate then
gets a full evaluation row built in `_find_match` (multiple dict keys,
`_as_list()` numpy-to-list conversions of centroid/bbox fields, several
geometric computations), so cost scales with candidate count per
observation, which this hypothesis predicts grows over the run.

This is deliberately framed as a hypothesis requiring measurement, not a
conclusion — Part 2's Step 2 attempt showed plausible code-level reasoning
about performance is not reliable on this hardware without a controlled run.

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
ls -dt debug/optimisation/optimisation_part3/optimisation3_* | head -n 1
python3 debug/optimisation/optimisation_part3/analyse_timing.py \
  debug/optimisation/optimisation_part3/optimisation3_SESSION_ID/phase1_timing.csv
```

`phase1.performance.timing_csv_path` in `config/rsg_pipeline.yaml` already
points new sessions at `optimisation_part3/optimisation3_<timestamp>/`.

## Step 1 — profiling instrumentation (no behavior change, implemented)

`PersistentObjectTracker.prepare_frame_assignments()` now accepts an
optional `stage_ms` accumulator (default `None`, so the method's return
value — the keep-mask — and every existing side effect (forced-match
installation) are unchanged when not supplied). Phase 1 passes a per-frame
dict through when `performance.measure_timing` is `true` (current config
value), attributing time to four sub-steps plus two non-timing diagnostic
counts:

| New column | What it measures |
|---|---|
| `assignment_candidate_search_ms` | the per-observation `_find_match()` loop — candidate lookup via the spatial index plus per-candidate multi-cue scoring |
| `assignment_a2_redundancy_ms` | the A2 union-redundancy mask-pair pass |
| `assignment_a3_nested_ms` | the A3 same-track nested-duplicate mask-pair pass |
| `assignment_hungarian_ms` | building the assignment weight matrix, the Hungarian solve, and committing forced matches |
| `assignment_candidate_count_total` (non-timing) | sum, across every observation in the frame, of how many candidate tracks `_find_match` evaluated — tests the density-growth hypothesis directly |
| `assignment_candidate_count_max` (non-timing) | the largest single-observation candidate count in the frame |

`frame_assignment_ms` (the pre-existing aggregate, timed in `phase1.py`
around the whole `prepare_frame_assignments()` call) is unchanged and should
closely match the sum of the three timing sub-steps (candidate search + A2 +
A3 + Hungarian), modulo lock-acquisition and per-call overhead not
attributed to any sub-step.

This change touches:

- `nodes/support/phase1/persistent_object_tracker.py` — added `stage_ms`
  parameter to `prepare_frame_assignments()`; timing wraps around the
  candidate-search loop, the A2 block, the A3 block, and the
  Hungarian/commit block (including both the early-return-when-nothing-
  retained path and the normal path); a running candidate-count tally
  during the candidate-search loop. Zero change to the returned keep-mask or
  any track/forced-match state.
- `nodes/phase1.py` — `run_rap_and_metadata()` creates one per-frame dict
  (only when `timing_enabled`), passes it into `prepare_frame_assignments()`,
  and folds it into the returned stage-ms dict; `_publish_hydra_from_result()`
  reads the six new keys into the `timing_recorder.add_sample(...)`
  frame-trace row.
- `config/rsg_pipeline.yaml` — `phase1.performance.timing_csv_path` /
  `timing_sheet_name` repointed from `optimisation_part2` to
  `optimisation_part3`.
- `debug/optimisation/optimisation_part3/analyse_timing.py` — forked from
  Part 2's copy (not edited in place) with the four new timing columns added
  as `leaf` stages excluded from "largest bottleneck" candidate selection
  (sub-components of `frame_assignment_ms`, same treatment as the existing
  `geometry_*`/`sam_*` splits), plus a new quartile-by-arrival-order
  breakdown of `frame_assignment_ms` and both candidate-count columns in the
  generated `stage_summary.md`, automating the growth-trend check that was
  previously done by hand at Part 1's close.
- `tests/test_global_frame_assignment.py` — new
  `test_prepare_frame_assignments_stage_ms_sink_does_not_change_keep_mask`,
  proving the profiling sink has zero effect on the returned keep-mask or
  forced-match assignment, reusing the file's existing tracker/config
  fixtures.

Verified: `python3 -m pytest -q tests -p no:anyio` → **33/33 pass** (32
pre-existing + 1 new); `colcon build --packages-select rsg` → succeeds;
`git status` confirms only `persistent_object_tracker.py`, `phase1.py`,
`rsg_pipeline.yaml`, and `test_global_frame_assignment.py` changed for this
step.

## Constraints specific to this part

(In addition to the global constraints in `../OPTIMISATION_REPORT.md`.)

- The A2/A3/E assignment **algorithm** (`docs/TRACK_A2_GLOBAL_ASSIGNMENT.md`)
  is a deliberate correctness feature, not incidental cost — this part
  targets computational efficiency only (redundant work, data-structure
  overhead, candidate-set sizing), never matching thresholds, route
  priorities, or which track wins an assignment.
- Any fix must be proven with an output-equivalence test (identical keep-mask
  and forced-match assignment for the same input) before being proposed as
  more than a profiling change — same bar Part 1 and Part 2 held.
- Given Part 2's lesson, no fix proceeds on code-level reasoning alone —
  every candidate change gets a controlled before/after run.
- Do not change NanoSAM, geometry estimation, or crop maintenance in this
  part.

## Step 2 — planning (not yet implemented, pending your direction)

Run 1 confirmed the hypothesis cleanly: `assignment_candidate_search_ms` is
82.2% of `frame_assignment_ms`, and candidate count grows 6.4x (total) /
9.2x (max, reaching ~303 candidates for a single observation by the run's
last quartile) — directly explaining the previously-observed growth trend.
A2/A3/Hungarian are minor (17.8% combined) despite looking algorithmically
heavier on paper.

**Investigated one specific lead:** `_find_match`'s per-candidate loop
(`persistent_object_tracker.py` ~line 1500) eagerly builds a row dict with
five `_as_list()` numpy-to-Python-list conversions
(`candidate_centroid_3d`, `candidate_bbox_3d_min/max`,
`candidate_last_bbox_3d_min/max`) for *every* candidate, before either cheap
gate (`respect_frame_used`, `known_label_mismatch`) is checked — a plausible
"defer the expensive part" target. **However**, tracing where these fields
go: `_find_match`'s full evaluation list flows into `associate()`'s
`record["candidate_evaluations"]`, into `track_records`, into
`build_result_metadata()`'s `metadata["unknown_track_records"]`, into
`result.metadata_json`, into `hydra_msg.perception_metadata_json` — **this
is a published field**, not dead code. Grep confirms these five keys are
never *read* anywhere in `phase1.py` or `persistent_object_tracker.py`
itself, but I have not yet checked whether any other package in this
workspace (`risk_scene_graph_core`, `risk_scene_graph_ros`,
`rsg_dsg_visualizer_package`, `rsg_semantic_adapter`) parses
`perception_metadata_json` and depends on these fields existing. This is
exactly the kind of "looks obviously safe, verify before touching a
published field" case the campaign's rules exist for — not proceeding on
the assumption alone.

**Two candidate directions, not yet chosen:**

1. **Reduce per-candidate cost** (attack the 82%'s constant factor): defer
   the five `_as_list()` conversions until a candidate passes the cheap
   gates (most candidates that fail `respect_frame_used`/`known_label_mismatch`
   never need them), or keep them but confirm via cross-package grep that
   no consumer needs them and drop them entirely. Lower algorithmic risk,
   needs the cross-package verification step first.
2. **Reduce candidate count** (attack the 6-9x growth directly): tighten
   `_candidate_track_ids()`'s spatial-cell padding or add a cheap
   centroid-distance-squared pre-filter before entering the expensive
   per-candidate loop. Higher-leverage (candidate count is the actual growth
   driver) but higher risk — this is exactly the kind of change that could
   silently exclude a legitimate revisit match if the tightened radius is
   even slightly wrong, which is the one thing this campaign is most
   forbidden from doing. Would need extensive equivalence testing against
   the existing revisit/continuation test suite in
   `tests/test_global_frame_assignment.py` before being considered.

**Finer profiling implemented (this step).** `_find_match()` now accepts the
same optional `stage_ms` sink, splitting its per-candidate work (accumulated
across every candidate evaluated by all `_find_match` calls in the frame)
into four further sub-steps:

| New column | What it measures |
|---|---|
| `assignment_row_init_ms` | dict construction (5 `_as_list()` numpy→list conversions) plus the two cheap gates (`respect_frame_used`, `known_label_mismatch`), for every candidate including quickly-rejected ones |
| `assignment_3d_geometry_ms` | the two AABB blocks — historical footprint vs. `track.bbox_3d_*` and recent continuation vs. `track.last_bbox_3d_*` (10 geometric function calls per candidate with 3D data) |
| `assignment_centroid_iou_ms` | centroid-distance + 2D IoU scoring |
| `assignment_scoring_ms` | the weighted multi-cue combination and `consider()` calls (both the `global_enabled=False` fallback path and the main quorum-gated path) |

These four should sum close to `assignment_candidate_search_ms` (the
existing Step 1 aggregate for this section), same reconciliation pattern as
Part 2. Implementation: `nodes/support/phase1/persistent_object_tracker.py`
— `_find_match()` gained the `stage_ms` parameter and per-candidate timer
wraps around each of the four sections (handling both early-continue paths
so quickly-rejected candidates are still attributed correctly to
`assignment_row_init_ms`); `prepare_frame_assignments()` forwards its own
`stage_ms` sink into every `_find_match()` call. `nodes/phase1.py` reads the
four new keys into the frame-trace CSV row.
`debug/optimisation/optimisation_part3/analyse_timing.py` updated with the
four new leaf stages (excluded from candidate selection, same treatment as
every other sub-step split). New tests
`test_find_match_stage_ms_sink_does_not_change_result` and updates to
`test_prepare_frame_assignments_stage_ms_sink_does_not_change_keep_mask`
prove zero effect on `_find_match`'s returned match/score/evaluations or
`prepare_frame_assignments`'s keep-mask.

Verified: `python3 -m pytest -q tests -p no:anyio` → **34/34 pass**;
`colcon build --packages-select rsg` → succeeds; `git status` confirms only
`persistent_object_tracker.py`, `phase1.py`, `analyse_timing.py`, and
`test_global_frame_assignment.py` changed for this step.

**Result (session `optimisation3_20260806_205816`, full detail in that
session's `SESSION_REPORT.md`):**

| Sub-step | Mean ms | % of `assignment_candidate_search_ms` | Max ms |
|---|---:|---:|---:|
| `assignment_3d_geometry_ms` | 20.451 | 35.4% | 361.938 |
| `assignment_row_init_ms` | 12.531 | 21.7% | 351.020 |
| `assignment_centroid_iou_ms` | 11.176 | 19.3% | 45.418 |
| `assignment_scoring_ms` | 7.115 | 12.3% | 34.172 |
| Sum | 51.273 | 88.8% | — |
| `assignment_candidate_search_ms` (outer) | 57.763 | 100% | 425.805 |

**No single sub-step dominates** (35/22/19/12% split) — unlike Part 2's
geometry work, there is no one line to fix here. The remaining ~11% gap
is attributed to `_candidate_track_ids()`'s spatial-index lookup itself,
called once per `_find_match` invocation, outside all four timed sections.

Candidate-count growth was **reconfirmed on this second, independent run**
(6.6x total / 9.8x max, Q1→Q4 — closely matching Run 1's 6.4x/9.2x),
strengthening confidence this is a real, reproducible effect, not noise.

**New finding, not predicted by the growth hypothesis:** severe tail
spikes (300+ ms, 15-28x the relevant stage's mean) on
`assignment_3d_geometry_ms`, `assignment_row_init_ms` (four separate
frames, not just one), and even `track_association_ms` (a normally
near-zero-cost stage). These don't co-occur on the same frame across
stages and are far larger than gradual per-frame growth would produce.
Leading candidate explanation: lock contention with the async RAP/VLM
worker threads on `PersistentObjectTracker._lock`, which every one of
these calls holds for its full body — not yet confirmed, since the current
instrumentation can't distinguish "blocked waiting for the lock" from
"doing real work after acquiring it."

## Step 3 — two distinct paths forward, pending your decision

**Path A — attack the mean-cost growth (candidate-count reduction).**
Since no single sub-step dominates the per-candidate cost, the highest-
leverage fix is reducing *how many* candidates get evaluated at all
(benefits all four sub-steps and the spatial lookup simultaneously),
rather than optimizing any one of the four sub-steps individually. Concrete
lever: tighten `_candidate_track_ids()`'s spatial-cell padding radii
(currently unions bbox-footprint cells, centroid cells, and a padding
radius up to `max(continuation_gap_m, revisit_overlap_gap_m,
global_centroid_pass_m)` ≈ 0.75m). **Risk:** this is exactly the kind of
change that could silently exclude a legitimate revisit match if the
radius is tightened even slightly too far — the single thing this campaign
is most forbidden from doing. Would require extensive equivalence testing
against every existing test in `tests/test_global_frame_assignment.py`
(which specifically tests revisit/continuation/contradiction scenarios)
before being considered, and ideally a way to measure whether any
real match was lost on the actual bag, not just synthetic unit tests.

**Path B — investigate the tail-latency spikes (lock contention).** Add
explicit lock-wait timing (time blocked acquiring
`PersistentObjectTracker._lock`, separate from time spent working after
acquiring it) to confirm or rule out contention with the RAP/VLM threads.
If confirmed, the fix is narrowing lock granularity (e.g., a separate,
cheaper lock for read-only state checks like `is_semantic_labeling_open`
that doesn't block on whatever the RAP/VLM thread is doing with the main
tracker lock) rather than touching the assignment algorithm at all. Lower
risk to tracking correctness (a locking refactor, not a matching-threshold
change) but is concurrency-sensitive code, needing careful review of every
one of the tracker's 11 `self._lock` call sites.

These are independent — Path A targets the mean, Path B targets the tail
(p95/max), and both could eventually be pursued, but not simultaneously
(one major change per part/step).

## Step 4 — cheap-early-filter attempt: correctness risk found, pivoted to a safe subset

Before implementing "reduce candidate count via an early cheap exact-distance
check" (a lower-risk-sounding variant of Path A, discussed at length before
implementation), tracing `_find_match`'s actual scoring logic surfaced a real
correctness hazard that would have made a naive version of that filter unsafe:

`votes = {"footprint": historical_pass or recent_pass, "centroid": centroid_pass,
"vertical": vertical_pass, "image": image_pass}`, and a candidate is accepted
once `pass_count >= persistent_global_min_independent_groups` (default 2) of
these vote **independently**. Critically, `vertical_pass` is computed purely
from z-axis gap/score (`_aabb_gap_z`, vertical `_gaussian_compatibility`),
entirely independent of XY distance — and `image_pass` (2D bbox IoU) does not
require 3D proximity either. So a candidate that is arbitrarily far away in
3D XY space can still legitimately pass via `{vertical, image}` if its height
matches and its 2D projection happens to overlap (exactly the "3D depth was
noisy this frame but the object is still the same one" case this quorum
system exists to handle robustly). An early filter that rejects candidates
based on XY distance alone — before the vote-counting logic runs — would
silently break this specific, intentional matching path. Verifying this
required tracing the full scoring function; it would not have been obvious
from the profiling data alone, and would have been exactly the kind of
subtle regression that's hard to catch with synthetic unit tests unless the
specific edge case (XY-far, Z-close, high 2D IoU) is deliberately
constructed. **Not implemented for this reason.**

**What was implemented instead:** the row-construction overhead identified
back in Step 1/2's finer profiling (`assignment_row_init_ms`, 21.7% of
`assignment_candidate_search_ms` in Run 2). `_find_match`'s per-candidate row
dict was eagerly computing five numpy→list conversions
(`candidate_centroid_3d`, `candidate_bbox_3d_min`, `candidate_bbox_3d_max`,
`candidate_last_bbox_3d_min`, `candidate_last_bbox_3d_max`) for *every*
candidate, unconditionally, regardless of whether that candidate would ever
be accepted. Before removing them, confirmed via `grep` across every package
in this workspace (`risk_scene_graph_core`, `risk_scene_graph_ros`,
`rsg_dsg_visualizer_package`, `rsg_semantic_adapter`, `hydra`, `hydra_ros`,
plus `nodes/phase1.py` and `persistent_object_tracker.py` itself) that none
of these five keys are read anywhere — only the field-type declaration in
`msg/RsgHydraFrame.msg` matched, which is unrelated. This is different from
the Part 2 Step 2 case (where similar-looking "unused" fields turned out to
be threaded into a published field) — here, the cross-package search came
back clean, so removal carries no published-payload risk. **Zero effect on
any matching decision** — these fields were never inputs to `historical_pass`,
`recent_pass`, `centroid_pass`, `vertical_pass`, `image_pass`, or the
weighted score; they were purely descriptive dead weight in the row dict.

**Implemented in `nodes/support/phase1/persistent_object_tracker.py`:**
removed the five `_as_list()` calls from the per-candidate row construction
inside `_find_match()`. The `_as_list()` helper itself is untouched (still
used elsewhere, e.g. published segment/track records at lines ~2005-2074,
which are real outputs, not per-candidate search scratch data).

**Verified:** `python3 -m pytest -q tests -p no:anyio` → **34/34 pass**
(no test referenced the removed keys); `colcon build --packages-select rsg`
→ succeeds; `git status` confirms only `persistent_object_tracker.py`
changed for this step.

**Result (session `optimisation3_20260806_220604`, full detail in that
session's `SESSION_REPORT.md`) — ACCEPTED:**

| Metric | Run 2 (before) | Run 3 (after) | Change |
|---|---:|---:|---:|
| `assignment_row_init_ms` mean | 12.531 ms | 3.674 ms | **-70.7%** |
| `assignment_candidate_search_ms` mean | 57.763 ms | 51.551 ms | **-10.8%** |
| `frame_assignment_ms` mean | 70.363 ms | 64.016 ms | **-9.0%** |
| `sam_inference_ms` mean (context) | 315.190 ms | 316.303 ms | +0.35% (negligible — unusually clean comparison) |
| non-SAM portion (`classifier - sam`) | 224.438 ms | 213.766 ms | **-4.8%** |

`sam_inference_ms` barely differed between these two runs, so this
comparison is one of the least noise-confounded in the campaign — the
delta is a fair read of this change's real effect. Zero test regressions
(34/34), zero effect on any matching/scoring decision (verified — none of
the removed fields were ever inputs to `historical_pass`/`recent_pass`/
`centroid_pass`/`vertical_pass`/`image_pass`/the weighted score).

**Bonus finding supporting Path B:** `assignment_row_init_ms`'s *mean* fell
70.7%, but its *max* barely moved (351.020 → 317.840 ms, -9.5%). Cutting the
computation cost by 70% without a comparable cut in the worst case is strong
indirect evidence the scattered tail spikes are not computational — they're
caused by something external to this code, consistent with the
lock-contention hypothesis (still not directly confirmed).

**Decision: accepted and kept.** This is the first real, measured
improvement landed in Part 3.

## Step 5 — Path B: lock-wait instrumentation (no behavior change, implemented)

Enumerated all 11 `self._lock` acquisition sites in
`PersistentObjectTracker` and mapped each to its calling thread:
`begin_frame`/`associate`/`prepare_frame_assignments` (and, via
`_remember_track_crop`/`_dispatch_tracks_after_settling`,
`is_semantic_labeling_open`/`prepare_active_for_labeling`/
`release_labeling_request`/`set_labeling_status`) run on the **main
classification thread**; `apply_rap_result`/`apply_vlm_result`/
`complete_semantic_labeling` and further `set_labeling_status` calls run on
the **RAP and VLM background worker threads** (`_rap_loop`/`_vlm_loop` in
`nodes/phase1.py`). All of these share the single `self._lock` — real
contention between the main thread and either background thread is
structurally possible.

Scoped this step to the two call sites directly implicated by the observed
tail spikes: `prepare_frame_assignments()` (source of the
`assignment_row_init_ms`/`assignment_3d_geometry_ms` spikes) and
`associate()` (source of the `track_association_ms` spike — a stage whose
compute is normally near-zero once `prepare_frame_assignments` has already
installed forced matches, making lock-wait the most direct explanation for
any large value there). `begin_frame()` and the crop/labeling-dispatch call
sites are not instrumented in this step — deferred if this evidence
warrants a fuller investigation.

**Implementation:** in both methods, `with self._lock:` was replaced with
the behavior-identical expansion (`self._lock.acquire()` timed, then
`try: ... finally: self._lock.release()`), which is exactly what a `with`
statement does under the hood for a `Lock` — this substitution changes
nothing about locking semantics, only adds a timestamp around the
`acquire()` call. New profiling fields: `assignment_lock_wait_ms` (time
`prepare_frame_assignments` blocks acquiring the lock, measured *before*
any of Step 1/2's sub-steps run) and `association_lock_wait_ms` (same, for
`associate()`, threaded through a new optional `stage_ms` parameter from
`nodes/phase1.py`'s per-mask loop in `run_rap_and_metadata`).

**Touches:**
- `nodes/support/phase1/persistent_object_tracker.py` — lock-wait timing in
  `prepare_frame_assignments()` and `associate()` (new `stage_ms` parameter
  on `associate()`); no change to any lock acquisition order, hold
  duration, or critical-section logic.
- `nodes/phase1.py` — `association_stage_ms` per-frame accumulator dict
  (same pattern as `geometry_stage_ms`/`assignment_stage_ms`), threaded
  into every `associate()` call in the per-mask loop; two new fields
  (`assignment_lock_wait_ms`, `association_lock_wait_ms`) read into the
  frame-trace CSV row.
- `debug/optimisation/optimisation_part3/analyse_timing.py` — two new leaf
  columns, excluded from candidate selection (sub-components of
  `frame_assignment_ms`/`track_association_ms`).
- `tests/test_global_frame_assignment.py` — extended the existing
  `prepare_frame_assignments` stage_ms test with the new key; added
  `test_associate_stage_ms_sink_does_not_change_result`, proving the sink
  has zero effect on `associate()`'s returned metadata/track record.

**Verified:** `python3 -m pytest -q tests -p no:anyio` → **35/35 pass** (34
pre-existing + 1 new); `colcon build --packages-select rsg` → succeeds;
`git status` confirms only `persistent_object_tracker.py`, `phase1.py`,
`analyse_timing.py`, and `test_global_frame_assignment.py` changed.

**Result (session `optimisation3_20260806_222959`, full detail in that
session's `SESSION_REPORT.md`) — CONCLUSIVE, lock contention ruled out:**

`assignment_lock_wait_ms` (mean 0.010ms, max 1.369ms across 277 frames) and
`association_lock_wait_ms` (mean 0.051ms, max 3.939ms) are both negligible
— nowhere near the spikes under investigation. This session reproduced a
severe spike (`frame_assignment_ms` = 335.660ms, sequence 1954), and on
that *exact* frame lock-wait was still tiny (0.005ms / 0.334ms) while the
four instrumented candidate-loop sub-steps summed to a completely ordinary
24.155ms. The gap — `assignment_candidate_search_ms` (314.699ms) minus the
four sub-steps (24.155ms) = **290.544ms** — has nowhere else to be but
inside `_candidate_track_ids()` itself (called 7 times this frame, once per
observation, outside every timed section). This is frame-level evidence,
not a statistical correlation: on this specific frame, the only unexplained
cost sits exactly where the earlier-flagged full-track scan lives.

**Path B closed.** Lock contention is ruled out for these two call sites.
The tail-spike investigation redirects to `_candidate_track_ids()`'s final
line:
```python
return [track_id for track_id in self._tracks if track_id in candidate_ids]
```
which scans every track in `self._tracks` (not just the candidate subset)
to preserve dict-insertion order, likely for `_find_match`'s tie-breaking
determinism. This is the same finding flagged when Path B was chosen over
it — now with direct frame-level evidence it's the real cause of at least
the most severe outlier, not merely a plausible hypothesis. **Next:**
resolve whether that iteration order is actually load-bearing (does
`_find_match`'s `consider()` tie-breaking, or the Hungarian weight-matrix
row/column order, depend on it) before attempting any fix — same discipline
every prior change in this campaign has followed.

## Run registry

### Run 1 — profiling (`optimisation3_20260806_204104`)

- Folder: `debug/optimisation/optimisation_part3/optimisation3_20260806_204104/`
- Purpose: attribute `frame_assignment_ms`'s cost and test the candidate-
  count growth hypothesis.
- Received/processed/dropped/failed: 2,925 / 301 / 2,624 / 0. Mean
  masks/processed frame: 7.674 (consistent with every prior session).
- Sub-step breakdown: `assignment_candidate_search_ms` 58.625 ms (82.2%),
  `assignment_a2_redundancy_ms` 6.796 ms (9.5%), `assignment_a3_nested_ms`
  3.685 ms (5.2%), `assignment_hungarian_ms` 2.111 ms (3.0%). Sum 71.217 ms
  vs. unchanged outer `frame_assignment_ms` 71.308 ms (0.13% call overhead).
- Growth confirmation: candidate count (total/frame) 165.6 → 529.9 → 673.0 →
  1065.1 across quartiles (6.4x); candidate count (max/observation) 33.1 →
  130.3 → 233.6 → 302.8 (9.2x). `frame_assignment_ms` tracks this (36.4 →
  74.8 → 65.8 → 107.8 ms, ~3x, sublinear vs. candidate growth).
- Verification: 33/33 tests pass; `colcon build` succeeds; instrumentation
  changes no returned keep-mask or forced-match value (side-channel dict
  only).
- Decision: **Step 1 accepted as profiling evidence; hypothesis confirmed.**
  Proceed to the finer per-candidate-loop profiling split (Run 2).

### Run 2 — finer profiling (`optimisation3_20260806_205816`)

- Folder: `debug/optimisation/optimisation_part3/optimisation3_20260806_205816/`
- Purpose: split `assignment_candidate_search_ms` into row-construction /
  3D-geometry / centroid-IoU / scoring, and re-check the growth hypothesis
  on a second, independent run.
- Received/processed/dropped/failed: 2,517 / 273 / 2,244 / 0. Mean
  masks/processed frame: 7.722 (consistent with every prior session).
- Sub-step breakdown: `assignment_3d_geometry_ms` 20.451 ms (35.4%),
  `assignment_row_init_ms` 12.531 ms (21.7%), `assignment_centroid_iou_ms`
  11.176 ms (19.3%), `assignment_scoring_ms` 7.115 ms (12.3%). Sum
  51.273 ms vs. outer `assignment_candidate_search_ms` 57.763 ms — ~11.2%
  gap attributed to `_candidate_track_ids()`'s spatial lookup, not covered
  by any of the four timers.
- Growth reconfirmed: candidate count 6.6x (total) / 9.8x (max) Q1→Q4,
  closely matching Run 1's 6.4x/9.2x.
- New finding: severe scattered tail spikes (300+ ms, 15-28x mean) on
  `assignment_3d_geometry_ms`, `assignment_row_init_ms` (4 separate
  frames), and `track_association_ms` — not explained by gradual
  candidate-count growth; leading hypothesis is lock contention with
  async RAP/VLM threads, not yet confirmed.
- Verification: 34/34 tests pass; `colcon build` succeeds; two new
  equivalence tests prove zero effect on `_find_match`'s output.
- Decision: **Finer profiling validated.** Two independent findings (mean
  growth, well-attributed but no single dominant sub-step; tail spikes,
  unexplained by growth alone) — see "Step 3" above for both candidate
  paths forward, pending your decision on which to pursue.

### Run 3 — dead-weight removal, ACCEPTED (`optimisation3_20260806_220604`)

- Folder: `debug/optimisation/optimisation_part3/optimisation3_20260806_220604/`
- Purpose: measure removing the five unused `_as_list()` conversions from
  `_find_match`'s per-candidate row construction (Step 4 — the originally
  proposed XY-distance early-reject was investigated and rejected as unsafe
  first; see Step 4 above for that finding).
- Received/processed/dropped/failed: 2,740 / 304 / 2,436 / 0. Mean
  masks/processed frame: 7.661 (consistent with every prior session).
  `sam_inference_ms` 316.303 ms mean — essentially identical to Run 2's
  315.190 ms, making this an unusually clean, low-noise comparison.
- Result vs. Run 2: `assignment_row_init_ms` -70.7% mean (12.531→3.674 ms),
  `assignment_candidate_search_ms` -10.8%, `frame_assignment_ms` -9.0%,
  non-SAM portion -4.8%. `assignment_row_init_ms`'s max barely moved
  (351.020→317.840 ms, -9.5%) despite the 70.7% mean drop — strong
  supporting evidence the tail spikes are non-computational (favors the
  lock-contention hypothesis).
- Verification: 34/34 tests pass (no test referenced the removed keys);
  `colcon build` succeeds; zero effect on any matching/scoring decision
  (removed fields were never read by any accept/reject/score computation).
- Decision: **Accept and keep.** First real, measured improvement landed
  in Part 3.

### Run 4 — Path B lock-wait measurement, CONCLUSIVE (`optimisation3_20260806_222959`)

- Folder: `debug/optimisation/optimisation_part3/optimisation3_20260806_222959/`
- Purpose: confirm or rule out lock contention as the cause of the
  scattered severe tail spikes.
- Received/processed/dropped/failed: 2,383 / 277 / 2,106 / 0. Mean
  masks/processed frame: 7.646 (consistent with every prior session).
- Result: `assignment_lock_wait_ms` mean 0.010ms / max 1.369ms;
  `association_lock_wait_ms` mean 0.051ms / max 3.939ms — both negligible.
  On this session's own reproduced spike (sequence 1954,
  `frame_assignment_ms` = 335.660ms), lock-wait was 0.005ms/0.334ms while
  the four instrumented candidate-loop sub-steps summed to an ordinary
  24.155ms — leaving a 290.544ms gap attributable only to the
  uninstrumented `_candidate_track_ids()` calls (7 per this frame, one per
  observation). Frame-level, not statistical, evidence.
- Verification: 35/35 tests pass; `colcon build` succeeds; new equivalence
  test proves zero effect on `associate()`'s output.
- Decision: **Path B closed — lock contention ruled out.** Investigation
  redirects to `_candidate_track_ids()`'s final-line full-track scan (see
  "Step 5" result above), pending the order-dependency question.
