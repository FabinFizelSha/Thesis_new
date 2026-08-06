# Phase 1 optimisation Part 4 report

Level 2 of 3 in the campaign documentation hierarchy — see
[../OPTIMISATION_REPORT.md](../OPTIMISATION_REPORT.md) for the whole-campaign
record, global non-negotiable constraints, and the standard test protocol
this part follows. Level 3 (per-session) reports will live in each
`optimisation4_<timestamp>/SESSION_REPORT.md` below as runs complete.

## This part is a different kind of change than Parts 1-3 — read this first

Parts 1-3 each reduced the mean synchronous compute cost of one stage
(crop maintenance, geometry estimation, frame assignment) while the pipeline
stayed a single serial pass per frame. **Part 4 does not reduce any stage's
compute cost.** It splits the single classification thread into two threads
— segmentation (image conversion + SAM) and tracking/publish (geometry,
assignment, label maps, Hydra publish) — connected by a new one-slot FIFO,
so SAM for frame N+1 can run while tracking/publish is still finishing frame
N. SAM and tracking/publish now overlap in wall-clock time instead of
running back to back.

Consequence for what to measure: **a single processed frame's own
`total_delay_ms` is not the primary question here**, and may not shrink —
it could even grow slightly, since a frame now additionally waits in the new
`sam_output_queue_wait_ms` hop between the two threads. The metric this part
is actually testing is **throughput**: how many of the received frames get
processed and published to Hydra in the same 180 s window. The existing
"largest leaf stage" framing from Parts 1-3's analysers is kept in the
generated report for continuity, but it does not answer Part 4's question —
see `analyse_timing.py`'s "Throughput" section, which is the one that
matters.

**Part 3 status:** Part 3 remains **open**, not closed — its Step 5 ruled
out lock contention and redirected the tail-spike investigation to
`_candidate_track_ids()`'s full-track scan (see
`../optimisation_part3/PART3_REPORT.md` "Step 5" / "Run 4"), which is still
unresolved. Part 4 opens as a separate, parallel investigation on a
different optimization axis (concurrency/throughput vs. Part 3's per-frame
compute-cost reduction), not a continuation of Part 3's specific question.
Both remain open; neither blocks the other.

## Objective

Increase the fraction of received frames that reach Hydra in a fixed
playback window, by overlapping SAM (GPU-bound) with tracking/publish
(CPU-bound) across adjacent frames, without changing any stage's computed
output, without changing tracking/assignment decisions, and without
reducing object-tracking, fuser, RAP, or VLM accuracy.

## Start point

- Inherited from: Part 3's current accepted point (Part 3 remains open, but
  its last accepted, latency-affecting change is the reference — see
  `../OPTIMISATION_REPORT.md` "Campaign current point").
- Reference session: `optimisation_part3/optimisation3_20260806_220604`
  (Part 3 Run 3, the row-init dead-weight removal — the last change in the
  campaign that altered measured latency; Part 3's Run 4 was
  instrumentation-only with no behavior change).
- End-to-end latency mean/median/p95/max: **589.961 / 579.772 / 744.811 /
  1,119.836 ms**; classifier mean **530.069 ms**.
- Received/processed/dropped/failed: **2,740 / 302 / 2,436 / 0** —
  processing ratio **11.02%** (88.98% of received frames dropped at the
  single-slot pre-SAM FIFO because the serial pipeline could not keep up).
- Per-stage split at this start point (mean): SAM (`sam_delay_ms`) 337.6 ms
  (63.7% of `classifier_delay_ms`); geometry+tracking+dispatch
  (`rap_delay_ms`) 157.3 ms (29.7%); label map + Hydra build/publish
  ~17-20 ms (~3.5%); coordinator overhead ~12.7 ms. This is the split Part 4
  targets: `max(337.6, ~187) ≈ 337.6 ms` bound instead of `530.1 ms` serial
  is the theoretical throughput ceiling this change is testing for.
- NanoSAM inference: unchanged, still frozen — Part 4 does not touch SAM
  invocation, only what runs concurrently with it.

## End point

**Closed — accepted.** Reference session: `optimisation4_20260807_004917`
(Run 1, first valid controlled run — see "Run registry").

- Processing ratio **11.02% → 13.53%** (+2.51pp, **+23.51% more frames
  processed**: 302 → 373 of a comparable received count). Mean masks/frame
  unchanged (7.661 → 7.662) — no coverage/accuracy regression.
- Mean `total_delay_ms` rose 589.961 → 727.406 ms (+23.30%), and
  `classifier_delay_ms` rose 530.069 → 676.062 ms (+27.54%). Tail latency
  rose too: p95 744.811 → 896.909 ms (+20.42%), max 1,119.836 → 1,563.460 ms
  (+39.61%).
- Root cause of the latency rise: real CPU contention between the two
  threads, measurably larger than the "SAM releases the GIL almost
  entirely" hypothesis predicted (see "Run 1 finding" below) —
  `sam_inference_ms` itself rose 316.303 → 436.291 ms (+37.94%) when run
  concurrently with the tracking/publish thread.
- **Decision (user-confirmed 2026-08-07): accepted.** The per-frame latency
  reduction pursued in Parts 1-3 was always instrumental, not terminal —
  its purpose was to let more frames reach Hydra before the pipeline moves
  on, i.e. throughput. Part 4 reaches that same goal by a different
  mechanism (overlap instead of a cheaper per-stage computation), and the
  higher per-frame latency this specific mechanism costs is accepted as the
  price of that throughput gain, not treated as a regression. Satisfies
  this part's own stated success criterion (processing ratio up) and is
  accepted on that basis, with the latency cost explicitly acknowledged and
  recorded rather than hidden — see "Run 1 — accepted" in the Run registry.

## Selected Part 4 section

**SAM/tracking-publish thread split** — `Phase1SemanticCoordinator` in
`nodes/phase1.py`. The single `_classification_loop` consuming `frame_fifo`
and running `process_frame()` end-to-end is replaced by two threads:

- `_segmentation_loop` — dequeues `frame_fifo`, runs image conversion and
  SAM only (`run_segmentation_stage()`), then hands the result to a new
  `sam_output_fifo` (single-slot, same drop-oldest bias as `frame_fifo`).
- `_tracking_publish_loop` — dequeues `sam_output_fifo`, runs geometry,
  persistent-track assignment, label-map construction, result assembly, and
  the Hydra publish (`run_tracking_publish_stage()` +
  `_publish_hydra_from_result()`).

This is a pure decomposition of the previous `process_frame()` body into two
functions plus a queue handoff — no line inside either function changes what
it computes. The RAP and VLM worker threads are untouched; they already sat
downstream on track-ID-only queues before this change.

## Why this is expected to help (grounded in the start-point profile, not assumed)

- SAM inference is a single ~316 ms call into a GPU backend (NanoSAM via
  TensorRT). That call almost certainly releases CPython's GIL for the
  majority of its wall time while it blocks on the GPU — the same property
  the existing RAP/VLM worker threads already rely on for their overlap with
  the main thread.
- Tracking/publish is CPU/numpy work with no GPU dependency. Two threads
  where one is GPU-bound-and-GIL-releasing and the other is CPU-bound can
  get real wall-clock overlap under CPython's GIL, unlike two purely
  CPU-bound Python threads.
- Per Part 2's methodological lesson ("a plausible textbook optimization is
  not reliable on this hardware without a measured run"), this reasoning is
  the *hypothesis* Run 1 tests, not a conclusion — `sam_output_queue_wait_ms`
  is the specific number that will confirm or refute real overlap.

**Run 1 finding — hypothesis partially wrong, correction needed.**
`sam_output_queue_wait_ms` came back small (mean 0.704 ms, p95 0.888 ms) —
so the *scheduling* half of the hypothesis held: the tracking/publish
thread is essentially always idle and ready, waiting on SAM, confirming SAM
is the throughput-limiting stage and the two threads are genuinely
overlapping across frames. But the "almost entirely GIL-released, so
overlap is nearly free" half was **not** confirmed — `sam_inference_ms`
itself rose 316.303 → 436.291 ms (+37.94%) when run concurrently with the
tracking/publish thread's CPU work, and `geometry_metadata_ms`
(+19.62%)/`frame_assignment_ms` (+15.62%) rose too. This is evidence of
real contention (CPU cores, memory bandwidth, and/or Jetson thermal
behavior — not yet distinguished) between the two threads that the GIL
model alone did not predict. The throughput gain is real and the two
threads are correctly overlapping in wall-clock time, but the overlap is
not "free" the way the reasoning above assumed — each stage individually
runs slower while overlapped than it did alone in the old serial design.
See "Run 1 — ACCEPTED" in the Run registry for the full numbers and the
acceptance decision.

## Why not split further

`label_map_delay_ms` + `hydra_build_delay_ms` + `hydra_publish_delay_ms`
sum to ~17 ms at the start point — not worth a third thread/queue hop.
More importantly, those stages and the geometry/tracking stage are all
pure-Python/CPU work with no GPU wait, so additional threads among them
would not gain real concurrency under the GIL — that would need separate
*processes*, which reintroduces the cross-process frame-serialization cost
`phase1.py`'s own module docstring already documents as the reason this
node is single-process. Part 4 stays at exactly two threads, split at the
one GPU/CPU boundary that actually exists.

## Verification completed before requesting a run

- `python3 -m pytest -q tests -p no:anyio` → **35/35 pass** (no regression;
  no test currently exercises the new thread-split code path directly —
  see "What this part's tests do and don't prove" below).
- `colcon build --packages-select rsg` → succeeds.
- Module import + signature check (`run_segmentation_stage`,
  `run_tracking_publish_stage`, `_segmentation_loop`,
  `_tracking_publish_loop`, `_enqueue_sam_output` all present with expected
  signatures) — a live ROS/GPU context is required to instantiate the node
  itself, so this is the deepest static check possible outside a real run.
- `git status` confirms only `nodes/phase1.py` changed for the code, plus
  this report, `config/rsg_pipeline.yaml`'s timing repoint, and
  `debug/optimisation/optimisation_part4/analyse_timing.py` (new).

### What this part's tests do and don't prove

Parts 1-3 each proved output-equivalence with a synthetic unit test (e.g.
Part 1's byte-identical crop comparison, Part 3's keep-mask/match
equivalence tests) because each was a pure-function rewrite that a
synthetic array input could exercise directly. Part 4 is not a pure-function
rewrite — `run_tracking_publish_stage()` calls the exact same
`run_rap_and_metadata()`, `label_map_builder.build()`, etc. as before with
unchanged arguments, so the 35 existing unit tests covering those functions
still indirectly cover everything Part 4 did not touch. What Part 4 added —
two real threads, a new queue, and genuine cross-frame overlap — is a
concurrency/scheduling property that only exists once SAM, ROS, and the GPU
are actually running; it cannot be meaningfully asserted by a synthetic
unit test the way a numeric formula can. Its evidence is necessarily the
live bag run's `sam_output_queue_wait_ms` and throughput numbers, not a
pre-run equivalence test. This is stated explicitly rather than silently
skipped.

## Test protocol

Pointed at this part's own session folder and analyser, with one
Part-4-specific amendment to the bag-play command (see "Protocol amendment"
below) — otherwise identical to the standard protocol in
`../OPTIMISATION_REPORT.md`:

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
  --read-ahead-queue-size 10000 \
  --qos-profile-overrides-path ~/.tf_overrides.yaml
```

Stop the launch cleanly with Ctrl+C after playback ends. Analyze the newest
session:

```bash
ls -dt debug/optimisation/optimisation_part4/optimisation4_* | head -n 1
python3 debug/optimisation/optimisation_part4/analyse_timing.py \
  debug/optimisation/optimisation_part4/optimisation4_SESSION_ID/phase1_timing.csv
```

`phase1.performance.timing_csv_path` in `config/rsg_pipeline.yaml` already
points new sessions at `optimisation_part4/optimisation4_<timestamp>/`.

### Protocol amendment: `--read-ahead-queue-size`

Attempt 0 (see "Run registry" below) hit repeated `ros2 bag play` "Message
queue starved" warnings — the player's default read-ahead buffer (1000
messages) drained faster than it could be refilled from the 40.5 GiB / 742,621-
message bag, so messages were delivered late and irregularly. That breaks
the `--rate 1` real-time assumption every latency/throughput number in this
campaign depends on, so that attempt's trace is not valid evidence for
either axis (Part 4's throughput question or Part 1-3's per-stage latency
framing) — it does not measure Phase 1, it measures a starved player.
`--read-ahead-queue-size 10000` (10x default) is added to the bag-play
command above for every Part 4 run from here on. Not yet established
whether the starvation was purely a cold-disk-cache/first-replay-in-a-while
effect, or partly caused by Part 4 itself increasing peak simultaneous
CPU load (SAM and tracking/publish now genuinely run at the same time,
competing with the bag player process for CPU, where the old serial design
left CPU idle during GPU-bound SAM) — Run 1 will show whether the larger
queue alone is sufficient or whether this needs revisiting.

## Constraints specific to this part

(In addition to the global constraints in `../OPTIMISATION_REPORT.md`.)

- NanoSAM invocation, geometry, tracking/assignment logic, label-map
  construction, and Hydra message construction are byte-for-byte unchanged
  — verify this explicitly in Run 1 by comparing `num_masks`, `num_known`,
  `num_unknown`, and mean masks/frame against the Part 3 start-point session,
  same as every prior part's accuracy check.
- The new `sam_output_fifo` uses the same drop-oldest, size-1 policy as
  `frame_fifo` — this must not be silently changed to a blocking queue
  without an explicit decision, since a blocking queue would remove the
  real-time freshness bias the rest of the pipeline relies on.
- Report throughput (received/processed/dropped/processing ratio) as the
  primary result, not `total_delay_ms` alone — a flat or slightly higher
  mean `total_delay_ms` with a materially higher processing ratio is a
  successful outcome for this part, not a regression.
- One attributable change per iteration; retain every session folder; never
  overwrite an earlier one.

## Run registry

### Attempt 0 — invalidated and deleted (`optimisation4_20260807_004013`)

- Purpose: first controlled measurement attempt.
- Outcome: **invalid, not evidence.** `ros2 bag play` reported repeated
  "Message queue starved" warnings throughout playback (default
  `--read-ahead-queue-size` of 1000 insufficient for this 40.5 GiB /
  742,621-message bag), meaning messages were delivered late and
  irregularly rather than at the `--rate 1` real-time pace every timing
  number in this campaign depends on. The session did produce a
  `phase1_timing.csv` (349 frame traces, 2,137 drop rows), but it measures
  a starved bag player, not Phase 1, so it cannot be compared against the
  Part 3 start point or used to judge this part's throughput question.
- Action: session folder deleted per campaign evidence-retention rule —
  this exception applies because the run was invalidated by a playback
  tooling failure external to Phase 1, not because the code under test
  produced an unfavorable result. See "Protocol amendment" above for the
  fix (`--read-ahead-queue-size 10000`) applied before the next attempt.
- Decision: retry as Run 1 with the amended bag-play command.

### Run 1 — first valid measurement, ACCEPTED (`optimisation4_20260807_004917`)

- Folder: `debug/optimisation/optimisation_part4/optimisation4_20260807_004917/`
- Purpose: measure the SAM/tracking-publish thread split against the Part 3
  start point, with `--read-ahead-queue-size 10000` fixing Attempt 0's bag
  starvation.
- No "Message queue starved" warnings this run — protocol amendment
  confirmed working.
- Received/processed/dropped/failed: **2,757 / 373 / 2,381 / 0** (Part 3
  start point: 2,740 / 302 / 2,436 / 0).

**Throughput (this part's primary question):**

| Metric | Part 3 start point | Run 1 | Change |
|---|---:|---:|---:|
| Processed frames | 302 | 373 | **+71 (+23.51%)** |
| Processing ratio | 11.02% | 13.53% | **+2.51pp (+22.78% relative)** |
| Mean masks/processed frame | 7.661 | 7.662 | +0.001 (unchanged) |

**Latency (mean ms, secondary for this part but reported in full):**

| Stage | Part 3 start point | Run 1 | Change |
|---|---:|---:|---:|
| `total_delay_ms` | 589.961 | 727.406 | +137.445 (+23.30%) |
| `classifier_delay_ms` | 530.069 | 676.062 | +145.993 (+27.54%) |
| `sam_delay_ms` | 337.553 | 456.959 | +119.406 (+35.38%) |
| `sam_inference_ms` | 316.303 | 436.291 | **+119.988 (+37.94%)** |
| `sam_prepare_ms` | 0.358 | 0.246 | -0.112 (noise) |
| `sam_restore_ms` | 20.885 | 20.415 | -0.470 (noise) |
| `rap_delay_ms` (tracking/dispatch aggregate) | 157.259 | 184.078 | +26.819 (+17.06%) |
| `geometry_metadata_ms` | 85.150 | 101.858 | +16.708 (+19.62%) |
| `frame_assignment_ms` | 64.016 | 74.013 | +9.997 (+15.62%) |
| `track_association_ms` | 4.242 | 4.317 | +0.075 (noise) |
| `crop_update_ms` | 1.621 | 1.700 | +0.079 (noise) |
| `label_map_delay_ms` | 5.466 | 5.979 | +0.513 (+9.39%) |
| `hydra_build_delay_ms` | 7.676 | 8.251 | +0.575 (+7.49%) |
| `hydra_publish_delay_ms` | 4.115 | 4.569 | +0.454 (+11.03%) |
| `coordinator_delay_ms` | 12.660 | 13.654 | +0.994 (+7.85%) |
| `sam_output_queue_wait_ms` (new) | n/a | 0.704 | new metric, negligible |
| `total_delay_ms` p95 / max | 744.811 / 1,119.836 | 896.909 / 1,563.460 | +20.42% / +39.61% |

**Interpretation.** Every stage's mean got somewhat worse, not just SAM —
consistent with real thread contention rather than a measurement artifact
isolated to one stage. `sam_output_queue_wait_ms` staying tiny (mean
0.704 ms vs. `sam_inference_ms`'s 436.291 ms) confirms the tracking/publish
thread is essentially always caught up and waiting, i.e. SAM is the
throughput-limiting stage exactly as designed, and the two threads are
genuinely overlapping. A simple ceiling model supports this: old serial
cadence ≈ `classifier_delay_ms` mean (530.069 ms) → theoretical max
≈ 180,000 ms / 530.069 ms ≈ 340 frames (observed 302, 88.9% of ceiling);
new pipelined cadence ≈ the slower concurrent stage, `sam_delay_ms`
(456.959 ms) → theoretical max ≈ 180,000 / 456.959 ≈ 394 frames (observed
373, 94.7% of ceiling). Both runs land close to their respective ceilings,
and the new ceiling is higher — which is exactly why more frames get
through despite each one now costing more individually.

**Accuracy/function checks:** 0 failures; mean masks/frame unchanged
(7.661 → 7.662); no missing-crop or worker-exception symptoms in the trace.
No evidence of a tracking/geometry/assignment correctness regression — the
effect is purely a timing one.

**Decision (user-confirmed 2026-08-07): accept and keep.** This part's own
stated success criterion (`../PART4_REPORT.md` "Constraints specific to
this part") is satisfied — processing ratio rose materially, with no
accuracy/coverage regression. The user's explicit direction: the point of
reducing per-frame latency in Parts 1-3 was always to increase throughput,
not latency reduction as an end in itself — so a change that reaches higher
throughput by a different mechanism (overlap, at a measured per-frame
latency cost) satisfies the actual underlying goal and is accepted on that
basis. The latency cost (mean +23.30%, max +39.61%) is recorded above and
in "Run 1 finding", not hidden, but is not grounds for rejection given this
framing.

Not pursued further at this time (recorded for any future revisit, not
blocking this acceptance):

- **Contention root cause** — pin the two threads to separate CPU cores, or
  profile whether the slowdown is concentrated in TensorRT's CPU-side
  pre/post-processing specifically (would explain why `sam_inference_ms`
  moved so much more than `sam_prepare_ms`/`sam_restore_ms`, which stayed
  flat). Could plausibly push the processing ratio even higher if resolved,
  but the current result already clears this part's bar without it.

See "Final findings" below for the closing summary.

## Source files touched by this part

- `src/rsg/nodes/phase1.py` — `_classification_loop` replaced by
  `_segmentation_loop` + `_tracking_publish_loop`; `process_frame` split into
  `run_segmentation_stage` + `run_tracking_publish_stage`; new
  `sam_output_fifo`/`_enqueue_sam_output`; `sam_output_queue_wait_ms` folded
  into `pipeline_wait_ms` accounting and the timing CSV; thread
  creation/start and `destroy_node` join list updated; `publish_status`
  reports the new queue's size/drop count.
- `src/rsg/config/rsg_pipeline.yaml` — `phase1.performance.timing_csv_path` /
  `timing_sheet_name` repointed from `optimisation_part3` to
  `optimisation_part4`.
- `debug/optimisation/optimisation_part4/analyse_timing.py` — forked from
  Part 3's copy (not edited in place), with `sam_output_queue_wait_ms` added
  as a leaf stage and a new throughput-first section in the generated
  report.
- Branch: `pipeline-split` (off `rsg-uhumans-working`).

## Final findings

Run 1 confirms the throughput hypothesis directionally but not for free:
splitting SAM from tracking/publish across two threads increased processed
frames from 302 to 373 (+23.51%) and the processing ratio from 11.02% to
13.53% in the same 180 s window, with mean masks/processed frame unchanged
(7.661 → 7.662) — no coverage or accuracy regression. `sam_output_queue_wait_ms`
staying negligible (mean 0.704 ms) confirms the two threads are genuinely
overlapping and SAM is the throughput-limiting stage as designed.

However, the overlap is not free: `sam_inference_ms` itself rose 37.94%
(316.303 → 436.291 ms) and `geometry_metadata_ms`/`frame_assignment_ms` rose
19.62%/15.62% when the two threads ran concurrently, versus running alone in
the old serial design. This means the "SAM releases the GIL almost
entirely, so overlap is nearly free" half of this part's original hypothesis
was not correct — real contention between the two threads (CPU cores,
memory bandwidth, and/or Jetson thermal behavior, not yet distinguished)
measurably slows each one down while they overlap. Mean per-frame latency
rose 23.30% and max latency rose 39.61% as a result.

**Part 4 is closed — accepted, as a successful concept change on the
throughput axis.** The per-frame latency work in Parts 1-3 was always
instrumental toward one goal — more frames reaching Hydra — not an end in
itself. Part 4 reaches that same goal through a different mechanism
(overlapping SAM and tracking/publish across frames instead of cutting one
stage's own cost) and the measured result — +23.51% more frames processed,
zero accuracy/coverage regression — is accepted on that basis. The
per-frame latency cost (mean +23.30%, max +39.61%) is real and is recorded
above, not hidden, but is not grounds for rejection given what this
campaign is actually optimizing for. The unresolved contention root cause
(why the overlap costs more than the GIL-release model predicted) is noted
as a possible future refinement, not a blocker to this acceptance.

Hands off to whichever part is picked up next — Part 3 remains open and
unresolved (`_candidate_track_ids`'s full-track scan) independently of this
closure.
