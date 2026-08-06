# Phase 1 optimisation Part 1 report

Level 2 of 3 in the campaign documentation hierarchy — see
[../OPTIMISATION_REPORT.md](../OPTIMISATION_REPORT.md) for the whole-campaign
record, global non-negotiable constraints, and the standard test protocol
this part follows. Level 3 (per-session) reports live in each
`optimisation1_<timestamp>/SESSION_REPORT.md` below.

## Objective

Minimize synchronous per-frame latency from Phase 1 callback entry through
Hydra publication without reducing object-tracking, fuser, RAP, or VLM
accuracy, for exactly one major synchronous section: representative-crop
maintenance.

## Start point

- Inherited from: campaign start (this is the first part; see
  `../OPTIMISATION_REPORT.md` "Campaign start point").
- Reference session: `optimisation1_20260806_011841`.
- End-to-end latency mean/median/p95/max: **743.077 / 726.837 / 962.189 /
  1,381.612 ms**.
- Selected target: representative-crop maintenance (`crop_update_ms`),
  135.032 ms mean, 18.2% of mean end-to-end latency — the largest eligible
  synchronous leaf after NanoSAM inference (328.441 ms mean, frozen).

## End point

- **Closed and accepted.** Reference session: `optimisation1_20260806_015126`
  (Iteration 1).
- End-to-end latency mean/median/p95/max: **602.392 / 603.160 / 813.911 /
  1,257.563 ms**.
- Change vs. start point: **-140.685 ms mean (-18.93%)**, -148.278 ms p95
  (-15.41%), throughput +26.78% more frames processed in the same 180 s
  window.
- `crop_update_ms` fell from 135.032 ms mean to 1.666 ms mean (-98.77%) and
  is no longer a meaningful synchronous bottleneck.
- Hands off to Part 2 (geometry metadata) — see
  `../optimisation_part2/PART2_REPORT.md`.

## Test protocol

Identical to the standard protocol in `../OPTIMISATION_REPORT.md`. Clear the
generic RAP database while the pipeline is stopped, build/test, launch, then
play the bag for exactly 180 seconds:

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
python3 debug/optimisation/optimisation_part1/analyse_timing.py \
  debug/optimisation/optimisation_part1/optimisation1_SESSION_ID/phase1_timing.csv
```

## Baseline

- Session: `optimisation1_20260806_011841`
- Evidence: `debug/optimisation/optimisation_part1/optimisation1_20260806_011841/`
- Frames received/processed/dropped/failed: 2,809 / 239 / 2,570 / 0
- Processing ratio: 8.51%; drop ratio: 91.49%
- End-to-end latency mean/median/p95/max: 743.077 / 726.837 / 962.189 / 1,381.612 ms
- Classifier latency mean/median/p95/max: 683.766 / 673.546 / 861.458 / 1,316.400 ms
- Largest eligible synchronous leaf stage: NanoSAM inference, 328.441 ms mean and 393.383 ms p95
- Selected Part 1 section: synchronous representative-crop maintenance
- Explicit exclusion: NanoSAM inference remains untouched by user decision

## System and test setup

- Host: `student-desktop`
- Hardware: NVIDIA Jetson AGX Orin Developer Kit, 12-core ARM Cortex-A78AE
- Memory: 61 GiB RAM, 30 GiB swap
- OS/kernel: Linux aarch64, kernel `5.15.185-tegra`
- Jetson Linux: R36.5.0 (`nvidia-l4t-core 36.5.0-20260115194252`)
- Power mode: `MODE_50W`
- Jetson clocks: not captured because `jetson_clocks --show` required root
- ROS: Humble (`ros-humble-ros-base 0.10.0-1jammy.20260607.150006`)
- Python: 3.10.12
- Git branch/commit: `latex-edit` / `8064cf9db84a5fb2c4eefe140ae27a1b8b742d36`, with uncommitted diagnostic and configuration changes recorded in the session evidence
- Pipeline launch: `ros2 launch rsg rsg_all.launch.py`
- Bag: `~/datasets/uhumans2/office_s1_00h_v2`, rate 1, TF QoS overrides from `~/.tf_overrides.yaml`
- Requested playback window: 180 seconds
- Phase 1 queue: size 1, drop oldest when full
- NanoSAM: full 640×480 input, TensorRT, CUDA, 3×3 point grid, max 8 masks, NMS IoU 0.55, minimum mask area 10 px
- Object geometry: projection stride 1
- RAP/VLM: asynchronous and excluded from bottleneck selection

## Baseline interpretation

NanoSAM inference is the largest measured stage, but it is frozen by explicit
user decision. Part 1 therefore selects the largest remaining synchronous leaf:
representative-crop maintenance at 135.032 ms mean, 118.404 ms median,
307.942 ms p95, and 439.464 ms maximum. It consumes 18.2% of mean end-to-end
latency and 19.7% of mean classifier latency.

The trace reconciles numerically: callback/FIFO + classifier + timing-hook +
coordinator + residual pipeline wait equals the 743.077 ms mean total, and the
named classifier stages plus `classifier_other_ms` equal the 683.766 ms mean
classifier total. There is no large unidentified timing gap. Within SAM,
preparation is only 0.375 ms and restored-mask construction is 20.228 ms;
328.441 ms is inside the backend call itself.

The current mask cap is saturated: 180/239 processed frames produced 8 masks,
46 produced 7, 9 produced 6, and 4 produced 5. Reducing the grid, resolution,
or mask cap could reduce time but is prohibited in Part 1 because it could
reduce object coverage or tracking/fuser accuracy.

The wider post-segmentation classifier block contains crop update 135.032 ms,
geometry projection 84.395 ms, global assignment 68.523 ms, result construction
31.570 ms, and association 4.433 ms. Part 1 changes only crop maintenance.
Geometry, assignment, and result-message restructuring remain separately
attributable future sections.

Frame assignment also shows a time-dependent growth signal: its mean rises
from 39.2 ms in the first trace quartile to 100.3 ms in the fourth, with a
sequence correlation of 0.435. This is consistent with assignment work growing
as the persistent track registry grows. It is important but is not the Part 1
target. NanoSAM remains unchanged.

The queue cannot absorb the current compute rate: 2,570 of 2,809 received
frames were dropped. Queue wait averaged 44.150 ms, but increasing queue depth
would only increase latency and process older frames; it would not correct the
compute bottleneck.

## Crop-maintenance bottlenecks

For every retained observation whose quality score is not lower than the
current best, `_remember_track_crop()` currently performs all of this on the
synchronous frame path:

1. `extract_crop_with_context()` creates an RGB context crop even though the
   caller discards that image and keeps only the computed box.
2. `prepare_target_mask()` runs connected-components cleanup and `np.isin` on
   the complete 640×480 mask, once per selected object.
3. It renders both a RAP target-only crop and a VLM focus/context crop even
   though RAP/VLM workers are asynchronous and VLM may never be needed.
4. VLM rendering performs grayscale conversion, float conversion/scaling,
   exact elliptical dilation, exterior connected components, target copying,
   and contour drawing.
5. The newly allocated crops are copied again into the track registry and
   copied a third time when a worker snapshots the task.
6. Equal quality scores use `>=`, so an equally scored later observation can
   repeat all rendering and replace the stored crop. Changing this rule without
   a better tie-breaker is not automatically accuracy-safe.

## Part 1 change sequence

1. **Remove the discarded context RGB copy.** Add a bbox-only context helper.
   This is exact, local, low risk, and should be the first implementation.
2. **Use ROI-local mask cleanup.** The SAM bbox contains every target pixel;
   connected-component cleanup on that bounded ROI is equivalent to processing
   a full-frame zero background, while touching far fewer pixels.
3. **Defer semantic rendering to the asynchronous workers.** On the critical
   path, retain one immutable best raw RGB/mask ROI plus metadata. Build the RAP
   target crop when the RAP worker dequeues it, and build the VLM crop only if
   RAP returns unknown. This preserves the exact selected source observation
   while removing expensive rendering from per-frame classification.
4. **Eliminate redundant copies without sharing mutable buffers.** Builders
   already allocate output arrays. Store them directly as read-only arrays and
   either transfer ownership or make only the single worker-bound copy.
5. **Add deterministic equivalence tests.** For identical RGB/mask/bbox input,
   require byte-identical RAP and VLM crops. Preserve quality score, revision,
   source frame, and task metadata in the unchanged task-assembly contract and
   check runtime traces for missing or invalid crop tasks.
6. **Do not change crop scoring or tie behavior initially.** A later tie-break
   optimization is acceptable only with an explicit quality-equivalence metric.

## Other classifier opportunities (not Part 1)

- **Geometry projection — 84.395 ms mean:** `ObjectGeometryEstimator` performs
  `np.where`, full-depth gathering, camera back-projection, world transformation,
  medians, and bounds independently for every mask at stride 1. A future part
  can cache per-frame camera rays and reuse mask coordinates/statistics already
  produced after SAM. Keep stride 1 and all geometry values to preserve tracking.
- **Global assignment — 68.523 ms mean and rising:** every observation evaluates
  many persistent tracks and multiple full-resolution mask pairs before Hungarian
  assignment. A conservative 3D spatial candidate index and bbox-bounded exact
  mask intersections can reduce work without removing any eligible match. This
  needs dedicated revisit/continuation regression tests.
- **Result construction — 31.570 ms mean:** Phase 1 builds an intermediate ROS
  classification message, serializes JSON, then immediately parses/converts parts
  again to build Hydra output. A future structural part can build one internal
  result object and serialize each final payload once, reducing code size and
  memory copies while retaining identical published messages.
- **Hydra depth filtering — 7.284 ms mean:** filtering currently converts ROS
  images back to arrays after label-message construction. Applying the identical
  mask before the one final image-message conversion can eliminate a round trip.

Iteration 1 implements steps 1–5 as one cohesive data-lifetime change. The
frame-critical path now computes the context box without allocating a discarded
crop, checks the existing quality score, and stores one immutable contiguous
RGB/mask context ROI for an accepted revision. Mask cleanup and semantic
rendering run only when an asynchronous worker dequeues that revision. RAP
dequeue renders only the RAP target crop; VLM dequeue renders the VLM crop and
the matching RAP crop needed by the optional live-memory update. The registry
and dequeue stages no longer make redundant copies of already-rendered crops.

Crop scoring, equal-score replacement (`>=`), context ratio, mask cleanup,
target-only RAP representation, VLM halo/context rendering, source revision,
and task metadata are unchanged. NanoSAM and tracking/assignment code are
untouched. A deterministic random-image regression test compares the old
full-frame rendering route with ROI-local deferred rendering and requires exact
array equality for RAP and VLM outputs, including cleanup, halo, grayscale
context, and contour drawing.

The post-baseline framework correction also removes duplicate drop diagnostic
rows for future sessions; baseline counts above were deduplicated by sequence.

## Constraints

- Preserve input data, model resolution, thresholds, tracking decisions, label
  maps, object metadata, and published behavior unless equivalence is proven.
- Do not optimize asynchronous RAP/VLM retrieval or inference in this part.
- Make one attributable change per iteration and retain every session folder.
- Move the next major synchronous section to a separately numbered part.

## Iterations

| Iteration | Session | Code change | Mean total | P95 total | Accuracy/function checks | Decision |
|---|---|---|---:|---:|---|---|
| Baseline | `optimisation1_20260806_011841` | Timing only | 743.077 ms | 962.189 ms | 0 processing failures; accuracy equivalence not yet measured | Select crop maintenance; freeze NanoSAM |
| 1 | `optimisation1_20260806_015126` | Immutable context ROI; defer mask cleanup and RAP/VLM rendering to workers; remove redundant crop copies | 602.392 ms | 813.911 ms | 0 processing failures; 29 tests pass; exact crop pixels; no missing-crop records | Accept and close crop-maintenance section |

## Run registry

### Run 1 — baseline (`optimisation1_20260806_011841`)

- Folder: `debug/optimisation/optimisation_part1/optimisation1_20260806_011841/`
- Raw trace: `phase1_timing.csv` (5,497 data rows: 239 frame traces,
  5,140 duplicate drop rows representing 2,570 unique drops, plus async worker rows)
- Generated analysis: `stage_summary.csv`, `stage_summary.md`
- Session report: `SESSION_REPORT.md`
- Purpose: establish synchronous Phase 1 end-to-end latency and select one
  major section for Part 1 without changing perception behavior.
- Decision: leave NanoSAM untouched and optimize synchronous crop maintenance first.

### Run 2 — Iteration 1 (`optimisation1_20260806_015126`)

- Folder: `debug/optimisation/optimisation_part1/optimisation1_20260806_015126/`
- Purpose: quantify the reduction in `crop_update_ms`, classifier latency,
  end-to-end latency, and frame drops after deferred ROI rendering.
- Controlled variables: same launch file, YAML, bag, playback rate, 180-second
  window, generic RAP-memory location, and NanoSAM configuration as baseline.
- Verification: package build succeeded; all 29 package unit tests pass;
  deferred RAP/VLM output is byte-identical to the previous full-frame route.
- Received/processed/dropped/failed: 2,829 / 303 / 2,526 / 0.
- Mean/median/p95/max total latency: 602.392 / 603.160 / 813.911 /
  1,257.563 ms.
- Mean/median/p95/max crop update: 1.666 / 1.000 / 5.697 / 11.406 ms.
- Baseline-to-iteration change: crop mean -98.77%, total mean -18.93%,
  classifier mean -20.76%, and processed frames +26.78%.
- Functional checks: no synchronous failure, no missing-crop or worker-exception
  trace, and comparable mean masks per processed frame (7.682 versus 7.726).
- Asynchronous caveat: 28 VLM results report HTTP 503. This endpoint-availability
  issue is excluded from synchronous timing and prevents using this run as a
  VLM accuracy benchmark; it is not a deferred-crop missing-data failure.
- Decision: accept Iteration 1. Crop maintenance is no longer a meaningful
  synchronous bottleneck. See the session's `SESSION_REPORT.md`
  ("Comparison to previous session") for the full baseline-vs-iteration table.

## Source files touched by this part

- `src/rsg/nodes/phase1.py` — end-to-end instrumentation, consolidated frame
  traces, Hydra substage timing, immutable ROI registry, worker-deferred crop
  rendering (`_remember_track_crop`, `_snapshot_track_task`).
- `src/rsg/nodes/support/phase1/semantic_crop.py` — existing semantic-crop
  algorithms plus allocation-free `context_bbox_xywh()`.
- `src/rsg/nodes/support/phase1/frame_cache.py` — stores callback-enqueue and
  queue-wait timestamps used by frame tracing.
- `src/rsg/nodes/support/phase1/phase1_config.py` — buffered-CSV timing
  configuration and compatibility parsing.
- `src/rsg/nodes/support/phase1/phase1_timing_recorder.py` — RAM-buffered CSV
  recorder written on shutdown.
- `src/rsg/config/rsg_pipeline.yaml` — enabled buffered Part 1 CSV output
  (later repointed to `optimisation_part2/` at the start of Part 2 — see
  `../optimisation_part2/PART2_REPORT.md`); generic RAP memory and frozen
  NanoSAM values unchanged.
- `src/rsg/tests/test_semantic_crop.py` — added
  `test_roi_deferred_render_is_pixel_identical_to_full_frame_render` and
  supporting equivalence tests.
- `debug/optimisation/optimisation_part1/analyse_timing.py` — generates
  `stage_summary.csv`/`.md` from a session's `phase1_timing.csv`.
- `debug/optimisation/optimisation_part1/optimisation1_*/` — session evidence
  folders (raw traces, generated summaries, per-session reports, config/git
  snapshots).

## Final findings

Part 1 is complete. Immutable ROI storage and deferred rendering reduced mean
crop maintenance from 135.032 ms to 1.666 ms (-98.77%). Mean end-to-end latency
fell from 743.077 ms to 602.392 ms (-18.93%), and processed frames increased
from 239 to 303 (+26.78%) over the controlled 180-second playback.

The next largest eligible synchronous leaf is geometry metadata at 86.966 ms
mean. Frame assignment averages 76.349 ms but rises from 37.747 ms in the first
run quartile to 121.910 ms in the fourth. Both are outside Part 1 and must not
be changed here. NanoSAM remains frozen. Start the next selected section in a
separately numbered optimization folder and report.
