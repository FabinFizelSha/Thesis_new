# Session report: optimisation1_20260806_015126

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART1_REPORT.md](../PART1_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol.

## Purpose

Measure Part 1 Iteration 1 against the baseline session (`optimisation1_20260806_011841`)
after replacing synchronous semantic-crop rendering with an immutable context
ROI and worker-deferred rendering. The run tests whether `crop_update_ms` and
Phase 1 end-to-end latency fall without changing NanoSAM, tracking, fuser,
RAP, or VLM crop semantics.

## Code change under test

`Phase1SemanticCoordinator._remember_track_crop()` / `_snapshot_track_task()`
(`nodes/phase1.py`) and `nodes/support/phase1/semantic_crop.py`: store one
immutable RGB/mask ROI on the synchronous path; defer mask cleanup and
RAP/VLM rendering to the worker thread that actually dequeues the track. Full
root-cause and change description in `../PART1_REPORT.md`.

## Runtime configuration

- Session folder: `debug/optimisation/optimisation_part1/optimisation1_20260806_015126/`
- Configuration snapshot: `rsg_pipeline_snapshot.yaml`
- Repository state: `git_status.txt`, `working_tree.patch`
- System snapshot: `system_setup.txt` — NVIDIA Jetson AGX Orin Developer Kit,
  Jetson Linux R36.5.0, 12-core ARM Cortex-A78AE, 61 GiB RAM, `MODE_50W`,
  ROS 2 Humble, Python 3.10.12
- Git branch/commit: `latex-edit` / `8064cf9db84a5fb2c4eefe140ae27a1b8b742d36`
  plus the uncommitted Part 1 diagnostic/code changes recorded in
  `working_tree.patch`

## Test protocol

The generic RAP database and audit file were cleared while the pipeline was
stopped. The complete stack was launched using the existing config:

```bash
cd ~/rsg_ros2_ws
mkdir -p ~/rsg_rap_memory
find ~/rsg_rap_memory -mindepth 1 -delete
rm -f ~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rsg rsg_all.launch.py
```

The uHumans office bag was played at rate 1 for exactly 180 seconds:

```bash
timeout --signal=INT --kill-after=15s 180s \
  ros2 bag play ~/datasets/uhumans2/office_s1_00h_v2 \
  --rate 1 \
  --qos-profile-overrides-path ~/.tf_overrides.yaml
```

The launch was stopped cleanly so the buffered CSV was written once at
shutdown. RAP/VLM inference timing remained asynchronous and excluded.

## Evidence

- Raw trace: `phase1_timing.csv` (3,046 data rows)
- Generated stage analysis: `stage_summary.csv`, `stage_summary.md`
- Baseline session compared against: `optimisation1_20260806_011841`

## Results (this session)

- Received/processed/dropped/failed: 2,829 / 303 / 2,526 / 0.
- Processing ratio: 10.71%; drop ratio: 89.29%.
- Total latency mean/median/p95/max: 602.392 / 603.160 / 813.911 /
  1,257.563 ms.
- Classifier latency mean/median/p95/max: 541.846 / 545.937 / 722.995 /
  1,022.002 ms.
- Crop update mean/median/p95/max: 1.666 / 1.000 / 5.697 / 11.406 ms.
- Largest next eligible synchronous leaf: geometry metadata, 86.966 ms mean.

## Comparison to previous session (`optimisation1_20260806_011841`)

| Metric | Previous session (baseline) | This session | Change |
|---|---:|---:|---:|
| Frames received | 2,809 | 2,829 | +20 (+0.71%) |
| Frames processed/published | 239 | 303 | +64 (+26.78%) |
| Unique dropped frames | 2,570 | 2,526 | -44 (-1.71%) |
| Processing ratio | 8.51% | 10.71% | +2.20 percentage points |
| Mean end-to-end latency | 743.077 ms | 602.392 ms | -140.685 ms (-18.93%) |
| Median end-to-end latency | 726.837 ms | 603.160 ms | -123.677 ms (-17.02%) |
| P95 end-to-end latency | 962.189 ms | 813.911 ms | -148.278 ms (-15.41%) |
| Maximum end-to-end latency | 1,381.612 ms | 1,257.563 ms | -124.049 ms (-8.98%) |
| Mean classifier latency | 683.766 ms | 541.846 ms | -141.920 ms (-20.76%) |
| P95 classifier latency | 861.458 ms | 722.995 ms | -138.463 ms (-16.07%) |
| Mean crop maintenance | 135.032 ms | 1.666 ms | -133.366 ms (-98.77%) |
| P95 crop maintenance | 307.942 ms | 5.697 ms | -302.245 ms (-98.15%) |
| Processing failures | 0 | 0 | unchanged |

The crop-stage reduction accounts for approximately 94% of the measured mean
end-to-end improvement and classifier improvement — strong attribution to
this session's single code change rather than an unrelated timing
fluctuation.

## Functional / accuracy evidence

- NanoSAM configuration and implementation were not changed.
- Crop quality scoring and equal-score replacement behavior were not changed.
- Persistent tracking, global assignment, geometry, label-map, fuser, and
  Hydra publication logic were not changed.
- A deterministic regression test
  (`test_roi_deferred_render_is_pixel_identical_to_full_frame_render`) proves
  byte-identical RAP and VLM crop pixels between the previous full-frame
  rendering route and the new deferred ROI-local rendering route.
- All 29 package unit tests pass and the `rsg` package builds successfully.
- Mean masks per processed frame remained comparable: 7.682 baseline versus
  7.726 this session. Mask-count distribution: 235 frames with 8 masks, 57
  with 7, 7 with 6, 4 with 5.
- The trace contains no `missing_crop` or `worker_exception` record and no
  synchronous processing failure.

This timing run does not independently establish end-task object-tracking or
semantic-label accuracy. The unchanged tracking code and exact crop-image
equivalence provide regression evidence, but a thesis accuracy claim must use
the project's separate accuracy evaluation rather than timing traces alone.

## Asynchronous VLM caveat

RAP/VLM retrieval and inference are explicitly excluded from Part 1 latency
selection because they do not block Hydra publication. This trace contains 72
completed VLM records: 22 `vlm_done` and 50 `vlm_failed`. Of the failed
records, 28 explicitly report `HTTP Error 503: Service Unavailable`; the rest
are model/validation outcomes. The baseline also contained VLM failures (16
of 39 completed records). There are no missing-crop failures, so this does
not point to deferred crop rendering as the cause — it is an endpoint
availability issue, tracked separately, and this run must not be used as a
VLM accuracy benchmark.

## New synchronous bottleneck surfaced by this session

With crop maintenance reduced to 1.666 ms mean, the largest eligible leaf
stage becomes geometry metadata construction:

- `geometry_metadata_ms`: 86.966 ms mean, 82.380 ms median, 150.665 ms p95.
- `frame_assignment_ms`: 76.349 ms mean, 65.922 ms median, 122.547 ms p95,
  rising within the run (quartile means 37.747 / 67.782 / 77.452 /
  121.910 ms, sequence correlation 0.442).
- `result_message_build_delay_ms`: 32.071 ms mean.

NanoSAM inference remains the largest absolute leaf at 308.803 ms mean, but
stays frozen and ineligible for the whole campaign.

## Decision

**Accept Iteration 1 and retain the code. Close Part 1.** Representative-crop
maintenance is no longer a meaningful synchronous bottleneck. The next
optimization part targets geometry metadata construction — see
`../../optimisation_part2/PART2_REPORT.md`.
