# Session report: optimisation1_20260806_011841

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART1_REPORT.md](../PART1_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol. This is the **campaign baseline
session** — the first controlled run, with no prior session to compare
against.

## Purpose

Baseline the synchronous path of every processed frame from entry into Phase 1
through publication to Hydra. Select the highest-delay section for Part 1.
Asynchronous RAP/VLM inference and retrieval are recorded separately but are
not included in bottleneck selection.

## Evidence

- Session folder: `debug/optimisation/optimisation_part1/optimisation1_20260806_011841/`
- Raw buffered CSV: `phase1_timing.csv`
- Stage statistics: `stage_summary.csv`, `stage_summary.md`
- Configuration snapshot: `rsg_pipeline_snapshot.yaml`
- Repository state: `git_status.txt`, `working_tree.patch`
- System snapshot: `system_setup.txt`

## Test setup

- NVIDIA Jetson AGX Orin Developer Kit, Jetson Linux R36.5.0
- 12-core ARM Cortex-A78AE, 61 GiB RAM, `MODE_50W`
- ROS 2 Humble; Python 3.10.12
- `ros2 launch rsg rsg_all.launch.py`
- `ros2 bag play ~/datasets/uhumans2/office_s1_00h_v2 --rate 1 --qos-profile-overrides-path ~/.tf_overrides.yaml`
- Requested playback duration: 180 seconds
- NanoSAM: CUDA/TensorRT, full 640×480, 3×3 prompts, max 8 masks, NMS IoU 0.55
- RAP/VLM asynchronous; generic RAP memory cleared before the run

## Results

- Received: 2,809 frames
- Reached Hydra: 239 frames (8.51%)
- Dropped: 2,570 unique frames (91.49%)
- Processing failures: 0
- Total delay: mean 743.077 ms; median 726.837 ms; p95 962.189 ms; max 1,381.612 ms
- NanoSAM inference: mean 328.441 ms; p95 393.383 ms; max 475.044 ms
- Crop update: mean 135.032 ms; p95 307.942 ms
- Geometry metadata: mean 84.395 ms; p95 141.786 ms
- Frame assignment: mean 68.523 ms; p95 105.033 ms, with 439.794 ms maximum

The measured stages reconcile to the total; no substantial synchronous delay
is hidden between flags. Frame-assignment mean increased from 39.2 ms in the
first trace quartile to 100.3 ms in the fourth, indicating track-registry growth,
while NanoSAM remained the largest stable per-frame component.

## Decision

By explicit user decision, NanoSAM remains untouched. Part 1 will optimize the
largest remaining synchronous leaf stage: representative-crop maintenance.
The first fixes remove a discarded RGB context-copy, restrict exact mask cleanup
to the target ROI, and then defer byte-equivalent RAP/VLM crop rendering to the
asynchronous workers. Geometry and global assignment remain future sections.

No accuracy conclusion can be drawn from timing alone. Zero runtime failures
confirms execution stability, not segmentation/tracking equivalence.
