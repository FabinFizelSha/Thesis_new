# RSG Phase 1 latency optimisation — master report

This is the single top-level, continuously-updated record of the whole
optimisation campaign. It exists alongside exactly two other documentation
levels, and nothing else:

```
debug/optimisation/OPTIMISATION_REPORT.md                          <- this file (whole campaign)
debug/optimisation/optimisation_partN/PARTN_REPORT.md               <- one per part (one synchronous section)
debug/optimisation/optimisation_partN/optimisationN_<session>/SESSION_REPORT.md  <- one per controlled run
```

Update this file whenever a part opens or closes. Update a `PARTN_REPORT.md`
whenever a session inside that part completes. Write a `SESSION_REPORT.md`
for every controlled run. Never overwrite an earlier session folder or an
earlier part's closed report; append/extend instead.

## Campaign objective

Minimize synchronous Phase 1 latency — every frame's time from arrival at the
Phase 1 coordinator (`frame_callback`) to publication to Hydra
(`_publish_hydra_from_result` completing) — without reducing object-tracking,
fuser, RAP, or VLM accuracy. Work proceeds one major synchronous section at a
time (one "part"), each with a controlled before/after measurement and full
evidence retention, so the sequence of changes and their measured effect form
a defensible thesis record.

## Non-negotiable constraints (apply to every part)

1. **NanoSAM is completely frozen.** No change to inference, resolution,
   TensorRT engines, prompts, thresholds, mask cap, or any other model
   behavior, in any part.
2. **No accuracy regression.** Object-tracking, fuser behavior, label-map
   behavior, Hydra output correctness, and RAP/VLM accuracy must not
   regress. Timing evidence alone never proves accuracy is preserved —
   pair every behavior-relevant change with an explicit output-equivalence
   test (byte-identical output, unchanged geometry values, etc.).
3. **RAP/VLM are asynchronous** and excluded from synchronous bottleneck
   ranking, but no change may reduce their input quality.
4. **One major synchronous section per part.** Do not mix geometry,
   assignment, message construction, or other major sections into one part.
   A different major section starts a new, separately numbered part.
5. **Evidence is permanent.** Every controlled run gets its own timestamped
   session folder. Never overwrite, delete, or reuse an earlier session
   folder, and never edit a closed part's report to change its historical
   numbers — only append new findings.
6. **Low-overhead diagnostics only.** Timing uses `time.perf_counter()`,
   buffers rows in RAM, and writes one plain CSV at clean shutdown. No XLSX,
   no per-frame disk I/O, no custom diagnostic YAML or launch file — only
   `config/rsg_pipeline.yaml` and `ros2 launch rsg rsg_all.launch.py`.
7. **The user launches the pipeline and bag manually.** The assistant's role
   after a run is: inspect the generated session, update the relevant
   `SESSION_REPORT.md`/`PARTN_REPORT.md`/this file, explain the bottleneck's
   root cause, propose or implement the next authorized change, verify with
   `pytest`/`colcon build`, and hand back exact copy-paste commands for the
   next run.
8. **Diagnostic instrumentation code is temporary** — kept only until the
   optimisation campaign and thesis evidence are complete, then removed as a
   final cleanup step.
9. **Generic RAP memory only.** Clear `~/rsg_rap_memory` and
   `debug/phase1_rap_memory.jsonl` before every run. Never create a per-run
   or per-part RAP memory.
10. **The worktree is otherwise dirty with unrelated user-owned changes.**
    Every command scopes to RSG files relevant to this optimisation; nothing
    outside it is touched, reset, or cleaned.

## Standard test protocol (identical for every part unless a part's report
says otherwise)

Hardware/software (see each session's `system_setup.txt` for the exact
captured snapshot):

- NVIDIA Jetson AGX Orin Developer Kit, 12-core ARM Cortex-A78AE, 61 GiB RAM /
  30 GiB swap, `MODE_50W`, Jetson Linux R36.5.0, kernel `5.15.185-tegra`
- ROS 2 Humble, Python 3.10.12
- Bag: `~/datasets/uhumans2/office_s1_00h_v2`, `--rate 1`,
  `--qos-profile-overrides-path ~/.tf_overrides.yaml`, exactly 180 seconds

Before every run:

```bash
cd ~/rsg_ros2_ws
mkdir -p ~/rsg_rap_memory
find ~/rsg_rap_memory -mindepth 1 -delete
rm -f ~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl

cd ~/rsg_ros2_ws/src/rsg
python3 -m pytest -q tests -p no:anyio   # -p no:anyio works around a local venv/pytest-plugin mismatch, unrelated to RSG code

cd ~/rsg_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select rsg
```

Terminal 1 — launch:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rsg rsg_all.launch.py
```

Terminal 2 — play exactly three minutes:

```bash
timeout --signal=INT --kill-after=15s 180s \
  ros2 bag play ~/datasets/uhumans2/office_s1_00h_v2 \
  --rate 1 \
  --qos-profile-overrides-path ~/.tf_overrides.yaml
```

Stop Terminal 1 cleanly with **Ctrl+C** after the `timeout` in Terminal 2
finishes (exit status 124 is expected) — the CSV is written only at clean
shutdown. Then locate and analyze the newest session (path/analyser script
differ per part — see that part's `PARTN_REPORT.md`):

```bash
ls -dt debug/optimisation/optimisation_partN/optimisationN_* | head -n 1
python3 debug/optimisation/optimisation_partN/analyse_timing.py \
  debug/optimisation/optimisation_partN/optimisationN_SESSION_ID/phase1_timing.csv
```

Every session folder retains: `phase1_timing.csv` (raw buffered trace),
`stage_summary.csv`/`stage_summary.md` (generated statistics),
`SESSION_REPORT.md` (purpose/results/decision), `rsg_pipeline_snapshot.yaml`
(exact config used), `git_status.txt` + `working_tree.patch` (repository
state), `system_setup.txt` (hardware/software snapshot).

## Campaign start point (never changes — the pre-optimisation baseline)

- Session: `optimisation_part1/optimisation1_20260806_011841`
- End-to-end latency mean/median/p95/max: **743.077 / 726.837 / 962.189 /
  1,381.612 ms**
- Classifier (Phase 1 processing) latency mean/median/p95/max: 683.766 /
  673.546 / 861.458 / 1,316.400 ms
- Processed 239 of 2,809 received frames (8.51%) in the 180 s window
- NanoSAM inference: 328.441 ms mean (44.2% of total) — frozen for the whole
  campaign
- This is the number every part's improvement is ultimately measured against.

## Campaign current point (update after every accepted change, in any part)

- **As of Part 1 close (session `optimisation1_20260806_015126`):**
  end-to-end latency mean/median/p95/max **602.392 / 603.160 / 813.911 /
  1,257.563 ms**; classifier mean 541.846 ms.
- Cumulative change from campaign start: **-140.685 ms mean (-18.93%)**,
  **-148.278 ms p95 (-15.41%)**, throughput +26.78% more frames processed in
  the same 180 s window.
- Part 2 (geometry metadata) closed with no accepted change — one fix was
  tried, measured to regress performance, and reverted (confirmed clean).
  Net effect on the current point: zero. See
  `optimisation_part2/PART2_REPORT.md` "End point".
- **As of Part 3's accepted row-init fix (session `optimisation3_20260806_220604`):**
  end-to-end latency mean/median/p95/max **589.961 / 579.772 / 744.811 /
  1,119.836 ms**; classifier mean 530.069 ms. Part 3 remains open (the
  tail-latency question / Path B is still unresolved), but this specific
  change is accepted and verified, so the current point moves to reflect it.
- Cumulative change from campaign start: **-153.116 ms mean (-20.61%)**.
  Note this total mixes two durable, verified code-change deltas (Part 1's
  -140.685 ms, Part 3's row-init fix) with ordinary `sam_inference_ms`
  run-to-run noise on this hardware (see "Measurement caveats" below) — a
  fresh run right now would likely land within roughly ±20-30 ms of
  589.961 ms purely from that noise, independent of any further code
  change.

## Measurement caveats (apply to every part's comparisons)

- **`sam_inference_ms` is not perfectly stable run-to-run** on this Jetson
  even with zero NanoSAM-relevant code changes between two sessions —
  observed +9.17% (+28.3ms mean) between two back-to-back sessions with only
  a profiling side-channel added elsewhere in the code
  (`optimisation_part1/optimisation1_20260806_015126` vs
  `optimisation_part2/optimisation2_20260806_022259`). Likely causes: Jetson
  thermal state, background system load, or TensorRT/CUDA warm-up
  differences between separate `ros2 launch` invocations. Because Phase 1's
  pipeline is SAM-bound with `request_queue_size: 1` and drop-oldest, this
  also secondarily perturbs frames-processed and every downstream latency
  number by a small amount.
- **Consequence:** evaluate a part's own fix primarily by the specific
  targeted stage's delta (e.g. `geometry_projection_ms` for Part 2) and by
  `classifier_delay_ms - sam_delay_ms` (the non-SAM portion of processing),
  not by raw `total_delay_ms` alone, which always carries some SAM-driven
  noise on top of whatever the actual code change did. Always state this
  attribution explicitly in the session's `SESSION_REPORT.md` rather than
  reporting a raw total-latency delta as if it were solely caused by the
  change under test.
- **This campaign's controlled runs measure latency, not tracking
  accuracy.** The standard protocol plays the bag at `--rate 1` (real-time),
  which the pipeline cannot keep up with — every session so far has dropped
  85-90% of received frames. That's fine for latency measurement (each
  *processed* frame's cost doesn't depend on which frames were dropped), but
  it is **not** a valid setup for judging object-tracking accuracy: sparse,
  non-contiguous frame sampling can produce apparent tracking gaps or
  mismatches that reflect the playback rate, not a code defect. No accuracy
  conclusion should be drawn from these sessions. A future accuracy
  evaluation of any change from this campaign needs a separately configured
  run (slower rate, or a rate the pipeline can actually keep up with, giving
  a much lower drop ratio) — not reuse of these timing sessions.

## Master progress table

| Part | Section targeted | Status | Sessions | Start mean (ms) | End mean (ms) | Δ mean | Δ % | Report |
|---|---|---|---|---:|---:|---:|---:|---|
| 1 | Representative-crop maintenance | **Closed — accepted** | `011841` (baseline), `015126` (iter. 1) | 743.077 | 602.392 | -140.685 ms | -18.93% | [PART1_REPORT.md](optimisation_part1/PART1_REPORT.md) |
| 2 | Geometry metadata estimation | **Closed — no accepted change (net zero)** | `022259` (profiling), `192701` (Step 2, reverted), `202114` (revert verified) | 602.392 | 602.392 (605.903 measured, within noise) | 0.000 ms | 0.00% | [PART2_REPORT.md](optimisation_part2/PART2_REPORT.md) |
| 3 | Frame assignment (persistent-track association) | **Open — one improvement accepted; tail-spike cause identified (`_candidate_track_ids` full-track scan), lock contention ruled out** | `204104`, `205816` (profiling), `220604` (accepted fix), `222959` (Path B, conclusive) | 602.392 | 589.961 (part not closed) | -12.431 ms | -2.06% | [PART3_REPORT.md](optimisation_part3/PART3_REPORT.md) |

## Next candidates after Part 3 (not yet started, tracked for planning only)

- **Result message construction** (`result_message_build_delay_ms`): 32.071
  ms mean — Phase 1 currently builds an intermediate ROS message, serializes
  to JSON, then re-parses/converts parts of it again for Hydra output. A
  structural, not incremental, fix.
- **Hydra depth filtering** (`hydra_depth_filter_ms`): 7.284 ms mean — minor,
  round-trips ROS image messages back to arrays after label-message
  construction.

## Document map

- [optimisation_part1/PART1_REPORT.md](optimisation_part1/PART1_REPORT.md) —
  closed, crop-maintenance section, accepted.
- [optimisation_part2/PART2_REPORT.md](optimisation_part2/PART2_REPORT.md) —
  closed, geometry-metadata section, no accepted change (net zero;
  instrumentation kept, one fix tried and correctly reverted).
- [optimisation_part3/PART3_REPORT.md](optimisation_part3/PART3_REPORT.md) —
  open, frame-assignment section, profiling step in progress.
- Each part folder's session subfolders (e.g.
  `optimisation_part1/optimisation1_20260806_011841/`) hold that run's raw
  evidence and `SESSION_REPORT.md`.
