# Revisit / FPS Experiment — Diagnostic Setup

Status: implementation ready, 2026-09-09. No runs have been executed yet — this
is the setup and runbook. Results and analysis are a separate document once
the 4 runs are done.

## What this measures

Four consecutive runs of the same fixed-duration bag segment, each resuming
from the previous run's saved state (`resume_reset_trajectory` at its default
`true` — see `IMPLEMENTATION.md` Section 5, "Scenario B: next-day revisit").
The hypothesis under test: as the pipeline re-encounters objects it has
already mapped, it should stop minting new tracks for them (object count
plateaus), which means fewer objects need a VLM call for labelling (VLM call
count drops), which frees GPU time for SAM (Phase 1's own frame throughput —
FPS — should rise).

Per run, this setup captures:

1. **Object/track count** — Phase 1's `track_count` and Hydra's object-node
   count, so a flat count across runs 2-4 is directly visible.
2. **VLM call count** — every completed call to the object-labelling VLM,
   split success/fail.
3. **Frame throughput (FPS)** — Phase 1's real per-frame processing rate,
   plus its frame-drop count (a second, independent signal of GPU contention:
   drops rise when Phase 1 falls behind).
4. **CPU/GPU load** — sampled directly from this Jetson's own `tegrastats`
   (not `nvidia-smi` — confirmed on this hardware that reports no usable
   figures on Jetson/Tegra).

## One-time config, already set

- `phase1.risk_vlm.enabled: false` (`rsg_pipeline.yaml`) — a second VLM
  server sharing the same GPU would confound the "object-VLM calls decline
  -> GPU load drops -> SAM speeds up" measurement. This experiment is meant
  to sit *before* the risk-VLM experiment in the report; re-enable this for
  that one.
- `phase1.persistent_tracking.session_persistence.enabled: true` — required
  for anything to carry over between runs at all.
- `hydra.enable_object_merging` / `resume_reset_trajectory` — both at their
  documented defaults (`true`), no change needed. See `IMPLEMENTATION.md`.
- `phase1.performance.measure_timing` / `write_timing_csv` — both already
  `true`. Three new columns were added to the existing per-frame/per-VLM-call
  CSV specifically for this experiment (`wall_clock_unix_sec` on the
  `frame_trace` and VLM-result rows) — no config change, already live on
  `phase1.py`'s next process start (symlink-installed, no rebuild needed).

**Launch with the risk VLM server not even started**, not just disabled in
config — saves it competing for GPU memory at all:

```bash
ros2 launch rsg rsg_all.launch.py start_risk_vlm:=false
```

## Per-run procedure

Repeat this block **four times**. Two terminals per run (three if you also
watch logs live), the same each time except the memory-clear step, which is
**run 1 only**.

### Run 1 only — start from a clean baseline

```bash
python3 clear_memory.py
```

Confirms nothing is running and wipes `memory/{tracker,hydra,rap}`. Refuses
if a pipeline process is still alive — see its own `--help`.

### Every run — terminal A: the pipeline

```bash
ros2 launch rsg rsg_all.launch.py start_risk_vlm:=false
```

### Every run — terminal B: the resource monitor, started right after A comes up

```bash
python3 debug/revisit_experiment/monitor_resources.py --run-label run1
```

(`run2`, `run3`, `run4` on the later runs.) Prints a live gpu%/cpu% line
every 10 samples so you can see it's alive. Leave it running.

### Every run — terminal C (or reuse B once the bag exists): play the bag

Your own bag-play command, timed to ~300s as you already do. When it ends
(or you stop it), leave the pipeline running a few more seconds to let any
in-flight VLM calls finish, then:

### Shut down cleanly, in this order

1. **Ctrl+C in terminal B** (the resource monitor) — or let it run a moment
   longer; it does not need to stop before A.
2. **Ctrl+C in terminal A** (the pipeline). Phase 1 and Hydra both save their
   state on clean shutdown — **wait for the shell prompt to actually return**
   (or watch for both `hydra_ros_node` and `rsg_phase1_semantic_coordinator`
   to report finished) before doing anything else. Snapshotting before the
   save has actually finished writing will copy an incomplete or stale file.

### Then, before touching terminal A again: archive this run's data

```bash
python3 debug/revisit_experiment/snapshot_run.py --run-label run1
```

Prints a summary to the terminal immediately — track count, object count,
VLM calls, FPS, average GPU/CPU — so you can sanity-check each run as you go
rather than discovering a problem only after all four are done. Writes
`debug/revisit_experiment/runs/run1/` (tracker state, Hydra DSG, the timing
CSV, the resource CSV, and `summary.json`), and appends one row to
`debug/revisit_experiment/runs/summary_all_runs.csv`.

**Do not run `clear_memory.py` between runs 2, 3, and 4** — the entire point
is that each one resumes from the previous run's save.

### Then start the next run

Back to terminal A, same launch command, same `run2`/`run3`/`run4` label in
the other two scripts.

## What you'll have after 4 runs

- `debug/revisit_experiment/runs/summary_all_runs.csv` — one row per run,
  every headline metric in one file. This is what a track-count-vs-run,
  VLM-calls-vs-run, or FPS-vs-run plot reads directly.
- `debug/revisit_experiment/runs/run<N>/resource_usage.csv` — one row per
  second of GPU%/CPU%/RAM/temperature/power for that run, for a
  load-over-time plot within or across runs.
- `debug/revisit_experiment/runs/run<N>/phase1_timing.csv` — full per-frame
  and per-VLM-call detail (SAM inference ms, classifier delay, etc.) if the
  headline numbers need to be broken down further.
- `debug/revisit_experiment/runs/run<N>/{phase1_tracker_state.json,hydra_dsg.json}`
  — the raw state each run actually produced, in case a specific object needs
  tracing back by hand.

## Ideal outcome, restated precisely

- `track_count` and `hydra_object_node_count`: run 1 establishes the map;
  runs 2-4 should show **no growth** (per `IMPLEMENTATION.md` Section 6 — this
  is the same revisit-accuracy question, measured here as a side effect while
  measuring FPS).
- `vlm_call_count`: high on run 1 (everything is new), falling toward run 4
  (most objects already labelled from a previous run and never re-dispatched
  — see `IMPLEMENTATION.md` Section 2.3, restored tracks are never re-sent to
  the VLM).
- `avg_fps`: rising run 1 -> run 4, inversely tracking `vlm_call_count` and
  `avg_gpu_pct` — less VLM inference contending for the GPU should leave more
  of it for SAM.
- `frame_drop_count`: should fall alongside `avg_gpu_pct`, as a second,
  independent signal of the same effect.

Any run where `track_count` grows is evidence against the hypothesis and
should be looked at first — cross-reference against
`debug/revisit_experiment/IMPLEMENTATION.md` Section 6's discussion of the
`global_centroid_pass_m` gate as the leading suspect.
