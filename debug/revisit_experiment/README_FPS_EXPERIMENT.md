# Revisit Experiment — Diagnostic Setup

Status: **closed, at the revisit-accuracy result only** — 5 sessions run
2026-09-09, results and analysis in `EXPERIMENT_PART1_REVISIT_ACCURACY.md`.
The frame-throughput/FPS question this tooling could also measure (see
below) is explicitly **not** part of this experiment or its conclusion — it
will be its own separate experiment, run and reported independently, at a
higher bag playback rate than the 0.1x used here. This document's tooling
and per-session procedure remain valid for that future experiment; nothing
here implies it is scheduled or in progress.

## What this measures

Consecutive sessions of the same fixed-duration bag segment, each resuming
from the previous session's saved state (`resume_reset_trajectory` at its
default `true` — see `IMPLEMENTATION.md` Section 5, "Scenario B: next-day
revisit"). This experiment's result (`EXPERIMENT_PART1_REVISIT_ACCURACY.md`):
as the pipeline re-encounters objects it has already mapped, it stops
minting new tracks for them, resolving them as revisits instead — confirmed
at 97.5% accuracy across four resumed sessions.

A second question — whether fewer not-yet-labelled objects (fewer VLM
calls) frees GPU time for SAM and raises frame throughput on later sessions
— was tracked by the same tooling but is **not concluded here**. This
dataset showed VLM calls falling sharply without FPS moving, at `--rate
0.1`; `EXPERIMENT_PART1_REVISIT_ACCURACY.md` Section 6 found the pipeline
was already compute-saturated (not input-starved) even at that rate, so the
flat FPS result doesn't confirm or disprove the throughput question either
way. That question is left to the separate follow-up experiment mentioned
above.

Per run, this setup captures:

1. **Object/track count** — Phase 1's `track_count` and Hydra's object-node
   count, so a flat count across runs 2-4 is directly visible.
2. **VLM call count** — every completed call to the object-labelling VLM,
   split success/fail.
3. **Frame throughput (FPS)** — Phase 1's real per-frame processing rate,
   plus its frame-drop count (a second, independent signal of GPU contention:
   drops rise when Phase 1 falls behind).
4. **Per-frame latency (input to Hydra output)** — average/median/max
   milliseconds from a frame arriving at Phase 1 to Phase 1 finishing
   publishing it to Hydra (`total_delay_ms` on the existing `frame_trace`
   row), plus how much of that is spent in the classifier stage
   (SAM + association) versus pure SAM inference. This is the more direct
   answer to "does a frame move through Phase 1 faster on a later run" —
   FPS can be flat while this still moves, or vice versa, if frame arrival
   is the bottleneck rather than processing time (see the note on the first
   two runs below).
5. **CPU/GPU load** — sampled directly from this Jetson's own `tegrastats`
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

## Lessons from the manual-label attempts, and the fix

Three attempts hit the same class of problem, in order: (1) two runs done
back to back with no archiving at all -- run 1's track/object counts were
gone by the time anyone thought to check, since `memory/tracker/` and
`memory/hydra/backend/dsg.json` get overwritten every run; (2) a
`--run-label runN` typed literally instead of a real number, twice, because
the checklist's placeholder text got copy-pasted as-is. Every case is the
same root cause: a manual step, at a specific moment, easy to forget or
mistype.

**`run_session.py` removes the manual step.** One command, no label to
choose, run once per launch. It waits for the pipeline to be up, samples
resources the whole time, detects the pipeline exiting on its own (instead of
requiring a manually-timed follow-up command), waits for the shutdown save to
settle, and archives automatically into its own auto-generated, never-reused
`session_<timestamp>` directory. This is the primary way to run this
experiment now; `monitor_resources.py` and `snapshot_run.py` still work
standalone (documented further below) if you want to run either step by hand.

## Quick checklist — the recommended way

```
[ ] (first run of a fresh baseline only) python3 clear_memory.py
[ ] terminal A:  ros2 launch rsg rsg_all.launch.py start_risk_vlm:=false
[ ] terminal B:  python3 debug/revisit_experiment/run_session.py
[ ]              -- note the printed session id, or read it back later --
[ ] terminal C:  <your bag-play command with the 300s timeout>
[ ]              -- when the bag stops, Ctrl+C terminal A and wait --
[ ]              -- terminal B detects the exit and archives on its own --
[ ] confirm the printed summary has real numbers, not blank/None, before starting the next run
```

Terminal B does not need a Ctrl+C at all in the normal case -- it stops
itself once it sees both `hydra_ros_node` and `rsg_phase1_semantic_coordinator`
have exited. Repeat for as many runs as the test calls for; give me the
printed session ids (or point at
`debug/revisit_experiment/sessions/summary_all_sessions.csv`, which has all
of them) for a full test cycle.

## Per-run procedure (manual fallback, one script per step)

The section below documents `monitor_resources.py` + `snapshot_run.py` run by
hand, kept for reference and for anyone who wants to control each step
themselves. `run_session.py` above does the same thing automatically and is
the recommended path.

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

## What you'll have after N runs

Using `run_session.py` (recommended), everything lands under
`debug/revisit_experiment/sessions/<session_id>/`; using the manual scripts,
under `debug/revisit_experiment/runs/<label>/`. Same file layout either way:

- `sessions/summary_all_sessions.csv` (or `runs/summary_all_runs.csv`) — one
  row per run, every headline metric in one file. This is what a
  track-count-vs-run, VLM-calls-vs-run, or FPS-vs-run plot reads directly.
- `<session_id>/resource_usage.csv` — one row per second of
  GPU%/CPU%/RAM/temperature/power for that run, for a load-over-time plot
  within or across runs.
- `<session_id>/phase1_timing.csv` — full per-frame and per-VLM-call detail
  (SAM inference ms, classifier delay, etc.) if the headline numbers need to
  be broken down further.
- `<session_id>/{phase1_tracker_state.json,hydra_dsg.json}` — the raw state
  each run actually produced, in case a specific object needs tracing back
  by hand.

Neither directory is tracked in git (see `.gitignore`) — large and
regeneratable by re-running the bag with the same tooling.

## Results

**This experiment is closed.** Full results, the five-session data table,
the recurring-miss analysis, and the closing notes are in
`EXPERIMENT_PART1_REVISIT_ACCURACY.md` — not duplicated here. Headline:
97.5% of re-encountered objects across sessions 2-5 correctly resolved as
revisits rather than new tracks. `vlm_call_count` fell monotonically
(28→11→5→2→1) while FPS/latency stayed flat; that flat result does not
confirm or disprove the throughput half of the original hypothesis — see
that document's Section 6 for why (the pipeline was already
compute-saturated at `--rate 0.1`, not input-starved as first assumed).

**The frame-throughput question is a separate future experiment**, at a
higher bag rate, not a "Part 2" of this one. If and when it happens, its own
results document should follow the same structure (objective, hypothesis,
setup, results, analysis, conclusion) so it is ready to drop into a thesis
experiments chapter with minimal rewriting — but it is independent of the
result this experiment closed with.
