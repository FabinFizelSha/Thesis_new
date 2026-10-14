# Phase 1 optimisation Part 5 report

Level 2 of 3 in the campaign documentation hierarchy — see
[../OPTIMISATION_REPORT.md](../OPTIMISATION_REPORT.md) for the whole-campaign
record, global non-negotiable constraints, and the standard test protocol
this part follows. Level 3 (per-session) reports will live in each
`optimisation5_<timestamp>/SESSION_REPORT.md` below as runs complete.

This part investigates the contention Part 4 measured (every stage's mean
cost rose when segmentation and tracking/publish ran concurrently, more
than a GIL-release model predicted) via **two candidate causes, tested as
separate steps within this one part** — the same structure Part 3 used for
its Path A / Path B investigation, rather than spawning a new numbered part
per hypothesis. Both steps are on the same throughput axis Part 4 opened;
neither is a per-stage-cost change like Parts 1-3.

## Objective

Identify and, if possible, remove the cause of Part 4's measured thread
contention, without changing NanoSAM, geometry, tracking/assignment logic,
or Hydra output — same throughput axis as Part 4, still targeting more
frames processed per window, not per-stage compute cost.

## Start point

- Inherited from: Part 4's accepted end point
  (`../optimisation_part4/PART4_REPORT.md` "End point").
- Reference session: `optimisation_part4/optimisation4_20260807_004917`
  (Run 1, accepted).
- Processing ratio: **13.53%** (373/2,757 processed).
- Mean `total_delay_ms`: 727.406 ms; mean `classifier_delay_ms`: 676.062 ms.
- Mean `sam_inference_ms`: **436.291 ms** (vs. 316.303 ms when segmentation
  ran alone in the old serial design — +37.94% under concurrency).
- Mean `geometry_metadata_ms`: 101.858 ms, `frame_assignment_ms`: 74.013 ms
  (vs. 85.150 ms / 64.016 ms alone — +19.62% / +15.62% under concurrency).
- Mean `sam_output_queue_wait_ms`: 0.704 ms — the tracking/publish thread
  has roughly 228 ms of slack per frame before it would become the
  throughput-limiting stage instead of SAM.

## End point

**Closed — both steps tested, neither adopted, root cause undetermined
(user-confirmed 2026-08-07).** Step 1 (CPU core affinity): confirmed null
result across two runs, code subsequently **removed** (not just left
disabled) — see "Final findings" for the portability reasoning. Step 2
(`jetson_clocks`/power mode): one run showed a small, directionally
positive but statistically inconclusive effect; closed without a
confirmatory second run since the investigation's priority shifted. Part
4's accepted trade-off (+23.51% throughput, +23% mean latency) stands as
final; this part's contribution is negative results, honestly recorded, not
a fix.

## Hypotheses (two candidate causes, tested as separate steps)

Part 4's contention was measured, not explained — every stage's mean got
worse when the two threads ran concurrently, more than a "SAM releases the
GIL almost entirely" model predicted.

1. **CPU scheduling/cache contention** — the two threads can be scheduled
   onto the same physical cores, and/or migrate between cores, causing cache
   thrashing and scheduler overhead that a single-threaded design never hit.
   → **Step 1**, below. Closed — null result.
2. **Clock/thermal effects** — this Jetson's CPU governor is `schedutil`
   (dynamic frequency scaling), and `jetson_clocks` is not confirmed active
   (per Part 1's `system_setup.txt`, this was never checked). Orin shares a
   power/thermal budget across CPU, GPU, and memory controller; concurrent
   load may prevent the DVFS governor from boosting clocks as aggressively
   as it did for one thread at a time. → **Step 2**, below. Not yet run.

Testing both at once would conflate their effects, which the campaign's
discipline (one attributable change per iteration) rules out — hence two
sequential steps, not one combined run.

---

## Step 1 — CPU core affinity (closed — null result)

### What changed

**CPU core affinity for the two Phase 1 threads** —
`Phase1SemanticCoordinator._apply_thread_cpu_affinity()` in `nodes/phase1.py`,
called once at the top of `_segmentation_loop()` and
`_tracking_publish_loop()`. Uses `os.sched_setaffinity(0, cpu_set)`, which
sets the *calling* thread's affinity on Linux when called from within that
thread — not the whole process, so RAP/VLM threads and the main thread are
unaffected.

Default core split (12-core Orin AGX), set in `config/rsg_pipeline.yaml`:

- `segmentation_thread_cpus: [0, 1, 2, 3, 4, 5]` — 6 cores. SAM is ~2x
  tracking/publish's per-frame cost (458.65 ms vs. 231.07 ms at the Part 4
  start point), so it gets a larger, but not exclusive, core allocation.
- `tracking_publish_thread_cpus: [6, 7, 8, 9]` — 4 cores, fully disjoint.
- Cores 10-11 left unpinned for the main thread, RAP/VLM worker threads,
  ROS 2/DDS, and general OS/kernel work.

**Why not pin RAP/VLM threads too**: they're already asynchronous,
lower-frequency (one settled track per job, not one call per frame), and
excluded from this campaign's synchronous bottleneck ranking per the global
constraints. Pinning them adds complexity without a measured problem to fix.

### Verification before requesting a run

- Confirmed via a standalone script on this exact host that
  `os.sched_setaffinity(0, {...})` called from inside a `threading.Thread`
  target pins only that thread — `os.sched_getaffinity(0)` from a second
  thread and the main thread were unaffected.
- Added kernel-visible thread naming (`prctl(PR_SET_NAME)` →
  `phase1-seg`/`phase1-track`) and affinity-readback logging
  (`os.sched_getaffinity()` after `os.sched_setaffinity()`) so pinning can
  be confirmed externally via `/proc/<pid>/task/<tid>/{comm,status}`,
  independent of trusting scrolled log output.
- `python3 -m pytest -q tests -p no:anyio` → **35/35 pass**;
  `colcon build --packages-select rsg` → succeeds.
- New config is fail-safe by construction: `cpu_affinity_enabled` defaults
  to `false`, and an invalid/out-of-range core list logs a warning and skips
  pinning rather than raising.

### Run registry — Step 1

#### Run 1 — first measurement, pinning not directly confirmed at the time (`optimisation5_20260807_012626`)

- Received/processed/dropped/failed: 2,899 / 385 / 2,512 / 0.
- Processing ratio: 13.28% (vs. Part 4's 13.53% — essentially flat).
- `sam_inference_ms` 438.545 ms (vs. 436.291 ms unpinned, +0.52%) —
  effectively no change; did not recover toward the 316.303 ms
  serial-mode value.
- `geometry_metadata_ms` 103.686 ms (+1.79%), `frame_assignment_ms`
  71.809 ms (-2.98%) — both within normal run-to-run noise.
- Anomaly noted: `sam_output_queue_wait_ms` max spiked to 85.374 ms (mean
  stayed tiny at 0.891 ms) — far above Part 4's 4.453 ms max. Single-frame
  outlier, not investigated further.
- At the time, whether affinity had actually been applied was not directly
  verified (no external confirmation beyond the log line, which was not
  checked) — user correctly flagged this as a gap before trusting the null
  result, given how close the numbers were to unpinned Part 4.

#### Verification step — confirmed active (code change, no run)

Live-verified against the running node:

```
phase1-seg   (tid=458511): CPUs=0-5
phase1-track (tid=458512): CPUs=6-9
```

Confirms Run 1's pinning was in fact active as configured — the null result
was real, not a silent no-op.

#### Run 2 — confirmed-active pinning, corroborates Run 1 (`optimisation5_20260807_013627`)

- Received/processed/dropped/failed: 2,879 / 385 / 2,491 / 0.
- One `ros2 bag play` "Message queue starved" warning at ~178.8s into the
  180s window — a single late warning right as the `timeout` signal was
  about to fire, not a sustained/repeated starvation like the deleted
  Part 4 Attempt 0. Tail-frame `total_delay_ms` values (724-1240 ms) are
  consistent with normal tail variance, not a corrupted trace, and this
  run's numbers closely match Run 1's independent (differently-timed) run
  — treated as valid evidence, with this caveat recorded rather than
  silently omitted.
- Processing ratio: 13.37% (vs. Part 4's 13.53%, vs. Run 1's 13.28%).

| Metric | Part 4 (unpinned) | Run 2 (pinned, confirmed) | Change |
|---|---:|---:|---:|
| Processing ratio | 13.53% | 13.37% | -0.16pp (-1.18%) |
| `sam_inference_ms` | 436.291 ms | 432.900 ms | -0.78% |
| `geometry_metadata_ms` | 101.858 ms | 102.480 ms | +0.61% |
| `frame_assignment_ms` | 74.013 ms | 71.351 ms | -3.60% |
| `total_delay_ms` mean | 727.406 ms | 721.827 ms | -0.77% |
| `classifier_delay_ms` mean | 676.062 ms | 671.961 ms | -0.61% |

Every delta is under 4% — inside the campaign's own established
run-to-run noise band (Part 2 measured ±9% on `sam_inference_ms` alone
between two back-to-back sessions with zero relevant code change).
**Statistically indistinguishable from unpinned Part 4, replicated across
two independent runs, one with pinning externally confirmed active.**
`sam_output_queue_wait_ms` max was again elevated (28.245 ms) vs. Part 4's
4.453 ms, though smaller than Run 1's 85.374 ms spike — consistent
direction, not conclusive on its own.

### Step 1 result: CLOSED — null result

CPU core affinity was implemented correctly and confirmed genuinely active
at runtime — via kernel thread naming and `/proc` inspection, not just log
trust — across two independent 180 s runs. Both show every stage's mean
within ~4% of unpinned Part 4, well inside this campaign's established
noise band. **CPU scheduling/cache-locality contention is not confirmed as
the cause of Part 4's measured slowdown.** Pinning to disjoint cores does
not recover `sam_inference_ms`, `geometry_metadata_ms`, or
`frame_assignment_ms` toward their pre-Part-4 serial-mode values, and does
not move the processing ratio beyond noise.

One secondary, unresolved observation: `sam_output_queue_wait_ms`'s *max*
was elevated in both pinned runs (85.374 ms, 28.245 ms) versus Part 4's
4.453 ms, while the *mean* stayed similarly tiny in all three runs. Not
investigated further — noted in case it recurs.

**Decision (user-confirmed 2026-08-07): close Step 1 as a documented null
result; continue investigating within this part via Step 2, not a new
part** (folding the originally-separate "Part 6" plan back in — jetson_clocks
is a second hypothesis for the same Part-4 contention question, not a
different major section, matching how Part 3 ran Path A/Path B as steps
within one part rather than separate parts). Affinity code was initially
kept enabled (config-gated, harmless, zero measured downside) — **superseded
by a later decision to remove it entirely; see "Final findings."**

---

## Step 2 — `jetson_clocks` / power mode (in progress, no code change)

### Why this step has no code diff

Unlike Step 1, this step changes nothing in `src/rsg/`. It tests a
system-level configuration — `jetson_clocks` and the Orin's power mode —
external to Phase 1 entirely. There is no diff to review, build, or
unit-test; the "change" is a `sudo` command run once before launch.

### Hypothesis

Orin shares a power/thermal budget across CPU, GPU, and memory controller.
Under `schedutil`, the DVFS governor decides clock speeds dynamically based
on load; when segmentation (GPU + some CPU) and tracking/publish (CPU) run
concurrently instead of sequentially, the *combined* instantaneous load is
higher than either stage alone ever produced in the old serial design, even
though total work per frame is unchanged. If the governor doesn't boost
clocks as aggressively under that combined load — or if there's a shared
thermal/power ceiling being hit — that alone could explain Part 4's
measured slowdown, with no code-level explanation needed at all. Locking
clocks to maximum removes the governor's dynamic response as a variable
entirely. Measured, not assumed — same discipline as every other step.

### What changes, concretely

Nothing in `src/rsg/`. Before the controlled run:

```bash
sudo nvpmodel -q                 # confirm/record current power mode
sudo jetson_clocks --show        # record current vs max clocks (baseline)
sudo jetson_clocks               # lock CPU/GPU/EMC to max
```

`jetson_clocks` with no arguments locks clocks for the current boot session
(not persistent across reboot). No Phase 1 config, launch file, or code
changes — `cpu_affinity_enabled` and the `[0-5]`/`[6-9]` split from Step 1
stay exactly as they are, since Step 1 found them harmless and this step
tests a different, independent variable; combining an untested new variable
with a config change would confound which one caused any observed effect.

### Verification before requesting a run

No `pytest`/`colcon build` step applies — no source files changed. The only
pre-run check is confirming the clock lock actually took effect, since
(like Step 1's affinity) a system-level setting that silently failed to
apply would produce a meaningless null result:

```bash
sudo jetson_clocks --show | grep -i "cpu\|gpu\|online"
```

Compare the `MinFreq`/`MaxFreq`/current-freq fields before and after running
`sudo jetson_clocks` — after locking, current frequency should sit at (or
very near) `MaxFreq` for every online CPU core and the GPU, not fluctuating.

### Test protocol — Step 2

Same standard protocol as Step 1 (including the Part 4
`--read-ahead-queue-size` amendment), with the clock lock as the one new
step before launch:

```bash
cd ~/rsg_ros2_ws
mkdir -p ~/rsg_rap_memory
find ~/rsg_rap_memory -mindepth 1 -delete
rm -f ~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl
cd ~/rsg_ros2_ws/src/rsg && python3 -m pytest -q tests -p no:anyio   # confirms nothing else drifted
cd ~/rsg_ros2_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select rsg

# Step 2's one actual change:
sudo nvpmodel -q
sudo jetson_clocks --show
sudo jetson_clocks
sudo jetson_clocks --show | grep -i "cpu\|gpu\|online"   # confirm locked before proceeding

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
session (same folder and analyser as Step 1 — this step doesn't need a new
one, since there's no new column to add):

```bash
ls -dt debug/optimisation/optimisation_part5/optimisation5_* | head -n 1
python3 debug/optimisation/optimisation_part5/analyse_timing.py \
  debug/optimisation/optimisation_part5/optimisation5_SESSION_ID/phase1_timing.csv
```

**After this run**, consider reverting `jetson_clocks` if the machine isn't
meant to run at locked-max clocks permanently (thermal/power/fan-noise
implications outside this campaign's scope) — `sudo jetson_clocks --restore`
or a reboot returns to the default governor.

### Run registry — Step 2

#### Run 1 — clocks locked and confirmed (`optimisation5_20260807_015857`)

- `jetson_clocks --show` confirmed before/after: all 12 CPU cores went from
  `schedutil` dynamic scaling (729,600-1,497,600 kHz, varying) to pinned at
  `MaxFreq` (1,497,600 kHz), idle states disabled (`WFI=0 c7=0` vs.
  `WFI=1 c7=1`); GPU went from idle (306 MHz) to `MaxFreq` (816 MHz).
- Received/processed/dropped/failed: 2,889 / 402 / 2,485 / 0.
- Processing ratio: **13.91%** (vs. Step 1 close's 13.37%, vs. Part 4's
  original 13.53%).

| Metric | Step 1 close (pinned, unlocked) | Step 2 Run 1 (pinned, locked) | Δ |
|---|---:|---:|---:|
| Processing ratio | 13.37% | 13.91% | +0.54pp (+4.04%) |
| `sam_inference_ms` | 432.900 ms | 422.487 ms | -2.41% |
| `geometry_metadata_ms` | 102.480 ms | 101.322 ms | -1.13% |
| `frame_assignment_ms` | 71.351 ms | 70.881 ms | -0.66% |
| `total_delay_ms` mean | 721.827 ms | 701.177 ms | -2.86% |

Every metric moved in the helpful direction, but all deltas are smaller
than this campaign's own established noise band (±9% on `sam_inference_ms`
alone between two back-to-back runs with zero code/config change) —
suggestive, not conclusive, off a single run.

One more confidently attributable finding: `sam_output_queue_wait_ms`'s
*max* dropped sharply, from 28.245-85.374 ms (both Step 1 runs) to
**2.739 ms**. Plausible specific mechanism: disabling CPU idle states
(`WFI=0 c7=0`) removes core wake-up latency jitter, independent of the
noisier mean-level clock-speed question. Not chased further.

Even at this run's best numbers, `sam_inference_ms` (422 ms) remains ~34%
above the pre-Part-4 serial value (316.303 ms) — clocks do not fully
explain Part 4's contention even if they contribute partially.

### Step 2 result: CLOSED — inconclusive

One run showed a small, directionally-positive, but statistically
inconclusive effect. **Decision (user-confirmed 2026-08-07): close without
a confirmatory second run.** See "Final findings" for the reasoning — the
investigation's practical direction changed (Thor portability), which
supersedes finishing this specific measurement.

---

## Constraints specific to this part

(In addition to the global constraints in `../OPTIMISATION_REPORT.md`.)

- NanoSAM invocation, geometry, tracking/assignment logic, label-map
  construction, and Hydra message construction are byte-for-byte unchanged
  in either step — verify via `num_masks`/mean masks-per-frame against the
  Part 4 start point, same as every prior part's accuracy check.
- One variable per step: Step 1 changed only CPU affinity; Step 2 changes
  only clock/power state. Never combine an untested step with another
  untested change in the same run.
- Success is `sam_inference_ms` and/or
  `geometry_metadata_ms`/`frame_assignment_ms` moving back toward their
  pre-Part-4 (serial, uncontended) values while the processing ratio stays
  at or above Part 4's 13.53%. A processing-ratio regression would need
  explicit judgement (per-frame latency improving at a throughput cost is
  the opposite trade-off from what this campaign is optimizing for).
- One attributable change per iteration; retain every session folder; never
  overwrite an earlier one.

## Source files touched by this part

Step 1 — implemented, then reverted (net effect on `src/rsg/`: none):
- `src/rsg/nodes/phase1.py` — added, then removed,
  `_apply_thread_cpu_affinity()` and `_set_os_thread_name()` helpers and
  their call sites in `_segmentation_loop()`/`_tracking_publish_loop()`;
  the now-unused `import os` was removed with them.
- `src/rsg/nodes/support/phase1/phase1_config.py` — added, then removed,
  `cpu_affinity_enabled`, `segmentation_thread_cpus`,
  `tracking_publish_thread_cpus` fields and their YAML parsing.
- `src/rsg/config/rsg_pipeline.yaml` — added, then removed, the
  `[0-5]`/`[6-9]` affinity config. `phase1.performance.timing_csv_path` /
  `timing_sheet_name` remain repointed from `optimisation_part4` to
  `optimisation_part5` (that repoint is retained — it's just where session
  evidence lands, not part of the affinity feature).
- `debug/optimisation/optimisation_part5/analyse_timing.py` — forked from
  Part 4's copy (not edited in place); no new columns, updated framing text
  only. Kept, since it's just an evidence-analysis script, independent of
  whether the affinity code itself was kept.

Step 2: none in `src/rsg/`. System configuration only (outside git):
`sudo jetson_clocks`.

Branch: `pipeline-split-lean` (pushed; this part's commits will follow the
same branch unless directed otherwise). Verified after the Step 1 revert:
`python3 -m pytest -q tests -p no:anyio` → 35/35 pass; `colcon build
--packages-select rsg` → succeeds; `grep` confirms zero remaining references
to `cpu_affinity`/`_apply_thread_cpu_affinity`/`_set_os_thread_name` anywhere
in `src/rsg/`.

## Final findings

**Step 1 (CPU core affinity): confirmed null result.** Ruled out CPU
scheduling/cache-locality as the cause of Part 4's contention, with pinning
verified genuinely active via `/proc`, not just log trust, across two
independent runs — every metric within ~4% of unpinned Part 4, inside this
campaign's established noise band.

**Step 2 (`jetson_clocks`/power mode): inconclusive.** One run showed a
small, directionally-positive effect (+4% processing ratio, -2.4%
`sam_inference_ms`) plus one more confidently attributable finding (tail
latency on `sam_output_queue_wait_ms` dropped sharply, plausibly from
disabled CPU idle states) — but the mean-level result is still smaller than
the established noise band, off a single run.

**Decision (user-confirmed 2026-08-07): close Part 5 entirely, and remove
the Step 1 CPU-affinity code from the codebase rather than leave it enabled
as originally planned.** The reasoning changed from "is this fix free to
leave in" to "does this code belong in the codebase at all": there is a
concrete plan to port this code to Jetson Thor, whose core count and cache
topology are unknown and almost certainly different from this Orin's 12
cores. The affinity feature's tuned values (`[0-5]`/`[6-9]`) are Orin-only
knowledge, and since Step 1 already proved the feature provides zero
measured benefit on the hardware it *was* tuned for, there is no argument
left for carrying Orin-specific, unvalidated-elsewhere scheduling logic
into a codebase meant to run on different silicon. Removing it entirely —
not just disabling it — keeps the two-thread pipeline split from Part 4
(which is architecture-independent and stays) generic, with nothing in it
that assumes a specific core count or layout.

Root cause of Part 4's contention remains formally undetermined. That does
not threaten Part 4's acceptance — the campaign's own framing there was
explicit that the per-frame latency cost is an accepted trade-off for
throughput, not something requiring an explanation to justify. This part's
honest contribution is two ruled-out (or inconclusive) hypotheses,
correctly attributed and not oversold, plus a codebase that stayed
portable rather than accumulating hardware-specific tuning for a platform
change already on the roadmap.
