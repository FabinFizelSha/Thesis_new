# Memory/FPS Scaling Experiment

**Status: Iteration 1 closed after 3 of the 5 planned sessions
(calibration + run1-3).** Results, analysis, and conclusion: `RESULTS.md`.
Closed deliberately once run3 showed fps plateauing/slightly reversing
despite `vlm_call_count` continuing to fall sharply -- a considered stopping
point, not a shortfall; this document's procedure remains valid for anyone
continuing with run4/run5.

**Iteration 2 completed its full planned 3-session scope** (Section 10) as
an independent replication check on Iteration 1's early stop -- it
reproduced the same plateau/reversal pattern, corroborating rather than
overturning it. Combined results and conclusion across both iterations:
`RESULTS.md` Section 7.

This is the throughput question that
`debug/revisit_experiment/EXPERIMENT_PART1_REVISIT_ACCURACY.md` deliberately
left out of scope ("a separate, dedicated follow-up experiment... at a higher
bag playback rate"). It was blocked until now by a real SAM-throughput bug
(`sam_prep_summary` serializing ~8 MB of raw arrays into every frame's
metadata, see commit `9cbbb82`) that made `sam_inference_ms` ~4-5x worse than
it should have been; that bug is fixed and verified, so this experiment can
now produce a real result instead of measuring the bug.

## 1. Objective

Confirm and quantify the second half of the original two-part hypothesis from
the revisit experiment: as a pipeline resumes across sessions and re-observes
already-known objects, the object-detection VLM is dispatched less often
(already confirmed: `vlm_call_count` fell 28→11→5→2→1 across five sessions),
and that reduced VLM load frees enough GPU time that **frame throughput
(fps) rises** across sessions -- not just that VLM calls decline, but that
the decline actually shows up as more frames reaching Hydra per second.

## 2. Hypothesis

Across 5 consecutive sessions replaying the same environment, each resuming
from the previous session's saved memory (tracker state + Hydra DSG/mesh),
with the object-detection VLM enabled throughout:

1. Early sessions see many not-yet-labelled objects, so many VLM calls, so
   more GPU contention with SAM -- achieved fps is at its lowest.
2. As sessions progress, more objects are already labelled and skip VLM
   dispatch entirely, so achieved fps should rise.
3. The rise should be visible in both the raw `avg_fps` metric and in
   `sam_inference_ms`/`total_delay_ms` falling.

## 3. Why the input rate itself has to change between sessions

The bag's own native rate is far faster than the pipeline can consume:

```
/tesse/left_cam/rgb/image_raw: 8307 messages / 506.1499 s = 16.4121 Hz  (at --rate 1.0)
```

recomputed for a different bag with:
```bash
ros2 bag info <bag> | grep -A1 "left_cam/rgb/image_raw"
# native_hz = message_count / bag_duration_sec
```

Playing at `--rate 1.0` (or higher) floods the pipeline with far more frames
than it can process, so almost every frame beyond capacity is dropped. That's
fine as a one-off ceiling measurement (see Step 0 below), but not as the
experiment's actual per-session condition: dropping the large majority of
frames starves the tracker of the repeated observations its quorum-based
association needs, which would confound "fps changed" with "accuracy
collapsed because there was barely any data to track with" -- exactly what
the user flagged going in ("if we play at rate 1, too many frames will be
lost leading to very low accuracy").

The fix is to feed each session at a rate **derived from what the pipeline
just proved it could do**, not a fixed rate chosen up front:

```
next_rate = target_multiplier * this_session's_avg_fps / native_rate_hz
```

with `target_multiplier = 2.0` (the default `snapshot_session.py` uses).
Feeding at 2x the just-measured achieved fps keeps the pipeline genuinely
input-saturated (so the *next* fps reading is a real compute-bound
measurement, not an artifact of an input rate too slow to matter) while
leaving meaningfully more surviving frames than a full `--rate 1.0` overload
would -- a middle ground between "measurement is meaningless because the
input never taxed the pipeline" and "measurement is meaningless because
almost nothing survived to be tracked."

Each session's own achieved fps re-derives the *next* session's rate --
this is not "double the previous `--rate` value" (that would compound
independently of what the pipeline actually achieved). It is "convert the
fps this session actually hit back into the equivalent bag rate, then double
that," so a session that undershot or overshot its target doesn't distort
every session after it.

## 4. Fixed configuration for all sessions

Held constant across the calibration run and all 5 experiment sessions
(current `rsg_pipeline.yaml` state as of 2026-09-11 -- confirm before
starting, since several of these were toggled during the SAM investigation
that preceded this experiment):

| Setting | Value | Why |
|---|---|---|
| `phase1.persistent_tracking.session_persistence.enabled` | `true` | The entire experiment is about what accumulates across sessions. |
| `phase1.vlm.enabled` | `true` | The object-detection VLM is the thing whose declining load this experiment is about. |
| `phase1.risk_vlm.enabled` | `false` | A second concurrent VLM server would confound GPU-load attribution. |
| `phase1.rap.enabled` | `false` | Matches the revisit experiment's baseline; keeps the label source solely VLM so the VLM-load story stays clean. |
| `phase1.vlm.profiles.qwen3_5_4b_q4.server.extra_args` | includes `-np 1` | Required for the object-detection VLM to work at all -- see commit `8367300` (llama-server multimodal KV-cache corruption under >1 slot). |
| `phase1.performance.measure_timing` / `write_timing_excel` | `true` | The per-frame timing CSV this whole experiment reads. |

**Calibration, run1, and run2 were each played for a fixed 300s wall-clock
window** (`timeout --signal=INT 300` around the `ros2 bag play` command).
That approach was superseded starting with run3 (Section 9.2): a fixed
window at increasing rates covers increasingly more of the bag's own
timeline, so later sessions were reaching genuinely unseen scene content
within the same 300s that earlier sessions never touched -- a confound with
the memory-accumulation effect this experiment measures, not a rate/duration
detail. **From run3 onward, both the rate and the play duration are
computed upfront** (Section 9.2/9.3) so each session lands on run1's
input-frame budget (1317 frames) by construction, instead of playing a fixed
window and truncating the excess afterward.

Launch command (risk VLM server not even started, saving it competing for
GPU memory):
```bash
ros2 launch rsg rsg_all.launch.py start_risk_vlm:=false
```

**Before starting a full pipeline restart, kill the *entire* stack, not just
the nodes that write to `memory/`.** `rsg_scene_graph_fuser` and
`hydra_visualizer_node` hold live, in-process accumulated state with no disk
persistence at all -- a leftover instance from an earlier launch will keep
feeding stale markers into RViz (and, worse here, keep its own object-fusion
state alive) even after Hydra and Phase 1 have been restarted cleanly. Confirm
nothing is left with:
```bash
pgrep -fa "ros2 launch rsg|hydra_ros_node|rsg_phase1|rsg_preprocessor|llama-server|rviz2|hydra_visualizer|rsg_scene_graph_fuser|chroma"
```

## 5. Procedure

Bag: `uHumans2_office_s1_00h_ros2`, same topic set used throughout this
project's diagnostics:
```
/clock /tf /tf_static /tesse/left_cam/rgb/image_raw /tesse/depth_cam/mono/image_raw /tesse/left_cam/camera_info /tesse/odom
```

All runs manual (launch, play the bag, Ctrl+C, archive) -- no orchestration
script. `snapshot_session.py` only archives and computes the next rate after
a session ends; it does not launch or watch anything.

### Step 0 -- baseline clear
```bash
python3 clear_memory.py
```
Refuses if a pipeline process is still alive -- see its own `--help`.

### Step 1 -- calibration run (establishes the ceiling; not one of the 5 sessions)

Terminal A:
```bash
ros2 launch rsg rsg_all.launch.py start_risk_vlm:=false
```
Terminal B, once both `hydra_ros_node` and `rsg_phase1_semantic_coordinator`
are up:
```bash
timeout --signal=INT 300 ros2 bag play /home/student/datasets/uhumans2/uHumans2_office_s1_00h_ros2 --rate 1.0 \
    --read-ahead-queue-size 2000 \
    --qos-profile-overrides-path /home/student/datasets/uhumans2/tf_static_qos_override.yaml \
    --topics /clock /tf /tf_static /tesse/left_cam/rgb/image_raw /tesse/depth_cam/mono/image_raw /tesse/left_cam/camera_info /tesse/odom
```
300s, same fixed window as every other session (see Section 4).

When the bag command above exits on its own (the `timeout`), Ctrl+C
terminal A and **wait for the shell prompt to actually return** before
touching anything -- Phase 1 and Hydra both save state on clean shutdown, and
snapshotting before that finishes copies an incomplete file.

Archive and get the calibration's rate ceiling:
```bash
python3 debug/fps_scaling_experiment/snapshot_session.py --session calibration
```
This prints `avg_fps` and a suggested next rate at the bottom
(`target_multiplier x avg_fps / native_rate_hz`) -- that suggested rate is
what Step 3 plays at for run 1.

### Step 2 -- discard the calibration run's memory
```bash
python3 clear_memory.py
```
The calibration run's tracker/map state is a worst-case, massively
frame-dropped partial pass -- it must not be what run 1 resumes from. Run 1
needs the same genuinely-empty starting point the calibration run itself
started from.

### Step 3 -- run 1
Terminal A: same launch command as Step 1.
Terminal B, using the rate `snapshot_session.py` printed for calibration:
```bash
timeout --signal=INT 300 ros2 bag play /home/student/datasets/uhumans2/uHumans2_office_s1_00h_ros2 --rate <calibration's suggested rate> \
    --read-ahead-queue-size 2000 \
    --qos-profile-overrides-path /home/student/datasets/uhumans2/tf_static_qos_override.yaml \
    --topics /clock /tf /tf_static /tesse/left_cam/rgb/image_raw /tesse/depth_cam/mono/image_raw /tesse/left_cam/camera_info /tesse/odom
```
300s (matching the revisit experiment's own window) unless the chosen rate
runs past the bag's ~506s total length first, in which case `ros2 bag play`
simply exits on its own when the bag ends -- that's fine, just note the
actual elapsed time when archiving.

Shut down cleanly (Ctrl+C bag, then Ctrl+C the launch, wait for the prompt),
then:
```bash
python3 debug/fps_scaling_experiment/snapshot_session.py --session run1
```
Note the suggested rate it prints for run 2.

### Steps 4-7 -- runs 2 through 5

**Do not clear memory between runs 1-5** -- each must resume from the
previous run's save; that accumulation across sessions is the entire point.
Repeat Step 3's pattern exactly, each time:
1. Same launch command.
2. Play the bag at the rate `snapshot_session.py` printed after the
   *previous* run (`run2` uses run 1's suggested rate, `run3` uses run 2's,
   etc.) -- not a fixed schedule decided up front.
3. Shut down cleanly.
4. `python3 debug/fps_scaling_experiment/snapshot_session.py --session run2`
   (then `run3`, `run4`, `run5`).

## 6. What gets recorded, and where

`debug/fps_scaling_experiment/sessions/<session>/` per session (calibration,
run1..run5), each containing:
- `phase1_tracker_state.json`, `hydra_dsg.json` -- that session's saved
  memory, for tracing a specific object/track by hand if needed.
- `phase1_timing.csv` -- full per-frame and per-VLM-call detail
  (`sam_inference_ms`, `total_delay_ms`, VLM outcomes, etc.).
- `summary.json` -- every computed metric for that one session.
- `resource_usage.csv` -- only present if `debug/revisit_experiment/monitor_resources.py`
  was run alongside that session in a spare terminal (optional; not required
  by this experiment's core question, but useful for explaining *why* fps
  moved if GPU/CPU load is worth showing).

`debug/fps_scaling_experiment/sessions/summary_all_sessions.csv` -- one row
per session, every headline metric plus `suggested_next_rate` in one file --
this is what a fps-vs-session or vlm_call_count-vs-session plot reads
directly. Not tracked in git (see `.gitignore`); regenerate by re-running the
bag with the same tooling.

## 7. Evaluation plan (after the 5 sessions are done)

1. Confirm `vlm_call_count` falls across run1..run5 (expected, matching the
   revisit experiment's own 28→11→5→2→1 pattern -- though the absolute
   numbers here will differ since the input rate is different per session).
2. Plot/tabulate `avg_fps` across run1..run5. The hypothesis predicts a rise;
   report the actual trend even if it's flat or non-monotonic -- the revisit
   experiment's own closing lesson was to report what the data shows, not
   force-fit the hypothesis.
3. Cross-check `avg_sam_inference_ms` and `avg_frame_latency_ms` against the
   fps trend -- if fps changes without these moving, or vice versa, that's a
   real finding worth explaining, not a discrepancy to paper over (see the
   analysis method already established for the SAM-throughput investigation
   this session: `debug/revisit_experiment/analyze_stage_timing.py` can be
   pointed at any of these `phase1_timing.csv` files for a full stage
   breakdown).
4. Sanity-check `frame_drop_count` and `frame_count` per session so the
   comparison is measuring the hypothesis and not a session that happened to
   process far more or fewer frames than the others.
5. Write up results as `RESULTS.md` in this folder once all 5 sessions are
   archived, following the same structure as
   `debug/revisit_experiment/EXPERIMENT_PART1_REVISIT_ACCURACY.md`
   (objective, hypothesis, setup, results table, analysis, conclusion) --
   ready to drop into a thesis experiments chapter.

## 8. Known caveats, stated up front

- **`suggested_next_rate` is a recommendation, not a hard requirement.** If a
  session badly undershoots or overshoots its target rate (e.g. the bag ends
  before the window does, or the pipeline behaves very differently than the
  previous session), use judgement on whether to use the printed rate as-is,
  re-run that session, or note the deviation in `RESULTS.md`.
- **The calibration session's rate (1.0) is deliberately not one of the 5
  experiment sessions** -- it exists purely to give run 1 a starting rate
  grounded in a real measurement instead of an arbitrary guess, and its
  frame-drop-heavy memory is explicitly discarded (Step 2) before run 1
  starts.
- **This is a single run per rate, not a repeated/averaged measurement** --
  same limitation the revisit experiment's own Section 6 already flagged for
  GPU-load noise; report single-run numbers as such, not as a statistically
  averaged result.
- **`--rate` values below ~0.05 or above what keeps `ros2 bag play` stable
  are not sanity-checked by any tooling here** -- if a suggested rate looks
  degenerate (e.g. very close to 0, or absurdly high), stop and check the
  previous session's `avg_fps` reading before proceeding.
- **A fixed 300s wall-clock window at increasing rates covers increasingly
  more of the bag's own timeline** -- a higher-rate session's later portion
  is genuinely unseen scene content the previous session never reached, which
  adds VLM/fps effects that have nothing to do with accumulated memory. Every
  session from run2 onward must also be normalized to run1's total input
  frame count (`frame_count + frame_drop_count`, run1's own value = 1317)
  before comparing metrics across sessions -- see Section 9.1 and
  `normalize_session.py`. The raw (full-window) numbers in Section 9's table
  are still recorded for reference, but the *normalized* numbers are what
  `RESULTS.md`'s final comparison should actually use.

## 9. Session log (filled in as sessions complete)

Running record of what was actually done, kept here during the experiment;
folded into `RESULTS.md`'s results table once all 5 sessions are done.

| Session | Rate suggested | Rate used | avg_fps | frame_count | frame_drop_count | Notes |
|---|---:|---:|---:|---:|---:|---|
| calibration | 1.0 (fixed) | 1.0 | 2.3593 | 699 | 4142 | ~86% of input dropped, as expected at native rate; drop count checks out against 300s x 16.4121 Hz native. Not one of the 5 sessions -- not normalized. |
| run1 | 0.2875 | 0.25 | 2.7164 | 804 | 513 | Rounded down from the calculated 0.2875 -- deliberate, not a tooling error. First real accumulation-story session (started from cleared memory, Step 2) -- vlm_call_count (96) rose vs. calibration (63), expected: most objects are new here, same pattern as the revisit experiment's own session 1. Declining VLM load is the run2+ story. **This session is the normalization reference** -- its total input frame count, frame_count + frame_drop_count = 804 + 513 = **1317**, is the fixed budget every later session gets truncated to match (Section 9.1). |
| run2 | 0.331 | 0.3 | 2.9751 | 886 | 687 | Rounded down from 0.331, same rounding convention as run1. Memory carried over from run1 (no clear), as required. Raw vlm_call_count dropped sharply vs run1 (96->32); **normalized to run1's 1317-frame budget it drops further, to 18 (-81.3% vs run1's 96)** -- 14 of the raw 32 calls were in scene content beyond run1's reach, not part of the accumulation effect. Normalized fps (2.9775) barely differs from raw (2.9751) for this pair -- see Section 9.1. |
| run3 | 0.3628 | 0.33 | 2.9436 | 711 | 708 | Methodology change from here on -- see Section 9.2: rate and play *duration* are both chosen upfront so the run lands on the 1317-frame budget by construction, instead of playing 300s and truncating after the fact. Rate rounded down to 0.33; duration recalculated for that specific rate -- 243.2s, timeout 243 (actual raw total input 1419, only ~7.7% over target, confirming the upfront-duration approach works -- much closer than run2's fixed-300s overshoot). Normalized avg_fps 2.9375, a small **dip** vs run2's 2.9775 (-1.3%) despite vlm_call_count continuing to fall sharply (18->7) -- track_count/hydra_object_node_count keep growing (157/161 -> 158/169), a plausible but unconfirmed counter-cost from more per-frame tracking/geometry work against a bigger accumulated map. Within single-run noise so far; not yet a conclusion. |

### 9.1 Frame-budget normalization

After archiving a session with `snapshot_session.py`, also normalize it
against run1 (the reference, 1317-input-frame budget):
```bash
python3 debug/fps_scaling_experiment/normalize_session.py --reference run1 --session run2
python3 debug/fps_scaling_experiment/normalize_session.py --reference run1 --session run3
# ... run4, run5
```
This truncates that session's `phase1_timing.csv` to only the frames within
the wall-clock window it took that session's *own empirically observed*
input rate to reach 1317 total input frames (processed + dropped) -- see the
script's own docstring for the full reasoning and formula. Writes
`sessions/<session>/summary_normalized.json` and appends to
`sessions/summary_all_sessions_normalized.csv`. Run1 itself needs no
normalization (it *is* the reference, so its raw numbers already are its
normalized numbers).

| Session | raw avg_fps | raw vlm_call_count | normalized window (s) | normalized avg_fps | normalized vlm_call_count |
|---|---:|---:|---:|---:|---:|
| run1 (reference) | 2.7164 | 96 | 295.98 (full) | 2.7164 | 96 |
| run2 | 2.9751 | 32 | 249.34 | 2.9775 | 18 |
| run3 | 2.9436 | 7 | 224.18 | 2.9375 | 7 |

### 9.2 Methodology change from run3 onward: choose the play duration upfront

Run1 and run2 both played the fixed 300s window (Section 4) and were
normalized *after the fact* by truncating the excess. That works but wastes
real playback time on data that gets discarded, and only approximately hits
the target frame count (whatever the session's rate happened to produce
within 300s, then cut). From run3 onward, both the rate and the play
duration are computed upfront so the run lands on run1's 1317-frame budget
by construction, with no truncation needed:

```
next_rate         = target_multiplier * previous_normalized_avg_fps / native_rate_hz
next_duration_sec = target_frame_count / (target_multiplier * previous_normalized_avg_fps)
```

The duration formula simplifies to not depend on `native_rate_hz` at all
(it cancels out of `target_frame_count / (rate * native_rate_hz)` once
`rate` is substituted) -- so any imprecision in the measured native rate only
affects which rate is chosen, not whether that rate's session actually lands
on the frame budget. `normalize_session.py` computes and prints both numbers
automatically as part of normalizing the *previous* session (it did this for
run2 already -- see the run3 row in Section 9's table above and Section 9.3
for the derivation). Play command for a session sized this way:

```bash
timeout --signal=INT <next_duration_sec, rounded> ros2 bag play /home/student/datasets/uhumans2/uHumans2_office_s1_00h_ros2 --rate <next_rate> \
    --read-ahead-queue-size 2000 \
    --qos-profile-overrides-path /home/student/datasets/uhumans2/tf_static_qos_override.yaml \
    --topics /clock /tf /tf_static /tesse/left_cam/rgb/image_raw /tesse/depth_cam/mono/image_raw /tesse/left_cam/camera_info /tesse/odom
```

Since the resulting session should already land close to 1317 total input
frames, running `normalize_session.py` against it afterward should show a
`normalized_window_sec` close to the *entire* actual session (little to no
truncation) -- treat it as a verification step for run3 onward rather than
the main correction it was for run2.

### 9.3 Run3's rate and duration (derived from run2's normalized fps)

```
suggested_next_rate         = 2 x 2.9775 / 16.4121 = 0.3628
suggested_next_duration_sec = 1317 / (2 x 2.9775)  = 221.2
```
i.e. `--rate 0.3628` for `timeout --signal=INT 221` (or the user's rounded
choice of either number, recorded in Section 9's table either way).

## 10. Iteration 2

An independent repeat of the same 3-run experiment (run1-run3; the 4th/5th
sessions remain not-yet-attempted from Iteration 1 either), started fresh
rather than continuing from Iteration 1's accumulated memory -- a second
trial under the same conditions, not an extension of the first one.
Deviations from Iteration 1, both deliberate:

1. **No calibration bag-play.** The `--rate 1.0` ceiling measurement
   (2.3593 fps, Section 9's table) is a property of the pipeline's raw
   processing capacity, not of accumulated memory, and nothing about the
   pipeline/model/SAM config has changed since Iteration 1 measured it --
   so it is reused analytically instead of re-measured.
2. **`iter2_run1` mirrors Iteration 1's own run1 exactly**: fixed 300s
   window, `--rate 0.25` (the same rounded value Iteration 1 used, derived
   the same way -- 2 x 2.3593 / 16.4121 = 0.2875, rounded down). It
   establishes **Iteration 2's own** input-frame budget from its own
   `frame_count + frame_drop_count`, empirically -- deliberately *not*
   reusing Iteration 1's 1317, since a fresh independent trial's own frame
   count is what run2/run3 of *this* iteration should be normalized against
   (it may not land on exactly 1317 again, and that's fine -- if it comes
   out close, that is itself a useful reproducibility signal, not something
   to force-match).
3. **The frame-count logic (Section 9.2) applies from run2 onward**, exactly
   mirroring Iteration 1's own structure (which applied it from run3
   onward) -- the only change is which session (`iter2_run1`, not
   Iteration 1's `run1`) supplies the reference budget. Session names are
   prefixed `iter2_` to keep both iterations' archives side by side under
   `sessions/`.

### 10.1 iter2_run1

```bash
python3 clear_memory.py
```
then the standard launch (`ros2 launch rsg rsg_all.launch.py start_risk_vlm:=false`), then:
```bash
timeout --signal=INT 300 ros2 bag play /home/student/datasets/uhumans2/uHumans2_office_s1_00h_ros2 --rate 0.25 \
    --read-ahead-queue-size 2000 \
    --qos-profile-overrides-path /home/student/datasets/uhumans2/tf_static_qos_override.yaml \
    --topics /clock /tf /tf_static /tesse/left_cam/rgb/image_raw /tesse/depth_cam/mono/image_raw /tesse/left_cam/camera_info /tesse/odom
```
After clean shutdown:
```bash
python3 debug/fps_scaling_experiment/snapshot_session.py --session iter2_run1
```
No normalization step for `iter2_run1` -- it *is* Iteration 2's reference,
same as Iteration 1's run1 needed none. Its `frame_count + frame_drop_count`
becomes the fixed budget `iter2_run2`/`iter2_run3` are normalized against:
```bash
python3 debug/fps_scaling_experiment/normalize_session.py --reference iter2_run1 --session iter2_run2
python3 debug/fps_scaling_experiment/normalize_session.py --reference iter2_run1 --session iter2_run3
```
No memory clear between `iter2_run1`, `iter2_run2`, `iter2_run3` -- same
accumulation-across-sessions requirement as Iteration 1.

### 10.2 Session log

| Session | Rate suggested | Rate used | Duration used (s) | frame_count | frame_drop_count | normalized avg_fps | normalized vlm_call_count | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| iter2_run1 | 0.2875 | 0.25 | 300 | 816 | 508 | 2.7440 | 95 | Reference session for iter2_run2/3 -- frame budget = 816+508 = **1324** (vs Iteration 1 run1's 1317, +0.5% -- strong reproducibility: every headline metric within ~1-2% of iter1_run1, see comparison table in the response to "check run 1 results"). |
| iter2_run2 | 0.3344 | 0.3 | 269 | 821 | 609 | 3.0694 | 21 | Rate rounded down to 0.3; duration recalculated -- 268.9s, timeout 269. Normalized to iter2_run1's 1324-frame budget. Reproduces Iteration 1's run1->run2 pattern closely: fps +11.9% (iter1: +9.6%), vlm_call_count -77.9% (iter1: -81.3%), sam_ms -9.7% (iter1: -8.5%), latency -8.6% (iter1: -7.8%) -- same direction and magnitude on an independent trial. |
| iter2_run3 | 0.374 | 0.35 | 230 | 668 | 742 | 2.9512 | 2 | Rate rounded down to 0.35; duration recalculated -- 230.5s, timeout 230. Normalized to iter2_run1's 1324-frame budget. **Independently replicates Iteration 1's run2->run3 plateau/reversal**: fps -3.8% (iter1: -1.3%) despite vlm_call_count collapsing further, 21->2 (iter1: 18->7); avg_sam_inference_ms rose in step (300.5->314.9, iter1: 309.5->315.7). hydra_object_node_count kept growing (93->120->142, iter1: 95->161->169) -- same counter-cost candidate as Iteration 1, now seen twice. |
| iter2_run4 | -- | 0.325 | 248 | 729 | 698 | 2.9626 | **0** | Extension beyond the planned 3-session scope, continuing iter2_run3's memory (no clear). **vlm_call_count hits zero for the first time** -- everything in scene already labelled. fps stayed at the plateau (2.9512->2.9626, +0.4%) rather than recovering toward run2's peak (3.0694) despite VLM load now literally at its floor -- strong confirmation that the plateau is not merely "VLM load hasn't dropped enough yet." hydra_object_node_count kept climbing (120->142->160), now the only variable still moving in the plateau's direction with VLM eliminated as a confound. |
