# Memory/FPS Scaling Experiment — Results

**Status: Iteration 1 closed after 3 of the 5 planned sessions; Iteration 2
completed its full planned 3-session scope as an independent replication.**
Iteration 1 was closed deliberately, not due to a failure: by run3,
`vlm_call_count` had already fallen by 92.7% from run1 and fps had visibly
plateaued rather than continuing to rise, so two more sessions were judged
unlikely to add a materially different finding (Section 4). **Iteration 2
(Section 6) independently reproduced the same plateau/reversal pattern on a
fresh trial**, substantially strengthening the conclusion beyond what a
single 3-run trial could support (combined conclusion: Section 7).

## 1. Objective

See `EXPERIMENT_PLAN.md` Section 1-2 for the full objective and hypothesis.
In short: confirm whether declining object-detection VLM load, as a resumed
pipeline re-observes already-labelled objects, actually shows up as rising
frame throughput (fps) — the half of the original revisit-experiment
hypothesis that experiment deliberately left unanswered.

## 2. Setup

- Bag: `uHumans2_office_s1_00h_ros2`, same topic set used throughout this
  project's diagnostics.
- Config fixed across all sessions: `session_persistence.enabled: true`,
  `phase1.vlm.enabled: true`, `phase1.risk_vlm.enabled: false`,
  `phase1.rap.enabled: false` (see `EXPERIMENT_PLAN.md` Section 4 for the
  full table and reasoning).
- One calibration run at `--rate 1.0` (300s) established the pipeline's raw
  processing ceiling (2.3593 fps) and was discarded (memory cleared) before
  run1, per `EXPERIMENT_PLAN.md` Section 3's reasoning: the native bag rate
  (16.4121 Hz) is far faster than the pipeline can consume, so a `--rate 1.0`
  session drops ~86% of its input and is only useful as a one-off ceiling
  measurement, not as an experiment session in its own right.
- run1, run2, run3 each resumed from the previous session's saved memory (no
  `clear_memory.py` between them) and were each played at `2 x` the
  *previous* session's own achieved fps, converted back into an equivalent
  bag `--rate` (see `EXPERIMENT_PLAN.md` Section 3 for the exact formula and
  its rationale).
- **Methodology correction applied from run2 onward**: a fixed 300s window
  at increasing rates covers increasingly more of the bag's own timeline, so
  a later, higher-rate session's tail reaches genuinely unseen scene content
  the earlier session never touched — new objects requiring VLM dispatch for
  reasons unrelated to the memory-accumulation effect being measured. Every
  session is normalized to run1's total input frame count (`frame_count +
  frame_drop_count` = **1317** frames) before comparison; run2 needed a
  substantial post-hoc truncation (297.8s -> 249.3s), run3 needed only a
  small one (241.5s -> 224.2s) because run3's rate and play duration were
  both computed upfront specifically to land on that budget. See
  `EXPERIMENT_PLAN.md` Sections 9.1-9.2 and `normalize_session.py` for the
  full derivation. **All results below are the normalized figures** unless
  explicitly marked "raw."

## 3. Results

### 3.1 Headline metrics, normalized to run1's 1317-frame budget

| Session | rate used | avg_fps | Δ vs run1 | vlm_call_count | Δ vs run1 | avg_sam_inference_ms | avg_frame_latency_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| run1 (reference) | 0.25 | 2.7164 | — | 96 | — | 338.4 | 600.0 |
| run2 | 0.30 | 2.9775 | **+9.6%** | 18 | **-81.3%** | 309.5 | 553.5 |
| run3 | 0.33 | 2.9375 | **+8.1%** | 7 | **-92.7%** | 315.7 | 563.2 |

### 3.2 Accumulated map size, by session (context for Section 4)

| Session | track_count | allocated_slot_count | hydra_object_node_count | hydra_duplicate_slot_count |
|---|---:|---:|---:|---:|
| run1 | 121 | 140 | 95 | 12 |
| run2 | 157 | 197 | 161 | 30 |
| run3 | 158 | 207 | 169 | 32 |

## 4. Analysis

**The VLM-load half of the hypothesis is confirmed cleanly and strongly.**
`vlm_call_count` fell monotonically and steeply on the same frame budget:
96 -> 18 -> 7, a 92.7% reduction by run3. This is the sharpest, cleanest
trend in the dataset — a resumed, already-labelled track is not
re-dispatched to the VLM, and the count of not-yet-labelled objects
collapses fast as memory accumulates, exactly mirroring the revisit
experiment's own `vlm_call_count` finding (28->11->5->2->1) under a
completely different set of confounds (that experiment held the bag rate
fixed; this one holds the frame budget fixed while varying the rate).

**The throughput half is confirmed, but saturates rather than continuing to
rise.** fps rose sharply from run1 to run2 (+9.6%), matching the "less VLM
load, more room for SAM" prediction directly. But from run2 to run3, despite
`vlm_call_count` falling by more than half again (18 -> 7), fps did not rise
further — it dipped slightly (-1.3%), and `avg_sam_inference_ms` and
`avg_frame_latency_ms` both ticked up in step (309.5 -> 315.7 ms;
553.5 -> 563.2 ms). All three of those figures moving together, in the
opposite direction from what more VLM reduction alone would predict, is a
more specific signal than isolated single-run noise on fps alone would be.

**A plausible, not yet confirmed, explanation**: the accumulated map itself
keeps growing throughout (`track_count` 121->157->158, `hydra_object_node_count`
95->161->169), and per-frame work that scales with map size (tracking/
assignment candidate search, geometry checks against more existing objects)
is a real, separate cost this experiment did not isolate from the VLM-load
effect. It is consistent with the data that VLM load's contribution to the
fps ceiling saturates once the absolute call count is already small (18 and
7 calls, out of 650-750 frames, are both a small minority of frames actually
paying the VLM-dispatch cost either way), while a slower-growing,
monotonically-increasing map-size cost becomes comparatively more visible
once the VLM effect stops dominating. This experiment's data cannot
distinguish that explanation from ordinary single-run variance (see
`EXPERIMENT_PLAN.md` Section 8's own caveat: no run was repeated/averaged) —
it is a lead for a follow-up, not a confirmed mechanism.

## 5. Iteration 1 conclusion

The hypothesis is **supported in direction, not proven as an open-ended
relationship**: as a resumed pipeline re-observes known objects, VLM load
falls sharply and monotonically (confirmed strongly), and frame throughput
rises as a result (confirmed: +8.1% net over the three sessions run) — but
the rise is front-loaded into the first resume and plateaus quickly rather
than continuing to climb as VLM load keeps shrinking toward zero. Stating
this as "fps keeps improving as memory grows" would overclaim what three
data points actually show; stating it as "resuming from memory measurably
improves throughput, with the gain concentrated in the first resume and a
possible map-size counter-cost worth isolating in future work" is the
precise, defensible version -- **on the strength of one trial**. Section 6
tests whether it replicates.

## 6. Iteration 2 — independent replication

A second, independent trial: fresh memory (not a continuation), same bag,
same fixed config, same 3-session structure, deliberately run to check
whether Iteration 1's pattern was a genuine effect or an artifact of that
one trial. Two setup differences from Iteration 1, both in
`EXPERIMENT_PLAN.md` Section 10: the calibration bag-play was not repeated
(its ceiling fps is a pipeline property, not a memory-dependent one, so it
was reused analytically); and `iter2_run1` establishes its own frame budget
(1324 frames) rather than reusing Iteration 1's 1317, so run2/run3 are
normalized within Iteration 2's own terms.

### 6.1 Results, normalized within each iteration's own frame budget

| Step | Iteration 1 fps | Iteration 2 fps | Iteration 1 vlm_call_count | Iteration 2 vlm_call_count |
|---|---:|---:|---:|---:|
| run1 (reference) | 2.7164 | 2.7440 | 96 | 95 |
| run2 | 2.9775 (**+9.6%**) | 3.0694 (**+11.9%**) | 18 (**-81.3%**) | 21 (**-77.9%**) |
| run3 | 2.9375 (**-1.3%** vs run2) | 2.9512 (**-3.8%** vs run2) | 7 (-61.1% vs run2) | 2 (**-90.5%** vs run2) |

`avg_sam_inference_ms` moved the same way at the same step in both
iterations: run1->run2 fell (338.4->309.5 in Iteration 1; 332.7->300.5 in
Iteration 2), then run2->run3 rose back partway (->315.7; ->314.9) --
essentially the same absolute floor (~310-315 ms) reached from both
directions. `hydra_object_node_count` also kept growing in both
(95->161->169 in Iteration 1; 93->120->142 in Iteration 2), continuing to
track alongside the fps plateau rather than the fps rise.

Additionally, `iter2_run1` itself (same rate 0.25, same 300s window, same
fresh-memory start as Iteration 1's own run1) landed within ~1-2% of
Iteration 1's run1 on every headline metric (fps 2.7164 vs 2.7440; vlm_calls
96 vs 95; avg_sam_inference_ms 338.4 vs 332.7; avg_frame_latency_ms 600.0
vs 594.8) -- confirming the experimental setup itself reproduces cleanly
before even looking at the memory-scaling question.

### 6.2 Analysis

Every part of Iteration 1's finding reproduced on an independent trial: the
sharp VLM-load collapse, the sharp fps rise that tracks it on the first
resume, and — the part most worth double-checking, since a single
occurrence could plausibly have been noise — **the plateau/reversal on the
third session, despite VLM load continuing to fall even further**
(Iteration 2's run2->run3 VLM drop, -90.5%, is deeper than Iteration 1's
-61.1%, yet fps still fell rather than rose). Two independent trials
producing the same non-monotonic shape, with `avg_sam_inference_ms` and
`hydra_object_node_count` both moving in the same supporting direction each
time, is a materially stronger basis than either trial alone for treating
the plateau as a real, reproducible saturation effect rather than
run-to-run variance.

### 6.3 iter2_run4 — a 4th session, extending past the planned scope

One further session, continuing `iter2_run3`'s memory (no clear), played at
`--rate 0.325` for 248s (duration computed to hit the same 1324-frame
budget). Not part of either iteration's original 3-session plan; run because
`iter2_run3` left an open question the first three sessions alone could not
settle -- was the plateau a floor, or would fps recover once VLM load had
truly bottomed out?

| | run2 (peak) | run3 | **run4** |
|---|---:|---:|---:|
| normalized fps | 3.0694 | 2.9512 | 2.9626 |
| normalized vlm_call_count | 21 | 2 | **0** |
| avg_sam_inference_ms | 300.5 | 314.9 | 313.5 |
| hydra_object_node_count | 120 | 142 | 160 |

`vlm_call_count` reached exactly zero -- every object in the scene was
already labelled, the strongest possible statement of "VLM load has
bottomed out." fps did **not** recover toward run2's peak; it stayed at the
plateau (2.9512 -> 2.9626, +0.4%, still 3.5% below run2). This directly
rules out the remaining alternative explanation for the plateau ("VLM load
just hadn't fallen far enough yet by run3") -- with VLM completely
eliminated as a variable, `hydra_object_node_count` is now the only
quantity in this dataset still moving in the direction that tracks the
plateau (120 -> 142 -> 160, monotonically, while fps sits flat). This does
not prove the map-size mechanism (still not directly instrumented), but it
substantially narrows what else could be responsible.

## 7. Combined conclusion

**Two independent trials, plus a 4th confirming session, support the same
conclusion**: resuming from accumulated memory measurably improves frame
throughput, and the mechanism (declining VLM dispatch as more objects are
already labelled) is confirmed cleanly and reproducibly. But the benefit
**saturates after the first resume** rather than continuing to climb as
memory keeps accumulating and VLM load keeps shrinking toward zero — both
trials show fps rising sharply on session 2, then plateauing or mildly
reversing on session 3 even as VLM calls keep collapsing further, and
Iteration 2's `iter2_run4` (Section 6.3) pushed VLM load all the way to
**zero calls** without fps recovering toward its session-2 peak — ruling out
"VLM load just hadn't bottomed out yet" as the explanation for the plateau.
The most likely remaining explanation, consistent across all four sessions
examined this way but not yet directly isolated, is that per-frame cost tied
to the size of the accumulated map itself (`track_count`,
`hydra_object_node_count` — the only quantities still moving in the
plateau's direction once VLM is eliminated as a variable) becomes the
dominant remaining factor once VLM load is already small. This is now a
well-supported finding for a thesis chapter: not "memory accumulation
monotonically improves throughput," but "memory accumulation improves
throughput up to a point, after which a specific, named, reproducible
map-size cost is the more likely limiting factor — worth its own follow-up
to instrument directly."

## 8. Closing notes

- **Iteration 1 closed at 3 of 5 planned sessions**, by the user's decision
  once the plateau/slight reversal in run3 made two further sessions
  unlikely to change the conclusion. **Iteration 2 completed its full
  planned 3-session scope** and was run specifically to check whether that
  early stop was justified — it was: the same pattern reproduced
  independently. `EXPERIMENT_PLAN.md`'s procedure remains valid for anyone
  extending either iteration with a 4th/5th session, which would help
  further distinguish "genuine plateau" from "three-point noise on each of
  two trials" (unlikely given how consistently the direction and rough
  magnitude reproduced, but not ruled out by two 3-point trials alone).
- **Not attempted here, left to a follow-up**: isolating the map-size
  counter-cost candidate named in Sections 4 and 6.2 (e.g. by holding the
  map size roughly fixed across sessions and varying only VLM load, or by
  directly instrumenting the tracking/assignment stages' cost as a function
  of `track_count`).
- **Single run per rate per iteration, not repeated/averaged within an
  iteration** — Iteration 2 itself functions as the repeat/average check
  this limitation (flagged in `EXPERIMENT_PLAN.md` Section 8 and the
  revisit experiment's own Section 6) would otherwise leave unaddressed,
  but report per-session numbers as single-run figures, not statistically
  averaged ones.
- **Methodology note**: Iteration 1 only applied the upfront rate+duration
  calculation (`EXPERIMENT_PLAN.md` Section 9.2) from run3 onward; Iteration
  2 applied it from run2 onward (run1 in both iterations used the original
  fixed-300s-window approach, by design — see `EXPERIMENT_PLAN.md` Section
  10). This does not affect comparability: every reported number in this
  document is the *normalized* figure regardless of which approach produced
  the session's raw data.

## 9. Data and reproducibility

- Raw per-session archives: `debug/fps_scaling_experiment/sessions/{calibration,run1,run2,run3,iter2_run1,iter2_run2,iter2_run3,iter2_run4}/`
  (not tracked in git — see `.gitignore`; regenerate by re-running the bag
  with the same tooling and rates recorded in `EXPERIMENT_PLAN.md` Sections
  9 and 10).
- Combined raw table: `debug/fps_scaling_experiment/sessions/summary_all_sessions.csv`.
- Combined normalized table: `debug/fps_scaling_experiment/sessions/summary_all_sessions_normalized.csv`.
- Tooling: `debug/fps_scaling_experiment/{snapshot_session.py,normalize_session.py}`.
- Full procedure, formulas, and the run-by-run log (including the
  calibration session and every rate/rounding decision):
  `debug/fps_scaling_experiment/EXPERIMENT_PLAN.md`.
