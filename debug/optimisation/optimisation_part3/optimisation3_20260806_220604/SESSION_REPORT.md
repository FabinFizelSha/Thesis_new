# Session report: optimisation3_20260806_220604

Level 3 of 3 in the campaign documentation hierarchy — see
[../PART3_REPORT.md](../PART3_REPORT.md) for this part's full record and
[../../OPTIMISATION_REPORT.md](../../OPTIMISATION_REPORT.md) for the
whole-campaign record and standard protocol.

## Purpose

Measure the effect of removing the five unused `_as_list()` numpy→list
conversions from `_find_match`'s per-candidate row construction (Step 4 in
`../PART3_REPORT.md`), compared against Run 2
(`optimisation3_20260806_205816`, before this change).

## Code change under test

`nodes/support/phase1/persistent_object_tracker.py`: removed
`candidate_centroid_3d`, `candidate_bbox_3d_min`, `candidate_bbox_3d_max`,
`candidate_last_bbox_3d_min`, `candidate_last_bbox_3d_max` from the
per-candidate row dict in `_find_match()`. Confirmed via cross-package grep
before removal that none of these five keys are read anywhere in this
workspace. Zero input to any matching/scoring decision.

## Runtime configuration

- Session folder: `debug/optimisation/optimisation_part3/optimisation3_20260806_220604/`
- Standard test protocol per `../../OPTIMISATION_REPORT.md`.
- Pre-run verification: 34/34 tests pass, `colcon build` succeeds.

## Results (this session)

- Received/processed/dropped/failed: 2,740 / 304 / 2,436 / 0.
- Mean masks per processed frame: 7.661 — consistent with every prior
  session.
- Total latency mean/median/p95/max: 589.961 / 579.772 / 744.811 /
  1,119.836 ms. Classifier mean: 530.069 ms.
- `sam_inference_ms` this run: 316.303 ms mean — **essentially identical**
  to Run 2's 315.190 ms (+0.35%). This is an unusually clean comparison:
  the two runs' SAM cost barely differs, so the delta below is attributable
  to the code change with much less of the run-to-run noise that
  complicated earlier comparisons.

## Comparison to Run 2 (`optimisation3_20260806_205816`, before this change)

| Metric | Run 2 (before) | Run 3 (after) | Change |
|---|---:|---:|---:|
| `assignment_row_init_ms` mean | 12.531 ms | 3.674 ms | **-8.857 ms (-70.7%)** |
| `assignment_candidate_search_ms` mean | 57.763 ms | 51.551 ms | **-6.212 ms (-10.8%)** |
| `frame_assignment_ms` mean | 70.363 ms | 64.016 ms | **-6.347 ms (-9.0%)** |
| `sam_inference_ms` mean (context) | 315.190 ms | 316.303 ms | +1.113 ms (noise, negligible) |
| `classifier_delay_ms` mean | 539.628 ms | 530.069 ms | -9.559 ms (-1.8%) |
| non-SAM portion (`classifier - sam`) | 224.438 ms | 213.766 ms | **-10.672 ms (-4.8%)** |
| `total_delay_ms` mean | 601.630 ms | 589.961 ms | -11.669 ms (-1.9%) |
| Processed frames | 273 | 304 | +31 (+11.4%) |
| Mean masks/processed frame | 7.722 | 7.661 | comparable |

`assignment_row_init_ms` fell almost exactly as expected from eliminating 5
numpy→list conversions across up to ~300 candidates per frame. This
cascaded cleanly into `assignment_candidate_search_ms` (-10.8%) and
`frame_assignment_ms` (-9.0%). Because `sam_inference_ms` barely moved
between these two specific runs, the non-SAM-portion delta (-4.8%) is a
reasonably direct read of this change's real effect, not mostly SAM noise
like several earlier comparisons in this campaign.

## Unexpected secondary finding: strong evidence for the lock-contention hypothesis

`assignment_row_init_ms`'s **mean** fell 70.7% (12.531 → 3.674 ms), but its
**max** barely moved: **351.020 ms → 317.840 ms, only -9.5%**. If the
scattered severe spikes documented in Run 2's session report were caused by
the per-candidate computation itself, cutting that computation's cost by
70% should have cut the spike by a similar proportion — it didn't come
close. This is strong indirect evidence that those spikes are **not**
computational at all, and are caused by something external to this code
(most plausibly lock contention with the async RAP/VLM threads, as
hypothesized in Run 2's report, though still not directly confirmed without
explicit lock-wait timing — see `../PART3_REPORT.md` "Path B"). Removing
dead computation made the typical case faster without touching the tail,
exactly what you'd expect if the tail's cause is unrelated to the mean
cost.

## Functional / accuracy evidence

- No test references the removed keys; 34/34 tests pass both before and
  after.
- Zero processing failures. Mask-count distribution stable vs. all prior
  sessions.
- No matching/scoring logic was touched — the five removed fields were
  never read by any decision path.

**Methodology note (from this session's discussion, applies going
forward):** this campaign's controlled runs use `--rate 1` real-time bag
playback, which drops ~85-90% of frames under the current pipeline
throughput. That's an appropriate setting for measuring **per-frame
processing latency** (each processed frame's cost is independent of which
frames were dropped), but it is **not** an appropriate setup for evaluating
tracking **accuracy** — sparse, non-contiguous frame sampling can produce
tracking gaps or mismatches that reflect the playback rate, not code
correctness. Any future accuracy evaluation of this campaign's changes
needs a separately configured run (slower rate / lower drop ratio), not
these timing sessions.

## Decision

**Accept.** Removing the five unused row-construction fields is a real,
verified, low-risk latency improvement — `frame_assignment_ms` -9.0% mean,
non-SAM processing -4.8%, with zero effect on any matching decision (nothing
consumed the removed fields) and zero test regressions. Also produced a
valuable secondary finding reinforcing the lock-contention hypothesis for
the still-open tail-latency question. See `../PART3_REPORT.md` for the
updated Part 3 status and the remaining open items (Path B investigation,
whether to pursue further candidate-count reduction).
