# Phase 1 Pipeline Optimisation

*Draft chapter — Experiments. Compiled from the RSG Phase 1 latency/throughput
optimisation campaign (`debug/optimisation/`). All figures in this chapter
are taken directly from recorded controlled-run evidence; none are
estimated. Session identifiers are given so every number can be traced back
to its raw `phase1_timing.csv` in the accompanying archive.*

## 1. Motivation and scope

The Phase 1 semantic coordinator (`rsg_phase1_semantic_coordinator`) is the
component of the RSG pipeline responsible for turning each synchronised
RGB-D-pose frame into a Hydra-ready semantic frame: NanoSAM segmentation,
persistent object tracking, best-crop maintenance, and Hydra message
construction all happen inline on the frame-critical path. Before this
campaign, this pipeline processed only a small fraction of the frames the
sensor produced — an initial baseline measurement on the target hardware
(NVIDIA Jetson AGX Orin Developer Kit, `MODE_50W`, 12-core ARM
Cortex-A78AE) processed **239 of 2,809 received frames (8.51%)** during a
180-second, real-time (`--rate 1`) replay of the `office_s1_00h_v2` uHumans2
bag, with a mean end-to-end latency of **743.077 ms** per processed frame.

The campaign's objective was to reduce this bottleneck along two related
axes:

1. **Per-frame synchronous latency** — how long a single frame takes from
   arrival to Hydra publication (Sections 3–4).
2. **Throughput** — how many of the frames the sensor produces actually
   reach Hydra in a fixed time window (Sections 5–6), which is ultimately
   the axis that matters: a lower per-frame cost is only useful insofar as
   it lets more frames through before the pipeline falls behind real time.

Work proceeded as a sequence of controlled experiments ("parts"), each
targeting one specific section of the pipeline, with every change measured
by a matched before/after run on identical hardware, bag, playback rate,
and window length (180 s, `--rate 1`, `--qos-profile-overrides-path
~/.tf_overrides.yaml`). Every controlled run's raw trace, generated
statistics, and session report are retained as permanent evidence in the
accompanying archive.

One methodological caveat applies throughout this chapter: `sam_inference_ms`
(NanoSAM's own inference call) was observed to vary run-to-run on this
hardware by as much as ±9% with *zero* relevant code change between two
back-to-back sessions, attributed to thermal state, background load, or
TensorRT/CUDA warm-up variance. Deltas smaller than this are reported as
inconclusive rather than as confirmed effects — this distinction is made
explicitly wherever it applies below.

## 2. Method

Every controlled run followed the same protocol:

- Clear the visual-retrieval memory (Chroma-backed RAP store) to a generic
  state before each run, so no run benefits from a previous run's learned
  associations.
- Build and run the full unit test suite before every code change is
  measured (`pytest`, 35 tests by the end of the campaign).
- Launch the full stack (`ros2 launch rsg rsg_all.launch.py`) and replay the
  bag for exactly 180 simulated seconds at real-time rate.
- Stop the launch with a clean shutdown (SIGINT) so the buffered CSV timing
  trace is written; a hard kill discards the trace.
- Analyse the resulting `phase1_timing.csv` with a purpose-built script that
  reports mean/median/p95/max per pipeline stage and, where relevant,
  throughput (frames received/processed/dropped).

Two hard constraints were maintained across every part: **NanoSAM itself
was never modified** (its inference cost is treated as fixed, external
cost), and **no change was accepted without an explicit accuracy/output
equivalence check** (identical tracking decisions, identical published
geometry, no regression in masks-per-frame) — a timing improvement alone
was never sufficient grounds to accept a change.

## 3. Crop rendering optimisation

### 3.1 Problem

Every SAM mask that becomes (or updates) a persistent object track requires
Phase 1 to maintain a "best crop" for that track — the single best-quality
observation of that object seen so far, which is what the downstream visual
retrieval (RAP) and vision-language fallback (VLM) workers classify. At the
campaign's starting point, **representative-crop maintenance
(`crop_update_ms`) cost 135.032 ms mean per frame — 18.2% of total
end-to-end latency**, and was the single largest synchronous cost after
NanoSAM inference itself (328.441 ms, frozen).

Profiling traced this cost to `_remember_track_crop()` doing far more work
than a given frame's best-crop decision actually required:

- It rendered a full RGB context crop even when the caller only needed the
  resulting bounding box.
- Mask cleanup (connected-component filtering) ran on the full 640×480
  frame rather than the small region of interest actually containing the
  object.
- Both the RAP-facing crop *and* the VLM-facing crop were rendered
  synchronously on every qualifying observation, even though RAP and VLM
  are asynchronous workers and VLM is only needed when RAP fails.
- The rendered crops were then copied a further one to two times as they
  moved into the track registry and into a worker's task snapshot.

### 3.2 Change made

The fix restructured *when* rendering happens rather than *what* is
rendered: the frame-critical path now stores one immutable, contiguous
RGB/mask region of interest for the best-scoring observation of a track,
without rendering either the RAP-facing or VLM-facing crop. Rendering (mask
cleanup, RAP target-only compositing, VLM context/halo compositing) is
deferred entirely to the asynchronous RAP or VLM worker, and happens only
once — at the moment that worker actually dequeues the track. Crop scoring
itself, the equal-score tie-break rule, and every other selection criterion
were left unchanged; this was a data-lifetime restructuring, not an
algorithmic one, and its output equivalence was verified with a dedicated
regression test requiring byte-identical rendered crops between the old
full-frame route and the new deferred route.

### 3.3 Findings — before and after

| Metric | Before (baseline) | After (accepted) | Change |
|---|---:|---:|---:|
| `crop_update_ms` mean | 135.032 ms | 1.666 ms | **−98.77%** |
| End-to-end latency mean | 743.077 ms | 602.392 ms | **−18.93%** |
| End-to-end latency p95 | 962.189 ms | 813.911 ms | −15.41% |
| Classifier (Phase 1 processing) mean | 683.766 ms | ≈541.8 ms | −20.76% |
| Frames processed (180 s window) | 239 | 303 | **+26.78%** |

*Sessions: `optimisation_part1/optimisation1_20260806_011841` (baseline),
`optimisation_part1/optimisation1_20260806_015126` (accepted change).*

Representative-crop maintenance stopped being a meaningful synchronous cost
entirely, and — because Phase 1's frame queue is a single-slot,
drop-oldest FIFO — cutting the per-frame processing time directly increased
how many frames were processed before the next one arrived, independent of
any explicit throughput-focused change. This result established the
pattern the rest of the campaign followed: profile a stage, restructure
*when*/*how much* work it does without changing *what* it decides, and
verify with an explicit equivalence test before trusting the timing number.

## 4. Best-match candidate search optimisation

### 4.1 Problem

Every SAM mask that could plausibly belong to an existing persistent object
must be matched against that object's track before Phase 1 can decide
whether to update an existing track's best crop or create a new one. This
matching (`prepare_frame_assignments()` → `_find_match()` per observation,
followed by a global Hungarian assignment) is the "best-match" search that
gates which physical object — and therefore which crop — an observation is
attributed to.

At the point this stage was selected, `frame_assignment_ms` cost 76.349 ms
mean, but its more important characteristic was a **growth trend within a
single run**: mean cost rose from 37.7 ms in the first quarter of a session
to 121.9 ms in the last quarter (sequence correlation 0.442). Finer
profiling (splitting the per-candidate search loop into row-construction,
3D-geometry, centroid/IoU, and scoring sub-steps) confirmed the mechanism:
the *candidate track count* each observation had to be scored against grew
**6.4–6.6× (total) and 9.2–9.8× (max, single observation)** from the first
to the last quarter of a run, as more of the persistent-track registry
filled the explored scene. No single sub-step dominated the resulting cost
(35% / 22% / 19% / 12% split across the four sub-steps) — this ruled out a
single-line fix and pointed at reducing *how much work happens per
candidate* instead.

### 4.2 Change made

Tracing the actual data flow of the per-candidate evaluation row revealed
that it eagerly computed five NumPy-to-Python-list conversions
(`candidate_centroid_3d`, `candidate_bbox_3d_min/max`,
`candidate_last_bbox_3d_min/max`) for *every* candidate track considered,
regardless of whether that candidate would ever be accepted. A first
candidate fix — an early, cheap distance-based rejection filter before the
expensive per-candidate work — was investigated but **rejected before
implementation**: tracing the scoring logic showed a legitimate matching
path (`vertical` + `image` evidence alone, without XY proximity) that an
early XY-distance filter would have silently broken, exactly the kind of
correctness regression this campaign's accuracy constraint exists to catch.

Instead, the five per-candidate list conversions were removed after
confirming — by grep across every package in the workspace — that none of
the five fields were read anywhere, published, or otherwise load-bearing;
they were purely descriptive dead weight left over from an earlier
iteration of the evaluation-row structure.

A separate line of investigation tested whether the *tail* spikes seen in
this stage (300+ ms outliers, 15–28× the stage's own mean) were caused by
lock contention with the asynchronous RAP/VLM worker threads, which share
the persistent-tracker's lock. Explicit lock-wait instrumentation measured
this directly: mean lock-wait was 0.010–0.051 ms, and on a specific
reproduced 335.660 ms spike frame, lock-wait was 0.005–0.334 ms while the
instrumented candidate-loop sub-steps summed to an entirely ordinary
24.155 ms — leaving a ~290 ms gap attributable only to the one
uninstrumented call in the chain, `_candidate_track_ids()`'s final line,
which scans every track in the registry (not just the spatially relevant
candidate subset) to preserve deterministic tie-breaking order. Lock
contention was conclusively ruled out; this specific full-registry scan is
flagged as the probable cause of the tail spikes and remains **unresolved,
future work** at the close of this campaign.

### 4.3 Findings — before and after

| Metric | Before | After (accepted) | Change |
|---|---:|---:|---:|
| `assignment_row_init_ms` mean | 12.531 ms | 3.674 ms | **−70.7%** |
| `assignment_candidate_search_ms` mean | 57.763 ms | 51.551 ms | −10.8% |
| `frame_assignment_ms` mean | 70.363 ms | 64.016 ms | **−9.0%** |
| Non-SAM portion of classifier time | 224.438 ms | 213.766 ms | −4.8% |
| `assignment_row_init_ms` max | 351.020 ms | 317.840 ms | −9.5% |

*Sessions: `optimisation_part3/optimisation3_20260806_205816` (before),
`optimisation_part3/optimisation3_20260806_220604` (accepted change).
`sam_inference_ms` differed by only +0.35% between these two sessions,
making this an unusually low-noise, high-confidence comparison.*

A secondary finding: the removed field's *mean* cost fell 70.7%, but its
*max* barely moved (−9.5%) — strong indirect evidence that the tail spikes
investigated separately (above) are not computational in origin, consistent
with the subsequent finding that they trace to the full-registry scan
rather than any of the timed candidate-evaluation sub-steps.

One earlier candidate fix in this same investigative area — reordering a
linear-algebra formula in the object-geometry projection step
(`(rot_m @ points_cam.T).T` → `points_cam @ rot_m.T`, mathematically
identical, verified numerically equivalent) — was implemented, measured,
and **rejected**: it regressed `geometry_projection_ms` by +15.3% and
`geometry_metadata_ms` overall by +18.2% on this specific hardware, most
likely because a 3×3 matrix multiplication does not route through a BLAS
`dgemm` call the way the "avoid a transposed view" heuristic assumed. It
was reverted in full. This result is retained in the record as a
methodological lesson applied to every subsequent decision in the
campaign: a plausible, textbook-correct micro-optimisation is not reliable
on this hardware without a controlled before/after measurement.

## 5. Pipeline re-split (segmentation/tracking concurrency)

### 5.1 Problem

By the close of Section 4's work, Phase 1 still processed frames through
one strictly serial path per frame: image conversion → NanoSAM → geometry →
tracking/assignment → label-map construction → Hydra publish, all on a
single worker thread, with the next frame unable to begin until the
current one finished completely. At this point NanoSAM inference itself —
frozen throughout the campaign — was **337.6 ms mean, 63.7% of the
635.9 ms combined active per-frame cost**, with the remaining
tracking/publish work (geometry, assignment, label maps, Hydra I/O) costing
**231.1 ms, 33.5%**. Because Phase 1's input queue drops the oldest frame
whenever it is full, this serial cost directly capped throughput: only
**302 of 2,740 received frames (11.02%) were processed** in the standard
180-second window.

### 5.2 Change made

The single classification thread was split into two: a segmentation thread
(image conversion + NanoSAM only) and a tracking/publish thread (geometry,
assignment, label-map construction, and the Hydra publish call), connected
by a new single-slot, drop-oldest handoff queue. This allows NanoSAM
inference for frame *N+1* to run concurrently with tracking/publish work
for frame *N*, rather than waiting for it to finish. The design rationale:
NanoSAM's backend call is expected to release the Python interpreter's GIL
for most of its wall time while it blocks on the GPU, the same property the
pipeline's existing asynchronous RAP/VLM worker threads already depend on
for their own overlap with the main thread — meaning the CPU-bound
tracking/publish thread should, in principle, be able to run largely
"for free" alongside it. This is a change to *when* work happens (thread
scheduling), not to any computed value: geometry, tracking decisions,
label-map content, and published Hydra messages are byte-for-byte
unchanged from the serial design.

### 5.3 Findings — before and after

| Metric | Before (serial) | After (two-thread, accepted) | Change |
|---|---:|---:|---:|
| Frames processed (180 s window) | 302 | 373 | **+23.51%** |
| Processing ratio | 11.02% | 13.53% | **+2.51 pp** |
| Mean end-to-end latency | 589.961 ms | 727.406 ms | **+23.30%** |
| Max end-to-end latency | 1,119.836 ms | 1,563.460 ms | +39.61% |
| `sam_inference_ms` mean | 316.303 ms | 436.291 ms | **+37.94%** |
| Mean masks per processed frame | 7.661 | 7.662 | unchanged |

*Sessions: `optimisation_part4/optimisation4_20260807_004917`, compared
against `optimisation_part3/optimisation3_20260806_220604` (the Section 4
close point).*

This result required a genuine judgement call rather than a clean accept/
reject: **every individual frame got measurably slower** (mean +23.3%, max
+39.6%), because the two threads contended for CPU/memory resources more
than the GIL-release model predicted — `sam_inference_ms` alone was 37.94%
more expensive when run concurrently with the tracking/publish thread than
it was running alone. At the same time, **23.5% more frames reached Hydra**
in the same window, with zero measurable accuracy or coverage regression
(mean masks per processed frame unchanged). The change was accepted on the
basis that per-frame latency reduction throughout Sections 3–4 was always
instrumental toward one underlying goal — more frames reaching Hydra before
the pipeline falls behind — not an end in itself; a design that reaches
that same goal via a different mechanism, at a per-frame cost, satisfies
the actual objective. This reframing — from "minimise latency" to
"maximise throughput," made explicit at this point in the campaign — is
reflected in Section 6's investigation being judged by processing ratio
rather than mean latency.

## 6. Contention investigation (not adopted)

Section 5's result left an open question: *why* did concurrent execution
cost more than expected, beyond the two threads simply sharing CPU time?
Two candidate explanations were tested as parallel hypotheses.

### 6.1 CPU core affinity

**Hypothesis.** The segmentation and tracking/publish threads can be freely
scheduled onto the same physical CPU cores by the OS, causing cache
thrashing or scheduling overhead a single-threaded design never
encountered. Pinning each thread to a disjoint set of CPU cores (6 of 12
Orin cores for segmentation, 4 for tracking/publish, reflecting their
roughly 2:1 cost ratio) should remove this specific variable.

**Result.** Implemented, and verified genuinely active at runtime — not
merely assumed — via kernel-visible thread naming and direct `/proc`
inspection of each thread's confirmed CPU mask, after an initial run raised
a legitimate concern about whether the setting had silently failed to
apply. Across two independent 180-second runs, one with pinning externally
confirmed active, every relevant metric moved by less than 4% from the
unpinned baseline — inside the run-to-run noise band established earlier
in the campaign. **CPU scheduling/cache-locality contention was not
confirmed as a cause of Section 5's slowdown.**

| Metric | Unpinned (Section 5) | Pinned, confirmed active | Change |
|---|---:|---:|---:|
| Processing ratio | 13.53% | 13.37% | −0.16 pp (noise) |
| `sam_inference_ms` mean | 436.291 ms | 432.900 ms | −0.78% (noise) |
| `geometry_metadata_ms` mean | 101.858 ms | 102.480 ms | +0.61% (noise) |

*Session: `optimisation_part5/optimisation5_20260807_013627`.*

### 6.2 Clock and power-mode scaling

**Hypothesis.** The Jetson's default CPU governor (`schedutil`) scales
clocks dynamically by load; concurrent execution produces a higher combined
instantaneous CPU+GPU load than the old serial design ever did, which may
prevent the governor from boosting clocks as aggressively, or may hit a
shared thermal/power ceiling across CPU, GPU, and memory controller.
Locking all clocks to maximum (`jetson_clocks`) removes this as a variable.

**Result.** One controlled run, with the clock lock confirmed via
`jetson_clocks --show` (all 12 cores moved from dynamic 729.6–1,497.6 MHz
scaling to a fixed 1,497.6 MHz; GPU from an idle 306 MHz to a locked
816 MHz). Every metric moved in the favourable direction, but by less than
the established noise threshold — **suggestive, not conclusive**, from a
single run:

| Metric | Clocks unlocked | Clocks locked | Change |
|---|---:|---:|---:|
| Processing ratio | 13.37% | 13.91% | +0.54 pp |
| `sam_inference_ms` mean | 432.900 ms | 422.487 ms | −2.41% |
| `total_delay_ms` mean | 721.827 ms | 701.177 ms | −2.86% |

*Session: `optimisation_part5/optimisation5_20260807_015857`.* One more
confidently attributable secondary effect was observed regardless of the
inconclusive mean-level result: `sam_output_queue_wait_ms`'s worst case fell
from 28–85 ms (both affinity runs, clocks unlocked) to 2.739 ms, plausibly
because locking clocks also disabled CPU idle sleep states, removing
core wake-up latency jitter — a distinct and better-supported claim than
"locked clocks made SAM faster on average."

### 6.3 Outcome

Neither hypothesis was confirmed; the root cause of Section 5's contention
remains **formally undetermined** at the close of this campaign. This does
not affect Section 5's acceptance, which was made on throughput/accuracy
grounds independent of explaining the mechanism. The CPU-affinity code was
implemented, measured, and subsequently **removed from the codebase in
full** (not merely left disabled) rather than kept as an inert option: a
Jetson Thor deployment is planned next, Thor's core count and cache
topology are expected to differ substantially from this Orin's 12 cores,
and an Orin-tuned core split provided no benefit even on the hardware it
was tuned for. The two-thread pipeline split from Section 5 itself makes no
assumption about core count or topology and was retained unmodified.

## 7. Summary

| Section | Target | Outcome | Key metric, before → after |
|---|---|---|---|
| 3 | Crop rendering | **Accepted** | `crop_update_ms` 135.032 → 1.666 ms (−98.77%) |
| 4 | Best-match candidate search | **Accepted** | `assignment_row_init_ms` 12.531 → 3.674 ms (−70.7%) |
| — | Geometry projection reorder | **Rejected** (regressed) | `geometry_metadata_ms` +18.2%, reverted |
| 5 | Pipeline re-split (concurrency) | **Accepted** (trade-off) | Processing ratio 11.02% → 13.53% (+23.51% frames), latency +23.30% |
| 6.1 | CPU core affinity | **Rejected** (null result, removed) | All deltas <4%, within noise |
| 6.2 | Clock/power-mode lock | **Inconclusive** (not pursued further) | +4% ratio, within noise, single run |

Across Sections 3–4, mean synchronous latency fell from 743.077 ms to
589.961 ms (−20.6%) with a fully serial pipeline. Section 5's concurrency
change then traded some of that latency back (589.961 → 727.406 ms) for a
throughput gain the serial design's own latency reductions could not
reach on their own: **frames processed per 180-second window rose from an
initial 239 (8.51% of received) to a final 373 (13.53% of received) — a
1.56× improvement in how much of the sensor's output actually reaches
Hydra**, with no measured loss of tracking accuracy or object coverage at
any accepted step.

## 8. Open work

- **Tail-latency root cause** (Section 4): `_candidate_track_ids()`'s
  full-registry scan is the identified but unfixed source of severe
  (300+ ms) outlier frames in candidate search. A fix requires establishing
  whether the registry's dict-insertion iteration order is load-bearing for
  the Hungarian assignment's tie-breaking behaviour before it can safely be
  narrowed to the spatial candidate subset.
- **Section 5's contention mechanism**: neither CPU affinity nor clock
  locking conclusively explains why concurrent execution costs more per
  frame than serial execution predicted. Shared memory bandwidth (which
  disjoint core sets do not isolate) is the remaining untested candidate.
- **Jetson Thor validation**: the accepted two-thread design is
  architecture-generic by construction, but its actual throughput benefit
  on Thor is unmeasured — Thor's GPU generation is expected to reduce
  `sam_inference_ms` substantially, which would shift the balance between
  the two threads (segmentation vs. tracking/publish) away from the ~2:1
  ratio this design was tuned against on Orin, and needs re-profiling
  rather than assuming the Orin numbers transfer.
