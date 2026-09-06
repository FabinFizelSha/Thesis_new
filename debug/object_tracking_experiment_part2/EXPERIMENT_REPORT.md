# Phase 2 (Part 2): Cross-Object Merge Investigation
## Companion Report to "Phase 2: Persistent Object Tracking Weight Optimization"

---

## Executive Summary

Following the original tracking-weight optimization experiment (Part 1), qualitative
review of real pipeline runs surfaced a recurring, visually obvious defect that the
Part 1 metric (track count on a 300 s test) did not capture: object nodes whose 3D
bounding box silently absorbs a second, physically distinct object — most visibly a
floor node's box swallowing a sofa, a ceiling node swallowing a ceiling-mounted pipe,
and a wall node swallowing a potted plant and the furniture in front of it.

This investigation set out to answer one question — *why does this happen, and at
which layer of the pipeline* — and ended up finding **two independent, mechanically
distinct failure modes** that produce the same visible symptom ("two objects,
one bounding box"):

1. **Segmentation-time fusion (SAM).** NanoSAM occasionally produces a single
   connected mask spanning two touching-but-distinct surfaces in one frame (e.g. a
   sofa and the floor behind it, joined by their shared contact shadow). No
   tracking-layer parameter can fix it, because the bad geometry is already fused
   before the tracker ever sees two separate candidates. *(Corrected 2026-09-06 —
   an earlier revision of this report asserted this fusion is always present from
   the track's first observation. It is not: see "Correction: Failure Mode A can
   onset mid-track" below.)*
2. **Tracking-time cross-association (the persistent tracker).** A genuinely
   separate, correctly-segmented mask (e.g. a real, distinct sofa mask) gets
   matched into an *existing*, unrelated track (e.g. a floor) by the global
   association scorer, because the scorer's dominant term (`historical`, footprint
   overlap) is blind to object identity and only two independent evidence groups
   are required to accept a match.

### Key outcome

The phase ran in two stages. **Stage 1** (§"Attempted Fixes", Fixes 1–2) tried two
weight/quorum changes, could confirm neither as both sufficient and side-effect
free, and reverted both — leaving the configuration byte-identical to Part 1 and
the diagnostic instruments as the only deliverable.

**Stage 2** (§"Stage 2", added 2026-09-06) revisited the problem with those
instruments actually producing data, and this time produced a **shipped
configuration fix**: `global_min_independent_groups: 2 → 3` together with
`global_centroid_pass_m: 0.75 → 0.55`. **Stage 3** then found the same quorum
still had a third free vote (`containment`, which is the footprint measurement
under a different threshold) and raised `global_containment_threshold` to 0.92 —
**that one is tentative and awaiting a confirmation run.** The decisive new evidence was that the
quorum's five votes are *not* independent in practice — `vertical` passes on
100% of recorded associations and `footprint` on nearly all of them, so a quorum
of 2 was always satisfied before `centroid` or `image` could object. Simulated
against 582 real association rows, the change rejects both flagged merges and
retains both user-confirmed-correct continuations. Verification on a fresh run is
pending at the time of writing.

One committed code fix (vertical-gap gating on the historical "touch" shortcut)
closes a real, distinct sub-case, but does not cover the specific floor/sofa and
wall/floor cases documented here, because those involve an object resting *at*
floor level with zero vertical gap rather than *above* it.

---

## Background: Motivation for Part 2

Part 1 optimized the global association scorer's thresholds and weights against one
metric — total track count on a 300 s replay — and reported "false positives
minimal" based on qualitative review of that same run. That review did not include
a frame-by-frame audit of *which* pixels ended up inside *which* track's accumulated
3D envelope over the track's full lifetime.

While evaluating RAP (retrieval-augmented perception) effectiveness in a separate,
concurrent investigation, per-track crop inspection surfaced tracks whose "best
crop" image visibly contained two unrelated objects. This report documents the
investigation that followed.

---

## Two Distinct Failure Modes Discovered

### Failure Mode A — Segmentation-time mask fusion

**Signature:** the very first recorded observation of the track already shows a
single, continuously-connected SAM mask boundary spanning two objects. There is no
"clean, then contaminated" transition — the fusion is present at track creation.

**Confirmed instances** (all reproduced with actual crop images pulled from
`RAP-VLM crops/<session>/<track_id>/best_update_*.jpg` and, later,
`periodic_crop_diagnostics`):

| Track / session | Objects fused | First seen |
|---|---|---|
| ceiling track, session `20260906_143808` | ceiling + suspended pipe | revision 1 |
| ceiling track (earlier session) | ceiling + "VISITORS" sign area | revision 1 |
| wall track, session `20260906_145531` | wall + mounted bench | revision 1 |
| wall track (`rsg_obj_000003`), session `20260906_150815` | wall + potted plant | revision 1 |
| floor track (`rsg_obj_000010`), session `20260906_133600` | floor + sofa/armchairs | revision 2 (see note) |
| floor track (`rsg_obj_000006`), session `20260906_141231` | floor + sofa/armchairs | revision 1 |
| wall track (`rsg_obj_000003`), session `20260906_154536` | wall + plant + furniture | first re-observed frame (obs 50 / seq 284), confirmed *before* the two flagged observations (328, 369) that a user manually identified as wrongly consolidated |

*Note on the one exception (session `133600`):* this track's revision 1 was a
genuinely clean sofa-only crop; revision 2 was the fused blob. Root-caused to the
"same-track broader mask takeover" logic (`same_track_broader_max_volume_ratio:
6.0`, `same_track_min_added_area_fraction: 0.05`) accepting a later, larger,
already-fused SAM mask as the new "best crop" for the track, because that logic
only checks area/volume ratios, not depth-consistency or shape.

**Why this happens (root cause, most likely, in order of confidence):**

1. **Weak/ambiguous boundary at the objects' shared contact shadow.** A dark object
   (sofa, furniture) resting on or against a lighter surface casts a shadow that
   creates a soft tonal gradient rather than a hard edge — exactly the kind of
   boundary a lightweight segmentation decoder is most likely to misjudge.
2. **`points_per_side` (SAM prompt grid density).** The SAM optimization
   experiment's own Phase 2.1 sweep measured monotonically *worse* segmentation
   quality at sparser grids (F1 0.582 at 6×6, down to 0.469 at 3×3 — a 19.3% drop).
   The current production value (3) is the worst-scoring setting in that sweep.
   Tested at both 3×3 and 4×4 in this investigation; the fusion recurred at both.
3. **`pred_iou_thresh` is dead code for the active backend.** This whole-mask
   confidence filter exists in the codebase and defaults to a fairly strict `0.96`,
   but the production config disables it (`0.0`) and, critically, `NanoSamBackend`
   (the backend actually running on this deployment) never reads this parameter at
   all — it is only consumed by the unused fallback `SamAutomaticMaskGenerator`
   backend class. No value of this parameter has any effect here.
4. **`mask_threshold` (per-pixel inclusion confidence).** Unlike `pred_iou_thresh`,
   this parameter *is* live for NanoSAM — it directly thresholds the per-pixel
   decoder output. Raised from `0.70` to `0.80` (near-zero aggregate cost per the
   SAM experiment's own sweep: F1 0.4977 → 0.4975) as a targeted attempt to trim
   low-confidence boundary pixels. The fusion recurred at `0.80` too (session
   `150815`), suggesting the ambiguous region's pixels are confidently — not just
   marginally — assigned to the fused mask, not hovering near the decision
   boundary.

**Conclusion for Failure Mode A:** confirmed as a real, recurring, and (so far)
unresolved segmentation-quality issue, robust to every SAM-parameter change tried.
It cannot be fixed at the tracking layer by definition — the bad geometry already
exists before two candidates are ever compared.

#### Correction: Failure Mode A can onset mid-track

The signature stated above ("present at track creation") is **wrong as a general
rule**, and was corrected on 2026-09-06 by direct review of the periodic crops for
the floor track `rsg_obj_000005` in session `20260906_143346`. The contour drawn on
each saved crop is that frame's *own raw SAM mask*, so it dates the fusion exactly:

| Observation | Sequence | What the raw single-frame mask contour shows |
|---|---|---|
| 0 | 1 | Floor only. Contour runs along the wall/floor junction *behind* the chair row. |
| 40 | 157 | Floor only. Contour hugs the chair **bases**, notching down between the legs. |
| 50 | 198 | Floor only. Same base-hugging boundary, smaller visible floor area. |
| 60 | 238 | **Fused.** Contour climbs up and over the chair **backrests**, then drops back to floor level between chairs. |

The transition therefore happens somewhere in sequences 198–238, roughly 50
observations into a track that was clean until then — not at birth. Mask area is
*not* a usable detector for it: it reads 29 347 px at observation 49 (clean) and
28 813 px at observation 60 (fused), because the newly-absorbed chair pixels are
offset by floor pixels lost to the changing viewing angle.

**Why this correction matters:**

- "Clean for many observations, then contaminated" was previously treated as the
  distinguishing signature of Failure Mode B. It is not sufficient — Mode A
  produces the same temporal pattern. The two are separable only by looking at
  whether the *raw single-frame mask* is fused (Mode A) or clean-but-misassigned
  (Mode B), which is exactly the distinction instrument #2 was built to expose.
- It rules out "SAM was already wrong when the track was born" as an explanation
  here, and points instead at viewing-angle dependence: as the robot advances down
  the corridor, the floor/chair boundary evidently becomes ambiguous enough for the
  decoder to bridge it, having been unambiguous from further away.
- It means Mode A can contaminate a track that has *already* been validated as
  clean, so a one-time inspection of a track's early crops is not evidence of
  health for the rest of its life.

### Failure Mode B — Cross-track association merge

**Signature:** a track that was clean for many consecutive observations abruptly
absorbs a *separately, correctly segmented* candidate mask belonging to a different
physical object, via the global association scorer. Diagnosed directly from
`tracking_associations_*.jsonl`, correlating consecutive-frame centroid jumps within
one `matched_track_id`.

**Confirmed instances:**

- **Floor → wall** (track `rsg_obj_000004`, depth=4m run,
  `tracking_associations_20260906_124346_final.jsonl`): centroid held stable at
  `Z≈1.15 m` for 60 frames, then a second, competing observation at `Z≈2.54 m`
  (age 1, i.e. a fresh candidate) got accepted at frame 60, and centroid drifted
  progressively up to `Z≈3.7 m` and across ~8 m of XY by the end of the track's
  life. **Confirmed absent at depth=5m** for the equivalent floor patch (traced by
  spatial location, not track-ID number, since track numbering differs between
  runs): the same physical floor patch, re-identified as `rsg_obj_000002` in the
  5 m run, stayed at `Z=1.15 m` for its entire 65-observation history. Most likely
  mechanism: at `max_depth_m=4.0`, the wall's farther pixels get filtered as
  invalid depth, leaving a truncated fragment whose 3D footprint happens to be
  close enough to the floor's to score above threshold; at 5 m the wall's full,
  undistorted extent is captured and correctly scores as a mismatch. **This does
  not mean the underlying scorer gap is fixed at 5 m** — it means this specific
  wall's geometry doesn't happen to trigger it at that range.
- **Floor → sofa** (track `rsg_obj_000005`, session `20260906_150815`,
  `tracking_associations_20260906_131201_final.jsonl`): the case worked through
  quantitatively below.

**Root cause, quantitatively (the floor→sofa case):**

The critical transition is at `sequence=264`. The centroid jump into this
transition was `0.765 m` (`0.735 m` horizontal + `0.21 m` vertical) from the
track's prior position, and the match was accepted at a translated weighted score
of **0.326** — against a `min_score` gate of `0.30`. This is the single most
marginal accepted match in the track's entire recorded history; every subsequent
observation (267→315) then matches strongly (0.88–0.94) against the now-corrupted
track, because the track's own reference point had already been dragged toward the
sofa by the one bad acceptance.

The critical number: `persistent_global_centroid_pass_m` (the hard distance cutoff
for centroid to count as an independent vote) is `0.75 m`. The observed distance,
`0.765 m`, **already exceeded this cutoff** — centroid was the one cue correctly
signalling "reject this." It didn't matter, because:

- Only `persistent_global_min_independent_groups: 2` groups are required to pass
  the quorum, and `historical` (footprint touch/overlap, weight `0.70`, 54% of the
  total recent-mode weight) plus `image` (2D IoU, weight `0.45`, 35%) apparently
  supplied enough on their own.
- `historical_score` and `image_score` are both, by construction, **pure
  proximity/overlap measures with no concept of object identity**. A sofa standing
  on a floor patch will always show real XY-footprint overlap/touch with that
  floor, and its 2D screen-space box will often overlap the floor's, *regardless
  of whether it is really the same object*. `centroid` is the only one of the
  three terms that measures the kind of separation (real 3D distance/height) that
  actually distinguishes a sofa from the floor beneath it — and it carries the
  smallest weight of the three.

**Why this is architecturally distinct from what Part 1's own vertical-gap fix (see
below) can catch:** a sofa's own bounding box necessarily spans from floor level
(its feet touch the floor) upward — its Z-range *overlaps* the floor's own Z-range
at the base by construction. Any fix gated on "is there a vertical *gap*" is
structurally blind to "object resting directly on the tracked surface," which
describes almost all real standing furniture, not just this sofa.

---

## Diagnostic Methodology: New Instrumentation Built This Phase

Three new, permanent diagnostic capabilities were built specifically to support
this investigation and future ones. All are gated behind the existing
`phase1.diagnostics.enabled` master switch (or, for the two accuracy-specific
writers, additionally behind the relevant feature's own enable flag, so they don't
create empty output when that feature is off).

### 1. Direct RAP hit/miss logging + `rap_results.csv`
A ground-truth log of every RAP classification attempt (hit or miss), with the
measured embedding distance and threshold, plus a matching per-attempt CSV
(`debug/rap_accuracy_test/session_<timestamp>/`) with manual-verification columns.
Built for the concurrent RAP-effectiveness evaluation, but relevant here as the
tooling precedent for the diagnostics below.

### 2. `PeriodicCropDiagnostics` — per-track, per-N-observations sampling
(`src/rsg/nodes/support/phase1/periodic_crop_diagnostics.py`,
`debug/object_tracking_experiment_part2/periodic_crop_diagnostics/session_<timestamp>/`;
the `tracking_quality` logs were relocated alongside it, from a path outside the
repository, so a run's crops and its association log now live together)

Unlike the pre-existing "best crop" mechanism (which only keeps the single
highest-scoring revision per track), this samples every Nth observation
(configurable via `phase1.diagnostics.periodic_crop_interval`, default 10) of
*every* track unconditionally, saving:

- the raw crop with that exact frame's own SAM mask contour drawn on it, and
- **two** geometry columns side by side in one CSV row: the *raw* per-frame
  geometry for that exact observation, and the *accumulated* local-Hydra-segment
  envelope it was just merged into (which — being built by element-wise min/max
  across every observation ever assigned to it — only ever grows, never shrinks).

This is what made the wall+plant+sofa and floor+wall/sofa cases directly visible
rather than inferred: the two geometry columns let a reviewer see, frame by frame,
whether a later observation's *raw* mask itself changed shape (Failure Mode A) or
the *accumulated* envelope quietly grew while the raw mask stayed clean (Failure
Mode B).

### 3. Per-component association scores in `tracking_associations_*.jsonl`
(`persistent_object_tracker.py` / `tracking_quality_recorder.py`)

The existing per-frame association log already recorded one blended `match_score`
per decision. This phase added four more columns — `historical_score`,
`centroid_score`, `image_score`, `vertical_score` — populated from the exact
row that won the match (`None` for a new-track decision). Before this change, root
causing the floor/sofa case required back-solving the individual component values
algebraically from the one blended number (as done above for `centroid_score`);
after this change, all four are read directly from the log for any future case.
Verified end-to-end with a synthetic reproduction before shipping (new-track case
correctly logs `None`; a real match logs real per-component values, e.g.
`historical=0.83, centroid=0.9999, image=0.098, vertical=1.0` in the verification
run).

---

## Stage 1: Attempted Fixes for Failure Mode B (all reverted)

### Fix 0 (committed, kept): vertical-gap gating on the historical "touch" shortcut

`historical_pass`'s XY-only "touching" shortcut
(`accumulated_gap_xy <= persistent_global_touch_gap_pass_m`) previously ignored Z
entirely — a floor and anything resting on top of it always have ~0 XY gap, so this
shortcut fired regardless of height separation. Fixed by additionally requiring
`vertical_gap <= persistent_max_vertical_gap_m` (0.15 m). Verified via a direct
before/after synthetic reproduction (`git stash` diff): pre-fix, a sofa 0.25 m
above a floor got a false footprint vote purely via this shortcut; post-fix, the
footprint vote correctly fails.

**Confirmed limitation:** does not cover an object resting *at* floor level
(zero vertical gap) — i.e. does not cover either of the floor/sofa or floor/wall
cases documented in this report, which is exactly why they were still observed
after this fix shipped.

### Fix 1: raise `global_recent_weight_centroid`, lower `global_recent_weight_historical`
**Tried:** `0.15→0.35` / `0.70→0.50`, later also tried `0.15→0.30` alone.
**Outcome: reverted, quantitatively insufficient in the alone-case.**

Working the real numbers: at the seq=264 transition, `centroid_score ≈ 0.31`
(computed independently via the same Gaussian formula the code uses, at distance
0.765 m). Raising only the centroid weight to `0.30` (holding `historical` fixed at
`0.70`) moves the blended score from `0.326` to approximately `0.324` — the weight
denominator grows along with the numerator, diluting the effect. The combined
change (also lowering `historical`) moves the number more, but by how much depends
critically on the *actual* `historical_score` value for that frame, which was
**not directly logged at the time** (motivating diagnostic instrument #3 above).
A synthetic reproduction attempt was inconclusive both ways (its geometry did not
precisely reproduce the real case's numbers) and could not confirm the fix would
reliably clear the threshold.

### Fix 2: raise `global_min_independent_groups` from 2 to 3
**Outcome: reverted here — plausible side effect on a legitimate consolidation.
Later re-applied deliberately as part of Fix 3 (Stage 2), once the vote
distribution showed how narrow its real effect is.**

Directly confirmed via synthetic reproduction to block the exact evidence pattern
in the floor/sofa case: with only `footprint` + one other group passing and
`centroid` failing, `min_groups=3` adds an unconditional
`insufficient_independent_evidence` rejection that fires regardless of score —
whereas `min_groups=2` accepts the same pattern whenever the blended score merely
clears `0.30`. This is a clean, degree-independent fix for the documented case.

However, after applying it and re-running, two *other* tracks (14 and 15) that a
user identified as visually similar armchairs did not consolidate, which was
suspected to be a regression from this change. Investigation found their raw
centroid distance (~0.94 m) already exceeded `centroid_pass_m` (0.75 m) even before
this change, and the run's own association log was not available to confirm
`min_groups` was the actual blocking factor (the diagnostic file for that specific
run was never flushed, likely due to an abrupt stop) — so **the regression was not
conclusively confirmed**, but the change was reverted out of caution pending better
evidence, and both `min_groups` and the Fix 1 reweight were reverted together back
to the exact Part 1 baseline.

**Net result of Stage 1:** `persistent_tracking` configuration was left
byte-identical to `FINAL_OPTIMIZED_CONFIG.yaml` (Part 1's finalized output) in
every field, re-verified via an automated 71-key diff.

---

## Stage 2: Root cause found and a fix shipped

Stage 1 ended with the instruments built but no run yet analysed through them.
Stage 2 is that analysis, performed on session `20260906_143346`
(582 scored association rows, 13 tracks, full per-component score logging active
for the first time), driven by two specific merges flagged from manual crop review:

- **Object 3 / observation 60** (sequence 330): a sofa mask absorbed into the wall
  track. Observations 70 and 80 of the same track were confirmed *correct* and had
  to keep merging — an important constraint, since it rules out any fix that simply
  makes the scorer stricter across the board.
- **Object 5 / observations 60 and 70**: the recurring floor/sofa case.

### Finding 1 — the quorum's five votes are not independent in practice

`_find_match` counts five boolean votes (`footprint`, `centroid`, `vertical`,
`image`, `containment`) and requires `pass_count >= global_min_independent_groups`
*and* `score >= min_score`. Reconstructing all five votes for every one of the 582
recorded rows (deriving centroid distance and vertical gap by inverting the logged
Gaussian scores, and reading containment from the lifecycle log) gives this
distribution:

| Votes passed | Rows |
|---|---|
| 5 | 473 |
| 4 | 48 |
| 3 | 54 |
| 2 | 7 |

`vertical` passed on **100%** of rows, and `footprint` on nearly all. The reason is
structural and self-reinforcing: `vertical_score` is a Gaussian on the *bounding-box
gap* in Z, and once a track's accumulated envelope spans the room height — which
happens early here, because SAM fuses the wall with the ceiling pipes and the floor
with the chair bases — the Z-gap to any indoor-height candidate is ~0, so the vote
is unconditionally granted. `footprint` behaves similarly for the same reason.

**Consequence:** a quorum of 2 was always satisfied by these two structurally-free
votes before `centroid` or `image` — the only two cues that can actually
distinguish a sofa from the floor beneath it — got any say. This is the mechanism
behind Failure Mode B, stated more precisely than Stage 1 managed: the problem is
not primarily the *weights*, it is that the quorum was being met without either
discriminating cue having to agree.

### Finding 2 — the two flagged merges, by vote

| Row | footprint | centroid | vertical | image | containment | votes |
|---|---|---|---|---|---|---|
| obj3 seq 330 — sofa, **must reject** | ✓ 0.60 | ✓ 0.63 m | ✓ | ✗ 0.08 | ✗ 0.60 | 3 |
| obj3 seq 350 — obs 70, **must keep** | ✓ 0.95 | ✗ 1.16 m | ✓ | ✗ 0.00 | ✓ 0.95 | 3 |
| obj3 seq 368 — obs 80, **must keep** | ✓ 0.97 | ✗ 0.85 m | ✓ | ✓ 0.97 | ✓ 0.97 | 4 |
| obj5 seq 261 — floor jump, **must reject** | ✓ 0.42 | ✗ 0.77 m | ✓ | ✗ 0.18 | ✗ 0.42 | 2 |

Note the shape of the problem: the two must-keep rows are *not* stronger than the
must-reject rows on centroid or image — obj3 seq 350 scores 0.00 on image and fails
centroid outright. They are stronger only on `containment`. Any fix therefore has
to work through the vote *count*, not through raising a score threshold.

This also resolves an open question from Stage 1 in the negative: **reweighting
alone cannot separate these cases.** The sofa row's `centroid_score` (0.45) is
*higher* than that of both correct continuations (0.07 and 0.24), so raising the
centroid weight — the intuitive fix, and the one attempted as Fix 1 — actively
rewards the wrong row. Measured, not estimated, this time.

### Fix 3 (shipped): `min_independent_groups` 2→3 and `centroid_pass_m` 0.75→0.55

`min_groups=3` alone rejects the object-5 floor jump (2 votes) but not the object-3
sofa row, which survives on exactly 3. Removing its `centroid` vote — its distance
is 0.63 m, and `global_centroid_pass_m` was 0.75 — drops it to 2 and rejects it,
while neither must-keep row depends on that vote. `0.55` was chosen over `0.60`
because both reject the identical set of rows, so the lower value buys 8 cm of
margin instead of 3 cm at no cost — margin that matters because the distances are
derived from the logged Gaussian rather than logged directly (see Open Questions).

Simulated effect across all 582 rows: **9 flip from accept to reject, 0 from reject
to accept** (raising a quorum can only reject more). Every rejected row is a large
centroid jump with negligible 2D overlap:

| Track | Seq | historical | image IoU | centroid dist | containment |
|---|---|---|---|---|---|
| obj1 | 50 | 1.00 | 0.23 | 1.42 m | 0.57 |
| obj1 | 50 | 0.57 | 0.24 | 0.98 m | 0.57 |
| obj1 | 58 | 0.88 | 0.19 | 1.23 m | 0.88 |
| obj1 | 338 | 0.81 | 0.05 | 1.16 m | 0.81 |
| obj3 | 330 | 0.69 | 0.10 | 1.16 m | 0.60 |
| **obj3** | **330** | **0.60** | **0.08** | **0.63 m** | **0.60** |
| obj3 | 396 | 0.83 | 0.01 | 1.45 m | 0.83 |
| **obj5** | **261** | **0.42** | **0.18** | **0.77 m** | **0.42** |
| obj7 | 58 | 0.51 | 0.20 | 0.60 m | 0.51 |

(Bold = the two rows the fix was designed to catch. The other seven are incidental
and share the same signature.)

**Accepted cost, stated explicitly:** a stricter quorum makes merging strictly
harder, so the objects-14/15 under-merge from Stage 1 gets *worse*, not better.
This is the same parameter that was reverted in Fix 2 for precisely that reason;
re-applying it is a deliberate prioritisation of over-merges over under-merges,
now taken with the vote distribution above as evidence that the change is narrow
(9 of 582 rows, 1.5%) rather than a blanket tightening.

**Status: shipped to `rsg_pipeline.yaml`, not yet verified on a fresh run.** The
simulation is a replay of recorded decisions, which cannot model the second-order
effect: once a bad association is rejected, the candidate starts a new track, and
every subsequent frame's association landscape differs from the recorded one.

### Finding 3 — what the fix explicitly does not address

Object 5's observations 60 and 70 — the originally-flagged floor/sofa crops — are
**not** fixed by this change and cannot be fixed at the association layer at all.
Per the correction documented under Failure Mode A above, the chair pixels at
sequence 238 are inside the *raw single-frame SAM mask itself*. When SAM hands the
tracker one mask containing two objects, there is exactly one candidate and no
association decision to make differently. What the fix does catch within that same
track is the *separate* cross-track jump at sequence 261 (0.77 m, image IoU 0.18),
which is a genuine Mode B event layered on top of the Mode A contamination.

This is the clearest demonstration in either part of this report of why the two
failure modes must be diagnosed separately: one track exhibited both, ~20
observations apart, and a fix for one is structurally incapable of touching the
other.

---

## Stage 3: containment is a fourth free vote

> **Status: the analysis below stands; the fix in it was inert for three runs.**
> `global_containment_threshold` was never parsed — `Phase1Config` had no
> matching field, so the tracker's `getattr(..., 0.90)` always fell back to the
> hardcoded literal and every value written in YAML was ignored. This was found
> in Stage 4 and only then wired up. Sessions `152244`, `170651` and `190046`
> all ran with containment at 0.90 regardless of what the config said. The
> vote arithmetic in this section is unaffected — it was computed from logged
> component scores, not from the config — but any claim here that the change
> *took effect* is wrong.

The first run after Fix 3 (session `20260906_152244`, 805 associations, 565
frames) still produced a wall track absorbing a **potted plant** — visible in the
periodic crop for `rsg_obj_000003` observation 60, sequence 323.

### The merge is 21 frames older than it looks

Observation 60 is not where it broke. Tracing the track's associations from
sequence 270 onward:

| Seq | Mask area | Obs centroid Z | Step | historical | centroid | image |
|---|---|---|---|---|---|---|
| 274 | 46 661 | 2.48 | 0.24 m | 0.959 | 0.892 | 0.907 |
| 298 | 32 984 | 2.82 | 0.33 m | 0.943 | 0.803 | 0.682 |
| **302** | **5 167** | **1.97** | **0.94 m** | **0.906** | **0.174** | **0.122** |
| 307 | 6 503 | 1.98 | 0.51 m | 0.873 | 0.592 | 0.704 |
| 323 | 9 122 | 1.93 | 0.08 m | 0.961 | 0.986 | 0.708 |

At sequence 302 the mask area collapses from 32 984 px to 5 167 px and the
centroid drops 0.85 m — the frame where SAM stops returning the wall and returns
the plant. Both genuinely independent cues rejected it correctly: centroid
distance 0.935 m (over even the old 0.75 m gate) and image IoU 0.122. They were
outvoted 3–2 by `footprint` + `vertical` + `containment`.

By observation 60 the track centroid had already been dragged onto the plant, so
that frame scores a near-perfect five-vote match (centroid distance 8 cm). **A
merge should be diagnosed at the frame where the mask identity changes, not at
the frame where it was noticed** — the flagged observation looked innocent in
every logged number.

### Why containment was the third vote

`_aabb_3d_containment` and `_aabb_overlap_fraction_3d` are the **same formula** —
both return the observation-normalised overlap volume. Containment is therefore
not independent evidence; it is the footprint overlap measured against a stricter
threshold (0.90 vs 0.30). For any small object standing inside a big surface
track's oversized envelope, `footprint`, `containment` and `vertical` are three
votes for the single geometric fact "candidate box sits inside track box" — which
is exactly a quorum of 3.

This extends Stage 2's Finding 1: the quorum has not five independent cues but
effectively **two** (`centroid`, `image`) plus a cluster of three that co-fire on
one measurement. Raising the quorum to 4 is not an option — object 3's
observation 70 in the previous run, confirmed correct by review, passes on
exactly 3 votes.

### Tentative change: `global_containment_threshold` 0.90 → 0.92

The plant merge sat at containment **0.906** — barely over the threshold — in
*both* recorded runs (`obj3 seq 301` in the older session, `obj3 seq 302` in the
newer). Raising the bar to 0.92 removes its containment vote, dropping it to 2.

Simulated across both sessions (1387 associations), 5 rows are newly rejected,
all sharing one signature — a small mask absorbed into a big track from a
distance:

| Run | Track | Seq | containment | image IoU | centroid dist | area |
|---|---|---|---|---|---|---|
| old | obj1 | 250 | 0.913 | 0.000 | 2.66 m | 7 786 |
| old | obj1 | 396 | 0.918 | 0.251 | 1.96 m | 15 637 |
| old | obj3 | 301 | 0.904 | 0.160 | 0.96 m | 5 712 |
| old | obj3 | 400 | 0.905 | 0.290 | 0.89 m | 10 980 |
| new | obj3 | 302 | 0.906 | 0.122 | 0.94 m | 5 167 |

Both rows confirmed correct by review (containment 0.9515 and 0.9678) keep a
3-point margin.

**Caveats carried by this change:**

- Containment values form a **continuum** across 0.88–0.99 with no natural gap,
  so 0.92 is a dial position, not a discovered boundary. Every increment trades
  over-merge against fragmentation.
- The margin is thin: the plant sits at 0.906, so a similar merge landing at
  0.925 would still pass. The next step is 0.935 (13 rejections across both runs
  instead of 5, all the same signature).
- Simulation replays recorded decisions and cannot model the second-order effect
  of a rejected association spawning a new track.

### Method note: the run's log nearly did not exist

`save_snapshots` is called only from `destroy_node()`
(`src/rsg/nodes/phase1.py:3483`), so the association log survives only a clean
shutdown. The first attempt at this run left crops but no log, and the analysis
above was initially attempted from crop geometry alone — which produced a
**wrong** inference (an estimated containment of ~0.11, and the conclusion that
the observed vote pattern was impossible). The measured value was 0.906. Crop
geometry gives the segment envelope, not the track envelope the scorer actually
compares against; the two differ enough to invert the conclusion. Open Question 3
(hardening the flush) is therefore not a convenience item — without the log this
investigation reached the opposite answer.

---

## Stage 4: the real mechanism, two dead config keys, and closure

Stage 4 began with a floor track (`rsg_obj_000025`, session `205022`) that had
absorbed a trashcan, a cabinet, a file resting on that cabinet, and a chair. It
ended with the over-merge mechanism identified exactly, two configuration keys
found to be silently disconnected, and the optimisation closed.

### Finding 4 — the vote model, validated to zero error

Every earlier conclusion in this report rested on reconstructing the five votes
from logged component scores. That reconstruction was *approximate*: it modelled
the footprint vote as `historical_score >= threshold`, and it left 1.3–1.6% of
accepted associations looking impossible.

Adding the footprint **touch shortcut** to the model closed the gap completely:

| Model | Accepted rows the model says should have been rejected |
|---|---|
| footprint = overlap only | 19 / 1460 (1.30%) |
| footprint = overlap **or touch** | **0 / 1460 (0.00%)** |

Zero contradictions across 1460 associations. From this point the reconstruction
is exact, and it also settles which config each run used — a question that had
been guessed at twice and got wrong both times.

### Finding 5 — the touch shortcut is what feeds surfaces

`historical_pass` grants the footprint vote through either branch:

```
(overlap_volume >= min_hist AND overlap_x >= min_axis AND overlap_y >= min_axis)
OR (accumulated_gap_xy <= touch_gap_pass_m AND vertical_compatible)
```

The second branch asks only whether the boxes *touch*. **An object resting on a
surface always touches it** — zero XY gap, zero Z gap. So every object standing
on a tracked floor received a free footprint vote no matter how little it
actually overlapped.

The floor's contamination, traced to the frame:

| Cue | Value at seq 837 | Threshold | |
|---|---|---|---|
| footprint | overlap 0.368, but **touching** | 0.50 | pass, via shortcut |
| vertical | gap 0.00 m (it rests on the floor) | ≤ 0.15 | pass |
| centroid | 0.43 m (0.32 m of it is just its height) | ≤ 0.60 | pass |
| containment | 0.368 | ≥ 0.90 | fail |
| image IoU | 0.000 | ≥ 0.30 | fail |

Three votes, quorum met. The two cues that could tell something was wrong both
objected and were outvoted — and the three that accepted are **one fact stated
three ways**: *this thing is standing on the floor, near the middle of the
patch*. That is true of every object on every floor, which is why the floor
kept eating them.

The consequence compounds. Seq 837 lifted the floor's Z ceiling from 1.30 m to
1.83 m, later to 2.30 m. From then on anything standing there was *inside* the
box, so containment and overlap began passing too — the cabinet and chair
scored 4–5 votes, not a bare 3, putting them beyond the reach of any threshold.
**The damage is done by the first off-ground object; everything after is
downstream.**

### Finding 6 — two configuration keys were silently disconnected

| Key | Defect | Consequence |
|---|---|---|
| `global_containment_threshold` | No field in `Phase1Config`, no loader line. Tracker read it via `getattr(..., 0.90)`. | Every value written in YAML was ignored for the life of the key. Three runs were interpreted as testing it. |
| `global_touch_gap_pass_m` | Loader clamped it with `max(0.0, ...)`. An XY gap is never negative, and touching boxes report exactly 0. | The shortcut could not be switched off by configuration — `<= 0.0` still fires at gap 0. Half the parameter's range was unreachable. |
| `global_vertical_score_pass` | Parsed into config, never read by the tracker. | Still dead. Left as-is and documented. |

Both defects were fixed by plumbing, not by new logic: a dataclass field plus a
loader line for the first, and removing the clamp for the second so a negative
value means "off".

**A threshold that silently does nothing is worse than one set wrong**, because
it survives every experiment that appears to test it. Two of the three
"tentative" changes documented in Stage 3 were of this kind.

### Fix 4 (shipped): disable the footprint touch shortcut

Two candidates were simulated against every recorded run:

| | Option 1: `centroid_pass` 0.40 | Option 2: touch shortcut off |
|---|---|---|
| floor injection (seq 837) | rejected | rejected |
| trashcan | rejected | rejected |
| sofa | rejected | rejected |
| pipe into ceiling | **still merges** | rejected |
| obs70 / obs80 (must keep) | kept | kept |
| extra rejections | +6 / +2 / +1 | +19 / +9 / +1 |

Option 1 shows fewer rejections but was rejected anyway, because
`global_centroid_pass_m` **doubles as the candidate search radius**
(`_candidate_track_ids`, `persistent_object_tracker.py:2300`). Tracks excluded
by a smaller radius are never evaluated and therefore leave no row to count, so
Option 1's "+6" is a floor rather than an estimate — and that same hidden
mechanism had already produced a 13 → 47 track jump when the radius was last
tightened. Option 2 changes no radius, so its higher count is its whole cost.

Shipped as `global_touch_gap_pass_m: -1.0`, which the un-clamped loader now
reads as "shortcut disabled", forcing the footprint vote to be earned through
real overlap.

### Finding 7 — depth-less masks were a track factory

Objects 79, 80, 82, 87 and 88 were reported as one wall split five ways. None of
them had any 3D geometry at all: `valid_geometry=False`, centroid (0,0,0),
volume 0, each surviving exactly one observation.

The cause is a quorum interaction. Four of the five votes — footprint, centroid,
vertical, containment — require 3D. Without it the scorer enters an explicit
degraded mode whose own comment states the design:

> *"Explicit degraded mode: image overlap plus temporal freshness. This preserves
> tracking through isolated invalid-depth frames while still requiring **two**
> independent cues."*

Maximum available: **two votes** (image + temporal). Raising the quorum to three
therefore made the degraded path unreachable, so a depth-less observation could
never associate — it could only start a new track and then die, because the next
depth-less frame could not associate with it either.

The evidence is categorical: in every run, *every single* depth-less track has
exactly one observation.

| Run | Quorum | Tracks | Depth-less tracks | With >1 observation |
|---|---|---|---|---|
| 143346 | 2 | 13 | 1 | 0 |
| 190046 | 3 | 97 | 21 | 0 |
| 194600 | 3 | 101 | 16 | 0 |

Sixteen of 101 tracks — a sixth of the fragmentation — were phantoms.

### Fix 5 (shipped): align the depth gate with the geometry threshold

The masks were surviving a rejection gate that fired only at *exactly zero*
in-range depth points, while the geometry estimator needs
`min_valid_depth_points` (20). The 1–19 band fell through: a mask whose object
lies beyond `max_depth_m` still collects a few in-range points from near-field
speckle elsewhere in the same contour — enough to clear a `== 0` test, not
enough to produce geometry. The crops show this directly: a large contour over
the far wall of an open office, plus scattered near-field specks.

The gate now rejects whenever the geometry estimator cannot produce a box,
guarded so it stays inert when the depth gather never ran (geometry disabled, or
an empty mask) rather than silently dropping every mask in that configuration.

**Result: depth-less tracks went from 16 to 0**, confirmed across three
subsequent runs.

---

## Configuration State at End of Part 2

### Configuration

`persistent_tracking` differs from `FINAL_OPTIMIZED_CONFIG.yaml` in five fields:

| Parameter | Part 1 | Part 2 final | Why |
|---|---|---|---|
| `global_min_independent_groups` | 2 | **3** | Fix 3 — a quorum of 2 was met by structurally-free votes alone |
| `global_centroid_pass_m` | 0.75 | **0.60** | Fix 3 — the 0.63 m sofa merge was still earning a position vote at 0.75. Note this also controls the candidate search radius |
| `global_historical_overlap_pass` | 0.30 | **0.50** | Stage 3 — the pipe entered the ceiling track on a footprint vote of 0.426, while legitimate continuations of that same track score 0.83–1.00 |
| `global_touch_gap_pass_m` | 0.02 | **−1.0** | Fix 4 — negative disables the touch shortcut, which handed a free footprint vote to anything resting on a tracked surface |
| `global_containment_threshold` | 0.90 | **0.92** | Stage 3 — containment is the footprint measurement re-thresholded, and supplied a third free vote at 0.906. Inert until Stage 4 wired the key up |
| `local_segments_enabled` | false | true | Unrelated, deliberate: long-object splitting into local Hydra segments |

The association **weights are untouched** — `historical` 0.70, `centroid` 0.30,
`image` 0.45, `vertical` 0.00, both `min_score` gates at 0.30. Stage 2
established the defect lives in the quorum, not the blend, and Finding 2 showed
reweighting toward the position cue would have rewarded the wrong row.

### Code changes

| Change | File | Nature |
|---|---|---|
| Vertical-gap gating on the historical touch shortcut (Fix 0) | `persistent_object_tracker.py` | Correctness fix, Stage 1 |
| Per-component association scores in the log | `tracking_quality_recorder.py`, `persistent_object_tracker.py` | Diagnostics |
| `PeriodicCropDiagnostics` | new module | Diagnostics |
| `persistent_global_containment_threshold` field + loader line | `phase1_config.py` | Plumbing — key was never parsed |
| Removed the `max(0.0, …)` clamp on `global_touch_gap_pass_m` | `phase1_config.py` | Plumbing — half the range was unreachable |
| Depth-range gate widened to the geometry threshold (Fix 5) | `phase1.py` | Correctness fix, Stage 4 |

### Rollback guidance

In order of what to relax first if tracks fragment:

1. `global_touch_gap_pass_m` back to `0.02` — restores the touch shortcut. This
   is the largest single lever and it re-opens the floor/surface over-merges.
2. `global_historical_overlap_pass` back toward `0.30`.
3. `global_centroid_pass_m` toward `0.65` — but note this re-admits the sofa
   merge, which sits at 0.63 m.
4. The quorum back to `2` only as a last resort: it re-opens every over-merge
   documented here *and* silently disables the depth-less degraded path fix's
   rationale.

Do **not** "restore" these to the Part 1 values on the assumption they drifted by
accident. Each is deliberate and traced to a specific failure in this report.

Separately, and orthogonally, this phase also iterated on SAM-side parameters in
pursuit of Failure Mode A:

| Parameter | Part 1 baseline | Tried in Part 2 | Outcome |
|---|---|---|---|
| `points_per_side` | 3 (deliberate override; experiment picked 4) | 3, 4 | Fusion observed at both |
| `mask_threshold` | 0.70 | 0.80 | Fusion still observed |
| `max_depth_m` (all 6 locations) | 5.0 | 4.0, 5.0 | Failure Mode B's floor→wall instance depth-dependent (see above); Failure Mode A unaffected by depth |

At the time of writing, `points_per_side=3`, `mask_threshold=0.80`,
`max_depth_m=5.0` are the active values, none confirmed sufficient to eliminate
Failure Mode A.

---

## Results and Observations

- **Two mechanically distinct failure modes** produce the identical visible
  symptom ("bounding box contains two objects"), and must be diagnosed and fixed
  independently — a fix at one layer cannot resolve a defect originating at the
  other. This is the central methodological finding of this phase.
- **Six-plus confirmed instances of Failure Mode A** across three different
  object-category pairings (ceiling+pipe/sign, wall+bench/plant, floor+sofa),
  robust to every SAM-parameter change attempted, strongly suggesting a systemic
  segmentation-quality limit rather than a misconfiguration.
- **At least two confirmed instances of Failure Mode B** (floor→wall, floor→sofa),
  both traced to the same underlying mechanism: a footprint/overlap-dominant score
  that cannot distinguish "same surface, later frame" from "different object,
  touching."
- **Stage 1:** every fix attempted for Failure Mode B either could not be
  quantitatively confirmed sufficient (Fix 1) or risked an unconfirmed but
  plausible regression elsewhere (Fix 2), leading to a full revert.
- **Stage 2**, with the instruments producing real data, located the defect in the
  quorum rather than the weights (`vertical` passing on 100% of 582 rows,
  `footprint` on nearly all), and shipped a two-parameter fix that rejects 9 of
  582 recorded associations including both flagged merges, while retaining both
  user-confirmed-correct continuations.
- **Reweighting was measured, not just suspected, to be the wrong lever here:** the
  bad sofa row's `centroid_score` (0.45) exceeds that of both correct continuations
  (0.07, 0.24), so raising the centroid weight rewards the wrong row. Stage 1's
  Fix 1 was aimed in a direction the data does not support.
- **Stage 4 found the actual mechanism** — the footprint touch shortcut, which
  gives a free vote to anything resting on a tracked surface — and two config
  keys that were silently disconnected, one of which had been reported in this
  very document as a shipped fix.

### Measured outcome across the campaign

| Run | Configuration | Frames | Tracks | Tracks / 100 frames | Depth-less tracks |
|---|---|---|---|---|---|
| 143346 | Part 1 baseline (quorum 2) | 400 | 13 | 3.2 | 1 |
| 152244 | quorum 3, `cpass` 0.55 | 566 | 56 | 9.9 | 5 |
| 170651 | quorum 3, `cpass` 0.55 | 588 | 47 | 8.0 | 1 |
| 190046 | + `hist` 0.50, `cpass` 0.60 | 1056 | 97 | 9.2 | 21 |
| 194600 | + touch shortcut off | 906 | 101 | 11.1 | 16 |
| 202338 | + depth gate | 604 | 57 | 9.4 | **0** |
| 203242 | + depth gate | 494 | 41 | 8.3 | **0** |

Read honestly, this is a **trade, not a win on every axis**. Fragmentation rose
from 3.2 to roughly 8.5 tracks per 100 frames — about 2.6× — and in exchange the
confirmed over-merges (floor+sofa, floor+trashcan, wall+sofa, wall+plant,
ceiling+pipe) stopped occurring. The depth gate recovered the worst of the
regression, taking the rate from 11.1 back to 8.3–9.4 by eliminating phantom
tracks entirely.

Whether that trade is favourable depends on the downstream consumer. For a
semantic scene graph, a fragmented wall is a recoverable error — the pieces can
be merged later by a geometric or semantic pass. A floor node whose bounding box
contains a sofa, a cabinet and a chair is not recoverable: the geometry is
wrong, the crop feeding the VLM shows the wrong object, and the resulting label
is wrong for every piece of it. **The campaign deliberately optimised for the
non-recoverable error.**

## Known Limitations / Open Questions

1. ~~No real-run per-component score data yet exists.~~ **Resolved in Stage 2** —
   session `20260906_143346` carries all four component scores for all 582
   associations, and they are what Fix 3 was derived from.
1b. **The boolean votes themselves are still not logged.** `_find_match` computes
   `row["centroid_distance_m"]` and `row["evidence_group_passes"]` /
   `row["independent_pass_count"]`, but none reach the association JSONL, so the
   vote table in Stage 2 was reconstructed by *inverting* the logged Gaussian
   scores — which requires assuming the sigma bucket from the track's EMA volume.
   The reconstruction is self-consistent (it reproduces the accept/reject outcome
   of every recorded row) but the individual distances carry that assumption.
   Threading those three fields through the existing `log_association_decision`
   call is a small, behaviour-free change that would remove the assumption
   entirely; it was scoped but deliberately not bundled with Fix 3.
2. **Failure Mode A has no proposed fix candidate remaining from the existing SAM
   parameter set** — `points_per_side`, `mask_threshold`, and (found to be dead
   code) `pred_iou_thresh` have all been examined. Untried directions include
   `nms_iou` (already at the Part-1-experiment's best-tested value, 0.30, so any
   further change would be venturing outside validated territory) and
   depth-discontinuity-based mask post-processing (a genuinely new mechanism, not
   a parameter tune).
3. **The 14/15 "should have merged" report was never conclusively attributed** to
   the `min_groups=3` change or ruled independent of it, because the run's
   association log did not persist. A more robust shutdown-flush guarantee for
   `tracking_quality_recorder` would prevent this gap in future investigations.
   This question is now *more* pressing, not less: Fix 3 re-applies the same
   parameter, so the pending verification run must re-check those two objects.
4. **The floor→wall depth-dependence (4 m vs. 5 m)** is explained mechanistically
   but not fully verified at the pixel level — the hypothesis (depth-range
   truncation distorting the wall's captured 3D footprint) has not been directly
   confirmed by inspecting the actual valid-depth pixel counts for that specific
   wall observation at each depth setting.
5. **Fix 3 is validated only by replay, not by a fresh run.** Re-scoring recorded
   decisions cannot capture the second-order effect: a rejected association spawns
   a new track, after which every later frame's candidate set differs from the one
   recorded. The 9-row figure is therefore a lower bound on the change's reach, and
   the true fragmentation cost can only be measured by re-running the bag.
6. **`vertical` is currently a dead vote and a dead weight.** It passed on 100% of
   582 associations, and its recent-mode weight is already 0.00, so it contributes
   nothing to the blend while still counting toward the quorum. Whether it should
   be measured differently (e.g. against the *raw* per-frame footprint rather than
   the accumulated envelope, which is what makes it degenerate) or dropped from the
   quorum count entirely was not investigated.
7. **Two views of one large planar surface cannot be merged by any current cue.**
   Three confirmed instances, all the same shape:

   | Pair | Overlap fraction | Centroid distance | Votes | What it is |
   |---|---|---|---|---|
   | obj7 / obj11 | 0.043 | 0.861 m | 2 | adjacent wall sections meeting at a seam |
   | obj55 / obj63 | 0.177 | 1.48 m | 1 | one wall, two viewing distances |
   | obj21 / obj33 | 0.030 | 0.619 m | 1 | one column, edge-on vs face-on |

   Each mask captures a different extent of the same surface — obj21 caught a
   column as a 15 cm-thick slab, obj33 caught it 1.12 m deep — so the boxes share
   almost no volume and their centres are far apart *by construction*. Every
   overlap-based cue is near zero, which is indistinguishable from "two different
   things near each other". The signal that would separate them is **planarity or
   surface orientation** — whether both masks lie on the same plane — which the
   scorer does not compute. obj21/obj33 additionally sits 2 cm outside
   `centroid_pass_m`, but admitting it would re-admit the sofa merge at 0.63 m;
   the viable window is about 1 cm wide on distances derived by inverting a
   Gaussian, which is not a basis for a threshold.
8. **The degraded path for depth-less observations is unreachable, not repaired.**
   Fix 5 discards those masks rather than restoring their ability to associate.
   A genuinely close object that flickers out of depth for one frame is now
   dropped for that frame instead of being carried on image + temporal. The
   alternative — letting the degraded path require 2 votes regardless of the
   quorum, which is what its own comment says was intended — was scoped and not
   taken.
9. **Rejected candidates are still not logged.** For a `new_track` decision the
   log records no scores at all, so every under-merge question in this campaign
   ended in inference rather than measurement. `_find_match` computes all five
   cues for every candidate it evaluates; none of it leaves the function. This is
   the single highest-value diagnostic gap remaining, and it blocked a definitive
   answer on obj7/obj11, obj55/obj63 and obj21/obj33 alike.

## Lessons Learned

1. **A single quantitative metric (track count) can hide qualitatively distinct
   failure modes.** Part 1's 26% track-reduction result is a real, valid
   improvement for its target case (same object, different viewing angle), but it
   does not certify the absence of a *different* problem (cross-object fusion) that
   happens not to move that particular metric much on that particular test.
2. **A score composed of "proximity" cues (footprint overlap, 2D IoU) is
   structurally unable to reject a genuinely different but touching object** — no
   amount of reweighting fully compensates for a term that is, by definition,
   blind to identity; the fix needs either a hard quorum requirement on the one
   identity-sensitive cue (centroid) or a genuinely new signal (e.g. depth
   discontinuity, appearance/embedding similarity) that current scoring lacks
   entirely.
3. **Diagnose before reweighting.** Two reweighting attempts in this phase were
   evaluated only by rough estimation or an imperfect synthetic reproduction,
   because the actual per-component numbers were not being logged. Building the
   direct logging (instrument #3) should have been the *first* step, not a
   follow-up after two inconclusive tuning attempts — a lesson directly informing
   the recommended order of operations for Part 3.
4. **A quorum is only as strong as its weakest vote's independence.** The scorer
   was designed around five "independent evidence groups," but two of them
   (`vertical`, `footprint`) are computed against the track's *accumulated*
   envelope, which grows monotonically. Once that envelope is large, those votes
   become unconditional — the quorum silently degrades from "five cues, two must
   agree" to "two free passes." Any future evidence-group design should be checked
   for this: measure each vote's pass rate across a real run before trusting it as
   independent evidence. In this run the check took one pass over the association
   log and immediately explained a defect that two rounds of weight tuning had not.
5. **The intuitive lever and the correct lever can point opposite ways.** "It's
   over-merging, so weight the position cue higher" is a reasonable prior, and it
   is wrong here: the incorrect merge scored *better* on centroid than the correct
   continuations did, because the correct ones were large view-angle changes on a
   genuinely large object. Only measured per-component values exposed this; the
   estimate-based reasoning of Stage 1 could not have.
6. **Distinguish "the crop looks wrong" from "the accumulated geometry is wrong."**
   Several apparent merges (ceiling track's gradual XY drift, a track's Z range
   spanning a genuinely sloped architectural feature) turned out to be legitimate
   same-surface continuation, not bugs — the two-column raw/accumulated geometry
   logging in instrument #2 was specifically designed to make this distinction
   checkable rather than a judgment call from a single crop image.

## Future Work

1. ~~Capture a real Failure Mode B occurrence with instrument #3 active.~~
   **Done in Stage 2.** ~~Re-attempt `global_min_independent_groups: 3`.~~
   **Done as Fix 3.** Both superseded by item 1' below.
0'. **Confirm or revert the Stage 3 containment change** — the only open item
   blocking Part 2 from being closed. Re-run and check whether the wall track
   still absorbs the potted plant at the sequence-302 equivalent (the frame where
   mask area collapses, not the later frame where it becomes visible). If it
   still merges, read that row's containment: below 0.92 means the vote analysis
   is wrong somewhere; above 0.92 means step the threshold to 0.935. If the merge
   is gone, also confirm the wall track did not fragment where it should have
   continued.
1'. **Verify Fix 3 on a fresh run** — confirm the sofa mask at the sequence-330
   equivalent starts its own track rather than joining the wall track, confirm the
   same for the floor track's sequence-261 equivalent, compare total track count
   against the 13 of session `20260906_143346` to quantify the fragmentation cost,
   and re-check objects 14/15, which this change pushes in the wrong direction.
2. **Log the boolean votes and raw centroid distance** (Open Question 1b) so the
   next quorum analysis reads them instead of inverting Gaussians.
3. **Investigate a depth-discontinuity mask-splitting post-process** as a
   candidate fix for Failure Mode A, since parameter tuning within NanoSAM's
   existing knobs has been exhausted without resolving it.
4. **Harden `tracking_quality_recorder`'s shutdown flush** so an abrupt stop
   (Ctrl-C, crash) does not silently lose the run's association log, which
   happened at least twice during this investigation and blocked a conclusive
   answer on the 14/15 question.
5. **Consider an appearance-similarity or depth-consistency term** as a genuinely
   new (not reweighted) addition to the global association score, specifically to
   give the scorer a signal that can positively distinguish "same object" from
   "different object, touching" — something no combination of the current four
   terms (historical, centroid, image, vertical) is designed to provide.

## Conclusion

This phase replaced an assumed single problem ("over-merging") with a precise
taxonomy of two independent problems, and then — once the instrumentation built in
Stage 1 was actually producing data — root-caused one of them and shipped a fix
for it.

The sequence is the point. Stage 1 tried two parameter changes reasoned from
estimates, could confirm neither, and reverted both; its honest output was
diagnostic capability rather than a tuned config. Stage 2 used that capability on a
real run and found the defect was not where either Stage 1 attempt aimed: not in
the score weights, but in the quorum, where two of the five "independent" evidence
votes had degraded into unconditional passes because they are measured against an
accumulated envelope that only ever grows. That finding also retired the intuitive
reweighting hypothesis on measured grounds — the bad merge scored *better* on the
position cue than the correct merges did. The resulting fix
(`min_independent_groups` 2→3, `centroid_pass_m` 0.75→0.55) is narrow by
construction: 9 of 582 recorded associations, all with the same low-evidence
signature, including both flagged merges and neither confirmed-correct
continuation.

Stage 4 then found what Stage 2 had only approximated. The defect was not merely
that some votes were structurally easy — it was one specific branch, the
footprint **touch shortcut**, which grants its vote to anything whose box merely
touches the track's. An object resting on a floor always touches it. Every
object standing on any tracked surface therefore arrived with a free vote, and
needed only two more from cues that are equally automatic once the surface's
envelope has grown. Disabling that branch is the single change that stopped the
floor from eating a trashcan, a cabinet, a file and a chair.

The same stage found two configuration keys that had never been connected to
anything: one with no field in the config dataclass at all, and one whose loader
clamped away exactly the half of its range that would have switched a behaviour
off. **One of them had already been written up in this report as a shipped fix
and credited with a result it could not have produced.** A silently inert
threshold is worse than a wrong one, because it survives every experiment that
appears to test it — and here it survived three.

What closes the phase is a measured trade rather than a clean victory.
Fragmentation rose roughly 2.6-fold; the confirmed over-merges stopped. That
trade was chosen deliberately, because a fragmented wall is a recoverable error
and a floor node containing a chair is not. Three limitations travel with the
result and are documented above: two views of one large planar surface remain
unmergeable by any current cue; Failure Mode A is untouched, since no association
parameter can act when SAM delivers one mask containing two objects; and the
degraded path for depth-less observations is now unreachable under a quorum of
three, worked around by discarding those masks rather than by repairing the path.

The most transferable result is methodological. Every wrong turn in this campaign
came from reasoning about a number nobody was logging — an estimated component
score in Stage 1, a segment envelope mistaken for a track envelope in Stage 3, a
config key assumed to be wired in Stage 3 and disproved in Stage 4. Every
correct turn came from a measurement: the per-component scores, the crop
contours, the vote reconstruction validated to zero error against 1460 rows.
**Build the instrument before turning the dial** — and verify the dial is
connected to something before believing what it appears to tell you.
