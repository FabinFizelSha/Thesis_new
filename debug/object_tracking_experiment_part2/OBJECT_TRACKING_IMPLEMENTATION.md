# Object Tracking — Implementation Reference (A to Z)

Complete walkthrough of persistent object tracking in Phase 1: every stage a SAM
mask passes through, every decision rule, every parameter, and the known failure
modes of each. Companion to `EXPERIMENT_REPORT.md`, which documents *why* the
current parameter values were chosen; this document describes *what the code
does*.

**Primary sources**

| Component | File |
|---|---|
| Tracker core | `src/rsg/nodes/support/phase1/persistent_object_tracker.py` (3006 lines) |
| Per-frame driver | `src/rsg/nodes/phase1.py`, `run_rap_and_metadata` |
| Geometry | `src/rsg/nodes/support/phase1/object_geometry.py` |
| Config schema | `src/rsg/nodes/support/phase1/phase1_config.py` |
| Config values | `src/rsg/config/rsg_pipeline.yaml`, `phase1.persistent_tracking` |
| Diagnostics | `tracking_quality_recorder.py`, `periodic_crop_diagnostics.py` |

---

## 0. The problem being solved

SAM produces an unordered set of 2D masks per frame with no identity. The
tracker's job is to decide, for each mask, whether it is a **new physical
object** or **another observation of one already seen**, and to maintain a
persistent 3D estimate for each object across the session.

Two errors are possible and they are not symmetric:

- **Over-merge** — two physically distinct objects share one track. The
  bounding box is wrong, the crop shown to the VLM contains the wrong object,
  and the resulting semantic label is wrong for every part of it. Not
  recoverable downstream.
- **Under-merge (fragmentation)** — one physical object becomes several tracks.
  Wasteful and visually messy, but each fragment's geometry and label are
  individually correct, and a later geometric or semantic pass can merge them.

The current configuration deliberately trades the second for the first.

---

## 1. Pipeline overview

```
RGB-D frame
   │
   ▼
[A] SAM segmentation ─────────────► List[SamMask]  (mask, bbox_2d, area_px)
   │                                  backend NMS + area filter applied here
   ▼
[B] Per-mask geometry              ObjectGeometryEstimator.estimate()
   │   largest-contour filter          → centroid_3d, bbox_3d_min/max, volume
   │   depth gather + back-projection  → valid_geometry flag
   ▼
[C] Depth-range gate               drop masks that cannot produce geometry
   ▼
[D] Frame-level planning           prepare_frame_assignments()
   │   A2 redundancy suppression       one big mask vs. several small ones
   │   A3 nested suppression           duplicate masks of the same track
   │   Hungarian assignment            one track per mask, per frame
   ▼
[E] Association                    associate() → _find_match()
   │   candidate pre-filter            spatial hash
   │   five evidence votes             quorum + weighted score
   ▼
[F] Track update                   _update_track_geometry()
   │   EMA centroid/volume, min-max box accumulation
   ▼
[G] Local segment assignment       _assign_local_segment()
   │   splits long objects into per-Hydra-slot sections
   ▼
[H] Crop registry                  best-crop selection per track
   ▼
[I] Semantic dispatch              settling window → RAP → VLM
   ▼
[J] Publish                        Hydra semantic/instance label images
```

Stages A–C run on the segmentation thread; D–J on the tracking/publish thread.

---

## 2. [A] Segmentation

`NanoSamBackend.segment()` (`backends.py`) is the only backend used on the
Jetson deployment. It prompts a TensorRT MobileSAM decoder on a
`points_per_side × points_per_side` grid, thresholds the decoder output at
`mask_threshold`, applies IoU-based NMS at `nms_iou`, and caps the result at
`sam_max_masks`.

| Parameter | Value | Effect |
|---|---|---|
| `points_per_side` | 3 | Prompt grid density. Fewer prompts, coarser masks |
| `mask_threshold` | 0.80 | Per-pixel inclusion confidence |
| `nms_iou` | 0.30 | Overlap above which two masks are deduplicated |
| `max_masks` | 8 | Per-frame cap after NMS |
| `min_mask_pixels` | 3500 | Masks smaller than this are dropped |
| `pred_iou_thresh` | — | **Dead for this backend.** Only read by the unused `SamAutomaticMaskGenerator` fallback |

`SegmentationStage._restore_to_original` then rebuilds each mask at full image
resolution, re-checking `min_mask_pixels`. **This is the last size filter before
the tracker** — nothing downstream re-applies it, which matters if masks are
ever split or synthesised later in the pipeline.

**Known limitation (Failure Mode A):** SAM sometimes returns one connected mask
spanning two touching-but-distinct surfaces — floor + sofa, wall + ceiling
pipes, floor + chairs. No tracking parameter can undo this, because only one
candidate ever reaches the tracker. It is *not* restricted to a track's first
frame: a mask can be clean for 50 observations and fuse later as the viewing
angle changes.

---

## 3. [B] Geometry estimation

`ObjectGeometryEstimator.estimate(mask, depth, camera_info, tx, rot_m)` converts
one binary mask into 3D.

1. **Contour filter.** `tracking_crop_manager.get_filtered_mask()` keeps only the
   single largest contour (dropping contours under 200 px) and returns a new
   array. This affects **geometry only** — the mask stored on the track, used for
   crops, and passed to frame assignment is the original, unfiltered one.
2. **Sampling.** Mask pixels are subsampled with `projection_stride: 4`, so a
   26 000-px mask yields roughly 6 600 depth samples.
3. **Depth validity.** A sample is valid if finite and within
   `[min_depth_m, max_depth_m]` = `[0.30, 5.0]`. `depth_valid_points` counts them.
4. **Give-up test.** If `valid_count < min_valid_depth_points` (20), the function
   returns with `valid_geometry=False` and reason `too_few_valid_depth_points`,
   **before** back-projection — so the expensive step is already skipped.
5. **Back-projection.** Valid pixels → camera coordinates via `fx, fy, cx, cy`,
   then to world via `rot_m @ p + tx`.
6. **Statistics.** Centroid by `median` (robust to foreground/background depth
   outliers), plus axis-aligned min/max.
7. **Thin-object padding.** Two passes, because a floor or wall is nearly
   two-dimensional and a zero-thickness box breaks volume ratios:
   - Z: if thickness < `min_assumed_depth_m` (0.30), extend **down** if the box
     centre is below Z = 1.0, otherwise **up**.
   - X/Y: if thickness < 0.15 m, extend away from a hardcoded scene centre of 2.5.

> **Caveat worth knowing.** The Z-padding rule assumes floors sit below world
> Z = 1.0. In the recorded sessions the floor sits at Z ≈ 1.15, so it is padded
> *upward* as though it were a ceiling. The constants `1.0` and `2.5` are
> hardcoded and environment-specific.

Output keys consumed downstream: `centroid_3d`, `bbox_3d_min`, `bbox_3d_max`,
`bbox_volume_m3`, `bbox_2d`, `mask_area_px`, `depth_valid_points`,
`valid_geometry`.

---

## 4. [C] Depth-range gate

`phase1.py`, in the first pass of `run_rap_and_metadata`:

```python
depth_valid_points = metadata.get("depth_valid_points")
if (config.reject_masks_fully_outside_depth_range
        and depth_valid_points is not None
        and int(depth_valid_points) < int(config.min_valid_depth_points)):
    depth_range_keep.append(False)
```

A mask that cannot produce geometry is dropped before tracking, RAP, VLM and
Hydra. Two details matter:

- The threshold is the geometry estimator's own `min_valid_depth_points`, not
  zero. A mask whose object lies beyond `max_depth_m` still collects a few
  in-range points from near-field speckle elsewhere in the same contour —
  enough to clear a `== 0` test, not enough to produce geometry.
- The `is not None` guard keeps the gate inert when the depth gather never ran
  (geometry disabled, or an empty mask). Without it, disabling object geometry
  would silently drop every mask in the pipeline.

**Why this matters:** without 3D, four of the five evidence votes are
structurally false, so such an observation can never satisfy a quorum of three.
It could only ever start a new track and then die at one observation. Before
this gate was widened, 16 of 101 tracks in one run were exactly these phantoms.

Results are tracked in a list parallel to `prepared` rather than by skipping the
append, so `prepared` / `sam_masks` / `keep_mask` stay index-aligned.

---

## 5. [D] Frame-level planning

`prepare_frame_assignments(observations)` runs **once per frame**, after all
geometry is computed and before any track is mutated. It returns a keep/suppress
flag per observation and installs forced matches that `associate()` consumes.

### A2 — track-aware redundancy suppression

If one large mask's area is ≥ `redundancy_union_coverage_threshold` (0.80)
covered by the union of ≥ `redundancy_min_children` (2) smaller masks, and those
children each already match distinct established tracks with better utility, the
large mask is suppressed. This handles SAM emitting both a fused blob and its
constituent parts in the same frame — but only when the parts already have
identities. It cannot help when SAM emits *only* the fused blob.

### A3 — same-track nested suppression

When two masks with high mutual containment would both route to the same track,
one is suppressed, preventing double-counting.

### E — Hungarian assignment

`_find_match` is run in preview mode for every surviving observation, producing a
utility per (mask, track) pair. A Hungarian solve (`_hungarian_maximize`, with
`_greedy_maximize` as fallback) then enforces **one track per mask per frame**,
eliminating SAM output-order bias. Tracks already claimed in the current frame
are excluded via `_frame_used_track_ids`.

---

## 6. [E] Association — the core decision

`_find_match(centroid, volume, bbox_2d, bbox_3d_min, bbox_3d_max, timestamp_sec, …)`

### 6.1 Candidate pre-filter

`_candidate_track_ids()` narrows the search using a spatial hash:

- **Footprint cells** — grid cells covering the observation's 3D box, padded by
  `max(continuation_gap_m, revisit_overlap_gap_m)`.
- **Centroid cells** — cells within a radius of **`global_centroid_pass_m`**.

> **This double duty is a trap.** `global_centroid_pass_m` is both the centroid
> *vote threshold* and the *search radius*. Tightening it removes tracks from
> consideration entirely, and an unevaluated candidate leaves no row in any log —
> so the effect is invisible to replay analysis. Tightening it 0.75 → 0.55 once
> tripled the track count, roughly five times what row-level simulation predicted.

### 6.2 Mode selection

```
age_sec = timestamp_sec − track.last_seen_timestamp_sec
recent_mode = age_sec <= continuation_max_age_sec   (8.0 s)
```

Recent and revisit modes differ in weights, `min_score`, the image threshold, and
the centroid distance metric (3D in recent mode, XY-only in revisit).

### 6.3 The five evidence votes

| Vote | Passes when | Parameter |
|---|---|---|
| `footprint` | `(overlap_volume ≥ hist_pass AND overlap_x ≥ min_axis AND overlap_y ≥ min_axis)` **OR** `(gap_xy ≤ touch_gap_pass_m AND vertical_compatible)` | `global_historical_overlap_pass` 0.50, `global_min_axis_overlap` 0.20, `global_touch_gap_pass_m` **−1.0** |
| `centroid` | `distance ≤ global_centroid_pass_m` | 0.60 |
| `vertical` | `vertical_gap ≤ persistent_max_vertical_gap_m` | 0.15 |
| `image` | `bbox_2d IoU ≥ min_2d_iou` (recent) / `revisit_min_2d_iou` (revisit) | 0.30 / 0.75 |
| `containment` | `containment ≥ global_containment_threshold` | 0.92 |
| `temporal` | added **only** when `reliable_3d` is false | — |

Accepted when `pass_count ≥ global_min_independent_groups` (**3**) **and**
`score ≥ min_score` (0.30) **and** not `hard_2d_contradiction`.

#### The votes are not independent — three structural traps

1. **`containment` is `footprint` re-thresholded.** `_aabb_3d_containment` and
   `_aabb_overlap_fraction_3d` compute the *identical* quantity —
   observation-normalised overlap volume. Containment passing implies footprint
   passing, unless footprint came via the shortcut.
2. **The footprint touch shortcut fires for anything resting on a surface.** An
   object standing on a floor has zero XY gap and zero Z gap, so it earns the
   footprint vote regardless of how little it overlaps. This was the single
   largest source of over-merges and is now disabled by the negative threshold.
3. **`vertical` is a gap test, not a height test.** Resting on a surface means
   zero gap. It passed on **100%** of 582 associations in one analysed run. Its
   recent-mode weight is also 0.00, so it contributes nothing to the score while
   still counting toward the quorum.

Consequence: for a small object standing inside a large surface's grown
envelope, `footprint`, `containment` and `vertical` are **one geometric fact
stated three ways** — exactly a quorum of three — while `centroid` and `image`,
the only cues that can object, are outvoted.

### 6.4 Scoring

```
score = (w_hist·historical + w_recent·recent + w_centroid·centroid
         + w_vertical·vertical + w_image·iou + w_contain·containment) / Σw
```

Recent-mode weights: historical 0.70, centroid 0.30, image 0.45, vertical 0.00,
recent 0.00, containment 0.00 (Σ = 1.45).

Component definitions:

- `historical_score = max(overlap_volume, gap_score if gap_xy > 0 else overlap_volume)`,
  where `gap_score = gaussian(gap_xy, revisit_overlap_gap_m=0.4)`.
- `centroid_score = gaussian(distance, sigma)` with sigma **size-scaled** by the
  track's EMA volume: 0.50 below 5 m³, 0.70 below 20 m³, else 0.90. Separate from
  the hard `centroid_pass` test, which is not size-scaled.
- `iou_score` = 2D IoU against the track's **last-seen** 2D box, not an
  accumulated one.
- `gaussian(v, σ) = exp(−0.5·(v/σ)²)`.

**Degraded mode.** When `reliable_3d` is false, scoring becomes
`0.70·iou + 0.30·temporal`, and a `temporal` vote is added. Its comment states
the design intent: *"preserves tracking through isolated invalid-depth frames
while still requiring two independent cues."* Since four votes need 3D, the
maximum reachable is **two** — so under a quorum of three this path is
unreachable. Currently worked around by discarding such masks at stage [C]
rather than by repairing the path.

**Hard 2D contradiction.** When `global_block_2d_on_3d_contradiction` is true and
reliable 3D exists, a match passing only on image IoU while failing both
footprint and centroid is rejected outright.

The accepted match is returned as `1.0 − score`, i.e. a **cost** where lower is
better. This is what appears as `match_score` in the association log — a value of
0.017 is an excellent match, not a poor one.

---

## 7. [F] Track update

`_update_track_geometry()` mixes two update rules, and confusing them causes
misreadings:

| Field | Rule |
|---|---|
| `centroid_3d` | EMA: `α·old + (1−α)·new`, `centroid_update_alpha` = 0.5 |
| `bbox_volume_m3` | **EMA**, same α |
| `bbox_3d_min` / `bbox_3d_max` | **Element-wise min/max — monotonic, only ever grows** |
| `last_bbox_3d_min/max` | Overwritten with the current observation |
| `bbox_2d` | Overwritten |

The accumulated box never shrinks. Once a wall track absorbs ceiling pipes, its
Z range spans the room forever, and every subsequent containment and vertical
test against it is trivially satisfied. **This is the mechanism that makes a
single bad association permanent** — the envelope it creates keeps admitting
further intruders at 4–5 votes rather than a marginal 3.

`bbox_volume_m3` being an EMA while the box is min/max means a track's reported
volume can *fall* while its box grows.

---

## 8. [G] Local segments

A long object — corridor floor, wall run, ceiling — should not be one Hydra node
with a 9 m box. `_assign_local_segment()` splits one persistent track into
several `PersistentObjectSegment`s, each owning its own Hydra slot, while the
`track_id` remains the identity for crops, RAP and VLM.

Matching order for each new observation:

1. **Open segment, would stay under cap** — `candidate_span ≤ max_xy_span_m`
   (6.0) and `gap ≤ local_segment_gap_m` (0.20). Score `gap + 0.01·centre_dist`.
   Geometry expands.
2. **Open segment, would exceed cap** — the seam belongs here: mark
   `segment.closed = True` and match without expanding. Freezing at the cap
   prevents an at-capacity segment from absorbing further observations through
   the centroid fallback, which previously produced multi-metre overlaps.
3. **Closed segment** — identity-only re-check within `gap_limit` or
   `revisit_distance` (1.5 m). Geometry never expands again.
4. **2D fallback** — only when either side lacks 3D, bounded by
   `local_segment_max_2d_iou_age_sec` and `local_segment_min_2d_iou`. If both
   sides have 3D, a failed 3D association is authoritative.
5. **No match** — allocate a new segment and Hydra slot.

**Documented limitation.** The cap governs the tracker's own bookkeeping only. It
cannot shrink or split an oversized *single-frame* mask: when a closed segment is
re-matched, the entire current mask is still published under that segment's
label, and Hydra's frontend integrates every pixel into that label's node with no
size cap of its own. A hallway floor can put 7–9 m of extent in one mask with no
multi-frame accumulation at all. The real fix is per-mask 3D splitting before
`_assign_local_segment`, which is not implemented.

---

## 9. Track lifecycle

**Birth** — `_new_track()` allocates a track id (`rsg_obj_%06d`), an instance id,
a Hydra slot, and the first segment. Capped by `max_tracks` (10 000) and
available slots.

**Death** — tracks are retired by the coordinator's staleness policy; the
lifecycle log records `death_frame`.

**Merge** — `_merge_track_pair(keep, drop)` folds one track into another, with
`adopt_drop_geometry` selecting whether the survivor takes the absorbed track's
geometry or blends by observation count.

**Re-anchoring** — `reanchor_all(rot, trans)` rigidly transforms every track's
geometry after a map→odom correction, so the cache survives loop closure.

**`merge_reanchor_duplicates()`** — two passes that fold split identities:
- *drift pass*: a freshly re-observed track folds into an older compatible track
  within `|correction|·1.25 + distance_slack_m`.
- *overlap pass*: any pair with 3D AABB IoU ≥ `min_iou_3d` (0.30) and centroids
  within `max_centroid_distance_m` is folded — described in its own docstring as
  "ordinary fragmentation cleanup".

> **This is dead code in practice.** It is called from exactly one place, the
> loop-closure handler, and fires only on a map→odom correction. On ground-truth
> odometry no correction ever occurs, so the only track-to-track fragmentation
> cleanup in the system never runs. Its own docstring says "never in the
> steady-state association path."

---

## 10. [H]–[J] Crops, semantics, publishing

**Crop registry.** Each observation offers a crop; the best per track is retained
by a quality score. `_retire_track_crop` clears the entry once the classification
result is emitted — so anything needing the crop must capture it at enqueue time,
not look it up later.

**Semantic dispatch.** After a settling window, `prepare_active_for_labeling()`
returns tracks ready to classify. Each is dispatched **once**, asynchronously:
RAP retrieval first when enabled, falling back to the VLM on a miss or low
confidence. The SAM→Hydra path never blocks on either.

**Publishing.** The active segment's slot id is written into the Hydra semantic
and instance label images. Labels resolve per slot, so a track owning several
segments propagates one label to all of them.

---

## 11. Parameter reference

### Quorum and gates

| Parameter | Value | Meaning |
|---|---|---|
| `global_min_independent_groups` | 3 | Votes required to accept |
| `global_recent_min_score` | 0.30 | Weighted-score floor, recent mode |
| `global_revisit_min_score` | 0.30 | Weighted-score floor, revisit mode |
| `global_historical_overlap_pass` | 0.50 | Overlap for the footprint vote |
| `global_min_axis_overlap` | 0.20 | Per-axis minimum alongside the above |
| `global_touch_gap_pass_m` | **−1.0** | Negative ⇒ touch shortcut disabled |
| `global_centroid_pass_m` | 0.60 | Centroid vote **and search radius** |
| `global_centroid_sigma_m` | 0.50 | Base sigma, size-scaled at 5 / 20 m³ |
| `global_containment_threshold` | 0.92 | Containment vote |
| `max_vertical_gap_m` | 0.15 | Vertical vote and shortcut compatibility |
| `min_2d_iou` / `revisit_min_2d_iou` | 0.30 / 0.75 | Image vote |
| `global_block_2d_on_3d_contradiction` | true | Veto image-only matches |

### Weights (recent / revisit)

| Component | Recent | Revisit |
|---|---|---|
| historical | 0.70 | 0.60 |
| centroid | 0.30 | 0.25 |
| image | 0.45 | 0.35 |
| vertical | 0.00 | 0.00 |
| recent | 0.00 | 0.00 |
| containment | 0.00 | 0.00 |

### Geometry and segments

| Parameter | Value |
|---|---|
| `projection_stride` | 4 |
| `min_valid_depth_points` | 20 |
| `min_depth_m` / `max_depth_m` | 0.30 / 5.0 |
| `centroid_method` | median |
| `reject_masks_fully_outside_depth_range` | true |
| `centroid_update_alpha` | 0.5 |
| `continuation_max_age_sec` | 8.0 |
| `local_segments_enabled` | true |
| `local_segment_max_xy_span_m` | 6.0 |
| `local_segment_gap_m` | 0.20 |
| `local_segment_revisit_distance_m` | 1.5 |

### Keys that are parsed but never read

| Key | Status |
|---|---|
| `global_vertical_score_pass` | Loaded into `Phase1Config`, never read by the tracker. The vertical vote uses `max_vertical_gap_m` |
| `pred_iou_thresh` | Not read by `NanoSamBackend` |
| `max_vertical_center_delta_m` | `_aabb_center_delta_z` is computed and stored in the evaluation row, but nothing in the global scorer reads it |

Two further keys were disconnected until 2026-09-06 and are now wired:
`global_containment_threshold` had no dataclass field at all, and
`global_touch_gap_pass_m` was clamped with `max(0.0, …)`, making exactly the
half of its range that disables the shortcut unreachable.

---

## 12. Diagnostics

All gated behind `phase1.diagnostics.enabled`.

| Instrument | Output | Purpose |
|---|---|---|
| `TrackingQualityRecorder` | `tracking_associations_*.jsonl`, `tracking_lifecycles_*.json`, `tracking_frames_*.jsonl` | One row per accepted association with all five component scores; per-track lifecycle with `global_association_components` |
| `PeriodicCropDiagnostics` | `session_*/crops/*.jpg` + `periodic_observations.csv` | Every Nth observation of every track: crop with that frame's own mask contour, plus raw *and* accumulated geometry side by side |
| `RapAccuracyDiagnostics` | `rap_results.csv` | Every RAP attempt, hit or miss, with distance and threshold |

**Reading the logs correctly:**

- `match_score` is a **cost** (`1 − score`); lower is better.
- `prev_bbox_volume_m3` is an **EMA**, not the accumulated box volume.
- The crop contour is that frame's **raw mask**, not the accumulated envelope —
  which is what makes SAM-level fusion visually distinguishable from
  tracking-level merges.
- The CSV's segment columns are the **segment** envelope, not the **track**
  envelope the scorer compares against. The two differ, and confusing them
  inverted one root-cause conclusion during the campaign.
- Files are named with `datetime.utcnow()` at **save** time, while crop session
  directories use local time at **construction**. On a UTC+2 host these differ by
  two hours and do not indicate different runs.

### Gaps

- **Rejected candidates are not logged.** A `new_track` row carries no scores, so
  "why didn't these two merge?" can only be answered by inference. `_find_match`
  computes all five cues for every candidate; none of it leaves the function.
  This is the highest-value diagnostic gap remaining.
- **Boolean votes are not logged.** `row["evidence_group_passes"]` and
  `row["independent_pass_count"]` are computed and discarded. Vote counts must be
  reconstructed by inverting the logged Gaussians and assuming a sigma bucket.
- **Snapshots are written only in `destroy_node()`.** A hard kill loses the whole
  run's association log. This has happened more than once.

---

## 13. Failure modes and where they live

| Failure | Layer | Signature | Fixable in the tracker? |
|---|---|---|---|
| SAM mask fuses two objects | Segmentation | One connected contour spanning both, visible in the crop | **No** — only one candidate ever exists |
| Object on a surface joins that surface's track | Association | Footprint via touch + vertical + centroid = 3 votes | Yes — touch shortcut disabled |
| Small object inside a grown envelope | Association | Containment ≈ 1.0, 4–5 votes | Only by preventing the *first* intruder |
| Depth-less mask spawns a phantom track | Geometry / quorum | `valid_geometry=False`, exactly 1 observation | Yes — depth gate |
| Two views of one large plane stay split | Association | Overlap < 0.2, centroid > 0.6 m, ≤ 2 votes | **No** — needs a planarity cue that does not exist |
| Adjacent sections of one surface stay split | Association | Meet at a seam, overlap ≈ 0.04 | **No** — same reason |

### Diagnosing a merge

Do **not** start at the observation where the problem is visible. By then the
track has usually been contaminated for many frames and every cue looks
excellent. Instead:

1. Find the frame where **mask area collapses or jumps**, or where the observation
   centroid steps by more than the usual per-frame delta. That is the injection.
2. Read that row's five component scores from the association log.
3. Check which votes were structurally free — touch-based footprint, zero-gap
   vertical, containment inside a grown envelope — against which actually
   discriminated.
4. Confirm visually with the periodic crop for that observation, remembering the
   contour is the raw single-frame mask.

Worked example (obj25, floor): the reported problem was at observation 120, the
cabinet. The injection was at **sequence 837**, roughly 90 frames earlier —
footprint 0.368 via touch, vertical gap 0, centroid 0.43 m, image IoU 0.000.
That lifted the floor's Z ceiling from 1.30 m to 1.83 m, after which everything
standing on the floor was contained and scored 4–5 votes.

---

## 14. Change history

| Date | Change | Rationale |
|---|---|---|
| Part 1 | Weight optimisation, quorum 2 | 86 → 64 tracks on a 300 s replay |
| 2026-09-06 | Vertical-gap gating on the touch shortcut | Floor vs. object resting above it |
| 2026-09-06 | Per-component scores logged | Root-causing from measurement, not estimate |
| 2026-09-06 | `PeriodicCropDiagnostics` | Separate SAM fusion from tracking merges |
| 2026-09-06 | Quorum 2 → 3, `centroid_pass` 0.75 → 0.60 | Free votes were meeting a quorum of 2 |
| 2026-09-06 | `historical_overlap_pass` 0.30 → 0.50 | Pipe entered a ceiling track at 0.426 |
| 2026-09-06 | `containment_threshold` wired + 0.92 | Key had no config field; inert for three runs |
| 2026-09-06 | Touch shortcut disabled (`−1.0`) | Free footprint vote for anything on a surface |
| 2026-09-06 | Depth gate widened to the geometry threshold | 16 of 101 tracks were depth-less phantoms |

See `EXPERIMENT_REPORT.md` for the measurements behind each.
