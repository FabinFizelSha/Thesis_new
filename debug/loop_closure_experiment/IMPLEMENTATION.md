# Loop-closure re-anchoring — implementation reference

Branch `pipeline-split-lean`. Commits `21b8f0d` (core) + `575562d` (`/tf` design
+ regression tests). This document describes **what was built and how it
works**; `IMPLEMENTATION_PLAN.md` holds the design rationale and the 5-step
plan history, `README.md` the loop-window analysis and test bags.

---

## 1. Problem

Phase 1 accumulates a session-persistent cache of physical-object geometry —
per track and per local segment: `centroid_3d`, the global-union 3D box
(`bbox_3d_min/max`), the last-observation box (`last_bbox_3d_min/max`), and a
hashed 2-D spatial index built from those. Every coordinate is in the mapping
frame as it stood **when the observation was projected**.

When a SLAM back-end (Hydra LCD) detects a loop and corrects accumulated
odometry drift, it folds the correction into the **`map → odom`** TF — there is
no explicit "loop closed" event, the changed transform *is* the signal, and the
corrected scene graph is re-published. At that instant the entire phase-1 cache
is stale by the same rigid step `ΔT`. A previously-mapped object re-observed
after the correction lands `‖ΔT‖` away from its cached track and the associator
spawns a **duplicate** instead of re-recognising it.

**Goal:** when `map → odom` steps, rigid-transform the whole cache by that step
so re-observations re-associate, and fold the duplicate that the drift window
already produced.

---

## 2. Design decisions

| decision | why |
|---|---|
| **Single rigid `ΔT` applied to the whole cache.** | A pose-graph loop closure is non-rigid (the correction is distributed along the trajectory), but reproducing that needs the re-published DSG. A single `ΔT` from the live `map → odom` step is exact for objects near the loop point and a good first-order approximation elsewhere. Accepted for v1. |
| **No `/hydra/backend/dsg` subscription.** | That message is large and high-rate. `map → odom` on `/tf` carries all the signal we act on. |
| **Two plain `/tf` + `/tf_static` subscriptions, not `tf2_ros.TransformListener`.** | `TransformListener(spin_thread=True)` adds the phase-1 node to a *second* executor on its own thread — a real risk of changing `frame_callback` threading. Plain subscriptions run on the node's existing executor exactly like every other phase-1 callback. |
| **Disabled by default (`phase1.loop_closure.enabled: false`).** | The production pipeline runs ground-truth odometry with `map == odom == world` and Hydra LCD off, so `map → odom` is always identity. The feature is inert until a front end actually publishes a non-identity `map → odom`. |
| **Decision maths in a ROS-free module.** | `nodes/support/phase1/loop_closure.py` holds the SE(3) step computation so it is unit-testable without a ROS node. |
| **Re-anchor + merge run on the tracking thread, under the tracker lock.** | Same lock `associate()` takes, called from the same thread immediately before `begin_frame()` — race-free with association and with the RAP/VLM result appliers. |

---

## 3. Data flow

```
        /tf, /tf_static  (map -> odom, published by the back-end)
                 │
                 ▼   phase1 executor thread
   Phase1SemanticCoordinator._on_tf(msg)
     • match header.frame_id==map_frame & child_frame_id==odom_frame
     • quat_to_rot + translation  ->  self._pending_map_odom   (under _map_odom_lock)
                 │
                 ▼   phase1 tracking/publish thread, once per frame,
                     in run_rap_and_metadata() just before begin_frame()
   Phase1SemanticCoordinator._maybe_reanchor_on_loop_closure(timestamp_sec)
     • enabled?  pending is not None?           else return
     • first reading -> store baseline, return
     • loop_closure_delta(old, new, thresholds)
          -> None if |Δt| < min_translation_m AND Δθ < min_rotation_deg  -> return
          -> (R_delta, t_delta, |t_delta|, Δθ_deg)
                 │
                 ▼   PersistentObjectTracker, under self._lock
   reanchor_all(R_delta, t_delta, stamp=timestamp_sec)
     • every track + every segment:
         centroid_3d          -> R·p + t
         bbox_3d_min/max       -> 8-corner AABB
         last_bbox_3d_min/max  -> 8-corner AABB
     • _refresh_spatial_index(track)   (rebuild hashed cells)
     • record last_reanchor() summary
                 │
                 ▼   (if loop_closure.merge_duplicates)
   merge_reanchor_duplicates(correction_translation_m=|t_delta|,
                             now_sec=timestamp_sec, ...)
     • drift pass   : fold the fresh revisit track into its stale older identity
     • overlap pass : fold any now-coincident fragments
                 │
                 ▼
   WARN log  +  /rsg/phase1/loop_closure_event   (std_msgs/String, JSON)
```

---

## 4. Component reference

### 4.1 `src/rsg/nodes/support/phase1/loop_closure.py` (new, ROS-free)

**`quat_to_rot(x, y, z, w) -> np.ndarray (3×3)`**
Unit-quaternion to rotation matrix; returns identity for a near-zero quaternion.

**`loop_closure_delta(rot_old, trans_old, rot_new, trans_new, *, min_translation_m, min_rotation_deg) -> Optional[(R_delta, t_delta, |t_delta|, angle_deg)]`**
The rigid step between two `map → odom` readings:

```
R_delta = R_new · R_oldᵀ
t_delta = t_new − R_delta · t_old        # so that  (R_delta,t_delta) · T_old == T_new
angle   = degrees(arccos(clip((tr(R_delta) − 1) / 2, −1, 1)))
```

Returns **`None`** when `|t_delta| < min_translation_m` **and**
`angle < min_rotation_deg` — the jitter gate that keeps a noisy but
un-corrected `map → odom` from ever triggering a re-anchor.

### 4.2 `persistent_object_tracker.py` — module helpers (new)

| helper | contract |
|---|---|
| `_rigid_point(p, R, t)` | `R @ p + t`; `None` passes through. |
| `_rigid_aabb(min, max, R, t)` | Transforms all **8 corners**, returns fresh axis-aligned `(min, max)`. Exact for pure translation; tightest AABB envelope under rotation. `None`/`None` passes through. |
| `_aabb_iou_3d(a_min, a_max, b_min, b_max)` | Symmetric 3-D IoU = `∩ / (a + b − ∩)`. |
| `_track_sort_key(track_id)` | Deterministic ordering key (`int` when the id is all-digits, else the string). |
| `_forget_spatial_index(track_id)` | Removes every hashed-cell entry for a track id. Extracted from `_refresh_spatial_index`, which now calls it. |

### 4.3 `PersistentObjectTracker.reanchor_all(rotation, translation, *, stamp=None) -> int`

Rigid-transforms **every** cached track and segment, holding `self._lock`:

* moved: track & segment `centroid_3d`, `bbox_3d_min/max`, `last_bbox_3d_min/max`
* rebuilt: the spatial index, once per track via `_refresh_spatial_index`
* **untouched:** EMA state, `seen_count`, all labels / evidence, Hydra slot ids,
  `active_segment_slot_id`, the shared best-crop registries, `bbox_2d`
* **left as-is:** `bbox_volume_m3` — invariant under a rigid motion

Records a summary retrievable via **`last_reanchor()`**
(`{stamp_sec, translation[3], rotation[9], track_count}`), returns the track
count.

### 4.4 `PersistentObjectTracker.merge_reanchor_duplicates(...) -> int`

```
merge_reanchor_duplicates(*, correction_translation_m=0.0, now_sec=None,
                          recent_window_sec=5.0, distance_slack_m=0.6,
                          min_iou_3d=0.30, max_centroid_distance_m=None) -> int
```

Two conservative, label-gated passes under `self._lock`; returns the number of
tracks removed. `max_centroid_distance_m` defaults to
`persistent_global_centroid_pass_m`.

**Drift pass** (only when `now_sec` is given). The loop-closure duplicate pair
straddles the drift — the older track was mapped in early odom, the fresh one in
late odom — so a single rigid `ΔT` cannot bring them together. For each track
last seen within `recent_window_sec` (the *fresh* copy), find the older
label-compatible track (`first_seen` strictly earlier, itself *not* fresh) whose
XY centroid gap is smallest and `≤ drift_radius`:

```
drift_radius = max(distance_slack_m, |correction_translation_m| · 1.25 + distance_slack_m)
```

Merge via `_merge_track_pair(keep=older, drop=fresh, reason="loop_closure_drift",
adopt_drop_geometry=True)` — the older identity survives but **takes the fresh
track's drift-corrected geometry**.

**Overlap pass** (always). Over the remaining tracks, ordered by
`(−seen_count, id)`, use `_candidate_track_ids` to find any pair that genuinely
overlaps now (`_aabb_iou_3d ≥ min_iou_3d`) with centroid gap
`≤ max_centroid_distance_m` and compatible labels; fold the lower-`seen_count`
one via `_merge_track_pair(..., reason="post_reanchor_overlap",
adopt_drop_geometry=False)` — observation-weighted centroid, union box. Ordinary
fragmentation cleanup.

### 4.5 `PersistentObjectTracker._merge_track_pair(keep, drop, *, iou_3d, distance_m, reason, adopt_drop_geometry)`

`keep` always retains `track_id`, label evidence, and the earlier `first_seen`.

| field | `adopt_drop_geometry=True` (drift) | `False` (overlap) |
|---|---|---|
| `centroid_3d` | `drop`'s, verbatim | `(kc·keep + dc·drop) / (kc+dc)` |
| `bbox_3d_min/max` | `drop`'s, verbatim | union |
| `last_bbox_3d_*` | `drop`'s | `drop`'s if newer |

Always: `bbox_volume_m3` recomputed; `seen_count += drop.seen_count`;
`last_seen_*` taken from whichever is later; `label_evidence` /
`label_observations` summed; **`drop`'s segments re-keyed onto `keep`**
(`segment.segment_id` rewritten to `"{keep.track_id}:slot_{hydra_label_id}"`;
Hydra slot ids are globally unique so there is no key collision, and each
segment keeps its own slot so fuser presence continuity is preserved); an audit
entry appended to `keep.metadata["reanchor_merged_from"]`
(`{track_id, seen_count, iou_3d, distance_m, reason, adopted_geometry, slot_ids}`);
`drop` dropped from the spatial index and, by the caller, from `self._tracks`.

### 4.6 `PersistentObjectTracker._reanchor_labels_compatible(a, b) -> bool`

`False` if `a.semantic_kind != b.semantic_kind`. Otherwise compares the first
non-generic label found per track across
`semantic_label → canonical_label → raw_vlm_label → raw_rap_label`
(generic set: `"", unknown, unknown object, object, thing, stuff, background`).
Both strongly labelled → must be equal; otherwise geometry decides.

### 4.7 `PersistentObjectTracker.debug_snapshot() -> List[dict]`

Read-only per-track dump for tests / diagnostics:
`{track_id, internal_object_id, seen_count, semantic_kind, canonical_label,
semantic_label, centroid_3d, bbox_3d_min, bbox_3d_max, segment_slot_ids,
reanchor_merged_from}`.

### 4.8 `phase1.py` — `Phase1SemanticCoordinator`

**`__init__`** (only when `loop_closure_enabled`): stores `_lc_map_frame` /
`_lc_odom_frame`; creates `_tf_sub` (`/tf`, RELIABLE, depth 100) and
`_tf_static_sub` (`/tf_static`, RELIABLE + TRANSIENT_LOCAL) both routed to
`_on_tf`; creates the `loop_closure_pub`. State: `_last_map_odom`,
`_pending_map_odom` (both `Optional[(R, t)]`), `_map_odom_lock`.

**`_on_tf(msg)`** — executor thread. For each transform matching
`map_frame → odom_frame` (leading `/` stripped), converts to `(R, t)` and
stores it in `_pending_map_odom` under the lock. No decision logic.

**`_maybe_reanchor_on_loop_closure(timestamp_sec)`** — tracking thread, called
in `run_rap_and_metadata()` immediately before
`self.persistent_tracker.begin_frame()`:

1. `if not self._loop_closure_enabled: return` — the disabled fast path.
2. Read `_pending_map_odom` under the lock; `None` → return.
3. First reading → store as `_last_map_odom`, return.
4. `loop_closure_delta(...)`; `None` → refresh baseline, return.
5. `reanchor_all(R_delta, t_delta, stamp=timestamp_sec)`.
6. If `loop_closure_merge_duplicates`: `merge_reanchor_duplicates(...)`.
7. Refresh baseline; `WARN` log; publish `loop_closure_event`.

### 4.9 Config — `phase1.loop_closure` (`phase1_config.py` + `rsg_pipeline.yaml`)

| key | field | default | meaning |
|---|---|---|---|
| `enabled` | `loop_closure_enabled` | `false` | master switch; when false, zero runtime footprint |
| `map_frame` | `loop_closure_map_frame` | `map` | drift-corrected frame |
| `odom_frame` | `loop_closure_odom_frame` | `odom` | drifting odometry frame |
| `min_translation_m` | `loop_closure_min_translation_m` | `0.05` | re-anchor gate (translation) |
| `min_rotation_deg` | `loop_closure_min_rotation_deg` | `0.5` | re-anchor gate (yaw) |
| `merge_duplicates` | `loop_closure_merge_duplicates` | `true` | run `merge_reanchor_duplicates` after a re-anchor |
| `merge_recent_window_sec` | `loop_closure_merge_recent_window_sec` | `5.0` | "fresh" cutoff for the drift pass |
| `merge_distance_slack_m` | `loop_closure_merge_distance_slack_m` | `0.6` | drift-pass radius slack |
| `event_topic` | `loop_closure_event_topic` | `/rsg/phase1/loop_closure_event` | diagnostic topic |

---

## 5. `/rsg/phase1/loop_closure_event`

`std_msgs/String`, one JSON object per re-anchor:

```json
{
  "timestamp_sec": 389.47,
  "delta_translation_m": [-0.80, 0.02, 0.0],
  "delta_translation_norm_m": 0.8003,
  "delta_rotation_deg": 1.13,
  "tracks_reanchored": 214,
  "tracks_merged": 1,
  "map_frame": "map",
  "odom_frame": "odom"
}
```

---

## 6. Behaviour by state (regression stance)

| state | runtime effect |
|---|---|
| `enabled: false` (default) | `_maybe_reanchor_on_loop_closure` returns on its first line. No `/tf` subscriptions, no publisher. **Byte-identical to the pre-feature pipeline.** |
| `enabled: true`, no `map → odom` ever published (current GT-odom setup) | Two `/tf` subscriptions + `_on_tf` string-compares each transform (never matches) + one lock-guarded `None` read per frame. `_pending_map_odom` stays `None`; `reanchor_all` / `merge_reanchor_duplicates` are **never called**; tracker state is untouched. |
| `enabled: true`, `map → odom` present but below both thresholds | As above plus `loop_closure_delta` returns `None` each frame; baseline is refreshed, nothing else. |
| `enabled: true`, `map → odom` steps past a threshold | One `reanchor_all` (+ optional merge), one WARN line, one `loop_closure_event`. |

---

## 7. Tests

Run: `PYTHONPATH=src/rsg python3 -m pytest src/rsg/tests/ -q -p no:anyio`
(the `-p no:anyio` disables a broken system pytest plugin).
Result: **54 passed, 2 pre-existing unrelated failures**
(`test_global_frame_assignment.py::test_spatial_search_keeps_only_plausible_3d_candidates`,
`test_object_geometry.py::test_estimate_matches_independent_reference` — both
predate this work).

### `src/rsg/tests/test_loop_closure_decision.py` (8) — ROS-free

`quat_to_rot` identity + Rz(90°); `loop_closure_delta` returns `None` for
identity / sub-threshold translation / sub-threshold rotation / no-change from a
non-identity baseline; translation step; rotation step; and the composition
identity `ΔT · T_old == T_new` for a general `(R, t)` pair.

### `src/rsg/tests/test_reanchor.py` (10)

`_seed_track` inserts fully-formed `PersistentObjectTrack` + one
`PersistentObjectSegment` directly, so geometry is controlled exactly.

| test | asserts |
|---|---|
| `test_reanchor_pure_translation_shifts_every_coordinate` | every track & segment centroid/box shifts by exactly `t`; `bbox_volume_m3` unchanged; `last_reanchor()` summary correct |
| `test_reanchor_identity_is_a_noop` | `reanchor_all(I, 0)` changes no coordinate and no spatial-index cell; a no-signal `merge_reanchor_duplicates()` removes nothing |
| `test_reanchor_rotation_uses_corner_aabb` | Rz(90°) box matches the independent 8-corner AABB; volume preserved; centroid rotated |
| `test_reanchor_rebuilds_spatial_index` | after `reanchor_all(I, [20,0,0])` a `_candidate_track_ids` query at the new location returns the track and one at the old location does not |
| `test_drift_pass_folds_revisit_duplicate_into_older_identity` | older id survives, `seen_count` summed, adopts the fresh centroid, carries both slots, one `reanchor_merged_from` entry `reason="loop_closure_drift"` |
| `test_drift_pass_ignores_incompatible_labels` | `"chair"` vs `"table"` → no merge |
| `test_drift_pass_respects_the_correction_radius` | 3 m gap with a 0.2 m reported correction → no merge |
| `test_overlap_pass_folds_coincident_fragments` | two overlapping same-label tracks → one survivor, `reason="post_reanchor_overlap"` |
| `test_no_merge_without_signal` | two well-separated tracks, no args → nothing removed |
| `test_end_to_end_reanchor_then_merge` | pre-loop tracks share one rigid `ΔT`; the revisit duplicate is then folded by the drift pass |

---

## 8. Not done / prerequisites for a live run

The current pipeline cannot exercise the re-anchor path end to end:
ground-truth odometry, `map_frame == odom_frame == world`, static identity
`world → odom`, Hydra `enable_lcd: false` ⇒ `map → odom` is always identity.

A real end-to-end test needs **either**:

* a genuinely drifting front end (VIO on `/tesse/imu/noisy/imu` + stereo, or
  dead-reckoned noisy IMU) with Hydra `enable_lcd: true` and distinct
  `map` / `odom` frames, so Hydra's own LCD produces the correction; **or**
* the same, with the correction supplied by a stand-in
  (`loop_jump_injector.py` steps `map → odom` by a scripted `ΔT`).

A synthetic-drift substitute on the GT bag was considered and rejected — with no
real drift to cancel it only re-tests the arithmetic the unit tests already
cover.

Also deferred: on a drifting front end the preprocessor stamps
`frame.camera_pose` in raw `odom` (`preprocessor.py:479-483`), so phase 1 would
additionally need to compose `map → odom` into each incoming pose before
projection. No-op under GT odom; not yet written.

---

## 9. File manifest

| file | change |
|---|---|
| `src/rsg/nodes/support/phase1/loop_closure.py` | **new** — `quat_to_rot`, `loop_closure_delta` |
| `src/rsg/nodes/support/phase1/persistent_object_tracker.py` | `_rigid_point`, `_rigid_aabb`, `_aabb_iou_3d`, `_track_sort_key`, `_forget_spatial_index`, `reanchor_all`, `last_reanchor`, `merge_reanchor_duplicates`, `_merge_track_pair`, `_reanchor_labels_compatible`, `debug_snapshot` |
| `src/rsg/nodes/phase1.py` | `_on_tf`, `_maybe_reanchor_on_loop_closure`, call site, `/tf` subs + publisher in `__init__`, `DurabilityPolicy` import |
| `src/rsg/nodes/support/phase1/phase1_config.py` | 9 `loop_closure_*` fields + `phase1.loop_closure` parse block |
| `src/rsg/config/rsg_pipeline.yaml` | `phase1.loop_closure:` block (`enabled: false`) |
| `src/rsg/tests/test_loop_closure_decision.py` | **new** — 8 tests |
| `src/rsg/tests/test_reanchor.py` | **new** — 10 tests |
| `debug/loop_closure_experiment/loop_jump_injector.py` | **new** — scripted `map → odom` step (mechanism aid) |

Commits: `21b8f0d` (core: tracker + phase1 + config + `test_reanchor.py` +
`loop_jump_injector.py`), `575562d` (`/tf` subs replace `TransformListener`;
`loop_closure.py` extraction; `test_loop_closure_decision.py` +
`test_reanchor_identity_is_a_noop`).
