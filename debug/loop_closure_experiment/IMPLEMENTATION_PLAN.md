# Loop-closure re-anchoring — implementation plan

Status: **step 3 implemented + step 4 harness partly built.** Steps 1–2 done
(code pushed & verified; loop windows identified and test bags cut).

Landed (behind `phase1.loop_closure.enabled: false`, so production is unchanged):

| piece | where |
|---|---|
| `_rigid_point`, `_rigid_aabb`, `_aabb_iou_3d`, `_track_sort_key` | `persistent_object_tracker.py` (module helpers) |
| `PersistentObjectTracker.reanchor_all(rotation, translation, *, stamp=None) -> int` | rigid-transforms every track+segment `centroid_3d` / `bbox_3d_min/max` / `last_bbox_3d_min/max`, rebuilds spatial index, records `last_reanchor()` |
| `PersistentObjectTracker.merge_reanchor_duplicates(*, correction_translation_m, now_sec, recent_window_sec, distance_slack_m, min_iou_3d, max_centroid_distance_m) -> int` | drift pass (stale↔fresh pair within `\|correction\|·1.25 + slack`, survivor adopts fresh geometry) + overlap pass (weighted blend) |
| `PersistentObjectTracker.debug_snapshot()`, `last_reanchor()` | read-only dumps for tests / diagnostics |
| `_forget_spatial_index(track_id)` | extracted from `_refresh_spatial_index` |
| `Phase1SemanticCoordinator._maybe_reanchor_on_loop_closure(timestamp_sec)` + `_quat_to_rot` | `phase1.py`; TF `Buffer`/`TransformListener(spin_thread=True)` in `__init__`; call site at the `begin_frame()` line |
| `/rsg/phase1/loop_closure_event` (`std_msgs/String` JSON) | published per re-anchor |
| `phase1.loop_closure` config block | `phase1_config.py` (`loop_closure_*` fields) + `rsg_pipeline.yaml` |
| `src/rsg/tests/test_reanchor.py` | 9 unit tests (translation / rotation-AABB / spatial-index / drift-merge / overlap-merge / label-gate / radius-gate / end-to-end), all green |
| `debug/loop_closure_experiment/loop_jump_injector.py` | ROS node: `map→odom` identity until `t_jump_sec`, then step `(dx,dy,dz,dyaw_deg)` (optional `ramp_sec`) |

Still to do: §5c `run_loop_test.sh` end-to-end script + §5d baseline, and the
front-end / frame-split prerequisite for step 5 (below).

---

## 0. The mental model (confirmed)

Camera odometry (`odom`) drifts. The SLAM back-end (Hydra LCD) detects a loop,
estimates the accumulated drift, and folds the correction into the
**`map → odom`** TF. It does *not* emit an explicit "loop closed" event; the
corrected `map → odom` transform *is* the signal. The corrected DSG is
re-published by the back-end.

Phase 1 caches persistent-object geometry (`centroid_3d`, 3D boxes, spatial
index) that was accumulated under the *pre-correction* `map → odom`. After a
jump, that whole cache is stale by the same rigid `ΔT`. Re-anchor it once,
in place, and re-observations of the same object fall back inside the revisit
gates and re-associate instead of spawning duplicates.

```
ΔT = mapTodom_new  ∘  inverse(mapTodom_old)          # SE(3)
for every cached point p:   p' = ΔT · p
```

---

## 1. Scope

### In
* **L1 — re-anchor on `map → odom` jump.** Phase 1 watches `map → odom`; on a
  step change it computes `ΔT` and rigid-transforms every track & segment, then
  rebuilds the spatial index.
* **L1.5 — post-re-anchor duplicate merge.** One pass that folds tracks which
  were created for the *same* physical object during the pre-correction window
  (they only overlap *after* re-anchoring).

### Optional (fast follow, own commit)
* **L4 — revisit-aware 2D fallback.** After a re-anchor, briefly widen the
  `_find_match` "revisit" branch (raise age ceiling / relax 2D-IoU) so the first
  few frames post-correction re-associate more readily.

### Out (explicitly not this change)
* No subscription to `/hydra/backend/dsg` in phase 1 (large message; TF is
  enough).
* No change to the physical-object association maths (`_find_match`,
  `_update_track_geometry`) beyond L4's temporary gate widening.
* No change to local-segment split logic.
* No re-projection of *past* crops / evidence images — only the cached geometry
  moves.

---

## 2. Prerequisite (blocks step 5, not step 3/4)

The current pipeline runs GT odom, `map_frame == odom_frame == world`, static
identity `world → odom`, Hydra `enable_lcd: false` → `map → odom` is always
identity. Nothing to react to.

To exercise the path end-to-end, **one** of:

* **(A) real drift** — front end on VIO / noisy IMU, Hydra `enable_lcd: true`,
  distinct `map` / `odom` frames. Largest change, depends on a working drifting
  front end.
* **(B) synthetic jump** — keep GT odom; a tiny `loop_jump_injector` node
  publishes `map → odom` = identity until a scripted bag time `T_jump`, then a
  fixed `ΔT` (e.g. 0.8 m + 3°). Directly tests §3/§4. **This is what the step-4
  harness uses.** `uHumans2_loop_L1` / `_L3` provide the genuine revisit
  geometry so the post-jump re-anchored cache lines up with real
  re-observations.

Decision needed: build the injector (B) now, and treat (A) as later
integration? (recommended)

---

## 3. Implementation — L1 core

### 3a. `PersistentObjectTracker.reanchor_all(delta_R, delta_t, *, stamp)`  (new)

`persistent_object_tracker.py`, near `_refresh_spatial_index`.

```
def reanchor_all(self, R, t, *, stamp=None):
    with self._lock:
        n = 0
        for track in self._tracks.values():
            _rigid_track_inplace(track, R, t)
            for seg in track.segments.values():
                _rigid_segment_inplace(seg, R, t)
            self._refresh_spatial_index(track)
            n += 1
        self._last_reanchor = (stamp, R, t, n)
        return n
```

Fields to transform (all `Optional[np.ndarray]`, skip `None`):

| dataclass | fields |
|---|---|
| `PersistentObjectTrack` | `centroid_3d`, `bbox_3d_min/max`, `last_bbox_3d_min/max` |
| `PersistentObjectSegment` | `centroid_3d`, `bbox_3d_min/max`, `last_bbox_3d_min/max` |

* point:  `p' = R @ p + t`
* AABB:   transform all **8 corners**, retake `min` / `max` (a non-zero yaw
  correction tilts an axis-aligned box; corner method is exact and cheap).
* `bbox_volume_m3` — invariant under rigid motion, leave. `bbox_2d` — image
  space, leave. EMA/`seen_count`/labels — untouched.
* Must hold `self._lock` (same lock `associate()` takes).

Helpers `_rigid_track_inplace` / `_rigid_segment_inplace` + `_aabb_rigid(mn,mx,R,t)`
in the same module.

### 3b. Phase 1 — watch `map → odom`, fire the re-anchor

`phase1.py`:

1. **Add a TF listener** (phase 1 currently only *broadcasts* TF, via
   `TransformBroadcaster`; it reads camera pose off the `RsgFrame` message, not
   from TF). Add `tf2_ros.Buffer` + `tf2_ros.TransformListener(buffer, self)` in
   `__init__`, guarded by a new config flag.
2. Config (`rsg_pipeline.yaml`, `phase1.loop_closure`):
   ```
   loop_closure:
     enabled: false            # master switch (keep false until prereq met)
     map_frame: map
     odom_frame: odom
     min_translation_m: 0.05   # ignore sub-threshold jitter
     min_rotation_deg: 0.5
     merge_duplicates: true    # run L1.5 after a re-anchor
   ```
3. In `process_frame`, just before `self.persistent_tracker.begin_frame()`
   (phase1.py:1069), call `self._maybe_reanchor_on_loop_closure()`:
   * `tf = tf_buffer.lookup_transform(map_frame, odom_frame, Time())` — newest.
     `LookupException` / `ExtrapolationException` → return (no correction yet).
   * On first success store `self._last_map_odom` and return.
   * Else `ΔT = T_new ∘ inv(T_old)`. If `‖Δt‖ < min_translation_m` **and**
     `Δyaw < min_rotation_deg` → just refresh `_last_map_odom`, return.
   * Else: `n = persistent_tracker.reanchor_all(ΔR, Δt, stamp=frame.header.stamp)`;
     log `WARN "loop-closure re-anchor: Δt=… Δyaw=… tracks=n"`; if
     `merge_duplicates` → `persistent_tracker.merge_reanchor_duplicates()`;
     refresh `_last_map_odom`.
   * Also publish a one-shot diagnostic on `/rsg/phase1/loop_closure_event`
     (`std_msgs/String` JSON: stamp, Δt, Δyaw, n_tracks, n_merged) for the
     harness + fuser visibility.

   Cost: one `lookup_transform` + a cheap SE(3) compare per frame; the O(tracks)
   walk only on an actual jump (rare). Negligible.

### 3c. Camera-pose frame — RESOLVED

`preprocessor.py:401,479-483`: the camera pose is
`t_odom_camera = t_odom_base @ t_base_camera`, where `t_odom_base` is an
interpolated `/tesse/odom` sample straight out of `OdomBuffer.lookup()`. No TF,
no `map → odom`. `make_pose_msg` then stamps it `frame_id = world_frame`.

**So the incoming `camera_pose` is in the raw `odom` frame** (labelled `world`
because today `world == odom == map`). It is **not** corrected by anything.

Consequence for a drifting-front-end + LCD config: phase 1 must, going forward,
left-multiply each incoming pose by the live `map → odom` before 3D projection,
so new observations land in the `map` (corrected) frame and stay consistent with
the re-anchored cache. That is an added ~3 lines in the frame path *plus* the
§3b jump handler. (With GT odom / identity `map → odom` it is a no-op, so it can
land now and stay dormant.)

### 3d. Rigid ΔT is an approximation — accepted for v1

A real pose-graph loop closure re-optimises the **whole trajectory**; the
correction is non-rigid (distributed along the path) and the back-end
re-publishes the full corrected DSG. Applying the single live `map → odom` step
`ΔT` rigidly to every cached track is a first-order approximation — exact for
objects near the loop point, increasingly loose for objects mapped far from it.

v1 accepts this: it is O(tracks) arithmetic off one TF read, needs no DSG
subscription, and fixes the dominant failure (duplicate spawn on re-observation
right after the closure). A later refinement could ingest the re-published DSG
node deltas for a per-object correction, but that is explicitly out of scope
here (and re-introduces the big-message subscription we are avoiding).

---

## 4. Implementation — L1.5 duplicate merge

`PersistentObjectTracker.merge_reanchor_duplicates()` (new). Right after a
re-anchor, for track pairs that now (post-ΔT) satisfy the existing "revisit"
association gate (`persistent_revisit_min_2d_iou`,
`persistent_global_revisit_min_score`, `revisit_overlap_gap_m`) **and** are
label-compatible:

* keep the older / more-observed track (lower `track_id` index, higher
  `seen_count`);
* union its 3D box with the loser's, `seen_count += loser.seen_count`, merge
  `label_evidence` / `label_observations`;
* **segments**: re-key the loser's segments onto the winner; each keeps its own
  Hydra slot (no slot freed — presence continuity matters more than slot
  economy at 10 000-slot pool);
* record `winner.metadata["merged_from"] = [...]`;
* emit a `duplicate_merged` diagnostic per merge;
* delete the loser from `self._tracks`; `_refresh_spatial_index(winner)`.

Candidate generation via the existing `_candidate_track_ids` on the winner's
re-anchored box, so it stays O(local), not O(tracks²).

Leaves the physical-association code path unchanged for the steady state; this
runs *only* in the re-anchor callback.

---

## 5. Test harness (step 4)

`debug/loop_closure_experiment/`:

### 5a. `loop_jump_injector.py`  (ROS node)
Params: `t_jump_sec` (bag-clock), `dx dy dz`, `dyaw_deg`, `map_frame`,
`odom_frame`, `rate_hz` (default 50). Publishes `map → odom` on `/tf`:
identity before `t_jump`, fixed `ΔT` after. Subscribe `/clock` for bag time.
Replaces the static identity `world → odom` bridge for the test.

### 5b. `test_reanchor.py`  (pytest, no ROS)
Pure unit test of `reanchor_all` / `merge_reanchor_duplicates`:
* build a tracker, inject 3 synthetic tracks + segments with known geometry;
* `reanchor_all(R(10°), [1,0.5,0])`;
* assert every centroid / box moved by exactly ΔT (8-corner AABB expectation),
  `bbox_volume_m3` unchanged, spatial index cells updated (a
  `_candidate_track_ids` query at the new location returns the track, the old
  location does not);
* duplicate case: two tracks 0.3 m apart *after* ΔT, same label → one survives,
  `seen_count` summed, both segment sets on the winner.

### 5c. `run_loop_test.sh`  (end-to-end)
1. launch phase-1 pipeline with `phase1.loop_closure.enabled: true`,
   `map`/`odom` split, LCD off, static bridge replaced by `loop_jump_injector`
   (`t_jump` set to land inside the bag's revisit window — L3: ~53 s in);
2. `ros2 bag play uHumans2_loop_L3 --clock --rate 0.1`;
3. record `/rsg/phase1/loop_closure_event`, the phase-1 track dump, and the
   fuser DSG.

**Pass criteria**
* exactly one `loop_closure_event`, `n_tracks` ≈ live track count, `Δt` matches
  the injected jump;
* track count does **not** step up after the revisit (no duplicate spawn);
  `internal_object_id` of re-observed objects is stable across the jump;
* re-observed objects' `rsg_presence` in the fuser refreshes to `OBSERVED`
  (confidence reset) — the original point of the local-segment design;
* `debug/loop_closure_experiment/` diff of the phase-1 track dump before/after
  shows the cache translated by ΔT, not rebuilt.

### 5d. Baseline
Same run with `loop_closure.enabled: false` → expect duplicate tracks / a track
count step at the revisit. This is the "before" for the thesis figure.

---

## 6. Files touched

| file | change |
|---|---|
| `src/rsg/nodes/support/phase1/persistent_object_tracker.py` | `reanchor_all`, `merge_reanchor_duplicates`, rigid helpers |
| `src/rsg/nodes/phase1.py` | TF listener, `_maybe_reanchor_on_loop_closure`, `/rsg/phase1/loop_closure_event` pub, call site at :1069 |
| `src/rsg/config/rsg_pipeline.yaml` | `phase1.loop_closure` block |
| `debug/loop_closure_experiment/loop_jump_injector.py` | new |
| `debug/loop_closure_experiment/test_reanchor.py` | new |
| `debug/loop_closure_experiment/run_loop_test.sh` | new |
| (later) fuser | consume `loop_closure_event` for a DSG annotation — optional |

No fuser change is strictly required for step 3/4: the fuser already rebuilds
its DSG clone from `/hydra/backend/dsg` each update and re-keys presence per
slot, so a re-anchored, non-duplicated slot set flows through as-is.

---

## 7. Order of work

1. ~~resolve §3c (camera-pose frame)~~ — done: incoming pose is raw `odom`.
2. ~~`reanchor_all` + rigid helpers + `test_reanchor.py` §5b~~ — done, 9/9 green.
3. ~~`loop_jump_injector.py`~~ — done.
4. ~~phase-1 TF listener + call site + config + event topic~~ — done.
5. ~~`merge_reanchor_duplicates` + its unit test~~ — done (drift + overlap passes).
6. `run_loop_test.sh`, baseline vs enabled on `uHumans2_loop_L3`. — TODO
7. large-drift check on `uHumans2_loop_L1`. — TODO
8. document results + restore `loop_closure.enabled: false` default. — TODO

### Legacy ordered list (kept for reference)

1. resolve §3c (camera-pose frame) — 15 min reading upstream node.
2. `reanchor_all` + rigid helpers + `test_reanchor.py` §5b (unit, no ROS). ✅ gate
3. `loop_jump_injector.py`.
4. phase-1 TF listener + call site + config + event topic.
5. `merge_reanchor_duplicates` + its unit test.
6. `run_loop_test.sh`, baseline vs enabled on `uHumans2_loop_L3`.
7. large-drift check on `uHumans2_loop_L1`.
8. document results here + a thesis subsection; restore config default
   (`loop_closure.enabled: false`).
