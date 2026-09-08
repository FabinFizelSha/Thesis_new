# Multi-Session Resume and Revisit — Implementation Reference (A to Z)

Status at time of writing: implementation complete and manually verified working
end to end (2026-09-08). No controlled experiment has been run yet — Section 6
defines the methodology for that; the results are a separate, later document.

## 0. The problem being solved

A mapping pipeline that starts from an empty map every time it launches cannot
model two situations that matter for a real deployment:

1. **A robot that loses power mid-deployment and is restarted.** It is still
   physically in the same place. A useful system should not forget everything
   it had already mapped and re-discover the entire environment from scratch.
2. **A robot that revisits an environment it has already mapped** — a daily
   patrol, a second inspection pass, a robot redeployed the next day. A useful
   system should recognise objects it has already seen and update its existing
   model of them, rather than silently accumulating a duplicate copy of the
   environment every time it returns.

Both require the same underlying capability: **persist what has been learned,
and correctly re-identify it against new observations.** This document covers
the implementation that makes that possible in this pipeline, built and
debugged across several sessions culminating 2026-09-08.

## 1. Architecture overview

Three independent processes each own a piece of the world model, and each
persists its own state independently — there is no shared database:

| Owner | What it holds | Persisted to |
|---|---|---|
| Phase 1 (`rsg_phase1_semantic_coordinator`, Python) | Per-object tracks: 3D geometry, observation history, semantic label, mobility class, Hydra slot assignment | `memory/tracker/phase1_tracker_state.json` |
| Hydra (`hydra_ros_node`, C++) | The scene graph (objects/places/rooms/agents) and its mesh, TSDF, and pose-graph state | `memory/hydra/backend/{dsg_with_mesh.json,mesh.ply,deformation_graph.dgrf,...}` |
| RAP (Chroma, external process) | Visual memory embeddings used for label retrieval | `memory/rap/chroma/` |

The fuser (`rsg_scene_graph_fuser`, C++) holds no persisted state of its own —
it is a pure function of whatever Hydra and Phase 1 are currently publishing,
rebuilt fresh every process start. It matters here anyway because several
resume-correctness fixes live there: the fuser is the last place a duplicate
or a stale-looking object could still slip through even when both upstream
saves are correct.

All of this lives under `memory/` at the repo root (see `memory/README.md`).
`clear_memory.py` (repo root) wipes all three stores in one step and refuses
to run while an owning process is still alive — see its own docstring for
usage; Section 7 covers a bug found and fixed in its process-liveness check.

## 2. Phase 1 persistence

### 2.1 What is saved

`PersistentObjectTracker._tracks` (a dict of `track_id -> PersistentObjectTrack`)
is the entire state. Each track carries:

- 3D bounding box, centroid, volume, observation count, first/last-seen timestamps
- every local Hydra segment belonging to the track, keyed by **slot id** — the
  field that ties a label to a specific Hydra object node
- semantic label, its confidence and source (VLM / RAP / prior), accumulated
  `label_evidence` / `label_observations`
- mobility classification (static / dynamic / unknown) and its confidence

Also saved: `_allocated_slot_ids` and the `_next_track_index` /
`_next_slot_index` counters, so a restored session cannot re-issue an id that
a restored track already owns. The spatial index used for fast candidate
lookup (`_spatial_bbox_cells`, `_spatial_centroid_cells`) is deliberately
**not** serialised — its cell size is config-dependent, and it is cheaply
rebuilt on load instead (`_refresh_spatial_index` per track,
`tracker_state_store.py:384`).

Implementation: `src/rsg/nodes/support/phase1/tracker_state_store.py`
(`SCHEMA_VERSION = 1`), functions `save_tracker_state` /
`load_tracker_state`. The file is written atomically (temp file + rename) so
a run killed mid-save cannot corrupt the store.

### 2.2 The timestamp shift — required, not cosmetic

This is the single most important, least obvious mechanic in the whole
feature. `persistent_object_tracker.py`'s mode-selection logic is:

```python
age_sec = max(0.0, timestamp_sec - track.last_seen_timestamp_sec)
recent_mode = age_sec <= persistent_continuation_max_age_sec   # 8 seconds
mode = "recent" if recent_mode else "revisit"
```

Bag replay always restarts the simulated clock at the bag's own recording
start time. That start time is **earlier** than whatever
`last_seen_timestamp_sec` a track was saved with on the previous run. Raw age
is therefore *negative*, and `max(0.0, ...)` clamps it to exactly `0.0` — which
selects **recent** mode, the opposite of what a resumed session needs (recent
mode is calibrated for frame-to-frame continuity with an in-memory,
uninterrupted evidence trail; revisit mode's looser thresholds are what a
reloaded, evidence-reset track actually needs).

The fix: every stored timestamp is shifted `time_shift_sec` (default 86400s,
one day) further into the past **every time state is saved**
(`cumulative_time_shift_sec` in the saved JSON tracks the running total across
repeated resumes). This makes `age_sec` land at roughly one day on the very
first post-resume observation of any restored track — far past the 8-second
recent-mode threshold — reliably forcing revisit-mode association regardless
of how much real or simulated time actually elapsed between the save and the
next launch. **This is why every resume, no matter how quickly you relaunch,
is treated by Phase 1 as a "day-later" re-encounter — see Section 5 for why
that is actually the correct, uniform choice rather than a quirk.**

Config: `phase1.persistent_tracking.session_persistence` in
`rsg_pipeline.yaml` — `enabled`, `state_path`, `time_shift_sec`.

### 2.3 What happens on load

- `load_tracker_state` restores every track, its segments, and the id
  counters, then rebuilds the spatial index.
- `_restored_label_pending`: every restored track that already carries a
  label. Restored tracks are **never** re-sent to RAP/VLM (they already have
  an answer) — but the fuser's label cache is per-process and starts empty,
  so the label has to be pushed to it at least once. `_drain_restored_semantic_labels`
  (phase1.py) does this proactively, a few tracks per frame, from the moment
  the process starts — **not** only when a track happens to be re-observed
  (an earlier version waited for re-observation, which left every
  not-yet-revisited object unlabelled for the whole run; fixed in `545c599`).
- `_restored_presence_pending`: tracked **per slot**, not per track. The
  fuser needs a `slot_id -> internal_object_id` mapping to know that several
  Hydra nodes belong to one physical object (otherwise its own duplicate
  handling and same-object dotted-edge rendering both break). This is
  published every frame for any not-yet-reobserved slot
  (`_restored_presence_segments`, tagged `restored_from_previous_session:
  true`) and retires **that one slot** the moment it is genuinely
  reobserved — retiring at track granularity instead (an earlier version) let
  one reobserved segment of a multi-segment object silently strand every
  other segment of the same object, since the ordinary per-frame heartbeat
  only ever republishes the one slot seen in the current frame. Fixed in
  `63296d0`.

## 3. Hydra persistence — what changed and why

Hydra already saved everything needed
(`hydra.launch.yaml` → `hydra_node.cpp`'s `hydra.save(output)` at shutdown
writes `backend/dsg_with_mesh.json`, `backend/mesh.ply`,
`backend/deformation_graph.dgrf`, plus trajectory and layer-statistics CSVs).
**It had no load path at all before this work** —
`BackendModule::loadState` existed with zero callers, and nothing ever called
`DynamicSceneGraph::load` on startup. Everything in this section is new.

### 3.1 New config

Three parameters, threaded through `hydra.launch.yaml` →
`rsg_hydra_from_phase1.launch.py` → `rsg_all.launch.py` as ordinary launch
args so they are visible and overridable at every level rather than buried in
a yaml file:

| Parameter | Default | Effect |
|---|---|---|
| `load_state_path` | `memory/hydra/backend/dsg_with_mesh.json` | Path to resume from. Sentinel `"none"` disables resume entirely (an empty string cannot be used — the launch frontend renders it as YAML null, which config-utilities cannot convert to a string, breaking every launch, not just resume). |
| `resume_reset_trajectory` | `true` | Whether restored agent/trajectory graph nodes are dropped (`true`) or kept (`false`) on load. See Section 5 for what this does and does not control. |
| `enable_object_merging` | `true` | Overrides `backend.enable_node_merging` from `hydra/config/datasets/uhumans2.yaml`, which sets it `false`. Without this, resumed objects can never reconcile with their previous-session counterpart — see Section 3.4. |

A missing save file is treated as a normal first run (logged, not an error),
so resume can be left permanently enabled without special-casing "is this the
first launch."

### 3.2 The resume sequence (`hydra_ros_pipeline.cpp::init()`)

Runs **before** the backend is constructed — `BackendModule`'s constructor
does `unmerged_graph_ = private_dsg_->graph->clone()`, so a graph injected
after construction would never reach the backend's own working copy. In
order:

1. **Load.** `DynamicSceneGraph::load(load_state_path)`, guarded by
   `std::filesystem::exists` (the call throws on a missing path rather than
   returning null) and wrapped in try/catch (a truncated save from a run
   killed mid-write must not stop the pipeline from starting).

2. **Trajectory: drop or keep.** If `resume_reset_trajectory` (default),
   every node whose `NodeSymbol` category matches the robot prefix is removed
   from the restored graph before it is installed anywhere. See Section 5 for
   the real-world meaning of this choice — it is narrower than the phrase
   "reset trajectory" suggests.

3. **Archive every restored node.** Every node's `is_active` is forced
   `false`. This is what makes re-identification possible at all: every
   Hydra merge-candidate strategy (`Pairwise`, `SemanticPairwise`,
   `SemanticNearestNode` — see 3.4) requires the *candidate* side of a merge
   to be archived. A node left as saved (`is_active` reflects whatever it was
   at shutdown, in practice `true` for everything still in the active window)
   is never eligible as a merge target, so a fresh detection of the same
   object could never merge into it, however identical it looked.

4. **Install into both backend-side graphs.** `backend_dsg_->graph = restored`
   and `shared_state_->backend_graph->graph = restored->clone()` — the
   frontend merges into the latter every spin, so seeding it means the very
   first merge extends history instead of silently resetting the backend's
   view of the world back to empty.

5. **Give the frontend its own copy of the mesh (not the graph).**
   `frontend_dsg_->graph->setMesh(restored->mesh()->clone())`. The frontend
   emits new objects' `mesh_connections` as indices into *its own* mesh, and
   the backend resolves those indices against *its own* mesh — if only the
   backend had the restored mesh, every newly detected object's geometry
   would silently resolve against the wrong (previous-session) vertex range.
   A clone, not a shared pointer: the two meshes evolve independently from
   here on. Frontend **nodes** are deliberately *not* seeded — only its mesh —
   for the reason in the next step.

6. **Seed frontend id counters, not frontend nodes.**
   `GraphBuilder::seedNodeIdCounters` (`graph_builder.cpp:202`) scans the
   restored graph's `node_lookup()`, finds the highest already-used index for
   every `NodeSymbol` prefix, and calls `setNextNodeIndex` on whichever
   segmenter owns that prefix (`MeshSegmenter` for objects, `Place2dSegmenter`
   for 2D places, `FrontierExtractor` for frontiers — a `setNextNodeIndex`
   method added to each specifically for this).

   **Why nodes are not also injected into the frontend graph:** every
   segmenter keeps its own per-process node cache
   (`MeshSegmenter::active_nodes_`, keyed by label) that starts empty on every
   process start, with no seeding mechanism of its own.
   `DynamicSceneGraph::emplaceNode` on an id that already exists is a
   **silent no-op returning `false`**
   (`spark_dsg/src/dynamic_scene_graph.cpp`), and none of the three
   segmenters check that return value — they record the id and increment
   their counter regardless. Seeding the frontend graph with restored nodes
   would therefore silently discard the geometry of every newly detected
   object in the resumed session, the moment its id happened to collide with
   a restored one. Seeding only the *counter* avoids the collision outright:
   new ids simply start above the highest restored one.

7. **Mesh offset alignment.** `kimera_pgmo::MeshOffsetInfo` is seeded with
   `archived_vertices = prev_archived_vertices = restored_mesh.numVertices()`
   before the mesh compressor starts (`graph_builder.cpp:162`). This matters
   because `updateVertices`/`remapVertexIndices` treat any index below
   `prev_archived_vertices` as already-valid and pass it through unchanged;
   without this seed, the very first live mesh delta this session would
   either misinterpret the restored (session-1) vertex range or throw.
   Verified arithmetically during implementation: with the seed in place, a
   node whose `mesh_connections` are entirely restored (session-1) indices
   passes through the remap untouched and is correctly reported as archived,
   never corrupted or deleted.

8. **Backend mesh re-install.** `BackendModule`'s constructor unconditionally
   installs a fresh, empty mesh, discarding anything injected in step 4 —
   `loadState(load_state_path, /*dgrf=*/"", /*force_loopclosures=*/false)`
   exists precisely to re-apply it after construction. `force_loopclosures =
   false` is load-bearing, not a default: it keeps `have_loopclosures_` false,
   which is what prevents `deformPoints` from later overwriting restored mesh
   vertices with deformed copies of the *new* session's geometry (see 3.3).
   The empty dgrf path skips loading the deformation graph entirely — see 3.3
   for why.

9. **Force one publish.** `backend->step(/*force_optimize=*/false)` — the
   backend's own spin loop does nothing while its input queue is empty, so
   without this the restored map would sit invisible in RViz until the bag's
   first frame propagates all the way through the frontend. Safe specifically
   because `have_loopclosures_` is false at this point, so `step()` takes the
   plain `updateDsgMesh` path rather than `optimize()`/`deformPoints`; it
   publishes once with timestamp 0, immediately superseded by the first real
   frame.

### 3.3 What is deliberately *not* restored

- **The deformation graph.** This deployment runs `enable_lcd = false` (GT
  odometry, no loop closure), so the entire pose-graph-optimisation path is
  already inert. Restoring the dgrf would risk `addPrior`'s unconditional
  double-weighting of the prior at `Symbol(prefix, 0)`, and restoring
  `trajectory_` specifically would make the ODOM-edge loop's
  `to_key.index() != initial_trajectory.size()` check silently drop every
  session-2 odometry edge. None of this matters while LCD stays off; it would
  need revisiting first if it were ever turned on.
- **The frontend's own graph nodes** (Section 3.2, step 6) — only its mesh
  and id counters are seeded.

### 3.4 Enabling backend object merging

This is the fix that actually stops duplicate object nodes on resume, and it
was the hardest of these to find: `hydra/config/datasets/uhumans2.yaml` sets
`backend.enable_node_merging: false`, which this launch loads for every run.
That flag gates `UpdateObjectsFunctor::findMerges` entirely
(`dsg_updater.cpp`) — confirmed directly against a live run's own glog
output: the functor's own throttled diagnostic
(`update_objects_functor.cpp`, added specifically to investigate this) never
printed once across a 30-plus-minute resumed run, meaning the merge logic was
never even entered.

With merging off, a freshly re-observed object can never reconcile with the
node restored from the previous session — every re-observation mints a
second, permanent, duplicate node. `enable_object_merging` (Section 3.1)
overrides the dataset default.

**How the merge actually works**, traced end to end during implementation:
the default strategy, `SemanticNearestNode`, restricts candidates to nodes
sharing the *same semantic label* — rebuilt fresh from the current graph on
**every single spin** (`MergeProposer::findMerges` constructs the associator
per call, not once), so it correctly sees archived nodes carried over from
resume as well as anything archived later in the same run. A found candidate
is then required to pass the same bounding-box-containment test the fuser's
own duplicate collapse uses (`lhs.bbox.contains(rhs.position) ||
rhs.bbox.contains(lhs.position)`) before a merge is proposed, and the check
is retried every spin for as long as an unmatched duplicate exists.

**Why this is safe specifically in this deployment:** `SemanticNearestNode`
merging by label is normally risky — in a generic closed-set semantic
labelling scheme, a label is a *class* shared by many physical instances
("chair"), so merging by label risks fusing two different chairs that happen
to sit close together. In this pipeline a label **is** Phase 1's
per-*physical-object* slot id (`rsg_slot_only_frozen_label_space.yaml`
reserves 10,000 distinct object-label ids) — two different real objects
structurally cannot share a label, so the classic failure mode this gate
likely exists to prevent upstream cannot occur here.

## 4. Fuser-side complementary work

Three fixes live downstream of Hydra, in the process that renders the final
scene graph. None of them are optional extras — each closes a gap the
upstream fixes alone left open.

### 4.1 Duplicate-node collapse (backstop)

Even with backend merging enabled, the fuser collapses any object nodes still
sharing a semantic slot, as a safety net for whatever the backend's per-spin
retry has not yet caught (`object_slot_collapse_enabled`,
`rsg_scene_graph_fuser.yaml`). The critical correctness rule, learned the
hard way (an earlier version got this backwards and deleted a restored node):
**an archived node is never removed, and always outranks an active one as the
surviving node.** Only active-window duplicates are ever folded together;
their box is unioned into the survivor. Verified against real saved data:
before this rule, a slot legitimately holding two archived nodes (a large
object Hydra had clustered in two pieces the previous run) silently lost one
of them on resume.

### 4.2 Restored-object visibility (presence-decay fix)

The fuser fades an object's opacity based on how long it has been unobserved
(`resolvePresenceForSlot`, exponential decay, 600s static / 120s dynamic
half-life, floored at `minimum_object_alpha` = 0.03). A restored-but-not-yet
-reobserved object's saved timestamp is, by design (Section 2.2), a day-plus
old — fed through that decay curve, its confidence underflows to roughly
`4.5×10⁻⁴⁴`, and the object renders at 3% opacity: functionally invisible in
RViz, even though the node and its edges are correctly published. This was
initially mistaken for a "Hydra hasn't published the node yet" bug; it is
purely a rendering-alpha problem, confirmed by the fact that edges pointing
at the "missing" node always rendered correctly (edges only draw when both
endpoints exist in the fuser's model).

Fix: `PresenceObservation` carries an `is_restored` flag (set from Phase 1's
`restored_from_previous_session` tag, Section 2.3).
`objectDisplayColor` renders full opacity specifically when that flag is set,
while `resolvePresenceForSlot` itself keeps reporting the true, un-overridden
confidence value everywhere else — kept as two separate concerns
deliberately, because the halo feature (4.3) needed the *honest* value.

### 4.3 In-view halo

A translucent cyan halo (`highlight_active_objects`) around any object
currently at presence confidence above `active_object_halo_min_confidence`
(default 0.99). This falls out of the existing decay curve almost for free —
confidence is exactly `1.0` while inside `presence_observed_epsilon_sec`
(observed this frame) and stays above 0.99 for a further ~8.7s (static
half-life) / ~1.7s (dynamic), so it reads as "in view right now" without a
dedicated age parameter. Because it uses the *honest* confidence value (4.2),
a restored-but-unconfirmed object — confidence ~10⁻⁴⁴ — correctly never
glows, even though it now also renders fully opaque. An earlier version keyed
this off Hydra's `is_active` flag instead, which lit up nearly every restored
object regardless of whether the robot was actually looking at it; keying off
presence confidence instead ties the halo to what Phase 1 has genuinely
observed this frame.

## 5. The two deployment scenarios

This is the part most relevant to a report's methodology section, and the
part where an earlier working assumption needed correcting during
implementation — recorded here precisely so it is not re-derived incorrectly
later.

**The robot's actual position at any moment is determined entirely by
whatever `/tf` is currently being published — never by any Hydra resume
setting.** `resume_reset_trajectory` (Section 3.1) only controls whether
Hydra's own agent/trajectory graph nodes are kept or dropped; it has no
connection to sensor input. This was confirmed by a real test: setting it
`false` (keep old trajectory nodes) produced exactly the predicted failure
mode — the previous run's trace loaded correctly, but the new run's trace
stopped extending, because Hydra's agent-layer node ids come from an
external `pose_graph_tools` publisher whose own numbering is never seeded
against the restored graph (unlike the object/place/frontier layers,
Section 3.2 step 6), so a new pose can silently fail to insert
(`DynamicSceneGraph::hasNode` → skip) if its id collides with a retained old
one. The default was reverted to `true` specifically to sidestep this bug —
dropping old nodes costs nothing, since position continuity was never tied to
this flag in the first place, and it also happens to give the cleaner
visual result (each run's own trace, not the previous one overlaid).

What actually distinguishes the two scenarios below is **whether the sensor
stream (bag, or a real robot's localisation) is paused-and-resumed versus
restarted from its own beginning** — entirely outside Hydra's control, and
the same for whatever `resume_reset_trajectory` is set to.

### Scenario A — power cycle and continue

**Real-world case:** a field robot loses power (battery swap, safety stop,
software crash, deliberate restart) while still physically in the
environment, and is expected to come back up mapping the same space
seamlessly, recognising what it had already found.

**How to reproduce with a bag:** pause the bag (space bar — `ros2 bag play`
has interactive keyboard controls on by default; killing and relaunching the
bag process is *not* equivalent, since a fresh invocation always restarts at
the recording's own beginning), restart only the Hydra/Phase 1/fuser
pipeline, then resume the same paused bag process. `/tf` continues seamlessly
across the pipeline restart because the bag process itself never stopped.

**What is exercised:** Hydra's full resume path (Section 3), continuing the
TSDF/mesh/object-layer state exactly where it left off. Phase 1 still applies
the day-plus timestamp shift on every save (Section 2.2) and so still
evaluates every re-observed object under **revisit**-mode thresholds, not
recent-mode ones — deliberately uniform regardless of how quickly the restart
actually happened. This is the correct choice, not a limitation: a restart of
any kind discards Phase 1's live, in-memory evidence trail (buffered
detections, in-flight VLM/RAP requests), so treating any reload — whether
five seconds or five days after the save — with the more conservative,
looser-but-bounded revisit thresholds is the right conservative default
either way.

### Scenario B — next-day revisit

**Real-world case:** a robot doing a repeated patrol of the same space — a
daily inspection round, a warehouse robot redeployed the next morning — that
should recognise most of what it saw on the previous pass while correctly
adding anything genuinely new.

**How to reproduce with a bag:** replay the *same* bag from its own beginning
(a fresh `ros2 bag play` invocation) after a previous run has already saved
state. The robot re-traverses a similar path and re-encounters the same
objects from broadly similar (not necessarily identical) viewpoints to the
first pass — the natural effect of driving the same recorded route twice.

**What is exercised:** the same underlying mechanism as Scenario A — nothing
in Phase 1 or Hydra's resume path distinguishes the two cases. The
difference in outcome, if any, comes only from how different the second
pass's viewpoints and partial views of each object are compared to the
first, which is exactly what Section 6's experiment is designed to measure.

## 6. Planned experiment — revisit accuracy (methodology, not yet run)

**Objective.** Measure how reliably the pipeline re-identifies
previously-tracked objects across repeated exposure to the same environment,
as opposed to minting new tracks/nodes for objects it has already mapped.

**Method.**
1. Select a fixed-duration segment of the bag (exact duration to be decided
   before running).
2. Run it once from a cleared memory state (`python3 clear_memory.py`) — this
   is run 1, the baseline map.
3. Run the identical segment again, unmodified, resuming from run 1's save —
   run 2.
4. Repeat for a third run, resuming from run 2's save.
5. After each run, record:
   - Phase 1's `track_count` (from the saved tracker state JSON)
   - the association-mode breakdown from that run's
     `tracking_associations_*.jsonl` (counts of `new_track` /
     `global_revisit_association` / `global_recent_association`)
   - Hydra's object-node count and duplicate-slot count
     (`debug/check_duplicate_objects.py` against the saved
     `memory/hydra/backend/dsg.json`)

**Ideal outcome.** Because runs 2 and 3 replay a scene the pipeline has
already fully mapped, a perfectly accurate revisit mechanism should produce
**zero `new_track` events** and an **unchanged `track_count`** from run 2
onward — every object encountered on the repeat pass should resolve to its
existing track, not mint a new one. Any track-count growth on a repeat run is
a direct, quantifiable measure of revisit-association failure.

**What a deviation would point back to.** This pipeline has an open,
deliberately-paused investigation into exactly this failure mode: Phase 1's
`global_centroid_pass_m` gate (currently 0.60) was tightened specifically to
fix a same-run over-merge case, and there is reason to believe it may be too
tight for the larger viewpoint-driven centroid drift a genuine cross-session
revisit produces on the first re-observation of an object (see the project
memory `tracking-quorum-gate-decision.md`). If this experiment shows
track-count growth concentrated across many distinct objects rather than a
few specific ones, that gate — not a Hydra-side issue — is the first place to
look; Hydra's own object-merge mechanism (Section 3.4) is independent of and
downstream of whatever track id Phase 1 assigns.

**Not yet decided (for tomorrow):** exact bag segment and duration, number of
repeated runs, and what track-count delta should count as a pass/fail
threshold versus measurement noise.

## 7. Known limitations and open items at time of writing

- **Agent-layer id collision** when `resume_reset_trajectory=false` (Section
  5) — documented and worked around via the default, not fixed at the root.
  Root fix would require seeding the agent-layer id space from the restored
  graph the same way the object/place/frontier layers already are, which in
  turn requires first confirming how the external `pose_graph_tools`
  publisher numbers its poses (not established during this work — the
  publisher lives outside this repository's own source).
- **The centroid-pass-gate cross-session investigation** referenced in
  Section 6 is deliberately paused pending the results of that experiment,
  per explicit direction to exhaust measurement before touching the gate
  again.
- **`clear_memory.py`'s Hydra process guard** originally checked for the
  substring `"hydra_node"`, which does not match the real binary name
  `hydra_ros_node` (the `_ros_` in the middle breaks the substring match) —
  found and fixed by testing directly against a live process
  (`fb69168`). Worth remembering if this script is ever copied or adapted.
- **`rsg_all.launch.py`'s `clear_hydra_cache` step** prints "Clearing all
  Hydra persistent state..." on every launch but only touches legacy paths
  (`~/.hydra/*`, `/tmp/hydra_*`, `~/.local/share/hydra*`) that predate the
  `memory/hydra/` resume path — it does not actually affect resume state, but
  the message is misleading. Left in place (out of scope of this work);
  worth cleaning up before this becomes report material someone else reads.

## 8. Change history

In implementation order. All commits on branch `pipeline-split-lean`.

| Commit | What it did |
|---|---|
| `71b99ba` | Initial Hydra multi-session resume: load the saved DSG+mesh, restart from it |
| `c1a069b` | Fixed resume never actually loading — needed a dedicated launch arg, `extra_yaml` was being overridden by a parent launch file |
| `287946e` | Resume on by default; a missing save file treated as a normal first run |
| `39612cf` | Added `resume_reset_trajectory`, gated by config, default true |
| `3852b4d` | Seeded backend mesh offsets from the restored mesh |
| `75fa47a` | Aligned frontend and backend mesh index spaces (superseded/corrected the previous commit's claim about what mesh-offset seeding actually fixes) |
| `b14bee9` | Archive every restored node, so re-observations become eligible merge candidates |
| `ceec5f2` | Republish restored segment identity so same-object edges render dotted, not solid |
| `09543bf` | Added the `findMerges` diagnostic used to prove backend merging was never firing |
| `057e45e` | First fuser-side duplicate-object collapse (later found to wrongly delete archived nodes) |
| `1ca38b5` | Added `clear_memory.py` |
| `63296d0` | Fixed restored-presence retirement to be per-slot, not per-track |
| `545c599` | Restored labels now drained proactively, not only on re-observation |
| `919e0ee` | Fixed the collapse to never remove an archived node; first halo attempt (keyed on `is_active`, later found too broad) |
| `e461313` | Halo re-keyed to actual in-view observation instead of `is_active` |
| `ac6940e` | Fixed restored objects rendering at 3% (functionally invisible) opacity |
| `4a3865b` | **The core fix**: enabled `backend.enable_node_merging`, previously silently `false` |
| `246dadb` | Halo re-implemented cleanly on presence confidence > 0.99, correctly excluding restored-unconfirmed objects |
| `fb69168` | Fixed `clear_memory.py`'s Hydra process-name guard |
| `d370456` | Briefly defaulted `resume_reset_trajectory` to false, to test trajectory continuity |
| `c25afab` | Reverted to true after the id-collision bug was confirmed on a real test; corrected the parameter's description |
| `7e49a54` | Fixed a launch-file YAML syntax crash introduced by the previous commit's description text |
