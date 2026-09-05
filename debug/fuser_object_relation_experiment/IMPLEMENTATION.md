# Fuser object-relation edges — implementation

Everything in this document lives in `src/rsg/nodes/fuser.cpp` (a single-file
C++ ROS 2 node). It covers two derived, display-and-DSG-write-back relation
types the fuser adds on top of Hydra's native scene graph:

- **object-contact edges** (amber double-headed arrows) — two physical
  objects whose 3D geometry touches or nearly touches.
- **object-segment edges** (blue dotted lines) — two nodes that are really
  local segments of *one* physical object, split apart by phase 1 because
  the object was too large to track as a single segment.

Both are computed fresh every fusion cycle from the current Hydra DSG
snapshot; neither is incremental or persisted between cycles beyond the
bookkeeping needed to prune last cycle's derived edges before writing new
ones.

---

## 1. Why "main objects" instead of raw DSG nodes

Phase 1's persistent-object tracker can split one large physical object
into several **local segments** — each a separate `OBJECTS`-layer DSG node
with its own bounding box, but sharing a common
`PresenceObservation.internal_object_id` that identifies them as pieces of
the same real thing.

The very first version of object-contact detection ran on raw nodes: every
`OBJECTS`-layer node with a bbox was a candidate, all-pairs (later
grid-narrowed). That produces the wrong picture as soon as either side of a
contact is split: a single object next to a 3-way-split object doesn't have
one relationship, it has (up to) three redundant edges — one per segment —
even though there is exactly one real physical relationship. Worse, once
*both* sides are split, the raw-node approach can produce a confusing tangle
that doesn't reflect the real geometry at all.

The fix, and the core idea of the current design: group nodes by
`internal_object_id` into **main objects** first, and always reason about
contact at the main-object level, only descending to individual segments to
find and anchor the real touch point(s).

```cpp
struct SegmentMember {
  NodeId id = 0;
  uint32_t semantic_slot = 0;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  bool has_bbox = false;
  Eigen::Vector3d bbox_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_size = Eigen::Vector3d::Zero();
};

struct MainObjectGroup {
  std::string internal_object_id;      // empty for a standalone (unsplit) object
  std::vector<SegmentMember> members;  // sorted by NodeId ascending
};
```

`buildMainObjectGroups(model, presence)` does the grouping: every
`OBJECTS`-layer node is keyed by its resolved `internal_object_id`; a node
with no `internal_object_id` gets a synthetic unique key
(`"__solo_" + node_id`) so it never accidentally merges with another
unrelated standalone object. The result is computed **once per fusion
cycle** in `publishFusedOutputs` and passed to both
`computeObjectContacts` and `computeObjectSegmentEdges` — no duplicated
grouping work between the two.

A standalone (never-split) object is simply a `MainObjectGroup` with one
member. This means the contact-detection code below never needs a special
case for "not actually split" — it's a group of size 1, which naturally
degenerates into the same behavior as the pre-refactor per-node comparison.

---

## 2. Object-contact edges — two levels

### Level 1 — broad phase, main-object aggregate bboxes

Each main object's **aggregate bbox** is the union of its members' bboxes
(members without a bbox are ignored; a group with no bboxed member at all
is skipped — it has no geometry to test):

```cpp
for (const auto& member : group.members) {
  if (!member.has_bbox) continue;
  const Eigen::Vector3d half = 0.5 * member.bbox_size.cwiseMax(0.0);
  agg_min = agg_min.cwiseMin(member.bbox_center - half);
  agg_max = agg_max.cwiseMax(member.bbox_center + half);
}
```

Candidate main-object *pairs* are narrowed with the same uniform
spatial-hash grid used for the original raw-node search (see §5 for the
grid mechanics) — just built over aggregate centers instead of individual
segment centers. For each grid-narrowed pair, the broad-phase test is a
plain `aabbContact(agg_a, agg_b, tolerance).touching`: true if the
aggregates overlap, or are separated by no more than
`object_contact_tolerance_m` (default 0.05 m) along their most-separated
axis.

Because the aggregate always **contains** every member's real geometry,
this test can only ever *over-approximate* — it can produce a false
positive (two aggregates are close but no real segment inside them
actually touches — see the L-shape / non-convex case below) but it can
never produce a false negative. A pair whose real segments truly touch is
guaranteed to have aggregates that are at least as close.

### Level 2 — narrow phase, nearest-centroid matching (reworked 2026-09-05, twice)

For every main-object pair that passes the broad phase, every node on the
side with **fewer real (bboxed) members** connects to its single closest
counterpart on the other side by plain 3D centroid distance — no
bbox-intersection test, no per-pair distance cutoff, and matching runs from
the smaller side only (a tie keeps `group_a`, already the side with the
smaller minimum NodeId by the broad phase's canonical ordering):

```cpp
const bool a_is_smaller = count_bboxed(group_a.members) <= count_bboxed(group_b.members);
const auto& smaller_members = a_is_smaller ? group_a.members : group_b.members;
const auto& larger_members  = a_is_smaller ? group_b.members : group_a.members;

for (const auto& small_member : smaller_members) {
  if (!small_member.has_bbox) continue;
  const SegmentMember* best = nullptr;
  double best_dist = std::numeric_limits<double>::max();
  for (const auto& large_member : larger_members) {
    if (!large_member.has_bbox) continue;
    const double dist = (small_member.bbox_center - large_member.bbox_center).norm();
    if (dist < best_dist) { best_dist = dist; best = &large_member; }
  }
  if (best != nullptr) selected.push_back(a_is_smaller ? ClosestPair{&small_member, best}
                                                        : ClosestPair{best, &small_member});
}
```

Then, for each *selected* pair only, `aabbContact()` is computed once —
purely to populate the edge's display/diagnostic metadata (gap, IoU,
contact axis). Geometry no longer decides *which* pairs are selected, only
how the selected pairs are described.

**Why smaller-side-only, not both directions:** the first version of this
rework matched from both sides and deduped the union — correct when the two
groups have comparable member counts, but wrong whenever one side has very
few nodes. A 1-node wall next to a 5-node floor: matching from the floor's
side alone already forces *every* floor node to pick the wall (it's the
only candidate on that side), so all 5 floor nodes got a wall edge
regardless of what the wall-side loop found — even the floor nodes at the
far end, nowhere near the wall. Running only from the smaller side (the
wall, here) fixes this structurally: exactly one edge, to whichever floor
node is genuinely closest, and the other four floor nodes stay free. Total
edge count is capped at the smaller side's own member count. A larger-side
node can still end up with more than one edge if it's independently the
closest match for more than one smaller-side node (e.g. two separate wall
segments both nearest the same floor tile) — that's a genuine multi-point
contact, not a repeat of the bug above.

**Why this replaced the old `k_a × k_b` intersection matrix**: the previous
design (every genuinely-overlapping member pair became its own edge, gated
by `contact_footprint_overlaps` and `object_contact_min_iou_3d`) worked, but
tied edge existence to bbox geometry lining up between two independently-
tracked objects — which it often doesn't (see the old worked example this
section used to carry: two parallel objects split at different points could
leave a member with two overlapping neighbors, or leave two main objects
whose aggregates clearly touch with *zero* real member-pair overlap,
producing no edge at all despite the objects visibly being in contact).
Nearest-centroid matching sidesteps this: once the broad phase has
established the two main objects are in contact, every real member is
*guaranteed* an edge to its closest counterpart, regardless of whether their
bboxes happen to intersect. There is deliberately no per-pair distance
cutoff — a member far from everything on the other side still gets
connected to its nearest neighbor, since the aggregate-level touch test is
what decided contact exists; the narrow phase's only job is picking which
real segments best represent that already-established contact.

`AabbContact::contact_footprint_overlaps` and `object_contact_min_iou_3d`
still exist (the struct field is still computed by `aabbContact()`, and the
param is still declared/reported) but are no longer consulted by this
matching step — see §7 for the full rationale and history of the
intersection-matrix design that preceded this one, including why
`contact_footprint_overlaps` was originally added.

### Anchoring and metadata

The two real member nodes become the edge's `source`/`target` directly —
no canonicalization/reordering (unlike segment edges, contact edges don't
need a stable chain order, and `spark_dsg::EdgeKey` already normalizes
`(a,b)`/`(b,a)` internally for graph storage — see §4). Each edge carries:

```jsonc
{
  "schema": "rsg_object_contact_v1",
  "relation": "bbox_contact",
  "centroid_distance_m": ...,
  "bbox_gap_m": ...,
  "bbox_overlap_volume_m3": ...,
  "bbox_overlap_xy_m2": ...,
  "bbox_iou_3d": ...,
  "bbox_iou_xz": ...,           // 2D IoU of the XZ-plane projections
  "bbox_iou_yz": ...,           // 2D IoU of the YZ-plane projections
  "bbox_iou_2d_max": ...,       // max(iou_xz, iou_yz) -- drives the RViz label
  "contact_axis": "x" | "y" | "z" | "none",
  "source_slot_id": ..., "target_slot_id": ...,
  "source_internal_object_id": ..., "target_internal_object_id": ...,
  "source_group_size": ..., "target_group_size": ...,
  "updated_timestamp_sec": ...
}
```

**Why 2D IoU, not 3D:** volumetric 3D IoU is a poor summary for most real
contacts — e.g. a small object against a large flat surface has tiny 3D
IoU even when solidly resting on it. `iou_xz`/`iou_yz` project the two
boxes onto each plane and compute ordinary 2D rectangle IoU. They're
padded by `tolerance_m` before the projection (capped at
`min(size_a, size_b)`, the true max possible 1D overlap on that axis) so
that a real surface contact — which by definition has *zero* literal
overlap on the separating axis, since the two objects don't interpenetrate
— doesn't get zeroed out on whichever plane includes that axis. `bbox_iou_3d`
is still computed and logged, and still drives the DSG edge's `weight`, but
is not used for the label (and, since the 2026-09-05 nearest-centroid
rework, no longer gates whether a pair gets an edge at all — see §2's
narrow-phase section).

---

## 3. Object-segment edges — the same-physical-object chain

Segments of one physical object are connected as an identity relation, not
a geometric one — grouped purely by `internal_object_id`
(`buildMainObjectGroups`, same grouping as above), with **no**
tolerance/grid/geometry involved at all.

**Formation-order chain, not all-pairs:** a `k`-way split gets `k-1` edges
(each segment linked to the *previous* one by NodeId), not `k*(k-1)/2`. The
ordering key is ascending NodeId — a proxy for creation order, since phase
1 assigns each newly-split-off segment a higher id than what already
exists at that point:

```cpp
// members is already sorted by NodeId ascending (buildMainObjectGroups)
for (size_t i = 0; i + 1 < members.size(); ++i) {
  connect(members[i], members[i + 1]);
}
```

So: `1a` forms alone (group size 1, no edge yet); when `1b` splits off,
`1a-1b`; when `1c` splits off, `1b-1c` (linking to the *most recently
existing* segment, not `1a`); and so on. This is explicitly the simpler of
two designs that were tried — an earlier version computed a Euclidean
minimum spanning tree over segment centroids instead (§7) — replaced by
this simpler NodeId-order chain per explicit instruction, since the
project owner judged that phase 1's segment-id assignment order already
captures the right topology without needing geometry at all.

Metadata:

```jsonc
{
  "schema": "rsg_object_segment_v1",
  "relation": "same_physical_object",
  "internal_object_id": ...,
  "source_slot_id": ..., "target_slot_id": ...,
  "group_size": ...,
  "updated_timestamp_sec": ...
}
```

Edge weight is a flat `1.0` (a deterministic identity relation, not a
fuzzy geometric measure).

**Structural invariant:** a contact edge and a segment edge can never
connect the *same* pair of nodes. Segment edges only ever connect two
members of the *same* `MainObjectGroup`; contact edges only ever connect
members of *two different* groups (the whole point of collapsing each
group to one aggregate before pairing even starts). The diagnostic tool
(§6) has an explicit check for this (`BOTH TYPES`) in case it's ever
violated in practice.

---

## 4. Rendering

- **Contact edges** — two overlapping `Marker::ARROW`s along the same
  segment (one per direction), so it reads as `<-------->`: arrowheads at
  both ends since it's a symmetric relation. Shaft diameter = `edge_width_m_`;
  head diameter = `1.5x` that; head length is a small fixed value
  (`3x edge_width_m_`), capped at `0.4x` the segment length so it never
  overshoots a very short segment (`appendObjectContactArrows`). A small
  floating text label at the segment's midpoint shows centroid distance and
  `max(iou_xz, iou_yz)` (`appendObjectContactLabel`).
- **Segment edges** — no native "dashed line" marker type exists in
  `visualization_msgs/Marker`, so it's built from evenly-spaced small dots
  (`appendDottedSegmentPoints`, spacing `object_segment_dot_spacing_m`,
  default 0.08 m) along the segment, all batched into one `Marker::POINTS`
  marker per cycle (`rsg_object_segment_edges` namespace) rather than one
  marker per edge.

Both are colored distinctly: amber (`0.91, 0.64, 0.24`) for contact,
blue (`0.20, 0.55, 0.90`) for segment — deliberately different from the
plain black used for every other native/derived edge type in the fuser, so
these two derived relation types read as visually distinct at a glance.

Stale-marker cleanup is automatic and generic: `addMarker` keys every
marker by `(namespace, id)`; `publishMarkerArray` diffs this cycle's key
set against last cycle's and issues `DELETE` for anything missing. Neither
edge type needs its own cleanup logic.

---

## 5. Performance

### Broad-phase grid

Main objects are bucketed into a uniform 3D spatial hash grid keyed by
`floor(aggregate_center / cell_size)`. Cell size auto-sizes each cycle to
`2 * (largest per-object "reach" seen)`, where `reach = 0.5 *
longest_aggregate_dimension + tolerance` — this guarantees a **single ring**
of 27 neighbor cells is always enough to find every possible touching
partner, proven by construction: two objects can only touch if their
centers are within `reach_a + reach_b <= 2 * global_max_reach = cell_size`,
so they can never be more than one cell apart.
`object_contact_grid_cell_size_m` can force a fixed cell size instead, in
which case the per-object search ring widens as needed
(`ceil((own_reach + global_max_reach) / cell_size)`) to keep the same
coverage guarantee even with an undersized fixed cell.

This grid is what makes level 1 sub-quadratic: measured on a 2000-object
synthetic scene (uniform density), 99.92% fewer pairs tested versus brute
force (1,999,000 → 1,599), ~14.6x faster wall time in a standalone
benchmark.

### Narrow-phase cost (nearest-centroid matching, 2026-09-05)

The current narrow phase still touches every `(member_a, member_b)` pair —
`O(k_a * k_b)` per candidate main-object pair — but each cell is now a
single `Eigen::Vector3d::norm()` (a centroid-distance computation) instead
of a full `aabbContact()` call, run twice over (once finding each
`member_a`'s closest `member_b`, once finding each `member_b`'s closest
`member_a`). `aabbContact()` itself is only invoked afterward, once per
*selected* pair (bounded by `k_a + k_b`, not `k_a * k_b`), purely to
populate that edge's display metadata — so the expensive geometry call runs
far less often than under the old intersection-matrix design, even though
the distance-comparison pass still visits every cell.

An earlier version of the narrow phase (the intersection-matrix design this
replaced) computed `aabbContact()` for the same `(member_a, member_b)` pair
up to **three times**: once evaluating from group A's perspective, once
from group B's (back when there was a relative-IoU ranking step requiring
both directions), and once more at edge-emission time. That version was
fixed to build the matrix exactly once and reuse each cached result
directly for emission — roughly 2-3x fewer geometry calls in the narrow
phase, verified by literally counting `aabbContact()` invocations on the
standard test scenarios (e.g. the misaligned 3-vs-3 case: 21 calls -> 9
calls). That whole matrix — cached or not — no longer exists; see §2 for
why it was replaced.

---

## 6. Diagnostic system

Off by default, zero cost on production runs
(`object_contact_diagnostics_enabled`, default `false`).

### What gets recorded

`writeContactDiagnostics(groups, resolved)` runs once per fusion cycle,
right after `computeObjectContacts`/`computeObjectSegmentEdges`, and
appends one JSON line to a log file: every main object's aggregate bbox,
every real member's own bbox (`node_id`, `semantic_slot_id`, `has_bbox`,
`bbox_center`, `bbox_size`, `label`), the params in effect
(`object_contact_tolerance_m`, `object_contact_min_iou_3d`), and the
contact/segment edges actually produced that cycle (mirroring
`last_object_relations_` / `last_object_segment_relations_`, already
computed — no extra geometry work to log them).

`label` is each member's resolved semantic label (e.g. `floor`, `chair`;
empty if phase 1/RAP hadn't labeled it yet that cycle), looked up from the
same `resolved` overlay map `publishFusedOutputs` already computes via
`resolveOverlays` — purely for human identification when reading the
log/CSV later, never consulted by the verification logic itself (which
only ever reasons about geometry). Each main object also gets a
`label` field of its own (the first non-empty label among its members),
so identifying which display id ("1a", "2a", ...) corresponds to which
real-world object doesn't require digging into individual members.

The file is opened once (lazily, on first write) and kept open, appending
+ flushing every cycle, so a killed/crashed node still leaves a usable
partial log.

Two params on this node are additionally live-toggleable via `ros2 param
set /rsg_scene_graph_fuser <name> <value>` without a restart —
`object_contact_diagnostics_enabled` and `object_contact_diagnostics_path`
— via a narrowly-scoped `add_on_set_parameters_callback` registered in the
constructor. No other param on this node supports this; every other
`declare_parameter` call is read once at startup and cached, so a live
`ros2 param set` on anything else silently has no effect.

### Where it's stored

If `object_contact_diagnostics_path` is left empty (the default), the
fuser creates a **new timestamped folder per node lifetime** under this
experiment's `results/` directory:

```
debug/fuser_object_relation_experiment/results/run_<YYYYMMDD_HHMMSS>/contact_diagnostics.jsonl
```

created via `std::filesystem::create_directories` on first write, so
successive debugging runs never clobber each other's logs. Set
`object_contact_diagnostics_path` explicitly to override this and write to
one fixed file instead.

### Independent verification, not a wrapper

`analyze_contact_diagnostics.py` is a **from-scratch Python
reimplementation** of `aabbContact`, `contact_footprint_overlaps`, the
main-object broad-phase + narrow-phase matrix, and the segment
formation-order chain — it never calls into the C++ or re-runs the fuser's
own edge-producing code path, only reads geometry. That's deliberate: a
tool that just re-executes the same algorithm agrees with itself by
construction and catches nothing. This one independently recomputes what
*should* have happened from raw bounding boxes and diffs against what was
actually logged, so a real implementation bug (wrong filter, wrong
tie-break, a stale edge, a duplicate) shows up as a mismatch.

It reports, per cycle:
- `CONTACT missing` / `CONTACT extra` — a real geometric contact the fuser
  didn't produce, or an edge the fuser produced that the geometry doesn't
  support.
- `SEGMENT missing` / `SEGMENT extra` — same, for the identity chain.
- `BOTH TYPES` — the structural invariant from §3 violated (a pair logged
  as both a contact and a segment edge).
- `STRUCTURAL` — a node appearing in two different main-object groups in
  the same cycle (a logging-level inconsistency, not an edge-logic bug).

Exit code `1` if anything was found, `0` if every cycle was clean.

**This is a verification report, not a data browser** — `--csv` mirrors it
exactly (one row per finding), so a clean run produces a header-only CSV.
To browse what the fuser actually logged regardless of correctness (e.g.
"what are the object ids of the floor objects"), `--export-csv` flattens
every `contact_edges`/`segment_edges` entry across all cycles into
`contact_edges.csv`/`segment_edges.csv` instead — no recomputation, no
comparison, just the raw data with readable display ids and resolved
labels attached. See the README for both flags' exact usage.

**Keeping it in sync:** there's no shared source of truth between the C++
and the Python oracle by design (see above), which also means: if the
C++ algorithm changes, `analyze_contact_diagnostics.py` needs the matching
hand update, or every cycle will "fail" against outdated rules. Whenever
`computeObjectContacts`/`computeObjectSegmentEdges`/`aabbContact` change in
`fuser.cpp`, the corresponding Python functions need the same change.

### Validated, not just written

The tool's own correctness was checked before trusting it: fed a fixture
reproducing the exact real "3-vs-3 offset boundary" bug from development
(§7) — caught all three symptoms (two missing edges, one wrong extra edge)
precisely; fed an injected "same pair as both a contact and a segment
edge" case — the `BOTH TYPES` check fired correctly; the Python oracle's
`expected_contact_edges` was cross-checked against every hand-verified
scenario from development and matched exactly in each case.

---

## 7. Design history — bugs found and fixes applied

Chronological, since later entries assume earlier context. Each was found
either by direct code review or by a concrete scenario reported during
development; each fix was verified with a standalone reproduction before
being folded into `fuser.cpp`.

1. **Raw all-pairs, no grouping (original design).** One object next to a
   3-way-split object produced three redundant contact edges instead of
   one. → introduced `buildMainObjectGroups` + the broad/narrow-phase
   split (§1-2).

2. **Segment-chain source/target vs. metadata mismatch.** An early
   segment-chain implementation canonicalized `source =
   min(id_i, id_{i+1})` / `target = max(...)`, but the metadata's
   `source_slot_id`/`target_slot_id` were still pulled directly from
   `members[i]`/`members[i+1]` — whenever the chain's *spatial* order
   didn't match ascending NodeId order, the slot IDs ended up silently
   swapped relative to the edge's actual source/target. Fixed by using
   chain order directly for source/target (no re-canonicalization needed —
   `spark_dsg::EdgeKey` already normalizes storage internally regardless
   of argument order, confirmed by reading `EdgeKey::EdgeKey` in
   `spark_dsg/src/scene_graph_types.cpp`).

3. **Segment chain: axis-sort, then MST, then simplified back down.**
   First attempt ordered a group's members by whichever axis had the
   greatest spread, then chained consecutively — breaks down for any
   non-1D layout (an L-shaped or 2D-slab split). Replaced with a Euclidean
   minimum spanning tree (Prim's) over segment centroids — correct for any
   layout, verified against a 2x2-grid adversarial case where axis-sort
   would pick an arbitrary/wrong pairing. Then, per explicit instruction,
   simplified back down to the current NodeId-ascending consecutive chain
   (§3) — phase 1's segment-id assignment order was judged to already
   capture the right topology, making the geometric MST unnecessary
   complexity for this codebase.

4. **Multi-point contact vs. single-closest-pair.** An intermediate
   version selected only the single closest real member-pair per
   main-object pair. That's wrong whenever two main objects touch along an
   *extended* shared boundary — e.g. object 1 split a-b-c-d, object 2
   split a-b, touching in the middle: `1b-2a` and `1c-2b` are both
   genuinely real contact points. Fixed by emitting one edge per
   independently-qualifying real pair instead of just the best one.

5. **Same-object neighbor-chain artifact.** Fixing (4) by "emit every
   touching pair" reintroduced spurious edges for a *contiguous* same-object
   split (§2, `contact_footprint_overlaps` worked example): adjacent
   segments share a face, and a face can register as "touching" something
   on the other object via a zero-width boundary graze alone. Fixed by
   `contact_footprint_overlaps` (§2). A follow-on floating-point issue in
   that same check — a literal zero-gap boundary touch can round to a hair
   below zero, which a strict `< 0` comparison misreads as real overlap —
   was fixed by requiring the overlap to exceed a small margin (`0.001 m`)
   instead of just being negative.

6. **Misaligned-boundary sliver vs. genuine match ("3-vs-3 offset
   boundary" bug).** Two main objects with segmentation boundaries that
   don't land at the same points can leave one segment geometrically
   qualifying (footprint-overlapping) against two neighbors on the other
   side, one a genuine match and one a marginal sliver — "emit every
   qualifying pair" then produces e.g. `1a-2a`, `1b-2a` (wrong, stealing
   `1b`'s line from the real match `2b`), `1c`-nothing (wrong, its real
   match `2c` lost out too), instead of the correct clean `1a-2a`,
   `1b-2b`, `1c-2c`. Root cause: `gap_m`, the natural "closeness" metric,
   *ties* across an entire row when two main objects run parallel (every
   pair shares the same separating-axis distance), so it can't discriminate
   a well-aligned neighbor from one that only grazes a corner — 2D IoU
   can. Fixed (temporarily — see next entry) with a relative-IoU
   threshold: each member keeps every candidate within some fraction
   (0.5 default) of its own best-found IoU, not just the single winner,
   evaluated from both sides and unioned. Verified this didn't reintroduce
   (4)'s bug (still produced `1b-2a`+`1c-2b` on the extended-boundary
   scene) nor over-collapse a genuine 50/50 straddle (both candidates tie
   at IoU 0.5, both pass the threshold, both kept).

7. **Relative-IoU threshold removed by explicit instruction.** After (6)
   was implemented and verified, the project owner decided the added
   ranking complexity wasn't wanted in this codebase, accepting the
   tradeoff that the misaligned-boundary symptom from (6) can reappear.
   Reverted to the simpler "every footprint-overlapping pair becomes an
   edge" rule from (5), with only `contact_footprint_overlaps` (not
   ranking) filtering candidates. This is the behavior as of this writing.

8. **Narrow-phase 2-3x redundant computation.** The two-direction
   evaluation added for (6) computed `aabbContact()` for the same
   `(member_a, member_b)` pair up to three times (§5). Fixed by caching
   the `k_a x k_b` contact matrix once and reusing it — this optimization
   survived (7)'s simplification since the matrix-caching structure is
   independent of whether a ranking step runs on top of it.

9. **Freeze-on-cap for phase 1 local segments (2026-09-04).** This one is
   in `persistent_object_tracker.py`, not the fuser, but it directly
   produced the object-contact symptoms this experiment exists to catch,
   so it's recorded here too. A local segment that touched
   `max_xy_span_m` kept absorbing further-drifted observations via the
   centroid-distance revisit fallback instead of handing off to a new
   segment, producing multi-metre overlaps between adjacent segments
   instead of a clean seam (see real-run examples that motivated this:
   `rsg_obj_000004`'s two segments overlapping by ~3m). Fixed by adding
   `PersistentObjectSegment.closed`: a not-yet-closed segment now freezes
   (`closed=True`, no further bbox expansion) the instant a touching
   observation would push it past the cap, instead of falling through to
   the ambiguous fallback. Verified via a standalone simulation (15m
   floor → 3 panels, no overlap; revisit from the far side still
   re-associates onto the right panel) and a real-code regression test,
   `src/rsg/tests/test_local_segment_freeze_on_cap.py`, exercising the
   actual `PersistentObjectTracker` (not a reimplementation).

   A related, separate bug found the same day: `_merge_track_pair`
   (used by `merge_reanchor_duplicates`, the loop-closure duplicate-track
   merge path) re-keyed *every* one of the dropped track's segments onto
   the survivor verbatim, even when one geometrically overlapped a
   segment the survivor already had — leaving two permanently-static,
   overlapping segments side by side forever (`iou_3d` up to 0.74 in a
   real run). Fixed by reconciling a touching/overlapping drop segment
   into the matching keep segment (bbox union, `closed` set if the union
   crosses the cap) instead of adding it as a second, redundant slot.
   Regression test:
   `test_overlap_pass_reconciles_duplicate_segments_instead_of_stacking_them`
   in `src/rsg/tests/test_reanchor.py`. Note this can only reconcile the
   *tracker's* forward bookkeeping — it can't retroactively erase mesh
   geometry Hydra already committed under the retired slot before the
   merge.

   **Both fixes are real and unit-tested, but neither actually bounds the
   geometry a real run produces — see "Still open" below.** Re-running the
   real pipeline three times after these fixes (each confirmed via
   `~/.ros/log/.../launch.log` to be a genuinely fresh process, ruling out
   a stale-node theory that looked plausible at first) still showed
   segments well past `max_xy_span_m` (9.4m, 7.8m against a 6.0m cap) with
   near-identical numbers across runs — a deterministic bag replaying the
   same views, hitting the same untouched code path every time.

10. **Narrow phase reworked from bbox-intersection matrix to
    nearest-centroid matching (2026-09-05).** Replaced entirely, not
    patched: the `k_a x k_b` intersection matrix (`aabbContact()` +
    `contact_footprint_overlaps` + `object_contact_min_iou_3d` filtering,
    items 4-8 above) is gone. The new narrow phase finds, for every real
    member on each side, its single closest counterpart on the other side
    by plain 3D centroid distance — no intersection test, no per-pair
    distance cutoff — run from both directions and deduped. Motivation: the
    old design tied edge existence to bbox geometry actually intersecting
    between two independently-tracked objects, which isn't guaranteed even
    when the objects are genuinely in contact (misaligned segmentation
    boundaries could leave a real touching pair with zero bbox overlap,
    producing no edge at all despite the broad phase correctly detecting
    aggregate contact). Nearest-centroid matching guarantees an edge exists
    once the broad phase says the main objects touch, regardless of whether
    any individual member pair's bbox happens to intersect.
    `contact_footprint_overlaps` and `object_contact_min_iou_3d` still exist
    (computed/declared) but are no longer consulted by this step — see §2
    for the full current algorithm and worked reasoning, and §5 for the
    updated complexity picture. `analyze_contact_diagnostics.py`'s
    `expected_contact_edges()` was updated in lockstep (per §6's
    keep-in-sync requirement) and verified against a hand-worked synthetic
    scenario before considering this done; not yet verified against a real
    pipeline run.

11. **Narrow phase matched from both sides, corrected to smaller-side-only,
    same day (2026-09-05).** Item 10's "both directions, deduped" version
    had a real bug the project owner caught by inspection: a 1-node wall
    next to a 5-node floor produced 5 wall-floor edges, one per floor node
    — because matching from the floor's side alone already forces every
    floor node to pick the wall (it's the only candidate over there),
    regardless of what the wall's own search finds. Fixed by matching from
    the side with fewer real members *only* (a tie keeps `group_a`); the
    `remember()`/dedup-by-canonical-key machinery item 10 needed for the
    two-direction union is gone entirely, since a single-direction search
    can't rediscover the same pair twice. Result: edge count is capped at
    the smaller side's own member count, and every non-matched larger-side
    node stays free — exactly the wall/floor case that motivated this.
    Verified with a hand-worked synthetic wall(1 node)/floor(5 node)
    scenario through the updated `analyze_contact_diagnostics.py` mirror
    (exactly one edge produced, to the correct nearest floor node); not yet
    verified against a real pipeline run. See §2 for the current algorithm.

### Still open

- **Local-segment span cap can't bound what actually gets published, when
  a single frame's own detection mask is already larger than the cap
  (2026-09-04).** Traced end-to-end, not just suspected:
  1. `tracking.py` (`TrackingStage.associate`, ~line 54-64) hands
     `_assign_local_segment` one whole per-frame NanoSAM detection mask's
     `bbox_3d_min/max` at a time — the full connected blob for e.g.
     "floor" in *that single frame*. A long, unobstructed sightline (a
     hallway floor) can put 7-9m of real-world extent in one mask, no
     multi-frame accumulation needed.
  2. Freeze-on-cap (item 9 above) correctly stops phase 1's own
     bookkeeping (`segment.bbox_3d_min/max`) from growing past the cap.
     But when a closed segment is re-matched (gap ≤
     `local_segment_gap_m`, same identity, no expand), phase 1 still
     routes the *entire current mask* to that segment's Hydra label — it
     only decided which label wins, not which pixels get published. There
     is no code path that crops/splits a mask to fit inside a segment's
     frozen bounds.
  3. Hydra's own frontend (`hydra/src/frontend/mesh_segmenter.cpp`,
     `nodesMatch`/`addNodeToGraph`/`updateNodeInGraph`, ~line 90-92 and
     299-315) integrates every pixel published under a label into that
     label's DSG node and recomputes its bounding box as the union of all
     mesh ever connected to it — with **no size cap of its own**, entirely
     independent of phase 1's `max_xy_span_m`.
  4. If the same oversized mask (or near-identical reprojections of it)
     recurs frame after frame — exactly what a slow bag replay of a
     static hallway view produces — Hydra's bbox for that label keeps
     absorbing it indefinitely. This is what the smooth 1.9m→9.4m growth
     curve in real runs actually is: not a phase-1 decision error, but
     Hydra revising the same label's geometry every frame from a mask
     phase 1 never shrank.

  **The real fix has to happen earlier than the tracker**: when a single
  observation's own footprint already exceeds `max_xy_span_m`, split it
  by 3D position into cap-sized pieces before it ever reaches
  `_assign_local_segment`, so no one observation — let alone the
  accumulated bookkeeping — can hand Hydra more than a cap-sized mask in
  one frame. That's mask/geometry-construction work in `tracking.py`
  (per-pixel 3D partitioning), not another tweak to the matching logic.
  Deliberately not implemented yet — flagged for follow-up.

- **Hydra's own frontend can split one phase-1 label into multiple DSG
  nodes on its own, independent of phase 1 entirely (confirmed
  2026-09-04).** Tested by setting `local_segments_enabled: false` (so
  phase 1 never assigns more than one Hydra slot per track) and re-running
  fresh (`run_20260904_234601`, confirmed via `~/.ros/log/.../launch.log`
  timestamps to be a genuinely new process on the new config). The dotted
  multi-member lines did **not** go away: `rsg_obj_000003`/`_000006` still
  had 2 members each, `rsg_obj_000011` had 4 — and critically, every
  member within each group shared the *identical* `slot_id`. That rules
  out phase 1 as the source for these: it's `hydra/src/frontend/
  mesh_segmenter.cpp` re-clustering all mesh vertices under one semantic
  label from scratch every frame, using `ClusteringConfig::cluster_tolerance`
  (`hydra/include/hydra/frontend/mesh_delta_clustering.h`, default 0.25m).
  If two regions carrying the *same* `hydra_label_id` end up more than
  25cm apart in the reconstructed mesh — a depth dropout, an occluding
  object, a patch never seen at a clean angle — Hydra treats them as
  separate object nodes regardless of the shared label. Hydra never
  consults the label to decide "same object," only mesh connectivity
  within it. `local_segments_enabled` reverted back to `true` afterward
  (it does help the separate long-object-span problem above; disabling it
  bought nothing for this one). Any real fix for *this* symptom lives in
  Hydra's clustering config/frontend, not `persistent_object_tracker.py`
  — not investigated further this session.

- **"Dotted and solid line between the same two nodes."** Reported once
  during development; never reproduced despite a structural argument that
  it should be impossible (§3's invariant) and a synthetic-scenario search.
  The diagnostic tool's `BOTH TYPES` check (§6) exists specifically to
  catch this for real if it recurs — if it ever fires, that's the place to
  start.
- **NodeId-as-formation-order assumption (§3).** The segment chain assumes
  NodeId increases monotonically with segment creation time within phase
  1. Not independently verified against phase 1's actual ID-assignment
  code; if that assumption doesn't hold in practice, the chain order could
  come out wrong. `PresenceObservation.local_segment_id`/
  `persistent_track_id` (opaque strings from phase 1) might encode a more
  reliable sequence, but their exact format wasn't confirmed before this
  writing.
- **Nearest-centroid narrow phase (items 10-11, §7) not yet verified against
  a real pipeline run.** Verified so far: compiles clean at each step, and
  `analyze_contact_diagnostics.py`'s updated `expected_contact_edges()`
  reproduces the exact expected pairs on hand-worked synthetic scenarios —
  a 2x2 case for item 10's "no distance cutoff" behavior, and a 1-node
  wall/5-node floor case for item 11's smaller-side-only fix (exactly one
  edge, to the correct nearest floor node, other four left free). Not yet
  checked against a fresh real-run diagnostic log with the rebuilt fuser,
  which would additionally exercise the spatial-hash broad phase and real
  multi-member groups together with the current narrow phase.
