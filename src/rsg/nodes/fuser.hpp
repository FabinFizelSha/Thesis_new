#pragma once

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <geometry_msgs/msg/point.hpp>
#include <hydra_msgs/msg/dsg_update.hpp>
#include <nlohmann/json.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/edge_attributes.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph_types.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>

#include "fuser_types.hpp"

namespace rsg {

/**
 * Hydra-authoritative RAP metadata fuser.
 *
 * The node maintains a local Spark-DSG clone from /hydra/backend/dsg, attaches
 * RAP/VLM metadata to matching object attributes without changing Hydra node
 * IDs, geometry, slot IDs, or topology. Slot equality is the primary semantic
 * association rule: every Hydra object node carrying a resolved slot receives
 * that slot label. Centroids are used only when contradictory labels arrive
 * for one slot. The node publishes a mesh-free DsgUpdate copy,
 * and renders a typed layered projection as RViz MarkerArray messages. Missing
 * room-to-place membership is completed only in that derived display graph by
 * a nearest-visible-place majority vote with mesh wall validation. Derived
 * object-to-place display edges use the same conservative mesh policy. When a
 * mesh is unavailable, obstacle-aware fallbacks fail closed and no unvalidated
 * edge is drawn.
 */
class SemanticSceneGraphFuser : public rclcpp::Node {
 public:
  SemanticSceneGraphFuser();
  ~SemanticSceneGraphFuser() override = default;

 private:
  bool layerVisible(LayerKind kind) const;
  /**
   * Update the local Hydra DSG and request a later render pass.
   *
   * This callback intentionally does not build markers, snapshots, or a
   * layered projection. Those operations scale with map size and belong to
   * the capped renderer, not to the input path.
   */
  void handleDsg(const DsgUpdate::SharedPtr msg);
  void applyDsgDeletions(const DsgUpdate& msg);
  /**
   * Suppress one Hydra slot's object node from all future rendered output.
   *
   * Sent by phase1's persistent_dynamic_track_expiry: a confirmed-dynamic
   * object (person, animal, mobile robot) whose presence confidence decayed
   * past the configured threshold, so it has likely moved elsewhere. This
   * never touches graph_/Hydra's own DSG mirror -- that lifecycle is owned
   * by Hydra's backend (see applyDsgDeletions) and mixing the two risks
   * fighting each other. Instead the slot is filtered out of collectModel's
   * output, which every downstream consumer (markers, edges, exports)
   * already reads from -- a single choke point, not scattered per-consumer
   * checks.
   */
  void handleTrackDeleted(const Json& payload);
  /**
   * Cache one final Phase-1 result and return immediately.
   *
   * The callback performs no DSG traversal, marker creation, or RViz publish.
   * This prevents a growing fused graph from blocking the reliable semantic
   * result queue near the end of a rosbag replay.
   */
  void handleSemanticLabel(const std_msgs::msg::String::SharedPtr msg);
  void handleActiveSegments(const std_msgs::msg::String::SharedPtr msg);
  /**
   * Ingest one risk_result message (phase1's one-shot Risk VLM output for a
   * track) into risk_overlays_by_slot_, keyed by hydra_slot_id exactly like
   * a semantic label. Unlike handleSemanticLabel there is no multi-candidate
   * tie-breaking: risk is computed once per track and never re-evaluated, so
   * a later message for the same slot (which should not normally happen)
   * simply overwrites the earlier one.
   */
  void handleRiskResult(const std_msgs::msg::String::SharedPtr msg);
  /**
   * Render the latest graph at a bounded rate.
   *
   * Label updates are copied before the graph mutex is acquired. Therefore the
   * semantic callback can continue storing incoming slot labels while a long
   * marker/snapshot render traverses the Hydra graph.
   */
  void renderDirtyState();
  SceneModel collectModel(const OverlayCache& overlay_snapshot) const;
  /// True when @p point falls inside @p node's axis-aligned bounding box.
  static bool bboxContains(const NodeView& node, const Eigen::Vector3d& point);
  /**
   * Collapse Hydra object nodes that are the same physical object into one.
   *
   * In this pipeline a semantic slot is allocated per physical object by
   * Phase 1, so two object nodes carrying the same slot ARE the same object --
   * a guarantee generic Hydra does not have, since it treats semantic_label as
   * a class shared by many objects and therefore creates a node per mesh
   * cluster. That gap is what produces a duplicate node stacked on top of a
   * restored one when a session resumes: Phase 1 correctly re-identifies the
   * object and reuses its slot, but Hydra still emits a second node for it.
   *
   * The fuser is the right place to enforce it because the fuser's output is
   * the final scene graph, and the slot is the authoritative cross-pipeline
   * identity here (the same reasoning resolveOverlays already relies on).
   *
   * Only nodes in the robot's current view are ever removed. Everything
   * restored from a previous session is archived, and archived nodes are kept
   * exactly as loaded -- both as survivors and as geometry. Merging those
   * would delete parts of the map the robot is not currently looking at, and
   * would silently drop one of the two nodes in a slot that legitimately held
   * a large object Hydra clustered in two pieces last run.
   *
   * So an archived node always outranks an active one as survivor, and among
   * equals the lowest node id (the oldest) wins. A resumed session therefore
   * keeps the object it already had instead of replacing it with a freshly
   * created one, and keeps its saved extent rather than being reshaped by a
   * partial new view. Only when every node in a slot is active -- an ordinary
   * within-run duplicate, no restored node involved -- is the survivor's box
   * expanded to cover what it absorbed.
   *
   * @returns map of absorbed node id -> survivor node id.
   */
  std::unordered_map<NodeId, NodeId> collapseDuplicateObjects(SceneModel& model) const;
  void collectLayer(SceneModel& model, const std::string& layer_name, LayerKind kind) const;
  bool centroidUsable(const SemanticOverlay& overlay, const std::string& dsg_frame) const;
  std::unordered_map<NodeId, ResolvedOverlay> resolveOverlays(
      const SceneModel& model,
      const std::string& dsg_frame,
      const OverlayCache& overlays) const;
  /**
   * Remove the object-contact edges this node added to the DSG clone last
   * cycle. Called before collectModel() so the model only carries native
   * Hydra topology. Never touches an edge the fuser did not create.
   */
  void pruneOwnedContactEdges();
  /** Same as pruneOwnedContactEdges(), for the same-physical-object segment edges. */
  void pruneOwnedSegmentEdges();
    struct SegmentMember {
      NodeId id = 0;
      uint32_t semantic_slot = 0;
      Eigen::Vector3d position = Eigen::Vector3d::Zero();
      bool has_bbox = false;
      Eigen::Vector3d bbox_center = Eigen::Vector3d::Zero();
      Eigen::Vector3d bbox_size = Eigen::Vector3d::Zero();
    };

    struct MainObjectGroup {
      std::string internal_object_id;      //!< empty for a standalone (unsplit) object
      std::vector<SegmentMember> members;  //!< sorted by NodeId ascending
    };
  /**
   * Group every OBJECTS-layer node by PresenceObservation.internal_object_id
   * into "main objects": nodes sharing a non-empty internal_object_id are
   * local segments phase1 split off from one physical object too large to
   * track as a single segment, and are grouped together; a node with no
   * internal_object_id is its own singleton main object. Feeds both the
   * same-object segment chain and, aggregated into one bbox per group,
   * object-contact detection at the physical-object level.
   */
  std::vector<MainObjectGroup> buildMainObjectGroups(const SceneModel& model,
                                                      const PresenceCache& presence) const;
  /**
   * Connect main objects whose bounding boxes touch/overlap. A "main object"
   * is either a standalone node or a whole buildMainObjectGroups() group —
   * candidate pairing always happens main-object-to-main-object, never
   * between two segments of the same physical object (impossible: they're
   * merged into one candidate before pairing starts) and never against
   * non-touching segments of an otherwise-unrelated split object (e.g. a
   * single object next to a 3-way-split object where only one of the three
   * segments is actually close no longer produces three edges, one per
   * segment; it produces exactly one).
   *
   * Two phases per candidate main-object pair:
   *  - Broad phase (aabbContact() on the aggregates): each main object's
   *    aggregate bbox is the union of its members' bboxes. A uniform
   *    spatial-hash grid narrows main-object pairs to candidates whose
   *    aggregates could possibly touch (same grid/ring scheme as a flat
   *    per-node search would use — cell size = 2x the largest per-object
   *    reach this cycle, guaranteeing a single ring of neighbors is enough
   *    when auto-sized; object_contact_grid_cell_size_m can force a fixed
   *    cell size instead, with the per-object ring widened as needed). The
   *    aggregate always contains every member's geometry, so this can only
   *    ever over-approximate — it cannot miss a pair whose real segments
   *    touch.
   *  - Narrow phase (plain centroid distance, no intersection test): once
   *    the aggregates are confirmed touching, every real member connects to
   *    its single closest counterpart on the other side by nearest 3D
   *    centroid distance — run from both directions and deduped, so a
   *    mutually-closest pair yields one edge while a member whose neighbor's
   *    own closest match is someone else still gets its own edge. There is
   *    no per-pair distance cutoff and no bbox-overlap requirement here: the
   *    aggregate touch test already established the two main objects are in
   *    contact, so the narrow phase's only job is picking which real
   *    segments best represent that contact, not re-deciding whether it
   *    exists. AabbContact is still computed once per selected pair, purely
   *    to populate the edge's display/diagnostic metadata (gap, IoU, contact
   *    axis) — geometry no longer decides which pairs are selected.
   */
  void computeObjectContacts(const std::vector<MainObjectGroup>& groups,
                             LayeredProjection& projection);
  /**
   * Connect local segments phase1 split off from one physical object that
   * was too large to track as a single segment (buildMainObjectGroups). This
   * is an identity relation, not a geometric one: segments are chained in
   * ascending NodeId order — a proxy for formation order, since phase1
   * assigns each newly-split-off segment a higher id than what already
   * exists — rather than fully interconnected, so each new segment links to
   * the most recently existing one and a k-way split gets k-1 edges, not
   * k*(k-1)/2.
   */
  void computeObjectSegmentEdges(const std::vector<MainObjectGroup>& groups,
                                 LayeredProjection& projection);
  /**
   * Offline-verification dump: every main object's aggregate bbox, every
   * member's own bbox (plus its resolved semantic label, purely for human
   * identification when reading the log/CSV -- never used by the
   * verification logic itself, which only reasons about geometry), and the
   * contact/segment edges actually produced this cycle — one JSON line per
   * fusion cycle. Gated behind object_contact_diagnostics_enabled_ (default
   * off, no effect on production runs); the label lookup only happens on
   * this already-diagnostics-only path, so it costs nothing when disabled.
   * See debug/fuser_object_relation_experiment/analyze_contact_diagnostics.py
   * for the offline script that reads this log, independently recomputes
   * the expected edges from the raw geometry, and diffs against what's
   * logged here; debug/fuser_object_relation_experiment/IMPLEMENTATION.md
   * for the full design writeup.
   */
  void writeContactDiagnostics(const std::vector<MainObjectGroup>& groups,
                               const std::unordered_map<NodeId, ResolvedOverlay>& resolved);
  void updateLocalGraphMetadata(const SceneModel& model,
                                const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                                const RiskOverlayCache& risk_overlays,
                                const PresenceCache& presence);
  Eigen::Vector3d sourcePosition(const NodeView& node) const;
  Eigen::Vector3d displayPosition(const NodeView& node) const;
  GridKey gridKey(const Eigen::Vector3d& position, double cell_size_m) const;
  uint64_t hashCombine(uint64_t seed, uint64_t value) const;
  uint64_t placeFingerprint(const SceneModel& model, const std::vector<NodeId>& place_ids) const;
  void ensurePlaceSpatialIndex(const SceneModel& model, const std::vector<NodeId>& place_ids);
  uint64_t meshFingerprint(const spark_dsg::Mesh& mesh) const;
  void indexTriangle(size_t triangle_index, const MeshTriangle& triangle);
  bool ensureMeshWallIndex();
  bool segmentIntersectsTriangle(const Eigen::Vector3d& start,
                                 const Eigen::Vector3d& end,
                                 const MeshTriangle& triangle) const;
  bool meshSegmentIsClear(const Eigen::Vector3d& start, const Eigen::Vector3d& end) const;
  Eigen::Vector3d objectFreeSpaceAnchor(const NodeView& object,
                                        const NodeView& place) const;
  std::vector<std::pair<NodeId, double>> localPlaceCandidates(
      const SceneModel& model, const NodeView& object) const;
  bool cachedAssociationUsable(const SceneModel& model, const NodeView& object,
                               NodeId& place_id);
  std::optional<NodeId> findValidatedLocalPlace(const SceneModel& model,
                                                 const NodeView& object,
                                                 LayeredProjection& projection);
  static int64_t stampNanoseconds(const builtin_interfaces::msg::Time& stamp);
  void updateRoomPlaceCompletionGraceClock(const std_msgs::msg::Header& header);
  double roomPlaceCompletionGraceElapsedSec() const;
  bool roomPlaceCompletionReady(LayeredProjection& projection) const;
  std::optional<NodeId> majorityVisibleNeighbourRoom(
      const SceneModel& model,
      const NodeView& orphan_place,
      const std::unordered_map<NodeId, NodeId>& assigned_places,
      LayeredProjection& projection) const;
  void completeMissingPlaceRoomMembership(
      const SceneModel& model,
      const std::vector<NodeId>& place_ids,
      std::set<std::pair<NodeId, NodeId>>& room_place_pairs,
      LayeredProjection& projection) const;
  LayeredProjection buildLayeredProjection(const SceneModel& model);
  // Room names and colours are display-only. They deliberately do not modify
  // Hydra node labels or the authoritative DSG. Ordinals are allocated once per
  // node ID and never reused during a process lifetime, so Room1 keeps the same
  // label and colour across incremental updates.
  void syncRoomDisplayOrdinals(const SceneModel& model);
  uint32_t roomDisplayOrdinal(NodeId room_id) const;
  std::string roomDisplayLabel(NodeId room_id) const;
  Color roomDisplayColor(NodeId room_id, float alpha) const;
  static Color blackEdgeColor(float alpha = 0.90F);
  static Color objectContactEdgeColor(float alpha = 0.95F);
  static Color objectSegmentEdgeColor(float alpha = 0.95F);
  double currentReferenceTimeSec() const;
  /** Return the semantic overlay already resolved to one Hydra DSG node. */
  const SemanticOverlay* overlayForNode(
      const NodeView& node,
      const std::unordered_map<NodeId, ResolvedOverlay>& resolved) const;
  /** Select the configured static/unknown or dynamic presence half-life. */
  double presenceHalfLifeSec(const std::string& mobility_class) const;
  /** Resolve time-dependent presence confidence for one observed semantic slot. */
  ResolvedPresence resolvePresenceForSlot(
      uint32_t slot_id,
      const PresenceCache& presence,
      const std::string& mobility_class) const;
  /**
   * Build compact multiline RViz text for one Hydra object node. Default
   * format is exactly two lines, e.g.:
   *   floor(0.95)
   *   static(1.00)
   */
  std::string objectDisplayLabel(
      const NodeView& node,
      const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
      const PresenceCache& presence) const;
  /** Apply semantic colour and mobility-aware presence alpha to one node. */
  Color objectDisplayColor(
      const NodeView& node,
      const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
      const PresenceCache& presence) const;
  void addMarker(MarkerArray& markers, MarkerSet& next_keys, Marker marker) const;
  void publishMarkerArray(const SceneModel& model,
                          const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                          const PresenceCache& presence,
                          const LayeredProjection& projection);
  double objectSphereDiameter(const NodeView& node) const;
  void appendObjectMarkers(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                           const builtin_interfaces::msg::Time& stamp, const NodeView& node,
                           const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                           const PresenceCache& presence) const;
  Color placeDisplayColor(const NodeView& node, const LayeredProjection& projection) const;
  void appendLayerNodeMarker(MarkerArray& markers, MarkerSet& next_keys,
                             const std::string& frame,
                             const builtin_interfaces::msg::Time& stamp, const NodeView& node,
                             const std::string& node_namespace, const std::string& text_namespace,
                             const Color& color, double node_size, double text_height,
                             bool show_text, const std::string& text_label) const;
  void appendSemanticVolumeMarkers(MarkerArray& markers, MarkerSet& next_keys,
                                   const std::string& frame,
                                   const builtin_interfaces::msg::Time& stamp, const NodeView& node,
                                   const std::string& volume_namespace, const std::string& text_namespace,
                                   const Color& color, double text_height, bool show_text,
                                   const std::string& text_label) const;
  void appendOptionalLayerMarker(MarkerArray& markers, MarkerSet& next_keys,
                                 const std::string& frame,
                                 const builtin_interfaces::msg::Time& stamp,
                                 const NodeView& node) const;
  geometry_msgs::msg::Point markerPoint(const Eigen::Vector3d& position) const;
  void appendLineList(MarkerArray& markers, MarkerSet& next_keys,
                      const std::string& frame, const builtin_interfaces::msg::Time& stamp,
                      const std::string& marker_namespace, int32_t id, const Color& color,
                      const std::vector<geometry_msgs::msg::Point>& points) const;
  void appendPointList(MarkerArray& markers, MarkerSet& next_keys,
                       const std::string& frame, const builtin_interfaces::msg::Time& stamp,
                       const std::string& marker_namespace, int32_t id, const Color& color,
                       double point_size_m, const std::vector<geometry_msgs::msg::Point>& points) const;
  /** Evenly spaced points from start to end (inclusive), for a dotted-line marker. */
  static void appendDottedSegmentPoints(std::vector<geometry_msgs::msg::Point>& points,
                                        const geometry_msgs::msg::Point& start,
                                        const geometry_msgs::msg::Point& end, double spacing_m);
  /**
   * Contact edges are a symmetric relation, so they render as a shaft with an
   * arrowhead at each end rather than a plain undirected line: two overlapping
   * ARROW markers along the same segment, one per direction.
   *
   * `start`/`end` are the two objects' own center positions; source_radius_m/
   * target_radius_m are their sphere radii, so the drawn segment can be
   * pulled back to start at each object's own outer surface instead of its
   * center -- otherwise the arrow visibly disappears into the sphere at
   * both ends rather than appearing to originate from it.
   */
  void appendObjectContactArrows(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                                 const builtin_interfaces::msg::Time& stamp, NodeId source, NodeId target,
                                 const geometry_msgs::msg::Point& start,
                                 const geometry_msgs::msg::Point& end,
                                 double source_radius_m, double target_radius_m) const;
  /**
   * Small floating label at a contact edge's midpoint: the Euclidean
   * centroid distance (e.g. "1.45m").
   */
  void appendObjectContactLabel(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                                const builtin_interfaces::msg::Time& stamp, NodeId source, NodeId target,
                                const geometry_msgs::msg::Point& start,
                                const geometry_msgs::msg::Point& end,
                                double centroid_distance_m) const;
  void appendEdgeMarkers(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                         const builtin_interfaces::msg::Time& stamp, const SceneModel& model,
                         const LayeredProjection& projection) const;
  bool shouldPublishMarkers(bool force) const;
  void publishFusedDsg();
  /**
   * Build and publish one fused view from immutable label cache data.
   *
   * The graph lock intentionally covers the complete read/annotation pass so
   * Spark-DSG is never read while Hydra applies an incremental update. The
   * independent label mutex is not held here, so semantic ingestion remains
   * fast even when rendering a large map.
   */
  bool publishFusedOutputs(bool force_markers,
                           const std::string& reason,
                           const OverlayCache& overlay_snapshot,
                           const RiskOverlayCache& risk_overlay_snapshot,
                           const PresenceCache& presence_snapshot);
  size_t objectNodeCount(const SceneModel& model) const;
  /** Publish lightweight fuser diagnostics without traversing the live DSG. */
  void publishStatus(const std::string& state,
                     const std::string& detail,
                     const SceneModel* model = nullptr,
                     const std::unordered_map<NodeId, ResolvedOverlay>* resolved = nullptr,
                     const LayeredProjection* projection = nullptr,
                     const OverlayCache* overlay_snapshot = nullptr) const;

  mutable size_t collapse_log_countdown_ = 0;
  bool object_slot_collapse_enabled_ = true;
  double object_slot_collapse_max_distance_m_ = 2.0;
  bool highlight_active_objects_ = true;
  double active_object_halo_min_confidence_ = 0.995;
  double active_object_halo_scale_ = 1.7;
  double active_object_halo_alpha_ = 0.45;
  double active_object_halo_min_margin_m_ = 0.12;
  std::string input_dsg_topic_;
  std::string semantic_label_topic_;
  std::string active_segments_topic_;
  std::string risk_result_topic_;
  std::string fused_dsg_topic_;
  std::string markers_topic_;
  std::string status_topic_;
  std::string fallback_frame_id_;
  size_t semantic_label_qos_depth_ = 4096U;
  double semantic_refresh_rate_hz_ = 1.0;

  bool publish_fused_dsg_ = false;
  bool drop_local_mesh_ = true;
  double marker_publish_rate_hz_ = kDefaultMarkerRateHz;

  bool show_objects_ = true;
  bool show_rooms_ = true;
  bool show_buildings_ = false;
  bool show_places_ = true;
  bool show_segments_ = false;
  bool show_agents_ = false;
  bool show_edges_ = true;
  bool show_room_place_edges_ = true;
  bool show_place_object_edges_ = true;
  bool show_place_connectivity_edges_ = true;
  bool show_native_object_object_edges_ = false;
  bool show_native_room_room_edges_ = false;
  bool object_contact_edges_enabled_ = true;
  double object_contact_tolerance_m_ = 0.05;
  // No longer consulted by computeObjectContacts()'s narrow phase (nearest-
  // centroid matching has no IoU gate); kept declared for backward
  // compatibility with existing launch configs and reported in diagnostics.
  double object_contact_min_iou_3d_ = 0.0;
  bool object_contact_write_dsg_edges_ = true;
  size_t object_contact_max_objects_ = 500;
  double object_contact_grid_cell_size_m_ = 0.0;
  double object_contact_grid_min_cell_size_m_ = 0.05;
  bool show_object_contact_edges_ = true;
  bool show_object_contact_labels_ = true;
  double object_contact_text_height_m_ = 0.16;
  double object_contact_arrow_head_length_m_ = 0.20;
  double object_contact_arrow_head_diameter_m_ = 0.11;
  bool object_segment_edges_enabled_ = true;
  bool object_segment_write_dsg_edges_ = true;
  size_t object_segment_max_group_size_ = 50;
  bool show_object_segment_edges_ = true;
  double object_segment_dot_spacing_m_ = 0.08;
  double object_segment_dot_size_m_ = 0.03;
  bool object_contact_diagnostics_enabled_ = false;
  std::string object_contact_diagnostics_path_;
  bool show_object_labels_ = true;
  bool show_room_labels_ = true;
  bool show_place_labels_ = false;
  bool show_building_labels_ = true;
  bool show_slot_ids_ = false;
  bool show_presence_confidence_ = false;
  bool show_label_confidence_ = true;
  bool show_mobility_metadata_ = true;
  bool show_object_detail_ = true;
  double static_presence_half_life_sec_ = 600.0;
  double dynamic_presence_half_life_sec_ = 120.0;
  double presence_observed_epsilon_sec_ = 1.5;
  float minimum_object_alpha_ = 0.03F;
  bool dynamic_object_use_cube_ = true;
  bool presence_decay_continuous_refresh_ = true;
  double last_presence_refresh_reference_time_sec_ = -1.0;

  double object_z_offset_m_ = 0.0;
  double place_z_offset_m_ = 10.0;
  double room_z_offset_m_ = 20.0;
  double room_node_size_m_ = 1.10;
  double place_node_size_m_ = 0.45;
  double place_text_height_m_ = 0.18;
  float place_alpha_ = 0.90F;
  bool room_place_completion_enabled_ = true;
  double room_place_completion_grace_period_sec_ = 30.0;
  int room_place_completion_neighbours_ = 7;
  double room_place_completion_max_distance_m_ = 0.0;
  double room_place_completion_max_height_difference_m_ = 0.50;
  bool room_place_completion_require_mesh_validation_ = true;
  int room_place_completion_min_majority_votes_ = 1;
  bool object_place_use_local_validated_fallback_ = true;
  double object_place_max_distance_m_ = 3.0;
  double object_place_index_voxel_size_m_ = 2.0;
  int object_place_index_search_radius_cells_ = 2;
  int object_place_max_candidates_ = 6;
  bool object_place_require_mesh_validation_ = true;
  double object_place_mesh_voxel_size_m_ = 0.50;
  int object_place_mesh_max_triangle_tests_ = 2048;
  int object_place_mesh_max_cells_per_triangle_ = 256;
  double object_place_anchor_outset_m_ = 0.12;
  double object_place_cache_recompute_translation_m_ = 0.25;

  PlaceSpatialIndex place_spatial_index_;
  MeshWallIndex mesh_wall_index_;
  std::unordered_map<NodeId, CachedObjectPlaceAssociation> object_place_cache_;

  // Map-time warm-up clock for derived room-place completion. Source time is
  // preferred so rosbag replay at a different playback rate still waits for
  // the requested amount of mapping data, with wall time as a safe fallback.
  bool room_place_completion_grace_clock_started_ = false;
  int64_t room_place_completion_source_start_ns_ = 0;
  int64_t room_place_completion_last_source_stamp_ns_ = 0;
  std::chrono::steady_clock::time_point room_place_completion_grace_wall_start_ =
      std::chrono::steady_clock::now();

  // Persistent RViz-only room numbering and colours. Hydra room IDs stay untouched.
  std::unordered_map<NodeId, uint32_t> room_display_ordinals_;
  uint32_t next_room_display_ordinal_ = 1U;

  bool centroid_association_enabled_ = true;
  bool require_centroid_frame_match_ = true;
  bool allow_unframed_centroid_ = false;
  size_t max_label_candidates_per_slot_ = 8U;
  std::string unlabeled_object_display_label_ = "object_unknown";

  double object_min_size_m_ = 0.12;
  double object_max_size_m_ = 0.60;
  double object_sphere_volume_scale_ = 0.60;
  std::string object_sphere_size_mode_ = "volume_scaled";
  bool object_use_fixed_sphere_size_ = false;
  double object_fixed_sphere_size_m_ = 0.28;
  double object_text_height_m_ = 0.28;
  double object_label_vertical_offset_m_ = 0.10;
  double room_text_height_m_ = 0.35;
  double building_text_height_m_ = 0.42;
  float room_alpha_ = 0.14F;
  float building_alpha_ = 0.10F;
  double edge_width_m_ = 0.025;

  // Graph state is protected independently from semantic labels. Rendering
  // holds graph_mutex_ while reading Spark-DSG, but label callbacks only need
  // overlays_mutex_ and continue draining during expensive marker generation.
  mutable std::mutex graph_mutex_;
  mutable std::mutex overlays_mutex_;
  mutable std::mutex presence_mutex_;
  // Separate from overlays_mutex_ even though both guard "async result keyed
  // by slot_id" caches: risk results arrive from an entirely independent
  // phase 1 pipeline (a second VLM call), so there's no reason for risk
  // ingestion to ever contend with label ingestion for the same lock.
  mutable std::mutex risk_overlays_mutex_;
  mutable std::mutex render_mutex_;
  spark_dsg::DynamicSceneGraph::Ptr graph_;
  // One slot normally has one class. Multiple candidates are retained only
  // when contradictory semantic messages are received for that same slot.
  OverlayCache overlays_by_slot_;
  // One slot has at most one risk result, ever -- it's a one-shot
  // assessment (see RiskOverlay's own comment), so unlike overlays_by_slot_
  // there's no candidate list or tie-breaking here.
  RiskOverlayCache risk_overlays_by_slot_;
  PresenceCache presence_by_slot_;
  // Hydra slots phase1 has explicitly deleted (persistent_dynamic_track_
  // expiry -- a confirmed-dynamic object whose presence confidence decayed
  // past the configured threshold). Never cleared: like a merge-dropped
  // track's slot on the phase1 side, a deleted slot is retired permanently,
  // not recycled, so there is no "un-suppress on reuse" case to handle.
  // Protected by overlays_mutex_, same as overlays_by_slot_.
  std::unordered_set<uint32_t> deleted_slot_ids_;
  std_msgs::msg::Header latest_header_;
  int64_t latest_sequence_ = 0;

  // Object-contact edges written into the fused DSG last cycle, so they can be
  // pruned before the next model is collected (they are not native topology).
  // Guarded by graph_mutex_.
  std::set<std::pair<NodeId, NodeId>> contact_edges_prev_;
  // Full contact metadata for the current cycle, consumed by the status JSON.
  // Guarded by graph_mutex_.
  Json last_object_relations_ = Json::array();
  size_t last_object_contact_edge_count_ = 0;
  double last_object_contact_max_iou_ = 0.0;
  // Diagnostics for the spatial-grid pairing search (last computeObjectContacts call).
  double last_object_contact_grid_cell_size_m_ = 0.0;
  size_t last_object_contact_pairs_tested_ = 0;

  // Same-physical-object segment edges written into the fused DSG last
  // cycle, pruned the same way as contact_edges_prev_. Guarded by graph_mutex_.
  std::set<std::pair<NodeId, NodeId>> segment_edges_prev_;
  Json last_object_segment_relations_ = Json::array();
  size_t last_object_segment_edge_count_ = 0;

  // object_contact_diagnostics_enabled_ output stream, opened lazily on the
  // first write and kept open for the node's lifetime (append + flush per
  // cycle, so a killed/crashed node still leaves a usable partial log).
  std::ofstream object_contact_diagnostics_stream_;
  bool object_contact_diagnostics_stream_failed_ = false;
  uint64_t object_contact_diagnostics_sequence_ = 0;
  // Kept alive so the live-toggle callback registered in the constructor
  // (object_contact_diagnostics_enabled/_path only) stays registered;
  // letting this handle be destroyed silently de-registers the callback.
  OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  std::atomic<uint64_t> raw_dsg_updates_{0};
  std::atomic<uint64_t> full_dsg_updates_{0};
  std::atomic<uint64_t> incremental_dsg_updates_{0};
  std::atomic<uint64_t> skipped_initial_incremental_updates_{0};
  std::atomic<uint64_t> deleted_nodes_applied_{0};
  std::atomic<uint64_t> deleted_edges_applied_{0};
  std::atomic<uint64_t> malformed_deleted_edge_updates_{0};
  std::atomic<uint64_t> deserialize_failures_{0};
  std::atomic<uint64_t> accepted_label_messages_{0};
  std::atomic<uint64_t> accepted_semantic_result_events_{0};
  std::atomic<uint64_t> semantic_refresh_publications_{0};
  std::atomic<uint64_t> ignored_label_messages_{0};
  std::atomic<uint64_t> invalid_label_messages_{0};
  std::atomic<uint64_t> accepted_active_segment_messages_{0};
  std::atomic<uint64_t> invalid_active_segment_messages_{0};
  std::atomic<uint64_t> accepted_risk_messages_{0};
  std::atomic<uint64_t> invalid_risk_messages_{0};
  std::atomic<uint64_t> fused_dsg_publications_{0};
  std::atomic<uint64_t> marker_publications_{0};
  std::atomic<uint64_t> semantic_label_generation_{0};
  std::chrono::steady_clock::time_point last_marker_publish_ = std::chrono::steady_clock::now();
  bool first_marker_publication_ = true;
  std::atomic<bool> semantic_refresh_pending_{false};
  std::atomic<bool> render_dirty_{false};
  MarkerSet active_marker_keys_;

  rclcpp::CallbackGroup::SharedPtr graph_callback_group_;
  rclcpp::CallbackGroup::SharedPtr semantic_callback_group_;
  rclcpp::CallbackGroup::SharedPtr render_callback_group_;
  rclcpp::Subscription<DsgUpdate>::SharedPtr dsg_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr label_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr active_segments_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr risk_sub_;
  rclcpp::TimerBase::SharedPtr semantic_refresh_timer_;
  rclcpp::Publisher<DsgUpdate>::SharedPtr fused_dsg_pub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
};

}  // namespace rsg
