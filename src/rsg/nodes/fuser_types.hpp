#pragma once

/**
 * @file fuser_types.hpp
 * @brief Shared value types and small pure helpers used across the fuser's
 * split implementation files (fuser_*.cpp) and declared in fuser.hpp.
 *
 * These types cross translation-unit boundaries (as class method
 * parameter/return types, and as helper functions called from multiple
 * fuser_*.cpp files), so they live in the ordinary `rsg` namespace, not an
 * anonymous one -- an anonymous namespace gives each TU a distinct,
 * incompatible copy of the same type, which is unsafe once the type is used
 * in a cross-TU function signature. Free functions are marked `inline` so
 * defining them identically in every including TU is legal.
 */

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <hydra_msgs/msg/dsg_update.hpp>
#include <nlohmann/json.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <spark_dsg/node_symbol.h>

namespace rsg {

using DsgUpdate = hydra_msgs::msg::DsgUpdate;
using Json = nlohmann::json;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;
using NodeId = spark_dsg::NodeId;
using MarkerKey = std::pair<std::string, int32_t>;
using MarkerSet = std::set<MarkerKey>;

constexpr double kDefaultMarkerRateHz = 1.0;
constexpr double kSmallExtentM = 0.05;
constexpr double kPi = 3.14159265358979323846;

struct Color {
  float r = 0.8F;
  float g = 0.8F;
  float b = 0.8F;
  float a = 1.0F;
};

enum class LayerKind {
  kObjects,
  kRooms,
  kBuildings,
  kPlaces,
  kSegments,
  kAgents,
  kOther,
};

struct NodeView {
  NodeId id = 0;
  LayerKind kind = LayerKind::kOther;
  bool visible = false;
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  bool has_bbox = false;
  Eigen::Vector3f bbox_center = Eigen::Vector3f::Zero();
  Eigen::Vector3f bbox_size = Eigen::Vector3f::Zero();
  uint32_t semantic_slot = 0;
  // Hydra's "in the active window" flag: true while the node is in the robot's
  // current view and still being updated, false once archived -- which is what
  // every node restored from a previous session is marked as.
  bool is_active = false;
  std::string name;
};

struct RawEdge {
  NodeId source = 0;
  NodeId target = 0;
};

enum class DisplayEdgeType {
  kRoomPlaceHierarchy,
  kPlaceObjectMembership,
  kPlaceConnectivity,
  kNativeObjectObjectDebug,
  kNativeRoomRoomDebug,
  // Derived: two object bounding boxes touch/overlap. Written back into the
  // fused DSG as a real object-object edge carrying contact metadata.
  kDerivedObjectContact,
  // Derived: two object nodes are local segments phase1 split off from one
  // physical object that was too large to track as a single segment (same
  // PresenceObservation internal_object_id). An identity relation, not a
  // geometric one. Written back into the fused DSG; rendered as a dotted line.
  kDerivedObjectSegment,
};

enum class EdgeOrigin {
  kNativeHydraEdge,
  // Derived only after an indexed local 3D lookup and a mesh wall-intersection
  // test. The fuser never writes this display relation back into Hydra's DSG.
  kDerivedMeshValidatedLocalPlace,
  // Derived display-only room membership for a place with no native Hydra room
  // edge. The room is selected by a majority vote from its nearest visible
  // room-owned place neighbours.
  kDerivedVisibleNeighbourRoom,
};

struct DisplayEdge {
  NodeId source = 0;
  NodeId target = 0;
  DisplayEdgeType type = DisplayEdgeType::kPlaceConnectivity;
  EdgeOrigin origin = EdgeOrigin::kNativeHydraEdge;
  // Parent room used only for display color. The authoritative Hydra graph is
  // never modified and this does not create a new DSG edge.
  NodeId color_owner = 0;
  // Populated only for kDerivedObjectContact; drives the optional contact-label marker.
  double contact_iou_2d_max = 0.0;  //!< max(iou_xz, iou_yz)
  double contact_centroid_distance_m = 0.0;
};

struct LayeredProjection {
  // One deterministic room color is assigned to each room. A place inherits
  // the color of its direct native or derived display-only room parent.
  std::unordered_map<NodeId, NodeId> place_to_room;
  std::unordered_map<NodeId, EdgeOrigin> place_room_origin;
  std::unordered_map<NodeId, NodeId> object_to_place;
  std::unordered_map<NodeId, EdgeOrigin> object_place_origin;
  std::vector<DisplayEdge> edges;

  // Diagnostics for local mesh-validated association. A missing mesh makes
  // obstacle-aware fallbacks fail closed rather than crossing a wall.
  bool mesh_validation_available = false;
  size_t local_index_candidates_examined = 0;
  size_t mesh_rejected_candidates = 0;
  size_t validated_local_place_associations = 0;
  size_t local_association_cache_hits = 0;
  size_t fallback_suppressed_without_mesh = 0;
  size_t room_completion_candidates_examined = 0;
  size_t room_completion_mesh_rejected_candidates = 0;
  size_t derived_room_place_associations = 0;
  size_t room_completion_ties = 0;
  size_t room_completion_suppressed_without_mesh = 0;
  // During early mapping, room centres, place connectivity, and the mesh are
  // still evolving. Keep native Hydra room-place edges visible, but postpone
  // all derived neighbour-majority links until this warm-up has elapsed.
  bool room_completion_waiting_for_grace = false;
  double room_completion_grace_elapsed_sec = 0.0;
  double room_completion_grace_remaining_sec = 0.0;
  size_t room_completion_suppressed_by_grace = 0;
};

struct GridKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const GridKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct GridKeyHash {
  size_t operator()(const GridKey& key) const {
    const uint64_t x = static_cast<uint32_t>(key.x);
    const uint64_t y = static_cast<uint32_t>(key.y);
    const uint64_t z = static_cast<uint32_t>(key.z);
    uint64_t seed = x * 0x9e3779b185ebca87ULL;
    seed ^= y + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    seed ^= z + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return static_cast<size_t>(seed);
  }
};

struct PlaceSpatialIndex {
  double cell_size_m = 2.0;
  uint64_t fingerprint = 0;
  uint64_t revision = 0;
  std::unordered_map<GridKey, std::vector<NodeId>, GridKeyHash> cells;
};

struct MeshTriangle {
  Eigen::Vector3d first = Eigen::Vector3d::Zero();
  Eigen::Vector3d second = Eigen::Vector3d::Zero();
  Eigen::Vector3d third = Eigen::Vector3d::Zero();
  Eigen::Vector3d min_corner = Eigen::Vector3d::Zero();
  Eigen::Vector3d max_corner = Eigen::Vector3d::Zero();
};

struct MeshWallIndex {
  double cell_size_m = 0.5;
  uint64_t fingerprint = 0;
  uint64_t revision = 0;
  bool available = false;
  std::vector<MeshTriangle> triangles;
  std::vector<size_t> overflow_triangles;
  std::unordered_map<GridKey, std::vector<size_t>, GridKeyHash> cells;
};

struct CachedObjectPlaceAssociation {
  NodeId place_id = 0;
  Eigen::Vector3d object_position = Eigen::Vector3d::Zero();
  uint64_t place_index_revision = 0;
  uint64_t mesh_index_revision = 0;
};

struct SemanticOverlay {
  uint32_t slot_id = 0;
  std::string label;
  double confidence = 0.0;
  std::string mobility_class = "unknown";
  double mobility_confidence = 0.0;
  std::string mobility_source = "none";
  std::string object_detail;
  std::string source;
  double timestamp_sec = 0.0;
  bool has_centroid = false;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  std::string centroid_frame_id;
  // Only used to inject a synthetic object NodeView (see collectModel) for a
  // dynamic object phase1 confirmed but no Hydra mesh-cluster node exists for
  // yet -- a moving object's mesh-cluster vertex count can stay under
  // MeshSegmenter's min_cluster_size in every single update pass, so Hydra
  // may never form a node for it at all, independent of how correctly/
  // confidently phase1 classified it. Sourced from last_bbox_3d_min/max (one
  // observation, never accumulated), not bbox_3d_min/max (the track's
  // ever-growing accumulated envelope) -- for a moving object the
  // accumulated one spans its whole travelled path, not its current extent.
  bool has_bbox = false;
  Eigen::Vector3d bbox_min = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_max = Eigen::Vector3d::Zero();
};

struct ResolvedOverlay {
  SemanticOverlay overlay;
  std::string association = "slot_id";
  double centroid_distance_m = -1.0;
};

/// A one-shot hazard assessment for one Hydra slot (see phase1's Risk VLM
/// dispatch, phase1.py's _publish_risk_result). Deliberately simpler than
/// SemanticOverlay: risk is computed exactly once per track and never
/// re-evaluated, so there's no centroid-fallback matching or multi-candidate
/// tie-breaking to do -- a slot either has a risk result or it doesn't, and
/// the lookup is a plain slot_id match, same key phase 1 already publishes.
struct RiskOverlay {
  uint32_t slot_id = 0;
  double risk_score = 0.0;  //!< Signed [-1.0, 1.0]: negative = risk-reducing (safety equipment/signage), 0 = neutral, positive = hazard.
  std::vector<std::string> risk_factors;
  std::string source;
  double timestamp_sec = 0.0;
};

/// Same "ingest under a short-lived mutex, copy for the render pass" shape
/// as OverlayCache below -- see that type's own comment for why.
using RiskOverlayCache = std::unordered_map<uint32_t, RiskOverlay>;


struct PresenceObservation {
  uint32_t slot_id = 0;
  std::string internal_object_id;
  std::string persistent_track_id;
  std::string local_segment_id;
  double last_observed_timestamp_sec = 0.0;
  bool has_centroid = false;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  bool has_bbox = false;
  Eigen::Vector3d bbox_min = Eigen::Vector3d::Zero();
  Eigen::Vector3d bbox_max = Eigen::Vector3d::Zero();
  double local_segment_xy_span_m = 0.0;
  // Set when this observation came from phase1's restored-slot republish
  // (_restored_presence_segments), not the ordinary per-frame heartbeat --
  // i.e. the object is known from a previous session but not yet re-observed
  // this one. See resolvePresenceForSlot for why this needs special handling.
  bool is_restored = false;
  Json raw = Json::object();
};

struct ResolvedPresence {
  PresenceObservation observation;
  std::string state = "UNK";
  double confidence = 0.0;
  double age_sec = 0.0;
};

using PresenceCache = std::unordered_map<uint32_t, PresenceObservation>;

/// Immutable copy of the slot-label cache used by one render pass.
///
/// The semantic callback updates the live cache under a short-lived mutex.
/// The renderer copies that map, releases the semantic lock, and performs all
/// expensive DSG/RViz work against the copy. This keeps label ingestion O(1).
using OverlayCache = std::unordered_map<uint32_t, std::vector<SemanticOverlay>>;

struct SceneModel {
  std::unordered_map<NodeId, NodeView> nodes;
  std::vector<RawEdge> raw_edges;
  std::unordered_map<NodeId, std::vector<NodeId>> adjacency;
};

inline std::string idString(NodeId id) {
  return std::to_string(static_cast<uint64_t>(id));
}

inline bool isObject(const NodeView& node) {
  return node.kind == LayerKind::kObjects;
}

inline bool isRoom(const NodeView& node) {
  return node.kind == LayerKind::kRooms;
}

inline bool isPlace(const NodeView& node) {
  return node.kind == LayerKind::kPlaces;
}

inline bool isBuilding(const NodeView& node) {
  return node.kind == LayerKind::kBuildings;
}

inline std::string normaliseLabel(const std::string& raw) {
  std::string result;
  result.reserve(raw.size());
  bool previous_space = true;
  for (const unsigned char character : raw) {
    const char value = character == '_' ? ' ' : static_cast<char>(std::tolower(character));
    if (std::isspace(static_cast<unsigned char>(value))) {
      if (!previous_space) {
        result.push_back(' ');
      }
      previous_space = true;
    } else {
      result.push_back(value);
      previous_space = false;
    }
  }
  if (!result.empty() && result.back() == ' ') {
    result.pop_back();
  }
  return result;
}

/** Convert mobility aliases to the three-class fuser vocabulary. */
inline std::string normaliseMobilityClass(const std::string& raw) {
  const std::string value = normaliseLabel(raw);
  if (value == "dynamic" || value == "mobile") {
    return "dynamic";
  }
  if (value == "static" || value == "stationary" || value == "fixed") {
    return "static";
  }
  return "unknown";
}

inline bool usableLabel(const std::string& label) {
  const auto normalized = normaliseLabel(label);
  return !normalized.empty() && normalized != "unknown" && normalized != "unknown object" &&
         normalized != "unclassified" && normalized != "unclassified object";
}

inline bool parseVector3(const Json& input, Eigen::Vector3d& output) {
  if (!input.is_array() || input.size() != 3) {
    return false;
  }
  try {
    const double x = input.at(0).get<double>();
    const double y = input.at(1).get<double>();
    const double z = input.at(2).get<double>();
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      return false;
    }
    output = Eigen::Vector3d(x, y, z);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

inline uint64_t fnv1a(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char character : value) {
    hash ^= static_cast<uint64_t>(character);
    hash *= 1099511628211ULL;
  }
  return hash;
}

/**
 * @brief Axis-aligned bounding-box contact test between two centered boxes.
 *
 * Boxes are (center, full-size). Rotation is ignored (see fuser NodeView, which
 * only retains center + dimensions). Per axis i:
 *   gap_i = |cA_i - cB_i| - (sA_i + sB_i)/2
 * gap_i > 0 => separated on that axis; gap_i <= 0 => overlapping by -gap_i.
 */
struct AabbContact {
  bool touching = false;
  double gap_m = 0.0;               //!< max_i gap_i; <= 0 means the boxes overlap
  double overlap_volume_m3 = 0.0;
  double overlap_xy_m2 = 0.0;
  double iou_3d = 0.0;
  double iou_xz = 0.0;               //!< 2D IoU of the two boxes' projections onto the XZ plane
  double iou_yz = 0.0;               //!< 2D IoU of the two boxes' projections onto the YZ plane
  double centroid_distance_m = 0.0;
  int contact_axis = -1;            //!< argmax_i gap_i (near-separating axis); -1 if degenerate
  // Do the boxes genuinely overlap (strictly negative gap, not just a
  // zero-width graze) on BOTH axes other than contact_axis? touching alone
  // allows a same-object neighbor-chain artifact through: two boxes that
  // share a face with a third box (e.g. adjacent segments of one elongated
  // object) can register "touching" against something merely because that
  // shared face happens to sit within tolerance, even with zero real
  // footprint overlap. This flags that case so callers can require an
  // actual physical contact area, not just an incidental boundary touch.
  bool contact_footprint_overlaps = false;
};

/** 2D intersection-over-union of two axis-aligned rectangles, given their per-axis overlap lengths. */
inline double planarIou(double overlap_u, double overlap_v, double size_a_u, double size_a_v,
                        double size_b_u, double size_b_v) {
  const double overlap_area = overlap_u * overlap_v;
  const double area_a = std::max(0.0, size_a_u) * std::max(0.0, size_a_v);
  const double area_b = std::max(0.0, size_b_u) * std::max(0.0, size_b_v);
  const double denom = area_a + area_b - overlap_area;
  return denom > 1.0e-9 ? overlap_area / denom : 0.0;
}

inline AabbContact aabbContact(const Eigen::Vector3d& center_a,
                               const Eigen::Vector3d& size_a,
                               const Eigen::Vector3d& center_b,
                               const Eigen::Vector3d& size_b,
                               double tolerance_m) {
  AabbContact out;
  out.centroid_distance_m = (center_a - center_b).norm();

  Eigen::Vector3d gap;
  Eigen::Vector3d overlap;
  for (int i = 0; i < 3; ++i) {
    const double half_sum = 0.5 * (std::max(0.0, size_a[i]) + std::max(0.0, size_b[i]));
    const double sep = std::abs(center_a[i] - center_b[i]);
    gap[i] = sep - half_sum;
    overlap[i] = std::max(0.0, -gap[i]);
  }

  Eigen::Index axis_index = 0;
  out.gap_m = gap.maxCoeff(&axis_index);
  out.contact_axis = static_cast<int>(axis_index);
  out.touching = out.gap_m <= tolerance_m;
  // Require overlap to exceed a small margin, not just be negative: a
  // literal boundary touch (two adjacent same-object segments sharing a
  // face) should land at gap == 0, but floating-point rounding on the
  // (center, size) arithmetic can just as easily push it a hair below zero,
  // which a strict "< 0" check would misread as genuine overlap.
  constexpr double kFootprintOverlapMarginM = 0.001;
  out.contact_footprint_overlaps = true;
  for (int i = 0; i < 3; ++i) {
    if (i == out.contact_axis) {
      continue;
    }
    if (gap[i] >= -kFootprintOverlapMarginM) {
      out.contact_footprint_overlaps = false;
      break;
    }
  }

  // Planar IoUs only need their own two axes to overlap (e.g. XZ doesn't
  // care whether the boxes also overlap along Y), so these are independent
  // of the full 3-axis volumetric overlap gated below. A real surface
  // contact normally has zero literal overlap on the separating axis (boxes
  // touch, they don't interpenetrate) — using raw `overlap` there would zero
  // out the whole plane even when the *other* in-plane axis genuinely
  // overlaps. Pad by tolerance_m instead, so any axis within touching
  // tolerance contributes a small positive projected extent rather than
  // clamping to zero; capped at min(size_a, size_b), the true maximum
  // possible 1D overlap, so an axis that already fully overlaps can't be
  // padded past 100% and push the resulting IoU above 1.
  Eigen::Vector3d padded_overlap;
  for (int i = 0; i < 3; ++i) {
    const double max_possible = std::min(std::max(0.0, size_a[i]), std::max(0.0, size_b[i]));
    padded_overlap[i] = std::min(max_possible, std::max(0.0, tolerance_m - gap[i]));
  }
  out.iou_xz = planarIou(padded_overlap[0], padded_overlap[2], size_a[0], size_a[2], size_b[0], size_b[2]);
  out.iou_yz = planarIou(padded_overlap[1], padded_overlap[2], size_a[1], size_a[2], size_b[1], size_b[2]);

  if ((gap.array() <= 0.0).all()) {
    out.overlap_volume_m3 = overlap[0] * overlap[1] * overlap[2];
    out.overlap_xy_m2 = overlap[0] * overlap[1];
    const double vol_a = std::max(0.0, size_a[0]) * std::max(0.0, size_a[1]) * std::max(0.0, size_a[2]);
    const double vol_b = std::max(0.0, size_b[0]) * std::max(0.0, size_b[1]) * std::max(0.0, size_b[2]);
    const double denom = vol_a + vol_b - out.overlap_volume_m3;
    out.iou_3d = denom > 1.0e-9 ? out.overlap_volume_m3 / denom : 0.0;
  }
  return out;
}

inline const char* contactAxisName(int axis) {
  switch (axis) {
    case 0: return "x";
    case 1: return "y";
    case 2: return "z";
    default: return "none";
  }
}

/**
 * @brief Uniform-grid cell index used to bucket object bbox centers so the
 * contact search only tests spatially nearby pairs instead of every pair.
 */
struct GridCell {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;
  bool operator==(const GridCell& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct GridCellHash {
  size_t operator()(const GridCell& cell) const {
    size_t seed = std::hash<int32_t>()(cell.x);
    seed ^= std::hash<int32_t>()(cell.y) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int32_t>()(cell.z) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
    return seed;
  }
};

inline GridCell cellForPoint(const Eigen::Vector3d& point, double cell_size_m) {
  return GridCell{static_cast<int32_t>(std::floor(point.x() / cell_size_m)),
                  static_cast<int32_t>(std::floor(point.y() / cell_size_m)),
                  static_cast<int32_t>(std::floor(point.z() / cell_size_m))};
}

inline int32_t markerId(NodeId id) {
  // RViz keys a marker by namespace and signed 32-bit id. Keep IDs stable
  // across graph updates so removed nodes can receive targeted DELETE markers.
  return static_cast<int32_t>(fnv1a(idString(id)) & 0x7fffffffULL);
}

inline int32_t markerIdForPair(NodeId source, NodeId target) {
  return static_cast<int32_t>(fnv1a(idString(source) + "_" + idString(target)) & 0x7fffffffULL);
}

inline Color hsvToRgb(double hue, double saturation, double value, float alpha = 1.0F) {
  const double h = hue - std::floor(hue);
  const double c = value * saturation;
  const double x = c * (1.0 - std::abs(std::fmod(h * 6.0, 2.0) - 1.0));
  const double m = value - c;
  double r = 0.0;
  double g = 0.0;
  double b = 0.0;
  const int sector = static_cast<int>(std::floor(h * 6.0)) % 6;
  switch (sector) {
    case 0:
      r = c;
      g = x;
      break;
    case 1:
      r = x;
      g = c;
      break;
    case 2:
      g = c;
      b = x;
      break;
    case 3:
      g = x;
      b = c;
      break;
    case 4:
      r = x;
      b = c;
      break;
    default:
      r = c;
      b = x;
      break;
  }
  return Color{static_cast<float>(r + m), static_cast<float>(g + m), static_cast<float>(b + m), alpha};
}

inline Color colorForLabel(const std::string& label) {
  if (!usableLabel(label)) {
    return Color{0.55F, 0.55F, 0.55F, 0.95F};
  }
  constexpr double denominator = static_cast<double>(std::numeric_limits<uint32_t>::max());
  const double hue = static_cast<double>(fnv1a(normaliseLabel(label)) & 0xffffffffULL) / denominator;
  return hsvToRgb(hue, 0.63, 0.93, 0.95F);
}

template <typename T>
T clampValue(const T& value, const T& low, const T& high) {
  return std::max(low, std::min(value, high));
}

inline int sourcePriority(const std::string& source) {
  std::string normalized;
  normalized.reserve(source.size());
  for (const unsigned char character : source) {
    normalized.push_back(static_cast<char>(std::tolower(character)));
  }
  if (normalized.find("vlm") != std::string::npos) {
    return 2;
  }
  if (normalized.find("rap") != std::string::npos) {
    return 1;
  }
  return 0;
}

inline bool overlayPreferred(const SemanticOverlay& incoming, const SemanticOverlay& existing) {
  const int incoming_rank = sourcePriority(incoming.source);
  const int existing_rank = sourcePriority(existing.source);
  if (incoming_rank != existing_rank) {
    return incoming_rank > existing_rank;
  }
  if (incoming.confidence != existing.confidence) {
    return incoming.confidence > existing.confidence;
  }
  return incoming.timestamp_sec >= existing.timestamp_sec;
}

}  // namespace rsg
