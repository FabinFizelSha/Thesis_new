/**
 * @file hydra_semantic_fuser.cpp
 * @brief Fuses final Phase-1 semantic labels with Hydra DSG object nodes.
 *
 * Hydra remains authoritative for spatial geometry and topology. The fuser
 * caches final labels by stable slot ID, batches incoming label callbacks, and
 * reapplies the cache to every current Hydra object node whenever the DSG or
 * semantic state changes.
 */
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
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
#include <spark_dsg/scene_graph_types.h>
#include <spark_dsg/serialization/graph_binary_serialization.h>

namespace rsg {
namespace {

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
  std::string source;
  double timestamp_sec = 0.0;
  bool has_centroid = false;
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  std::string centroid_frame_id;
};

struct ResolvedOverlay {
  SemanticOverlay overlay;
  std::string association = "slot_id";
  double centroid_distance_m = -1.0;
};


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

std::string idString(NodeId id) {
  return std::to_string(static_cast<uint64_t>(id));
}

std::string layerName(LayerKind kind) {
  switch (kind) {
    case LayerKind::kObjects:
      return "objects";
    case LayerKind::kRooms:
      return "rooms";
    case LayerKind::kBuildings:
      return "buildings";
    case LayerKind::kPlaces:
      return "places";
    case LayerKind::kSegments:
      return "segments";
    case LayerKind::kAgents:
      return "agents";
    default:
      return "other";
  }
}

bool isObject(const NodeView& node) {
  return node.kind == LayerKind::kObjects;
}

bool isRoom(const NodeView& node) {
  return node.kind == LayerKind::kRooms;
}

bool isPlace(const NodeView& node) {
  return node.kind == LayerKind::kPlaces;
}

bool isBuilding(const NodeView& node) {
  return node.kind == LayerKind::kBuildings;
}

std::string normaliseLabel(const std::string& raw) {
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
std::string normaliseMobilityClass(const std::string& raw) {
  const std::string value = normaliseLabel(raw);
  if (value == "dynamic" || value == "mobile") {
    return "dynamic";
  }
  if (value == "static" || value == "stationary" || value == "fixed") {
    return "static";
  }
  return "unknown";
}

bool usableLabel(const std::string& label) {
  const auto normalized = normaliseLabel(label);
  return !normalized.empty() && normalized != "unknown" && normalized != "unknown object" &&
         normalized != "unclassified" && normalized != "unclassified object";
}

Json vector3(const Eigen::Vector3d& value) {
  return Json::array({value.x(), value.y(), value.z()});
}

Json vector3f(const Eigen::Vector3f& value) {
  return Json::array({value.x(), value.y(), value.z()});
}

bool parseVector3(const Json& input, Eigen::Vector3d& output) {
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

uint64_t fnv1a(const std::string& value) {
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

const char* contactAxisName(int axis) {
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

int32_t markerId(NodeId id) {
  // RViz keys a marker by namespace and signed 32-bit id. Keep IDs stable
  // across graph updates so removed nodes can receive targeted DELETE markers.
  return static_cast<int32_t>(fnv1a(idString(id)) & 0x7fffffffULL);
}

int32_t markerIdForPair(NodeId source, NodeId target) {
  return static_cast<int32_t>(fnv1a(idString(source) + "_" + idString(target)) & 0x7fffffffULL);
}

Color hsvToRgb(double hue, double saturation, double value, float alpha = 1.0F) {
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

Color colorForLabel(const std::string& label) {
  if (!usableLabel(label)) {
    return Color{0.55F, 0.55F, 0.55F, 0.95F};
  }
  constexpr double denominator = static_cast<double>(std::numeric_limits<uint32_t>::max());
  const double hue = static_cast<double>(fnv1a(normaliseLabel(label)) & 0xffffffffULL) / denominator;
  return hsvToRgb(hue, 0.63, 0.93, 0.95F);
}

std::string edgeTypeName(DisplayEdgeType type) {
  switch (type) {
    case DisplayEdgeType::kRoomPlaceHierarchy:
      return "room_place_hierarchy";
    case DisplayEdgeType::kPlaceObjectMembership:
      return "place_object_membership";
    case DisplayEdgeType::kPlaceConnectivity:
      return "hydra_place_connectivity";
    case DisplayEdgeType::kNativeObjectObjectDebug:
      return "native_object_object_debug";
    case DisplayEdgeType::kNativeRoomRoomDebug:
      return "native_room_room_debug";
    case DisplayEdgeType::kDerivedObjectContact:
      return "derived_object_contact";
    case DisplayEdgeType::kDerivedObjectSegment:
      return "derived_object_segment";
  }
  return "unknown";
}

std::string edgeOriginName(EdgeOrigin origin) {
  switch (origin) {
    case EdgeOrigin::kNativeHydraEdge:
      return "native_hydra_edge";
    case EdgeOrigin::kDerivedMeshValidatedLocalPlace:
      return "derived_mesh_validated_local_place";
    case EdgeOrigin::kDerivedVisibleNeighbourRoom:
      return "derived_visible_neighbour_room_majority";
  }
  return "unknown";
}

template <typename T>
T clampValue(const T& value, const T& low, const T& high) {
  return std::max(low, std::min(value, high));
}

int sourcePriority(const std::string& source) {
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

bool overlayPreferred(const SemanticOverlay& incoming, const SemanticOverlay& existing) {
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

}  // namespace

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
  SemanticSceneGraphFuser() : Node("rsg_scene_graph_fuser") {
    input_dsg_topic_ = declare_parameter<std::string>("input_dsg_topic", "/hydra/backend/dsg");
    semantic_label_topic_ = declare_parameter<std::string>(
        "semantic_label_topic", "/rsg/objects/semantic_label_result");
    active_segments_topic_ = declare_parameter<std::string>(
        "active_segments_topic", "/rsg/objects/active_local_segments");
    semantic_label_qos_depth_ = static_cast<size_t>(std::max<int64_t>(
        1, declare_parameter<int64_t>("semantic_label_qos_depth", 4096)));
    semantic_refresh_rate_hz_ = std::max(
        0.1, declare_parameter<double>("semantic_refresh_rate_hz", 1.0));
    fused_dsg_topic_ = declare_parameter<std::string>("fused_dsg_topic", "/rsg/hydra/fused_dsg");
    markers_topic_ = declare_parameter<std::string>("markers_topic", "/rsg/scene_graph/markers");
    status_topic_ = declare_parameter<std::string>("status_topic", "/rsg/hydra/rap_fuser/status");
    fallback_frame_id_ = declare_parameter<std::string>("fallback_frame_id", "odom");

    // RViz consumes MarkerArray directly. Full DSG publication is optional.
    publish_fused_dsg_ = declare_parameter<bool>("publish_fused_dsg", false);
    // Semantic labels are batched by a short timer. This prevents a burst of
    // RAP/VLM completions from repeatedly rebuilding the full DSG projection.
    drop_local_mesh_ = declare_parameter<bool>("drop_local_mesh", true);
    marker_publish_rate_hz_ = std::max(0.0, declare_parameter<double>(
        "marker_publish_rate_hz", kDefaultMarkerRateHz));

    // Layered fused display. The local DSG remains an unchanged Hydra clone;
    // these offsets exist only in RViz MarkerArray positions.
    show_objects_ = declare_parameter<bool>("show_objects", true);
    show_rooms_ = declare_parameter<bool>("show_rooms", true);
    show_buildings_ = declare_parameter<bool>("show_buildings", false);
    show_places_ = declare_parameter<bool>("show_places", true);
    show_segments_ = declare_parameter<bool>("show_segments", false);
    show_agents_ = declare_parameter<bool>("show_agents", false);
    show_edges_ = declare_parameter<bool>("show_edges", true);
    show_room_place_edges_ = declare_parameter<bool>("show_room_place_edges", true);
    show_place_object_edges_ = declare_parameter<bool>("show_place_object_edges", true);
    show_place_connectivity_edges_ = declare_parameter<bool>("show_place_connectivity_edges", true);
    show_native_object_object_edges_ = declare_parameter<bool>("show_native_object_object_edges", false);
    show_native_room_room_edges_ = declare_parameter<bool>("show_native_room_room_edges", false);

    // Object-contact relationships: connect objects whose bounding boxes touch.
    object_contact_edges_enabled_ = declare_parameter<bool>("object_contact_edges_enabled", true);
    object_contact_tolerance_m_ = declare_parameter<double>("object_contact_tolerance_m", 0.05);
    object_contact_min_iou_3d_ = declare_parameter<double>("object_contact_min_iou_3d", 0.0);
    object_contact_write_dsg_edges_ = declare_parameter<bool>("object_contact_write_dsg_edges", true);
    object_contact_max_objects_ =
        static_cast<size_t>(std::max<int64_t>(0, declare_parameter<int>("object_contact_max_objects", 500)));
    // Spatial-hash grid narrows the pairing search: 0.0 = auto-size the cell
    // each cycle from the largest object reach seen (guarantees a single ring
    // of neighbor cells is enough); a positive override fixes the cell size
    // instead, with the per-object search ring widened as needed to keep
    // guaranteed coverage.
    object_contact_grid_cell_size_m_ = declare_parameter<double>("object_contact_grid_cell_size_m", 0.0);
    object_contact_grid_min_cell_size_m_ =
        declare_parameter<double>("object_contact_grid_min_cell_size_m", 0.05);
    show_object_contact_edges_ = declare_parameter<bool>("show_object_contact_edges", true);
    show_object_contact_labels_ = declare_parameter<bool>("show_object_contact_labels", true);
    object_contact_text_height_m_ =
        std::max(0.05, declare_parameter<double>("object_contact_text_height_m", 0.16));

    // Object-segment relationships: connect local segments phase1 split off
    // from one physical object that was too large to track as a single
    // segment (same PresenceObservation internal_object_id). Identity
    // relation, not geometric — no tolerance/grid params needed.
    object_segment_edges_enabled_ = declare_parameter<bool>("object_segment_edges_enabled", true);
    object_segment_write_dsg_edges_ =
        declare_parameter<bool>("object_segment_write_dsg_edges", true);
    object_segment_max_group_size_ = static_cast<size_t>(
        std::max<int64_t>(0, declare_parameter<int>("object_segment_max_group_size", 50)));
    show_object_segment_edges_ = declare_parameter<bool>("show_object_segment_edges", true);
    object_segment_dot_spacing_m_ =
        std::max(0.01, declare_parameter<double>("object_segment_dot_spacing_m", 0.08));
    object_segment_dot_size_m_ =
        std::max(0.005, declare_parameter<double>("object_segment_dot_size_m", 0.03));

    // Offline verification: dumps every main object's aggregate bbox, every
    // member's bbox, and the contact/segment edges actually produced, one
    // JSON line per fusion cycle, so
    // debug/fuser_object_relation_experiment/analyze_contact_diagnostics.py
    // can independently recompute the expected edges from raw geometry and
    // diff against what the fuser decided. Off by default — no effect on
    // production runs.
    object_contact_diagnostics_enabled_ =
        declare_parameter<bool>("object_contact_diagnostics_enabled", false);
    object_contact_diagnostics_path_ =
        declare_parameter<std::string>("object_contact_diagnostics_path", "");

    show_object_labels_ = declare_parameter<bool>("show_object_labels", true);
    show_room_labels_ = declare_parameter<bool>("show_room_labels", true);
    show_place_labels_ = declare_parameter<bool>("show_place_labels", false);
    show_building_labels_ = declare_parameter<bool>("show_building_labels", true);
    show_slot_ids_ = declare_parameter<bool>("show_slot_ids", false);
    show_presence_confidence_ = declare_parameter<bool>("show_presence_confidence", true);
    show_label_confidence_ = declare_parameter<bool>("show_label_confidence", true);
    show_mobility_metadata_ = declare_parameter<bool>("show_mobility_metadata", true);
    static_presence_half_life_sec_ = std::max(
        0.1, declare_parameter<double>("static_presence_half_life_sec", 600.0));
    dynamic_presence_half_life_sec_ = std::max(
        0.1, declare_parameter<double>("dynamic_presence_half_life_sec", 120.0));
    presence_observed_epsilon_sec_ = std::max(0.0, declare_parameter<double>("presence_observed_epsilon_sec", 1.5));
    minimum_object_alpha_ = static_cast<float>(clampValue(
        declare_parameter<double>("minimum_object_alpha", 0.03), 0.001, 1.0));
    dynamic_object_use_cube_ = declare_parameter<bool>("dynamic_object_use_cube", true);
    presence_decay_continuous_refresh_ = declare_parameter<bool>(
        "presence_decay_continuous_refresh", true);
    object_z_offset_m_ = declare_parameter<double>("object_z_offset_m", 0.0);
    place_z_offset_m_ = declare_parameter<double>("place_z_offset_m", 10.0);
    room_z_offset_m_ = declare_parameter<double>("room_z_offset_m", 20.0);
    room_node_size_m_ = std::max(0.10, declare_parameter<double>("room_node_size_m", 1.10));
    place_node_size_m_ = std::max(0.05, declare_parameter<double>("place_node_size_m", 0.45));
    place_text_height_m_ = std::max(0.05, declare_parameter<double>("place_text_height_m", 0.18));
    place_alpha_ = static_cast<float>(clampValue(declare_parameter<double>("place_alpha", 0.90), 0.0, 1.0));

    // Complete missing room/place membership in the fuser-owned display graph.
    // For each orphaned place, the nearest visible room-owned places vote for
    // the parent room. The test is mesh validated and therefore fails closed.
    room_place_completion_enabled_ = declare_parameter<bool>("room_place_completion_enabled", true);
    room_place_completion_grace_period_sec_ = std::max(0.0, declare_parameter<double>(
        "room_place_completion_grace_period_sec", 30.0));
    const int64_t room_place_completion_neighbours =
        declare_parameter<int64_t>("room_place_completion_neighbours", 7);
    room_place_completion_neighbours_ = static_cast<int>(
        std::max<int64_t>(1, room_place_completion_neighbours));
    room_place_completion_max_distance_m_ = std::max(0.0, declare_parameter<double>(
        "room_place_completion_max_distance_m", 0.0));
    room_place_completion_max_height_difference_m_ = std::max(0.0, declare_parameter<double>(
        "room_place_completion_max_height_difference_m", 0.50));
    room_place_completion_require_mesh_validation_ = declare_parameter<bool>(
        "room_place_completion_require_mesh_validation", true);
    const int64_t room_place_completion_min_majority_votes =
        declare_parameter<int64_t>("room_place_completion_min_majority_votes", 1);
    room_place_completion_min_majority_votes_ = static_cast<int>(
        std::max<int64_t>(1, room_place_completion_min_majority_votes));

    object_place_use_local_validated_fallback_ = declare_parameter<bool>(
        "object_place_use_local_validated_fallback", true);
    object_place_max_distance_m_ = std::max(0.0, declare_parameter<double>(
        "object_place_max_distance_m", 3.0));
    object_place_index_voxel_size_m_ = std::max(0.10, declare_parameter<double>(
        "object_place_index_voxel_size_m", 2.0));
    const int64_t object_place_index_search_radius_cells =
        declare_parameter<int64_t>("object_place_index_search_radius_cells", 2);
    object_place_index_search_radius_cells_ = static_cast<int>(
        std::max<int64_t>(0, object_place_index_search_radius_cells));

    const int64_t object_place_max_candidates =
        declare_parameter<int64_t>("object_place_max_candidates", 6);
    object_place_max_candidates_ = static_cast<int>(
        std::max<int64_t>(1, object_place_max_candidates));
    object_place_require_mesh_validation_ = declare_parameter<bool>(
        "object_place_require_mesh_validation", true);
    object_place_mesh_voxel_size_m_ = std::max(0.10, declare_parameter<double>(
        "object_place_mesh_voxel_size_m", 0.50));
    const int64_t object_place_mesh_max_triangle_tests =
        declare_parameter<int64_t>("object_place_mesh_max_triangle_tests", 2048);
    object_place_mesh_max_triangle_tests_ = static_cast<int>(
        std::max<int64_t>(1, object_place_mesh_max_triangle_tests));

    const int64_t object_place_mesh_max_cells_per_triangle =
        declare_parameter<int64_t>("object_place_mesh_max_cells_per_triangle", 256);
    object_place_mesh_max_cells_per_triangle_ = static_cast<int>(
        std::max<int64_t>(1, object_place_mesh_max_cells_per_triangle));
    object_place_anchor_outset_m_ = std::max(0.0, declare_parameter<double>(
        "object_place_anchor_outset_m", 0.12));
    object_place_cache_recompute_translation_m_ = std::max(0.0, declare_parameter<double>(
        "object_place_cache_recompute_translation_m", 0.25));

    centroid_association_enabled_ = declare_parameter<bool>("centroid_association_enabled", true);
    require_centroid_frame_match_ = declare_parameter<bool>("require_centroid_frame_match", true);
    allow_unframed_centroid_ = declare_parameter<bool>("allow_unframed_centroid", false);
    const int64_t max_label_candidates_per_slot = declare_parameter<int64_t>(
        "max_label_candidates_per_slot", 8);
    max_label_candidates_per_slot_ = static_cast<size_t>(
        std::max<int64_t>(1, max_label_candidates_per_slot));
    unlabeled_object_display_label_ = normaliseLabel(declare_parameter<std::string>(
        "unlabeled_object_display_label", "unknown object"));
    if (unlabeled_object_display_label_.empty()) {
      unlabeled_object_display_label_ = "unknown object";
    }

    object_min_size_m_ = std::max(kSmallExtentM, declare_parameter<double>("object_min_size_m", 0.12));
    object_max_size_m_ = std::max(object_min_size_m_, declare_parameter<double>("object_max_size_m", 0.60));
    object_sphere_volume_scale_ = std::max(0.0, declare_parameter<double>(
        "object_sphere_volume_scale", 0.60));
    object_sphere_size_mode_ = normaliseLabel(declare_parameter<std::string>(
        "object_sphere_size_mode", "volume_scaled"));
    const bool mode_requests_fixed_size =
        object_sphere_size_mode_ == "fixed" ||
        object_sphere_size_mode_ == "constant" ||
        object_sphere_size_mode_ == "uniform";
    object_use_fixed_sphere_size_ = declare_parameter<bool>(
        "object_use_fixed_sphere_size", mode_requests_fixed_size);
    object_fixed_sphere_size_m_ = std::max(kSmallExtentM, declare_parameter<double>(
        "object_fixed_sphere_size_m", 0.28));
    object_text_height_m_ = std::max(0.05, declare_parameter<double>("object_text_height_m", 0.28));
    object_label_vertical_offset_m_ = std::max(
        0.0, declare_parameter<double>("object_label_vertical_offset_m", 0.10));
    room_text_height_m_ = std::max(0.05, declare_parameter<double>("room_text_height_m", 0.35));
    building_text_height_m_ = std::max(0.05, declare_parameter<double>("building_text_height_m", 0.42));
    room_alpha_ = static_cast<float>(clampValue(declare_parameter<double>("room_alpha", 0.14), 0.0, 1.0));
    building_alpha_ = static_cast<float>(clampValue(declare_parameter<double>("building_alpha", 0.10), 0.0, 1.0));
    edge_width_m_ = std::max(0.005, declare_parameter<double>("edge_width_m", 0.025));

    const auto input_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    const auto semantic_qos = rclcpp::QoS(rclcpp::KeepLast(semantic_label_qos_depth_))
                                  .reliable()
                                  .durability_volatile();
    const auto retained_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    // Label ingestion, Hydra graph updates, and rendering run in separate
    // callback groups. A multi-threaded executor can therefore drain the
    // reliable final-label queue while a slower render pass builds markers.
    graph_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    semantic_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    render_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions graph_options;
    graph_options.callback_group = graph_callback_group_;
    rclcpp::SubscriptionOptions semantic_options;
    semantic_options.callback_group = semantic_callback_group_;

    dsg_sub_ = create_subscription<DsgUpdate>(
        input_dsg_topic_, input_qos,
        std::bind(&SemanticSceneGraphFuser::handleDsg, this, std::placeholders::_1),
        graph_options);
    label_sub_ = create_subscription<std_msgs::msg::String>(
        semantic_label_topic_, semantic_qos,
        std::bind(&SemanticSceneGraphFuser::handleSemanticLabel, this, std::placeholders::_1),
        semantic_options);
    active_segments_sub_ = create_subscription<std_msgs::msg::String>(
        active_segments_topic_, semantic_qos,
        std::bind(&SemanticSceneGraphFuser::handleActiveSegments, this, std::placeholders::_1),
        semantic_options);

    fused_dsg_pub_ = create_publisher<DsgUpdate>(fused_dsg_topic_, retained_qos);
    marker_pub_ = create_publisher<MarkerArray>(markers_topic_, retained_qos);
    status_pub_ = create_publisher<std_msgs::msg::String>(status_topic_, retained_qos);

    const auto refresh_period = std::chrono::milliseconds(std::max<int64_t>(
        1, static_cast<int64_t>(std::llround(1000.0 / semantic_refresh_rate_hz_))));
    semantic_refresh_timer_ = create_wall_timer(
        refresh_period,
        std::bind(&SemanticSceneGraphFuser::renderDirtyState, this),
        render_callback_group_);

    RCLCPP_INFO(
        get_logger(),
        "Hydra semantic fuser ready: DSG='%s', final labels='%s', label QoS depth=%zu, markers='%s'",
        input_dsg_topic_.c_str(), semantic_label_topic_.c_str(), semantic_label_qos_depth_,
        markers_topic_.c_str());
    publishStatus("started", "ready");

    // No other param on this node is live-updatable (every declare_parameter
    // above is read once here and cached into a member); a `ros2 param set`
    // on an already-running node normally has no effect at all. That's the
    // wrong UX specifically for a diagnostics on/off switch someone wants to
    // flip mid-run without restarting the whole pipeline, so these two are
    // deliberately the exception.
    parameter_callback_handle_ = add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = true;
          for (const auto& parameter : parameters) {
            if (parameter.get_name() == "object_contact_diagnostics_enabled" &&
                parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) {
              object_contact_diagnostics_enabled_ = parameter.as_bool();
              RCLCPP_INFO(get_logger(), "object contact diagnostics: %s (live update)",
                         object_contact_diagnostics_enabled_ ? "enabled" : "disabled");
            } else if (parameter.get_name() == "object_contact_diagnostics_path" &&
                       parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
              object_contact_diagnostics_path_ = parameter.as_string();
            }
          }
          return result;
        });
  }



  ~SemanticSceneGraphFuser() override = default;
 private:
  bool layerVisible(LayerKind kind) const {
    switch (kind) {
      case LayerKind::kObjects:
        return show_objects_;
      case LayerKind::kRooms:
        return show_rooms_;
      case LayerKind::kBuildings:
        return show_buildings_;
      case LayerKind::kPlaces:
        return show_places_;
      case LayerKind::kSegments:
        return show_segments_;
      case LayerKind::kAgents:
        return show_agents_;
      default:
        return false;
    }
  }

  /**
   * Update the local Hydra DSG and request a later render pass.
   *
   * This callback intentionally does not build markers, snapshots, or a
   * layered projection. Those operations scale with map size and belong to
   * the capped renderer, not to the input path.
   */
  void handleDsg(const DsgUpdate::SharedPtr msg) {
    std::string failure_state;
    std::string failure_detail;
    {
      std::lock_guard<std::mutex> graph_lock(graph_mutex_);
      latest_header_ = msg->header;
      latest_sequence_ = msg->sequence_number;

      try {
        if (msg->full_update) {
          // A full update is authoritative: omitted nodes and edges are removed
          // by replacing the local graph, which also handles every deletion.
          graph_ = spark_dsg::io::binary::readGraph(msg->layer_contents);
          ++full_dsg_updates_;
        } else {
          // Incremental payloads cannot initialise a complete graph.
          if (!graph_) {
            ++skipped_initial_incremental_updates_;
            failure_state = "waiting_for_full_update";
            failure_detail = "received_incremental_update_without_local_graph";
          } else if (!spark_dsg::io::binary::updateGraph(*graph_, msg->layer_contents)) {
            ++deserialize_failures_;
            failure_state = "incremental_update_failed";
            failure_detail = "Spark-DSG rejected incremental payload";
          } else {
            applyDsgDeletions(*msg);
            ++incremental_dsg_updates_;
          }
        }
        if (failure_state.empty() && drop_local_mesh_ && graph_ && graph_->hasMesh()) {
          graph_->setMesh(std::shared_ptr<spark_dsg::Mesh>{});
        }
      } catch (const std::exception& error) {
        ++deserialize_failures_;
        failure_state = "deserialize_failed";
        failure_detail = error.what();
      }

      if (failure_state.empty() && !graph_) {
        failure_state = "empty_graph";
        failure_detail = "Hydra full update produced no graph";
      }
      if (failure_state.empty()) {
        ++raw_dsg_updates_;
        updateRoomPlaceCompletionGraceClock(msg->header);
      }
    }

    if (!failure_state.empty()) {
      publishStatus(failure_state, failure_detail);
      if (failure_state == "incremental_update_failed" || failure_state == "deserialize_failed") {
        RCLCPP_ERROR(get_logger(), "%s", failure_detail.c_str());
      }
      return;
    }

    // Any DSG change can introduce a new Hydra object node for a slot whose
    // semantic result was already received. The next render reapplies labels
    // across the full current object set.
    render_dirty_.store(true, std::memory_order_release);
  }

  void applyDsgDeletions(const DsgUpdate& msg) {
    if (!graph_) {
      return;
    }

    // hydra_msgs/DsgUpdate serialises deleted edge endpoints consecutively:
    // [source_0, target_0, source_1, target_1, ...]. Nodes are removed after
    // the binary update so the local copy exactly follows Hydra's current map.
    for (const uint64_t raw_id : msg.deleted_nodes) {
      if (graph_->removeNode(static_cast<NodeId>(raw_id))) {
        ++deleted_nodes_applied_;
      }
    }

    if ((msg.deleted_edges.size() % 2U) != 0U) {
      ++malformed_deleted_edge_updates_;
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Hydra DSG update contains an odd deleted_edges count (%zu); ignoring final unmatched endpoint",
          msg.deleted_edges.size());
    }
    for (size_t index = 0; index + 1U < msg.deleted_edges.size(); index += 2U) {
      const auto source = static_cast<NodeId>(msg.deleted_edges[index]);
      const auto target = static_cast<NodeId>(msg.deleted_edges[index + 1U]);
      if (graph_->removeEdge(source, target)) {
        ++deleted_edges_applied_;
      }
    }
  }

  /**
   * Cache one final Phase-1 result and return immediately.
   *
   * The callback performs no DSG traversal, marker creation, or RViz publish.
   * This prevents a growing fused graph from blocking the reliable semantic
   * result queue near the end of a rosbag replay.
   */
  void handleSemanticLabel(const std_msgs::msg::String::SharedPtr msg) {
    Json payload;
    try {
      payload = Json::parse(msg->data);
    } catch (const std::exception&) {
      ++invalid_label_messages_;
      return;
    }
    if (!payload.is_object()) {
      ++invalid_label_messages_;
      return;
    }
    const std::string event = payload.value("event", std::string());
    // The fuser consumes only terminal Phase-1 outcomes. Raw RAP retrieval
    // messages remain available for diagnostics but never colour the map.
    if (event != "semantic_label_result") {
      return;
    }

    SemanticOverlay overlay;
    try {
      const auto raw_slot = payload.value("hydra_slot_id", payload.value("slot_id", 0));
      const int64_t slot = raw_slot;
      if (slot <= 0 || slot > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        ++invalid_label_messages_;
        return;
      }
      overlay.slot_id = static_cast<uint32_t>(slot);
    } catch (const std::exception&) {
      ++invalid_label_messages_;
      return;
    }

    overlay.label = normaliseLabel(payload.value("label", payload.value("final_label", std::string())));
    if (!usableLabel(overlay.label)) {
      ++ignored_label_messages_;
      return;
    }
    try {
      overlay.confidence = payload.value("confidence", payload.value("final_label_confidence", 0.0));
    } catch (const std::exception&) {
      overlay.confidence = 0.0;
    }
    overlay.mobility_class = normaliseMobilityClass(
        payload.value("mobility_class", std::string("unknown")));
    try {
      overlay.mobility_confidence = clampValue(
          payload.value("mobility_confidence", 0.0), 0.0, 1.0);
    } catch (const std::exception&) {
      overlay.mobility_confidence = 0.0;
    }
    overlay.mobility_source = payload.value("mobility_source", std::string("none"));
    overlay.source = payload.value("source", std::string("none"));
    ++accepted_semantic_result_events_;
    try {
      overlay.timestamp_sec = payload.value("timestamp_sec", 0.0);
    } catch (const std::exception&) {
      overlay.timestamp_sec = 0.0;
    }
    overlay.centroid_frame_id = payload.value("centroid_frame_id", std::string());
    const auto centroid_it = payload.find("centroid_3d");
    if (centroid_it != payload.end()) {
      overlay.has_centroid = parseVector3(*centroid_it, overlay.centroid);
    }

    {
      std::lock_guard<std::mutex> labels_lock(overlays_mutex_);
      auto& candidates = overlays_by_slot_[overlay.slot_id];
      auto same_label = std::find_if(
          candidates.begin(), candidates.end(),
          [&overlay](const SemanticOverlay& existing) {
            return existing.label == overlay.label;
          });

      if (same_label != candidates.end()) {
        // Keep the strongest/current record for an already-known class while
        // retaining genuinely conflicting labels for centroid tie-breaking.
        if (!overlayPreferred(overlay, *same_label)) {
          ++ignored_label_messages_;
          return;
        }
        *same_label = overlay;
      } else {
        candidates.push_back(overlay);
        if (candidates.size() > max_label_candidates_per_slot_) {
          const auto weakest = std::min_element(
              candidates.begin(), candidates.end(),
              [](const SemanticOverlay& lhs, const SemanticOverlay& rhs) {
                const int lhs_rank = sourcePriority(lhs.source);
                const int rhs_rank = sourcePriority(rhs.source);
                if (lhs_rank != rhs_rank) {
                  return lhs_rank < rhs_rank;
                }
                if (lhs.confidence != rhs.confidence) {
                  return lhs.confidence < rhs.confidence;
                }
                return lhs.timestamp_sec < rhs.timestamp_sec;
              });
          if (weakest != candidates.end()) {
            candidates.erase(weakest);
          }
        }
      }
    }

    ++accepted_label_messages_;
    ++semantic_label_generation_;
    semantic_refresh_pending_.store(true, std::memory_order_release);
    render_dirty_.store(true, std::memory_order_release);
  }

  void handleActiveSegments(const std_msgs::msg::String::SharedPtr msg) {
    Json payload;
    try {
      payload = Json::parse(msg->data);
    } catch (const std::exception&) {
      ++invalid_active_segment_messages_;
      return;
    }
    if (!payload.is_object() || payload.value("event", std::string()) != "local_segment_observations") {
      ++invalid_active_segment_messages_;
      return;
    }
    const auto segments_it = payload.find("segments");
    if (segments_it == payload.end() || !segments_it->is_array()) {
      ++invalid_active_segment_messages_;
      return;
    }
    size_t accepted = 0;
    std::lock_guard<std::mutex> presence_lock(presence_mutex_);
    for (const auto& segment : *segments_it) {
      if (!segment.is_object()) {
        continue;
      }
      int64_t raw_slot = segment.value("hydra_slot_id", segment.value("slot_id", 0));
      if (raw_slot <= 0 || raw_slot > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        continue;
      }
      PresenceObservation obs;
      obs.slot_id = static_cast<uint32_t>(raw_slot);
      obs.internal_object_id = segment.value("internal_object_id", std::string());
      obs.persistent_track_id = segment.value("persistent_track_id", obs.internal_object_id);
      obs.local_segment_id = segment.value("local_segment_id", segment.value("semantic_segment_id", std::string()));
      obs.last_observed_timestamp_sec = segment.value("last_observed_timestamp_sec", payload.value("timestamp_sec", 0.0));
      const auto centroid_it = segment.find("centroid_3d");
      if (centroid_it != segment.end()) {
        obs.has_centroid = parseVector3(*centroid_it, obs.centroid);
      }
      const auto bbox_min_it = segment.find("bbox_3d_min");
      const auto bbox_max_it = segment.find("bbox_3d_max");
      if (bbox_min_it != segment.end() && bbox_max_it != segment.end()) {
        obs.has_bbox = parseVector3(*bbox_min_it, obs.bbox_min) && parseVector3(*bbox_max_it, obs.bbox_max);
      }
      try {
        obs.local_segment_xy_span_m = segment.value("local_segment_xy_span_m", 0.0);
      } catch (const std::exception&) {
        obs.local_segment_xy_span_m = 0.0;
      }
      obs.raw = segment;
      presence_by_slot_[obs.slot_id] = obs;
      ++accepted;
    }
    if (accepted == 0U) {
      return;
    }
    accepted_active_segment_messages_.fetch_add(static_cast<uint64_t>(accepted), std::memory_order_relaxed);
    render_dirty_.store(true, std::memory_order_release);
  }

  /**
   * Render the latest graph at a bounded rate.
   *
   * Label updates are copied before the graph mutex is acquired. Therefore the
   * semantic callback can continue storing incoming slot labels while a long
   * marker/snapshot render traverses the Hydra graph.
   */
  void renderDirtyState() {
    const bool explicitly_dirty = render_dirty_.exchange(false, std::memory_order_acq_rel);
    if (!explicitly_dirty) {
      if (!presence_decay_continuous_refresh_) {
        return;
      }
      {
        std::lock_guard<std::mutex> presence_lock(presence_mutex_);
        if (presence_by_slot_.empty()) {
          return;
        }
      }
      const double reference_time_sec = currentReferenceTimeSec();
      if (std::abs(reference_time_sec - last_presence_refresh_reference_time_sec_) < 1.0e-6) {
        return;
      }
      last_presence_refresh_reference_time_sec_ = reference_time_sec;
    }

    std::unique_lock<std::mutex> render_lock(render_mutex_, std::try_to_lock);
    if (!render_lock.owns_lock()) {
      render_dirty_.store(true, std::memory_order_release);
      return;
    }

    const uint64_t label_generation_before = semantic_label_generation_.load();
    OverlayCache overlay_snapshot;
    {
      std::lock_guard<std::mutex> labels_lock(overlays_mutex_);
      overlay_snapshot = overlays_by_slot_;
    }
    PresenceCache presence_snapshot;
    {
      std::lock_guard<std::mutex> presence_lock(presence_mutex_);
      presence_snapshot = presence_by_slot_;
    }

    if (!publishFusedOutputs(false, "bounded_fused_graph_refresh", overlay_snapshot, presence_snapshot)) {
      return;
    }

    last_presence_refresh_reference_time_sec_ = currentReferenceTimeSec();
    ++semantic_refresh_publications_;
    if (semantic_label_generation_.load() == label_generation_before) {
      semantic_refresh_pending_.store(false, std::memory_order_release);
    } else {
      // A label arrived during rendering. Preserve the request for the next
      // bounded refresh instead of dropping the newest semantic state.
      semantic_refresh_pending_.store(true, std::memory_order_release);
      render_dirty_.store(true, std::memory_order_release);
    }
  }

  SceneModel collectModel() const {
    SceneModel model;
    if (!graph_) {
      return model;
    }

    collectLayer(model, spark_dsg::DsgLayers::OBJECTS, LayerKind::kObjects);
    collectLayer(model, spark_dsg::DsgLayers::ROOMS, LayerKind::kRooms);
    collectLayer(model, spark_dsg::DsgLayers::BUILDINGS, LayerKind::kBuildings);
    collectLayer(model, spark_dsg::DsgLayers::PLACES, LayerKind::kPlaces);
    collectLayer(model, spark_dsg::DsgLayers::SEGMENTS, LayerKind::kSegments);
    collectLayer(model, spark_dsg::DsgLayers::AGENTS, LayerKind::kAgents);

    std::set<std::pair<NodeId, NodeId>> seen_edges;
    const auto collect_edges = [&model, &seen_edges](const auto& edges) {
      for (const auto& [edge_key, edge] : edges) {
        (void)edge_key;
        if (!model.nodes.count(edge.source) || !model.nodes.count(edge.target)) {
          continue;
        }
        const auto key = std::make_pair(std::min(edge.source, edge.target), std::max(edge.source, edge.target));
        if (!seen_edges.insert(key).second) {
          continue;
        }
        model.raw_edges.push_back(RawEdge{edge.source, edge.target});
        model.adjacency[edge.source].push_back(edge.target);
        model.adjacency[edge.target].push_back(edge.source);
      }
    };

    for (const auto& layer_name : {spark_dsg::DsgLayers::OBJECTS,
                                   spark_dsg::DsgLayers::ROOMS,
                                   spark_dsg::DsgLayers::BUILDINGS,
                                   spark_dsg::DsgLayers::PLACES,
                                   spark_dsg::DsgLayers::SEGMENTS,
                                   spark_dsg::DsgLayers::AGENTS}) {
      const auto* layer = graph_->findLayer(layer_name);
      if (layer) {
        collect_edges(layer->edges());
      }
    }
    collect_edges(graph_->interlayer_edges());
    return model;
  }

  void collectLayer(SceneModel& model, const std::string& layer_name, LayerKind kind) const {
    const auto* layer = graph_->findLayer(layer_name);
    if (!layer) {
      return;
    }
    for (const auto& [node_id, node] : layer->nodes()) {
      const auto* attrs = node->tryAttributes<spark_dsg::NodeAttributes>();
      if (!attrs) {
        continue;
      }
      NodeView view;
      view.id = node_id;
      view.kind = kind;
      view.visible = layerVisible(kind);
      view.position = attrs->position;

      const auto* semantic_attrs = node->tryAttributes<spark_dsg::SemanticNodeAttributes>();
      if (semantic_attrs) {
        view.name = semantic_attrs->name;
        view.semantic_slot = static_cast<uint32_t>(semantic_attrs->semantic_label);
        if (semantic_attrs->bounding_box.isValid()) {
          view.has_bbox = true;
          view.bbox_center = semantic_attrs->bounding_box.world_P_center;
          view.bbox_size = semantic_attrs->bounding_box.dimensions;
        }
      }
      model.nodes.emplace(node_id, std::move(view));
    }
  }

  bool centroidUsable(const SemanticOverlay& overlay, const std::string& dsg_frame) const {
    if (!centroid_association_enabled_ || !overlay.has_centroid) {
      return false;
    }
    if (overlay.centroid_frame_id.empty()) {
      return allow_unframed_centroid_;
    }
    if (!require_centroid_frame_match_) {
      return true;
    }
    return overlay.centroid_frame_id == dsg_frame;
  }

  std::unordered_map<NodeId, ResolvedOverlay> resolveOverlays(
      const SceneModel& model,
      const std::string& dsg_frame,
      const OverlayCache& overlays) const {
    std::unordered_map<uint32_t, std::vector<const NodeView*>> objects_by_slot;
    for (const auto& [node_id, node] : model.nodes) {
      (void)node_id;
      if (node.kind == LayerKind::kObjects && node.semantic_slot > 0) {
        objects_by_slot[node.semantic_slot].push_back(&node);
      }
    }

    std::unordered_map<NodeId, ResolvedOverlay> resolved;
    for (const auto& [slot_id, candidates] : overlays) {
      const auto found = objects_by_slot.find(slot_id);
      if (found == objects_by_slot.end() || found->second.empty() || candidates.empty()) {
        continue;
      }

      // The semantic slot is the primary and authoritative cross-pipeline
      // identifier. Hydra may temporarily contain multiple object nodes with
      // one slot after object fragmentation or DSG updates. When all available
      // Phase-1 results agree on a label, every Hydra node carrying that slot
      // receives the same label. SAM centroid data is intentionally not used
      // as a gate in this normal path.
      std::unordered_map<std::string, const SemanticOverlay*> best_by_label;
      for (const auto& candidate : candidates) {
        const auto existing = best_by_label.find(candidate.label);
        if (existing == best_by_label.end() || overlayPreferred(candidate, *existing->second)) {
          best_by_label[candidate.label] = &candidate;
        }
      }

      if (best_by_label.size() == 1U) {
        const auto* selected = best_by_label.begin()->second;
        for (const auto* node : found->second) {
          resolved[node->id] = ResolvedOverlay{*selected, "slot_id_all_matching_nodes", -1.0};
        }
        continue;
      }

      // A single slot should normally resolve to one label. If it carries
      // genuinely different labels, centroid data is used only as a tie-breaker
      // between those competing semantic events. There is deliberately no
      // distance rejection threshold: slot equality remains the first check.
      std::vector<const SemanticOverlay*> centroid_candidates;
      centroid_candidates.reserve(best_by_label.size());
      for (const auto& [label, candidate] : best_by_label) {
        (void)label;
        if (centroidUsable(*candidate, dsg_frame)) {
          centroid_candidates.push_back(candidate);
        }
      }

      if (!centroid_candidates.empty()) {
        for (const auto* node : found->second) {
          const SemanticOverlay* selected = nullptr;
          double best_distance = std::numeric_limits<double>::max();
          for (const auto* candidate : centroid_candidates) {
            const double distance = (node->position - candidate->centroid).norm();
            if (distance < best_distance ||
                (distance == best_distance && selected && overlayPreferred(*candidate, *selected))) {
              best_distance = distance;
              selected = candidate;
            }
          }
          if (selected) {
            resolved[node->id] = ResolvedOverlay{
                *selected, "slot_id_conflict_centroid", best_distance};
          }
        }
        continue;
      }

      // If conflicting labels have no comparable centroids, apply a stable
      // source/confidence/timestamp winner to every Hydra node carrying the
      // slot. This keeps the fused graph complete while surfacing the conflict
      // explicitly in diagnostics through the association field.
      const SemanticOverlay* selected = nullptr;
      for (const auto& [label, candidate] : best_by_label) {
        (void)label;
        if (!selected || overlayPreferred(*candidate, *selected)) {
          selected = candidate;
        }
      }
      if (selected) {
        for (const auto* node : found->second) {
          resolved[node->id] = ResolvedOverlay{
              *selected, "slot_id_conflict_priority_fallback", -1.0};
        }
      }
    }
    return resolved;
  }

  /**
   * Remove the object-contact edges this node added to the DSG clone last
   * cycle. Called before collectModel() so the model only carries native
   * Hydra topology. Never touches an edge the fuser did not create.
   */
  void pruneOwnedContactEdges() {
    if (!graph_) {
      contact_edges_prev_.clear();
      return;
    }
    for (const auto& [source, target] : contact_edges_prev_) {
      graph_->removeEdge(source, target);
    }
    contact_edges_prev_.clear();
  }

  /** Same as pruneOwnedContactEdges(), for the same-physical-object segment edges. */
  void pruneOwnedSegmentEdges() {
    if (!graph_) {
      segment_edges_prev_.clear();
      return;
    }
    for (const auto& [source, target] : segment_edges_prev_) {
      graph_->removeEdge(source, target);
    }
    segment_edges_prev_.clear();
  }

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
                                                      const PresenceCache& presence) const {
    const auto internal_id_for = [&](uint32_t slot) -> std::string {
      const auto it = presence.find(slot);
      return it == presence.end() ? std::string() : it->second.internal_object_id;
    };

    std::unordered_map<std::string, MainObjectGroup> by_key;
    for (const auto& [node_id, node] : model.nodes) {
      if (node.kind != LayerKind::kObjects) {
        continue;
      }
      const std::string internal_id = internal_id_for(node.semantic_slot);
      // Unique per-node key for a standalone object, so it never merges with
      // another standalone object under an empty internal_object_id.
      const std::string key = internal_id.empty() ? ("__solo_" + idString(node_id)) : internal_id;

      SegmentMember member;
      member.id = node_id;
      member.semantic_slot = node.semantic_slot;
      member.position = node.position;
      member.has_bbox = node.has_bbox;
      member.bbox_center = node.bbox_center.cast<double>();
      member.bbox_size = node.bbox_size.cast<double>();

      auto& group = by_key[key];
      group.internal_object_id = internal_id;
      group.members.push_back(std::move(member));
    }

    std::vector<MainObjectGroup> groups;
    groups.reserve(by_key.size());
    for (auto& [key, group] : by_key) {
      std::sort(group.members.begin(), group.members.end(),
               [](const SegmentMember& lhs, const SegmentMember& rhs) { return lhs.id < rhs.id; });
      groups.push_back(std::move(group));
    }
    std::sort(groups.begin(), groups.end(), [](const MainObjectGroup& lhs, const MainObjectGroup& rhs) {
      return lhs.members.front().id < rhs.members.front().id;
    });
    return groups;
  }

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
                             LayeredProjection& projection) {
    last_object_relations_ = Json::array();
    last_object_contact_edge_count_ = 0;
    last_object_contact_max_iou_ = 0.0;
    last_object_contact_grid_cell_size_m_ = 0.0;
    last_object_contact_pairs_tested_ = 0;
    if (!object_contact_edges_enabled_ || !graph_) {
      return;
    }

    struct MainObjectGeom {
      size_t group_index = 0;
      Eigen::Vector3d agg_center = Eigen::Vector3d::Zero();
      Eigen::Vector3d agg_size = Eigen::Vector3d::Zero();
      double reach_m = 0.0;  //!< half the longest aggregate dimension + tolerance
      GridCell cell;
    };

    // Pass 1: aggregate bbox per main object (union of member bboxes; groups
    // with no bbox anywhere have no geometry to test and are skipped), and
    // the largest per-object reach, so the grid cell size can guarantee full
    // coverage from a single ring of neighboring cells.
    std::vector<MainObjectGeom> geoms;
    geoms.reserve(groups.size());
    double global_max_reach_m = 0.0;
    for (size_t g = 0; g < groups.size(); ++g) {
      bool any_bbox = false;
      Eigen::Vector3d agg_min = Eigen::Vector3d::Zero();
      Eigen::Vector3d agg_max = Eigen::Vector3d::Zero();
      for (const auto& member : groups[g].members) {
        if (!member.has_bbox) {
          continue;
        }
        const Eigen::Vector3d half = 0.5 * member.bbox_size.cwiseMax(0.0);
        const Eigen::Vector3d mn = member.bbox_center - half;
        const Eigen::Vector3d mx = member.bbox_center + half;
        if (!any_bbox) {
          agg_min = mn;
          agg_max = mx;
          any_bbox = true;
        } else {
          agg_min = agg_min.cwiseMin(mn);
          agg_max = agg_max.cwiseMax(mx);
        }
      }
      if (!any_bbox) {
        continue;
      }
      MainObjectGeom geom;
      geom.group_index = g;
      geom.agg_center = 0.5 * (agg_min + agg_max);
      geom.agg_size = agg_max - agg_min;
      geom.reach_m = 0.5 * geom.agg_size.cwiseMax(0.0).maxCoeff() + object_contact_tolerance_m_;
      global_max_reach_m = std::max(global_max_reach_m, geom.reach_m);
      geoms.push_back(geom);
    }

    if (object_contact_max_objects_ > 0 && geoms.size() > object_contact_max_objects_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                           "object contact: %zu main objects exceeds object_contact_max_objects=%zu; "
                           "skipping", geoms.size(), object_contact_max_objects_);
      return;
    }
    if (geoms.empty()) {
      return;
    }

    const double now_sec = currentReferenceTimeSec();
    const double cell_size_m =
        object_contact_grid_cell_size_m_ > 0.0
            ? object_contact_grid_cell_size_m_
            : std::max(object_contact_grid_min_cell_size_m_, 2.0 * global_max_reach_m);
    last_object_contact_grid_cell_size_m_ = cell_size_m;

    // Pass 2: bucket main objects by cell.
    std::unordered_map<GridCell, std::vector<size_t>, GridCellHash> grid;
    grid.reserve(geoms.size());
    for (size_t idx = 0; idx < geoms.size(); ++idx) {
      geoms[idx].cell = cellForPoint(geoms[idx].agg_center, cell_size_m);
      grid[geoms[idx].cell].push_back(idx);
    }

    // Canonical per-pair key (each group's smallest member id, distinct
    // across groups since they partition all object nodes) gives the same
    // dedup guarantee the old id-based i<j check gave over raw nodes.
    const auto geom_key = [&groups](const MainObjectGeom& geom) {
      return groups[geom.group_index].members.front().id;
    };

    // Pass 3: for each main object, test only candidates within its
    // reach-derived neighborhood.
    size_t pairs_tested = 0;
    for (size_t i = 0; i < geoms.size(); ++i) {
      const MainObjectGeom& a = geoms[i];
      const int ring = std::max(
          1, static_cast<int>(std::ceil((a.reach_m + global_max_reach_m) / cell_size_m)));
      for (int dz = -ring; dz <= ring; ++dz) {
        for (int dy = -ring; dy <= ring; ++dy) {
          for (int dx = -ring; dx <= ring; ++dx) {
            const auto neighbor_it =
                grid.find(GridCell{a.cell.x + dx, a.cell.y + dy, a.cell.z + dz});
            if (neighbor_it == grid.end()) {
              continue;
            }
            for (const size_t j : neighbor_it->second) {
              const MainObjectGeom& b = geoms[j];
              if (geom_key(b) <= geom_key(a)) {
                continue;  // lower-key side already tests this pair
              }
              ++pairs_tested;

              const auto broad = aabbContact(a.agg_center, a.agg_size, b.agg_center, b.agg_size,
                                             object_contact_tolerance_m_);
              if (!broad.touching) {
                continue;  // aggregates don't even come close; skip the narrow phase
              }

              // Narrow phase: once the aggregates are confirmed touching,
              // every node on the side with FEWER real (bboxed) members
              // connects to its single closest counterpart on the other
              // side, by plain centroid distance — no bbox-intersection
              // test, no per-pair distance cutoff, and matching runs from
              // the smaller side only. Running both directions (an earlier
              // version of this code did) forces every node on the LARGER
              // side to match whenever the smaller side has very few nodes
              // — e.g. a 1-node wall next to a 5-node floor produced 5
              // wall-floor edges (every floor node "finds" the wall, since
              // it's the only candidate on that side), even though only the
              // one floor node actually nearest the wall is a sensible
              // contact point. Iterating only the smaller side's nodes caps
              // the edge count at the smaller side's own node count and
              // leaves every non-matched larger-side node free, matching
              // what a real, sparse point of contact should look like. A
              // larger-side node can still receive more than one edge if
              // it's independently the closest match for more than one
              // smaller-side node — that's a genuine multi-point contact,
              // not the bug above. On a tie in member count, group_a is
              // used (it's already the side with the smaller minimum
              // NodeId, by the canonical ordering established above).
              const MainObjectGroup& group_a = groups[a.group_index];
              const MainObjectGroup& group_b = groups[b.group_index];

              const auto count_bboxed = [](const std::vector<SegmentMember>& members) {
                return std::count_if(members.begin(), members.end(),
                                     [](const SegmentMember& m) { return m.has_bbox; });
              };
              const bool a_is_smaller = count_bboxed(group_a.members) <= count_bboxed(group_b.members);
              const auto& smaller_members = a_is_smaller ? group_a.members : group_b.members;
              const auto& larger_members = a_is_smaller ? group_b.members : group_a.members;

              struct ClosestPair {
                const SegmentMember* member_a = nullptr;
                const SegmentMember* member_b = nullptr;
              };
              std::vector<ClosestPair> selected;
              selected.reserve(smaller_members.size());
              for (const auto& small_member : smaller_members) {
                if (!small_member.has_bbox) {
                  continue;
                }
                const SegmentMember* best = nullptr;
                double best_dist = std::numeric_limits<double>::max();
                for (const auto& large_member : larger_members) {
                  if (!large_member.has_bbox) {
                    continue;
                  }
                  const double dist = (small_member.bbox_center - large_member.bbox_center).norm();
                  if (dist < best_dist) {
                    best_dist = dist;
                    best = &large_member;
                  }
                }
                if (best == nullptr) {
                  continue;
                }
                // member_a always comes from group_a and member_b always
                // from group_b, regardless of which side turned out to be
                // "smaller", so downstream source/target assignment and
                // metadata stay consistent with the broad phase's canonical
                // group ordering.
                if (a_is_smaller) {
                  selected.push_back(ClosestPair{&small_member, best});
                } else {
                  selected.push_back(ClosestPair{best, &small_member});
                }
              }

              // Each selected (nearest-neighbor) pair still gets the full
              // AabbContact computed once, purely to populate the edge's
              // display/diagnostic metadata (gap, IoU, overlap volume,
              // contact axis) — geometry no longer decides which pairs are
              // selected, only how the selected pairs are described.
              for (const auto& pc : selected) {
                const SegmentMember* member_a = pc.member_a;
                const SegmentMember* member_b = pc.member_b;
                const auto contact = aabbContact(member_a->bbox_center, member_a->bbox_size,
                                                 member_b->bbox_center, member_b->bbox_size,
                                                 object_contact_tolerance_m_);
                const NodeId source = member_a->id;
                const NodeId target = member_b->id;
                // Volumetric 3D IoU is a poor summary for most contacts (e.g.
                // a small object against a large flat surface); the larger of
                // the two 2D planar IoUs reads better on the label.
                const double iou_2d_max = std::max(contact.iou_xz, contact.iou_yz);
                projection.edges.push_back(DisplayEdge{source, target,
                                                       DisplayEdgeType::kDerivedObjectContact,
                                                       EdgeOrigin::kNativeHydraEdge, 0,
                                                       iou_2d_max, contact.centroid_distance_m});

                Json meta = {
                    {"schema", "rsg_object_contact_v1"},
                    {"relation", "bbox_contact"},
                    {"centroid_distance_m", contact.centroid_distance_m},
                    {"bbox_gap_m", contact.gap_m},
                    {"bbox_overlap_volume_m3", contact.overlap_volume_m3},
                    {"bbox_overlap_xy_m2", contact.overlap_xy_m2},
                    {"bbox_iou_3d", contact.iou_3d},
                    {"bbox_iou_xz", contact.iou_xz},
                    {"bbox_iou_yz", contact.iou_yz},
                    {"bbox_iou_2d_max", iou_2d_max},
                    {"contact_axis", contactAxisName(contact.contact_axis)},
                    {"source_slot_id", member_a->semantic_slot},
                    {"target_slot_id", member_b->semantic_slot},
                    {"source_internal_object_id", group_a.internal_object_id},
                    {"target_internal_object_id", group_b.internal_object_id},
                    {"source_group_size", group_a.members.size()},
                    {"target_group_size", group_b.members.size()},
                    {"updated_timestamp_sec", now_sec},
                };

                if (object_contact_write_dsg_edges_) {
                  auto attrs = std::make_unique<spark_dsg::EdgeAttributes>(
                      clampValue(contact.iou_3d, 0.0, 1.0));
                  attrs->metadata.set(meta);
                  if (graph_->addOrUpdateEdge(source, target, std::move(attrs))) {
                    contact_edges_prev_.emplace(source, target);
                  }
                }

                Json record = meta;
                record["source"] = idString(source);
                record["target"] = idString(target);
                last_object_relations_.push_back(std::move(record));
                ++last_object_contact_edge_count_;
                last_object_contact_max_iou_ = std::max(last_object_contact_max_iou_, contact.iou_3d);
              }
            }
          }
        }
      }
    }
    last_object_contact_pairs_tested_ = pairs_tested;
  }

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
                                 LayeredProjection& projection) {
    last_object_segment_relations_ = Json::array();
    last_object_segment_edge_count_ = 0;
    if (!object_segment_edges_enabled_ || !graph_) {
      return;
    }

    const double now_sec = currentReferenceTimeSec();

    for (const auto& group : groups) {
      if (group.internal_object_id.empty() || group.members.size() < 2) {
        continue;
      }
      if (object_segment_max_group_size_ > 0 &&
          group.members.size() > object_segment_max_group_size_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                             "object segment: internal object '%s' has %zu segments, exceeds "
                             "object_segment_max_group_size=%zu; skipping",
                             group.internal_object_id.c_str(), group.members.size(),
                             object_segment_max_group_size_);
        continue;
      }

      // members is already sorted by NodeId ascending (buildMainObjectGroups).
      const auto& members = group.members;
      for (size_t i = 0; i + 1 < members.size(); ++i) {
        const NodeId source = members[i].id;
        const NodeId target = members[i + 1].id;
        projection.edges.push_back(DisplayEdge{source, target,
                                               DisplayEdgeType::kDerivedObjectSegment,
                                               EdgeOrigin::kNativeHydraEdge, 0});

        Json meta = {
            {"schema", "rsg_object_segment_v1"},
            {"relation", "same_physical_object"},
            {"internal_object_id", group.internal_object_id},
            {"source_slot_id", members[i].semantic_slot},
            {"target_slot_id", members[i + 1].semantic_slot},
            {"group_size", members.size()},
            {"updated_timestamp_sec", now_sec},
        };

        if (object_segment_write_dsg_edges_) {
          auto attrs = std::make_unique<spark_dsg::EdgeAttributes>(1.0);
          attrs->metadata.set(meta);
          if (graph_->addOrUpdateEdge(source, target, std::move(attrs))) {
            segment_edges_prev_.emplace(source, target);
          }
        }

        Json record = meta;
        record["source"] = idString(source);
        record["target"] = idString(target);
        last_object_segment_relations_.push_back(std::move(record));
        ++last_object_segment_edge_count_;
      }
    }
  }

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
                               const std::unordered_map<NodeId, ResolvedOverlay>& resolved) {
    if (!object_contact_diagnostics_enabled_ || object_contact_diagnostics_stream_failed_) {
      return;
    }
    if (!object_contact_diagnostics_stream_.is_open()) {
      std::string path = object_contact_diagnostics_path_;
      if (path.empty()) {
        // One fresh timestamped folder per node lifetime (the stream stays
        // open for the rest of the run once created), under this
        // experiment's results/ directory, so successive debugging runs
        // never clobber each other's logs.
        const std::time_t now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm_buf{};
        localtime_r(&now_time, &tm_buf);
        std::ostringstream stamp;
        stamp << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
        const std::filesystem::path run_dir =
            std::filesystem::path(
                "/home/student/Thesis_new/debug/fuser_object_relation_experiment/results") /
            ("run_" + stamp.str());
        std::error_code mkdir_error;
        std::filesystem::create_directories(run_dir, mkdir_error);
        if (mkdir_error) {
          RCLCPP_WARN(get_logger(), "object contact diagnostics: failed to create '%s' (%s); disabling",
                     run_dir.c_str(), mkdir_error.message().c_str());
          object_contact_diagnostics_stream_failed_ = true;
          return;
        }
        path = (run_dir / "contact_diagnostics.jsonl").string();
      }
      object_contact_diagnostics_stream_.open(path, std::ios::out | std::ios::app);
      if (!object_contact_diagnostics_stream_.is_open()) {
        RCLCPP_WARN(get_logger(), "object contact diagnostics: failed to open '%s'; disabling",
                   path.c_str());
        object_contact_diagnostics_stream_failed_ = true;
        return;
      }
      RCLCPP_INFO(get_logger(), "object contact diagnostics: writing to '%s'", path.c_str());
    }

    Json main_objects = Json::array();
    for (const auto& group : groups) {
      bool any_bbox = false;
      Eigen::Vector3d agg_min = Eigen::Vector3d::Zero();
      Eigen::Vector3d agg_max = Eigen::Vector3d::Zero();
      std::string group_label;  // first member with a non-empty resolved label
      Json members = Json::array();
      for (const auto& member : group.members) {
        const auto overlay_it = resolved.find(member.id);
        const std::string label =
            overlay_it != resolved.end() ? overlay_it->second.overlay.label : std::string();
        if (group_label.empty() && !label.empty()) {
          group_label = label;
        }
        Json member_json = {
            {"node_id", idString(member.id)},
            {"semantic_slot_id", member.semantic_slot},
            {"has_bbox", member.has_bbox},
            {"label", label},
        };
        if (member.has_bbox) {
          member_json["bbox_center"] = {member.bbox_center.x(), member.bbox_center.y(),
                                        member.bbox_center.z()};
          member_json["bbox_size"] = {member.bbox_size.x(), member.bbox_size.y(),
                                      member.bbox_size.z()};
          const Eigen::Vector3d half = 0.5 * member.bbox_size.cwiseMax(0.0);
          const Eigen::Vector3d mn = member.bbox_center - half;
          const Eigen::Vector3d mx = member.bbox_center + half;
          if (!any_bbox) {
            agg_min = mn;
            agg_max = mx;
            any_bbox = true;
          } else {
            agg_min = agg_min.cwiseMin(mn);
            agg_max = agg_max.cwiseMax(mx);
          }
        }
        members.push_back(std::move(member_json));
      }
      Json group_json = {
          {"internal_object_id", group.internal_object_id},
          {"member_count", group.members.size()},
          {"label", group_label},  // representative label for the whole physical object
          {"members", std::move(members)},
      };
      if (any_bbox) {
        const Eigen::Vector3d agg_center = 0.5 * (agg_min + agg_max);
        const Eigen::Vector3d agg_size = agg_max - agg_min;
        group_json["aggregate_bbox_center"] = {agg_center.x(), agg_center.y(), agg_center.z()};
        group_json["aggregate_bbox_size"] = {agg_size.x(), agg_size.y(), agg_size.z()};
      }
      main_objects.push_back(std::move(group_json));
    }

    // params included on every line so each line is independently
    // analyzable without needing to correlate against a separate header.
    const Json record = {
        {"schema", "rsg_object_contact_diagnostics_v1"},
        {"sequence", object_contact_diagnostics_sequence_},
        {"stamp_sec", currentReferenceTimeSec()},
        {"params",
         {
             {"object_contact_tolerance_m", object_contact_tolerance_m_},
             {"object_contact_min_iou_3d", object_contact_min_iou_3d_},
         }},
        {"main_objects", main_objects},
        {"contact_edges", last_object_relations_},
        {"segment_edges", last_object_segment_relations_},
    };
    object_contact_diagnostics_stream_ << record.dump() << "\n";
    object_contact_diagnostics_stream_.flush();
    ++object_contact_diagnostics_sequence_;
  }

  void updateLocalGraphMetadata(const SceneModel& model,
                                const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                                const PresenceCache& presence) {
    if (!graph_) {
      return;
    }
    for (const auto& [node_id, node] : model.nodes) {
      if (node.kind != LayerKind::kObjects) {
        continue;
      }
      auto& attrs = graph_->getNode(node_id).attributes<spark_dsg::NodeAttributes>();
      Json metadata = attrs.metadata.get();
      if (!metadata.is_object()) {
        metadata = Json::object();
      }
      metadata.erase("rsg_rap");
      metadata.erase("rsg_presence");
      metadata.erase("rsg_identity");
      const auto presence_it = presence.find(node.semantic_slot);
      if (presence_it != presence.end()) {
        const auto& obs = presence_it->second;
        metadata["rsg_presence"] = {
            {"slot_id", obs.slot_id},
            {"last_observed_timestamp_sec", obs.last_observed_timestamp_sec},
            {"internal_object_id", obs.internal_object_id},
            {"persistent_track_id", obs.persistent_track_id},
            {"local_segment_id", obs.local_segment_id},
            {"local_segment_xy_span_m", obs.local_segment_xy_span_m},
        };
        metadata["rsg_identity"] = {
            {"schema", "rsg_local_segment_identity_v1"},
            {"semantic_slot_id", obs.slot_id},
            {"hydra_slot_id", obs.slot_id},
            {"internal_object_id", obs.internal_object_id},
            {"persistent_track_id", obs.persistent_track_id},
            {"local_segment_id", obs.local_segment_id},
        };
      }
      const auto it = resolved.find(node_id);
      if (it != resolved.end()) {
        const auto& label = it->second;
        metadata["rsg_rap"] = {
            {"slot_id", label.overlay.slot_id},
            {"label", label.overlay.label},
            {"confidence", label.overlay.confidence},
            {"label_confidence", label.overlay.confidence},
            {"mobility_class", label.overlay.mobility_class},
            {"mobility_confidence", label.overlay.mobility_confidence},
            {"mobility_source", label.overlay.mobility_source},
            {"source", label.overlay.source},
            {"timestamp_sec", label.overlay.timestamp_sec},
            {"association", label.association},
            {"centroid_distance_m", label.centroid_distance_m},
        };
      }
      attrs.metadata.set(metadata);
    }
  }

  Eigen::Vector3d sourcePosition(const NodeView& node) const {
    if (node.has_bbox) {
      return node.bbox_center.cast<double>();
    }
    return node.position;
  }

  Eigen::Vector3d displayPosition(const NodeView& node) const {
    Eigen::Vector3d output = sourcePosition(node);
    switch (node.kind) {
      case LayerKind::kObjects:
        output.z() += object_z_offset_m_;
        break;
      case LayerKind::kPlaces:
        output.z() += place_z_offset_m_;
        break;
      case LayerKind::kRooms:
        output.z() += room_z_offset_m_;
        break;
      default:
        break;
    }
    return output;
  }

  GridKey gridKey(const Eigen::Vector3d& position, double cell_size_m) const {
    return GridKey{
        static_cast<int>(std::floor(position.x() / cell_size_m)),
        static_cast<int>(std::floor(position.y() / cell_size_m)),
        static_cast<int>(std::floor(position.z() / cell_size_m)),
    };
  }

  uint64_t hashCombine(uint64_t seed, uint64_t value) const {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed;
  }

  uint64_t placeFingerprint(const SceneModel& model, const std::vector<NodeId>& place_ids) const {
    uint64_t fingerprint = fnv1a("rsg_place_spatial_index");
    for (const NodeId place_id : place_ids) {
      const auto found = model.nodes.find(place_id);
      if (found == model.nodes.end()) {
        continue;
      }
      const Eigen::Vector3d position = sourcePosition(found->second);
      fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(place_id));
      // Quantising to centimetres avoids needless index rebuilds from floating
      // point noise while still invalidating after a meaningful place motion.
      fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(std::llround(position.x() * 100.0)));
      fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(std::llround(position.y() * 100.0)));
      fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(std::llround(position.z() * 100.0)));
    }
    return fingerprint;
  }

  void ensurePlaceSpatialIndex(const SceneModel& model, const std::vector<NodeId>& place_ids) {
    const uint64_t fingerprint = placeFingerprint(model, place_ids);
    if (place_spatial_index_.fingerprint == fingerprint &&
        std::abs(place_spatial_index_.cell_size_m - object_place_index_voxel_size_m_) < 1e-9) {
      return;
    }

    place_spatial_index_.cell_size_m = object_place_index_voxel_size_m_;
    place_spatial_index_.fingerprint = fingerprint;
    place_spatial_index_.cells.clear();
    for (const NodeId place_id : place_ids) {
      const auto found = model.nodes.find(place_id);
      if (found == model.nodes.end()) {
        continue;
      }
      place_spatial_index_.cells[gridKey(sourcePosition(found->second), place_spatial_index_.cell_size_m)]
          .push_back(place_id);
    }
    for (auto& [cell, members] : place_spatial_index_.cells) {
      (void)cell;
      std::sort(members.begin(), members.end());
    }
    ++place_spatial_index_.revision;
    object_place_cache_.clear();
  }

  uint64_t meshFingerprint(const spark_dsg::Mesh& mesh) const {
    uint64_t fingerprint = fnv1a("rsg_mesh_wall_index");
    fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(mesh.points.size()));
    fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(mesh.faces.size()));
    if (!mesh.points.empty()) {
      const size_t samples = std::min<size_t>(mesh.points.size(), 16U);
      const size_t stride = std::max<size_t>(1U, mesh.points.size() / samples);
      for (size_t index = 0; index < mesh.points.size(); index += stride) {
        const auto& point = mesh.points[index];
        fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(std::llround(point.x() * 100.0F)));
        fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(std::llround(point.y() * 100.0F)));
        fingerprint = hashCombine(fingerprint, static_cast<uint64_t>(std::llround(point.z() * 100.0F)));
      }
    }
    return fingerprint;
  }

  void indexTriangle(size_t triangle_index, const MeshTriangle& triangle) {
    const GridKey lower = gridKey(triangle.min_corner, mesh_wall_index_.cell_size_m);
    const GridKey upper = gridKey(triangle.max_corner, mesh_wall_index_.cell_size_m);
    const int64_t dx = static_cast<int64_t>(upper.x) - static_cast<int64_t>(lower.x) + 1;
    const int64_t dy = static_cast<int64_t>(upper.y) - static_cast<int64_t>(lower.y) + 1;
    const int64_t dz = static_cast<int64_t>(upper.z) - static_cast<int64_t>(lower.z) + 1;
    const int64_t cell_count = dx * dy * dz;
    if (cell_count <= 0 || cell_count > object_place_mesh_max_cells_per_triangle_) {
      mesh_wall_index_.overflow_triangles.push_back(triangle_index);
      return;
    }
    for (int x = lower.x; x <= upper.x; ++x) {
      for (int y = lower.y; y <= upper.y; ++y) {
        for (int z = lower.z; z <= upper.z; ++z) {
          mesh_wall_index_.cells[GridKey{x, y, z}].push_back(triangle_index);
        }
      }
    }
  }

  bool ensureMeshWallIndex() {
    const bool mesh_validation_required =
        object_place_require_mesh_validation_ ||
        (room_place_completion_enabled_ && room_place_completion_require_mesh_validation_);
    if (!mesh_validation_required) {
      return true;
    }
    if (!graph_ || !graph_->hasMesh()) {
      mesh_wall_index_.available = false;
      return false;
    }
    const auto mesh = graph_->mesh();
    if (!mesh || mesh->points.empty() || mesh->faces.empty()) {
      mesh_wall_index_.available = false;
      return false;
    }

    const uint64_t fingerprint = meshFingerprint(*mesh);
    if (mesh_wall_index_.available && mesh_wall_index_.fingerprint == fingerprint &&
        std::abs(mesh_wall_index_.cell_size_m - object_place_mesh_voxel_size_m_) < 1e-9) {
      return true;
    }

    mesh_wall_index_.cell_size_m = object_place_mesh_voxel_size_m_;
    mesh_wall_index_.fingerprint = fingerprint;
    mesh_wall_index_.available = false;
    mesh_wall_index_.triangles.clear();
    mesh_wall_index_.overflow_triangles.clear();
    mesh_wall_index_.cells.clear();
    mesh_wall_index_.triangles.reserve(mesh->faces.size());

    for (const auto& face : mesh->faces) {
      if (face[0] >= mesh->points.size() || face[1] >= mesh->points.size() ||
          face[2] >= mesh->points.size()) {
        continue;
      }
      MeshTriangle triangle;
      triangle.first = mesh->points[face[0]].cast<double>();
      triangle.second = mesh->points[face[1]].cast<double>();
      triangle.third = mesh->points[face[2]].cast<double>();
      triangle.min_corner = triangle.first.cwiseMin(triangle.second).cwiseMin(triangle.third);
      triangle.max_corner = triangle.first.cwiseMax(triangle.second).cwiseMax(triangle.third);
      const size_t index = mesh_wall_index_.triangles.size();
      mesh_wall_index_.triangles.push_back(triangle);
      indexTriangle(index, triangle);
    }

    mesh_wall_index_.available = !mesh_wall_index_.triangles.empty();
    if (mesh_wall_index_.available) {
      ++mesh_wall_index_.revision;
      object_place_cache_.clear();
    }
    return mesh_wall_index_.available;
  }

  bool segmentIntersectsTriangle(const Eigen::Vector3d& start,
                                 const Eigen::Vector3d& end,
                                 const MeshTriangle& triangle) const {
    constexpr double kEpsilon = 1e-8;
    const Eigen::Vector3d direction = end - start;
    const Eigen::Vector3d edge_first = triangle.second - triangle.first;
    const Eigen::Vector3d edge_second = triangle.third - triangle.first;
    const Eigen::Vector3d cross = direction.cross(edge_second);
    const double determinant = edge_first.dot(cross);
    if (std::abs(determinant) < kEpsilon) {
      return false;
    }
    const double inverse_determinant = 1.0 / determinant;
    const Eigen::Vector3d relative = start - triangle.first;
    const double u = relative.dot(cross) * inverse_determinant;
    if (u < -kEpsilon || u > 1.0 + kEpsilon) {
      return false;
    }
    const Eigen::Vector3d q = relative.cross(edge_first);
    const double v = direction.dot(q) * inverse_determinant;
    if (v < -kEpsilon || u + v > 1.0 + kEpsilon) {
      return false;
    }
    const double t = edge_second.dot(q) * inverse_determinant;
    // Ignore a tiny endpoint neighbourhood. The object anchor is deliberately
    // offset from its bounding box; intersections in the interior represent a
    // reconstructed wall, object, floor, or other occupied mesh surface.
    return t > 1e-4 && t < (1.0 - 1e-4);
  }

  bool meshSegmentIsClear(const Eigen::Vector3d& start, const Eigen::Vector3d& end) const {
    if (!mesh_wall_index_.available) {
      return false;
    }
    Eigen::Vector3d min_corner = start.cwiseMin(end);
    Eigen::Vector3d max_corner = start.cwiseMax(end);
    constexpr double kPadding = 1e-4;
    min_corner.array() -= kPadding;
    max_corner.array() += kPadding;
    const GridKey lower = gridKey(min_corner, mesh_wall_index_.cell_size_m);
    const GridKey upper = gridKey(max_corner, mesh_wall_index_.cell_size_m);

    std::unordered_set<size_t> candidates;
    for (int x = lower.x; x <= upper.x; ++x) {
      for (int y = lower.y; y <= upper.y; ++y) {
        for (int z = lower.z; z <= upper.z; ++z) {
          const auto found = mesh_wall_index_.cells.find(GridKey{x, y, z});
          if (found == mesh_wall_index_.cells.end()) {
            continue;
          }
          candidates.insert(found->second.begin(), found->second.end());
          if (candidates.size() > static_cast<size_t>(object_place_mesh_max_triangle_tests_)) {
            return false;  // fail closed rather than introducing an unbounded query.
          }
        }
      }
    }
    candidates.insert(mesh_wall_index_.overflow_triangles.begin(), mesh_wall_index_.overflow_triangles.end());
    if (candidates.size() > static_cast<size_t>(object_place_mesh_max_triangle_tests_)) {
      return false;
    }
    for (const size_t triangle_index : candidates) {
      if (triangle_index >= mesh_wall_index_.triangles.size()) {
        continue;
      }
      if (segmentIntersectsTriangle(start, end, mesh_wall_index_.triangles[triangle_index])) {
        return false;
      }
    }
    return true;
  }

  Eigen::Vector3d objectFreeSpaceAnchor(const NodeView& object,
                                        const NodeView& place) const {
    const Eigen::Vector3d center = sourcePosition(object);
    const Eigen::Vector3d delta = sourcePosition(place) - center;
    const double distance = delta.norm();
    if (distance < 1e-6) {
      return center;
    }
    const Eigen::Vector3d direction = delta / distance;
    double support_distance = 0.0;
    if (object.has_bbox) {
      const Eigen::Vector3d half_extent = 0.5 * object.bbox_size.cast<double>();
      support_distance = std::abs(direction.x()) * half_extent.x() +
                         std::abs(direction.y()) * half_extent.y() +
                         std::abs(direction.z()) * half_extent.z();
    }
    return center + direction * (support_distance + object_place_anchor_outset_m_);
  }

  std::vector<std::pair<NodeId, double>> localPlaceCandidates(
      const SceneModel& model, const NodeView& object) const {
    std::unordered_set<NodeId> ids;
    const GridKey center = gridKey(sourcePosition(object), place_spatial_index_.cell_size_m);
    for (int radius = 0; radius <= object_place_index_search_radius_cells_; ++radius) {
      for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
          for (int dz = -radius; dz <= radius; ++dz) {
            if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != radius) {
              continue;
            }
            const auto found = place_spatial_index_.cells.find(
                GridKey{center.x + dx, center.y + dy, center.z + dz});
            if (found != place_spatial_index_.cells.end()) {
              ids.insert(found->second.begin(), found->second.end());
            }
          }
        }
      }
    }

    std::vector<std::pair<NodeId, double>> candidates;
    candidates.reserve(ids.size());
    for (const NodeId place_id : ids) {
      const auto found = model.nodes.find(place_id);
      if (found == model.nodes.end()) {
        continue;
      }
      const double distance = (sourcePosition(object) - sourcePosition(found->second)).norm();
      if (distance <= object_place_max_distance_m_) {
        candidates.emplace_back(place_id, distance);
      }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      if (std::abs(lhs.second - rhs.second) > 1e-9) {
        return lhs.second < rhs.second;
      }
      return lhs.first < rhs.first;
    });
    if (candidates.size() > static_cast<size_t>(object_place_max_candidates_)) {
      candidates.resize(static_cast<size_t>(object_place_max_candidates_));
    }
    return candidates;
  }

  bool cachedAssociationUsable(const SceneModel& model, const NodeView& object,
                               NodeId& place_id) {
    const auto found = object_place_cache_.find(object.id);
    if (found == object_place_cache_.end()) {
      return false;
    }
    const auto& cache = found->second;
    if (cache.place_index_revision != place_spatial_index_.revision ||
        cache.mesh_index_revision != mesh_wall_index_.revision ||
        !model.nodes.count(cache.place_id)) {
      return false;
    }
    if ((sourcePosition(object) - cache.object_position).norm() >
        object_place_cache_recompute_translation_m_) {
      return false;
    }
    place_id = cache.place_id;
    return true;
  }

  std::optional<NodeId> findValidatedLocalPlace(const SceneModel& model,
                                                 const NodeView& object,
                                                 LayeredProjection& projection) {
    NodeId cached_place = 0;
    if (cachedAssociationUsable(model, object, cached_place)) {
      ++projection.local_association_cache_hits;
      return cached_place;
    }

    const auto candidates = localPlaceCandidates(model, object);
    projection.local_index_candidates_examined += candidates.size();
    for (const auto& [place_id, distance] : candidates) {
      (void)distance;
      const auto place_it = model.nodes.find(place_id);
      if (place_it == model.nodes.end()) {
        continue;
      }
      const Eigen::Vector3d anchor = objectFreeSpaceAnchor(object, place_it->second);
      if (object_place_require_mesh_validation_ &&
          !meshSegmentIsClear(anchor, sourcePosition(place_it->second))) {
        ++projection.mesh_rejected_candidates;
        continue;
      }
      object_place_cache_[object.id] = CachedObjectPlaceAssociation{
          place_id, sourcePosition(object), place_spatial_index_.revision, mesh_wall_index_.revision};
      ++projection.validated_local_place_associations;
      return place_id;
    }
    return std::nullopt;
  }

  static int64_t stampNanoseconds(const builtin_interfaces::msg::Time& stamp) {
    constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
    return static_cast<int64_t>(stamp.sec) * kNanosecondsPerSecond +
           static_cast<int64_t>(stamp.nanosec);
  }

  void updateRoomPlaceCompletionGraceClock(const std_msgs::msg::Header& header) {
    const auto now = std::chrono::steady_clock::now();
    const int64_t source_stamp_ns = stampNanoseconds(header.stamp);

    if (!room_place_completion_grace_clock_started_) {
      room_place_completion_grace_clock_started_ = true;
      room_place_completion_grace_wall_start_ = now;
      if (source_stamp_ns > 0) {
        room_place_completion_source_start_ns_ = source_stamp_ns;
        room_place_completion_last_source_stamp_ns_ = source_stamp_ns;
      }
      return;
    }

    // A backwards jump occurs when a rosbag is replayed from the beginning or
    // the upstream Hydra clock restarts. Treat it as a fresh map and apply the
    // same settling interval again. Ordinary full DSG messages do not reset it.
    constexpr int64_t kClockRewindToleranceNs = 1000000000LL;
    if (source_stamp_ns > 0 &&
        room_place_completion_last_source_stamp_ns_ > 0 &&
        source_stamp_ns + kClockRewindToleranceNs <
            room_place_completion_last_source_stamp_ns_) {
      room_place_completion_grace_wall_start_ = now;
      room_place_completion_source_start_ns_ = source_stamp_ns;
    } else if (source_stamp_ns > 0 && room_place_completion_source_start_ns_ == 0) {
      // If the first message was stamped exactly zero, begin source-time
      // accounting as soon as the clock advances.
      room_place_completion_source_start_ns_ = source_stamp_ns;
      room_place_completion_grace_wall_start_ = now;
    }

    if (source_stamp_ns > 0) {
      room_place_completion_last_source_stamp_ns_ = source_stamp_ns;
    }
  }

  double roomPlaceCompletionGraceElapsedSec() const {
    if (!room_place_completion_grace_clock_started_) {
      return 0.0;
    }

    const int64_t latest_source_stamp_ns = stampNanoseconds(latest_header_.stamp);
    if (room_place_completion_source_start_ns_ > 0 &&
        latest_source_stamp_ns >= room_place_completion_source_start_ns_) {
      return static_cast<double>(latest_source_stamp_ns -
                                 room_place_completion_source_start_ns_) /
             1.0e9;
    }

    return std::max(0.0, std::chrono::duration<double>(
        std::chrono::steady_clock::now() - room_place_completion_grace_wall_start_).count());
  }

  bool roomPlaceCompletionReady(LayeredProjection& projection) const {
    const double elapsed = roomPlaceCompletionGraceElapsedSec();
    projection.room_completion_grace_elapsed_sec = elapsed;
    projection.room_completion_grace_remaining_sec = std::max(
        0.0, room_place_completion_grace_period_sec_ - elapsed);
    projection.room_completion_waiting_for_grace =
        room_place_completion_grace_period_sec_ > 0.0 &&
        elapsed < room_place_completion_grace_period_sec_;
    return !projection.room_completion_waiting_for_grace;
  }

  std::optional<NodeId> majorityVisibleNeighbourRoom(
      const SceneModel& model,
      const NodeView& orphan_place,
      const std::unordered_map<NodeId, NodeId>& assigned_places,
      LayeredProjection& projection) const {
    if (room_place_completion_require_mesh_validation_ && !mesh_wall_index_.available) {
      return std::nullopt;
    }

    std::vector<std::pair<NodeId, double>> candidates;
    candidates.reserve(assigned_places.size());
    const Eigen::Vector3d orphan_position = sourcePosition(orphan_place);
    for (const auto& [candidate_id, room_id] : assigned_places) {
      (void)room_id;
      if (candidate_id == orphan_place.id) {
        continue;
      }
      const auto candidate_it = model.nodes.find(candidate_id);
      if (candidate_it == model.nodes.end() || !isPlace(candidate_it->second)) {
        continue;
      }
      const Eigen::Vector3d candidate_position = sourcePosition(candidate_it->second);
      if (std::abs(candidate_position.z() - orphan_position.z()) >
          room_place_completion_max_height_difference_m_) {
        continue;
      }
      const double distance = (candidate_position - orphan_position).norm();
      // A zero or negative cap means "use the actual nearest neighbours" with
      // no arbitrary range cutoff. Keep the optional cap for unusually large
      // graphs where the user wants to bound the local search.
      if (room_place_completion_max_distance_m_ <= 0.0 ||
          distance <= room_place_completion_max_distance_m_) {
        candidates.emplace_back(candidate_id, distance);
      }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
      if (std::abs(lhs.second - rhs.second) > 1e-9) {
        return lhs.second < rhs.second;
      }
      return lhs.first < rhs.first;
    });

    std::unordered_map<NodeId, size_t> room_votes;
    size_t visible_neighbours = 0;
    for (const auto& [candidate_id, distance] : candidates) {
      (void)distance;
      const auto candidate_it = model.nodes.find(candidate_id);
      const auto room_it = assigned_places.find(candidate_id);
      if (candidate_it == model.nodes.end() || room_it == assigned_places.end()) {
        continue;
      }
      ++projection.room_completion_candidates_examined;
      if (room_place_completion_require_mesh_validation_ &&
          !meshSegmentIsClear(orphan_position, sourcePosition(candidate_it->second))) {
        ++projection.room_completion_mesh_rejected_candidates;
        continue;
      }
      ++room_votes[room_it->second];
      ++visible_neighbours;
      if (visible_neighbours >= static_cast<size_t>(room_place_completion_neighbours_)) {
        break;
      }
    }

    NodeId winning_room = 0;
    size_t winning_votes = 0;
    bool tie = false;
    for (const auto& [room_id, votes] : room_votes) {
      if (votes > winning_votes) {
        winning_room = room_id;
        winning_votes = votes;
        tie = false;
      } else if (votes == winning_votes) {
        tie = true;
      }
    }
    if (winning_votes < static_cast<size_t>(room_place_completion_min_majority_votes_)) {
      return std::nullopt;
    }
    if (tie) {
      ++projection.room_completion_ties;
      return std::nullopt;
    }
    return winning_room;
  }

  void completeMissingPlaceRoomMembership(
      const SceneModel& model,
      const std::vector<NodeId>& place_ids,
      std::set<std::pair<NodeId, NodeId>>& room_place_pairs,
      LayeredProjection& projection) const {
    if (!room_place_completion_enabled_ || place_ids.empty()) {
      return;
    }
    if (!roomPlaceCompletionReady(projection)) {
      for (const NodeId place_id : place_ids) {
        projection.room_completion_suppressed_by_grace +=
            projection.place_to_room.count(place_id) == 0U ? 1U : 0U;
      }
      return;
    }
    if (room_place_completion_require_mesh_validation_ && !mesh_wall_index_.available) {
      for (const NodeId place_id : place_ids) {
        projection.room_completion_suppressed_without_mesh +=
            projection.place_to_room.count(place_id) == 0U ? 1U : 0U;
      }
      return;
    }

    // New links are applied only after one complete pass. This makes the
    // outcome independent of the unordered-map iteration order, while later
    // passes let a room propagate through neighbouring visible places.
    for (size_t pass = 0; pass < place_ids.size(); ++pass) {
      const auto assignments_at_pass_start = projection.place_to_room;
      std::vector<std::pair<NodeId, NodeId>> inferred;
      for (const NodeId place_id : place_ids) {
        if (assignments_at_pass_start.count(place_id) != 0U) {
          continue;
        }
        const auto place_it = model.nodes.find(place_id);
        if (place_it == model.nodes.end()) {
          continue;
        }
        const auto room_id = majorityVisibleNeighbourRoom(
            model, place_it->second, assignments_at_pass_start, projection);
        if (room_id) {
          inferred.emplace_back(place_id, *room_id);
        }
      }
      if (inferred.empty()) {
        break;
      }
      for (const auto& [place_id, room_id] : inferred) {
        projection.place_to_room[place_id] = room_id;
        projection.place_room_origin[place_id] = EdgeOrigin::kDerivedVisibleNeighbourRoom;
        room_place_pairs.emplace(room_id, place_id);
        ++projection.derived_room_place_associations;
      }
    }
  }

  LayeredProjection buildLayeredProjection(const SceneModel& model) {
    LayeredProjection projection;
    std::set<std::pair<NodeId, NodeId>> room_place_pairs;
    std::set<std::pair<NodeId, NodeId>> object_place_pairs;
    std::set<std::pair<NodeId, NodeId>> place_place_pairs;
    std::set<std::pair<NodeId, NodeId>> object_object_pairs;
    std::set<std::pair<NodeId, NodeId>> room_room_pairs;
    std::vector<NodeId> object_ids;
    std::vector<NodeId> place_ids;

    for (const auto& [node_id, node] : model.nodes) {
      if (isObject(node)) {
        object_ids.push_back(node_id);
      } else if (isPlace(node)) {
        place_ids.push_back(node_id);
      }
    }
    std::sort(object_ids.begin(), object_ids.end());
    std::sort(place_ids.begin(), place_ids.end());

    // Reuse only direct native Hydra topology for room/place and object/place
    // membership. No broad graph traversal is used, so a room cannot be
    // inferred through an unrelated reachable place.
    for (const auto& edge : model.raw_edges) {
      const auto source_it = model.nodes.find(edge.source);
      const auto target_it = model.nodes.find(edge.target);
      if (source_it == model.nodes.end() || target_it == model.nodes.end() || edge.source == edge.target) {
        continue;
      }
      const auto& source = source_it->second;
      const auto& target = target_it->second;
      if (isRoom(source) && isPlace(target)) {
        room_place_pairs.emplace(source.id, target.id);
      } else if (isPlace(source) && isRoom(target)) {
        room_place_pairs.emplace(target.id, source.id);
      } else if (isObject(source) && isPlace(target)) {
        object_place_pairs.emplace(source.id, target.id);
      } else if (isPlace(source) && isObject(target)) {
        object_place_pairs.emplace(target.id, source.id);
      } else if (isPlace(source) && isPlace(target)) {
        place_place_pairs.emplace(std::min(source.id, target.id), std::max(source.id, target.id));
      } else if (isObject(source) && isObject(target)) {
        object_object_pairs.emplace(std::min(source.id, target.id), std::max(source.id, target.id));
      } else if (isRoom(source) && isRoom(target)) {
        room_room_pairs.emplace(std::min(source.id, target.id), std::max(source.id, target.id));
      }
    }

    // A place normally has one room parent. If a DSG contains more than one
    // direct room edge, retain every native hierarchy edge for display but use
    // the lowest stable node ID as its deterministic color parent.
    for (const auto& [room_id, place_id] : room_place_pairs) {
      const auto found = projection.place_to_room.find(place_id);
      if (found == projection.place_to_room.end() || room_id < found->second) {
        projection.place_to_room[place_id] = room_id;
        projection.place_room_origin[place_id] = EdgeOrigin::kNativeHydraEdge;
      }
    }

    // A direct object/place relation is authoritative. Again, retain one
    // deterministic membership parent if Hydra temporarily exposes duplicates.
    for (const auto& [object_id, place_id] : object_place_pairs) {
      const auto found = projection.object_to_place.find(object_id);
      if (found == projection.object_to_place.end() || place_id < found->second) {
        projection.object_to_place[object_id] = place_id;
        projection.object_place_origin[object_id] = EdgeOrigin::kNativeHydraEdge;
      }
    }

    // Fallback association is deliberately conservative. A local 3D hash index
    // shortlists only nearby places, then the direct line from an object-free
    // anchor to the candidate must be clear of the retained Hydra mesh. When
    // no mesh is available, no fallback edge is created; a missing relation is
    // safer and more truthful than an edge that could cross a wall.
    ensurePlaceSpatialIndex(model, place_ids);
    const bool mesh_validation_needed =
        object_place_require_mesh_validation_ ||
        (room_place_completion_enabled_ && room_place_completion_require_mesh_validation_);
    projection.mesh_validation_available = !mesh_validation_needed || ensureMeshWallIndex();

    completeMissingPlaceRoomMembership(model, place_ids, room_place_pairs, projection);

    if (object_place_use_local_validated_fallback_ && !place_ids.empty()) {
      for (const NodeId object_id : object_ids) {
        if (projection.object_to_place.count(object_id) != 0U) {
          continue;
        }
        const auto object_it = model.nodes.find(object_id);
        if (object_it == model.nodes.end()) {
          continue;
        }
        if (object_place_require_mesh_validation_ && !projection.mesh_validation_available) {
          ++projection.fallback_suppressed_without_mesh;
          continue;
        }
        const auto place_id = findValidatedLocalPlace(model, object_it->second, projection);
        if (place_id) {
          projection.object_to_place[object_id] = *place_id;
          projection.object_place_origin[object_id] = EdgeOrigin::kDerivedMeshValidatedLocalPlace;
        }
      }
    }

    if (!show_edges_) {
      return projection;
    }

    if (show_room_place_edges_ && show_rooms_ && show_places_) {
      for (const auto& [room_id, place_id] : room_place_pairs) {
        const auto origin_it = projection.place_room_origin.find(place_id);
        const EdgeOrigin origin = origin_it == projection.place_room_origin.end()
                                      ? EdgeOrigin::kNativeHydraEdge
                                      : origin_it->second;
        projection.edges.push_back(DisplayEdge{
            room_id, place_id, DisplayEdgeType::kRoomPlaceHierarchy, origin, room_id});
      }
    }
    if (show_place_object_edges_ && show_places_ && show_objects_) {
      std::vector<NodeId> associated_objects;
      associated_objects.reserve(projection.object_to_place.size());
      for (const auto& [object_id, place_id] : projection.object_to_place) {
        (void)place_id;
        associated_objects.push_back(object_id);
      }
      std::sort(associated_objects.begin(), associated_objects.end());
      for (const NodeId object_id : associated_objects) {
        const NodeId place_id = projection.object_to_place.at(object_id);
        const auto origin_it = projection.object_place_origin.find(object_id);
        const EdgeOrigin origin = origin_it == projection.object_place_origin.end()
                                      ? EdgeOrigin::kNativeHydraEdge
                                      : origin_it->second;
        const auto room_it = projection.place_to_room.find(place_id);
        const NodeId color_owner = room_it == projection.place_to_room.end() ? 0 : room_it->second;
        projection.edges.push_back(DisplayEdge{
            place_id, object_id, DisplayEdgeType::kPlaceObjectMembership, origin, color_owner});
      }
    }
    if (show_place_connectivity_edges_ && show_places_) {
      for (const auto& [source, target] : place_place_pairs) {
        projection.edges.push_back(DisplayEdge{
            source, target, DisplayEdgeType::kPlaceConnectivity,
            EdgeOrigin::kNativeHydraEdge, 0});
      }
    }
    if (show_native_object_object_edges_ && show_objects_) {
      for (const auto& [source, target] : object_object_pairs) {
        projection.edges.push_back(DisplayEdge{
            source, target, DisplayEdgeType::kNativeObjectObjectDebug,
            EdgeOrigin::kNativeHydraEdge, 0});
      }
    }
    if (show_native_room_room_edges_ && show_rooms_) {
      for (const auto& [source, target] : room_room_pairs) {
        projection.edges.push_back(DisplayEdge{
            source, target, DisplayEdgeType::kNativeRoomRoomDebug,
            EdgeOrigin::kNativeHydraEdge, 0});
      }
    }
    return projection;
  }

  // Room names and colours are display-only. They deliberately do not modify
  // Hydra node labels or the authoritative DSG. Ordinals are allocated once per
  // node ID and never reused during a process lifetime, so Room1 keeps the same
  // label and colour across incremental updates.
  void syncRoomDisplayOrdinals(const SceneModel& model) {
    std::vector<NodeId> room_ids;
    room_ids.reserve(model.nodes.size());
    for (const auto& [node_id, node] : model.nodes) {
      if (isRoom(node)) {
        room_ids.push_back(node_id);
      }
    }
    std::sort(room_ids.begin(), room_ids.end());
    for (const NodeId room_id : room_ids) {
      if (room_display_ordinals_.count(room_id) == 0U) {
        room_display_ordinals_[room_id] = next_room_display_ordinal_++;
      }
    }
  }

  uint32_t roomDisplayOrdinal(NodeId room_id) const {
    const auto found = room_display_ordinals_.find(room_id);
    return found == room_display_ordinals_.end() ? 0U : found->second;
  }

  std::string roomDisplayLabel(NodeId room_id) const {
    const uint32_t ordinal = roomDisplayOrdinal(room_id);
    return ordinal == 0U ? "Room" : "Room" + std::to_string(ordinal);
  }

  Color roomDisplayColor(NodeId room_id, float alpha) const {
    const uint32_t ordinal = roomDisplayOrdinal(room_id);
    if (ordinal == 0U) {
      return Color{0.45F, 0.45F, 0.45F, alpha};
    }

    // Golden-ratio hue stepping maximises visual separation for sequential
    // rooms while keeping each room colour deterministic and stable.
    constexpr double kGoldenRatioConjugate = 0.6180339887498948482;
    const double hue = std::fmod((static_cast<double>(ordinal) - 1.0) *
                                 kGoldenRatioConjugate,
                                 1.0);
    return hsvToRgb(hue, 0.72, 0.94, alpha);
  }

  static Color blackEdgeColor(float alpha = 0.90F) {
    return Color{0.0F, 0.0F, 0.0F, alpha};
  }

  static Color objectContactEdgeColor(float alpha = 0.95F) {
    return Color{0.91F, 0.64F, 0.24F, alpha};
  }

  static Color objectSegmentEdgeColor(float alpha = 0.95F) {
    return Color{0.20F, 0.55F, 0.90F, alpha};
  }

  double currentReferenceTimeSec() const {
    const auto clock = const_cast<SemanticSceneGraphFuser*>(this)->get_clock();
    const double clock_sec = static_cast<double>(clock->now().nanoseconds()) * 1.0e-9;
    if (clock_sec > 0.0) {
      return clock_sec;
    }
    const double stamp_sec = static_cast<double>(latest_header_.stamp.sec) +
                             static_cast<double>(latest_header_.stamp.nanosec) * 1.0e-9;
    if (stamp_sec > 0.0) {
      return stamp_sec;
    }
    return 0.0;
  }

  /** Return the semantic overlay already resolved to one Hydra DSG node. */
  const SemanticOverlay* overlayForNode(
      const NodeView& node,
      const std::unordered_map<NodeId, ResolvedOverlay>& resolved) const {
    const auto found = resolved.find(node.id);
    return found == resolved.end() ? nullptr : &found->second.overlay;
  }

  /** Select the configured static/unknown or dynamic presence half-life. */
  double presenceHalfLifeSec(const std::string& mobility_class) const {
    return normaliseMobilityClass(mobility_class) == "dynamic"
               ? dynamic_presence_half_life_sec_
               : static_presence_half_life_sec_;
  }

  /** Resolve time-dependent presence confidence for one observed semantic slot. */
  ResolvedPresence resolvePresenceForSlot(
      uint32_t slot_id,
      const PresenceCache& presence,
      const std::string& mobility_class) const {
    ResolvedPresence resolved;
    if (slot_id == 0U) {
      return resolved;
    }
    const auto it = presence.find(slot_id);
    if (it == presence.end()) {
      return resolved;
    }
    resolved.observation = it->second;
    const double now_sec = currentReferenceTimeSec();
    resolved.age_sec = std::max(0.0, now_sec - it->second.last_observed_timestamp_sec);
    const bool observed = resolved.age_sec <= presence_observed_epsilon_sec_;
    resolved.state = observed ? "OBSERVED" : "DECAYING";
    const double decay_age_sec = std::max(0.0, resolved.age_sec - presence_observed_epsilon_sec_);
    resolved.confidence = observed
                              ? 1.0
                              : std::pow(0.5, decay_age_sec / presenceHalfLifeSec(mobility_class));
    resolved.confidence = clampValue(resolved.confidence, 0.0, 1.0);
    return resolved;
  }

  /** Build compact multiline RViz text for one Hydra object node. */
  std::string objectDisplayLabel(
      const NodeView& node,
      const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
      const PresenceCache& presence) const {
    const SemanticOverlay* overlay = overlayForNode(node, resolved);
    const uint32_t slot_id = overlay ? overlay->slot_id : node.semantic_slot;
    const std::string mobility_class = overlay ? overlay->mobility_class : "unknown";
    std::string label = overlay
                            ? overlay->label
                            : (node.semantic_slot > 0U
                                   ? unlabeled_object_display_label_
                                   : (node.name.empty() ? unlabeled_object_display_label_ : node.name));

    if (show_label_confidence_ && overlay) {
      std::ostringstream line;
      line.setf(std::ios::fixed);
      line.precision(2);
      line << "label " << overlay->confidence;
      label += "\n" + line.str();
    }
    if (show_mobility_metadata_ && overlay) {
      std::ostringstream line;
      line.setf(std::ios::fixed);
      line.precision(2);
      line << "mob " << mobility_class << " " << overlay->mobility_confidence;
      label += "\n" + line.str();
    }
    if (show_presence_confidence_ && slot_id > 0U) {
      const ResolvedPresence presence_state = resolvePresenceForSlot(slot_id, presence, mobility_class);
      if (presence_state.observation.slot_id == 0U) {
        label += "\npresence n/a";
      } else {
        std::ostringstream line;
        line.setf(std::ios::fixed);
        line.precision(2);
        line << "presence " << (presence_state.state == "OBSERVED" ? "obs " : "dec ")
             << presence_state.confidence;
        label += "\n" + line.str();
      }
    }
    if (show_slot_ids_ && slot_id > 0U) {
      label += "\nslot " + std::to_string(slot_id);
    }
    return label;
  }

  /** Apply semantic colour and mobility-aware presence alpha to one node. */
  Color objectDisplayColor(
      const NodeView& node,
      const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
      const PresenceCache& presence) const {
    const SemanticOverlay* overlay = overlayForNode(node, resolved);
    Color color = overlay ? colorForLabel(overlay->label)
                          : Color{0.55F, 0.55F, 0.55F, 0.95F};
    const std::string mobility_class = overlay ? overlay->mobility_class : "unknown";
    const uint32_t slot_id = overlay ? overlay->slot_id : node.semantic_slot;
    const ResolvedPresence presence_state = resolvePresenceForSlot(slot_id, presence, mobility_class);
    if (presence_state.observation.slot_id > 0U) {
      color.a = std::max(
          minimum_object_alpha_,
          static_cast<float>(static_cast<double>(color.a) * presence_state.confidence));
    }
    return color;
  }

  void addMarker(MarkerArray& markers, MarkerSet& next_keys, Marker marker) const {
    const MarkerKey key{marker.ns, marker.id};
    if (!next_keys.insert(key).second) {
      // Stable IDs must be unique within a namespace. Do not emit duplicate
      // marker keys; a later node could otherwise overwrite an earlier one.
      return;
    }
    markers.markers.push_back(std::move(marker));
  }

  void publishMarkerArray(const SceneModel& model,
                          const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                          const PresenceCache& presence,
                          const LayeredProjection& projection) {
    const std::string frame = latest_header_.frame_id.empty() ? fallback_frame_id_ : latest_header_.frame_id;
    builtin_interfaces::msg::Time stamp = latest_header_.stamp;
    if (stamp.sec == 0 && stamp.nanosec == 0U) {
      // rclcpp::Time on ROS 2 Iron does not expose to_msg(). Convert through
      // nanoseconds so this fuser stays compatible with Iron and newer ROS 2 releases.
      const int64_t now_ns = get_clock()->now().nanoseconds();
      constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
      stamp.sec = static_cast<int32_t>(now_ns / kNanosecondsPerSecond);
      stamp.nanosec = static_cast<uint32_t>(now_ns % kNanosecondsPerSecond);
    }

    MarkerArray markers;
    MarkerSet next_keys;
    if (first_marker_publication_) {
      Marker clear;
      clear.header.frame_id = frame;
      clear.header.stamp = stamp;
      clear.action = Marker::DELETEALL;
      markers.markers.push_back(std::move(clear));
      first_marker_publication_ = false;
    }

    for (const auto& [node_id, node] : model.nodes) {
      (void)node_id;
      if (!node.visible) {
        continue;
      }
      if (isObject(node)) {
        appendObjectMarkers(markers, next_keys, frame, stamp, node, resolved, presence);
      } else if (isRoom(node)) {
        appendLayerNodeMarker(
            markers, next_keys, frame, stamp, node, "rsg_layered_rooms", "rsg_layered_room_labels",
            roomDisplayColor(node.id, room_alpha_), room_node_size_m_, room_text_height_m_, show_room_labels_,
            roomDisplayLabel(node.id));
      } else if (isPlace(node)) {
        appendLayerNodeMarker(
            markers, next_keys, frame, stamp, node, "rsg_layered_places", "rsg_layered_place_labels",
            placeDisplayColor(node, projection), place_node_size_m_, place_text_height_m_, show_place_labels_,
            node.name.empty() ? "place " + idString(node.id) : node.name);
      } else if (isBuilding(node)) {
        appendSemanticVolumeMarkers(
            markers, next_keys, frame, stamp, node, "rsg_buildings", "rsg_building_labels",
            Color{0.20F, 0.72F, 0.35F, building_alpha_}, building_text_height_m_, show_building_labels_,
            node.name.empty() ? "building " + idString(node.id) : node.name);
      } else {
        appendOptionalLayerMarker(markers, next_keys, frame, stamp, node);
      }
    }

    appendEdgeMarkers(markers, next_keys, frame, stamp, model, projection);

    // Delete only markers whose source DSG node/edge disappeared or became
    // hidden. This is the operation that makes Hydra-side removals visible in
    // RViz without clearing and redrawing every marker each update.
    for (const auto& stale_key : active_marker_keys_) {
      if (next_keys.count(stale_key) != 0U) {
        continue;
      }
      Marker erase;
      erase.header.frame_id = frame;
      erase.header.stamp = stamp;
      erase.ns = stale_key.first;
      erase.id = stale_key.second;
      erase.action = Marker::DELETE;
      markers.markers.push_back(std::move(erase));
    }
    active_marker_keys_.swap(next_keys);

    marker_pub_->publish(markers);
    ++marker_publications_;
    last_marker_publish_ = std::chrono::steady_clock::now();
  }

  double objectSphereDiameter(const NodeView& node) const {
    if (object_use_fixed_sphere_size_) {
      return object_fixed_sphere_size_m_;
    }
    if (!node.has_bbox) {
      return object_min_size_m_;
    }
    const Eigen::Vector3d dimensions = node.bbox_size.cast<double>().cwiseAbs();
    const double volume = std::max(
        kSmallExtentM * kSmallExtentM * kSmallExtentM,
        dimensions.x() * dimensions.y() * dimensions.z());
    const double equivalent_sphere_diameter = std::cbrt((6.0 * volume) / kPi);
    return clampValue(object_sphere_volume_scale_ * equivalent_sphere_diameter,
                      object_min_size_m_, object_max_size_m_);
  }

  void appendObjectMarkers(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                           const builtin_interfaces::msg::Time& stamp, const NodeView& node,
                           const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                           const PresenceCache& presence) const {
    const Color color = objectDisplayColor(node, resolved, presence);
    const SemanticOverlay* overlay = overlayForNode(node, resolved);
    const bool dynamic_object = overlay && overlay->mobility_class == "dynamic";
    const int32_t id = markerId(node.id);
    Marker marker;
    marker.header.frame_id = frame;
    marker.header.stamp = stamp;
    marker.ns = "rsg_objects";
    marker.id = id;
    marker.action = Marker::ADD;
    const Eigen::Vector3d display_position = displayPosition(node);
    marker.pose.position.x = display_position.x();
    marker.pose.position.y = display_position.y();
    marker.pose.position.z = display_position.z();
    marker.pose.orientation.w = 1.0;
    marker.color.r = color.r;
    marker.color.g = color.g;
    marker.color.b = color.b;
    marker.color.a = color.a;
    const double marker_diameter = objectSphereDiameter(node);
    marker.type = dynamic_object && dynamic_object_use_cube_ ? Marker::CUBE : Marker::SPHERE;
    marker.scale.x = marker_diameter;
    marker.scale.y = marker_diameter;
    marker.scale.z = marker_diameter;
    addMarker(markers, next_keys, std::move(marker));

    if (!show_object_labels_) {
      return;
    }
    Marker text;
    text.header.frame_id = frame;
    text.header.stamp = stamp;
    text.ns = "rsg_object_labels";
    text.id = id;
    text.type = Marker::TEXT_VIEW_FACING;
    text.action = Marker::ADD;
    text.pose.position.x = display_position.x();
    text.pose.position.y = display_position.y();
    const double z_top = display_position.z() + marker_diameter / 2.0;
    text.pose.position.z = z_top + object_label_vertical_offset_m_;
    text.pose.orientation.w = 1.0;
    text.scale.z = object_text_height_m_;
    text.color.r = 0.0F;
    text.color.g = 0.0F;
    text.color.b = 0.0F;
    text.color.a = 1.0F;
    text.text = objectDisplayLabel(node, resolved, presence);
    addMarker(markers, next_keys, std::move(text));
  }

  Color placeDisplayColor(const NodeView& node, const LayeredProjection& projection) const {
    const auto parent = projection.place_to_room.find(node.id);
    if (parent == projection.place_to_room.end()) {
      return Color{0.45F, 0.45F, 0.45F, place_alpha_};
    }
    return roomDisplayColor(parent->second, place_alpha_);
  }

  void appendLayerNodeMarker(MarkerArray& markers, MarkerSet& next_keys,
                             const std::string& frame,
                             const builtin_interfaces::msg::Time& stamp, const NodeView& node,
                             const std::string& node_namespace, const std::string& text_namespace,
                             const Color& color, double node_size, double text_height,
                             bool show_text, const std::string& text_label) const {
    const int32_t id = markerId(node.id);
    const Eigen::Vector3d position = displayPosition(node);
    Marker sphere;
    sphere.header.frame_id = frame;
    sphere.header.stamp = stamp;
    sphere.ns = node_namespace;
    sphere.id = id;
    sphere.type = Marker::SPHERE;
    sphere.action = Marker::ADD;
    sphere.pose.position.x = position.x();
    sphere.pose.position.y = position.y();
    sphere.pose.position.z = position.z();
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = node_size;
    sphere.scale.y = node_size;
    sphere.scale.z = node_size;
    sphere.color.r = color.r;
    sphere.color.g = color.g;
    sphere.color.b = color.b;
    sphere.color.a = color.a;
    addMarker(markers, next_keys, std::move(sphere));

    if (!show_text) {
      return;
    }
    Marker text;
    text.header.frame_id = frame;
    text.header.stamp = stamp;
    text.ns = text_namespace;
    text.id = id;
    text.type = Marker::TEXT_VIEW_FACING;
    text.action = Marker::ADD;
    text.pose.position.x = position.x();
    text.pose.position.y = position.y();
    text.pose.position.z = position.z() + node_size * 0.65;
    text.pose.orientation.w = 1.0;
    text.scale.z = text_height;
    text.color.r = 0.0F;
    text.color.g = 0.0F;
    text.color.b = 0.0F;
    text.color.a = 1.0F;
    text.text = text_label;
    addMarker(markers, next_keys, std::move(text));
  }

  void appendSemanticVolumeMarkers(MarkerArray& markers, MarkerSet& next_keys,
                                   const std::string& frame,
                                   const builtin_interfaces::msg::Time& stamp, const NodeView& node,
                                   const std::string& volume_namespace, const std::string& text_namespace,
                                   const Color& color, double text_height, bool show_text,
                                   const std::string& text_label) const {
    const int32_t id = markerId(node.id);
    if (node.has_bbox) {
      Marker volume;
      volume.header.frame_id = frame;
      volume.header.stamp = stamp;
      volume.ns = volume_namespace;
      volume.id = id;
      volume.type = Marker::CUBE;
      volume.action = Marker::ADD;
      const Eigen::Vector3d volume_position = displayPosition(node);
      volume.pose.position.x = volume_position.x();
      volume.pose.position.y = volume_position.y();
      volume.pose.position.z = volume_position.z();
      volume.pose.orientation.w = 1.0;
      volume.scale.x = std::max(kSmallExtentM, static_cast<double>(node.bbox_size.x()));
      volume.scale.y = std::max(kSmallExtentM, static_cast<double>(node.bbox_size.y()));
      volume.scale.z = std::max(kSmallExtentM, static_cast<double>(node.bbox_size.z()));
      volume.color.r = color.r;
      volume.color.g = color.g;
      volume.color.b = color.b;
      volume.color.a = color.a;
      addMarker(markers, next_keys, std::move(volume));
    } else {
      Marker sphere;
      sphere.header.frame_id = frame;
      sphere.header.stamp = stamp;
      sphere.ns = volume_namespace;
      sphere.id = id;
      sphere.type = Marker::SPHERE;
      sphere.action = Marker::ADD;
      const Eigen::Vector3d sphere_position = displayPosition(node);
      sphere.pose.position.x = sphere_position.x();
      sphere.pose.position.y = sphere_position.y();
      sphere.pose.position.z = sphere_position.z();
      sphere.pose.orientation.w = 1.0;
      sphere.scale.x = 0.28;
      sphere.scale.y = 0.28;
      sphere.scale.z = 0.28;
      sphere.color.r = color.r;
      sphere.color.g = color.g;
      sphere.color.b = color.b;
      sphere.color.a = std::max(0.45F, color.a);
      addMarker(markers, next_keys, std::move(sphere));
    }

    if (!show_text) {
      return;
    }
    Marker text;
    text.header.frame_id = frame;
    text.header.stamp = stamp;
    text.ns = text_namespace;
    text.id = id;
    text.type = Marker::TEXT_VIEW_FACING;
    text.action = Marker::ADD;
    const Eigen::Vector3d text_position = displayPosition(node);
    text.pose.position.x = text_position.x();
    text.pose.position.y = text_position.y();
    text.pose.position.z = text_position.z() + (node.has_bbox ? node.bbox_size.z() / 2.0 : 0.0) + 0.20;
    text.pose.orientation.w = 1.0;
    text.scale.z = text_height;
    text.color.r = 0.0F;
    text.color.g = 0.0F;
    text.color.b = 0.0F;
    text.color.a = 1.0F;
    text.text = text_label;
    addMarker(markers, next_keys, std::move(text));
  }

  void appendOptionalLayerMarker(MarkerArray& markers, MarkerSet& next_keys,
                                 const std::string& frame,
                                 const builtin_interfaces::msg::Time& stamp,
                                 const NodeView& node) const {
    Marker sphere;
    sphere.header.frame_id = frame;
    sphere.header.stamp = stamp;
    sphere.ns = "rsg_optional_layers";
    sphere.id = markerId(node.id);
    sphere.type = Marker::SPHERE;
    sphere.action = Marker::ADD;
    const Eigen::Vector3d optional_position = displayPosition(node);
    sphere.pose.position.x = optional_position.x();
    sphere.pose.position.y = optional_position.y();
    sphere.pose.position.z = optional_position.z();
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = 0.12;
    sphere.scale.y = 0.12;
    sphere.scale.z = 0.12;
    sphere.color.r = 0.70F;
    sphere.color.g = 0.40F;
    sphere.color.b = 0.95F;
    sphere.color.a = 0.80F;
    addMarker(markers, next_keys, std::move(sphere));
  }

  geometry_msgs::msg::Point markerPoint(const Eigen::Vector3d& position) const {
    geometry_msgs::msg::Point point;
    point.x = position.x();
    point.y = position.y();
    point.z = position.z();
    return point;
  }

  void appendLineList(MarkerArray& markers, MarkerSet& next_keys,
                      const std::string& frame, const builtin_interfaces::msg::Time& stamp,
                      const std::string& marker_namespace, int32_t id, const Color& color,
                      const std::vector<geometry_msgs::msg::Point>& points) const {
    if (points.empty()) {
      return;
    }
    Marker lines;
    lines.header.frame_id = frame;
    lines.header.stamp = stamp;
    lines.ns = marker_namespace;
    lines.id = id;
    lines.type = Marker::LINE_LIST;
    lines.action = Marker::ADD;
    lines.pose.orientation.w = 1.0;
    lines.scale.x = edge_width_m_;
    lines.color.r = color.r;
    lines.color.g = color.g;
    lines.color.b = color.b;
    lines.color.a = color.a;
    lines.points = points;
    addMarker(markers, next_keys, std::move(lines));
  }

  void appendPointList(MarkerArray& markers, MarkerSet& next_keys,
                       const std::string& frame, const builtin_interfaces::msg::Time& stamp,
                       const std::string& marker_namespace, int32_t id, const Color& color,
                       double point_size_m, const std::vector<geometry_msgs::msg::Point>& points) const {
    if (points.empty()) {
      return;
    }
    Marker dots;
    dots.header.frame_id = frame;
    dots.header.stamp = stamp;
    dots.ns = marker_namespace;
    dots.id = id;
    dots.type = Marker::POINTS;
    dots.action = Marker::ADD;
    dots.pose.orientation.w = 1.0;
    dots.scale.x = point_size_m;
    dots.scale.y = point_size_m;
    dots.color.r = color.r;
    dots.color.g = color.g;
    dots.color.b = color.b;
    dots.color.a = color.a;
    dots.points = points;
    addMarker(markers, next_keys, std::move(dots));
  }

  /** Evenly spaced points from start to end (inclusive), for a dotted-line marker. */
  static void appendDottedSegmentPoints(std::vector<geometry_msgs::msg::Point>& points,
                                        const geometry_msgs::msg::Point& start,
                                        const geometry_msgs::msg::Point& end, double spacing_m) {
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double dz = end.z - start.z;
    const double length_m = std::sqrt(dx * dx + dy * dy + dz * dz);
    const int n = std::max(2, static_cast<int>(std::round(length_m / std::max(1.0e-6, spacing_m))) + 1);
    for (int k = 0; k < n; ++k) {
      const double t = static_cast<double>(k) / static_cast<double>(n - 1);
      geometry_msgs::msg::Point point;
      point.x = start.x + t * dx;
      point.y = start.y + t * dy;
      point.z = start.z + t * dz;
      points.push_back(point);
    }
  }

  /**
   * Contact edges are a symmetric relation, so they render as a shaft with an
   * arrowhead at each end rather than a plain undirected line: two overlapping
   * ARROW markers along the same segment, one per direction.
   */
  void appendObjectContactArrows(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                                 const builtin_interfaces::msg::Time& stamp, NodeId source, NodeId target,
                                 const geometry_msgs::msg::Point& start,
                                 const geometry_msgs::msg::Point& end) const {
    if (!show_object_contact_edges_) {
      return;
    }
    const Color color = objectContactEdgeColor(0.95F);
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double dz = end.z - start.z;
    const double segment_length_m = std::sqrt(dx * dx + dy * dy + dz * dz);
    // Fixed small head size (not auto-scaled off segment length, which would
    // otherwise make longer contact edges sprout disproportionately large
    // heads), capped so it never overshoots a very short segment.
    const double head_length_m = std::min(edge_width_m_ * 3.0, 0.4 * segment_length_m);
    const auto make_arrow = [&](int32_t id, const geometry_msgs::msg::Point& tail,
                                const geometry_msgs::msg::Point& head) {
      Marker arrow;
      arrow.header.frame_id = frame;
      arrow.header.stamp = stamp;
      arrow.ns = "rsg_object_contact_edges";
      arrow.id = id;
      arrow.type = Marker::ARROW;
      arrow.action = Marker::ADD;
      arrow.pose.orientation.w = 1.0;
      arrow.points = {tail, head};
      arrow.scale.x = edge_width_m_;        // shaft diameter
      arrow.scale.y = edge_width_m_ * 1.5;  // head diameter
      arrow.scale.z = head_length_m;
      arrow.color.r = color.r;
      arrow.color.g = color.g;
      arrow.color.b = color.b;
      arrow.color.a = color.a;
      addMarker(markers, next_keys, std::move(arrow));
    };
    make_arrow(markerIdForPair(source, target), start, end);
    make_arrow(markerIdForPair(target, source), end, start);
  }

  /**
   * Small floating label at a contact edge's midpoint: centroid distance and
   * 2D IoU (the larger of the XZ- and YZ-plane projections; volumetric 3D
   * IoU is a poor summary here, e.g. for a small object against a large
   * flat surface).
   */
  void appendObjectContactLabel(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                                const builtin_interfaces::msg::Time& stamp, NodeId source, NodeId target,
                                const geometry_msgs::msg::Point& start,
                                const geometry_msgs::msg::Point& end, double iou_2d_max,
                                double centroid_distance_m) const {
    if (!show_object_contact_labels_) {
      return;
    }
    Marker text;
    text.header.frame_id = frame;
    text.header.stamp = stamp;
    text.ns = "rsg_object_contact_labels";
    text.id = markerIdForPair(source, target);
    text.type = Marker::TEXT_VIEW_FACING;
    text.action = Marker::ADD;
    text.pose.position.x = 0.5 * (start.x + end.x);
    text.pose.position.y = 0.5 * (start.y + end.y);
    text.pose.position.z = 0.5 * (start.z + end.z);
    text.pose.orientation.w = 1.0;
    text.scale.z = object_contact_text_height_m_;
    text.color.r = 0.0F;
    text.color.g = 0.0F;
    text.color.b = 0.0F;
    text.color.a = 1.0F;
    std::ostringstream line;
    line.setf(std::ios::fixed);
    line.precision(2);
    line << centroid_distance_m << " m\n" << "iou_2d " << iou_2d_max;
    text.text = line.str();
    addMarker(markers, next_keys, std::move(text));
  }

  void appendEdgeMarkers(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                         const builtin_interfaces::msg::Time& stamp, const SceneModel& model,
                         const LayeredProjection& projection) const {
    std::unordered_map<NodeId, std::vector<geometry_msgs::msg::Point>> room_place_points;
    std::unordered_map<NodeId, std::vector<geometry_msgs::msg::Point>> place_object_points;
    std::vector<geometry_msgs::msg::Point> place_connectivity_points;
    std::vector<geometry_msgs::msg::Point> object_object_points;
    std::vector<geometry_msgs::msg::Point> object_segment_dot_points;
    std::vector<geometry_msgs::msg::Point> room_room_points;

    for (const auto& edge : projection.edges) {
      const auto source_it = model.nodes.find(edge.source);
      const auto target_it = model.nodes.find(edge.target);
      if (source_it == model.nodes.end() || target_it == model.nodes.end()) {
        continue;
      }
      const auto start = markerPoint(displayPosition(source_it->second));
      const auto end = markerPoint(displayPosition(target_it->second));
      switch (edge.type) {
        case DisplayEdgeType::kRoomPlaceHierarchy: {
          auto& points = room_place_points[edge.color_owner];
          points.push_back(start);
          points.push_back(end);
          break;
        }
        case DisplayEdgeType::kPlaceObjectMembership: {
          auto& points = place_object_points[edge.source];
          points.push_back(start);
          points.push_back(end);
          break;
        }
        case DisplayEdgeType::kPlaceConnectivity:
          place_connectivity_points.push_back(start);
          place_connectivity_points.push_back(end);
          break;
        case DisplayEdgeType::kNativeObjectObjectDebug:
          object_object_points.push_back(start);
          object_object_points.push_back(end);
          break;
        case DisplayEdgeType::kDerivedObjectContact:
          appendObjectContactArrows(markers, next_keys, frame, stamp, edge.source, edge.target, start, end);
          appendObjectContactLabel(markers, next_keys, frame, stamp, edge.source, edge.target, start,
                                   end, edge.contact_iou_2d_max, edge.contact_centroid_distance_m);
          break;
        case DisplayEdgeType::kDerivedObjectSegment:
          if (show_object_segment_edges_) {
            appendDottedSegmentPoints(object_segment_dot_points, start, end, object_segment_dot_spacing_m_);
          }
          break;
        case DisplayEdgeType::kNativeRoomRoomDebug:
          room_room_points.push_back(start);
          room_room_points.push_back(end);
          break;
      }
    }

    for (const auto& [room_id, points] : room_place_points) {
      appendLineList(markers, next_keys, frame, stamp, "rsg_room_place_edges", markerId(room_id),
                     blackEdgeColor(0.90F), points);
    }
    for (const auto& [place_id, points] : place_object_points) {
      appendLineList(markers, next_keys, frame, stamp, "rsg_place_object_edges", markerId(place_id),
                     blackEdgeColor(0.90F), points);
    }
    appendLineList(markers, next_keys, frame, stamp, "rsg_place_connectivity_edges", 0,
                   blackEdgeColor(0.90F), place_connectivity_points);
    appendLineList(markers, next_keys, frame, stamp, "rsg_native_object_object_edges", 0,
                   blackEdgeColor(0.90F), object_object_points);
    appendLineList(markers, next_keys, frame, stamp, "rsg_native_room_room_edges", 0,
                   blackEdgeColor(0.90F), room_room_points);
    appendPointList(markers, next_keys, frame, stamp, "rsg_object_segment_edges", 0,
                    objectSegmentEdgeColor(0.95F), object_segment_dot_size_m_, object_segment_dot_points);
  }

  Json buildSnapshot(const SceneModel& model,
                     const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                     const LayeredProjection& projection) const {
    const std::string frame = latest_header_.frame_id.empty() ? fallback_frame_id_ : latest_header_.frame_id;
    Json snapshot = {
        {"event", "hydra_rap_fused_snapshot"},
        {"sequence", latest_sequence_},
        {"frame_id", frame},
        {"stamp", {{"sec", latest_header_.stamp.sec}, {"nanosec", latest_header_.stamp.nanosec}}},
        {"mesh_included", false},
        {"nodes", Json::array()},
        {"edges", Json::array()},
    };

    for (const auto& [node_id, node] : model.nodes) {
      if (!node.visible) {
        continue;
      }
      Json entry = {
          {"hydra_node_id", idString(node_id)},
          {"layer", layerName(node.kind)},
          {"source_position", vector3(sourcePosition(node))},
          {"display_position", vector3(displayPosition(node))},
          {"name", node.name},
          {"semantic_slot_id", node.semantic_slot},
          {"bbox_valid", node.has_bbox},
      };
      if (node.has_bbox) {
        entry["bbox_center"] = vector3f(node.bbox_center);
        entry["bbox_size"] = vector3f(node.bbox_size);
      }
      if (isPlace(node)) {
        const auto room_it = projection.place_to_room.find(node_id);
        if (room_it != projection.place_to_room.end()) {
          entry["parent_room"] = idString(room_it->second);
          const auto origin_it = projection.place_room_origin.find(node_id);
          if (origin_it != projection.place_room_origin.end()) {
            entry["room_association_origin"] = edgeOriginName(origin_it->second);
          }
        }
      } else if (isObject(node)) {
        const auto place_it = projection.object_to_place.find(node_id);
        if (place_it != projection.object_to_place.end()) {
          entry["parent_place"] = idString(place_it->second);
          const auto origin_it = projection.object_place_origin.find(node_id);
          if (origin_it != projection.object_place_origin.end()) {
            entry["place_association_origin"] = edgeOriginName(origin_it->second);
          }
        }
      }
      const auto overlay_it = resolved.find(node_id);
      if (overlay_it != resolved.end()) {
        entry["rap"] = {
            {"label", overlay_it->second.overlay.label},
            {"confidence", overlay_it->second.overlay.confidence},
            {"label_confidence", overlay_it->second.overlay.confidence},
            {"mobility_class", overlay_it->second.overlay.mobility_class},
            {"mobility_confidence", overlay_it->second.overlay.mobility_confidence},
            {"mobility_source", overlay_it->second.overlay.mobility_source},
            {"source", overlay_it->second.overlay.source},
            {"association", overlay_it->second.association},
            {"centroid_distance_m", overlay_it->second.centroid_distance_m},
        };
      } else if (isObject(node)) {
        // Keep the diagnostic snapshot aligned with RViz: every object has a
        // visible display label, even before a Phase-1 semantic result exists.
        entry["fuser_display"] = {
            {"label", unlabeled_object_display_label_},
            {"source", "fuser_display_fallback"},
            {"association", "no_phase1_semantic_result"},
        };
      }
      snapshot["nodes"].push_back(std::move(entry));
    }

    for (const auto& edge : projection.edges) {
      snapshot["edges"].push_back({
          {"source", idString(edge.source)},
          {"target", idString(edge.target)},
          {"type", edgeTypeName(edge.type)},
          {"origin", edgeOriginName(edge.origin)},
          {"color_owner_room", edge.color_owner == 0 ? "" : idString(edge.color_owner)},
      });
    }
    snapshot["object_relationships"] = last_object_relations_;
    snapshot["object_segment_relationships"] = last_object_segment_relations_;
    return snapshot;
  }

  bool shouldPublishMarkers(bool force) const {
    if (force || marker_publish_rate_hz_ <= 0.0 || marker_publications_ == 0) {
      return true;
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_marker_publish_).count();
    return elapsed >= (1.0 / marker_publish_rate_hz_);
  }

  void publishFusedDsg() {
    if (!publish_fused_dsg_ || !graph_) {
      return;
    }
    DsgUpdate output;
    output.header = latest_header_;
    if (output.header.frame_id.empty()) {
      output.header.frame_id = fallback_frame_id_;
    }
    output.sequence_number = latest_sequence_;
    output.full_update = true;
    spark_dsg::io::binary::writeGraph(*graph_, output.layer_contents, false);
    fused_dsg_pub_->publish(output);
    ++fused_dsg_publications_;
  }

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
                           const PresenceCache& presence_snapshot) {
    std::lock_guard<std::mutex> graph_lock(graph_mutex_);
    if (!graph_) {
      return false;
    }

    const bool publish_markers = shouldPublishMarkers(force_markers);
    if (!publish_fused_dsg_ && !publish_markers) {
      return true;
    }

    // Drop last cycle's derived edges before the model is collected so
    // collectModel() only sees native Hydra topology.
    pruneOwnedContactEdges();
    pruneOwnedSegmentEdges();

    const SceneModel model = collectModel();
    const std::string frame = latest_header_.frame_id.empty() ? fallback_frame_id_ : latest_header_.frame_id;
    const auto resolved = resolveOverlays(model, frame, overlay_snapshot);
    updateLocalGraphMetadata(model, resolved, presence_snapshot);
    syncRoomDisplayOrdinals(model);
    auto projection = buildLayeredProjection(model);
    const auto main_object_groups = buildMainObjectGroups(model, presence_snapshot);
    computeObjectContacts(main_object_groups, projection);
    computeObjectSegmentEdges(main_object_groups, projection);
    writeContactDiagnostics(main_object_groups, resolved);

    publishFusedDsg();
    if (publish_markers) {
      publishMarkerArray(model, resolved, presence_snapshot, projection);
    }
    publishStatus(reason, "ok", &model, &resolved, &projection, &overlay_snapshot);
    return true;
  }

  Json rsgMetadataForNode(NodeId node_id) const {
    if (!graph_) {
      return Json::object();
    }
    try {
      const auto& attrs = graph_->getNode(node_id).attributes<spark_dsg::NodeAttributes>();
      Json metadata = attrs.metadata.get();
      if (metadata.is_object()) {
        return metadata;
      }
    } catch (const std::exception&) {
    }
    return Json::object();
  }

  Json buildObjectMetadataExport(const SceneModel& model,
                                 const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                                 const PresenceCache& presence,
                                 const LayeredProjection& projection) const {
    const std::string frame = latest_header_.frame_id.empty() ? fallback_frame_id_ : latest_header_.frame_id;
    Json output = {
        {"event", "rsg_fuser_object_metadata_export"},
        {"schema", "rsg_fuser_objects_v3_mobility_decay"},
        {"sequence", latest_sequence_},
        {"frame_id", frame},
        {"stamp", {{"sec", latest_header_.stamp.sec}, {"nanosec", latest_header_.stamp.nanosec}}},
        {"reference_time_sec", currentReferenceTimeSec()},
        {"objects", Json::array()},
    };

    std::unordered_map<uint32_t, size_t> slot_counts;
    std::unordered_map<std::string, size_t> internal_counts;
    for (const auto& [node_id, node] : model.nodes) {
      if (isObject(node)) {
        slot_counts[node.semantic_slot] += 1U;
      }
    }

    for (const auto& [node_id, node] : model.nodes) {
      if (!isObject(node)) {
        continue;
      }
      Json metadata = rsgMetadataForNode(node_id);
      Json identity = metadata.contains("rsg_identity") && metadata["rsg_identity"].is_object()
                          ? metadata["rsg_identity"]
                          : Json::object();
      Json presence_json = metadata.contains("rsg_presence") && metadata["rsg_presence"].is_object()
                               ? metadata["rsg_presence"]
                               : Json::object();
      Json rap_json = metadata.contains("rsg_rap") && metadata["rsg_rap"].is_object()
                          ? metadata["rsg_rap"]
                          : Json::object();

      const auto overlay_it = resolved.find(node_id);
      if (overlay_it != resolved.end()) {
        rap_json = {
            {"label", overlay_it->second.overlay.label},
            {"confidence", overlay_it->second.overlay.confidence},
            {"label_confidence", overlay_it->second.overlay.confidence},
            {"mobility_class", overlay_it->second.overlay.mobility_class},
            {"mobility_confidence", overlay_it->second.overlay.mobility_confidence},
            {"mobility_source", overlay_it->second.overlay.mobility_source},
            {"source", overlay_it->second.overlay.source},
            {"association", overlay_it->second.association},
            {"centroid_distance_m", overlay_it->second.centroid_distance_m},
            {"slot_id", overlay_it->second.overlay.slot_id},
            {"timestamp_sec", overlay_it->second.overlay.timestamp_sec},
        };
      }

      const std::string mobility_class = overlay_it != resolved.end()
                                             ? overlay_it->second.overlay.mobility_class
                                             : "unknown";
      const auto presence_resolved = resolvePresenceForSlot(
          node.semantic_slot, presence, mobility_class);
      const double selected_half_life_sec = presenceHalfLifeSec(mobility_class);
      const Color display_color = objectDisplayColor(node, resolved, presence);
      if (presence_resolved.observation.slot_id > 0U) {
        const auto& obs = presence_resolved.observation;
        presence_json["slot_id"] = obs.slot_id;
        presence_json["state"] = presence_resolved.state;
        presence_json["confidence"] = presence_resolved.confidence;
        presence_json["age_sec"] = presence_resolved.age_sec;
        presence_json["half_life_sec"] = selected_half_life_sec;
        presence_json["minimum_alpha"] = minimum_object_alpha_;
        presence_json["rendered_alpha"] = display_color.a;
        presence_json["last_observed_timestamp_sec"] = obs.last_observed_timestamp_sec;
        presence_json["internal_object_id"] = obs.internal_object_id;
        presence_json["persistent_track_id"] = obs.persistent_track_id;
        presence_json["local_segment_id"] = obs.local_segment_id;
        presence_json["local_segment_xy_span_m"] = obs.local_segment_xy_span_m;
        if (obs.has_centroid) {
          presence_json["centroid_3d"] = vector3(obs.centroid);
        }
        if (obs.has_bbox) {
          presence_json["bbox_3d_min"] = vector3(obs.bbox_min);
          presence_json["bbox_3d_max"] = vector3(obs.bbox_max);
        }
      } else if (node.semantic_slot > 0U) {
        presence_json["slot_id"] = node.semantic_slot;
        presence_json["state"] = "UNKNOWN";
        presence_json["confidence"] = nullptr;
        presence_json["age_sec"] = nullptr;
        presence_json["half_life_sec"] = selected_half_life_sec;
        presence_json["minimum_alpha"] = minimum_object_alpha_;
        presence_json["rendered_alpha"] = display_color.a;
      }

      if (!identity.is_object()) {
        identity = Json::object();
      }
      if (!identity.contains("semantic_slot_id")) {
        identity["semantic_slot_id"] = node.semantic_slot;
      }
      if (!identity.contains("hydra_slot_id")) {
        identity["hydra_slot_id"] = node.semantic_slot;
      }
      if (presence_json.contains("internal_object_id") && !identity.contains("internal_object_id")) {
        identity["internal_object_id"] = presence_json["internal_object_id"];
      }
      if (presence_json.contains("persistent_track_id") && !identity.contains("persistent_track_id")) {
        identity["persistent_track_id"] = presence_json["persistent_track_id"];
      }
      if (presence_json.contains("local_segment_id") && !identity.contains("local_segment_id")) {
        identity["local_segment_id"] = presence_json["local_segment_id"];
      }

      Json entry = {
          {"hydra_node_id", idString(node_id)},
          {"layer", layerName(node.kind)},
          {"name", node.name},
          {"semantic_slot_id", node.semantic_slot},
          {"source_position", vector3(sourcePosition(node))},
          {"display_position", vector3(displayPosition(node))},
          {"bbox_valid", node.has_bbox},
          {"node_metadata", metadata},
          {"rsg_identity", identity},
          {"rsg_presence", presence_json},
          {"rsg_rap", rap_json},
          {"rviz_marker_shape",
           mobility_class == "dynamic" && dynamic_object_use_cube_ ? "cube" : "sphere"},
          {"rviz_marker_alpha", display_color.a},
      };
      if (node.has_bbox) {
        const Eigen::Vector3f half = 0.5F * node.bbox_size;
        const Eigen::Vector3f min_corner = node.bbox_center - half;
        const Eigen::Vector3f max_corner = node.bbox_center + half;
        const double xy_diag = std::hypot(static_cast<double>(node.bbox_size.x()),
                                          static_cast<double>(node.bbox_size.y()));
        const double volume = std::max(0.0, static_cast<double>(node.bbox_size.x())) *
                              std::max(0.0, static_cast<double>(node.bbox_size.y())) *
                              std::max(0.0, static_cast<double>(node.bbox_size.z()));
        entry["bbox_center"] = vector3f(node.bbox_center);
        entry["bbox_size"] = vector3f(node.bbox_size);
        entry["bbox_min"] = vector3f(min_corner);
        entry["bbox_max"] = vector3f(max_corner);
        entry["bbox_xy_diagonal_m"] = xy_diag;
        entry["bbox_volume_m3"] = volume;
      }
      const auto place_it = projection.object_to_place.find(node_id);
      if (place_it != projection.object_to_place.end()) {
        entry["parent_place"] = idString(place_it->second);
        const auto origin_it = projection.object_place_origin.find(node_id);
        if (origin_it != projection.object_place_origin.end()) {
          entry["place_association_origin"] = edgeOriginName(origin_it->second);
        }
        const auto room_it = projection.place_to_room.find(place_it->second);
        if (room_it != projection.place_to_room.end()) {
          entry["parent_room"] = idString(room_it->second);
        }
      }

      Json tracking_diagnostics = Json::object();
      tracking_diagnostics["internal_object_id"] = identity.value("internal_object_id", std::string());
      tracking_diagnostics["local_segment_id"] = identity.value("local_segment_id", std::string());
      tracking_diagnostics["semantic_slot_id"] = node.semantic_slot;
      if (node.has_bbox) {
        tracking_diagnostics["hydra_bbox_center_z"] = static_cast<double>(node.bbox_center.z());
        tracking_diagnostics["hydra_bbox_height_m"] = static_cast<double>(node.bbox_size.z());
      }
      if (presence_json.contains("centroid_3d") && presence_json["centroid_3d"].is_array() &&
          presence_json["centroid_3d"].size() == 3U) {
        const double local_centroid_z = presence_json["centroid_3d"][2].get<double>();
        tracking_diagnostics["local_segment_centroid_z"] = local_centroid_z;
        if (node.has_bbox) {
          tracking_diagnostics["hydra_minus_local_centroid_z_m"] =
              static_cast<double>(node.bbox_center.z()) - local_centroid_z;
        }
      }
      if (presence_json.contains("bbox_3d_min") && presence_json.contains("bbox_3d_max") &&
          presence_json["bbox_3d_min"].is_array() && presence_json["bbox_3d_max"].is_array() &&
          presence_json["bbox_3d_min"].size() == 3U && presence_json["bbox_3d_max"].size() == 3U) {
        const double local_min_z = presence_json["bbox_3d_min"][2].get<double>();
        const double local_max_z = presence_json["bbox_3d_max"][2].get<double>();
        const double local_center_z = 0.5 * (local_min_z + local_max_z);
        tracking_diagnostics["local_segment_bbox_center_z"] = local_center_z;
        tracking_diagnostics["local_segment_bbox_height_m"] = std::max(0.0, local_max_z - local_min_z);
        if (node.has_bbox) {
          tracking_diagnostics["hydra_minus_local_bbox_center_z_m"] =
              static_cast<double>(node.bbox_center.z()) - local_center_z;
        }
      }
      entry["rsg_tracking_diagnostics"] = std::move(tracking_diagnostics);

      const std::string internal_id = identity.value("internal_object_id", std::string());
      if (!internal_id.empty()) {
        internal_counts[internal_id] += 1U;
      }
      output["objects"].push_back(std::move(entry));
    }

    Json duplicate_slots = Json::array();
    for (const auto& [slot_id, count] : slot_counts) {
      if (slot_id > 0U && count > 1U) {
        duplicate_slots.push_back({{"semantic_slot_id", slot_id}, {"hydra_object_nodes", count}});
      }
    }
    Json duplicate_internal = Json::array();
    for (const auto& [internal_id, count] : internal_counts) {
      if (count > 1U) {
        duplicate_internal.push_back({{"internal_object_id", internal_id}, {"hydra_object_nodes", count}});
      }
    }
    output["summary"] = {
        {"object_count", output["objects"].size()},
        {"unique_semantic_slots", slot_counts.size()},
        {"duplicate_slot_count", duplicate_slots.size()},
        {"duplicate_slots", duplicate_slots},
        {"duplicate_internal_object_count", duplicate_internal.size()},
        {"duplicate_internal_objects", duplicate_internal},
        {"accepted_active_segment_messages", accepted_active_segment_messages_.load()},
        {"accepted_label_messages", accepted_label_messages_.load()},
        {"raw_dsg_updates", raw_dsg_updates_.load()},
        {"marker_publications", marker_publications_.load()},
    };
    return output;
  }

  size_t objectNodeCount(const SceneModel& model) const {
    size_t count = 0;
    for (const auto& [node_id, node] : model.nodes) {
      (void)node_id;
      count += isObject(node) ? 1U : 0U;
    }
    return count;
  }

  /** Publish lightweight fuser diagnostics without traversing the live DSG. */
  void publishStatus(const std::string& state,
                     const std::string& detail,
                     const SceneModel* model = nullptr,
                     const std::unordered_map<NodeId, ResolvedOverlay>* resolved = nullptr,
                     const LayeredProjection* projection = nullptr,
                     const OverlayCache* overlay_snapshot = nullptr) const {
    size_t label_slot_count = 0;
    size_t label_candidate_count = 0;
    OverlayCache copied_labels;
    if (overlay_snapshot) {
      copied_labels = *overlay_snapshot;
    } else {
      std::lock_guard<std::mutex> labels_lock(overlays_mutex_);
      copied_labels = overlays_by_slot_;
    }
    label_slot_count = copied_labels.size();
    for (const auto& [slot_id, candidates] : copied_labels) {
      (void)slot_id;
      label_candidate_count += candidates.size();
    }

    Json status = {
        {"event", "hydra_rap_fuser_status"},
        {"state", state},
        {"detail", detail},
        {"raw_dsg_updates", raw_dsg_updates_.load()},
        {"full_dsg_updates", full_dsg_updates_.load()},
        {"incremental_dsg_updates", incremental_dsg_updates_.load()},
        {"skipped_initial_incremental_updates", skipped_initial_incremental_updates_.load()},
        {"deleted_nodes_applied", deleted_nodes_applied_.load()},
        {"deleted_edges_applied", deleted_edges_applied_.load()},
        {"malformed_deleted_edge_updates", malformed_deleted_edge_updates_.load()},
        {"deserialize_failures", deserialize_failures_.load()},
        {"accepted_label_messages", accepted_label_messages_.load()},
        {"accepted_semantic_result_events", accepted_semantic_result_events_.load()},
        {"semantic_refresh_publications", semantic_refresh_publications_.load()},
        {"semantic_refresh_pending", semantic_refresh_pending_.load()},
        {"render_dirty", render_dirty_.load()},
        {"semantic_label_generation", semantic_label_generation_.load()},
        {"semantic_label_qos_depth", semantic_label_qos_depth_},
        {"ignored_label_messages", ignored_label_messages_.load()},
        {"invalid_label_messages", invalid_label_messages_.load()},
        {"labels_buffered_by_slot", label_slot_count},
        {"label_candidates_buffered", label_candidate_count},
        {"fused_dsg_publications", fused_dsg_publications_.load()},
        {"marker_publications", marker_publications_.load()},
        {"local_mesh_dropped", drop_local_mesh_},
        {"mesh_rendered", false},
    };
    if (model) {
      size_t visible_nodes = 0;
      size_t object_nodes = 0;
      size_t object_nodes_with_slot = 0;
      std::unordered_set<uint32_t> hydra_object_slots;
      for (const auto& [node_id, node] : model->nodes) {
        (void)node_id;
        visible_nodes += node.visible ? 1U : 0U;
        if (isObject(node)) {
          ++object_nodes;
          if (node.semantic_slot > 0) {
            ++object_nodes_with_slot;
            hydra_object_slots.insert(node.semantic_slot);
          }
        }
      }
      size_t buffered_slots_present_in_hydra = 0;
      for (const auto& [slot_id, candidates] : copied_labels) {
        (void)candidates;
        buffered_slots_present_in_hydra += hydra_object_slots.count(slot_id) ? 1U : 0U;
      }
      status["known_nodes"] = model->nodes.size();
      status["visible_nodes"] = visible_nodes;
      status["object_nodes"] = object_nodes;
      status["object_nodes_with_slot_id"] = object_nodes_with_slot;
      status["distinct_hydra_object_slots"] = hydra_object_slots.size();
      status["buffered_label_slots_present_in_hydra"] = buffered_slots_present_in_hydra;
    }
    if (resolved) {
      status["rap_annotations_applied"] = resolved->size();
    }
    if (projection) {
      size_t room_place_edges = 0;
      size_t place_object_edges = 0;
      size_t place_connectivity_edges = 0;
      size_t native_object_object_edges = 0;
      size_t native_room_room_edges = 0;
      for (const auto& edge : projection->edges) {
        if (edge.type == DisplayEdgeType::kRoomPlaceHierarchy) {
          ++room_place_edges;
        } else if (edge.type == DisplayEdgeType::kPlaceObjectMembership) {
          ++place_object_edges;
        } else if (edge.type == DisplayEdgeType::kPlaceConnectivity) {
          ++place_connectivity_edges;
        } else if (edge.type == DisplayEdgeType::kNativeObjectObjectDebug) {
          ++native_object_object_edges;
        } else if (edge.type == DisplayEdgeType::kNativeRoomRoomDebug) {
          ++native_room_room_edges;
        }
      }
      status["layered_display"] = true;
      status["visible_edges"] = projection->edges.size();
      status["room_place_edges"] = room_place_edges;
      status["place_object_edges"] = place_object_edges;
      status["place_connectivity_edges"] = place_connectivity_edges;
      status["native_object_object_edges"] = native_object_object_edges;
      status["native_room_room_edges"] = native_room_room_edges;
      status["object_contact_edges"] = last_object_contact_edge_count_;
      status["object_contact_max_iou_3d"] = last_object_contact_max_iou_;
      status["object_contact_grid_cell_size_m"] = last_object_contact_grid_cell_size_m_;
      status["object_contact_pairs_tested"] = last_object_contact_pairs_tested_;
      status["object_segment_edges"] = last_object_segment_edge_count_;
      status["mesh_validation_available"] = projection->mesh_validation_available;
      status["local_index_candidates_examined"] = projection->local_index_candidates_examined;
      status["mesh_rejected_candidates"] = projection->mesh_rejected_candidates;
      status["validated_local_place_associations"] = projection->validated_local_place_associations;
      status["local_association_cache_hits"] = projection->local_association_cache_hits;
      status["fallback_suppressed_without_mesh"] = projection->fallback_suppressed_without_mesh;
      status["room_completion_candidates_examined"] = projection->room_completion_candidates_examined;
      status["room_completion_mesh_rejected_candidates"] = projection->room_completion_mesh_rejected_candidates;
      status["derived_room_place_associations"] = projection->derived_room_place_associations;
      status["room_completion_ties"] = projection->room_completion_ties;
      status["room_completion_suppressed_without_mesh"] = projection->room_completion_suppressed_without_mesh;
      status["room_completion_waiting_for_grace"] = projection->room_completion_waiting_for_grace;
      status["room_completion_grace_elapsed_sec"] = projection->room_completion_grace_elapsed_sec;
      status["room_completion_grace_remaining_sec"] = projection->room_completion_grace_remaining_sec;
      status["room_completion_suppressed_by_grace"] = projection->room_completion_suppressed_by_grace;
      size_t place_nodes = 0;
      if (model) {
        for (const auto& [node_id, node] : model->nodes) {
          (void)node_id;
          place_nodes += isPlace(node) ? 1U : 0U;
        }
      }
      status["unassigned_places"] = model ? place_nodes - projection->place_to_room.size() : 0U;
      status["unassigned_objects"] = model ? objectNodeCount(*model) - projection->object_to_place.size() : 0U;
    }
    std_msgs::msg::String status_msg;
    status_msg.data = status.dump();
    status_pub_->publish(status_msg);
  }

  std::string input_dsg_topic_;
  std::string semantic_label_topic_;
  std::string active_segments_topic_;
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
  bool show_presence_confidence_ = true;
  bool show_label_confidence_ = true;
  bool show_mobility_metadata_ = true;
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
  std::string unlabeled_object_display_label_ = "unknown object";

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
  mutable std::mutex render_mutex_;
  spark_dsg::DynamicSceneGraph::Ptr graph_;
  // One slot normally has one class. Multiple candidates are retained only
  // when contradictory semantic messages are received for that same slot.
  OverlayCache overlays_by_slot_;
  PresenceCache presence_by_slot_;
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
  rclcpp::TimerBase::SharedPtr semantic_refresh_timer_;
  rclcpp::Publisher<DsgUpdate>::SharedPtr fused_dsg_pub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
};

}  // namespace rsg

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rsg::SemanticSceneGraphFuser>();
  // Three worker threads allow fast semantic ingestion, Hydra DSG updates, and
  // bounded rendering to make progress independently through callback groups.
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 3U);
  executor.add_node(node);
  executor.spin();
  executor.remove_node(node->get_node_base_interface());
  node.reset();
  rclcpp::shutdown();
  return 0;
}
