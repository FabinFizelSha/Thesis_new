/**
 * @file fuser_edge_markers.cpp
 * @brief RViz marker construction for contact/segment/hierarchy edges.
 */
#include "fuser.hpp"

namespace rsg {

  geometry_msgs::msg::Point SemanticSceneGraphFuser::markerPoint(const Eigen::Vector3d& position) const {
    geometry_msgs::msg::Point point;
    point.x = position.x();
    point.y = position.y();
    point.z = position.z();
    return point;
  }

  void SemanticSceneGraphFuser::appendLineList(MarkerArray& markers, MarkerSet& next_keys,
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

  void SemanticSceneGraphFuser::appendPointList(MarkerArray& markers, MarkerSet& next_keys,
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

  void SemanticSceneGraphFuser::appendDottedSegmentPoints(std::vector<geometry_msgs::msg::Point>& points,
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

  void SemanticSceneGraphFuser::appendObjectContactArrows(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                                 const builtin_interfaces::msg::Time& stamp, NodeId source, NodeId target,
                                 const geometry_msgs::msg::Point& start,
                                 const geometry_msgs::msg::Point& end,
                                 double source_radius_m, double target_radius_m) const {
    if (!show_object_contact_edges_) {
      return;
    }
    const Color color = objectContactEdgeColor(1.0F);  // Fully solid, not translucent.
    const double dx = end.x - start.x;
    const double dy = end.y - start.y;
    const double dz = end.z - start.z;
    const double segment_length_m = std::sqrt(dx * dx + dy * dy + dz * dz);
    constexpr double kMinSegmentLengthM = 1e-6;
    // Fraction of the segment to trim off each end so it starts/ends at the
    // object's own surface. Clamped to 0.5 each so the two trims can never
    // cross past the segment's midpoint (objects already touching or
    // overlapping) -- otherwise the "surface" points would invert past one
    // another and the shaft would point the wrong way.
    geometry_msgs::msg::Point surface_start = start;
    geometry_msgs::msg::Point surface_end = end;
    if (segment_length_m > kMinSegmentLengthM) {
      const double source_frac = std::min(0.5, source_radius_m / segment_length_m);
      const double target_frac = std::min(0.5, target_radius_m / segment_length_m);
      surface_start.x = start.x + dx * source_frac;
      surface_start.y = start.y + dy * source_frac;
      surface_start.z = start.z + dz * source_frac;
      surface_end.x = end.x - dx * target_frac;
      surface_end.y = end.y - dy * target_frac;
      surface_end.z = end.z - dz * target_frac;
    }
    // Fixed absolute head size for every contact arrow, regardless of this
    // segment's own length -- deliberately not derived from segment_length_m,
    // so a short edge and a long edge sprout the same-looking arrowhead.
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
      arrow.scale.x = edge_width_m_;                          // shaft diameter
      arrow.scale.y = object_contact_arrow_head_diameter_m_;  // head diameter
      arrow.scale.z = object_contact_arrow_head_length_m_;    // head length
      arrow.color.r = color.r;
      arrow.color.g = color.g;
      arrow.color.b = color.b;
      arrow.color.a = color.a;
      addMarker(markers, next_keys, std::move(arrow));
    };
    make_arrow(markerIdForPair(source, target), surface_start, surface_end);
    make_arrow(markerIdForPair(target, source), surface_end, surface_start);
  }

  void SemanticSceneGraphFuser::appendObjectContactLabel(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
                                const builtin_interfaces::msg::Time& stamp, NodeId source, NodeId target,
                                const geometry_msgs::msg::Point& start,
                                const geometry_msgs::msg::Point& end,
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
    line << centroid_distance_m << "m";
    text.text = line.str();
    addMarker(markers, next_keys, std::move(text));
  }

  void SemanticSceneGraphFuser::appendEdgeMarkers(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
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
          appendObjectContactArrows(markers, next_keys, frame, stamp, edge.source, edge.target, start, end,
                                    0.5 * objectSphereDiameter(source_it->second),
                                    0.5 * objectSphereDiameter(target_it->second));
          appendObjectContactLabel(markers, next_keys, frame, stamp, edge.source, edge.target, start,
                                   end, edge.contact_centroid_distance_m);
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

}  // namespace rsg
