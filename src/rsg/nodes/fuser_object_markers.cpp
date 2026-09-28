/**
 * @file fuser_object_markers.cpp
 * @brief RViz marker construction for object/room/place/building nodes.
 */
#include "fuser.hpp"

namespace rsg {

  std::string SemanticSceneGraphFuser::objectDisplayLabel(
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

    // Prepended, not appended: the object id is the one line useful for
    // cross-referencing this node against other diagnostics (bbox_diagnostics
    // track_id, VLM crop folders, etc.) by eye, so it belongs first, above
    // the label/confidence/mobility lines, not buried at the bottom.
    if (show_slot_ids_ && slot_id > 0U) {
      label = "id_" + std::to_string(slot_id) + "\n" + label;
    }

    if (show_label_confidence_ && overlay) {
      std::ostringstream line;
      line.setf(std::ios::fixed);
      line.precision(2);
      line << label << "(" << overlay->confidence << ")";
      label = line.str();
    }
    if (show_mobility_metadata_ && overlay) {
      std::ostringstream line;
      line.setf(std::ios::fixed);
      line.precision(2);
      line << mobility_class << "(" << overlay->mobility_confidence << ")";
      label += "\n" + line.str();
    }
    if (show_object_detail_ && overlay && !overlay->object_detail.empty()) {
      label += "\n" + overlay->object_detail;
    }
    if (show_presence_confidence_ && slot_id > 0U) {
      const ResolvedPresence presence_state = resolvePresenceForSlot(slot_id, presence, mobility_class);
      if (presence_state.observation.slot_id == 0U) {
        label += "\npresence n/a";
      } else {
        std::ostringstream line;
        line.setf(std::ios::fixed);
        line.precision(2);
        const char* presence_tag = presence_state.state == "OBSERVED" ? "obs "
                                   : presence_state.state == "RESTORED" ? "rst "
                                                                        : "dec ";
        line << "presence " << presence_tag << presence_state.confidence;
        label += "\n" + line.str();
      }
    }
    return label;
  }

  Color SemanticSceneGraphFuser::objectDisplayColor(
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
      // A restored-but-not-yet-reobserved slot's confidence is deliberately
      // near-zero (see resolvePresenceForSlot) -- correct for recency, wrong
      // for opacity. Its day-plus save-time shift is not a session's worth of
      // real inactivity; it just hasn't been checked yet this run. Render it
      // fully opaque instead of applying the same alpha the decay curve would
      // give an object that has genuinely sat unseen this long.
      const double alpha_confidence =
          presence_state.observation.is_restored ? 1.0 : presence_state.confidence;
      color.a = std::max(
          minimum_object_alpha_,
          static_cast<float>(static_cast<double>(color.a) * alpha_confidence));
    }
    return color;
  }

  void SemanticSceneGraphFuser::addMarker(MarkerArray& markers, MarkerSet& next_keys, Marker marker) const {
    const MarkerKey key{marker.ns, marker.id};
    if (!next_keys.insert(key).second) {
      // Stable IDs must be unique within a namespace. Do not emit duplicate
      // marker keys; a later node could otherwise overwrite an earlier one.
      return;
    }
    markers.markers.push_back(std::move(marker));
  }

  void SemanticSceneGraphFuser::publishMarkerArray(const SceneModel& model,
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

  double SemanticSceneGraphFuser::objectSphereDiameter(const NodeView& node) const {
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

  void SemanticSceneGraphFuser::appendObjectMarkers(MarkerArray& markers, MarkerSet& next_keys, const std::string& frame,
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

    // Halo: a translucent grey sphere around anything recently observed.
    // Driven by the same presence confidence that sets alpha above,
    // thresholded high (default 0.995) rather than gated on a separate age
    // cutoff. This falls out of the decay formula almost for free:
    // confidence is exactly 1.0 while inside presence_observed_epsilon_sec
    // (freshly seen this frame), and clears the threshold for only a few
    // seconds after -- so it reads as "in view right now" without needing
    // its own age parameter.
    //
    // Deliberately uses the RAW confidence, not objectDisplayColor's alpha
    // value: a restored-but-not-yet-reobserved slot is rendered fully opaque
    // (see objectDisplayColor) so it isn't invisible, but its true recency is
    // still near zero -- it must not glow just because it's known map
    // content that hasn't actually been checked yet this session.
    if (highlight_active_objects_) {
      const SemanticOverlay* halo_overlay = overlayForNode(node, resolved);
      const uint32_t halo_slot = halo_overlay ? halo_overlay->slot_id : node.semantic_slot;
      const std::string halo_mobility = halo_overlay ? halo_overlay->mobility_class : "unknown";
      const ResolvedPresence halo_presence = resolvePresenceForSlot(halo_slot, presence, halo_mobility);
      if (halo_presence.observation.slot_id > 0U &&
          halo_presence.confidence > active_object_halo_min_confidence_) {
        Marker halo;
        halo.header.frame_id = frame;
        halo.header.stamp = stamp;
        halo.ns = "rsg_active_objects";
        halo.id = id;
        halo.type = Marker::SPHERE;
        halo.action = Marker::ADD;
        halo.pose.position.x = display_position.x();
        halo.pose.position.y = display_position.y();
        halo.pose.position.z = display_position.z();
        halo.pose.orientation.w = 1.0;
        const double halo_diameter =
            std::max(marker_diameter * active_object_halo_scale_,
                     marker_diameter + active_object_halo_min_margin_m_);
        halo.scale.x = halo_diameter;
        halo.scale.y = halo_diameter;
        halo.scale.z = halo_diameter;
        halo.color.r = 0.6F;
        halo.color.g = 0.6F;
        halo.color.b = 0.6F;
        halo.color.a = static_cast<float>(active_object_halo_alpha_);
        addMarker(markers, next_keys, std::move(halo));
      }
    }

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

  Color SemanticSceneGraphFuser::placeDisplayColor(const NodeView& node, const LayeredProjection& projection) const {
    const auto parent = projection.place_to_room.find(node.id);
    if (parent == projection.place_to_room.end()) {
      return Color{0.45F, 0.45F, 0.45F, place_alpha_};
    }
    return roomDisplayColor(parent->second, place_alpha_);
  }

  void SemanticSceneGraphFuser::appendLayerNodeMarker(MarkerArray& markers, MarkerSet& next_keys,
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

  void SemanticSceneGraphFuser::appendSemanticVolumeMarkers(MarkerArray& markers, MarkerSet& next_keys,
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

  void SemanticSceneGraphFuser::appendOptionalLayerMarker(MarkerArray& markers, MarkerSet& next_keys,
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

}  // namespace rsg
