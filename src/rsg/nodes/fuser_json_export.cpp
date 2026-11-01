/**
 * @file fuser_json_export.cpp
 * @brief Periodic JSON export of the fused scene graph with all metadata.
 *
 * Built from the same model, overlays, presence and projection the RViz
 * markers are drawn from, so it matches what RViz shows.
 */
#include "fuser.hpp"

namespace rsg {

  namespace {

    Json timeToJson(const builtin_interfaces::msg::Time& stamp) {
      return {{"sec", stamp.sec}, {"nanosec", stamp.nanosec}};
    }

  }  // namespace

  namespace {

    const char* layerKindName(LayerKind kind) {
      switch (kind) {
        case LayerKind::kObjects: return "object";
        case LayerKind::kRooms: return "room";
        case LayerKind::kBuildings: return "building";
        case LayerKind::kPlaces: return "place";
        case LayerKind::kSegments: return "segment";
        case LayerKind::kAgents: return "agent";
        default: return "other";
      }
    }

    const char* edgeTypeName(DisplayEdgeType type) {
      switch (type) {
        case DisplayEdgeType::kRoomPlaceHierarchy: return "room_place_hierarchy";
        case DisplayEdgeType::kPlaceObjectMembership: return "place_object_membership";
        case DisplayEdgeType::kPlaceConnectivity: return "place_connectivity";
        case DisplayEdgeType::kNativeObjectObjectDebug: return "native_object_object";
        case DisplayEdgeType::kNativeRoomRoomDebug: return "native_room_room";
        case DisplayEdgeType::kDerivedObjectContact: return "derived_object_contact";
        case DisplayEdgeType::kDerivedObjectSegment: return "derived_object_segment";
      }
      return "unknown";
    }

    const char* edgeOriginName(EdgeOrigin origin) {
      switch (origin) {
        case EdgeOrigin::kNativeHydraEdge: return "native_hydra_edge";
        case EdgeOrigin::kDerivedMeshValidatedLocalPlace: return "derived_mesh_validated_local_place";
        case EdgeOrigin::kDerivedVisibleNeighbourRoom: return "derived_visible_neighbour_room";
      }
      return "unknown";
    }

    Json vec3ToJson(const Eigen::Vector3d& v) {
      return {{"x", v.x()}, {"y", v.y()}, {"z", v.z()}};
    }

    Json vec3ToJson(const Eigen::Vector3f& v) {
      return {{"x", v.x()}, {"y", v.y()}, {"z", v.z()}};
    }

    Json overlayToJson(const ResolvedOverlay& resolved) {
      const auto& o = resolved.overlay;
      Json out = {
          {"slot_id", o.slot_id},
          {"label", o.label},
          {"confidence", o.confidence},
          {"mobility_class", o.mobility_class},
          {"mobility_confidence", o.mobility_confidence},
          {"mobility_source", o.mobility_source},
          {"object_detail", o.object_detail},
          {"source", o.source},
          {"timestamp_sec", o.timestamp_sec},
          {"association", resolved.association},
          {"centroid_distance_m", resolved.centroid_distance_m},
          {"has_centroid", o.has_centroid},
          {"centroid_frame_id", o.centroid_frame_id},
          {"has_bbox", o.has_bbox},
      };
      if (o.has_centroid) {
        out["centroid"] = vec3ToJson(o.centroid);
      }
      if (o.has_bbox) {
        out["bbox_min"] = vec3ToJson(o.bbox_min);
        out["bbox_max"] = vec3ToJson(o.bbox_max);
      }
      return out;
    }

    Json presenceToJson(const ResolvedPresence& p) {
      const auto& obs = p.observation;
      Json out = {
          {"state", p.state},
          {"confidence", p.confidence},
          {"age_sec", p.age_sec},
          {"slot_id", obs.slot_id},
          {"internal_object_id", obs.internal_object_id},
          {"persistent_track_id", obs.persistent_track_id},
          {"local_segment_id", obs.local_segment_id},
          {"last_observed_timestamp_sec", obs.last_observed_timestamp_sec},
          {"local_segment_xy_span_m", obs.local_segment_xy_span_m},
          {"is_restored", obs.is_restored},
          {"has_centroid", obs.has_centroid},
          {"has_bbox", obs.has_bbox},
          {"raw", obs.raw},
      };
      if (obs.has_centroid) {
        out["centroid"] = vec3ToJson(obs.centroid);
      }
      if (obs.has_bbox) {
        out["bbox_min"] = vec3ToJson(obs.bbox_min);
        out["bbox_max"] = vec3ToJson(obs.bbox_max);
      }
      return out;
    }

    Json colorFromStruct(const Color& c) {
      return {{"r", c.r}, {"g", c.g}, {"b", c.b}, {"a", c.a}};
    }

  }  // namespace

  Json SemanticSceneGraphFuser::buildSceneJson(const SceneModel& model,
                                               const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                                               const PresenceCache& presence,
                                               const LayeredProjection& projection) const {
    Json nodes = Json::array();
    for (const auto& [node_id, node] : model.nodes) {
      if (!node.visible) {
        continue;
      }
      Json entry = {
          {"id", idString(node_id)},
          {"symbol", spark_dsg::NodeSymbol(node_id).str()},
          {"marker_id", markerId(node_id)},
          {"layer", layerKindName(node.kind)},
          {"name", node.name},
          {"visible", node.visible},
          {"is_active", node.is_active},
          {"semantic_slot", node.semantic_slot},
          {"position", vec3ToJson(node.position)},
          {"display_position", vec3ToJson(displayPosition(node))},
          {"has_bbox", node.has_bbox},
      };
      if (node.has_bbox) {
        entry["bbox"] = {{"center", vec3ToJson(node.bbox_center)}, {"size", vec3ToJson(node.bbox_size)}};
      }

      const auto adjacency = model.adjacency.find(node_id);
      Json neighbours = Json::array();
      if (adjacency != model.adjacency.end()) {
        for (const NodeId neighbour : adjacency->second) {
          neighbours.push_back(idString(neighbour));
        }
      }
      entry["neighbours"] = std::move(neighbours);

      if (isObject(node)) {
        const SemanticOverlay* overlay = overlayForNode(node, resolved);
        const auto resolved_it = resolved.find(node_id);
        entry["semantic"] = resolved_it != resolved.end() ? overlayToJson(resolved_it->second) : Json(nullptr);
        const std::string mobility = overlay ? overlay->mobility_class : "unknown";
        const uint32_t slot = overlay ? overlay->slot_id : node.semantic_slot;
        entry["presence"] = slot > 0U ? presenceToJson(resolvePresenceForSlot(slot, presence, mobility))
                                      : Json(nullptr);
        entry["display"] = {
            {"label", objectDisplayLabel(node, resolved, presence)},
            {"color", colorFromStruct(objectDisplayColor(node, resolved, presence))},
            {"diameter_m", objectSphereDiameter(node)},
        };
        const auto place = projection.object_to_place.find(node_id);
        if (place != projection.object_to_place.end()) {
          entry["parent_place"] = idString(place->second);
          const auto origin = projection.object_place_origin.find(node_id);
          if (origin != projection.object_place_origin.end()) {
            entry["parent_place_origin"] = edgeOriginName(origin->second);
          }
        }
      } else if (isRoom(node)) {
        entry["display"] = {
            {"label", roomDisplayLabel(node_id)},
            {"ordinal", roomDisplayOrdinal(node_id)},
            {"color", colorFromStruct(roomDisplayColor(node_id, room_alpha_))},
        };
      } else if (isPlace(node)) {
        entry["display"] = {{"color", colorFromStruct(placeDisplayColor(node, projection))}};
        const auto room = projection.place_to_room.find(node_id);
        if (room != projection.place_to_room.end()) {
          entry["parent_room"] = idString(room->second);
          const auto origin = projection.place_room_origin.find(node_id);
          if (origin != projection.place_room_origin.end()) {
            entry["parent_room_origin"] = edgeOriginName(origin->second);
          }
        }
      }

      // Everything the DSG itself carries for this node, including the
      // rsg_rap / rsg_presence / rsg_identity / rsg_risk blocks the fuser
      // synchronised into it just before rendering.
      if (graph_ && graph_->hasNode(node_id)) {
        const auto& attrs = graph_->getNode(node_id).attributes();
        Json dsg_metadata = attrs.metadata.get();
        entry["dsg_metadata"] = dsg_metadata.is_null() ? Json::object() : dsg_metadata;
        entry["dsg_is_active"] = attrs.is_active;
        entry["dsg_last_update_time_ns"] = attrs.last_update_time_ns;
        if (dsg_metadata.is_object() && dsg_metadata.contains("rsg_risk")) {
          entry["risk"] = dsg_metadata["rsg_risk"];
        }
      }
      nodes.push_back(std::move(entry));
    }

    Json edges = Json::array();
    for (const auto& edge : projection.edges) {
      const auto source_it = model.nodes.find(edge.source);
      const auto target_it = model.nodes.find(edge.target);
      if (source_it == model.nodes.end() || target_it == model.nodes.end()) {
        continue;
      }
      const Eigen::Vector3d start = displayPosition(source_it->second);
      const Eigen::Vector3d end = displayPosition(target_it->second);
      Json entry = {
          {"source", idString(edge.source)},
          {"target", idString(edge.target)},
          {"type", edgeTypeName(edge.type)},
          {"origin", edgeOriginName(edge.origin)},
          {"color_owner", edge.color_owner == 0 ? Json(nullptr) : Json(idString(edge.color_owner))},
          {"length_m", (end - start).norm()},
          {"source_position", vec3ToJson(start)},
          {"target_position", vec3ToJson(end)},
      };
      if (edge.type == DisplayEdgeType::kDerivedObjectContact) {
        entry["contact_iou_2d_max"] = edge.contact_iou_2d_max;
        entry["contact_centroid_distance_m"] = edge.contact_centroid_distance_m;
      }
      if (graph_) {
        const auto* dsg_edge = graph_->findEdge(edge.source, edge.target);
        if (dsg_edge && dsg_edge->info) {
          entry["dsg_edge"] = {
              {"weighted", dsg_edge->info->weighted},
              {"weight", dsg_edge->info->weight},
              {"metadata", dsg_edge->info->metadata.get().is_null() ? Json::object()
                                                                     : dsg_edge->info->metadata.get()},
          };
        }
      }
      edges.push_back(std::move(entry));
    }

    return {
        {"frame_id", latest_header_.frame_id.empty() ? fallback_frame_id_ : latest_header_.frame_id},
        {"stamp", timeToJson(latest_header_.stamp)},
        {"sequence_number", latest_sequence_},
        {"node_count", nodes.size()},
        {"edge_count", edges.size()},
        {"nodes", std::move(nodes)},
        {"edges", std::move(edges)},
        {"projection",
         {{"mesh_validation_available", projection.mesh_validation_available},
          {"local_index_candidates_examined", projection.local_index_candidates_examined},
          {"mesh_rejected_candidates", projection.mesh_rejected_candidates},
          {"validated_local_place_associations", projection.validated_local_place_associations},
          {"local_association_cache_hits", projection.local_association_cache_hits},
          {"fallback_suppressed_without_mesh", projection.fallback_suppressed_without_mesh},
          {"room_completion_candidates_examined", projection.room_completion_candidates_examined},
          {"room_completion_mesh_rejected_candidates", projection.room_completion_mesh_rejected_candidates},
          {"derived_room_place_associations", projection.derived_room_place_associations},
          {"room_completion_ties", projection.room_completion_ties},
          {"room_completion_suppressed_without_mesh", projection.room_completion_suppressed_without_mesh},
          {"room_completion_waiting_for_grace", projection.room_completion_waiting_for_grace},
          {"room_completion_grace_elapsed_sec", projection.room_completion_grace_elapsed_sec},
          {"room_completion_grace_remaining_sec", projection.room_completion_grace_remaining_sec},
          {"room_completion_suppressed_by_grace", projection.room_completion_suppressed_by_grace}}},
    };
  }

  void SemanticSceneGraphFuser::publishGraphJson(const SceneModel& model,
                                                 const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                                                 const PresenceCache& presence,
                                                 const LayeredProjection& projection) {
    if (graph_json_period_sec_ <= 0.0 || !graph_json_pub_) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (graph_json_publications_ > 0 &&
        std::chrono::duration<double>(now - last_graph_json_publish_).count() < graph_json_period_sec_) {
      return;
    }
    Json graph = buildSceneJson(model, resolved, presence, projection);
    graph["event"] = "hydra_rap_fuser_graph";
    std_msgs::msg::String message;
    message.data = graph.dump();
    graph_json_pub_->publish(message);
    ++graph_json_publications_;
    last_graph_json_publish_ = now;
  }

}  // namespace rsg
