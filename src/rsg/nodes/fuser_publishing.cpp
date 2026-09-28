/**
 * @file fuser_publishing.cpp
 * @brief Final output publishing: fused DSG, MarkerArray, and the status/diagnostics JSON.
 */
#include "fuser.hpp"

namespace rsg {

  bool SemanticSceneGraphFuser::shouldPublishMarkers(bool force) const {
    if (force || marker_publish_rate_hz_ <= 0.0 || marker_publications_ == 0) {
      return true;
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_marker_publish_).count();
    return elapsed >= (1.0 / marker_publish_rate_hz_);
  }

  void SemanticSceneGraphFuser::publishFusedDsg() {
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

  bool SemanticSceneGraphFuser::publishFusedOutputs(bool force_markers,
                           const std::string& reason,
                           const OverlayCache& overlay_snapshot,
                           const RiskOverlayCache& risk_overlay_snapshot,
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

    const SceneModel model = collectModel(overlay_snapshot);
    const std::string frame = latest_header_.frame_id.empty() ? fallback_frame_id_ : latest_header_.frame_id;
    const auto resolved = resolveOverlays(model, frame, overlay_snapshot);
    updateLocalGraphMetadata(model, resolved, risk_overlay_snapshot, presence_snapshot);
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

  size_t SemanticSceneGraphFuser::objectNodeCount(const SceneModel& model) const {
    size_t count = 0;
    for (const auto& [node_id, node] : model.nodes) {
      (void)node_id;
      count += isObject(node) ? 1U : 0U;
    }
    return count;
  }

  void SemanticSceneGraphFuser::publishStatus(const std::string& state,
                     const std::string& detail,
                     const SceneModel* model,
                     const std::unordered_map<NodeId, ResolvedOverlay>* resolved,
                     const LayeredProjection* projection,
                     const OverlayCache* overlay_snapshot) const {
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

}  // namespace rsg
