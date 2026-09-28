/**
 * @file fuser_ingestion.cpp
 * @brief Message ingestion: DSG/label/presence/risk callbacks (cache under a short lock, mark dirty).
 */
#include "fuser.hpp"

namespace rsg {

  bool SemanticSceneGraphFuser::layerVisible(LayerKind kind) const {
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

  void SemanticSceneGraphFuser::handleDsg(const DsgUpdate::SharedPtr msg) {
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

  void SemanticSceneGraphFuser::applyDsgDeletions(const DsgUpdate& msg) {
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

  void SemanticSceneGraphFuser::handleTrackDeleted(const Json& payload) {
    const auto raw_slot = payload.value("hydra_slot_id", payload.value("slot_id", 0));
    const int64_t slot = raw_slot;
    if (slot <= 0 || slot > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return;
    }
    {
      std::lock_guard<std::mutex> labels_lock(overlays_mutex_);
      deleted_slot_ids_.insert(static_cast<uint32_t>(slot));
    }
    render_dirty_.store(true, std::memory_order_release);
  }

  void SemanticSceneGraphFuser::handleSemanticLabel(const std_msgs::msg::String::SharedPtr msg) {
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
    if (event == "track_deleted") {
      handleTrackDeleted(payload);
      return;
    }
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
    overlay.object_detail = payload.value("object_detail", std::string());
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
    // Deliberately "last_bbox_3d_*", not "bbox_3d_*": the latter is the
    // track's accumulated envelope (monotonic min/max over every observation
    // ever seen -- see PersistentObjectTrack._update_track_geometry), which
    // for a genuinely moving object spans its entire travelled path, not its
    // current extent. "last_bbox_3d_*" is the single most recent observation,
    // never accumulated -- the only one of the two that means anything for a
    // marker meant to show where the object actually is right now.
    const auto bbox_min_it = payload.find("last_bbox_3d_min");
    const auto bbox_max_it = payload.find("last_bbox_3d_max");
    if (bbox_min_it != payload.end() && bbox_max_it != payload.end()) {
      overlay.has_bbox = parseVector3(*bbox_min_it, overlay.bbox_min) &&
                          parseVector3(*bbox_max_it, overlay.bbox_max);
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

  void SemanticSceneGraphFuser::handleActiveSegments(const std_msgs::msg::String::SharedPtr msg) {
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
      obs.is_restored = segment.value("restored_from_previous_session", false);
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

  void SemanticSceneGraphFuser::handleRiskResult(const std_msgs::msg::String::SharedPtr msg) {
    Json payload;
    try {
      payload = Json::parse(msg->data);
    } catch (const std::exception&) {
      ++invalid_risk_messages_;
      return;
    }
    if (!payload.is_object() || payload.value("event", std::string()) != "risk_result") {
      ++invalid_risk_messages_;
      return;
    }

    RiskOverlay overlay;
    try {
      const auto raw_slot = payload.value("hydra_slot_id", 0);
      const int64_t slot = raw_slot;
      if (slot <= 0 || slot > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        ++invalid_risk_messages_;
        return;
      }
      overlay.slot_id = static_cast<uint32_t>(slot);
    } catch (const std::exception&) {
      ++invalid_risk_messages_;
      return;
    }
    try {
      // Signed range: negative means the object actively reduces risk
      // (safety equipment, hazard-warning signage), 0 is neutral, positive
      // is a hazard -- see RiskVlmBackend's prompt / _normalise_risk_score.
      overlay.risk_score = clampValue(payload.value("risk_score", 0.0), -1.0, 1.0);
    } catch (const std::exception&) {
      overlay.risk_score = 0.0;
    }
    const auto factors_it = payload.find("risk_factors");
    if (factors_it != payload.end() && factors_it->is_array()) {
      for (const auto& factor : *factors_it) {
        if (factor.is_string()) {
          overlay.risk_factors.push_back(factor.get<std::string>());
        }
      }
    }
    overlay.source = payload.value("source", std::string("none"));
    try {
      overlay.timestamp_sec = payload.value("timestamp_sec", 0.0);
    } catch (const std::exception&) {
      overlay.timestamp_sec = 0.0;
    }

    {
      std::lock_guard<std::mutex> lock(risk_overlays_mutex_);
      risk_overlays_by_slot_[overlay.slot_id] = overlay;
    }
    ++accepted_risk_messages_;
    render_dirty_.store(true, std::memory_order_release);
  }

}  // namespace rsg
