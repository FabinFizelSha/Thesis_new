/**
 * @file fuser_model.cpp
 * @brief Scene-model collection: build the SceneModel snapshot from the local Hydra DSG clone.
 */
#include "fuser.hpp"

namespace rsg {

  void SemanticSceneGraphFuser::renderDirtyState() {
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
    RiskOverlayCache risk_overlay_snapshot;
    {
      std::lock_guard<std::mutex> risk_lock(risk_overlays_mutex_);
      risk_overlay_snapshot = risk_overlays_by_slot_;
    }
    PresenceCache presence_snapshot;
    {
      std::lock_guard<std::mutex> presence_lock(presence_mutex_);
      presence_snapshot = presence_by_slot_;
    }

    if (!publishFusedOutputs(false, "bounded_fused_graph_refresh", overlay_snapshot, risk_overlay_snapshot, presence_snapshot)) {
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

  SceneModel SemanticSceneGraphFuser::collectModel(const OverlayCache& overlay_snapshot) const {
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

    // Drop any object node whose Hydra slot was explicitly deleted by phase1
    // (see handleTrackDeleted). Single choke point: every downstream
    // consumer of `model` (markers, edges, exports) already reads from here,
    // so nothing else needs its own suppression check.
    if (!deleted_slot_ids_.empty()) {
      std::unordered_set<uint32_t> deleted_snapshot;
      {
        std::lock_guard<std::mutex> labels_lock(overlays_mutex_);
        deleted_snapshot = deleted_slot_ids_;
      }
      for (auto it = model.nodes.begin(); it != model.nodes.end();) {
        if (isObject(it->second) && deleted_snapshot.count(it->second.semantic_slot) > 0) {
          it = model.nodes.erase(it);
        } else {
          ++it;
        }
      }
    }

    // Collapse object nodes that are the same physical object. Done here, before
    // edges are collected, so every downstream consumer -- markers, contact
    // edges, segment edges, node metadata -- sees one node per object.
    const auto collapsed = collapseDuplicateObjects(model);

    std::set<std::pair<NodeId, NodeId>> seen_edges;
    const auto collect_edges = [&model, &seen_edges, &collapsed](const auto& edges) {
      for (const auto& [edge_key, edge] : edges) {
        (void)edge_key;
        // Rewire onto the survivor rather than dropping the edge, so a real
        // relationship (object->place, contact) is not lost with the duplicate.
        auto source = edge.source;
        auto target = edge.target;
        const auto src_it = collapsed.find(source);
        if (src_it != collapsed.end()) {
          source = src_it->second;
        }
        const auto tgt_it = collapsed.find(target);
        if (tgt_it != collapsed.end()) {
          target = tgt_it->second;
        }
        if (source == target) {
          continue;  // both sides collapsed into the same node
        }
        if (!model.nodes.count(source) || !model.nodes.count(target)) {
          continue;
        }
        const auto key = std::make_pair(std::min(source, target), std::max(source, target));
        if (!seen_edges.insert(key).second) {
          continue;
        }
        model.raw_edges.push_back(RawEdge{source, target});
        model.adjacency[source].push_back(target);
        model.adjacency[target].push_back(source);
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

    // Render dynamic objects phase1 confirmed but Hydra never formed a mesh
    // node for. MeshSegmenter only creates a node when 50+ vertices of one
    // label appear together within a single ~0.4s update pass -- there is no
    // accumulation across passes for a not-yet-existing node, so a genuinely
    // moving object's labeled mesh region can simply never clear that
    // one-shot bar, independent of how correctly/confidently phase1
    // classified it (confirmed: several VLM-confirmed "person" detections at
    // 0.95 confidence, zero matching Hydra object nodes for any of them).
    //
    // Injected as an ordinary object NodeView -- never added to graph_ itself
    // -- so every existing consumer (markers via appendObjectMarkers, contact
    // edges via computeObjectContacts/buildMainObjectGroups, overlay
    // resolution via resolveOverlays, presence fading) treats it exactly like
    // a real Hydra node with zero special-casing. Uses geometry from
    // last_bbox_3d_min/max (see SemanticOverlay's bbox fields), the single
    // most recent observation, never the track's accumulated envelope, which
    // for a moving object would span its entire travelled path. A disjoint
    // NodeSymbol prefix ('D') guarantees this can never collide with Hydra's
    // own object ids (prefix 'O', see MeshSegmenter::kNodePrefix).
    if (!overlay_snapshot.empty()) {
      std::unordered_set<uint32_t> occupied_slots;
      for (const auto& [existing_id, existing_node] : model.nodes) {
        (void)existing_id;
        if (isObject(existing_node) && existing_node.semantic_slot > 0U) {
          occupied_slots.insert(existing_node.semantic_slot);
        }
      }
      std::unordered_set<uint32_t> deleted_snapshot;
      {
        std::lock_guard<std::mutex> labels_lock(overlays_mutex_);
        deleted_snapshot = deleted_slot_ids_;
      }
      for (const auto& [slot_id, candidates] : overlay_snapshot) {
        if (slot_id == 0U || candidates.empty() || occupied_slots.count(slot_id) > 0U ||
            deleted_snapshot.count(slot_id) > 0U) {
          continue;
        }
        const SemanticOverlay* best = &candidates.front();
        for (const auto& candidate : candidates) {
          if (candidate.timestamp_sec > best->timestamp_sec) {
            best = &candidate;
          }
        }
        if (best->mobility_class != "dynamic" || !best->has_bbox || !usableLabel(best->label)) {
          continue;
        }
        const Eigen::Vector3d dims = (best->bbox_max - best->bbox_min).cwiseAbs();
        if (dims.x() <= 0.0 || dims.y() <= 0.0 || dims.z() <= 0.0) {
          continue;
        }
        const Eigen::Vector3d center = (best->bbox_min + best->bbox_max) * 0.5;

        NodeView view;
        view.id = spark_dsg::NodeSymbol('D', slot_id);
        view.kind = LayerKind::kObjects;
        view.visible = layerVisible(LayerKind::kObjects);
        view.semantic_slot = slot_id;
        view.position = center;
        view.is_active = true;
        view.has_bbox = true;
        view.bbox_center = center.cast<float>();
        view.bbox_size = dims.cast<float>();
        model.nodes.emplace(view.id, std::move(view));
      }
    }

    return model;
  }

  bool SemanticSceneGraphFuser::bboxContains(const NodeView& node, const Eigen::Vector3d& point) {
    const Eigen::Vector3d center = node.bbox_center.cast<double>();
    const Eigen::Vector3d half = node.bbox_size.cast<double>() / 2.0;
    const Eigen::Vector3d delta = (point - center).cwiseAbs();
    return (delta.array() <= half.array()).all();
  }

  std::unordered_map<NodeId, NodeId> SemanticSceneGraphFuser::collapseDuplicateObjects(SceneModel& model) const {
    std::unordered_map<NodeId, NodeId> collapsed;
    if (!object_slot_collapse_enabled_) {
      return collapsed;
    }

    std::unordered_map<uint32_t, std::vector<NodeId>> by_slot;
    for (const auto& [node_id, node] : model.nodes) {
      if (node.kind == LayerKind::kObjects && node.semantic_slot > 0) {
        by_slot[node.semantic_slot].push_back(node_id);
      }
    }

    size_t absorbed_total = 0;
    for (auto& [slot_id, ids] : by_slot) {
      if (ids.size() < 2) {
        continue;
      }
      // Node ids increase monotonically, so lowest is oldest.
      std::sort(ids.begin(), ids.end());

      // Only a node in the robot's current view is ever removed. A node
      // restored from a previous session is archived, and archived nodes are
      // left exactly as they were loaded -- collapsing those would delete map
      // the robot is not even looking at, and a slot legitimately holding two
      // archived nodes (a large object Hydra clustered in two pieces last run)
      // would silently lose one of them.
      const NodeId survivor_id = *std::min_element(
          ids.begin(), ids.end(), [&model](NodeId lhs, NodeId rhs) {
            const bool lhs_archived = !model.nodes.at(lhs).is_active;
            const bool rhs_archived = !model.nodes.at(rhs).is_active;
            if (lhs_archived != rhs_archived) {
              return lhs_archived;  // an archived node always outranks an active one
            }
            return lhs < rhs;
          });
      auto& survivor = model.nodes.at(survivor_id);
      // Restored geometry stays as loaded: the point of a resume is that the
      // object keeps the extent it was saved with, rather than being reshaped
      // by a fresh partial view of it.
      const bool survivor_restored = !survivor.is_active;

      for (const NodeId other_id : ids) {
        if (other_id == survivor_id) {
          continue;
        }
        const auto& other = model.nodes.at(other_id);
        if (!other.is_active) {
          continue;  // archived: loaded from memory, left alone
        }

        // Proximity guard: only collapse nodes that really are co-located.
        // Same slot should already imply same object, but if Phase 1 ever
        // re-used a slot for something genuinely elsewhere, this keeps the two
        // apart rather than fusing distant geometry into one box.
        //
        // Either box containing the other's centroid is the same predicate
        // Hydra's own UpdateObjectsFunctor::findMerges uses, and it is the one
        // that carries large objects: a wall or floor seen from a new angle can
        // move its centroid metres while the two boxes still plainly overlap.
        // Centroid distance is the fallback for small or box-less nodes.
        // When the survivor's box grows as it absorbs (below, active-only
        // slots), this becomes transitive across a slot's members rather than
        // pairwise-to-first. A restored survivor keeps its saved box, so there
        // it stays a straight comparison against the extent from last run.
        const bool boxes_overlap =
            (survivor.has_bbox && bboxContains(survivor, other.position)) ||
            (other.has_bbox && bboxContains(other, survivor.position));
        const double distance = (survivor.position - other.position).norm();
        if (!boxes_overlap && distance > object_slot_collapse_max_distance_m_) {
          continue;
        }

        // Union the bounding boxes so the surviving object covers everything it
        // absorbed, rather than the newer partial observation replacing it.
        // Skipped when the survivor was restored, which keeps its saved extent.
        if (other.has_bbox && !survivor_restored) {
          if (!survivor.has_bbox) {
            survivor.has_bbox = true;
            survivor.bbox_center = other.bbox_center;
            survivor.bbox_size = other.bbox_size;
          } else {
            const Eigen::Vector3f a_min = survivor.bbox_center - survivor.bbox_size / 2.0F;
            const Eigen::Vector3f a_max = survivor.bbox_center + survivor.bbox_size / 2.0F;
            const Eigen::Vector3f b_min = other.bbox_center - other.bbox_size / 2.0F;
            const Eigen::Vector3f b_max = other.bbox_center + other.bbox_size / 2.0F;
            const Eigen::Vector3f u_min = a_min.cwiseMin(b_min);
            const Eigen::Vector3f u_max = a_max.cwiseMax(b_max);
            survivor.bbox_center = (u_min + u_max) / 2.0F;
            survivor.bbox_size = u_max - u_min;
          }
        }

        model.nodes.erase(other_id);
        collapsed.emplace(other_id, survivor_id);
        ++absorbed_total;
      }
    }

    // Own counter rather than RCLCPP_*_THROTTLE: this method is const and the
    // throttle macro needs a mutable clock.
    if (absorbed_total > 0 && (collapse_log_countdown_++ % 100) == 0) {
      RCLCPP_INFO(get_logger(),
                  "object slot collapse: absorbed %zu duplicate object node(s) into "
                  "their same-slot survivors",
                  absorbed_total);
    }
    return collapsed;
  }

  void SemanticSceneGraphFuser::collectLayer(SceneModel& model, const std::string& layer_name, LayerKind kind) const {
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
      view.is_active = attrs->is_active;

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

}  // namespace rsg
