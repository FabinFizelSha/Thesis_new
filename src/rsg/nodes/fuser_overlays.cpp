/**
 * @file fuser_overlays.cpp
 * @brief Semantic-overlay resolution: match buffered RAP/VLM labels onto model nodes.
 */
#include "fuser.hpp"

namespace rsg {

  bool SemanticSceneGraphFuser::centroidUsable(const SemanticOverlay& overlay, const std::string& dsg_frame) const {
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

  std::unordered_map<NodeId, ResolvedOverlay> SemanticSceneGraphFuser::resolveOverlays(
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

  void SemanticSceneGraphFuser::pruneOwnedContactEdges() {
    if (!graph_) {
      contact_edges_prev_.clear();
      return;
    }
    for (const auto& [source, target] : contact_edges_prev_) {
      graph_->removeEdge(source, target);
    }
    contact_edges_prev_.clear();
  }

  void SemanticSceneGraphFuser::pruneOwnedSegmentEdges() {
    if (!graph_) {
      segment_edges_prev_.clear();
      return;
    }
    for (const auto& [source, target] : segment_edges_prev_) {
      graph_->removeEdge(source, target);
    }
    segment_edges_prev_.clear();
  }

}  // namespace rsg
