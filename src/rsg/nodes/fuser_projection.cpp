/**
 * @file fuser_projection.cpp
 * @brief Layered display projection assembly: ties model + overlays + spatial association together.
 */
#include "fuser.hpp"

namespace rsg {

  LayeredProjection SemanticSceneGraphFuser::buildLayeredProjection(const SceneModel& model) {
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

  void SemanticSceneGraphFuser::syncRoomDisplayOrdinals(const SceneModel& model) {
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

  uint32_t SemanticSceneGraphFuser::roomDisplayOrdinal(NodeId room_id) const {
    const auto found = room_display_ordinals_.find(room_id);
    return found == room_display_ordinals_.end() ? 0U : found->second;
  }

  std::string SemanticSceneGraphFuser::roomDisplayLabel(NodeId room_id) const {
    const uint32_t ordinal = roomDisplayOrdinal(room_id);
    return ordinal == 0U ? "Room" : "Room" + std::to_string(ordinal);
  }

  Color SemanticSceneGraphFuser::roomDisplayColor(NodeId room_id, float alpha) const {
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

  Color SemanticSceneGraphFuser::blackEdgeColor(float alpha) {
    return Color{0.0F, 0.0F, 0.0F, alpha};
  }

  Color SemanticSceneGraphFuser::objectContactEdgeColor(float alpha) {
    return Color{0.91F, 0.64F, 0.24F, alpha};
  }

  Color SemanticSceneGraphFuser::objectSegmentEdgeColor(float alpha) {
    return Color{0.20F, 0.55F, 0.90F, alpha};
  }

}  // namespace rsg
