/**
 * @file fuser_spatial_index.cpp
 * @brief Place spatial-hash index and mesh wall index used for local place association.
 */
#include "fuser.hpp"

namespace rsg {

  Eigen::Vector3d SemanticSceneGraphFuser::sourcePosition(const NodeView& node) const {
    if (node.has_bbox) {
      return node.bbox_center.cast<double>();
    }
    return node.position;
  }

  Eigen::Vector3d SemanticSceneGraphFuser::displayPosition(const NodeView& node) const {
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

  GridKey SemanticSceneGraphFuser::gridKey(const Eigen::Vector3d& position, double cell_size_m) const {
    return GridKey{
        static_cast<int>(std::floor(position.x() / cell_size_m)),
        static_cast<int>(std::floor(position.y() / cell_size_m)),
        static_cast<int>(std::floor(position.z() / cell_size_m)),
    };
  }

  uint64_t SemanticSceneGraphFuser::hashCombine(uint64_t seed, uint64_t value) const {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed;
  }

  uint64_t SemanticSceneGraphFuser::placeFingerprint(const SceneModel& model, const std::vector<NodeId>& place_ids) const {
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

  void SemanticSceneGraphFuser::ensurePlaceSpatialIndex(const SceneModel& model, const std::vector<NodeId>& place_ids) {
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

  uint64_t SemanticSceneGraphFuser::meshFingerprint(const spark_dsg::Mesh& mesh) const {
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

  void SemanticSceneGraphFuser::indexTriangle(size_t triangle_index, const MeshTriangle& triangle) {
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

  bool SemanticSceneGraphFuser::ensureMeshWallIndex() {
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

  bool SemanticSceneGraphFuser::segmentIntersectsTriangle(const Eigen::Vector3d& start,
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

  bool SemanticSceneGraphFuser::meshSegmentIsClear(const Eigen::Vector3d& start, const Eigen::Vector3d& end) const {
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

  Eigen::Vector3d SemanticSceneGraphFuser::objectFreeSpaceAnchor(const NodeView& object,
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

  std::vector<std::pair<NodeId, double>> SemanticSceneGraphFuser::localPlaceCandidates(
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

  bool SemanticSceneGraphFuser::cachedAssociationUsable(const SceneModel& model, const NodeView& object,
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

  std::optional<NodeId> SemanticSceneGraphFuser::findValidatedLocalPlace(const SceneModel& model,
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

}  // namespace rsg
