/**
 * @file fuser_room_place.cpp
 * @brief Derived room<->place membership completion (nearest-visible-room majority vote).
 */
#include "fuser.hpp"

namespace rsg {

  int64_t SemanticSceneGraphFuser::stampNanoseconds(const builtin_interfaces::msg::Time& stamp) {
    constexpr int64_t kNanosecondsPerSecond = 1000000000LL;
    return static_cast<int64_t>(stamp.sec) * kNanosecondsPerSecond +
           static_cast<int64_t>(stamp.nanosec);
  }

  void SemanticSceneGraphFuser::updateRoomPlaceCompletionGraceClock(const std_msgs::msg::Header& header) {
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

  double SemanticSceneGraphFuser::roomPlaceCompletionGraceElapsedSec() const {
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

  bool SemanticSceneGraphFuser::roomPlaceCompletionReady(LayeredProjection& projection) const {
    const double elapsed = roomPlaceCompletionGraceElapsedSec();
    projection.room_completion_grace_elapsed_sec = elapsed;
    projection.room_completion_grace_remaining_sec = std::max(
        0.0, room_place_completion_grace_period_sec_ - elapsed);
    projection.room_completion_waiting_for_grace =
        room_place_completion_grace_period_sec_ > 0.0 &&
        elapsed < room_place_completion_grace_period_sec_;
    return !projection.room_completion_waiting_for_grace;
  }

  std::optional<NodeId> SemanticSceneGraphFuser::majorityVisibleNeighbourRoom(
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

  void SemanticSceneGraphFuser::completeMissingPlaceRoomMembership(
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

}  // namespace rsg
