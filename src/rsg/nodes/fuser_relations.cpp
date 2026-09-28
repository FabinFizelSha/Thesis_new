/**
 * @file fuser_relations.cpp
 * @brief Object-contact and object-segment edge derivation, plus their offline diagnostics dump.
 */
#include "fuser.hpp"

namespace rsg {

  std::vector<SemanticSceneGraphFuser::MainObjectGroup> SemanticSceneGraphFuser::buildMainObjectGroups(const SceneModel& model,
                                                      const PresenceCache& presence) const {
    const auto internal_id_for = [&](uint32_t slot) -> std::string {
      const auto it = presence.find(slot);
      return it == presence.end() ? std::string() : it->second.internal_object_id;
    };

    std::unordered_map<std::string, MainObjectGroup> by_key;
    for (const auto& [node_id, node] : model.nodes) {
      if (node.kind != LayerKind::kObjects) {
        continue;
      }
      const std::string internal_id = internal_id_for(node.semantic_slot);
      // Unique per-node key for a standalone object, so it never merges with
      // another standalone object under an empty internal_object_id.
      const std::string key = internal_id.empty() ? ("__solo_" + idString(node_id)) : internal_id;

      SegmentMember member;
      member.id = node_id;
      member.semantic_slot = node.semantic_slot;
      member.position = node.position;
      member.has_bbox = node.has_bbox;
      member.bbox_center = node.bbox_center.cast<double>();
      member.bbox_size = node.bbox_size.cast<double>();

      auto& group = by_key[key];
      group.internal_object_id = internal_id;
      group.members.push_back(std::move(member));
    }

    std::vector<MainObjectGroup> groups;
    groups.reserve(by_key.size());
    for (auto& [key, group] : by_key) {
      std::sort(group.members.begin(), group.members.end(),
               [](const SegmentMember& lhs, const SegmentMember& rhs) { return lhs.id < rhs.id; });
      groups.push_back(std::move(group));
    }
    std::sort(groups.begin(), groups.end(), [](const MainObjectGroup& lhs, const MainObjectGroup& rhs) {
      return lhs.members.front().id < rhs.members.front().id;
    });
    return groups;
  }

  void SemanticSceneGraphFuser::computeObjectContacts(const std::vector<MainObjectGroup>& groups,
                             LayeredProjection& projection) {
    last_object_relations_ = Json::array();
    last_object_contact_edge_count_ = 0;
    last_object_contact_max_iou_ = 0.0;
    last_object_contact_grid_cell_size_m_ = 0.0;
    last_object_contact_pairs_tested_ = 0;
    if (!object_contact_edges_enabled_ || !graph_) {
      return;
    }

    struct MainObjectGeom {
      size_t group_index = 0;
      Eigen::Vector3d agg_center = Eigen::Vector3d::Zero();
      Eigen::Vector3d agg_size = Eigen::Vector3d::Zero();
      double reach_m = 0.0;  //!< half the longest aggregate dimension + tolerance
      GridCell cell;
    };

    // Pass 1: aggregate bbox per main object (union of member bboxes; groups
    // with no bbox anywhere have no geometry to test and are skipped), and
    // the largest per-object reach, so the grid cell size can guarantee full
    // coverage from a single ring of neighboring cells.
    std::vector<MainObjectGeom> geoms;
    geoms.reserve(groups.size());
    double global_max_reach_m = 0.0;
    for (size_t g = 0; g < groups.size(); ++g) {
      bool any_bbox = false;
      Eigen::Vector3d agg_min = Eigen::Vector3d::Zero();
      Eigen::Vector3d agg_max = Eigen::Vector3d::Zero();
      for (const auto& member : groups[g].members) {
        if (!member.has_bbox) {
          continue;
        }
        const Eigen::Vector3d half = 0.5 * member.bbox_size.cwiseMax(0.0);
        const Eigen::Vector3d mn = member.bbox_center - half;
        const Eigen::Vector3d mx = member.bbox_center + half;
        if (!any_bbox) {
          agg_min = mn;
          agg_max = mx;
          any_bbox = true;
        } else {
          agg_min = agg_min.cwiseMin(mn);
          agg_max = agg_max.cwiseMax(mx);
        }
      }
      if (!any_bbox) {
        continue;
      }
      MainObjectGeom geom;
      geom.group_index = g;
      geom.agg_center = 0.5 * (agg_min + agg_max);
      geom.agg_size = agg_max - agg_min;
      geom.reach_m = 0.5 * geom.agg_size.cwiseMax(0.0).maxCoeff() + object_contact_tolerance_m_;
      global_max_reach_m = std::max(global_max_reach_m, geom.reach_m);
      geoms.push_back(geom);
    }

    if (object_contact_max_objects_ > 0 && geoms.size() > object_contact_max_objects_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                           "object contact: %zu main objects exceeds object_contact_max_objects=%zu; "
                           "skipping", geoms.size(), object_contact_max_objects_);
      return;
    }
    if (geoms.empty()) {
      return;
    }

    const double now_sec = currentReferenceTimeSec();
    const double cell_size_m =
        object_contact_grid_cell_size_m_ > 0.0
            ? object_contact_grid_cell_size_m_
            : std::max(object_contact_grid_min_cell_size_m_, 2.0 * global_max_reach_m);
    last_object_contact_grid_cell_size_m_ = cell_size_m;

    // Pass 2: bucket main objects by cell.
    std::unordered_map<GridCell, std::vector<size_t>, GridCellHash> grid;
    grid.reserve(geoms.size());
    for (size_t idx = 0; idx < geoms.size(); ++idx) {
      geoms[idx].cell = cellForPoint(geoms[idx].agg_center, cell_size_m);
      grid[geoms[idx].cell].push_back(idx);
    }

    // Canonical per-pair key (each group's smallest member id, distinct
    // across groups since they partition all object nodes) gives the same
    // dedup guarantee the old id-based i<j check gave over raw nodes.
    const auto geom_key = [&groups](const MainObjectGeom& geom) {
      return groups[geom.group_index].members.front().id;
    };

    // Pass 3: for each main object, test only candidates within its
    // reach-derived neighborhood.
    size_t pairs_tested = 0;
    for (size_t i = 0; i < geoms.size(); ++i) {
      const MainObjectGeom& a = geoms[i];
      const int ring = std::max(
          1, static_cast<int>(std::ceil((a.reach_m + global_max_reach_m) / cell_size_m)));
      for (int dz = -ring; dz <= ring; ++dz) {
        for (int dy = -ring; dy <= ring; ++dy) {
          for (int dx = -ring; dx <= ring; ++dx) {
            const auto neighbor_it =
                grid.find(GridCell{a.cell.x + dx, a.cell.y + dy, a.cell.z + dz});
            if (neighbor_it == grid.end()) {
              continue;
            }
            for (const size_t j : neighbor_it->second) {
              const MainObjectGeom& b = geoms[j];
              if (geom_key(b) <= geom_key(a)) {
                continue;  // lower-key side already tests this pair
              }
              ++pairs_tested;

              const auto broad = aabbContact(a.agg_center, a.agg_size, b.agg_center, b.agg_size,
                                             object_contact_tolerance_m_);
              if (!broad.touching) {
                continue;  // aggregates don't even come close; skip the narrow phase
              }

              // Narrow phase: once the aggregates are confirmed touching,
              // every node on the side with FEWER real (bboxed) members
              // connects to its single closest counterpart on the other
              // side, by plain centroid distance — no bbox-intersection
              // test, no per-pair distance cutoff, and matching runs from
              // the smaller side only. Running both directions (an earlier
              // version of this code did) forces every node on the LARGER
              // side to match whenever the smaller side has very few nodes
              // — e.g. a 1-node wall next to a 5-node floor produced 5
              // wall-floor edges (every floor node "finds" the wall, since
              // it's the only candidate on that side), even though only the
              // one floor node actually nearest the wall is a sensible
              // contact point. Iterating only the smaller side's nodes caps
              // the edge count at the smaller side's own node count and
              // leaves every non-matched larger-side node free, matching
              // what a real, sparse point of contact should look like. A
              // larger-side node can still receive more than one edge if
              // it's independently the closest match for more than one
              // smaller-side node — that's a genuine multi-point contact,
              // not the bug above. On a tie in member count, group_a is
              // used (it's already the side with the smaller minimum
              // NodeId, by the canonical ordering established above).
              const MainObjectGroup& group_a = groups[a.group_index];
              const MainObjectGroup& group_b = groups[b.group_index];

              const auto count_bboxed = [](const std::vector<SegmentMember>& members) {
                return std::count_if(members.begin(), members.end(),
                                     [](const SegmentMember& m) { return m.has_bbox; });
              };
              const bool a_is_smaller = count_bboxed(group_a.members) <= count_bboxed(group_b.members);
              const auto& smaller_members = a_is_smaller ? group_a.members : group_b.members;
              const auto& larger_members = a_is_smaller ? group_b.members : group_a.members;

              struct ClosestPair {
                const SegmentMember* member_a = nullptr;
                const SegmentMember* member_b = nullptr;
              };
              std::vector<ClosestPair> selected;
              selected.reserve(smaller_members.size());
              for (const auto& small_member : smaller_members) {
                if (!small_member.has_bbox) {
                  continue;
                }
                const SegmentMember* best = nullptr;
                double best_dist = std::numeric_limits<double>::max();
                for (const auto& large_member : larger_members) {
                  if (!large_member.has_bbox) {
                    continue;
                  }
                  const double dist = (small_member.bbox_center - large_member.bbox_center).norm();
                  if (dist < best_dist) {
                    best_dist = dist;
                    best = &large_member;
                  }
                }
                if (best == nullptr) {
                  continue;
                }
                // member_a always comes from group_a and member_b always
                // from group_b, regardless of which side turned out to be
                // "smaller", so downstream source/target assignment and
                // metadata stay consistent with the broad phase's canonical
                // group ordering.
                if (a_is_smaller) {
                  selected.push_back(ClosestPair{&small_member, best});
                } else {
                  selected.push_back(ClosestPair{best, &small_member});
                }
              }

              // Each selected (nearest-neighbor) pair still gets the full
              // AabbContact computed once, purely to populate the edge's
              // display/diagnostic metadata (gap, IoU, overlap volume,
              // contact axis) — geometry no longer decides which pairs are
              // selected, only how the selected pairs are described.
              for (const auto& pc : selected) {
                const SegmentMember* member_a = pc.member_a;
                const SegmentMember* member_b = pc.member_b;
                const auto contact = aabbContact(member_a->bbox_center, member_a->bbox_size,
                                                 member_b->bbox_center, member_b->bbox_size,
                                                 object_contact_tolerance_m_);
                const NodeId source = member_a->id;
                const NodeId target = member_b->id;
                // Volumetric 3D IoU is a poor summary for most contacts (e.g.
                // a small object against a large flat surface); the larger of
                // the two 2D planar IoUs reads better on the label.
                const double iou_2d_max = std::max(contact.iou_xz, contact.iou_yz);
                projection.edges.push_back(DisplayEdge{source, target,
                                                       DisplayEdgeType::kDerivedObjectContact,
                                                       EdgeOrigin::kNativeHydraEdge, 0,
                                                       iou_2d_max, contact.centroid_distance_m});

                Json meta = {
                    {"schema", "rsg_object_contact_v1"},
                    {"relation", "bbox_contact"},
                    {"centroid_distance_m", contact.centroid_distance_m},
                    {"bbox_gap_m", contact.gap_m},
                    {"bbox_overlap_volume_m3", contact.overlap_volume_m3},
                    {"bbox_overlap_xy_m2", contact.overlap_xy_m2},
                    {"bbox_iou_3d", contact.iou_3d},
                    {"bbox_iou_xz", contact.iou_xz},
                    {"bbox_iou_yz", contact.iou_yz},
                    {"bbox_iou_2d_max", iou_2d_max},
                    {"contact_axis", contactAxisName(contact.contact_axis)},
                    {"source_slot_id", member_a->semantic_slot},
                    {"target_slot_id", member_b->semantic_slot},
                    {"source_internal_object_id", group_a.internal_object_id},
                    {"target_internal_object_id", group_b.internal_object_id},
                    {"source_group_size", group_a.members.size()},
                    {"target_group_size", group_b.members.size()},
                    {"updated_timestamp_sec", now_sec},
                };

                if (object_contact_write_dsg_edges_) {
                  auto attrs = std::make_unique<spark_dsg::EdgeAttributes>(
                      clampValue(contact.iou_3d, 0.0, 1.0));
                  attrs->metadata.set(meta);
                  if (graph_->addOrUpdateEdge(source, target, std::move(attrs))) {
                    contact_edges_prev_.emplace(source, target);
                  }
                }

                Json record = meta;
                record["source"] = idString(source);
                record["target"] = idString(target);
                last_object_relations_.push_back(std::move(record));
                ++last_object_contact_edge_count_;
                last_object_contact_max_iou_ = std::max(last_object_contact_max_iou_, contact.iou_3d);
              }
            }
          }
        }
      }
    }
    last_object_contact_pairs_tested_ = pairs_tested;
  }

  void SemanticSceneGraphFuser::computeObjectSegmentEdges(const std::vector<MainObjectGroup>& groups,
                                 LayeredProjection& projection) {
    last_object_segment_relations_ = Json::array();
    last_object_segment_edge_count_ = 0;
    if (!object_segment_edges_enabled_ || !graph_) {
      return;
    }

    const double now_sec = currentReferenceTimeSec();

    for (const auto& group : groups) {
      if (group.internal_object_id.empty() || group.members.size() < 2) {
        continue;
      }
      if (object_segment_max_group_size_ > 0 &&
          group.members.size() > object_segment_max_group_size_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                             "object segment: internal object '%s' has %zu segments, exceeds "
                             "object_segment_max_group_size=%zu; skipping",
                             group.internal_object_id.c_str(), group.members.size(),
                             object_segment_max_group_size_);
        continue;
      }

      // members is already sorted by NodeId ascending (buildMainObjectGroups).
      const auto& members = group.members;
      for (size_t i = 0; i + 1 < members.size(); ++i) {
        const NodeId source = members[i].id;
        const NodeId target = members[i + 1].id;
        projection.edges.push_back(DisplayEdge{source, target,
                                               DisplayEdgeType::kDerivedObjectSegment,
                                               EdgeOrigin::kNativeHydraEdge, 0});

        Json meta = {
            {"schema", "rsg_object_segment_v1"},
            {"relation", "same_physical_object"},
            {"internal_object_id", group.internal_object_id},
            {"source_slot_id", members[i].semantic_slot},
            {"target_slot_id", members[i + 1].semantic_slot},
            {"group_size", members.size()},
            {"updated_timestamp_sec", now_sec},
        };

        if (object_segment_write_dsg_edges_) {
          auto attrs = std::make_unique<spark_dsg::EdgeAttributes>(1.0);
          attrs->metadata.set(meta);
          if (graph_->addOrUpdateEdge(source, target, std::move(attrs))) {
            segment_edges_prev_.emplace(source, target);
          }
        }

        Json record = meta;
        record["source"] = idString(source);
        record["target"] = idString(target);
        last_object_segment_relations_.push_back(std::move(record));
        ++last_object_segment_edge_count_;
      }
    }
  }

  void SemanticSceneGraphFuser::writeContactDiagnostics(const std::vector<MainObjectGroup>& groups,
                               const std::unordered_map<NodeId, ResolvedOverlay>& resolved) {
    if (!object_contact_diagnostics_enabled_ || object_contact_diagnostics_stream_failed_) {
      return;
    }
    if (!object_contact_diagnostics_stream_.is_open()) {
      std::string path = object_contact_diagnostics_path_;
      if (path.empty()) {
        // One fresh timestamped folder per node lifetime (the stream stays
        // open for the rest of the run once created), under this
        // experiment's results/ directory, so successive debugging runs
        // never clobber each other's logs.
        const std::time_t now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm_buf{};
        localtime_r(&now_time, &tm_buf);
        std::ostringstream stamp;
        stamp << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
        // No portable C++ equivalent of Python's __file__-based workspace-root
        // lookup (see nodes/support/workspace_paths.py for that approach), so
        // this off-by-default debug fallback assumes the conventional
        // $HOME/Thesis_pipeline_split_lean layout. Pass
        // object_contact_diagnostics_path explicitly (a real declared
        // parameter, see above) if this diagnostic is enabled on a workspace
        // cloned somewhere else -- no rebuild required.
        const char* home_env = std::getenv("HOME");
        const std::filesystem::path home_dir =
            home_env ? std::filesystem::path(home_env) : std::filesystem::path("/tmp");
        const std::filesystem::path run_dir =
            home_dir / "Thesis_pipeline_split_lean" / "debug" / "fuser_object_relation_experiment" / "results" /
            ("run_" + stamp.str());
        std::error_code mkdir_error;
        std::filesystem::create_directories(run_dir, mkdir_error);
        if (mkdir_error) {
          RCLCPP_WARN(get_logger(), "object contact diagnostics: failed to create '%s' (%s); disabling",
                     run_dir.c_str(), mkdir_error.message().c_str());
          object_contact_diagnostics_stream_failed_ = true;
          return;
        }
        path = (run_dir / "contact_diagnostics.jsonl").string();
      }
      object_contact_diagnostics_stream_.open(path, std::ios::out | std::ios::app);
      if (!object_contact_diagnostics_stream_.is_open()) {
        RCLCPP_WARN(get_logger(), "object contact diagnostics: failed to open '%s'; disabling",
                   path.c_str());
        object_contact_diagnostics_stream_failed_ = true;
        return;
      }
      RCLCPP_INFO(get_logger(), "object contact diagnostics: writing to '%s'", path.c_str());
    }

    Json main_objects = Json::array();
    for (const auto& group : groups) {
      bool any_bbox = false;
      Eigen::Vector3d agg_min = Eigen::Vector3d::Zero();
      Eigen::Vector3d agg_max = Eigen::Vector3d::Zero();
      std::string group_label;  // first member with a non-empty resolved label
      Json members = Json::array();
      for (const auto& member : group.members) {
        const auto overlay_it = resolved.find(member.id);
        const std::string label =
            overlay_it != resolved.end() ? overlay_it->second.overlay.label : std::string();
        if (group_label.empty() && !label.empty()) {
          group_label = label;
        }
        Json member_json = {
            {"node_id", idString(member.id)},
            {"semantic_slot_id", member.semantic_slot},
            {"has_bbox", member.has_bbox},
            {"label", label},
        };
        if (member.has_bbox) {
          member_json["bbox_center"] = {member.bbox_center.x(), member.bbox_center.y(),
                                        member.bbox_center.z()};
          member_json["bbox_size"] = {member.bbox_size.x(), member.bbox_size.y(),
                                      member.bbox_size.z()};
          const Eigen::Vector3d half = 0.5 * member.bbox_size.cwiseMax(0.0);
          const Eigen::Vector3d mn = member.bbox_center - half;
          const Eigen::Vector3d mx = member.bbox_center + half;
          if (!any_bbox) {
            agg_min = mn;
            agg_max = mx;
            any_bbox = true;
          } else {
            agg_min = agg_min.cwiseMin(mn);
            agg_max = agg_max.cwiseMax(mx);
          }
        }
        members.push_back(std::move(member_json));
      }
      Json group_json = {
          {"internal_object_id", group.internal_object_id},
          {"member_count", group.members.size()},
          {"label", group_label},  // representative label for the whole physical object
          {"members", std::move(members)},
      };
      if (any_bbox) {
        const Eigen::Vector3d agg_center = 0.5 * (agg_min + agg_max);
        const Eigen::Vector3d agg_size = agg_max - agg_min;
        group_json["aggregate_bbox_center"] = {agg_center.x(), agg_center.y(), agg_center.z()};
        group_json["aggregate_bbox_size"] = {agg_size.x(), agg_size.y(), agg_size.z()};
      }
      main_objects.push_back(std::move(group_json));
    }

    // params included on every line so each line is independently
    // analyzable without needing to correlate against a separate header.
    const Json record = {
        {"schema", "rsg_object_contact_diagnostics_v1"},
        {"sequence", object_contact_diagnostics_sequence_},
        {"stamp_sec", currentReferenceTimeSec()},
        {"params",
         {
             {"object_contact_tolerance_m", object_contact_tolerance_m_},
             {"object_contact_min_iou_3d", object_contact_min_iou_3d_},
         }},
        {"main_objects", main_objects},
        {"contact_edges", last_object_relations_},
        {"segment_edges", last_object_segment_relations_},
    };
    object_contact_diagnostics_stream_ << record.dump() << "\n";
    object_contact_diagnostics_stream_.flush();
    ++object_contact_diagnostics_sequence_;
  }

  void SemanticSceneGraphFuser::updateLocalGraphMetadata(const SceneModel& model,
                                const std::unordered_map<NodeId, ResolvedOverlay>& resolved,
                                const RiskOverlayCache& risk_overlays,
                                const PresenceCache& presence) {
    if (!graph_) {
      return;
    }
    for (const auto& [node_id, node] : model.nodes) {
      if (node.kind != LayerKind::kObjects) {
        continue;
      }
      // A synthetic dynamic-object node (see collectModel) is never added to
      // graph_ itself, so it has no real DSG metadata to sync -- skip rather
      // than let getNode() throw on an id the graph doesn't have.
      if (!graph_->hasNode(node_id)) {
        continue;
      }
      auto& attrs = graph_->getNode(node_id).attributes<spark_dsg::NodeAttributes>();
      Json metadata = attrs.metadata.get();
      if (!metadata.is_object()) {
        metadata = Json::object();
      }
      metadata.erase("rsg_rap");
      metadata.erase("rsg_presence");
      metadata.erase("rsg_identity");
      metadata.erase("rsg_risk");
      const auto presence_it = presence.find(node.semantic_slot);
      if (presence_it != presence.end()) {
        const auto& obs = presence_it->second;
        metadata["rsg_presence"] = {
            {"slot_id", obs.slot_id},
            {"last_observed_timestamp_sec", obs.last_observed_timestamp_sec},
            {"internal_object_id", obs.internal_object_id},
            {"persistent_track_id", obs.persistent_track_id},
            {"local_segment_id", obs.local_segment_id},
            {"local_segment_xy_span_m", obs.local_segment_xy_span_m},
        };
        metadata["rsg_identity"] = {
            {"schema", "rsg_local_segment_identity_v1"},
            {"semantic_slot_id", obs.slot_id},
            {"hydra_slot_id", obs.slot_id},
            {"internal_object_id", obs.internal_object_id},
            {"persistent_track_id", obs.persistent_track_id},
            {"local_segment_id", obs.local_segment_id},
        };
      }
      const auto it = resolved.find(node_id);
      if (it != resolved.end()) {
        const auto& label = it->second;
        metadata["rsg_rap"] = {
            {"slot_id", label.overlay.slot_id},
            {"label", label.overlay.label},
            {"confidence", label.overlay.confidence},
            {"label_confidence", label.overlay.confidence},
            {"mobility_class", label.overlay.mobility_class},
            {"mobility_confidence", label.overlay.mobility_confidence},
            {"mobility_source", label.overlay.mobility_source},
            {"object_detail", label.overlay.object_detail},
            {"source", label.overlay.source},
            {"timestamp_sec", label.overlay.timestamp_sec},
            {"association", label.association},
            {"centroid_distance_m", label.centroid_distance_m},
        };
      }
      // Plain slot_id lookup, same key as rsg_presence above -- risk has no
      // centroid-fallback matching to do (see RiskOverlay's own comment), so
      // it doesn't need the resolved/ResolvedOverlay machinery label lookup
      // above uses.
      const auto risk_it = risk_overlays.find(node.semantic_slot);
      if (risk_it != risk_overlays.end()) {
        const auto& risk = risk_it->second;
        metadata["rsg_risk"] = {
            {"schema", "rsg_object_risk_v1"},
            {"slot_id", risk.slot_id},
            {"risk_score", risk.risk_score},
            {"risk_factors", risk.risk_factors},
            {"source", risk.source},
            {"timestamp_sec", risk.timestamp_sec},
        };
      }
      attrs.metadata.set(metadata);
    }
  }

}  // namespace rsg
