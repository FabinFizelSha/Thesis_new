/**
 * @file fuser_core.cpp
 * @brief Node construction (parameter declarations, subscriptions,
 * publishers, timer) and process entry point.
 */
#include "fuser.hpp"

namespace rsg {

  SemanticSceneGraphFuser::SemanticSceneGraphFuser() : Node("rsg_scene_graph_fuser") {
    input_dsg_topic_ = declare_parameter<std::string>("input_dsg_topic", "/hydra/backend/dsg");
    semantic_label_topic_ = declare_parameter<std::string>(
        "semantic_label_topic", "/rsg/objects/semantic_label_result");
    active_segments_topic_ = declare_parameter<std::string>(
        "active_segments_topic", "/rsg/objects/active_local_segments");
    risk_result_topic_ = declare_parameter<std::string>(
        "risk_result_topic", "/rsg/objects/risk_result");
    semantic_label_qos_depth_ = static_cast<size_t>(std::max<int64_t>(
        1, declare_parameter<int64_t>("semantic_label_qos_depth", 4096)));
    semantic_refresh_rate_hz_ = std::max(
        0.1, declare_parameter<double>("semantic_refresh_rate_hz", 1.0));
    fused_dsg_topic_ = declare_parameter<std::string>("fused_dsg_topic", "/rsg/hydra/fused_dsg");
    markers_topic_ = declare_parameter<std::string>("markers_topic", "/rsg/scene_graph/markers");
    status_topic_ = declare_parameter<std::string>("status_topic", "/rsg/hydra/rap_fuser/status");
    fallback_frame_id_ = declare_parameter<std::string>("fallback_frame_id", "odom");

    // RViz consumes MarkerArray directly. Full DSG publication is optional.
    publish_fused_dsg_ = declare_parameter<bool>("publish_fused_dsg", false);
    // Semantic labels are batched by a short timer. This prevents a burst of
    // RAP/VLM completions from repeatedly rebuilding the full DSG projection.
    drop_local_mesh_ = declare_parameter<bool>("drop_local_mesh", true);
    marker_publish_rate_hz_ = std::max(0.0, declare_parameter<double>(
        "marker_publish_rate_hz", kDefaultMarkerRateHz));

    // Layered fused display. The local DSG remains an unchanged Hydra clone;
    // these offsets exist only in RViz MarkerArray positions.
    show_objects_ = declare_parameter<bool>("show_objects", true);
    show_rooms_ = declare_parameter<bool>("show_rooms", true);
    show_buildings_ = declare_parameter<bool>("show_buildings", false);
    show_places_ = declare_parameter<bool>("show_places", true);
    show_segments_ = declare_parameter<bool>("show_segments", false);
    show_agents_ = declare_parameter<bool>("show_agents", false);
    show_edges_ = declare_parameter<bool>("show_edges", true);
    show_room_place_edges_ = declare_parameter<bool>("show_room_place_edges", true);
    show_place_object_edges_ = declare_parameter<bool>("show_place_object_edges", true);
    show_place_connectivity_edges_ = declare_parameter<bool>("show_place_connectivity_edges", true);
    show_native_object_object_edges_ = declare_parameter<bool>("show_native_object_object_edges", false);
    show_native_room_room_edges_ = declare_parameter<bool>("show_native_room_room_edges", false);

    // Object-contact relationships: connect objects whose bounding boxes touch.
    object_contact_edges_enabled_ = declare_parameter<bool>("object_contact_edges_enabled", true);
    object_contact_tolerance_m_ = declare_parameter<double>("object_contact_tolerance_m", 0.05);
    object_contact_min_iou_3d_ = declare_parameter<double>("object_contact_min_iou_3d", 0.0);
    object_contact_write_dsg_edges_ = declare_parameter<bool>("object_contact_write_dsg_edges", true);
    object_contact_max_objects_ =
        static_cast<size_t>(std::max<int64_t>(0, declare_parameter<int>("object_contact_max_objects", 500)));
    // Spatial-hash grid narrows the pairing search: 0.0 = auto-size the cell
    // each cycle from the largest object reach seen (guarantees a single ring
    // of neighbor cells is enough); a positive override fixes the cell size
    // instead, with the per-object search ring widened as needed to keep
    // guaranteed coverage.
    object_contact_grid_cell_size_m_ = declare_parameter<double>("object_contact_grid_cell_size_m", 0.0);
    object_contact_grid_min_cell_size_m_ =
        declare_parameter<double>("object_contact_grid_min_cell_size_m", 0.05);
    show_object_contact_edges_ = declare_parameter<bool>("show_object_contact_edges", true);
    show_object_contact_labels_ = declare_parameter<bool>("show_object_contact_labels", true);
    object_contact_text_height_m_ =
        std::max(0.05, declare_parameter<double>("object_contact_text_height_m", 0.16));
    // Fixed absolute size, deliberately NOT derived from segment length --
    // every contact arrowhead is the same size regardless of how long or
    // short the edge is. (Previously scaled down on short segments to avoid
    // overshoot; the fixed size is used unconditionally now, per direction.)
    object_contact_arrow_head_length_m_ =
        std::max(0.01, declare_parameter<double>("object_contact_arrow_head_length_m", 0.20));
    object_contact_arrow_head_diameter_m_ =
        std::max(0.01, declare_parameter<double>("object_contact_arrow_head_diameter_m", 0.11));

    // Object-segment relationships: connect local segments phase1 split off
    // from one physical object that was too large to track as a single
    // segment (same PresenceObservation internal_object_id). Identity
    // relation, not geometric — no tolerance/grid params needed.
    object_segment_edges_enabled_ = declare_parameter<bool>("object_segment_edges_enabled", true);
    // One semantic slot == one physical object in this pipeline, but Hydra
    // creates a node per mesh cluster, so a resumed session gets a duplicate
    // node stacked on the restored one. Collapse them in the fuser, whose
    // output is the final scene graph.
    object_slot_collapse_enabled_ = declare_parameter<bool>("object_slot_collapse_enabled", true);
    object_slot_collapse_max_distance_m_ =
        declare_parameter<double>("object_slot_collapse_max_distance_m", 2.0);
    // In-view halo: a translucent filled sphere around anything currently at
    // high presence confidence. Grey rather than cyan so it reads as a
    // neutral highlight instead of tinting the object's own color.
    highlight_active_objects_ = declare_parameter<bool>("highlight_active_objects", true);
    active_object_halo_min_confidence_ = clampValue(
        declare_parameter<double>("active_object_halo_min_confidence", 0.995), 0.0, 1.0);
    active_object_halo_scale_ = declare_parameter<double>("active_object_halo_scale", 1.7);
    active_object_halo_alpha_ = declare_parameter<double>("active_object_halo_alpha", 0.45);
    active_object_halo_min_margin_m_ =
        declare_parameter<double>("active_object_halo_min_margin_m", 0.12);
    object_segment_write_dsg_edges_ =
        declare_parameter<bool>("object_segment_write_dsg_edges", true);
    object_segment_max_group_size_ = static_cast<size_t>(
        std::max<int64_t>(0, declare_parameter<int>("object_segment_max_group_size", 50)));
    show_object_segment_edges_ = declare_parameter<bool>("show_object_segment_edges", true);
    object_segment_dot_spacing_m_ =
        std::max(0.01, declare_parameter<double>("object_segment_dot_spacing_m", 0.08));
    object_segment_dot_size_m_ =
        std::max(0.005, declare_parameter<double>("object_segment_dot_size_m", 0.03));

    // Offline verification: dumps every main object's aggregate bbox, every
    // member's bbox, and the contact/segment edges actually produced, one
    // JSON line per fusion cycle, so
    // debug/fuser_object_relation_experiment/analyze_contact_diagnostics.py
    // can independently recompute the expected edges from raw geometry and
    // diff against what the fuser decided. Off by default — no effect on
    // production runs.
    object_contact_diagnostics_enabled_ =
        declare_parameter<bool>("object_contact_diagnostics_enabled", false);
    object_contact_diagnostics_path_ =
        declare_parameter<std::string>("object_contact_diagnostics_path", "");

    show_object_labels_ = declare_parameter<bool>("show_object_labels", true);
    show_room_labels_ = declare_parameter<bool>("show_room_labels", true);
    show_place_labels_ = declare_parameter<bool>("show_place_labels", false);
    show_building_labels_ = declare_parameter<bool>("show_building_labels", true);
    show_slot_ids_ = declare_parameter<bool>("show_slot_ids", false);
    show_presence_confidence_ = declare_parameter<bool>("show_presence_confidence", false);
    show_label_confidence_ = declare_parameter<bool>("show_label_confidence", true);
    show_mobility_metadata_ = declare_parameter<bool>("show_mobility_metadata", true);
    show_object_detail_ = declare_parameter<bool>("show_object_detail", true);
    static_presence_half_life_sec_ = std::max(
        0.1, declare_parameter<double>("static_presence_half_life_sec", 600.0));
    dynamic_presence_half_life_sec_ = std::max(
        0.1, declare_parameter<double>("dynamic_presence_half_life_sec", 120.0));
    presence_observed_epsilon_sec_ = std::max(0.0, declare_parameter<double>("presence_observed_epsilon_sec", 1.5));
    minimum_object_alpha_ = static_cast<float>(clampValue(
        declare_parameter<double>("minimum_object_alpha", 0.03), 0.001, 1.0));
    dynamic_object_use_cube_ = declare_parameter<bool>("dynamic_object_use_cube", true);
    presence_decay_continuous_refresh_ = declare_parameter<bool>(
        "presence_decay_continuous_refresh", true);
    object_z_offset_m_ = declare_parameter<double>("object_z_offset_m", 0.0);
    place_z_offset_m_ = declare_parameter<double>("place_z_offset_m", 10.0);
    room_z_offset_m_ = declare_parameter<double>("room_z_offset_m", 20.0);
    room_node_size_m_ = std::max(0.10, declare_parameter<double>("room_node_size_m", 1.10));
    place_node_size_m_ = std::max(0.05, declare_parameter<double>("place_node_size_m", 0.45));
    place_text_height_m_ = std::max(0.05, declare_parameter<double>("place_text_height_m", 0.18));
    place_alpha_ = static_cast<float>(clampValue(declare_parameter<double>("place_alpha", 0.90), 0.0, 1.0));

    // Complete missing room/place membership in the fuser-owned display graph.
    // For each orphaned place, the nearest visible room-owned places vote for
    // the parent room. The test is mesh validated and therefore fails closed.
    room_place_completion_enabled_ = declare_parameter<bool>("room_place_completion_enabled", true);
    room_place_completion_grace_period_sec_ = std::max(0.0, declare_parameter<double>(
        "room_place_completion_grace_period_sec", 30.0));
    const int64_t room_place_completion_neighbours =
        declare_parameter<int64_t>("room_place_completion_neighbours", 7);
    room_place_completion_neighbours_ = static_cast<int>(
        std::max<int64_t>(1, room_place_completion_neighbours));
    room_place_completion_max_distance_m_ = std::max(0.0, declare_parameter<double>(
        "room_place_completion_max_distance_m", 0.0));
    room_place_completion_max_height_difference_m_ = std::max(0.0, declare_parameter<double>(
        "room_place_completion_max_height_difference_m", 0.50));
    room_place_completion_require_mesh_validation_ = declare_parameter<bool>(
        "room_place_completion_require_mesh_validation", true);
    const int64_t room_place_completion_min_majority_votes =
        declare_parameter<int64_t>("room_place_completion_min_majority_votes", 1);
    room_place_completion_min_majority_votes_ = static_cast<int>(
        std::max<int64_t>(1, room_place_completion_min_majority_votes));

    object_place_use_local_validated_fallback_ = declare_parameter<bool>(
        "object_place_use_local_validated_fallback", true);
    object_place_max_distance_m_ = std::max(0.0, declare_parameter<double>(
        "object_place_max_distance_m", 3.0));
    object_place_index_voxel_size_m_ = std::max(0.10, declare_parameter<double>(
        "object_place_index_voxel_size_m", 2.0));
    const int64_t object_place_index_search_radius_cells =
        declare_parameter<int64_t>("object_place_index_search_radius_cells", 2);
    object_place_index_search_radius_cells_ = static_cast<int>(
        std::max<int64_t>(0, object_place_index_search_radius_cells));

    const int64_t object_place_max_candidates =
        declare_parameter<int64_t>("object_place_max_candidates", 6);
    object_place_max_candidates_ = static_cast<int>(
        std::max<int64_t>(1, object_place_max_candidates));
    object_place_require_mesh_validation_ = declare_parameter<bool>(
        "object_place_require_mesh_validation", true);
    object_place_mesh_voxel_size_m_ = std::max(0.10, declare_parameter<double>(
        "object_place_mesh_voxel_size_m", 0.50));
    const int64_t object_place_mesh_max_triangle_tests =
        declare_parameter<int64_t>("object_place_mesh_max_triangle_tests", 2048);
    object_place_mesh_max_triangle_tests_ = static_cast<int>(
        std::max<int64_t>(1, object_place_mesh_max_triangle_tests));

    const int64_t object_place_mesh_max_cells_per_triangle =
        declare_parameter<int64_t>("object_place_mesh_max_cells_per_triangle", 256);
    object_place_mesh_max_cells_per_triangle_ = static_cast<int>(
        std::max<int64_t>(1, object_place_mesh_max_cells_per_triangle));
    object_place_anchor_outset_m_ = std::max(0.0, declare_parameter<double>(
        "object_place_anchor_outset_m", 0.12));
    object_place_cache_recompute_translation_m_ = std::max(0.0, declare_parameter<double>(
        "object_place_cache_recompute_translation_m", 0.25));

    centroid_association_enabled_ = declare_parameter<bool>("centroid_association_enabled", true);
    require_centroid_frame_match_ = declare_parameter<bool>("require_centroid_frame_match", true);
    allow_unframed_centroid_ = declare_parameter<bool>("allow_unframed_centroid", false);
    const int64_t max_label_candidates_per_slot = declare_parameter<int64_t>(
        "max_label_candidates_per_slot", 8);
    max_label_candidates_per_slot_ = static_cast<size_t>(
        std::max<int64_t>(1, max_label_candidates_per_slot));
    // Deliberately NOT passed through normaliseLabel(): that helper converts
    // underscores to spaces (for real semantic labels like "coffee_table"
    // from a VLM/RAP source), which would turn this fixed placeholder back
    // into "object unknown" and defeat the point of using an underscore here.
    unlabeled_object_display_label_ =
        declare_parameter<std::string>("unlabeled_object_display_label", "object_unknown");
    if (unlabeled_object_display_label_.empty()) {
      unlabeled_object_display_label_ = "object_unknown";
    }

    object_min_size_m_ = std::max(kSmallExtentM, declare_parameter<double>("object_min_size_m", 0.12));
    object_max_size_m_ = std::max(object_min_size_m_, declare_parameter<double>("object_max_size_m", 0.60));
    object_sphere_volume_scale_ = std::max(0.0, declare_parameter<double>(
        "object_sphere_volume_scale", 0.60));
    object_sphere_size_mode_ = normaliseLabel(declare_parameter<std::string>(
        "object_sphere_size_mode", "volume_scaled"));
    const bool mode_requests_fixed_size =
        object_sphere_size_mode_ == "fixed" ||
        object_sphere_size_mode_ == "constant" ||
        object_sphere_size_mode_ == "uniform";
    object_use_fixed_sphere_size_ = declare_parameter<bool>(
        "object_use_fixed_sphere_size", mode_requests_fixed_size);
    object_fixed_sphere_size_m_ = std::max(kSmallExtentM, declare_parameter<double>(
        "object_fixed_sphere_size_m", 0.28));
    object_text_height_m_ = std::max(0.05, declare_parameter<double>("object_text_height_m", 0.28));
    object_label_vertical_offset_m_ = std::max(
        0.0, declare_parameter<double>("object_label_vertical_offset_m", 0.10));
    room_text_height_m_ = std::max(0.05, declare_parameter<double>("room_text_height_m", 0.35));
    building_text_height_m_ = std::max(0.05, declare_parameter<double>("building_text_height_m", 0.42));
    room_alpha_ = static_cast<float>(clampValue(declare_parameter<double>("room_alpha", 0.14), 0.0, 1.0));
    building_alpha_ = static_cast<float>(clampValue(declare_parameter<double>("building_alpha", 0.10), 0.0, 1.0));
    edge_width_m_ = std::max(0.005, declare_parameter<double>("edge_width_m", 0.025));

    const auto input_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    const auto semantic_qos = rclcpp::QoS(rclcpp::KeepLast(semantic_label_qos_depth_))
                                  .reliable()
                                  .durability_volatile();
    const auto retained_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    // Label ingestion, Hydra graph updates, and rendering run in separate
    // callback groups. A multi-threaded executor can therefore drain the
    // reliable final-label queue while a slower render pass builds markers.
    graph_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    semantic_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    render_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions graph_options;
    graph_options.callback_group = graph_callback_group_;
    rclcpp::SubscriptionOptions semantic_options;
    semantic_options.callback_group = semantic_callback_group_;

    dsg_sub_ = create_subscription<DsgUpdate>(
        input_dsg_topic_, input_qos,
        std::bind(&SemanticSceneGraphFuser::handleDsg, this, std::placeholders::_1),
        graph_options);
    label_sub_ = create_subscription<std_msgs::msg::String>(
        semantic_label_topic_, semantic_qos,
        std::bind(&SemanticSceneGraphFuser::handleSemanticLabel, this, std::placeholders::_1),
        semantic_options);
    active_segments_sub_ = create_subscription<std_msgs::msg::String>(
        active_segments_topic_, semantic_qos,
        std::bind(&SemanticSceneGraphFuser::handleActiveSegments, this, std::placeholders::_1),
        semantic_options);
    risk_sub_ = create_subscription<std_msgs::msg::String>(
        risk_result_topic_, semantic_qos,
        std::bind(&SemanticSceneGraphFuser::handleRiskResult, this, std::placeholders::_1),
        semantic_options);

    fused_dsg_pub_ = create_publisher<DsgUpdate>(fused_dsg_topic_, retained_qos);
    marker_pub_ = create_publisher<MarkerArray>(markers_topic_, retained_qos);
    status_pub_ = create_publisher<std_msgs::msg::String>(status_topic_, retained_qos);

    const auto refresh_period = std::chrono::milliseconds(std::max<int64_t>(
        1, static_cast<int64_t>(std::llround(1000.0 / semantic_refresh_rate_hz_))));
    semantic_refresh_timer_ = create_wall_timer(
        refresh_period,
        std::bind(&SemanticSceneGraphFuser::renderDirtyState, this),
        render_callback_group_);

    RCLCPP_INFO(
        get_logger(),
        "Hydra semantic fuser ready: DSG='%s', final labels='%s', label QoS depth=%zu, markers='%s'",
        input_dsg_topic_.c_str(), semantic_label_topic_.c_str(), semantic_label_qos_depth_,
        markers_topic_.c_str());
    publishStatus("started", "ready");

    // No other param on this node is live-updatable (every declare_parameter
    // above is read once here and cached into a member); a `ros2 param set`
    // on an already-running node normally has no effect at all. That's the
    // wrong UX specifically for a diagnostics on/off switch someone wants to
    // flip mid-run without restarting the whole pipeline, so these two are
    // deliberately the exception.
    parameter_callback_handle_ = add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& parameters) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = true;
          for (const auto& parameter : parameters) {
            if (parameter.get_name() == "object_contact_diagnostics_enabled" &&
                parameter.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) {
              object_contact_diagnostics_enabled_ = parameter.as_bool();
              RCLCPP_INFO(get_logger(), "object contact diagnostics: %s (live update)",
                         object_contact_diagnostics_enabled_ ? "enabled" : "disabled");
            } else if (parameter.get_name() == "object_contact_diagnostics_path" &&
                       parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
              object_contact_diagnostics_path_ = parameter.as_string();
            }
          }
          return result;
        });
  }

}  // namespace rsg

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rsg::SemanticSceneGraphFuser>();
  // Three worker threads allow fast semantic ingestion, Hydra DSG updates, and
  // bounded rendering to make progress independently through callback groups.
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 3U);
  executor.add_node(node);
  executor.spin();
  executor.remove_node(node->get_node_base_interface());
  node.reset();
  rclcpp::shutdown();
  return 0;
}
