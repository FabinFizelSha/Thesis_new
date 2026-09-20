"""Launch Hydra mapping and visualization for the official TESSE uHumans2 bag."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from nodes.support.workspace_paths import workspace_path


def generate_launch_description() -> LaunchDescription:
    """Build the ROS 2 launch description for this component."""
    rsg_share = FindPackageShare("rsg")
    hydra_ros_share = FindPackageShare("hydra_ros")
    input_config = LaunchConfiguration("input_config")
    rviz_config = PathJoinSubstitution([rsg_share, "config", "rviz", "rsg_hydra_rap_fused_scene_graph.rviz"])

    # Hydra writes its save artifacts (backend/dsg_with_mesh.json, mesh.ply,
    # deformation_graph.dgrf, and the active window's volumetric map) under
    # log_path at shutdown.
    #
    # This used to be an unconditionally fresh /tmp/hydra_uhumans2_<timestamp>
    # directory, deliberately, so each launch got a clean Hydra with no chance
    # of picking up stale state. It is no longer the default, because
    # multi-session resume needs the previous run's artifacts to still be on
    # disk and findable. The default now points into the workspace memory/
    # folder alongside the phase-1 tracker state and the RAP store. To get the
    # old throwaway behaviour back, pass a /tmp path:
    #   hydra_log_path:=/tmp/hydra_scratch
    #
    # Note this only controls where Hydra SAVES. Loading is opt-in and separate
    # (hydra_load_state_path below), so a stable log_path on its own cannot
    # resurrect old state -- it only stops it from being thrown away.
    #
    # Namespaced by dataset (memory/hydra/<dataset>/...) so switching datasets
    # can never silently resume another dataset's saved DSG/mesh -- confirmed
    # 2026-09-19: an OpenLoRIS run loaded uhumans2's leftover mesh/objects
    # because both shared this one fixed path.
    hydra_memory_root = str(workspace_path("memory", "hydra"))
    default_log_path = PathJoinSubstitution([hydra_memory_root, LaunchConfiguration("dataset")])

    visualization_odom_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="rsg_world_to_odom_visualization_bridge",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "world", "--child-frame-id", LaunchConfiguration("visualization_odom_bridge_child_frame"),
        ],
        condition=IfCondition(LaunchConfiguration("publish_visualization_odom_bridge")),
        output="screen",
    )

    # Cosmetic-only fixed rotation from sensor_frame (REP-103 optical:
    # X=right, Y=down, Z=forward) to robot_frame (body convention: X=forward,
    # Y=left, Z=up) -- confirmed 2026-09-19 on OpenLoRIS: with robot_frame set
    # equal to sensor_frame (required so Hydra's one-time
    # robot_frame_T_sensor_frame extrinsics cache is trivially identity,
    # forever correct, while the real per-frame motion flows through the
    # separate dynamic odom_frame_T_robot_frame TFLookup -- see
    # hydra_ros/src/utils/tf_lookup.cpp's getBodyPose), the agent/pose-array
    # arrow in RViz was drawn along the optical frame's local X (camera
    # right) instead of forward, ~90 degrees off. This publishes robot_frame
    # as a fixed child of sensor_frame instead, using the exact inverse of
    # the standard ROS camera driver camera_link->*_optical_frame quaternion
    # (-0.5, 0.5, -0.5, 0.5) -- does not touch sensor_frame itself, so voxel
    # integration/TSDF geometry is unaffected either way. Off by default: a
    # profile where robot_frame is already a real body frame (e.g. uhumans2's
    # base_link_gt) does not need or want this.
    sensor_body_frame_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="rsg_sensor_to_robot_body_frame_bridge",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--qx", "0.5", "--qy", "-0.5", "--qz", "0.5", "--qw", "0.5",
            "--frame-id", LaunchConfiguration("sensor_frame"),
            "--child-frame-id", LaunchConfiguration("robot_frame"),
        ],
        condition=IfCondition(LaunchConfiguration("publish_sensor_body_frame_bridge")),
        output="screen",
    )

    return LaunchDescription([
        # Both bundled .rviz configs hardcode Fixed Frame: world. TESSE's own
        # TF root really is named "world" so this bridge stays off by default.
        # A profile whose odom_frame is a different name (e.g. OpenLoRIS's
        # base_odom) has no "world" frame at all -- RViz then shows nothing,
        # silently, with no error beyond its own "Fixed Frame does not exist"
        # status. Set publish_visualization_odom_bridge:=true and this to
        # that profile's actual root frame (e.g. base_odom) to fix it without
        # editing the .rviz files.
        DeclareLaunchArgument("visualization_odom_bridge_child_frame", default_value="odom"),
        DeclareLaunchArgument(
            "input_config",
            default_value=PathJoinSubstitution(
                [rsg_share, "config", "hydra", "rsg_phase1_input_tesse.yaml"]
            ),
            description="Hydra ROS input config; defaults to the official TESSE uHumans2 bag.",
        ),
        DeclareLaunchArgument("dataset", default_value="uhumans2"),
        DeclareLaunchArgument("labelspace", default_value="rsg_slot_only_frozen"),
        # Where Hydra writes its shutdown artifacts. See default_log_path above.
        DeclareLaunchArgument("hydra_log_path", default_value=default_log_path),
        # Multi-session resume. ON by default and pointed at where
        # hydra_log_path saves, so runs chain automatically: the first run finds
        # no file and starts fresh, saves at shutdown, and the next one resumes
        # from it. A missing file is handled as a normal first run, not an error.
        # Pass "none" to disable. Must not be empty -- the launch frontend
        # renders an empty arg as YAML null, which config-utilities cannot
        # convert to a string and throws on.
        # On resume, start the trajectory at the map origin (correct when
        # replaying the same bag). Set false to continue the previous run's
        # trajectory, which is what you want when genuinely exploring onward.
        # See rsg_all.launch.py's comment on this same arg: controls only
        # whether restored agent/trajectory nodes are kept or dropped, not the
        # robot's actual position. Default true avoids a real bug where a
        # kept old node can silently block a new pose with the same id from
        # ever being inserted.
        DeclareLaunchArgument("hydra_resume_reset_trajectory", default_value="true"),
        # datasets/uhumans2.yaml sets backend.enable_node_merging: false. True is
        # required for a resumed run to reconcile a freshly re-observed object
        # with the one restored from the previous session (bbox-overlap match
        # against the archived candidate) instead of creating a duplicate node
        # for it -- see UpdateObjectsFunctor::findMerges.
        DeclareLaunchArgument("hydra_enable_object_merging", default_value="true"),
        DeclareLaunchArgument(
            "hydra_load_state_path",
            default_value=PathJoinSubstitution(
                [hydra_memory_root, LaunchConfiguration("dataset"), "backend", "dsg_with_mesh.json"]
            ),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("sensor_frame", default_value="left_cam"),
        DeclareLaunchArgument("robot_frame", default_value="base_link_gt"),
        DeclareLaunchArgument("odom_frame", default_value="world"),
        DeclareLaunchArgument("map_frame", default_value="world"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument("start_hydra_visualizer", default_value="true"),
        DeclareLaunchArgument("publish_visualization_odom_bridge", default_value="false"),
        # See sensor_body_frame_bridge's comment above.
        DeclareLaunchArgument("publish_sensor_body_frame_bridge", default_value="false"),
        DeclareLaunchArgument("glog_level", default_value="0"),
        DeclareLaunchArgument("glog_verbosity", default_value="0"),
        DeclareLaunchArgument("hydra_extra_yaml", default_value="{show_run_settings: false, config_verbosity: 0}"),

        visualization_odom_bridge,
        sensor_body_frame_bridge,

        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution([hydra_ros_share, "launch", "hydra.launch.yaml"])
            ),
            launch_arguments={
                "dataset": LaunchConfiguration("dataset"),
                "labelspace": LaunchConfiguration("labelspace"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "input_config_path": input_config,
                "labelspace_path": PathJoinSubstitution([rsg_share, "config", "hydra", "rsg_slot_only_frozen_label_space.yaml"]),
                "sensor_frame": LaunchConfiguration("sensor_frame"),
                "robot_frame": LaunchConfiguration("robot_frame"),
                "odom_frame": LaunchConfiguration("odom_frame"),
                "map_frame": LaunchConfiguration("map_frame"),
                "start_visualizer": LaunchConfiguration("start_hydra_visualizer"),
                "glog_level": LaunchConfiguration("glog_level"),
                "glog_verbosity": LaunchConfiguration("glog_verbosity"),
                "extra_yaml": LaunchConfiguration("hydra_extra_yaml"),
                "log_path": LaunchConfiguration("hydra_log_path"),
                # Dedicated argument rather than folded into extra_yaml: any
                # parent launch file that declares its own hydra_extra_yaml
                # default would otherwise silently drop resume. rsg_all.launch.py
                # does exactly that, which is why the first attempt never loaded.
                "load_state_path": LaunchConfiguration("hydra_load_state_path"),
                "resume_reset_trajectory": LaunchConfiguration("hydra_resume_reset_trajectory"),
                "enable_object_merging": LaunchConfiguration("hydra_enable_object_merging"),
            }.items(),
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rsg_fused_rviz",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            condition=IfCondition(LaunchConfiguration("start_rviz")),
            output="screen",
        ),
    ])
