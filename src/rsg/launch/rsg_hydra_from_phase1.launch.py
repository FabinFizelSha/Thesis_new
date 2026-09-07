"""Launch Hydra mapping and visualization for the official TESSE uHumans2 bag."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the ROS 2 launch description for this component."""
    rsg_share = FindPackageShare("rsg")
    hydra_ros_share = FindPackageShare("hydra_ros")
    input_config = PathJoinSubstitution(
        [rsg_share, "config", "hydra", "rsg_phase1_input_tesse.yaml"]
    )
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
    default_log_path = "/home/student/Thesis_new/memory/hydra"

    visualization_odom_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="rsg_world_to_odom_visualization_bridge",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "world", "--child-frame-id", "odom",
        ],
        condition=IfCondition(LaunchConfiguration("publish_visualization_odom_bridge")),
        output="screen",
    )

    return LaunchDescription([
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
        DeclareLaunchArgument(
            "hydra_load_state_path",
            default_value="/home/student/Thesis_new/memory/hydra/backend/dsg_with_mesh.json",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("sensor_frame", default_value="left_cam"),
        DeclareLaunchArgument("robot_frame", default_value="base_link_gt"),
        DeclareLaunchArgument("odom_frame", default_value="world"),
        DeclareLaunchArgument("map_frame", default_value="world"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument("start_hydra_visualizer", default_value="true"),
        DeclareLaunchArgument("publish_visualization_odom_bridge", default_value="false"),
        DeclareLaunchArgument("glog_level", default_value="0"),
        DeclareLaunchArgument("glog_verbosity", default_value="0"),
        DeclareLaunchArgument("hydra_extra_yaml", default_value="{show_run_settings: false, config_verbosity: 0}"),

        visualization_odom_bridge,

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
