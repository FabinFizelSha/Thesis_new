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
