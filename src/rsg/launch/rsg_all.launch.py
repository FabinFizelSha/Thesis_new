"""Launch the complete RSG, Hydra, and RViz stack in one command.

The launch combines the existing three-node RSG stack with the existing Hydra
input/mapping/visualization launch. It keeps the baseline launch defaults and
only exposes them as top-level arguments. Rosbag playback is intentionally not
started here; run ``ros2 run rsg rsg_play_uhumans2`` in a separate terminal.
The helper defaults to the external dataset at ``/home/student/datasets``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create the complete RSG + Hydra launch description."""
    share = FindPackageShare("rsg")
    start_hydra = LaunchConfiguration("start_hydra")

    rsg_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "rsg_full_stack.launch.py"])
        ),
        launch_arguments={
            "pipeline_config": LaunchConfiguration("pipeline_config"),
            "start_chroma": LaunchConfiguration("start_chroma"),
            "start_qwen": LaunchConfiguration("start_qwen"),
        }.items(),
    )

    hydra_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "rsg_hydra_from_phase1.launch.py"])
        ),
        launch_arguments={
            "dataset": LaunchConfiguration("dataset"),
            "labelspace": LaunchConfiguration("labelspace"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "sensor_frame": LaunchConfiguration("sensor_frame"),
            "robot_frame": LaunchConfiguration("robot_frame"),
            "odom_frame": LaunchConfiguration("odom_frame"),
            "map_frame": LaunchConfiguration("map_frame"),
            "start_rviz": LaunchConfiguration("start_rviz"),
            "start_hydra_visualizer": LaunchConfiguration("start_hydra_visualizer"),
            "publish_visualization_odom_bridge": LaunchConfiguration(
                "publish_visualization_odom_bridge"
            ),
            "hydra_extra_yaml": LaunchConfiguration("hydra_extra_yaml"),
            "glog_level": LaunchConfiguration("glog_level"),
            "glog_verbosity": LaunchConfiguration("glog_verbosity"),
        }.items(),
        condition=IfCondition(start_hydra),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "pipeline_config",
            default_value=PathJoinSubstitution(
                [share, "config", "rsg_pipeline_tesse.yaml"]
            ),
            description="RSG pipeline profile; defaults to the official TESSE uHumans2 bag.",
        ),
        DeclareLaunchArgument("start_chroma", default_value="true"),
        DeclareLaunchArgument("start_qwen", default_value="true"),
        DeclareLaunchArgument("start_hydra", default_value="true"),
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
        DeclareLaunchArgument("hydra_extra_yaml", default_value="{show_run_settings: false, config_verbosity: 0}"),
        DeclareLaunchArgument("glog_level", default_value="0"),
        DeclareLaunchArgument("glog_verbosity", default_value="0"),
        rsg_stack,
        hydra_stack,
    ])
