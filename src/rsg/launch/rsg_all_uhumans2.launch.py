"""Launch the complete RSG + Hydra + RViz stack for the Tesse/uHumans2 bag.

Thin wrapper around rsg_all.launch.py with the Tesse/uHumans2 dataset
arguments made explicit -- they already are rsg_all.launch.py's defaults;
this file exists so the dataset choice is a filename, not a wall of launch
arguments to remember. Rosbag playback is intentionally not started here;
run ``ros2 run rsg rsg_play_uhumans2`` in a separate terminal.
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the ROS 2 launch description for this component."""
    share = FindPackageShare("rsg")

    rsg_all = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "rsg_all.launch.py"])
        ),
        launch_arguments={
            "pipeline_config": PathJoinSubstitution([share, "config", "rsg_pipeline.yaml"]),
            "dataset": "uhumans2",
            "sensor_frame": "left_cam",
            "robot_frame": "base_link_gt",
            "odom_frame": "world",
            "map_frame": "world",
            "input_config": PathJoinSubstitution(
                [share, "config", "hydra", "rsg_phase1_input_tesse.yaml"]
            ),
            "rviz_config": PathJoinSubstitution(
                [share, "config", "rviz", "rsg_hydra_rap_fused_scene_graph.rviz"]
            ),
        }.items(),
    )

    return LaunchDescription([rsg_all])
