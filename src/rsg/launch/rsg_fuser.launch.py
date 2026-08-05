"""Launch only the RSG Hydra semantic scene-graph fuser."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create the standalone fuser launch description."""
    share = FindPackageShare("rsg")
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution([share, "config", "rsg_scene_graph_fuser.yaml"]),
        ),
        Node(
            package="rsg",
            executable="rsg_scene_graph_fuser",
            name="rsg_scene_graph_fuser",
            output="screen",
            parameters=[config_file],
        ),
    ])
