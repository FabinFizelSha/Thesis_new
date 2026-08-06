"""Launch only the RSG RGB-D preprocessing node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create the standalone preprocessing launch description."""
    share = FindPackageShare("rsg")
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution([share, "config", "rsg_pipeline.yaml"]),
        ),
        ExecuteProcess(
            cmd=[
                "python3",
                PathJoinSubstitution([share, "scripts", "rsg_preprocessor"]),
                "--ros-args",
                "-p",
                ["config_file:=", config_file],
            ],
            name="rsg_preprocessor",
            output="screen",
            emulate_tty=True,
        ),
    ])
