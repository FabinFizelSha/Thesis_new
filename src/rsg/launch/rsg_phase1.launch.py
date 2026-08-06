"""Launch only the RSG Phase 1 semantic coordinator."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create the standalone Phase 1 launch description."""
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
                PathJoinSubstitution([share, "scripts", "rsg_phase1_semantic_coordinator"]),
                "--ros-args",
                "-p",
                ["config_file:=", config_file],
            ],
            name="rsg_phase1_semantic_coordinator",
            output="screen",
            emulate_tty=True,
        ),
    ])
