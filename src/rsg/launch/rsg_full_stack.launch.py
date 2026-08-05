"""Launch the three RSG nodes and local RAP/VLM service processes.

This launch deliberately does not start Hydra. Use ``rsg_all.launch.py`` when
Hydra, its visualizer, and RViz should be started together as well.
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _python_node(name, script, config_file, python_executable):
    """Launch an installed Python ROS node with the configured RSG environment."""
    return ExecuteProcess(
        cmd=[python_executable, script, "--ros-args", "-p", ["config_file:=", config_file]],
        name=name,
        output="screen",
        emulate_tty=True,
    )


def generate_launch_description() -> LaunchDescription:
    """Create the three-node RSG stack using the validated baseline defaults."""
    share = FindPackageShare("rsg")
    pipeline_config = LaunchConfiguration("pipeline_config")
    fuser_config = PathJoinSubstitution([share, "config", "rsg_scene_graph_fuser.yaml"])
    preprocessor_script = PathJoinSubstitution([share, "scripts", "rsg_preprocessor"])
    phase1_script = PathJoinSubstitution([share, "scripts", "rsg_phase1_semantic_coordinator"])
    vlm_server_script = PathJoinSubstitution([share, "scripts", "rsg_vlm_server"])

    start_chroma = LaunchConfiguration("start_chroma")
    start_qwen = LaunchConfiguration("start_qwen")

    home_dir = Path.home()
    venv_dir = Path(
        os.environ.get(
            "RSG_VENV_PATH",
            os.environ.get("VIRTUAL_ENV", str(home_dir / ".venvs" / "rsg_thor")),
        )
    ).expanduser()
    python_executable = str(venv_dir / "bin" / "python3")
    chroma_executable = str(venv_dir / "bin" / "chroma")
    rap_storage_path = str(
        Path(
            os.environ.get(
                "RSG_RAP_STORAGE_PATH",
                str(home_dir / "rsg_rap_memory"),
            )
        ).expanduser()
    )

    chroma = ExecuteProcess(
        cmd=[
            chroma_executable,
            "run",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
            "--path",
            rap_storage_path,
        ],
        name="chroma_rap_memory",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(start_chroma),
    )

    qwen = ExecuteProcess(
        cmd=[python_executable, vlm_server_script, "--config", pipeline_config],
        name="qwen_vlm_server",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(start_qwen),
    )

    preprocessor = _python_node(
        "rsg_preprocessor", preprocessor_script, pipeline_config, python_executable
    )
    phase1 = _python_node(
        "rsg_phase1_semantic_coordinator",
        phase1_script,
        pipeline_config,
        python_executable,
    )
    fuser = Node(
        package="rsg",
        executable="rsg_scene_graph_fuser",
        name="rsg_scene_graph_fuser",
        output="screen",
        parameters=[fuser_config],
        additional_env={"RSG_PYTHON_EXECUTABLE": python_executable},
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
        chroma,
        qwen,
        preprocessor,
        TimerAction(period=2.0, actions=[phase1]),
        TimerAction(period=3.0, actions=[fuser]),
    ])
