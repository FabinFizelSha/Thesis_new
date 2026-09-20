"""Launch the complete RSG, Hydra, and RViz stack in one command.

The launch combines the existing three-node RSG stack with the existing Hydra
input/mapping/visualization launch. It keeps the baseline launch defaults and
only exposes them as top-level arguments. Rosbag playback is intentionally not
started here; run ``ros2 run rsg rsg_play_uhumans2`` in a separate terminal.
The helper defaults to the external dataset at ``/home/student/datasets``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from nodes.support.workspace_paths import workspace_path

# Namespaced by dataset (memory/hydra/<dataset>/...) so switching datasets
# can never silently resume another dataset's saved DSG/mesh -- confirmed
# 2026-09-19: an OpenLoRIS run loaded uhumans2's leftover mesh/objects because
# both shared one fixed path here. Phase1's own tracker state_path was
# already namespaced per pipeline-config file (see
# rsg_pipeline_openloris.yaml's phase1.persistent_tracking.session_persistence.state_path);
# this was the one piece of resumable state that wasn't.
_HYDRA_MEMORY_ROOT = str(workspace_path("memory", "hydra"))


def _default_hydra_log_path(dataset: str) -> str:
    return str(workspace_path("memory", "hydra", dataset))


def _default_hydra_load_state_path(dataset: str) -> str:
    return str(workspace_path("memory", "hydra", dataset, "backend", "dsg_with_mesh.json"))


def _launch_hydra_stack(context, share, rsg_stack_include_source):
    """Build the Hydra IncludeLaunchDescription with hydra_load_state_path
    corrected for phase1.persistent_tracking.session_persistence.enabled.

    Without this, hydra_load_state_path always defaults to the real state
    file's path, and Hydra resumes from it whenever that file exists on disk
    -- completely independent of the persistence config. There was no way to
    turn resume off short of deleting memory/hydra/ before every single
    launch. This reads the actual pipeline config being used and disables
    resume (the documented "none" sentinel -- see hydra.launch.yaml and
    hydra_ros_pipeline.cpp's load_state_path check) whenever persistence is
    configured off, unless the caller explicitly passed a non-default
    hydra_load_state_path (a deliberate one-off resume for testing, which is
    still honoured either way).
    """
    import yaml

    pipeline_config_path = LaunchConfiguration("pipeline_config").perform(context)
    requested_state_path = LaunchConfiguration("hydra_load_state_path").perform(context)
    dataset = LaunchConfiguration("dataset").perform(context)

    persistence_enabled = True
    try:
        with open(pipeline_config_path) as f:
            cfg = yaml.safe_load(f) or {}
        persistence_enabled = bool(
            cfg.get("phase1", {})
            .get("persistent_tracking", {})
            .get("session_persistence", {})
            .get("enabled", True)
        )
    except Exception:
        pass  # config unreadable -- fail open to the previous always-resume behaviour

    if not persistence_enabled and requested_state_path == _default_hydra_load_state_path(dataset):
        effective_state_path = "none"
    else:
        effective_state_path = requested_state_path

    hydra_stack = IncludeLaunchDescription(
        rsg_stack_include_source,
        launch_arguments={
            "input_config": LaunchConfiguration("input_config"),
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
            "visualization_odom_bridge_child_frame": LaunchConfiguration(
                "visualization_odom_bridge_child_frame"
            ),
            "publish_sensor_body_frame_bridge": LaunchConfiguration(
                "publish_sensor_body_frame_bridge"
            ),
            "hydra_extra_yaml": LaunchConfiguration("hydra_extra_yaml"),
            "hydra_log_path": LaunchConfiguration("hydra_log_path"),
            "hydra_load_state_path": effective_state_path,
            "hydra_resume_reset_trajectory": LaunchConfiguration("hydra_resume_reset_trajectory"),
            "hydra_enable_object_merging": LaunchConfiguration("hydra_enable_object_merging"),
            "glog_level": LaunchConfiguration("glog_level"),
            "glog_verbosity": LaunchConfiguration("glog_verbosity"),
        }.items(),
    )
    return [hydra_stack]


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
            "fuser_config": LaunchConfiguration("fuser_config"),
            "start_chroma": LaunchConfiguration("start_chroma"),
            "start_qwen": LaunchConfiguration("start_qwen"),
            "start_risk_vlm": LaunchConfiguration("start_risk_vlm"),
        }.items(),
    )

    hydra_stack_source = PythonLaunchDescriptionSource(
        PathJoinSubstitution([share, "launch", "rsg_hydra_from_phase1.launch.py"])
    )
    hydra_stack = OpaqueFunction(
        function=_launch_hydra_stack,
        args=[share, hydra_stack_source],
        condition=IfCondition(start_hydra),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "pipeline_config",
            default_value=PathJoinSubstitution(
                [share, "config", "rsg_pipeline.yaml"]
            ),
            description="RSG pipeline profile; defaults to the official TESSE uHumans2 bag.",
        ),
        DeclareLaunchArgument(
            "fuser_config",
            default_value=PathJoinSubstitution(
                [share, "config", "rsg_scene_graph_fuser.yaml"]
            ),
            description="Scene graph fuser (marker/visualization) parameters; per-profile overrides (e.g. object marker/text size) live in their own file.",
        ),
        DeclareLaunchArgument("start_chroma", default_value="true"),
        DeclareLaunchArgument("start_qwen", default_value="true"),
        DeclareLaunchArgument("start_risk_vlm", default_value="true"),
        DeclareLaunchArgument("start_hydra", default_value="true"),
        DeclareLaunchArgument(
            "input_config",
            default_value=PathJoinSubstitution(
                [share, "config", "hydra", "rsg_phase1_input_tesse.yaml"]
            ),
            description="Hydra ROS input config; defaults to the official TESSE uHumans2 bag.",
        ),
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
        DeclareLaunchArgument("visualization_odom_bridge_child_frame", default_value="odom"),
        # See rsg_hydra_from_phase1.launch.py's sensor_body_frame_bridge comment.
        DeclareLaunchArgument("publish_sensor_body_frame_bridge", default_value="false"),
        DeclareLaunchArgument("hydra_extra_yaml", default_value="{show_run_settings: false, config_verbosity: 0}"),
        # Forwarded to rsg_hydra_from_phase1.launch.py. Declared here too because
        # a launch argument not declared at this level cannot be set from the
        # command line when launching rsg_all.
        # Namespaced by dataset -- see the _HYDRA_MEMORY_ROOT comment above.
        DeclareLaunchArgument(
            "hydra_log_path",
            default_value=PathJoinSubstitution([_HYDRA_MEMORY_ROOT, LaunchConfiguration("dataset")]),
        ),
        # Default kept as the real per-dataset state-file path for backward
        # compatibility (an explicit override here is always honoured) -- but
        # _launch_hydra_stack above forces this to "none" (disabled) whenever
        # phase1.persistent_tracking.session_persistence.enabled is false in the
        # active pipeline_config, regardless of what is sitting on disk at this
        # path. See that function's docstring.
        DeclareLaunchArgument(
            "hydra_load_state_path",
            default_value=PathJoinSubstitution(
                [_HYDRA_MEMORY_ROOT, LaunchConfiguration("dataset"), "backend", "dsg_with_mesh.json"]
            ),
        ),
        # This flag ONLY controls whether Hydra's restored agent/trajectory
        # nodes are kept (false) or dropped (true, default) -- it has no
        # effect on the robot's actual position, which comes entirely from
        # whatever /tf the bag or live sensor is currently publishing. Keeping
        # the old nodes (false) risks a NEW pose silently failing to insert if
        # its id collides with a retained one (graph.hasNode -> skip), which
        # is what "trace stops updating after resume" looks like -- confirmed
        # 2026-09-08. Default true: drop them, so each run's visible trace is
        # just its own, with no collision risk, while a properly paused (not
        # restarted) bag still continues the real position seamlessly.
        DeclareLaunchArgument("hydra_resume_reset_trajectory", default_value="true"),
        # datasets/uhumans2.yaml sets backend.enable_node_merging: false. See the
        # comment at rsg_hydra_from_phase1.launch.py's declaration of this arg.
        DeclareLaunchArgument("hydra_enable_object_merging", default_value="true"),
        DeclareLaunchArgument("glog_level", default_value="0"),
        DeclareLaunchArgument("glog_verbosity", default_value="0"),
        rsg_stack,
        hydra_stack,
    ])
