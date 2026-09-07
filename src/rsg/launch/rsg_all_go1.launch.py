"""Launch the complete RSG + Hydra + RViz stack for the real-world Go1/D455 bag.

Thin wrapper around rsg_all.launch.py with the go1 dataset's arguments
pre-filled, plus the two TF bridges this bag needs that the Tesse bag does
not: this bag carries no /tf at all, but Hydra's RosInputModule::getBodyPose
does an unconditional TF lookup for the moving robot pose, and
ros_sensors.cpp does a one-time TF lookup for the static camera extrinsic
(see rsg_pipeline_go1.yaml's header comment for the full story). So this
file also starts:

  - odom_to_tf (hydra_ros): republishes /go1_controller/odom as a live
    odom -> base transform.
  - a static_transform_publisher for the measured base ->
    camera_color_optical_frame mounting calibration.

Rosbag playback is intentionally not started here; run

    ros2 bag play <path-to-go1-bag> --clock --rate <r>

in a separate terminal (see the go1 bag under
~/datasets/rebased_bags/go1_d455_20180128_170537/ for the one already
converted to ROS 2).
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Build the ROS 2 launch description for this component."""
    share = FindPackageShare("rsg")

    odom_to_tf = Node(
        package="hydra_ros",
        executable="odom_to_tf",
        name="odom_to_tf",
        remappings=[("~/odom", "/go1_controller/odom")],
        parameters=[{
            "parent_frame": "odom",
            "child_frame": "base",
            "use_sim_time": True,
        }],
        output="screen",
    )

    base_to_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="rsg_base_to_camera_static_tf",
        arguments=[
            "--x", "0.10", "--y", "0.00", "--z", "0.159",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "base", "--child-frame-id", "camera_color_optical_frame",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    rsg_all = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([share, "launch", "rsg_all.launch.py"])
        ),
        launch_arguments={
            "pipeline_config": PathJoinSubstitution(
                [share, "config", "rsg_pipeline_go1.yaml"]
            ),
            "dataset": "simmons_a1",
            "sensor_frame": "camera_color_optical_frame",
            "robot_frame": "base",
            "odom_frame": "odom",
            "map_frame": "odom",
            "input_config": PathJoinSubstitution(
                [share, "config", "hydra", "rsg_phase1_input_go1.yaml"]
            ),
            "rviz_config": PathJoinSubstitution(
                [share, "config", "rviz", "rsg_hydra_rap_fused_scene_graph_go1.rviz"]
            ),
        }.items(),
    )

    return LaunchDescription([odom_to_tf, base_to_camera_tf, rsg_all])
