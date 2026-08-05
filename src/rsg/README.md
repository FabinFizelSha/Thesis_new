# Risk Scene Graph (RSG)

RSG is one ROS 2 package with three project-owned nodes:

- `rsg_preprocessor` — synchronizes and validates RGB-D, CameraInfo, and pose.
- `rsg_phase1_semantic_coordinator` — performs NanoSAM segmentation, persistent
  slot tracking, RAP retrieval, VLM fallback, and Hydra semantic-image output.
- `rsg_scene_graph_fuser` — fuses Hydra DSG object nodes with final Phase 1
  semantic and mobility metadata, applies configurable presence decay, and
  publishes RViz markers.

Hydra, Chroma, and the Qwen server are external services.

## Build

```bash
cd ~/rsg_ros2_ws
source /opt/ros/iron/setup.bash
colcon build --symlink-install --packages-select rsg --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
```

## Launch

```bash
# Three RSG nodes plus Chroma and Qwen. Hydra is launched separately.
ros2 launch rsg rsg_full_stack.launch.py

# Hydra, native visualization, fused RViz, and the three RSG nodes together.
ros2 launch rsg rsg_all.launch.py
```

See `docs/ARCHITECTURE.md`, `docs/MOBILITY_AWARE_DECAY.md`, and
`docs/PROJECT_LAYOUT.md`.
