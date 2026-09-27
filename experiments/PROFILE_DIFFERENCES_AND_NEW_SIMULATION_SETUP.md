# RSG Pipeline Profiles: uhumans2 (simulation) vs. OpenLoRIS (real sensor), and how to add a new simulation profile

This documents the actual, verified differences between the two profiles that ship with
this pipeline today, and gives a concrete step-by-step for adding a **new simulation**
profile alongside them.

Core fact to hold onto throughout: **there is only one node graph and one launch file
tree.** `rsg_all.launch.py` / `rsg_full_stack.launch.py` are identical for every profile.
A "profile" is not a fork of the code — it is a bundle of config file paths and launch
argument values fed into that same shared graph. Adding a new profile means creating a
new bundle of config files, never touching the launch files or node source.


## 1. What actually differs between uhumans2 and OpenLoRIS today

### 1.1 Phase1 pipeline config (`phase1.<node>` parameters)
- `src/rsg/config/rsg_pipeline.yaml` (uhumans2, the default)
- `src/rsg/config/rsg_pipeline_openloris.yaml` (real sensor)

This is the single biggest source of divergence: VLM prompt text, SAM/crop-quality
thresholds, `persistent_tracking` association thresholds (centroid/volume/quorum gates),
RAP settings, VLM label-confidence thresholds, diagnostics toggles. Most of the tuning
work in this project happens in this file per-profile.

### 1.2 Fuser config (RViz marker/visualization parameters)
- `src/rsg/config/rsg_scene_graph_fuser.yaml` (uhumans2)
- `src/rsg/config/rsg_scene_graph_fuser_openloris.yaml` (real sensor)

Marker/text sizes, `show_slot_ids`, presence half-lives, object-collapse distance
thresholds. Scaled independently because the two scenes are at different physical
scales and camera distances.

### 1.3 Hydra dataset config (voxel/TSDF/semantic-integration parameters)
- `src/hydra/config/datasets/uhumans2.yaml`
- `src/hydra/config/datasets/openloris.yaml`

Confirmed real behavioral difference, not just a tuning number:
`backend.enable_node_merging` is `false` for uhumans2 and `true` for openloris — Hydra's
own geometric object-merge pass (`UpdateObjectsFunctor::findMerges`, bbox-contains-
centroid only, no semantic check) runs for OpenLoRIS but not for uhumans2.

### 1.4 ROS2 QoS override files
- `src/rsg/config/rsg_pipeline_uhumans2_qos_overrides.yaml`
- `src/rsg/config/rsg_pipeline_openloris_qos_overrides.yaml`

Separate reliability/history-depth tuning, presumably because a real-sensor bag replay
and a simulated TESSE bag have different publishing/timing characteristics.

### 1.5 Hydra ROS input config (topic wiring)
- `src/rsg/config/hydra/rsg_phase1_input_tesse.yaml`
- `src/rsg/config/hydra/rsg_phase1_input_openloris.yaml`

Which topics Hydra's input adapter subscribes to (RGB/depth/camera-info/pose) — these
differ because the two bags publish under different topic names/conventions.

### 1.6 Launch-time arguments (same launch file, different values)
Declared in `rsg_all.launch.py`, passed straight through to the shared `rsg_hydra_from_
phase1.launch.py` / `rsg_full_stack.launch.py`:

| Argument | uhumans2 default | OpenLoRIS value |
|---|---|---|
| `dataset` | `uhumans2` | `openloris` |
| `sensor_frame` | `left_cam` | `camera_color_optical_frame` |
| `robot_frame` | `base_link_gt` | `camera_link` |
| `odom_frame` | `world` | `base_odom_rot45` (see 1.7) |
| `map_frame` | `world` | `base_odom_rot45` |
| `pipeline_config` | `rsg_pipeline.yaml` | `rsg_pipeline_openloris.yaml` |
| `fuser_config` | `rsg_scene_graph_fuser.yaml` | `rsg_scene_graph_fuser_openloris.yaml` |
| `input_config` | `rsg_phase1_input_tesse.yaml` | `rsg_phase1_input_openloris.yaml` |

The `dataset` argument alone also namespaces **all persisted state** —
`memory/hydra/<dataset>/...` and `hydra_load_state_path` are both derived from it — so
switching datasets can never silently resume another profile's saved map. This was a
real incident (2026-09-19: an OpenLoRIS run loaded uhumans2's leftover mesh because both
used one fixed path, before the namespacing was added).

### 1.7 OpenLoRIS-only: manual TF workaround
`odom_frame`/`map_frame:=base_odom_rot45` is a **new frame that doesn't exist in the
bag** — it's a small fixed-yaw (45°) static transform bridge from `base_odom`, published
by a separate, manually-run static transform publisher, **not part of the main launch
files**. It exists to correct for the real rig's world-frame heading being arbitrary
relative to the room, which was causing thin/angled real objects (walls, doors) to
render as fat, square-ish AABBs in RViz. A clean simulator with a sane world frame
should not need this at all — see §2.5.

### 1.8 RAP/Chroma memory path (env var, not a config file)
- uhumans2: `storage_path: ${WORKSPACE_ROOT}/memory/rap/chroma` (the default if
  `RSG_RAP_STORAGE_PATH` is unset)
- OpenLoRIS: `storage_path: ${WORKSPACE_ROOT}/memory/rap/chroma_openloris`

**Important trap**: `rsg_full_stack.launch.py` starts the actual Chroma server process
using **only** the `RSG_RAP_STORAGE_PATH` environment variable (defaulting to the
uhumans2 path) — it does **not** read `storage_path` out of whichever `pipeline_config`
you pass. The phase1 node, separately, reads `storage_path` from its own yaml to know
where to query. If these two don't match, the server and the node are pointed at
different stores. You must manually `export RSG_RAP_STORAGE_PATH=...` to match your
profile's yaml before launching, every time you switch profiles.

### 1.9 Everything not listed above is shared
Node graph, launch file logic, all Python (`phase1.py`, `persistent_object_tracker.py`,
etc.) and C++ (`fuser.cpp`) source, the VLM/RAP backend server processes, the label
space (`rsg_slot_only_frozen_label_space.yaml`) — all identical across profiles.


## 2. Adding a new profile that is ALSO a simulation

Since the new setup is a simulator (not a real sensor rig), **clone the uhumans2 files
as your starting template, not OpenLoRIS's** — OpenLoRIS's real-sensor concerns (§1.7's
TF workaround, its QoS tuning for real-bag jitter) most likely don't apply to a clean
new sim, unless your new simulator turns out to have similarly awkward frames.

Call the new profile `<new_dataset>` below (e.g. `carla`, `isaac_office`, whatever name
fits) — use the same string everywhere it appears.

### 2.1 Work out what your new simulator actually publishes, first
Before touching any config, run the new sim/bag standalone and check:
- `ros2 topic list` — RGB, depth, camera_info, and pose/odom topic names.
- `ros2 run tf2_tools view_frames` (or `ros2 topic echo /tf` / `/tf_static`) — the real
  frame names it publishes, and whether the world frame is sanely aligned (axis-aligned
  to the room/scene) or arbitrary like OpenLoRIS's was.
- Whether it publishes ground-truth pose (like uhumans2's `base_link_gt`) or only noisy
  odometry.

This determines your `sensor_frame`/`robot_frame`/`odom_frame`/`map_frame` values and
whether you need anything like §1.7's TF bridge (hopefully not, for a clean sim).

### 2.2 Hydra dataset config
Create `src/hydra/config/datasets/<new_dataset>.yaml` by copying
`src/hydra/config/datasets/uhumans2.yaml`. Adjust:
- `tsdf`/voxel size to match your new sensor's effective depth noise and scene scale.
- `tsdf.semantic_integrator.label_confidence` and related MLE settings if your labeling
  pipeline behaves differently (unlikely to need changes if you keep the same VLM/RAP
  backend).
- `backend.enable_node_merging` — start `false` (matching uhumans2) unless you have a
  specific reason to want Hydra's own geometric merge pass active; it's a real, distinct
  behavior (see §1.3), not a tuning knob to flip casually.

### 2.3 Hydra ROS input config
Create `src/rsg/config/hydra/rsg_phase1_input_<new_dataset>.yaml` by copying
`rsg_phase1_input_tesse.yaml`. Rewire every topic name to match what you found in §2.1.

### 2.4 Phase1 pipeline config
Create `src/rsg/config/rsg_pipeline_<new_dataset>.yaml` by copying `rsg_pipeline.yaml`
(the uhumans2 one) in full. Things you will likely need to change:
- `phase1.rap.storage_path` → a new, dedicated directory, e.g.
  `${WORKSPACE_ROOT}/memory/rap/chroma_<new_dataset>` — never reuse another profile's
  path (see §1.8's trap).
- `phase1.persistent_tracking.session_persistence.state_path` → give it its own path
  too, so resuming this profile can never load another profile's saved tracker state.
- `phase1.vlm.prompt` — only if your new scene's object vocabulary is meaningfully
  different from uhumans2's (e.g. very different room types/object classes); otherwise
  the existing prompt should transfer fine since it's not scene-specific.
- SAM/crop thresholds (`min_crop_quality_score`, `max_masks`, `points_per_side`, etc.)
  — only if your new sim's image resolution/FOV is very different from uhumans2's.
  Start with uhumans2's values unchanged and only tune if you see a specific problem
  in a real run — this whole session's history is a long list of examples of what
  over-tuning without evidence looks like.
- `persistent_tracking` association thresholds (`global_centroid_pass_m`,
  `max_volume_ratio`, `global_min_independent_groups`, etc.) — leave these at
  uhumans2's already-tuned values initially. These were arrived at through extensive,
  evidence-based tuning (see the OpenLoRIS profile's own commit history/comments for
  the reasoning behind each one) and are a reasonable starting point for any profile
  until you have concrete evidence of a specific failure mode in your new sim.

### 2.5 Fuser config
Create `src/rsg/config/rsg_scene_graph_fuser_<new_dataset>.yaml` by copying
`rsg_scene_graph_fuser.yaml`. Adjust marker/text sizes only if your new scene's physical
scale is very different from uhumans2's apartment-scale scene (e.g. a warehouse-scale
sim would want larger markers, a small-room sim smaller ones).

### 2.6 QoS overrides
Create `src/rsg/config/rsg_pipeline_<new_dataset>_qos_overrides.yaml` by copying
`rsg_pipeline_uhumans2_qos_overrides.yaml`. A clean simulator publishing reliably at a
steady rate (like uhumans2) usually needs no changes here at all.

### 2.7 TF frame launch arguments
From §2.1's findings, set:
- `sensor_frame:=<your camera optical frame>`
- `robot_frame:=<your body/base frame>`
- `odom_frame:=<your odometry/world frame>`
- `map_frame:=<usually the same as odom_frame for a sim with ground-truth pose>`

If (and only if) your sim's world frame turns out to be arbitrarily rotated relative to
the scene the way OpenLoRIS's was, you'll need §1.7's kind of fixed-yaw static
transform bridge — publish it as a **separate, manual step** (not baked into the shared
launch files), matching how it's done for OpenLoRIS today.

### 2.8 Full launch command
```bash
export RSG_RAP_STORAGE_PATH=/path/to/workspace/memory/rap/chroma_<new_dataset>

ros2 launch rsg rsg_all.launch.py \
  dataset:=<new_dataset> \
  pipeline_config:=<path>/rsg_pipeline_<new_dataset>.yaml \
  fuser_config:=<path>/rsg_scene_graph_fuser_<new_dataset>.yaml \
  input_config:=<path>/rsg_phase1_input_<new_dataset>.yaml \
  sensor_frame:=<your sensor frame> \
  robot_frame:=<your robot frame> \
  odom_frame:=<your odom frame> \
  map_frame:=<your map frame> \
  use_sim_time:=true
```
(`use_sim_time:=true` matches uhumans2's default — keep it true for any bag-replay
simulation so Hydra's clock follows the bag, not the wall clock.)

### 2.9 First-run verification checklist
- Chroma starts pointed at the **new** storage path (check the server's own startup log
  line, not just that it started).
- `ros2 run tf2_tools view_frames` shows the expected frame tree with no missing links.
- RViz's camera-facing arrow and the first few object bounding boxes are sane
  (axis-aligned to real room geometry, not rotated — the exact symptom the OpenLoRIS
  TF bridge was built to fix, so this is the first sanity check for a *new* profile
  that skips that bridge).
- `memory/hydra/<new_dataset>/` and the new RAP storage directory are being created
  fresh, not silently resuming/mixing with another profile's saved state.
- A handful of VLM crops in the new profile's session folder look reasonable for your
  scene's object vocabulary before doing any prompt tuning.

Only after a baseline run like this looks broadly correct is it worth tuning any of the
association/SAM thresholds — and even then, change one value at a time with a concrete
observed failure to point at, the same way every tuning change in the OpenLoRIS profile
was justified in this project's history.
