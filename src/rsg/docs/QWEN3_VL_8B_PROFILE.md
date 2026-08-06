# Qwen3-VL Model Profiles

## Purpose

The RSG package retains the current `qwen3_vl_2b_q4` configuration and adds a
selectable `qwen3_vl_8b_f16` profile. Both profiles use the same local
OpenAI-compatible llama.cpp endpoint on port `8000`; select one profile at a
time and never start both models concurrently.

The scheduler, crop-quality gate, RAP settings, prompt, and VLM request format
are shared by both profiles. This makes 2B-versus-8B runs comparable: the
selected VLM receives the same crop for a given track.

## Profiles

| Profile | Model weights | Intended use |
|---|---|---|
| `qwen3_vl_2b_q4` | Qwen3-VL-2B-Instruct, Q4_K_M GGUF | Current low-memory, lower-latency model. Default. |
| `qwen3_vl_8b_f16` | Qwen3-VL-8B-Instruct, FP16 GGUF + FP16 vision projector | Quality-oriented evaluation model. Requires a high-memory Jetson. |

## Selecting a profile

Edit the source configuration:

```bash
nano ~/rsg_ros2_ws/src/rsg/config/rsg_pipeline.yaml
```

For the 8B FP16 model:

```yaml
phase1:
  vlm:
    active_profile: qwen3_vl_8b_f16
```

For the original model:

```yaml
phase1:
  vlm:
    active_profile: qwen3_vl_2b_q4
```

Rebuild and source after any source-YAML change:

```bash
cd ~/rsg_ros2_ws
colcon build --packages-select rsg --symlink-install
source install/setup.bash
```

## Download options

### Automatic Hugging Face download

Leave `model_path` and `mmproj_path` empty. The profile-aware launcher passes
its `hf_model` value to `llama-server`, which downloads and caches the selected
GGUF model and vision projector on first launch.

### Manual download to a fixed local directory

Use the official `huggingface-cli` commands documented in the main setup
instructions, then set both paths in the selected profile:

```yaml
model_path: ~/rsg_models/qwen3_vl_8b_f16/Qwen3VL-8B-Instruct-F16.gguf
mmproj_path: ~/rsg_models/qwen3_vl_8b_f16/mmproj-Qwen3VL-8B-Instruct-F16.gguf
```

With `model_path` set, the launcher uses `-m` and `--mmproj` instead of `-hf`.
Both paths must exist before the stack launches.

## Resource guard

The 8B FP16 profile has `min_system_memory_gib: 32`. This is a startup guard,
not a promise that 32 GiB will provide enough free memory once ROS 2, Hydra,
NanoSAM, Chroma, and queues are active. Do not attempt the profile on a 16 GB
Orin. If it fails to load or causes memory pressure, return to the 2B Q4
profile; do not run both models simultaneously.

## Preflight command

After rebuilding, print the exact server command without launching ROS nodes:

```bash
python3 ~/rsg_ros2_ws/install/rsg/share/rsg/scripts/rsg_vlm_server \
  --config ~/rsg_ros2_ws/install/rsg/share/rsg/config/rsg_pipeline.yaml \
  --dry-run
```

Expected 8B output contains `Qwen/Qwen3-VL-8B-Instruct-GGUF:F16` unless local
model paths were configured.
