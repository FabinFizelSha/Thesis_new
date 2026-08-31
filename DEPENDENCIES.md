# RSG Pipeline — Comprehensive Dependency List

Everything required to build and run the **Risk Scene Graph (`rsg`)** ROS 2 pipeline
(`ros2 launch rsg rsg_all.launch.py` / `rsg_full_stack.launch.py`).

Captured **2026-08-31** from a working install.

---

## 0. Target platform

| | |
|---|---|
| Device | NVIDIA **Jetson Orin** (`aarch64`) |
| JetPack / L4T | **6.2** / R36.5.0 |
| OS | Ubuntu **22.04.5** LTS (`jammy`) |
| Python | **3.10.12** (system) |
| CUDA | **12.6** (V12.6.68), driver 540.5.0 |
| ROS 2 | **Humble** (Ubuntu 22.04 → Humble; repo READMEs say Iron/Jazzy but those are for 24.04) |
| Compute arch | `sm_87` (Orin) |
| CPU / RAM | 12 cores / 61 GB + 30 GB swap |

Disk footprint of the full install: ROS ~200 MB, CUDA 12.6 ~4.2 GB, venv ~3.3 GB,
llama.cpp build ~0.5 GB, colcon build+install ~0.35 GB, model weights (user) ~3 GB+.

---

## 1. ROS 2 Humble (apt)

Repo added via the modern `ros2-apt-source` package (`ros-apt-source` v1.2.0), then:

```bash
sudo apt install ros-humble-desktop ros-dev-tools
```

Pulls ~280 `ros-humble-*` packages. Key ones the workspace uses directly:

| Package | Version | Purpose |
|---|---|---|
| `ros-humble-desktop` | 0.10.0-1jammy | full ROS 2 desktop (rclpy, rclcpp, rviz2, …) |
| `ros-dev-tools` | 1.0.1 | build tooling metapackage |
| `ros-humble-cv-bridge` | 3.2.1 | ROS image ↔ OpenCV/NumPy (used by `phase1.py`, preprocessor) |
| `ros-humble-gtsam` | **4.2.0** | factor-graph backend for Hydra (apt-packaged — no source build) |
| `ros-humble-rviz2` | 11.2.28 | visualization |
| `ros-humble-tf2-ros` | 0.25.22 | transforms |
| `ros-humble-message-filters` | 4.3.19 | RGB-D + pose sync |
| `ros-humble-image-transport` | 3.1.13 | image transport |
| `ros-humble-vision-opencv` | — | OpenCV ROS integration |

### Build tooling (apt)

| Package | Version |
|---|---|
| `python3-colcon-common-extensions` | 0.3.0-100 |
| `python3-rosdep` | 0.26.0-1 |
| `python3-vcstool` | 0.3.0-1 |

`sudo rosdep init && rosdep update` was run.

---

## 2. Workspace C/C++ system dependencies (resolved by `rosdep`)

```bash
cd ~/Thesis_new
rosdep install --from-paths src --ignore-src --rosdistro humble --skip-keys ament_python -y
```

> `--skip-keys ament_python` is needed: `rsg_semantic_adapter` and `rsg_dsg_visualizer`
> declare `<buildtool_depend>ament_python</buildtool_depend>`, which has no rosdep key.

| Package | Version | Needed by |
|---|---|---|
| `ros-humble-gtsam` | 4.2.0 | hydra, kimera_rpgo |
| `nlohmann-json3-dev` | 3.10.5-2 | rsg (C++ fuser), config_utilities |
| `libcli11-dev` | 2.1.2 | hydra |
| `libzmq3-dev` | 4.3.4-2 | spark_dsg, hydra visualization bridge |
| `libgoogle-glog-dev` (+ `libgflags-dev`) | 0.5.0 / 2.2.2 | hydra, kimera |
| `python3-flask` | 2.0.1 | hydra python tooling |
| `python3-openpyxl` | 3.0.9 | rsg (timing Excel recorder) |

(Eigen comes via GTSAM; OpenCV via `cv_bridge`; `spatial_hash` replaces voxblox and is vendored.)

---

## 3. CUDA toolkit / TensorRT (apt) — for torch, llama.cpp CUDA, NanoSAM

The base image ships only a minimal CUDA runtime. These were added:

| Package | Version | Why |
|---|---|---|
| `cuda-libraries-dev-12-6` | 12.6.11-1 | cuBLAS/cuSPARSE/cuSOLVER/cuRAND/cuFFT/NPP **dev** — llama.cpp CUDA build needs `CUDA::cublas` |
| `cuda-cupti-12-6` | 12.6.68-1 | `libcupti.so.12` — required by torch 2.11 |
| `cuda-nvcc-12-6` | 12.6.68-1 | CUDA compiler (already present) |
| `cuda-cudart-dev-12-6` | 12.6.68-1 | CUDA runtime dev (already present) |
| `tensorrt` | **10.3.0.30**-1+cuda12.5 | NanoSAM TensorRT backend |
| `python3-libnvinfer` / `-dev` | 10.3.0.30 | TensorRT Python API (visible in venv via system-site-packages) |
| `libnvinfer-bin` | 10.3.0.30 | provides `trtexec` at `/usr/src/tensorrt/bin/trtexec` (builds the `.engine` files) |
| `libcurl4-openssl-dev` | 7.81.0 | llama.cpp `-DLLAMA_CURL=ON` (`-hf` model refs) |

**`libcudss.so.0`** (a torch 2.11 dependency) is **not** in the Jetson apt repos. It's
supplied by the pip wheel `nvidia-cudss-cu12` and made discoverable system-wide via:

```
/etc/ld.so.conf.d/zz-rsg-venv-nvidia.conf
  → /home/student/.venvs/rsg_thor/lib/python3.10/site-packages/nvidia/cu12/lib
sudo ldconfig
```

---

## 4. Python virtual environment

Path: **`/home/student/.venvs/rsg_thor`** — the `rsg` launch files hard-code this
(`RSG_VENV_PATH` env → `VIRTUAL_ENV` → default `~/.venvs/rsg_thor`). Also used for
`$venv/bin/chroma` (RAP-memory server) and `$venv/bin/python3` (all RSG nodes).

```bash
python3 -m venv --system-site-packages /home/student/.venvs/rsg_thor
V=/home/student/.venvs/rsg_thor
$V/bin/pip install 'setuptools<80' wheel
```

### 4a. PyTorch — Jetson CUDA wheels (install FIRST, own index)

```bash
$V/bin/pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
    torch==2.11.0 torchvision==0.26.0
```

| Package | Version | Notes |
|---|---|---|
| `torch` | **2.11.0** | CUDA 12.6, `sm_87`, `torch.cuda.is_available() → True` on Orin |
| `torchvision` | **0.26.0** | matched |
| `nvidia-cudss-cu12` | 0.8.0.10 | pulled as dep; provides `libcudss.so.0` (see §3) |
| `nvidia-cublas-cu12` | 12.9.2.10 | pulled as dep |
| `nvidia-cuda-nvrtc-cu12` | 12.9.86 | pulled as dep |

> **Do NOT `pip install torch` from PyPI** — that is a CPU-only build on aarch64.

### 4b. ⚠️ NumPy pin — hard constraint

**`numpy==1.26.4` (must stay `<2`).** ROS Humble's compiled extensions (`cv_bridge`,
`tf2_py`, …) are built against the NumPy 1.x C-ABI. NumPy 2 → `_ARRAY_API not found`
and runtime segfaults on the first image conversion.
Companion pins: `scipy==1.13.1`, `ml_dtypes==0.4.1`.

### 4c. RSG + ML stack (PyPI)

Install the pinned set from **Appendix A** (save it as `requirements.txt` and
`$V/bin/pip install -r requirements.txt`, or install the notable ones below and let
pip resolve the rest).

Direct/notable dependencies (full pinned list in **Appendix A**):

| Package | Version | Used for |
|---|---|---|
| `numpy` | 1.26.4 | **pinned <2** (see 4b) |
| `scipy` | 1.13.1 | geometry / filtering |
| `opencv-python` | 4.11.0.86 | image ops (venv copy; system `cv2` also present) |
| `pillow` | 12.3.0 | crops |
| `matplotlib` | 3.10.9 | debug plots |
| `networkx` | 3.4.2 | graph ops |
| `pandas` | 2.2.3 | experiment tables |
| `pyyaml` | 6.0.3 | config |
| `shapely` | 2.1.2 | spark_dsg python dep |
| `pyzmq` | 27.2.0 | spark_dsg / hydra viz |
| `transformers` | 5.16.1 | VLM tokenizer / utils |
| `chromadb` | 1.5.9 | RAP visual memory (server on :8001, `chroma` CLI) |
| `openai` | 3.6.0 | OpenAI-compatible client → llama.cpp endpoint |
| `fastapi` / `uvicorn[standard]` | 0.141.1 / 0.52.4 | internal HTTP services |
| `pydantic` | 2.13.5 | config models |
| `tenacity` | 8.5.0 | **<9** (chromadb 1.5 imports `retry_if_exception`) |
| `segment-anything` | 1.0 | SAM fallback backend |
| `ultralytics` | 8.4.136 | semantic_inference / detection utils |
| `onnx` / `onnxruntime` | 1.22.0 / 1.23.2 | model tooling |
| `pyqtgraph` | 0.14.0 | hydra python |
| `openpyxl` | 3.0.9 | timing Excel recorder |
| `toon_format` | 0.1.0 | `FormatConverter.py` (TOON encoding) |
| `rosbags` | 0.11.5 | bag utilities |
| `einops`, `imageio`, `seaborn`, `distinctipy`, `rich`, `ruamel.yaml` | — | semantic_inference python deps |

### 4d. NanoSAM + torch2trt (git, Phase 1 TensorRT segmentation)

```bash
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH
$V/bin/pip install --no-build-isolation "git+https://github.com/NVIDIA-AI-IOT/torch2trt.git"
git clone https://github.com/NVIDIA-AI-IOT/nanosam.git /home/student/nanosam
$V/bin/pip install --no-build-isolation -e /home/student/nanosam
```

| Package | Version / commit |
|---|---|
| `torch2trt` | 0.5.0 (`4e820ae`) |
| `nanosam` | 0.0 (`6536336`, editable at `/home/student/nanosam`) |
| `tensorrt` (py) | 10.3.0 — from apt `python3-libnvinfer`, visible via `--system-site-packages` |

---

## 5. llama.cpp — CUDA VLM server (built from source)

```bash
git clone https://github.com/ggml-org/llama.cpp.git /home/student/llama.cpp   # commit 8e53fce
cd /home/student/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build -j 8 --target llama-server llama-cli llama-mtmd-cli
```

Produces **`/home/student/llama.cpp/build/bin/llama-server`** (CUDA-linked:
`libggml-cuda`, `libcublas.so.12`, `libcudart.so.12`). Verified: loads Qwen3.5-VL-4B on
GPU, `/health` → `{"status":"ok"}`.

`rsg_pipeline.yaml` active profile = **`qwen3_5_4b_q4`**, resolves to:
```
llama-server -m .../Qwen3.5-4B-Q4_K_M.gguf --mmproj .../mmproj-BF16.gguf \
             --host 0.0.0.0 --port 8000 -ngl 99 -c 4096
```

---

## 6. ROS 2 workspace packages (vendored in `src/`, built by colcon)

`colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF`
→ **26 packages, exit 0**. No `vcs import` needed (all source is in-tree).

Project packages: **`rsg`** (the pipeline), `rsg_semantic_adapter`, `rsg_dsg_visualizer`,
`risk_scene_graph_core`, `risk_scene_graph_ros`.

Upstream (MIT-SPARK) packages: `hydra`, `hydra_ros` (+ `hydra_msgs`, `hydra_visualizer`),
`spark_dsg`, `spatial_hash`, `config_utilities` (+ `_ros`, `_msgs`), `kimera_pgmo`
(+ `_ros`, `_msgs`, `_rviz`), `kimera_rpgo`, `pose_graph_tools` (+ `_ros`, `_msgs`),
`teaser_plusplus` (`teaserpp`), `ianvs`, `semantic_inference` (+ `_msgs`).

---

## 7. Runtime services started by the launch file

| Service | Command | Port | Provided by |
|---|---|---|---|
| ChromaDB (RAP memory) | `$venv/bin/chroma run --host 127.0.0.1 --port 8001 --path ~/rsg_rap_memory` | 8001 | `chromadb[cli]` |
| Qwen VLM server | `$venv/bin/python <rsg>/scripts/rsg_vlm_server --config rsg_pipeline.yaml` | 8000 | llama.cpp `llama-server` |
| RSG preprocessor / phase1 / fuser | ROS nodes | — | `rsg` package |
| Hydra + visualizer + RViz | `rsg_hydra_from_phase1.launch.py` | — | `hydra_ros` |

---

## 8. External model assets — NOT dependencies, must be supplied

Referenced by `src/rsg/config/rsg_pipeline.yaml`:

| Asset | Path | Status |
|---|---|---|
| Qwen3.5-4B GGUF | `~/rsg_models/qwen3_5_4b/Qwen3.5-4B-Q4_K_M.gguf` | ✅ present |
| Qwen3.5-4B vision projector | `~/rsg_models/qwen3_5_4b/mmproj-BF16.gguf` | ✅ present |
| NanoSAM image encoder engine | `~/rsg_models/nanosam/resnet18_image_encoder.engine` | ❌ build with `trtexec` from NanoSAM ONNX |
| NanoSAM mask decoder engine | `~/rsg_models/nanosam/mobile_sam_mask_decoder.engine` | ❌ build with `trtexec` |
| SAM fallback checkpoint | `~/rsg_models/sam/sam_vit_b_01ec64.pth` | ❌ optional fallback (`vit_b`) |
| uHumans2 TESSE rosbag | `~/datasets/…` | ❌ dataset |

> TensorRT `.engine` files are device- and TRT-version-specific — build them **on this Orin**.

---

## 9. Environment setup for every run

```bash
source /opt/ros/humble/setup.bash
source ~/Thesis_new/install/setup.bash        # puts /opt/ros/humble/lib on LD_LIBRARY_PATH (libcv_bridge.so)
# RSG nodes run under ~/.venvs/rsg_thor/bin/python3 (launch files handle this)
```

Do **not** strip `LD_LIBRARY_PATH` — ROS libs and the ldconfig `libcudss` entry must both apply.

`sudo` note: passwordless sudo was enabled at `/etc/sudoers.d/91-student-nopasswd` for the
install; remove it if not wanted (`sudo rm /etc/sudoers.d/91-student-nopasswd`).

---

## Appendix A — Python venv pinned freeze

Save as `requirements.txt` and `pip install -r` it into `~/.venvs/rsg_thor` **after**
installing torch/torchvision from the Jetson index (§4a) and pinning `numpy==1.26.4` (§4b).
`torch*`, `torch2trt`, `nanosam`, `pip`, `setuptools`, `wheel` are commented — install per
§4a / §4d. Generated 2026-08-31 (venv-only packages; system-site inheritances excluded).

```text
# NOTE: numpy MUST stay <2 (ROS Humble binary ABI).
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.14.2
apsw==3.53.4.0
async-timeout==5.0.1
attrs==26.1.0
bcrypt==5.0.0
build==1.6.0
certifi==2026.7.22
charset-normalizer==3.5.1
chromadb==1.5.9
click==8.5.0
coloredlogs==15.0.1
contourpy==1.3.2
cuda-toolkit==12.9.2.0
cycler==0.12.1
distinctipy==1.3.4
durationpy==0.11
einops==0.8.2
exceptiongroup==1.3.1
fastapi==0.141.1
filelock==3.32.4
flatbuffers==25.12.19
fonttools==4.63.0
frozenlist==1.8.0
fsspec==2026.7.0
googleapis-common-protos==1.75.2
grpcio==1.83.1
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpcore2==2.12.0
httptools==0.8.0
httpx==0.28.1
httpx2==2.12.0
huggingface_hub==1.29.0
humanfriendly==10.0
idna==3.19
ImageIO==2.37.4
importlib_resources==7.1.0
jiter==0.16.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
kubernetes==36.0.3
markdown-it-py==4.2.0
matplotlib==3.10.9
mdurl==0.1.2
ml-dtypes==0.4.1
mmh3==5.3.0
mpmath==1.3.0
multidict==6.7.1
# nanosam==0.0   # git install - see §4d
networkx==3.4.2
numpy==1.26.4
nvidia-cublas-cu12==12.9.2.10
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cudss-cu12==0.8.0.10
nvidia-ml-py==13.610.43
onnx==1.22.0
onnxruntime==1.23.2
openai==3.6.0
opencv-python==4.11.0.86
opentelemetry-api==1.44.0
opentelemetry-exporter-otlp-proto-common==1.44.0
opentelemetry-exporter-otlp-proto-grpc==1.44.0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
orjson==3.12.0
overrides==7.7.0
packaging==26.3
pandas==2.2.3
pillow==12.3.0
# pip==26.2.1
polars==1.44.1
polars-runtime-32==1.44.1
propcache==0.5.2
protobuf==7.36.0
pybase64==1.5.0
pydantic==2.13.5
pydantic-settings==2.15.0
pydantic_core==2.46.5
Pygments==2.21.0
pyparsing==3.3.2
PyPika==0.51.1
pyproject_hooks==1.2.0
pyqtgraph==0.14.0
python-dateutil==2.9.0.post0
python-dotenv==1.2.3
pytz==2026.3.post1
PyYAML==6.0.3
pyzmq==27.2.0
referencing==0.37.0
regex==2026.8.31
requests==2.34.2
requests-oauthlib==2.0.0
rich==15.0.0
rosbags==0.11.5
rpds-py==0.30.0
ruamel.yaml==0.19.1
safetensors==0.8.0
scipy==1.13.1
seaborn==0.13.2
segment-anything==1.0
# setuptools==79.0.1
shapely==2.1.2
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
starlette==1.6.0
sympy==1.14.0
tenacity==8.5.0
tokenizers==0.23.1
tomli==2.4.1
toon-format==0.1.0
# torch==2.11.0   # Jetson index - see §4a
# torch2trt==0.5.0   # git install - see §4d
# torchvision==0.26.0   # Jetson index - see §4a
tqdm==4.70.0
transformers==5.16.1
truststore==0.10.4
typer==0.27.2
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.3
ultralytics==8.4.136
ultralytics-thop==2.1.6
urllib3==2.7.0
uvicorn==0.52.4
uvloop==0.22.1
watchfiles==1.2.0
websocket-client==1.9.2
websockets==16.1.1
# wheel==0.48.0
yarl==1.24.5
zstandard==0.25.0
```
