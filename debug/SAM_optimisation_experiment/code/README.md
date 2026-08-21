# SAM Optimization Experiment - Code Module

## Overview

This directory contains all preprocessing, evaluation, and inference code used in the SAM optimization experiment.

**Key Modules**:
- `ground_truth.py` — F1/Precision/Recall evaluation
- `data_loader.py` — Dataset loading (RGB-D frames + GT masks)
- `runners.py` — Backend inference (NanoSAM, ViT-B)
- `timing.py` — Latency measurement and FPS computation
- `test_runner_template.py` — Complete test runner example

---

## Module Details

### 1. ground_truth.py - Evaluation

**Purpose**: Compute F1 score, precision, recall by matching predicted masks to ground truth

**Key Class**: `GroundTruthEvaluator`

**Algorithm**:
```python
# Instance-level matching
for each predicted_mask:
    find GT mask with max IoU
    if IoU >= 0.3:
        True Positive
    else:
        False Positive

for unmatched GT masks:
    False Negative

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Usage**:
```python
evaluator = GroundTruthEvaluator(iou_threshold=0.3)

for frame in dataset:
    masks = model.segment(frame['rgb'])
    
    metrics = evaluator.compute_metrics(
        predicted_masks=masks,
        ground_truth=frame['semantic'],
        depth_valid=frame['depth_valid'],
        min_mask_pixels=3500,
        min_gt_pixels=3500
    )
    
    print(f"F1: {metrics.f1_score:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
```

**Key Parameters**:
- `iou_threshold` (0.3): IoU threshold for considering a match valid
- `min_mask_pixels`: Minimum object size (filters noise)
- `min_gt_pixels`: Minimum GT object size (must match min_mask_pixels for fairness)

**Ground Truth Extraction**:
- Per-class connected component labeling (8-connectivity)
- Each connected component treated as separate instance
- Supports 2D [H, W] and 3D [H, W, C] semantic masks

---

### 2. data_loader.py - Dataset Loading

**Purpose**: Load Phase 1 Suite 1 dataset (300 RGB-D frames + ground truth)

**Key Class**: `Phase1DatasetLoader`

**Dataset Structure**:
```
dataset_root/
├── rgb_000000.npy           (1280×720×3, uint8)
├── rgb_000001.npy
├── ...
├── depth_000000.npy         (1280×720, float32, meters)
├── depth_000001.npy
├── ...
├── semantic_000000.npy      (1280×720, uint8)
└── semantic_000001.npy
```

**Usage**:
```python
loader = Phase1DatasetLoader("path/to/phase1_frames_300")

# Iterate over frames
for frame_data in loader:
    rgb = frame_data['rgb']              # [1280, 720, 3] uint8
    depth = frame_data['depth']          # [1280, 720] float32
    semantic = frame_data['semantic']    # [1280, 720] uint8
    frame_id = frame_data['frame_id']    # "000000"

# Random shuffle option
loader = Phase1DatasetLoader("path/...", random_shuffle=True)

# Direct access
frame = loader.get_frame(0)
```

**Properties**:
- 300 frames with full ground truth
- RGB-D format (color + depth)
- Per-class semantic segmentation
- Robotic manipulation domain

---

### 3. runners.py - Backend Inference

**Purpose**: Inference runners for NanoSAM and ViT-B backends

**Key Classes**:
- `BaseRunner` — Abstract base class
- `NanoSAMRunner` — Lightweight SAM (ResNet18)
- `ViTBRunner` — Full SAM (ViT-B)
- `get_backend_runner()` — Factory function

**Configuration Parameters**:
```yaml
backend: nanosam              # 'nanosam' or 'vitb'
device: cuda                  # 'cuda' or 'cpu'

# Model paths (TensorRT optimized)
image_encoder_engine: /path/to/encoder.engine
mask_decoder_engine: /path/to/decoder.engine

# SAM Parameters
points_per_side: 6            # Grid density (6×6 = 36 prompts)
max_masks: 24                 # Max masks per frame
mask_threshold: 0.80          # Confidence threshold
min_mask_pixels: 3500         # Minimum object size
nms_iou: 0.20                 # NMS suppression threshold
```

**Usage**:
```python
import yaml

# Load config
with open('phase2_final_config.yaml') as f:
    config = yaml.safe_load(f)

# Get runner
runner = get_backend_runner(config['backend'], config)

# Run inference
rgb = frame['rgb']
depth = frame['depth']
masks = runner.segment(rgb, depth)  # [H, W] uint8
```

**Output**:
- Mask predictions: [H, W] uint8 array
  - 0 = background (no object)
  - 1+ = object instance IDs

**Inference Parameters**:
- `points_per_side`: Determines prompt grid density
  - Higher = more prompts = higher accuracy, slower
  - PPS=6 means 6×6=36 prompts per image
  
- `mask_threshold`: Confidence filter for mask acceptance
  - Only keep masks with confidence ≥ threshold
  
- `nms_iou`: Non-maximum suppression
  - Suppresses masks overlapping by ≥ threshold
  - Higher threshold = less aggressive suppression

---

### 4. timing.py - Performance Measurement

**Purpose**: Track per-frame latency and compute FPS

**Key Class**: `FrameTimer`

**Usage**:
```python
timer = FrameTimer()

for frame_idx, frame_data in enumerate(dataset):
    with timer.record_frame(frame_idx, backend='nanosam'):
        masks = model.segment(frame_data['rgb'])
    
    # Get timing for this frame
    latency_ms = timer.frames[frame_idx].total_latency_ms

# Get aggregate statistics
stats = timer.get_stats()
print(f"Mean latency: {stats['mean_latency_ms']:.1f} ms")
print(f"FPS: {stats['fps']:.2f}")
print(f"Min/Max: {stats['min_latency_ms']:.1f} / {stats['max_latency_ms']:.1f} ms")
```

**Metrics Computed**:
- Mean latency (ms/frame)
- FPS (frames/second)
- Min/Max latency
- Per-frame timing details

**What's Included in Latency**:
- Image encoder forward pass
- Mask decoder forward pass
- Post-processing (NMS, filtering)
- Excludes: I/O, data loading

---

### 5. test_runner_template.py - Complete Example

**Purpose**: Template showing how to use all modules together

**Workflow**:
1. Load configuration YAML
2. Initialize dataset loader
3. Initialize evaluator
4. Initialize backend runner
5. Initialize timer
6. For each frame:
   - Load frame (RGB, depth, GT)
   - Run inference (with timing)
   - Evaluate (compute F1, precision, recall)
7. Aggregate statistics
8. Report results

**Example Output**:
```
Config: PPS=4, Masks=12, Threshold=0.70, NMS=0.30 ... F1=0.5017, FPS=3.15

Config                    F1         Precision  Recall     Latency        FPS
───────────────────────────────────────────────────────────────────────────
phase2_5_nms03            0.5017     0.6214     0.4417     317.4          3.15
```

---

## Integration with Experiment

### How These Modules Are Used

**Phase 1: Backend Comparison**
```python
# phase_1/PHASE_1_EXPLANATION.md describes results from:
# 1. NanoSAMRunner with PPS=6, masks=24, threshold=0.80
# 2. ViTBRunner with same config
# Using ground_truth.py evaluation
# Timing via timing.py
```

**Phase 2: Parameter Optimization**
```python
# Phase 2.1: Test PPS values
# Actual test runners (run_phase2_pps_test.py) use:
# - data_loader.py to load frames
# - runners.py to run inference with different PPS
# - timing.py to measure latency
# - ground_truth.py to compute F1

# Same pattern for other phases (2.2, 2.3, 2.5, 2.6)
```

### Actual Test Runners

The test runners referenced in documentation follow this pattern:

```python
# run_phase2_1_pps_test.py (example)
configs = [
    ("PPS=6", "config_pps6.yaml"),
    ("PPS=5", "config_pps5.yaml"),
    ("PPS=4", "config_pps4.yaml"),
    ("PPS=3", "config_pps3.yaml"),
]

for label, config_file in configs:
    # Load and run test
    result = run_experiment(config_file, dataset_path)
    # Compare results
```

---

## Reproducibility Guide

To reproduce experiments:

1. **Ensure Dataset Exists**:
   ```bash
   ls phase_1/datasets/phase1_frames_300/
   # Should have: rgb_*.npy, depth_*.npy, semantic_*.npy
   ```

2. **Load Configuration**:
   ```bash
   ls phase_2/2_1_points_per_side/configs/
   # Should have: phase2_pps*.yaml files
   ```

3. **Run Test**:
   ```python
   python test_runner_template.py \
       --config phase_2/2_1_points_per_side/configs/phase2_pps4.yaml \
       --dataset phase_1/datasets/phase1_frames_300 \
       --frames 300
   ```

4. **Verify Results**:
   - Expected F1 ≈ 0.5307 (±0.05)
   - Expected FPS ≈ 2.76 (±0.1)
   - Variance due to GPU/timing variation

---

## Dependencies

**Required**:
- numpy
- yaml (PyYAML)
- scipy (for connected components)
- torch (PyTorch)

**Optional**:
- tensorrt (for engine optimization)
- mobile-sam (for NanoSAM backend)
- segment-anything (for ViT-B backend)

**Installation**:
```bash
pip install numpy pyyaml scipy torch

# Optional
pip install tensorrt
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## Code Quality Notes

**Preprocessing**:
- ✓ Handles both 2D and 3D semantic masks
- ✓ Validates input shapes
- ✓ Proper depth filtering
- ✓ Connected component extraction (8-connectivity)

**Evaluation**:
- ✓ Instance-level IoU matching
- ✓ Greedy matching algorithm
- ✓ Proper TP/FP/FN counting
- ✓ Handles edge cases (no objects, no predictions)

**Timing**:
- ✓ Context manager for clean latency tracking
- ✓ Per-frame and aggregate statistics
- ✓ FPS computed from mean latency (not mean FPS)

**Runners**:
- ✓ Abstract base class for extensibility
- ✓ Configuration-driven parameters
- ✓ Device-agnostic (CPU/GPU)
- ✓ Clear error messages

---

## Extension Points

To add new features:

1. **New Backend**: Inherit from `BaseRunner` in `runners.py`
2. **New Metrics**: Add to `GroundTruthEvaluator.compute_metrics()`
3. **New Dataset**: Create new loader inheriting from dataset loader pattern
4. **Custom Timing**: Extend `FrameTimer` with additional metrics

---

**Code Status**: Complete ✅  
**Test Coverage**: Phase 1 & 2 fully documented  
**Ready for**: Production use, research, reproducibility
