# Phase1 Pipeline Architecture

## Overview

The Phase1 pipeline processes RGB-D frames and outputs semantic object tracking for the Hydra spatial map. The pipeline is now organized into **modular stages** that can be developed, tested, and reused independently.

## Modular Pipeline Stages

### 1. **SegmentationStage** (`phase1_pipeline/segmentation.py`)
**Responsibility**: SAM segmentation of input images

**Key Methods**:
- `run(rgb, depth)` → List[SamMask]
- Handles image resizing, normalization, and coordinate restoration
- Configurable interpolation and minimum mask area

**Used by**: Frame processing pipeline
**Config Prefix**: `sam_*`

---

### 2. **TrackingStage** (`phase1_pipeline/tracking.py`)
**Responsibility**: Persistent object association across frames

**Key Methods**:
- `associate(masks, frame, timestamp)` → List[Dict]
- Associates new masks with existing tracks using Hungarian algorithm
- Handles track continuity and lifecycle

**Used by**: Frame processing pipeline
**Dependencies**: PersistentObjectTracker
**Config Prefix**: `persistent_tracking_*`

---

### 3. **SemanticsStage** (`phase1_pipeline/semantics.py`)
**Responsibility**: Semantic labeling (RAP/VLM classification)

**Key Methods**:
- `enqueue_rap_task(track_id, crop)` → task_id
- `enqueue_vlm_task(track_id, crop)` → task_id
- `build_rap_crop(rgb, mask, bbox)` → crop
- `build_vlm_crop(rgb, mask, bbox)` → crop
- `evaluate_crop_quality(crop, mask, bbox)` → metrics

**Used by**: Semantic classification workers
**Config Prefix**: `rap_*`, `vlm_*`, `semantic_crop_*`

---

### 4. **PublishingStage** (`phase1_pipeline/publishing.py`)
**Responsibility**: Hydra message construction and publishing

**Key Methods**:
- `build_hydra_frame(frame, track_records)` → RsgHydraFrame
- `apply_depth_filter(hydra_msg)` → None
- Converts tracking results to ROS messages

**Used by**: Publishing pipeline
**Config Prefix**: `hydra_*`

---

## Configuration Management

### Phase1Config (`phase1_pipeline/config.py`)
Unified configuration wrapper that:
1. Delegates to original phase1_config for all parameters
2. Adds **optional diagnostic configuration**
3. Allows runtime enabling/disabling of diagnostics

**Diagnostic Options**:
```python
config = Phase1Config(base_config)
config.diagnostics.timing_enabled = False  # Disable timing
config.diagnostics.crop_extraction_enabled = True  # Enable crops
config.diagnostics.disable_all_diagnostics()  # Disable all
```

---

## Pipeline Flow

```
Input Frame (RGB-D)
    ↓
[SegmentationStage.run()]
    ↓
List of Masks
    ↓
[TrackingStage.associate()]
    ↓
Track Records + Associations
    ↓
[SemanticsStage.build_rap_crop() / build_vlm_crop()]
    ↓
RAP & VLM tasks enqueued
    ↓
[PublishingStage.build_hydra_frame()]
    ↓
RsgHydraFrame
    ↓
ROS Publishers
```

---

## Separation of Concerns

| Component | Responsibility | Dependencies |
|-----------|---|---|
| **SegmentationStage** | Mask generation | SAM backend |
| **TrackingStage** | Track association | PersistentObjectTracker |
| **SemanticsStage** | Crop building, task queueing | None (stateless) |
| **PublishingStage** | Message creation | ROS message types |
| **Diagnostics** | Logging/recording | Optional modules |

---

## Extending the Pipeline

### Adding a New Stage

1. Create `phase1_pipeline/my_stage.py`:
```python
class MyStage:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def process(self, data):
        # Do work
        return result
```

2. Export in `__init__.py`
3. Use in phase1.py coordinator

### Adding Diagnostics

Diagnostic modules (TrackingQualityRecorder, TrackingCropManager, etc.) are kept optional through `config.diagnostics`:

```python
if self.config.diagnostics.crop_extraction_enabled:
    self.crop_manager.extract_crop(...)
```

---

## Current Status

✅ **Modularized**:
- SegmentationStage
- TrackingStage
- SemanticsStage
- PublishingStage
- Configuration wrapper

⏳ **Not Yet Refactored** (still in phase1.py):
- Frame callback & threading
- RAP/VLM result handling
- Hydra metadata enrichment
- Unknown object tracking

These can be extracted in future phases as needed.

---

## Design Principles

1. **Single Responsibility**: Each stage handles one aspect of the pipeline
2. **Composability**: Stages can be developed/tested independently
3. **Configurability**: All behavior via config, no hardcoded constants
4. **Optional Diagnostics**: Logging doesn't pollute core logic
5. **Backward Compatible**: Original phase1.py still works as-is
