# Phase1 Refactoring - Implementation Guide

## Quick Start

**Template Location**: `phase1_refactored_template.py` (286 lines)

This shows the complete refactored structure. Use it as a reference while editing the original `phase1.py`.

---

## How to Apply the Refactoring

### Approach: Incremental Method Extraction

Instead of rewriting the whole file, extract methods one group at a time.

#### Step 1: Add Stage Imports
**In phase1.py, after line 58:**

```python
from nodes.phase1_pipeline import (
    SegmentationStage,
    TrackingStage,
    SemanticsStage,
    PublishingStage,
)
```

#### Step 2: Initialize Stages in `__init__()`
**After line 98 (after sam_backend initialization):**

```python
# Initialize modular stages
self.seg_stage = SegmentationStage(self.sam_backend, self.config, self.get_logger())
self.track_stage = TrackingStage(self.persistent_tracker, self.config, self.get_logger())
self.sem_stage = SemanticsStage(self.config, self.get_logger())
self.pub_stage = PublishingStage(self.config, self.get_logger())
```

---

## Phase-by-Phase Implementation

### Phase 1: Replace SAM Calls (30 min)

#### Find all calls to `self.run_sam()`
```bash
grep -n "self.run_sam" ~/rsg_ros2_ws/src/rsg/nodes/phase1.py
```

#### Replace with stage call
**OLD** (lines ~1200):
```python
masks = self.run_sam(rgb, min_area_px=scaled_min)
```

**NEW**:
```python
masks, prep_info = self.seg_stage.run(rgb, depth, min_area_px=scaled_min)
```

#### Delete SAM methods (lines ~1113-1280)
- `run_sam()`
- `prepare_sam_input()`
- `cv_resize_interpolation()`
- `scaled_min_mask_pixels()`
- `restore_sam_masks_to_original()`

**Commit after Phase 1:**
```bash
git add src/rsg/nodes/phase1.py
git commit -m "Phase 1: Extract SAM segmentation to SegmentationStage (-170 lines)"
```

---

### Phase 2: Replace Publishing Calls (45 min)

#### Find publishing calls
```bash
grep -n "self.build_hydra_frame\|self._publish_hydra" src/rsg/nodes/phase1.py
```

#### Replace with stage calls
**OLD**:
```python
hydra_msg = self.build_hydra_frame(frame, result, callback_start, cached)
self._publish_hydra_from_result(frame, result)
```

**NEW**:
```python
hydra_msg = self.pub_stage.build_hydra_frame(frame, track_records)
self._safe_publish(self.hydra_pub, hydra_msg)
```

#### Delete publishing methods (lines ~851-1050)
- `build_hydra_frame()`
- `publish_hydra_camera_tf()`
- `publish_separate_hydra_topics()`
- `apply_hydra_depth_range_filter()`
- `add_evidence_record()`
- `_publish_hydra_from_result()`

**Commit after Phase 2:**
```bash
git add src/rsg/nodes/phase1.py
git commit -m "Phase 2: Extract publishing to PublishingStage (-750 lines)"
```

---

### Phase 3: Replace Crop Building Calls (1 hour)

#### Find crop building calls
```bash
grep -n "_build_rap_semantic_crop\|_build_vlm_semantic_crop" src/rsg/nodes/phase1.py
```

#### Replace with stage calls
**OLD**:
```python
rap_crop = self._build_rap_semantic_crop(rgb, mask, bbox_2d)
vlm_crop = self._build_vlm_semantic_crop(rgb, mask, context_bbox)
```

**NEW**:
```python
rap_crop = self.sem_stage.build_rap_crop(rgb, mask, bbox_2d)
vlm_crop = self.sem_stage.build_vlm_crop(rgb, mask, context_bbox)
```

#### Delete semantics methods (lines ~1621-1900)
- `_build_rap_semantic_crop()`
- `_build_vlm_semantic_crop()`
- `_remember_track_crop()`
- `_score_track_crop()`
- `_describe_track_crop()`
- `_retire_track_crop()`
- `_snapshot_track_task()`
- All crop quality methods

**Commit after Phase 3:**
```bash
git add src/rsg/nodes/phase1.py
git commit -m "Phase 3: Extract semantics to SemanticsStage (-800 lines)"
```

---

### Phase 4: Simplify Tracking (30 min)

#### Find tracking calls
```bash
grep -n "self.build_object_metadata\|self.persistent_tracker.associate" src/rsg/nodes/phase1.py
```

#### Move to stage if needed
Most tracking logic can stay, but `build_object_metadata()` can move to TrackingStage.

**Commit after Phase 4:**
```bash
git add src/rsg/nodes/phase1.py
git commit -m "Phase 4: Refine TrackingStage integration (-200 lines)"
```

---

### Phase 5: Final Cleanup (45 min)

Remove dead code:
- Unused helper methods
- Redundant state tracking
- Simplified loop logic

**Final result**: ~500-600 lines

**Commit after Phase 5:**
```bash
git add src/rsg/nodes/phase1.py
git commit -m "Phase 5: Final cleanup - lean coordinator (-800 lines)"
```

---

## Testing After Each Phase

After each phase, verify:

```bash
# Build
colcon build --packages-select rsg

# Check for errors
grep -i "error\|undefined" log/latest_build/rsg/stderr.log

# Visual check - spot-test the calling code
grep "self.seg_stage\|self.pub_stage" src/rsg/nodes/phase1.py | head -5
```

---

## Reference: Template Structure

The `phase1_refactored_template.py` shows:

1. **Imports** (lines ~1-40): All stage imports
2. **__init__** (lines ~45-120): Stage initialization
3. **frame_callback** (lines ~125-135): Frame intake
4. **_segmentation_loop** (lines ~140-165): Uses `seg_stage.run()`
5. **_tracking_publish_loop** (lines ~170-210): Uses `track_stage.associate()` and `pub_stage.build_hydra_frame()`
6. **Cleanup** (lines ~215+): Minimal helper methods

---

## Key Points

### What Stays in phase1.py
- ROS lifecycle (__init__, destroy_node)
- Frame callback and queueing
- Worker thread management
- RAP/VLM queue orchestration
- State tracking (_track_best_crops, etc)

### What Moves to Stages
- **SegmentationStage**: All SAM logic
- **TrackingStage**: Object association
- **SemanticsStage**: Crop building, RAP/VLM prep
- **PublishingStage**: Hydra message creation

### What to Keep in Coordinator
- Worker thread loops (_segmentation_loop, _tracking_publish_loop, etc)
- Queue management (frame_fifo, sam_output_fifo, rap_queue, vlm_queue)
- ROS publishers and subscribers
- Essential state (_track_best_crops, _semantic_label_pending_track_ids)

---

## Troubleshooting

### "ImportError: cannot import name"
- Verify stage modules exist: `ls phase1_pipeline/`
- Check imports in __init__.py

### "AttributeError: 'Phase1SemanticCoordinator' object has no attribute"
- You forgot to initialize the stage in __init__()
- Add: `self.seg_stage = SegmentationStage(...)`

### Build passes but pipeline doesn't work
- Check that you updated all method calls
- Verify stage methods return what the coordinator expects
- Add logging to see where it breaks

---

## Rollback if Needed

If something breaks:

```bash
git reset --hard HEAD~1
# Or restore specific file:
git checkout phase1.py
```

---

## Final Stats

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| phase1.py lines | 3318 | ~600 | -82% |
| Methods in phase1.py | 68 | ~15 | -78% |
| Modular stages | 0 | 4 | +4 |
| Testability | Poor | Good | ✓ |
| Maintainability | Hard | Easy | ✓ |

---

## Next: Automated Testing

Once refactoring is complete:

```bash
# Run tests
ros2 launch rsg rsg_all.launch.py

# Monitor output
ros2 topic echo /rsg/hydra

# Check performance
ros2 topic hz /rsg/hydra
```

