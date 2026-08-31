# Phase1.py Full Refactoring Guide

## Goal
Reduce phase1.py from 3318 → ~500 lines by extracting business logic to modular stages.

## Refactoring Checklist

### Phase 1: SAM Segmentation Extraction ✓ (Mostly Done)
- [x] SegmentationStage created
- [x] run_sam() → SegmentationStage.run()
- [x] prepare_sam_input() → SegmentationStage._prepare_input()
- [ ] Update phase1.py: call `self.seg_stage.run(rgb, depth)` instead of `self.run_sam()`
- [ ] Remove SAM methods from phase1.py (lines ~1113-1280)

**Lines to delete from phase1.py**: 1113-1280 (~170 lines)

---

### Phase 2: Publishing Extraction
- [ ] Enhance PublishingStage with:
  - `publish_hydra_camera_tf()`
  - `publish_separate_hydra_topics()`
  - `add_evidence_record()`
  - Hydra TF publishing logic
  
- [ ] Move methods from phase1.py:
  - `build_hydra_frame()` (lines ~851-900)
  - `publish_hydra_camera_tf()` (lines ~901-920)
  - `publish_separate_hydra_topics()` (lines ~921-1000)
  - `apply_hydra_depth_range_filter()` (lines ~1001-1050)
  - `_publish_hydra_from_result()` (lines ~2100-2150)
  - `add_evidence_record()` (lines ~2151-2200)

**Lines to delete from phase1.py**: ~750 lines total

---

### Phase 3: Semantic Labeling Extraction
- [ ] Enhance SemanticsStage with:
  - `_build_rap_semantic_crop()` 
  - `_build_vlm_semantic_crop()`
  - `_remember_track_crop()`
  - `_score_track_crop()`
  - `_describe_track_crop()`
  - `_retire_track_crop()`
  
- [ ] Move methods from phase1.py:
  - `_build_rap_semantic_crop()` (lines ~1621-1640)
  - `_build_vlm_semantic_crop()` (lines ~1642-1669)
  - `_remember_track_crop()` (lines ~1671-1710)
  - `_score_track_crop()` (lines ~1512-1580)
  - `_describe_track_crop()` (lines ~1780-1800)
  - `_retire_track_crop()` (lines ~1801-1820)
  - `_snapshot_track_task()` (lines ~1821-1850)
  - `_normalise_label_key()` (lines ~1851-1870)
  - Crop quality methods (~300 lines)

**Lines to delete from phase1.py**: ~800 lines total

---

### Phase 4: Tracking Extraction
- [ ] Enhance TrackingStage with:
  - `build_object_metadata()` (whole method)
  - `_candidate_track_ids()` logic
  - Metadata building helpers
  
- [ ] Move methods from phase1.py:
  - `build_object_metadata()` (lines ~2350-2500)
  - All metadata builders (~200 lines)

**Lines to delete from phase1.py**: ~200 lines

---

### Phase 5: Coordinator Cleanup
Keep ONLY:
- `__init__()` - initialization
- `_log_startup_summary()` - startup
- `frame_callback()` - frame intake
- `_safe_publish()` - publishing helper
- `_segmentation_loop()` - segmentation threading
- `_tracking_publish_loop()` - tracking threading
- `_enqueue_sam_output()` - queue management
- `run_segmentation_stage()` - orchestration
- `run_tracking_publish_stage()` - orchestration
- `destroy_node()` - cleanup
- RAP/VLM queue management methods (~100 lines)
- UnknownObjectTracker integration (~50 lines)

**Result**: ~500-600 line coordinator focused on:
1. ROS lifecycle
2. Thread management
3. Queue orchestration
4. Minimal state tracking

---

## Extraction Pattern

For each method to extract:

### Before (in phase1.py)
```python
def _build_rap_semantic_crop(self, rgb, mask, bbox_2d, prepared_mask=None):
    # 20 lines of logic
    return crop
```

### After (in SemanticsStage)
```python
def build_rap_crop(self, rgb, mask, bbox_2d):
    # Same logic
    return crop
```

### In phase1.py
```python
# Delete the old method
# Update callers:
# OLD: crop = self._build_rap_semantic_crop(rgb, mask, bbox)
# NEW: crop = self.sem_stage.build_rap_crop(rgb, mask, bbox)
```

---

## Phase by Phase Line Count

| Phase | What | Lines Removed | New Total |
|-------|------|---|---|
| Start | Original | - | 3318 |
| 1 | SAM | -170 | 3148 |
| 2 | Publishing | -750 | 2398 |
| 3 | Semantics | -800 | 1598 |
| 4 | Tracking | -200 | 1398 |
| 5 | Cleanup | -800 | 598 |

**Final: 598 lines** (82% reduction)

---

## How to Apply

### Step 1: Update phase1.py imports
```python
from phase1_pipeline import (
    SegmentationStage,
    TrackingStage,
    SemanticsStage,
    PublishingStage,
)
```

### Step 2: Initialize stages in __init__()
```python
self.seg_stage = SegmentationStage(self.sam_backend, self.config, self.get_logger())
self.track_stage = TrackingStage(self.persistent_tracker, self.config, self.get_logger())
self.sem_stage = SemanticsStage(self.config, self.get_logger())
self.pub_stage = PublishingStage(self.config, self.get_logger())
```

### Step 3: Replace method calls
```python
# OLD
masks = self.run_sam(rgb)

# NEW
masks, prep_info = self.seg_stage.run(rgb, depth)
```

### Step 4: Move methods one phase at a time
- Commit after each phase
- Test build after each phase
- Makes debugging easier

---

## Testing Checklist

After each phase:
- [ ] `colcon build --packages-select rsg` passes
- [ ] No import errors
- [ ] Semantics still work (RAP labels assigned)
- [ ] Hydra messages still published
- [ ] Tracking still associates objects

---

## Notes

- Some methods reference `self.config` → pass as parameter to stage
- Some methods modify `self._track_*` → keep that state in coordinator
- Queue management stays in coordinator (essential for orchestration)
- RAP/VLM queues stay in coordinator (they're coordinator responsibilities)

---

## Estimated Effort

- **Phase 1** (SAM): 30 min
- **Phase 2** (Publishing): 45 min
- **Phase 3** (Semantics): 1 hour
- **Phase 4** (Tracking): 30 min
- **Phase 5** (Cleanup): 45 min
- **Total**: ~3-4 hours

Can be done incrementally across multiple sessions.
