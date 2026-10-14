# Source Code Reference

## Implementation Files

### Primary: tracking_crop_manager.py
**Location:** `src/rsg/nodes/support/phase1/tracking_crop_manager.py`

**Key Constants (lines 26-40):**
```python
PIXEL_NORMALIZATION_BASE = 100000
SHARPNESS_DIVISOR = 400.0
SHARPNESS_SKIP_THRESHOLD = 0.1
DEFAULT_SHARPNESS = 0.5
HYSTERESIS_MARGIN = 0.005
MARGIN_EDGE_PROXIMITY = 3

# Weight constants
WEIGHT_PIXEL = 2.0
WEIGHT_SHARPNESS = 2.0
WEIGHT_MARGIN = 1.0
```

**Key Methods:**

1. **_score_crop(crop_rgb, mask)** (lines 463-500)
   - Returns: (composite_score, pixel_score, sharpness_score, margin_score)
   - Calculates all three metrics and combines them

2. **_compute_margin_score(mask)** (lines 501-545)
   - Defines 3px edge zone
   - Counts object pixels in border
   - Returns normalized margin score

3. **extract_crop(...)** (lines 230-350)
   - Main entry point for crop processing
   - Calls _score_crop()
   - Handles update acceptance with hysteresis
   - Saves crops and logs diagnostics

4. **_log_crop_update_diagnostics(...)** (lines 547-572)
   - Logs individual crop updates with all metrics
   - Stores in internal diagnostics list

5. **save_crop_progression_diagnostics()** (lines 574-620)
   - Exports CSV file with complete metric history
   - Called on pipeline shutdown

### Integration: phase1.py
**Location:** `src/rsg/nodes/phase1.py`

**Crop Extraction Call (lines 1153-1161):**
```python
if external_track_id and bbox_2d:
    is_new_track = track_record.get("persistent_match_reason") == "new_track"
    self.tracking_crop_manager.extract_crop(
        rgb_image=rgb,
        mask=mask.mask,
        bbox_2d=bbox_2d,
        track_id=external_track_id,
        frame_id=frame.rsg_frame_id,
        sequence=int(frame.sequence),
        is_new_track=is_new_track,
    )
```

**Shutdown Logging (in destroy_node()):**
```python
if self.tracking_crop_manager:
    self.tracking_crop_manager.save_crop_progression_diagnostics()
```

---

## Formula References

### Pixel Score
```python
pixel_count = int(np.sum(mask > 0))
pixel_score = np.log(1 + pixel_count) / np.log(1 + PIXEL_NORMALIZATION_BASE)
pixel_score = min(pixel_score, 1.0)

# Normalization base: 100000
# Result range: [0, 1]
# Examples:
#   100 pixels   → 0.35
#   5000 pixels  → 0.74
#   50000 pixels → 0.94
```

### Sharpness Score
```python
if pixel_score < SHARPNESS_SKIP_THRESHOLD:
    sharpness_score = DEFAULT_SHARPNESS
else:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / SHARPNESS_DIVISOR, 1.0)

# Divisor: 400.0
# Skip threshold: 0.1
# Default when skipped: 0.5
# Result range: [0, 1]
```

### Margin Score
```python
# Define edge zone (3px from all edges)
edge_zone = np.zeros_like(mask_binary)
edge_zone[0:MARGIN_EDGE_PROXIMITY, :] = 1
edge_zone[-MARGIN_EDGE_PROXIMITY:, :] = 1
edge_zone[:, 0:MARGIN_EDGE_PROXIMITY] = 1
edge_zone[:, -MARGIN_EDGE_PROXIMITY:] = 1

# Count pixels
total_edge_pixels = np.sum(edge_zone)
object_in_edge = np.sum(mask_binary * edge_zone)

# Score
edge_fraction = object_in_edge / total_edge_pixels
margin_score = 1.0 - edge_fraction

# Result range: [0, 1]
# Interpretation:
#   1.0 = 0% object in edge zone (perfect)
#   0.95 = 5% object in edge zone (good)
#   0.80 = 20% object in edge zone (tight)
#   0.60 = 40% object in edge zone (cropped)
```

### Composite Score (Weighted Additive)
```python
total_weight = WEIGHT_PIXEL + WEIGHT_SHARPNESS + WEIGHT_MARGIN  # 5.0
composite_score = (
    WEIGHT_PIXEL * pixel_score +
    WEIGHT_SHARPNESS * sharpness_score +
    WEIGHT_MARGIN * margin_score
) / total_weight

# Weights: 2.0, 2.0, 1.0
# Result range: [0, 1]
# Interpretation:
#   0.95+ = Excellent crop
#   0.90-0.95 = Good crop
#   0.80-0.90 = Acceptable
#   0.70-0.80 = Poor
#   < 0.70 = Reject
```

### Update Acceptance (Hysteresis)
```python
HYSTERESIS_MARGIN = 0.005  # 0.5% threshold

if is_new_track:
    accept_as_best()
elif composite_score > old_score * (1.0 + HYSTERESIS_MARGIN):
    accept_as_best()
else:
    reject_update()

# Requires: new_score > old_score × 1.005
# Examples:
#   old=0.922, new=0.927 → 0.927 > 0.927 (reject, Δ=0.005)
#   old=0.922, new=0.928 → 0.928 > 0.927 (accept, Δ=0.006)
```

---

## How to Modify

### Change a Weight
Edit line in `tracking_crop_manager.py`:
```python
WEIGHT_PIXEL = 3.0  # Increase to 3 (was 2.0)
colcon build --packages-select rsg
```

### Change Hysteresis Threshold
```python
HYSTERESIS_MARGIN = 0.010  # 1.0% threshold (was 0.5%)
colcon build --packages-select rsg
```

### Change Edge Zone Size
```python
MARGIN_EDGE_PROXIMITY = 5  # 5px instead of 3px
colcon build --packages-select rsg
```

### Skip Sharpness on More Crops
```python
SHARPNESS_SKIP_THRESHOLD = 0.2  # Skip if pixel < 0.2 (was 0.1)
colcon build --packages-select rsg
```

---

## Debugging

### Print Scores During Runtime
Add to `_score_crop()` after line 495:
```python
print(f"[CROP_DEBUG] pixel={pixel_score:.4f} sharp={sharpness_score:.4f} " +
      f"margin={margin_score:.4f} composite={composite_score:.4f}")
```

### Check CSV Output
```bash
cat debug/best_crop_analysis/crops/session_*/crop_progression_diagnostics.csv
```

### Inspect Crop Images
```bash
ls -lh debug/best_crop_analysis/crops/session_20260830_003232_final/rsg_obj_*/
display debug/best_crop_analysis/crops/session_20260830_003232_final/rsg_obj_000001/best_update_1.jpg
```

---

## Performance Notes

- Pixel count: ~0.05 ms
- Margin calculation: ~0.15 ms  
- Sharpness (when computed): ~0.80 ms
- Sharpness (when skipped): ~0.0 ms
- Total per crop: ~1.0 ms (0.7 ms with skip optimization)

Sharpness is skipped ~30% of the time (small crops with low pixel scores), providing significant speedup.

---

## Testing

### Rebuild and Test
```bash
cd /home/student/rsg_ros2_ws
colcon build --packages-select rsg
ros2 launch rsg phase1_launch.py
```

### Verify Changes
Check the latest CSV file:
```bash
find debug/best_crop_analysis/crops -name "crop_progression_diagnostics.csv" \
  -type f -exec ls -lt {} + | head -1
```

Look for changes in:
- Number of best_update_N.jpg files (fewer if stricter)
- Margin score distribution (wider range if working)
- Improvement deltas (larger if stricter threshold)

