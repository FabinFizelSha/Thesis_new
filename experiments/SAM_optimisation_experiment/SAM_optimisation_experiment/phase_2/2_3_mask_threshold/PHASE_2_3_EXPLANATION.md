# Phase 2.3: Mask Threshold Tuning

## Objective
Test whether adjusting confidence threshold for mask acceptance balances precision/recall to optimize F1.

## Test Configuration
**Variable**: mask_threshold = [0.60, 0.70, 0.80, 0.90]  
**Fixed**: PPS=4, max_masks=12, min_mask_pixels=3500, nms_iou=0.20

## Results

| Threshold | F1 Score | Precision | Recall | FPS | Latency (ms) | Change |
|-----------|----------|-----------|--------|-----|--------------|--------|
| 0.60 | 0.4861 | 0.5758 | 0.4193 | 2.77 | 360.7 | Lower recall |
| **0.70** | **0.4977** | **0.5888** | **0.4293** | **2.79** | **358.4** | **+0.04% F1** |
| 0.80 | 0.4975 | 0.5886 | 0.4290 | 2.78 | 358.9 | Baseline |
| 0.90 | 0.4881 | 0.5933 | 0.4156 | 2.79 | 358.3 | Lower recall |

## Selection: mask_threshold=0.70

**Rationale**:
- Marginal F1 improvement (+0.04% vs 0.80 baseline)
- Best balance between precision (0.5888) and recall (0.4293)
- FPS unchanged across all thresholds (~2.78)
- Maintains competitive accuracy with slight recall boost

## Key Insight

Threshold tuning is **fine-tuning** because:
- SAM generates masks with confidence scores
- Lower threshold accepts more masks → higher recall, lower precision
- 0.70 sweet spot for this dataset
- FPS impact minimal (threshold filtering low-cost)

## Impact

Minor F1 recovery (+0.04%) after PPS loss. Demonstrates that threshold tuning can't fully recover from aggressive parameter reduction, but helps.

## Next Step
Fix PPS=4, max_masks=12, threshold=0.70, proceed to Phase 2.5 (NMS optimization).

---

**Configs in `/configs/`**: phase2_3_thresh60.yaml, phase2_3_thresh70.yaml, phase2_3_thresh80.yaml, phase2_3_thresh90.yaml
