# Phase 2.5: NMS IoU Optimization

## Objective
Test whether adjusting Non-Maximum Suppression threshold improves balance between duplicate removal and valid adjacent detections.

## Test Configuration
**Variable**: nms_iou = [0.10, 0.20, 0.30]  
**Fixed**: PPS=4, max_masks=12, mask_threshold=0.70, min_mask_pixels=3500

## Results

| NMS IoU | F1 Score | Precision | Recall | FPS | Latency (ms) | Change |
|---------|----------|-----------|--------|-----|--------------|--------|
| 0.10 | 0.4921 | 0.6360 | 0.4198 | 3.04 | 329.1 | Stricter (worse) |
| 0.20 | 0.4975 | 0.6255 | 0.4332 | 3.11 | 321.8 | Baseline |
| **0.30** | **0.5017** | **0.6214** | **0.4417** | **3.15** | **317.4** | **+0.8% F1, +1.4% FPS** |

## Selection: nms_iou=0.30 (Looser NMS)

**Rationale**:
- Best F1 score (0.5017)
- Improves recall (+2% from baseline)
- Faster FPS (+1.4%) with looser suppression
- Counterintuitive: loose is better than strict for this dataset

## Key Insight

**Counterintuitive Finding**: Looser NMS performs better because:
- Phase 1 baseline (0.20) was over-suppressing
- Valid adjacent objects were being removed
- Looser suppression (0.30) retains nearby masks
- Clutter dataset benefits from adjacency retention

This suggests Phase 1 baseline was sub-optimal and could have been improved with NMS tuning alone.

## Partial Recovery

NMS optimization recovers +0.8% F1 after Phase 2.3 tuning. Together with threshold adjustment (+0.04%), partial recovery offsets ~1% of PPS loss (8.8% total).

## Next Step
Fix all previous params, proceed to Phase 2.6 (min_mask_pixels validation).

---

**Configs in `/configs/`**: phase2_5_nms01.yaml, phase2_5_nms02.yaml, phase2_5_nms03.yaml
