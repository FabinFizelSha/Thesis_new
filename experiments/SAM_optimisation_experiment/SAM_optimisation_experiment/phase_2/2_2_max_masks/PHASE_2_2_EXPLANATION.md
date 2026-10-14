# Phase 2.2: Max Masks Reduction

## Objective
Test whether reducing maximum mask output limit improves FPS with zero F1 loss (dataset rarely saturates limit).

## Test Configuration
**Variable**: max_masks = [24, 20, 16, 12, 8]  
**Fixed**: PPS=4, mask_threshold=0.80, min_mask_pixels=3500, nms_iou=0.20

## Results

| Masks | F1 Score | Precision | Recall | FPS | Latency (ms) | Change |
|-------|----------|-----------|--------|-----|--------------|--------|
| 24 | 0.5307 | 0.5969 | 0.4881 | 2.76 | 362.9 | Baseline |
| 20 | 0.5307 | 0.5969 | 0.4881 | 2.77 | 361.2 | 0% F1 |
| 16 | 0.5307 | 0.5969 | 0.4881 | 2.78 | 360.0 | 0% F1 |
| **12** | **0.5307** | **0.5969** | **0.4881** | **2.78** | **359.6** | **0% F1, +1% FPS** |
| 8 | 0.5306 | 0.5968 | 0.4880 | 2.79 | 358.9 | 0% F1 |

## Selection: max_masks=12

**Rationale**:
- Zero F1 loss down to 8 masks
- Dataset typically has 7-12 objects per frame
- max_masks=12 provides safety margin above average
- Marginal FPS gain (+1%), but no accuracy penalty

## Key Insight

Max masks is a **zero-cost optimization** because:
- Suite 1 dataset rarely has >12 objects per frame
- Mask decoder doesn't saturate at 12 limit
- Reducing limit saves downstream processing
- No detection capability loss

## Next Step
Fix PPS=4, max_masks=12, proceed to Phase 2.3 (threshold tuning).

---

**Configs in `/configs/`**: phase2_2_masks8.yaml, phase2_2_masks12.yaml, phase2_2_masks16.yaml, phase2_2_masks20.yaml, phase2_2_masks24.yaml
