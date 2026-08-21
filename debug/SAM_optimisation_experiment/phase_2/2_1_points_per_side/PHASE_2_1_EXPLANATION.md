# Phase 2.1: Points Per Side (PPS) Reduction

## Objective
Test whether reducing SAM prompt grid density (fewer encoder forward passes) improves FPS with manageable accuracy trade-off.

## Test Configuration
**Variable**: points_per_side = [6, 5, 4, 3]  
**Fixed**: max_masks=24, mask_threshold=0.80, min_mask_pixels=3500, nms_iou=0.20

## Results

| PPS | Grid | Prompts | F1 Score | Precision | Recall | FPS | Latency (ms) | Change |
|-----|------|---------|----------|-----------|--------|-----|--------------|--------|
| 6 | 6×6 | 36 | 0.5817 | 0.6181 | 0.5758 | 1.53 | 653.6 | Baseline |
| 5 | 5×5 | 25 | 0.5643 | 0.6136 | 0.5367 | 1.85 | 541.3 | -2.9% F1 |
| **4** | **4×4** | **16** | **0.5307** | **0.5969** | **0.4881** | **2.76** | **362.9** | **-8.8% F1, +80% FPS** |
| 3 | 3×3 | 9 | 0.4693 | 0.5472 | 0.4148 | 3.46 | 289.5 | -19.3% F1 |

## Selection: PPS=4

**Rationale**:
- Achieves 80% FPS improvement (1.53 → 2.76)
- F1 loss (-8.8%) manageable for speed gain
- PPS=3 is too sparse (-19.3% F1)
- PPS=5 marginal improvement over baseline

## Key Insight

PPS is the **primary speed driver** because:
- Each prompt requires encoder forward pass
- PPS=6: 36 encoder evaluations per frame
- PPS=4: 16 encoder evaluations per frame
- ResNet18 encoder dominates latency (600+ ms → 360 ms)

## Next Step
Fix PPS=4, proceed to Phase 2.2 (max_masks optimization).

---

**Configs in `/configs/`**: phase2_pps3.yaml, phase2_pps4.yaml, phase2_pps5.yaml
