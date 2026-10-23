# Experiment B selection log

Rule: highest PQ; among values within 0.005 PQ of the best, the highest FPS.

## B1 (grid): chosen by the user = 4x4 (PQ 0.320, 42.2 ms, 23.7 FPS) as a balance between PQ and latency. The rule alone would have picked 7x7 (highest PQ).

## B2 (mask_threshold) at grid 4x4, nms_iou 0.10 (0.8 reused from B1_g4)

| mask_threshold | PQ | SQ | RQ | median ms | p95 ms | FPS | masks/frame | overlap px frac |
|---|---|---|---|---|---|---|---|---|
| 0.7 | 0.321 | 0.819 | 0.392 | 42.9 | 45.1 | 23.3 | 7.02 | 0.014 |
| 0.8 | 0.320 | 0.818 | 0.391 | 42.2 | 43.7 | 23.7 | 7.00 | nan |
| 0.9 | 0.318 | 0.818 | 0.388 | 42.7 | 49.4 | 23.4 | 6.98 | 0.013 |

Best PQ 0.321; within 0.005: [0.7, 0.8, 0.9]. **Chosen: mask_threshold = 0.8**

## B3 (nms_iou) at grid 4x4, mask_threshold 0.8 (0.10 is a reference row, not part of the selection list)

| nms_iou | PQ | SQ | RQ | median ms | p95 ms | FPS | masks/frame | overlap px frac |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.298 | 0.821 | 0.363 | 42.8 | 53.8 | 23.4 | 6.04 | 0.000 |
| 0.1 | 0.320 | 0.818 | 0.391 | 43.1 | 51.4 | 23.2 | 7.00 | 0.013 |
| 0.15 | 0.320 | 0.818 | 0.392 | 42.5 | 49.9 | 23.6 | 7.08 | 0.020 |
| 0.3 | 0.319 | 0.817 | 0.390 | 42.4 | 43.9 | 23.6 | 7.26 | 0.039 |

Best PQ among {0, 0.15, 0.30} 0.320; within 0.005: [0.15, 0.3]. **Chosen: nms_iou = 0.3**

## Final B configuration: grid 4x4, mask_threshold 0.8, nms_iou 0.3, max_masks 8, min_mask_pixels 3500, input_scale 0.85

