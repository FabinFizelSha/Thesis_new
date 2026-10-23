# Experiment B (revised): NanoSAM grid size, mask threshold and NMS IoU, at the deployed pipeline settings

## Question
How small can the NanoSAM prompt grid be, and which mask threshold and NMS IoU are best, before class-agnostic PQ drops, when every other setting is the one the live pipeline uses?
Experiment A showed NanoSAM beats SAM ViT-B at every grid size; B tunes NanoSAM only.

## Fixed for every run: active values of the live pipeline (`src/rsg/config/rsg_pipeline.yaml`, `phase1.sam` / `phase1.sam.nanosam`)
| Parameter | Value | Pipeline key |
|---|---|---|
| model | NanoSAM (TensorRT, FP16 encoder) | `sam.backend: nanosam` |
| `input_scale` | **0.85** (area interpolation; masks resized back to full resolution, nearest) | `sam.input_scale_ratio` |
| `max_masks` | **8** | `sam.max_masks` |
| `min_mask_pixels` | **3500** (not swept; full-resolution pixels; the backend scales it by input_scale^2 internally, equivalent to the benchmark) | `sam.min_mask_pixels` |
| `nms_iou` | **0.10** in B1/B2 (same as Experiment A; swept in B3; pipeline value is 0.30) | `sam.nanosam.nms_iou` |
| `point_margin_fraction` | 0.167 | `sam.nanosam.point_margin_fraction` |
| `skip_black_points`, `min_prompt_luma` | true, 4 | `sam.nanosam.*` |
| data / scoring / timing | 500 frames, `pq_eval.py` (GT void limit 3500), 10 warm-up + 3 passes | as Experiment A |

Swept: `points_per_side` (B1), `mask_threshold` (B2), and `nms_iou` (B3). Pipeline defaults for these: 3, 0.80 and 0.30. B1 and B2 use the Experiment A values `mask_threshold` 0.80 and `nms_iou` 0.10.
Note: B still differs from Experiment A in `input_scale` (0.85 vs 1.0) and `max_masks` (8 vs 49), so B results are not directly comparable with A; a re-run of A-style references is not planned.

## Stage B1: grid size
`points_per_side` in {7, 6, 5, 4, 3}, `mask_threshold` = 0.80, `nms_iou` = 0.10 (both as in Experiment A). All five are run (max_masks 8 binds at the large grids).
Output: PQ, SQ, RQ, precision, recall, latency (median, p95), FPS, masks/frame per grid + plot. **Checkpoint:** you choose the grid.

## Stage B2: mask threshold
At the chosen grid, `mask_threshold` in {0.7, 0.8, 0.9} at `nms_iou` 0.10 (0.8 reused from B1; 2 new runs).

## Stage B3: NMS IoU
At the chosen grid and threshold, `nms_iou` in {0, 0.15, 0.30}: 3 new runs (the 0.10 used in B1/B2 is not in this list; its B2 result serves as the reference row, so 0.10 is effectively also compared).
0 = any pixel overlap between kept masks rejects the candidate (disjoint masks); 0.30 = pipeline value. Besides PQ, also report masks/frame and the fraction of overlapping pixels among kept masks (pre-painting), since reducing overlap is the reason for testing low values.

## Selection rule (all stages)
Highest PQ; among values within 0.005 PQ of the best, highest FPS. B1 is decided by you at the checkpoint (rule shown as a suggestion). Decisions logged in `results/selection_log.md`.

## Cost
10 NanoSAM runs on 500 frames (5 + 2 + 3), about 2-5 min each: roughly 30-40 min total.

## Outputs
`results/B1_g{7,6,5,4,3}/`, `results/B2_thr{0.7,0.9}_g<chosen>/`, `results/B3_nms{0,0.15,0.30}_g<chosen>_thr<chosen>/`, `results/comparison_B.csv`, `results/latex_tables_B.txt`, B1 plot, `selection_log.md`.

## Limits
- One-factor-at-a-time: the best threshold may depend on the grid. Optional cheap fix: full factorial 5 grids x 3 thresholds x 3 NMS values (45 runs, ~2.5 h).
- Same 500 frames used to select and to report (no held-out set); the 200 extra frames come from the same recording.
- `input_scale`, `max_masks` and `min_mask_pixels` are not tuned here (`min_mask_pixels` fixed at the pipeline value 3500).
- Sequential order matters: the threshold is chosen at nms 0.10 and NMS is then chosen at that threshold, so a joint optimum can be missed.
- The yaml comment for `input_scale_ratio` says "65% resolution" but the active value is 0.85.
