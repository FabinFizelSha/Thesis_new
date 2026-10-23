# NanoSAM vs SAM ViT-B as point-prompted proposal generators: class-agnostic Panoptic Quality (Experiments A and B)

Self-contained summary for writing the thesis section. Experiment A (backend comparison) and Experiment B (NanoSAM parameter study). All numbers come from `comparison.csv`, `comparison_B.csv` and `summary_*.json` in this folder.

## Setup
- **Hardware/software:** NVIDIA Jetson AGX Thor (JetPack 7.2, L4T R39, CUDA 13.2, TensorRT 10.16.2, PyTorch 2.12.0+cu132), MAXN power mode, `jetson_clocks` enabled, no other GPU jobs (`env.json`).
- **Models:** NanoSAM (ResNet18 image encoder FP16 + MobileSAM mask decoder, TensorRT engines built on this device, official `nanosam` predictor, encoder input 1024). SAM ViT-B (`sam_vit_b_01ec64.pth`, strict FP32, TF32 off, `segment_anything` `SamPredictor`).
- **Data:** 500 frames (480x720) of the uHumans2 office scene: 300 frames from the thesis set `phase1_frames_300` plus 200 extra frames from the same bag, interleaved (bag is ~506 s long, so a 1.5 s spacing cannot give 500 frames). Ground truth from the simulator's semantic image: each exact RGB colour = one class, each 8-connected region = one object (20 class colours, 10.46 objects/frame on average, 10.4 % void pixels). Regions < 3500 px are void (ignored) for every configuration. Total GT objects over 500 frames: 5231.
- **Prompting (identical for both models, `method.md`):** a points_per_side x points_per_side grid of single foreground points (margin 16.7 % of the image size, black pixels skipped), output token 0 (single-mask output) with the model's predicted IoU as score, mask = logit > 0.80, drop masks < 3500 px, NMS sorted by (score, area) with `nms_iou = 0.10`, `max_masks = 49` (never binding), accepted masks painted into one label image (highest score wins on overlap). `input_scale = 1.0`.
- **Metric:** class-agnostic Panoptic Quality (Kirillov et al., CVPR 2019): match iff IoU > 0.5, PQ = SQ x RQ, pooled over all frames; void pixels removed from the union, unmatched predictions with > 50 % void pixels not counted as FP (`pq_eval.py`, sanity checks in `sanity.md`).
- **Timing:** per frame, from the RGB array in memory to the final label image (resize, encoder, all prompts, thresholding, NMS, painting), GPU-synchronised, excluding disk I/O and saving. 10 warm-up frames, then 3 passes over all 500 frames (1500 timed frames per configuration); median and 95th percentile reported. Pass 1 vs pass 2 labels identical on the first 20 frames for all six configurations.

## Results (500 frames)

| Grid | Model | PQ | SQ | RQ | Precision | Recall | Median latency (ms) | p95 (ms) | FPS | Encoder (ms) | Decoder, all prompts (ms) | Masks/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3x3 | NanoSAM | 0.298 | 0.830 | 0.359 | 0.536 | 0.270 | 29.2 | 31.1 | 34.2 | 11.4 | 15.0 | 5.6 |
| 3x3 | SAM ViT-B | 0.214 | 0.848 | 0.252 | 0.406 | 0.183 | 610 | 615 | 1.64 | 368.5 | 238.6 | 4.9 |
| 5x5 | NanoSAM | 0.352 | 0.825 | 0.427 | 0.494 | 0.376 | 57.7 | 59.4 | 17.3 | 11.3 | 41.5 | 8.8 |
| 5x5 | SAM ViT-B | 0.283 | 0.834 | 0.340 | 0.394 | 0.299 | 1037 | 1049 | 0.97 | 369.0 | 662.3 | 8.3 |
| 7x7 | NanoSAM | 0.375 | 0.821 | 0.456 | 0.478 | 0.437 | 101.2 | 103.2 | 9.9 | 11.3 | 80.9 | 10.7 |
| 7x7 | SAM ViT-B | 0.308 | 0.830 | 0.372 | 0.384 | 0.360 | 1677 | 1685 | 0.60 | 367.4 | 1301.0 | 10.4 |

TP/FP/FN per configuration are in `comparison.csv`. Plot: `pq_vs_grid_and_latency.png`. LaTeX-ready block: `latex_tables_A.txt`.

## Findings
1. NanoSAM has higher PQ than SAM ViT-B at every grid size (+0.084, +0.069, +0.067 at 3x3, 5x5, 7x7).
2. The gap comes from recognition quality (RQ), not segmentation quality: SQ is nearly equal and slightly higher for ViT-B (0.830-0.848 vs 0.821-0.830). When ViT-B finds an object its mask is as good, but it finds fewer objects at the shared settings.
3. NanoSAM is about 21x (3x3), 18x (5x5) and 17x (7x7) faster per frame. At 7x7 ViT-B spends ~1.3 s in the per-prompt decoder loop and ~0.37 s in the encoder; NanoSAM spends ~11 ms in the encoder.
4. PQ increases with grid density for both models with diminishing returns (NanoSAM 0.298 -> 0.352 -> 0.375). Even NanoSAM 7x7 recovers only 2286 of 5231 ground-truth objects (recall 0.44).
5. Number of kept masks per frame is limited mainly by repeated hits on the same large objects (many prompts yield near-identical masks), the 3500 px minimum size and NMS, not by `max_masks`. Ground truth has ~10.5 objects/frame.

## Experiment B: NanoSAM parameter study (grid, mask threshold, NMS IoU)
Purpose: find a faster NanoSAM configuration with acceptable PQ. Plan: `EXPERIMENT_B_PROPOSAL.md`; every decision: `selection_log.md`; all rows: `comparison_B.csv`; plot: `B1_plot.png`.

**Fixed for all B runs** (taken from the live pipeline config `src/rsg/config/rsg_pipeline.yaml`): NanoSAM, `input_scale` 0.85 (masks resized back to full resolution), `max_masks` 8, `min_mask_pixels` 3500 (full-resolution px; GT void limit also 3500), `point_margin_fraction` 0.167, black-point skipping on (`min_prompt_luma` 4). B1 and B2 use `mask_threshold` 0.80 and `nms_iou` 0.10 (the Experiment A values). Same 500 frames, scoring and timing protocol as Experiment A. The machine had no other GPU/CPU workload during B (Hydra, VLM servers and the pipeline node were stopped; an earlier B1 attempt under load was discarded and re-run).

**Selection rule:** highest PQ; among values within 0.005 PQ of the best, highest FPS. For B1 the grid was chosen by the author.

### B1: grid size (`points_per_side`; threshold 0.80, NMS 0.10)
| grid | PQ | SQ | RQ | Precision | Recall | Median ms | p95 ms | FPS | Masks/frame |
|---|---|---|---|---|---|---|---|---|---|
| 7x7 | 0.359 | 0.829 | 0.434 | 0.522 | 0.371 | 103.2 | 107.7 | 9.7 | 7.92 |
| 6x6 | 0.347 | 0.824 | 0.422 | 0.510 | 0.359 | 78.7 | 91.2 | 12.7 | 7.86 |
| 5x5 | 0.344 | 0.827 | 0.416 | 0.518 | 0.348 | 60.3 | 69.0 | 16.6 | 7.63 |
| 4x4 | 0.320 | 0.818 | 0.391 | 0.508 | 0.317 | 42.2 | 43.7 | 23.7 | 7.00 |
| 3x3 | 0.294 | 0.825 | 0.357 | 0.530 | 0.269 | 29.5 | 31.8 | 33.9 | 5.63 |

Reading: PQ falls with the grid almost entirely through recall (SQ stays about 0.82-0.83, precision about 0.51-0.53). The drop is gradual from 7x7 to 5x5 (0.359 to 0.344) and steeper below (4x4 0.320, 3x3 0.294). Latency is roughly linear in the number of prompts (decoder-dominated; encoder constant at about 10.7 ms). By the rule alone 7x7 (highest PQ) would be selected; the author chose **4x4** as a balance between PQ and latency (42 ms, 23.7 FPS, p95 only 44 ms, with PQ 0.039 below 7x7 and 0.024 below 5x5; 5x5 was the knee of the curve).

### B2: mask threshold (grid 4x4, NMS 0.10; the 0.8 row is the B1 4x4 run)
| threshold | PQ | SQ | RQ | Precision | Recall | Median ms | p95 ms | FPS | Masks/frame |
|---|---|---|---|---|---|---|---|---|---|
| 0.7 | 0.321 | 0.819 | 0.392 | 0.509 | 0.319 | 42.9 | 45.1 | 23.3 | 7.02 |
| 0.8 | 0.320 | 0.818 | 0.391 | 0.508 | 0.317 | 42.2 | 43.7 | 23.7 | 7.00 |
| 0.9 | 0.318 | 0.818 | 0.388 | 0.507 | 0.315 | 42.7 | 49.4 | 23.4 | 6.98 |

Reading: PQ varies by only 0.003 across 0.7-0.9, so the threshold is insensitive in this range (within the noise of one recording). All values are within 0.005 PQ, so the rule's FPS tie-break picked 0.8, but the FPS differences (42.2-42.9 ms) are timing noise. 0.8 equals the pipeline's current value.

### B3: NMS IoU (grid 4x4, threshold 0.8)
| NMS IoU | PQ | SQ | RQ | Precision | Recall | Median ms | p95 ms | FPS | Masks/frame | Overlapping px (before painting) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.298 | 0.821 | 0.363 | 0.509 | 0.282 | 42.8 | 53.8 | 23.4 | 6.04 | 0.0 % |
| 0.10 | 0.320 | 0.818 | 0.391 | 0.508 | 0.317 | 43.1 | 51.4 | 23.2 | 7.00 | 1.3 % |
| 0.15 | 0.320 | 0.818 | 0.392 | 0.507 | 0.319 | 42.5 | 49.9 | 23.6 | 7.08 | 2.0 % |
| 0.30 | 0.319 | 0.817 | 0.390 | 0.499 | 0.320 | 42.4 | 43.9 | 23.6 | 7.26 | 3.9 % |

Reading: only strict zero overlap hurts (PQ 0.298, one mask per frame fewer, -0.022 PQ versus 0.10); 0.10, 0.15 and 0.30 are within 0.001 PQ of each other, while the share of overlapping pixels among kept masks grows with the threshold (1.3 %, 2.0 %, 3.9 %). NMS has no measurable latency cost. The rule selected 0.30 from {0, 0.15, 0.30} only because of a 0.1 ms FPS tie-break; since the aim of testing low values was to reduce overlap between the generated crops, 0.10-0.15 gives the same PQ with about half the overlap. 0.10 was added as a reference row (recomputed to obtain the overlap metric); it was not in the selection list.

### Resulting B configuration (as selected by the rule, with the author's grid)
4x4 grid, mask_threshold 0.8, nms_iou 0.30, max_masks 8, min_mask_pixels 3500, input_scale 0.85: PQ 0.319, 23.6 FPS. Not run: the minimum-mask-size sweep (dropped by the author; fixed at the pipeline value 3500) and a joint/factorial search.

B limitations: one-factor-at-a-time sequential selection (thresholds chosen at NMS 0.10, then NMS at that threshold); the same 500 frames are used to select and to report (PQ optimistic); no confidence intervals; B differs from Experiment A in `input_scale` (0.85 vs 1.0) and `max_masks` (8 vs 49), so B rows are not directly comparable with the A rows (7x7 B1 PQ 0.359 vs A_nano_7 0.375).

## Caveats / limitations to state in the thesis
- Class-agnostic PQ with pooled TP/FP/FN, not the per-category-averaged COCO PQ; values are not comparable with published PQ numbers.
- Ground-truth objects are connected colour regions of the simulator's semantic image: touching same-class objects merge into one object; large surfaces (walls, floor) are split into one object per connected region; regions < 3500 px are void (small objects are not evaluated).
- Predictions are made non-overlapping by painting (highest score wins), as PQ requires.
- Thresholds (mask logit 0.80, `nms_iou` 0.10) are shared by both models and were not tuned per model; 0.80 was chosen for NanoSAM and may disadvantage ViT-B.
- 200 of the 500 frames come from the same recording as the 300 thesis frames (interleaved, 0.75 s offset), so frames are not independent; no confidence intervals were computed (the optional paired bootstrap was not run).
- The surface-vs-object breakdown was skipped; only the all-objects score exists.
- Experiments A and B were run. The original B plan was simplified (grid, mask threshold and NMS only; no `max_masks` or `min_mask_pixels` sweep); the deployed-configuration run (Experiment C) was not run.
- `nms_iou` was changed before Experiment A (0.20 -> 0.0 -> 0.10 as base value). Earlier 50-frame pilots exist only in the repo (`results/archive_nms020`, `results/archive_nms000`) and are not part of these results; a 50-frame pilot at 0.0 gave NanoSAM 7x7 PQ 0.417 and ViT-B 7x7 0.384, at 0.20 gave 0.439 and 0.396.
- Latency is the whole per-frame pipeline on this Jetson, not the model alone; ViT-B decoder time includes SAM's own mask upscaling per prompt.

## Files in this folder
`REPORT.md` (this), `comparison.csv`, `latex_tables_A.txt` (Experiment A), `comparison_B.csv`, `latex_tables_B.txt`, `selection_log.md`, `B1_plot.png`, `EXPERIMENT_B_PROPOSAL.md` (Experiment B), `summary_B*_*.json`, `config_B*_*.yaml`, `proposals.py`/`runner.py`/`models.py` (benchmark code), `summary_A_*.json`, `config_A_*.yaml`, `method.md`, `sanity.md`, `env.json`, `prep_summary.json`, `class_colours.csv`, `pq_eval.py` (metric code), `pq_vs_grid_and_latency.png`.
