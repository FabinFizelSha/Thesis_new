# Method (common to both models)

- Dataset: 500 frames of the uHumans2 office scene (frames 0-299 = thesis `phase1_frames_300`, 300-499 = 200 extra frames from the same bag, interleaved). RGB 720x480 uint8; GT from the simulator class image (exact RGB colour = class, 8-connected region = instance, regions < 3500 px = void; fixed for all configurations).
- Prompts: one foreground point (label 1) per call. Same grid, margins, black-point skipping for both models (`src/proposals.py`).
- Mask choice: output token 0 (single-mask output) for both. SAM: `multimask_output=False`; NanoSAM: index 0 of the 4 decoder outputs. Score = the model's predicted IoU for that mask.
- Threshold: pixel in mask iff logit > `mask_threshold` (0.80 in the base config) for both; thresholding is done by the common code, not by the wrappers. SAM is called with `return_logits=True`.
- Masks returned by a wrapper have the size of the image given to `set_image`; with `input_scale < 1` the thresholded mask is resized back to 720x480 with nearest neighbour *before* the area filter, so `min_mask_pixels` and NMS operate on full-resolution pixels.
- NanoSAM: official `nanosam` package (`Predictor`, torch2trt `TRTModule`), TensorRT 10.16.2 engines rebuilt on this Thor with `trtexec` (encoder FP16, decoder default precision, decoder profile 1..10 points), image encoder input 1024. Its own pre-/post-processing is used unchanged.
- SAM ViT-B: `segment_anything` `vit_b`, checkpoint `sam_vit_b_01ec64.pth`, strict FP32 on CUDA (TF32 disabled, cuDNN deterministic), prompts via `SamPredictor.predict_torch`.
- NMS: sorted by (score, area) descending, accept unless IoU > `nms_iou` with an accepted mask (base `nms_iou` = 0.10; earlier 50-frame pilots with 0.20 and 0.0 are archived in `results/archive_nms020` and `results/archive_nms000`), stop at `max_masks`. Accepted masks are painted with `masks_to_label_image` (highest score wins).
- Metric: `src/pq_eval.py` verbatim (class-agnostic PQ, IoU > 0.5), pooled over all frames. Surfaces/objects split skipped (`surface_colours.json` left empty).
- Timing: per frame from the RGB array in memory to the label image (resize, encoder, all prompts, threshold, NMS, painting), GPU-synchronised; excludes disk I/O and saving. 10 warm-up frames, then 3 passes over all frames; pass 1 is saved and scored; determinism checked pass 1 vs pass 2 on the first 20 frames.
