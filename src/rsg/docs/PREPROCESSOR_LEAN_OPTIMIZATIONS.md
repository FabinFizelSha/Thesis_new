# Lean Preprocessor Optimizations

This revision implements the selected preprocessing optimizations while
preserving the `RsgFrame` output interface.

## Default hot path

1. Count raw RGB and depth arrivals.
2. Synchronize only RGB and depth.
3. Read the latest cached CameraInfo.
4. Validate RGB/depth resolution.
5. Shallow-forward `rgb8` without conversion.
6. Shallow-forward `32FC1` depth without conversion, range masking, or
   invalid-depth analysis.
7. Associate odometry using a moving ordered-buffer cursor.
8. Compose camera pose and publish `RsgFrame`.

Depth range masking remains in Phase 1/Hydra output. The optional preprocessor
single-pass depth path is disabled by default.

## New configuration

```yaml
preprocessing:
  synchronization:
    sync_queue_size: 50

  validation:
    single_pass_depth_processing_enabled: false
    compute_invalid_depth_ratio: false
    invalid_depth_ratio_every_n_frames: 5
    check_invalid_depth_ratio: false
```

When the optional depth path is disabled, input depth must already match
`image.output_depth_encoding` (currently `32FC1`). This prevents silent depth
unit errors.

## Status counters

Every status JSON message now includes:

- `rgb_received_count`
- `depth_received_count`
- `camera_info_received_count`
- `synchronized_pair_count`
- `published_frame_count`
- `explicit_rejected_count`
- `rate_limited_count`

These counters distinguish subscription/synchronizer loss from explicit frame
rejection.
