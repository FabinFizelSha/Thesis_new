# Phase 1 optimisation Part 5 timing summary

Complete frame traces: 387; drops: 2512; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

## Throughput (the primary question for this part)

Part 5 removes cross-thread CPU contention (via core affinity) from the concurrency model Part 4 introduced, so the number that matters here is still how many frames got through the 180 s window -- and whether sam_inference_ms/geometry_metadata_ms/frame_assignment_ms fall back toward their pre-Part-4 serial-mode values.

| Received | Processed | Dropped | Failed | Hydra published | Processing ratio |
|---:|---:|---:|---:|---:|---:|
| 2899 | 385 | 2512 | 0 | 385 | 13.28% |

`sam_output_queue_wait_ms` -- time a completed SAM stage waits for the tracking/publish thread to become free. Small relative to `sam_inference_ms` means the two stages are overlapping well; large means tracking/publish has become the new bottleneck.

Mean 0.891 ms, median 0.625 ms, p95 0.877 ms, max 85.374 ms (n=387).

## Per-stage breakdown (same framing as Parts 1-3, for continuity)

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 103.686 ms, p95 176.984 ms). This framing answers "which stage costs the most," which is not the question Part 5 is testing -- see the throughput section above.

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 387 | 726.780 | 722.829 | 938.774 | 1141.757 |
| classifier_delay_ms | aggregate | False | 387 | 678.426 | 676.294 | 894.380 | 1094.949 |
| sam_delay_ms | aggregate | False | 387 | 459.071 | 456.770 | 569.689 | 832.965 |
| sam_inference_ms | leaf | False | 387 | 438.545 | 437.908 | 549.698 | 814.606 |
| rap_delay_ms | aggregate | False | 387 | 184.166 | 175.237 | 280.808 | 584.910 |
| geometry_metadata_ms | leaf | True | 387 | 103.686 | 99.349 | 176.984 | 304.730 |
| frame_assignment_ms | leaf | True | 387 | 71.809 | 66.290 | 117.458 | 408.143 |
| assignment_candidate_search_ms | leaf | False | 387 | 57.369 | 53.233 | 98.912 | 376.827 |
| geometry_projection_ms | leaf | False | 387 | 45.133 | 38.317 | 99.170 | 227.120 |
| sent_to_classifier_delay_ms | aggregate | False | 387 | 32.431 | 27.932 | 80.407 | 153.334 |
| frame_queue_wait_ms | leaf | True | 387 | 32.274 | 27.787 | 80.195 | 153.193 |
| result_message_build_delay_ms | leaf | True | 387 | 27.150 | 27.347 | 47.573 | 68.079 |
| geometry_mask_extract_ms | leaf | False | 387 | 26.728 | 24.883 | 45.638 | 71.615 |
| assignment_3d_geometry_ms | leaf | False | 387 | 24.181 | 21.083 | 48.441 | 329.465 |
| geometry_stats_ms | leaf | False | 387 | 22.284 | 19.685 | 40.753 | 79.857 |
| sam_restore_ms | leaf | False | 387 | 20.292 | 18.999 | 29.351 | 102.665 |
| assignment_centroid_iou_ms | leaf | False | 387 | 15.957 | 12.821 | 31.187 | 338.684 |
| coordinator_delay_ms | aggregate | False | 387 | 13.172 | 13.148 | 18.876 | 24.937 |
| geometry_depth_gather_ms | leaf | False | 387 | 8.681 | 6.258 | 22.430 | 66.878 |
| hydra_build_delay_ms | aggregate | False | 387 | 8.073 | 7.890 | 12.940 | 18.393 |
| assignment_a2_redundancy_ms | leaf | False | 387 | 7.919 | 7.720 | 14.886 | 31.284 |
| hydra_depth_filter_ms | leaf | True | 387 | 7.419 | 7.144 | 12.193 | 17.792 |
| assignment_scoring_ms | leaf | False | 387 | 7.290 | 6.367 | 16.151 | 32.175 |
| label_map_delay_ms | leaf | True | 387 | 6.204 | 5.242 | 13.042 | 28.734 |
| track_association_ms | leaf | True | 387 | 4.633 | 4.072 | 8.222 | 22.530 |
| hydra_publish_delay_ms | leaf | True | 387 | 4.230 | 3.778 | 7.440 | 12.230 |
| assignment_a3_nested_ms | leaf | False | 387 | 4.198 | 3.518 | 8.406 | 17.128 |
| assignment_row_init_ms | leaf | False | 387 | 2.821 | 2.226 | 6.808 | 28.277 |
| assignment_hungarian_ms | leaf | False | 387 | 2.234 | 2.161 | 4.078 | 8.762 |
| pipeline_wait_ms | leaf | False | 387 | 1.855 | 1.838 | 3.414 | 5.943 |
| crop_update_ms | leaf | True | 387 | 1.779 | 1.120 | 5.643 | 14.878 |
| image_conversion_delay_ms | leaf | True | 387 | 1.615 | 1.350 | 3.113 | 14.689 |
| active_segments_publish_ms | leaf | True | 387 | 1.477 | 1.391 | 1.815 | 11.399 |
| sam_output_queue_wait_ms | leaf | True | 387 | 0.891 | 0.625 | 0.877 | 85.374 |
| unknown_publish_delay_ms | leaf | True | 387 | 0.863 | 0.739 | 1.389 | 5.208 |
| hydra_build_other_ms | leaf | True | 387 | 0.550 | 0.533 | 0.691 | 2.377 |
| semantic_dispatch_ms | leaf | False | 387 | 0.382 | 0.379 | 0.707 | 3.693 |
| run_rap_other_ms | leaf | True | 387 | 0.372 | 0.345 | 0.444 | 6.362 |
| sam_prepare_ms | leaf | False | 387 | 0.226 | 0.194 | 0.333 | 1.670 |
| callback_enqueue_delay_ms | leaf | True | 387 | 0.157 | 0.150 | 0.199 | 0.369 |
| metadata_delay_ms | leaf | True | 387 | 0.154 | 0.142 | 0.192 | 2.484 |
| hydra_metadata_build_ms | leaf | True | 387 | 0.104 | 0.100 | 0.131 | 0.331 |
| association_lock_wait_ms | leaf | False | 387 | 0.072 | 0.015 | 0.021 | 8.400 |
| classifier_other_ms | leaf | True | 387 | 0.067 | 0.065 | 0.083 | 0.255 |
| sam_other_ms | leaf | False | 387 | 0.008 | 0.007 | 0.009 | 0.086 |
| coordinator_other_ms | leaf | True | 387 | 0.006 | 0.005 | 0.007 | 0.294 |
| classifier_debug_record_delay_ms | leaf | False | 387 | 0.005 | 0.005 | 0.008 | 0.010 |
| quality_deferred_release_ms | leaf | False | 387 | 0.005 | 0.005 | 0.006 | 0.014 |
| assignment_lock_wait_ms | leaf | False | 387 | 0.004 | 0.004 | 0.006 | 0.045 |
