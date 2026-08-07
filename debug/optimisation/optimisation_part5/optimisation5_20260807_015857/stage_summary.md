# Phase 1 optimisation Part 5 timing summary

Complete frame traces: 404; drops: 2485; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

## Throughput (the primary question for this part)

Part 5 removes cross-thread CPU contention (via core affinity) from the concurrency model Part 4 introduced, so the number that matters here is still how many frames got through the 180 s window -- and whether sam_inference_ms/geometry_metadata_ms/frame_assignment_ms fall back toward their pre-Part-4 serial-mode values.

| Received | Processed | Dropped | Failed | Hydra published | Processing ratio |
|---:|---:|---:|---:|---:|---:|
| 2889 | 402 | 2485 | 0 | 402 | 13.91% |

`sam_output_queue_wait_ms` -- time a completed SAM stage waits for the tracking/publish thread to become free. Small relative to `sam_inference_ms` means the two stages are overlapping well; large means tracking/publish has become the new bottleneck.

Mean 0.641 ms, median 0.592 ms, p95 0.822 ms, max 2.739 ms (n=404).

## Per-stage breakdown (same framing as Parts 1-3, for continuity)

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 101.322 ms, p95 184.269 ms). This framing answers "which stage costs the most," which is not the question Part 5 is testing -- see the throughput section above.

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 404 | 701.177 | 702.976 | 913.272 | 1292.479 |
| classifier_delay_ms | aggregate | False | 404 | 653.974 | 647.503 | 858.338 | 1255.933 |
| sam_delay_ms | aggregate | False | 404 | 438.669 | 435.993 | 568.362 | 888.493 |
| sam_inference_ms | leaf | False | 404 | 422.487 | 420.193 | 552.671 | 869.450 |
| rap_delay_ms | aggregate | False | 404 | 180.372 | 174.650 | 269.009 | 662.519 |
| geometry_metadata_ms | leaf | True | 404 | 101.322 | 98.570 | 184.269 | 235.974 |
| frame_assignment_ms | leaf | True | 404 | 70.881 | 66.965 | 118.244 | 463.378 |
| assignment_candidate_search_ms | leaf | False | 404 | 56.864 | 54.434 | 101.736 | 435.351 |
| geometry_projection_ms | leaf | False | 404 | 42.784 | 38.963 | 93.626 | 141.328 |
| sent_to_classifier_delay_ms | aggregate | False | 404 | 31.871 | 25.424 | 82.089 | 129.579 |
| frame_queue_wait_ms | leaf | True | 404 | 31.726 | 25.286 | 81.943 | 129.435 |
| result_message_build_delay_ms | leaf | True | 404 | 27.293 | 27.541 | 49.008 | 69.278 |
| geometry_mask_extract_ms | leaf | False | 404 | 26.899 | 24.508 | 44.900 | 89.654 |
| assignment_3d_geometry_ms | leaf | False | 404 | 23.745 | 22.077 | 47.039 | 75.479 |
| geometry_stats_ms | leaf | False | 404 | 22.296 | 18.817 | 44.306 | 63.103 |
| sam_restore_ms | leaf | False | 404 | 15.977 | 15.206 | 20.893 | 38.972 |
| assignment_centroid_iou_ms | leaf | False | 404 | 15.671 | 12.456 | 30.619 | 358.162 |
| coordinator_delay_ms | aggregate | False | 404 | 12.858 | 12.841 | 17.936 | 26.075 |
| geometry_depth_gather_ms | leaf | False | 404 | 8.511 | 5.957 | 21.853 | 41.963 |
| hydra_build_delay_ms | aggregate | False | 404 | 7.759 | 7.208 | 11.842 | 20.454 |
| assignment_scoring_ms | leaf | False | 404 | 7.629 | 6.554 | 18.559 | 48.643 |
| assignment_a2_redundancy_ms | leaf | False | 404 | 7.365 | 7.041 | 12.829 | 25.454 |
| hydra_depth_filter_ms | leaf | True | 404 | 7.122 | 6.562 | 11.167 | 19.692 |
| label_map_delay_ms | leaf | True | 404 | 6.047 | 5.154 | 12.619 | 21.916 |
| track_association_ms | leaf | True | 404 | 4.372 | 3.850 | 8.145 | 18.317 |
| assignment_a3_nested_ms | leaf | False | 404 | 4.327 | 3.425 | 8.641 | 33.385 |
| hydra_publish_delay_ms | leaf | True | 404 | 4.246 | 3.833 | 7.082 | 14.933 |
| assignment_row_init_ms | leaf | False | 404 | 2.830 | 2.113 | 7.194 | 23.563 |
| assignment_hungarian_ms | leaf | False | 404 | 2.221 | 2.177 | 4.426 | 13.418 |
| pipeline_wait_ms | leaf | False | 404 | 1.829 | 1.876 | 3.408 | 3.999 |
| crop_update_ms | leaf | True | 404 | 1.610 | 1.101 | 4.817 | 12.063 |
| active_segments_publish_ms | leaf | True | 404 | 1.422 | 1.383 | 1.764 | 4.781 |
| image_conversion_delay_ms | leaf | True | 404 | 1.376 | 1.244 | 2.444 | 5.331 |
| unknown_publish_delay_ms | leaf | True | 404 | 0.847 | 0.715 | 1.341 | 5.109 |
| sam_output_queue_wait_ms | leaf | True | 404 | 0.641 | 0.592 | 0.822 | 2.739 |
| hydra_build_other_ms | leaf | True | 404 | 0.534 | 0.532 | 0.633 | 0.934 |
| semantic_dispatch_ms | leaf | False | 404 | 0.382 | 0.376 | 0.754 | 3.013 |
| run_rap_other_ms | leaf | True | 404 | 0.356 | 0.338 | 0.449 | 2.704 |
| sam_prepare_ms | leaf | False | 404 | 0.197 | 0.180 | 0.248 | 2.030 |
| metadata_delay_ms | leaf | True | 404 | 0.156 | 0.140 | 0.183 | 4.375 |
| callback_enqueue_delay_ms | leaf | True | 404 | 0.144 | 0.139 | 0.170 | 0.543 |
| hydra_metadata_build_ms | leaf | True | 404 | 0.103 | 0.099 | 0.131 | 0.296 |
| classifier_other_ms | leaf | True | 404 | 0.063 | 0.063 | 0.075 | 0.109 |
| association_lock_wait_ms | leaf | False | 404 | 0.062 | 0.015 | 0.020 | 10.850 |
| sam_other_ms | leaf | False | 404 | 0.007 | 0.006 | 0.008 | 0.010 |
| coordinator_other_ms | leaf | True | 404 | 0.006 | 0.005 | 0.007 | 0.214 |
| assignment_lock_wait_ms | leaf | False | 404 | 0.005 | 0.004 | 0.006 | 0.045 |
| classifier_debug_record_delay_ms | leaf | False | 404 | 0.004 | 0.004 | 0.007 | 0.011 |
| quality_deferred_release_ms | leaf | False | 404 | 0.004 | 0.004 | 0.006 | 0.008 |
