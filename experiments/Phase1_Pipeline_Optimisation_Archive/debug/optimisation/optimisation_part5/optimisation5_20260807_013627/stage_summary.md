# Phase 1 optimisation Part 5 timing summary

Complete frame traces: 388; drops: 2491; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

## Throughput (the primary question for this part)

Part 5 removes cross-thread CPU contention (via core affinity) from the concurrency model Part 4 introduced, so the number that matters here is still how many frames got through the 180 s window -- and whether sam_inference_ms/geometry_metadata_ms/frame_assignment_ms fall back toward their pre-Part-4 serial-mode values.

| Received | Processed | Dropped | Failed | Hydra published | Processing ratio |
|---:|---:|---:|---:|---:|---:|
| 2879 | 385 | 2491 | 0 | 385 | 13.37% |

`sam_output_queue_wait_ms` -- time a completed SAM stage waits for the tracking/publish thread to become free. Small relative to `sam_inference_ms` means the two stages are overlapping well; large means tracking/publish has become the new bottleneck.

Mean 0.770 ms, median 0.639 ms, p95 0.811 ms, max 28.245 ms (n=388).

## Per-stage breakdown (same framing as Parts 1-3, for continuity)

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 102.480 ms, p95 170.411 ms). This framing answers "which stage costs the most," which is not the question Part 5 is testing -- see the throughput section above.

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 388 | 721.827 | 705.548 | 920.379 | 1239.767 |
| classifier_delay_ms | aggregate | False | 388 | 671.961 | 661.264 | 856.193 | 1197.798 |
| sam_delay_ms | aggregate | False | 388 | 454.412 | 448.485 | 569.317 | 942.624 |
| sam_inference_ms | leaf | False | 388 | 432.900 | 427.816 | 547.525 | 926.848 |
| rap_delay_ms | aggregate | False | 388 | 182.206 | 174.977 | 267.251 | 555.332 |
| geometry_metadata_ms | leaf | True | 388 | 102.480 | 95.519 | 170.411 | 246.502 |
| frame_assignment_ms | leaf | True | 388 | 71.351 | 66.771 | 116.182 | 441.173 |
| assignment_candidate_search_ms | leaf | False | 388 | 56.278 | 54.037 | 98.028 | 403.608 |
| geometry_projection_ms | leaf | False | 388 | 42.006 | 36.369 | 91.534 | 166.782 |
| sent_to_classifier_delay_ms | aggregate | False | 388 | 33.856 | 28.720 | 87.057 | 302.091 |
| frame_queue_wait_ms | leaf | True | 388 | 33.694 | 28.559 | 86.890 | 301.946 |
| geometry_mask_extract_ms | leaf | False | 388 | 28.274 | 25.714 | 49.148 | 67.175 |
| result_message_build_delay_ms | leaf | True | 388 | 27.041 | 27.271 | 47.001 | 66.580 |
| assignment_3d_geometry_ms | leaf | False | 388 | 23.390 | 21.326 | 46.760 | 69.440 |
| geometry_stats_ms | leaf | False | 388 | 22.454 | 19.145 | 41.932 | 72.393 |
| sam_restore_ms | leaf | False | 388 | 21.255 | 20.655 | 29.337 | 99.417 |
| assignment_centroid_iou_ms | leaf | False | 388 | 13.784 | 12.679 | 27.581 | 55.487 |
| coordinator_delay_ms | aggregate | False | 388 | 13.382 | 13.250 | 18.823 | 35.776 |
| geometry_depth_gather_ms | leaf | False | 388 | 8.839 | 6.220 | 22.243 | 61.676 |
| assignment_a2_redundancy_ms | leaf | False | 388 | 8.513 | 7.445 | 15.425 | 351.767 |
| assignment_scoring_ms | leaf | False | 388 | 8.454 | 6.798 | 16.388 | 336.305 |
| hydra_build_delay_ms | aggregate | False | 388 | 8.199 | 7.842 | 13.727 | 25.289 |
| hydra_depth_filter_ms | leaf | True | 388 | 7.549 | 7.135 | 13.028 | 24.632 |
| label_map_delay_ms | leaf | True | 388 | 6.477 | 5.476 | 13.443 | 41.682 |
| track_association_ms | leaf | True | 388 | 4.329 | 3.886 | 7.329 | 16.920 |
| hydra_publish_delay_ms | leaf | True | 388 | 4.277 | 3.901 | 7.145 | 11.885 |
| assignment_a3_nested_ms | leaf | False | 388 | 4.220 | 3.505 | 8.436 | 20.841 |
| assignment_row_init_ms | leaf | False | 388 | 2.896 | 2.160 | 7.732 | 21.619 |
| assignment_hungarian_ms | leaf | False | 388 | 2.226 | 2.225 | 4.123 | 6.980 |
| pipeline_wait_ms | leaf | False | 388 | 1.853 | 1.930 | 3.221 | 4.375 |
| crop_update_ms | leaf | True | 388 | 1.819 | 1.217 | 5.803 | 12.339 |
| image_conversion_delay_ms | leaf | True | 388 | 1.614 | 1.388 | 2.753 | 10.744 |
| active_segments_publish_ms | leaf | True | 388 | 1.448 | 1.390 | 1.893 | 4.135 |
| unknown_publish_delay_ms | leaf | True | 388 | 0.900 | 0.736 | 1.873 | 6.350 |
| sam_output_queue_wait_ms | leaf | True | 388 | 0.770 | 0.639 | 0.811 | 28.245 |
| hydra_build_other_ms | leaf | True | 388 | 0.549 | 0.543 | 0.647 | 0.980 |
| semantic_dispatch_ms | leaf | False | 388 | 0.379 | 0.363 | 0.738 | 2.181 |
| run_rap_other_ms | leaf | True | 388 | 0.372 | 0.344 | 0.478 | 3.758 |
| sam_prepare_ms | leaf | False | 388 | 0.250 | 0.196 | 0.320 | 4.789 |
| callback_enqueue_delay_ms | leaf | True | 388 | 0.162 | 0.153 | 0.213 | 0.536 |
| metadata_delay_ms | leaf | True | 388 | 0.147 | 0.142 | 0.187 | 0.316 |
| hydra_metadata_build_ms | leaf | True | 388 | 0.101 | 0.099 | 0.123 | 0.205 |
| classifier_other_ms | leaf | True | 388 | 0.065 | 0.063 | 0.078 | 0.150 |
| association_lock_wait_ms | leaf | False | 388 | 0.026 | 0.014 | 0.021 | 1.760 |
| sam_other_ms | leaf | False | 388 | 0.008 | 0.007 | 0.009 | 0.082 |
| coordinator_other_ms | leaf | True | 388 | 0.006 | 0.005 | 0.007 | 0.235 |
| assignment_lock_wait_ms | leaf | False | 388 | 0.006 | 0.005 | 0.006 | 0.389 |
| quality_deferred_release_ms | leaf | False | 388 | 0.004 | 0.004 | 0.005 | 0.008 |
| classifier_debug_record_delay_ms | leaf | False | 388 | 0.004 | 0.004 | 0.006 | 0.027 |
