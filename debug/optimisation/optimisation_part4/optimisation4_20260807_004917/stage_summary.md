# Phase 1 optimisation Part 4 timing summary

Complete frame traces: 376; drops: 2381; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

## Throughput (the primary question for this part)

Part 4 changes concurrency (SAM for frame N+1 overlaps tracking/publish for frame N), not any stage's own compute cost, so the number that matters here is how many frames got through the 180 s window, not which single leaf stage is largest.

| Received | Processed | Dropped | Failed | Hydra published | Processing ratio |
|---:|---:|---:|---:|---:|---:|
| 2757 | 373 | 2381 | 0 | 373 | 13.53% |

`sam_output_queue_wait_ms` -- time a completed SAM stage waits for the tracking/publish thread to become free. Small relative to `sam_inference_ms` means the two stages are overlapping well; large means tracking/publish has become the new bottleneck.

Mean 0.704 ms, median 0.632 ms, p95 0.888 ms, max 4.453 ms (n=376).

## Per-stage breakdown (same framing as Parts 1-3, for continuity)

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 101.858 ms, p95 172.932 ms). This framing answers "which stage costs the most," which is not the question Part 4 is testing -- see the throughput section above.

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 376 | 727.406 | 712.513 | 896.909 | 1563.460 |
| classifier_delay_ms | aggregate | False | 376 | 676.062 | 667.356 | 822.642 | 1150.665 |
| sam_delay_ms | aggregate | False | 376 | 456.959 | 449.904 | 551.186 | 937.752 |
| sam_inference_ms | leaf | False | 376 | 436.291 | 427.936 | 527.019 | 921.258 |
| rap_delay_ms | aggregate | False | 376 | 184.078 | 178.163 | 261.565 | 520.952 |
| geometry_metadata_ms | leaf | True | 376 | 101.858 | 95.809 | 172.932 | 242.858 |
| frame_assignment_ms | leaf | True | 376 | 74.013 | 69.897 | 113.085 | 452.982 |
| assignment_candidate_search_ms | leaf | False | 376 | 60.555 | 57.755 | 99.308 | 421.399 |
| geometry_projection_ms | leaf | False | 376 | 40.986 | 36.722 | 86.369 | 138.881 |
| sent_to_classifier_delay_ms | aggregate | False | 376 | 35.142 | 26.937 | 87.605 | 535.785 |
| frame_queue_wait_ms | leaf | True | 376 | 34.980 | 26.783 | 87.446 | 535.702 |
| geometry_mask_extract_ms | leaf | False | 376 | 28.271 | 26.111 | 46.731 | 71.393 |
| assignment_3d_geometry_ms | leaf | False | 376 | 27.166 | 25.501 | 49.735 | 66.348 |
| result_message_build_delay_ms | leaf | True | 376 | 27.141 | 25.623 | 48.752 | 64.258 |
| geometry_stats_ms | leaf | False | 376 | 22.925 | 20.379 | 44.913 | 80.550 |
| sam_restore_ms | leaf | False | 376 | 20.415 | 19.921 | 29.569 | 59.070 |
| assignment_centroid_iou_ms | leaf | False | 376 | 14.761 | 12.674 | 28.175 | 341.840 |
| coordinator_delay_ms | aggregate | False | 376 | 13.654 | 13.373 | 19.001 | 29.273 |
| geometry_depth_gather_ms | leaf | False | 376 | 8.734 | 6.565 | 20.248 | 46.212 |
| hydra_build_delay_ms | aggregate | False | 376 | 8.251 | 7.903 | 13.243 | 23.658 |
| hydra_depth_filter_ms | leaf | True | 376 | 7.594 | 7.218 | 12.421 | 23.022 |
| assignment_scoring_ms | leaf | False | 376 | 7.215 | 6.656 | 14.073 | 31.494 |
| assignment_a2_redundancy_ms | leaf | False | 376 | 7.077 | 6.480 | 12.662 | 41.658 |
| label_map_delay_ms | leaf | True | 376 | 5.979 | 5.247 | 11.878 | 21.541 |
| hydra_publish_delay_ms | leaf | True | 376 | 4.569 | 4.118 | 7.886 | 10.229 |
| track_association_ms | leaf | True | 376 | 4.317 | 3.914 | 7.603 | 11.397 |
| assignment_a3_nested_ms | leaf | False | 376 | 4.138 | 3.512 | 8.265 | 24.647 |
| assignment_row_init_ms | leaf | False | 376 | 3.573 | 2.181 | 7.239 | 341.105 |
| assignment_hungarian_ms | leaf | False | 376 | 2.145 | 2.084 | 3.964 | 8.853 |
| pipeline_wait_ms | leaf | False | 376 | 1.840 | 1.828 | 3.169 | 4.357 |
| crop_update_ms | leaf | True | 376 | 1.700 | 1.149 | 5.019 | 10.615 |
| image_conversion_delay_ms | leaf | True | 376 | 1.687 | 1.363 | 3.550 | 16.755 |
| active_segments_publish_ms | leaf | True | 376 | 1.416 | 1.383 | 1.751 | 4.954 |
| unknown_publish_delay_ms | leaf | True | 376 | 0.829 | 0.718 | 1.162 | 4.978 |
| sam_output_queue_wait_ms | leaf | True | 376 | 0.704 | 0.632 | 0.888 | 4.453 |
| hydra_build_other_ms | leaf | True | 376 | 0.557 | 0.541 | 0.671 | 4.562 |
| run_rap_other_ms | leaf | True | 376 | 0.374 | 0.343 | 0.433 | 9.380 |
| semantic_dispatch_ms | leaf | False | 376 | 0.372 | 0.363 | 0.736 | 1.375 |
| sam_prepare_ms | leaf | False | 376 | 0.246 | 0.195 | 0.355 | 3.400 |
| callback_enqueue_delay_ms | leaf | True | 376 | 0.161 | 0.156 | 0.204 | 0.400 |
| metadata_delay_ms | leaf | True | 376 | 0.152 | 0.144 | 0.193 | 0.737 |
| hydra_metadata_build_ms | leaf | True | 376 | 0.099 | 0.097 | 0.123 | 0.231 |
| classifier_other_ms | leaf | True | 376 | 0.067 | 0.066 | 0.081 | 0.171 |
| association_lock_wait_ms | leaf | False | 376 | 0.038 | 0.015 | 0.021 | 1.913 |
| sam_other_ms | leaf | False | 376 | 0.007 | 0.007 | 0.009 | 0.011 |
| quality_deferred_release_ms | leaf | False | 376 | 0.005 | 0.005 | 0.007 | 0.011 |
| assignment_lock_wait_ms | leaf | False | 376 | 0.005 | 0.005 | 0.006 | 0.109 |
| coordinator_other_ms | leaf | True | 376 | 0.005 | 0.005 | 0.007 | 0.028 |
| classifier_debug_record_delay_ms | leaf | False | 376 | 0.004 | 0.004 | 0.007 | 0.026 |
