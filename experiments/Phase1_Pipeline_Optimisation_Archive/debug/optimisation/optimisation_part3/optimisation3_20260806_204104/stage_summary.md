# Phase 1 optimisation Part 3 timing summary

Complete frame traces: 301; drops: 2624; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 83.433 ms, p95 143.794 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 301 | 606.377 | 600.158 | 756.509 | 1062.869 |
| classifier_delay_ms | aggregate | False | 301 | 552.680 | 547.471 | 673.051 | 1037.040 |
| sam_delay_ms | aggregate | False | 301 | 348.978 | 350.037 | 425.915 | 513.865 |
| sam_inference_ms | leaf | False | 301 | 327.045 | 328.337 | 406.760 | 483.098 |
| rap_delay_ms | aggregate | False | 301 | 162.653 | 150.471 | 251.412 | 601.209 |
| geometry_metadata_ms | leaf | True | 301 | 83.433 | 77.004 | 143.794 | 211.376 |
| frame_assignment_ms | leaf | True | 301 | 71.308 | 62.398 | 116.621 | 475.532 |
| assignment_candidate_search_ms | leaf | False | 301 | 58.625 | 51.860 | 103.294 | 450.494 |
| sent_to_classifier_delay_ms | aggregate | False | 301 | 38.123 | 33.625 | 101.947 | 374.943 |
| frame_queue_wait_ms | leaf | True | 301 | 37.973 | 33.356 | 101.824 | 374.814 |
| geometry_projection_ms | leaf | False | 301 | 34.641 | 28.925 | 77.885 | 155.112 |
| result_message_build_delay_ms | leaf | True | 301 | 32.444 | 33.055 | 58.505 | 81.557 |
| geometry_mask_extract_ms | leaf | False | 301 | 23.223 | 21.192 | 39.966 | 61.369 |
| sam_restore_ms | leaf | False | 301 | 21.577 | 21.425 | 31.218 | 36.127 |
| geometry_stats_ms | leaf | False | 301 | 18.167 | 15.415 | 36.545 | 49.745 |
| coordinator_delay_ms | aggregate | False | 301 | 13.215 | 12.868 | 19.105 | 29.859 |
| hydra_build_delay_ms | aggregate | False | 301 | 8.267 | 8.154 | 13.679 | 25.339 |
| hydra_depth_filter_ms | leaf | True | 301 | 7.574 | 7.525 | 12.940 | 20.393 |
| assignment_a2_redundancy_ms | leaf | False | 301 | 6.796 | 6.183 | 14.335 | 35.074 |
| geometry_depth_gather_ms | leaf | False | 301 | 6.619 | 5.216 | 13.906 | 34.338 |
| label_map_delay_ms | leaf | True | 301 | 5.223 | 4.245 | 10.814 | 23.678 |
| hydra_publish_delay_ms | leaf | True | 301 | 4.046 | 3.755 | 5.815 | 10.020 |
| track_association_ms | leaf | True | 301 | 3.967 | 3.671 | 6.499 | 16.660 |
| assignment_a3_nested_ms | leaf | False | 301 | 3.685 | 3.209 | 7.218 | 22.761 |
| image_conversion_delay_ms | leaf | True | 301 | 3.183 | 2.318 | 6.954 | 15.065 |
| pipeline_wait_ms | leaf | False | 301 | 2.331 | 2.333 | 4.577 | 9.256 |
| assignment_hungarian_ms | leaf | False | 301 | 2.111 | 2.066 | 4.110 | 8.717 |
| crop_update_ms | leaf | True | 301 | 1.698 | 1.037 | 5.184 | 21.396 |
| active_segments_publish_ms | leaf | True | 301 | 1.489 | 1.394 | 1.918 | 16.372 |
| unknown_publish_delay_ms | leaf | True | 301 | 0.894 | 0.722 | 2.094 | 6.153 |
| hydra_build_other_ms | leaf | True | 301 | 0.590 | 0.536 | 0.680 | 5.708 |
| semantic_dispatch_ms | leaf | False | 301 | 0.378 | 0.358 | 0.713 | 5.484 |
| run_rap_other_ms | leaf | True | 301 | 0.357 | 0.345 | 0.464 | 0.720 |
| sam_prepare_ms | leaf | False | 301 | 0.348 | 0.209 | 1.238 | 4.510 |
| metadata_delay_ms | leaf | True | 301 | 0.152 | 0.146 | 0.189 | 0.353 |
| callback_enqueue_delay_ms | leaf | True | 301 | 0.150 | 0.145 | 0.187 | 0.398 |
| hydra_metadata_build_ms | leaf | True | 301 | 0.104 | 0.098 | 0.135 | 0.397 |
| classifier_other_ms | leaf | True | 301 | 0.048 | 0.046 | 0.060 | 0.221 |
| classifier_debug_record_delay_ms | leaf | False | 301 | 0.027 | 0.004 | 0.007 | 3.840 |
| coordinator_other_ms | leaf | True | 301 | 0.008 | 0.005 | 0.007 | 0.766 |
| sam_other_ms | leaf | False | 301 | 0.008 | 0.008 | 0.010 | 0.028 |
| quality_deferred_release_ms | leaf | False | 301 | 0.005 | 0.005 | 0.006 | 0.030 |

## frame_assignment_ms and candidate-count growth across the run

Tests the hypothesis that per-frame assignment cost grows as the persistent-track registry fills the explored scene, independent of whole-run mean.

| Quartile | Frames | frame_assignment_ms mean | candidate_count_total mean | candidate_count_max mean |
|---:|---:|---:|---:|---:|
| 1 | 75 | 36.427 | 165.60 | 33.09 |
| 2 | 75 | 74.758 | 529.91 | 130.31 |
| 3 | 75 | 65.774 | 673.03 | 233.55 |
| 4 | 76 | 107.787 | 1065.13 | 302.78 |
