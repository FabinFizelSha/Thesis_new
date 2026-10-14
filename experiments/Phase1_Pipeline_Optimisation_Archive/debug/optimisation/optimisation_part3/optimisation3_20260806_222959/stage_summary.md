# Phase 1 optimisation Part 3 timing summary

Complete frame traces: 277; drops: 2106; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 83.939 ms, p95 149.818 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 277 | 568.379 | 554.715 | 722.210 | 1293.912 |
| classifier_delay_ms | aggregate | False | 277 | 508.370 | 504.265 | 639.460 | 834.433 |
| sam_delay_ms | aggregate | False | 277 | 326.259 | 323.471 | 406.747 | 516.752 |
| sam_inference_ms | leaf | False | 277 | 304.238 | 300.589 | 383.414 | 490.459 |
| rap_delay_ms | aggregate | False | 277 | 150.850 | 142.502 | 236.154 | 481.756 |
| geometry_metadata_ms | leaf | True | 277 | 83.939 | 77.057 | 149.818 | 249.965 |
| frame_assignment_ms | leaf | True | 277 | 58.944 | 54.736 | 106.419 | 335.660 |
| assignment_candidate_search_ms | leaf | False | 277 | 46.937 | 44.849 | 93.019 | 314.699 |
| sent_to_classifier_delay_ms | aggregate | False | 277 | 45.719 | 30.175 | 117.945 | 690.985 |
| frame_queue_wait_ms | leaf | True | 277 | 45.568 | 30.006 | 117.810 | 690.885 |
| geometry_projection_ms | leaf | False | 277 | 34.677 | 27.366 | 80.502 | 130.336 |
| geometry_mask_extract_ms | leaf | False | 277 | 23.505 | 20.575 | 40.470 | 76.234 |
| result_message_build_delay_ms | leaf | True | 277 | 23.044 | 21.828 | 41.112 | 51.672 |
| sam_restore_ms | leaf | False | 277 | 21.622 | 21.479 | 29.534 | 34.882 |
| assignment_3d_geometry_ms | leaf | False | 277 | 20.719 | 17.885 | 43.834 | 60.334 |
| geometry_stats_ms | leaf | False | 277 | 18.240 | 16.379 | 34.864 | 58.060 |
| coordinator_delay_ms | aggregate | False | 277 | 12.606 | 12.337 | 17.612 | 33.542 |
| assignment_centroid_iou_ms | leaf | False | 277 | 10.950 | 8.846 | 23.892 | 42.159 |
| hydra_build_delay_ms | aggregate | False | 277 | 7.576 | 6.876 | 12.138 | 23.065 |
| hydra_depth_filter_ms | leaf | True | 277 | 6.925 | 6.194 | 11.469 | 22.500 |
| geometry_depth_gather_ms | leaf | False | 277 | 6.707 | 5.327 | 14.116 | 29.902 |
| assignment_a2_redundancy_ms | leaf | False | 277 | 6.298 | 5.830 | 12.427 | 19.502 |
| assignment_scoring_ms | leaf | False | 277 | 6.199 | 4.978 | 14.165 | 35.256 |
| label_map_delay_ms | leaf | True | 277 | 5.226 | 4.311 | 11.007 | 19.899 |
| hydra_publish_delay_ms | leaf | True | 277 | 4.218 | 3.779 | 7.144 | 16.746 |
| track_association_ms | leaf | True | 277 | 4.185 | 3.655 | 7.971 | 14.190 |
| assignment_a3_nested_ms | leaf | False | 277 | 3.717 | 3.205 | 7.344 | 12.481 |
| image_conversion_delay_ms | leaf | True | 277 | 2.760 | 1.827 | 6.505 | 11.520 |
| assignment_row_init_ms | leaf | False | 277 | 2.275 | 1.671 | 5.880 | 17.604 |
| assignment_hungarian_ms | leaf | False | 277 | 1.902 | 1.808 | 3.622 | 8.700 |
| pipeline_wait_ms | leaf | False | 277 | 1.680 | 1.590 | 2.931 | 8.422 |
| active_segments_publish_ms | leaf | True | 277 | 1.514 | 1.389 | 1.845 | 16.062 |
| crop_update_ms | leaf | True | 277 | 1.486 | 1.026 | 4.567 | 13.730 |
| unknown_publish_delay_ms | leaf | True | 277 | 0.793 | 0.702 | 1.011 | 4.754 |
| hydra_build_other_ms | leaf | True | 277 | 0.550 | 0.534 | 0.653 | 2.536 |
| sam_prepare_ms | leaf | False | 277 | 0.391 | 0.209 | 1.269 | 5.699 |
| semantic_dispatch_ms | leaf | False | 277 | 0.380 | 0.328 | 0.718 | 5.456 |
| run_rap_other_ms | leaf | True | 277 | 0.377 | 0.355 | 0.495 | 2.520 |
| metadata_delay_ms | leaf | True | 277 | 0.171 | 0.148 | 0.195 | 3.388 |
| callback_enqueue_delay_ms | leaf | True | 277 | 0.151 | 0.145 | 0.188 | 0.377 |
| hydra_metadata_build_ms | leaf | True | 277 | 0.101 | 0.097 | 0.124 | 0.355 |
| classifier_other_ms | leaf | True | 277 | 0.060 | 0.046 | 0.059 | 3.135 |
| association_lock_wait_ms | leaf | False | 277 | 0.051 | 0.015 | 0.023 | 3.939 |
| coordinator_other_ms | leaf | True | 277 | 0.019 | 0.005 | 0.007 | 3.613 |
| assignment_lock_wait_ms | leaf | False | 277 | 0.010 | 0.005 | 0.006 | 1.369 |
| sam_other_ms | leaf | False | 277 | 0.008 | 0.007 | 0.009 | 0.032 |
| quality_deferred_release_ms | leaf | False | 277 | 0.005 | 0.005 | 0.006 | 0.010 |
| classifier_debug_record_delay_ms | leaf | False | 277 | 0.004 | 0.003 | 0.006 | 0.136 |

## frame_assignment_ms and candidate-count growth across the run

Tests the hypothesis that per-frame assignment cost grows as the persistent-track registry fills the explored scene, independent of whole-run mean.

| Quartile | Frames | frame_assignment_ms mean | candidate_count_total mean | candidate_count_max mean |
|---:|---:|---:|---:|---:|
| 1 | 69 | 35.806 | 152.04 | 32.72 |
| 2 | 69 | 53.722 | 435.19 | 130.06 |
| 3 | 69 | 57.800 | 577.81 | 208.20 |
| 4 | 70 | 88.028 | 944.53 | 273.71 |
