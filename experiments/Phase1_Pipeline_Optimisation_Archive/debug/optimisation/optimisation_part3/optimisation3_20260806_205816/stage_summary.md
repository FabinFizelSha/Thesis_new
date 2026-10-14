# Phase 1 optimisation Part 3 timing summary

Complete frame traces: 273; drops: 2244; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 83.748 ms, p95 145.307 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 273 | 601.630 | 587.148 | 761.702 | 1357.068 |
| classifier_delay_ms | aggregate | False | 273 | 539.628 | 529.926 | 677.435 | 1019.632 |
| sam_delay_ms | aggregate | False | 273 | 337.694 | 336.754 | 409.806 | 461.537 |
| sam_inference_ms | leaf | False | 273 | 315.190 | 313.638 | 389.522 | 432.599 |
| rap_delay_ms | aggregate | False | 273 | 163.255 | 149.651 | 262.277 | 607.774 |
| geometry_metadata_ms | leaf | True | 273 | 83.748 | 78.469 | 145.307 | 218.112 |
| frame_assignment_ms | leaf | True | 273 | 70.363 | 62.701 | 107.792 | 464.120 |
| assignment_candidate_search_ms | leaf | False | 273 | 57.763 | 52.146 | 95.006 | 425.805 |
| sent_to_classifier_delay_ms | aggregate | False | 273 | 47.222 | 33.871 | 121.077 | 514.717 |
| frame_queue_wait_ms | leaf | True | 273 | 47.075 | 33.680 | 120.953 | 514.620 |
| geometry_projection_ms | leaf | False | 273 | 34.120 | 27.844 | 79.712 | 135.973 |
| result_message_build_delay_ms | leaf | True | 273 | 30.246 | 29.139 | 54.803 | 69.978 |
| geometry_mask_extract_ms | leaf | False | 273 | 23.794 | 21.646 | 39.897 | 56.352 |
| sam_restore_ms | leaf | False | 273 | 22.129 | 21.688 | 30.689 | 41.981 |
| assignment_3d_geometry_ms | leaf | False | 273 | 20.451 | 18.110 | 38.561 | 361.938 |
| geometry_stats_ms | leaf | False | 273 | 18.177 | 15.580 | 33.260 | 51.264 |
| coordinator_delay_ms | aggregate | False | 273 | 12.722 | 12.730 | 17.722 | 24.111 |
| assignment_row_init_ms | leaf | False | 273 | 12.531 | 6.105 | 19.768 | 351.020 |
| assignment_centroid_iou_ms | leaf | False | 273 | 11.176 | 9.423 | 24.472 | 45.418 |
| hydra_build_delay_ms | aggregate | False | 273 | 7.819 | 7.787 | 12.511 | 20.071 |
| hydra_depth_filter_ms | leaf | True | 273 | 7.171 | 7.195 | 11.783 | 19.456 |
| assignment_scoring_ms | leaf | False | 273 | 7.115 | 6.091 | 18.380 | 34.172 |
| geometry_depth_gather_ms | leaf | False | 273 | 6.867 | 5.534 | 13.807 | 37.649 |
| assignment_a2_redundancy_ms | leaf | False | 273 | 6.752 | 5.824 | 14.262 | 30.651 |
| track_association_ms | leaf | True | 273 | 5.402 | 3.661 | 7.681 | 330.456 |
| label_map_delay_ms | leaf | True | 273 | 5.347 | 4.204 | 12.256 | 24.906 |
| hydra_publish_delay_ms | leaf | True | 273 | 4.069 | 3.673 | 6.584 | 18.766 |
| assignment_a3_nested_ms | leaf | False | 273 | 3.775 | 3.144 | 7.703 | 23.735 |
| image_conversion_delay_ms | leaf | True | 273 | 2.875 | 1.838 | 6.843 | 11.625 |
| pipeline_wait_ms | leaf | False | 273 | 2.053 | 2.014 | 3.883 | 4.663 |
| assignment_hungarian_ms | leaf | False | 273 | 1.984 | 1.968 | 3.588 | 5.250 |
| crop_update_ms | leaf | True | 273 | 1.507 | 0.989 | 4.611 | 15.655 |
| active_segments_publish_ms | leaf | True | 273 | 1.452 | 1.384 | 1.904 | 4.687 |
| unknown_publish_delay_ms | leaf | True | 273 | 0.829 | 0.707 | 1.353 | 4.811 |
| hydra_build_other_ms | leaf | True | 273 | 0.549 | 0.529 | 0.650 | 1.928 |
| run_rap_other_ms | leaf | True | 273 | 0.396 | 0.339 | 0.453 | 11.112 |
| sam_prepare_ms | leaf | False | 273 | 0.368 | 0.209 | 0.940 | 10.795 |
| semantic_dispatch_ms | leaf | False | 273 | 0.362 | 0.340 | 0.752 | 2.873 |
| metadata_delay_ms | leaf | True | 273 | 0.153 | 0.147 | 0.198 | 0.696 |
| callback_enqueue_delay_ms | leaf | True | 273 | 0.147 | 0.141 | 0.186 | 0.317 |
| hydra_metadata_build_ms | leaf | True | 273 | 0.099 | 0.097 | 0.124 | 0.169 |
| classifier_other_ms | leaf | True | 273 | 0.059 | 0.045 | 0.056 | 3.583 |
| sam_other_ms | leaf | False | 273 | 0.008 | 0.007 | 0.009 | 0.066 |
| quality_deferred_release_ms | leaf | False | 273 | 0.005 | 0.005 | 0.007 | 0.009 |
| coordinator_other_ms | leaf | True | 273 | 0.005 | 0.005 | 0.007 | 0.023 |
| classifier_debug_record_delay_ms | leaf | False | 273 | 0.005 | 0.004 | 0.007 | 0.026 |

## frame_assignment_ms and candidate-count growth across the run

Tests the hypothesis that per-frame assignment cost grows as the persistent-track registry fills the explored scene, independent of whole-run mean.

| Quartile | Frames | frame_assignment_ms mean | candidate_count_total mean | candidate_count_max mean |
|---:|---:|---:|---:|---:|
| 1 | 68 | 37.862 | 155.85 | 29.72 |
| 2 | 68 | 62.753 | 461.57 | 119.18 |
| 3 | 68 | 66.970 | 656.51 | 223.28 |
| 4 | 69 | 113.236 | 1027.16 | 290.91 |
