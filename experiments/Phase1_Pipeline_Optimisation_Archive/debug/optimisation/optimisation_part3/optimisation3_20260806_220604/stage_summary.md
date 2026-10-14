# Phase 1 optimisation Part 3 timing summary

Complete frame traces: 304; drops: 2436; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 85.150 ms, p95 151.731 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 304 | 589.961 | 579.772 | 744.811 | 1119.836 |
| classifier_delay_ms | aggregate | False | 304 | 530.069 | 529.241 | 657.721 | 1006.016 |
| sam_delay_ms | aggregate | False | 304 | 337.553 | 336.099 | 415.057 | 465.676 |
| sam_inference_ms | leaf | False | 304 | 316.303 | 313.640 | 394.257 | 442.543 |
| rap_delay_ms | aggregate | False | 304 | 157.259 | 147.400 | 239.117 | 571.476 |
| geometry_metadata_ms | leaf | True | 304 | 85.150 | 74.440 | 151.731 | 218.756 |
| frame_assignment_ms | leaf | True | 304 | 64.016 | 61.861 | 100.225 | 456.336 |
| assignment_candidate_search_ms | leaf | False | 304 | 51.551 | 50.190 | 86.043 | 425.995 |
| sent_to_classifier_delay_ms | aggregate | False | 304 | 45.389 | 32.464 | 111.386 | 542.393 |
| frame_queue_wait_ms | leaf | True | 304 | 45.241 | 32.323 | 111.247 | 542.279 |
| geometry_projection_ms | leaf | False | 304 | 34.846 | 28.008 | 86.245 | 150.076 |
| result_message_build_delay_ms | leaf | True | 304 | 26.549 | 26.652 | 46.734 | 57.014 |
| geometry_mask_extract_ms | leaf | False | 304 | 24.018 | 21.041 | 41.749 | 66.970 |
| assignment_3d_geometry_ms | leaf | False | 304 | 21.695 | 19.600 | 40.414 | 377.591 |
| sam_restore_ms | leaf | False | 304 | 20.885 | 20.778 | 29.795 | 35.164 |
| geometry_stats_ms | leaf | False | 304 | 18.740 | 15.125 | 40.790 | 62.313 |
| coordinator_delay_ms | aggregate | False | 304 | 12.660 | 12.529 | 17.590 | 30.897 |
| assignment_centroid_iou_ms | leaf | False | 304 | 12.494 | 11.104 | 29.695 | 43.193 |
| hydra_build_delay_ms | aggregate | False | 304 | 7.676 | 7.217 | 11.575 | 26.835 |
| assignment_scoring_ms | leaf | False | 304 | 7.180 | 6.312 | 16.912 | 29.051 |
| hydra_depth_filter_ms | leaf | True | 304 | 7.007 | 6.411 | 10.848 | 26.239 |
| geometry_depth_gather_ms | leaf | False | 304 | 6.750 | 5.322 | 15.657 | 53.903 |
| assignment_a2_redundancy_ms | leaf | False | 304 | 6.644 | 6.465 | 12.516 | 24.167 |
| label_map_delay_ms | leaf | True | 304 | 5.466 | 4.428 | 12.471 | 29.939 |
| track_association_ms | leaf | True | 304 | 4.242 | 3.692 | 7.609 | 25.615 |
| hydra_publish_delay_ms | leaf | True | 304 | 4.115 | 3.775 | 6.444 | 9.132 |
| assignment_row_init_ms | leaf | False | 304 | 3.674 | 2.054 | 7.969 | 317.840 |
| assignment_a3_nested_ms | leaf | False | 304 | 3.625 | 3.159 | 7.748 | 16.941 |
| image_conversion_delay_ms | leaf | True | 304 | 3.043 | 2.218 | 6.939 | 12.989 |
| assignment_hungarian_ms | leaf | False | 304 | 2.105 | 2.067 | 3.761 | 12.195 |
| pipeline_wait_ms | leaf | False | 304 | 1.838 | 1.856 | 3.228 | 5.399 |
| crop_update_ms | leaf | True | 304 | 1.621 | 1.001 | 5.536 | 12.600 |
| active_segments_publish_ms | leaf | True | 304 | 1.436 | 1.388 | 1.827 | 4.591 |
| unknown_publish_delay_ms | leaf | True | 304 | 0.864 | 0.719 | 1.200 | 8.985 |
| hydra_build_other_ms | leaf | True | 304 | 0.569 | 0.537 | 0.697 | 6.160 |
| semantic_dispatch_ms | leaf | False | 304 | 0.389 | 0.369 | 0.731 | 3.367 |
| run_rap_other_ms | leaf | True | 304 | 0.380 | 0.349 | 0.459 | 4.487 |
| sam_prepare_ms | leaf | False | 304 | 0.358 | 0.212 | 0.874 | 7.039 |
| metadata_delay_ms | leaf | True | 304 | 0.154 | 0.150 | 0.198 | 0.331 |
| callback_enqueue_delay_ms | leaf | True | 304 | 0.148 | 0.143 | 0.186 | 0.490 |
| hydra_metadata_build_ms | leaf | True | 304 | 0.100 | 0.097 | 0.128 | 0.167 |
| classifier_other_ms | leaf | True | 304 | 0.045 | 0.045 | 0.057 | 0.103 |
| sam_other_ms | leaf | False | 304 | 0.007 | 0.007 | 0.009 | 0.014 |
| quality_deferred_release_ms | leaf | False | 304 | 0.006 | 0.005 | 0.007 | 0.243 |
| coordinator_other_ms | leaf | True | 304 | 0.005 | 0.005 | 0.007 | 0.087 |
| classifier_debug_record_delay_ms | leaf | False | 304 | 0.005 | 0.005 | 0.007 | 0.011 |

## frame_assignment_ms and candidate-count growth across the run

Tests the hypothesis that per-frame assignment cost grows as the persistent-track registry fills the explored scene, independent of whole-run mean.

| Quartile | Frames | frame_assignment_ms mean | candidate_count_total mean | candidate_count_max mean |
|---:|---:|---:|---:|---:|
| 1 | 76 | 36.225 | 195.89 | 39.87 |
| 2 | 76 | 63.848 | 556.74 | 137.76 |
| 3 | 76 | 63.412 | 714.22 | 232.26 |
| 4 | 76 | 92.579 | 1056.22 | 300.59 |
