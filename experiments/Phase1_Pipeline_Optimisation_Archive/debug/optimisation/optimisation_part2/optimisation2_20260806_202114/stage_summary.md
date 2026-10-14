# Phase 1 optimisation Part 2 timing summary

Complete frame traces: 304; drops: 2603; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 85.669 ms, p95 146.181 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 304 | 605.903 | 595.679 | 770.376 | 1229.409 |
| classifier_delay_ms | aggregate | False | 304 | 548.102 | 542.662 | 682.875 | 983.873 |
| sam_delay_ms | aggregate | False | 304 | 343.558 | 343.474 | 419.477 | 476.022 |
| sam_inference_ms | leaf | False | 304 | 322.547 | 323.195 | 400.706 | 460.727 |
| rap_delay_ms | aggregate | False | 304 | 163.526 | 154.617 | 266.072 | 574.920 |
| geometry_metadata_ms | leaf | True | 304 | 85.669 | 80.550 | 146.181 | 192.587 |
| frame_assignment_ms | leaf | True | 304 | 69.484 | 62.216 | 109.872 | 450.933 |
| sent_to_classifier_delay_ms | aggregate | False | 304 | 42.912 | 33.834 | 110.941 | 623.532 |
| frame_queue_wait_ms | leaf | True | 304 | 42.760 | 33.676 | 110.848 | 623.449 |
| geometry_projection_ms | leaf | False | 304 | 36.293 | 32.550 | 82.181 | 139.282 |
| result_message_build_delay_ms | leaf | True | 304 | 32.549 | 32.130 | 57.337 | 79.325 |
| geometry_mask_extract_ms | leaf | False | 304 | 24.104 | 21.545 | 40.100 | 66.092 |
| sam_restore_ms | leaf | False | 304 | 20.619 | 20.280 | 29.717 | 36.468 |
| geometry_stats_ms | leaf | False | 304 | 18.145 | 15.189 | 37.095 | 54.408 |
| coordinator_delay_ms | aggregate | False | 304 | 12.611 | 12.482 | 17.635 | 25.689 |
| hydra_build_delay_ms | aggregate | False | 304 | 7.774 | 7.744 | 11.949 | 18.899 |
| hydra_depth_filter_ms | leaf | True | 304 | 7.123 | 7.067 | 11.289 | 18.259 |
| geometry_depth_gather_ms | leaf | False | 304 | 6.292 | 5.136 | 13.367 | 32.202 |
| label_map_delay_ms | leaf | True | 304 | 5.219 | 4.433 | 10.843 | 35.695 |
| track_association_ms | leaf | True | 304 | 4.368 | 3.670 | 8.176 | 24.046 |
| hydra_publish_delay_ms | leaf | True | 304 | 3.954 | 3.661 | 5.566 | 8.959 |
| image_conversion_delay_ms | leaf | True | 304 | 3.039 | 2.158 | 6.521 | 11.287 |
| pipeline_wait_ms | leaf | False | 304 | 2.274 | 2.300 | 4.064 | 9.233 |
| active_segments_publish_ms | leaf | True | 304 | 1.630 | 1.369 | 2.002 | 15.976 |
| crop_update_ms | leaf | True | 304 | 1.624 | 1.032 | 5.176 | 11.974 |
| unknown_publish_delay_ms | leaf | True | 304 | 0.878 | 0.717 | 1.972 | 5.819 |
| hydra_build_other_ms | leaf | True | 304 | 0.549 | 0.532 | 0.645 | 4.347 |
| sam_prepare_ms | leaf | False | 304 | 0.384 | 0.210 | 1.012 | 7.081 |
| semantic_dispatch_ms | leaf | False | 304 | 0.377 | 0.362 | 0.751 | 2.417 |
| run_rap_other_ms | leaf | True | 304 | 0.351 | 0.342 | 0.439 | 0.605 |
| metadata_delay_ms | leaf | True | 304 | 0.155 | 0.150 | 0.196 | 0.346 |
| callback_enqueue_delay_ms | leaf | True | 304 | 0.152 | 0.144 | 0.198 | 0.837 |
| hydra_metadata_build_ms | leaf | True | 304 | 0.101 | 0.097 | 0.125 | 0.277 |
| classifier_other_ms | leaf | True | 304 | 0.056 | 0.045 | 0.055 | 3.080 |
| sam_other_ms | leaf | False | 304 | 0.008 | 0.007 | 0.009 | 0.022 |
| coordinator_other_ms | leaf | True | 304 | 0.006 | 0.005 | 0.007 | 0.041 |
| quality_deferred_release_ms | leaf | False | 304 | 0.004 | 0.004 | 0.006 | 0.015 |
| classifier_debug_record_delay_ms | leaf | False | 304 | 0.003 | 0.003 | 0.005 | 0.006 |
