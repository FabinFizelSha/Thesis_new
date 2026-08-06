# Phase 1 optimisation Part 3 timing summary

Complete frame traces: 303; drops: 2526; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 86.966 ms, p95 150.665 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 303 | 602.392 | 603.160 | 813.911 | 1257.563 |
| classifier_delay_ms | aggregate | False | 303 | 541.846 | 545.937 | 722.995 | 1022.002 |
| sam_delay_ms | aggregate | False | 303 | 328.554 | 335.547 | 427.756 | 506.548 |
| sam_inference_ms | leaf | False | 303 | 308.803 | 314.789 | 403.353 | 488.738 |
| rap_delay_ms | aggregate | False | 303 | 171.765 | 158.447 | 269.370 | 566.709 |
| geometry_metadata_ms | leaf | True | 303 | 86.966 | 82.380 | 150.665 | 207.999 |
| frame_assignment_ms | leaf | True | 303 | 76.349 | 65.922 | 122.547 | 472.879 |
| sent_to_classifier_delay_ms | aggregate | False | 303 | 45.372 | 34.667 | 119.520 | 704.381 |
| frame_queue_wait_ms | leaf | True | 303 | 45.221 | 34.520 | 119.398 | 704.272 |
| result_message_build_delay_ms | leaf | True | 303 | 32.071 | 31.618 | 60.659 | 85.155 |
| sam_restore_ms | leaf | False | 303 | 19.406 | 18.564 | 26.988 | 34.278 |
| coordinator_delay_ms | aggregate | False | 303 | 12.860 | 12.713 | 18.001 | 37.366 |
| hydra_build_delay_ms | aggregate | False | 303 | 7.825 | 7.669 | 11.954 | 22.731 |
| hydra_depth_filter_ms | leaf | True | 303 | 7.120 | 6.953 | 11.129 | 18.587 |
| label_map_delay_ms | leaf | True | 303 | 6.089 | 4.893 | 14.056 | 32.553 |
| track_association_ms | leaf | True | 303 | 4.388 | 3.774 | 8.483 | 18.391 |
| hydra_publish_delay_ms | leaf | True | 303 | 4.065 | 3.705 | 6.350 | 13.923 |
| image_conversion_delay_ms | leaf | True | 303 | 3.137 | 2.163 | 7.164 | 16.643 |
| pipeline_wait_ms | leaf | False | 303 | 2.309 | 2.348 | 4.376 | 9.019 |
| crop_update_ms | leaf | True | 303 | 1.666 | 1.000 | 5.697 | 11.406 |
| active_segments_publish_ms | leaf | True | 303 | 1.548 | 1.405 | 1.889 | 33.360 |
| unknown_publish_delay_ms | leaf | True | 303 | 0.964 | 0.724 | 2.287 | 10.168 |
| hydra_build_other_ms | leaf | True | 303 | 0.602 | 0.547 | 0.738 | 5.437 |
| semantic_dispatch_ms | leaf | False | 303 | 0.411 | 0.340 | 0.871 | 5.965 |
| run_rap_other_ms | leaf | True | 303 | 0.410 | 0.356 | 0.519 | 6.073 |
| sam_prepare_ms | leaf | False | 303 | 0.337 | 0.202 | 0.876 | 9.378 |
| metadata_delay_ms | leaf | True | 303 | 0.178 | 0.151 | 0.203 | 5.061 |
| callback_enqueue_delay_ms | leaf | True | 303 | 0.151 | 0.143 | 0.188 | 0.410 |
| hydra_metadata_build_ms | leaf | True | 303 | 0.102 | 0.096 | 0.131 | 0.452 |
| classifier_other_ms | leaf | True | 303 | 0.052 | 0.045 | 0.064 | 1.254 |
| sam_other_ms | leaf | False | 303 | 0.007 | 0.007 | 0.010 | 0.036 |
| coordinator_other_ms | leaf | True | 303 | 0.006 | 0.005 | 0.007 | 0.189 |
| quality_deferred_release_ms | leaf | False | 303 | 0.006 | 0.005 | 0.008 | 0.022 |
| classifier_debug_record_delay_ms | leaf | False | 303 | 0.005 | 0.005 | 0.008 | 0.011 |
