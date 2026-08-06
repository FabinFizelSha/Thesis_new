# Phase 1 optimisation Part 3 timing summary

Complete frame traces: 239; drops: 2570; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **crop_update_ms** (mean 135.032 ms, p95 307.942 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 239 | 743.077 | 726.837 | 962.189 | 1381.612 |
| classifier_delay_ms | aggregate | False | 239 | 683.766 | 673.546 | 861.458 | 1316.400 |
| sam_delay_ms | aggregate | False | 239 | 349.051 | 348.281 | 415.158 | 493.541 |
| sam_inference_ms | leaf | False | 239 | 328.441 | 326.262 | 393.383 | 475.044 |
| rap_delay_ms | aggregate | False | 239 | 294.757 | 275.427 | 490.021 | 938.616 |
| crop_update_ms | leaf | True | 239 | 135.032 | 118.404 | 307.942 | 439.464 |
| geometry_metadata_ms | leaf | True | 239 | 84.395 | 80.940 | 141.786 | 221.979 |
| frame_assignment_ms | leaf | True | 239 | 68.523 | 62.984 | 105.033 | 439.794 |
| sent_to_classifier_delay_ms | aggregate | False | 239 | 44.319 | 33.861 | 106.639 | 525.018 |
| frame_queue_wait_ms | leaf | True | 239 | 44.150 | 33.701 | 106.474 | 524.856 |
| result_message_build_delay_ms | leaf | True | 239 | 31.570 | 31.728 | 56.477 | 75.215 |
| sam_restore_ms | leaf | False | 239 | 20.228 | 19.575 | 29.425 | 36.082 |
| coordinator_delay_ms | aggregate | False | 239 | 12.715 | 12.585 | 18.228 | 23.928 |
| hydra_build_delay_ms | aggregate | False | 239 | 7.941 | 7.916 | 12.019 | 19.610 |
| hydra_depth_filter_ms | leaf | True | 239 | 7.284 | 7.191 | 11.381 | 18.967 |
| label_map_delay_ms | leaf | True | 239 | 4.928 | 4.700 | 9.721 | 17.932 |
| track_association_ms | leaf | True | 239 | 4.433 | 4.331 | 5.994 | 10.504 |
| hydra_publish_delay_ms | leaf | True | 239 | 3.935 | 3.654 | 5.571 | 10.069 |
| image_conversion_delay_ms | leaf | True | 239 | 3.265 | 2.186 | 8.164 | 16.518 |
| pipeline_wait_ms | leaf | False | 239 | 2.272 | 2.208 | 4.276 | 8.004 |
| active_segments_publish_ms | leaf | True | 239 | 1.419 | 1.416 | 1.663 | 2.185 |
| unknown_publish_delay_ms | leaf | True | 239 | 0.806 | 0.712 | 0.997 | 4.413 |
| hydra_build_other_ms | leaf | True | 239 | 0.558 | 0.523 | 0.633 | 4.281 |
| run_rap_other_ms | leaf | True | 239 | 0.535 | 0.495 | 0.632 | 5.541 |
| semantic_dispatch_ms | leaf | False | 239 | 0.392 | 0.394 | 0.709 | 1.418 |
| sam_prepare_ms | leaf | False | 239 | 0.375 | 0.208 | 1.242 | 4.881 |
| callback_enqueue_delay_ms | leaf | True | 239 | 0.169 | 0.162 | 0.216 | 0.331 |
| metadata_delay_ms | leaf | True | 239 | 0.150 | 0.142 | 0.191 | 0.712 |
| hydra_metadata_build_ms | leaf | True | 239 | 0.099 | 0.095 | 0.118 | 0.246 |
| classifier_other_ms | leaf | True | 239 | 0.045 | 0.044 | 0.055 | 0.072 |
| coordinator_other_ms | leaf | True | 239 | 0.033 | 0.005 | 0.007 | 4.888 |
| sam_other_ms | leaf | False | 239 | 0.007 | 0.007 | 0.009 | 0.011 |
| classifier_debug_record_delay_ms | leaf | False | 239 | 0.005 | 0.005 | 0.007 | 0.010 |
| quality_deferred_release_ms | leaf | False | 239 | 0.005 | 0.005 | 0.006 | 0.009 |
