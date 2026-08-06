# Phase 1 optimisation Part 2 timing summary

Complete frame traces: 308; drops: 2587; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 102.888 ms, p95 172.249 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 308 | 603.371 | 596.320 | 770.743 | 1403.228 |
| classifier_delay_ms | aggregate | False | 308 | 539.546 | 543.376 | 649.224 | 932.017 |
| sam_delay_ms | aggregate | False | 308 | 323.263 | 320.274 | 411.484 | 484.052 |
| sam_inference_ms | leaf | False | 308 | 302.563 | 299.526 | 390.083 | 463.647 |
| rap_delay_ms | aggregate | False | 308 | 176.356 | 171.342 | 261.734 | 567.724 |
| geometry_metadata_ms | leaf | True | 308 | 102.888 | 95.901 | 172.249 | 257.407 |
| frame_assignment_ms | leaf | True | 308 | 65.766 | 59.073 | 108.124 | 476.368 |
| sent_to_classifier_delay_ms | aggregate | False | 308 | 48.891 | 34.713 | 127.465 | 571.436 |
| frame_queue_wait_ms | leaf | True | 308 | 48.739 | 34.585 | 127.330 | 571.307 |
| geometry_projection_ms | leaf | False | 308 | 42.296 | 37.204 | 97.842 | 165.030 |
| result_message_build_delay_ms | leaf | True | 308 | 31.372 | 30.676 | 59.191 | 80.890 |
| geometry_stats_ms | leaf | False | 308 | 28.276 | 24.501 | 52.813 | 90.215 |
| geometry_mask_extract_ms | leaf | False | 308 | 24.496 | 21.501 | 41.521 | 99.838 |
| sam_restore_ms | leaf | False | 308 | 20.288 | 19.892 | 28.336 | 34.062 |
| coordinator_delay_ms | aggregate | False | 308 | 12.664 | 12.504 | 18.248 | 22.608 |
| hydra_build_delay_ms | aggregate | False | 308 | 7.889 | 7.869 | 12.370 | 16.162 |
| hydra_depth_filter_ms | leaf | True | 308 | 7.207 | 7.197 | 11.560 | 15.494 |
| geometry_depth_gather_ms | leaf | False | 308 | 6.975 | 5.187 | 16.872 | 38.908 |
| label_map_delay_ms | leaf | True | 308 | 5.357 | 4.694 | 11.597 | 27.730 |
| track_association_ms | leaf | True | 308 | 4.075 | 3.690 | 7.006 | 23.452 |
| hydra_publish_delay_ms | leaf | True | 308 | 3.976 | 3.676 | 5.828 | 10.286 |
| image_conversion_delay_ms | leaf | True | 308 | 2.994 | 2.006 | 6.948 | 13.696 |
| pipeline_wait_ms | leaf | False | 308 | 2.268 | 2.183 | 4.350 | 9.866 |
| active_segments_publish_ms | leaf | True | 308 | 1.423 | 1.385 | 1.820 | 3.459 |
| crop_update_ms | leaf | True | 308 | 1.402 | 0.932 | 5.049 | 9.347 |
| unknown_publish_delay_ms | leaf | True | 308 | 0.783 | 0.717 | 1.013 | 4.200 |
| hydra_build_other_ms | leaf | True | 308 | 0.583 | 0.531 | 0.649 | 11.538 |
| run_rap_other_ms | leaf | True | 308 | 0.412 | 0.338 | 0.451 | 16.170 |
| sam_prepare_ms | leaf | False | 308 | 0.403 | 0.211 | 1.091 | 6.371 |
| semantic_dispatch_ms | leaf | False | 308 | 0.368 | 0.369 | 0.722 | 3.854 |
| metadata_delay_ms | leaf | True | 308 | 0.158 | 0.143 | 0.199 | 1.535 |
| callback_enqueue_delay_ms | leaf | True | 308 | 0.152 | 0.144 | 0.193 | 0.364 |
| hydra_metadata_build_ms | leaf | True | 308 | 0.100 | 0.096 | 0.132 | 0.214 |
| classifier_other_ms | leaf | True | 308 | 0.047 | 0.044 | 0.054 | 0.534 |
| coordinator_other_ms | leaf | True | 308 | 0.016 | 0.005 | 0.007 | 3.164 |
| sam_other_ms | leaf | False | 308 | 0.009 | 0.007 | 0.009 | 0.175 |
| quality_deferred_release_ms | leaf | False | 308 | 0.005 | 0.005 | 0.007 | 0.024 |
| classifier_debug_record_delay_ms | leaf | False | 308 | 0.003 | 0.003 | 0.006 | 0.007 |
