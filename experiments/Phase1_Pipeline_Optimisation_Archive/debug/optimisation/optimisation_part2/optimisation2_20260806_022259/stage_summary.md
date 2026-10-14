# Phase 1 optimisation Part 2 timing summary

Complete frame traces: 294; drops: 2635; failures: 0.

Asynchronous RAP/VLM inference and retrieval are outside this analysis.

Largest eligible synchronous leaf stage: **geometry_metadata_ms** (mean 87.017 ms, p95 148.585 ms).

| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| total_delay_ms | aggregate | False | 294 | 626.940 | 612.556 | 800.173 | 1203.036 |
| classifier_delay_ms | aggregate | False | 294 | 569.159 | 563.537 | 719.875 | 1031.938 |
| sam_delay_ms | aggregate | False | 294 | 357.926 | 358.685 | 439.350 | 519.326 |
| sam_inference_ms | leaf | False | 294 | 337.122 | 338.421 | 415.430 | 500.102 |
| rap_delay_ms | aggregate | False | 294 | 168.327 | 159.073 | 249.715 | 558.799 |
| geometry_metadata_ms | leaf | True | 294 | 87.017 | 82.687 | 148.585 | 195.524 |
| frame_assignment_ms | leaf | True | 294 | 73.183 | 66.064 | 114.123 | 450.660 |
| sent_to_classifier_delay_ms | aggregate | False | 294 | 42.315 | 36.042 | 111.487 | 618.277 |
| frame_queue_wait_ms | leaf | True | 294 | 42.166 | 35.880 | 111.356 | 618.195 |
| geometry_projection_ms | leaf | False | 294 | 36.690 | 32.235 | 80.139 | 118.609 |
| result_message_build_delay_ms | leaf | True | 294 | 33.888 | 33.759 | 62.516 | 87.094 |
| geometry_mask_extract_ms | leaf | False | 294 | 24.135 | 21.329 | 40.803 | 63.663 |
| sam_restore_ms | leaf | False | 294 | 20.413 | 19.991 | 28.207 | 35.425 |
| geometry_stats_ms | leaf | False | 294 | 18.467 | 16.514 | 33.465 | 52.132 |
| coordinator_delay_ms | aggregate | False | 294 | 13.014 | 12.856 | 18.392 | 28.624 |
| hydra_build_delay_ms | aggregate | False | 294 | 7.945 | 7.794 | 11.892 | 24.516 |
| hydra_depth_filter_ms | leaf | True | 294 | 7.246 | 7.143 | 11.128 | 23.780 |
| geometry_depth_gather_ms | leaf | False | 294 | 6.860 | 5.411 | 16.844 | 30.897 |
| label_map_delay_ms | leaf | True | 294 | 5.638 | 4.964 | 11.904 | 26.910 |
| hydra_publish_delay_ms | leaf | True | 294 | 4.219 | 3.808 | 6.730 | 14.397 |
| track_association_ms | leaf | True | 294 | 4.188 | 3.748 | 7.504 | 17.543 |
| image_conversion_delay_ms | leaf | True | 294 | 3.183 | 2.229 | 7.047 | 13.538 |
| pipeline_wait_ms | leaf | False | 294 | 2.435 | 2.450 | 4.393 | 10.737 |
| crop_update_ms | leaf | True | 294 | 1.558 | 1.085 | 4.834 | 8.781 |
| active_segments_publish_ms | leaf | True | 294 | 1.525 | 1.377 | 1.890 | 13.420 |
| unknown_publish_delay_ms | leaf | True | 294 | 0.846 | 0.703 | 1.115 | 8.294 |
| hydra_build_other_ms | leaf | True | 294 | 0.596 | 0.549 | 0.697 | 5.078 |
| semantic_dispatch_ms | leaf | False | 294 | 0.447 | 0.365 | 0.798 | 15.206 |
| sam_prepare_ms | leaf | False | 294 | 0.383 | 0.213 | 1.067 | 5.083 |
| run_rap_other_ms | leaf | True | 294 | 0.377 | 0.349 | 0.466 | 4.299 |
| metadata_delay_ms | leaf | True | 294 | 0.150 | 0.146 | 0.190 | 0.400 |
| callback_enqueue_delay_ms | leaf | True | 294 | 0.149 | 0.141 | 0.198 | 0.357 |
| hydra_metadata_build_ms | leaf | True | 294 | 0.102 | 0.098 | 0.131 | 0.231 |
| classifier_other_ms | leaf | True | 294 | 0.047 | 0.046 | 0.059 | 0.090 |
| classifier_debug_record_delay_ms | leaf | False | 294 | 0.016 | 0.003 | 0.006 | 3.761 |
| quality_deferred_release_ms | leaf | False | 294 | 0.011 | 0.006 | 0.008 | 1.627 |
| sam_other_ms | leaf | False | 294 | 0.008 | 0.007 | 0.009 | 0.082 |
| coordinator_other_ms | leaf | True | 294 | 0.005 | 0.005 | 0.007 | 0.009 |
