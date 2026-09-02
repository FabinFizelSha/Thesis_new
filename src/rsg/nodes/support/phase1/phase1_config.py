"""Configuration loader for Phase 1 RSG nodes.

The Phase 1 nodes use the same central ``rsg_pipeline.yaml`` file as the
preprocessor.  This keeps runtime switches, topic names, debug controls, and
performance settings in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _as_bool(value: Any, default: bool = False) -> bool:
    """Return a robust bool from YAML values."""
    if value is None:
        return default
    return bool(value)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base and return base."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass(kw_only=True)
class Phase1Config:
    """Configuration values shared by ``rsg_object_detection`` and classifier."""

    node_key: str
    use_sim_time: bool
    profile: str
    allow_dummy_fallback: bool

    # Topics.
    preprocessed_frame_topic: str
    perception_request_topic: str
    vlm_result_topic: str
    rap_result_topic: str
    semantic_label_result_topic: str
    track_observations_topic: str
    hydra_frame_topic: str
    status_topic: str
    timing_topic: str
    unknown_candidates_topic: str
    hydra_rgb_topic: str
    hydra_depth_topic: str
    hydra_camera_info_topic: str
    hydra_pose_topic: str
    hydra_semantic_topic: str
    hydra_instance_topic: str
    hydra_metadata_topic: str

    # QoS / queueing.
    input_qos_depth: int
    output_qos_depth: int
    semantic_label_qos_depth: int
    request_queue_size: int
    classifier_queue_size: int
    frame_cache_size: int
    max_result_age_sec: float
    drop_oldest_when_full: bool

    # Debug / timing.
    timing_measurement_enabled: bool
    publish_timing_topic: bool
    write_timing_csv: bool
    timing_csv_path: str
    timing_sheet_name: str
    timing_excel_autosave_every: int
    publish_status: bool
    status_every_n_frames: int

    # Master switch for Phase 1's per-run diagnostic writers: bbox dumps, crop
    # evolution, tracking-quality logs, RAP/VLM crop images, and the VLM test
    # crop+CSV.  Off = nothing written, near-zero overhead.  Functional crop
    # scoring / mask filtering / best-crop selection is unaffected.
    diagnostics_enabled: bool = False

    # Hydra output.
    publish_hydra_combined: bool
    publish_hydra_separate_topics: bool
    semantic_label_encoding: str
    instance_label_encoding: str
    publish_hydra_tf: bool
    # Output-side range gate. Semantic/instance ID 0 is written only for
    # consistency; invalid depth is what blocks mesh integration.
    hydra_depth_range_filter_enabled: bool
    hydra_depth_min_range_m: float
    hydra_depth_max_range_m: float
    hydra_far_range_semantic_label_id: int

    # Metadata switches. Turning these off reduces JSON size and processing.
    include_object_metadata: bool
    include_known_objects: bool
    include_unknown_objects: bool
    include_bbox_2d: bool
    include_centroid_2d: bool
    include_centroid_3d: bool
    include_bbox_3d: bool
    include_bbox_volume: bool
    include_depth_stats: bool
    include_mask_area: bool
    include_timing_metadata: bool
    include_frame_relation_metadata: bool

    # SAM / RAP / VLM settings.
    sam_enabled: bool
    sam_backend: str
    sam_min_mask_pixels: int
    sam_max_masks: int
    sam_dummy_num_masks: int
    sam_model_type: str = "vit_h"
    sam_checkpoint_path: str = ""
    sam_checkpoint_cache_dir: str = "~/.cache/sam"
    sam_device: str = ""
    sam_mask_threshold: float = 0.88
    sam_padding: float = 0.1
    sam_points_per_side: int = 16
    sam_pred_iou_thresh: float = 0.96
    sam_auto_download: bool = False
    sam_input_scale_ratio: float = 1.0
    sam_resize_interpolation: str = "area"
    sam_depth_filter_enabled: bool = False
    sam_depth_filter_min_m: float = 0.0
    sam_depth_filter_max_m: float = 0.0
    sam_depth_filter_min_valid_ratio: float = 0.0
    sam_depth_filter_crop_to_roi: bool = True
    sam_depth_filter_roi_margin_px: int = 0
    sam_depth_filter_background_value: Tuple[int, int, int] = (0, 0, 0)
    sam_nanosam_image_encoder_engine: str = ""
    sam_nanosam_mask_decoder_engine: str = ""
    sam_nanosam_points_per_side: int = 3
    sam_nanosam_mask_threshold: float = 0.0
    sam_nanosam_nms_iou: float = 0.70
    sam_nanosam_point_margin_fraction: float = 0.08
    sam_nanosam_skip_black_points: bool = True
    sam_nanosam_min_prompt_luma: int = 4

    rap_enabled: bool
    rap_backend: str
    rap_confidence_threshold: float
    rap_distance_threshold: float
    rap_dummy_unknown_every_n: int
    rap_dummy_known_labels: List[str] = field(default_factory=list)
    rap_update_enabled: bool = True
    rap_update_min_confidence: float = 0.50
    rap_memory_path: str = "~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl"
    rap_model_name: str = "openai/clip-vit-base-patch32"
    rap_device: str = ""
    rap_storage_path: str = "~/rsg_ros2_ws/visual_memory"
    rap_chroma_host: str = "localhost"
    rap_chroma_port: int = 8001
    rap_collection_name: str = "visual_rag"
    rap_auto_start_server: bool = True
    # RAP stores only track IDs in a bounded worker queue.  A deferred-ID
    # registry preserves unique unresolved tracks when the FIFO is full.
    rap_async: bool = True
    rap_queue_size: int = 300
    rap_queue_drop_policy: str = "defer_track_id"

    # Shared mask-aware semantic crop representations. RAP queries and RAP
    # memory use a target-only crop. VLM keeps orientation context, but that
    # context is grayscale and strongly dimmed so it cannot dominate the target.
    semantic_crop_rap_target_only_enabled: bool = True
    semantic_crop_rap_background_rgb: Tuple[int, int, int] = (32, 32, 32)
    semantic_crop_vlm_target_focus_enabled: bool = True
    semantic_crop_vlm_context_alpha: float = 0.10
    semantic_crop_vlm_context_grayscale: bool = True
    semantic_crop_vlm_near_context_enabled: bool = True
    semantic_crop_vlm_near_context_alpha: float = 0.45
    semantic_crop_vlm_near_context_dilation_px: int = 15
    semantic_crop_vlm_near_context_grayscale: bool = False
    semantic_crop_mask_cleanup_enabled: bool = True
    semantic_crop_mask_cleanup_min_component_area_ratio: float = 0.02
    semantic_crop_mask_cleanup_component_max_gap_px: int = 15
    semantic_crop_draw_target_contour: bool = True
    semantic_crop_target_contour_rgb: Tuple[int, int, int] = (255, 255, 255)
    semantic_crop_target_contour_thickness_px: int = 2

    # Semantic labelling is decoupled from the SAM-to-Hydra path. A track keeps
    # updating its representative crop during the fixed settling window, then
    # RAP/VLM receive only its track ID.
    semantic_labeling_enabled: bool = True
    semantic_labeling_min_observations: int = 1
    semantic_labeling_settle_time_sec: float = 3.0
    semantic_labeling_force_dispatch_on_shutdown: bool = True
    semantic_labeling_publish_topic: str = "/rsg/objects/semantic_label_result"
    semantic_labeling_shutdown_wait_sec: float = 8.0

    vlm_enabled: bool = False
    vlm_active_profile: str = ""
    vlm_mode: str = "dummy"
    vlm_async: bool = True
    # VLM also stores track IDs only in normal RAP-enabled operation. A
    # deferred registry avoids discarding unresolved tracks under backpressure.
    vlm_queue_size: int = 300
    vlm_queue_drop_policy: str = "defer_track_id"
    vlm_dummy_delay_sec: float = 0.0
    vlm_dummy_label_prefix: str = "vlm_dummy_object"
    vlm_confidence: float = 0.55
    vlm_endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    vlm_model: str = "Qwen2.5-VL-7B-Instruct"
    vlm_timeout_sec: float = 30.0
    vlm_api_key: str = ""
    vlm_max_tokens: int = 32
    vlm_temperature: float = 0.0
    vlm_jpeg_quality: int = 85
    vlm_prompt: str = "Identify the main object. Reply with one lowercase noun only. If unclear, reply unknown_object."
    vlm_prompt_opt_enabled: bool = False
    vlm_prompt_opt_run_id: str = ""
    vlm_prompt_opt_prompt_version: str = ""
    vlm_prompt_opt_output_root: str = ""
    vlm_result_min_label_confidence: float = 0.35
    vlm_result_min_mobility_confidence: float = 0.50
    vlm_dynamic_label_hints: List[str] = field(default_factory=list)
    vlm_static_label_hints: List[str] = field(default_factory=list)
    # VLM crop selection is kept separate from the semantic settling interval:
    # a track becomes eligible after settling, but a weak crop may wait for a
    # later observation rather than consuming one irreversible VLM request.
    vlm_crop_context_ratio: float = 0.15
    vlm_crop_min_area_px: int = 12000
    vlm_crop_min_short_side_px: int = 64
    vlm_crop_min_quality_score: float = 0.34
    vlm_crop_target_area_px: int = 30000
    vlm_crop_target_short_side_px: int = 120
    vlm_crop_border_penalty: float = 0.18
    # A weak crop can remain mutable after RAP-unknown, but this bounded delay
    # guarantees that a small or partly visible real object still reaches VLM.
    vlm_crop_quality_max_wait_sec: float = 2.0
    vlm_crop_quality_force_on_timeout: bool = True

    # Projection / object geometry.
    estimate_object_geometry: bool = True
    projection_stride: int = 4
    min_valid_depth_points: int = 20
    min_depth_m: float = 0.2
    max_depth_m: float = 6.0
    centroid_method: str = "median"

    # Evidence buffer for future risk annotation.
    store_evidence_frames: bool = True
    evidence_buffer_size: int = 50

    # Persistent unknown-object tracking. This prevents repeated VLM calls for
    # the same physical unknown object across consecutive frames.
    unknown_tracking_enabled: bool = True
    unknown_max_match_distance_m: float = 0.30
    unknown_max_volume_ratio: float = 3.0
    unknown_max_track_age_sec: float = 2.0
    unknown_min_observations_before_vlm: int = 3
    unknown_max_wait_before_vlm_sec: float = 0.75
    unknown_call_vlm_only_once_per_track: bool = True
    unknown_retry_failed_tracks: bool = False
    unknown_update_track_centroid: bool = True
    unknown_centroid_update_alpha: float = 0.7
    unknown_best_frame_selection_enabled: bool = True
    unknown_min_quality_for_vlm: float = 0.0
    unknown_max_tracks: int = 200
    unknown_use_2d_iou_fallback: bool = True
    unknown_min_2d_iou: float = 0.30

    # Fixed Hydra semantic ontology and persistent RSG object identity.
    # The semantic lookup must match the Hydra label-space YAML loaded at
    # startup. Persistent IDs are carried by the separate instance map and
    # object metadata, never by dynamically created semantic classes.
    hydra_label_lookup: Dict[str, int] = field(default_factory=dict)
    hydra_label_names: Dict[int, str] = field(default_factory=dict)
    hydra_unclassified_label_id: int = 0
    hydra_unclassified_label_name: str = "unknown"

    rap_execution_mode: str = "auto"  # auto | async | synchronous

    persistent_tracking_enabled: bool = False
    persistent_track_prefix: str = "rsg_obj_"
    persistent_max_tracks: int = 1024
    persistent_max_match_distance_m: float = 0.30
    persistent_max_volume_ratio: float = 3.0
    persistent_continuation_max_age_sec: float = 2.5
    persistent_continuation_gap_m: float = 0.18
    persistent_revisit_overlap_gap_m: float = 0.05
    persistent_max_vertical_gap_m: float = 0.12
    persistent_max_vertical_center_delta_m: float = 0.12
    persistent_max_2d_iou_age_sec: float = 1.0
    persistent_min_2d_iou: float = 0.30
    persistent_centroid_update_alpha: float = 0.7
    persistent_track_aware_redundancy_enabled: bool = True
    persistent_redundancy_union_coverage_threshold: float = 0.90
    persistent_redundancy_child_containment_threshold: float = 0.85
    persistent_redundancy_min_children: int = 2
    persistent_same_track_nested_suppression_enabled: bool = True
    persistent_same_track_nested_containment_threshold: float = 0.90
    persistent_same_track_max_parent_child_area_ratio: float = 3.0
    persistent_same_track_min_added_area_fraction: float = 0.05
    persistent_same_track_broader_max_route_priority: int = 2
    persistent_same_track_broader_max_volume_ratio: float = 6.0
    persistent_global_association_enabled: bool = True
    persistent_global_min_independent_groups: int = 2
    persistent_global_recent_min_score: float = 0.55
    persistent_global_revisit_min_score: float = 0.70
    persistent_global_historical_overlap_pass: float = 0.30
    persistent_global_recent_overlap_pass: float = 0.25
    persistent_global_min_axis_overlap: float = 0.20
    persistent_global_touch_gap_pass_m: float = 0.02
    persistent_global_centroid_pass_m: float = 0.75
    persistent_global_centroid_sigma_m: float = 0.50
    persistent_global_vertical_score_pass: float = 0.60
    persistent_global_vertical_sigma_m: float = 0.15
    persistent_global_block_2d_on_3d_contradiction: bool = True
    persistent_global_recent_weight_historical: float = 0.20
    persistent_global_recent_weight_recent: float = 0.30
    persistent_global_recent_weight_centroid: float = 0.25
    persistent_global_recent_weight_vertical: float = 0.15
    persistent_global_recent_weight_image: float = 0.10
    persistent_global_revisit_weight_historical: float = 0.45
    persistent_global_revisit_weight_recent: float = 0.00
    persistent_global_revisit_weight_centroid: float = 0.30
    persistent_global_revisit_weight_vertical: float = 0.20
    persistent_global_revisit_weight_image: float = 0.05
    persistent_global_min_depth_z: float = 0.30
    persistent_global_min_depth_xy: float = 0.15
    persistent_local_segments_enabled: bool = True
    persistent_local_segment_max_xy_span_m: float = 4.0
    persistent_local_segment_revisit_distance_m: float = 1.5
    persistent_local_segment_gap_m: float = 0.20
    persistent_local_segment_2d_fallback_enabled: bool = True
    persistent_local_segment_max_2d_iou_age_sec: float = 2.0
    persistent_local_segment_min_2d_iou: float = 0.30
    persistent_active_segments_topic: str = "/rsg/objects/active_local_segments"
    persistent_require_known_label_match: bool = True
    persistent_unclassified_label_id: int = 0

    # Loop-closure re-anchoring.  When the SLAM back-end folds an accumulated
    # drift correction into ``map -> odom``, phase 1 rigid-transforms every
    # cached track/segment by the same step so a re-observed object
    # re-associates instead of spawning a duplicate.  Disabled by default; a
    # front end that publishes a non-identity ``map -> odom`` (LCD on, distinct
    # map/odom frames) is required for it to do anything.
    loop_closure_enabled: bool = False
    loop_closure_map_frame: str = "map"
    loop_closure_odom_frame: str = "odom"
    loop_closure_min_translation_m: float = 0.05
    loop_closure_min_rotation_deg: float = 0.5
    loop_closure_merge_duplicates: bool = True
    loop_closure_merge_recent_window_sec: float = 5.0
    loop_closure_merge_distance_slack_m: float = 0.6
    loop_closure_event_topic: str = "/rsg/phase1/loop_closure_event"

    # Optional fixed Hydra-slot mode. Every new physical object receives a
    # predeclared semantic slot ID, stable for the full mapping session.
    persistent_use_hydra_slots: bool = False
    persistent_slot_first_label_id: int = 21
    persistent_slot_count: int = 10000
    persistent_slot_label_prefix: str = "unknown_slot_"
    persistent_slot_label_width: int = 5

    # Immediate semantic-result evidence thresholds. These do not impose an
    # inactivity delay; they only govern how RAP/VLM evidence is accepted.
    semantic_result_min_observations: int = 1
    semantic_result_min_consensus: float = 0.0
    semantic_result_min_evidence: float = 0.0
    persistent_rap_evidence_weight: float = 0.70
    persistent_vlm_evidence_weight: float = 1.00
    persistent_label_aliases: Dict[str, str] = field(default_factory=dict)

    @property
    def timing_enabled(self) -> bool:
        """Return whether timing measurement is enabled for this node."""
        return self.timing_measurement_enabled

    @staticmethod
    def from_yaml(path: str, node_key: str) -> "Phase1Config":
        """Load Phase 1 configuration from the central YAML file."""
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as stream:
            root = yaml.safe_load(stream) or {}

        preprocessing = root.get("preprocessing", {}) or {}
        phase1 = root.get("phase1", {}) or {}

        # Master profile switch.  To switch the complete Phase 1 perception
        # stack, edit only ``phase1.profile`` in YAML.  The selected profile
        # is merged into phase1 before individual keys are read.  Explicit keys
        # outside the profile can still override profile defaults when needed.
        profile_name = str(phase1.get("profile", "dummy"))
        profiles = phase1.get("profiles", {}) or {}
        if profile_name in profiles:
            selected_profile = dict(profiles.get(profile_name) or {})
            # Keep the profile name itself and the profiles table; merge selected
            # values into a copy to avoid mutating the root structure.
            base_phase1 = {k: v for k, v in phase1.items() if k != "profiles"}
            phase1 = _deep_update(base_phase1, selected_profile)
            phase1["profile"] = profile_name
        else:
            phase1 = dict(phase1)
            phase1["profile"] = profile_name

        runtime = phase1.get("runtime", {}) or {}
        topics = phase1.get("topics", {}) or {}
        qos = phase1.get("qos", {}) or {}
        coordinator = phase1.get("coordinator", {}) or {}
        classifier = phase1.get("object_classifier", {}) or {}
        hydra = phase1.get("hydra_output", {}) or {}
        metadata = phase1.get("metadata", {}) or {}
        sam = phase1.get("sam", {}) or {}
        rap = phase1.get("rap", {}) or {}
        semantic_crop = phase1.get("semantic_crop", {}) or {}
        vlm = phase1.get("vlm", {}) or {}
        # VLM profile selection is intentionally independent from the broader
        # Phase 1 profile. The selected profile overrides only VLM endpoint,
        # model, and server settings; crop scheduling remains shared so model
        # comparisons use identical inputs.
        active_vlm_profile = str(vlm.get("active_profile", "")).strip()
        vlm_profiles = vlm.get("profiles", {}) or {}
        if active_vlm_profile:
            if active_vlm_profile not in vlm_profiles:
                raise ValueError(
                    f"Unknown phase1.vlm.active_profile='{active_vlm_profile}'. "
                    f"Available profiles: {sorted(vlm_profiles)}"
                )
            base_vlm = {
                key: value
                for key, value in vlm.items()
                if key not in {"active_profile", "profiles"}
            }
            vlm = _deep_update(base_vlm, dict(vlm_profiles[active_vlm_profile] or {}))
            vlm["active_profile"] = active_vlm_profile
        else:
            vlm = dict(vlm)
        geometry = phase1.get("object_geometry", {}) or {}
        evidence = phase1.get("evidence_buffer", {}) or {}
        semantic_labeling = phase1.get("semantic_labeling", {}) or {}
        unknown_tracking = phase1.get("unknown_tracking", {}) or {}
        hydra_semantic = phase1.get("hydra_semantic", {}) or {}
        persistent_tracking = phase1.get("persistent_tracking", {}) or {}
        persistent_slots = persistent_tracking.get("slots", {}) or {}
        semantic_result = persistent_tracking.get("semantic_result", {}) or {}
        loop_closure = phase1.get("loop_closure", {}) or {}
        diagnostics = phase1.get("diagnostics", {}) or {}
        deployment = phase1.get("deployment", {}) or {}
        performance = phase1.get("performance", {}) or {}

        # Slot-only semantics: no built-in object classes are emitted by Phase 1.
        default_hydra_label_lookup = {
            "unknown": 0,
            "unknown object": 0,
            "unknown_object": 0,
        }
        raw_lookup = hydra_semantic.get("label_lookup", default_hydra_label_lookup) or {}
        hydra_label_lookup = {
            " ".join(str(key).strip().lower().replace("_", " ").split()): int(value)
            for key, value in raw_lookup.items()
        }
        raw_names = hydra_semantic.get("label_names", {}) or {}
        hydra_label_names = {int(key): str(value) for key, value in raw_names.items()}
        if not hydra_label_names:
            for name, label_id in hydra_label_lookup.items():
                hydra_label_names.setdefault(int(label_id), name.replace(" ", "_"))
        unclassified_name = str(hydra_semantic.get("unclassified_label", "unknown_object"))
        unclassified_key = " ".join(unclassified_name.strip().lower().replace("_", " ").split())
        unclassified_id = int(hydra_label_lookup.get(unclassified_key, hydra_label_lookup.get("unknown object", 0)))
        raw_aliases = persistent_tracking.get("label_aliases", hydra_semantic.get("label_aliases", {})) or {}
        persistent_label_aliases = {
            " ".join(str(key).strip().lower().replace("_", " ").split()):
            " ".join(str(value).strip().lower().replace("_", " ").split())
            for key, value in raw_aliases.items()
        }
        slot_mode = str(persistent_tracking.get("mode", "instance_only")).strip().lower() == "hydra_slots"
        slot_first_id = int(persistent_slots.get("first_label_id", persistent_tracking.get("slot_first_label_id", 1)))
        slot_count = int(persistent_slots.get("count", persistent_tracking.get("slot_count", persistent_tracking.get("max_tracks", 1024))))
        slot_prefix = str(persistent_slots.get("label_prefix", persistent_tracking.get("slot_label_prefix", "unknown_slot_")))
        slot_width = int(persistent_slots.get("label_width", persistent_tracking.get("slot_label_width", 5)))
        rap_execution_mode = str(rap.get("execution_mode", "auto")).strip().lower()
        if rap_execution_mode not in {"auto", "async", "synchronous"}:
            raise ValueError("phase1.rap.execution_mode must be auto, async, or synchronous")

        creation_time = datetime.now().strftime("%H%M%S")
        session_date = datetime.now().strftime("%Y%m%d")
        timing_csv_path = str(
            performance.get(
                "timing_csv_path",
                performance.get(
                    "timing_excel_path",
                    f"~/rsg_ros2_ws/debug/{node_key}_debug_{{session_date}}_{{creation_time}}.csv",
                ),
            )
        ).format(session_date=session_date, creation_time=creation_time, node_name=node_key)
        preproc_topics = preprocessing.get("topics", {}) or {}
        preproc_runtime = preprocessing.get("runtime", {}) or {}
        preproc_image = preprocessing.get("image", {}) or {}

        config = Phase1Config(
            node_key=node_key,
            use_sim_time=bool(runtime.get("use_sim_time", preproc_runtime.get("use_sim_time", True))),
            profile=str(phase1.get("profile", "dummy")),
            allow_dummy_fallback=bool(deployment.get("allow_dummy_fallback", False)),
            preprocessed_frame_topic=str(topics.get("preprocessed_frame", preproc_topics.get("prepared_frame", "/rsg/preprocessed/frame"))),
            perception_request_topic=str(topics.get("perception_request", "/rsg/phase1/object_classifier/input")),
            vlm_result_topic=str(topics.get("vlm_result", "/rsg/phase1/object_classifier/vlm_result")),
            rap_result_topic=str(topics.get("rap_result", "/rsg/objects/rap_result")),
            semantic_label_result_topic=str(topics.get("semantic_label_result", "/rsg/objects/semantic_label_result")),
            track_observations_topic=str(topics.get("track_observations", "/rsg/objects/track_observations")),
            hydra_frame_topic=str(topics.get("hydra_frame", "/rsg/phase1/hydra/input_frame")),
            status_topic=str(topics.get(f"{node_key}_status", f"/rsg/phase1/{node_key}/status")),
            timing_topic=str(topics.get(f"{node_key}_timing", f"/rsg/phase1/{node_key}/timing")),
            unknown_candidates_topic=str(topics.get("unknown_candidates", "/rsg/phase1/unknown_candidates")),
            hydra_rgb_topic=str(topics.get("hydra_rgb", "/rsg/hydra/rgb")),
            hydra_depth_topic=str(topics.get("hydra_depth", "/rsg/hydra/depth")),
            hydra_camera_info_topic=str(topics.get("hydra_camera_info", "/rsg/hydra/camera_info")),
            hydra_pose_topic=str(topics.get("hydra_pose", "/rsg/hydra/pose")),
            hydra_semantic_topic=str(topics.get("hydra_semantic_labels", "/rsg/hydra/semantic_labels")),
            hydra_instance_topic=str(topics.get("hydra_instance_labels", "/rsg/hydra/instance_labels")),
            hydra_metadata_topic=str(topics.get("hydra_metadata", "/rsg/hydra/metadata")),
            input_qos_depth=max(1, int(qos.get("input_depth", 10))),
            output_qos_depth=max(1, int(qos.get("output_depth", 10))),
            # Final semantic events must survive while the C++ fuser batches
            # expensive DSG redraws. This is separate from high-bandwidth image
            # outputs, which keep the small normal output queue.
            semantic_label_qos_depth=max(1, int(qos.get("semantic_label_depth", 4096))),
            request_queue_size=max(1, int(coordinator.get("request_queue_size", 2))),
            classifier_queue_size=max(1, int(classifier.get("queue_size", 1))),
            frame_cache_size=max(1, int(coordinator.get("frame_cache_size", 75))),
            max_result_age_sec=float(coordinator.get("max_result_age_sec", 1.0)),
            drop_oldest_when_full=bool(coordinator.get("drop_oldest_when_full", True)),
            timing_measurement_enabled=bool(performance.get("measure_timing", True)),
            publish_timing_topic=bool(performance.get("publish_timing", True)),
            write_timing_csv=bool(performance.get("write_timing_csv", performance.get("write_timing_excel", True))),
            diagnostics_enabled=bool(diagnostics.get("enabled", False)),
            timing_csv_path=timing_csv_path,
            timing_sheet_name=str(performance.get("timing_sheet_name", node_key[:31])),
            timing_excel_autosave_every=int(performance.get("timing_excel_autosave_every", 0)),
            publish_status=bool(coordinator.get("publish_status", True)),
            status_every_n_frames=max(1, int(coordinator.get("status_every_n_frames", 30))),
            publish_hydra_combined=bool(hydra.get("publish_combined", True)),
            publish_hydra_separate_topics=bool(hydra.get("publish_separate_topics", False)),
            semantic_label_encoding=str(hydra.get("semantic_label_encoding", "16UC1")),
            instance_label_encoding=str(hydra.get("instance_label_encoding", "16UC1")),
            publish_hydra_tf=bool(hydra.get("publish_camera_tf", True)),
            hydra_depth_range_filter_enabled=bool((hydra.get("depth_range_filter", {}) or {}).get("enabled", True)),
            hydra_depth_min_range_m=float((hydra.get("depth_range_filter", {}) or {}).get("min_range_m", preproc_image.get("min_depth_m", 0.2))),
            hydra_depth_max_range_m=float((hydra.get("depth_range_filter", {}) or {}).get("max_range_m", preproc_image.get("max_depth_m", 6.0))),
            hydra_far_range_semantic_label_id=int((hydra.get("depth_range_filter", {}) or {}).get("semantic_label_id", 0)),
            include_object_metadata=bool(metadata.get("include_object_metadata", True)),
            include_known_objects=bool(metadata.get("include_known_objects", True)),
            include_unknown_objects=bool(metadata.get("include_unknown_objects", True)),
            include_bbox_2d=bool(metadata.get("include_bbox_2d", True)),
            include_centroid_2d=bool(metadata.get("include_centroid_2d", True)),
            include_centroid_3d=bool(metadata.get("include_centroid_3d", True)),
            include_bbox_3d=bool(metadata.get("include_bbox_3d", True)),
            include_bbox_volume=bool(metadata.get("include_bbox_volume", True)),
            include_depth_stats=bool(metadata.get("include_depth_stats", True)),
            include_mask_area=bool(metadata.get("include_mask_area", True)),
            include_timing_metadata=bool(metadata.get("include_timing_metadata", True)),
            include_frame_relation_metadata=bool(metadata.get("include_frame_relation_metadata", True)),
            sam_enabled=bool(sam.get("enabled", True)),
            sam_backend=str(sam.get("backend", "dummy")),
            sam_min_mask_pixels=max(1, int(sam.get("min_mask_pixels", 500))),
            sam_max_masks=max(1, int(sam.get("max_masks", 64))),
            sam_dummy_num_masks=max(0, int(sam.get("dummy_num_masks", 1))),
            sam_model_type=str(sam.get("model_type", "vit_h")),
            sam_checkpoint_path=str(sam.get("checkpoint_path", "")),
            sam_checkpoint_cache_dir=str(sam.get("checkpoint_cache_dir", "~/.cache/sam")),
            sam_device=str(sam.get("device", "")),
            sam_mask_threshold=float(sam.get("mask_threshold", sam.get("stability_score_thresh", 0.88))),
            sam_padding=float(sam.get("padding", 0.1)),
            sam_points_per_side=max(1, int(sam.get("points_per_side", 16))),
            sam_pred_iou_thresh=float(sam.get("pred_iou_thresh", 0.96)),
            sam_auto_download=bool(sam.get("auto_download", False)),
            sam_input_scale_ratio=max(0.05, min(1.0, float(sam.get("input_scale_ratio", sam.get("resize_ratio", 1.0))))),
            sam_resize_interpolation=str(sam.get("resize_interpolation", "area")),
            sam_depth_filter_enabled=bool((sam.get("depth_filter", {}) or {}).get("enabled", False)),
            sam_depth_filter_min_m=float((sam.get("depth_filter", {}) or {}).get("min_depth_m", geometry.get("min_depth_m", preproc_image.get("min_depth_m", 0.2)))),
            sam_depth_filter_max_m=float((sam.get("depth_filter", {}) or {}).get("max_depth_m", geometry.get("max_depth_m", preproc_image.get("max_depth_m", 6.0)))),
            sam_depth_filter_min_valid_ratio=max(0.0, min(1.0, float((sam.get("depth_filter", {}) or {}).get("min_valid_ratio", 0.05)))),
            sam_depth_filter_crop_to_roi=bool((sam.get("depth_filter", {}) or {}).get("crop_to_valid_roi", True)),
            sam_depth_filter_roi_margin_px=max(0, int((sam.get("depth_filter", {}) or {}).get("roi_margin_px", 8))),
            sam_depth_filter_background_value=tuple(int(v) for v in ((sam.get("depth_filter", {}) or {}).get("background_value", [0, 0, 0]))[:3]),
            sam_nanosam_image_encoder_engine=str((sam.get("nanosam", {}) or {}).get("image_encoder_engine", "")),
            sam_nanosam_mask_decoder_engine=str((sam.get("nanosam", {}) or {}).get("mask_decoder_engine", "")),
            sam_nanosam_points_per_side=max(1, int((sam.get("nanosam", {}) or {}).get("points_per_side", 3))),
            sam_nanosam_mask_threshold=float((sam.get("nanosam", {}) or {}).get("mask_threshold", 0.0)),
            sam_nanosam_nms_iou=max(0.0, min(1.0, float((sam.get("nanosam", {}) or {}).get("nms_iou", 0.70)))),
            sam_nanosam_point_margin_fraction=max(0.0, min(0.45, float((sam.get("nanosam", {}) or {}).get("point_margin_fraction", 0.08)))),
            sam_nanosam_skip_black_points=bool((sam.get("nanosam", {}) or {}).get("skip_black_points", True)),
            sam_nanosam_min_prompt_luma=max(0, int((sam.get("nanosam", {}) or {}).get("min_prompt_luma", 4))),
            rap_enabled=bool(rap.get("enabled", True)),
            rap_backend=str(rap.get("backend", "dummy")),
            rap_confidence_threshold=float(rap.get("confidence_threshold", 0.30)),
            rap_distance_threshold=float(rap.get("distance_threshold", 0.30)),
            rap_dummy_unknown_every_n=max(1, int(rap.get("dummy_unknown_every_n", 1))),
            rap_dummy_known_labels=list(rap.get("dummy_known_labels", [])),
            rap_update_enabled=bool(rap.get("update_memory_from_vlm", True)),
            rap_update_min_confidence=float(rap.get("update_min_confidence", 0.50)),
            rap_memory_path=str(rap.get("memory_update_path", "~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl")),
            rap_model_name=str(rap.get("model_name", "openai/clip-vit-base-patch32")),
            rap_device=str(rap.get("device", "")),
            rap_storage_path=str(rap.get("storage_path", "~/rsg_ros2_ws/visual_memory")),
            rap_chroma_host=str(rap.get("chroma_host", "localhost")),
            rap_chroma_port=int(rap.get("chroma_port", 8001)),
            rap_collection_name=str(rap.get("collection_name", "visual_rag")),
            rap_auto_start_server=bool(rap.get("auto_start_server", True)),
            rap_async=bool(rap.get("async", True)),
            rap_queue_size=max(1, int(rap.get("queue_size", 300))),
            rap_queue_drop_policy=str(rap.get("queue_drop_policy", "defer_track_id")),
            semantic_crop_rap_target_only_enabled=bool(semantic_crop.get("rap_target_only", True)),
            semantic_crop_rap_background_rgb=tuple(
                max(0, min(255, int(value)))
                for value in list(semantic_crop.get("rap_background_rgb", [32, 32, 32]))[:3]
            ),
            semantic_crop_vlm_target_focus_enabled=bool(semantic_crop.get("vlm_target_focus", True)),
            semantic_crop_vlm_context_alpha=max(0.0, min(1.0, float(semantic_crop.get("vlm_context_alpha", 0.10)))),
            semantic_crop_vlm_context_grayscale=bool(semantic_crop.get("vlm_context_grayscale", True)),
            semantic_crop_vlm_near_context_enabled=bool(semantic_crop.get("vlm_near_context_enabled", True)),
            semantic_crop_vlm_near_context_alpha=max(0.0, min(1.0, float(semantic_crop.get("vlm_near_context_alpha", 0.45)))),
            semantic_crop_vlm_near_context_dilation_px=max(0, int(semantic_crop.get("vlm_near_context_dilation_px", 15))),
            semantic_crop_vlm_near_context_grayscale=bool(semantic_crop.get("vlm_near_context_grayscale", False)),
            semantic_crop_mask_cleanup_enabled=bool(semantic_crop.get("mask_cleanup_enabled", True)),
            semantic_crop_mask_cleanup_min_component_area_ratio=max(0.0, min(1.0, float(semantic_crop.get("mask_cleanup_min_component_area_ratio", 0.02)))),
            semantic_crop_mask_cleanup_component_max_gap_px=max(0, int(semantic_crop.get("mask_cleanup_component_max_gap_px", 15))),
            semantic_crop_draw_target_contour=bool(semantic_crop.get("draw_target_contour", True)),
            semantic_crop_target_contour_rgb=tuple(
                max(0, min(255, int(value)))
                for value in list(semantic_crop.get("target_contour_rgb", [255, 255, 255]))[:3]
            ),
            semantic_crop_target_contour_thickness_px=max(0, int(semantic_crop.get("target_contour_thickness_px", 2))),
            semantic_labeling_enabled=bool(semantic_labeling.get("enabled", True)),
            semantic_labeling_min_observations=max(1, int(semantic_labeling.get("min_observations", 1))),
            semantic_labeling_settle_time_sec=max(0.0, float(semantic_labeling.get("settle_time_sec", 3.0))),
            semantic_labeling_force_dispatch_on_shutdown=bool(semantic_labeling.get("force_dispatch_on_shutdown", True)),
            semantic_labeling_publish_topic=str(semantic_labeling.get("result_topic", topics.get("semantic_label_result", "/rsg/objects/semantic_label_result"))),
            semantic_labeling_shutdown_wait_sec=max(0.0, float(semantic_labeling.get("shutdown_wait_sec", 8.0))),
            vlm_enabled=bool(vlm.get("enabled", True)),
            vlm_active_profile=str(vlm.get("active_profile", "")),
            vlm_mode=str(vlm.get("mode", "dummy")),
            vlm_async=bool(vlm.get("async", True)),
            vlm_queue_size=max(1, int(vlm.get("queue_size", 300))),
            vlm_queue_drop_policy=str(vlm.get("queue_drop_policy", "defer_track_id")),
            vlm_dummy_delay_sec=float(vlm.get("dummy_delay_sec", 0.0)),
            vlm_dummy_label_prefix=str(vlm.get("dummy_label_prefix", "vlm_dummy_object")),
            vlm_confidence=float(vlm.get("dummy_confidence", vlm.get("confidence", 0.55))),
            vlm_endpoint=str(vlm.get("endpoint", "http://127.0.0.1:8000/v1/chat/completions")),
            vlm_model=str(vlm.get("model", "Qwen2.5-VL-7B-Instruct")),
            vlm_timeout_sec=float(vlm.get("timeout_sec", 30.0)),
            vlm_api_key=str(vlm.get("api_key", "")),
            vlm_max_tokens=max(1, int(vlm.get("max_tokens", 32))),
            vlm_temperature=float(vlm.get("temperature", 0.0)),
            vlm_jpeg_quality=max(1, min(100, int(vlm.get("jpeg_quality", 85)))),
            vlm_prompt=str(vlm.get("prompt", "Identify the main object. Reply with one lowercase noun only. If unclear, reply unknown_object.")),
            vlm_prompt_opt_enabled=bool((vlm.get("prompt_optimisation", {}) or {}).get("enabled", False)),
            vlm_prompt_opt_run_id=str((vlm.get("prompt_optimisation", {}) or {}).get("run_id", "")),
            vlm_prompt_opt_prompt_version=str((vlm.get("prompt_optimisation", {}) or {}).get("prompt_version", "")),
            vlm_prompt_opt_output_root=str((vlm.get("prompt_optimisation", {}) or {}).get("output_root", "")),
            vlm_result_min_label_confidence=max(0.0, min(1.0, float((vlm.get("result_validation", {}) or {}).get("min_label_confidence", 0.35)))),
            vlm_result_min_mobility_confidence=max(0.0, min(1.0, float((vlm.get("result_validation", {}) or {}).get("min_mobility_confidence", 0.50)))),
            vlm_dynamic_label_hints=list((vlm.get("result_validation", {}) or {}).get("dynamic_label_hints", [])),
            vlm_static_label_hints=list((vlm.get("result_validation", {}) or {}).get("static_label_hints", [])),
            vlm_crop_context_ratio=max(0.0, min(0.50, float(vlm.get("crop_context_ratio", 0.15)))),
            vlm_crop_min_area_px=max(1, int(vlm.get("min_crop_area_px", 12000))),
            vlm_crop_min_short_side_px=max(1, int(vlm.get("min_crop_short_side_px", 64))),
            vlm_crop_min_quality_score=max(0.0, min(1.0, float(vlm.get("min_crop_quality_score", 0.34)))),
            vlm_crop_target_area_px=max(1, int(vlm.get("target_crop_area_px", 30000))),
            vlm_crop_target_short_side_px=max(1, int(vlm.get("target_crop_short_side_px", 120))),
            vlm_crop_border_penalty=max(0.0, min(0.25, float(vlm.get("border_penalty", 0.18)))),
            vlm_crop_quality_max_wait_sec=max(0.0, float(vlm.get("quality_defer_max_wait_sec", 2.0))),
            vlm_crop_quality_force_on_timeout=bool(vlm.get("quality_defer_force_on_timeout", True)),
            estimate_object_geometry=bool(geometry.get("enabled", True)),
            projection_stride=max(1, int(geometry.get("projection_stride", 4))),
            min_valid_depth_points=max(1, int(geometry.get("min_valid_depth_points", 20))),
            min_depth_m=float(geometry.get("min_depth_m", preproc_image.get("min_depth_m", 0.2))),
            max_depth_m=float(geometry.get("max_depth_m", preproc_image.get("max_depth_m", 6.0))),
            centroid_method=str(geometry.get("centroid_method", "median")),
            store_evidence_frames=bool(evidence.get("enabled", True)),
            evidence_buffer_size=max(1, int(evidence.get("max_frames", 50))),
            unknown_tracking_enabled=bool(unknown_tracking.get("enabled", True)),
            unknown_max_match_distance_m=float(unknown_tracking.get("max_match_distance_m", 0.30)),
            unknown_max_volume_ratio=float(unknown_tracking.get("max_volume_ratio", 3.0)),
            unknown_max_track_age_sec=float(unknown_tracking.get("max_track_age_sec", 2.0)),
            unknown_min_observations_before_vlm=max(1, int(unknown_tracking.get("min_observations_before_vlm", 3))),
            unknown_max_wait_before_vlm_sec=float(unknown_tracking.get("max_wait_before_vlm_sec", 0.75)),
            unknown_call_vlm_only_once_per_track=bool(unknown_tracking.get("call_vlm_only_once_per_track", True)),
            unknown_retry_failed_tracks=bool(unknown_tracking.get("retry_failed_tracks", False)),
            unknown_update_track_centroid=bool(unknown_tracking.get("update_track_centroid", True)),
            unknown_centroid_update_alpha=float(unknown_tracking.get("centroid_update_alpha", 0.7)),
            unknown_best_frame_selection_enabled=bool(unknown_tracking.get("best_frame_selection", True)),
            unknown_min_quality_for_vlm=float(unknown_tracking.get("min_quality_for_vlm", 0.0)),
            unknown_max_tracks=max(1, int(unknown_tracking.get("max_tracks", 200))),
            unknown_use_2d_iou_fallback=bool(unknown_tracking.get("use_2d_iou_fallback", True)),
            unknown_min_2d_iou=float(unknown_tracking.get("min_2d_iou", 0.30)),
            hydra_label_lookup=hydra_label_lookup,
            hydra_label_names=hydra_label_names,
            hydra_unclassified_label_id=unclassified_id,
            hydra_unclassified_label_name=str(hydra_label_names.get(unclassified_id, unclassified_name)),
            rap_execution_mode=rap_execution_mode,
            persistent_tracking_enabled=bool(persistent_tracking.get("enabled", False)),
            persistent_track_prefix=str(persistent_tracking.get("track_prefix", "rsg_obj_")),
            persistent_max_tracks=max(1, min(65535, int(persistent_tracking.get("max_tracks", slot_count if slot_mode else 1024)))),
            persistent_max_match_distance_m=float(persistent_tracking.get("max_match_distance_m", 0.30)),
            persistent_max_volume_ratio=float(persistent_tracking.get("max_volume_ratio", 3.0)),
            persistent_continuation_max_age_sec=max(0.0, float(persistent_tracking.get("continuation_max_age_sec", 2.5))),
            persistent_continuation_gap_m=max(0.0, float(persistent_tracking.get("continuation_gap_m", 0.18))),
            persistent_revisit_overlap_gap_m=max(0.0, float(persistent_tracking.get("revisit_overlap_gap_m", 0.05))),
            persistent_max_vertical_gap_m=max(0.0, float(persistent_tracking.get("max_vertical_gap_m", 0.12))),
            persistent_max_vertical_center_delta_m=max(0.0, float(persistent_tracking.get("max_vertical_center_delta_m", 0.12))),
            persistent_max_2d_iou_age_sec=float(persistent_tracking.get("max_2d_iou_age_sec", 1.0)),
            persistent_min_2d_iou=float(persistent_tracking.get("min_2d_iou", 0.30)),
            persistent_centroid_update_alpha=float(persistent_tracking.get("centroid_update_alpha", 0.7)),
            persistent_track_aware_redundancy_enabled=bool(persistent_tracking.get("track_aware_redundancy_enabled", True)),
            persistent_redundancy_union_coverage_threshold=float(persistent_tracking.get("redundancy_union_coverage_threshold", 0.90)),
            persistent_redundancy_child_containment_threshold=float(persistent_tracking.get("redundancy_child_containment_threshold", 0.85)),
            persistent_redundancy_min_children=max(2, int(persistent_tracking.get("redundancy_min_children", 2))),
            persistent_same_track_nested_suppression_enabled=bool(persistent_tracking.get("same_track_nested_suppression_enabled", True)),
            persistent_same_track_nested_containment_threshold=float(persistent_tracking.get("same_track_nested_containment_threshold", 0.90)),
            persistent_same_track_max_parent_child_area_ratio=max(1.0, float(persistent_tracking.get("same_track_max_parent_child_area_ratio", 3.0))),
            persistent_same_track_min_added_area_fraction=max(0.0, min(1.0, float(persistent_tracking.get("same_track_min_added_area_fraction", 0.05)))),
            persistent_same_track_broader_max_route_priority=max(0, min(3, int(persistent_tracking.get("same_track_broader_max_route_priority", 2)))),
            persistent_same_track_broader_max_volume_ratio=max(1.0, float(persistent_tracking.get("same_track_broader_max_volume_ratio", 6.0))),
            persistent_global_association_enabled=bool(persistent_tracking.get("global_association_enabled", True)),
            persistent_global_min_independent_groups=max(2, int(persistent_tracking.get("global_min_independent_groups", 2))),
            persistent_global_recent_min_score=min(1.0, max(0.0, float(persistent_tracking.get("global_recent_min_score", 0.55)))),
            persistent_global_revisit_min_score=min(1.0, max(0.0, float(persistent_tracking.get("global_revisit_min_score", 0.70)))),
            persistent_global_historical_overlap_pass=min(1.0, max(0.0, float(persistent_tracking.get("global_historical_overlap_pass", 0.30)))),
            persistent_global_recent_overlap_pass=min(1.0, max(0.0, float(persistent_tracking.get("global_recent_overlap_pass", 0.25)))),
            persistent_global_min_axis_overlap=min(1.0, max(0.0, float(persistent_tracking.get("global_min_axis_overlap", 0.20)))),
            persistent_global_touch_gap_pass_m=max(0.0, float(persistent_tracking.get("global_touch_gap_pass_m", 0.02))),
            persistent_global_centroid_pass_m=max(0.0, float(persistent_tracking.get("global_centroid_pass_m", 0.75))),
            persistent_global_centroid_sigma_m=max(1e-6, float(persistent_tracking.get("global_centroid_sigma_m", 0.50))),
            persistent_global_vertical_score_pass=min(1.0, max(0.0, float(persistent_tracking.get("global_vertical_score_pass", 0.60)))),
            persistent_global_vertical_sigma_m=max(1e-6, float(persistent_tracking.get("global_vertical_sigma_m", 0.15))),
            persistent_global_block_2d_on_3d_contradiction=bool(persistent_tracking.get("global_block_2d_on_3d_contradiction", True)),
            persistent_global_recent_weight_historical=max(0.0, float(persistent_tracking.get("global_recent_weight_historical", 0.20))),
            persistent_global_recent_weight_recent=max(0.0, float(persistent_tracking.get("global_recent_weight_recent", 0.30))),
            persistent_global_recent_weight_centroid=max(0.0, float(persistent_tracking.get("global_recent_weight_centroid", 0.25))),
            persistent_global_recent_weight_vertical=max(0.0, float(persistent_tracking.get("global_recent_weight_vertical", 0.15))),
            persistent_global_recent_weight_image=max(0.0, float(persistent_tracking.get("global_recent_weight_image", 0.10))),
            persistent_global_revisit_weight_historical=max(0.0, float(persistent_tracking.get("global_revisit_weight_historical", 0.45))),
            persistent_global_revisit_weight_recent=max(0.0, float(persistent_tracking.get("global_revisit_weight_recent", 0.00))),
            persistent_global_revisit_weight_centroid=max(0.0, float(persistent_tracking.get("global_revisit_weight_centroid", 0.30))),
            persistent_global_revisit_weight_vertical=max(0.0, float(persistent_tracking.get("global_revisit_weight_vertical", 0.20))),
            persistent_global_revisit_weight_image=max(0.0, float(persistent_tracking.get("global_revisit_weight_image", 0.05))),
            persistent_global_min_depth_z=max(0.05, float(persistent_tracking.get("global_min_depth_z", 0.30))),
            persistent_global_min_depth_xy=max(0.05, float(persistent_tracking.get("global_min_depth_xy", 0.15))),
            persistent_local_segments_enabled=bool(persistent_tracking.get("local_segments_enabled", True)),
            persistent_local_segment_max_xy_span_m=max(0.25, float(persistent_tracking.get("local_segment_max_xy_span_m", 4.0))),
            persistent_local_segment_revisit_distance_m=max(0.05, float(persistent_tracking.get("local_segment_revisit_distance_m", 1.5))),
            persistent_local_segment_gap_m=max(0.0, float(persistent_tracking.get("local_segment_gap_m", 0.20))),
            persistent_local_segment_2d_fallback_enabled=bool(persistent_tracking.get("local_segment_2d_fallback_enabled", True)),
            persistent_local_segment_max_2d_iou_age_sec=max(
                0.0,
                float(
                    persistent_tracking.get(
                        "local_segment_max_2d_iou_age_sec",
                        persistent_tracking.get("max_2d_iou_age_sec", 2.0),
                    )
                ),
            ),
            persistent_local_segment_min_2d_iou=min(
                1.0,
                max(
                    0.0,
                    float(
                        persistent_tracking.get(
                            "local_segment_min_2d_iou",
                            persistent_tracking.get("min_2d_iou", 0.30),
                        )
                    ),
                ),
            ),
            persistent_active_segments_topic=str(persistent_tracking.get("active_segments_topic", "/rsg/objects/active_local_segments")),
            persistent_require_known_label_match=bool(persistent_tracking.get("require_known_label_match", True)),
            persistent_unclassified_label_id=unclassified_id,
            persistent_use_hydra_slots=slot_mode,
            persistent_slot_first_label_id=slot_first_id,
            persistent_slot_count=max(1, min(65535, slot_count)),
            persistent_slot_label_prefix=slot_prefix,
            persistent_slot_label_width=max(1, slot_width),
            semantic_result_min_observations=max(1, int(semantic_result.get("min_observations", 1))),
            semantic_result_min_consensus=min(1.0, max(0.0, float(semantic_result.get("min_consensus", 0.0)))),
            semantic_result_min_evidence=max(0.0, float(semantic_result.get("min_evidence", 0.0))),
            persistent_rap_evidence_weight=max(0.0, float(persistent_tracking.get("rap_evidence_weight", 0.70))),
            persistent_vlm_evidence_weight=max(0.0, float(persistent_tracking.get("vlm_evidence_weight", 1.00))),
            persistent_label_aliases=persistent_label_aliases,
            loop_closure_enabled=bool(loop_closure.get("enabled", False)),
            loop_closure_map_frame=str(loop_closure.get("map_frame", "map")),
            loop_closure_odom_frame=str(loop_closure.get("odom_frame", "odom")),
            loop_closure_min_translation_m=max(0.0, float(loop_closure.get("min_translation_m", 0.05))),
            loop_closure_min_rotation_deg=max(0.0, float(loop_closure.get("min_rotation_deg", 0.5))),
            loop_closure_merge_duplicates=bool(loop_closure.get("merge_duplicates", True)),
            loop_closure_merge_recent_window_sec=max(0.0, float(loop_closure.get("merge_recent_window_sec", 5.0))),
            loop_closure_merge_distance_slack_m=max(0.0, float(loop_closure.get("merge_distance_slack_m", 0.6))),
            loop_closure_event_topic=str(loop_closure.get("event_topic", "/rsg/phase1/loop_closure_event")),
        )
        if config.persistent_tracking_enabled and not config.estimate_object_geometry:
            raise ValueError(
                "persistent_tracking.enabled requires object_geometry.enabled=true "
                "so that physical-object association uses 3D geometry"
            )
        # Slot-only tracking intentionally reserves ID 0 for background/no observation.
        if config.persistent_tracking_enabled and config.persistent_use_hydra_slots:
            last_slot = config.persistent_slot_first_label_id + config.persistent_slot_count - 1
            if config.persistent_slot_first_label_id <= 0 or last_slot > 65535:
                raise ValueError(
                    "persistent_tracking.slots must satisfy 1 <= first_label_id and "
                    "first_label_id + count - 1 <= 65535 for 16UC1 images"
                )
            if config.persistent_max_tracks > config.persistent_slot_count:
                config.persistent_max_tracks = config.persistent_slot_count
        if config.hydra_depth_range_filter_enabled:
            if config.hydra_depth_min_range_m < 0.0 or config.hydra_depth_max_range_m <= config.hydra_depth_min_range_m:
                raise ValueError("phase1.hydra_output.depth_range_filter must satisfy 0 <= min_range_m < max_range_m")
            if config.hydra_far_range_semantic_label_id != 0:
                raise ValueError(
                    "phase1.hydra_output.depth_range_filter.semantic_label_id must be 0; "
                    "far-range mesh exclusion is depth-driven, not semantic-label-driven"
                )
        return config
