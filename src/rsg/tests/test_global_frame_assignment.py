from types import SimpleNamespace
import numpy as np

from nodes.support.phase1.persistent_object_tracker import PersistentObjectTracker


class Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def config():
    return SimpleNamespace(
        persistent_track_prefix="rsg_obj_", persistent_max_tracks=100,
        persistent_max_match_distance_m=0.5, persistent_max_volume_ratio=4.0,
        persistent_continuation_max_age_sec=8.0, persistent_continuation_gap_m=0.5,
        persistent_revisit_overlap_gap_m=0.4, persistent_max_vertical_gap_m=0.15,
        persistent_max_vertical_center_delta_m=0.15, persistent_max_2d_iou_age_sec=3.0,
        persistent_min_2d_iou=0.30, persistent_centroid_update_alpha=0.5,
        persistent_local_segments_enabled=False, persistent_local_segment_max_xy_span_m=6.0,
        persistent_local_segment_revisit_distance_m=1.5, persistent_local_segment_gap_m=0.2,
        persistent_local_segment_2d_fallback_enabled=True,
        persistent_local_segment_max_2d_iou_age_sec=2.0,
        persistent_local_segment_min_2d_iou=0.3, persistent_require_known_label_match=False,
        persistent_unclassified_label_id=0, persistent_use_hydra_slots=True,
        persistent_slot_first_label_id=1, persistent_slot_count=100,
        persistent_slot_label_prefix="unknown_slot_", persistent_slot_label_width=5,
        persistent_label_aliases={}, persistent_rap_evidence_weight=0.7,
        persistent_vlm_evidence_weight=1.0, semantic_result_min_observations=1,
        semantic_result_min_consensus=0.0, semantic_result_min_evidence=0.0,
        persistent_track_aware_redundancy_enabled=True,
        persistent_redundancy_union_coverage_threshold=0.90,
        persistent_redundancy_child_containment_threshold=0.85,
        persistent_redundancy_min_children=2,
        persistent_same_track_nested_suppression_enabled=True,
        persistent_same_track_nested_containment_threshold=0.90,
        persistent_same_track_max_parent_child_area_ratio=3.0,
        persistent_same_track_min_added_area_fraction=0.05,
        persistent_same_track_broader_max_route_priority=2,
        persistent_same_track_broader_max_volume_ratio=6.0,
        persistent_global_association_enabled=False,
    )


def add_track(tracker, cid, bbox, ts=0.0):
    md = {"candidate_id": cid, "bbox_2d": bbox}
    return tracker.associate(metadata=md, frame_id="f0", sequence=0, timestamp_sec=ts,
        desired_hydra_label_id=0, desired_hydra_label_name="unknown", raw_label="",
        label_source="pending", label_confidence=0.0)


def test_global_assignment_removes_observation_order_bias():
    tracker = PersistentObjectTracker(config(), Logger())
    tracker.begin_frame()
    add_track(tracker, "seed_a", [0, 0, 100, 100])
    add_track(tracker, "seed_b", [80, 0, 100, 100])
    tracker.begin_frame()
    observations = [
        {"metadata": {"candidate_id": "ambiguous", "bbox_2d": [60, 0, 100, 100]}, "mask": np.ones((20,20), bool), "timestamp_sec": 0.1, "desired_hydra_label_id": 0},
        {"metadata": {"candidate_id": "perfect_a", "bbox_2d": [0, 0, 100, 100]}, "mask": np.ones((20,20), bool), "timestamp_sec": 0.1, "desired_hydra_label_id": 0},
    ]
    keep = tracker.prepare_frame_assignments(observations)
    assert keep == [True, True]
    assert tracker._forced_frame_matches["perfect_a"][0] == "rsg_obj_000001"
    assert tracker._forced_frame_matches["ambiguous"][0] == "rsg_obj_000002"


def test_a2_rejects_enclosing_union_when_children_map_to_distinct_tracks():
    tracker = PersistentObjectTracker(config(), Logger())
    tracker.begin_frame()
    add_track(tracker, "seed_a", [0, 0, 50, 100])
    add_track(tracker, "seed_b", [50, 0, 50, 100])
    tracker.begin_frame()
    left = np.zeros((100,100), bool); left[:, :50] = True
    right = np.zeros((100,100), bool); right[:, 50:] = True
    large = np.ones((100,100), bool)
    observations = [
        {"metadata": {"candidate_id": "left", "bbox_2d": [0,0,50,100]}, "mask": left, "timestamp_sec": .1, "desired_hydra_label_id": 0},
        {"metadata": {"candidate_id": "right", "bbox_2d": [50,0,50,100]}, "mask": right, "timestamp_sec": .1, "desired_hydra_label_id": 0},
        {"metadata": {"candidate_id": "large", "bbox_2d": [0,0,100,100]}, "mask": large, "timestamp_sec": .1, "desired_hydra_label_id": 0},
    ]
    keep = tracker.prepare_frame_assignments(observations)
    assert keep == [True, True, False]
    assert observations[2]["suppression_reason"] == "track_aware_union_redundancy"


def test_same_track_nested_duplicate_keeps_full_historical_continuation():
    tracker = PersistentObjectTracker(config(), Logger())
    tracker.begin_frame()
    # Historical right-floor track with a stable full observation.
    add_track(tracker, "seed_floor", [374, 258, 346, 221])
    tracker.begin_frame()

    full = np.zeros((480, 720), bool)
    full[258:479, 374:720] = True
    subsection = np.zeros((480, 720), bool)
    subsection[327:480, 393:720] = True
    observations = [
        {
            "metadata": {"candidate_id": "full_floor", "bbox_2d": [374, 258, 346, 221]},
            "mask": full,
            "timestamp_sec": 0.1,
            "desired_hydra_label_id": 0,
        },
        {
            "metadata": {"candidate_id": "floor_subsection", "bbox_2d": [393, 327, 327, 153]},
            "mask": subsection,
            "timestamp_sec": 0.1,
            "desired_hydra_label_id": 0,
        },
    ]

    keep = tracker.prepare_frame_assignments(observations)

    assert keep == [True, False]
    assert observations[1]["suppression_reason"] == "same_track_nested_duplicate"
    assert observations[1]["suppression_preferred_track_id"] == "rsg_obj_000001"
    assert tracker._forced_frame_matches["full_floor"][0] == "rsg_obj_000001"


def test_same_track_broader_wall_mask_inherits_existing_track():
    tracker = PersistentObjectTracker(config(), Logger())
    tracker.begin_frame()
    seed = {
        "candidate_id": "seed_wall",
        "bbox_2d": [62, 96, 98, 268],
        "centroid_3d": [0.0, 0.0, 1.5],
        "bbox_volume_m3": 1.0,
        "bbox_3d_min": [-0.2, -0.2, 0.0],
        "bbox_3d_max": [0.2, 0.2, 3.0],
    }
    tracker.associate(
        metadata=seed, frame_id="f0", sequence=0, timestamp_sec=0.0,
        desired_hydra_label_id=0, desired_hydra_label_name="unknown", raw_label="",
        label_source="pending", label_confidence=0.0,
    )
    tracker.begin_frame()

    narrow = np.zeros((420, 200), bool)
    narrow[96:364, 62:160] = True
    broad = np.zeros((420, 200), bool)
    broad[71:399, 0:161] = True
    observations = [
        {
            "metadata": {
                "candidate_id": "narrow_wall", "bbox_2d": [62, 96, 98, 268],
                "centroid_3d": [0.01, 0.01, 1.5], "bbox_volume_m3": 1.0,
                "bbox_3d_min": [-0.19, -0.19, 0.0], "bbox_3d_max": [0.21, 0.21, 3.0],
            },
            "mask": narrow, "timestamp_sec": 0.1, "desired_hydra_label_id": 0,
        },
        {
            "metadata": {
                "candidate_id": "broad_wall", "bbox_2d": [0, 71, 161, 328],
                "centroid_3d": [0.15, 0.02, 1.5], "bbox_volume_m3": 1.9,
                "bbox_3d_min": [-0.45, -0.22, 0.0], "bbox_3d_max": [0.25, 0.22, 3.0],
            },
            "mask": broad, "timestamp_sec": 0.1, "desired_hydra_label_id": 0,
        },
    ]
    keep = tracker.prepare_frame_assignments(observations)

    assert keep == [False, True]
    assert observations[0]["suppression_reason"] == "same_track_broader_mask_takeover"
    assert observations[0]["suppression_broader_promoted"] is True
    assert tracker._forced_frame_matches["broad_wall"][0] == "rsg_obj_000001"


def global_config():
    cfg = config()
    cfg.persistent_global_association_enabled = True
    cfg.persistent_global_min_independent_groups = 2
    cfg.persistent_global_recent_min_score = 0.55
    cfg.persistent_global_revisit_min_score = 0.70
    cfg.persistent_global_historical_overlap_pass = 0.30
    cfg.persistent_global_recent_overlap_pass = 0.25
    cfg.persistent_global_min_axis_overlap = 0.20
    cfg.persistent_global_touch_gap_pass_m = 0.02
    cfg.persistent_global_centroid_pass_m = 0.75
    cfg.persistent_global_centroid_sigma_m = 0.50
    cfg.persistent_global_vertical_score_pass = 0.60
    cfg.persistent_global_vertical_sigma_m = 0.15
    cfg.persistent_global_block_2d_on_3d_contradiction = True
    cfg.persistent_global_recent_weight_historical = 0.20
    cfg.persistent_global_recent_weight_recent = 0.30
    cfg.persistent_global_recent_weight_centroid = 0.25
    cfg.persistent_global_recent_weight_vertical = 0.15
    cfg.persistent_global_recent_weight_image = 0.10
    cfg.persistent_global_revisit_weight_historical = 0.45
    cfg.persistent_global_revisit_weight_recent = 0.00
    cfg.persistent_global_revisit_weight_centroid = 0.30
    cfg.persistent_global_revisit_weight_vertical = 0.20
    cfg.persistent_global_revisit_weight_image = 0.05
    return cfg


def add_3d_track(tracker, cid, *, ts=0.0):
    md = {
        "candidate_id": cid, "bbox_2d": [0, 0, 100, 100],
        "centroid_3d": [0.5, 0.5, 0.5], "bbox_volume_m3": 1.0,
        "bbox_3d_min": [0.0, 0.0, 0.0], "bbox_3d_max": [1.0, 1.0, 1.0],
    }
    return tracker.associate(
        metadata=md, frame_id="f0", sequence=0, timestamp_sec=ts,
        desired_hydra_label_id=0, desired_hydra_label_name="unknown", raw_label="",
        label_source="pending", label_confidence=0.0,
    )


def test_global_score_rejects_single_accumulated_contact_cue():
    tracker = PersistentObjectTracker(global_config(), Logger())
    tracker.begin_frame(); add_3d_track(tracker, "seed")
    tracker.begin_frame()
    # Near-contact with the accumulated box, but no centroid, recent, or image support.
    observation = {
        "metadata": {
            "candidate_id": "false_contact", "bbox_2d": [300, 300, 50, 50],
            "centroid_3d": [3.5, 0.5, 0.5], "bbox_volume_m3": 1.0,
            "bbox_3d_min": [1.01, 0.0, 0.0], "bbox_3d_max": [2.01, 1.0, 1.0],
        },
        "mask": np.ones((20, 20), bool), "timestamp_sec": 9.0,
        "desired_hydra_label_id": 0,
    }
    tracker.prepare_frame_assignments([observation])
    forced = tracker._forced_frame_matches["false_contact"]
    assert forced[0] is None
    evaluation = forced[3][0]
    assert evaluation["independent_pass_count"] < 2 or evaluation["global_association_score"] < 0.70
    assert "global_association_score_below_threshold" in evaluation["rejection_reasons"]


def test_global_score_accepts_valid_long_term_revisit_with_multiple_cues():
    tracker = PersistentObjectTracker(global_config(), Logger())
    tracker.begin_frame(); add_3d_track(tracker, "seed")
    tracker.begin_frame()
    observation = {
        "metadata": {
            "candidate_id": "valid_revisit", "bbox_2d": [0, 0, 100, 100],
            "centroid_3d": [0.55, 0.52, 0.5], "bbox_volume_m3": 1.1,
            "bbox_3d_min": [0.05, 0.02, 0.0], "bbox_3d_max": [1.05, 1.02, 1.0],
        },
        "mask": np.ones((20, 20), bool), "timestamp_sec": 20.0,
        "desired_hydra_label_id": 0,
    }
    tracker.prepare_frame_assignments([observation])
    forced = tracker._forced_frame_matches["valid_revisit"]
    assert forced[0] == "rsg_obj_000001"
    assert forced[1] == "global_revisit_association"
    evaluation = forced[3][0]
    assert evaluation["independent_pass_count"] >= 2
    assert evaluation["global_association_score"] >= 0.70


def test_spatial_search_rejects_2d_overlap_with_contradictory_3d():
    tracker = PersistentObjectTracker(global_config(), Logger())
    tracker.begin_frame(); add_3d_track(tracker, "seed")
    tracker.begin_frame()
    observation = {
        "metadata": {
            "candidate_id": "bad_2d", "bbox_2d": [0, 0, 100, 100],
            "centroid_3d": [5.5, 5.5, 0.5], "bbox_volume_m3": 1.0,
            "bbox_3d_min": [5.0, 5.0, 0.0], "bbox_3d_max": [6.0, 6.0, 1.0],
        },
        "mask": np.ones((20, 20), bool), "timestamp_sec": 0.1,
        "desired_hydra_label_id": 0,
    }
    tracker.prepare_frame_assignments([observation])
    forced = tracker._forced_frame_matches["bad_2d"]
    assert forced[0] is None
    assert forced[3] == []


def test_spatial_search_keeps_only_plausible_3d_candidates():
    tracker = PersistentObjectTracker(global_config(), Logger())
    tracker.begin_frame()
    add_3d_track(tracker, "target")
    for index in range(32):
        offset = 10.0 + 2.0 * index
        tracker.associate(
            metadata={
                "candidate_id": f"remote_{index}", "bbox_2d": [0, 0, 100, 100],
                "centroid_3d": [offset + 0.5, 0.5, 0.5], "bbox_volume_m3": 1.0,
                "bbox_3d_min": [offset, 0.0, 0.0], "bbox_3d_max": [offset + 1.0, 1.0, 1.0],
            },
            frame_id="f0", sequence=index + 1, timestamp_sec=0.0,
            desired_hydra_label_id=0, desired_hydra_label_name="unknown", raw_label="",
            label_source="pending", label_confidence=0.0,
        )
    tracker.begin_frame()

    match_id, _, _, evaluations = tracker._find_match(
        centroid=np.array([0.55, 0.52, 0.5]), volume=1.0, bbox_2d=[0, 0, 100, 100],
        bbox_3d_min=np.array([0.05, 0.02, 0.0]), bbox_3d_max=np.array([1.05, 1.02, 1.0]),
        timestamp_sec=0.1, desired_hydra_label_id=0,
    )

    assert match_id == "rsg_obj_000001"
    assert [row["candidate_track_id"] for row in evaluations] == ["rsg_obj_000001"]


def test_prepare_frame_assignments_stage_ms_sink_does_not_change_keep_mask():
    """Part 3 profiling side-channel must never alter assignment output."""
    tracker = PersistentObjectTracker(config(), Logger())
    tracker.begin_frame()
    add_track(tracker, "seed_a", [0, 0, 100, 100])
    add_track(tracker, "seed_b", [80, 0, 100, 100])
    tracker.begin_frame()
    observations = [
        {"metadata": {"candidate_id": "ambiguous", "bbox_2d": [60, 0, 100, 100]}, "mask": np.ones((20, 20), bool), "timestamp_sec": 0.1, "desired_hydra_label_id": 0},
        {"metadata": {"candidate_id": "perfect_a", "bbox_2d": [0, 0, 100, 100]}, "mask": np.ones((20, 20), bool), "timestamp_sec": 0.1, "desired_hydra_label_id": 0},
    ]
    keep_without_sink = tracker.prepare_frame_assignments([dict(obs) for obs in observations])

    tracker2 = PersistentObjectTracker(config(), Logger())
    tracker2.begin_frame()
    add_track(tracker2, "seed_a", [0, 0, 100, 100])
    add_track(tracker2, "seed_b", [80, 0, 100, 100])
    tracker2.begin_frame()
    stage_ms: dict = {}
    keep_with_sink = tracker2.prepare_frame_assignments([dict(obs) for obs in observations], stage_ms=stage_ms)

    assert keep_without_sink == keep_with_sink == [True, True]
    for key in (
        "assignment_candidate_search_ms", "assignment_row_init_ms",
        "assignment_3d_geometry_ms", "assignment_centroid_iou_ms",
        "assignment_scoring_ms", "assignment_a2_redundancy_ms",
        "assignment_a3_nested_ms", "assignment_hungarian_ms",
        "assignment_lock_wait_ms",
    ):
        assert key in stage_ms
        assert stage_ms[key] >= 0.0
    assert stage_ms["assignment_candidate_count_total"] >= 0.0
    assert stage_ms["assignment_candidate_count_max"] >= 0.0


def test_associate_stage_ms_sink_does_not_change_result():
    """Part 3 Path B profiling side-channel must never alter associate() output."""
    def run(stage_ms):
        tracker = PersistentObjectTracker(config(), Logger())
        tracker.begin_frame()
        add_track(tracker, "seed_a", [0, 0, 100, 100])
        tracker.begin_frame()
        return tracker.associate(
            metadata={"candidate_id": "obs_1", "bbox_2d": [5, 0, 100, 100]},
            frame_id="f1", sequence=1, timestamp_sec=0.1,
            desired_hydra_label_id=0, desired_hydra_label_name="unknown",
            raw_label="", label_source="pending", label_confidence=0.0,
            stage_ms=stage_ms,
        )

    metadata_without_sink, record_without_sink = run(None)
    stage_ms: dict = {}
    metadata_with_sink, record_with_sink = run(stage_ms)

    assert metadata_without_sink == metadata_with_sink
    assert record_without_sink["persistent_track_id"] == record_with_sink["persistent_track_id"]
    assert record_without_sink["track_event"] == record_with_sink["track_event"]
    assert "association_lock_wait_ms" in stage_ms
    assert stage_ms["association_lock_wait_ms"] >= 0.0


def test_find_match_stage_ms_sink_does_not_change_result():
    """Part 3 finer profiling side-channel must never alter _find_match output."""
    tracker = PersistentObjectTracker(config(), Logger())
    tracker.begin_frame()
    add_track(tracker, "seed_a", [0, 0, 100, 100])
    tracker.begin_frame()

    without_sink = tracker._find_match(
        centroid=None, volume=None, bbox_2d=[10, 0, 100, 100],
        bbox_3d_min=None, bbox_3d_max=None,
        timestamp_sec=0.1, desired_hydra_label_id=0,
    )
    stage_ms: dict = {}
    with_sink = tracker._find_match(
        centroid=None, volume=None, bbox_2d=[10, 0, 100, 100],
        bbox_3d_min=None, bbox_3d_max=None,
        timestamp_sec=0.1, desired_hydra_label_id=0, stage_ms=stage_ms,
    )

    assert without_sink[0] == with_sink[0]
    assert without_sink[1] == with_sink[1]
    assert without_sink[2] == with_sink[2]
    for key in (
        "assignment_row_init_ms", "assignment_3d_geometry_ms",
        "assignment_centroid_iou_ms", "assignment_scoring_ms",
    ):
        assert key in stage_ms
        assert stage_ms[key] >= 0.0
