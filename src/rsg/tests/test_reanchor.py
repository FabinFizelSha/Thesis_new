"""Unit tests for loop-closure re-anchoring in the persistent object tracker.

Covers ``PersistentObjectTracker.reanchor_all`` (rigid transform of every
cached track/segment + spatial-index rebuild) and
``merge_reanchor_duplicates`` (drift pass + overlap pass).  Pure Python, no ROS.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nodes.support.phase1.persistent_object_tracker import (
    PersistentObjectSegment,
    PersistentObjectTrack,
    PersistentObjectTracker,
    _rigid_aabb,
)


class _Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def _config(**overrides):
    values = dict(
        persistent_track_prefix="rsg_obj_",
        persistent_max_tracks=100,
        persistent_max_match_distance_m=0.5,
        persistent_max_volume_ratio=4.0,
        persistent_continuation_max_age_sec=8.0,
        persistent_continuation_gap_m=0.5,
        persistent_revisit_overlap_gap_m=0.4,
        persistent_max_vertical_gap_m=0.15,
        persistent_max_vertical_center_delta_m=0.15,
        persistent_max_2d_iou_age_sec=3.0,
        persistent_min_2d_iou=0.30,
        persistent_centroid_update_alpha=0.5,
        persistent_local_segments_enabled=False,
        persistent_local_segment_max_xy_span_m=6.0,
        persistent_local_segment_revisit_distance_m=1.5,
        persistent_local_segment_gap_m=0.2,
        persistent_local_segment_2d_fallback_enabled=True,
        persistent_local_segment_max_2d_iou_age_sec=2.0,
        persistent_local_segment_min_2d_iou=0.3,
        persistent_require_known_label_match=False,
        persistent_unclassified_label_id=0,
        persistent_use_hydra_slots=True,
        persistent_slot_first_label_id=1,
        persistent_slot_count=100,
        persistent_slot_label_prefix="unknown_slot_",
        persistent_slot_label_width=5,
        persistent_label_aliases={},
        persistent_rap_evidence_weight=0.7,
        persistent_vlm_evidence_weight=1.0,
        semantic_result_min_observations=1,
        semantic_result_min_consensus=0.0,
        semantic_result_min_evidence=0.0,
        persistent_global_association_enabled=True,
        persistent_global_block_2d_on_3d_contradiction=True,
        persistent_global_centroid_pass_m=0.6,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _associate(tracker, *, sequence, timestamp_sec, bbox_2d, bbox_3d_min, bbox_3d_max):
    metadata = {
        "candidate_id": f"c{sequence}",
        "bbox_2d": list(bbox_2d),
        "bbox_3d_min": list(bbox_3d_min),
        "bbox_3d_max": list(bbox_3d_max),
    }
    return tracker.associate(
        metadata=metadata,
        frame_id=f"frame_{sequence}",
        sequence=sequence,
        timestamp_sec=timestamp_sec,
        desired_hydra_label_id=0,
        desired_hydra_label_name="unknown",
        raw_label="",
        label_source="none",
        label_confidence=0.0,
    )


def _seed_track(
    tracker,
    track_id,
    *,
    center,
    half=(0.5, 0.5, 0.5),
    seen=1,
    label="",
    first_ts=0.0,
    last_ts=0.0,
    slot=1,
):
    """Insert a fully-formed track directly (bypasses association)."""
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half, dtype=np.float64)
    bmin, bmax = c - h, c + h
    track = PersistentObjectTrack(
        track_id=track_id,
        instance_id=slot,
        hydra_label_id=slot,
        hydra_label_name=f"unknown_slot_{slot:05d}",
        semantic_kind="slot",
        canonical_label=label,
        label_source="vlm" if label else "none",
        label_confidence=0.9 if label else 0.0,
        first_seen_frame_id="f0",
        first_seen_sequence=0,
        first_seen_timestamp_sec=first_ts,
        last_seen_frame_id="f1",
        last_seen_sequence=1,
        last_seen_timestamp_sec=last_ts,
        centroid_3d=c.copy(),
        bbox_volume_m3=float(np.prod(bmax - bmin)),
        bbox_2d=[0, 0, 10, 10],
        bbox_3d_min=bmin.copy(),
        bbox_3d_max=bmax.copy(),
        last_bbox_3d_min=bmin.copy(),
        last_bbox_3d_max=bmax.copy(),
        seen_count=seen,
    )
    if label:
        track.raw_vlm_label = label
        track.semantic_label = label
    seg = PersistentObjectSegment(
        segment_id=f"{track_id}:slot_{slot}",
        hydra_label_id=slot,
        hydra_label_name=f"unknown_slot_{slot:05d}",
        instance_id=slot,
        first_seen_timestamp_sec=first_ts,
        last_seen_timestamp_sec=last_ts,
        first_seen_frame_id="f0",
        last_seen_frame_id="f1",
        first_seen_sequence=0,
        last_seen_sequence=1,
        centroid_3d=c.copy(),
        bbox_2d=[0, 0, 10, 10],
        bbox_3d_min=bmin.copy(),
        bbox_3d_max=bmax.copy(),
        last_bbox_3d_min=bmin.copy(),
        last_bbox_3d_max=bmax.copy(),
        seen_count=seen,
    )
    track.segments[slot] = seg
    track.active_segment_slot_id = slot
    tracker._tracks[track_id] = track
    tracker._refresh_spatial_index(track)
    return track


# --------------------------------------------------------------------------
# reanchor_all
# --------------------------------------------------------------------------
def test_reanchor_pure_translation_shifts_every_coordinate():
    tracker = PersistentObjectTracker(_config(), _Logger())
    for i, cx in enumerate((0.0, 5.0, -4.0)):
        _seed_track(tracker, f"t{i}", center=(cx, 0.0, 0.0), slot=i + 1)

    before = {d["track_id"]: d for d in tracker.debug_snapshot()}
    delta = np.array([1.0, 0.5, -0.25])
    n = tracker.reanchor_all(np.eye(3), delta, stamp=42.0)

    assert n == 3
    after = {d["track_id"]: d for d in tracker.debug_snapshot()}
    for tid, pre in before.items():
        post = after[tid]
        assert np.allclose(np.array(post["centroid_3d"]) - np.array(pre["centroid_3d"]), delta)
        assert np.allclose(np.array(post["bbox_3d_min"]) - np.array(pre["bbox_3d_min"]), delta)
        assert np.allclose(np.array(post["bbox_3d_max"]) - np.array(pre["bbox_3d_max"]), delta)
        # segment moved too
        seg = tracker._tracks[tid].segments[tracker._tracks[tid].active_segment_slot_id]
        assert np.allclose(seg.centroid_3d - np.array(pre["centroid_3d"]), delta)
        # volume invariant under a rigid motion
        pre_vol = np.prod(np.array(pre["bbox_3d_max"]) - np.array(pre["bbox_3d_min"]))
        assert np.isclose(tracker._tracks[tid].bbox_volume_m3, pre_vol)

    summary = tracker.last_reanchor()
    assert summary["track_count"] == 3
    assert summary["stamp_sec"] == 42.0
    assert np.allclose(summary["translation"], delta)


def test_reanchor_identity_is_a_noop():
    """An identity ΔT (or a below-threshold one the caller still forwards) must
    leave every coordinate, volume and spatial-index cell untouched."""
    tracker = PersistentObjectTracker(_config(), _Logger())
    for i, cx in enumerate((0.0, 3.0, -2.5)):
        _seed_track(tracker, f"t{i}", center=(cx, 1.0, 0.5), seen=4, slot=i + 1)
    before = {d["track_id"]: d for d in tracker.debug_snapshot()}
    before_cells = {
        tid: sorted(tracker._spatial_bbox_cells_by_track.get(tid, set()))
        for tid in before
    }

    n = tracker.reanchor_all(np.eye(3), np.zeros(3), stamp=1.0)

    assert n == 3
    after = {d["track_id"]: d for d in tracker.debug_snapshot()}
    for tid, pre in before.items():
        post = after[tid]
        assert np.allclose(post["centroid_3d"], pre["centroid_3d"])
        assert np.allclose(post["bbox_3d_min"], pre["bbox_3d_min"])
        assert np.allclose(post["bbox_3d_max"], pre["bbox_3d_max"])
        assert post["segment_slot_ids"] == pre["segment_slot_ids"]
        assert (
            sorted(tracker._spatial_bbox_cells_by_track.get(tid, set()))
            == before_cells[tid]
        )
    # and a no-signal merge still removes nothing
    assert tracker.merge_reanchor_duplicates() == 0
    assert len(tracker._tracks) == 3


def test_reanchor_rotation_uses_corner_aabb():
    tracker = PersistentObjectTracker(_config(), _Logger())
    _seed_track(tracker, "t0", center=(1.0, 0.5, 0.5), half=(1.0, 0.5, 0.5), slot=1)
    # box is min=[0,0,0] max=[2,1,1]
    rot_z90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    exp_min, exp_max = _rigid_aabb(
        np.array([0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0]), rot_z90, np.zeros(3)
    )
    tracker.reanchor_all(rot_z90, np.zeros(3))
    track = tracker._tracks["t0"]

    assert np.allclose(track.bbox_3d_min, exp_min)
    assert np.allclose(track.bbox_3d_max, exp_max)
    # Rz90 of an axis-aligned box is still axis-aligned -> volume preserved
    assert np.isclose(track.bbox_volume_m3, 2.0)
    assert np.allclose(track.centroid_3d, rot_z90 @ np.array([1.0, 0.5, 0.5]))


def test_reanchor_rebuilds_spatial_index():
    tracker = PersistentObjectTracker(_config(), _Logger())
    _seed_track(tracker, "t0", center=(0.0, 0.0, 0.0), slot=1)

    far_min, far_max = np.array([19.0, -1.0, -1.0]), np.array([21.0, 1.0, 1.0])
    far_centroid = np.array([20.0, 0.0, 0.0])
    assert "t0" not in tracker._candidate_track_ids(far_centroid, far_min, far_max)

    tracker.reanchor_all(np.eye(3), np.array([20.0, 0.0, 0.0]))

    assert "t0" in tracker._candidate_track_ids(far_centroid, far_min, far_max)
    origin_min, origin_max = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    assert "t0" not in tracker._candidate_track_ids(
        np.array([0.0, 0.0, 0.0]), origin_min, origin_max
    )


# --------------------------------------------------------------------------
# merge_reanchor_duplicates
# --------------------------------------------------------------------------
def test_drift_pass_folds_revisit_duplicate_into_older_identity():
    tracker = PersistentObjectTracker(_config(), _Logger())
    old = _seed_track(
        tracker, "t_old", center=(0.0, 0.0, 0.0), seen=6,
        label="chair", first_ts=0.0, last_ts=0.5, slot=1,
    )
    fresh = _seed_track(
        tracker, "t_new", center=(3.0, 0.0, 0.0), seen=2,
        label="chair", first_ts=100.0, last_ts=100.05, slot=2,
    )

    removed = tracker.merge_reanchor_duplicates(
        correction_translation_m=3.0, now_sec=100.1, recent_window_sec=5.0
    )

    assert removed == 1
    snap = tracker.debug_snapshot()
    assert [d["track_id"] for d in snap] == ["t_old"]
    survivor = tracker._tracks["t_old"]
    assert survivor.seen_count == 8
    # survivor keeps its identity but adopts the drift-corrected geometry
    assert np.allclose(survivor.centroid_3d, [3.0, 0.0, 0.0])
    assert survivor.first_seen_timestamp_sec == 0.0
    assert survivor.last_seen_timestamp_sec == 100.05
    # the revisit slot is now carried by the survivor
    assert {1, 2} <= set(survivor.segments)
    merged = survivor.metadata["reanchor_merged_from"]
    assert len(merged) == 1
    assert merged[0]["track_id"] == "t_new"
    assert merged[0]["reason"] == "loop_closure_drift"
    assert merged[0]["adopted_geometry"] is True
    _ = old, fresh


def test_drift_pass_ignores_incompatible_labels():
    tracker = PersistentObjectTracker(_config(), _Logger())
    _seed_track(tracker, "t_old", center=(0.0, 0.0, 0.0), seen=6,
                label="chair", first_ts=0.0, last_ts=0.5, slot=1)
    _seed_track(tracker, "t_new", center=(3.0, 0.0, 0.0), seen=2,
                label="table", first_ts=100.0, last_ts=100.05, slot=2)

    removed = tracker.merge_reanchor_duplicates(
        correction_translation_m=3.0, now_sec=100.1, recent_window_sec=5.0
    )
    assert removed == 0
    assert len(tracker._tracks) == 2


def test_drift_pass_respects_the_correction_radius():
    tracker = PersistentObjectTracker(_config(), _Logger())
    _seed_track(tracker, "t_old", center=(0.0, 0.0, 0.0), seen=6,
                label="chair", first_ts=0.0, last_ts=0.5, slot=1)
    # 3 m apart but the reported correction is only 0.2 m -> outside the radius
    _seed_track(tracker, "t_new", center=(3.0, 0.0, 0.0), seen=2,
                label="chair", first_ts=100.0, last_ts=100.05, slot=2)

    removed = tracker.merge_reanchor_duplicates(
        correction_translation_m=0.2, now_sec=100.1, recent_window_sec=5.0
    )
    assert removed == 0


def test_overlap_pass_folds_coincident_fragments():
    tracker = PersistentObjectTracker(_config(), _Logger())
    _seed_track(tracker, "t_a", center=(0.0, 0.0, 0.0), seen=5,
                label="sofa", first_ts=0.0, last_ts=1.0, slot=1)
    _seed_track(tracker, "t_b", center=(0.2, 0.1, 0.0), seen=1,
                label="sofa", first_ts=0.4, last_ts=0.9, slot=2)

    removed = tracker.merge_reanchor_duplicates(min_iou_3d=0.3)

    assert removed == 1
    assert [d["track_id"] for d in tracker.debug_snapshot()] == ["t_a"]
    survivor = tracker._tracks["t_a"]
    assert survivor.seen_count == 6
    assert survivor.metadata["reanchor_merged_from"][0]["reason"] == "post_reanchor_overlap"


def test_overlap_pass_reconciles_duplicate_segments_instead_of_stacking_them():
    """Before this fix, ``_merge_track_pair`` re-keyed *every* drop segment
    onto the survivor verbatim, even when it geometrically overlapped a
    segment the survivor already had. That left two permanently-static,
    heavily-overlapping local segments sitting side by side forever -- this
    is exactly what showed up as a near-duplicate SEGMENT_OVERLAP finding
    (iou_3d up to 0.74) in a real run. A touching/overlapping drop segment
    must now be folded into the matching keep segment instead of added as a
    second, redundant one."""
    tracker = PersistentObjectTracker(_config(), _Logger())
    _seed_track(tracker, "t_a", center=(0.0, 0.0, 0.0), half=(1.0, 1.5, 0.05),
                seen=5, label="ceiling", first_ts=0.0, last_ts=1.0, slot=1)
    _seed_track(tracker, "t_b", center=(0.1, 0.2, 0.0), half=(1.05, 1.2, 0.05),
                seen=3, label="ceiling", first_ts=1.2, last_ts=1.6, slot=2)

    removed = tracker.merge_reanchor_duplicates(min_iou_3d=0.3)

    assert removed == 1
    survivor = tracker._tracks["t_a"]
    # the two overlapping segments must be reconciled into one -- not left
    # stacked as two separate, permanently-overlapping slots.
    assert list(survivor.segments.keys()) == [1]
    merged_segment = survivor.segments[1]
    assert np.allclose(merged_segment.bbox_3d_min, [-1.0, -1.5, -0.05])
    assert np.allclose(merged_segment.bbox_3d_max, [1.15, 1.5, 0.05])
    assert merged_segment.seen_count == 5 + 3
    assert merged_segment.last_seen_timestamp_sec == 1.6
    assert merged_segment.first_seen_timestamp_sec == 0.0
    assert merged_segment.closed is False


def test_no_merge_without_signal():
    tracker = PersistentObjectTracker(_config(), _Logger())
    _seed_track(tracker, "t_a", center=(0.0, 0.0, 0.0), seen=5, label="sofa", slot=1)
    _seed_track(tracker, "t_b", center=(8.0, 0.0, 0.0), seen=5, label="sofa", slot=2)

    assert tracker.merge_reanchor_duplicates() == 0
    assert len(tracker._tracks) == 2


def test_end_to_end_reanchor_then_merge():
    """Tracks observed before the loop share one rigid correction; the revisit
    duplicate is cleaned by the drift pass afterwards."""
    tracker = PersistentObjectTracker(_config(), _Logger())
    # two objects mapped early, in the pre-correction frame
    _seed_track(tracker, "wall", center=(0.0, 0.0, 0.0), half=(2.0, 0.2, 1.0),
                seen=10, label="wall", first_ts=0.0, last_ts=2.0, slot=1)
    _seed_track(tracker, "desk", center=(1.0, 3.0, 0.0), seen=8,
                label="desk", first_ts=1.0, last_ts=3.0, slot=2)
    # the robot comes back and re-detects the desk 0.8 m off (drifted odom)
    _seed_track(tracker, "desk_dup", center=(1.8, 3.0, 0.0), seen=2,
                label="desk", first_ts=120.0, last_ts=120.1, slot=3)

    # loop closes: map->odom jumps by (-0.8, 0, 0)
    tracker.reanchor_all(np.eye(3), np.array([-0.8, 0.0, 0.0]))
    assert np.allclose(tracker._tracks["wall"].centroid_3d, [-0.8, 0.0, 0.0])
    assert np.allclose(tracker._tracks["desk_dup"].centroid_3d, [1.0, 3.0, 0.0])

    removed = tracker.merge_reanchor_duplicates(
        correction_translation_m=0.8, now_sec=120.2, recent_window_sec=5.0
    )
    assert removed == 1
    ids = sorted(d["track_id"] for d in tracker.debug_snapshot())
    assert ids == ["desk", "wall"]
    assert tracker._tracks["desk"].seen_count == 10
