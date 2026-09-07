"""Unit tests for cross-run persistence of the persistent object tracker.

Covers ``tracker_state_store.save_tracker_state`` / ``load_tracker_state``:
round-trip fidelity, the past-shift of timestamps that routes restored tracks
through the revisit-association branch, and id/slot collision safety.

Pure Python, no ROS.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nodes.support.phase1.persistent_object_tracker import (
    PersistentObjectSegment,
    PersistentObjectTrack,
    PersistentObjectTracker,
)
from nodes.support.phase1.tracker_state_store import (
    SCHEMA_VERSION,
    load_tracker_state,
    save_tracker_state,
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
        persistent_global_min_independent_groups=3,
        persistent_global_historical_overlap_pass=0.5,
        persistent_global_min_axis_overlap=0.2,
        persistent_global_touch_gap_pass_m=-1.0,
        persistent_global_containment_threshold=0.92,
        persistent_global_centroid_sigma_m=0.5,
        persistent_global_vertical_sigma_m=0.15,
        persistent_global_recent_min_score=0.30,
        persistent_global_revisit_min_score=0.30,
        persistent_revisit_min_2d_iou=0.75,
        persistent_global_recent_weight_historical=0.70,
        persistent_global_recent_weight_recent=0.0,
        persistent_global_recent_weight_centroid=0.30,
        persistent_global_recent_weight_vertical=0.0,
        persistent_global_recent_weight_image=0.45,
        persistent_global_recent_weight_containment=0.0,
        persistent_global_revisit_weight_historical=0.60,
        persistent_global_revisit_weight_recent=0.0,
        persistent_global_revisit_weight_centroid=0.25,
        persistent_global_revisit_weight_vertical=0.0,
        persistent_global_revisit_weight_image=0.35,
        persistent_global_revisit_weight_containment=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _tracker(**overrides):
    return PersistentObjectTracker(_config(**overrides), _Logger())


def _seed(tracker, track_id, *, center, slot, first_ts, last_ts, seen=12, closed=False):
    """Insert a fully-populated track (two segments, evidence, semantics)."""
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray([0.5, 0.5, 0.5])
    bmin, bmax = c - h, c + h
    track = PersistentObjectTrack(
        track_id=track_id,
        instance_id=slot,
        hydra_label_id=slot,
        hydra_label_name=f"unknown_slot_{slot:05d}",
        semantic_kind="slot",
        canonical_label="chair",
        label_source="vlm",
        label_confidence=0.87,
        first_seen_frame_id="frame_a",
        first_seen_sequence=3,
        first_seen_timestamp_sec=first_ts,
        last_seen_frame_id="frame_z",
        last_seen_sequence=91,
        last_seen_timestamp_sec=last_ts,
        centroid_3d=c.copy(),
        bbox_volume_m3=float(np.prod(bmax - bmin)),
        bbox_2d=[10, 20, 30, 40],
        bbox_3d_min=bmin.copy(),
        bbox_3d_max=bmax.copy(),
        last_bbox_3d_min=bmin.copy(),
        last_bbox_3d_max=bmax.copy(),
        seen_count=seen,
    )
    track.raw_vlm_label = "chair"
    track.raw_rap_label = "seat"
    track.mobility_class = "movable"
    track.mobility_confidence = 0.71
    track.mobility_source = "vlm"
    track.metadata = {"candidate_id": "c17", "depth_valid_points": 812}
    track.semantic_label = "chair"
    track.semantic_label_source = "vlm"
    track.semantic_label_confidence = 0.87
    track.semantic_timestamp_sec = last_ts
    track.semantic_update_count = 2
    track.semantic_hydra_class_id = 5
    track.semantic_reason = "consensus"
    track.label_evidence = {"chair": 1.7, "stool": 0.3}
    track.label_observations = {"chair": 2, "stool": 1}
    track.labeling_dispatched = True
    track.labeling_completed = True
    track.labeling_status = "completed"
    track.last_segment_event = "matched_segment"
    track.last_segment_match_reason = "segment_bbox_local"
    track.last_segment_match_score = 0.021

    for offset, seg_slot in enumerate((slot, slot + 50)):
        seg_c = c + np.asarray([offset * 1.5, 0.0, 0.0])
        seg = PersistentObjectSegment(
            segment_id=f"{track_id}:slot_{seg_slot}",
            hydra_label_id=seg_slot,
            hydra_label_name=f"unknown_slot_{seg_slot:05d}",
            instance_id=seg_slot,
            first_seen_timestamp_sec=first_ts,
            last_seen_timestamp_sec=last_ts,
            first_seen_frame_id="frame_a",
            last_seen_frame_id="frame_z",
            first_seen_sequence=3,
            last_seen_sequence=91,
            centroid_3d=seg_c.copy(),
            bbox_2d=[1, 2, 3, 4],
            bbox_3d_min=(seg_c - h).copy(),
            bbox_3d_max=(seg_c + h).copy(),
            last_bbox_3d_min=(seg_c - h).copy(),
            last_bbox_3d_max=(seg_c + h).copy(),
            seen_count=seen,
        )
        seg.closed = closed and offset == 1
        track.segments[seg_slot] = seg
        tracker._allocated_slot_ids.add(seg_slot)

    track.active_segment_slot_id = slot
    tracker._tracks[track_id] = track
    tracker._allocated_slot_ids.add(slot)
    tracker._refresh_spatial_index(track)
    return track


def _roundtrip(source, shift=86400.0):
    """Save `source`, load into a fresh tracker, return (restored, payload)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "state.json"
        assert save_tracker_state(source, path, shift, _Logger(), source_run="run_a")
        assert path.exists()
        payload = json.loads(path.read_text())
        restored = _tracker()
        count = load_tracker_state(restored, path, _Logger())
    return restored, payload, count


# --------------------------------------------------------------------------
# round-trip fidelity
# --------------------------------------------------------------------------


def test_roundtrip_preserves_every_track_field():
    source = _tracker()
    _seed(source, "rsg_obj_000004", center=(1.0, 2.0, 1.2), slot=4, first_ts=100.0, last_ts=160.0)
    source._next_track_index = 5

    restored, _payload, count = _roundtrip(source)

    assert count == 1
    a = source._tracks["rsg_obj_000004"]
    b = restored._tracks["rsg_obj_000004"]
    for name in (
        "track_id", "instance_id", "hydra_label_id", "hydra_label_name", "semantic_kind",
        "canonical_label", "label_source", "label_confidence", "first_seen_frame_id",
        "first_seen_sequence", "last_seen_frame_id", "last_seen_sequence", "bbox_volume_m3",
        "bbox_2d", "seen_count", "raw_rap_label", "raw_vlm_label", "mobility_class",
        "mobility_confidence", "mobility_source", "slot_state", "semantic_update_count",
        "semantic_label", "semantic_label_source", "semantic_label_confidence",
        "semantic_hydra_class_id", "semantic_reason", "label_evidence", "label_observations",
        "labeling_dispatched", "labeling_completed", "labeling_status",
        "active_segment_slot_id", "last_segment_event", "last_segment_match_reason",
        "last_segment_match_score", "metadata",
    ):
        assert getattr(b, name) == getattr(a, name), name

    for name in ("centroid_3d", "bbox_3d_min", "bbox_3d_max", "last_bbox_3d_min", "last_bbox_3d_max"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), err_msg=name)


def test_roundtrip_preserves_segments_with_int_keys():
    """JSON stringifies dict keys; segments must come back keyed by int slot id."""
    source = _tracker()
    _seed(source, "rsg_obj_000002", center=(0.0, 0.0, 1.0), slot=2, first_ts=10.0,
          last_ts=50.0, closed=True)

    restored, _payload, _count = _roundtrip(source)

    a = source._tracks["rsg_obj_000002"]
    b = restored._tracks["rsg_obj_000002"]
    assert set(b.segments) == set(a.segments)
    assert all(isinstance(k, int) for k in b.segments)
    for slot_id, seg_a in a.segments.items():
        seg_b = b.segments[slot_id]
        assert seg_b.segment_id == seg_a.segment_id
        assert seg_b.closed == seg_a.closed          # freeze-on-cap must survive
        assert seg_b.seen_count == seg_a.seen_count
        assert seg_b.bbox_2d == seg_a.bbox_2d
        np.testing.assert_allclose(seg_b.centroid_3d, seg_a.centroid_3d)
        np.testing.assert_allclose(seg_b.bbox_3d_min, seg_a.bbox_3d_min)


def test_spatial_index_is_rebuilt_not_serialised():
    source = _tracker()
    _seed(source, "rsg_obj_000001", center=(3.0, 3.0, 1.0), slot=1, first_ts=0.0, last_ts=40.0)

    restored, payload, _count = _roundtrip(source)

    assert "spatial" not in json.dumps(payload)
    assert restored._spatial_bbox_cells, "index should be populated after load"
    candidates = restored._candidate_track_ids(
        np.asarray([3.0, 3.0, 1.0]), np.asarray([2.5, 2.5, 0.5]), np.asarray([3.5, 3.5, 1.5])
    )
    assert "rsg_obj_000001" in candidates


# --------------------------------------------------------------------------
# the time shift -- the entire point of the feature
# --------------------------------------------------------------------------


def test_every_timestamp_is_shifted_into_the_past():
    source = _tracker()
    _seed(source, "rsg_obj_000001", center=(0.0, 0.0, 1.0), slot=1, first_ts=100.0, last_ts=160.0)
    shift = 86400.0

    restored, _payload, _count = _roundtrip(source, shift=shift)

    b = restored._tracks["rsg_obj_000001"]
    assert b.first_seen_timestamp_sec == 100.0 - shift
    assert b.last_seen_timestamp_sec == 160.0 - shift
    assert b.semantic_timestamp_sec == 160.0 - shift
    for seg in b.segments.values():
        assert seg.first_seen_timestamp_sec == 100.0 - shift
        assert seg.last_seen_timestamp_sec == 160.0 - shift


def test_restored_track_associates_in_revisit_mode():
    """Without the shift this lands in *recent* mode -- the opposite of intent.

    Replay timestamps restart at the bag's start time, which is smaller than a
    saved last_seen_timestamp_sec, so the raw age is negative and _find_match's
    max(0.0, ...) clamps it to 0.0 -> recent mode. The past-shift is what makes
    the age large and positive.
    """
    source = _tracker()
    _seed(source, "rsg_obj_000001", center=(0.0, 0.0, 1.0), slot=1, first_ts=100.0, last_ts=160.0)

    restored, _payload, _count = _roundtrip(source, shift=86400.0)

    # A replay timestamp near the *start* of the bag, i.e. earlier than the
    # original session's last_seen.
    _match, _reason, _score, evaluations = restored._find_match(
        centroid=np.asarray([0.0, 0.0, 1.0]),
        volume=1.0,
        bbox_2d=[10, 20, 30, 40],
        bbox_3d_min=np.asarray([-0.5, -0.5, 0.5]),
        bbox_3d_max=np.asarray([0.5, 0.5, 1.5]),
        timestamp_sec=105.0,
        desired_hydra_label_id=0,
    )
    modes = {row.get("association_mode") for row in evaluations if "association_mode" in row}
    assert modes == {"revisit"}, f"expected revisit mode, got {modes}"


def test_unshifted_state_would_fall_into_recent_mode():
    """Pin the trap itself, so a future change to the shift cannot silently
    reintroduce it."""
    source = _tracker()
    _seed(source, "rsg_obj_000001", center=(0.0, 0.0, 1.0), slot=1, first_ts=100.0, last_ts=160.0)

    restored, _payload, _count = _roundtrip(source, shift=0.0)

    _match, _reason, _score, evaluations = restored._find_match(
        centroid=np.asarray([0.0, 0.0, 1.0]),
        volume=1.0,
        bbox_2d=[10, 20, 30, 40],
        bbox_3d_min=np.asarray([-0.5, -0.5, 0.5]),
        bbox_3d_max=np.asarray([0.5, 0.5, 1.5]),
        timestamp_sec=105.0,
        desired_hydra_label_id=0,
    )
    modes = {row.get("association_mode") for row in evaluations if "association_mode" in row}
    assert modes == {"recent"}


def test_cumulative_shift_accumulates_across_cycles():
    source = _tracker()
    _seed(source, "rsg_obj_000001", center=(0.0, 0.0, 1.0), slot=1, first_ts=100.0, last_ts=160.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        save_tracker_state(source, path, 86400.0, _Logger())
        second = _tracker()
        load_tracker_state(second, path, _Logger())
        save_tracker_state(second, path, 86400.0, _Logger())
        payload = json.loads(path.read_text())

    assert payload["cumulative_time_shift_sec"] == 2 * 86400.0
    assert payload["time_shift_sec"] == 86400.0


# --------------------------------------------------------------------------
# identity safety
# --------------------------------------------------------------------------


def test_new_track_after_load_does_not_collide():
    """_new_track does `_tracks[id] = track`, so a reused id is silent data loss."""
    source = _tracker()
    for idx, slot in ((1, 1), (2, 2), (3, 3)):
        _seed(source, f"rsg_obj_{idx:06d}", center=(idx, 0.0, 1.0), slot=slot,
              first_ts=0.0, last_ts=60.0)
    source._next_track_index = 4

    restored, _payload, _count = _roundtrip(source)

    assert restored._next_track_index == 4
    track = restored._new_track(
        metadata={},
        frame_id="f",
        sequence=1,
        timestamp_sec=1.0,
        centroid=np.asarray([9.0, 9.0, 1.0]),
        volume=1.0,
        bbox_2d=[0, 0, 5, 5],
        bbox_3d_min=np.asarray([8.5, 8.5, 0.5]),
        bbox_3d_max=np.asarray([9.5, 9.5, 1.5]),
        desired_hydra_label_id=0,
        desired_hydra_label_name="unknown",
        raw_label="",
        label_source="none",
        label_confidence=0.0,
        use_hydra_slot=True,
    )
    # _new_track builds and returns the track; `associate` is what inserts it,
    # so the collision that matters is on the id and the slot, not on _tracks.
    assert track.track_id not in restored._tracks
    assert track.track_id == "rsg_obj_000004"
    assert track.hydra_label_id not in {1, 2, 3, 51, 52, 53}


def test_next_track_index_recovers_from_a_stale_counter():
    source = _tracker()
    _seed(source, "rsg_obj_000009", center=(0.0, 0.0, 1.0), slot=9, first_ts=0.0, last_ts=60.0)
    source._next_track_index = 1          # deliberately stale

    restored, _payload, _count = _roundtrip(source)

    assert restored._next_track_index == 10


def test_allocated_slots_survive_even_without_a_live_track():
    """Slots are never released, so the set is a superset of live slot ids and
    cannot be rebuilt from _tracks alone."""
    source = _tracker()
    _seed(source, "rsg_obj_000001", center=(0.0, 0.0, 1.0), slot=1, first_ts=0.0, last_ts=60.0)
    source._allocated_slot_ids.add(77)    # retired track's slot

    restored, _payload, _count = _roundtrip(source)

    assert 77 in restored._allocated_slot_ids


# --------------------------------------------------------------------------
# label republication to the fuser
# --------------------------------------------------------------------------


def test_restored_track_carries_everything_the_fuser_fanout_needs():
    """A restored track is never re-classified, so its stored label is the only
    copy phase 1 has. The fuser's overlay cache is per-process and starts empty,
    so that label has to be republished per Hydra slot -- which requires the
    semantic fields and every segment's slot id to survive the round trip."""
    source = _tracker()
    _seed(source, "rsg_obj_000003", center=(2.0, 1.0, 1.0), slot=3, first_ts=0.0, last_ts=90.0)

    restored, _payload, _count = _roundtrip(source)

    track = restored._tracks["rsg_obj_000003"]
    assert track.semantic_label == "chair"
    assert track.semantic_label_confidence == 0.87
    assert track.mobility_class == "movable"
    assert track.mobility_source == "vlm"
    # Not re-dispatched to RAP/VLM: this is what makes republication necessary.
    assert track.labeling_completed is True

    records = [PersistentObjectTracker._segment_record(s) for s in track.segments.values()]
    assert {r["hydra_slot_id"] for r in records} == {3, 53}
    assert all(r["hydra_slot_name"] for r in records)
    assert all(r["local_segment_id"] for r in records)


def test_unlabelled_restored_track_has_nothing_to_republish():
    source = _tracker()
    track = _seed(source, "rsg_obj_000001", center=(0.0, 0.0, 1.0), slot=1,
                  first_ts=0.0, last_ts=60.0)
    track.semantic_label = ""
    track.canonical_label = ""

    restored, _payload, _count = _roundtrip(source)

    b = restored._tracks["rsg_obj_000001"]
    assert not (b.semantic_label or b.canonical_label)


# --------------------------------------------------------------------------
# failure handling -- must never block startup
# --------------------------------------------------------------------------


def test_missing_file_is_a_normal_first_run():
    tracker = _tracker()
    assert load_tracker_state(tracker, Path("/nonexistent/state.json"), _Logger()) == 0
    assert tracker._tracks == {}


def test_corrupt_file_is_ignored():
    tracker = _tracker()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        path.write_text("{not valid json")
        assert load_tracker_state(tracker, path, _Logger()) == 0
    assert tracker._tracks == {}


def test_schema_mismatch_is_ignored():
    tracker = _tracker()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1, "tracks": {}}))
        assert load_tracker_state(tracker, path, _Logger()) == 0


def test_save_writes_atomically():
    source = _tracker()
    _seed(source, "rsg_obj_000001", center=(0.0, 0.0, 1.0), slot=1, first_ts=0.0, last_ts=60.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        save_tracker_state(source, path, 86400.0, _Logger())
        assert path.exists()
        assert not path.with_suffix(path.suffix + ".tmp").exists()
