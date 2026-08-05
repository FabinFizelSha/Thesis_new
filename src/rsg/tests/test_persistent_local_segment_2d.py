"""Regression tests for local-segment 2D fallback association."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from nodes.support.phase1.persistent_object_tracker import PersistentObjectTracker


class _Logger:
    """Minimal logger accepted by the tracker."""

    def debug(self, *_args, **_kwargs) -> None:
        pass

    def info(self, *_args, **_kwargs) -> None:
        pass

    def warning(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *_args, **_kwargs) -> None:
        pass


def _config(**overrides):
    values = {
        "persistent_track_prefix": "rsg_obj_",
        "persistent_max_tracks": 100,
        "persistent_max_match_distance_m": 0.30,
        "persistent_max_volume_ratio": 3.0,
        "persistent_continuation_max_age_sec": 2.5,
        "persistent_continuation_gap_m": 0.18,
        "persistent_revisit_overlap_gap_m": 0.05,
        "persistent_max_vertical_gap_m": 0.12,
        "persistent_max_vertical_center_delta_m": 0.12,
        "persistent_max_2d_iou_age_sec": 2.0,
        "persistent_min_2d_iou": 0.30,
        "persistent_centroid_update_alpha": 0.7,
        "persistent_local_segments_enabled": True,
        "persistent_local_segment_max_xy_span_m": 4.0,
        "persistent_local_segment_revisit_distance_m": 1.5,
        "persistent_local_segment_gap_m": 0.20,
        "persistent_local_segment_2d_fallback_enabled": True,
        "persistent_local_segment_max_2d_iou_age_sec": 2.0,
        "persistent_local_segment_min_2d_iou": 0.30,
        "persistent_require_known_label_match": False,
        "persistent_unclassified_label_id": 0,
        "persistent_use_hydra_slots": True,
        "persistent_slot_first_label_id": 1,
        "persistent_slot_count": 100,
        "persistent_slot_label_prefix": "unknown_slot_",
        "persistent_slot_label_width": 5,
        "persistent_label_aliases": {},
        "persistent_rap_evidence_weight": 0.70,
        "persistent_vlm_evidence_weight": 1.0,
        "semantic_result_min_observations": 1,
        "semantic_result_min_consensus": 0.0,
        "semantic_result_min_evidence": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _associate(
    tracker: PersistentObjectTracker,
    *,
    sequence: int,
    timestamp_sec: float,
    bbox_2d,
    bbox_3d_min=None,
    bbox_3d_max=None,
):
    metadata = {"bbox_2d": list(bbox_2d)}
    if bbox_3d_min is not None:
        metadata["bbox_3d_min"] = list(bbox_3d_min)
    if bbox_3d_max is not None:
        metadata["bbox_3d_max"] = list(bbox_3d_max)
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


class LocalSegment2DFallbackTests(unittest.TestCase):
    def test_reuses_local_slot_when_depth_geometry_is_missing(self) -> None:
        tracker = PersistentObjectTracker(_config(), _Logger())

        tracker.begin_frame()
        first_metadata, _ = _associate(
            tracker,
            sequence=1,
            timestamp_sec=0.0,
            bbox_2d=[443, 270, 59, 47],
        )

        tracker.begin_frame()
        second_metadata, second_record = _associate(
            tracker,
            sequence=2,
            timestamp_sec=0.05,
            bbox_2d=[443, 270, 59, 47],
        )

        self.assertEqual(first_metadata["persistent_track_id"], "rsg_obj_000001")
        self.assertEqual(second_metadata["persistent_track_id"], "rsg_obj_000001")
        self.assertEqual(
            second_metadata["local_segment_slot_id"],
            first_metadata["local_segment_slot_id"],
        )
        self.assertEqual(second_record["local_segment_event"], "matched_segment")
        self.assertEqual(
            second_record["local_segment_match_reason"],
            "segment_bbox_2d_iou",
        )
        self.assertAlmostEqual(second_record["local_segment_match_score"], 0.0)
        self.assertEqual(len(tracker._tracks["rsg_obj_000001"].segments), 1)
        self.assertEqual(
            next(iter(tracker._tracks["rsg_obj_000001"].segments.values())).seen_count,
            2,
        )

    def test_does_not_override_conflicting_valid_3d_geometry(self) -> None:
        tracker = PersistentObjectTracker(_config(), _Logger())

        tracker.begin_frame()
        first_metadata, _ = _associate(
            tracker,
            sequence=1,
            timestamp_sec=0.0,
            bbox_2d=[100, 100, 50, 50],
            bbox_3d_min=[0.0, 0.0, 0.0],
            bbox_3d_max=[1.0, 1.0, 1.0],
        )

        tracker.begin_frame()
        second_metadata, second_record = _associate(
            tracker,
            sequence=2,
            timestamp_sec=0.05,
            bbox_2d=[100, 100, 50, 50],
            bbox_3d_min=[10.0, 0.0, 0.0],
            bbox_3d_max=[11.0, 1.0, 1.0],
        )

        # Reliable 3D contradiction now blocks the parent 2D-only fallback.
        # The observation must create a new physical track rather than silently
        # extending the original object with a distant local segment.
        self.assertEqual(first_metadata["persistent_track_id"], "rsg_obj_000001")
        self.assertEqual(second_metadata["persistent_track_id"], "rsg_obj_000002")
        self.assertEqual(second_record["track_event"], "new_track")
        self.assertEqual(len(tracker._tracks["rsg_obj_000001"].segments), 1)


if __name__ == "__main__":
    unittest.main()
