"""Regression tests for the freeze-on-cap fix in
PersistentObjectTracker._assign_local_segment(): once a local segment's own
XY span reaches persistent_local_segment_max_xy_span_m, it must freeze
(closed=True) instead of continuing to absorb nearby observations via the
centroid-distance fallback -- that fallback path previously produced
multi-metre overlaps between adjacent segments instead of a clean seam.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from nodes.support.phase1.persistent_object_tracker import PersistentObjectTracker


class _Logger:
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
        "persistent_continuation_max_age_sec": 200.0,
        "persistent_continuation_gap_m": 0.18,
        "persistent_revisit_overlap_gap_m": 0.05,
        "persistent_max_vertical_gap_m": 0.12,
        "persistent_max_vertical_center_delta_m": 0.12,
        "persistent_max_2d_iou_age_sec": 2.0,
        "persistent_min_2d_iou": 0.30,
        "persistent_centroid_update_alpha": 0.7,
        "persistent_local_segments_enabled": True,
        "persistent_local_segment_max_xy_span_m": 6.0,
        "persistent_local_segment_revisit_distance_m": 1.5,
        "persistent_local_segment_gap_m": 0.20,
        "persistent_local_segment_2d_fallback_enabled": True,
        "persistent_local_segment_max_2d_iou_age_sec": 2.0,
        "persistent_local_segment_min_2d_iou": 0.30,
        "persistent_require_known_label_match": False,
        "persistent_unclassified_label_id": 0,
        "persistent_use_hydra_slots": True,
        "persistent_slot_first_label_id": 1,
        "persistent_slot_count": 1000,
        "persistent_slot_label_prefix": "unknown_slot_",
        "persistent_slot_label_width": 5,
        "persistent_label_aliases": {},
        "persistent_rap_evidence_weight": 0.70,
        "persistent_vlm_evidence_weight": 1.0,
        "semantic_result_min_observations": 1,
        "semantic_result_min_consensus": 0.0,
        "semantic_result_min_evidence": 0.0,
        # Isolate segment-level (freeze-on-cap) behaviour from the track-level
        # global quorum scorer, which needs a realistic centroid_3d and its
        # own weight tuning to pass on its own and is exercised separately in
        # test_global_frame_assignment.py. Track-level continuation here uses
        # the simpler accumulated-3D-footprint fallback (same as the plain
        # config() helper in that file).
        "persistent_global_association_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _associate(tracker, *, sequence, timestamp_sec, bbox_2d, bbox_3d_min, bbox_3d_max):
    metadata = {
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


class LocalSegmentFreezeOnCapTests(unittest.TestCase):
    def test_long_floor_splits_cleanly_without_overlap(self) -> None:
        """A 15m floor explored end-to-end in 0.5m steps (1m-wide boxes, so
        each step overlaps the previous by 0.5m) must split into segments
        that never exceed the span cap and never overlap each other."""
        tracker = PersistentObjectTracker(_config(), _Logger())
        seq = 0
        t = 0.0
        for i in range(29):
            xmin = 0.5 * i
            xmax = xmin + 1.0
            seq += 1
            t += 0.1
            tracker.begin_frame()
            metadata, record = _associate(
                tracker, sequence=seq, timestamp_sec=t,
                bbox_2d=[0, 0, 100, 100],
                bbox_3d_min=[xmin, 0.0, 0.0],
                bbox_3d_max=[xmax, 2.0, 0.1],
            )

        track_id = metadata["persistent_track_id"]
        self.assertEqual(track_id, "rsg_obj_000001")
        track = tracker._tracks[track_id]
        segments = sorted(track.segments.values(), key=lambda s: s.bbox_3d_min[0])

        self.assertEqual(len(segments), 3)
        max_span = 6.0
        eps = 0.05
        for segment in segments:
            span = segment.bbox_3d_max[0] - segment.bbox_3d_min[0]
            self.assertLessEqual(span, max_span + eps, f"segment {segment.hydra_label_id} exceeded span cap")
        for a, b in zip(segments, segments[1:]):
            overlap = a.bbox_3d_max[0] - b.bbox_3d_min[0]
            self.assertLessEqual(overlap, eps, f"segments {a.hydra_label_id}/{b.hydra_label_id} overlap by {overlap:.2f}m")

        # The two segments that actually hit the cap must be frozen; the
        # trailing partial segment (3m, never reached the cap) stays open.
        self.assertTrue(segments[0].closed)
        self.assertTrue(segments[1].closed)
        self.assertFalse(segments[2].closed)

    def test_revisit_from_far_side_reuses_existing_segment(self) -> None:
        """After a time gap, re-approaching the far end of the same floor
        from fresh (differently-framed) observations must re-associate onto
        the existing last segment, not spawn a spurious extra one."""
        tracker = PersistentObjectTracker(_config(), _Logger())
        seq = 0
        t = 0.0
        for i in range(29):
            xmin = 0.5 * i
            xmax = xmin + 1.0
            seq += 1
            t += 0.1
            tracker.begin_frame()
            metadata, _ = _associate(
                tracker, sequence=seq, timestamp_sec=t,
                bbox_2d=[0, 0, 100, 100],
                bbox_3d_min=[xmin, 0.0, 0.0],
                bbox_3d_max=[xmax, 2.0, 0.1],
            )

        track_id = metadata["persistent_track_id"]
        track = tracker._tracks[track_id]
        segments_before = sorted(track.segments.values(), key=lambda s: s.bbox_3d_min[0])
        last_slot_id = segments_before[-1].hydra_label_id

        t += 100.0  # well past any short-lived fallback window
        revisit_boxes = [(13.6, 14.6), (13.2, 14.2), (12.8, 13.8), (12.4, 13.4), (13.0, 14.0)]
        for xmin, xmax in revisit_boxes:
            seq += 1
            t += 0.1
            tracker.begin_frame()
            metadata, record = _associate(
                tracker, sequence=seq, timestamp_sec=t,
                bbox_2d=[0, 0, 100, 100],
                bbox_3d_min=[xmin, 0.0, 0.0],
                bbox_3d_max=[xmax, 2.0, 0.1],
            )

        self.assertEqual(metadata["persistent_track_id"], track_id)
        self.assertEqual(metadata["local_segment_slot_id"], last_slot_id)
        track = tracker._tracks[track_id]
        segments_after = sorted(track.segments.values(), key=lambda s: s.bbox_3d_min[0])
        self.assertEqual(len(segments_after), len(segments_before))


if __name__ == "__main__":
    unittest.main()
