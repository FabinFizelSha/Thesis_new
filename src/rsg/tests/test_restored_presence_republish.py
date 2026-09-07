"""Regression tests for restored-slot presence republication on a resumed run.

The fuser's presence cache is per-process and starts empty, so a resumed
session has to republish every restored slot's `internal_object_id` before the
fuser can tell that two object nodes are segments of one physical object.
Without it `internal_id_for()` returns empty, each node becomes its own
"__solo_" group, and their overlap renders as a *solid* contact edge between
two different objects rather than a *dotted* same-object edge -- and the
restored label never reaches the node either.

The bug these pin: the pending set was retired at *track* granularity, while
the ordinary heartbeat republishes only the single slot observed in the
current frame. Re-observing any one segment therefore stranded every other
segment of that object, which had often never been published at all -- the
retirement happens in run_rap_and_metadata, which runs *before* the publish
stage in the same frame.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from nodes.phase1 import Phase1SemanticCoordinator


def _segment(slot_id: int):
    return SimpleNamespace(
        segment_id=f"seg_{slot_id}",
        hydra_label_name=f"slot_{slot_id}",
        last_seen_timestamp_sec=1000.0,
        # numpy arrays, as the tracker stores them: _as_list calls .tolist()
        centroid_3d=np.array([float(slot_id), 0.0, 0.0]),
        bbox_3d_min=np.zeros(3),
        bbox_3d_max=np.ones(3),
        last_bbox_3d_min=np.zeros(3),
        last_bbox_3d_max=np.ones(3),
    )


class _Coordinator:
    """Minimal stand-in exposing only what _restored_presence_segments uses."""

    _restored_presence_segments = Phase1SemanticCoordinator._restored_presence_segments

    def __init__(self, tracks):
        self.persistent_tracker = SimpleNamespace(_tracks=tracks)
        self._restored_presence_pending = {
            int(slot)
            for track in tracks.values()
            for slot in track.segments
            if int(slot) > 0
        }


def _one_object_three_segments():
    """One physical object restored as three local segments."""
    track = SimpleNamespace(
        instance_id=1,
        canonical_label="sofa",
        segments={5: _segment(5), 6: _segment(6), 7: _segment(7)},
    )
    return {"track_a": track}


class RestoredPresenceRepublishTest(unittest.TestCase):
    def test_publishes_every_restored_slot_before_re_observation(self):
        coordinator = _Coordinator(_one_object_three_segments())

        out = coordinator._restored_presence_segments(set())

        self.assertEqual({entry["hydra_slot_id"] for entry in out}, {5, 6, 7})
        # The identity the fuser groups on has to be present and shared.
        self.assertEqual({entry["internal_object_id"] for entry in out}, {"track_a"})
        self.assertEqual({entry["canonical_label"] for entry in out}, {"sofa"})

    def test_re_observing_one_segment_keeps_the_others_published(self):
        """The regression: slot 6 seen, slots 5 and 7 must keep coming."""
        coordinator = _Coordinator(_one_object_three_segments())

        # Frame where slot 6 is re-observed: the heartbeat covers it, so it
        # arrives in seen_slots and must not be duplicated here.
        out = coordinator._restored_presence_segments({6})
        self.assertEqual({entry["hydra_slot_id"] for entry in out}, {5, 7})

        # A later frame observing nothing must still carry 5 and 7, or the
        # fuser loses their grouping and draws solid edges.
        out = coordinator._restored_presence_segments(set())
        self.assertEqual({entry["hydra_slot_id"] for entry in out}, {5, 7})

    def test_slot_retires_only_once_that_slot_is_observed(self):
        coordinator = _Coordinator(_one_object_three_segments())

        coordinator._restored_presence_segments({6})
        self.assertNotIn(6, coordinator._restored_presence_pending)
        self.assertEqual(coordinator._restored_presence_pending, {5, 7})

        coordinator._restored_presence_segments({5, 7})
        self.assertEqual(coordinator._restored_presence_pending, set())
        # Fully re-observed: the heartbeat owns all three now.
        self.assertEqual(coordinator._restored_presence_segments(set()), [])

    def test_does_not_republish_a_slot_seen_in_the_same_frame(self):
        """seen_slots wins, so the heartbeat's own entry is never duplicated."""
        coordinator = _Coordinator(_one_object_three_segments())

        seen = {5}
        out = coordinator._restored_presence_segments(seen)

        self.assertNotIn(5, [entry["hydra_slot_id"] for entry in out])
        # Emitted slots are added to seen_slots so later callers skip them.
        self.assertEqual(seen, {5, 6, 7})

    def test_multi_track_objects_stay_separate(self):
        tracks = _one_object_three_segments()
        tracks["track_b"] = SimpleNamespace(
            instance_id=2, canonical_label="chair", segments={9: _segment(9)}
        )
        coordinator = _Coordinator(tracks)

        out = coordinator._restored_presence_segments(set())

        by_slot = {entry["hydra_slot_id"]: entry["internal_object_id"] for entry in out}
        self.assertEqual(by_slot[5], "track_a")
        self.assertEqual(by_slot[9], "track_b")

    def test_no_restored_state_is_a_no_op(self):
        coordinator = _Coordinator({})
        self.assertEqual(coordinator._restored_presence_segments(set()), [])


class _LabelCoordinator:
    """Stand-in for the restored-label drain, with only publishing stubbed."""

    _drain_restored_semantic_labels = (
        Phase1SemanticCoordinator._drain_restored_semantic_labels
    )
    _emit_restored_semantic_label = (
        Phase1SemanticCoordinator._emit_restored_semantic_label
    )

    def __init__(self, tracks):
        self.persistent_tracker = SimpleNamespace(
            _tracks=tracks,
            _segment_record=lambda segment: {"segment_id": segment.segment_id},
        )
        self._restored_label_pending = set(tracks)
        self.emitted = []

    def get_logger(self):
        return SimpleNamespace(
            info=lambda *a, **k: None,
            warn=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        )

    def _emit_semantic_label_result(self, event, task, source, finalize_track):
        self.emitted.append((event["persistent_track_id"], event["semantic_label"], source))


def _labelled_track(name, label, slots):
    return SimpleNamespace(
        instance_id=1,
        semantic_label=label,
        canonical_label=label,
        semantic_label_confidence=0.9,
        label_confidence=0.9,
        mobility_class="static",
        mobility_confidence=0.8,
        mobility_source="prior",
        hydra_label_id=slots[0],
        hydra_label_name=f"slot_{slots[0]}",
        segments={s: _segment(s) for s in slots},
    )


class RestoredLabelDrainTest(unittest.TestCase):
    """A resumed map must open labelled, not blank until each object is revisited."""

    def _coordinator(self, count=10):
        tracks = {
            f"t{i}": _labelled_track(f"t{i}", f"label_{i}", [i + 1])
            for i in range(count)
        }
        return _LabelCoordinator(tracks)

    def test_emits_without_waiting_for_re_observation(self):
        coordinator = self._coordinator(count=3)
        frame = SimpleNamespace(rsg_frame_id="f", sequence=1)

        coordinator._drain_restored_semantic_labels(frame, 100.0)

        self.assertEqual({e[0] for e in coordinator.emitted}, {"t0", "t1", "t2"})
        self.assertEqual({e[2] for e in coordinator.emitted}, {"restored_session"})
        self.assertEqual(coordinator._restored_label_pending, set())

    def test_spreads_over_frames_instead_of_one_burst(self):
        coordinator = self._coordinator(count=10)
        frame = SimpleNamespace(rsg_frame_id="f", sequence=1)

        coordinator._drain_restored_semantic_labels(frame, 100.0, max_per_frame=4)
        self.assertEqual(len(coordinator.emitted), 4)
        self.assertEqual(len(coordinator._restored_label_pending), 6)

        for _ in range(10):
            coordinator._drain_restored_semantic_labels(frame, 100.0, max_per_frame=4)

        # Every track emitted exactly once, despite repeated drains.
        self.assertEqual(len(coordinator.emitted), 10)
        self.assertEqual(len(set(e[0] for e in coordinator.emitted)), 10)
        self.assertEqual(coordinator._restored_label_pending, set())

    def test_multi_segment_track_fans_out_to_all_segments(self):
        tracks = {"t0": _labelled_track("t0", "sofa", [5, 6, 7])}
        coordinator = _LabelCoordinator(tracks)
        frame = SimpleNamespace(rsg_frame_id="f", sequence=1)

        coordinator._drain_restored_semantic_labels(frame, 100.0)

        self.assertEqual(len(coordinator.emitted), 1)

    def test_nothing_restored_is_a_no_op(self):
        coordinator = _LabelCoordinator({})
        frame = SimpleNamespace(rsg_frame_id="f", sequence=1)
        coordinator._drain_restored_semantic_labels(frame, 100.0)
        self.assertEqual(coordinator.emitted, [])


if __name__ == "__main__":
    unittest.main()
