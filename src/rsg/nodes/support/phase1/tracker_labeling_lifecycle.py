"""RAP/VLM labeling-request lifecycle for PersistentObjectTracker.

Tracks which tracks are settled and ready for RAP/VLM, and commits their
RAP/VLM outcomes. Crop selection itself is owned by Phase 1's shared crop
registry (Phase1TrackCropRegistry, in nodes/phase1_pipeline/), not here -- this
only tracks the *scheduling* state (dispatched/completed/status/attempt
count) for each track. Every public method here is an externally-called
entry point (from phase1.py's dispatch stages) and acquires
``tracker._lock`` itself, exactly as before the extraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from nodes.support.phase1 import tracker_serialization
from nodes.support.phase1.vlm_result import DEFAULT_OBJECT_DETAIL


class TrackerLabelingLifecycle:
    """Owns per-track RAP/VLM scheduling state and outcome commit."""

    def __init__(self, tracker: Any):
        self.tracker = tracker

    def prepare_active_for_labeling(
        self,
        current_timestamp_sec: float,
        *,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return RAP task records after the fixed crop-settling window.

        A semantic job is intentionally not scheduled from the first valid crop.
        The persistent track remains live while later observations can replace
        that crop in Phase 1's shared best-crop registry. ``current_timestamp_sec``
        uses recorded message time, so the delay is deterministic during bag
        replay and independent of worker latency.
        """
        tracker = self.tracker
        ready: List[Dict[str, Any]] = []
        min_observations = int(
            getattr(tracker.config, "semantic_labeling_min_observations", 1)
        )
        settle_time_sec = max(
            0.0,
            float(getattr(tracker.config, "semantic_labeling_settle_time_sec", 0.0)),
        )
        now_sec = float(current_timestamp_sec)

        with tracker._lock:
            # Iterate over every live track rather than only the current-frame
            # detections. This lets an object that has left the camera view be
            # released once its fixed collection interval has elapsed.
            for track in tracker._tracks.values():
                if track.labeling_dispatched or track.labeling_completed:
                    continue
                if int(track.seen_count) < min_observations:
                    continue

                settling_age_sec = max(0.0, now_sec - float(track.first_seen_timestamp_sec))
                if not force and settling_age_sec < settle_time_sec:
                    track.labeling_status = "collecting"
                    continue

                track.labeling_dispatched = True
                track.labeling_status = "queued"
                reason = "shutdown_best_available_crop" if force else "fixed_settling_window_elapsed"
                record = tracker_serialization.track_record(track, "labeling_ready", reason, None)
                record["settling_age_sec"] = float(settling_age_sec)
                record["settle_time_sec"] = float(settle_time_sec)
                record["forced_dispatch"] = bool(force)
                ready.append(record)
        return ready

    def release_labeling_request(self, track_id: str, reason: str) -> None:
        """Allow a later observation to retry RAP after enqueue/worker failure."""
        tracker = self.tracker
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            if track is None or track.labeling_completed:
                return
            track.labeling_dispatched = False
            track.labeling_status = str(reason)

    def set_labeling_status(self, track_id: str, status: str) -> None:
        """Record the asynchronous semantic stage without changing track identity.

        Crop selection remains owned by Phase 1's shared registry.  This state
        is diagnostic and makes it explicit that the representative crop stays
        mutable while a track ID waits in RAP or VLM.
        """
        tracker = self.tracker
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            if track is None or track.labeling_completed:
                return
            track.labeling_status = str(status)

    def is_semantic_labeling_open(self, track_id: str) -> bool:
        """Return whether a track may still accept crop updates."""
        tracker = self.tracker
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            return bool(track is not None and not track.labeling_completed)

    def increment_vlm_attempt_count(self, track_id: str) -> Optional[int]:
        """Record one more failed VLM attempt and return the new total.

        Returns ``None`` if the track no longer exists or is already
        finalized -- the caller should then treat this as exhausted rather
        than retry a track that isn't live anymore.
        """
        tracker = self.tracker
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            if track is None or track.labeling_completed:
                return None
            track.vlm_attempt_count += 1
            return int(track.vlm_attempt_count)

    def record_vlm_attempt_detail(self, track_id: str, object_detail: str) -> None:
        """Keep a failed attempt's object_detail without committing a label.

        validate_vlm_response extracts ``object_detail`` unconditionally,
        independent of label_confidence -- so even a rejected label can carry
        a real, useful description. This only updates that one field; it
        deliberately does not touch semantic_label/mobility (those still
        require a successful attempt via apply_vlm_result).
        """
        detail = str(object_detail or "").strip()
        if not detail:
            return
        tracker = self.tracker
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            if track is None or track.labeling_completed:
                return
            track.object_detail = detail

    def get_waiting_record(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Return a read-only snapshot for a track waiting on a VLM retry.

        Unlike complete_semantic_labeling, this does not mark the track
        completed or touch any state -- it only builds the same record shape
        Phase 1 publishes to the fuser, so a "waiting_for_better_crop" status
        (and whatever object_detail is currently known) can be shown while
        the track keeps collecting crops.
        """
        tracker = self.tracker
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            if track is None:
                return None
            return tracker_serialization.track_record(track, "vlm_retry_pending", "vlm_retry_pending", None)

    def complete_semantic_labeling(
        self,
        track_id: str,
        timestamp_sec: float,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Commit the first RAP/VLM outcome without changing the Hydra slot.

        The returned record is consumed by Phase 1 to publish a
        ``semantic_label_result`` event.  The slot remains the same physical
        object identity and is never replaced by a class ID.
        """
        tracker = self.tracker
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            if track is None:
                return None
            tracker.semantic_evidence._commit_semantic_label(track, float(timestamp_sec), str(reason))
            track.labeling_completed = True
            track.labeling_status = "completed"
            return tracker_serialization.track_record(track, "semantic_label_completed", str(reason), None)

    def apply_vlm_result(
        self,
        track_id: str,
        label: str,
        confidence: float,
        mobility_class: str = "unknown",
        mobility_confidence: float = 0.0,
        object_detail: str = DEFAULT_OBJECT_DETAIL,
    ) -> Optional[Dict[str, Any]]:
        """Attach one validated VLM label and mobility decision to a slot."""
        tracker = self.tracker
        normalised = tracker.semantic_evidence._canonicalise_label(label)
        if not normalised:
            return None
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            if track is None:
                return None
            track.raw_vlm_label = normalised
            track.object_detail = str(object_detail or DEFAULT_OBJECT_DETAIL)
            tracker.semantic_evidence._update_semantics(track, normalised, "vlm", float(confidence))
            tracker.semantic_evidence._update_mobility(
                track,
                mobility_class=mobility_class,
                confidence=mobility_confidence,
                source="vlm",
            )
            return tracker_serialization.track_record(track, "vlm_semantic_update", "vlm_result", None)

    def apply_rap_result(
        self,
        track_id: str,
        label: str,
        confidence: float,
        is_known: bool,
        mobility_class: str = "unknown",
        mobility_confidence: float = 0.0,
        mobility_source: str = "rap",
        object_detail: str = DEFAULT_OBJECT_DETAIL,
    ) -> Optional[Dict[str, Any]]:
        """Attach one RAP label and stored mobility metadata to a slot."""
        tracker = self.tracker
        with tracker._lock:
            track = tracker._tracks.get(str(track_id))
            if track is None:
                return None
            resolved = tracker.semantic_evidence._canonicalise_label(label) if bool(is_known) else ""
            if resolved:
                tracker.semantic_evidence._update_semantics(track, resolved, "rap", float(confidence))
                tracker.semantic_evidence._update_mobility(
                    track,
                    mobility_class=mobility_class,
                    confidence=mobility_confidence,
                    source=mobility_source,
                )
                track.object_detail = str(object_detail or DEFAULT_OBJECT_DETAIL)
            return tracker_serialization.track_record(
                track,
                "rap_semantic_update" if resolved else "rap_unknown",
                "rap_result",
                None,
            )
