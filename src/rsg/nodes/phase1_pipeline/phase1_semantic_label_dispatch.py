"""Semantic-label dispatch: when to send a settled track for RAP/VLM, and
fanning a classification result out to every local Hydra segment it owns.
"""

from __future__ import annotations

from typing import Any, Dict, List

import rclpy
from std_msgs.msg import String

from nodes.support.phase1.json_utils import safe_json_dumps
from nodes.support.phase1.vlm_result import DEFAULT_OBJECT_DETAIL


class Phase1SemanticLabelDispatchStage:
    """Decides when a track is ready for RAP/VLM and publishes the outcome."""

    def __init__(self, coordinator: Any, config: Any, logger: Any):
        self.coordinator = coordinator
        self.config = config
        self.logger = logger

    def _dispatch_tracks_after_settling(
        self,
        current_timestamp_sec: float,
        *,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Schedule settled tracks for RAP using their current best crop.

        The raw crop is not placed into the queue. It remains in the shared
        registry throughout the fixed collection window and may still improve
        while the worker FIFO is waiting. Queue pressure defers an ID instead of
        losing the semantic job.
        """
        coordinator = self.coordinator
        records: List[Dict[str, Any]] = []
        ready = coordinator.persistent_tracker.prepare_active_for_labeling(
            current_timestamp_sec,
            force=force,
        )
        for event in ready:
            track_id = str(event.get("persistent_track_id", ""))
            crop_state = coordinator.crop_registry._describe_track_crop(track_id) if track_id else None
            if not track_id or crop_state is None:
                if track_id:
                    coordinator.persistent_tracker.release_labeling_request(track_id, "missing_representative_crop")
                continue

            if self.config.rap_enabled:
                # The RAP FIFO contains only the persistent track ID.  Do not copy
                # or freeze the crop here: later observations remain eligible until
                # the RAP worker actually dequeues this ID.
                status = coordinator.rap_stage.enqueue_rap_task(track_id)
                if status in {"queued_for_rap", "deferred_for_rap"}:
                    coordinator.persistent_tracker.set_labeling_status(
                        track_id,
                        "rap_queued" if status == "queued_for_rap" else "rap_deferred",
                    )
                    with coordinator._semantic_label_lock:
                        coordinator._semantic_label_pending_track_ids.add(track_id)
                else:
                    coordinator.persistent_tracker.release_labeling_request(track_id, status)
            else:
                # RAP disabled: there is no retrieval short-circuit, so every
                # settled track goes straight to the VLM by ID.  This reuses the
                # exact same crop registry, quality gate, hysteresis and worker
                # diagnostics as the post-RAP VLM path.  The legacy
                # ``unknown_tracker`` dispatch is inert on this branch (its
                # ``_tracks`` map is never populated) and must not be relied on.
                status = coordinator.vlm_stage.enqueue_after_rap(track_id)
                if coordinator.vlm_stage.schedule_accepted(status):
                    with coordinator._semantic_label_lock:
                        coordinator._semantic_label_pending_track_ids.add(track_id)
                else:
                    coordinator.persistent_tracker.release_labeling_request(track_id, status)
            records.append({
                "persistent_track_id": track_id,
                "hydra_slot_id": int(event.get("hydra_slot_id", 0) or 0),
                "status": status,
                "best_crop_score": float(crop_state.get("best_frame_score", 0.0) or 0.0),
                "crop_revision": int(crop_state.get("crop_revision", 0) or 0),
                "crop_timestamp_sec": float(crop_state.get("crop_timestamp_sec", 0.0) or 0.0),
                "last_crop_update_timestamp_sec": float(crop_state.get("last_crop_update_timestamp_sec", 0.0) or 0.0),
                "settling_age_sec": float(event.get("settling_age_sec", 0.0) or 0.0),
                "settle_time_sec": float(event.get("settle_time_sec", 0.0) or 0.0),
                "forced_dispatch": bool(event.get("forced_dispatch", False)),
            })
        return records

    @staticmethod
    def _semantic_segments_for_fanout(payload: Dict[str, Any], task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return every local Hydra segment a classified track owns.

        A track is classified once, but may own several local Hydra slots
        (phase 1 splits a long physical object into multiple segments -- see
        persistent_object_tracker.py). Shared by every consumer that needs to
        fan a single track-level result out to all of that track's segments:
        today that's semantic labeling (below) and risk-result publishing
        (_publish_risk_result). Falls back to one synthetic segment built
        from the top-level hydra_slot_id/hydra_slot_name when the event
        carries no explicit multi-segment list -- the common case, since most
        tracks own exactly one segment.
        """
        segments = list(payload.get("semantic_segments") or [])
        if not segments:
            segments = [{
                "hydra_slot_id": int(payload.get("hydra_slot_id", task.get("hydra_slot_id", 0)) or 0),
                "hydra_slot_name": str(payload.get("hydra_slot_name", task.get("hydra_slot_name", ""))),
                "local_segment_id": str(payload.get("local_segment_id", "")),
                "centroid_3d": payload.get("centroid_3d", (task.get("object_metadata") or {}).get("centroid_3d")),
            }]
        return segments

    def _drain_restored_semantic_labels(
        self, frame: Any, timestamp_sec: float, max_per_frame: int = 4
    ) -> None:
        """Republish restored labels up front, not only on re-observation.

        A resumed session is meant to open on the previous run's map with its
        objects already labelled. Emitting only when a track is re-observed
        leaves every object the robot has not revisited *yet* rendering
        unlabeled -- which on a partial run is most of them, and reads as the
        restored map having lost its semantics.

        The fuser's overlay cache is insert-only, so one emit per track holds
        for the rest of the run. Spread over frames rather than sent in one
        burst: a restored session can carry dozens of tracks, and each emit
        fans out to every segment of its track.
        """
        pending = getattr(self.coordinator, "_restored_label_pending", None)
        if not pending:
            return
        # _emit_restored_semantic_label discards from the pending set, so this
        # drains without needing to mutate the set while iterating it.
        for track_id in sorted(pending)[:max_per_frame]:
            self._emit_restored_semantic_label(track_id, frame, timestamp_sec)

    def _emit_restored_semantic_label(self, track_id: str, frame: Any, timestamp_sec: float) -> None:
        """Republish a restored track's stored label to the fuser, once.

        Restored tracks are deliberately not re-classified -- they already have
        a label, and re-running RAP/VLM would waste inference and risk a worse
        answer. But the label lives only in phase 1's tracker, while the fuser
        keys its overlay cache by Hydra slot id in its own process, which is
        empty on a fresh run. So the label has to be pushed once when the track
        is re-observed, using the same fan-out the RAP/VLM paths use so every
        local segment of a multi-segment track is covered.
        """
        coordinator = self.coordinator
        coordinator._restored_label_pending.discard(track_id)
        track = coordinator.persistent_tracker._tracks.get(track_id)
        if track is None:
            return
        label = str(track.semantic_label or track.canonical_label or "")
        if not label:
            return

        segments = [
            coordinator.persistent_tracker._segment_record(segment)
            for segment in track.segments.values()
        ]
        event = {
            "persistent_track_id": track_id,
            "internal_object_id": track_id,
            "semantic_label": label,
            "canonical_label": track.canonical_label,
            "semantic_label_confidence": float(track.semantic_label_confidence or track.label_confidence or 0.0),
            "mobility_class": track.mobility_class,
            "mobility_confidence": float(track.mobility_confidence or 0.0),
            "mobility_source": track.mobility_source,
            "object_detail": track.object_detail,
            "semantic_segments": segments,
            "hydra_slot_id": int(track.hydra_label_id),
            "hydra_slot_name": str(track.hydra_label_name),
        }
        task = {
            "persistent_track_id": track_id,
            "frame_id": frame.rsg_frame_id,
            "sequence": int(frame.sequence),
            "timestamp_sec": float(timestamp_sec),
        }
        # finalize_track=False: this track was never added to the pending set
        # that _emit_semantic_label_result discards from, since it never went
        # through a dispatch.
        self._emit_semantic_label_result(
            event, task, source="restored_session", finalize_track=False
        )
        self.logger.info(
            f"Republished restored label '{label}' for {track_id} "
            f"across {len(segments)} segment(s)"
        )

    def _emit_semantic_label_result(self, event: Dict[str, Any], task: Dict[str, Any], *, source: str, finalize_track: bool = True) -> None:
        """Publish final class labels for every local semantic slot of one object.

        RAP/VLM still classifies the internal object track once.  If that object
        owns several local Hydra slots, the same class label is emitted once per
        slot so the fuser can label every Hydra node while keeping confidence
        timestamps slot-local.
        """
        coordinator = self.coordinator
        # Do not start or continue semantic fan-out during shutdown.
        if coordinator._stop_event.is_set() or not rclpy.ok():
            return

        payload = dict(event)
        source_name = str(source)
        label = str(payload.get("semantic_label") or payload.get("canonical_label") or "unclassified_object")
        if source_name in {"vlm_failed", "rap_unknown", "rap_error"}:
            label = "unknown_object"
        elif source_name == "vlm_retry_pending":
            # A failed attempt that hasn't exhausted persistent_max_vlm_attempts
            # yet -- object_detail (extracted independent of label_confidence)
            # rides along in payload unchanged, so the fuser can show partial
            # information even while the label itself is still unresolved.
            label = "waiting_for_better_crop"

        segments = self._semantic_segments_for_fanout(payload, task)

        for segment in segments:
            # Abort a partially-started fan-out when ROS shutdown begins.
            if coordinator._stop_event.is_set() or not rclpy.ok():
                break
            slot_id = int(segment.get("hydra_slot_id", segment.get("hydra_label_id", 0)) or 0)
            if slot_id <= 0:
                continue
            segment_payload = dict(payload)
            segment_payload.update({
                "event": "semantic_label_result",
                "source": source_name,
                "persistent_track_id": str(payload.get("persistent_track_id", task.get("persistent_track_id", ""))),
                "internal_object_id": str(payload.get("internal_object_id", payload.get("persistent_track_id", task.get("persistent_track_id", "")))),
                "local_segment_id": str(segment.get("local_segment_id", segment.get("semantic_segment_id", f"slot_{slot_id}"))),
                "semantic_segment_id": str(segment.get("semantic_segment_id", segment.get("local_segment_id", f"slot_{slot_id}"))),
                "hydra_slot_id": slot_id,
                "hydra_slot_name": str(segment.get("hydra_slot_name", segment.get("hydra_label_name", task.get("hydra_slot_name", payload.get("hydra_slot_name", ""))))),
                "frame_id": str(task.get("frame_id", "")),
                "sequence": int(task.get("sequence", 0) or 0),
                "timestamp_sec": float(task.get("timestamp_sec", payload.get("semantic_timestamp_sec", 0.0)) or 0.0),
                "label": label,
                "confidence": float(payload.get("semantic_label_confidence", 0.0) or 0.0),
                "label_confidence": float(payload.get("semantic_label_confidence", 0.0) or 0.0),
                "mobility_class": str(payload.get("mobility_class", "unknown") or "unknown"),
                "mobility_confidence": float(payload.get("mobility_confidence", 0.0) or 0.0),
                "mobility_source": str(payload.get("mobility_source", "none") or "none"),
                "object_detail": str(payload.get("object_detail") or DEFAULT_OBJECT_DETAIL),
                "crop_revision": int(task.get("crop_revision", 0) or 0),
                "vlm_crop_quality_score": float(task.get("vlm_crop_quality_score", 0.0) or 0.0),
                "vlm_crop_quality_eligible": bool(task.get("vlm_crop_quality_eligible", False)),
                "vlm_crop_quality_reasons": list(task.get("vlm_crop_quality_reasons", []) or []),
                "vlm_crop_quality_timeout_forced": bool(task.get("vlm_crop_quality_timeout_forced", False)),
                "vlm_failure_reason": str(task.get("vlm_failure_reason", "")),
                "vlm_raw_response": str(task.get("vlm_raw_response", ""))[:500],
                "centroid_3d": segment.get("centroid_3d", payload.get("centroid_3d", (task.get("object_metadata") or {}).get("centroid_3d"))),
                "centroid_frame_id": str(task.get("centroid_frame_id", "")),
            })
            coordinator._safe_publish(coordinator.semantic_label_result_pub, String(data=safe_json_dumps(segment_payload)))

        track_id = str(payload.get("persistent_track_id", ""))
        if finalize_track and track_id:
            with coordinator._semantic_label_lock:
                coordinator._semantic_label_pending_track_ids.discard(track_id)
            self._finalize_track_queue_state(track_id)
            coordinator.crop_registry._retire_track_crop(track_id)

    def _finalize_track_queue_state(self, track_id: str) -> None:
        """Release scheduler de-duplication after a final semantic outcome."""
        key = str(track_id)
        if not key:
            return
        self.coordinator.rap_stage.finalize_track(key)
        self.coordinator.vlm_stage.finalize_track(key)

    def _finish_unknown_without_vlm(self, track_id: str, task: Dict[str, Any], reason: str) -> None:
        """Publish a terminal unknown state when no VLM work can run."""
        coordinator = self.coordinator
        completed = coordinator.persistent_tracker.complete_semantic_labeling(
            str(track_id), float(task.get("timestamp_sec", 0.0) or 0.0), str(reason)
        )
        if completed is not None:
            self._emit_semantic_label_result(completed, task, source="vlm_failed")
