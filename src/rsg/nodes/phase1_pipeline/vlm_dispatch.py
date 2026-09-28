"""VLM dispatch: object-detection VLM scheduling, quality gating, and retry.

RAP/VLM scheduling is intentionally decoupled from Hydra output. A persistent
track receives a unique Hydra slot immediately, while this stage receives
only its track ID. The current best crop remains mutable until the VLM
worker dequeues that ID; an immutable crop snapshot is then used for that
worker's single inference request.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy

from rsg.msg import Phase1VlmResult
from nodes.phase1_pipeline.crop_utils import extract_crop_with_context
from nodes.support.phase1.vlm_result import DEFAULT_OBJECT_DETAIL
from nodes.support.phase1.json_utils import safe_json_dumps


class VlmDispatchStage:
    """Owns the VLM queue, worker thread, backend, and quality/retry pools."""

    def __init__(self, coordinator: Any, config: Any, logger: Any, *, backend: Any, test_diagnostics: Any):
        self.coordinator = coordinator
        self.config = config
        self.logger = logger
        self.backend = backend
        self.test_diagnostics = test_diagnostics

        # VLM uses the same ID-only/deferred scheduling policy as RAP. A
        # legacy dictionary task remains supported only for the RAP-disabled
        # fallback path; normal RAP-enabled operation always queues a track ID.
        self.queue: "queue.Queue[Any]" = queue.Queue(maxsize=config.vlm_queue_size)
        self.queue_dropped_count = 0
        self.queue_deferred_count = 0
        self._task_keys: set = set()
        self._deferred_track_ids = deque()
        self._deferred_track_id_set: set = set()
        self._enqueued_monotonic: Dict[str, float] = {}
        self._task_lock = threading.Lock()

        # Tracks that reached RAP but still have a weak VLM crop are retained
        # outside the VLM FIFO while later observations may improve the crop.
        # The defer-start timestamp uses recorded bag time. A bounded timeout
        # prevents small or partly visible real objects from waiting forever.
        self._quality_deferred_track_ids: set = set()
        self._quality_deferred_since_timestamp_sec: Dict[str, float] = {}
        self._quality_force_track_ids: set = set()
        self._quality_deferred_lock = threading.Lock()

        # Tracks that failed VLM (low label_confidence) but have not yet
        # exhausted persistent_max_vlm_attempts. Unlike the quality-deferred
        # pool above, there is no timeout here -- a track waits indefinitely
        # for a strictly better crop than the one its last attempt used.
        self._retry_waiting_track_ids: set = set()
        self._retry_last_attempt_score: Dict[str, float] = {}
        self._retry_lock = threading.Lock()

        # Exclusively a VLM counter -- every increment site is inside this
        # stage's own dispatch methods.
        self.queued_count = 0

    def finalize_track(self, track_id: str) -> None:
        """Release scheduler de-duplication after a final semantic outcome."""
        key = str(track_id)
        if not key:
            return
        with self._task_lock:
            self._task_keys.discard(key)
            self._deferred_track_id_set.discard(key)
            self._enqueued_monotonic.pop(key, None)
        with self._quality_deferred_lock:
            self._quality_deferred_track_ids.discard(key)
            self._quality_deferred_since_timestamp_sec.pop(key, None)
            self._quality_force_track_ids.discard(key)

    def clear_retry_state(self, track_id: str) -> None:
        """Drop a track from the post-failure retry-waiting pool, if present."""
        key = str(track_id)
        with self._retry_lock:
            self._retry_waiting_track_ids.discard(key)
            self._retry_last_attempt_score.pop(key, None)

    def is_quality_force(self, track_id: str) -> bool:
        """Whether this track's next dequeue snapshot was force-released by timeout."""
        key = str(track_id)
        with self._quality_deferred_lock:
            return bool(key in self._quality_force_track_ids)

    def _pump_vlm_deferred(self) -> None:
        """Move deferred VLM IDs into the bounded FIFO when capacity exists."""
        with self._task_lock:
            while self._deferred_track_ids:
                track_id = str(self._deferred_track_ids[0])
                if track_id not in self._task_keys or track_id not in self._deferred_track_id_set:
                    self._deferred_track_ids.popleft()
                    self._deferred_track_id_set.discard(track_id)
                    continue
                try:
                    self.queue.put_nowait(track_id)
                except queue.Full:
                    return
                self._deferred_track_ids.popleft()
                self._deferred_track_id_set.discard(track_id)

    def _is_vlm_crop_eligible(self, task: Dict[str, Any], track_id: str = "") -> bool:
        """Return whether a normal VLM request has a sufficiently useful crop."""
        key = str(track_id or task.get("persistent_track_id", ""))
        with self._quality_deferred_lock:
            if key and key in self._quality_force_track_ids:
                return True
        return bool(task.get("vlm_crop_quality_eligible", (task.get("object_metadata") or {}).get("vlm_crop_quality_eligible", True)))

    def _defer_vlm_for_better_crop(
        self,
        track_id: str,
        reason: str,
        *,
        current_timestamp_sec: Optional[float] = None,
    ) -> str:
        """Hold one RAP-unknown track for a better crop, with a bounded wait.

        The timer is measured in recorded bag time so its behaviour is stable
        across rosbag playback rates. The ID remains outside the VLM FIFO while
        the crop registry is still mutable; a later eligible crop resumes it
        immediately, otherwise the timeout releases the best available crop.
        """
        key = str(track_id)
        if not key:
            return "missing_track_id"
        defer_timestamp_sec = float(
            self.coordinator._latest_processed_timestamp_sec
            if current_timestamp_sec is None
            else current_timestamp_sec
        )
        with self._task_lock:
            self._task_keys.discard(key)
            self._deferred_track_id_set.discard(key)
            self._enqueued_monotonic.pop(key, None)
        with self._quality_deferred_lock:
            self._quality_deferred_track_ids.add(key)
            self._quality_deferred_since_timestamp_sec.setdefault(key, defer_timestamp_sec)
        self.coordinator.persistent_tracker.set_labeling_status(key, "vlm_waiting_for_better_crop")
        return str(reason)

    def _resume_quality_deferred_vlm_if_ready(self, track_id: str) -> None:
        """Queue a deferred track as soon as a stronger crop satisfies the VLM gate."""
        key = str(track_id)
        with self._quality_deferred_lock:
            if key not in self._quality_deferred_track_ids:
                return
        crop_state = self.coordinator.crop_registry._describe_track_crop(key)
        if crop_state is None or not bool(crop_state.get("vlm_crop_quality_eligible", False)):
            return
        with self._quality_deferred_lock:
            self._quality_deferred_track_ids.discard(key)
            self._quality_deferred_since_timestamp_sec.pop(key, None)
        status = self.enqueue_vlm_track(key)
        if status in {"queued_for_vlm_fifo", "deferred_for_vlm"}:
            self.coordinator.persistent_tracker.set_labeling_status(
                key,
                "vlm_queued_after_crop_quality" if status == "queued_for_vlm_fifo" else "vlm_deferred_after_crop_quality",
            )

    def _resume_vlm_retry_if_better_crop(self, track_id: str) -> None:
        """Re-queue a track that failed VLM once a strictly better crop arrives.

        Unlike _resume_quality_deferred_vlm_if_ready (a pre-first-attempt
        minimum-quality gate with a bounded timeout), this waits indefinitely:
        a track only leaves this pool by beating the crop score its last
        (failed) VLM attempt used, or by exhausting persistent_max_vlm_attempts
        on a later attempt (handled in the VLM worker, not here).
        """
        key = str(track_id)
        with self._retry_lock:
            if key not in self._retry_waiting_track_ids:
                return
            last_score = float(self._retry_last_attempt_score.get(key, 0.0))
        crop_state = self.coordinator.crop_registry._describe_track_crop(key)
        if crop_state is None:
            return
        current_score = float(crop_state.get("best_frame_score", 0.0) or 0.0)
        if current_score <= last_score:
            return
        with self._retry_lock:
            self._retry_waiting_track_ids.discard(key)
        status = self.enqueue_vlm_track(key)
        if status in {"queued_for_vlm_fifo", "deferred_for_vlm"}:
            self.coordinator.persistent_tracker.set_labeling_status(key, "vlm_queued_after_retry_crop_improved")

    def _release_quality_deferred_vlm_if_expired(self, current_timestamp_sec: float) -> None:
        """Release weak crops after the bounded post-RAP collection interval.

        A track keeps accepting better crops until this method queues its ID.
        It is called once for each processed frame, so the deadline is driven by
        bag time even when no later observation of the deferred object arrives.
        """
        if not bool(self.config.vlm_crop_quality_force_on_timeout):
            return
        max_wait_sec = max(0.0, float(self.config.vlm_crop_quality_max_wait_sec))
        now_sec = float(current_timestamp_sec)
        due_track_ids: List[str] = []
        with self._quality_deferred_lock:
            for track_id in list(self._quality_deferred_track_ids):
                deferred_since_sec = float(
                    self._quality_deferred_since_timestamp_sec.get(track_id, now_sec)
                )
                if max(0.0, now_sec - deferred_since_sec) < max_wait_sec:
                    continue
                self._quality_deferred_track_ids.discard(track_id)
                self._quality_deferred_since_timestamp_sec.pop(track_id, None)
                self._quality_force_track_ids.add(track_id)
                due_track_ids.append(track_id)

        for track_id in due_track_ids:
            status = self.enqueue_vlm_track(track_id)
            if status in {"queued_for_vlm_fifo", "deferred_for_vlm"}:
                self.coordinator.persistent_tracker.set_labeling_status(
                    track_id,
                    "vlm_queued_after_quality_timeout"
                    if status == "queued_for_vlm_fifo"
                    else "vlm_deferred_after_quality_timeout",
                )
                crop_state = self.coordinator.crop_registry._describe_track_crop(track_id) or {}
                self.record_vlm_queue_event(
                    event="quality_timeout_force",
                    task={
                        "persistent_track_id": track_id,
                        "unknown_track_id": track_id,
                        "crop_revision": int(crop_state.get("crop_revision", 0) or 0),
                        "best_frame_score": float(crop_state.get("best_frame_score", 0.0) or 0.0),
                    },
                    queue_wait_ms=0.0,
                    reason="vlm_crop_quality_timeout",
                )

    def enqueue_vlm_track(self, track_id: str) -> str:
        """Schedule one unresolved persistent track for VLM by ID only."""
        key = str(track_id)
        if not key:
            return "missing_track_id"
        with self._task_lock:
            if key in self._task_keys:
                return "vlm_already_requested_for_track"
            self._task_keys.add(key)
            self._enqueued_monotonic[key] = time.perf_counter()
            try:
                self.queue.put_nowait(key)
                status = "queued_for_vlm_fifo"
            except queue.Full:
                self._deferred_track_ids.append(key)
                self._deferred_track_id_set.add(key)
                self.queue_deferred_count += 1
                status = "deferred_for_vlm"
        # The legacy unknown tracker may not own this persistent-track ID; this
        # call is harmless in that case and keeps fallback diagnostics coherent.
        self.coordinator.unknown_tracker.mark_vlm_queued(key)
        self.queued_count += 1
        return status

    def enqueue_after_rap(self, track_id: str) -> str:
        """Queue an unresolved track only when its current VLM crop is useful.

        The track ID, rather than an image payload, remains the queued unit.
        Weak fragments are retained for later observations and resume as soon as
        the shared best crop reaches the quality threshold.
        """
        if not self.config.vlm_enabled:
            return "vlm_disabled"
        key = str(track_id)
        crop_state = self.coordinator.crop_registry._describe_track_crop(key)
        if crop_state is None:
            return "vlm_missing_crop"
        if not bool(crop_state.get("vlm_crop_quality_eligible", False)):
            return self._defer_vlm_for_better_crop(
                key,
                "deferred_for_better_crop",
                current_timestamp_sec=float(self.coordinator._latest_processed_timestamp_sec),
            )
        status = self.enqueue_vlm_track(key)
        if status in {"queued_for_vlm_fifo", "deferred_for_vlm"}:
            self.coordinator.persistent_tracker.set_labeling_status(
                key,
                "vlm_queued" if status == "queued_for_vlm_fifo" else "vlm_deferred",
            )
        return status

    @staticmethod
    def schedule_accepted(status: str) -> bool:
        return str(status) in {
            "queued_for_vlm_fifo",
            "deferred_for_vlm",
            "deferred_for_better_crop",
            "vlm_already_requested_for_track",
        }

    def dispatch_unknowns_to_vlm(
        self,
        frame: Any,
        rgb: np.ndarray,
        depth: np.ndarray,
        unknowns: List[Dict[str, Any]],
        classified: List[Any],
    ) -> List[Dict[str, Any]]:
        """Queue one VLM request per persistent unknown track.

        Legacy per-frame dispatch path, currently unreachable: normal
        operation drives VLM dispatch entirely through
        ``Phase1SemanticCoordinator._dispatch_tracks_after_settling``
        (RAP-enabled: RAP miss -> ``enqueue_after_rap``; RAP-disabled:
        straight to this stage's ``enqueue_vlm_track``), for both
        RAP-enabled and RAP-disabled operation. Kept for the RAP-disabled
        legacy ``unknown_tracker``-based fallback it was written for.

        A frame may contain several unknown detections, and the same physical
        unknown may appear in many consecutive frames. The tracker assigns the
        persistent ``unknown_track_id`` and this function only queues the VLM
        task when the track is ready and has not already been queued/done.
        """
        coordinator = self.coordinator
        dispatch_records: List[Dict[str, Any]] = []
        mask_lookup = {item.candidate_id: item for item in classified}
        image_area_px = int(rgb.shape[0] * rgb.shape[1]) if rgb.ndim >= 2 else None

        for unknown in unknowns:
            candidate_id = str(unknown.get("candidate_id", ""))
            classified_mask = mask_lookup.get(candidate_id)
            bbox_2d = unknown.get("bbox_2d")
            _, context_bbox_2d = extract_crop_with_context(
                rgb,
                bbox_2d,
                context_ratio=float(self.config.vlm_crop_context_ratio),
            )
            rgb_crop = coordinator.sem_stage.build_vlm_crop(
                rgb,
                None if classified_mask is None else classified_mask.mask,
                context_bbox_2d,
            )

            if not self.config.vlm_enabled:
                dispatch_info = {"vlm_dispatch_status": "vlm_disabled", "best_frame_score": 0.0}
                task = None
            else:
                task, dispatch_info = coordinator.unknown_tracker.update_evidence_and_build_vlm_task(
                    unknown=unknown,
                    rgb_crop=rgb_crop,
                    frame_header=frame.header,
                    frame_id=frame.rsg_frame_id,
                    sequence=int(frame.sequence),
                    image_area_px=image_area_px,
                )

            # Store VLM crop for later analysis (saved once tracking assigns track_id)
            if rgb_crop is not None and rgb_crop.size > 0:
                if not hasattr(coordinator, '_current_frame_vlm_crops'):
                    coordinator._current_frame_vlm_crops = {}
                unknown_track_id = str(unknown.get("unknown_track_id", ""))
                if unknown_track_id:
                    coordinator._current_frame_vlm_crops[unknown_track_id] = {
                        "crop": rgb_crop.copy(),
                        "quality_score": float(dispatch_info.get("vlm_crop_quality_score", 0.0) or 0.0),
                    }

            dispatch_status = str(dispatch_info.get("vlm_dispatch_status", "not_queued"))
            if task is not None:
                dispatch_status = self.enqueue_vlm_task(task, dispatch_status)

            record = {
                "candidate_id": candidate_id,
                "unknown_track_id": str(unknown.get("unknown_track_id", "")),
                "status": dispatch_status,
                "track_seen_count": int(unknown.get("track_seen_count", 1) or 1),
                "best_frame_score": float(dispatch_info.get("best_frame_score", 0.0) or 0.0),
            }
            dispatch_records.append(record)

        return dispatch_records

    def enqueue_vlm_task(self, task: Dict[str, Any], previous_status: str) -> str:
        """Compatibility enqueue path for RAP-disabled legacy unknown tracking.

        Normal RAP-enabled operation uses :meth:`enqueue_vlm_track` and stores
        only a track ID.  This path remains bounded for compatibility with older
        launch configurations that bypass RAP.
        """
        track_id = str(task.get("unknown_track_id", ""))
        task["enqueued_monotonic"] = time.perf_counter()
        try:
            self.queue.put_nowait(task)
            self.coordinator.unknown_tracker.mark_vlm_queued(track_id)
            self.queued_count += 1
            self.record_vlm_queue_event(event="enqueued", task=task, queue_wait_ms=0.0, reason=previous_status)
            return "queued_for_vlm_fifo"
        except queue.Full:
            # Preserve the old non-RAP fallback behavior. The requested
            # no-drop guarantee applies to the RAP-enabled ID-only scheduler.
            self.queue_dropped_count += 1
            self.coordinator.unknown_tracker.mark_vlm_queue_rejected(track_id, reason="vlm_fifo_queue_full")
            self.record_vlm_queue_event(event="dropped_queue_full", task=task, queue_wait_ms=0.0, reason=self.config.vlm_queue_drop_policy)
            return "vlm_fifo_queue_full_dropped"

    def record_vlm_queue_event(self, event: str, task: Dict[str, Any], queue_wait_ms: float = 0.0, reason: str = "") -> None:
        """Record VLM queue latency when timing diagnostics are enabled."""
        if not self.config.timing_enabled:
            return
        self.coordinator.timing_recorder.add_sample(
            # Wall-clock -- see the matching comment in publish_timing_event.
            # This is the row the revisit/FPS experiment counts VLM calls
            # from (event == "completed"), so its own timestamp avoids having
            # to cross-reference the publish_vlm_timing row from the same
            # call for a VLM-calls-over-time plot.
            wall_clock_unix_sec=time.time(),
            node="rsg_object_detection",
            event=event,
            sequence=int(task.get("sequence", 0) or 0),
            frame_id=str(task.get("rsg_frame_id", "")),
            unknown_track_id=str(task.get("unknown_track_id", "")),
            candidate_id=str(task.get("candidate_id", "")),
            queue_size=int(self.queue.qsize()),
            queue_max_size=int(self.config.vlm_queue_size),
            queue_wait_ms=float(queue_wait_ms),
            track_seen_count=int(task.get("track_seen_count", 0) or 0),
            best_frame_score=float(task.get("best_frame_score", 0.0) or 0.0),
            reason=reason,
        )

    def _vlm_loop(self) -> None:
        """Run VLM on the latest crop available when its track ID is dequeued."""
        coordinator = self.coordinator
        while not coordinator._stop_event.is_set():
            self._pump_vlm_deferred()
            try:
                queued_item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._pump_vlm_deferred()

            is_track_id_task = isinstance(queued_item, str)
            track_id = str(queued_item) if is_track_id_task else str(queued_item.get("unknown_track_id", queued_item.get("persistent_track_id", "")))
            if is_track_id_task:
                task = coordinator.crop_registry._snapshot_track_task(track_id, "vlm_dequeue")
                if task is not None:
                    coordinator.persistent_tracker.set_labeling_status(track_id, "vlm_dequeued")
                    # Save diagnostic crop for VLM
                    try:
                        vlm_crop_result = coordinator.tracking_crop_manager.save_vlm_dequeue_crop(
                            track_id=track_id,
                            vlm_crop=task.get("vlm_rgb_crop", task.get("rgb_crop")),
                            crop_revision=int(task.get("crop_revision", 0)),
                            crop_score=float(task.get("crop_score", 0.0)),
                            sequence=int(task.get("sequence", 0)),
                        )
                        if vlm_crop_result:
                            self.logger.debug(f"Saved VLM crop for {track_id}: {vlm_crop_result}")
                    except Exception as e:
                        self.logger.warn(f"Failed to save VLM crop for {track_id}: {e}")
                if task is None:
                    fallback = {
                        "persistent_track_id": track_id,
                        "unknown_track_id": track_id,
                        "hydra_slot_id": 0,
                        "timestamp_sec": 0.0,
                        "object_metadata": {},
                    }
                    self.logger.warn(f"VLM track {track_id} has no active crop; finalizing as unknown.")
                    coordinator._finish_unknown_without_vlm(track_id, fallback, "vlm_missing_crop")
                    continue
                task["unknown_track_id"] = track_id
                if not self._is_vlm_crop_eligible(task, track_id):
                    self._defer_vlm_for_better_crop(
                        track_id,
                        "deferred_for_better_crop",
                        current_timestamp_sec=float(coordinator._latest_processed_timestamp_sec),
                    )
                    self.record_vlm_queue_event(
                        event="quality_deferred",
                        task=task,
                        queue_wait_ms=0.0,
                        reason="vlm_crop_quality_gate",
                    )
                    continue
            else:
                # Compatibility only for RAP-disabled legacy dispatch.
                task = dict(queued_item)
                track_id = str(task.get("unknown_track_id", task.get("persistent_track_id", "")))

            start = time.perf_counter()
            queue_wait_ms = max(0.0, (start - float(task.get("created_monotonic", start))) * 1000.0)
            self.record_vlm_queue_event(event="dequeued", task=task, queue_wait_ms=queue_wait_ms, reason="fifo_order")
            try:
                result = self.backend.identify(
                    task.get("vlm_rgb_crop", task.get("rgb_crop")),
                    task.get("object_metadata", {}),
                )
            except Exception as exc:
                # VLM availability is not allowed to strand a unique slot in
                # the pending state.  Publish an explicit unknown-object result
                # and allow the fuser to retain the stable physical slot.
                result = {
                    "success": False,
                    "label": "unknown_object",
                    "confidence": 0.0,
                    "label_confidence": 0.0,
                    "mobility_class": "unknown",
                    "mobility_confidence": 0.0,
                    "backend": self.config.vlm_mode,
                    "model": self.config.vlm_model,
                    "raw_response": f"vlm_error: {exc}",
                    "failure_reason": f"worker_exception:{type(exc).__name__}",
                    "validation_status": "rejected",
                    "validation_reason": f"worker_exception:{type(exc).__name__}",
                }
                if rclpy.ok() and not coordinator._stop_event.is_set():
                    self.logger.error(f"VLM failed for track={track_id}: {exc}")

            # Log VLM result for testing diagnostics
            vlm_processing_time_ms = (time.perf_counter() - start) * 1000.0
            try:
                crop_rgb = task.get("vlm_rgb_crop", task.get("rgb_crop"))
                self.test_diagnostics.log_vlm_result(
                    crop_rgb=crop_rgb,
                    vlm_output=result,
                    end_to_end_ms=vlm_processing_time_ms,
                    inference_ms=float(result.get("vlm_inference_ms", 0.0) or 0.0),
                    timestamp=float(task.get("timestamp_sec", 0.0)),
                    track_id=track_id,
                    hydra_label_id=coordinator.persistent_tracker.get_hydra_label_id(track_id),
                )
            except Exception as e:
                self.logger.warn(f"Failed to log VLM test result: {e}")

            # The VLM label must be paired with the exact immutable crop that
            # was supplied at VLM dequeue.  Do not replace it with a later crop
            # observed during inference: that later crop was not classified by
            # this request and could contaminate RAP memory with a mismatched
            # label/image pair.
            memory_task = task
            memory_metadata = dict(memory_task.get("object_metadata") or {})
            memory_metadata.update({
                "rsg_slot_id": int(memory_task.get("hydra_slot_id", memory_metadata.get("hydra_label_id", 0)) or 0),
                "hydra_slot_id": int(memory_task.get("hydra_slot_id", memory_metadata.get("hydra_label_id", 0)) or 0),
                "persistent_track_id": track_id,
                "crop_revision": int(memory_task.get("crop_revision", 0) or 0),
                "crop_score": float(memory_task.get("best_frame_score", 0.0) or 0.0),
                "memory_crop_stage": "best_available_when_vlm_completed",
                "label_confidence": float(result.get("label_confidence", result.get("confidence", 0.0)) or 0.0),
                "mobility_class": str(result.get("mobility_class", "unknown") or "unknown"),
                "mobility_confidence": float(result.get("mobility_confidence", 0.0) or 0.0),
                "mobility_source": "vlm",
                "object_detail": str(result.get("object_detail") or DEFAULT_OBJECT_DETAIL),
            })
            rap_update_status = coordinator.rap_memory_updater.update_from_vlm(result, memory_metadata)
            try:
                if self.config.rap_update_enabled and bool(result.get("success", False)) and float(result.get("confidence", 0.0)) >= float(self.config.rap_update_min_confidence):
                    label_for_rap = str(result.get("label", "")).replace("_", " ").strip()
                    memory_crop = memory_task.get("rgb_crop")
                    rap_backend = coordinator.rap_stage.backend
                    if label_for_rap and memory_crop is not None and hasattr(rap_backend, "add_image"):
                        rap_backend.add_image(memory_crop, label_for_rap, metadata={
                            "rsg_slot_id": memory_metadata["rsg_slot_id"],
                            "hydra_slot_id": memory_metadata["hydra_slot_id"],
                            "persistent_track_id": track_id,
                            "crop_revision": memory_metadata["crop_revision"],
                            "source": "vlm_label_with_rap_target_only_crop_at_dequeue",
                            "label_confidence": memory_metadata["label_confidence"],
                            "mobility_class": memory_metadata["mobility_class"],
                            "mobility_confidence": memory_metadata["mobility_confidence"],
                            "mobility_source": "vlm",
                            "object_detail": memory_metadata["object_detail"],
                        })
                        rap_update_status["rap_memory_live_update"] = "added_to_visual_rap"
                        rap_update_status["memory_slot_id"] = memory_metadata["rsg_slot_id"]
                        rap_update_status["memory_crop_revision"] = memory_metadata["crop_revision"]
                        self.logger.info(
                            f"RAP memory grew: track={track_id} label='{label_for_rap}' "
                            f"confidence={float(result.get('confidence', 0.0)):.2f}"
                        )
                    else:
                        self.logger.debug(
                            f"RAP live memory update skipped for track={track_id}: "
                            f"label_for_rap={label_for_rap!r} memory_crop_present="
                            f"{memory_crop is not None} has_add_image="
                            f"{hasattr(rap_backend, 'add_image')}"
                        )
            except Exception as exc:
                rap_update_status["rap_memory_live_update"] = "failed"
                rap_update_status["live_update_error"] = str(exc)
                # Previously swallowed with no trace: the Chroma collection RAP
                # queries against stayed empty across many real sessions with
                # update_memory_from_vlm=true, and there was no way to tell
                # why -- this failure carried no console/log signal at all.
                self.logger.error(
                    f"RAP live memory update failed for track={track_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
            result["rap_update"] = rap_update_status
            result["memory_slot_id"] = memory_metadata.get("rsg_slot_id", 0)
            result["memory_crop_revision"] = memory_metadata.get("crop_revision", 0)

            vlm_delay_ms = (time.perf_counter() - start) * 1000.0
            total_age_ms = (time.perf_counter() - float(task.get("created_monotonic", start))) * 1000.0
            msg = Phase1VlmResult()
            msg.header = task["frame_header"]
            msg.rsg_frame_id = str(task["rsg_frame_id"])
            msg.sequence = int(task["sequence"])
            msg.candidate_id = str(task["candidate_id"])
            msg.unknown_track_id = track_id
            msg.mask_id = str(task["mask_id"])
            msg.success = bool(result.get("success", False))
            msg.status = "vlm_done" if msg.success else "vlm_failed"
            msg.reason = "ok" if msg.success else str(result.get("raw_response", "vlm_failed"))
            msg.predicted_label = str(result.get("label", "unknown_object")) if msg.success else "unknown_object"
            msg.confidence = float(result.get("confidence", 0.0)) if msg.success else 0.0
            msg.label_confidence = float(result.get("label_confidence", msg.confidence)) if msg.success else 0.0
            msg.mobility_class = str(result.get("mobility_class", "unknown")) if msg.success else "unknown"
            msg.mobility_confidence = float(result.get("mobility_confidence", 0.0)) if msg.success else 0.0
            msg.object_detail = str(result.get("object_detail", DEFAULT_OBJECT_DETAIL))
            msg.backend = str(result.get("backend", self.config.vlm_mode))
            msg.model = str(result.get("model", self.config.vlm_model))
            msg.vlm_delay_ms = float(vlm_delay_ms)
            msg.total_age_ms = float(total_age_ms)
            msg.object_metadata_json = safe_json_dumps(memory_metadata)
            result["unknown_track_id"] = msg.unknown_track_id
            result["track_seen_count"] = int(memory_task.get("track_seen_count", task.get("track_seen_count", 0)) or 0)
            result["best_frame_score"] = float(memory_task.get("best_frame_score", task.get("best_frame_score", 0.0)) or 0.0)
            memory_task["vlm_failure_reason"] = str(result.get("failure_reason", ""))
            memory_task["vlm_raw_response"] = str(result.get("raw_response", ""))
            memory_task["mobility_class"] = msg.mobility_class
            memory_task["mobility_confidence"] = float(msg.mobility_confidence)
            memory_task["mobility_source"] = "vlm" if msg.success else "none"
            msg.vlm_metadata_json = safe_json_dumps(result)
            coordinator.unknown_tracker.mark_vlm_result(msg.unknown_track_id, result)
            if self.config.persistent_tracking_enabled:
                persistent_update = None
                semantic_task = memory_task if is_track_id_task else dict(task.get("semantic_label_task") or task)
                if msg.success:
                    persistent_update = coordinator.persistent_tracker.apply_vlm_result(
                        msg.unknown_track_id,
                        msg.predicted_label,
                        msg.confidence,
                        msg.mobility_class,
                        msg.mobility_confidence,
                        msg.object_detail,
                    )
                    completed = coordinator.persistent_tracker.complete_semantic_labeling(
                        msg.unknown_track_id,
                        float(semantic_task.get("timestamp_sec", 0.0) or 0.0),
                        "vlm_known",
                    )
                    if completed is not None:
                        coordinator._emit_semantic_label_result(completed, semantic_task, source="vlm")
                        coordinator.risk_stage.enqueue_risk_task(
                            event=completed,
                            task=semantic_task,
                            crop=task.get("vlm_rgb_crop", task.get("rgb_crop")),
                            label=msg.predicted_label,
                            mobility_class=msg.mobility_class,
                            source="vlm",
                        )
                        persistent_update = completed
                else:
                    # object_detail is extracted unconditionally in
                    # validate_vlm_response, independent of label_confidence --
                    # a rejected label can still carry a real description, so
                    # keep it even though the label itself isn't trustworthy.
                    coordinator.persistent_tracker.record_vlm_attempt_detail(
                        msg.unknown_track_id, msg.object_detail,
                    )
                    attempt_count = coordinator.persistent_tracker.increment_vlm_attempt_count(
                        msg.unknown_track_id,
                    )
                    max_attempts = int(self.config.persistent_max_vlm_attempts)
                    if attempt_count is None or attempt_count >= max_attempts:
                        completed = coordinator.persistent_tracker.complete_semantic_labeling(
                            msg.unknown_track_id,
                            float(semantic_task.get("timestamp_sec", 0.0) or 0.0),
                            "vlm_failed",
                        )
                        if completed is not None:
                            coordinator._emit_semantic_label_result(completed, semantic_task, source="vlm_failed")
                            persistent_update = completed
                    else:
                        # Attempts remain: leave the track open (is_semantic_
                        # labeling_open keeps crop updates running) and wait
                        # indefinitely in a non-FIFO pool for a strictly
                        # better crop than the one this attempt used, rather
                        # than finalizing now.
                        waiting_record = coordinator.persistent_tracker.get_waiting_record(msg.unknown_track_id)
                        if waiting_record is not None:
                            coordinator._emit_semantic_label_result(
                                waiting_record, semantic_task, source="vlm_retry_pending",
                                finalize_track=False,
                            )
                        coordinator._finalize_track_queue_state(track_id)
                        with self._retry_lock:
                            self._retry_last_attempt_score[track_id] = float(
                                task.get("crop_score", 0.0) or 0.0
                            )
                            self._retry_waiting_track_ids.add(track_id)
                        coordinator.persistent_tracker.set_labeling_status(
                            msg.unknown_track_id, "vlm_retry_pending",
                        )
                if persistent_update is not None:
                    result["persistent_track_update"] = persistent_update
                    msg.vlm_metadata_json = safe_json_dumps(result)
            coordinator._safe_publish(coordinator.vlm_result_pub, msg)
            coordinator.publish_vlm_timing(msg)
            self.record_vlm_queue_event(event="completed", task=task, queue_wait_ms=queue_wait_ms, reason=msg.status)
