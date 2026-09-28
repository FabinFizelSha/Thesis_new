"""RAP dispatch: async retrieval-augmented classification for settled tracks.

The bounded worker FIFO stores only persistent track IDs. When it is full,
the ID is retained in a small deferred registry rather than being dropped.
Crops are never held in the queue -- ``_process_rap_task`` snapshots the
track's current best crop at dequeue time.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import rclpy
from std_msgs.msg import String

from nodes.support.phase1.backends import SamMask
from nodes.support.phase1.json_utils import safe_json_dumps
from nodes.support.phase1.vlm_result import DEFAULT_OBJECT_DETAIL, infer_mobility_from_label


class RapDispatchStage:
    """Owns the RAP queue, worker thread, backend, and accuracy diagnostics.

    A RAP miss/error hands off to ``coordinator.vlm_stage`` to queue the
    track for VLM.
    """

    def __init__(self, coordinator: Any, config: Any, logger: Any, *, backend: Any, accuracy_diagnostics: Any):
        self.coordinator = coordinator
        self.config = config
        self.logger = logger
        self.backend = backend
        self.accuracy_diagnostics = accuracy_diagnostics

        self.queue: "queue.Queue[str]" = queue.Queue(maxsize=config.rap_queue_size)
        self.queue_dropped_count = 0
        self.queue_deferred_count = 0
        self.completed_count = 0
        self._task_keys: set = set()
        self._deferred_track_ids = deque()
        self._deferred_track_id_set: set = set()
        self._enqueued_monotonic: Dict[str, float] = {}
        self._task_lock = threading.Lock()

    def finalize_track(self, track_id: str) -> None:
        """Release scheduler de-duplication after a final semantic outcome."""
        key = str(track_id)
        if not key:
            return
        with self._task_lock:
            self._task_keys.discard(key)
            self._deferred_track_id_set.discard(key)
            self._enqueued_monotonic.pop(key, None)

    def _pump_rap_deferred(self) -> None:
        """Move deferred RAP IDs into the bounded FIFO when capacity exists."""
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

    def enqueue_rap_task(self, track_id: str) -> str:
        """Schedule one persistent track for RAP without queuing its crop.

        A bounded FIFO protects the worker, while the deferred registry preserves
        every unique unresolved track ID during temporary overload.
        """
        key = str(track_id)
        if not key:
            return "missing_track_id"
        with self._task_lock:
            if key in self._task_keys:
                return "rap_already_requested_for_track"
            self._task_keys.add(key)
            self._enqueued_monotonic[key] = time.perf_counter()
            try:
                self.queue.put_nowait(key)
                return "queued_for_rap"
            except queue.Full:
                self._deferred_track_ids.append(key)
                self._deferred_track_id_set.add(key)
                self.queue_deferred_count += 1
                return "deferred_for_rap"

    def _rap_loop(self) -> None:
        """Run VisualRAP over the latest crop available at ID dequeue time."""
        coordinator = self.coordinator
        while not coordinator._stop_event.is_set():
            self._pump_rap_deferred()
            try:
                track_id = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._pump_rap_deferred()
            task = coordinator.crop_registry._snapshot_track_task(str(track_id), "rap_dequeue")
            if task is not None:
                coordinator.persistent_tracker.set_labeling_status(str(track_id), "rap_dequeued")
                # Save diagnostic crop for RAP
                try:
                    rap_crop_result = coordinator.tracking_crop_manager.save_rap_dequeue_crop(
                        track_id=str(track_id),
                        rap_crop=task.get("rgb_crop"),
                        crop_revision=int(task.get("crop_revision", 0)),
                        crop_score=float(task.get("crop_score", 0.0)),
                        sequence=int(task.get("sequence", 0)),
                    )
                    if rap_crop_result:
                        self.logger.debug(f"Saved RAP crop for {track_id}: {rap_crop_result}")
                except Exception as e:
                    self.logger.warn(f"Failed to save RAP crop for {track_id}: {e}")
            if task is None:
                fallback = {
                    "persistent_track_id": str(track_id),
                    "hydra_slot_id": 0,
                    "timestamp_sec": 0.0,
                    "object_metadata": {},
                }
                self.logger.warn(f"RAP track {track_id} has no active crop; finalizing as unknown.")
                self._publish_rap_result(fallback, label="unknown_object", confidence=0.0,
                                         is_known=False, status="rap_missing_crop", reason="no_active_crop")
                coordinator._finish_unknown_without_vlm(str(track_id), fallback, "rap_missing_crop")
                continue
            try:
                self._process_rap_task(task)
            except Exception as exc:
                if rclpy.ok() and not coordinator._stop_event.is_set():
                    self.logger.error(
                        f"Async RAP failed for slot={task.get('hydra_slot_id', 0)} "
                        f"track={task.get('persistent_track_id', '')}: {exc}"
                    )
                vlm_status = coordinator.vlm_stage.enqueue_after_rap(str(track_id))
                self._publish_rap_result(task, label="unknown_object", confidence=0.0,
                                         is_known=False, status="rap_error", reason=str(exc),
                                         vlm_dispatch_status=vlm_status)
                if not coordinator.vlm_stage.schedule_accepted(vlm_status):
                    coordinator._finish_unknown_without_vlm(str(track_id), task, "rap_worker_error")

    def _process_rap_task(self, task: Dict[str, Any]) -> str:
        """Run RAP on a dequeue-time snapshot of one track's best crop."""
        coordinator = self.coordinator
        start = time.perf_counter()
        crop = task.get("rgb_crop")
        if crop is None or getattr(crop, "size", 0) == 0:
            raise RuntimeError("Asynchronous RAP task has no representative crop")
        height, width = crop.shape[:2]
        synthetic_mask = SamMask(
            mask_id=str(task.get("candidate_id", "semantic_crop")),
            mask=np.ones((height, width), dtype=bool),
            bbox_2d=[0, 0, int(width), int(height)],
            area_px=int(height * width),
            crop=crop,
            score=1.0,
            metadata={"semantic_track_labeling": True, "crop_revision": task.get("crop_revision", 0)},
        )
        rap = self.backend.classify(crop, synthetic_mask, 0)
        is_known = bool(rap.is_known and rap.confidence >= self.config.rap_confidence_threshold)
        # Direct RAP hit/miss signal, independent of Risk VLM's source column
        # (which requires a separate feature enabled) -- this is the ground
        # truth for whether RAP identified the crop or deferred to VLM.
        self.logger.info(
            f"RAP {'HIT' if is_known else 'MISS'}: track={task.get('persistent_track_id', '?')} "
            f"label='{rap.label}' distance={float(rap.distance):.4f} "
            f"threshold={float(self.config.rap_distance_threshold):.4f} confidence={float(rap.confidence):.3f}"
        )
        try:
            self.accuracy_diagnostics.log_rap_result(
                crop,
                is_known,
                str(rap.label),
                float(rap.distance),
                float(rap.confidence),
                float(self.config.rap_distance_threshold),
                track_id=str(task.get("persistent_track_id", "")),
                hydra_slot_id=int(task.get("hydra_slot_id", 0) or 0),
                timestamp=float(task.get("timestamp_sec", 0.0) or 0.0),
            )
        except Exception as e:
            self.logger.warn(f"Failed to log RAP result diagnostics: {e}")
        label = str(rap.label or "unknown_object")
        rap_metadata = dict(rap.metadata or {})
        rap_has_mobility_metadata = "mobility_class" in rap_metadata
        mobility_class = str(rap_metadata.get("mobility_class", "unknown") or "unknown").strip().lower()
        if mobility_class not in {"static", "dynamic", "unknown"}:
            mobility_class = "unknown"
        try:
            stored_mobility_confidence = float(rap_metadata.get("mobility_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            stored_mobility_confidence = 0.0
        try:
            stored_label_confidence = float(rap_metadata.get("label_confidence", rap.confidence) or 0.0)
        except (TypeError, ValueError):
            stored_label_confidence = float(rap.confidence)
        label_confidence = min(
            max(0.0, min(1.0, stored_label_confidence)),
            max(0.0, min(1.0, float(rap.confidence))),
        )
        mobility_confidence = min(
            max(0.0, min(1.0, stored_mobility_confidence)),
            max(0.0, min(1.0, float(rap.confidence))),
        )
        mobility_source = "rap_memory" if mobility_class != "unknown" else "none"
        object_detail = str(rap_metadata.get("object_detail") or DEFAULT_OBJECT_DETAIL)
        if is_known and mobility_class == "unknown" and not rap_has_mobility_metadata:
            mobility_class = infer_mobility_from_label(
                label,
                dynamic_label_hints=self.config.vlm_dynamic_label_hints,
                static_label_hints=self.config.vlm_static_label_hints,
            )
            if mobility_class != "unknown":
                mobility_confidence = max(0.0, min(1.0, float(rap.confidence)))
                mobility_source = "rap_label_hint"
        track_id = str(task.get("persistent_track_id", ""))
        persistent_update = None
        if self.config.persistent_tracking_enabled and track_id:
            persistent_update = coordinator.persistent_tracker.apply_rap_result(
                track_id=track_id,
                label=label,
                confidence=float(label_confidence),
                is_known=is_known,
                mobility_class=mobility_class,
                mobility_confidence=mobility_confidence,
                mobility_source=mobility_source,
                object_detail=object_detail,
            )

        vlm_status = "not_requested"
        if is_known and track_id:
            completed = coordinator.persistent_tracker.complete_semantic_labeling(
                track_id, float(task.get("timestamp_sec", 0.0) or 0.0), "rap_known"
            )
            if completed is not None:
                coordinator._emit_semantic_label_result(completed, task, source="rap")
                coordinator.risk_stage.enqueue_risk_task(
                    event=completed,
                    task=task,
                    crop=task.get("vlm_rgb_crop", task.get("rgb_crop")),
                    label=label,
                    mobility_class=mobility_class,
                    source="rap",
                )
        elif not is_known:
            vlm_status = coordinator.vlm_stage.enqueue_after_rap(track_id)
            if not coordinator.vlm_stage.schedule_accepted(vlm_status):
                coordinator._finish_unknown_without_vlm(track_id, task, "rap_unknown_vlm_unavailable")

        self.completed_count += 1
        self._publish_rap_result(
            task,
            label=label,
            confidence=float(rap.confidence),
            label_confidence=float(label_confidence),
            is_known=is_known,
            status="known" if is_known else "unknown",
            reason="async_retrieval_complete",
            rap_metadata=rap_metadata,
            mobility_class=mobility_class,
            mobility_confidence=mobility_confidence,
            mobility_source=mobility_source,
            persistent_update=persistent_update,
            vlm_dispatch_status=vlm_status,
            rap_delay_ms=(time.perf_counter() - start) * 1000.0,
        )
        return "completed"

    def _publish_rap_result(
        self,
        task: Dict[str, Any],
        *,
        label: str,
        confidence: float,
        label_confidence: Optional[float] = None,
        is_known: bool,
        status: str,
        reason: str,
        rap_metadata: Optional[Dict[str, Any]] = None,
        mobility_class: str = "unknown",
        mobility_confidence: float = 0.0,
        mobility_source: str = "none",
        persistent_update: Optional[Dict[str, Any]] = None,
        vlm_dispatch_status: str = "",
        rap_delay_ms: float = 0.0,
    ) -> None:
        """Publish the asynchronous RAP contract: stable slot ID and label."""
        payload = {
            "event": "rap_result",
            "status": str(status),
            "reason": str(reason),
            "persistent_track_id": str(task.get("persistent_track_id", "")),
            "hydra_slot_id": int(task.get("hydra_slot_id", 0) or 0),
            "hydra_slot_name": str(task.get("hydra_slot_name", "")),
            "candidate_id": str(task.get("candidate_id", "")),
            "frame_id": str(task.get("frame_id", "")),
            "sequence": int(task.get("sequence", 0) or 0),
            "timestamp_sec": float(task.get("timestamp_sec", 0.0) or 0.0),
            "label": str(label),
            "confidence": float(confidence),
            "label_confidence": float(confidence if label_confidence is None else label_confidence),
            "retrieval_confidence": float(confidence),
            "mobility_class": str(mobility_class),
            "mobility_confidence": float(mobility_confidence),
            "mobility_source": str(mobility_source),
            "is_known": bool(is_known),
            "rap_delay_ms": float(rap_delay_ms),
            "vlm_dispatch_status": str(vlm_dispatch_status),
            "rap_metadata": rap_metadata or {},
            "persistent_update": persistent_update or {},
        }
        self.coordinator._safe_publish(self.coordinator.rap_result_pub, String(data=safe_json_dumps(payload)))
