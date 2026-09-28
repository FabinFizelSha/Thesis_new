"""Risk-VLM dispatch: one-shot risk assessment for tracks that just got a label."""

from __future__ import annotations

import queue
import time
from typing import Any, Dict, Optional

import numpy as np
import rclpy
from std_msgs.msg import String

from nodes.support.phase1.json_utils import safe_json_dumps


class RiskVlmDispatchStage:
    """Owns the risk-assessment queue, worker thread, backend, and diagnostics.

    Runs entirely independent of the object-detection RAP/VLM queues/thread/
    backend -- risk assessment uses a separate model/server by design, so a
    slow or unreachable risk server can never block classification. See
    ``_risk_loop`` for the one place it takes a hint from the VLM queue.
    """

    def __init__(self, coordinator: Any, config: Any, logger: Any, *, backend: Any, diagnostics: Any):
        self.coordinator = coordinator
        self.config = config
        self.logger = logger
        self.backend = backend
        self.diagnostics = diagnostics

        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=config.risk_vlm_queue_size)
        self.queue_dropped_count = 0
        self.completed_count = 0

    def enqueue_risk_task(
        self,
        *,
        event: Dict[str, Any],
        task: Dict[str, Any],
        crop: Optional[np.ndarray],
        label: str,
        mobility_class: str,
        source: str,
    ) -> None:
        """Fire-and-forget: queue a one-shot risk assessment for a track that
        was just classified (RAP hit, or successful object-detection VLM).

        Call this *after* ``_emit_semantic_label_result`` at both dispatch
        points -- risk assessment only ever runs once a label already exists.
        ``crop`` must be the exact array already in hand at the call site,
        never looked up again later: ``_emit_semantic_label_result`` retires
        the track's live crop registry entry (``_retire_track_crop``)
        immediately after publishing the classification result, so a later
        by-track-ID lookup from the risk worker thread would find nothing.
        Capturing the crop here, at enqueue time, sidesteps that ordering
        problem entirely -- the risk queue carries the crop itself, not an ID.
        """
        if not self.config.risk_vlm_enabled:
            return
        if crop is None or getattr(crop, "size", 0) == 0:
            return
        track_id = str(event.get("persistent_track_id", task.get("persistent_track_id", "")))
        if not track_id:
            return
        segments = self.coordinator._semantic_segments_for_fanout(event, task)
        hydra_slot_ids = sorted({
            slot_id
            for segment in segments
            for slot_id in (int(segment.get("hydra_slot_id", segment.get("hydra_label_id", 0)) or 0),)
            if slot_id > 0
        })
        if not hydra_slot_ids:
            return
        risk_task = {
            "track_id": track_id,
            "hydra_slot_ids": hydra_slot_ids,
            "crop": crop,
            "label": str(label or "unknown_object"),
            "mobility_class": str(mobility_class or "unknown"),
            "source": str(source),
            "frame_id": str(task.get("frame_id", "")),
            "sequence": int(task.get("sequence", 0) or 0),
            "timestamp_sec": float(task.get("timestamp_sec", 0.0) or 0.0),
            "created_monotonic": time.perf_counter(),
        }
        try:
            self.queue.put_nowait(risk_task)
        except queue.Full:
            # Bounded by design: a slow/unreachable risk server can only ever
            # delay or drop risk results, never block classification,
            # tracking, or Hydra publishing. There's no "wait for a better
            # crop" concept to defer to instead (the crop is already final),
            # so the only sane full-queue behavior is drop the oldest
            # pending task to make room for this one.
            try:
                self.queue.get_nowait()
                self.queue_dropped_count += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(risk_task)
            except queue.Full:
                self.queue_dropped_count += 1

    def _risk_loop(self) -> None:
        """Dispatch one risk assessment at a time from ``queue``.

        Runs on its own daemon thread. The one thing this loop shares with
        the object-detection VLM loop is Jetson hardware: even though
        they're logically separate models, this deployment may run both VLM
        servers on the same physical GPU, so this loop backs off while the
        object-detection VLM has pending work rather than dispatching
        concurrently. That's a priority hint, not a guarantee -- see
        risk_vlm_yield_to_object_vlm's doc comment in phase1_config.py for
        the tradeoff, and the design doc (debug/risk_assessment_feature/
        DESIGN.md) for the full reasoning. As RAP's hit rate improves over a
        session, object-detection VLM traffic naturally drops (fewer RAP
        misses need it), so the VLM queue sits empty more often on its own
        and risk throughput rises without any adaptive tuning here.
        """
        while not self.coordinator._stop_event.is_set():
            if self.config.risk_vlm_yield_to_object_vlm and not self.coordinator.vlm_queue.empty():
                time.sleep(float(self.config.risk_vlm_yield_backoff_sec))
                continue
            try:
                task = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            start = time.perf_counter()
            try:
                result = self.backend.assess(
                    task["crop"], task["label"], task["mobility_class"], task["source"]
                )
            except Exception as exc:
                # Mirrors the object-detection VLM loop's own contract: a
                # worker-level exception must never crash this thread or
                # strand a track without ever finishing its risk task.
                result = {
                    "success": False,
                    "risk_score": 0.0,
                    "risk_factors": [],
                    "failure_reason": f"worker_exception:{type(exc).__name__}",
                }
                if rclpy.ok() and not self.coordinator._stop_event.is_set():
                    self.logger.error(f"Risk VLM failed for track={task['track_id']}: {exc}")
            risk_delay_ms = (time.perf_counter() - start) * 1000.0
            # Logged unconditionally (success or failure) -- a failed or
            # malformed risk response is exactly the case worth inspecting
            # later, and this mirrors vlm_test_diagnostics' own convention
            # for the object-detection VLM.
            try:
                self.diagnostics.log_risk_result(
                    task["crop"],
                    result,
                    risk_delay_ms,
                    track_id=task["track_id"],
                    hydra_slot_id=task["hydra_slot_ids"][0] if task["hydra_slot_ids"] else 0,
                    label=task["label"],
                    mobility_class=task["mobility_class"],
                    source=task["source"],
                    timestamp=task["timestamp_sec"],
                )
            except Exception as e:
                self.logger.warn(f"Failed to log risk VLM diagnostics: {e}")
            if bool(result.get("success", False)):
                self._publish_risk_result(task, result, risk_delay_ms)
            self.completed_count += 1

    def _publish_risk_result(self, task: Dict[str, Any], result: Dict[str, Any], risk_delay_ms: float) -> None:
        """Publish one risk result, fanned out to every Hydra slot the track owns.

        Matches ``_emit_semantic_label_result``'s fan-out shape (one message
        per slot) so a track that owns several local Hydra segments gets its
        risk value attached to every one of them, exactly like its label.
        Published on a dedicated topic (``risk_result_topic``), not folded
        into ``semantic_label_result``: risk always arrives later, from a
        second independent VLM call, and carries an unrelated payload shape.
        """
        if self.coordinator._stop_event.is_set() or not rclpy.ok():
            return
        risk_score = float(result.get("risk_score", 0.0) or 0.0)
        risk_factors = [str(item) for item in (result.get("risk_factors") or [])]
        for slot_id in task["hydra_slot_ids"]:
            payload = {
                "event": "risk_result",
                "persistent_track_id": task["track_id"],
                "internal_object_id": task["track_id"],
                "hydra_slot_id": int(slot_id),
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "label": task["label"],
                "mobility_class": task["mobility_class"],
                "source": task["source"],
                "frame_id": task["frame_id"],
                "sequence": task["sequence"],
                "timestamp_sec": task["timestamp_sec"],
                "risk_delay_ms": float(risk_delay_ms),
            }
            self.coordinator._safe_publish(self.coordinator.risk_result_pub, String(data=safe_json_dumps(payload)))
