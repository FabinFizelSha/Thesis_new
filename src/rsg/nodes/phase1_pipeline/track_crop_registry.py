"""Per-track best-crop registry: one immutable source ROI per unresolved track.

Semantic rendering is intentionally deferred to the RAP/VLM worker that
dequeues the track. The frame-critical path only scores and stores; the
worker snapshot (``_snapshot_track_task``) does the (more expensive) target
crop rendering, outside the registry lock, at dequeue time.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from nodes.support.phase1.semantic_crop import context_bbox_xywh, prepare_target_mask
from nodes.support.phase1.time_utils import stamp_to_float


class TrackCropRegistry:
    """Owns the best-crop-per-track store and its lock."""

    def __init__(self, coordinator: Any, config: Any, logger: Any):
        self.coordinator = coordinator
        self.config = config
        self.logger = logger
        self._track_best_crops: Dict[str, Dict[str, Any]] = {}
        self._track_crop_lock = threading.RLock()

    def _experiment_crop_score(
        self, rgb: np.ndarray, mask: Optional[np.ndarray], bbox_2d: Any
    ) -> Optional[float]:
        """Crop-quality score: the 2:2:1 weighted additive scorer (log pixel
        count : Laplacian sharpness : 3px edge margin) in
        ``TrackingCropManager._score_crop``, evaluated on the tight mask
        bounding-box crop exactly as ``extract_crop`` does. Returns ``None``
        when the crop cannot be scored (no mask / degenerate box), so the
        caller can fall back to the geometry score.
        """
        if mask is None or not bbox_2d or len(bbox_2d) < 4:
            return None
        h, w = rgb.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox_2d[:4]]
        x0 = max(0, min(w, x))
        y0 = max(0, min(h, y))
        x1 = max(0, min(w, x + bw))
        y1 = max(0, min(h, y + bh))
        if x1 <= x0 or y1 <= y0:
            return None
        mask_array = np.asarray(mask)
        if mask_array.shape != (h, w):
            return None
        crop_rgb = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        crop_mask = np.ascontiguousarray(mask_array[y0:y1, x0:x1])
        if crop_rgb.size == 0 or crop_mask.size == 0:
            return None
        composite, _pixel, _sharpness, _margin = self.coordinator.tracking_crop_manager._score_crop(
            crop_rgb, crop_mask
        )
        return float(composite)

    def _remember_track_crop(
        self,
        track_id: Optional[str],
        rgb: np.ndarray,
        metadata: Dict[str, Any],
        frame: Any,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Keep one immutable source ROI for the best observation of a track.

        Semantic rendering is intentionally deferred to the RAP/VLM worker
        that dequeues the track.  The frame-critical path copies the tight
        mask crop once to score it (finalized crop-scoring experiment,
        ``_experiment_crop_score``) and, only when that score beats the stored
        best by more than ``HYSTERESIS_MARGIN``, copies the bounded context
        ROI once more to store it; it never renders two crops that may be
        replaced before either worker consumes them.
        """
        coordinator = self.coordinator
        if not track_id:
            return
        key = str(track_id)
        if not coordinator.persistent_tracker.is_semantic_labeling_open(key):
            return

        bbox_2d = metadata.get("bbox_2d")
        context_bbox_2d = context_bbox_xywh(
            rgb.shape[:2],
            bbox_2d,
            context_ratio=float(self.config.vlm_crop_context_ratio),
        )
        if not context_bbox_2d:
            return
        timestamp_sec = float(stamp_to_float(frame.header.stamp))
        # Geometry-based eligibility (min area / short side / border clip) is
        # kept from score_track_crop; the *selection* score is the finalized
        # crop-scoring experiment's 2:2:1 composite on the tight mask crop.
        crop_quality = coordinator.sem_stage.score_track_crop(metadata, rgb.shape[:2], bbox_2d)
        experiment_score = self._experiment_crop_score(rgb, mask, bbox_2d)
        score = float(
            experiment_score
            if experiment_score is not None
            else crop_quality.get("vlm_crop_quality_score", 0.0) or 0.0
        )
        crop_quality["vlm_crop_quality_score"] = score
        _reasons = [
            r for r in (crop_quality.get("vlm_crop_quality_reasons") or [])
            if r != "crop_quality_below_minimum"
        ]
        if score < float(self.config.vlm_crop_min_quality_score):
            _reasons.append("crop_quality_below_minimum")
        crop_quality["vlm_crop_quality_reasons"] = _reasons
        crop_quality["vlm_crop_quality_eligible"] = not _reasons

        # Experiment acceptance rule: a new observation replaces the stored
        # best only when it beats it by more than HYSTERESIS_MARGIN.
        hysteresis = 1.0 + float(coordinator.tracking_crop_manager.HYSTERESIS_MARGIN)

        # Score before mask cleanup or rendering. A non-improving observation
        # cannot replace the current crop, so only refresh track recency.
        with self._track_crop_lock:
            current = self._track_best_crops.get(key)
            if current is not None and score <= float(current.get("score", -1.0)) * hysteresis:
                current["last_observed_timestamp_sec"] = timestamp_sec
                return

        context_x, context_y, context_width, context_height = context_bbox_2d
        image_height, image_width = rgb.shape[:2]
        bbox_x, bbox_y, bbox_width, bbox_height = [int(value) for value in bbox_2d]
        target_x0 = max(0, min(image_width, bbox_x))
        target_y0 = max(0, min(image_height, bbox_y))
        target_x1 = max(0, min(image_width, bbox_x + bbox_width))
        target_y1 = max(0, min(image_height, bbox_y + bbox_height))
        if target_x1 <= target_x0 or target_y1 <= target_y0:
            return
        target_bbox_in_roi = [
            int(target_x0 - context_x),
            int(target_y0 - context_y),
            int(target_x1 - target_x0),
            int(target_y1 - target_y0),
        ]

        source_rgb = np.array(
            rgb[
                context_y:context_y + context_height,
                context_x:context_x + context_width,
            ],
            copy=True,
            order="C",
        )
        if source_rgb.size == 0:
            return

        source_mask = None
        semantic_rendering_enabled = bool(
            self.config.semantic_crop_rap_target_only_enabled
            or self.config.semantic_crop_vlm_target_focus_enabled
        )
        if semantic_rendering_enabled:
            if mask is None:
                return
            mask_array = np.asarray(mask)
            if mask_array.shape != rgb.shape[:2]:
                return
            source_mask = np.array(
                mask_array[
                    context_y:context_y + context_height,
                    context_x:context_x + context_width,
                ],
                dtype=bool,
                copy=True,
                order="C",
            )
            if source_mask.shape != source_rgb.shape[:2]:
                return

        # Apply boundary marking when crop becomes the best (once-off, not every frame)
        # This happens before making it read-only, so RAP/VLM receive marked version
        if source_mask is not None:
            try:
                source_rgb = coordinator.tracking_crop_manager._highlight_contours(
                    source_rgb, source_mask,
                    color=(0, 255, 255),  # Cyan
                    thickness=1
                )
            except Exception:
                pass  # If marking fails, use unmarked version

        # Revisions are immutable after publication to the registry. Workers
        # can safely retain these references after releasing the registry lock.
        source_rgb.setflags(write=False)
        if source_mask is not None:
            source_mask.setflags(write=False)

        crop_metadata = dict(metadata)
        crop_metadata.update(crop_quality)
        crop_metadata.update({
            "vlm_crop_context_bbox_2d": context_bbox_2d,
            "vlm_crop_context_ratio": float(self.config.vlm_crop_context_ratio),
            "vlm_crop_width_px": int(context_width),
            "vlm_crop_height_px": int(context_height),
            "rap_crop_representation": "target_only" if self.config.semantic_crop_rap_target_only_enabled else "raw_bbox",
            "vlm_crop_representation": "target_full_colour_local_halo_dimmed_context" if self.config.semantic_crop_vlm_target_focus_enabled else "raw_context",
            "vlm_context_alpha": float(self.config.semantic_crop_vlm_context_alpha),
            "vlm_context_grayscale": bool(self.config.semantic_crop_vlm_context_grayscale),
            "vlm_near_context_enabled": bool(self.config.semantic_crop_vlm_near_context_enabled),
            "vlm_near_context_alpha": float(self.config.semantic_crop_vlm_near_context_alpha),
            "vlm_near_context_dilation_px": int(self.config.semantic_crop_vlm_near_context_dilation_px),
            "vlm_near_context_grayscale": bool(self.config.semantic_crop_vlm_near_context_grayscale),
            "semantic_crop_mask_cleanup_enabled": bool(self.config.semantic_crop_mask_cleanup_enabled),
            "semantic_crop_mask_cleanup_min_component_area_ratio": float(self.config.semantic_crop_mask_cleanup_min_component_area_ratio),
            "semantic_crop_mask_cleanup_component_max_gap_px": int(self.config.semantic_crop_mask_cleanup_component_max_gap_px),
        })
        updated = False
        previous_score: Optional[float] = None
        revision = 0
        selection_reason = "lower_score_than_current_best"

        with self._track_crop_lock:
            current = self._track_best_crops.get(key)
            if current is None:
                revision = 1
                selection_reason = "first_valid_crop"
                self._track_best_crops[key] = {
                    "score": score,
                    "source_rgb": source_rgb,
                    "source_mask": source_mask,
                    "target_bbox_in_roi": target_bbox_in_roi,
                    "crop_revision": revision,
                    "first_crop_timestamp_sec": timestamp_sec,
                    "last_crop_update_timestamp_sec": timestamp_sec,
                    "last_observed_timestamp_sec": timestamp_sec,
                    "frame_header": frame.header,
                    "frame_id": frame.rsg_frame_id,
                    "sequence": int(frame.sequence),
                    "timestamp_sec": timestamp_sec,
                    "candidate_id": str(metadata.get("candidate_id", "")),
                    "centroid_frame_id": str(frame.camera_pose.header.frame_id or frame.header.frame_id or ""),
                    "object_metadata": crop_metadata,
                }
                updated = True
            else:
                previous_score = float(current.get("score", -1.0))
                current["last_observed_timestamp_sec"] = timestamp_sec
                revision = int(current.get("crop_revision", 0) or 0)
                if score > previous_score * hysteresis:
                    revision += 1
                    selection_reason = "score_above_hysteresis_over_current_best"
                    self._track_best_crops[key] = {
                        "score": score,
                        "source_rgb": source_rgb,
                        "source_mask": source_mask,
                        "target_bbox_in_roi": target_bbox_in_roi,
                        "crop_revision": revision,
                        "first_crop_timestamp_sec": float(current.get("first_crop_timestamp_sec", timestamp_sec) or timestamp_sec),
                        "last_crop_update_timestamp_sec": timestamp_sec,
                        "last_observed_timestamp_sec": timestamp_sec,
                        "frame_header": frame.header,
                        "frame_id": frame.rsg_frame_id,
                        "sequence": int(frame.sequence),
                        "timestamp_sec": timestamp_sec,
                        "candidate_id": str(metadata.get("candidate_id", "")),
                        "centroid_frame_id": str(frame.camera_pose.header.frame_id or frame.header.frame_id or ""),
                        "object_metadata": crop_metadata,
                    }
                    updated = True

        if updated:
            # Save the best crop when it's accepted (already marked with boundaries)
            try:
                best_crop_path = coordinator.tracking_crop_manager.save_best_crop(
                    track_id=key,
                    source_rgb=source_rgb,
                    crop_revision=revision,
                    crop_score=score,
                    sequence=int(frame.sequence),
                )
                if best_crop_path:
                    self.logger.debug(f"Saved best crop for {key}: {best_crop_path}")
            except Exception as e:
                self.logger.warn(f"Failed to save best crop for {key}: {e}")

            coordinator.vlm_stage._resume_quality_deferred_vlm_if_ready(key)
            coordinator.vlm_stage._resume_vlm_retry_if_better_crop(key)

    def _describe_track_crop(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Return queue-safe crop metadata without copying image payloads."""
        key = str(track_id)
        with self._track_crop_lock:
            best = self._track_best_crops.get(key)
            if best is None:
                return None
            metadata = dict(best.get("object_metadata") or {})
            return {
                "best_frame_score": float(best.get("score", 0.0) or 0.0),
                "crop_revision": int(best.get("crop_revision", 0) or 0),
                "crop_timestamp_sec": float(best.get("timestamp_sec", 0.0) or 0.0),
                "last_crop_update_timestamp_sec": float(best.get("last_crop_update_timestamp_sec", 0.0) or 0.0),
                "last_observed_timestamp_sec": float(best.get("last_observed_timestamp_sec", 0.0) or 0.0),
                "vlm_crop_quality_score": float(metadata.get("vlm_crop_quality_score", 0.0) or 0.0),
                "vlm_crop_quality_eligible": bool(metadata.get("vlm_crop_quality_eligible", False)),
                "vlm_crop_quality_reasons": list(metadata.get("vlm_crop_quality_reasons", []) or []),
                "vlm_crop_width_px": int(metadata.get("vlm_crop_width_px", 0) or 0),
                "vlm_crop_height_px": int(metadata.get("vlm_crop_height_px", 0) or 0),
            }

    def _retire_track_crop(self, track_id: str) -> None:
        """Release a completed track's crop after its final semantic result."""
        key = str(track_id)
        if not key:
            return
        with self._track_crop_lock:
            self._track_best_crops.pop(key, None)

    def _snapshot_track_task(self, track_id: str, stage: str) -> Optional[Dict[str, Any]]:
        """Render one immutable ROI revision for the dequeuing worker."""
        coordinator = self.coordinator
        key = str(track_id)
        with self._track_crop_lock:
            best = self._track_best_crops.get(key)
            if best is None:
                return None
            source_rgb = best.get("source_rgb")
            source_mask = best.get("source_mask")
            target_bbox_in_roi = list(best.get("target_bbox_in_roi") or [])
            if source_rgb is None or getattr(source_rgb, "size", 0) == 0:
                return None
            if len(target_bbox_in_roi) != 4:
                return None
            metadata = dict(best.get("object_metadata") or {})
            score = float(best.get("score", 0.0) or 0.0)
            revision = int(best.get("crop_revision", 0) or 0)
            frame_header = best.get("frame_header")
            frame_id = str(best.get("frame_id", ""))
            sequence = int(best.get("sequence", 0) or 0)
            timestamp_sec = float(best.get("timestamp_sec", 0.0) or 0.0)
            candidate_id = str(best.get("candidate_id", ""))
            centroid_frame_id = str(best.get("centroid_frame_id", ""))
            last_crop_update_timestamp_sec = float(best.get("last_crop_update_timestamp_sec", timestamp_sec) or timestamp_sec)
            last_observed_timestamp_sec = float(best.get("last_observed_timestamp_sec", timestamp_sec) or timestamp_sec)

        # Render outside the registry lock. A VLM task also retains the RAP
        # representation from this exact revision so any later memory update
        # cannot pair the VLM label with a different observation.
        prepared_mask = None
        if bool(self.config.semantic_crop_rap_target_only_enabled) or bool(
            self.config.semantic_crop_vlm_target_focus_enabled
        ):
            prepared_mask = prepare_target_mask(
                source_rgb,
                source_mask,
                cleanup_enabled=self.config.semantic_crop_mask_cleanup_enabled,
                cleanup_min_component_area_ratio=self.config.semantic_crop_mask_cleanup_min_component_area_ratio,
                cleanup_component_max_gap_px=self.config.semantic_crop_mask_cleanup_component_max_gap_px,
            )
            if prepared_mask is None:
                return None

        object_crop = coordinator.sem_stage.build_rap_crop(
            source_rgb,
            source_mask,
            target_bbox_in_roi,
            prepared_mask=prepared_mask,
        )
        if object_crop is None or object_crop.size == 0:
            return None

        vlm_crop = None
        if str(stage).startswith("vlm"):
            # Reuse source_rgb which already has boundary marked
            # (no additional rendering to avoid double-marking)
            vlm_crop = source_rgb
            if vlm_crop is None or vlm_crop.size == 0:
                return None

        quality_timeout_forced = coordinator.vlm_stage.is_quality_force(key)
        slot_id = int(metadata.get("hydra_label_id", metadata.get("hydra_slot_id", 0)) or 0)
        slot_name = str(metadata.get("hydra_label_name", metadata.get("hydra_slot_name", "")))
        metadata.update({
            "persistent_track_id": key,
            "internal_object_id": str(metadata.get("internal_object_id", key)),
            "hydra_label_id": slot_id,
            "hydra_slot_id": slot_id,
            "hydra_label_name": slot_name,
            "hydra_slot_name": slot_name,
            "crop_score": score,
            "crop_revision": revision,
            "crop_stage": str(stage),
            "last_crop_update_timestamp_sec": last_crop_update_timestamp_sec,
            "last_observed_timestamp_sec": last_observed_timestamp_sec,
            "vlm_crop_quality_timeout_forced": quality_timeout_forced,
        })

        queued_time = coordinator.rap_stage._enqueued_monotonic.get(key) if stage.startswith("rap") else coordinator.vlm_stage._enqueued_monotonic.get(key)
        return {
            "persistent_track_id": key,
            "hydra_slot_id": slot_id,
            "hydra_slot_name": slot_name,
            "frame_header": frame_header,
            "frame_id": frame_id,
            "rsg_frame_id": frame_id,
            "sequence": sequence,
            "timestamp_sec": timestamp_sec,
            "candidate_id": candidate_id,
            "mask_id": str(metadata.get("mask_id", candidate_id)),
            "rgb_crop": object_crop,
            "vlm_rgb_crop": vlm_crop,
            "source_rgb": source_rgb,
            "source_mask": source_mask,
            "target_bbox_in_roi": target_bbox_in_roi,
            "object_metadata": metadata,
            "centroid_frame_id": centroid_frame_id,
            "created_monotonic": float(queued_time if queued_time is not None else time.perf_counter()),
            "track_seen_count": int(metadata.get("persistent_track_seen_count", 0) or 0),
            "best_frame_score": score,
            "crop_revision": revision,
            "vlm_crop_quality_score": float(metadata.get("vlm_crop_quality_score", 0.0) or 0.0),
            "vlm_crop_quality_eligible": bool(metadata.get("vlm_crop_quality_eligible", False)),
            "vlm_crop_quality_reasons": list(metadata.get("vlm_crop_quality_reasons", []) or []),
            "last_crop_update_timestamp_sec": last_crop_update_timestamp_sec,
            "last_observed_timestamp_sec": last_observed_timestamp_sec,
            "vlm_crop_quality_timeout_forced": quality_timeout_forced,
            "queue_stage": str(stage),
        }
