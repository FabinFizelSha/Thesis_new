"""Local Hydra-segment allocation and matching for PersistentObjectTracker.

Splits one physical object track into multiple local Hydra segments (each
with its own slot) so presence confidence stays spatially bounded for long
objects such as carpets, walls, shelves, or ceilings. Every method here
assumes the caller already holds ``tracker._lock`` -- nothing in this file
acquires or releases it.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from nodes.support.phase1.tracker_geometry import (
    _aabb_center_distance_xy,
    _aabb_gap_xy,
    _aabb_union_xy_diagonal,
    _aabb_xy_diagonal,
    _bbox_iou,
)


class TrackerLocalSegments:
    """Owns local-segment allocation, matching, and geometry updates."""

    def __init__(self, tracker: Any):
        self.tracker = tracker

    def _allocate_slot_for_segment(self) -> Tuple[int, str, int]:
        tracker = self.tracker
        if not bool(getattr(tracker.config, "persistent_use_hydra_slots", False)):
            instance_id = tracker._next_instance_id
            if instance_id > 65535:
                raise RuntimeError("Persistent instance ID space exhausted for 16UC1 output")
            tracker._next_instance_id += 1
            return int(instance_id), str(instance_id), int(instance_id)

        allocated = tracker._next_available_slot_id()
        if allocated is None:
            raise RuntimeError("No unreserved Hydra local segment slots remain")
        tracker._allocated_slot_ids.add(int(allocated))
        first = int(tracker.config.persistent_slot_first_label_id)
        width = max(1, int(tracker.config.persistent_slot_label_width))
        slot_index = int(allocated) - first + 1
        name = f"{tracker.config.persistent_slot_label_prefix}{slot_index:0{width}d}"
        return int(allocated), name, int(allocated)

    def _new_segment_for_track(
        self,
        *,
        track: Any,
        frame_id: str,
        sequence: int,
        timestamp_sec: float,
        centroid: Optional[np.ndarray],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
        forced_slot_id: int = 0,
        forced_slot_name: str = "",
    ) -> Any:
        from nodes.support.phase1.persistent_object_tracker import PersistentObjectSegment

        if forced_slot_id > 0:
            slot_id = int(forced_slot_id)
            slot_name = str(forced_slot_name) if forced_slot_name else str(forced_slot_id)
            instance_id = int(slot_id)
        else:
            slot_id, slot_name, instance_id = self._allocate_slot_for_segment()
        return PersistentObjectSegment(
            segment_id=f"{track.track_id}:slot_{slot_id}",
            hydra_label_id=int(slot_id),
            hydra_label_name=str(slot_name),
            instance_id=int(instance_id),
            first_seen_timestamp_sec=float(timestamp_sec),
            last_seen_timestamp_sec=float(timestamp_sec),
            first_seen_frame_id=str(frame_id),
            last_seen_frame_id=str(frame_id),
            first_seen_sequence=int(sequence),
            last_seen_sequence=int(sequence),
            centroid_3d=centroid.copy() if centroid is not None else None,
            bbox_2d=bbox_2d,
            bbox_3d_min=bbox_3d_min.copy() if bbox_3d_min is not None else None,
            bbox_3d_max=bbox_3d_max.copy() if bbox_3d_max is not None else None,
            last_bbox_3d_min=bbox_3d_min.copy() if bbox_3d_min is not None else None,
            last_bbox_3d_max=bbox_3d_max.copy() if bbox_3d_max is not None else None,
        )

    @staticmethod
    def _activate_segment(track: Any, segment: Any) -> None:
        track.active_segment_slot_id = int(segment.hydra_label_id)
        track.hydra_label_id = int(segment.hydra_label_id)
        track.hydra_label_name = str(segment.hydra_label_name)
        track.instance_id = int(segment.instance_id)

    def _segment_xy_span_after_update(
        self,
        segment: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> Optional[float]:
        if bbox_3d_min is None or bbox_3d_max is None:
            return None
        if segment.bbox_3d_min is None or segment.bbox_3d_max is None:
            return _aabb_xy_diagonal(bbox_3d_min, bbox_3d_max)
        return _aabb_union_xy_diagonal(segment.bbox_3d_min, segment.bbox_3d_max, bbox_3d_min, bbox_3d_max)

    def _update_segment_geometry(
        self,
        segment: Any,
        *,
        frame_id: str,
        sequence: int,
        timestamp_sec: float,
        centroid: Optional[np.ndarray],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
        expand_bbox: bool = True,
    ) -> None:
        alpha = float(self.tracker.config.persistent_centroid_update_alpha)
        if centroid is not None:
            segment.centroid_3d = centroid.copy() if segment.centroid_3d is None else alpha * segment.centroid_3d + (1.0 - alpha) * centroid
        if bbox_2d:
            segment.bbox_2d = bbox_2d
        if bbox_3d_min is not None and bbox_3d_max is not None:
            if expand_bbox:
                segment.bbox_3d_min = bbox_3d_min.copy() if segment.bbox_3d_min is None else np.minimum(segment.bbox_3d_min, bbox_3d_min)
                segment.bbox_3d_max = bbox_3d_max.copy() if segment.bbox_3d_max is None else np.maximum(segment.bbox_3d_max, bbox_3d_max)
            segment.last_bbox_3d_min = bbox_3d_min.copy()
            segment.last_bbox_3d_max = bbox_3d_max.copy()
        segment.last_seen_frame_id = str(frame_id)
        segment.last_seen_sequence = int(sequence)
        segment.last_seen_timestamp_sec = float(timestamp_sec)
        segment.seen_count += 1

    def _assign_local_segment(
        self,
        *,
        track: Any,
        frame_id: str,
        sequence: int,
        timestamp_sec: float,
        centroid: Optional[np.ndarray],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> Tuple[str, str, Optional[float]]:
        tracker = self.tracker
        if not bool(getattr(tracker.config, "persistent_local_segments_enabled", True)):
            segment = track.segments.get(int(track.active_segment_slot_id)) or next(iter(track.segments.values()))
            self._activate_segment(track, segment)
            self._update_segment_geometry(
                segment, frame_id=frame_id, sequence=sequence, timestamp_sec=timestamp_sec,
                centroid=centroid, bbox_2d=bbox_2d, bbox_3d_min=bbox_3d_min, bbox_3d_max=bbox_3d_max,
            )
            return "matched_segment", "local_segments_disabled", None

        max_span = float(getattr(tracker.config, "persistent_local_segment_max_xy_span_m", 4.0))
        revisit_distance = float(getattr(tracker.config, "persistent_local_segment_revisit_distance_m", 1.5))
        gap_limit = float(getattr(tracker.config, "persistent_local_segment_gap_m", 0.20))
        use_2d_fallback = bool(
            getattr(tracker.config, "persistent_local_segment_2d_fallback_enabled", True)
        )
        max_2d_age_sec = max(
            0.0,
            float(
                getattr(
                    tracker.config,
                    "persistent_local_segment_max_2d_iou_age_sec",
                    getattr(tracker.config, "persistent_max_2d_iou_age_sec", 2.0),
                )
            ),
        )
        min_2d_iou = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        tracker.config,
                        "persistent_local_segment_min_2d_iou",
                        getattr(tracker.config, "persistent_min_2d_iou", 0.30),
                    )
                ),
            ),
        )
        best_segment: Optional[Any] = None
        best_score: Optional[float] = None
        best_reason = ""
        timestamp_only = False

        for segment in track.segments.values():
            score: Optional[float] = None
            reason = ""
            if (
                bbox_3d_min is not None and bbox_3d_max is not None
                and segment.bbox_3d_min is not None and segment.bbox_3d_max is not None
            ):
                gap = _aabb_gap_xy(bbox_3d_min, bbox_3d_max, segment.bbox_3d_min, segment.bbox_3d_max)
                center_distance = _aabb_center_distance_xy(bbox_3d_min, bbox_3d_max, segment.bbox_3d_min, segment.bbox_3d_max)
                if not segment.closed:
                    candidate_span = self._segment_xy_span_after_update(segment, bbox_3d_min, bbox_3d_max)
                    if candidate_span is not None and candidate_span <= max_span and gap <= gap_limit:
                        score = gap + 0.01 * center_distance
                        reason = "segment_bbox_local"
                    elif gap <= gap_limit:
                        # Touches this still-growing segment, but merging would
                        # push it past the span cap: the seam belongs exactly
                        # here. Freeze the segment now instead of leaving it
                        # open to keep absorbing further-drifted observations
                        # via the centroid-distance fallback below -- that is
                        # what previously produced multi-metre overlaps instead
                        # of a clean cut between segments.
                        segment.closed = True
                        score = gap
                        reason = "segment_centroid_revisit_no_expand"
                    elif center_distance <= revisit_distance:
                        score = center_distance
                        reason = "segment_centroid_revisit_no_expand"
                else:
                    # Already closed: identity-only re-check (e.g. revisiting
                    # this section from a different angle later) -- geometry
                    # never expands again regardless of how close this is.
                    if gap <= gap_limit:
                        score = gap
                        reason = "segment_centroid_revisit_no_expand"
                    elif center_distance <= revisit_distance:
                        score = center_distance
                        reason = "segment_centroid_revisit_no_expand"
            elif centroid is not None and segment.centroid_3d is not None:
                distance = float(np.linalg.norm(centroid[:2] - segment.centroid_3d[:2]))
                if distance <= revisit_distance:
                    score = distance
                    reason = "segment_centroid_revisit"

            if score is not None and (best_score is None or score < best_score):
                best_segment = segment
                best_score = float(score)
                best_reason = reason
                timestamp_only = reason.endswith("no_expand")

        if best_segment is not None:
            self._activate_segment(track, best_segment)
            self._update_segment_geometry(
                best_segment, frame_id=frame_id, sequence=sequence, timestamp_sec=timestamp_sec,
                centroid=centroid, bbox_2d=bbox_2d, bbox_3d_min=bbox_3d_min, bbox_3d_max=bbox_3d_max,
                expand_bbox=not timestamp_only,
            )
            return "matched_segment", best_reason, best_score

        # Local segments previously had no image-space fallback. When depth was
        # missing, the physical track could survive through 2D IoU while every
        # observation allocated a fresh local Hydra slot. Use recent 2D overlap
        # only when either side lacks a usable 3D footprint. If both the current
        # observation and a candidate segment have 3D geometry, the failed 3D
        # association is authoritative and must not be overridden by image IoU.
        if use_2d_fallback:
            current_has_3d = bbox_3d_min is not None and bbox_3d_max is not None
            best_2d_segment: Optional[Any] = None
            best_2d_iou = 0.0

            for segment in track.segments.values():
                segment_has_3d = (
                    segment.bbox_3d_min is not None and segment.bbox_3d_max is not None
                )
                if current_has_3d and segment_has_3d:
                    continue

                age_sec = max(
                    0.0,
                    float(timestamp_sec) - float(segment.last_seen_timestamp_sec),
                )
                if age_sec > max_2d_age_sec:
                    continue

                iou = _bbox_iou(bbox_2d, segment.bbox_2d)
                if iou < min_2d_iou:
                    continue
                if best_2d_segment is None or iou > best_2d_iou:
                    best_2d_segment = segment
                    best_2d_iou = float(iou)

            if best_2d_segment is not None:
                self._activate_segment(track, best_2d_segment)
                self._update_segment_geometry(
                    best_2d_segment,
                    frame_id=frame_id,
                    sequence=sequence,
                    timestamp_sec=timestamp_sec,
                    centroid=centroid,
                    bbox_2d=bbox_2d,
                    bbox_3d_min=bbox_3d_min,
                    bbox_3d_max=bbox_3d_max,
                )
                return "matched_segment", "segment_bbox_2d_iou", 1.0 - best_2d_iou

        segment = self._new_segment_for_track(
            track=track, frame_id=frame_id, sequence=sequence, timestamp_sec=timestamp_sec,
            centroid=centroid, bbox_2d=bbox_2d, bbox_3d_min=bbox_3d_min, bbox_3d_max=bbox_3d_max,
        )
        track.segments[int(segment.hydra_label_id)] = segment
        self._activate_segment(track, segment)
        return "new_segment", "local_span_exceeded_or_new_local_identity", None
