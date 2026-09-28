"""Persistent physical-object tracking with fixed Hydra unknown slots.

The tracker separates a physical object identity from its final semantic class:

* ``track_id`` is the RSG identity for one physical object.
* ``hydra_label_id`` is a session-stable semantic *slot* (for example 21).
* ``canonical_label`` is evidence from RAP/VLM and may change while a slot is active.
* RAP/VLM can attach a semantic label asynchronously without changing the
  slot or Hydra's already-integrated geometry.

When ``persistent_use_hydra_slots`` is enabled, each new physical object gets a
unique label from a predeclared range in Hydra's startup label-space YAML.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from nodes.support.phase1.vlm_result import DEFAULT_OBJECT_DETAIL
from nodes.support.phase1.tracker_spatial_index import TrackerSpatialIndex
from nodes.support.phase1.tracker_local_segments import TrackerLocalSegments
from nodes.support.phase1.tracker_semantic_evidence import TrackerSemanticEvidence
from nodes.support.phase1.tracker_reanchor import TrackerReanchor
from nodes.support.phase1.tracker_labeling_lifecycle import TrackerLabelingLifecycle
from nodes.support.phase1 import tracker_serialization
# Re-exported so `from nodes.support.phase1.persistent_object_tracker import
# _as_list` (and friends) keeps working for every existing external importer
# -- the definitions live in tracker_geometry.py, which every tracker_*.py
# collaborator can import from without risking a circular import back here.
# _rigid_aabb specifically: no longer used inside this file (moved to
# tracker_reanchor.py), kept solely because tests/test_reanchor.py imports
# it from here.
from nodes.support.phase1.tracker_geometry import (
    _as_xyz,
    _safe_float,
    _bbox_iou,
    _volume_ratio,
    _aabb_gap_xy,
    _aabb_gap_z,
    _aabb_center_distance_xy,
    _aabb_center_delta_z,
    _aabb_overlap_fraction_3d,
    _aabb_3d_containment,
    _gaussian_compatibility,
    _rigid_aabb,
    _as_list,
)


@dataclass
class PersistentObjectSegment:
    """Local Hydra semantic section belonging to one internal object track.

    The internal object ID is used for crop/RAP/VLM identity.  Each local
    segment owns a separate Hydra slot so presence confidence remains spatially
    bounded even for long objects such as carpets, walls, shelves, or ceilings.
    """

    segment_id: str
    hydra_label_id: int
    hydra_label_name: str
    instance_id: int
    first_seen_timestamp_sec: float
    last_seen_timestamp_sec: float
    first_seen_frame_id: str
    last_seen_frame_id: str
    first_seen_sequence: int
    last_seen_sequence: int
    centroid_3d: Optional[np.ndarray]
    bbox_2d: Any
    bbox_3d_min: Optional[np.ndarray]
    bbox_3d_max: Optional[np.ndarray]
    last_bbox_3d_min: Optional[np.ndarray]
    last_bbox_3d_max: Optional[np.ndarray]
    seen_count: int = 1
    # Permanently frozen once this segment's own XY span reaches
    # persistent_local_segment_max_xy_span_m -- its bbox never expands again
    # after that point. Without this, an already-at-cap segment kept
    # absorbing nearby observations via the centroid-distance revisit
    # fallback (matched, geometry frozen, but still claimed under the old
    # identity) instead of handing off to a new segment right at the cap,
    # producing multi-metre overlaps instead of a clean seam between
    # segments. See debug/fuser_object_relation_experiment/IMPLEMENTATION.md
    # for the real-run examples that motivated this.
    closed: bool = False


@dataclass
class PersistentObjectTrack:
    """Session-persistent estimate for one physical object."""

    track_id: str
    instance_id: int
    hydra_label_id: int
    hydra_label_name: str
    semantic_kind: str  # slot | class
    canonical_label: str
    label_source: str
    label_confidence: float

    first_seen_frame_id: str
    first_seen_sequence: int
    first_seen_timestamp_sec: float
    last_seen_frame_id: str
    last_seen_sequence: int
    last_seen_timestamp_sec: float

    centroid_3d: Optional[np.ndarray]
    bbox_volume_m3: Optional[float]
    bbox_2d: Any

    # The global extent is the unsmoothed union of all observations. The last
    # extent remains separate so a partially observed large object can grow
    # continuously as the robot moves along it.
    bbox_3d_min: Optional[np.ndarray]
    bbox_3d_max: Optional[np.ndarray]
    last_bbox_3d_min: Optional[np.ndarray]
    last_bbox_3d_max: Optional[np.ndarray]
    seen_count: int = 1
    raw_rap_label: str = ""
    raw_vlm_label: str = ""
    mobility_class: str = "unknown"
    mobility_confidence: float = 0.0
    mobility_source: str = "none"
    object_detail: str = DEFAULT_OBJECT_DETAIL
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Semantic-label worker state. The live Hydra slot remains unchanged.
    slot_state: str = "active"
    semantic_timestamp_sec: Optional[float] = None
    semantic_update_count: int = 0
    semantic_label: str = ""
    semantic_label_source: str = ""
    semantic_label_confidence: float = 0.0
    semantic_hydra_class_id: int = 0
    semantic_reason: str = ""
    label_evidence: Dict[str, float] = field(default_factory=dict)
    label_observations: Dict[str, int] = field(default_factory=dict)

    # One asynchronous RAP/VLM attempt is issued after the configured crop
    # settling window. The main SAM-to-Hydra path never waits.
    labeling_dispatched: bool = False
    labeling_completed: bool = False
    labeling_status: str = "collecting"  # collecting | rap_queued | rap_dequeued | vlm_queued | vlm_dequeued | vlm_waiting_for_better_crop | vlm_retry_pending | completed
    # Total VLM attempts so far (across retries after a failed/low-confidence
    # result). Only used when persistent_max_vlm_attempts > 1.
    vlm_attempt_count: int = 0

    # Local Hydra sections.  ``track_id`` remains the object-level identity for
    # best-crop tracking and semantic labelling, while the active segment slot is
    # written into the Hydra semantic image for spatially local confidence.
    segments: Dict[int, PersistentObjectSegment] = field(default_factory=dict)
    active_segment_slot_id: int = 0
    last_segment_event: str = ""
    last_segment_match_reason: str = ""
    last_segment_match_score: Optional[float] = None


class PersistentObjectTracker:
    """Associate masks across frames and allocate session-stable Hydra slots.

    The tracker never recycles a slot during one mapping session. A later
    geometric match reuses the same physical-object slot.
    """

    def __init__(self, config: Any, logger: Any, coordinator: Any = None) -> None:
        self.config = config
        self.logger = logger
        self.coordinator = coordinator
        self._tracks: Dict[str, PersistentObjectTrack] = {}
        self._next_track_index = 1
        self._next_slot_index = 1
        self._allocated_slot_ids: Set[int] = set()
        self._reserved_slot_ids: Set[int] = set()
        self._next_instance_id = 1
        self._frame_used_track_ids: Set[str] = set()
        self._forced_frame_matches: Dict[str, Tuple[Optional[str], str, Optional[float], List[Dict[str, Any]]]] = {}
        self._spatial_bbox_cells: Dict[Tuple[int, int], Set[str]] = {}
        self._spatial_centroid_cells: Dict[Tuple[int, int], Set[str]] = {}
        self._spatial_bbox_cells_by_track: Dict[str, Set[Tuple[int, int]]] = {}
        self._spatial_centroid_cell_by_track: Dict[str, Tuple[int, int]] = {}
        self._spatial_fallback_track_ids: Set[str] = set()
        self._last_reanchor: Optional[Dict[str, Any]] = None
        self._lock = Lock()
        self.spatial_index = TrackerSpatialIndex(self)
        self.local_segments = TrackerLocalSegments(self)
        self.semantic_evidence = TrackerSemanticEvidence(self)
        self.reanchor = TrackerReanchor(self)
        self.labeling_lifecycle = TrackerLabelingLifecycle(self)

    def begin_frame(self) -> None:
        """Reset one-to-one association state for the next image frame."""
        with self._lock:
            self._frame_used_track_ids.clear()

    def set_reserved_slot_ids(self, slot_ids: Set[int]) -> None:
        """Reserve confirmed slots loaded from a prior session.

        Reserved slots are not handed to a newly discovered object.  They can
        only be reused later by an explicit global-map association hint.
        """
        with self._lock:
            first = int(self.config.persistent_slot_first_label_id)
            last = first + int(self.config.persistent_slot_count) - 1
            self._reserved_slot_ids = {int(slot) for slot in slot_ids if first <= int(slot) <= last}

    def associate(
        self,
        *,
        metadata: Dict[str, Any],
        frame_id: str,
        sequence: int,
        timestamp_sec: float,
        desired_hydra_label_id: int,
        desired_hydra_label_name: str,
        raw_label: str,
        label_source: str,
        label_confidence: float,
        new_track_use_hydra_slot: bool = True,
        forced_hydra_slot_id: int = 0,
        stage_ms: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return metadata annotated with a persistent track and Hydra slot.

        ``stage_ms`` is an optional profiling side-channel only (Part 3 Path
        B). When provided, time blocked acquiring ``self._lock`` is
        accumulated into ``stage_ms["association_lock_wait_ms"]``, separate
        from time spent doing work after acquiring it. Never changes the
        returned metadata/record.
        """
        centroid = _as_xyz(metadata.get("centroid_3d"))
        volume = _safe_float(metadata.get("bbox_volume_m3"))
        bbox_2d = metadata.get("bbox_2d")
        bbox_3d_min = _as_xyz(metadata.get("bbox_3d_min"))
        bbox_3d_max = _as_xyz(metadata.get("bbox_3d_max"))


        normalised_raw_label = self._canonicalise_label(raw_label)

        # Filter out masks with invalid/zero volume (ghost tracks from invalid depth)
        if volume is not None and volume < 0.001:
            self.logger.debug(f"Skipping mask with invalid volume: {volume} m³ (frame {frame_id})")
            return metadata, {
                "persistent_track_id": "",
                "internal_object_id": "",
                "persistent_instance_id": 0,
                "persistent_track_event": "skipped_invalid_volume",
                "reason": "bbox_volume_below_threshold",
            }

        # try/finally below is exactly what `with self._lock:` expands to;
        # this is a behavior-identical substitution, not a locking change.
        _lock_wait_t0 = time.perf_counter() if stage_ms is not None else 0.0
        self._lock.acquire()
        if stage_ms is not None:
            stage_ms["association_lock_wait_ms"] = stage_ms.get("association_lock_wait_ms", 0.0) + (time.perf_counter() - _lock_wait_t0) * 1000.0
        try:
            forced = self._forced_frame_matches.pop(str(metadata.get("candidate_id", "")), None)
            if forced is not None:
                match_id, match_reason, match_score, candidate_evaluations = forced
            else:
                match_id, match_reason, match_score, candidate_evaluations = self._find_match(
                centroid=centroid,
                volume=volume,
                bbox_2d=bbox_2d,
                bbox_3d_min=bbox_3d_min,
                bbox_3d_max=bbox_3d_max,
                timestamp_sec=timestamp_sec,
                desired_hydra_label_id=int(desired_hydra_label_id),
                )

            # Log association decision for tracking quality evaluation
            if self.coordinator and hasattr(self.coordinator, 'tracking_quality_recorder'):
                prev_track_age = None
                prev_centroid_3d = None
                prev_bbox_volume = None
                prev_observations = None
                if match_id is not None and match_id in self._tracks:
                    track = self._tracks[match_id]
                    prev_track_age = len(track._timestamps) if hasattr(track, '_timestamps') else track.seen_count
                    prev_centroid_3d = list(track.centroid_3d) if track.centroid_3d is not None else None
                    prev_bbox_volume = track.bbox_volume_m3
                    prev_observations = track.seen_count

                # Extract mask_id from candidate_id (handle both int and string formats)
                candidate_id = metadata.get("candidate_id", -1)
                try:
                    mask_id = int(candidate_id) if isinstance(candidate_id, int) else -1
                except (ValueError, TypeError):
                    mask_id = -1

                # Per-component scores for the row that actually won (if any),
                # so a merge can be diagnosed from the real numbers next time
                # instead of estimating them after the fact.
                historical_score_out = centroid_score_out = image_score_out = vertical_score_out = None
                if match_id is not None:
                    for candidate_row in candidate_evaluations:
                        if candidate_row.get("candidate_track_id") == match_id and candidate_row.get("selected"):
                            components = candidate_row.get("global_association_components") or {}
                            historical_score_out = components.get("historical_overlap")
                            centroid_score_out = components.get("centroid_3d")
                            image_score_out = components.get("bbox_2d_iou")
                            vertical_score_out = components.get("vertical_compatibility")
                            break

                self.coordinator.tracking_quality_recorder.log_association_decision(
                    frame_id=frame_id,
                    sequence=sequence,
                    mask_id=mask_id,
                    mask_area_px=float(metadata.get("mask_area_px", 0.0)),
                    mask_centroid_3d=list(centroid) if centroid is not None else [0, 0, 0],
                    matched_track_id=match_id,
                    match_type=match_reason,
                    match_iou_3d=float(metadata.get("mask_iou_3d", 0.0)) if metadata.get("mask_iou_3d") else None,
                    match_score=match_score,
                    historical_score=historical_score_out,
                    centroid_score=centroid_score_out,
                    image_score=image_score_out,
                    vertical_score=vertical_score_out,
                    prev_track_age_frames=prev_track_age,
                    prev_track_observations=prev_observations,
                    prev_centroid_3d=prev_centroid_3d,
                    prev_bbox_volume_m3=prev_bbox_volume,
                    reason=match_reason or "association_applied"
                )

            # Only create new track if still no match found
            if match_id is None:
                if bool(new_track_use_hydra_slot) and not int(forced_hydra_slot_id or 0) and not self._has_slot_capacity():
                    metadata.update(
                        {
                            "persistent_track_id": "",
                            "internal_object_id": "",
                            "persistent_instance_id": 0,
                            "persistent_track_event": "slot_capacity_exhausted",
                            "persistent_track_seen_count": 0,
                            "hydra_label_id": int(desired_hydra_label_id),
                            "hydra_label_name": str(desired_hydra_label_name),
                        }
                    )
                    return metadata, {
                        "persistent_track_event": "slot_capacity_exhausted",
                        "persistent_track_id": "",
                        "internal_object_id": "",
                        "persistent_instance_id": 0,
                        "reason": "persistent_slot_capacity_reached",
                    }

                track = self._new_track(
                    frame_id=frame_id,
                    sequence=sequence,
                    timestamp_sec=timestamp_sec,
                    centroid=centroid,
                    volume=volume,
                    bbox_2d=bbox_2d,
                    bbox_3d_min=bbox_3d_min,
                    bbox_3d_max=bbox_3d_max,
                    desired_hydra_label_id=int(desired_hydra_label_id),
                    desired_hydra_label_name=str(desired_hydra_label_name),
                    raw_label=normalised_raw_label,
                    label_source=str(label_source),
                    label_confidence=float(label_confidence),
                    metadata=metadata,
                    use_hydra_slot=bool(new_track_use_hydra_slot),
                    forced_hydra_slot_id=int(forced_hydra_slot_id),
                )
                self._tracks[track.track_id] = track
                self._refresh_spatial_index(track)
                track_event = "new_track"
                match_reason = "new_track"
                match_score = None
                segment_event = "new_segment"
                segment_reason = "first_segment"
                segment_score = None
            else:
                track = self._tracks[match_id]
                was_semantic_resolved = track.slot_state in {"semantic_resolved", "label_pending"}
                track.slot_state = "active"

                segment_event, segment_reason, segment_score = self._assign_local_segment(
                    track=track,
                    frame_id=frame_id,
                    sequence=int(sequence),
                    timestamp_sec=float(timestamp_sec),
                    centroid=centroid,
                    bbox_2d=bbox_2d,
                    bbox_3d_min=bbox_3d_min,
                    bbox_3d_max=bbox_3d_max,
                )

                # Object-level geometry remains global and is used only for
                # internal object continuation and crop/RAP/VLM identity.  The
                # active Hydra slot is owned by the selected local segment.
                self._update_track_geometry(track, centroid, volume, bbox_2d, bbox_3d_min, bbox_3d_max)
                self._refresh_spatial_index(track)
                track.last_seen_frame_id = frame_id
                track.last_seen_sequence = int(sequence)
                track.last_seen_timestamp_sec = float(timestamp_sec)
                track.seen_count += 1
                track.metadata = dict(metadata)
                self._update_semantics(track, normalised_raw_label, str(label_source), float(label_confidence))
                track_event = "reactivated_track" if was_semantic_resolved else "matched_track"

                # Log track observation for tracking quality evaluation
                if self.coordinator and hasattr(self.coordinator, 'tracking_quality_recorder'):
                    assoc_components = None
                    if candidate_evaluations:
                        for row in candidate_evaluations:
                            if row.get("candidate_track_id") == track.track_id and row.get("selected"):
                                assoc_components = row.get("global_association_components")
                                break

                    self.coordinator.tracking_quality_recorder.log_track_observation(
                        track_id=track.track_id,
                        frame_id=frame_id,
                        sequence=sequence,
                        centroid_3d=list(centroid) if centroid is not None else [0, 0, 0],
                        centroid_2d=list(bbox_2d[:2]) if bbox_2d is not None else [0, 0],
                        bbox_volume_m3=float(volume) if volume is not None else 0.0,
                        mask_area_px=int(metadata.get("mask_area_px", 0)) if metadata else 0,
                        depth_mean_m=float(metadata.get("depth_mean_m", 0.0)) if metadata else 0.0,
                        mask_iou_3d=float(metadata.get("mask_iou_3d", 0.0)) if metadata and metadata.get("mask_iou_3d") else None,
                        quality_score=1.0,
                        global_association_components=assoc_components,
                        match_reason=match_reason
                    )

            self._frame_used_track_ids.add(track.track_id)
            track.last_segment_event = segment_event
            track.last_segment_match_reason = segment_reason
            track.last_segment_match_score = segment_score
            self._annotate_metadata(metadata, track, track_event, match_reason, match_score)
            record = self._track_record(track, track_event, match_reason, match_score)
            record["candidate_evaluations"] = candidate_evaluations
            record["segment_event"] = segment_event
            record["local_segment_event"] = segment_event
            record["local_segment_match_reason"] = segment_reason
            record["local_segment_match_score"] = None if segment_score is None else float(segment_score)
            return metadata, record
        finally:
            self._lock.release()


    @staticmethod
    def _best_route_from_evaluation(row: Dict[str, Any]) -> Optional[Tuple[int, float, str]]:
        routes = list(row.get("accepted_routes") or [])
        if not routes:
            return None
        best = min(routes, key=lambda item: (int(item.get("priority", 99)), float(item.get("score", 1e9))))
        return int(best["priority"]), float(best["score"]), str(best["reason"])

    @staticmethod
    def _assignment_utility(priority: int, score: float) -> float:
        """Convert lexicographic route quality into one global-assignment utility."""
        route_base = {0: 1000.0, 1: 800.0, 2: 600.0, 3: 300.0}.get(int(priority), -1e6)
        # Scores are distances for 3D routes and 1-IoU for the 2D route.
        return route_base - min(199.0, max(0.0, float(score)) * 100.0)

    @classmethod
    def _same_track_continuity_key(
        cls,
        evaluation: Dict[str, Any],
        mask_area: int,
    ) -> Tuple[float, float, float, float, float, float]:
        """Rank nested observations that prefer the same established track.

        Lower is better. Route strength remains primary, but temporal image
        continuity, centroid consistency and volume consistency decide between
        multiple representations of the same object. Mask area is only the last
        tie-breaker, so this does not blindly prefer a larger mask.
        """
        route = cls._best_route_from_evaluation(evaluation)
        priority = float(route[0]) if route is not None else 99.0
        route_score = float(route[1]) if route is not None else float("inf")
        iou = float(evaluation.get("bbox_2d_iou", 0.0) or 0.0)
        centroid = float(evaluation.get("centroid_distance_m", float("inf")))
        if not math.isfinite(centroid):
            centroid = 1e6
        ratio = float(evaluation.get("volume_ratio", float("inf")))
        if ratio > 0.0 and math.isfinite(ratio):
            volume_error = abs(math.log(ratio))
        else:
            volume_error = 1e6
        return (priority, -iou, centroid, volume_error, route_score, -float(mask_area))

    def _same_track_broader_mask_is_coherent(
        self,
        evaluation: Dict[str, Any],
        *,
        area_ratio: float,
        added_area_fraction: float,
    ) -> bool:
        """Return whether a broader nested mask may inherit one existing track.

        A2 is evaluated before this function and rejects enclosing masks that
        combine multiple established tracks. This helper therefore handles only
        a single-track expansion. Strong 3D support is required so a weak 2D-only
        overlap cannot make a wall/floor union take over an identity.
        """
        max_area_ratio = float(getattr(
            self.config, "persistent_same_track_max_parent_child_area_ratio", 3.0
        ))
        min_added_fraction = float(getattr(
            self.config, "persistent_same_track_min_added_area_fraction", 0.05
        ))
        max_route_priority = int(getattr(
            self.config, "persistent_same_track_broader_max_route_priority", 2
        ))
        if area_ratio <= 1.0 or area_ratio > max_area_ratio:
            return False
        if added_area_fraction < min_added_fraction:
            return False
        route = self._best_route_from_evaluation(evaluation)
        if route is None or int(route[0]) > max_route_priority:
            return False

        # Any explicit 3D contradiction blocks expansion. These rejections may
        # coexist with another accepted route in the diagnostics, so inspect them
        # directly rather than relying only on the selected route.
        rejections = set(evaluation.get("rejection_reasons") or [])
        hard_contradictions = {
            "accumulated_vertical_gap_exceeded",
            "accumulated_vertical_center_delta_exceeded",
            "accumulated_xy_gap_exceeded",
            "continuation_vertical_gap_exceeded",
            "continuation_vertical_center_delta_exceeded",
            "continuation_xy_gap_exceeded",
            "centroid_distance_exceeded",
        }
        if rejections & hard_contradictions:
            return False

        ratio = float(evaluation.get("volume_ratio", 1.0) or 1.0)
        max_volume_ratio = float(getattr(
            self.config, "persistent_same_track_broader_max_volume_ratio", 6.0
        ))
        if math.isfinite(ratio) and ratio > max_volume_ratio:
            return False
        return True

    @staticmethod
    def _greedy_maximize(
        weights: List[List[float]],
        threshold: float = 0.0,
        return_diagnostics: bool = False
    ) -> tuple:
        """Greedy independent matching: each row picks best column independently.

        Unlike Hungarian (1-to-1), this allows multiple rows to pick the same column.
        Each row is assigned to its highest-scoring column if score >= threshold,
        otherwise assigned to its private dummy column (new track).

        Args:
            weights: List[row_idx][col_idx] where cols 0..len(track_ids)-1 are tracks
                    and cols len(track_ids)..end are private dummy columns (one per row)
            threshold: Minimum score to accept a match (scores below create new track)
            return_diagnostics: If True, return (assignment, diagnostics) tuple

        Returns:
            assignment[row] = selected column index (track or dummy)
            diagnostics (if return_diagnostics=True): List of {best_col, best_score, passed_threshold, second_best_col, second_best_score}
        """
        if not weights:
            return ([], []) if return_diagnostics else []

        n = len(weights)
        m = len(weights[0])
        assignment = [-1] * n
        diagnostics = []

        for row_idx in range(n):
            row_weights = weights[row_idx]
            best_col = -1
            best_score = float('-inf')
            second_best_col = -1
            second_best_score = float('-inf')

            # Find highest and second-highest scoring columns for this row
            for col_idx in range(m):
                if row_weights[col_idx] > best_score:
                    second_best_score = best_score
                    second_best_col = best_col
                    best_score = row_weights[col_idx]
                    best_col = col_idx
                elif row_weights[col_idx] > second_best_score:
                    second_best_score = row_weights[col_idx]
                    second_best_col = col_idx

            # Assign to best column if above threshold, else to private dummy
            passed_threshold = best_score >= threshold
            if passed_threshold:
                assignment[row_idx] = best_col
            else:
                # Assign to this row's private dummy column (one per row after track cols)
                num_track_cols = m - n
                assignment[row_idx] = num_track_cols + row_idx

            if return_diagnostics:
                diagnostics.append({
                    'best_col': best_col,
                    'best_score': best_score,
                    'second_best_col': second_best_col,
                    'second_best_score': second_best_score,
                    'passed_threshold': passed_threshold,
                    'threshold': threshold,
                })

        return (assignment, diagnostics) if return_diagnostics else assignment

    def prepare_frame_assignments(
        self,
        observations: List[Dict[str, Any]],
        stage_ms: Optional[Dict[str, float]] = None,
    ) -> List[bool]:
        """Plan A2 redundancy suppression and E global assignment without mutating tracks.

        Each observation must contain ``metadata`` and may contain a boolean NumPy
        ``mask``. The method installs forced matches consumed by subsequent calls
        to :meth:`associate` and returns a keep/suppress flag per observation.

        ``stage_ms`` is an optional profiling side-channel only (Part 3). When
        provided, elapsed time for each internal sub-step is accumulated into
        it under fixed keys (``assignment_candidate_search_ms``,
        ``assignment_a2_redundancy_ms``, ``assignment_a3_nested_ms``,
        ``assignment_hungarian_ms``), plus two non-timing diagnostic counts
        (``assignment_candidate_count_total``, ``assignment_candidate_count_max``)
        recording how many candidate tracks ``_find_match`` evaluated per
        observation. None of this changes the returned keep-mask or any track
        state.
        """
        if not observations:
            return []
        # Part 3 Path B profiling: measure time blocked acquiring the lock,
        # separately from time spent doing work after acquiring it, to test
        # whether async RAP/VLM-thread lock contention explains the severe
        # scattered latency spikes observed in this method's timed sub-steps.
        # try/finally below is exactly what `with self._lock:` expands to;
        # this is a behavior-identical substitution, not a locking change.
        _lock_wait_t0 = time.perf_counter() if stage_ms is not None else 0.0
        self._lock.acquire()
        if stage_ms is not None:
            stage_ms["assignment_lock_wait_ms"] = stage_ms.get("assignment_lock_wait_ms", 0.0) + (time.perf_counter() - _lock_wait_t0) * 1000.0
        try:
            t0 = time.perf_counter() if stage_ms is not None else 0.0
            previews: List[Tuple[Optional[str], str, Optional[float], List[Dict[str, Any]]]] = []
            best_by_obs: List[Tuple[Optional[str], float]] = []
            best_eval_by_obs: List[Optional[Dict[str, Any]]] = []
            candidate_count_total = 0
            candidate_count_max = 0
            for item in observations:
                metadata = dict(item.get("metadata") or {})
                preview = self._find_match(
                    centroid=_as_xyz(metadata.get("centroid_3d")),
                    volume=_safe_float(metadata.get("bbox_volume_m3")),
                    bbox_2d=metadata.get("bbox_2d"),
                    bbox_3d_min=_as_xyz(metadata.get("bbox_3d_min")),
                    bbox_3d_max=_as_xyz(metadata.get("bbox_3d_max")),
                    timestamp_sec=float(item.get("timestamp_sec", 0.0)),
                    desired_hydra_label_id=int(item.get("desired_hydra_label_id", 0)),
                    respect_frame_used=False,
                    stage_ms=stage_ms,
                )
                previews.append(preview)
                candidate_count = len(preview[3])
                candidate_count_total += candidate_count
                candidate_count_max = max(candidate_count_max, candidate_count)
                best_track = None
                best_utility = -1e6
                best_evaluation: Optional[Dict[str, Any]] = None
                for row in preview[3]:
                    route = self._best_route_from_evaluation(row)
                    if route is None:
                        continue
                    utility = self._assignment_utility(route[0], route[1])
                    if utility > best_utility:
                        best_track = str(row.get("candidate_track_id", ""))
                        best_utility = utility
                        best_evaluation = row
                best_by_obs.append((best_track, best_utility))
                best_eval_by_obs.append(best_evaluation)
            if stage_ms is not None:
                stage_ms["assignment_candidate_search_ms"] = stage_ms.get("assignment_candidate_search_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
                stage_ms["assignment_candidate_count_total"] = stage_ms.get("assignment_candidate_count_total", 0.0) + float(candidate_count_total)
                stage_ms["assignment_candidate_count_max"] = max(stage_ms.get("assignment_candidate_count_max", 0.0), float(candidate_count_max))

            t0 = time.perf_counter() if stage_ms is not None else 0.0
            keep = [True] * len(observations)
            enabled = bool(getattr(self.config, "persistent_track_aware_redundancy_enabled", True))
            coverage_threshold = float(getattr(self.config, "persistent_redundancy_union_coverage_threshold", 0.90))
            contained_threshold = float(getattr(self.config, "persistent_redundancy_child_containment_threshold", 0.85))
            min_children = int(getattr(self.config, "persistent_redundancy_min_children", 2))
            if enabled:
                masks = [np.asarray(item.get("mask"), dtype=bool) if item.get("mask") is not None else None for item in observations]
                areas = [int(np.count_nonzero(mask)) if mask is not None else 0 for mask in masks]
                for large_idx in sorted(range(len(masks)), key=lambda idx: areas[idx], reverse=True):
                    large = masks[large_idx]
                    if large is None or areas[large_idx] <= 0:
                        continue
                    children: List[int] = []
                    for child_idx, child in enumerate(masks):
                        if child_idx == large_idx or child is None or areas[child_idx] >= areas[large_idx]:
                            continue
                        intersection = int(np.count_nonzero(child & large))
                        containment = intersection / max(1, areas[child_idx])
                        if containment >= contained_threshold:
                            children.append(child_idx)
                    if len(children) < min_children:
                        continue
                    union = np.zeros_like(large, dtype=bool)
                    for child_idx in children:
                        union |= masks[child_idx]
                    union_coverage = float(np.count_nonzero(union & large)) / max(1, areas[large_idx])
                    distinct_tracks = {best_by_obs[idx][0] for idx in children if best_by_obs[idx][0]}
                    child_utility = sum(max(0.0, best_by_obs[idx][1]) for idx in children if best_by_obs[idx][0])
                    large_utility = max(0.0, best_by_obs[large_idx][1])
                    # A2: preserve a decomposition when it explains the enclosing
                    # mask and supports at least two distinct established tracks.
                    if (union_coverage >= coverage_threshold and len(distinct_tracks) >= min_children
                            and child_utility > large_utility):
                        keep[large_idx] = False
                        observations[large_idx]["suppression_reason"] = "track_aware_union_redundancy"
                        observations[large_idx]["suppression_union_coverage"] = union_coverage
                        observations[large_idx]["suppression_child_indices"] = children
            if stage_ms is not None:
                stage_ms["assignment_a2_redundancy_ms"] = stage_ms.get("assignment_a2_redundancy_ms", 0.0) + (time.perf_counter() - t0) * 1000.0

            # A3: SAM can emit a stable whole-object mask and a nested partial
            # mask that both prefer the same established track. Global one-to-one
            # assignment alone would give the track to whichever has the slightly
            # better route residual and force the other into a new ID. Suppress
            # the weaker duplicate before assignment, using temporal continuity
            # rather than a fixed larger-mask preference.
            t0 = time.perf_counter() if stage_ms is not None else 0.0
            same_track_enabled = bool(getattr(
                self.config, "persistent_same_track_nested_suppression_enabled", True
            ))
            same_track_containment = float(getattr(
                self.config, "persistent_same_track_nested_containment_threshold", 0.90
            ))
            if same_track_enabled:
                masks = [np.asarray(item.get("mask"), dtype=bool) if item.get("mask") is not None else None for item in observations]
                areas = [int(np.count_nonzero(mask)) if mask is not None else 0 for mask in masks]
                # Evaluate the most strongly nested pairs first. Each suppression
                # is final for this frame, preventing chains from creating a new ID.
                nested_pairs: List[Tuple[float, int, int]] = []
                for a in range(len(masks)):
                    if not keep[a] or masks[a] is None or areas[a] <= 0:
                        continue
                    for b in range(a + 1, len(masks)):
                        if not keep[b] or masks[b] is None or areas[b] <= 0:
                            continue
                        small_idx, large_idx = (a, b) if areas[a] <= areas[b] else (b, a)
                        intersection = int(np.count_nonzero(masks[small_idx] & masks[large_idx]))
                        containment = intersection / max(1, areas[small_idx])
                        if containment >= same_track_containment:
                            nested_pairs.append((containment, small_idx, large_idx))
                nested_pairs.sort(reverse=True)
                for containment, small_idx, large_idx in nested_pairs:
                    if not keep[small_idx] or not keep[large_idx]:
                        continue
                    preferred_small = best_by_obs[small_idx][0]
                    preferred_large = best_by_obs[large_idx][0]
                    if not preferred_small or preferred_small != preferred_large:
                        continue
                    eval_small = best_eval_by_obs[small_idx]
                    eval_large = best_eval_by_obs[large_idx]
                    if eval_small is None or eval_large is None:
                        continue
                    key_small = self._same_track_continuity_key(eval_small, areas[small_idx])
                    key_large = self._same_track_continuity_key(eval_large, areas[large_idx])

                    # A3 expansion policy: when both nested observations describe
                    # one established track, allow a coherent broader mask to take
                    # over the track so its persistent geometry can grow. A2 has
                    # already removed enclosing masks that combine multiple
                    # established tracks. We therefore prefer the larger mask when
                    # it adds meaningful area, stays within a bounded expansion
                    # ratio, and has a strong 3D-supported association without
                    # contradictory geometry. Otherwise, retain the strongest
                    # temporal continuation as the conservative fallback.
                    area_ratio = float(areas[large_idx]) / max(1.0, float(areas[small_idx]))
                    added_area_fraction = max(
                        0.0,
                        float(areas[large_idx] - int(np.count_nonzero(masks[small_idx] & masks[large_idx])))
                        / max(1.0, float(areas[large_idx])),
                    )
                    promote_broader = self._same_track_broader_mask_is_coherent(
                        eval_large,
                        area_ratio=area_ratio,
                        added_area_fraction=added_area_fraction,
                    )
                    if promote_broader:
                        winner_idx, loser_idx = large_idx, small_idx
                        decision = "same_track_broader_mask_takeover"
                    else:
                        winner_idx, loser_idx = (
                            (small_idx, large_idx) if key_small < key_large else (large_idx, small_idx)
                        )
                        decision = "same_track_nested_duplicate"

                    keep[loser_idx] = False
                    observations[loser_idx]["suppression_reason"] = decision
                    observations[loser_idx]["suppression_containment"] = float(containment)
                    observations[loser_idx]["suppression_preferred_track_id"] = str(preferred_small)
                    observations[loser_idx]["suppression_winner_index"] = int(winner_idx)
                    observations[loser_idx]["suppression_area_ratio"] = float(area_ratio)
                    observations[loser_idx]["suppression_added_area_fraction"] = float(added_area_fraction)
                    observations[loser_idx]["suppression_broader_promoted"] = bool(promote_broader)
                    observations[loser_idx]["suppression_winner_continuity_key"] = list(
                        key_small if winner_idx == small_idx else key_large
                    )
                    observations[loser_idx]["suppression_loser_continuity_key"] = list(
                        key_large if loser_idx == large_idx else key_small
                    )
            if stage_ms is not None:
                stage_ms["assignment_a3_nested_ms"] = stage_ms.get("assignment_a3_nested_ms", 0.0) + (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter() if stage_ms is not None else 0.0
            retained = [idx for idx, flag in enumerate(keep) if flag]
            if not retained:
                if stage_ms is not None:
                    stage_ms["assignment_hungarian_ms"] = stage_ms.get("assignment_hungarian_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
                return keep
            candidate_track_ids = {
                str(row.get("candidate_track_id", ""))
                for preview in previews for row in preview[3]
                if self._best_route_from_evaluation(row) is not None
            }
            track_ids = [track_id for track_id in self._tracks if track_id in candidate_track_ids]
            track_columns = {track_id: index for index, track_id in enumerate(track_ids)}
            # Add one private dummy/new-track column per observation so no match
            # is ever forced. Invalid real-track pairs receive a very low utility.
            weights: List[List[float]] = []
            route_lookup: Dict[Tuple[int, int], Tuple[str, float, str, Dict[str, Any]]] = {}
            for row_pos, obs_idx in enumerate(retained):
                row_weights = [-1e6] * len(track_ids) + [0.0] * len(retained)
                for evaluation in previews[obs_idx][3]:
                    route = self._best_route_from_evaluation(evaluation)
                    if route is None:
                        continue
                    track_id = str(evaluation.get("candidate_track_id", ""))
                    col = track_columns.get(track_id)
                    if col is None:
                        continue
                    utility = self._assignment_utility(route[0], route[1])
                    row_weights[col] = utility
                    route_lookup[(row_pos, col)] = (track_id, route[1], route[2], evaluation)
                weights.append(row_weights)
            # Use greedy independent matching: each crop picks best track independently
            # This allows multiple crops to match same track (handles wall segmentation)
            # Threshold=-0.2: allows negative-score matches (e.g., ceiling revisits from different angle)
            # that are still better than creating a completely new track (which has score=0.0)
            assignment, diag = self._greedy_maximize(weights, threshold=-0.2, return_diagnostics=True)

            # Apply temporal tie-breaking for spatially overlapping objects (e.g., rug on floor)
            # When two tracks have very similar scores, prefer the one that was recently active
            for row_pos, obs_idx in enumerate(retained):
                col = assignment[row_pos]
                diag_info = diag[row_pos] if row_pos < len(diag) else {}
                best_score = diag_info.get('best_score', float('-inf'))
                second_best_score = diag_info.get('second_best_score', float('-inf'))
                best_col = diag_info.get('best_col', -1)
                second_best_col = diag_info.get('second_best_col', -1)

                # If scores are tied (within 0.05), prefer more recently active track
                # This keeps semantically different overlapping objects separate (e.g., rug vs floor)
                score_difference = best_score - second_best_score
                tie_threshold = 0.05

                if (0 <= best_col < len(track_ids) and 0 <= second_best_col < len(track_ids)
                    and score_difference >= 0 and score_difference <= tie_threshold):
                    # Scores are close; apply temporal consistency
                    best_track = self._tracks.get(track_ids[best_col])
                    second_best_track = self._tracks.get(track_ids[second_best_col])

                    if best_track and second_best_track:
                        # Prefer the track that was more recently observed
                        if second_best_track.last_seen_timestamp_sec > best_track.last_seen_timestamp_sec:
                            assignment[row_pos] = second_best_col
                            # Update diagnostics to reflect tie-break decision
                            diag_info['temporal_tie_break_applied'] = True
                            diag_info['tie_winner'] = track_ids[second_best_col]
                            diag_info['tie_loser'] = track_ids[best_col]

            for row_pos, obs_idx in enumerate(retained):
                col = assignment[row_pos]
                candidate_evals = previews[obs_idx][3]
                selected_track: Optional[str] = None
                selected_reason = "new_track"
                selected_score: Optional[float] = None

                # Log matching diagnostics
                diag_info = diag[row_pos] if row_pos < len(diag) else {}
                best_score = diag_info.get('best_score', float('-inf'))
                passed_threshold = diag_info.get('passed_threshold', False)

                if 0 <= col < len(track_ids) and (row_pos, col) in route_lookup:
                    selected_track, selected_score, selected_reason, eval_info = route_lookup[(row_pos, col)]
                    # Log accepted match with score details
                    mode = "revisit" if selected_track and selected_track in self._tracks and self._tracks[selected_track].seen_count > 1 else "recent"
                    self.logger.debug(
                        f"Match accepted: obs={obs_idx} → track={selected_track} | "
                        f"score={best_score:.4f} | mode={mode} | reason={selected_reason}"
                    )
                else:
                    # Log rejected match (new track)
                    best_track = track_ids[col] if 0 <= col < len(track_ids) else "none"
                    self.logger.debug(
                        f"New track: obs={obs_idx} | best_score={best_score:.4f} | "
                        f"best_track={best_track} | threshold=0.0"
                    )

                for evaluation in candidate_evals:
                    selected = str(evaluation.get("candidate_track_id", "")) == selected_track
                    evaluation["selected"] = bool(selected)
                    evaluation["selected_reason"] = selected_reason if selected else ""
                    evaluation["selected_score"] = selected_score if selected else None
                    evaluation["global_assignment"] = True
                candidate_id = str((observations[obs_idx].get("metadata") or {}).get("candidate_id", ""))
                self._forced_frame_matches[candidate_id] = (
                    selected_track, selected_reason, selected_score, candidate_evals
                )
            if stage_ms is not None:
                stage_ms["assignment_hungarian_ms"] = stage_ms.get("assignment_hungarian_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
            return keep
        finally:
            self._lock.release()

    def prepare_active_for_labeling(
        self,
        current_timestamp_sec: float,
        *,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.labeling_lifecycle.prepare_active_for_labeling(current_timestamp_sec, force=force)

    def release_labeling_request(self, track_id: str, reason: str) -> None:
        return self.labeling_lifecycle.release_labeling_request(track_id, reason)

    def set_labeling_status(self, track_id: str, status: str) -> None:
        return self.labeling_lifecycle.set_labeling_status(track_id, status)

    def is_semantic_labeling_open(self, track_id: str) -> bool:
        return self.labeling_lifecycle.is_semantic_labeling_open(track_id)

    def increment_vlm_attempt_count(self, track_id: str) -> Optional[int]:
        return self.labeling_lifecycle.increment_vlm_attempt_count(track_id)

    def record_vlm_attempt_detail(self, track_id: str, object_detail: str) -> None:
        return self.labeling_lifecycle.record_vlm_attempt_detail(track_id, object_detail)

    def get_waiting_record(self, track_id: str) -> Optional[Dict[str, Any]]:
        return self.labeling_lifecycle.get_waiting_record(track_id)

    def prune_expired_dynamic_tracks(self, current_timestamp_sec: float) -> List[Dict[str, Any]]:
        """Delete confirmed-dynamic tracks once presence confidence decays.

        Mirrors the fuser's own presence-confidence formula
        (resolvePresenceForSlot in fuser_presence.cpp) so both sides agree on what
        "gone" means -- but here it actually removes the track instead of
        just fading its rendering, once it's unlikely to still be where it
        was last seen. A person, animal, or mobile robot that hasn't been
        re-observed in a while has a real chance of having moved elsewhere,
        so keeping searching against its old position/geometry is wasted
        work for every future association call.

        Only tracks with a VLM-confirmed mobility_class of "dynamic" are
        eligible -- never a still-pending or static/unknown track. Gated by
        persistent_dynamic_track_expiry_enabled (off by default).

        Matches the existing merge/reanchor precedent (see
        merge_reanchor_duplicates -> _forget_spatial_index): the track is
        removed from _tracks and the spatial index, but its Hydra slot id is
        never freed for reuse -- same as a dropped track after a merge today.
        That is deliberate: reusing the slot for a genuinely different future
        object would need the fuser's suppression to auto-clear on a new
        observation, which is exactly the failure mode this avoids.

        Returns one record per deleted track (deletion has already happened
        by the time this returns) so the caller can notify the fuser and
        clean up its own per-track bookkeeping (crop registry, VLM retry
        pool, etc.).
        """
        if not bool(getattr(self.config, "persistent_dynamic_track_expiry_enabled", False)):
            return []
        half_life = max(1e-3, float(
            getattr(self.config, "persistent_dynamic_track_expiry_half_life_sec", 120.0)
        ))
        threshold = float(
            getattr(self.config, "persistent_dynamic_track_expiry_confidence_threshold", 0.5)
        )
        epsilon = max(0.0, float(
            getattr(self.config, "persistent_dynamic_track_expiry_observed_epsilon_sec", 1.5)
        ))
        now_sec = float(current_timestamp_sec)
        deleted: List[Dict[str, Any]] = []
        with self._lock:
            for track_id, track in list(self._tracks.items()):
                if not track.labeling_completed:
                    continue
                if str(track.mobility_class or "").strip().lower() != "dynamic":
                    continue
                age_sec = max(0.0, now_sec - float(track.last_seen_timestamp_sec))
                if age_sec <= epsilon:
                    continue
                decay_age_sec = age_sec - epsilon
                confidence = 0.5 ** (decay_age_sec / half_life)
                if confidence >= threshold:
                    continue
                deleted.append({
                    "track_id": track.track_id,
                    "internal_object_id": track.track_id,
                    "hydra_slot_id": int(track.hydra_label_id),
                    "hydra_label_id": int(track.hydra_label_id),
                    "hydra_slot_name": track.hydra_label_name,
                    "canonical_label": track.canonical_label,
                    "semantic_label": track.semantic_label,
                    "mobility_class": track.mobility_class,
                    "presence_confidence": float(confidence),
                    "age_sec": float(age_sec),
                    "reason": "dynamic_track_expired",
                })
                del self._tracks[track_id]
                self._forget_spatial_index(track_id)
        return deleted

    def complete_semantic_labeling(
        self,
        track_id: str,
        timestamp_sec: float,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        return self.labeling_lifecycle.complete_semantic_labeling(track_id, timestamp_sec, reason)

    def apply_vlm_result(
        self,
        track_id: str,
        label: str,
        confidence: float,
        mobility_class: str = "unknown",
        mobility_confidence: float = 0.0,
        object_detail: str = DEFAULT_OBJECT_DETAIL,
    ) -> Optional[Dict[str, Any]]:
        return self.labeling_lifecycle.apply_vlm_result(
            track_id,
            label,
            confidence,
            mobility_class=mobility_class,
            mobility_confidence=mobility_confidence,
            object_detail=object_detail,
        )

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
        return self.labeling_lifecycle.apply_rap_result(
            track_id,
            label,
            confidence,
            is_known,
            mobility_class=mobility_class,
            mobility_confidence=mobility_confidence,
            mobility_source=mobility_source,
            object_detail=object_detail,
        )

    def track_counts(self) -> Dict[str, int]:
        """Snapshot of track/slot totals, for the fuser-vs-phase1 node-count diagnostic.

        ``total_tracks_created``/``total_hydra_slots_allocated`` are cumulative
        (track ids and slot ids are never reused, so these only grow) and are
        the counterparts to compare against the fuser's ``object_nodes`` /
        ``distinct_hydra_object_slots``. ``active_track_count`` is the live
        snapshot (merged/dropped tracks excluded).
        """
        with self._lock:
            return {
                "active_track_count": len(self._tracks),
                "total_tracks_created": int(self._next_track_index) - 1,
                "total_hydra_slots_allocated": int(self._next_slot_index) - 1,
            }

    def get_hydra_label_id(self, track_id: str) -> int:
        """Return the Hydra semantic slot allocated to ``track_id`` (0 if unknown).

        This is exactly the number the fuser renders as the ``id_N`` line on
        each object node (show_slot_ids). Diagnostics that want to be
        cross-referenceable against the RViz display by eye -- e.g. VLM test
        crop/CSV naming -- should use this instead of an independent counter.
        """
        with self._lock:
            track = self._tracks.get(str(track_id))
            return int(track.hydra_label_id) if track is not None else 0

    def get_seen_count(self, track_id: str) -> int:
        """Return how many times ``track_id`` has been observed (1 on creation).

        Used to gate whether a track's real Hydra slot label is safe to paint
        into the semantic image sent to Hydra yet -- see phase1.py's
        publish_confirmed_label check. Returns 0 for an unknown track id
        (already merged away, or never existed).
        """
        with self._lock:
            track = self._tracks.get(str(track_id))
            return int(track.seen_count) if track is not None else 0

    def _has_slot_capacity(self) -> bool:
        if not bool(getattr(self.config, "persistent_use_hydra_slots", False)):
            return len(self._tracks) < int(self.config.persistent_max_tracks)
        first = int(self.config.persistent_slot_first_label_id)
        last = first + int(self.config.persistent_slot_count) - 1
        return any(
            slot_id not in self._reserved_slot_ids and slot_id not in self._allocated_slot_ids
            for slot_id in range(first, last + 1)
        )

    def _next_available_slot_id(self) -> Optional[int]:
        first = int(self.config.persistent_slot_first_label_id)
        last = first + int(self.config.persistent_slot_count) - 1
        start = max(first, first + self._next_slot_index - 1)
        for slot_id in range(start, last + 1):
            if slot_id not in self._reserved_slot_ids and slot_id not in self._allocated_slot_ids:
                self._next_slot_index = slot_id - first + 2
                return slot_id
        for slot_id in range(first, start):
            if slot_id not in self._reserved_slot_ids and slot_id not in self._allocated_slot_ids:
                self._next_slot_index = slot_id - first + 2
                return slot_id
        return None

    def _new_track(
        self,
        *,
        frame_id: str,
        sequence: int,
        timestamp_sec: float,
        centroid: Optional[np.ndarray],
        volume: Optional[float],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
        desired_hydra_label_id: int,
        desired_hydra_label_name: str,
        raw_label: str,
        label_source: str,
        label_confidence: float,
        metadata: Dict[str, Any],
        use_hydra_slot: bool,
        forced_hydra_slot_id: int = 0,
    ) -> PersistentObjectTrack:
        track_index = self._next_track_index
        track_id = f"{self.config.persistent_track_prefix}{track_index:06d}"
        self._next_track_index += 1

        if bool(getattr(self.config, "persistent_use_hydra_slots", False)) and bool(use_hydra_slot):
            first = int(self.config.persistent_slot_first_label_id)
            last = first + int(self.config.persistent_slot_count) - 1
            forced = int(forced_hydra_slot_id or 0)
            if forced:
                if not (first <= forced <= last):
                    raise RuntimeError(f"Forced Hydra slot {forced} is outside the configured pool")
                # A duplicate reference in one frame must not merge two masks.
                # Fall back to an unused slot instead of raising or aliasing.
                hydra_label_id = forced if forced not in self._allocated_slot_ids else 0
            else:
                hydra_label_id = 0
            if not hydra_label_id:
                allocated = self._next_available_slot_id()
                if allocated is None:
                    raise RuntimeError("No unreserved Hydra object slots remain")
                hydra_label_id = int(allocated)
            self._allocated_slot_ids.add(hydra_label_id)
            width = max(1, int(self.config.persistent_slot_label_width))
            slot_index = hydra_label_id - first + 1
            # For a frozen cross-session reference, preserve the name that
            # Phase 1 loaded from the generated Hydra lookup table.
            if int(forced_hydra_slot_id or 0) == hydra_label_id and str(desired_hydra_label_name):
                hydra_label_name = str(desired_hydra_label_name)
            else:
                hydra_label_name = f"{self.config.persistent_slot_label_prefix}{slot_index:0{width}d}"
            # Slot IDs are intentionally visible in both maps during discovery.
            instance_id = hydra_label_id
            semantic_kind = "slot"
        else:
            instance_id = self._next_instance_id
            if instance_id > 65535:
                raise RuntimeError("Persistent instance ID space exhausted for 16UC1 output")
            self._next_instance_id += 1
            hydra_label_id = int(desired_hydra_label_id)
            hydra_label_name = str(desired_hydra_label_name)
            semantic_kind = "class"

        canonical_label = raw_label or "unknown_object"
        track = PersistentObjectTrack(
            track_id=track_id,
            instance_id=int(instance_id),
            hydra_label_id=int(hydra_label_id),
            hydra_label_name=str(hydra_label_name),
            semantic_kind=semantic_kind,
            canonical_label=canonical_label,
            label_source=str(label_source),
            label_confidence=float(label_confidence),
            first_seen_frame_id=frame_id,
            first_seen_sequence=int(sequence),
            first_seen_timestamp_sec=float(timestamp_sec),
            last_seen_frame_id=frame_id,
            last_seen_sequence=int(sequence),
            last_seen_timestamp_sec=float(timestamp_sec),
            centroid_3d=centroid.copy() if centroid is not None else None,
            bbox_volume_m3=volume,
            bbox_2d=bbox_2d,
            bbox_3d_min=bbox_3d_min.copy() if bbox_3d_min is not None else None,
            bbox_3d_max=bbox_3d_max.copy() if bbox_3d_max is not None else None,
            last_bbox_3d_min=bbox_3d_min.copy() if bbox_3d_min is not None else None,
            last_bbox_3d_max=bbox_3d_max.copy() if bbox_3d_max is not None else None,
            raw_rap_label=raw_label if label_source == "rap" else "",
            metadata=dict(metadata),
        )
        segment = self._new_segment_for_track(
            track=track,
            frame_id=frame_id,
            sequence=int(sequence),
            timestamp_sec=float(timestamp_sec),
            centroid=centroid,
            bbox_2d=bbox_2d,
            bbox_3d_min=bbox_3d_min,
            bbox_3d_max=bbox_3d_max,
            forced_slot_id=int(track.hydra_label_id),
            forced_slot_name=str(track.hydra_label_name),
        )
        track.segments[int(segment.hydra_label_id)] = segment
        self._activate_segment(track, segment)
        self._add_evidence(track, raw_label, label_source, label_confidence)

        # Log track birth for tracking quality evaluation
        if self.coordinator and hasattr(self.coordinator, 'tracking_quality_recorder'):
            self.coordinator.tracking_quality_recorder.log_track_birth(
                frame_id=frame_id,
                sequence=sequence,
                track_id=track.track_id
            )

            # Also log initial observation
            self.coordinator.tracking_quality_recorder.log_track_observation(
                track_id=track.track_id,
                frame_id=frame_id,
                sequence=sequence,
                centroid_3d=list(centroid) if centroid is not None else [0, 0, 0],
                centroid_2d=list(bbox_2d[:2]) if bbox_2d is not None else [0, 0],
                bbox_volume_m3=float(volume) if volume is not None else 0.0,
                mask_area_px=int(metadata.get("mask_area_px", 0)) if metadata else 0,
                depth_mean_m=float(metadata.get("depth_mean_m", 0.0)) if metadata else 0.0,
                mask_iou_3d=float(metadata.get("mask_iou_3d", 0.0)) if metadata and metadata.get("mask_iou_3d") else None,
                quality_score=1.0,
                match_reason="new_track"
            )

        return track

    def _allocate_slot_for_segment(self) -> Tuple[int, str, int]:
        return self.local_segments._allocate_slot_for_segment()

    def _new_segment_for_track(
        self,
        *,
        track: PersistentObjectTrack,
        frame_id: str,
        sequence: int,
        timestamp_sec: float,
        centroid: Optional[np.ndarray],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
        forced_slot_id: int = 0,
        forced_slot_name: str = "",
    ) -> PersistentObjectSegment:
        return self.local_segments._new_segment_for_track(
            track=track, frame_id=frame_id, sequence=sequence, timestamp_sec=timestamp_sec,
            centroid=centroid, bbox_2d=bbox_2d, bbox_3d_min=bbox_3d_min, bbox_3d_max=bbox_3d_max,
            forced_slot_id=forced_slot_id, forced_slot_name=forced_slot_name,
        )

    @staticmethod
    def _activate_segment(track: PersistentObjectTrack, segment: PersistentObjectSegment) -> None:
        return TrackerLocalSegments._activate_segment(track, segment)

    def _segment_xy_span_after_update(
        self,
        segment: PersistentObjectSegment,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> Optional[float]:
        return self.local_segments._segment_xy_span_after_update(segment, bbox_3d_min, bbox_3d_max)

    def _update_segment_geometry(
        self,
        segment: PersistentObjectSegment,
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
        return self.local_segments._update_segment_geometry(
            segment, frame_id=frame_id, sequence=sequence, timestamp_sec=timestamp_sec,
            centroid=centroid, bbox_2d=bbox_2d, bbox_3d_min=bbox_3d_min, bbox_3d_max=bbox_3d_max,
            expand_bbox=expand_bbox,
        )

    def _assign_local_segment(
        self,
        *,
        track: PersistentObjectTrack,
        frame_id: str,
        sequence: int,
        timestamp_sec: float,
        centroid: Optional[np.ndarray],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> Tuple[str, str, Optional[float]]:
        return self.local_segments._assign_local_segment(
            track=track, frame_id=frame_id, sequence=sequence, timestamp_sec=timestamp_sec,
            centroid=centroid, bbox_2d=bbox_2d, bbox_3d_min=bbox_3d_min, bbox_3d_max=bbox_3d_max,
        )

    # ------------------------------------------------------------------
    # Loop-closure re-anchoring
    #
    # When the SLAM back-end folds an accumulated-drift correction into the
    # ``map -> odom`` transform, every cached track/segment coordinate is stale
    # by that same rigid step.  ``reanchor_all`` moves the geometry in place;
    # ``merge_reanchor_duplicates`` folds identities that were split before the
    # correction and only coincide once the geometry has been moved.  Neither
    # runs in the steady-state association path.
    # ------------------------------------------------------------------
    def reanchor_all(
        self,
        rotation: Any,
        translation: Any,
        *,
        stamp: Optional[float] = None,
    ) -> int:
        return self.reanchor.reanchor_all(rotation, translation, stamp=stamp)

    def last_reanchor(self) -> Optional[Dict[str, Any]]:
        return self.reanchor.last_reanchor()

    def _reanchor_labels_compatible(
        self, a: PersistentObjectTrack, b: PersistentObjectTrack
    ) -> bool:
        return self.reanchor._reanchor_labels_compatible(a, b)

    def _merge_track_pair(
        self,
        keep: PersistentObjectTrack,
        drop: PersistentObjectTrack,
        *,
        iou_3d: float,
        distance_m: float,
        reason: str,
        adopt_drop_geometry: bool,
    ) -> None:
        return self.reanchor._merge_track_pair(
            keep, drop, iou_3d=iou_3d, distance_m=distance_m, reason=reason,
            adopt_drop_geometry=adopt_drop_geometry,
        )

    def merge_reanchor_duplicates(
        self,
        *,
        correction_translation_m: float = 0.0,
        now_sec: Optional[float] = None,
        recent_window_sec: float = 5.0,
        distance_slack_m: float = 0.6,
        min_iou_3d: float = 0.30,
        max_centroid_distance_m: Optional[float] = None,
    ) -> int:
        return self.reanchor.merge_reanchor_duplicates(
            correction_translation_m=correction_translation_m, now_sec=now_sec,
            recent_window_sec=recent_window_sec, distance_slack_m=distance_slack_m,
            min_iou_3d=min_iou_3d, max_centroid_distance_m=max_centroid_distance_m,
        )

    def debug_snapshot(self) -> List[Dict[str, Any]]:
        """Read-only geometry dump for tests and loop-closure diagnostics."""
        with self._lock:
            return [
                {
                    "track_id": track.track_id,
                    "internal_object_id": track.track_id,
                    "seen_count": int(track.seen_count),
                    "semantic_kind": track.semantic_kind,
                    "canonical_label": track.canonical_label,
                    "semantic_label": track.semantic_label,
                    "centroid_3d": _as_list(track.centroid_3d),
                    "bbox_3d_min": _as_list(track.bbox_3d_min),
                    "bbox_3d_max": _as_list(track.bbox_3d_max),
                    "segment_slot_ids": sorted(int(s) for s in track.segments),
                    "reanchor_merged_from": list(
                        track.metadata.get("reanchor_merged_from", [])
                    ),
                }
                for track in self._tracks.values()
            ]

    def _spatial_cell_size(self) -> float:
        return self.spatial_index._spatial_cell_size()

    def _spatial_cells(
        self, bbox_min: np.ndarray, bbox_max: np.ndarray, padding: float = 0.0
    ) -> Optional[Set[Tuple[int, int]]]:
        return self.spatial_index._spatial_cells(bbox_min, bbox_max, padding)

    def _forget_spatial_index(self, track_id: str) -> None:
        return self.spatial_index._forget_spatial_index(track_id)

    def _refresh_spatial_index(self, track: PersistentObjectTrack) -> None:
        return self.spatial_index._refresh_spatial_index(track)

    def _candidate_track_ids(
        self,
        centroid: Optional[np.ndarray],
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> List[str]:
        return self.spatial_index._candidate_track_ids(centroid, bbox_3d_min, bbox_3d_max)

    def _find_match(
        self,
        *,
        centroid: Optional[np.ndarray],
        volume: Optional[float],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
        timestamp_sec: float,
        desired_hydra_label_id: int,
        respect_frame_used: bool = True,
        stage_ms: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[str], str, Optional[float], List[Dict[str, Any]]]:
        """Evaluate tracks using a quorum-gated, mode-aware global score.

        No individual route may approve a match. At least the configured number
        of independent evidence groups must pass, and the weighted score must
        exceed the recent or revisit threshold. The score is exposed as one
        accepted route so the existing Hungarian frame assignment remains intact.

        ``stage_ms`` is an optional profiling side-channel only (Part 3). When
        provided, elapsed time across all candidates evaluated by this call is
        accumulated into it under fixed keys (``assignment_row_init_ms``,
        ``assignment_3d_geometry_ms``, ``assignment_centroid_iou_ms``,
        ``assignment_scoring_ms``). This never changes the returned match,
        score, or evaluation rows.
        """
        best_id: Optional[str] = None
        best_key: Optional[Tuple[int, float]] = None
        best_reason = ""
        evaluations: List[Dict[str, Any]] = []
        has_3d_footprint = bbox_3d_min is not None and bbox_3d_max is not None
        global_enabled = bool(getattr(self.config, "persistent_global_association_enabled", True))
        row_init_ms = 0.0
        geometry_3d_ms = 0.0
        centroid_iou_ms = 0.0
        scoring_ms = 0.0

        def consider(track_id: str, priority: int, residual: float, reason: str, row: Dict[str, Any]) -> None:
            nonlocal best_id, best_key, best_reason
            row.setdefault("accepted_routes", []).append({
                "priority": int(priority), "score": float(residual), "reason": reason
            })
            key = (int(priority), float(residual))
            if best_key is None or key < best_key:
                best_id, best_key, best_reason = track_id, key, reason

        for track_id in self._candidate_track_ids(centroid, bbox_3d_min, bbox_3d_max):
            _t0 = time.perf_counter() if stage_ms is not None else 0.0
            track = self._tracks[track_id]
            # Part 3: candidate_centroid_3d/bbox_3d_min/bbox_3d_max/
            # last_bbox_3d_min/last_bbox_3d_max were previously computed here
            # via _as_list() (5 numpy->list conversions) for every candidate,
            # but were never read anywhere in this file, nodes/phase1.py, or
            # any other package in this workspace (confirmed by search before
            # removal) -- purely wasted work on the hot path. Removed; every
            # other row field (routing/rejection bookkeeping actually used by
            # prepare_frame_assignments and downstream track records) is
            # unchanged.
            row: Dict[str, Any] = {
                "candidate_track_id": track_id,
                "candidate_seen_count": int(track.seen_count),
                "candidate_last_seen_timestamp_sec": float(track.last_seen_timestamp_sec),
                "candidate_bbox_volume_m3": track.bbox_volume_m3,
                "accepted_routes": [],
                "rejection_reasons": [],
            }
            if respect_frame_used and track_id in self._frame_used_track_ids:
                row["rejection_reasons"].append("track_already_used_in_current_frame")
                evaluations.append(row)
                if stage_ms is not None:
                    row_init_ms += (time.perf_counter() - _t0) * 1000.0
                continue

            if (
                not bool(getattr(self.config, "persistent_use_hydra_slots", False))
                and bool(self.config.persistent_require_known_label_match)
                and desired_hydra_label_id != self.config.persistent_unclassified_label_id
                and track.hydra_label_id != self.config.persistent_unclassified_label_id
                and desired_hydra_label_id != track.hydra_label_id
            ):
                row["rejection_reasons"].append("known_label_mismatch")
                evaluations.append(row)
                if stage_ms is not None:
                    row_init_ms += (time.perf_counter() - _t0) * 1000.0
                continue

            if stage_ms is not None:
                row_init_ms += (time.perf_counter() - _t0) * 1000.0
                _t0 = time.perf_counter()
            age_sec = max(0.0, float(timestamp_sec) - float(track.last_seen_timestamp_sec))
            recent_mode = age_sec <= float(self.config.persistent_continuation_max_age_sec)
            mode = "recent" if recent_mode else "revisit"
            row["age_sec"] = float(age_sec)
            row["association_mode"] = mode
            track_has_3d = track.bbox_3d_min is not None and track.bbox_3d_max is not None
            track_has_last_3d = track.last_bbox_3d_min is not None and track.last_bbox_3d_max is not None

            historical_score = 0.0
            recent_score = 0.0
            centroid_score = 0.0
            vertical_score = 0.0
            iou_score = 0.0
            historical_pass = False
            recent_pass = False
            centroid_pass = False
            vertical_pass = False
            image_pass = False
            reliable_3d = bool(has_3d_footprint and track_has_3d)

            if has_3d_footprint and track_has_3d:
                vertical_gap = _aabb_gap_z(bbox_3d_min, bbox_3d_max, track.bbox_3d_min, track.bbox_3d_max)
                vertical_center_delta = _aabb_center_delta_z(bbox_3d_min, bbox_3d_max, track.bbox_3d_min, track.bbox_3d_max)
                accumulated_gap_xy = _aabb_gap_xy(bbox_3d_min, bbox_3d_max, track.bbox_3d_min, track.bbox_3d_max)
                accumulated_center_xy = _aabb_center_distance_xy(bbox_3d_min, bbox_3d_max, track.bbox_3d_min, track.bbox_3d_max)
                # Use 3D overlap (with padding, Z-ranges are now consistent)
                overlap_volume, overlap_x, overlap_y, overlap_z = _aabb_overlap_fraction_3d(
                    bbox_3d_min, bbox_3d_max, track.bbox_3d_min, track.bbox_3d_max
                )
                row.update({
                    "accumulated_gap_xy_m": accumulated_gap_xy,
                    "accumulated_vertical_gap_m": vertical_gap,
                    "accumulated_vertical_center_delta_m": vertical_center_delta,
                    "accumulated_center_distance_xy_m": accumulated_center_xy,
                    "historical_overlap_fraction_3d": overlap_volume,
                    "historical_overlap_fraction_x": overlap_x,
                    "historical_overlap_fraction_y": overlap_y,
                    "historical_overlap_fraction_z": overlap_z,
                })
                if accumulated_gap_xy > self.config.persistent_revisit_overlap_gap_m:
                    row["rejection_reasons"].append("accumulated_xy_gap_exceeded")

                gap_score = _gaussian_compatibility(
                    accumulated_gap_xy,
                    max(float(self.config.persistent_revisit_overlap_gap_m), 1e-3),
                )
                # Historical now includes Z (3D volume overlap)
                # For touching/overlapping: use volume. For separated: use distance score (no 0.50 damper)
                historical_score = max(overlap_volume, gap_score if accumulated_gap_xy > 0.0 else overlap_volume)
                min_hist = float(getattr(self.config, "persistent_global_historical_overlap_pass", 0.30))
                min_axis = float(getattr(self.config, "persistent_global_min_axis_overlap", 0.20))
                max_vertical_gap = float(getattr(self.config, "persistent_max_vertical_gap_m", 0.12))
                vertical_compatible = bool(vertical_gap <= max_vertical_gap)
                # The XY-touch shortcut alone cannot tell a floor from something
                # resting ON it -- their footprints touch/overlap in XY with a
                # ~0 gap either way. Require vertical compatibility too, so a
                # track only counts as "touching" when it is also plausibly at
                # the same height, not just anywhere above/below the same spot.
                historical_pass = bool(
                    (overlap_volume >= min_hist and overlap_x >= min_axis and overlap_y >= min_axis)
                    or (
                        accumulated_gap_xy <= float(getattr(self.config, "persistent_global_touch_gap_pass_m", 0.02))
                        and vertical_compatible
                    )
                )
                vertical_score = _gaussian_compatibility(
                    vertical_gap,
                    max(float(getattr(self.config, "persistent_global_vertical_sigma_m", 0.15)), 1e-3),
                )
                vertical_pass = vertical_compatible

            # Recent Continuation removed as redundant (3D Historical now handles continuity)
            recent_score = 0.0
            recent_pass = False

            if stage_ms is not None:
                geometry_3d_ms += (time.perf_counter() - _t0) * 1000.0
                _t0 = time.perf_counter()
            ratio = _volume_ratio(volume, track.bbox_volume_m3)
            row["volume_ratio"] = float(ratio)
            # Hard veto, not a vote: footprint and containment below are both
            # computed as "fraction of the new, small observation contained in
            # the track's box" -- once a track's accumulated box is inflated
            # (e.g. one bad early merge), those two normally-separate votes
            # both pass almost for free for anything spatially inside it, and
            # quorum is satisfied without the match ever being physically
            # plausible.
            max_volume_ratio = float(getattr(self.config, "persistent_max_volume_ratio", 3.0))
            hard_volume_contradiction = bool(
                reliable_3d and math.isfinite(ratio) and ratio > max_volume_ratio
            )
            if hard_volume_contradiction:
                row["rejection_reasons"].append("volume_ratio_exceeded")

            containment_score = 0.0
            containment_pass = False
            if has_3d_footprint and track_has_3d:
                containment = _aabb_3d_containment(
                    bbox_3d_min, bbox_3d_max, track.bbox_3d_min, track.bbox_3d_max
                )
                containment_score = float(containment)
                containment_pass = bool(
                    containment >= float(getattr(self.config, "persistent_global_containment_threshold", 0.90))
                )
                row["bbox_3d_containment"] = float(containment)

            if centroid is not None and track.centroid_3d is not None:
                if recent_mode:
                    distance = float(np.linalg.norm(centroid - track.centroid_3d))
                else:
                    distance = float(np.linalg.norm(centroid[:2] - track.centroid_3d[:2]))
                row["centroid_distance_m"] = distance
                base_sigma = float(getattr(self.config, "persistent_global_centroid_sigma_m", 0.50))
                track_size = float(track.bbox_volume_m3) if track.bbox_volume_m3 and track.bbox_volume_m3 > 0 else 1.0
                # Discrete sigma scaling by size to handle fragmentation without over-merging
                if track_size < 5.0:
                    sigma = base_sigma  # small objects: tight matching
                elif track_size < 20.0:
                    sigma = 0.70  # medium objects: moderate tolerance
                else:
                    sigma = 0.90  # large accumulated: loose but capped
                centroid_score = _gaussian_compatibility(distance, sigma)
                row["centroid_sigma_m"] = float(sigma)
                row["centroid_track_size_m3"] = float(track_size)
                row["centroid_distance_mode"] = "3d_recent" if recent_mode else "2d_revisit"
                row["centroid_sigma_category"] = "small" if track_size < 5 else ("medium" if track_size < 20 else "large")
                centroid_pass = distance <= float(getattr(
                    self.config, "persistent_global_centroid_pass_m", self.config.persistent_max_match_distance_m
                ))

            iou = _bbox_iou(bbox_2d, track.bbox_2d)
            row["bbox_2d_iou"] = float(iou)
            iou_score = float(iou)
            if recent_mode:
                image_pass = bool(iou >= self.config.persistent_min_2d_iou)
            else:
                revisit_iou_threshold = float(getattr(self.config, "persistent_revisit_min_2d_iou", 0.75))
                image_pass = bool(iou >= revisit_iou_threshold)

            if stage_ms is not None:
                centroid_iou_ms += (time.perf_counter() - _t0) * 1000.0
                _t0 = time.perf_counter()

            if not global_enabled:
                # Compatibility fallback for controlled rollback.
                if historical_pass:
                    consider(track_id, 0, 1.0 - historical_score, "accumulated_3d_footprint", row)
                if recent_pass:
                    consider(track_id, 1, 1.0 - recent_score, "recent_3d_footprint_continuation", row)
                if centroid_pass:
                    consider(track_id, 2, 1.0 - centroid_score, "centroid_3d", row)
                if image_pass:
                    consider(track_id, 3, 1.0 - iou_score, "bbox_2d_iou", row)
                evaluations.append(row)
                if stage_ms is not None:
                    scoring_ms += (time.perf_counter() - _t0) * 1000.0
                continue

            footprint_vote = bool(historical_pass or recent_pass)
            temporal_score = _gaussian_compatibility(
                age_sec, max(float(self.config.persistent_continuation_max_age_sec), 1e-3)
            )
            temporal_pass = bool(recent_mode)
            votes = {
                "footprint": footprint_vote,
                "centroid": bool(centroid_pass),
                "vertical": bool(vertical_pass),
                "image": bool(image_pass),
                "containment": bool(containment_pass),
            }
            # When depth is unavailable, temporal freshness is an independent
            # fallback cue paired with image overlap. It is never counted when
            # reliable 3D exists, so it cannot conceal a physical contradiction.
            if not reliable_3d:
                votes["temporal"] = temporal_pass
            pass_count = sum(1 for passed in votes.values() if passed)
            row["evidence_group_passes"] = votes
            row["independent_pass_count"] = int(pass_count)
            row["temporal_compatibility_score"] = float(temporal_score)

            if recent_mode:
                weights = {
                    "historical": float(getattr(self.config, "persistent_global_recent_weight_historical", 0.20)),
                    "recent": float(getattr(self.config, "persistent_global_recent_weight_recent", 0.30)),
                    "centroid": float(getattr(self.config, "persistent_global_recent_weight_centroid", 0.25)),
                    "vertical": float(getattr(self.config, "persistent_global_recent_weight_vertical", 0.15)),
                    "image": float(getattr(self.config, "persistent_global_recent_weight_image", 0.10)),
                    "containment": float(getattr(self.config, "persistent_global_recent_weight_containment", 0.00)),
                }
                min_score = float(getattr(self.config, "persistent_global_recent_min_score", 0.55))
            else:
                weights = {
                    "historical": float(getattr(self.config, "persistent_global_revisit_weight_historical", 0.45)),
                    "recent": float(getattr(self.config, "persistent_global_revisit_weight_recent", 0.00)),
                    "centroid": float(getattr(self.config, "persistent_global_revisit_weight_centroid", 0.30)),
                    "vertical": float(getattr(self.config, "persistent_global_revisit_weight_vertical", 0.20)),
                    "image": float(getattr(self.config, "persistent_global_revisit_weight_image", 0.05)),
                    "containment": float(getattr(self.config, "persistent_global_revisit_weight_containment", 0.00)),
                }
                min_score = float(getattr(self.config, "persistent_global_revisit_min_score", 0.70))

            if not reliable_3d:
                # Explicit degraded mode: image overlap plus temporal freshness.
                # This preserves tracking through isolated invalid-depth frames
                # while still requiring two independent cues.
                score = 0.70 * iou_score + 0.30 * temporal_score
            else:
                total_weight = max(sum(weights.values()), 1e-9)
                score = (
                    weights["historical"] * historical_score
                    + weights["recent"] * recent_score
                    + weights["centroid"] * centroid_score
                    + weights["vertical"] * vertical_score
                    + weights["image"] * iou_score
                    + weights["containment"] * containment_score
                ) / total_weight
            row["global_association_components"] = {
                "historical_overlap": float(historical_score),
                "recent_overlap": float(recent_score),
                "centroid_3d": float(centroid_score),
                "vertical_compatibility": float(vertical_score),
                "bbox_2d_iou": float(iou_score),
                "bbox_3d_containment": float(containment_score),
            }
            row["global_association_weights"] = weights
            row["global_association_score"] = float(score)
            row["global_association_min_score"] = float(min_score)

            min_groups = int(getattr(self.config, "persistent_global_min_independent_groups", 2))
            hard_2d_contradiction = bool(
                getattr(self.config, "persistent_global_block_2d_on_3d_contradiction", True)
                and reliable_3d
                and image_pass
                and not footprint_vote
                and not centroid_pass
            )
            if hard_2d_contradiction:
                row["rejection_reasons"].append("reliable_3d_contradicts_2d_match")
            if pass_count < min_groups:
                row["rejection_reasons"].append("insufficient_independent_evidence")
            if score < min_score:
                row["rejection_reasons"].append("global_association_score_below_threshold")

            if (
                not hard_2d_contradiction
                and not hard_volume_contradiction
                and pass_count >= min_groups
                and score >= min_score
            ):
                # Priority zero lets the existing assignment utility rank all valid
                # candidates directly by the common global score.
                consider(track_id, 0, 1.0 - score, f"global_{mode}_association", row)

            evaluations.append(row)
            if stage_ms is not None:
                scoring_ms += (time.perf_counter() - _t0) * 1000.0

        for row in evaluations:
            row["selected"] = bool(row.get("candidate_track_id") == best_id)
            row["selected_reason"] = best_reason if row["selected"] else ""
            row["selected_score"] = None if not row["selected"] or best_key is None else float(best_key[1])
        if stage_ms is not None:
            stage_ms["assignment_row_init_ms"] = stage_ms.get("assignment_row_init_ms", 0.0) + row_init_ms
            stage_ms["assignment_3d_geometry_ms"] = stage_ms.get("assignment_3d_geometry_ms", 0.0) + geometry_3d_ms
            stage_ms["assignment_centroid_iou_ms"] = stage_ms.get("assignment_centroid_iou_ms", 0.0) + centroid_iou_ms
            stage_ms["assignment_scoring_ms"] = stage_ms.get("assignment_scoring_ms", 0.0) + scoring_ms
        return best_id, best_reason, None if best_key is None else float(best_key[1]), evaluations

    def _update_track_geometry(
        self,
        track: PersistentObjectTrack,
        centroid: Optional[np.ndarray],
        volume: Optional[float],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> None:
        alpha = float(self.config.persistent_centroid_update_alpha)
        if centroid is not None:
            track.centroid_3d = centroid.copy() if track.centroid_3d is None else alpha * track.centroid_3d + (1.0 - alpha) * centroid
        if volume is not None and volume > 0.0:
            track.bbox_volume_m3 = volume if track.bbox_volume_m3 is None or track.bbox_volume_m3 <= 0.0 else alpha * float(track.bbox_volume_m3) + (1.0 - alpha) * float(volume)
        if bbox_2d:
            track.bbox_2d = bbox_2d
        if bbox_3d_min is not None and bbox_3d_max is not None:
            track.bbox_3d_min = bbox_3d_min.copy() if track.bbox_3d_min is None else np.minimum(track.bbox_3d_min, bbox_3d_min)
            track.bbox_3d_max = bbox_3d_max.copy() if track.bbox_3d_max is None else np.maximum(track.bbox_3d_max, bbox_3d_max)
            track.last_bbox_3d_min = bbox_3d_min.copy()
            track.last_bbox_3d_max = bbox_3d_max.copy()

    @staticmethod
    def _source_rank(source: str) -> int:
        return TrackerSemanticEvidence._source_rank(source)

    def _canonicalise_label(self, raw_label: Any) -> str:
        return self.semantic_evidence._canonicalise_label(raw_label)

    def _source_weight(self, source: str) -> float:
        return self.semantic_evidence._source_weight(source)

    def _add_evidence(self, track: PersistentObjectTrack, raw_label: str, source: str, confidence: float) -> None:
        return self.semantic_evidence._add_evidence(track, raw_label, source, confidence)

    def _update_semantics(self, track: PersistentObjectTrack, raw_label: str, source: str, confidence: float) -> None:
        return self.semantic_evidence._update_semantics(track, raw_label, source, confidence)

    @staticmethod
    def _update_mobility(
        track: PersistentObjectTrack,
        *,
        mobility_class: Any,
        confidence: float,
        source: str,
    ) -> None:
        return TrackerSemanticEvidence._update_mobility(track, mobility_class=mobility_class, confidence=confidence, source=source)

    def _commit_semantic_label(self, track: PersistentObjectTrack, timestamp_sec: float, reason: str) -> None:
        return self.semantic_evidence._commit_semantic_label(track, timestamp_sec, reason)

    def _choose_semantic_label(self, track: PersistentObjectTrack) -> Tuple[str, str, float, int]:
        return self.semantic_evidence._choose_semantic_label(track)

    @staticmethod
    def _annotate_metadata(
        metadata: Dict[str, Any],
        track: PersistentObjectTrack,
        track_event: str,
        match_reason: str,
        match_score: Optional[float],
    ) -> None:
        return tracker_serialization.annotate_metadata(metadata, track, track_event, match_reason, match_score)

    @staticmethod
    def _segment_record(segment: PersistentObjectSegment) -> Dict[str, Any]:
        return tracker_serialization.segment_record(segment)

    def _track_record(
        self,
        track: PersistentObjectTrack,
        event: str,
        reason: str,
        score: Optional[float],
    ) -> Dict[str, Any]:
        return tracker_serialization.track_record(track, event, reason, score)
