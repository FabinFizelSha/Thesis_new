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


def _as_xyz(value: Any) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        return None
    try:
        array = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return array if np.all(np.isfinite(array)) else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bbox_iou(a: Any, b: Any) -> float:
    """Return IoU for [x, y, width, height] boxes."""
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return 0.0
    if len(a) != 4 or len(b) != 4:
        return 0.0
    try:
        ax, ay, aw, ah = (float(value) for value in a)
        bx, by, bw, bh = (float(value) for value in b)
    except (TypeError, ValueError):
        return 0.0
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _volume_ratio(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return 1.0
    return max(float(a), float(b)) / max(min(float(a), float(b)), 1e-9)


def _aabb_gap_xy(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    """Return the shortest horizontal separation between two 3D boxes."""
    dx = max(float(a_min[0] - b_max[0]), float(b_min[0] - a_max[0]), 0.0)
    dy = max(float(a_min[1] - b_max[1]), float(b_min[1] - a_max[1]), 0.0)
    return float(math.hypot(dx, dy))


def _aabb_gap_z(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    """Return the shortest vertical separation between two 3D boxes."""
    return max(float(a_min[2] - b_max[2]), float(b_min[2] - a_max[2]), 0.0)


def _aabb_center_distance_xy(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    """Return horizontal distance between two 3D-box centres."""
    a_center = 0.5 * (a_min[:2] + a_max[:2])
    b_center = 0.5 * (b_min[:2] + b_max[:2])
    return float(np.linalg.norm(a_center - b_center))


def _aabb_center_delta_z(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    """Return the vertical distance between two 3D-box centres."""
    a_center_z = 0.5 * float(a_min[2] + a_max[2])
    b_center_z = 0.5 * float(b_min[2] + b_max[2])
    return abs(a_center_z - b_center_z)




def _aabb_overlap_fraction_xy(
    observation_min: np.ndarray,
    observation_max: np.ndarray,
    track_min: np.ndarray,
    track_max: np.ndarray,
) -> Tuple[float, float, float]:
    """Return observation-normalised XY overlap area and per-axis fractions."""
    obs_dx = max(0.0, float(observation_max[0] - observation_min[0]))
    obs_dy = max(0.0, float(observation_max[1] - observation_min[1]))
    overlap_x = max(
        0.0,
        min(float(observation_max[0]), float(track_max[0]))
        - max(float(observation_min[0]), float(track_min[0])),
    )
    overlap_y = max(
        0.0,
        min(float(observation_max[1]), float(track_max[1]))
        - max(float(observation_min[1]), float(track_min[1])),
    )
    fraction_x = overlap_x / max(obs_dx, 1e-9)
    fraction_y = overlap_y / max(obs_dy, 1e-9)
    area_fraction = (overlap_x * overlap_y) / max(obs_dx * obs_dy, 1e-9)
    return (
        max(0.0, min(1.0, float(area_fraction))),
        max(0.0, min(1.0, float(fraction_x))),
        max(0.0, min(1.0, float(fraction_y))),
    )


def _aabb_overlap_fraction_3d(
    observation_min: np.ndarray,
    observation_max: np.ndarray,
    track_min: np.ndarray,
    track_max: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return observation-normalised 3D volume overlap (XYZ) with per-axis fractions.

    With 30cm depth padding, Z-ranges are normalized and 3D overlap is stable.
    Returns: (volume_fraction, x_fraction, y_fraction, z_fraction)
    """
    obs_dx = max(0.0, float(observation_max[0] - observation_min[0]))
    obs_dy = max(0.0, float(observation_max[1] - observation_min[1]))
    obs_dz = max(0.0, float(observation_max[2] - observation_min[2]))
    obs_volume = max(obs_dx * obs_dy * obs_dz, 1e-9)

    overlap_x = max(
        0.0,
        min(float(observation_max[0]), float(track_max[0]))
        - max(float(observation_min[0]), float(track_min[0])),
    )
    overlap_y = max(
        0.0,
        min(float(observation_max[1]), float(track_max[1]))
        - max(float(observation_min[1]), float(track_min[1])),
    )
    overlap_z = max(
        0.0,
        min(float(observation_max[2]), float(track_max[2]))
        - max(float(observation_min[2]), float(track_min[2])),
    )
    overlap_volume = overlap_x * overlap_y * overlap_z

    fraction_x = overlap_x / max(obs_dx, 1e-9)
    fraction_y = overlap_y / max(obs_dy, 1e-9)
    fraction_z = overlap_z / max(obs_dz, 1e-9)
    volume_fraction = overlap_volume / obs_volume

    return (
        max(0.0, min(1.0, float(volume_fraction))),
        max(0.0, min(1.0, float(fraction_x))),
        max(0.0, min(1.0, float(fraction_y))),
        max(0.0, min(1.0, float(fraction_z))),
    )


def _aabb_3d_containment(
    observation_min: np.ndarray,
    observation_max: np.ndarray,
    track_min: np.ndarray,
    track_max: np.ndarray,
) -> float:
    """Return fraction of observation bbox contained within track bbox (0.0 to 1.0)."""
    obs_dx = max(0.0, float(observation_max[0] - observation_min[0]))
    obs_dy = max(0.0, float(observation_max[1] - observation_min[1]))
    obs_dz = max(0.0, float(observation_max[2] - observation_min[2]))
    obs_volume = max(1e-9, obs_dx * obs_dy * obs_dz)

    contained_x = max(
        0.0,
        min(float(observation_max[0]), float(track_max[0]))
        - max(float(observation_min[0]), float(track_min[0])),
    )
    contained_y = max(
        0.0,
        min(float(observation_max[1]), float(track_max[1]))
        - max(float(observation_min[1]), float(track_min[1])),
    )
    contained_z = max(
        0.0,
        min(float(observation_max[2]), float(track_max[2]))
        - max(float(observation_min[2]), float(track_min[2])),
    )
    contained_volume = contained_x * contained_y * contained_z
    return max(0.0, min(1.0, float(contained_volume / obs_volume)))


def _gaussian_compatibility(value: float, sigma: float) -> float:
    """Map a non-negative residual to [0, 1], where one is ideal."""
    sigma = max(float(sigma), 1e-9)
    value = max(0.0, float(value))
    return float(math.exp(-0.5 * (value / sigma) ** 2))

def _aabb_xy_diagonal(a_min: np.ndarray, a_max: np.ndarray) -> float:
    """Return horizontal XY diagonal of a 3D axis-aligned bounding box."""
    return float(math.hypot(float(a_max[0] - a_min[0]), float(a_max[1] - a_min[1])))


def _aabb_union_xy_diagonal(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    """Return horizontal XY diagonal after merging two 3D boxes."""
    union_min = np.minimum(a_min, b_min)
    union_max = np.maximum(a_max, b_max)
    return _aabb_xy_diagonal(union_min, union_max)


def _aabb_iou_3d(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
) -> float:
    """Symmetric 3D intersection-over-union of two axis-aligned boxes."""
    inter = np.maximum(
        0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
    )
    inter_vol = float(inter[0] * inter[1] * inter[2])
    if inter_vol <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.maximum(0.0, a_max - a_min)))
    vol_b = float(np.prod(np.maximum(0.0, b_max - b_min)))
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 1e-9 else 0.0


def _rigid_point(point: Optional[np.ndarray], rot: np.ndarray, trans: np.ndarray) -> Optional[np.ndarray]:
    """Apply ``p -> R @ p + t`` to a single 3D point (``None`` passes through)."""
    if point is None:
        return None
    return (rot @ np.asarray(point, dtype=np.float64)) + trans


def _rigid_aabb(
    bbox_min: Optional[np.ndarray],
    bbox_max: Optional[np.ndarray],
    rot: np.ndarray,
    trans: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Rigid-transform an axis-aligned box.

    A non-zero rotation tilts the box, so all eight corners are transformed and
    a fresh axis-aligned min/max is taken.  Exact for a pure translation, and
    the tightest axis-aligned envelope otherwise.
    """
    if bbox_min is None or bbox_max is None:
        return bbox_min, bbox_max
    lo = np.asarray(bbox_min, dtype=np.float64)
    hi = np.asarray(bbox_max, dtype=np.float64)
    corners = np.array(
        [[lo[0], lo[1], lo[2]], [lo[0], lo[1], hi[2]],
         [lo[0], hi[1], lo[2]], [lo[0], hi[1], hi[2]],
         [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]],
         [hi[0], hi[1], lo[2]], [hi[0], hi[1], hi[2]]],
        dtype=np.float64,
    )
    moved = (corners @ rot.T) + trans
    return moved.min(axis=0), moved.max(axis=0)


def _normalise_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _track_sort_key(track_id: Any) -> Tuple[int, Any]:
    """Deterministic ordering for track ids that are usually plain integers."""
    text = str(track_id)
    return (0, int(text)) if text.isdigit() else (1, text)


def _as_list(value: Optional[np.ndarray]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(v) for v in value.tolist()]


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
    labeling_status: str = "collecting"  # collecting | rap_queued | rap_dequeued | vlm_queued | vlm_dequeued | completed

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
    def _hungarian_maximize(weights: List[List[float]]) -> List[int]:
        """Return one selected column per row using a rectangular Hungarian solver."""
        if not weights:
            return []
        n = len(weights)
        m = len(weights[0])
        if n > m:
            raise ValueError("Hungarian solver requires columns >= rows")
        maximum = max(max(row) for row in weights)
        cost = [[maximum - value for value in row] for row in weights]
        u = [0.0] * (n + 1)
        v = [0.0] * (m + 1)
        p = [0] * (m + 1)
        way = [0] * (m + 1)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [float("inf")] * (m + 1)
            used = [False] * (m + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        assignment = [-1] * n
        for j in range(1, m + 1):
            if p[j] > 0:
                assignment[p[j] - 1] = j - 1
        return assignment

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
        """Return RAP task records after the fixed crop-settling window.

        A semantic job is intentionally not scheduled from the first valid crop.
        The persistent track remains live while later observations can replace
        that crop in Phase 1's shared best-crop registry. ``current_timestamp_sec``
        uses recorded message time, so the delay is deterministic during bag
        replay and independent of worker latency.
        """
        ready: List[Dict[str, Any]] = []
        min_observations = int(
            getattr(self.config, "semantic_labeling_min_observations", 1)
        )
        settle_time_sec = max(
            0.0,
            float(getattr(self.config, "semantic_labeling_settle_time_sec", 0.0)),
        )
        now_sec = float(current_timestamp_sec)

        with self._lock:
            # Iterate over every live track rather than only the current-frame
            # detections. This lets an object that has left the camera view be
            # released once its fixed collection interval has elapsed.
            for track in self._tracks.values():
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
                record = self._track_record(track, "labeling_ready", reason, None)
                record["settling_age_sec"] = float(settling_age_sec)
                record["settle_time_sec"] = float(settle_time_sec)
                record["forced_dispatch"] = bool(force)
                ready.append(record)
        return ready

    def release_labeling_request(self, track_id: str, reason: str) -> None:
        """Allow a later observation to retry RAP after enqueue/worker failure."""
        with self._lock:
            track = self._tracks.get(str(track_id))
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
        with self._lock:
            track = self._tracks.get(str(track_id))
            if track is None or track.labeling_completed:
                return
            track.labeling_status = str(status)

    def is_semantic_labeling_open(self, track_id: str) -> bool:
        """Return whether a track may still accept crop updates."""
        with self._lock:
            track = self._tracks.get(str(track_id))
            return bool(track is not None and not track.labeling_completed)

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
        with self._lock:
            track = self._tracks.get(str(track_id))
            if track is None:
                return None
            self._commit_semantic_label(track, float(timestamp_sec), str(reason))
            track.labeling_completed = True
            track.labeling_status = "completed"
            return self._track_record(track, "semantic_label_completed", str(reason), None)

    def apply_vlm_result(
        self,
        track_id: str,
        label: str,
        confidence: float,
        mobility_class: str = "unknown",
        mobility_confidence: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Attach one validated VLM label and mobility decision to a slot."""
        normalised = self._canonicalise_label(label)
        if not normalised:
            return None
        with self._lock:
            track = self._tracks.get(str(track_id))
            if track is None:
                return None
            track.raw_vlm_label = normalised
            self._update_semantics(track, normalised, "vlm", float(confidence))
            self._update_mobility(
                track,
                mobility_class=mobility_class,
                confidence=mobility_confidence,
                source="vlm",
            )
            return self._track_record(track, "vlm_semantic_update", "vlm_result", None)

    def apply_rap_result(
        self,
        track_id: str,
        label: str,
        confidence: float,
        is_known: bool,
        mobility_class: str = "unknown",
        mobility_confidence: float = 0.0,
        mobility_source: str = "rap",
    ) -> Optional[Dict[str, Any]]:
        """Attach one RAP label and stored mobility metadata to a slot."""
        with self._lock:
            track = self._tracks.get(str(track_id))
            if track is None:
                return None
            resolved = self._canonicalise_label(label) if bool(is_known) else ""
            if resolved:
                self._update_semantics(track, resolved, "rap", float(confidence))
                self._update_mobility(
                    track,
                    mobility_class=mobility_class,
                    confidence=mobility_confidence,
                    source=mobility_source,
                )
            return self._track_record(
                track,
                "rap_semantic_update" if resolved else "rap_unknown",
                "rap_result",
                None,
            )

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
        if not bool(getattr(self.config, "persistent_use_hydra_slots", False)):
            instance_id = self._next_instance_id
            if instance_id > 65535:
                raise RuntimeError("Persistent instance ID space exhausted for 16UC1 output")
            self._next_instance_id += 1
            return int(instance_id), str(instance_id), int(instance_id)

        allocated = self._next_available_slot_id()
        if allocated is None:
            raise RuntimeError("No unreserved Hydra local segment slots remain")
        self._allocated_slot_ids.add(int(allocated))
        first = int(self.config.persistent_slot_first_label_id)
        width = max(1, int(self.config.persistent_slot_label_width))
        slot_index = int(allocated) - first + 1
        name = f"{self.config.persistent_slot_label_prefix}{slot_index:0{width}d}"
        return int(allocated), name, int(allocated)

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
    def _activate_segment(track: PersistentObjectTrack, segment: PersistentObjectSegment) -> None:
        track.active_segment_slot_id = int(segment.hydra_label_id)
        track.hydra_label_id = int(segment.hydra_label_id)
        track.hydra_label_name = str(segment.hydra_label_name)
        track.instance_id = int(segment.instance_id)

    def _segment_xy_span_after_update(
        self,
        segment: PersistentObjectSegment,
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
        alpha = float(self.config.persistent_centroid_update_alpha)
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
        track: PersistentObjectTrack,
        frame_id: str,
        sequence: int,
        timestamp_sec: float,
        centroid: Optional[np.ndarray],
        bbox_2d: Any,
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> Tuple[str, str, Optional[float]]:
        if not bool(getattr(self.config, "persistent_local_segments_enabled", True)):
            segment = track.segments.get(int(track.active_segment_slot_id)) or next(iter(track.segments.values()))
            self._activate_segment(track, segment)
            self._update_segment_geometry(
                segment, frame_id=frame_id, sequence=sequence, timestamp_sec=timestamp_sec,
                centroid=centroid, bbox_2d=bbox_2d, bbox_3d_min=bbox_3d_min, bbox_3d_max=bbox_3d_max,
            )
            return "matched_segment", "local_segments_disabled", None

        max_span = float(getattr(self.config, "persistent_local_segment_max_xy_span_m", 4.0))
        revisit_distance = float(getattr(self.config, "persistent_local_segment_revisit_distance_m", 1.5))
        gap_limit = float(getattr(self.config, "persistent_local_segment_gap_m", 0.20))
        use_2d_fallback = bool(
            getattr(self.config, "persistent_local_segment_2d_fallback_enabled", True)
        )
        max_2d_age_sec = max(
            0.0,
            float(
                getattr(
                    self.config,
                    "persistent_local_segment_max_2d_iou_age_sec",
                    getattr(self.config, "persistent_max_2d_iou_age_sec", 2.0),
                )
            ),
        )
        min_2d_iou = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        self.config,
                        "persistent_local_segment_min_2d_iou",
                        getattr(self.config, "persistent_min_2d_iou", 0.30),
                    )
                ),
            ),
        )
        best_segment: Optional[PersistentObjectSegment] = None
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
            best_2d_segment: Optional[PersistentObjectSegment] = None
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
        """Rigid-transform every cached track and segment by ``p -> R @ p + t``.

        Only geometry moves.  EMA state, observation counts, labels, Hydra
        slots and the shared crop registries are untouched.  ``bbox_volume_m3``
        is invariant under a rigid motion and is left as-is.  Returns the number
        of tracks re-anchored.
        """
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        trans = np.asarray(translation, dtype=np.float64).reshape(3)
        with self._lock:
            for track in self._tracks.values():
                track.centroid_3d = _rigid_point(track.centroid_3d, rot, trans)
                track.bbox_3d_min, track.bbox_3d_max = _rigid_aabb(
                    track.bbox_3d_min, track.bbox_3d_max, rot, trans
                )
                track.last_bbox_3d_min, track.last_bbox_3d_max = _rigid_aabb(
                    track.last_bbox_3d_min, track.last_bbox_3d_max, rot, trans
                )
                for segment in track.segments.values():
                    segment.centroid_3d = _rigid_point(segment.centroid_3d, rot, trans)
                    segment.bbox_3d_min, segment.bbox_3d_max = _rigid_aabb(
                        segment.bbox_3d_min, segment.bbox_3d_max, rot, trans
                    )
                    segment.last_bbox_3d_min, segment.last_bbox_3d_max = _rigid_aabb(
                        segment.last_bbox_3d_min, segment.last_bbox_3d_max, rot, trans
                    )
                self._refresh_spatial_index(track)
            count = len(self._tracks)
            self._last_reanchor = {
                "stamp_sec": float(stamp) if stamp is not None else None,
                "translation": [float(v) for v in trans],
                "rotation": [float(v) for v in rot.reshape(9)],
                "track_count": int(count),
            }
        return count

    def last_reanchor(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the most recent :meth:`reanchor_all` summary."""
        with self._lock:
            return dict(self._last_reanchor) if self._last_reanchor else None

    def _reanchor_labels_compatible(
        self, a: PersistentObjectTrack, b: PersistentObjectTrack
    ) -> bool:
        if a.semantic_kind != b.semantic_kind:
            return False
        generic = {"", "unknown", "unknown object", "object", "thing", "stuff", "background"}

        def strong_label(track: PersistentObjectTrack) -> str:
            for candidate in (
                track.semantic_label,
                track.canonical_label,
                track.raw_vlm_label,
                track.raw_rap_label,
            ):
                name = _normalise_label(candidate)
                if name and name not in generic:
                    return name
            return ""

        label_a = strong_label(a)
        label_b = strong_label(b)
        if label_a and label_b:
            return label_a == label_b
        return True

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
        """Fold ``drop`` into ``keep``.

        ``keep`` always retains the surviving ``track_id``, label evidence and
        (earlier) first-seen.  With ``adopt_drop_geometry`` the survivor takes
        ``drop``'s centroid and last box verbatim -- used for the loop-closure
        case, where ``drop`` is the freshly re-observed, drift-corrected copy
        and ``keep`` is the stale original.  Otherwise the centroid is an
        observation-weighted blend and the global box is the union.
        """
        kc = max(1, int(keep.seen_count))
        dc = max(1, int(drop.seen_count))

        union_min = None
        union_max = None
        if keep.bbox_3d_min is not None and drop.bbox_3d_min is not None:
            union_min = np.minimum(keep.bbox_3d_min, drop.bbox_3d_min)
            union_max = np.maximum(keep.bbox_3d_max, drop.bbox_3d_max)

        if adopt_drop_geometry:
            if drop.centroid_3d is not None:
                keep.centroid_3d = np.asarray(drop.centroid_3d, dtype=np.float64).copy()
            if drop.bbox_3d_min is not None:
                keep.bbox_3d_min = np.asarray(drop.bbox_3d_min, dtype=np.float64).copy()
                keep.bbox_3d_max = np.asarray(drop.bbox_3d_max, dtype=np.float64).copy()
            if drop.last_bbox_3d_min is not None:
                keep.last_bbox_3d_min = np.asarray(drop.last_bbox_3d_min, dtype=np.float64).copy()
                keep.last_bbox_3d_max = np.asarray(drop.last_bbox_3d_max, dtype=np.float64).copy()
        else:
            if keep.centroid_3d is not None and drop.centroid_3d is not None:
                keep.centroid_3d = (kc * keep.centroid_3d + dc * drop.centroid_3d) / float(kc + dc)
            elif keep.centroid_3d is None:
                keep.centroid_3d = drop.centroid_3d
            if union_min is not None:
                keep.bbox_3d_min, keep.bbox_3d_max = union_min, union_max
            if drop.last_bbox_3d_min is not None and (
                keep.last_bbox_3d_min is None
                or float(drop.last_seen_timestamp_sec) >= float(keep.last_seen_timestamp_sec)
            ):
                keep.last_bbox_3d_min = drop.last_bbox_3d_min
                keep.last_bbox_3d_max = drop.last_bbox_3d_max

        if keep.bbox_3d_min is not None and keep.bbox_3d_max is not None:
            keep.bbox_volume_m3 = float(
                np.prod(np.maximum(0.0, keep.bbox_3d_max - keep.bbox_3d_min))
            )
        keep.seen_count = kc + dc

        if float(drop.first_seen_timestamp_sec) < float(keep.first_seen_timestamp_sec):
            keep.first_seen_timestamp_sec = drop.first_seen_timestamp_sec
            keep.first_seen_frame_id = drop.first_seen_frame_id
            keep.first_seen_sequence = drop.first_seen_sequence
        if float(drop.last_seen_timestamp_sec) > float(keep.last_seen_timestamp_sec):
            keep.last_seen_timestamp_sec = drop.last_seen_timestamp_sec
            keep.last_seen_frame_id = drop.last_seen_frame_id
            keep.last_seen_sequence = drop.last_seen_sequence

        for key, value in (drop.label_evidence or {}).items():
            keep.label_evidence[key] = keep.label_evidence.get(key, 0.0) + float(value)
        for key, value in (drop.label_observations or {}).items():
            keep.label_observations[key] = keep.label_observations.get(key, 0) + int(value)

        # Hydra slot ids are globally unique, so a non-overlapping ``drop``
        # segment can be re-keyed straight onto the survivor without
        # collision. But ``keep`` and ``drop`` were tracked independently
        # until now, so each may already have built its own local segment
        # covering the same physical patch (e.g. the same ceiling tile seen
        # moments apart before the duplicate was recognised). Re-keying both
        # verbatim would leave two permanently-static, heavily-overlapping
        # segments sitting side by side forever -- geometry is reconciled
        # into the closest touching existing segment instead of adding a
        # new, redundant one. Only genuinely disjoint drop segments keep
        # their own slot; presence continuity matters more than slot economy
        # for those.
        gap_limit = float(getattr(self.config, "persistent_local_segment_gap_m", 0.20))
        max_span = float(getattr(self.config, "persistent_local_segment_max_xy_span_m", 4.0))
        for slot_id, segment in drop.segments.items():
            absorbing: Optional[PersistentObjectSegment] = None
            best_gap: Optional[float] = None
            if segment.bbox_3d_min is not None and segment.bbox_3d_max is not None:
                for existing in keep.segments.values():
                    if existing.bbox_3d_min is None or existing.bbox_3d_max is None:
                        continue
                    gap = _aabb_gap_xy(
                        segment.bbox_3d_min, segment.bbox_3d_max,
                        existing.bbox_3d_min, existing.bbox_3d_max,
                    )
                    if gap <= gap_limit and (best_gap is None or gap < best_gap):
                        absorbing, best_gap = existing, gap

            if absorbing is None:
                segment.segment_id = f"{keep.track_id}:slot_{int(segment.hydra_label_id)}"
                keep.segments.setdefault(int(slot_id), segment)
                continue

            absorbing.bbox_3d_min = np.minimum(absorbing.bbox_3d_min, segment.bbox_3d_min)
            absorbing.bbox_3d_max = np.maximum(absorbing.bbox_3d_max, segment.bbox_3d_max)
            if _aabb_xy_diagonal(absorbing.bbox_3d_min, absorbing.bbox_3d_max) > max_span:
                # Same freeze-on-cap rule as live association: a merge can
                # bridge the cap once, but the result must not keep growing.
                absorbing.closed = True
            if float(segment.last_seen_timestamp_sec) >= float(absorbing.last_seen_timestamp_sec):
                absorbing.last_bbox_3d_min = segment.last_bbox_3d_min
                absorbing.last_bbox_3d_max = segment.last_bbox_3d_max
                absorbing.last_seen_frame_id = segment.last_seen_frame_id
                absorbing.last_seen_sequence = segment.last_seen_sequence
                absorbing.last_seen_timestamp_sec = segment.last_seen_timestamp_sec
                if segment.bbox_2d:
                    absorbing.bbox_2d = segment.bbox_2d
            if float(segment.first_seen_timestamp_sec) < float(absorbing.first_seen_timestamp_sec):
                absorbing.first_seen_frame_id = segment.first_seen_frame_id
                absorbing.first_seen_sequence = segment.first_seen_sequence
                absorbing.first_seen_timestamp_sec = segment.first_seen_timestamp_sec
            absorbing.seen_count += int(segment.seen_count)
            # `segment`'s own Hydra slot is retired here: it is dropped from
            # the merged track's bookkeeping so no future observation is
            # ever routed to it again, and its existing DSG node stops
            # growing -- but note this cannot retroactively erase mesh
            # geometry Hydra already committed under that slot before the
            # merge; only the *tracker's* forward association state is
            # reconciled.

        keep.metadata.setdefault("reanchor_merged_from", []).append(
            {
                "track_id": drop.track_id,
                "internal_object_id": drop.track_id,
                "seen_count": int(dc),
                "iou_3d": round(float(iou_3d), 4),
                "distance_m": round(float(distance_m), 4),
                "reason": str(reason),
                "adopted_geometry": bool(adopt_drop_geometry),
                "slot_ids": sorted(int(s) for s in drop.segments),
            }
        )
        self._forget_spatial_index(drop.track_id)

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
        """Fold together track identities that describe one physical object but
        were split before a loop closure.

        Two complementary passes, both conservative and both label-gated:

        * **drift pass** (needs ``now_sec``): a track re-observed within
          ``recent_window_sec`` is folded into an older compatible track whose
          centroid lies within ``|correction| * 1.25 + distance_slack_m``.  A
          rigid re-anchor cannot close this gap because the pair straddles the
          drift; the survivor is the older identity but it adopts the fresh
          (drift-corrected) geometry.
        * **overlap pass**: any remaining track pair that now genuinely overlaps
          (``min_iou_3d``) and sits within ``max_centroid_distance_m`` is folded
          with an observation-weighted blend -- ordinary fragmentation cleanup.

        Runs once per correction, never in the steady-state association path.
        Returns the number of tracks removed.
        """
        if max_centroid_distance_m is None:
            max_centroid_distance_m = float(
                getattr(
                    self.config,
                    "persistent_global_centroid_pass_m",
                    getattr(self.config, "persistent_max_match_distance_m", 1.0),
                )
            )
        drift_radius = max(
            float(distance_slack_m),
            abs(float(correction_translation_m)) * 1.25 + float(distance_slack_m),
        )
        removed = 0
        with self._lock:
            def _xy_gap(a: PersistentObjectTrack, b: PersistentObjectTrack) -> Optional[float]:
                if a.centroid_3d is None or b.centroid_3d is None:
                    return None
                return float(np.linalg.norm(a.centroid_3d[:2] - b.centroid_3d[:2]))

            gone: Set[str] = set()
            absorbed: Set[str] = set()

            # --- drift pass -------------------------------------------------
            if now_sec is not None:
                recent = sorted(
                    (
                        t
                        for t in self._tracks.values()
                        if t.bbox_3d_min is not None
                        and t.centroid_3d is not None
                        and float(now_sec) - float(t.last_seen_timestamp_sec) <= float(recent_window_sec)
                    ),
                    key=lambda t: _track_sort_key(t.track_id),
                )
                for fresh in recent:
                    if fresh.track_id in gone:
                        continue
                    best: Optional[PersistentObjectTrack] = None
                    best_gap = drift_radius
                    for cand in self._tracks.values():
                        if cand.track_id == fresh.track_id or cand.track_id in gone or cand.track_id in absorbed:
                            continue
                        if cand.bbox_3d_min is None or cand.centroid_3d is None:
                            continue
                        if float(now_sec) - float(cand.last_seen_timestamp_sec) <= float(recent_window_sec):
                            continue  # both fresh -> not a stale/fresh pair
                        if float(cand.first_seen_timestamp_sec) > float(fresh.first_seen_timestamp_sec):
                            continue  # survivor must be the older identity
                        if not self._reanchor_labels_compatible(cand, fresh):
                            continue
                        gap = _xy_gap(cand, fresh)
                        if gap is None or gap > best_gap:
                            continue
                        best, best_gap = cand, gap
                    if best is None:
                        continue
                    iou = _aabb_iou_3d(
                        best.bbox_3d_min, best.bbox_3d_max,
                        fresh.bbox_3d_min, fresh.bbox_3d_max,
                    )
                    self._merge_track_pair(
                        best, fresh,
                        iou_3d=iou, distance_m=best_gap,
                        reason="loop_closure_drift", adopt_drop_geometry=True,
                    )
                    gone.add(fresh.track_id)
                    absorbed.add(best.track_id)
                    self._refresh_spatial_index(best)
                    removed += 1

            # --- overlap pass --------------------------------------------------
            ordered = sorted(
                (
                    t
                    for t in self._tracks.values()
                    if t.track_id not in gone
                    and t.bbox_3d_min is not None
                    and t.bbox_3d_max is not None
                ),
                key=lambda t: (-int(t.seen_count), _track_sort_key(t.track_id)),
            )
            for keep in ordered:
                if keep.track_id in gone:
                    continue
                for cid in self._candidate_track_ids(
                    keep.centroid_3d, keep.bbox_3d_min, keep.bbox_3d_max
                ):
                    if cid == keep.track_id or cid in gone or cid in absorbed:
                        continue
                    drop = self._tracks.get(cid)
                    if (
                        drop is None
                        or drop.bbox_3d_min is None
                        or drop.bbox_3d_max is None
                        or int(drop.seen_count) > int(keep.seen_count)
                        or not self._reanchor_labels_compatible(keep, drop)
                    ):
                        continue
                    iou = _aabb_iou_3d(
                        keep.bbox_3d_min, keep.bbox_3d_max,
                        drop.bbox_3d_min, drop.bbox_3d_max,
                    )
                    if iou < float(min_iou_3d):
                        continue
                    gap = _xy_gap(keep, drop)
                    if gap is not None and gap > float(max_centroid_distance_m):
                        continue
                    self._merge_track_pair(
                        keep, drop,
                        iou_3d=iou, distance_m=(gap if gap is not None else 0.0),
                        reason="post_reanchor_overlap", adopt_drop_geometry=False,
                    )
                    gone.add(cid)
                    absorbed.add(keep.track_id)
                    removed += 1
                if keep.track_id not in gone:
                    self._refresh_spatial_index(keep)

            for cid in gone:
                self._tracks.pop(cid, None)
        return removed

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
        return max(0.25, min(2.0, max(
            float(getattr(self.config, "persistent_global_centroid_pass_m", self.config.persistent_max_match_distance_m)),
            float(self.config.persistent_continuation_gap_m),
            float(self.config.persistent_revisit_overlap_gap_m),
        )))

    def _spatial_cells(
        self, bbox_min: np.ndarray, bbox_max: np.ndarray, padding: float = 0.0
    ) -> Optional[Set[Tuple[int, int]]]:
        size = self._spatial_cell_size()
        x0, y0 = np.floor((bbox_min[:2] - padding) / size).astype(int)
        x1, y1 = np.floor((bbox_max[:2] + padding) / size).astype(int)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 256:
            return None
        return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}

    def _forget_spatial_index(self, track_id: str) -> None:
        """Drop every spatial-index entry for one track id."""
        for cell in self._spatial_bbox_cells_by_track.pop(track_id, set()):
            ids = self._spatial_bbox_cells.get(cell)
            if ids is None:
                continue
            ids.discard(track_id)
            if not ids:
                del self._spatial_bbox_cells[cell]
        centroid_cell = self._spatial_centroid_cell_by_track.pop(track_id, None)
        if centroid_cell is not None:
            ids = self._spatial_centroid_cells.get(centroid_cell)
            if ids is not None:
                ids.discard(track_id)
                if not ids:
                    del self._spatial_centroid_cells[centroid_cell]
        self._spatial_fallback_track_ids.discard(track_id)

    def _refresh_spatial_index(self, track: PersistentObjectTrack) -> None:
        track_id = track.track_id
        self._forget_spatial_index(track_id)

        if track.bbox_3d_min is None or track.bbox_3d_max is None:
            self._spatial_fallback_track_ids.add(track_id)
        else:
            cells = self._spatial_cells(track.bbox_3d_min, track.bbox_3d_max)
            if cells is None:
                self._spatial_fallback_track_ids.add(track_id)
            else:
                self._spatial_bbox_cells_by_track[track_id] = cells
                for cell in cells:
                    self._spatial_bbox_cells.setdefault(cell, set()).add(track_id)
        if track.centroid_3d is not None:
            cell = tuple(np.floor(track.centroid_3d[:2] / self._spatial_cell_size()).astype(int))
            self._spatial_centroid_cell_by_track[track_id] = cell
            self._spatial_centroid_cells.setdefault(cell, set()).add(track_id)

    def _candidate_track_ids(
        self,
        centroid: Optional[np.ndarray],
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> List[str]:
        """Return every track that can still pass the exact association gates."""
        global_enabled = bool(getattr(self.config, "persistent_global_association_enabled", True))
        block_2d = bool(getattr(self.config, "persistent_global_block_2d_on_3d_contradiction", True))
        if not global_enabled or not block_2d or bbox_3d_min is None or bbox_3d_max is None:
            return list(self._tracks)

        candidate_ids = set(self._spatial_fallback_track_ids)
        footprint_cells = self._spatial_cells(
            bbox_3d_min,
            bbox_3d_max,
            max(float(self.config.persistent_continuation_gap_m), float(self.config.persistent_revisit_overlap_gap_m)),
        )
        if footprint_cells is None:
            return list(self._tracks)
        for cell in footprint_cells:
            candidate_ids.update(self._spatial_bbox_cells.get(cell, ()))
        if centroid is not None:
            radius = float(getattr(
                self.config, "persistent_global_centroid_pass_m", self.config.persistent_max_match_distance_m
            ))
            centroid_cells = self._spatial_cells(centroid, centroid, radius)
            if centroid_cells is None:
                return list(self._tracks)
            for cell in centroid_cells:
                candidate_ids.update(self._spatial_centroid_cells.get(cell, ()))
        return [track_id for track_id in self._tracks if track_id in candidate_ids]

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
                # Removed vertical checks from historical_pass; 3D overlap handles height separation
                historical_pass = bool(
                    (overlap_volume >= min_hist and overlap_x >= min_axis and overlap_y >= min_axis)
                    or accumulated_gap_xy <= float(getattr(self.config, "persistent_global_touch_gap_pass_m", 0.02))
                )
                # Vertical score removed (now in 3D overlap); keep vertical_pass=False for quorum
                vertical_pass = False

            # Recent Continuation removed as redundant (3D Historical now handles continuity)
            recent_score = 0.0
            recent_pass = False

            if stage_ms is not None:
                geometry_3d_ms += (time.perf_counter() - _t0) * 1000.0
                _t0 = time.perf_counter()
            ratio = _volume_ratio(volume, track.bbox_volume_m3)
            row["volume_ratio"] = float(ratio)

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

            if not hard_2d_contradiction and pass_count >= min_groups and score >= min_score:
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

    def analyze_bbox_overlaps(self) -> Dict[str, Any]:
        """Analyze overlapping bboxes in current tracks for fragmentation diagnosis.

        Returns dict with overlap statistics for threshold optimization.
        """
        overlaps = []
        track_list = list(self._tracks.items())

        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                tid_i, track_i = track_list[i]
                tid_j, track_j = track_list[j]

                # Skip if either track has no valid bbox
                if (track_i.bbox_3d_min is None or track_i.bbox_3d_max is None or
                    track_j.bbox_3d_min is None or track_j.bbox_3d_max is None):
                    continue

                # Calculate 3D bbox overlap fraction
                min_i = track_i.bbox_3d_min
                max_i = track_i.bbox_3d_max
                min_j = track_j.bbox_3d_min
                max_j = track_j.bbox_3d_max

                # Overlap bounds
                overlap_min = np.maximum(min_i, min_j)
                overlap_max = np.minimum(max_i, max_j)

                # Check if there's overlap in all 3 dimensions
                overlap = np.all(overlap_min < overlap_max)
                if not overlap:
                    continue

                # Calculate overlap volume
                overlap_dims = overlap_max - overlap_min
                overlap_volume = float(np.prod(overlap_dims))

                # Calculate original volumes
                vol_i = float(np.prod(max_i - min_i))
                vol_j = float(np.prod(max_j - min_j))

                # Overlap as percentage of smaller track
                overlap_pct = 100.0 * overlap_volume / min(vol_i, vol_j)

                # XY distance between centroids
                cent_i = track_i.centroid_3d or ((min_i + max_i) / 2.0)
                cent_j = track_j.centroid_3d or ((min_j + max_j) / 2.0)
                xy_dist = float(np.linalg.norm(cent_i[:2] - cent_j[:2]))

                if overlap_pct > 0:
                    overlaps.append({
                        'track_i': tid_i,
                        'track_j': tid_j,
                        'overlap_pct': overlap_pct,
                        'xy_distance_m': xy_dist,
                        'vol_i': vol_i,
                        'vol_j': vol_j,
                        'overlap_volume': overlap_volume,
                        'age_i': track_i.seen_count,
                        'age_j': track_j.seen_count,
                    })

        # Summary statistics
        stats = {
            'total_track_pairs': len(track_list) * (len(track_list) - 1) // 2,
            'overlapping_pairs': len(overlaps),
            'overlap_pct_min': min([o['overlap_pct'] for o in overlaps], default=0),
            'overlap_pct_max': max([o['overlap_pct'] for o in overlaps], default=0),
            'overlap_pct_mean': np.mean([o['overlap_pct'] for o in overlaps]) if overlaps else 0,
            'xy_distance_min': min([o['xy_distance_m'] for o in overlaps], default=0),
            'detailed_overlaps': overlaps,
        }

        self.logger.info(
            f"BBox overlap analysis: {stats['overlapping_pairs']} overlapping pairs "
            f"(max overlap: {stats['overlap_pct_max']:.1f}%, mean: {stats['overlap_pct_mean']:.1f}%)"
        )

        return stats

    @staticmethod
    def _source_rank(source: str) -> int:
        return {"pending": 0, "rap": 1, "vlm": 2}.get(str(source), 0)

    def _canonicalise_label(self, raw_label: Any) -> str:
        label = _normalise_label(raw_label)
        if not label or label in {"unknown", "unknown object", "unknown_object"}:
            return ""
        aliases = getattr(self.config, "persistent_label_aliases", {}) or {}
        return _normalise_label(aliases.get(label, label))

    def _source_weight(self, source: str) -> float:
        if source == "vlm":
            return float(getattr(self.config, "persistent_vlm_evidence_weight", 1.0))
        if source == "rap":
            return float(getattr(self.config, "persistent_rap_evidence_weight", 0.70))
        return 0.0

    def _add_evidence(self, track: PersistentObjectTrack, raw_label: str, source: str, confidence: float) -> None:
        label = self._canonicalise_label(raw_label)
        if not label:
            return
        score = max(0.0, min(1.0, float(confidence))) * max(0.0, self._source_weight(source))
        if score <= 0.0:
            return
        track.label_evidence[label] = float(track.label_evidence.get(label, 0.0) + score)
        track.label_observations[label] = int(track.label_observations.get(label, 0) + 1)

    def _update_semantics(self, track: PersistentObjectTrack, raw_label: str, source: str, confidence: float) -> None:
        label = self._canonicalise_label(raw_label)
        if not label:
            return
        if source == "rap":
            track.raw_rap_label = label
        elif source == "vlm":
            track.raw_vlm_label = label
        self._add_evidence(track, label, source, confidence)
        candidate_rank = self._source_rank(source)
        current_rank = self._source_rank(track.label_source)
        if candidate_rank > current_rank or (candidate_rank == current_rank and confidence >= track.label_confidence):
            track.canonical_label = label
            track.label_source = source
            track.label_confidence = float(confidence)

    @staticmethod
    def _update_mobility(
        track: PersistentObjectTrack,
        *,
        mobility_class: Any,
        confidence: float,
        source: str,
    ) -> None:
        """Keep the strongest valid static/dynamic/unknown mobility decision."""
        mobility = str(mobility_class or "unknown").strip().lower()
        if mobility not in {"static", "dynamic", "unknown"}:
            mobility = "unknown"
        score = max(0.0, min(1.0, float(confidence)))
        if mobility == "unknown" and track.mobility_class in {"static", "dynamic"}:
            return
        source_name = str(source)
        current_source = str(track.mobility_source)
        source_rank = 2 if source_name.startswith("vlm") else 1 if source_name.startswith("rap") else 0
        current_rank = 2 if current_source.startswith("vlm") else 1 if current_source.startswith("rap") else 0
        if source_rank > current_rank or (source_rank == current_rank and score >= track.mobility_confidence):
            track.mobility_class = mobility
            track.mobility_confidence = score
            track.mobility_source = str(source)

    def _commit_semantic_label(self, track: PersistentObjectTrack, timestamp_sec: float, reason: str) -> None:
        semantic_label, final_source, final_confidence, semantic_hydra_class_id = self._choose_semantic_label(track)
        track.slot_state = "semantic_resolved"
        track.semantic_timestamp_sec = float(timestamp_sec)
        track.semantic_update_count += 1
        track.semantic_label = semantic_label
        track.semantic_label_source = final_source
        track.semantic_label_confidence = float(final_confidence)
        track.semantic_hydra_class_id = int(semantic_hydra_class_id)
        track.semantic_reason = str(reason)

    def _choose_semantic_label(self, track: PersistentObjectTrack) -> Tuple[str, str, float, int]:
        total = float(sum(track.label_evidence.values()))
        if not track.label_evidence:
            return "unclassified_object", "none", 0.0, 0
        best_label, best_score = max(track.label_evidence.items(), key=lambda item: item[1])
        consensus = float(best_score / total) if total > 0.0 else 0.0
        observations = int(track.label_observations.get(best_label, 0))
        enough_observations = track.seen_count >= int(self.config.semantic_result_min_observations)
        enough_consensus = consensus >= float(self.config.semantic_result_min_consensus)
        enough_confidence = best_score >= float(self.config.semantic_result_min_evidence)
        if not (enough_observations and enough_consensus and enough_confidence):
            return "unclassified_object", "insufficient_evidence", consensus, 0
        # The slot remains the physical identity. The selected label is stored
        # only as semantic metadata for the downstream Hydra/RAP fuser.
        class_id = int(track.hydra_label_id) if int(track.hydra_label_id) > 0 else 0
        source = "vlm" if _normalise_label(track.raw_vlm_label) == best_label else "rap"
        # ``consensus`` measures agreement between candidate labels; it is not
        # model confidence. With one observation it is always 1.0 and hid the
        # actual RAP/VLM confidence from the fuser and diagnostics.
        raw_confidence = float(track.label_confidence)
        if _normalise_label(track.canonical_label) != best_label:
            raw_confidence = min(1.0, max(0.0, float(best_score)))
        return best_label, source, raw_confidence, class_id

    @staticmethod
    def _annotate_metadata(
        metadata: Dict[str, Any],
        track: PersistentObjectTrack,
        track_event: str,
        match_reason: str,
        match_score: Optional[float],
    ) -> None:
        active_segment = track.segments.get(int(track.active_segment_slot_id))
        active_segment_record = (
            PersistentObjectTracker._segment_record(active_segment)
            if active_segment is not None else {}
        )
        all_segment_records = [
            PersistentObjectTracker._segment_record(segment)
            for segment in track.segments.values()
        ]
        metadata.update(
            {
                "persistent_track_id": track.track_id,
                "internal_object_id": track.track_id,
                "persistent_instance_id": int(track.instance_id),
                "local_segment_id": f"{track.track_id}:slot_{int(track.hydra_label_id)}",
                "semantic_segment_id": f"{track.track_id}:slot_{int(track.hydra_label_id)}",
                "local_segment_slot_id": int(track.hydra_label_id),
                "local_segment_event": str(track.last_segment_event),
                "local_segment_match_reason": str(track.last_segment_match_reason),
                "local_segment_match_score": None if track.last_segment_match_score is None else float(track.last_segment_match_score),
                "local_segment_xy_span_m": active_segment_record.get("local_segment_xy_span_m"),
                "local_segment_centroid_3d": active_segment_record.get("centroid_3d"),
                "local_segment_bbox_3d_min": active_segment_record.get("bbox_3d_min"),
                "local_segment_bbox_3d_max": active_segment_record.get("bbox_3d_max"),
                "local_segment_seen_count": active_segment_record.get("seen_count", 0),
                "semantic_segments": all_segment_records,
                "semantic_slot_ids": [int(item.get("hydra_slot_id", 0)) for item in all_segment_records],
                "semantic_segment_count": len(all_segment_records),
                "persistent_track_event": track_event,
                "persistent_track_seen_count": int(track.seen_count),
                "persistent_match_reason": match_reason,
                "persistent_match_score": None if match_score is None else float(match_score),
                "hydra_label_id": int(track.hydra_label_id),
                "hydra_label_name": track.hydra_label_name,
                "semantic_kind": track.semantic_kind,
                "canonical_label": track.canonical_label,
                "semantic_label_source": track.label_source,
                "semantic_label_confidence": float(track.label_confidence),
                "raw_rap_label": track.raw_rap_label,
                "raw_vlm_label": track.raw_vlm_label,
                "mobility_class": track.mobility_class,
                "mobility_confidence": float(track.mobility_confidence),
                "mobility_source": track.mobility_source,
                "slot_state": track.slot_state,
                "labeling_dispatched": bool(track.labeling_dispatched),
                "labeling_completed": bool(track.labeling_completed),
                "labeling_status": str(track.labeling_status),
            }
        )

    @staticmethod
    def _segment_record(segment: PersistentObjectSegment) -> Dict[str, Any]:
        span = None
        if segment.bbox_3d_min is not None and segment.bbox_3d_max is not None:
            span = _aabb_xy_diagonal(segment.bbox_3d_min, segment.bbox_3d_max)
        return {
            "local_segment_id": str(segment.segment_id),
            "semantic_segment_id": str(segment.segment_id),
            "hydra_slot_id": int(segment.hydra_label_id),
            "hydra_slot_name": str(segment.hydra_label_name),
            "hydra_label_id": int(segment.hydra_label_id),
            "hydra_label_name": str(segment.hydra_label_name),
            "instance_id": int(segment.instance_id),
            "seen_count": int(segment.seen_count),
            "first_seen_timestamp_sec": float(segment.first_seen_timestamp_sec),
            "last_seen_timestamp_sec": float(segment.last_seen_timestamp_sec),
            "centroid_3d": _as_list(segment.centroid_3d),
            "bbox_2d": segment.bbox_2d,
            "bbox_3d_min": _as_list(segment.bbox_3d_min),
            "bbox_3d_max": _as_list(segment.bbox_3d_max),
            "last_bbox_3d_min": _as_list(segment.last_bbox_3d_min),
            "last_bbox_3d_max": _as_list(segment.last_bbox_3d_max),
            "local_segment_xy_span_m": span,
            "closed": bool(segment.closed),
        }

    def _track_record(
        self,
        track: PersistentObjectTrack,
        event: str,
        reason: str,
        score: Optional[float],
    ) -> Dict[str, Any]:
        active_segment = track.segments.get(int(track.active_segment_slot_id))
        active_segment_record = self._segment_record(active_segment) if active_segment is not None else {}
        all_segment_records = [self._segment_record(segment) for segment in track.segments.values()]
        return {
            "event": event,
            "track_event": event,
            "persistent_track_event": event,
            "persistent_track_id": track.track_id,
            "internal_object_id": track.track_id,
            "persistent_instance_id": int(track.instance_id),
            "local_segment_id": f"{track.track_id}:slot_{int(track.hydra_label_id)}",
            "semantic_segment_id": f"{track.track_id}:slot_{int(track.hydra_label_id)}",
            "local_segment_slot_id": int(track.hydra_label_id),
            "local_segment_event": str(track.last_segment_event),
            "local_segment_match_reason": str(track.last_segment_match_reason),
            "local_segment_match_score": None if track.last_segment_match_score is None else float(track.last_segment_match_score),
            "local_segment_xy_span_m": active_segment_record.get("local_segment_xy_span_m"),
            "local_segment_centroid_3d": active_segment_record.get("centroid_3d"),
            "local_segment_bbox_3d_min": active_segment_record.get("bbox_3d_min"),
            "local_segment_bbox_3d_max": active_segment_record.get("bbox_3d_max"),
            "local_segment_seen_count": active_segment_record.get("seen_count", 0),
            "persistent_track_seen_count": int(track.seen_count),
            "persistent_match_reason": reason,
            "persistent_match_score": None if score is None else float(score),
            "hydra_slot_id": int(track.hydra_label_id),
            "hydra_slot_name": track.hydra_label_name,
            "hydra_label_id": int(track.hydra_label_id),
            "hydra_label_name": track.hydra_label_name,
            "canonical_label": track.canonical_label,
            "semantic_label_source": track.label_source,
            "semantic_label_confidence": float(track.label_confidence),
            "slot_state": track.slot_state,
            "labeling_dispatched": bool(track.labeling_dispatched),
            "labeling_completed": bool(track.labeling_completed),
            "labeling_status": str(track.labeling_status),
            "semantic_label": track.semantic_label,
            "semantic_label_source": track.semantic_label_source,
            "semantic_label_confidence": float(track.semantic_label_confidence),
            "mobility_class": track.mobility_class,
            "mobility_confidence": float(track.mobility_confidence),
            "mobility_source": track.mobility_source,
            "semantic_hydra_class_id": int(track.semantic_hydra_class_id),
            "semantic_reason": track.semantic_reason,
            "semantic_timestamp_sec": track.semantic_timestamp_sec,
            "semantic_update_count": int(track.semantic_update_count),
            "first_seen_timestamp_sec": float(track.first_seen_timestamp_sec),
            "last_seen_timestamp_sec": float(track.last_seen_timestamp_sec),
            "centroid_3d": _as_list(track.centroid_3d),
            "bbox_volume_m3": track.bbox_volume_m3,
            "bbox_2d": track.bbox_2d,
            "bbox_3d_min": _as_list(track.bbox_3d_min),
            "bbox_3d_max": _as_list(track.bbox_3d_max),
            "last_bbox_3d_min": _as_list(track.last_bbox_3d_min),
            "last_bbox_3d_max": _as_list(track.last_bbox_3d_max),
            "label_evidence": {str(k): float(v) for k, v in track.label_evidence.items()},
            "label_observations": {str(k): int(v) for k, v in track.label_observations.items()},
            "semantic_slot_ids": [int(item.get("hydra_slot_id", 0)) for item in all_segment_records],
            "semantic_segment_count": len(all_segment_records),
            "active_segment": active_segment_record,
            "semantic_segments": all_segment_records,
        }
