"""Pure geometry/label helpers shared by PersistentObjectTracker and its
tracker_*.py collaborators.

Dependency-free by design: nothing here imports from persistent_object_
tracker.py or any other tracker_*.py file, so every one of those can import
from this module without risk of a circular import.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

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
