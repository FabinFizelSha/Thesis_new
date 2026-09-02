"""Pure loop-closure re-anchor decision logic (no ROS dependencies).

Kept separate from ``phase1.py`` so the SE(3) delta maths can be unit-tested
without standing up a ROS node.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Unit-quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    n = float(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def loop_closure_delta(
    rot_old: np.ndarray,
    trans_old: np.ndarray,
    rot_new: np.ndarray,
    trans_new: np.ndarray,
    *,
    min_translation_m: float,
    min_rotation_deg: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Rigid step between two ``map -> odom`` readings.

    Returns ``(rot_delta, trans_delta, |trans_delta|, angle_deg)`` for
    ``delta = T_new . inverse(T_old)`` -- i.e. ``delta . T_old == T_new`` -- or
    ``None`` when the step is below *both* thresholds (nothing to re-anchor).
    """
    rot_old = np.asarray(rot_old, dtype=np.float64).reshape(3, 3)
    trans_old = np.asarray(trans_old, dtype=np.float64).reshape(3)
    rot_new = np.asarray(rot_new, dtype=np.float64).reshape(3, 3)
    trans_new = np.asarray(trans_new, dtype=np.float64).reshape(3)

    rot_delta = rot_new @ rot_old.T
    trans_delta = trans_new - rot_delta @ trans_old
    trans_norm = float(np.linalg.norm(trans_delta))
    cos_angle = float(np.clip((np.trace(rot_delta) - 1.0) / 2.0, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    if trans_norm < float(min_translation_m) and angle_deg < float(min_rotation_deg):
        return None
    return rot_delta, trans_delta, trans_norm, angle_deg
