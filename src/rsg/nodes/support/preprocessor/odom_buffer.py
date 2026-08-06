"""Odometry buffering and timestamp association."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from nav_msgs.msg import Odometry

from nodes.support.preprocessor.time_utils import stamp_to_float
from nodes.support.preprocessor.transform_math import TransformMath


class OdomBuffer:
    """Time-ordered odometry buffer with cursor-based timestamp lookup."""

    def __init__(self, max_size: int, tolerance_sec: float, use_interpolation: bool, assume_ordered: bool = True) -> None:
        self.max_size = max_size
        self.tolerance_sec = tolerance_sec
        self.use_interpolation = use_interpolation
        self.assume_ordered = assume_ordered
        self._messages: List[Odometry] = []
        self._cursor_index = 0
        self._last_lookup_target: Optional[float] = None

    def add(self, msg: Odometry) -> None:
        """Add odometry while keeping the moving lookup cursor valid."""
        self._messages.append(msg)
        if not self.assume_ordered:
            self._messages.sort(key=lambda item: stamp_to_float(item.header.stamp))
            self._cursor_index = 0
            self._last_lookup_target = None

        overflow = len(self._messages) - self.max_size
        if overflow > 0:
            del self._messages[:overflow]
            self._cursor_index = max(0, self._cursor_index - overflow)

        if self._messages:
            self._cursor_index = min(self._cursor_index, len(self._messages) - 1)

    def lookup(self, target_time: float) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        """Find or interpolate odometry for an RGB timestamp."""
        if not self._messages:
            return None, None, "odom_buffer_empty"

        self._seek_cursor(target_time)
        if self.use_interpolation:
            interpolated = self._lookup_interpolated(target_time)
            if interpolated[0] is not None:
                self._last_lookup_target = target_time
                return interpolated

        nearest = self._lookup_nearest(target_time)
        self._last_lookup_target = target_time
        return nearest

    def _seek_cursor(self, target_time: float) -> None:
        """Move the cursor to the last ordered sample not after target time."""
        if not self._messages:
            self._cursor_index = 0
            return

        if (
            self._last_lookup_target is not None
            and target_time < self._last_lookup_target
        ):
            self._cursor_index = 0

        self._cursor_index = min(self._cursor_index, len(self._messages) - 1)

        while self._cursor_index > 0:
            current_time = stamp_to_float(self._messages[self._cursor_index].header.stamp)
            if current_time <= target_time:
                break
            self._cursor_index -= 1

        while self._cursor_index + 1 < len(self._messages):
            next_time = stamp_to_float(self._messages[self._cursor_index + 1].header.stamp)
            if next_time > target_time:
                break
            self._cursor_index += 1

    def _lookup_nearest(self, target_time: float) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        """Return the nearest sample using only the cursor neighbourhood."""
        candidate_indices = {
            max(0, self._cursor_index - 1),
            self._cursor_index,
            min(len(self._messages) - 1, self._cursor_index + 1),
        }
        candidates = [self._messages[index] for index in sorted(candidate_indices)]
        nearest = min(
            candidates,
            key=lambda item: abs(stamp_to_float(item.header.stamp) - target_time),
        )
        delta = abs(stamp_to_float(nearest.header.stamp) - target_time)
        if delta > self.tolerance_sec:
            return None, delta, "nearest_odom_outside_tolerance"
        return TransformMath.odom_to_transform(nearest), delta, "nearest_odom"

    def _lookup_interpolated(self, target_time: float) -> Tuple[Optional[np.ndarray], Optional[float], str]:
        """Interpolate between cursor-adjacent ordered odometry samples."""
        if len(self._messages) < 2:
            return None, None, "interpolation_bounds_missing"

        before_index = self._cursor_index
        if before_index + 1 >= len(self._messages):
            return None, None, "interpolation_bounds_missing"

        before = self._messages[before_index]
        after = self._messages[before_index + 1]
        before_time = stamp_to_float(before.header.stamp)
        after_time = stamp_to_float(after.header.stamp)

        if before_time > target_time or after_time < target_time:
            return None, None, "interpolation_bounds_missing"

        nearest_delta = min(abs(target_time - before_time), abs(after_time - target_time))
        if nearest_delta > self.tolerance_sec:
            return None, nearest_delta, "interpolated_odom_outside_tolerance"
        if after_time <= before_time:
            return TransformMath.odom_to_transform(before), 0.0, "duplicate_odom_time"

        alpha = (target_time - before_time) / (after_time - before_time)
        p0 = before.pose.pose.position
        p1 = after.pose.pose.position
        translation = np.array([
            p0.x + alpha * (p1.x - p0.x),
            p0.y + alpha * (p1.y - p0.y),
            p0.z + alpha * (p1.z - p0.z),
        ], dtype=np.float64)

        o0 = before.pose.pose.orientation
        o1 = after.pose.pose.orientation
        q0 = np.array([o0.x, o0.y, o0.z, o0.w], dtype=np.float64)
        q1 = np.array([o1.x, o1.y, o1.z, o1.w], dtype=np.float64)
        q = TransformMath.slerp(q0, q1, alpha)
        rotation = TransformMath.quaternion_to_matrix(q[0], q[1], q[2], q[3])
        return TransformMath.make_transform(rotation, translation), nearest_delta, "interpolated_odom"
