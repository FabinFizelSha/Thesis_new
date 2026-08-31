"""Object geometry estimation from masks, depth, camera intrinsics, and pose."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ObjectGeometryEstimator:
    """Estimate 2D/3D metadata for SAM mask candidates.

    The estimator is deliberately lightweight. It samples mask pixels using a
    configurable stride, rejects invalid depth values, back-projects image
    points to camera coordinates, and transforms them using the existing
    framework-compatible ``tx`` and ``rotM`` fields.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def estimate(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        camera_info: Any,
        tx: np.ndarray,
        rot_m: np.ndarray,
        stage_ms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Return geometry metadata for one binary mask.

        ``stage_ms`` is an optional profiling side-channel only. When provided,
        elapsed time for each internal sub-step is accumulated into it under
        fixed keys (``geometry_mask_extract_ms``, ``geometry_depth_gather_ms``,
        ``geometry_projection_ms``, ``geometry_stats_ms``). This never changes
        any value in the returned ``geometry`` dict; it exists to attribute
        the aggregate ``geometry_metadata_ms`` cost to a specific sub-step
        without altering behaviour, for the Part 2 (geometry) profiling pass.
        """
        geometry: Dict[str, Any] = {}

        t0 = time.perf_counter() if stage_ms is not None else 0.0
        ys, xs = np.where(mask)
        if xs.size == 0:
            geometry["valid_geometry"] = False
            geometry["reason"] = "empty_mask"
            if stage_ms is not None:
                stage_ms["geometry_mask_extract_ms"] = stage_ms.get("geometry_mask_extract_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
            return geometry

        bbox_2d = self._bbox_2d(xs, ys)
        centroid_2d = [float(np.median(xs)), float(np.median(ys))]
        geometry["bbox_2d"] = bbox_2d
        geometry["centroid_2d"] = centroid_2d
        geometry["mask_area_px"] = int(xs.size)
        if stage_ms is not None:
            stage_ms["geometry_mask_extract_ms"] = stage_ms.get("geometry_mask_extract_ms", 0.0) + (time.perf_counter() - t0) * 1000.0

        if not self.config.estimate_object_geometry:
            geometry["valid_geometry"] = False
            geometry["reason"] = "geometry_disabled"
            return geometry

        t0 = time.perf_counter() if stage_ms is not None else 0.0
        stride = max(1, int(self.config.projection_stride))
        xs_sample = xs[::stride]
        ys_sample = ys[::stride]
        z = depth_m[ys_sample, xs_sample].astype(np.float64, copy=False)
        valid = np.isfinite(z) & (z >= self.config.min_depth_m) & (z <= self.config.max_depth_m)
        xs_valid = xs_sample[valid].astype(np.float64)
        ys_valid = ys_sample[valid].astype(np.float64)
        z_valid = z[valid]

        total = int(z.size)
        valid_count = int(z_valid.size)
        geometry["depth_valid_points"] = valid_count
        geometry["depth_sample_points"] = total
        geometry["depth_valid_ratio"] = 0.0 if total == 0 else float(valid_count / total)
        if stage_ms is not None:
            stage_ms["geometry_depth_gather_ms"] = stage_ms.get("geometry_depth_gather_ms", 0.0) + (time.perf_counter() - t0) * 1000.0

        if valid_count < self.config.min_valid_depth_points:
            geometry["valid_geometry"] = False
            geometry["reason"] = "too_few_valid_depth_points"
            return geometry

        t0 = time.perf_counter() if stage_ms is not None else 0.0
        fx, fy, cx, cy = self._intrinsics(camera_info)
        x_cam = (xs_valid - cx) * z_valid / fx
        y_cam = (ys_valid - cy) * z_valid / fy
        points_cam = np.stack([x_cam, y_cam, z_valid], axis=1)
        points_world = (rot_m @ points_cam.T).T + tx.reshape(1, 3)
        if stage_ms is not None:
            stage_ms["geometry_projection_ms"] = stage_ms.get("geometry_projection_ms", 0.0) + (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter() if stage_ms is not None else 0.0
        if self.config.centroid_method.lower() == "mean":
            centroid = np.mean(points_world, axis=0)
        else:
            centroid = np.median(points_world, axis=0)

        min_xyz = np.min(points_world, axis=0)
        max_xyz = np.max(points_world, axis=0)
        size_xyz = np.maximum(max_xyz - min_xyz, 0.0)

        # Apply minimum assumed depth padding for thin objects (ceiling/floor).
        # For sheet-like objects with very small thickness, expand that dimension
        # to improve volume-ratio comparisons during track deduplication.
        min_assumed_depth = float(getattr(self.config, 'min_assumed_depth_m', 0.30))
        size_xyz_padded = np.array(size_xyz)
        for i in range(3):
            if size_xyz[i] < min_assumed_depth:
                size_xyz_padded[i] = min_assumed_depth

        volume = float(size_xyz_padded[0] * size_xyz_padded[1] * size_xyz_padded[2])

        # Apply hybrid two-pass directional padding to bbox coordinates
        # PASS 1: Z-axis (vertical) padding for floors/ceilings
        bbox_min_padded = np.array(min_xyz, dtype=np.float64)
        bbox_max_padded = np.array(max_xyz, dtype=np.float64)

        depth_z = float(bbox_max_padded[2] - bbox_min_padded[2])
        if depth_z < min_assumed_depth:
            deficit_z = min_assumed_depth - depth_z
            z_center = (bbox_min_padded[2] + bbox_max_padded[2]) / 2.0
            # Floor (low Z): extend downward
            if z_center < 1.0:
                bbox_min_padded[2] -= deficit_z
            # Ceiling (high Z): extend upward
            else:
                bbox_max_padded[2] += deficit_z

        # PASS 2: X/Y-axes (horizontal) padding for thin walls
        min_depth_xy = 0.15  # More conservative for horizontal
        for axis in [0, 1]:
            depth = float(bbox_max_padded[axis] - bbox_min_padded[axis])
            if depth < min_depth_xy:
                deficit = min_depth_xy - depth
                axis_center = (bbox_min_padded[axis] + bbox_max_padded[axis]) / 2.0
                scene_center = 2.5  # Typical room half-width
                # Pad toward edge
                if axis_center < scene_center:
                    bbox_min_padded[axis] -= deficit
                else:
                    bbox_max_padded[axis] += deficit

        # Recalculate centroid from padded bbox
        centroid_padded = (bbox_min_padded + bbox_max_padded) / 2.0

        geometry["valid_geometry"] = True
        geometry["centroid_3d"] = [float(v) for v in centroid_padded]
        geometry["bbox_3d_min"] = [float(v) for v in bbox_min_padded]
        geometry["bbox_3d_max"] = [float(v) for v in bbox_max_padded]
        geometry["bbox_3d_size"] = [float(v) for v in size_xyz]  # Keep original for diagnostics
        geometry["bbox_3d_size_padded"] = [float(v) for v in size_xyz_padded]  # Used for volume calc
        geometry["bbox_volume_m3"] = volume
        geometry["depth_min_m"] = float(np.min(z_valid))
        geometry["depth_max_m"] = float(np.max(z_valid))
        geometry["depth_median_m"] = float(np.median(z_valid))
        if stage_ms is not None:
            stage_ms["geometry_stats_ms"] = stage_ms.get("geometry_stats_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
        return geometry

    @staticmethod
    def _bbox_2d(xs: np.ndarray, ys: np.ndarray) -> list[int]:
        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        return [x_min, y_min, int(x_max - x_min + 1), int(y_max - y_min + 1)]

    @staticmethod
    def _intrinsics(camera_info: Any) -> Tuple[float, float, float, float]:
        k = list(camera_info.k)
        fx = float(k[0]) if k[0] else 1.0
        fy = float(k[4]) if k[4] else 1.0
        cx = float(k[2])
        cy = float(k[5])
        return fx, fy, cx, cy


def filter_metadata(metadata: Dict[str, Any], config: Any) -> Dict[str, Any]:
    """Remove optional metadata fields according to config switches."""
    if not config.include_bbox_2d:
        metadata.pop("bbox_2d", None)
    if not config.include_centroid_2d:
        metadata.pop("centroid_2d", None)
    if not config.include_centroid_3d:
        metadata.pop("centroid_3d", None)
    if not config.include_bbox_3d:
        for key in ["bbox_3d_min", "bbox_3d_max", "bbox_3d_size"]:
            metadata.pop(key, None)
    if not config.include_bbox_volume:
        metadata.pop("bbox_volume_m3", None)
    if not config.include_depth_stats:
        for key in ["depth_valid_points", "depth_sample_points", "depth_valid_ratio", "depth_min_m", "depth_max_m", "depth_median_m"]:
            metadata.pop(key, None)
    if not config.include_mask_area:
        metadata.pop("mask_area_px", None)
    return metadata
