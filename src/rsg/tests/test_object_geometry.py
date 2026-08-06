"""Unit tests for ObjectGeometryEstimator.

A general regression suite for estimate()'s output plus a check on the
optional Step 1 (Part 2) profiling side-channel. Added while investigating
the Part 2 geometry-metadata bottleneck; see
debug/optimisation/optimisation_part2/PART2_REPORT.md for that context. An
earlier version of this file also validated a projection-formula rewrite
(``points_cam @ rot_m.T`` in place of ``(rot_m @ points_cam.T).T``) that was
measured to regress performance on this hardware and was reverted — see
PART2_REPORT.md "Step 2" for that result. These tests remain useful as a
general correctness regression suite independent of that reverted attempt.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from nodes.support.phase1.object_geometry import ObjectGeometryEstimator


def make_config(**overrides: object) -> SimpleNamespace:
    values = dict(
        estimate_object_geometry=True,
        projection_stride=1,
        min_valid_depth_points=5,
        min_depth_m=0.2,
        max_depth_m=6.0,
        centroid_method="median",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_camera_info(fx: float = 525.0, fy: float = 525.0, cx: float = 319.5, cy: float = 239.5) -> SimpleNamespace:
    k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    return SimpleNamespace(k=k)


class ObjectGeometryEstimatorTests(unittest.TestCase):
    """Regression-test estimate() against an independently computed reference."""

    def setUp(self) -> None:
        self.config = make_config()
        self.estimator = ObjectGeometryEstimator(self.config)
        self.camera_info = make_camera_info()
        angle = 0.37
        c, s = np.cos(angle), np.sin(angle)
        self.rot_m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.tx = np.array([1.0, -2.0, 0.5])

        self.image_shape = (20, 20)
        self.mask = np.zeros(self.image_shape, dtype=bool)
        self.mask[6:14, 5:13] = True  # 8x8 block, 64 pixels

        rng = np.random.default_rng(7)
        self.depth = np.full(self.image_shape, 2.0, dtype=np.float32)
        self.depth[self.mask] += rng.uniform(-0.05, 0.05, size=int(self.mask.sum())).astype(np.float32)

    def _expected_world_points(self) -> np.ndarray:
        """Independently reproduce the estimator's pixel selection and back-projection."""
        ys, xs = np.where(self.mask)
        stride = max(1, int(self.config.projection_stride))
        xs_sample, ys_sample = xs[::stride], ys[::stride]
        z = self.depth[ys_sample, xs_sample].astype(np.float64)
        valid = np.isfinite(z) & (z >= self.config.min_depth_m) & (z <= self.config.max_depth_m)
        xs_valid = xs_sample[valid].astype(np.float64)
        ys_valid = ys_sample[valid].astype(np.float64)
        z_valid = z[valid]

        fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
        x_cam = (xs_valid - cx) * z_valid / fx
        y_cam = (ys_valid - cy) * z_valid / fy
        points_cam = np.stack([x_cam, y_cam, z_valid], axis=1)
        return (self.rot_m @ points_cam.T).T + self.tx.reshape(1, 3)

    def test_estimate_matches_independent_reference(self) -> None:
        expected_world = self._expected_world_points()
        expected_centroid = np.median(expected_world, axis=0)
        expected_min = np.min(expected_world, axis=0)
        expected_max = np.max(expected_world, axis=0)
        expected_size = np.maximum(expected_max - expected_min, 0.0)
        expected_volume = float(expected_size[0] * expected_size[1] * expected_size[2])

        geometry = self.estimator.estimate(self.mask, self.depth, self.camera_info, self.tx, self.rot_m)

        self.assertTrue(geometry["valid_geometry"])
        np.testing.assert_allclose(geometry["centroid_3d"], expected_centroid, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(geometry["bbox_3d_min"], expected_min, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(geometry["bbox_3d_max"], expected_max, rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(geometry["bbox_volume_m3"], expected_volume, places=9)
        self.assertEqual(geometry["mask_area_px"], int(self.mask.sum()))

    def test_estimate_output_types_are_plain_python_floats(self) -> None:
        geometry = self.estimator.estimate(self.mask, self.depth, self.camera_info, self.tx, self.rot_m)
        for key in ("centroid_3d", "bbox_3d_min", "bbox_3d_max", "bbox_3d_size"):
            for value in geometry[key]:
                self.assertIsInstance(value, float)

    def test_stage_ms_sink_does_not_change_geometry_output(self) -> None:
        """Passing a profiling stage_ms sink must not alter any returned value."""
        without_sink = self.estimator.estimate(self.mask, self.depth, self.camera_info, self.tx, self.rot_m)
        stage_ms: dict = {}
        with_sink = self.estimator.estimate(self.mask, self.depth, self.camera_info, self.tx, self.rot_m, stage_ms=stage_ms)
        self.assertEqual(without_sink, with_sink)
        for key in (
            "geometry_mask_extract_ms", "geometry_depth_gather_ms",
            "geometry_projection_ms", "geometry_stats_ms",
        ):
            self.assertIn(key, stage_ms)
            self.assertGreaterEqual(stage_ms[key], 0.0)


if __name__ == "__main__":
    unittest.main()
