"""Unit tests for target-only RAP and three-level VLM crops."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from nodes.support.phase1.semantic_crop import (
    _dilate_target_roi_exact,
    build_rap_target_only_crop,
    build_vlm_target_focus_crop,
    clean_target_mask_components,
    context_bbox_xywh,
    dilate_elliptical_mask_exact,
    prepare_target_mask,
)


class SemanticCropTests(unittest.TestCase):
    """Verify target, local halo, and distant context rendering."""

    def setUp(self) -> None:
        self.rgb = np.zeros((9, 9, 3), dtype=np.uint8)
        self.rgb[..., 0] = 120
        self.rgb[..., 1] = 60
        self.rgb[..., 2] = 30
        self.mask = np.zeros((9, 9), dtype=bool)
        self.mask[4:6, 4:6] = True

    def test_rap_replaces_every_non_target_pixel(self) -> None:
        crop = build_rap_target_only_crop(
            self.rgb,
            self.mask,
            [2, 2, 6, 6],
            background_rgb=(32, 32, 32),
            cleanup_mask=False,
        )
        self.assertIsNotNone(crop)
        assert crop is not None
        np.testing.assert_array_equal(crop[0, 0], [32, 32, 32])
        np.testing.assert_array_equal(crop[2, 2], self.rgb[4, 4])
        self.assertEqual(int(np.count_nonzero(np.any(crop != 32, axis=2))), 4)

    def test_vlm_uses_target_near_halo_and_far_context(self) -> None:
        crop = build_vlm_target_focus_crop(
            self.rgb,
            self.mask,
            [0, 0, 9, 9],
            context_alpha=0.10,
            grayscale_context=True,
            near_context_enabled=True,
            near_context_alpha=0.50,
            near_context_dilation_px=1,
            near_context_grayscale=False,
            cleanup_mask=False,
            draw_target_contour=False,
        )
        self.assertIsNotNone(crop)
        assert crop is not None
        np.testing.assert_array_equal(crop[4, 4], self.rgb[4, 4])
        np.testing.assert_array_equal(crop[3, 4], [60, 30, 15])
        self.assertEqual(int(crop[0, 0, 0]), int(crop[0, 0, 1]))
        self.assertEqual(int(crop[0, 0, 1]), int(crop[0, 0, 2]))
        self.assertLess(int(crop[0, 0, 0]), 20)

    def test_near_halo_does_not_brighten_enclosed_mask_hole(self) -> None:
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:7, 2:7] = True
        mask[4, 4] = False
        crop = build_vlm_target_focus_crop(
            self.rgb,
            mask,
            [0, 0, 9, 9],
            context_alpha=0.10,
            grayscale_context=True,
            near_context_enabled=True,
            near_context_alpha=0.50,
            near_context_dilation_px=1,
            near_context_grayscale=False,
            cleanup_mask=False,
            draw_target_contour=False,
        )
        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(int(crop[4, 4, 0]), int(crop[4, 4, 1]))
        self.assertLess(int(crop[4, 4, 0]), 20)

    def test_cleanup_removes_small_isolated_speck(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True
        mask[18, 18] = True
        cleaned = clean_target_mask_components(
            mask,
            enabled=True,
            min_component_area_ratio=0.10,
            component_max_gap_px=2,
        )
        self.assertTrue(bool(cleaned[6, 6]))
        self.assertFalse(bool(cleaned[18, 18]))

    def test_cleanup_keeps_nearby_small_component(self) -> None:
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True
        mask[10, 10] = True
        cleaned = clean_target_mask_components(
            mask,
            enabled=True,
            min_component_area_ratio=0.10,
            component_max_gap_px=2,
        )
        self.assertTrue(bool(cleaned[10, 10]))

    def test_invalid_mask_is_rejected(self) -> None:
        self.assertIsNone(build_rap_target_only_crop(self.rgb, np.zeros((2, 2)), [0, 0, 2, 2]))

    @staticmethod
    def _native_ellipse_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
        size = (2 * radius) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    def test_exact_ellipse_dilation_matches_random_masks(self) -> None:
        rng = np.random.default_rng(1729)
        for radius in (1, 2, 7, 15, 70):
            mask = rng.random((83, 109)) > 0.94
            np.testing.assert_array_equal(
                dilate_elliptical_mask_exact(mask, radius),
                self._native_ellipse_dilation(mask, radius),
            )
            np.testing.assert_array_equal(
                _dilate_target_roi_exact(mask, radius),
                self._native_ellipse_dilation(mask, radius),
            )

    def test_exact_ellipse_dilation_matches_at_image_borders(self) -> None:
        mask = np.zeros((31, 37), dtype=bool)
        mask[0, 0] = True
        mask[0, -1] = True
        mask[-1, 0] = True
        mask[-1, -1] = True
        mask[10:15, 0] = True
        for radius in (1, 4, 12):
            np.testing.assert_array_equal(
                dilate_elliptical_mask_exact(mask, radius),
                self._native_ellipse_dilation(mask, radius),
            )
            np.testing.assert_array_equal(
                _dilate_target_roi_exact(mask, radius),
                self._native_ellipse_dilation(mask, radius),
            )

    def test_exact_ellipse_dilation_matches_when_kernel_exceeds_image(self) -> None:
        mask = np.zeros((3, 5), dtype=bool)
        mask[1, 2] = True
        for radius in (2, 7, 70):
            np.testing.assert_array_equal(
                dilate_elliptical_mask_exact(mask, radius),
                self._native_ellipse_dilation(mask, radius),
            )
            np.testing.assert_array_equal(
                _dilate_target_roi_exact(mask, radius),
                self._native_ellipse_dilation(mask, radius),
            )

    def test_zero_radius_dilation_is_identity(self) -> None:
        mask = np.zeros((8, 11), dtype=bool)
        mask[2:6, 4:8] = True
        np.testing.assert_array_equal(dilate_elliptical_mask_exact(mask, 0), mask)

    def test_shared_prepared_mask_preserves_both_crop_pixels(self) -> None:
        prepared = prepare_target_mask(
            self.rgb,
            self.mask,
            cleanup_enabled=False,
            cleanup_min_component_area_ratio=0.02,
            cleanup_component_max_gap_px=15,
        )
        self.assertIsNotNone(prepared)
        rap_direct = build_rap_target_only_crop(
            self.rgb, self.mask, [2, 2, 6, 6], cleanup_mask=False
        )
        rap_shared = build_rap_target_only_crop(
            self.rgb, self.mask, [2, 2, 6, 6], cleanup_mask=False,
            prepared_mask=prepared,
        )
        vlm_direct = build_vlm_target_focus_crop(
            self.rgb, self.mask, [0, 0, 9, 9], cleanup_mask=False,
            near_context_dilation_px=2,
        )
        vlm_shared = build_vlm_target_focus_crop(
            self.rgb, self.mask, [0, 0, 9, 9], cleanup_mask=False,
            near_context_dilation_px=2, prepared_mask=prepared,
        )
        np.testing.assert_array_equal(rap_shared, rap_direct)
        np.testing.assert_array_equal(vlm_shared, vlm_direct)

    def test_roi_deferred_render_is_pixel_identical_to_full_frame_render(self) -> None:
        """A stored context ROI must preserve the previous crop semantics."""
        rng = np.random.default_rng(20260806)
        rgb = rng.integers(0, 256, size=(72, 96, 3), dtype=np.uint8)
        mask = np.zeros((72, 96), dtype=bool)
        mask[24:45, 34:61] = True
        mask[28:33, 63:67] = True
        target_bbox = [34, 24, 33, 21]
        context_bbox = context_bbox_xywh(
            rgb.shape[:2], target_bbox, context_ratio=0.20
        )
        self.assertEqual(context_bbox, [27, 20, 47, 29])

        prepared_full = prepare_target_mask(
            rgb,
            mask,
            cleanup_enabled=True,
            cleanup_min_component_area_ratio=0.02,
            cleanup_component_max_gap_px=15,
        )
        expected_rap = build_rap_target_only_crop(
            rgb,
            mask,
            target_bbox,
            cleanup_mask=True,
            cleanup_min_component_area_ratio=0.02,
            cleanup_component_max_gap_px=15,
            prepared_mask=prepared_full,
        )
        expected_vlm = build_vlm_target_focus_crop(
            rgb,
            mask,
            context_bbox,
            context_alpha=0.10,
            grayscale_context=True,
            near_context_enabled=True,
            near_context_alpha=0.50,
            near_context_dilation_px=7,
            near_context_grayscale=False,
            cleanup_mask=True,
            cleanup_min_component_area_ratio=0.02,
            cleanup_component_max_gap_px=15,
            draw_target_contour=True,
            contour_rgb=(0, 255, 255),
            contour_thickness_px=2,
            prepared_mask=prepared_full,
        )

        context_x, context_y, context_width, context_height = context_bbox
        roi_rgb = rgb[
            context_y:context_y + context_height,
            context_x:context_x + context_width,
        ].copy()
        roi_mask = mask[
            context_y:context_y + context_height,
            context_x:context_x + context_width,
        ].copy()
        local_bbox = [
            target_bbox[0] - context_x,
            target_bbox[1] - context_y,
            target_bbox[2],
            target_bbox[3],
        ]
        prepared_roi = prepare_target_mask(
            roi_rgb,
            roi_mask,
            cleanup_enabled=True,
            cleanup_min_component_area_ratio=0.02,
            cleanup_component_max_gap_px=15,
        )
        actual_rap = build_rap_target_only_crop(
            roi_rgb,
            roi_mask,
            local_bbox,
            cleanup_mask=True,
            cleanup_min_component_area_ratio=0.02,
            cleanup_component_max_gap_px=15,
            prepared_mask=prepared_roi,
        )
        actual_vlm = build_vlm_target_focus_crop(
            roi_rgb,
            roi_mask,
            [0, 0, context_width, context_height],
            context_alpha=0.10,
            grayscale_context=True,
            near_context_enabled=True,
            near_context_alpha=0.50,
            near_context_dilation_px=7,
            near_context_grayscale=False,
            cleanup_mask=True,
            cleanup_min_component_area_ratio=0.02,
            cleanup_component_max_gap_px=15,
            draw_target_contour=True,
            contour_rgb=(0, 255, 255),
            contour_thickness_px=2,
            prepared_mask=prepared_roi,
        )

        np.testing.assert_array_equal(actual_rap, expected_rap)
        np.testing.assert_array_equal(actual_vlm, expected_vlm)


if __name__ == "__main__":
    unittest.main()
