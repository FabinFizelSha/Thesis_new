"""Semantic labeling (RAP/VLM) pipeline stage."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from nodes.support.phase1.semantic_crop import build_rap_target_only_crop, build_vlm_target_focus_crop


class SemanticsStage:
    """Wraps RAP/VLM semantic labeling logic."""

    def __init__(self, config: Any, logger: Any):
        """Initialize semantics stage.

        Args:
            config: Phase1 configuration
            logger: ROS logger
        """
        self.config = config
        self.logger = logger

    def enqueue_rap_task(self, track_id: str, crop: np.ndarray) -> Optional[str]:
        """Enqueue a mask for RAP semantic classification.

        Args:
            track_id: Track ID
            crop: Image crop to classify

        Returns:
            Task ID if enqueued, None otherwise
        """
        if not bool(self.config.rap_enabled):
            return None

        # Task would be enqueued with coordinator
        return f"rap_task_{track_id}"

    def enqueue_vlm_task(self, track_id: str, crop: np.ndarray) -> Optional[str]:
        """Enqueue a mask for VLM semantic description.

        Args:
            track_id: Track ID
            crop: Image crop to describe

        Returns:
            Task ID if enqueued, None otherwise
        """
        if not bool(self.config.vlm_enabled):
            return None

        # Task would be enqueued with coordinator
        return f"vlm_task_{track_id}"

    def build_rap_crop(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        object_bbox_2d: Any,
        prepared_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return the exact target-only image supplied to RAP."""
        if not bool(self.config.semantic_crop_rap_target_only_enabled):
            return self._extract_crop(rgb, object_bbox_2d)
        return build_rap_target_only_crop(
            rgb,
            mask,
            object_bbox_2d,
            background_rgb=self.config.semantic_crop_rap_background_rgb,
            cleanup_mask=self.config.semantic_crop_mask_cleanup_enabled,
            cleanup_min_component_area_ratio=self.config.semantic_crop_mask_cleanup_min_component_area_ratio,
            cleanup_component_max_gap_px=self.config.semantic_crop_mask_cleanup_component_max_gap_px,
            prepared_mask=prepared_mask,
        )

    def build_vlm_crop(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        context_bbox_2d: Any,
        prepared_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return the exact target-focused image supplied to the VLM."""
        if not bool(self.config.semantic_crop_vlm_target_focus_enabled):
            return self._extract_crop(rgb, context_bbox_2d)
        return build_vlm_target_focus_crop(
            rgb,
            mask,
            context_bbox_2d,
            context_alpha=self.config.semantic_crop_vlm_context_alpha,
            grayscale_context=self.config.semantic_crop_vlm_context_grayscale,
            near_context_enabled=self.config.semantic_crop_vlm_near_context_enabled,
            near_context_alpha=self.config.semantic_crop_vlm_near_context_alpha,
            near_context_dilation_px=self.config.semantic_crop_vlm_near_context_dilation_px,
            near_context_grayscale=self.config.semantic_crop_vlm_near_context_grayscale,
            cleanup_mask=self.config.semantic_crop_mask_cleanup_enabled,
            cleanup_min_component_area_ratio=self.config.semantic_crop_mask_cleanup_min_component_area_ratio,
            cleanup_component_max_gap_px=self.config.semantic_crop_mask_cleanup_component_max_gap_px,
            draw_target_contour=self.config.semantic_crop_draw_target_contour,
            contour_rgb=self.config.semantic_crop_target_contour_rgb,
            contour_thickness_px=self.config.semantic_crop_target_contour_thickness_px,
            prepared_mask=prepared_mask,
        )

    def _extract_crop(self, rgb: np.ndarray, bbox_2d: Any) -> Optional[np.ndarray]:
        """Extract simple axis-aligned crop from image."""
        if bbox_2d is None or len(bbox_2d) < 4:
            return None
        x, y, w, h = int(bbox_2d[0]), int(bbox_2d[1]), int(bbox_2d[2]), int(bbox_2d[3])
        x_max = min(x + w, rgb.shape[1])
        y_max = min(y + h, rgb.shape[0])
        if x >= x_max or y >= y_max:
            return None
        return rgb[y:y_max, x:x_max].copy()

    def score_track_crop(
        self, metadata: Dict[str, Any], image_shape: Tuple[int, int], bbox_2d: Any
    ) -> Dict[str, Any]:
        """Score one candidate crop for semantic usefulness."""
        image_height, image_width = [max(1, int(value)) for value in image_shape]
        if not bbox_2d or len(bbox_2d) != 4:
            return {
                "vlm_crop_quality_score": 0.0,
                "vlm_crop_quality_eligible": False,
                "vlm_crop_quality_reasons": ["invalid_bbox"],
                "vlm_crop_border_edges": 0,
            }

        x, y, width, height = [int(value) for value in bbox_2d]
        width = max(0, min(width, image_width))
        height = max(0, min(height, image_height))
        bbox_area = int(width * height)
        short_side = int(min(width, height))
        mask_area = max(0.0, float(metadata.get("mask_area_px", 0) or 0.0))
        depth_valid_ratio = max(0.0, min(1.0, float(metadata.get("depth_valid_ratio", 0.0) or 0.0)))
        fill_ratio = max(0.0, min(1.0, mask_area / float(max(1, bbox_area))))

        border_edges = int(x <= 0) + int(y <= 0) + int(x + width >= image_width) + int(y + height >= image_height)
        area_score = min(1.0, bbox_area / float(max(1, self.config.vlm_crop_target_area_px)))
        mask_score = min(1.0, mask_area / float(max(1, self.config.vlm_crop_target_area_px)))
        short_side_score = min(1.0, short_side / float(max(1, self.config.vlm_crop_target_short_side_px)))
        border_factor = max(0.35, 1.0 - float(self.config.vlm_crop_border_penalty) * border_edges)

        score = (
            0.28 * area_score
            + 0.24 * mask_score
            + 0.22 * short_side_score
            + 0.16 * depth_valid_ratio
            + 0.10 * fill_ratio
        ) * border_factor
        score = max(0.0, min(1.0, float(score)))

        reasons: List[str] = []
        if bbox_area < int(self.config.vlm_crop_min_area_px):
            reasons.append("crop_area_below_minimum")
        if short_side < int(self.config.vlm_crop_min_short_side_px):
            reasons.append("crop_short_side_below_minimum")
        if score < float(self.config.vlm_crop_min_quality_score):
            reasons.append("crop_quality_below_minimum")
        if border_edges >= 2:
            reasons.append("object_box_heavily_border_clipped")

        return {
            "vlm_crop_quality_score": score,
            "vlm_crop_quality_eligible": not reasons,
            "vlm_crop_quality_reasons": reasons,
            "vlm_crop_border_edges": border_edges,
            "vlm_crop_object_bbox_area_px": bbox_area,
            "vlm_crop_object_short_side_px": short_side,
            "vlm_crop_mask_fill_ratio": fill_ratio,
            "vlm_crop_depth_valid_ratio": depth_valid_ratio,
        }
