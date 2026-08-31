"""SAM Segmentation pipeline stage."""

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
from nodes.support.phase1.backends import SamMask


class SegmentationStage:
    """Wraps SAM segmentation logic."""

    def __init__(self, backend: Any, config: Any, logger: Any):
        """Initialize segmentation stage.

        Args:
            backend: SAM backend instance
            config: Phase1 configuration
            logger: ROS logger
        """
        self.backend = backend
        self.config = config
        self.logger = logger

    def run(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        min_area_px: Optional[int] = None
    ) -> Tuple[List[SamMask], Dict[str, Any], Dict[str, float]]:
        """Run SAM segmentation with full timing metrics.

        Args:
            rgb: RGB image
            depth: Depth image
            min_area_px: Minimum mask area threshold

        Returns:
            Tuple of (masks, preparation info, timing metrics)
        """
        timing = {}

        # Check if SAM is enabled
        if not self.config.sam_enabled:
            return [], {}, timing

        prepare_start = time.perf_counter()
        prep_info = self._prepare_input(rgb, depth)
        timing["sam_prepare_ms"] = (time.perf_counter() - prepare_start) * 1000.0

        if min_area_px is None:
            min_area_px = self._scaled_min_pixels(prep_info)

        infer_start = time.perf_counter()
        masks = self.backend.segment(prep_info["rgb"])
        timing["sam_inference_ms"] = (time.perf_counter() - infer_start) * 1000.0

        # Sort masks by confidence score (descending) and take top N
        sorted_masks = sorted(
            masks,
            key=lambda m: float(m.score),
            reverse=True
        )

        # Filter masks by area and max count (using sorted order)
        filtered = []
        for mask in sorted_masks[: int(self.config.sam_max_masks)]:
            if int(mask.area_px) >= min_area_px:
                filtered.append(mask)

        restore_start = time.perf_counter()
        # Restore to original image coordinates
        filtered = self._restore_to_original(filtered, prep_info)
        timing["sam_restore_ms"] = (time.perf_counter() - restore_start) * 1000.0

        return filtered, prep_info, timing

    def _prepare_input(
        self,
        rgb: np.ndarray,
        depth: np.ndarray
    ) -> Dict[str, Any]:
        """Prepare SAM input with resizing, normalization, and depth filtering.

        Processing order:
        1. Resize RGB/depth by sam_input_scale_ratio
        2. Apply depth range filter if enabled
        3. Optionally crop to valid depth ROI
        4. Replace invalid-depth pixels with background color

        Args:
            rgb: Original RGB image
            depth: Original depth image

        Returns:
            Dictionary with prepared image and full metadata
        """
        orig_h, orig_w = rgb.shape[:2]
        scale = float(self.config.sam_input_scale_ratio)
        scale = max(0.05, min(1.0, scale))
        proc_h = max(1, int(round(orig_h * scale)))
        proc_w = max(1, int(round(orig_w * scale)))

        # Resize if needed
        if proc_h != orig_h or proc_w != orig_w:
            try:
                import cv2
                interp = self._interpolation_method()
                rgb_proc = cv2.resize(rgb, (proc_w, proc_h), interpolation=interp)
                depth_proc = cv2.resize(depth, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                # Fallback without OpenCV
                y_idx = np.linspace(0, orig_h - 1, proc_h).astype(np.int64)
                x_idx = np.linspace(0, orig_w - 1, proc_w).astype(np.int64)
                rgb_proc = rgb[np.ix_(y_idx, x_idx)]
                depth_proc = depth[np.ix_(y_idx, x_idx)]
        else:
            rgb_proc = rgb.copy()
            depth_proc = depth

        # Build metadata dict
        meta: Dict[str, Any] = {
            "scale_ratio": float(scale),
            "original_shape_hw": [int(orig_h), int(orig_w)],
            "processed_shape_hw": [int(proc_h), int(proc_w)],
            "sam_shape_hw": [int(proc_h), int(proc_w)],
            "roi_xyxy": None,
            "depth_filter_enabled": bool(self.config.sam_depth_filter_enabled),
            "depth_filter_valid_ratio": None,
            "depth_filter_effective_range_m": [
                float(self.config.sam_depth_filter_min_m),
                float(self.config.sam_depth_filter_max_m),
            ],
        }

        # Apply depth filtering if enabled
        if not self.config.sam_depth_filter_enabled:
            meta["valid_depth_mask_sam"] = None
            return {"rgb": rgb_proc, "depth": depth_proc, **meta}

        # Compute valid depth mask
        valid = (
            np.isfinite(depth_proc)
            & (depth_proc >= float(self.config.sam_depth_filter_min_m))
            & (depth_proc <= float(self.config.sam_depth_filter_max_m))
        )
        valid_ratio = float(np.count_nonzero(valid)) / float(valid.size) if valid.size else 0.0
        meta["depth_filter_valid_ratio"] = valid_ratio

        background = np.array(self.config.sam_depth_filter_background_value, dtype=rgb_proc.dtype).reshape(1, 1, 3)

        # If too little valid depth, give SAM a blank image
        if valid_ratio < float(self.config.sam_depth_filter_min_valid_ratio):
            rgb_blank = np.zeros_like(rgb_proc)
            if rgb_blank.ndim == 3 and rgb_blank.shape[2] == 3:
                rgb_blank[:] = background
            meta["valid_depth_mask_sam"] = np.zeros(rgb_blank.shape[:2], dtype=bool)
            meta["depth_filter_rejected_by_valid_ratio"] = True
            return {"rgb": rgb_blank, "depth": depth_proc, **meta}

        # Optionally crop to valid depth ROI
        if bool(self.config.sam_depth_filter_crop_to_roi) and np.any(valid):
            ys, xs = np.where(valid)
            margin = int(self.config.sam_depth_filter_roi_margin_px)
            x0 = max(0, int(xs.min()) - margin)
            y0 = max(0, int(ys.min()) - margin)
            x1 = min(proc_w, int(xs.max()) + margin + 1)
            y1 = min(proc_h, int(ys.max()) + margin + 1)
            rgb_sam = rgb_proc[y0:y1, x0:x1].copy()
            valid_sam = valid[y0:y1, x0:x1]
            meta["roi_xyxy"] = [int(x0), int(y0), int(x1), int(y1)]
            meta["sam_shape_hw"] = [int(rgb_sam.shape[0]), int(rgb_sam.shape[1])]
        else:
            rgb_sam = rgb_proc.copy()
            valid_sam = valid

        # Replace invalid-depth pixels with background color
        if rgb_sam.ndim == 3 and rgb_sam.shape[2] == 3:
            rgb_sam[~valid_sam] = background
        meta["valid_depth_mask_sam"] = valid_sam

        return {"rgb": rgb_sam, "depth": depth_proc, **meta}

    def _interpolation_method(self) -> int:
        """Get OpenCV interpolation method from config."""
        import cv2
        methods = {
            "LINEAR": cv2.INTER_LINEAR,
            "NEAREST": cv2.INTER_NEAREST,
            "CUBIC": cv2.INTER_CUBIC,
            "AREA": cv2.INTER_AREA,
        }
        method_name = str(self.config.sam_resize_interpolation).upper()
        return methods.get(method_name, cv2.INTER_LINEAR)

    def _scaled_min_pixels(self, prep_info: Dict[str, Any]) -> int:
        """Get minimum mask pixels scaled for resize."""
        base_min = int(self.config.sam_min_mask_pixels)
        scale = prep_info.get("scale_ratio", 1.0)
        return max(1, int(base_min * scale * scale))

    def _restore_to_original(
        self,
        masks: List[SamMask],
        prep_info: Dict[str, Any]
    ) -> List[SamMask]:
        """Restore masks from SAM input coordinates back to original RGB coordinates.

        This handles:
        - Resizing from processed resolution back to original
        - ROI cropping restoration (if depth filter was used)
        - Valid depth mask reapplication (if depth filter was used)
        - Area filtering at original resolution
        """
        if not masks:
            return []

        try:
            import cv2
        except Exception as exc:
            self.logger.error(f"OpenCV required for mask restoration: {exc}")
            return []

        orig_h, orig_w = prep_info["original_shape_hw"]
        proc_h, proc_w = prep_info["processed_shape_hw"]
        roi = prep_info.get("roi_xyxy")
        valid_sam = prep_info.get("valid_depth_mask_sam")
        min_mask_pixels = int(self.config.sam_min_mask_pixels)

        restored: List[SamMask] = []
        for idx, mask in enumerate(masks):
            sam_mask = np.asarray(mask.mask, dtype=bool)

            # Reapply valid depth mask if present
            if valid_sam is not None and sam_mask.shape == np.asarray(valid_sam).shape:
                sam_mask = sam_mask & np.asarray(valid_sam, dtype=bool)

            # Restore from ROI or processed resolution to processed resolution
            proc_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
            if roi is not None:
                x0, y0, x1, y1 = [int(v) for v in roi]
                roi_h = max(0, y1 - y0)
                roi_w = max(0, x1 - x0)
                if sam_mask.shape[:2] != (roi_h, roi_w):
                    sam_mask_u8 = cv2.resize(sam_mask.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                else:
                    sam_mask_u8 = sam_mask.astype(np.uint8)
                proc_mask[y0:y1, x0:x1] = sam_mask_u8[:roi_h, :roi_w]
            else:
                if sam_mask.shape[:2] != (proc_h, proc_w):
                    proc_mask = cv2.resize(sam_mask.astype(np.uint8), (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
                else:
                    proc_mask = sam_mask.astype(np.uint8)

            # Resize to original resolution
            full_mask = cv2.resize(proc_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            area = int(np.count_nonzero(full_mask))

            # Filter by minimum area at original resolution
            if area < min_mask_pixels:
                continue

            ys, xs = np.where(full_mask)
            if xs.size == 0 or ys.size == 0:
                continue

            x_min = int(xs.min())
            y_min = int(ys.min())
            x_max = int(xs.max()) + 1
            y_max = int(ys.max()) + 1

            # Add metadata about restoration
            metadata = dict(mask.metadata or {})
            metadata.update({
                "restored_from_scaled_sam": True,
                "sam_input_scale_ratio": prep_info.get("scale_ratio", 1.0),
                "sam_depth_filter_enabled": prep_info.get("depth_filter_enabled", False),
                "sam_depth_filter_valid_ratio": prep_info.get("depth_filter_valid_ratio"),
                "sam_roi_xyxy": prep_info.get("roi_xyxy"),
            })

            restored.append(
                SamMask(
                    mask_id=str(mask.mask_id or f"mask_{idx:03d}"),
                    mask=full_mask,
                    bbox_2d=[x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min)],
                    area_px=area,
                    crop=mask.crop,
                    score=float(mask.score),
                    metadata=metadata,
                )
            )

            if len(restored) >= int(self.config.sam_max_masks):
                break

        return restored
