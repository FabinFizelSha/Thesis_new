"""Build mask-aware semantic crops for RAP and VLM classification.

RAP receives a tight target-only crop so visual retrieval cannot match a
recognisable context object. VLM receives three visibility levels:

1. the SAM target at original colour and brightness;
2. a narrow, medium-bright colour halo that may contain target parts missed by
   segmentation; and
3. heavily dimmed grayscale distant context for scene orientation only.

The narrow halo improves recognition of partially segmented furniture without
restoring the previous failure mode where a prominent unrelated context object
dominated the VLM decision.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np


def _clipped_bbox_xywh(image_shape: Tuple[int, int], bbox_xywh: Any) -> Optional[Tuple[int, int, int, int]]:
    """Return a clipped ``(x0, y0, x1, y1)`` box or ``None``."""
    if not bbox_xywh or len(bbox_xywh) != 4:
        return None
    image_height, image_width = [int(value) for value in image_shape]
    x, y, width, height = [int(value) for value in bbox_xywh]
    if image_height <= 0 or image_width <= 0 or width <= 0 or height <= 0:
        return None
    x0 = max(0, min(image_width, x))
    y0 = max(0, min(image_height, y))
    x1 = max(0, min(image_width, x + width))
    y1 = max(0, min(image_height, y + height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def context_bbox_xywh(
    image_shape: Tuple[int, int],
    bbox_xywh: Any,
    *,
    context_ratio: float,
) -> list[int]:
    """Return the legacy clipped context box without allocating its crop."""
    if not bbox_xywh or len(bbox_xywh) != 4:
        return []
    image_height, image_width = [int(value) for value in image_shape]
    x, y, width, height = [int(value) for value in bbox_xywh]
    width = max(1, width)
    height = max(1, height)
    ratio = max(0.0, min(0.50, float(context_ratio)))
    pad_x = int(round(width * ratio))
    pad_y = int(round(height * ratio))
    context_x0 = max(0, min(image_width, x - pad_x))
    context_y0 = max(0, min(image_height, y - pad_y))
    context_x1 = max(0, min(image_width, x + width + pad_x))
    context_y1 = max(0, min(image_height, y + height + pad_y))
    if context_x1 <= context_x0 or context_y1 <= context_y0:
        return []
    return [
        int(context_x0), int(context_y0),
        int(context_x1 - context_x0), int(context_y1 - context_y0),
    ]


def _validated_mask(rgb: np.ndarray, mask: Any) -> Optional[np.ndarray]:
    """Return a boolean mask aligned with ``rgb`` or ``None``."""
    if rgb is None or getattr(rgb, "ndim", 0) != 3 or rgb.shape[2] != 3 or mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != rgb.shape[:2]:
        return None
    return mask_array


def _rgb_triplet(value: Sequence[int], fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Normalise a configurable RGB triplet."""
    try:
        values = list(value)
        if len(values) < 3:
            return fallback
        return tuple(max(0, min(255, int(values[index]))) for index in range(3))
    except Exception:
        return fallback


def _bbox_gap_px(a: Sequence[int], b: Sequence[int]) -> float:
    """Return the Euclidean pixel gap between two OpenCV component boxes."""
    ax, ay, aw, ah = [int(value) for value in a[:4]]
    bx, by, bw, bh = [int(value) for value in b[:4]]
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    dx = max(0, ax - bx1, bx - ax1)
    dy = max(0, ay - by1, by - ay1)
    return float(np.hypot(dx, dy))


def clean_target_mask_components(
    mask: np.ndarray,
    *,
    enabled: bool = True,
    min_component_area_ratio: float = 0.02,
    component_max_gap_px: int = 15,
) -> np.ndarray:
    """Remove only small, isolated mask fragments.

    The largest connected component is always retained. Other components are
    retained when they are large enough relative to the largest component or
    lie close to it. This conservative cleanup removes detached speckles while
    preserving meaningful disconnected parts of chairs and large surfaces.
    """
    mask_u8 = np.asarray(mask, dtype=bool).astype(np.uint8)
    if not enabled or not bool(np.any(mask_u8)):
        return mask_u8.astype(bool)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if component_count <= 2:
        return mask_u8.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest_label = int(np.argmax(areas)) + 1
    largest_area = max(1, int(stats[largest_label, cv2.CC_STAT_AREA]))
    minimum_area = max(1, int(round(largest_area * max(0.0, float(min_component_area_ratio)))))
    maximum_gap = max(0, int(component_max_gap_px))
    largest_box = stats[largest_label, :4]

    keep_labels = {largest_label}
    for label_id in range(1, component_count):
        if label_id == largest_label:
            continue
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        gap = _bbox_gap_px(stats[label_id, :4], largest_box)
        if area >= minimum_area or gap <= maximum_gap:
            keep_labels.add(label_id)

    cleaned = np.isin(labels, list(keep_labels))
    return cleaned.astype(bool)


def prepare_target_mask(
    rgb: np.ndarray,
    mask: Any,
    *,
    cleanup_enabled: bool,
    cleanup_min_component_area_ratio: float,
    cleanup_component_max_gap_px: int,
) -> Optional[np.ndarray]:
    """Validate and conservatively clean one target mask."""
    mask_array = _validated_mask(rgb, mask)
    if mask_array is None:
        return None
    return clean_target_mask_components(
        mask_array,
        enabled=cleanup_enabled,
        min_component_area_ratio=cleanup_min_component_area_ratio,
        component_max_gap_px=cleanup_component_max_gap_px,
    )


@lru_cache(maxsize=32)
def _ellipse_span_decomposition(radius: int) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    """Return cached ``(span_width, vertical_offsets)`` for an OpenCV ellipse.

    OpenCV's discrete ellipse consists of contiguous horizontal spans.  Grouping
    equal spans lets us perform each one-dimensional dilation once and combine
    it at every corresponding vertical offset.
    """
    radius = max(0, int(radius))
    size = (2 * radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    offsets_by_width = {}
    for row_index, row in enumerate(kernel):
        nonzero = np.flatnonzero(row)
        if nonzero.size == 0:
            continue
        width = int(nonzero[-1] - nonzero[0] + 1)
        offsets_by_width.setdefault(width, []).append(int(row_index - radius))
    return tuple(
        (width, tuple(offsets))
        for width, offsets in sorted(offsets_by_width.items())
    )


@lru_cache(maxsize=256)
def _horizontal_dilation_kernel(width: int) -> np.ndarray:
    """Return one cached horizontal rectangular morphology kernel."""
    return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(width)), 1))


def dilate_elliptical_mask_exact(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask with exactly OpenCV's discrete elliptical kernel.

    The operation is mathematically identical to a two-dimensional OpenCV
    dilation, but reuses one optimized horizontal dilation for every unique
    ellipse row width.
    """
    source = np.asarray(mask, dtype=bool).astype(np.uint8)
    radius = max(0, int(radius))
    if radius == 0 or source.size == 0:
        return source.astype(bool)

    result = np.zeros_like(source, dtype=np.uint8)
    height = int(source.shape[0])
    for width, offsets in _ellipse_span_decomposition(radius):
        horizontal = cv2.dilate(
            source,
            _horizontal_dilation_kernel(width),
            iterations=1,
        )
        for offset in offsets:
            # cv2.dilate samples source row ``output_y + offset`` for this
            # kernel row. Slices preserve OpenCV's zero border semantics.
            if abs(offset) >= height:
                continue
            if offset < 0:
                result[-offset:height] = np.maximum(
                    result[-offset:height], horizontal[0:height + offset]
                )
            elif offset > 0:
                result[0:height - offset] = np.maximum(
                    result[0:height - offset], horizontal[offset:height]
                )
            else:
                result = np.maximum(result, horizontal)
    return result.astype(bool)


def _dilate_target_roi_exact(target_mask: np.ndarray, radius: int) -> np.ndarray:
    """Apply exact elliptical dilation only where the result can be nonzero."""
    target = np.asarray(target_mask, dtype=bool)
    radius = max(0, int(radius))
    if radius == 0 or not bool(np.any(target)):
        return target.copy()
    ys, xs = np.nonzero(target)
    height, width = target.shape
    x0 = max(0, int(xs.min()) - radius)
    x1 = min(width, int(xs.max()) + radius + 1)
    y0 = max(0, int(ys.min()) - radius)
    y1 = min(height, int(ys.max()) + radius + 1)
    result = np.zeros_like(target, dtype=bool)
    result[y0:y1, x0:x1] = dilate_elliptical_mask_exact(
        target[y0:y1, x0:x1], radius
    )
    return result


def _exterior_background_mask(target_mask: np.ndarray) -> np.ndarray:
    """Return non-target pixels connected to the crop border.

    Dilation around an architectural surface must not brighten objects located
    inside enclosed holes of that surface mask. Restricting the local halo to
    border-connected background preserves context around an incomplete compact
    object while keeping enclosed wall/ceiling occlusions in distant context.
    """
    inverse = (~np.asarray(target_mask, dtype=bool)).astype(np.uint8)
    if not bool(np.any(inverse)):
        return np.zeros_like(inverse, dtype=bool)
    _, labels = cv2.connectedComponents(inverse, connectivity=8)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    return np.isin(labels, border_labels) & (inverse > 0)


def build_rap_target_only_crop(
    rgb: np.ndarray,
    mask: Any,
    bbox_xywh: Any,
    *,
    background_rgb: Sequence[int] = (32, 32, 32),
    cleanup_mask: bool = True,
    cleanup_min_component_area_ratio: float = 0.02,
    cleanup_component_max_gap_px: int = 15,
    prepared_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Return a tight RAP crop containing only SAM target pixels.

    Non-target pixels inside the target bounding box are replaced by a constant
    neutral background. The same conservative mask cleanup used by the VLM
    representation removes detached speckles but keeps substantial or nearby
    disconnected target components.
    """
    mask_array = (
        _validated_mask(rgb, prepared_mask)
        if prepared_mask is not None
        else prepare_target_mask(
            rgb,
            mask,
            cleanup_enabled=cleanup_mask,
            cleanup_min_component_area_ratio=cleanup_min_component_area_ratio,
            cleanup_component_max_gap_px=cleanup_component_max_gap_px,
        )
    )
    clipped = _clipped_bbox_xywh(rgb.shape[:2], bbox_xywh)
    if mask_array is None or clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    target_mask = mask_array[y0:y1, x0:x1]
    if not bool(np.any(target_mask)):
        return None
    source = np.asarray(rgb[y0:y1, x0:x1], dtype=np.uint8)
    background = np.asarray(_rgb_triplet(background_rgb, (32, 32, 32)), dtype=np.uint8)
    result = np.empty_like(source)
    result[...] = background
    result[target_mask] = source[target_mask]
    return result


def build_vlm_target_focus_crop(
    rgb: np.ndarray,
    mask: Any,
    context_bbox_xywh: Any,
    *,
    context_alpha: float = 0.10,
    grayscale_context: bool = True,
    near_context_enabled: bool = True,
    near_context_alpha: float = 0.45,
    near_context_dilation_px: int = 15,
    near_context_grayscale: bool = False,
    cleanup_mask: bool = True,
    cleanup_min_component_area_ratio: float = 0.02,
    cleanup_component_max_gap_px: int = 15,
    draw_target_contour: bool = True,
    contour_rgb: Sequence[int] = (255, 255, 255),
    contour_thickness_px: int = 2,
    prepared_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Return a VLM crop with target, local halo, and distant context.

    The target remains at original RGB values. A configurable dilation around
    the target creates a medium-bright local colour halo, allowing the VLM to
    see chair backs, seats, legs, and similar parts omitted by an incomplete SAM
    mask. Remaining context is heavily dimmed and optionally grayscale so a
    distant wall, chair, plant, or sign cannot dominate classification.
    """
    mask_array = (
        _validated_mask(rgb, prepared_mask)
        if prepared_mask is not None
        else prepare_target_mask(
            rgb,
            mask,
            cleanup_enabled=cleanup_mask,
            cleanup_min_component_area_ratio=cleanup_min_component_area_ratio,
            cleanup_component_max_gap_px=cleanup_component_max_gap_px,
        )
    )
    clipped = _clipped_bbox_xywh(rgb.shape[:2], context_bbox_xywh)
    if mask_array is None or clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    target_mask = mask_array[y0:y1, x0:x1]
    if not bool(np.any(target_mask)):
        return None

    source = np.asarray(rgb[y0:y1, x0:x1], dtype=np.uint8)
    far_alpha = max(0.0, min(1.0, float(context_alpha)))
    if grayscale_context:
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        far_context = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        far_context = source
    result = np.clip(far_context.astype(np.float32) * far_alpha, 0, 255).astype(np.uint8)

    dilation_radius = max(0, int(near_context_dilation_px))
    if near_context_enabled and dilation_radius > 0:
        dilated_target = _dilate_target_roi_exact(target_mask, dilation_radius)
        exterior_background = _exterior_background_mask(target_mask)
        near_mask = dilated_target & ~target_mask & exterior_background
        if bool(np.any(near_mask)):
            if near_context_grayscale:
                near_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
                near_source = cv2.cvtColor(near_gray, cv2.COLOR_GRAY2RGB)
            else:
                near_source = source
            near_alpha = max(0.0, min(1.0, float(near_context_alpha)))
            near_render = np.clip(near_source.astype(np.float32) * near_alpha, 0, 255).astype(np.uint8)
            result[near_mask] = near_render[near_mask]

    result[target_mask] = source[target_mask]

    thickness = max(0, int(contour_thickness_px))
    if draw_target_contour and thickness > 0:
        contours, _ = cv2.findContours(target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            result,
            contours,
            -1,
            _rgb_triplet(contour_rgb, (255, 255, 255)),
            thickness,
        )
    return result
