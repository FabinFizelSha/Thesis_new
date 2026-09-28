"""Pure crop-extraction helpers shared by segmentation, tracking, and dispatch."""

from typing import Any, List, Optional, Tuple
import numpy as np
from nodes.support.phase1.semantic_crop import context_bbox_xywh
from rsg.msg import RsgFrame


def extract_crop(rgb: np.ndarray, bbox_2d: Any) -> Optional[np.ndarray]:
    """Extract a tight RGB crop from [x, y, w, h] metadata."""
    crop, _ = extract_crop_with_context(rgb, bbox_2d, context_ratio=0.0)
    return crop


def extract_crop_with_context(
    rgb: np.ndarray,
    bbox_2d: Any,
    *,
    context_ratio: float,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """Extract a clipped crop with symmetric context around an object box."""
    context_bbox = context_bbox_xywh(rgb.shape[:2], bbox_2d, context_ratio=context_ratio)
    if not context_bbox:
        return None, []
    x, y, width, height = context_bbox
    return rgb[y:y + height, x:x + width].copy(), context_bbox


def make_candidate_id(frame: RsgFrame, mask_id: str, index: int, is_known: bool) -> str:
    """Build a reproducible identifier for one mask candidate."""
    prefix = "rsg_known" if is_known else "rsg_unknown"
    safe_frame_id = frame.rsg_frame_id.replace("/", "_")
    return f"{prefix}_{safe_frame_id}_{index:03d}_{mask_id}"
