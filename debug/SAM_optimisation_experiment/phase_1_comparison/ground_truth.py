"""Ground-truth evaluation module using TESSE semantic labels.

Computes precision, recall, F1, and IoU metrics by comparing SAM masks
against TESSE semantic segmentation ground truth.

Usage:
    evaluator = GroundTruthEvaluator(iou_threshold=0.3)

    # Per-frame evaluation
    masks = backend.segment(rgb, depth)
    semantic_gt = load_semantic_labels(frame_id)

    metrics = evaluator.compute_metrics(masks, semantic_gt)
    # Returns: {precision, recall, f1, avg_iou, num_accepted, num_detected}
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    num_detected: int           # Total masks detected
    num_accepted: int           # Masks that match ground truth
    num_gt_objects: int         # Ground truth object count
    precision: float            # accepted / detected
    recall: float               # accepted / gt_objects
    f1_score: float             # Harmonic mean of precision & recall
    iou_scores: List[float]     # IoU for each accepted mask
    avg_iou: float              # Mean IoU across accepted masks
    min_iou: float              # Minimum IoU
    max_iou: float              # Maximum IoU


class GroundTruthEvaluator:
    """Evaluate SAM masks against TESSE semantic ground truth.

    Uses IoU (Intersection over Union) between SAM masks and semantic
    labels to determine mask acceptance and compute accuracy metrics.
    """

    def __init__(self, iou_threshold: float = 0.3, background_class: int = 0):
        """Initialize evaluator.

        Args:
            iou_threshold: Minimum IoU for mask acceptance (default 0.3)
            background_class: Semantic label for background (default 0)
        """
        self.iou_threshold = iou_threshold
        self.background_class = background_class

    def compute_metrics(
        self,
        masks: List[np.ndarray],
        semantic_gt: np.ndarray,
        depth_valid: Optional[np.ndarray] = None,
    ) -> FrameMetrics:
        """Compute accuracy metrics for a frame.

        Args:
            masks: List of boolean masks (H×W each) from SAM
            semantic_gt: Semantic ground truth (H×W uint8)
            depth_valid: Optional boolean mask (H×W) for valid depth pixels

        Returns:
            FrameMetrics with precision, recall, F1, and IoU
        """
        # Apply depth filtering to both masks and ground truth
        if depth_valid is not None:
            # Filter masks: set pixels outside valid depth range to False
            masks = [mask.astype(bool) & depth_valid for mask in masks]
            # Filter ground truth: set pixels outside valid depth range to background
            semantic_gt = semantic_gt.copy()
            semantic_gt[~depth_valid] = self.background_class

        # Find ground truth objects (non-background pixels)
        gt_object_mask = semantic_gt != self.background_class
        num_gt_objects = len(np.unique(semantic_gt[gt_object_mask]))

        if num_gt_objects == 0:
            num_gt_objects = 1  # At least background

        # Evaluate each mask
        iou_scores = []
        num_accepted = 0

        for mask in masks:
            # Ensure boolean type
            mask = mask.astype(bool)

            # Compute IoU with semantic ground truth
            iou = self._compute_iou_with_gt(mask, semantic_gt)

            # Accept mask if IoU exceeds threshold
            if iou >= self.iou_threshold:
                iou_scores.append(iou)
                num_accepted += 1

        num_detected = len(masks)

        # Compute metrics
        if num_detected > 0:
            precision = num_accepted / num_detected
        else:
            precision = 0.0

        if num_gt_objects > 0:
            recall = num_accepted / num_gt_objects
        else:
            recall = 0.0

        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # IoU statistics
        if iou_scores:
            avg_iou = np.mean(iou_scores)
            min_iou = np.min(iou_scores)
            max_iou = np.max(iou_scores)
        else:
            avg_iou = 0.0
            min_iou = 0.0
            max_iou = 0.0

        return FrameMetrics(
            num_detected=num_detected,
            num_accepted=num_accepted,
            num_gt_objects=num_gt_objects,
            precision=precision,
            recall=recall,
            f1_score=f1,
            iou_scores=iou_scores,
            avg_iou=avg_iou,
            min_iou=min_iou,
            max_iou=max_iou,
        )

    def _compute_iou_with_gt(
        self,
        mask: np.ndarray,
        semantic_gt: np.ndarray,
    ) -> float:
        """Compute IoU between mask and dominant ground truth class.

        Algorithm:
        1. Extract pixels where mask = True
        2. Find dominant semantic class in those pixels
        3. Compute IoU with that class in ground truth
        4. Return IoU score (0.0-1.0)

        Args:
            mask: Boolean mask (H×W)
            semantic_gt: Semantic labels (H×W uint8)

        Returns:
            IoU score (0.0-1.0)
        """
        # Ensure boolean
        mask = mask.astype(bool)

        # Get pixels where mask is True
        mask_pixels = semantic_gt[mask]

        if len(mask_pixels) == 0:
            # Empty mask
            return 0.0

        # Find dominant class (ignore background)
        valid_pixels = mask_pixels[mask_pixels != self.background_class]

        if len(valid_pixels) == 0:
            # Mask only covers background
            return 0.0

        # Get dominant class
        unique, counts = np.unique(valid_pixels, return_counts=True)
        dominant_class = unique[np.argmax(counts)]

        # Extract ground truth pixels for dominant class
        gt_mask = semantic_gt == dominant_class

        # Compute IoU
        intersection = np.sum(mask & gt_mask)
        union = np.sum(mask | gt_mask)

        if union == 0:
            return 0.0

        iou = intersection / union
        return float(iou)

    def compute_statistics(self, all_metrics: List[FrameMetrics]) -> Dict:
        """Compute aggregate statistics across frames.

        Args:
            all_metrics: List of FrameMetrics from all frames

        Returns:
            Dictionary with aggregate statistics
        """
        if not all_metrics:
            return {}

        import statistics

        precisions = [m.precision for m in all_metrics]
        recalls = [m.recall for m in all_metrics]
        f1_scores = [m.f1_score for m in all_metrics]
        ious = [m.avg_iou for m in all_metrics]

        return {
            'num_frames': len(all_metrics),
            'precision_mean': statistics.mean(precisions),
            'precision_stdev': statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
            'recall_mean': statistics.mean(recalls),
            'recall_stdev': statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
            'f1_mean': statistics.mean(f1_scores),
            'f1_stdev': statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
            'iou_mean': statistics.mean(ious),
            'iou_stdev': statistics.stdev(ious) if len(ious) > 1 else 0.0,
        }


if __name__ == "__main__":
    # Quick test
    print("Testing GroundTruthEvaluator...")

    evaluator = GroundTruthEvaluator(iou_threshold=0.3)

    # Create synthetic test data
    H, W = 640, 480

    # Semantic ground truth: 3 objects (classes 1, 2, 3) + background (0)
    semantic_gt = np.zeros((H, W), dtype=np.uint8)
    semantic_gt[100:200, 100:200] = 1  # Object 1
    semantic_gt[250:350, 250:350] = 2  # Object 2
    semantic_gt[400:450, 400:450] = 3  # Object 3

    # Create masks that overlap with ground truth objects
    masks = [
        np.zeros((H, W), dtype=bool),
        np.zeros((H, W), dtype=bool),
        np.zeros((H, W), dtype=bool),
    ]

    # Mask 1: Good overlap with object 1
    masks[0][90:210, 90:210] = True

    # Mask 2: Decent overlap with object 2
    masks[1][240:360, 240:360] = True

    # Mask 3: Poor overlap (in background)
    masks[2][10:50, 10:50] = True

    # Evaluate
    metrics = evaluator.compute_metrics(masks, semantic_gt)

    print(f"\nTest Results:")
    print(f"  Detected masks: {metrics.num_detected}")
    print(f"  Accepted masks: {metrics.num_accepted}")
    print(f"  Precision: {metrics.precision:.2f}")
    print(f"  Recall: {metrics.recall:.2f}")
    print(f"  F1 Score: {metrics.f1_score:.2f}")
    print(f"  Avg IoU: {metrics.avg_iou:.2f}")

    print("\n✓ GroundTruthEvaluator working correctly")
