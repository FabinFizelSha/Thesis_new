#!/usr/bin/env python3
"""
Ground Truth Evaluation Module
Computes F1, Precision, Recall by matching predicted masks to ground truth.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Metrics:
    """Container for per-frame evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    tp: int
    fp: int
    fn: int


class GroundTruthEvaluator:
    """
    Evaluates predicted masks against ground truth semantic segmentation.

    Uses instance-level IoU matching:
    - Extracts instances from semantic masks via connected components
    - Matches predicted masks to GT masks by IoU
    - Computes precision, recall, F1 score
    """

    def __init__(self, iou_threshold: float = 0.3):
        """
        Args:
            iou_threshold: IoU threshold for considering a match valid
        """
        self.iou_threshold = iou_threshold

    def compute_metrics(self, predicted_masks: np.ndarray,
                       ground_truth: np.ndarray,
                       depth_valid: np.ndarray = None,
                       min_mask_pixels: int = 3500,
                       min_gt_pixels: int = 3500) -> Metrics:
        """
        Compute F1, precision, recall for a single frame.

        Args:
            predicted_masks: [H, W] mask IDs (0 = background)
            ground_truth: [H, W] semantic class labels
            depth_valid: [H, W] boolean mask for valid depth
            min_mask_pixels: minimum object size for predicted masks
            min_gt_pixels: minimum object size for GT masks

        Returns:
            Metrics object with precision, recall, f1_score
        """

        # Apply depth filtering if provided
        if depth_valid is not None:
            predicted_masks = predicted_masks.copy()
            ground_truth = ground_truth.copy()
            predicted_masks[~depth_valid] = 0
            ground_truth[~depth_valid] = 0

        # Extract GT instances (connected components per class)
        gt_instances = self._extract_instances(ground_truth, min_gt_pixels)

        # Filter predicted masks by size
        pred_instances = self._filter_predictions(predicted_masks, min_mask_pixels)

        # Match predicted to GT using IoU
        matches = self._match_instances(pred_instances, gt_instances)

        # Compute metrics
        tp = len(matches)
        fp = len(pred_instances) - tp
        fn = len(gt_instances) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return Metrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            tp=tp,
            fp=fp,
            fn=fn
        )

    def _extract_instances(self, semantic_mask: np.ndarray,
                           min_pixels: int) -> List[np.ndarray]:
        """
        Extract instances from semantic segmentation.

        For each class, apply connected component labeling.
        Returns list of binary masks (one per instance).
        """
        from scipy import ndimage

        instances = []
        unique_classes = np.unique(semantic_mask)

        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue

            # Get mask for this class
            class_mask = (semantic_mask == class_id).astype(np.uint8)

            # Connected component labeling (8-connectivity)
            labeled, num_components = ndimage.label(class_mask, structure=np.ones((3, 3), dtype=np.uint8))

            # Extract each component as separate instance
            for comp_id in range(1, num_components + 1):
                instance_mask = (labeled == comp_id).astype(np.uint8)

                # Filter by size
                if np.sum(instance_mask) >= min_pixels:
                    instances.append(instance_mask)

        return instances

    def _filter_predictions(self, predicted_masks: np.ndarray,
                            min_pixels: int) -> List[np.ndarray]:
        """Filter predicted masks by minimum size."""
        instances = []
        unique_ids = np.unique(predicted_masks)

        for mask_id in unique_ids:
            if mask_id == 0:  # Skip background
                continue

            instance_mask = (predicted_masks == mask_id).astype(np.uint8)
            if np.sum(instance_mask) >= min_pixels:
                instances.append(instance_mask)

        return instances

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def _match_instances(self, predicted: List[np.ndarray],
                         ground_truth: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Match predicted instances to GT instances using IoU.

        Greedy matching: for each predicted mask, find GT with max IoU.
        If IoU >= threshold, count as match.

        Returns:
            List of (pred_idx, gt_idx) tuples for matches
        """
        matches = []
        matched_gt = set()

        for pred_idx, pred_mask in enumerate(predicted):
            best_iou = 0.0
            best_gt_idx = -1

            # Find GT with maximum IoU
            for gt_idx, gt_mask in enumerate(ground_truth):
                if gt_idx in matched_gt:
                    continue

                iou = self._compute_iou(pred_mask, gt_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Accept if above threshold
            if best_iou >= self.iou_threshold:
                matches.append((pred_idx, best_gt_idx))
                matched_gt.add(best_gt_idx)

        return matches
