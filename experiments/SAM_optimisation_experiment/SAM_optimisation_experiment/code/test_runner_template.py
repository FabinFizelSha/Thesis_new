#!/usr/bin/env python3
"""
Test Runner Template
Shows how to use all preprocessing and evaluation components together.

This is a generic template - actual test runners (run_phase2_*.py) follow this pattern
but with specific configuration values for each phase.
"""

import sys
import statistics
from pathlib import Path
import numpy as np
import yaml

# Import components
sys.path.insert(0, str(Path(__file__).parent))

from ground_truth import GroundTruthEvaluator
from runners import get_backend_runner
from data_loader import Phase1DatasetLoader
from timing import FrameTimer


def run_experiment(config_path: str, dataset_path: str, num_frames: int = 300):
    """
    Run experiment for a single configuration.

    Args:
        config_path: Path to configuration YAML
        dataset_path: Path to dataset root directory
        num_frames: Number of frames to evaluate
    """

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*80}")
    print(f"TESTING: {Path(config_path).stem}")
    print(f"{'='*80}\n")

    print(f"Config: PPS={config['points_per_side']}, Masks={config['max_masks']}, "
          f"Threshold={config['mask_threshold']}, NMS={config['nms_iou']}", end=" ... ")

    # Initialize components
    loader = Phase1DatasetLoader(dataset_path, random_shuffle=False)
    evaluator = GroundTruthEvaluator(iou_threshold=0.3)
    runner = get_backend_runner(config['backend'], config)
    timer = FrameTimer()

    # Run evaluation on dataset
    metrics_list = []

    for frame_idx, frame_data in enumerate(loader):
        if frame_idx >= num_frames:
            break

        rgb = frame_data['rgb']
        depth = frame_data['depth']
        semantic_gt = frame_data['semantic']

        # Normalize depth to 0-1 range for consistency
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        # Create depth validity mask (example: valid if depth > 0.3 and < 6.0 originally)
        # This would depend on actual depth scaling
        depth_valid = np.ones_like(depth, dtype=bool)

        # Run inference with timing
        with timer.record_frame(frame_idx, config['backend']):
            masks = runner.segment(rgb, depth)

        # Evaluate this frame
        metrics = evaluator.compute_metrics(
            masks,
            semantic_gt,
            depth_valid=depth_valid,
            min_mask_pixels=config['min_mask_pixels'],
            min_gt_pixels=config['min_mask_pixels']  # Match thresholds
        )

        metrics_list.append(metrics)

    # Compute aggregate statistics
    f1_scores = [m.f1_score for m in metrics_list]
    precisions = [m.precision for m in metrics_list]
    recalls = [m.recall for m in metrics_list]
    latencies = [timer.frames[i].total_latency_ms for i in range(len(metrics_list))]

    f1_mean = statistics.mean(f1_scores)
    precision_mean = statistics.mean(precisions)
    recall_mean = statistics.mean(recalls)
    latency_mean_ms = statistics.mean(latencies)
    fps = 1000.0 / latency_mean_ms if latencies else 0.0

    # Print results
    print(f"F1={f1_mean:.4f}, FPS={fps:.2f}")

    # Detailed results table
    print(f"\n{'Config':<25} {'F1':<10} {'Precision':<10} {'Recall':<10} "
          f"{'Latency':<15} {'FPS':<8}")
    print("-" * 80)
    print(f"{Path(config_path).stem:<25} {f1_mean:<10.4f} {precision_mean:<10.4f} "
          f"{recall_mean:<10.4f} {latency_mean_ms:<15.1f} {fps:<8.2f}")

    return {
        'config_file': str(config_path),
        'f1_mean': f1_mean,
        'precision_mean': precision_mean,
        'recall_mean': recall_mean,
        'latency_mean_ms': latency_mean_ms,
        'fps': fps,
        'num_frames': len(metrics_list)
    }


def main():
    """Example usage"""
    # Example configuration
    config_path = "path/to/config.yaml"
    dataset_path = "path/to/phase1_frames_300"

    result = run_experiment(config_path, dataset_path, num_frames=300)
    print(f"\nResults: {result}")


if __name__ == '__main__':
    main()
