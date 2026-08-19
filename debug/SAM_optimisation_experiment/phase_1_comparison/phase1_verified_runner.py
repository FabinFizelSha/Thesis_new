#!/usr/bin/env python3
"""Phase 1 Verified Test Runner - Ensures all requested frames are processed.

CRITICAL: Verifies that:
1. Dataset has at least 300 frames available
2. All 300 frames are processed (none skipped)
3. Results are recorded for all 300 frames
4. Reports any missing or skipped frames
"""

import argparse
import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import yaml

from timing import FrameTimer
from ground_truth import GroundTruthEvaluator
from recorder import MetricsRecorder
from runners import get_backend_runner
from comparison_runner import Phase1DatasetLoader


class VerifiedPhase1Runner:
    """Phase 1 test runner with frame verification."""

    def __init__(self, dataset_dir: str, config_dir: str, output_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verify dataset
        self._verify_dataset()

    def _verify_dataset(self):
        """Verify dataset has frames available."""
        print("\n" + "="*80)
        print("VERIFYING DATASET")
        print("="*80 + "\n")

        loader = Phase1DatasetLoader(str(self.dataset_dir), random_shuffle=False)
        frame_count = 0
        for _ in loader:
            frame_count += 1

        print(f"✓ Dataset location: {self.dataset_dir}")
        print(f"✓ Total frames available: {frame_count}\n")

        if frame_count < 300:
            print(f"⚠️  WARNING: Dataset has only {frame_count} frames (expected 300)")
            print(f"   Proceeding with {frame_count} frames\n")
            self.actual_frames = frame_count
        else:
            print(f"✓ Dataset has sufficient frames (>= 300)\n")
            self.actual_frames = frame_count

    def run_verified_test(self, backend: str, level: str, max_frames: int = None):
        """Run test with frame verification.

        Args:
            backend: Backend name
            level: Complexity level
            max_frames: Max frames to process (None = all)

        Returns:
            Number of frames actually processed
        """
        print("="*80)
        print(f"Running: {backend.upper()} - {level.upper()}")
        print("="*80 + "\n")

        # Determine how many frames to process
        target_frames = max_frames if max_frames else self.actual_frames
        print(f"Target: {target_frames} frames")
        print(f"Available: {self.actual_frames} frames\n")

        # Load config
        config_file = self.config_dir / f"{backend}_{level}.yaml"
        if not config_file.exists():
            print(f"❌ Config not found: {config_file}")
            return 0

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Load dataset
        loader = Phase1DatasetLoader(str(self.dataset_dir), random_shuffle=False)
        frames = []
        for frame_data in loader:
            frames.append(frame_data)
            if len(frames) >= target_frames:
                break

        actual_frames_loaded = len(frames)
        print(f"Loaded: {actual_frames_loaded} frames")

        if actual_frames_loaded < target_frames:
            print(f"⚠️  WARNING: Loaded only {actual_frames_loaded}/{target_frames} frames\n")
        else:
            print(f"✓ Loaded all {target_frames} frames\n")

        # Initialize components
        runner = get_backend_runner(backend, config)
        evaluator = GroundTruthEvaluator(iou_threshold=0.3)
        timer = FrameTimer()
        recorder = MetricsRecorder()

        # Process frames with verification
        processed_count = 0
        skipped_frames = []

        print("Processing frames...\n")

        for frame_idx, frame_data in enumerate(frames):
            frame_id = frame_data.get('frame_id', frame_idx)

            try:
                rgb = frame_data['rgb']
                depth = frame_data['depth']
                semantic_gt = frame_data['semantic']

                # Convert semantic to single channel if needed
                if semantic_gt.ndim == 3 and semantic_gt.shape[2] == 3:
                    semantic_gt = semantic_gt.mean(axis=2).astype(np.uint8)

                # Apply depth filtering - match RSG pipeline configuration
                # Valid depth range: 0.3m to 6.0m (from rsg_pipeline.yaml)
                depth_valid = (depth >= 0.3) & (depth <= 6.0)

                # Run inference
                with timer.record_frame(frame_id, backend) as frame:
                    with frame.record_stage("inference"):
                        masks = runner.segment(rgb, depth)

                latency_ms = timer.frames[-1].total_latency_ms

                # Evaluate - pass depth_valid mask for filtering both masks and GT
                metrics = evaluator.compute_metrics(masks, semantic_gt, depth_valid=depth_valid)

                # Record
                recorder.record_frame(
                    frame_id=frame_id,
                    backend=backend,
                    level=level,
                    timestamp_sec=time.time(),
                    latency_ms=latency_ms,
                    metrics={
                        'num_detected': metrics.num_detected,
                        'num_accepted': metrics.num_accepted,
                        'precision': metrics.precision,
                        'recall': metrics.recall,
                        'f1_score': metrics.f1_score,
                        'avg_iou': metrics.avg_iou,
                        'min_iou': metrics.min_iou,
                        'max_iou': metrics.max_iou,
                    }
                )

                processed_count += 1

                if processed_count % 50 == 0:
                    print(f"  Frame {frame_idx+1}/{actual_frames_loaded}: "
                          f"F1={metrics.f1_score:.3f}, "
                          f"latency={latency_ms:.1f}ms")

            except Exception as e:
                print(f"  ❌ Frame {frame_idx}: {e}")
                skipped_frames.append(frame_idx)
                continue

        # Save results
        output_subdir = self.output_dir / f"{backend}_{level}"
        output_subdir.mkdir(exist_ok=True)

        recorder.export_csv(str(output_subdir / "metrics.csv"))

        # Generate report
        report = {
            'backend': backend,
            'level': level,
            'target_frames': target_frames,
            'loaded_frames': actual_frames_loaded,
            'processed_frames': processed_count,
            'skipped_frames': len(skipped_frames),
            'skipped_frame_indices': skipped_frames,
            'verification': {
                'all_frames_processed': (processed_count == actual_frames_loaded),
                'no_skipped_frames': (len(skipped_frames) == 0),
            }
        }

        with open(output_subdir / "frame_verification.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*80}")
        print("FRAME VERIFICATION REPORT")
        print(f"{'='*80}\n")
        print(f"Target frames: {target_frames}")
        print(f"Loaded frames: {actual_frames_loaded}")
        print(f"Processed frames: {processed_count}")
        print(f"Skipped frames: {len(skipped_frames)}")

        if processed_count == target_frames:
            print(f"\n✅ VERIFIED: All {target_frames} frames processed successfully")
        else:
            print(f"\n❌ VERIFICATION FAILED")
            print(f"   Expected: {target_frames} frames")
            print(f"   Processed: {processed_count} frames")
            print(f"   Missing: {target_frames - processed_count} frames")

        if skipped_frames:
            print(f"\n⚠️  Skipped frames: {skipped_frames}")

        return processed_count

    def run_all(self, backends: List[str], levels: List[str], max_frames: int = None):
        """Run all tests with verification."""
        print("\n" + "="*80)
        print("PHASE 1 VERIFIED COMPARISON TEST")
        print("="*80)

        for backend in backends:
            for level in levels:
                self.run_verified_test(backend, level, max_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 1 Verified Test Runner')
    parser.add_argument('--dataset', default='datasets/phase1_frames',
                        help='Dataset directory')
    parser.add_argument('--config-dir', default='configs',
                        help='Config directory')
    parser.add_argument('--output', default='results/phase1_verified',
                        help='Output directory')
    parser.add_argument('--backends', nargs='+', default=['nanosam', 'vitb'],
                        help='Backends to test')
    parser.add_argument('--levels', nargs='+', default=['strict', 'medium', 'loose'],
                        help='Levels to test')
    parser.add_argument('--max-frames', type=int, default=300,
                        help='Max frames to process (0 = all available)')

    args = parser.parse_args()

    runner = VerifiedPhase1Runner(args.dataset, args.config_dir, args.output)
    runner.run_all(args.backends, args.levels, args.max_frames if args.max_frames > 0 else None)
