"""Metrics recording and export module for Phase 1 comparison.

Collects per-frame metrics and exports to CSV for analysis.

Usage:
    recorder = MetricsRecorder()

    recorder.record_frame(
        frame_id=0,
        backend="nanosam",
        level="strict",
        latency_ms=32.5,
        metrics={...}
    )

    recorder.export_csv("results/comparison.csv")
"""

import csv
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class FrameRecord:
    """Complete record for one frame."""
    frame_id: int
    backend: str
    level: str
    timestamp_sec: float
    latency_ms: float
    num_detected: int
    num_accepted: int
    precision: float
    recall: float
    f1_score: float
    avg_iou: float
    min_iou: float
    max_iou: float


class MetricsRecorder:
    """Record and export frame-level metrics."""

    def __init__(self):
        """Initialize recorder."""
        self.records: List[FrameRecord] = []

    def record_frame(
        self,
        frame_id: int,
        backend: str,
        level: str,
        timestamp_sec: float,
        latency_ms: float,
        metrics: Dict[str, Any],
    ) -> None:
        """Record metrics for one frame.

        Args:
            frame_id: Frame identifier
            backend: Backend name ("nanosam" or "vitb")
            level: Complexity level ("strict", "medium", "loose")
            timestamp_sec: Frame timestamp
            latency_ms: Inference latency
            metrics: Dict with precision, recall, f1_score, avg_iou, min_iou, max_iou,
                    num_detected, num_accepted
        """
        record = FrameRecord(
            frame_id=frame_id,
            backend=backend,
            level=level,
            timestamp_sec=timestamp_sec,
            latency_ms=latency_ms,
            num_detected=metrics.get('num_detected', 0),
            num_accepted=metrics.get('num_accepted', 0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1_score=metrics.get('f1_score', 0.0),
            avg_iou=metrics.get('avg_iou', 0.0),
            min_iou=metrics.get('min_iou', 0.0),
            max_iou=metrics.get('max_iou', 0.0),
        )
        self.records.append(record)

    def export_csv(self, output_path: str) -> None:
        """Export all records to CSV.

        Args:
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'frame_id', 'backend', 'level', 'timestamp_sec',
            'latency_ms', 'num_detected', 'num_accepted',
            'precision', 'recall', 'f1_score',
            'avg_iou', 'min_iou', 'max_iou'
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in self.records:
                writer.writerow({
                    'frame_id': record.frame_id,
                    'backend': record.backend,
                    'level': record.level,
                    'timestamp_sec': f"{record.timestamp_sec:.6f}",
                    'latency_ms': f"{record.latency_ms:.2f}",
                    'num_detected': record.num_detected,
                    'num_accepted': record.num_accepted,
                    'precision': f"{record.precision:.4f}",
                    'recall': f"{record.recall:.4f}",
                    'f1_score': f"{record.f1_score:.4f}",
                    'avg_iou': f"{record.avg_iou:.4f}",
                    'min_iou': f"{record.min_iou:.4f}",
                    'max_iou': f"{record.max_iou:.4f}",
                })

    def get_summary(self) -> Dict[str, Dict]:
        """Get summary statistics by backend and level.

        Returns:
            Dict with per-backend and per-level aggregates
        """
        import statistics

        summary = {}

        # Group by backend and level
        groups = {}
        for record in self.records:
            key = (record.backend, record.level)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        # Compute stats
        for (backend, level), group in sorted(groups.items()):
            latencies = [r.latency_ms for r in group]
            f1_scores = [r.f1_score for r in group]

            key = f"{backend}_{level}"
            summary[key] = {
                'count': len(group),
                'latency_mean_ms': statistics.mean(latencies),
                'latency_p95_ms': sorted(latencies)[int(len(latencies) * 0.95)],
                'f1_mean': statistics.mean(f1_scores),
                'f1_stdev': statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
            }

        return summary
