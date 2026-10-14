#!/usr/bin/env python3
"""
Timing and Latency Measurement Module
Tracks per-frame inference latency and FPS.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FrameTiming:
    """Per-frame timing information"""
    frame_id: int
    backend: str
    total_latency_ms: float  # Total time from input to output


class FrameTimer:
    """
    Tracks per-frame latency for performance analysis.

    Usage:
        timer = FrameTimer()
        for frame_idx, frame_data in enumerate(dataset):
            with timer.record_frame(frame_idx, 'nanosam'):
                masks = model.segment(frame_data['rgb'])
            latency = timer.frames[frame_idx].total_latency_ms
    """

    def __init__(self):
        """Initialize timer."""
        self.frames: Dict[int, FrameTiming] = {}
        self._current_frame_id: Optional[int] = None
        self._start_time: Optional[float] = None

    @contextmanager
    def record_frame(self, frame_id: int, backend: str):
        """
        Context manager to record frame latency.

        Args:
            frame_id: Frame index
            backend: Backend name (e.g., 'nanosam', 'vitb')

        Usage:
            with timer.record_frame(0, 'nanosam'):
                masks = model.segment(rgb)
        """
        self._current_frame_id = frame_id
        self._start_time = time.time()

        try:
            yield
        finally:
            elapsed_ms = (time.time() - self._start_time) * 1000

            self.frames[frame_id] = FrameTiming(
                frame_id=frame_id,
                backend=backend,
                total_latency_ms=elapsed_ms
            )

    def get_fps(self) -> float:
        """
        Compute mean FPS across recorded frames.

        Returns:
            FPS (frames per second)
        """
        if not self.frames:
            return 0.0

        mean_latency_ms = sum(f.total_latency_ms for f in self.frames.values()) / len(self.frames)
        return 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    def get_mean_latency_ms(self) -> float:
        """Get mean latency in milliseconds."""
        if not self.frames:
            return 0.0

        return sum(f.total_latency_ms for f in self.frames.values()) / len(self.frames)

    def get_stats(self) -> Dict:
        """Get timing statistics."""
        if not self.frames:
            return {}

        latencies = [f.total_latency_ms for f in self.frames.values()]
        return {
            'num_frames': len(latencies),
            'mean_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'fps': self.get_fps()
        }
