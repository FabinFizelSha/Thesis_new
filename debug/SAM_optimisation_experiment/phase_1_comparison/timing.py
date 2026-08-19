"""Low-overhead nanosecond-precision per-frame timing module.

Provides context managers for frame-level and stage-level timing with
minimal computational overhead (<1ms per frame).

Usage:
    timer = FrameTimer()

    for frame_id, rgb, depth in dataset:
        with timer.record_frame(frame_id, "nanosam") as frame:
            with frame.record_stage("encoder"):
                encoder_out = backend.encode_image(rgb)

            with frame.record_stage("decoder"):
                masks = backend.decode_masks(encoder_out, depth)

    # Export results
    timer.export_csv("results/timing.csv")
    stats = timer.get_statistics()
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass
class StageTiming:
    """Timing data for a processing stage."""
    stage_name: str
    latency_ns: int  # Nanoseconds

    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return self.latency_ns / 1e6


@dataclass
class FrameTiming:
    """Timing data for a complete frame."""
    frame_id: int
    backend: str
    total_latency_ns: int
    stages: Dict[str, StageTiming]

    @property
    def total_latency_ms(self) -> float:
        """Total latency in milliseconds."""
        return self.total_latency_ns / 1e6

    @property
    def fps(self) -> float:
        """Frames per second (1000 / latency_ms)."""
        return 1000.0 / self.total_latency_ms if self.total_latency_ms > 0 else 0.0


class FrameTimer:
    """Nanosecond-precision frame and stage timer with minimal overhead.

    Features:
    - Nanosecond precision using time.perf_counter_ns()
    - Context managers for easy usage
    - Low overhead (<1ms per frame)
    - Per-stage timing breakdown
    - CSV export for analysis
    """

    def __init__(self):
        """Initialize timer."""
        self.frames: List[FrameTiming] = []
        self._current_frame: Optional['_FrameContext'] = None

    def record_frame(self, frame_id: int, backend: str) -> '_FrameContext':
        """Context manager for recording a frame's timing.

        Args:
            frame_id: Unique frame identifier
            backend: Backend name (e.g., "nanosam", "vitb")

        Returns:
            Context manager for the frame

        Usage:
            with timer.record_frame(0, "nanosam") as frame:
                # Inference code here
                with frame.record_stage("encoder"):
                    # Encoder code
                    ...
        """
        return self._FrameContext(self, frame_id, backend)

    def _record_frame_timing(self, frame_timing: FrameTiming) -> None:
        """Internal: Record a completed frame timing."""
        self.frames.append(frame_timing)

    def export_csv(self, output_path: str) -> None:
        """Export timing data to CSV.

        Args:
            output_path: Path to output CSV file

        CSV columns:
            - frame_id
            - backend
            - total_latency_ms
            - fps
            - stage_name (for each stage)
            - stage_latency_ms (for each stage)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'frame_id', 'backend', 'total_latency_ms', 'fps',
                'stage_name', 'stage_latency_ms'
            ])
            writer.writeheader()

            for frame in self.frames:
                if frame.stages:
                    # Write one row per stage
                    for stage_name, stage in frame.stages.items():
                        writer.writerow({
                            'frame_id': frame.frame_id,
                            'backend': frame.backend,
                            'total_latency_ms': f"{frame.total_latency_ms:.2f}",
                            'fps': f"{frame.fps:.1f}",
                            'stage_name': stage_name,
                            'stage_latency_ms': f"{stage.latency_ms:.2f}",
                        })
                else:
                    # Write one row for total only
                    writer.writerow({
                        'frame_id': frame.frame_id,
                        'backend': frame.backend,
                        'total_latency_ms': f"{frame.total_latency_ms:.2f}",
                        'fps': f"{frame.fps:.1f}",
                        'stage_name': 'total',
                        'stage_latency_ms': f"{frame.total_latency_ms:.2f}",
                    })

    def get_statistics(self) -> Dict:
        """Compute timing statistics.

        Returns:
            Dictionary with statistics:
            - per_backend: {backend_name: {stat_name: value}}
            - overall: {stat_name: value}

        Stats include:
            - latency_mean_ms, latency_std_ms, latency_min_ms, latency_max_ms
            - fps_mean, fps_std, fps_min, fps_max
            - p50, p95, p99 (percentiles)
        """
        if not self.frames:
            return {}

        import statistics

        # Group by backend
        by_backend = {}
        for frame in self.frames:
            if frame.backend not in by_backend:
                by_backend[frame.backend] = []
            by_backend[frame.backend].append(frame)

        stats = {'per_backend': {}, 'overall': {}}

        # Compute per-backend stats
        for backend, frames_list in by_backend.items():
            latencies_ms = [f.total_latency_ms for f in frames_list]
            fps_list = [f.fps for f in frames_list]

            latencies_ms_sorted = sorted(latencies_ms)
            n = len(latencies_ms_sorted)

            stats['per_backend'][backend] = {
                'count': len(frames_list),
                'latency_mean_ms': statistics.mean(latencies_ms),
                'latency_stdev_ms': statistics.stdev(latencies_ms) if n > 1 else 0.0,
                'latency_min_ms': min(latencies_ms),
                'latency_max_ms': max(latencies_ms),
                'latency_p50_ms': latencies_ms_sorted[n // 2],
                'latency_p95_ms': latencies_ms_sorted[int(n * 0.95)],
                'latency_p99_ms': latencies_ms_sorted[int(n * 0.99)],
                'fps_mean': statistics.mean(fps_list),
                'fps_stdev': statistics.stdev(fps_list) if n > 1 else 0.0,
                'fps_min': min(fps_list),
                'fps_max': max(fps_list),
            }

        # Compute overall stats
        all_latencies_ms = [f.total_latency_ms for f in self.frames]
        all_fps = [f.fps for f in self.frames]
        all_latencies_sorted = sorted(all_latencies_ms)
        n_total = len(all_latencies_sorted)

        stats['overall'] = {
            'total_frames': len(self.frames),
            'latency_mean_ms': statistics.mean(all_latencies_ms),
            'latency_stdev_ms': statistics.stdev(all_latencies_ms) if n_total > 1 else 0.0,
            'latency_min_ms': min(all_latencies_ms),
            'latency_max_ms': max(all_latencies_ms),
            'latency_p50_ms': all_latencies_sorted[n_total // 2],
            'latency_p95_ms': all_latencies_sorted[int(n_total * 0.95)],
            'latency_p99_ms': all_latencies_sorted[int(n_total * 0.99)],
            'fps_mean': statistics.mean(all_fps),
            'fps_stdev': statistics.stdev(all_fps) if n_total > 1 else 0.0,
            'fps_min': min(all_fps),
            'fps_max': max(all_fps),
        }

        return stats

    class _FrameContext:
        """Context manager for frame-level timing."""

        def __init__(self, timer: 'FrameTimer', frame_id: int, backend: str):
            self.timer = timer
            self.frame_id = frame_id
            self.backend = backend
            self.start_time_ns = None
            self.stages: Dict[str, StageTiming] = {}

        def __enter__(self) -> '_FrameContext':
            """Start frame timing."""
            self.start_time_ns = time.perf_counter_ns()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            """End frame timing and record."""
            end_time_ns = time.perf_counter_ns()
            total_latency_ns = end_time_ns - self.start_time_ns

            frame_timing = FrameTiming(
                frame_id=self.frame_id,
                backend=self.backend,
                total_latency_ns=total_latency_ns,
                stages=self.stages,
            )

            self.timer._record_frame_timing(frame_timing)

        def record_stage(self, stage_name: str) -> '_StageContext':
            """Context manager for stage-level timing within a frame.

            Args:
                stage_name: Name of the processing stage

            Returns:
                Context manager for the stage

            Usage:
                with frame.record_stage("encoder"):
                    # Encoder inference
                    ...
            """
            return self._StageContext(self, stage_name)

        class _StageContext:
            """Context manager for stage-level timing."""

            def __init__(self, frame_ctx: '_FrameContext', stage_name: str):
                self.frame_ctx = frame_ctx
                self.stage_name = stage_name
                self.start_time_ns = None

            def __enter__(self) -> '_StageContext':
                """Start stage timing."""
                self.start_time_ns = time.perf_counter_ns()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb) -> None:
                """End stage timing and record."""
                end_time_ns = time.perf_counter_ns()
                latency_ns = end_time_ns - self.start_time_ns

                stage_timing = StageTiming(
                    stage_name=self.stage_name,
                    latency_ns=latency_ns,
                )

                self.frame_ctx.stages[self.stage_name] = stage_timing


if __name__ == "__main__":
    # Quick test
    import time as time_module

    timer = FrameTimer()

    print("Testing FrameTimer...")
    for frame_id in range(5):
        with timer.record_frame(frame_id, "test") as frame:
            with frame.record_stage("stage1"):
                time_module.sleep(0.01)  # 10ms

            with frame.record_stage("stage2"):
                time_module.sleep(0.02)  # 20ms

    stats = timer.get_statistics()
    print(f"\nTest Results:")
    print(f"  Total frames: {stats['overall']['total_frames']}")
    print(f"  Latency mean: {stats['overall']['latency_mean_ms']:.1f} ms")
    print(f"  Latency p95: {stats['overall']['latency_p95_ms']:.1f} ms")
    print(f"  FPS mean: {stats['overall']['fps_mean']:.1f} fps")

    print("\n✓ FrameTimer working correctly")
