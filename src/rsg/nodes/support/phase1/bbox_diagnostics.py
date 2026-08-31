"""Bounding box diagnostics logger for track analysis.

Captures all track metadata during pipeline execution and saves to a JSON Lines
file for post-run analysis by Claude or other tools. Each line is one frame's
worth of track observations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class BboxDiagnosticsLogger:
    """Log all track bounding boxes and related metadata during pipeline runs.

    Usage:
        logger = BboxDiagnosticsLogger(enabled=True)

        # During frame processing:
        for track_id, metadata in tracks.items():
            logger.log_track(frame_number=123, track_id=track_id, metadata=metadata)

        # On shutdown:
        logger.save()
    """

    def __init__(self, enabled: bool = True, output_dir: Optional[str] = None):
        self.enabled = enabled
        self.frames: List[Dict[str, Any]] = []
        self.output_dir = output_dir or os.path.expanduser("~/rsg_ros2_ws/debug/bbox_diagnostics")
        self.frame_count = 0
        self.track_count_per_frame: Dict[int, int] = {}

        if self.enabled:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def log_frame_tracks(self, frame_number: int, tracks: Dict[str, Dict[str, Any]]) -> None:
        """Log all tracks in a single frame.

        Args:
            frame_number: Frame index (0-based or 1-based, as long as consistent).
            tracks: Dict mapping track_id -> metadata dict containing:
                - centroid_3d: [x, y, z]
                - bbox_3d_min: [x_min, y_min, z_min]
                - bbox_3d_max: [x_max, y_max, z_max]
                - bbox_3d_size: [width, depth, height]
                - bbox_volume_m3: float
                - depth_min_m, depth_max_m, depth_median_m: float
                - (optional) other metadata
        """
        if not self.enabled or not tracks:
            return

        self.frame_count = frame_number
        self.track_count_per_frame[frame_number] = len(tracks)

        for track_id, metadata in tracks.items():
            frame_record = {
                "frame_number": frame_number,
                "track_id": track_id,
                "centroid_3d": metadata.get("centroid_3d"),
                "bbox_3d_min": metadata.get("bbox_3d_min"),
                "bbox_3d_max": metadata.get("bbox_3d_max"),
                "bbox_3d_size": metadata.get("bbox_3d_size"),
                "bbox_volume_m3": metadata.get("bbox_volume_m3"),
                "depth_min_m": metadata.get("depth_min_m"),
                "depth_max_m": metadata.get("depth_max_m"),
                "depth_median_m": metadata.get("depth_median_m"),
                "depth_valid_ratio": metadata.get("depth_valid_ratio"),
                "mask_area_px": metadata.get("mask_area_px"),
            }
            self.frames.append(frame_record)

    def save(self) -> Optional[str]:
        """Save logged frames to JSON Lines file with timestamp.

        Returns:
            Path to saved file, or None if logging disabled or no data.
        """
        if not self.enabled or not self.frames:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bbox_log_{timestamp}.jsonl"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, "w") as f:
                for frame_record in self.frames:
                    f.write(json.dumps(frame_record) + "\n")

            print(f"", flush=True)
            print(f"=== Bounding Box Diagnostics Complete ===", flush=True)
            print(f"Total frames processed: {self.frame_count + 1}", flush=True)
            print(f"Total track observations: {len(self.frames)}", flush=True)
            print(f"Max tracks per frame: {max(self.track_count_per_frame.values()) if self.track_count_per_frame else 0}", flush=True)
            print(f"Avg tracks per frame: {len(self.frames) / (self.frame_count + 1) if self.frame_count >= 0 else 0:.1f}", flush=True)
            print(f"Output file: {filepath}", flush=True)
            print(f"Format: JSON Lines (one track observation per line)", flush=True)

            return filepath
        except Exception as exc:
            print(f"Failed to save bbox diagnostics: {exc}", flush=True)
            return None