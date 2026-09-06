"""Periodic per-track crop diagnostics -- raw crop + real 3D geometry at a
fixed observation interval, to trace how a track's mask and bounding box
evolve over time (e.g. as the robot approaches an object).

Unlike tracking_crop_manager.py's "best crop" saves (only the single
highest-scoring revision is kept), this samples every Nth *observation* of
a track unconditionally, so a merge investigation can see the full
sequence rather than just the winning snapshot.

Each row logs two different things side by side, which is the whole point:
  - the RAW per-observation geometry for *this exact frame's* mask
    (centroid_3d / bbox_3d_min / bbox_3d_max / bbox_volume_m3, as computed
    fresh by ObjectGeometryEstimator for this one mask), and
  - the ACCUMULATED envelope of the local Hydra segment this observation
    was just merged into (local_segment_centroid_3d / bbox_3d_min/max),
    which is what actually gets published -- and which, being built from
    element-wise min/max across every observation ever assigned to it,
    only ever grows and never shrinks.

Comparing the two columns for the same row answers the question that
motivated this diagnostic: does a later frame's *raw* mask itself already
include the other object (a segmentation-time issue), or does the raw mask
stay clean while the *accumulated* envelope quietly creeps outward anyway
(an accumulation-time issue)?

Output layout::

    <output_dir>/session_<YYYYmmdd_HHMMSS>/
        crops/<track_id>_obsNNNNN_seqNNNNNN.jpg
        periodic_observations.csv
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

CSV_HEADERS = [
    "track_id",
    "observation_index",
    "sequence",
    "frame_timestamp",
    "crop_filename",
    "mask_area_px",
    "valid_geometry",
    "raw_centroid_3d_x", "raw_centroid_3d_y", "raw_centroid_3d_z",
    "raw_bbox_3d_min_x", "raw_bbox_3d_min_y", "raw_bbox_3d_min_z",
    "raw_bbox_3d_max_x", "raw_bbox_3d_max_y", "raw_bbox_3d_max_z",
    "raw_bbox_volume_m3",
    "segment_seen_count",
    "segment_centroid_3d_x", "segment_centroid_3d_y", "segment_centroid_3d_z",
    "segment_bbox_3d_min_x", "segment_bbox_3d_min_y", "segment_bbox_3d_min_z",
    "segment_bbox_3d_max_x", "segment_bbox_3d_max_y", "segment_bbox_3d_max_z",
    "segment_xy_span_m",
]


def _xyz(value: Any) -> list:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return [value[0], value[1], value[2]]
    return [None, None, None]


class PeriodicCropDiagnostics:
    """Save every Nth observation of every track -- crop + raw and accumulated geometry."""

    def __init__(self, output_dir: Path = None, enabled: bool = True, interval: int = 10):
        """Initialize diagnostics.

        Args:
            output_dir: Parent directory; a fresh ``session_<timestamp>/`` is
                created under it.
            enabled: when False, no directory or CSV is created and
                ``log_observation`` is a no-op.
            interval: save 1 observation out of every this-many, counted
                per track (not globally) -- e.g. 10 means the 1st, 11th,
                21st, ... observation of each track is saved.
        """
        if output_dir is None:
            output_dir = Path("/home/student/rsg_ros2_ws/Periodic-Crop-Diagnostics")

        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / f"session_{timestamp}"
        self.crops_dir = self.session_dir / "crops"
        self.log_file = self.session_dir / "periodic_observations.csv"
        self._track_counts: Dict[str, int] = {}

        if self.enabled:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.crops_dir.mkdir(exist_ok=True)
            self._init_csv()

    def _init_csv(self) -> None:
        """Write the header once; never truncate an existing file."""
        if self.log_file.exists() and self.log_file.stat().st_size > 0:
            return
        with open(self.log_file, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()

    def log_observation(
        self,
        track_id: str,
        sequence: int,
        rgb: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        metadata: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Called for every observation of every track; saves only 1-in-N.

        Args:
            track_id: persistent track id this observation was assigned to.
            sequence: frame sequence number.
            rgb: full-frame RGB image (not yet cropped).
            mask: this observation's own 2D boolean SAM mask, in full-frame
                coordinates -- drawn as a contour on the saved crop so the
                mask's actual boundary is directly visible, independent of
                whatever crop the "best crop" selector picked.
            metadata: the object metadata dict for this observation, after
                PersistentObjectTracker.associate() has annotated it -- read
                for both the raw per-frame geometry (bbox_3d_min/max,
                centroid_3d, bbox_volume_m3, mask_area_px, valid_geometry)
                and the accumulated local-segment envelope
                (local_segment_bbox_3d_min/max, local_segment_centroid_3d,
                local_segment_seen_count, local_segment_xy_span_m).
            timestamp: bag time of this observation.
        """
        if not self.enabled:
            return None
        try:
            key = str(track_id)
            count = self._track_counts.get(key, 0)
            self._track_counts[key] = count + 1
            if count % self.interval != 0:
                return None

            crop_filename = "no_crop"
            if rgb is not None and mask is not None and getattr(mask, "size", 0) > 0 and bool(np.any(mask)):
                ys, xs = np.where(mask)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                pad = 20
                h, w = rgb.shape[:2]
                y0p, y1p = max(0, y0 - pad), min(h, y1 + pad)
                x0p, x1p = max(0, x0 - pad), min(w, x1 + pad)
                crop = np.array(rgb[y0p:y1p, x0p:x1p], dtype=np.uint8)
                local_mask = mask[y0p:y1p, x0p:x1p].astype(np.uint8)
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(crop_bgr, contours, -1, (255, 255, 0), 2)
                crop_filename = f"{key}_obs{count:05d}_seq{int(sequence):06d}.jpg"
                cv2.imwrite(str(self.crops_dir / crop_filename), crop_bgr)

            raw_c = _xyz(metadata.get("centroid_3d"))
            raw_min = _xyz(metadata.get("bbox_3d_min"))
            raw_max = _xyz(metadata.get("bbox_3d_max"))
            seg_c = _xyz(metadata.get("local_segment_centroid_3d"))
            seg_min = _xyz(metadata.get("local_segment_bbox_3d_min"))
            seg_max = _xyz(metadata.get("local_segment_bbox_3d_max"))

            row = {
                "track_id": key,
                "observation_index": count,
                "sequence": int(sequence),
                "frame_timestamp": timestamp if timestamp is not None else time.time(),
                "crop_filename": crop_filename,
                "mask_area_px": metadata.get("mask_area_px", ""),
                "valid_geometry": metadata.get("valid_geometry", ""),
                "raw_centroid_3d_x": raw_c[0], "raw_centroid_3d_y": raw_c[1], "raw_centroid_3d_z": raw_c[2],
                "raw_bbox_3d_min_x": raw_min[0], "raw_bbox_3d_min_y": raw_min[1], "raw_bbox_3d_min_z": raw_min[2],
                "raw_bbox_3d_max_x": raw_max[0], "raw_bbox_3d_max_y": raw_max[1], "raw_bbox_3d_max_z": raw_max[2],
                "raw_bbox_volume_m3": metadata.get("bbox_volume_m3", ""),
                "segment_seen_count": metadata.get("local_segment_seen_count", ""),
                "segment_centroid_3d_x": seg_c[0], "segment_centroid_3d_y": seg_c[1], "segment_centroid_3d_z": seg_c[2],
                "segment_bbox_3d_min_x": seg_min[0], "segment_bbox_3d_min_y": seg_min[1], "segment_bbox_3d_min_z": seg_min[2],
                "segment_bbox_3d_max_x": seg_max[0], "segment_bbox_3d_max_y": seg_max[1], "segment_bbox_3d_max_z": seg_max[2],
                "segment_xy_span_m": metadata.get("local_segment_xy_span_m", ""),
            }
            with open(self.log_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)
            return crop_filename
        except Exception as e:  # noqa: BLE001 - diagnostics must never break the pipeline
            print(f"Error logging periodic crop diagnostics: {e}")
            return None

    def get_session_dir(self) -> Path:
        return self.session_dir
