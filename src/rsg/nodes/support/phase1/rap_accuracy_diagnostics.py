"""RAP result diagnostics -- crops and per-attempt CSV for manual accuracy verification.

Every RAP classification attempt (hit or miss) saves the exact crop RAP was
given plus one CSV row, mirroring vlm_test_diagnostics.py's layout and
manual-verification columns so the same review workflow applies to both.
Output layout::

    <output_dir>/session_<YYYYmmdd_HHMMSS>/
        crops/obj_NNNNNN_crop.jpg
        rap_results.csv

Logged unconditionally (hit or miss), not just successes -- a miss is
exactly the case worth being able to inspect too (was the nearest match
genuinely dissimilar, or a near-threshold false negative?), matching
risk_vlm_diagnostics.py's own "log every call" convention. A miss still
gets routed to VLM afterward and appears again in vlm_test_diagnostics.py's
own CSV, so the same track can show up in both files -- rap_results.csv is
the complete RAP-side record on its own, cross-checkable independently.

The session folder is unique per process start, so repeated runs never
overwrite an earlier run's crops or CSV.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CSV_HEADERS = [
    "object_id",
    "crop_filename",
    "track_id",
    "hydra_slot_id",
    "frame_timestamp",
    "outcome",
    "predicted_label",
    "distance",
    "confidence",
    "distance_threshold",
    "manual_label",
    "manual_is_correct",
    "manual_notes",
]


class RapAccuracyDiagnostics:
    """Log crops and outcomes for every RAP classification attempt, for manual verification."""

    def __init__(self, output_dir: Path = None, enabled: bool = True):
        """Initialize diagnostics.

        Args:
            output_dir: Parent directory; a fresh ``session_<timestamp>/`` is
                created under it.
            enabled: when False, no directory or CSV is created and
                ``log_rap_result`` is a no-op.
        """
        if output_dir is None:
            output_dir = Path("/home/student/rsg_ros2_ws/RAP-Accuracy-Test-Session")

        self.enabled = bool(enabled)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / f"session_{timestamp}"
        self.crops_dir = self.session_dir / "crops"
        self.log_file = self.session_dir / "rap_results.csv"
        self.object_counter = 0

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

    def log_rap_result(
        self,
        crop_rgb: Optional[np.ndarray],
        is_hit: bool,
        predicted_label: str,
        distance: float,
        confidence: float,
        distance_threshold: float,
        *,
        track_id: str = "",
        hydra_slot_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Save the crop this RAP attempt classified and append one result row.

        Called for every RAP classification attempt, hit or miss.

        Args:
            crop_rgb: exact RGB crop RAP classified (same crop used for the
                classify()/add_image() calls -- see rap_target_only in
                rsg_pipeline.yaml for what shape this crop is in).
            is_hit: whether this attempt was accepted as a known match.
            predicted_label, distance, confidence, distance_threshold: the
                RAP match this attempt was decided from (on a miss, this is
                still the nearest candidate found, even though it was
                rejected).
            track_id, hydra_slot_id: context this attempt was made for
                (reference only).
            timestamp: bag time of the classification.
        """
        if not self.enabled:
            return None
        try:
            self.object_counter += 1
            object_id = f"{self.object_counter:06d}"

            if crop_rgb is not None and getattr(crop_rgb, "size", 0) > 0:
                crop_filename = f"obj_{object_id}_crop.jpg"
                crop = crop_rgb
                if crop.ndim == 3 and crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(self.crops_dir / crop_filename), crop)
            else:
                crop_filename = "no_crop"

            row = {
                "object_id": object_id,
                "crop_filename": crop_filename,
                "track_id": track_id,
                "hydra_slot_id": int(hydra_slot_id or 0),
                "frame_timestamp": timestamp if timestamp is not None else time.time(),
                "outcome": "hit" if is_hit else "miss",
                "predicted_label": predicted_label,
                "distance": f"{float(distance):.4f}",
                "confidence": f"{float(confidence):.3f}",
                "distance_threshold": f"{float(distance_threshold):.4f}",
                "manual_label": "",
                "manual_is_correct": "",
                "manual_notes": "",
            }
            with open(self.log_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)
            return object_id
        except Exception as e:  # noqa: BLE001 - diagnostics must never break the pipeline
            print(f"Error logging RAP result: {e}")
            return None

    def get_session_dir(self) -> Path:
        return self.session_dir
