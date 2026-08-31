"""Minimal VLM testing diagnostics - crops and outputs."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import cv2


class VLMTestDiagnostics:
    """Log crops and VLM outputs for manual verification."""

    def __init__(self, output_dir: Path = None):
        """Initialize diagnostics.

        Args:
            output_dir: Root directory for test logs (default: VLM-Test-Session/)
        """
        if output_dir is None:
            output_dir = Path("/home/student/rsg_ros2_ws/VLM-Test-Session")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.crops_dir = self.output_dir / "crops"
        self.crops_dir.mkdir(exist_ok=True)

        # Output file
        self.log_file = self.output_dir / "vlm_results.csv"

        # Initialize CSV
        self._init_csv()

        self.object_counter = 0
        self.test_start_time = datetime.now().isoformat()

    def _init_csv(self) -> None:
        """Initialize CSV with headers matching pipeline output format."""
        headers = [
            "object_id",
            "crop_filename",
            "timestamp",
            "label",
            "label_confidence",
            "mobility_class",
            "mobility_confidence",
            "vlm_processing_time_ms",
            "success",
            "validation_status",
            "raw_response",
            "manual_label",
            "manual_is_correct",
            "manual_notes",
        ]
        with open(self.log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def log_vlm_result(
        self,
        crop_rgb: np.ndarray,
        vlm_output: Dict[str, Any],
        processing_time_ms: float,
        timestamp: float = None,
        track_id: str = None,
    ) -> Optional[str]:
        """Log VLM result with crop.

        Args:
            crop_rgb: RGB crop image
            vlm_output: Full VLM output dict (from pipeline)
            processing_time_ms: VLM inference time
            timestamp: Frame timestamp
            track_id: Optional track ID for reference

        Returns:
            Object ID if logged successfully
        """
        try:
            self.object_counter += 1
            object_id = f"{self.object_counter:06d}"

            # Save crop
            if crop_rgb is not None and crop_rgb.size > 0:
                crop_filename = f"obj_{object_id}_crop.jpg"
                crop_path = self.crops_dir / crop_filename

                # Convert RGB to BGR for OpenCV
                if len(crop_rgb.shape) == 3 and crop_rgb.shape[2] == 3:
                    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                else:
                    crop_bgr = crop_rgb

                cv2.imwrite(str(crop_path), crop_bgr)
            else:
                crop_filename = "no_crop"

            # Extract fields from pipeline output (keep exact format)
            label = vlm_output.get("label", "unknown_object")
            label_confidence = float(vlm_output.get("label_confidence", 0.0))
            mobility_class = vlm_output.get("mobility_class", "unknown")
            mobility_confidence = float(vlm_output.get("mobility_confidence", 0.0))
            success = vlm_output.get("success", False)
            validation_status = vlm_output.get("validation_status", "unknown")
            raw_response = vlm_output.get("raw_response", "")

            # Log to CSV
            row = {
                "object_id": object_id,
                "crop_filename": crop_filename,
                "timestamp": timestamp or time.time(),
                "label": label,
                "label_confidence": f"{label_confidence:.3f}",
                "mobility_class": mobility_class,
                "mobility_confidence": f"{mobility_confidence:.3f}",
                "vlm_processing_time_ms": f"{processing_time_ms:.2f}",
                "success": str(success),
                "validation_status": validation_status,
                "raw_response": raw_response[:100] if isinstance(raw_response, str) else str(raw_response)[:100],
                "manual_label": "",
                "manual_is_correct": "",
                "manual_notes": "",
            }

            with open(self.log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

            return object_id

        except Exception as e:
            print(f"Error logging VLM result: {e}")
            return None

    def get_session_dir(self) -> Path:
        """Get session directory."""
        return self.output_dir

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged results."""
        return {
            "session_dir": str(self.output_dir),
            "crops_dir": str(self.crops_dir),
            "results_file": str(self.log_file),
            "objects_logged": self.object_counter,
            "test_start_time": self.test_start_time,
        }
