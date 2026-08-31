"""VLM prompt optimization diagnostic logging system."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import cv2


class VLMPromptLogger:
    """Log VLM outputs with timing and crop images for prompt optimization."""

    def __init__(self, output_dir: Path, prompt_version: str = "v1_production"):
        """Initialize VLM prompt logger.

        Args:
            output_dir: Root directory for optimization logs
            prompt_version: Current prompt version being tested
        """
        self.output_dir = Path(output_dir)
        self.prompt_version = prompt_version

        # Session folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"experiment_{timestamp}__{prompt_version}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.crops_dir = self.session_dir / "crops"
        self.crops_dir.mkdir(exist_ok=True)

        # Output files
        self.vlm_outputs_file = self.session_dir / "vlm_outputs.jsonl"
        self.verification_file = self.session_dir / "verification_results.csv"

        # Initialize CSV
        self._init_csv()

        # Track crop counter
        self.crop_counter = 0
        self.output_counter = 0

    def _init_csv(self) -> None:
        """Initialize verification CSV with headers."""
        headers = [
            "object_id",
            "timestamp",
            "crop_filename",
            "vlm_class",
            "vlm_confidence",
            "vlm_processing_time_ms",
            "prompt_version",
            "manual_verified",
            "manual_class",
            "manual_confidence",
            "is_correct",
            "verified_by",
            "verified_at",
            "notes",
        ]
        with open(self.verification_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def save_crop(
        self,
        crop_rgb: np.ndarray,
        object_id: str,
        track_id: str,
    ) -> Optional[str]:
        """Save crop image for later manual verification.

        Args:
            crop_rgb: RGB crop image
            object_id: Unique object identifier
            track_id: Track ID for reference

        Returns:
            Filename (without path) if saved, None if failed
        """
        try:
            if crop_rgb is None or crop_rgb.size == 0:
                return None

            self.crop_counter += 1
            filename = f"obj_{object_id:06d}_track_{track_id}_crop_{self.crop_counter:06d}.jpg"
            filepath = self.crops_dir / filename

            # Convert RGB to BGR for OpenCV
            if len(crop_rgb.shape) == 3 and crop_rgb.shape[2] == 3:
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = crop_rgb

            cv2.imwrite(str(filepath), crop_bgr)
            return filename
        except Exception:
            return None

    def log_vlm_output(
        self,
        object_id: str,
        track_id: str,
        crop_filename: str,
        vlm_output: Dict[str, Any],
        processing_time_ms: float,
        timestamp: float,
    ) -> Optional[str]:
        """Log VLM output with timing information.

        Args:
            object_id: Unique object identifier
            track_id: Track ID for reference
            crop_filename: Saved crop filename
            vlm_output: VLM response (dict with: class, confidence, reasoning, etc.)
            processing_time_ms: Time taken for VLM inference
            timestamp: Frame timestamp (seconds)

        Returns:
            Output ID if logged, None if failed
        """
        try:
            self.output_counter += 1
            output_id = f"{object_id:06d}_{self.output_counter:06d}"

            # Ensure vlm_output is dict
            if isinstance(vlm_output, str):
                try:
                    vlm_output = json.loads(vlm_output)
                except json.JSONDecodeError:
                    vlm_output = {"raw_output": vlm_output}

            # Extract standard fields (with defaults)
            vlm_class = vlm_output.get("class", "unknown")
            vlm_confidence = float(vlm_output.get("confidence", 0.0))

            # Ensure confidence is 0-1
            if vlm_confidence > 1.0:
                vlm_confidence = vlm_confidence / 100.0

            # Log to JSONL
            record = {
                "output_id": output_id,
                "object_id": str(object_id),
                "track_id": str(track_id),
                "timestamp": float(timestamp),
                "crop_filename": str(crop_filename),
                "vlm_class": str(vlm_class),
                "vlm_confidence": float(vlm_confidence),
                "vlm_processing_time_ms": float(processing_time_ms),
                "prompt_version": str(self.prompt_version),
                "vlm_full_output": vlm_output,
            }

            with open(self.vlm_outputs_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Add initial row to CSV
            self._add_to_verification_csv(
                object_id=str(object_id),
                timestamp=datetime.fromtimestamp(timestamp).isoformat(),
                crop_filename=str(crop_filename),
                vlm_class=str(vlm_class),
                vlm_confidence=f"{vlm_confidence:.3f}",
                vlm_processing_time_ms=f"{processing_time_ms:.2f}",
            )

            return output_id
        except Exception:
            return None

    def _add_to_verification_csv(
        self,
        object_id: str,
        timestamp: str,
        crop_filename: str,
        vlm_class: str,
        vlm_confidence: str,
        vlm_processing_time_ms: str,
    ) -> None:
        """Add initial row to verification CSV."""
        try:
            with open(self.verification_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "object_id",
                    "timestamp",
                    "crop_filename",
                    "vlm_class",
                    "vlm_confidence",
                    "vlm_processing_time_ms",
                    "prompt_version",
                    "manual_verified",
                    "manual_class",
                    "manual_confidence",
                    "is_correct",
                    "verified_by",
                    "verified_at",
                    "notes",
                ])
                writer.writerow({
                    "object_id": object_id,
                    "timestamp": timestamp,
                    "crop_filename": crop_filename,
                    "vlm_class": vlm_class,
                    "vlm_confidence": vlm_confidence,
                    "vlm_processing_time_ms": vlm_processing_time_ms,
                    "prompt_version": self.prompt_version,
                    "manual_verified": "",
                    "manual_class": "",
                    "manual_confidence": "",
                    "is_correct": "",
                    "verified_by": "",
                    "verified_at": "",
                    "notes": "",
                })
        except Exception:
            pass

    def get_session_dir(self) -> Path:
        """Get session directory path."""
        return self.session_dir

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            "session_dir": str(self.session_dir),
            "crops_dir": str(self.crops_dir),
            "vlm_outputs_file": str(self.vlm_outputs_file),
            "verification_file": str(self.verification_file),
            "prompt_version": self.prompt_version,
            "crops_saved": self.crop_counter,
            "outputs_logged": self.output_counter,
        }
