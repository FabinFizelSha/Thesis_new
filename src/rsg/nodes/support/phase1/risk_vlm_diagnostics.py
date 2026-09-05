"""Risk VLM diagnostics -- crops and per-call CSV for manual verification.

Every risk assessment (successful or not) saves the exact crop the Risk VLM
was given and one CSV row, mirroring vlm_test_diagnostics.py's layout for the
object-detection VLM. Output layout::

    <output_dir>/session_<YYYYmmdd_HHMMSS>/
        crops/obj_NNNNNN_crop.jpg
        risk_results.csv

The session folder is unique per process start, so repeated runs never
overwrite an earlier run's crops or CSV.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

CSV_HEADERS = [
    "object_id",
    "crop_filename",
    "track_id",
    "hydra_slot_id",
    "frame_timestamp",
    "source",
    "label",
    "mobility_class",
    "risk_score",
    "risk_factors",
    "risk_inference_ms",
    "end_to_end_ms",
    "success",
    "validation_status",
    "raw_response",
]


class RiskVlmDiagnostics:
    """Log crops and Risk VLM outputs for manual verification."""

    def __init__(self, output_dir: Path = None, enabled: bool = True):
        """Initialize diagnostics.

        Args:
            output_dir: Parent directory; a fresh ``session_<timestamp>/`` is
                created under it.
            enabled: when False, no directory or CSV is created and
                ``log_risk_result`` is a no-op.
        """
        if output_dir is None:
            output_dir = Path("/home/student/rsg_ros2_ws/Risk-VLM-Test-Session")

        self.enabled = bool(enabled)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / f"session_{timestamp}"
        self.crops_dir = self.session_dir / "crops"
        self.log_file = self.session_dir / "risk_results.csv"
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

    def log_risk_result(
        self,
        crop_rgb: Optional[np.ndarray],
        risk_output: Dict[str, Any],
        end_to_end_ms: float,
        *,
        track_id: str = "",
        hydra_slot_id: int = 0,
        label: str = "",
        mobility_class: str = "",
        source: str = "",
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """Save the crop this risk call used and append one result row.

        Called for every risk assessment regardless of success/failure --
        a failed or malformed response is exactly the case worth being able
        to inspect later, and matches vlm_test_diagnostics.log_vlm_result's
        own convention of logging unconditionally.

        Args:
            crop_rgb: exact RGB crop supplied to the Risk VLM (the same wide
                crop already used for classification -- see DESIGN.md §3 for
                why it can't be re-fetched later).
            risk_output: result dict from RiskVlmBackend.assess().
            end_to_end_ms: client perf_counter around the whole risk call.
            track_id, hydra_slot_id, label, mobility_class, source: context
                this risk assessment was made with/for (reference only).
            timestamp: bag time of the classification this risk task followed.
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

            raw = risk_output.get("raw_response", "")
            raw = raw[:200] if isinstance(raw, str) else str(raw)[:200]
            factors = risk_output.get("risk_factors") or []
            factors_text = " | ".join(str(item) for item in factors)

            row = {
                "object_id": object_id,
                "crop_filename": crop_filename,
                "track_id": track_id,
                "hydra_slot_id": int(hydra_slot_id or 0),
                "frame_timestamp": timestamp if timestamp is not None else time.time(),
                "source": source,
                "label": label,
                "mobility_class": mobility_class,
                "risk_score": f"{float(risk_output.get('risk_score', 0.0)):.3f}",
                "risk_factors": factors_text,
                "risk_inference_ms": f"{float(risk_output.get('risk_inference_ms', 0.0)):.2f}" if risk_output.get("risk_inference_ms") else "",
                "end_to_end_ms": f"{float(end_to_end_ms):.2f}",
                "success": str(risk_output.get("success", False)),
                "validation_status": risk_output.get("validation_status", "unknown"),
                "raw_response": raw,
            }
            with open(self.log_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)
            return object_id
        except Exception as e:  # noqa: BLE001 - diagnostics must never break the pipeline
            print(f"Error logging risk VLM result: {e}")
            return None

    def get_session_dir(self) -> Path:
        return self.session_dir
