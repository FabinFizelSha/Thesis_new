"""VLM testing diagnostics - crops and per-call CSV for manual verification.

Shared by two experiments: the (frozen) object-detection label/prompt
experiment in debug/prompt_optimisation_experiment/ (CSV schema in its
EXPERIMENT_REPORT.md §8, jpg crops) and the object_detail prompt-addition
experiment in debug/object_detail_prompt_experiment/ (its EXPERIMENT_REPORT.md
§7, png crops by default -- see ``crop_format``).

Every VLM call saves the exact crop it was given and one CSV row. Output layout::

    <output_dir>/session_<YYYYmmdd_HHMMSS>/
        crops/obj_NNNNNN_crop.<jpg|png>
        vlm_results.csv

The session folder is unique per process start, so repeated runs (of the same
experiment matrix row) never overwrite an earlier run's crops or CSV.
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from nodes.support.workspace_paths import workspace_path
from typing import Any, Dict, Optional

import cv2
import numpy as np

CSV_HEADERS = [
    "object_id",
    "crop_filename",
    "frame_timestamp",
    "run_id",
    "model_profile",
    "prompt_version",
    "vlm_label",
    "label_confidence",
    "mobility_class",
    "mobility_confidence",
    "object_detail",
    "vlm_inference_ms",
    "end_to_end_ms",
    "success",
    "validation_status",
    "raw_response",
    "manual_label",
    "manual_is_correct",
    "error_category",
    "manual_notes",
    # object_detail-specific manual grading (debug/object_detail_prompt_experiment/
    # EXPERIMENT_REPORT.md §8 rubric). Blank / unused for the label-prompt
    # experiment's rows.
    "manual_object_detail_rating",
    "manual_object_detail_notes",
]


class VLMTestDiagnostics:
    """Log crops and VLM outputs for manual verification / prompt optimisation."""

    def __init__(
        self,
        output_dir: Path = None,
        run_id: str = "",
        model_profile: str = "",
        prompt_version: str = "",
        enabled: bool = True,
        crop_format: str = "jpg",
    ):
        """Initialize diagnostics.

        Args:
            output_dir: Parent directory. For a prompt experiment pass
                ``<output_root>/<run_id>``; a fresh ``session_<timestamp>/`` is
                created under it.
            run_id: experiment matrix row id (e.g. ``R1__qwen3vl8b__v1_simplified``)
            model_profile: ``phase1.vlm.active_profile`` for the run
            prompt_version: which prompt is active (e.g. ``P1_v1_simplified``)
            enabled: when False, no directory or CSV is created and
                ``log_vlm_result`` is a no-op.
            crop_format: ``"jpg"`` (default, matches the label-prompt
                experiment's existing archived sessions) or ``"png"``
                (lossless -- used by the object_detail experiment, since
                jpeg compression artifacts can obscure the fine visual state
                cues that field depends on, e.g. a screen being on vs off).
        """
        if output_dir is None:
            output_dir = workspace_path("VLM-Test-Session")

        self.enabled = bool(enabled)
        self.run_id = str(run_id or "")
        self.model_profile = str(model_profile or "")
        self.prompt_version = str(prompt_version or "")
        self.crop_format = str(crop_format or "jpg").strip().lower()
        if self.crop_format not in {"jpg", "jpeg", "png"}:
            self.crop_format = "jpg"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / f"session_{timestamp}"
        self.crops_dir = self.session_dir / "crops"
        self.log_file = self.session_dir / "vlm_results.csv"
        self.object_counter = 0
        self.test_start_time = datetime.now().isoformat()

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

    def log_vlm_result(
        self,
        crop_rgb: np.ndarray,
        vlm_output: Dict[str, Any],
        end_to_end_ms: float,
        inference_ms: float = 0.0,
        timestamp: float = None,
        track_id: str = None,
    ) -> Optional[str]:
        """Save the crop this VLM call used and append one result row.

        Args:
            crop_rgb: exact RGB crop supplied to the VLM (boundary already drawn)
            vlm_output: VLM result dict from the pipeline
            end_to_end_ms: client perf_counter around the whole HTTP call
            inference_ms: model compute time (llama.cpp timings); 0.0 if unavailable
            timestamp: bag time of the crop
            track_id: track id (reference only)
        """
        if not self.enabled:
            return None
        try:
            self.object_counter += 1
            object_id = f"{self.object_counter:06d}"

            if crop_rgb is not None and getattr(crop_rgb, "size", 0) > 0:
                crop_filename = f"obj_{object_id}_crop.{self.crop_format}"
                crop = crop_rgb
                if crop.ndim == 3 and crop.shape[2] == 3:
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(self.crops_dir / crop_filename), crop)
            else:
                crop_filename = "no_crop"

            raw = vlm_output.get("raw_response", "")
            raw = raw[:200] if isinstance(raw, str) else str(raw)[:200]

            row = {
                "object_id": object_id,
                "crop_filename": crop_filename,
                "frame_timestamp": timestamp if timestamp is not None else time.time(),
                "run_id": self.run_id,
                "model_profile": self.model_profile,
                "prompt_version": self.prompt_version,
                "vlm_label": vlm_output.get("label", "unknown_object"),
                "label_confidence": f"{float(vlm_output.get('label_confidence', 0.0)):.3f}",
                "mobility_class": vlm_output.get("mobility_class", "unknown"),
                "mobility_confidence": f"{float(vlm_output.get('mobility_confidence', 0.0)):.3f}",
                "object_detail": vlm_output.get("object_detail", ""),
                "vlm_inference_ms": f"{float(inference_ms):.2f}" if inference_ms else "",
                "end_to_end_ms": f"{float(end_to_end_ms):.2f}",
                "success": str(vlm_output.get("success", False)),
                "validation_status": vlm_output.get("validation_status", "unknown"),
                "raw_response": raw,
                "manual_label": "",
                "manual_is_correct": "",
                "error_category": "",
                "manual_notes": "",
                "manual_object_detail_rating": "",
                "manual_object_detail_notes": "",
            }
            with open(self.log_file, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)
            return object_id
        except Exception as e:  # noqa: BLE001 - diagnostics must never break the pipeline
            print(f"Error logging VLM result: {e}")
            return None

    def get_session_dir(self) -> Path:
        return self.session_dir

    def get_summary(self) -> Dict[str, Any]:
        return {
            "session_dir": str(self.session_dir),
            "crops_dir": str(self.crops_dir),
            "results_file": str(self.log_file),
            "objects_logged": self.object_counter,
            "run_id": self.run_id,
            "model_profile": self.model_profile,
            "prompt_version": self.prompt_version,
            "test_start_time": self.test_start_time,
        }
