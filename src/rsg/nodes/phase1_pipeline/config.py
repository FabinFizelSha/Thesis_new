"""Phase1 pipeline configuration with optional diagnostics."""

from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class DiagnosticConfig:
    """Optional diagnostic/logging configuration."""

    # Performance timing
    timing_enabled: bool = True
    timing_output_dir: Optional[str] = None

    # Tracking quality
    tracking_quality_enabled: bool = True
    tracking_quality_output_dir: Optional[str] = None

    # Crop extraction
    crop_extraction_enabled: bool = True
    crop_output_dir: Optional[str] = None
    crop_include_rap: bool = True
    crop_include_vlm: bool = True
    crop_include_analysis: bool = True

    def is_any_enabled(self) -> bool:
        """Check if any diagnostics are enabled."""
        return (
            self.timing_enabled
            or self.tracking_quality_enabled
            or self.crop_extraction_enabled
        )


class Phase1Config:
    """Unified Phase1 configuration wrapper."""

    def __init__(self, base_config: Any):
        """Initialize config with diagnostics support.

        Args:
            base_config: Original phase1_config.Phase1Config instance
        """
        self._base = base_config
        self.diagnostics = DiagnosticConfig()
        self._setup_diagnostics()

    def _setup_diagnostics(self) -> None:
        """Setup diagnostic config from base config parameters."""
        # Check for diagnostic enable flags in base config
        if hasattr(self._base, "enable_timing_diagnostics"):
            self.diagnostics.timing_enabled = bool(self._base.enable_timing_diagnostics)

        if hasattr(self._base, "enable_tracking_quality_diagnostics"):
            self.diagnostics.tracking_quality_enabled = bool(
                self._base.enable_tracking_quality_diagnostics
            )

        if hasattr(self._base, "enable_crop_extraction"):
            self.diagnostics.crop_extraction_enabled = bool(
                self._base.enable_crop_extraction
            )

        # Output directories
        if hasattr(self._base, "timing_output_dir"):
            self.diagnostics.timing_output_dir = self._base.timing_output_dir

        if hasattr(self._base, "tracking_quality_output_dir"):
            self.diagnostics.tracking_quality_output_dir = (
                self._base.tracking_quality_output_dir
            )

        if hasattr(self._base, "crop_output_dir"):
            self.diagnostics.crop_output_dir = self._base.crop_output_dir

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to base config."""
        return getattr(self._base, name)

    def disable_all_diagnostics(self) -> None:
        """Disable all diagnostic output."""
        self.diagnostics.timing_enabled = False
        self.diagnostics.tracking_quality_enabled = False
        self.diagnostics.crop_extraction_enabled = False

    def enable_only_timing(self) -> None:
        """Enable only timing diagnostics."""
        self.diagnostics.timing_enabled = True
        self.diagnostics.tracking_quality_enabled = False
        self.diagnostics.crop_extraction_enabled = False

    def enable_only_quality(self) -> None:
        """Enable only tracking quality diagnostics."""
        self.diagnostics.timing_enabled = False
        self.diagnostics.tracking_quality_enabled = True
        self.diagnostics.crop_extraction_enabled = False

    def enable_only_crops(self) -> None:
        """Enable only crop extraction."""
        self.diagnostics.timing_enabled = False
        self.diagnostics.tracking_quality_enabled = False
        self.diagnostics.crop_extraction_enabled = True
