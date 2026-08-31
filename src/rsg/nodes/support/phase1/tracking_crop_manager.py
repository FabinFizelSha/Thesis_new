"""Extract and store object crops for manual inspection and analysis."""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np


@dataclass
class CropQuality:
    """Metrics for evaluating crop quality."""
    is_valid: bool
    blur_score: float = 0.0
    saturation: float = 0.0
    brightness: float = 0.0
    area_px: int = 0


class TrackingCropManager:
    """Manages extraction and storage of object crops for track progression analysis."""

    # 3-Metric Weighted Additive Scoring System
    # Score = (W_pixel×pixel + W_sharp×sharpness + W_margin×margin) / (W_pixel + W_sharp + W_margin)
    # All metrics normalized to [0, 1], weights: 2.0:2.0:1.0

    PIXEL_NORMALIZATION_BASE = 100000        # Log saturation point for pixel count
    SHARPNESS_DIVISOR = 400.0                # Laplacian variance normalization
    SHARPNESS_SKIP_THRESHOLD = 0.1           # Skip expensive Laplacian if pixel_score < this
    DEFAULT_SHARPNESS = 0.5                  # Default if skipped
    HYSTERESIS_MARGIN = 0.005                # Require 0.5% improvement (prevents trivial updates)
    MARGIN_EDGE_PROXIMITY = 3                # Check pixels within 3px of crop edges (sub-pixel alignment)

    # Composite score weights (2:2:1 = Pixel:Sharpness:Margin)
    WEIGHT_PIXEL = 2.0                       # Emphasize object prominence
    WEIGHT_SHARPNESS = 2.0                   # Emphasize image quality
    WEIGHT_MARGIN = 1.0                      # De-emphasize framing (lower priority)

    def __init__(self, output_dir: Path):
        """Initialize crop manager with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate session folder (timestamp-based) directly under output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # For backwards compatibility
        self.crops_root = self.output_dir

        # Track crops per track with subfolders for different types
        self.track_crops: Dict[str, Dict[str, Any]] = {}  # track_id -> {analysis, rap, vlm}

        # Diagnostics: detailed logging of each crop update
        self.track_diagnostics: Dict[str, list] = {}  # track_id -> list of {frame, scores...}

        # Create subfolders for crop types
        self.analysis_dir = None
        self.rap_dir = None
        self.vlm_dir = None

    def _clean_mask(self, mask_uint8: np.ndarray) -> np.ndarray:
        """
        Clean mask by removing small noise islands and keeping only largest component.

        Args:
            mask_uint8: Binary mask (uint8)

        Returns:
            Cleaned binary mask (or original if cleaning would eliminate it)
        """
        original_pixels = cv2.countNonZero(mask_uint8)

        # If mask is very small, return as-is (don't over-clean)
        if original_pixels < 20:
            return mask_uint8

        # Apply morphological close to remove small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask_closed)

        if num_labels <= 1:
            return mask_closed

        # Find largest component (excluding background=0)
        largest_label = 0
        largest_size = 0
        for label in range(1, num_labels):
            size = np.sum(labels == label)
            if size > largest_size:
                largest_size = size
                largest_label = label

        # Keep only largest component, but ensure it's substantial
        cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        # If cleaning removed too much, return original
        cleaned_pixels = cv2.countNonZero(cleaned)
        if cleaned_pixels < original_pixels * 0.3:  # Lost more than 70%
            return mask_uint8

        return cleaned

    def get_filtered_mask(self, mask: np.ndarray) -> np.ndarray:
        """Return mask containing only the largest contour (after area filtering).

        Used for 3D geometry calculation to exclude noise/island portions.
        Returns the filtered binary mask ready for depth/bbox estimation.
        """
        if mask is None or mask.size == 0:
            return mask

        try:
            # Convert mask to uint8 if needed
            if mask.dtype != np.uint8:
                mask_uint8 = (mask * 255).astype(np.uint8)
            else:
                mask_uint8 = mask.copy()

            # Threshold to binary
            _, mask_uint8 = cv2.threshold(mask_uint8, 50, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and keep only largest
            min_contour_area = 200
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_contour_area:
                    filtered_contours.append((contour, area))

            if not filtered_contours:
                return mask  # Return original if no valid contours

            # Keep only the largest contour
            filtered_contours.sort(key=lambda x: x[1], reverse=True)
            largest_contour = [filtered_contours[0][0]]

            # Create filtered mask with only largest contour
            filtered_mask = np.zeros_like(mask_uint8)
            cv2.drawContours(filtered_mask, largest_contour, -1, 255, -1)  # -1 thickness fills

            return filtered_mask.astype(mask.dtype)

        except Exception:
            return mask  # Return original mask on error

    def _highlight_contours(self, crop_rgb: np.ndarray, crop_mask: np.ndarray,
                            color: Tuple[int, int, int] = (0, 255, 255),
                            thickness: int = 1) -> np.ndarray:
        """
        Draw highlighted contours around object edges (not crop borders).

        Args:
            crop_rgb: RGB crop image
            crop_mask: Binary mask (0 or 1)
            color: RGB color for contours (bright cyan (0,255,255) by default)
            thickness: Line thickness in pixels

        Returns:
            RGB crop with blue contours drawn around object only
        """
        if crop_rgb is None:
            return crop_rgb

        result = crop_rgb.copy()

        try:
            if crop_mask is None or crop_mask.size == 0:
                return result

            # Convert mask to uint8 if needed
            if crop_mask.dtype != np.uint8:
                mask_uint8 = (crop_mask * 255).astype(np.uint8)
            else:
                mask_uint8 = crop_mask.copy()

            # Use lower threshold (50 instead of 127) to preserve sparse masks
            _, mask_uint8 = cv2.threshold(mask_uint8, 50, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area to remove noise pixels
            # Keep only contours with reasonable area (at least 200 pixels)
            min_contour_area = 200
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_contour_area:
                    filtered_contours.append((contour, area))

            # Keep ONLY the largest contour (no island portions)
            if filtered_contours:
                # Sort by area, keep only the largest
                filtered_contours.sort(key=lambda x: x[1], reverse=True)
                largest_contour = [filtered_contours[0][0]]  # Get the contour, not the area
                cv2.drawContours(result, largest_contour, -1, color, thickness)

        except Exception:
            pass

        return result

    def extract_crop(
        self,
        rgb_image: np.ndarray,
        mask: np.ndarray,
        bbox_2d: Tuple[int, int, int, int],
        track_id: str,
        frame_id: str,
        sequence: int,
        is_new_track: bool = False,
    ) -> Optional[str]:
        """
        Extract object crop from RGB image using mask.

        Args:
            rgb_image: RGB image array (H, W, 3)
            mask: Binary mask or confidence map
            bbox_2d: (x, y, width, height) in pixels (XYWH format)
            track_id: Persistent track ID
            frame_id: Frame identifier
            sequence: Frame sequence number
            is_new_track: True if this is the track's first observation

        Returns:
            Path to saved crop image, or None if extraction failed
        """
        try:
            # Validate inputs
            if rgb_image is None or rgb_image.size == 0:
                return None
            if mask is None or mask.size == 0:
                return None

            x, y, width, height = bbox_2d
            x_min, y_min = int(x), int(y)
            x_max, y_max = int(x + width), int(y + height)
            if x_min >= x_max or y_min >= y_max:
                return None

            # Clamp to image bounds
            h, w = rgb_image.shape[:2]
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(w, int(x_max))
            y_max = min(h, int(y_max))

            if x_min >= x_max or y_min >= y_max:
                return None

            # Extract crop from both RGB and mask using identical coordinates
            crop_rgb = rgb_image[y_min:y_max, x_min:x_max]
            crop_mask = mask[y_min:y_max, x_min:x_max]

            if crop_rgb.size == 0:
                return None

            # Convert RGB to BGR for OpenCV
            if len(crop_rgb.shape) == 3 and crop_rgb.shape[2] == 3:
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = crop_rgb

            # Compute quality metrics
            quality = self._compute_crop_quality(crop_rgb, crop_mask)

            # Create track-specific folder
            track_dir = self.session_dir / track_id
            track_dir.mkdir(parents=True, exist_ok=True)

            # Initialize track record if needed
            if track_id not in self.track_crops:
                self.track_crops[track_id] = {
                    "initial_path": None,
                    "initial_sequence": None,
                    "best_quality": -1.0,
                    "best_updates": [],  # List of {sequence, path, quality, metrics}
                }

            # Save initial crop if this is the first observation
            if is_new_track:
                initial_path = track_dir / "initial.jpg"
                # Highlight contours for clarity (helps VLM and visual inspection)
                crop_with_contours = self._highlight_contours(crop_rgb, crop_mask,
                                                              color=(0, 255, 255),
                                                              thickness=2)
                crop_with_contours_bgr = cv2.cvtColor(crop_with_contours, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(initial_path), crop_with_contours_bgr)
                self.track_crops[track_id]["initial_path"] = str(initial_path)
                self.track_crops[track_id]["initial_sequence"] = sequence

            # Update best crop - 3-metric additive scoring
            composite_score, pixel_score, sharpness_score, margin_score = self._score_crop(crop_rgb, crop_mask)
            best_quality = self.track_crops[track_id]["best_quality"]

            # Apply 10% hysteresis: only replace if meaningfully better
            if composite_score > best_quality * (1.0 + self.HYSTERESIS_MARGIN):
                # This is a new best crop - save it as an update
                update_number = len(self.track_crops[track_id]["best_updates"]) + 1
                best_path = track_dir / f"best_update_{update_number}.jpg"
                # Highlight contours for clarity (helps VLM and visual inspection)
                crop_with_contours = self._highlight_contours(crop_rgb, crop_mask,
                                                              color=(0, 255, 255),
                                                              thickness=2)
                crop_with_contours_bgr = cv2.cvtColor(crop_with_contours, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(best_path), crop_with_contours_bgr)

                # Record this update
                update_info = {
                    "update_number": update_number,
                    "sequence": sequence,
                    "path": str(best_path),
                    "quality_score": composite_score,
                    "pixel_score": pixel_score,
                    "sharpness_score": sharpness_score,
                    "margin_score": margin_score,
                }
                self.track_crops[track_id]["best_updates"].append(update_info)
                old_best = self.track_crops[track_id]["best_quality"]
                self.track_crops[track_id]["best_quality"] = composite_score

                # Log detailed diagnostics for this update
                # Get old metric scores from previous best update if it exists
                old_pixel_score = 0.0
                old_sharpness_score = 0.0
                old_margin_score = 0.0
                if self.track_crops[track_id]["best_updates"]:
                    last_update = self.track_crops[track_id]["best_updates"][-1]
                    old_pixel_score = last_update.get("pixel_score", 0.0)
                    old_sharpness_score = last_update.get("sharpness_score", 0.0)
                    old_margin_score = last_update.get("margin_score", 0.0)

                self._log_crop_update_diagnostics(
                    track_id, update_number, sequence,
                    old_best, composite_score,
                    old_pixel_score, old_sharpness_score, old_margin_score,
                    pixel_score, sharpness_score, margin_score
                )

            return self.track_crops[track_id]["best_updates"][-1]["path"] if self.track_crops[track_id]["best_updates"] else None

        except Exception as e:
            # Silently skip failed crops
            return None

    def save_rap_crop(
        self,
        rap_crop: np.ndarray,
        track_id: str,
        sequence: int,
        quality_score: Optional[float] = None,
    ) -> Optional[str]:
        """Save RAP crop (visual retrieval system input)."""
        try:
            if rap_crop is None or rap_crop.size == 0:
                return None

            # Create track folder if needed
            track_dir = self.session_dir / track_id
            track_dir.mkdir(parents=True, exist_ok=True)

            # Initialize track record if needed
            if track_id not in self.track_crops:
                self.track_crops[track_id] = {"analysis": [], "rap": [], "vlm": []}

            # Save RAP crop with frame info
            rap_path = track_dir / f"rap_frame_{sequence:06d}.jpg"

            # Convert BGR if needed (RAP crops typically in BGR from OpenCV operations)
            if len(rap_crop.shape) == 3 and rap_crop.shape[2] == 3:
                crop_to_save = rap_crop
            else:
                crop_to_save = rap_crop

            cv2.imwrite(str(rap_path), crop_to_save)
            self.track_crops[track_id]["rap"].append({
                "sequence": sequence,
                "path": str(rap_path),
                "quality": quality_score or 0.0,
            })
            return str(rap_path)
        except Exception:
            return None

    def save_vlm_crop(
        self,
        vlm_crop: np.ndarray,
        track_id: str,
        sequence: int,
        quality_score: Optional[float] = None,
    ) -> Optional[str]:
        """Save VLM crop (vision language model input)."""
        try:
            if vlm_crop is None or vlm_crop.size == 0:
                return None

            # Create track folder if needed
            track_dir = self.session_dir / track_id
            track_dir.mkdir(parents=True, exist_ok=True)

            # Initialize track record if needed
            if track_id not in self.track_crops:
                self.track_crops[track_id] = {"analysis": [], "rap": [], "vlm": []}

            # Save VLM crop with frame info
            vlm_path = track_dir / f"vlm_frame_{sequence:06d}.jpg"

            # Convert BGR if needed
            if len(vlm_crop.shape) == 3 and vlm_crop.shape[2] == 3:
                crop_to_save = vlm_crop
            else:
                crop_to_save = vlm_crop

            cv2.imwrite(str(vlm_path), crop_to_save)
            self.track_crops[track_id]["vlm"].append({
                "sequence": sequence,
                "path": str(vlm_path),
                "quality": quality_score or 0.0,
            })
            return str(vlm_path)
        except Exception:
            return None

    def _compute_crop_quality(self, crop_rgb: np.ndarray, mask: np.ndarray) -> CropQuality:
        """Compute quality metrics for a crop."""
        try:
            # Laplacian blur detection (higher = sharper)
            gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if len(crop_rgb.shape) == 3 else crop_rgb
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Saturation
            if len(crop_rgb.shape) == 3:
                hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv[:, :, 1]) / 255.0
            else:
                saturation = 0.0

            # Brightness (mean pixel value)
            brightness = np.mean(gray) / 255.0

            # Mask coverage
            mask_area = np.sum(mask > 0) if mask is not None else crop_rgb.shape[0] * crop_rgb.shape[1]

            return CropQuality(
                is_valid=True,
                blur_score=min(blur_score / 1000.0, 1.0),  # Normalize to ~0-1
                saturation=saturation,
                brightness=brightness,
                area_px=int(mask_area),
            )
        except Exception:
            return CropQuality(is_valid=False)

    def _score_crop(self, crop_rgb: np.ndarray, mask: np.ndarray) -> tuple:
        """
        Score crop quality using 3 normalized metrics (additive).

        Returns:
            (composite_score, pixel_score, sharpness_score, margin_score)
        """
        if crop_rgb is None or mask is None or mask.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        try:
            # Metric 1: Pixel Count (logarithmic normalization)
            pixel_count = int(np.sum(mask > 0))
            pixel_score = np.log(1 + pixel_count) / np.log(1 + self.PIXEL_NORMALIZATION_BASE)
            pixel_score = min(pixel_score, 1.0)

            # Metric 3: Sharpness (Laplacian variance, skip if low pixel count)
            if pixel_score < self.SHARPNESS_SKIP_THRESHOLD:
                sharpness_score = self.DEFAULT_SHARPNESS
            else:
                gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if len(crop_rgb.shape) == 3 else crop_rgb
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_score = min(laplacian_var / self.SHARPNESS_DIVISOR, 1.0)

            # Metric 4: Margin (edges touched)
            margin_score = self._compute_margin_score(mask)

            # Composite: weighted additive (Pixel:Sharpness:Margin weights)
            total_weight = self.WEIGHT_PIXEL + self.WEIGHT_SHARPNESS + self.WEIGHT_MARGIN
            composite_score = (self.WEIGHT_PIXEL * pixel_score +
                             self.WEIGHT_SHARPNESS * sharpness_score +
                             self.WEIGHT_MARGIN * margin_score) / total_weight

            return composite_score, pixel_score, sharpness_score, margin_score

        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def _compute_margin_score(self, mask: np.ndarray) -> float:
        """
        Margin score based on object pixels in crop's outer edge zone.

        Measures: 1.0 - (mask_pixels_in_edge_zone / total_edge_zone_pixels)
        Higher score = object well-framed (few pixels touching edges)
        Lower score = object cropped tight (many pixels in edge zone)

        Logic:
        1. Define outer 3px border of crop
        2. Count total pixels in this border
        3. Count how many border pixels contain object (mask > 0)
        4. Fraction = object_pixels / total_border_pixels
        5. Score = 1.0 - fraction

        Examples:
        - 0% object in border → 1.0 (perfect framing)
        - 5% object in border → 0.95 (good margin)
        - 10% object in border → 0.90 (tight)
        - 20% object in border → 0.80 (severely cropped)

        Returns [0, 1] where 1.0 = no object at edges (best), 0.0 = all edges have object (worst)
        """
        if mask.size == 0:
            return 0.5

        try:
            height, width = mask.shape
            mask_binary = (mask > 0).astype(np.uint8)

            # Define outer 3-pixel border zone
            edge_zone = np.zeros_like(mask_binary)
            edge_zone[0:self.MARGIN_EDGE_PROXIMITY, :] = 1                    # Top 3 rows
            edge_zone[-self.MARGIN_EDGE_PROXIMITY:, :] = 1                    # Bottom 3 rows
            edge_zone[:, 0:self.MARGIN_EDGE_PROXIMITY] = 1                    # Left 3 columns
            edge_zone[:, -self.MARGIN_EDGE_PROXIMITY:] = 1                    # Right 3 columns

            # Count total pixels in edge zone
            total_edge_pixels = np.sum(edge_zone)
            if total_edge_pixels == 0:
                return 1.0  # No edge zone (crop too small)

            # Count object pixels in edge zone
            object_in_edge = np.sum(mask_binary * edge_zone)

            # Calculate fraction and invert for score
            edge_fraction = object_in_edge / total_edge_pixels
            margin_score = 1.0 - edge_fraction

            return max(0.0, min(margin_score, 1.0))

        except Exception:
            return 0.5

    def _log_crop_update_diagnostics(self, track_id: str, update_number: int, sequence: int,
                                      old_score: float, new_score: float,
                                      old_pixel_score: float, old_sharpness_score: float, old_margin_score: float,
                                      pixel_score: float, sharpness_score: float, margin_score: float) -> None:
        """
        Log diagnostic info when a crop update is accepted.

        Args:
            track_id: Track identifier
            update_number: Which update number this is (1, 2, 3, ...)
            sequence: Frame sequence number
            old_score: Previous best composite score
            new_score: New best composite score
            old_pixel_score: Previous pixel count metric [0, 1]
            old_sharpness_score: Previous sharpness metric [0, 1]
            old_margin_score: Previous margin metric [0, 1]
            pixel_score: New pixel count metric [0, 1]
            sharpness_score: New sharpness metric [0, 1]
            margin_score: New margin metric [0, 1]
        """
        if track_id not in self.track_diagnostics:
            self.track_diagnostics[track_id] = []

        diagnostic_entry = {
            "track_id": track_id,
            "frame": sequence,
            "old_score": round(old_score, 6),
            "old_pixel_score": round(old_pixel_score, 6),
            "old_sharpness_score": round(old_sharpness_score, 6),
            "old_margin_score": round(old_margin_score, 6),
            "new_score": round(new_score, 6),
            "new_pixel_score": round(pixel_score, 6),
            "new_sharpness_score": round(sharpness_score, 6),
            "new_margin_score": round(margin_score, 6),
        }
        self.track_diagnostics[track_id].append(diagnostic_entry)

    def save_crop_progression_diagnostics(self) -> Path:
        """Save detailed diagnostics of crop score progression for each track."""
        if not self.track_diagnostics:
            return None

        csv_path = self.session_dir / "crop_progression_diagnostics.csv"

        all_diagnostics = []
        for track_id, diagnostics_list in self.track_diagnostics.items():
            for diag in diagnostics_list:
                all_diagnostics.append({
                    "track_id": diag.get("track_id", ""),
                    "frame": diag.get("frame", ""),
                    "old_score_total": f"{diag.get('old_score', 0):.6f}",
                    "old_score_pixel": f"{diag.get('old_pixel_score', 0):.6f}",
                    "old_score_sharpness": f"{diag.get('old_sharpness_score', 0):.6f}",
                    "old_score_margin": f"{diag.get('old_margin_score', 0):.6f}",
                    "new_score_total": f"{diag.get('new_score', 0):.6f}",
                    "new_score_pixel": f"{diag.get('new_pixel_score', 0):.6f}",
                    "new_score_sharpness": f"{diag.get('new_sharpness_score', 0):.6f}",
                    "new_score_margin": f"{diag.get('new_margin_score', 0):.6f}",
                })

        if all_diagnostics:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "track_id",
                        "frame",
                        "old_score_total",
                        "old_score_pixel",
                        "old_score_sharpness",
                        "old_score_margin",
                        "new_score_total",
                        "new_score_pixel",
                        "new_score_sharpness",
                        "new_score_margin",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_diagnostics)

        return csv_path

    def save_crop_summary(self) -> Path:
        """Save CSV summary of all track crops with best crop update sequence."""
        if not self.track_crops:
            return None

        csv_path = self.output_dir / "track_crops_summary.csv"

        summary_data = []
        for track_id, crop_info in self.track_crops.items():
            # Get info about best crop updates
            best_updates = crop_info.get("best_updates", [])
            best_crop_count = len(best_updates)

            if best_updates:
                # Use the latest (best) update for summary
                latest_update = best_updates[-1]
                best_crop_filename = Path(latest_update["path"]).name if latest_update.get("path") else "N/A"
                best_sequence = latest_update.get("sequence", "")
                best_quality = latest_update.get("quality_score", "")
                blur = latest_update.get("blur_score", "")
                saturation = latest_update.get("saturation", "")
                brightness = latest_update.get("brightness", "")
            else:
                best_crop_filename = "N/A"
                best_sequence = ""
                best_quality = ""
                blur = ""
                saturation = ""
                brightness = ""

            pixel_score = latest_update.get("pixel_score", "")
            sharpness_score = latest_update.get("sharpness_score", "")
            margin_score = latest_update.get("margin_score", "")

            summary_data.append({
                "track_id": track_id,
                "initial_crop": Path(crop_info["initial_path"]).name if crop_info["initial_path"] else "N/A",
                "initial_frame": crop_info.get("initial_sequence", ""),
                "best_crop_updates": best_crop_count,
                "latest_best_crop": best_crop_filename,
                "latest_best_frame": best_sequence,
                "latest_best_quality": f"{best_quality:.3f}" if best_quality else "",
                "pixel_score": f"{pixel_score:.3f}" if pixel_score else "",
                "sharpness_score": f"{sharpness_score:.3f}" if sharpness_score else "",
                "margin_score": f"{margin_score:.3f}" if margin_score else "",
            })

        if summary_data:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "track_id",
                        "initial_crop",
                        "initial_frame",
                        "best_crop_updates",
                        "latest_best_crop",
                        "latest_best_frame",
                        "latest_best_quality",
                        "pixel_score",
                        "sharpness_score",
                        "margin_score",
                    ],
                )
                writer.writeheader()
                writer.writerows(summary_data)

        return csv_path

    def save_best_crop_updates_log(self) -> Path:
        """Save detailed log of all best crop updates for each track."""
        if not self.track_crops:
            return None

        csv_path = self.output_dir / "best_crop_updates_log.csv"

        all_updates = []
        for track_id, crop_info in self.track_crops.items():
            for update in crop_info.get("best_updates", []):
                all_updates.append({
                    "track_id": track_id,
                    "update_number": update.get("update_number", ""),
                    "frame": update.get("sequence", ""),
                    "crop_file": Path(update.get("path", "")).name if update.get("path") else "N/A",
                    "quality_score": f"{update.get('quality_score', 0):.3f}",
                    "pixel_score": f"{update.get('pixel_score', 0):.3f}",
                    "sharpness_score": f"{update.get('sharpness_score', 0):.3f}",
                    "margin_score": f"{update.get('margin_score', 0):.3f}",
                })

        if all_updates:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "track_id",
                        "update_number",
                        "frame",
                        "crop_file",
                        "quality_score",
                        "pixel_score",
                        "sharpness_score",
                        "margin_score",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_updates)

        return csv_path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_updates = sum(len(crop_info.get("best_updates", [])) for crop_info in self.track_crops.values())
        return {
            "total_tracks": len(self.track_crops),
            "total_best_crop_updates": total_updates,
            "session_dir": str(self.session_dir),
            "crops_root": str(self.crops_root),
        }

    def save_best_crop(
        self,
        track_id: str,
        source_rgb: np.ndarray,
        crop_revision: int,
        crop_score: float,
        sequence: int,
    ) -> Optional[str]:
        """Save a crop when it becomes the best (already marked with boundaries).

        Args:
            track_id: Track identifier
            source_rgb: RGB crop (already marked with boundaries from _remember_track_crop)
            crop_revision: Crop revision number
            crop_score: Quality score of this crop
            sequence: Frame sequence number

        Returns:
            Path to saved crop file, or None if save failed
        """
        try:
            if source_rgb is None or source_rgb.size == 0:
                return None

            # Create track folder if needed
            track_dir = self.session_dir / track_id
            track_dir.mkdir(parents=True, exist_ok=True)

            # Initialize track record if needed
            if track_id not in self.track_crops:
                self.track_crops[track_id] = {
                    "best_updates": [],
                    "rap": [],
                    "vlm": [],
                }

            # Get next best update number
            update_number = len(self.track_crops[track_id]["best_updates"]) + 1
            best_path = track_dir / f"best_update_{update_number}.jpg"

            # Convert to BGR for saving
            if len(source_rgb.shape) == 3 and source_rgb.shape[2] == 3:
                crop_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = source_rgb

            cv2.imwrite(str(best_path), crop_bgr)

            # Record this update
            update_info = {
                "update_number": update_number,
                "revision": crop_revision,
                "sequence": sequence,
                "path": str(best_path),
                "score": crop_score,
                "filename": best_path.name,
            }
            self.track_crops[track_id]["best_updates"].append(update_info)

            return str(best_path)
        except Exception as e:
            return None

    def save_rap_dequeue_crop(
        self,
        track_id: str,
        rap_crop: np.ndarray,
        crop_revision: int,
        crop_score: float,
        sequence: int,
    ) -> Optional[str]:
        """Save the crop that RAP is using when it dequeues this track.

        Crop is already marked with boundaries when it became the best crop.

        Args:
            track_id: Track identifier
            rap_crop: RGB crop extracted for RAP (already marked with boundaries)
            crop_revision: Crop revision number this came from
            crop_score: Quality score of this crop
            sequence: Frame sequence number

        Returns:
            Path to saved crop file, or None if save failed
        """
        try:
            if rap_crop is None or rap_crop.size == 0:
                return None

            # Create track folder if needed
            track_dir = self.session_dir / track_id
            track_dir.mkdir(parents=True, exist_ok=True)

            # Create rap subfolder
            rap_dir = track_dir / "rap"
            rap_dir.mkdir(parents=True, exist_ok=True)

            # Save with revision and sequence info
            rap_path = rap_dir / f"rap_rev{crop_revision}_seq{sequence:06d}.jpg"

            # Convert to BGR for saving (crop already marked with boundaries)
            if len(rap_crop.shape) == 3 and rap_crop.shape[2] == 3:
                crop_bgr = cv2.cvtColor(rap_crop, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = rap_crop

            cv2.imwrite(str(rap_path), crop_bgr)

            # Track this in diagnostics
            if track_id not in self.track_crops:
                self.track_crops[track_id] = {"rap": [], "vlm": [], "best_updates": []}

            self.track_crops[track_id]["rap"].append({
                "revision": crop_revision,
                "sequence": sequence,
                "path": str(rap_path),
                "score": crop_score,
                "filename": rap_path.name,
            })

            return str(rap_path)
        except Exception as e:
            return None

    def save_vlm_dequeue_crop(
        self,
        track_id: str,
        vlm_crop: np.ndarray,
        crop_revision: int,
        crop_score: float,
        sequence: int,
    ) -> Optional[str]:
        """Save the crop that VLM is using when it dequeues this track.

        Crop is already marked with boundaries when it became the best crop.

        Args:
            track_id: Track identifier
            vlm_crop: RGB crop extracted for VLM (already marked with boundaries)
            crop_revision: Crop revision number this came from
            crop_score: Quality score of this crop
            sequence: Frame sequence number

        Returns:
            Path to saved crop file, or None if save failed
        """
        try:
            if vlm_crop is None or vlm_crop.size == 0:
                return None

            # Create track folder if needed
            track_dir = self.session_dir / track_id
            track_dir.mkdir(parents=True, exist_ok=True)

            # Create vlm subfolder
            vlm_dir = track_dir / "vlm"
            vlm_dir.mkdir(parents=True, exist_ok=True)

            # Save with revision and sequence info
            vlm_path = vlm_dir / f"vlm_rev{crop_revision}_seq{sequence:06d}.jpg"

            # Convert to BGR for saving (crop already marked with boundaries)
            if len(vlm_crop.shape) == 3 and vlm_crop.shape[2] == 3:
                crop_bgr = cv2.cvtColor(vlm_crop, cv2.COLOR_RGB2BGR)
            else:
                crop_bgr = vlm_crop

            cv2.imwrite(str(vlm_path), crop_bgr)

            # Track this in diagnostics
            if track_id not in self.track_crops:
                self.track_crops[track_id] = {"rap": [], "vlm": [], "best_updates": []}

            self.track_crops[track_id]["vlm"].append({
                "revision": crop_revision,
                "sequence": sequence,
                "path": str(vlm_path),
                "score": crop_score,
                "filename": vlm_path.name,
            })

            return str(vlm_path)
        except Exception as e:
            return None

    def save_system_usage_diagnostics(self) -> Path:
        """Save CSV showing which crops were used by RAP and VLM for each track.

        Returns:
            Path to CSV file with RAP/VLM usage diagnostics
        """
        csv_path = self.session_dir / "system_usage_diagnostics.csv"

        rows = []
        for track_id, crop_info in self.track_crops.items():
            # RAP crops
            for rap_crop in crop_info.get("rap", []):
                rows.append({
                    "track_id": track_id,
                    "system": "RAP",
                    "crop_revision": rap_crop.get("revision", ""),
                    "sequence": rap_crop.get("sequence", ""),
                    "crop_score": f"{rap_crop.get('score', 0):.3f}",
                    "crop_file": rap_crop.get("filename", ""),
                    "file_path": rap_crop.get("path", ""),
                })

            # VLM crops
            for vlm_crop in crop_info.get("vlm", []):
                rows.append({
                    "track_id": track_id,
                    "system": "VLM",
                    "crop_revision": vlm_crop.get("revision", ""),
                    "sequence": vlm_crop.get("sequence", ""),
                    "crop_score": f"{vlm_crop.get('score', 0):.3f}",
                    "crop_file": vlm_crop.get("filename", ""),
                    "file_path": vlm_crop.get("path", ""),
                })

        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "track_id",
                        "system",
                        "crop_revision",
                        "sequence",
                        "crop_score",
                        "crop_file",
                        "file_path",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

        return csv_path
