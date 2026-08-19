"""Backend runner wrappers for uniform SAM interface.

Provides ViT-B and NanoSAM runners with consistent API for masking operations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import time


class BackendRunner(ABC):
    """Abstract base for SAM backends."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize backend.

        Args:
            name: Backend identifier ("vitb" or "nanosam")
            config: Configuration dict with SAM parameters
        """
        self.name = name
        self.config = config
        self._init_backend()

    @abstractmethod
    def _init_backend(self) -> None:
        """Initialize the actual backend model."""
        pass

    @abstractmethod
    def segment(self, rgb: np.ndarray, depth: np.ndarray) -> List[np.ndarray]:
        """Generate segmentation masks.

        Args:
            rgb: RGB image (H×W×3 uint8)
            depth: Depth map (H×W float32, meters)

        Returns:
            List of boolean masks (H×W each)
        """
        pass


class ViTBRunner(BackendRunner):
    """ViT-B backend runner."""

    def _init_backend(self) -> None:
        """Initialize ViT-B model."""
        try:
            from segment_anything import sam_model_registry
            from pathlib import Path
            model_type = "vit_b"
            checkpoint = self.config.get("checkpoint", "sam_vit_b_01ec64.pth")
            checkpoint = str(Path(checkpoint).expanduser())  # Expand ~/
            device = self.config.get("device", "cuda")

            self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
            self.sam.to(device)

            from segment_anything import SamPredictor
            self.predictor = SamPredictor(self.sam)

            self.device = device
            print(f"✓ ViT-B backend initialized (device: {device})")

        except Exception as e:
            print(f"✗ Failed to initialize ViT-B: {e}")
            raise

    def segment(self, rgb: np.ndarray, depth: np.ndarray) -> List[np.ndarray]:
        """Segment image with ViT-B.

        Args:
            rgb: RGB image (H×W×3 uint8)
            depth: Depth map (unused for ViT-B, included for interface compat)

        Returns:
            List of masks (boolean arrays H×W)
        """
        try:
            import torch

            # Set image
            self.predictor.set_image(rgb)

            # Get configuration parameters
            points_per_side = self.config.get("points_per_side", 8)
            mask_threshold = self.config.get("mask_threshold", 0.0)
            pred_iou_thresh = self.config.get("pred_iou_thresh", 0.0)

            # Generate masks with prompt grid
            h, w = rgb.shape[:2]
            stride = max(1, w // points_per_side)

            # Create point grid
            points = []
            for y in range(stride // 2, h, stride):
                for x in range(stride // 2, w, stride):
                    points.append([x, y])

            if not points:
                return []

            points = np.array(points, dtype=np.float32)

            # Predict masks for each point
            all_masks = []
            all_scores = []

            # Process each point to get multiple mask candidates
            for point in points:
                masks, scores, _ = self.predictor.predict_torch(
                    point_coords=torch.from_numpy(np.array([[point]], dtype=np.float32)).to(self.device),
                    point_labels=torch.ones(1, dtype=torch.long).unsqueeze(0).to(self.device),
                    multimask_output=True,  # Get 3 mask candidates per point
                )

                # masks shape: (1, 3, H, W), scores shape: (1, 3)
                for mask, score in zip(masks[0], scores[0]):
                    if score >= mask_threshold:
                        all_masks.append(mask.cpu().numpy().astype(bool))
                        all_scores.append(score.cpu().item())

            # Apply NMS to remove overlapping masks
            if all_masks:
                nms_iou_thresh = self.config.get("nms_iou", 0.9)
                result_masks = self._apply_nms(all_masks, all_scores, nms_iou_thresh)
            else:
                result_masks = []

            # Limit to max_masks (keep highest confidence)
            max_masks = self.config.get("max_masks", 50)
            if len(result_masks) > max_masks:
                result_masks = result_masks[:max_masks]

            return result_masks

        except Exception as e:
            print(f"✗ ViT-B segmentation failed: {e}")
            return []

    def _apply_nms(self, masks: List[np.ndarray], scores: List[float], nms_iou_thresh: float = 0.9) -> List[np.ndarray]:
        """Apply Non-Maximum Suppression to remove overlapping masks.

        Args:
            masks: List of boolean masks
            scores: List of confidence scores
            nms_iou_thresh: IoU threshold for NMS

        Returns:
            Filtered list of masks
        """
        if not masks:
            return []

        # Sort by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        keep = []
        for i in sorted_indices:
            if i not in keep:
                # Check overlap with kept masks
                should_keep = True
                for j in keep:
                    # Calculate IoU
                    inter = np.logical_and(masks[i], masks[j]).sum()
                    union = np.logical_or(masks[i], masks[j]).sum()
                    iou = inter / union if union > 0 else 0

                    if iou > nms_iou_thresh:
                        should_keep = False
                        break

                if should_keep:
                    keep.append(i)

        return [masks[i] for i in keep]


class NanoSAMRunner(BackendRunner):
    """NanoSAM backend runner using TensorRT engines."""

    def _init_backend(self) -> None:
        """Initialize NanoSAM model from TensorRT engines."""
        try:
            from pathlib import Path

            self.device = self.config.get("device", "cuda")

            # Get engine paths - use nanosam-specific or fallback to checkpoint
            image_encoder = self.config.get("image_encoder_engine")
            mask_decoder = self.config.get("mask_decoder_engine")

            if not image_encoder or not mask_decoder:
                raise FileNotFoundError(
                    "NanoSAM requires image_encoder_engine and mask_decoder_engine paths"
                )

            image_encoder = str(Path(image_encoder).expanduser().resolve())
            mask_decoder = str(Path(mask_decoder).expanduser().resolve())

            if not Path(image_encoder).exists():
                raise FileNotFoundError(f"Image encoder engine not found: {image_encoder}")
            if not Path(mask_decoder).exists():
                raise FileNotFoundError(f"Mask decoder engine not found: {mask_decoder}")

            # Import and initialize NanoSAM predictor
            from nanosam.utils.predictor import Predictor

            self.predictor = Predictor(
                image_encoder_engine=image_encoder,
                mask_decoder_engine=mask_decoder,
            )

            print(f"✓ NanoSAM backend initialized (device: {self.device})")

        except Exception as e:
            print(f"✗ Failed to initialize NanoSAM: {e}")
            raise

    def segment(self, rgb: np.ndarray, depth: np.ndarray) -> List[np.ndarray]:
        """Segment image with NanoSAM.

        Args:
            rgb: RGB image (H×W×3 uint8)
            depth: Depth map (H×W float32, meters)

        Returns:
            List of masks (boolean arrays H×W)
        """
        try:
            from PIL import Image

            if self.predictor is None:
                return []

            h, w = rgb.shape[:2]
            if h <= 0 or w <= 0:
                return []

            # Encode image once; decoding is much cheaper
            self.predictor.set_image(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))

            # Get configuration parameters
            points_per_side = self.config.get("points_per_side", 8)
            mask_threshold = self.config.get("mask_threshold", 0.0)
            nms_iou_thresh = self.config.get("nms_iou", 0.9)
            max_masks = self.config.get("max_masks", 50)

            # Generate grid prompts
            prompts = self._make_grid_prompts(rgb, points_per_side)

            if not prompts:
                return []

            # Collect masks from all prompts
            all_masks = []
            all_scores = []

            for x, y in prompts:
                try:
                    raw_mask, score, _ = self.predictor.predict(
                        np.array([[float(x), float(y)]], dtype=np.float32),
                        np.array([1], dtype=np.int64),
                    )

                    mask = self._to_bool_mask(raw_mask, threshold=mask_threshold)

                    # Resize if needed
                    if mask.shape[:2] != (h, w):
                        mask = self._resize_mask(mask, w, h)

                    area = int(np.count_nonzero(mask))
                    if area >= 1:  # Keep any non-empty mask
                        all_masks.append(mask)
                        all_scores.append(self._score_to_float(score))

                except Exception as e:
                    # Skip individual prompt failures
                    continue

            # Apply NMS to remove overlapping masks
            if all_masks:
                result_masks = self._apply_nms(all_masks, all_scores, nms_iou_thresh)
            else:
                result_masks = []

            # Limit to max_masks (keep highest confidence)
            if len(result_masks) > max_masks:
                result_masks = result_masks[:max_masks]

            return result_masks

        except Exception as e:
            print(f"✗ NanoSAM segmentation failed: {e}")
            return []

    def _make_grid_prompts(self, rgb: np.ndarray, points_per_side: int) -> List[tuple]:
        """Generate grid of prompt points."""
        h, w = rgb.shape[:2]
        stride = max(1, w // points_per_side)

        prompts = []
        for y in range(stride // 2, h, stride):
            for x in range(stride // 2, w, stride):
                prompts.append((int(x), int(y)))

        return prompts

    def _apply_nms(self, masks: List[np.ndarray], scores: List[float], nms_iou_thresh: float = 0.9) -> List[np.ndarray]:
        """Apply Non-Maximum Suppression to remove overlapping masks."""
        if not masks:
            return []

        # Sort by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        keep = []
        for i in sorted_indices:
            if i not in keep:
                # Check overlap with kept masks
                should_keep = True
                for j in keep:
                    # Calculate IoU
                    inter = np.logical_and(masks[i], masks[j]).sum()
                    union = np.logical_or(masks[i], masks[j]).sum()
                    iou = inter / union if union > 0 else 0

                    if iou > nms_iou_thresh:
                        should_keep = False
                        break

                if should_keep:
                    keep.append(i)

        return [masks[i] for i in keep]

    @staticmethod
    def _score_to_float(score: Any) -> float:
        """Convert score to float."""
        try:
            if hasattr(score, "detach"):
                score = score.detach().cpu().numpy()
            arr = np.asarray(score, dtype=np.float32)
            return float(arr.reshape(-1)[0]) if arr.size else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _to_bool_mask(raw_mask: Any, threshold: float) -> np.ndarray:
        """Convert raw mask to boolean array."""
        if hasattr(raw_mask, "detach"):
            raw_mask = raw_mask.detach().cpu().numpy()
        arr = np.asarray(raw_mask)
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            # Keep the first candidate if decoder returns multiple masks
            arr = arr.reshape((-1,) + arr.shape[-2:])[0]
        if arr.dtype == np.bool_:
            return arr.astype(bool)
        return arr.astype(np.float32) > float(threshold)

    @staticmethod
    def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize mask to target dimensions."""
        try:
            import cv2
            return cv2.resize(
                mask.astype(np.uint8),
                (int(width), int(height)),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        except Exception:
            # Fallback to nearest-neighbor indexing
            y_idx = np.linspace(0, mask.shape[0] - 1, int(height)).astype(np.int64)
            x_idx = np.linspace(0, mask.shape[1] - 1, int(width)).astype(np.int64)
            return mask[np.ix_(y_idx, x_idx)].astype(bool)


class MockBackendRunner(BackendRunner):
    """Mock backend for testing without actual models."""

    def _init_backend(self) -> None:
        """Initialize mock backend."""
        self.seed = self.config.get("seed", 42)
        print(f"✓ Mock {self.name} backend initialized (seed: {self.seed})")

    def segment(self, rgb: np.ndarray, depth: np.ndarray) -> List[np.ndarray]:
        """Generate synthetic masks for testing.

        Args:
            rgb: RGB image (H×W×3 uint8)
            depth: Depth map (H×W float32)

        Returns:
            List of random masks (for testing only)
        """
        np.random.seed(self.seed)
        h, w = rgb.shape[:2]
        num_masks = self.config.get("num_masks", 5)

        masks = []
        for _ in range(num_masks):
            # Random mask
            mask = np.random.rand(h, w) > 0.7
            if mask.sum() > 100:  # Only keep if non-trivial
                masks.append(mask)

        return masks


def get_backend_runner(
    backend_name: str,
    config: Dict[str, Any],
    mock: bool = False,
) -> BackendRunner:
    """Factory function for backend runners.

    Args:
        backend_name: "vitb" or "nanosam"
        config: Backend configuration
        mock: If True, return mock runner for testing

    Returns:
        BackendRunner instance
    """
    if mock:
        return MockBackendRunner(backend_name, config)

    if backend_name == "vitb":
        return ViTBRunner(backend_name, config)
    elif backend_name == "nanosam":
        return NanoSAMRunner(backend_name, config)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
