"""SAM/RAP/VLM backend adapters for the Phase 1 object-detection node.

Two profiles are supported by configuration:

``dummy``
    Fully deterministic, dependency-light path for development PCs.

``real_orin`` / ``real``
    Deployment path intended for Jetson Orin/Thor.  It mirrors the baseline
    framework: SAM produces masks, VisualRAP performs CLIP + Chroma retrieval,
    and an OpenAI-compatible VLM endpoint identifies persistent unknown tracks.

The real adapters are optional at import time.  Missing heavy dependencies do
not break the ROS package build.  They are loaded only when the real profile is
selected.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nodes.support.phase1.vlm_result import validate_vlm_response


def _effective_sam_min_mask_pixels(config: Any) -> int:
    """Return min mask area in the image actually sent to SAM.

    ``sam_min_mask_pixels`` is specified in the original input image scale.
    If Phase 1 downsizes the image before SAM, the SAM backend must use a
    correspondingly smaller area threshold; otherwise small but valid objects
    are removed inside the SAM generator before the node can upscale masks back
    to the original image size.
    """
    scale = float(getattr(config, "sam_input_scale_ratio", 1.0) or 1.0)
    return max(1, int(round(float(config.sam_min_mask_pixels) * scale * scale)))


@dataclass
class SamMask:
    """Raw mask candidate produced by a SAM-like backend."""

    mask_id: str
    mask: np.ndarray
    bbox_2d: list[int]
    area_px: int
    crop: Optional[Any] = None
    score: float = 0.0
    metadata: Dict[str, Any] | None = None


@dataclass
class RapMatch:
    """RAP retrieval result for one object cutout/mask."""

    label: str
    confidence: float
    is_known: bool
    distance: float
    metadata: Dict[str, Any]


class DummySamBackend:
    """Deterministic segmentation backend for development PCs."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def segment(self, rgb: np.ndarray) -> List[SamMask]:
        """Generate segmentation masks for one RGB image."""
        height, width = rgb.shape[:2]
        masks: List[SamMask] = []
        count = min(int(self.config.sam_dummy_num_masks), int(self.config.sam_max_masks))
        if count <= 0:
            return masks

        for idx in range(count):
            box_w = max(20, width // 5)
            box_h = max(20, height // 5)
            if idx == 0:
                x0 = max(0, width // 2 - box_w // 2)
                y0 = max(0, height // 2 - box_h // 2)
            else:
                x0 = int((idx + 1) * width / (count + 2) - box_w / 2)
                y0 = int((idx + 1) * height / (count + 2) - box_h / 2)
                x0 = min(max(0, x0), max(0, width - box_w))
                y0 = min(max(0, y0), max(0, height - box_h))
            x1 = min(width, x0 + box_w)
            y1 = min(height, y0 + box_h)
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
            area = int(np.count_nonzero(mask))
            if area >= _effective_sam_min_mask_pixels(self.config):
                masks.append(
                    SamMask(
                        mask_id=f"mask_{idx:03d}",
                        mask=mask,
                        bbox_2d=[x0, y0, x1 - x0, y1 - y0],
                        area_px=area,
                        crop=rgb[y0:y1, x0:x1].copy(),
                        score=1.0,
                        metadata={"backend": "dummy"},
                    )
                )
        return masks


class RealSamBackend:
    """Segment Anything backend compatible with the baseline SamSegmenter.

    The adapter first tries to import the baseline ``scripts.SamSegmenter`` if
    available.  If that module is not on PYTHONPATH, it falls back to a native
    implementation using ``segment_anything`` directly.  This makes deployment
    possible either with the original baseline repository installed or with only
    the required Python dependencies installed on the Jetson.
    """

    CHECKPOINT_URLS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    CHECKPOINT_NAMES = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    def __init__(self, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.backend_name = "real_sam"
        self._baseline_segmenter = None
        self._mask_generator = None
        self._init_backend()

    def _init_backend(self) -> None:
        # Try baseline class first.  It uses SAM vit_h and returns dicts with
        # crop, bbox, mask, and score.
        for module_name in ("scripts.SamSegmenter", "SamSegmenter"):
            try:
                module = __import__(module_name, fromlist=["SamSegmenter"])
                SamSegmenter = getattr(module, "SamSegmenter")
                kwargs = {
                    "checkpoint_path": (
                        str(Path(str(self.config.sam_checkpoint_path)).expanduser().resolve())
                        if self.config.sam_checkpoint_path
                        else None
                    ),
                    "device": self.config.sam_device or None,
                    "mask_threshold": self.config.sam_mask_threshold,
                    "padding": self.config.sam_padding,
                    "min_segment_pixels": _effective_sam_min_mask_pixels(self.config),
                    "points_per_side": self.config.sam_points_per_side,
                    "pred_iou_thresh": self.config.sam_pred_iou_thresh,
                }
                self._baseline_segmenter = SamSegmenter(**kwargs)
                self.backend_name = f"baseline:{module_name}"
                self.logger.info(f"Using baseline SamSegmenter backend from {module_name}.")
                return
            except Exception as exc:
                last_exc = exc

        # Fall back to direct segment-anything implementation.
        try:
            import torch  # type: ignore
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Real SAM backend requested but neither baseline SamSegmenter nor "
                "segment_anything is available. Install the baseline repo on PYTHONPATH "
                "or install segment-anything + torch."
            ) from exc

        model_type = str(self.config.sam_model_type or "vit_h")
        checkpoint_path = (
            str(Path(str(self.config.sam_checkpoint_path)).expanduser().resolve())
            if self.config.sam_checkpoint_path
            else self._resolve_checkpoint(model_type)
        )
        device = self.config.sam_device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=int(self.config.sam_points_per_side),
            pred_iou_thresh=float(self.config.sam_pred_iou_thresh),
            stability_score_thresh=float(self.config.sam_mask_threshold),
            min_mask_region_area=_effective_sam_min_mask_pixels(self.config),
        )
        self.backend_name = f"segment_anything:{model_type}"
        self.logger.info(f"Using native Segment Anything backend: model_type={model_type}, device={device}.")

    def _resolve_checkpoint(self, model_type: str) -> str:
        cache_dir = Path(str(self.config.sam_checkpoint_cache_dir)).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = self.CHECKPOINT_NAMES.get(model_type, self.CHECKPOINT_NAMES["vit_h"])
        checkpoint_path = cache_dir / checkpoint_name
        if checkpoint_path.exists():
            return str(checkpoint_path)
        if not bool(self.config.sam_auto_download):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}. Set phase1.sam.checkpoint_path "
                "or enable phase1.sam.auto_download."
            )
        url = self.CHECKPOINT_URLS.get(model_type, self.CHECKPOINT_URLS["vit_h"])
        self.logger.warn(f"Downloading SAM checkpoint {model_type} to {checkpoint_path}. This can take several minutes.")
        urllib.request.urlretrieve(url, checkpoint_path)
        return str(checkpoint_path)

    def segment(self, rgb: np.ndarray) -> List[SamMask]:
        """Generate segmentation masks for one RGB image."""
        if self._baseline_segmenter is not None:
            try:
                from PIL import Image  # type: ignore
                detections = self._baseline_segmenter.segment(Image.fromarray(rgb))
            except Exception as exc:
                raise RuntimeError(f"Baseline SamSegmenter failed during segment(): {exc}") from exc
            return self._convert_baseline_detections(rgb, detections)

        if self._mask_generator is None:
            return []
        mask_dicts = self._mask_generator.generate(rgb)
        masks: List[SamMask] = []
        h, w = rgb.shape[:2]
        for idx, item in enumerate(mask_dicts[: int(self.config.sam_max_masks)]):
            seg = np.asarray(item.get("segmentation"), dtype=bool)
            bbox = item.get("bbox", [0, 0, 0, 0])
            x, y, bw, bh = [int(v) for v in bbox]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x0 + max(0, bw)), min(h, y0 + max(0, bh))
            area = int(np.count_nonzero(seg))
            if area < _effective_sam_min_mask_pixels(self.config):
                continue
            masks.append(
                SamMask(
                    mask_id=f"mask_{idx:03d}",
                    mask=seg,
                    bbox_2d=[x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                    area_px=area,
                    crop=rgb[y0:y1, x0:x1].copy() if x1 > x0 and y1 > y0 else None,
                    score=float(item.get("stability_score", item.get("predicted_iou", 0.0)) or 0.0),
                    metadata={"backend": self.backend_name},
                )
            )
        return masks

    @staticmethod
    def _convert_baseline_detections(rgb: np.ndarray, detections: List[Dict[str, Any]]) -> List[SamMask]:
        h, w = rgb.shape[:2]
        masks: List[SamMask] = []
        for idx, det in enumerate(detections):
            raw_bbox = det.get("bbox", [0, 0, 0, 0])
            if len(raw_bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in raw_bbox]
            # Baseline SamSegmenter returns padded bbox as (x1, y1, x2, y2).
            x0, y0 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask = np.asarray(det.get("mask"), dtype=bool)
            area = int(np.count_nonzero(mask)) if mask.size else int(max(0, x2 - x0) * max(0, y2 - y0))
            masks.append(
                SamMask(
                    mask_id=f"mask_{idx:03d}",
                    mask=mask,
                    bbox_2d=[x0, y0, max(0, x2 - x0), max(0, y2 - y0)],
                    area_px=area,
                    crop=det.get("crop", None),
                    score=float(det.get("score", 0.0) or 0.0),
                    metadata={"backend": "baseline_sam"},
                )
            )
        return masks


class NanoSamBackend:
    """NanoSAM TensorRT backend for lightweight prompt-based segmentation.

    NanoSAM is prompt-based; it does not provide a direct drop-in equivalent of
    ``SamAutomaticMaskGenerator``.  To keep the existing Phase 1 interface
    unchanged, this adapter approximates automatic segmentation by sampling a
    small grid of foreground point prompts over the already downscaled/depth-
    filtered SAM input image.

    This is intended for real-time profiling on Jetson Orin.  For best final
    performance, use this backend with a low ``points_per_side`` value or replace
    grid prompts with detector/depth-proposal box prompts.
    """

    def __init__(self, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.backend_name = "nanosam_tensorrt"
        self.predictor = None
        self._init_backend()

    def _init_backend(self) -> None:
        image_encoder = str(getattr(self.config, "sam_nanosam_image_encoder_engine", "") or "").strip()
        mask_decoder = str(getattr(self.config, "sam_nanosam_mask_decoder_engine", "") or "").strip()
        if not image_encoder or not mask_decoder:
            raise FileNotFoundError(
                "NanoSAM backend requested but engine paths are missing. Set "
                "phase1.sam.nanosam.image_encoder_engine and "
                "phase1.sam.nanosam.mask_decoder_engine in rsg_pipeline.yaml."
            )
        image_encoder = str(Path(image_encoder).expanduser().resolve())
        mask_decoder = str(Path(mask_decoder).expanduser().resolve())
        if not Path(image_encoder).exists():
            raise FileNotFoundError(f"NanoSAM image encoder engine not found: {image_encoder}")
        if not Path(mask_decoder).exists():
            raise FileNotFoundError(f"NanoSAM mask decoder engine not found: {mask_decoder}")
        try:
            from nanosam.utils.predictor import Predictor  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "NanoSAM backend requested but the nanosam Python package is not importable. "
                "Install NanoSAM and torch2trt in the same Python environment used by the ROS node; "
                "see README_NANOSAM_ORIN.md in this package."
            ) from exc

        self.predictor = Predictor(
            image_encoder_engine=image_encoder,
            mask_decoder_engine=mask_decoder,
        )
        self.logger.info(
            "Using NanoSAM TensorRT backend: "
            f"image_encoder={image_encoder}, mask_decoder={mask_decoder}, "
            f"grid={int(getattr(self.config, 'sam_nanosam_points_per_side', 3))}x"
            f"{int(getattr(self.config, 'sam_nanosam_points_per_side', 3))}."
        )

    def segment(self, rgb: np.ndarray) -> List[SamMask]:
        """Generate segmentation masks for one RGB image."""
        if self.predictor is None:
            return []
        h, w = rgb.shape[:2]
        if h <= 0 or w <= 0:
            return []

        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError("NanoSAM backend requires Pillow to convert numpy RGB images to PIL images") from exc

        # Encode image once.  Prompt decoder calls are much cheaper than the
        # original SAM image encoder, but the number of sampled prompts still
        # matters for end-to-end latency.
        self.predictor.set_image(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))

        prompts = self._make_grid_prompts(rgb)
        masks: List[SamMask] = []
        min_area = _effective_sam_min_mask_pixels(self.config)
        max_masks = int(self.config.sam_max_masks)
        nms_iou = float(getattr(self.config, "sam_nanosam_nms_iou", 0.70))
        threshold = float(getattr(self.config, "sam_nanosam_mask_threshold", getattr(self.config, "sam_mask_threshold", 0.0)))

        for prompt_idx, (x, y) in enumerate(prompts):
            try:
                raw_mask, score, _ = self.predictor.predict(
                    np.array([[float(x), float(y)]], dtype=np.float32),
                    np.array([1], dtype=np.int64),
                )
            except Exception as exc:
                self.logger.warning(f"NanoSAM prompt failed at ({x}, {y}): {exc}")
                continue

            mask = self._to_bool_mask(raw_mask, threshold=threshold)
            if mask.shape[:2] != (h, w):
                mask = self._resize_mask(mask, w, h)
            area = int(np.count_nonzero(mask))
            if area < min_area:
                continue
            bbox = self._bbox_from_mask(mask)
            if bbox is None:
                continue
            x0, y0, bw, bh = bbox
            masks.append(
                SamMask(
                    mask_id=f"nanosam_{prompt_idx:03d}",
                    mask=mask,
                    bbox_2d=[int(x0), int(y0), int(bw), int(bh)],
                    area_px=area,
                    crop=rgb[y0:y0 + bh, x0:x0 + bw].copy() if bw > 0 and bh > 0 else None,
                    score=self._score_to_float(score),
                    metadata={
                        "backend": self.backend_name,
                        "prompt_type": "foreground_point_grid",
                        "prompt_xy": [int(x), int(y)],
                        "nms_iou": nms_iou,
                    },
                )
            )

        # NanoSAM can generate duplicate masks from neighbouring prompts. Keep
        # higher-score candidates first and retain only non-overlapping masks.
        selected: List[SamMask] = []
        for candidate in sorted(masks, key=lambda item: (float(item.score), int(item.area_px)), reverse=True):
            if any(self._mask_iou(candidate.mask, accepted.mask) > nms_iou for accepted in selected):
                continue
            selected.append(candidate)
            if len(selected) >= max_masks:
                break
        return selected

    def _make_grid_prompts(self, rgb: np.ndarray) -> List[Tuple[int, int]]:
        h, w = rgb.shape[:2]
        pps = max(1, int(getattr(self.config, "sam_nanosam_points_per_side", getattr(self.config, "sam_points_per_side", 3))))
        margin_frac = max(0.0, min(0.45, float(getattr(self.config, "sam_nanosam_point_margin_fraction", 0.08))))
        x_margin = int(round(w * margin_frac))
        y_margin = int(round(h * margin_frac))
        xs = np.linspace(x_margin, max(x_margin, w - 1 - x_margin), pps).astype(np.int32)
        ys = np.linspace(y_margin, max(y_margin, h - 1 - y_margin), pps).astype(np.int32)

        skip_black = bool(getattr(self.config, "sam_nanosam_skip_black_points", True))
        min_luma = int(getattr(self.config, "sam_nanosam_min_prompt_luma", 4))
        prompts: List[Tuple[int, int]] = []
        for y in ys:
            for x in xs:
                x_i, y_i = int(x), int(y)
                if skip_black:
                    pixel = rgb[y_i, x_i]
                    if int(pixel[0]) + int(pixel[1]) + int(pixel[2]) <= min_luma * 3:
                        continue
                prompts.append((x_i, y_i))
        return prompts

    @staticmethod
    def _score_to_float(score: Any) -> float:
        try:
            if hasattr(score, "detach"):
                score = score.detach().cpu().numpy()
            arr = np.asarray(score, dtype=np.float32)
            return float(arr.reshape(-1)[0]) if arr.size else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _to_bool_mask(raw_mask: Any, threshold: float) -> np.ndarray:
        if hasattr(raw_mask, "detach"):
            raw_mask = raw_mask.detach().cpu().numpy()
        arr = np.asarray(raw_mask)
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            # Keep the first candidate if the decoder returns multiple masks.
            arr = arr.reshape((-1,) + arr.shape[-2:])[0]
        if arr.dtype == np.bool_:
            return arr.astype(bool)
        return arr.astype(np.float32) > float(threshold)

    @staticmethod
    def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        try:
            import cv2  # type: ignore
            return cv2.resize(mask.astype(np.uint8), (int(width), int(height)), interpolation=cv2.INTER_NEAREST).astype(bool)
        except Exception:
            y_idx = np.linspace(0, mask.shape[0] - 1, int(height)).astype(np.int64)
            x_idx = np.linspace(0, mask.shape[1] - 1, int(width)).astype(np.int64)
            return mask[np.ix_(y_idx, x_idx)].astype(bool)

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return None
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        return [x0, y0, max(0, x1 - x0), max(0, y1 - y0)]

    @staticmethod
    def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        inter = int(np.count_nonzero(a & b))
        union = int(np.count_nonzero(a | b))
        return float(inter) / float(union) if union > 0 else 0.0


class DummyRapBackend:
    """Deterministic RAP-like retrieval backend."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.known_labels = config.rap_dummy_known_labels or ["dummy_object"]

    def classify(self, rgb: np.ndarray, mask: SamMask, index: int) -> RapMatch:
        """Classify one object crop with the configured retrieval backend."""
        unknown_every = max(1, int(self.config.rap_dummy_unknown_every_n))
        is_unknown = (index % unknown_every) == 0
        if is_unknown or not self.config.rap_enabled:
            return RapMatch(
                label="unknown_object",
                confidence=0.0,
                is_known=False,
                distance=1.0,
                metadata={"backend": "dummy", "decision": "forced_unknown"},
            )
        label = self.known_labels[index % len(self.known_labels)]
        confidence = max(float(self.config.rap_confidence_threshold) + 0.1, 0.5)
        return RapMatch(
            label=label,
            confidence=float(confidence),
            is_known=True,
            distance=max(0.0, 1.0 - float(confidence)),
            metadata={"backend": "dummy", "decision": "known_label"},
        )


class RealRapBackend:
    """Baseline-style VisualRAP adapter using CLIP + ChromaDB."""

    def __init__(self, config: Any, logger: Any) -> None:
        self.config = config
        self.logger = logger
        self.backend_name = "real_rap"
        self._visual_rap = None
        self._init_backend()

    def _init_backend(self) -> None:
        # Prefer the baseline VisualRAP class if the original repo is installed.
        for module_name in ("scripts.VisualRAP", "VisualRAP"):
            try:
                module = __import__(module_name, fromlist=["VisualRAP"])
                VisualRAP = getattr(module, "VisualRAP")
                self._visual_rap = VisualRAP(
                    storage_path=self.config.rap_storage_path,
                    model_name=self.config.rap_model_name,
                    device=self.config.rap_device or None,
                    chroma_host=self.config.rap_chroma_host,
                    chroma_port=int(self.config.rap_chroma_port),
                    auto_start_server=bool(self.config.rap_auto_start_server),
                )
                self.backend_name = f"baseline:{module_name}"
                self.logger.info(f"Using baseline VisualRAP backend from {module_name}.")
                return
            except Exception as exc:
                last_exc = exc

        # Native implementation compatible with the baseline API.
        self._visual_rap = _NativeVisualRAP(
            storage_path=self.config.rap_storage_path,
            model_name=self.config.rap_model_name,
            device=self.config.rap_device or None,
            chroma_host=self.config.rap_chroma_host,
            chroma_port=int(self.config.rap_chroma_port),
            collection_name=self.config.rap_collection_name,
            auto_start_server=bool(self.config.rap_auto_start_server),
            logger=self.logger,
        )
        self.backend_name = "native_visual_rap"
        self.logger.info("Using native VisualRAP-compatible backend.")

    def classify(self, rgb: np.ndarray, mask: SamMask, index: int) -> RapMatch:
        """Classify one object crop with the configured retrieval backend."""
        crop = self._get_crop(rgb, mask)
        if crop is None:
            return RapMatch("unknown_object", 0.0, False, 1.0, {"backend": self.backend_name, "reason": "empty_crop"})
        try:
            result = self._visual_rap.query(image=crop, threshold=float(self.config.rap_distance_threshold))
            if isinstance(result, tuple) and len(result) >= 3:
                label, distance, reference_metadata = result[0], result[1], result[2]
            else:
                label, distance = result
                reference_metadata = {}
            label = str(label).strip()
            distance = float(distance)
        except Exception as exc:
            return RapMatch("unknown_object", 0.0, False, 1.0, {"backend": self.backend_name, "error": str(exc)})
        is_known = bool(label and label.lower() not in {"unknown", "unknown_object", "unclear", "unsure", "n/a"} and distance <= float(self.config.rap_distance_threshold))
        confidence = max(0.0, min(1.0, 1.0 - distance))
        metadata = {"backend": self.backend_name, "distance": distance}
        if isinstance(reference_metadata, dict):
            metadata.update(reference_metadata)
        if "rsg_slot_id" in metadata and "reference_slot_id" not in metadata:
            metadata["reference_slot_id"] = metadata["rsg_slot_id"]
        if not is_known:
            label = "unknown_object"
            confidence = 0.0
        return RapMatch(label, confidence, is_known, distance, metadata)

    def add_image(self, image: Any, label: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store one labelled reference crop in the retrieval backend."""
        if not hasattr(self._visual_rap, "add_image"):
            return
        try:
            self._visual_rap.add_image(image, label, metadata or {})
        except TypeError:
            # Baseline VisualRAP API lacks metadata. It can still learn the
            # label, but cannot later re-identify a specific frozen slot.
            self._visual_rap.add_image(image, label)

    @staticmethod
    def _get_crop(rgb: np.ndarray, mask: SamMask) -> Any:
        if mask.crop is not None:
            return mask.crop
        x, y, w, h = [int(v) for v in mask.bbox_2d]
        if w <= 0 or h <= 0:
            return None
        crop_np = rgb[y:y + h, x:x + w]
        if crop_np.size == 0:
            return None
        try:
            from PIL import Image  # type: ignore
            return Image.fromarray(crop_np)
        except Exception:
            return crop_np


class _NativeVisualRAP:
    """Small local implementation of baseline VisualRAP.

    It uses CLIP from ``transformers`` and ChromaDB's HTTP client.  Training
    images must be loaded once into the ``visual_rag`` collection.  The helper
    script ``rsg_load_rap_memory`` included in this package does that.
    """

    def __init__(
        self,
        storage_path: str,
        model_name: str,
        device: Optional[str],
        chroma_host: str,
        chroma_port: int,
        collection_name: str,
        auto_start_server: bool,
        logger: Any,
    ) -> None:
        try:
            import torch  # type: ignore
            from chromadb import HttpClient  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Real RAP backend requires torch, transformers, and chromadb. "
                "Install them or put the baseline VisualRAP module on PYTHONPATH."
            ) from exc

        self.storage_path = str(Path(storage_path).expanduser().resolve())
        self.chroma_host = chroma_host
        self.chroma_port = int(chroma_port)
        self.collection_name = collection_name
        self.logger = logger
        self.server_process = None
        self._owns_server = False
        if auto_start_server and not self._is_server_running():
            self._start_chroma_server()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.client = HttpClient(host=chroma_host, port=self.chroma_port)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def _is_server_running(self) -> bool:
        try:
            import requests  # type: ignore
            response = requests.get(f"http://{self.chroma_host}:{self.chroma_port}/api/v1/heartbeat", timeout=1)
            return response.status_code == 200
        except Exception:
            return False

    def _start_chroma_server(self) -> None:
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Starting ChromaDB server on port {self.chroma_port}; storage={self.storage_path}")
        self.server_process = subprocess.Popen(
            ["chroma", "run", "--port", str(self.chroma_port), "--path", self.storage_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._owns_server = True
        for _ in range(10):
            if self._is_server_running():
                return
            time.sleep(1.0)
        raise RuntimeError(f"Failed to start ChromaDB server on port {self.chroma_port}")

    def _to_pil(self, image: Any) -> Any:
        if hasattr(image, "mode"):
            return image
        from PIL import Image  # type: ignore
        return Image.fromarray(np.asarray(image, dtype=np.uint8))

    def embed_image(self, image: Any) -> np.ndarray:
        """Create the image embedding used for visual retrieval."""
        import torch  # type: ignore
        pil_image = self._to_pil(image)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.vision_model(**inputs)
            img_emb = self.model.visual_projection(image_features.pooler_output)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb.cpu().numpy().astype("float32")[0]

    def add_image(self, image: Any, label: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store one labelled reference crop in the retrieval backend."""
        import uuid
        emb = self.embed_image(image)
        safe_metadata = {str(k): str(v) for k, v in (metadata or {}).items() if v is not None}
        self.collection.add(
            embeddings=[emb.tolist()],
            documents=[str(label).strip()],
            metadatas=[safe_metadata],
            ids=[str(uuid.uuid4())],
        )

    def query(self, image: Any, top_k: int = 3, threshold: float = 0.3) -> Tuple[str, float, Dict[str, Any]]:
        """Query stored visual references for the supplied crop."""
        emb = self.embed_image(image)
        results = self.collection.query(query_embeddings=[emb.tolist()], n_results=int(top_k), include=["documents", "distances", "metadatas"])
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "unknown", 1.0, {}
        best_distance = float(results.get("distances", [[1.0]])[0][0])
        best_label = str(docs[0]).strip()
        metadatas = results.get("metadatas", [[]])[0]
        metadata = dict(metadatas[0]) if metadatas and isinstance(metadatas[0], dict) else {}
        if "rsg_slot_id" in metadata:
            metadata["reference_slot_id"] = metadata["rsg_slot_id"]
        if best_distance > float(threshold):
            return "unknown", best_distance, metadata
        return best_label, best_distance, metadata


class DummyVlmBackend:
    """Dummy unknown-object identification backend for non-VLM machines."""

    def __init__(self, config: Any) -> None:
        self.config = config
        # Support the baseline environment convention while keeping YAML as the
        # primary source of truth.  If OPENAI_BASE_URL is set, treat it as the
        # OpenAI-compatible /v1 base URL and append /chat/completions.
        env_base = os.environ.get("OPENAI_BASE_URL", "").strip()
        self.endpoint = (env_base.rstrip("/") + "/chat/completions") if env_base else str(config.vlm_endpoint)
        self.model = os.environ.get("MODEL", str(config.vlm_model))

    def identify(self, rgb_crop: Optional[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Request one semantic label from the configured VLM."""
        if float(self.config.vlm_dummy_delay_sec) > 0.0:
            time.sleep(float(self.config.vlm_dummy_delay_sec))
        candidate_id = str(metadata.get("candidate_id", "unknown"))
        suffix = candidate_id.split("_")[-1] if "_" in candidate_id else candidate_id
        return {
            "success": True,
            "label": f"{self.config.vlm_dummy_label_prefix}_{suffix}",
            "confidence": float(self.config.vlm_confidence),
            "label_confidence": float(self.config.vlm_confidence),
            "mobility_class": "unknown",
            "mobility_confidence": 0.0,
            "backend": "dummy",
            "model": "dummy",
            "raw_response": "dummy_vlm_path",
            "validation_status": "dummy",
            "validation_reason": "dummy_backend_has_no_mobility_inference",
        }


class OpenAICompatibleVlmBackend:
    """OpenAI-compatible HTTP adapter for local Qwen-style VLM servers."""

    def __init__(self, config: Any) -> None:
        self.config = config
        # Support the baseline environment convention while keeping YAML as the
        # primary source of truth.  If OPENAI_BASE_URL is set, treat it as the
        # OpenAI-compatible /v1 base URL and append /chat/completions.
        env_base = os.environ.get("OPENAI_BASE_URL", "").strip()
        self.endpoint = (env_base.rstrip("/") + "/chat/completions") if env_base else str(config.vlm_endpoint)
        self.model = os.environ.get("MODEL", str(config.vlm_model))

    def identify(self, rgb_crop: Optional[np.ndarray], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Request and validate one semantic-plus-mobility JSON result."""
        if rgb_crop is None or rgb_crop.size == 0:
            return {
                "success": False,
                "label": "unknown_object",
                "confidence": 0.0,
                "label_confidence": 0.0,
                "mobility_class": "unknown",
                "mobility_confidence": 0.0,
                "backend": self.config.vlm_mode,
                "model": self.config.vlm_model,
                "raw_response": "empty_crop",
                "failure_reason": "empty_crop",
                "validation_status": "rejected",
                "validation_reason": "empty_crop",
            }
        try:
            image_url = self._crop_to_data_url(rgb_crop)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.config.vlm_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": int(self.config.vlm_max_tokens),
                "temperature": float(self.config.vlm_temperature),
                # Qwen3 / Qwen3.5 are reasoning models: left on "auto" they emit
                # <think>...</think> first and, with a small max_tokens, return
                # empty message.content (all budget spent thinking). Force it off.
                # No-op for non-thinking models (Qwen3-VL-8B, R1-R3).
                "chat_template_kwargs": {"enable_thinking": False},
            }
            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            api_key = str(self.config.vlm_api_key or "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=float(self.config.vlm_timeout_sec)) as response:
                body = response.read().decode("utf-8")
            result = json.loads(body)
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "unknown_object")
            # llama.cpp reports model compute time in `timings` (prompt + generation).
            _t = result.get("timings") or {}
            _inference_ms = float(_t.get("prompt_ms", 0.0) or 0.0) + float(_t.get("predicted_ms", 0.0) or 0.0)
            validated = validate_vlm_response(
                text,
                min_label_confidence=float(self.config.vlm_result_min_label_confidence),
                min_mobility_confidence=float(self.config.vlm_result_min_mobility_confidence),
                dynamic_label_hints=self.config.vlm_dynamic_label_hints,
                static_label_hints=self.config.vlm_static_label_hints,
            )
            output = validated.as_dict()
            output.update({
                "backend": self.config.vlm_mode,
                "model": self.model,
                "raw_response": text,
                "vlm_inference_ms": _inference_ms,
                "failure_reason": "" if validated.success else validated.validation_reason,
            })
            return output
        except Exception as exc:
            return {
                "success": False,
                "label": "unknown_object",
                "confidence": 0.0,
                "label_confidence": 0.0,
                "mobility_class": "unknown",
                "mobility_confidence": 0.0,
                "backend": self.config.vlm_mode,
                "model": self.model,
                "raw_response": str(exc),
                "failure_reason": f"request_error:{type(exc).__name__}",
                "validation_status": "rejected",
                "validation_reason": f"request_error:{type(exc).__name__}",
            }

    def _crop_to_data_url(self, rgb_crop: np.ndarray) -> str:
        try:
            import cv2  # type: ignore
            bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.vlm_jpeg_quality)])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            from PIL import Image  # type: ignore
            stream = BytesIO()
            Image.fromarray(rgb_crop).save(stream, format="PNG")
            b64 = base64.b64encode(stream.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"

def _allow_fallback(config: Any) -> bool:
    return bool(getattr(config, "allow_dummy_fallback", False))


def make_sam_backend(config: Any, logger: Any) -> Any:
    """Create the configured SAM-compatible segmentation backend."""
    backend = str(config.sam_backend).strip().lower()
    if backend in {"real", "baseline", "sam", "segment_anything", "orin"}:
        try:
            return RealSamBackend(config, logger)
        except Exception as exc:
            if _allow_fallback(config):
                logger.error(f"Real SAM backend failed; falling back to dummy SAM because allow_dummy_fallback=true. Reason: {exc}")
                return DummySamBackend(config)
            raise
    if backend in {"nanosam", "nano_sam", "nano", "trt_sam"}:
        try:
            return NanoSamBackend(config, logger)
        except Exception as exc:
            if _allow_fallback(config):
                logger.error(f"NanoSAM backend failed; falling back to dummy SAM because allow_dummy_fallback=true. Reason: {exc}")
                return DummySamBackend(config)
            raise
    if backend in {"dummy", "mock", "test"}:
        return DummySamBackend(config)
    raise ValueError(
        f"Unsupported phase1.sam.backend='{config.sam_backend}'. "
        "Supported values: real, nanosam, dummy."
    )


def make_rap_backend(config: Any, logger: Any) -> Any:
    """Create the configured visual RAP retrieval backend."""
    backend = str(config.rap_backend).lower()
    if backend in {"real", "baseline", "visual_rap", "vrap", "orin"}:
        try:
            return RealRapBackend(config, logger)
        except Exception as exc:
            if _allow_fallback(config):
                logger.error(f"Real RAP backend failed; falling back to dummy RAP because allow_dummy_fallback=true. Reason: {exc}")
                return DummyRapBackend(config)
            raise
    return DummyRapBackend(config)


def make_vlm_backend(config: Any) -> Any:
    """Create the configured VLM backend."""
    mode = str(config.vlm_mode).lower()
    if mode in {"openai", "openai_vision", "qwen", "qwen_http", "real", "orin"}:
        return OpenAICompatibleVlmBackend(config)
    return DummyVlmBackend(config)
