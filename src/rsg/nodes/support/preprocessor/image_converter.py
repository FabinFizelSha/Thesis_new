"""Lean RGB and depth forwarding helpers for the RSG preprocessor.

The default path performs no image conversion, depth masking, or NumPy image
processing. The current uHumans2 bag already provides ``rgb8`` RGB and
``32FC1`` metric depth, so both payloads are forwarded by reference with only
header metadata replaced.

An optional single-pass depth-processing path remains available for future
bags that require unit/encoding conversion or sampled depth-validity metrics.
It is disabled by default in ``rsg_pipeline.yaml``.
"""

from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from nodes.support.preprocessor.config_loader import PreprocessorConfig


class ImageConverter:
    """Forward compatible images and optionally process depth in one pass."""

    def __init__(self, bridge: CvBridge, config: PreprocessorConfig) -> None:
        self.bridge = bridge
        self.config = config

    def _shallow_image_reference(self, msg: Image, frame_id: str) -> Image:
        """Return a new image message that reuses the original data buffer."""
        out = Image()
        out.header = copy.copy(msg.header)
        out.header.frame_id = frame_id
        out.height = msg.height
        out.width = msg.width
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = msg.step
        out.data = msg.data
        return out

    def forward_rgb(self, rgb_msg: Image) -> Image:
        """Forward an already-normalized ``rgb8`` image without conversion.

        RGB conversion code is intentionally absent from this optimized
        preprocessor. A mismatched encoding is rejected explicitly instead of
        silently applying an expensive color conversion in the hot path.
        """
        expected = self.config.output_rgb_encoding
        if rgb_msg.encoding != expected:
            raise ValueError(
                f"RGB passthrough requires encoding {expected!r}, "
                f"received {rgb_msg.encoding!r}"
            )
        return self._shallow_image_reference(rgb_msg, self.config.camera_frame)

    def prepare_depth(self, depth_msg: Image, sequence: int) -> Tuple[Image, float]:
        """Forward depth or run the optional single-pass depth operation.

        When ``single_pass_depth_processing_enabled`` is false, this method
        only verifies the expected encoding and forwards the original payload.
        No range masking or invalid-depth calculation occurs.

        When enabled, supported input depth is normalized to configured metric
        output in one materialization. A sampled invalid-depth ratio is
        calculated only on configured frames and only when its gate is enabled.
        Depth values are never range-masked in this preprocessor.
        """
        if not self.config.single_pass_depth_processing_enabled:
            expected = self.config.output_depth_encoding
            if depth_msg.encoding != expected:
                raise ValueError(
                    f"Depth passthrough requires encoding {expected!r}, "
                    f"received {depth_msg.encoding!r}. Enable "
                    "single_pass_depth_processing_enabled for conversion."
                )
            return self._shallow_image_reference(depth_msg, self.config.camera_frame), -1.0

        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_msg.encoding == "16UC1":
            depth_m = depth_raw.astype(np.float32)
            depth_m *= self.config.depth_scale_to_meter
        elif depth_msg.encoding == "32FC1":
            depth_m = depth_raw.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported depth encoding: {depth_msg.encoding}")

        invalid_ratio = -1.0
        if self._should_compute_invalid_ratio(sequence):
            invalid_ratio = self.compute_invalid_depth_ratio(depth_m)

        target_encoding = self.config.output_depth_encoding
        if depth_msg.encoding == target_encoding and target_encoding == "32FC1":
            depth_out = self._shallow_image_reference(depth_msg, self.config.camera_frame)
        else:
            if target_encoding != "32FC1":
                raise ValueError(
                    "Optional single-pass depth processing currently supports "
                    "only output_depth_encoding='32FC1'"
                )
            depth_out = self.bridge.cv2_to_imgmsg(depth_m, encoding=target_encoding)
            depth_out.header = copy.copy(depth_msg.header)
            depth_out.header.frame_id = self.config.camera_frame

        return depth_out, invalid_ratio

    def _should_compute_invalid_ratio(self, sequence: int) -> bool:
        """Return whether this frame should calculate depth validity."""
        if not self.config.compute_invalid_depth_ratio:
            return False
        every_n = max(1, int(self.config.invalid_depth_ratio_every_n_frames))
        return sequence % every_n == 0

    def compute_invalid_depth_ratio(self, depth_m: np.ndarray) -> float:
        """Calculate sampled invalid-depth ratio without modifying depth."""
        stride = max(1, int(self.config.depth_check_stride))
        sample = depth_m[::stride, ::stride] if stride > 1 else depth_m
        valid = (
            np.isfinite(sample)
            & (sample >= self.config.min_depth_m)
            & (sample <= self.config.max_depth_m)
        )
        if valid.size == 0:
            return 1.0
        return 1.0 - float(np.count_nonzero(valid)) / float(valid.size)
