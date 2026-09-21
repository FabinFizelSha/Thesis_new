"""Full-frame diagnostic: every SAM mask for a frame, overlaid on the raw RGB image.

Unlike the per-track crop diagnostics (periodic_crop_diagnostics.py,
tracking_crop_manager.py), which each show one object's own cropped region,
this saves the whole frame with every mask SAM produced that frame drawn on
top -- for answering "did SAM detect this object at all, and when" without
having to already know which track it ended up under.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from nodes.support.workspace_paths import workspace_path


class FrameMaskOverlayDiagnostics:
    """Save every Nth frame with all of that frame's SAM masks outlined."""

    def __init__(self, output_dir: Optional[str] = None, enabled: bool = True, interval: int = 10):
        self.enabled = bool(enabled)
        self.interval = max(1, int(interval))
        self.output_dir = Path(output_dir or str(workspace_path("debug", "frame_mask_overlays")))
        self._rng = random.Random(0)  # deterministic mask_id -> color across a run
        self._colors: dict = {}
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _color_for(self, key: str) -> tuple:
        color = self._colors.get(key)
        if color is None:
            color = (
                self._rng.randint(60, 255),
                self._rng.randint(60, 255),
                self._rng.randint(60, 255),
            )
            self._colors[key] = color
        return color

    def log_frame(
        self,
        frame_number: int,
        rgb: Optional[np.ndarray],
        masks: List[Any],
        track_ids: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Save this frame with every mask in ``masks`` outlined, if on-interval.

        Args:
            frame_number: sequence number used for the interval check and filename.
            rgb: full-frame RGB image (not cropped).
            masks: objects exposing ``.mask_id`` (str) and ``.mask`` (2D bool array
                in full-frame coordinates) -- e.g. the SamMask list for this frame,
                before or after classification.
            track_ids: optional ``mask_id -> persistent track_id`` map, so each
                mask's label also shows which track it was assigned to (e.g.
                "rsg_obj_000015"). A mask with no entry (filtered out before
                track association ran) is labelled with its mask_id alone.
        """
        if not self.enabled or rgb is None:
            return None
        if frame_number % self.interval != 0:
            return None
        try:
            overlay = cv2.cvtColor(np.array(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            for item in masks:
                mask = getattr(item, "mask", None)
                if mask is None or not np.any(mask):
                    continue
                mask_u8 = (np.asarray(mask, dtype=np.uint8)) * 255
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask_id = str(getattr(item, "mask_id", id(item)))
                color = self._color_for(mask_id)
                cv2.drawContours(overlay, contours, -1, color, 2)
                ys, xs = np.where(mask)
                if ys.size:
                    track_id = (track_ids or {}).get(mask_id)
                    label = f"{mask_id} {track_id}" if track_id else mask_id
                    cv2.putText(
                        overlay, label, (int(xs.min()), max(12, int(ys.min()) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
                    )
            filename = f"frame_{frame_number:06d}_masks{len(masks)}.jpg"
            path = self.output_dir / filename
            cv2.imwrite(str(path), overlay)
            return str(path)
        except Exception:
            return None
