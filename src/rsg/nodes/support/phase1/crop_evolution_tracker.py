"""Comprehensive diagnostic tracking of crop and semantic slot evolution over time."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class CropEvent:
    """Single event in a crop's lifecycle."""
    timestamp_iso: str
    frame_id: str
    sequence: int
    event_type: str  # sam_output, nms_filtered, track_assigned, crop_updated, label_assigned, slot_allocated
    track_id: Optional[str] = None
    slot_id: Optional[int] = None
    mask_index: Optional[int] = None
    mask_area_px: Optional[float] = None
    mask_centroid_2d: Optional[List[float]] = None
    mask_centroid_3d: Optional[List[float]] = None
    iou_score: Optional[float] = None
    depth_mean: Optional[float] = None
    predicted_label: Optional[str] = None
    label_confidence: Optional[float] = None
    mobility_class: Optional[str] = None
    reason: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None


class CropEvolutionTracker:
    """Track the lifecycle of every crop: from SAM output through tracking to final slot allocation."""

    def __init__(self, enabled: bool, output_dir: str, logger: Any) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.logger = logger
        self.events: List[CropEvent] = []
        self._lock = Lock()
        self.frame_count = 0
        self.crop_count = 0

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Crop evolution tracking enabled. Output: {self.output_dir}")
        else:
            self.logger.info("Crop evolution tracking disabled.")

    def log_sam_output(
        self,
        frame_id: str,
        sequence: int,
        mask_index: int,
        mask_area_px: float,
        centroid_2d: List[float],
        centroid_3d: List[float],
        depth_mean: float,
    ) -> None:
        """Log SAM mask output before NMS."""
        if not self.enabled:
            return
        event = CropEvent(
            timestamp_iso=datetime.utcnow().isoformat(),
            frame_id=frame_id,
            sequence=sequence,
            event_type="sam_output",
            mask_index=mask_index,
            mask_area_px=mask_area_px,
            mask_centroid_2d=centroid_2d,
            mask_centroid_3d=centroid_3d,
            depth_mean=depth_mean,
        )
        with self._lock:
            self.events.append(event)

    def log_nms_filtered(
        self,
        frame_id: str,
        sequence: int,
        mask_indices_kept: List[int],
        mask_indices_dropped: List[int],
    ) -> None:
        """Log NMS filtering results."""
        if not self.enabled:
            return
        for idx in mask_indices_dropped:
            event = CropEvent(
                timestamp_iso=datetime.utcnow().isoformat(),
                frame_id=frame_id,
                sequence=sequence,
                event_type="nms_filtered",
                mask_index=idx,
                reason="nms_overlap",
            )
            with self._lock:
                self.events.append(event)

    def log_track_assignment(
        self,
        frame_id: str,
        sequence: int,
        track_id: str,
        mask_index: int,
        mask_area_px: float,
        centroid_2d: List[float],
        centroid_3d: List[float],
        iou_score: float,
        reason: str,
    ) -> None:
        """Log when a SAM mask is assigned to a track."""
        if not self.enabled:
            return
        event = CropEvent(
            timestamp_iso=datetime.utcnow().isoformat(),
            frame_id=frame_id,
            sequence=sequence,
            event_type="track_assigned",
            track_id=track_id,
            mask_index=mask_index,
            mask_area_px=mask_area_px,
            mask_centroid_2d=centroid_2d,
            mask_centroid_3d=centroid_3d,
            iou_score=iou_score,
            reason=reason,
        )
        with self._lock:
            self.events.append(event)

    def log_crop_update(
        self,
        frame_id: str,
        sequence: int,
        track_id: str,
        slot_id: Optional[int],
        mask_area_px: float,
        centroid_2d: List[float],
        centroid_3d: List[float],
        depth_mean: float,
        reason: str,
    ) -> None:
        """Log when a track's crop is updated."""
        if not self.enabled:
            return
        event = CropEvent(
            timestamp_iso=datetime.utcnow().isoformat(),
            frame_id=frame_id,
            sequence=sequence,
            event_type="crop_updated",
            track_id=track_id,
            slot_id=slot_id,
            mask_area_px=mask_area_px,
            mask_centroid_2d=centroid_2d,
            mask_centroid_3d=centroid_3d,
            depth_mean=depth_mean,
            reason=reason,
        )
        with self._lock:
            self.events.append(event)

    def log_label_assigned(
        self,
        frame_id: str,
        sequence: int,
        track_id: str,
        slot_id: int,
        predicted_label: str,
        label_confidence: float,
        mobility_class: str,
        source: str,
    ) -> None:
        """Log when a semantic label is assigned to a track."""
        if not self.enabled:
            return
        event = CropEvent(
            timestamp_iso=datetime.utcnow().isoformat(),
            frame_id=frame_id,
            sequence=sequence,
            event_type="label_assigned",
            track_id=track_id,
            slot_id=slot_id,
            predicted_label=predicted_label,
            label_confidence=label_confidence,
            mobility_class=mobility_class,
            reason=source,
        )
        with self._lock:
            self.events.append(event)

    def log_slot_allocated(
        self,
        frame_id: str,
        sequence: int,
        track_id: str,
        slot_id: int,
        reason: str,
    ) -> None:
        """Log when a track is allocated a Hydra semantic slot."""
        if not self.enabled:
            return
        event = CropEvent(
            timestamp_iso=datetime.utcnow().isoformat(),
            frame_id=frame_id,
            sequence=sequence,
            event_type="slot_allocated",
            track_id=track_id,
            slot_id=slot_id,
            reason=reason,
        )
        with self._lock:
            self.events.append(event)

    def save_snapshot(self, suffix: str = "") -> None:
        """Save all tracked events to JSONL file."""
        if not self.enabled:
            return
        with self._lock:
            if not self.events:
                return
            snapshot = list(self.events)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"crop_evolution_{timestamp}{suffix}.jsonl"
        filepath = self.output_dir / filename

        try:
            with open(filepath, "w") as f:
                for event in snapshot:
                    f.write(json.dumps(asdict(event)) + "\n")
            self.logger.info(f"Saved {len(snapshot)} crop evolution events to {filepath}")
        except Exception as exc:
            self.logger.error(f"Failed to save crop evolution snapshot: {exc}")

    def save_analysis(self) -> None:
        """Generate human-readable analysis of crop lifecycles."""
        if not self.enabled or not self.events:
            return

        with self._lock:
            events = list(self.events)

        # Group by track_id
        tracks: Dict[str, List[CropEvent]] = {}
        for event in events:
            if event.track_id:
                if event.track_id not in tracks:
                    tracks[event.track_id] = []
                tracks[event.track_id].append(event)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"crop_evolution_analysis_{timestamp}.md"

        try:
            with open(filepath, "w") as f:
                f.write("# Crop Evolution Analysis\n\n")
                f.write(f"Total tracks observed: {len(tracks)}\n")
                f.write(f"Total events logged: {len(events)}\n\n")

                # Sort by track_id for readability
                for track_id in sorted(tracks.keys()):
                    track_events = tracks[track_id]
                    f.write(f"## Track: {track_id}\n\n")
                    f.write(f"Events: {len(track_events)}\n\n")

                    # Summarize lifecycle
                    first_frame = min(e.sequence for e in track_events)
                    last_frame = max(e.sequence for e in track_events)
                    frames_spanned = last_frame - first_frame + 1

                    f.write(f"- First seen: Frame {first_frame}\n")
                    f.write(f"- Last seen: Frame {last_frame}\n")
                    f.write(f"- Lifespan: {frames_spanned} frames\n\n")

                    # Event timeline
                    f.write("### Event Timeline\n\n")
                    for event in sorted(track_events, key=lambda e: e.sequence):
                        f.write(f"- **[{event.sequence:04d}]** {event.event_type}")
                        if event.slot_id is not None:
                            f.write(f" → Slot {event.slot_id}")
                        if event.predicted_label:
                            f.write(
                                f" | Label: {event.predicted_label} ({event.label_confidence:.2f})"
                            )
                        if event.reason:
                            f.write(f" | {event.reason}")
                        f.write("\n")

                    f.write("\n")

            self.logger.info(f"Saved crop evolution analysis to {filepath}")
        except Exception as exc:
            self.logger.error(f"Failed to save crop evolution analysis: {exc}")
