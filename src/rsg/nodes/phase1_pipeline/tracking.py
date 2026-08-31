"""Object Tracking pipeline stage."""

from typing import Any, Dict, List, Optional
import numpy as np
from rsg.msg import RsgFrame
from nodes.support.phase1.persistent_object_tracker import PersistentObjectTracker
from nodes.support.phase1.time_utils import stamp_to_float
from nodes.support.phase1.object_geometry import filter_metadata


class TrackingStage:
    """Wraps persistent object tracking logic."""

    def __init__(
        self,
        tracker: PersistentObjectTracker,
        config: Any,
        logger: Any,
        geometry_estimator: Any = None,
    ):
        """Initialize tracking stage.

        Args:
            tracker: PersistentObjectTracker instance
            config: Phase1 configuration
            logger: ROS logger
            geometry_estimator: ObjectGeometryEstimator for metadata building
        """
        self.tracker = tracker
        self.config = config
        self.logger = logger
        self.geometry_estimator = geometry_estimator

    def associate(
        self,
        masks: List[Any],
        frame: RsgFrame,
        timestamp_sec: float,
        coordinator: Any = None,
    ) -> List[Dict[str, Any]]:
        """Associate masks with existing tracks.

        Args:
            masks: List of detected masks
            frame: Input RsgFrame
            timestamp_sec: Frame timestamp
            coordinator: Optional coordinator reference for diagnostics

        Returns:
            List of track records
        """
        track_records = []

        for mask_idx, mask in enumerate(masks):
            # Build metadata for this mask
            metadata = self._build_mask_metadata(mask, frame)

            # Associate with existing track or create new
            metadata_out, track_record = self.tracker.associate(
                metadata=metadata,
                frame_id=frame.id,
                sequence=frame.sequence,
                timestamp_sec=timestamp_sec,
                desired_hydra_label_id=0,
                desired_hydra_label_name="unknown",
                raw_label=None,
                label_source="sam_prior",
                label_confidence=0.0,
            )

            # Update metadata with tracking result
            metadata_out.update(track_record)
            track_records.append(metadata_out)

        return track_records

    def _build_mask_metadata(self, mask: Any, frame: RsgFrame) -> Dict[str, Any]:
        """Build metadata dictionary for a mask."""
        return {
            "mask": mask.mask if hasattr(mask, "mask") else None,
            "bbox_2d": mask.bbox if hasattr(mask, "bbox") else None,
            "centroid_3d": mask.centroid_3d if hasattr(mask, "centroid_3d") else None,
            "bbox_3d_min": mask.bbox_3d_min if hasattr(mask, "bbox_3d_min") else None,
            "bbox_3d_max": mask.bbox_3d_max if hasattr(mask, "bbox_3d_max") else None,
            "volume": mask.volume if hasattr(mask, "volume") else 0.0,
            "area_px": mask.area_px if hasattr(mask, "area_px") else 0,
            "confidence": mask.confidence if hasattr(mask, "confidence") else 1.0,
        }

    def build_object_metadata(
        self,
        frame: RsgFrame,
        mask: Any,
        depth: np.ndarray,
        tx: np.ndarray,
        rot_m: np.ndarray,
        label: str,
        label_id: int,
        instance_id: int,
        confidence: float,
        status: str,
        candidate_id: str,
        rap_metadata: Dict[str, Any],
        geometry_stage_ms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create configurable object metadata used by Hydra/fusion/risk nodes."""
        if self.geometry_estimator is None:
            return {"error": "geometry_estimator not available"}

        geometry = self.geometry_estimator.estimate(
            mask.mask if hasattr(mask, "mask") else mask,
            depth,
            frame.camera_info,
            tx,
            rot_m,
            stage_ms=geometry_stage_ms,
        )
        metadata = {
            "source_frame_id": frame.rsg_frame_id,
            "timestamp_sec": stamp_to_float(frame.header.stamp),
            "candidate_id": candidate_id,
            "mask_id": mask.mask_id if hasattr(mask, "mask_id") else "",
            "label": label,
            "label_id": int(label_id),
            "instance_id": int(instance_id),
            "confidence": float(confidence),
            "status": status,
            "rap": rap_metadata,
            **geometry,
        }
        if self.config.persistent_tracking_enabled:
            return metadata
        return filter_metadata(metadata, self.config)

    def update_track_observations(self, track_records: List[Dict[str, Any]]) -> None:
        """Update track observations after association."""
        # This would be called after metadata is finalized
        for record in track_records:
            track_id = record.get("persistent_track_id")
            if track_id and track_id in self.tracker._tracks:
                track = self.tracker._tracks[track_id]
                # Update track with final metadata
                if "centroid_3d" in record:
                    track.centroid_3d = record["centroid_3d"]
                if "bbox_3d_min" in record and "bbox_3d_max" in record:
                    track.bbox_3d_min = record["bbox_3d_min"]
                    track.bbox_3d_max = record["bbox_3d_max"]
