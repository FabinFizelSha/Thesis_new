"""Dict-shaping helpers for PersistentObjectTracker's external contract.

Pure functions of the passed-in track/segment -- no tracker state (config,
lock, `_tracks`) is needed, so these take the dataclass instances directly
rather than a tracker reference.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from nodes.support.phase1.tracker_geometry import _aabb_xy_diagonal, _as_list


def segment_record(segment: Any) -> Dict[str, Any]:
    span = None
    if segment.bbox_3d_min is not None and segment.bbox_3d_max is not None:
        span = _aabb_xy_diagonal(segment.bbox_3d_min, segment.bbox_3d_max)
    return {
        "local_segment_id": str(segment.segment_id),
        "semantic_segment_id": str(segment.segment_id),
        "hydra_slot_id": int(segment.hydra_label_id),
        "hydra_slot_name": str(segment.hydra_label_name),
        "hydra_label_id": int(segment.hydra_label_id),
        "hydra_label_name": str(segment.hydra_label_name),
        "instance_id": int(segment.instance_id),
        "seen_count": int(segment.seen_count),
        "first_seen_timestamp_sec": float(segment.first_seen_timestamp_sec),
        "last_seen_timestamp_sec": float(segment.last_seen_timestamp_sec),
        "centroid_3d": _as_list(segment.centroid_3d),
        "bbox_2d": segment.bbox_2d,
        "bbox_3d_min": _as_list(segment.bbox_3d_min),
        "bbox_3d_max": _as_list(segment.bbox_3d_max),
        "last_bbox_3d_min": _as_list(segment.last_bbox_3d_min),
        "last_bbox_3d_max": _as_list(segment.last_bbox_3d_max),
        "local_segment_xy_span_m": span,
        "closed": bool(segment.closed),
    }


def track_record(
    track: Any,
    event: str,
    reason: str,
    score: Optional[float],
) -> Dict[str, Any]:
    active_segment = track.segments.get(int(track.active_segment_slot_id))
    active_segment_record = segment_record(active_segment) if active_segment is not None else {}
    all_segment_records = [segment_record(segment) for segment in track.segments.values()]
    return {
        "event": event,
        "track_event": event,
        "persistent_track_event": event,
        "persistent_track_id": track.track_id,
        "internal_object_id": track.track_id,
        "persistent_instance_id": int(track.instance_id),
        "local_segment_id": f"{track.track_id}:slot_{int(track.hydra_label_id)}",
        "semantic_segment_id": f"{track.track_id}:slot_{int(track.hydra_label_id)}",
        "local_segment_slot_id": int(track.hydra_label_id),
        "local_segment_event": str(track.last_segment_event),
        "local_segment_match_reason": str(track.last_segment_match_reason),
        "local_segment_match_score": None if track.last_segment_match_score is None else float(track.last_segment_match_score),
        "local_segment_xy_span_m": active_segment_record.get("local_segment_xy_span_m"),
        "local_segment_centroid_3d": active_segment_record.get("centroid_3d"),
        "local_segment_bbox_3d_min": active_segment_record.get("bbox_3d_min"),
        "local_segment_bbox_3d_max": active_segment_record.get("bbox_3d_max"),
        "local_segment_seen_count": active_segment_record.get("seen_count", 0),
        "persistent_track_seen_count": int(track.seen_count),
        "persistent_match_reason": reason,
        "persistent_match_score": None if score is None else float(score),
        "hydra_slot_id": int(track.hydra_label_id),
        "hydra_slot_name": track.hydra_label_name,
        "hydra_label_id": int(track.hydra_label_id),
        "hydra_label_name": track.hydra_label_name,
        "canonical_label": track.canonical_label,
        "semantic_label_source": track.label_source,
        "semantic_label_confidence": float(track.label_confidence),
        "slot_state": track.slot_state,
        "labeling_dispatched": bool(track.labeling_dispatched),
        "labeling_completed": bool(track.labeling_completed),
        "labeling_status": str(track.labeling_status),
        "semantic_label": track.semantic_label,
        "semantic_label_source": track.semantic_label_source,
        "semantic_label_confidence": float(track.semantic_label_confidence),
        "mobility_class": track.mobility_class,
        "mobility_confidence": float(track.mobility_confidence),
        "mobility_source": track.mobility_source,
        "object_detail": track.object_detail,
        "semantic_hydra_class_id": int(track.semantic_hydra_class_id),
        "semantic_reason": track.semantic_reason,
        "semantic_timestamp_sec": track.semantic_timestamp_sec,
        "semantic_update_count": int(track.semantic_update_count),
        "first_seen_timestamp_sec": float(track.first_seen_timestamp_sec),
        "last_seen_timestamp_sec": float(track.last_seen_timestamp_sec),
        "centroid_3d": _as_list(track.centroid_3d),
        "bbox_volume_m3": track.bbox_volume_m3,
        "bbox_2d": track.bbox_2d,
        "bbox_3d_min": _as_list(track.bbox_3d_min),
        "bbox_3d_max": _as_list(track.bbox_3d_max),
        "last_bbox_3d_min": _as_list(track.last_bbox_3d_min),
        "last_bbox_3d_max": _as_list(track.last_bbox_3d_max),
        "label_evidence": {str(k): float(v) for k, v in track.label_evidence.items()},
        "label_observations": {str(k): int(v) for k, v in track.label_observations.items()},
        "semantic_slot_ids": [int(item.get("hydra_slot_id", 0)) for item in all_segment_records],
        "semantic_segment_count": len(all_segment_records),
        "active_segment": active_segment_record,
        "semantic_segments": all_segment_records,
    }


def annotate_metadata(
    metadata: Dict[str, Any],
    track: Any,
    track_event: str,
    match_reason: str,
    match_score: Optional[float],
) -> None:
    active_segment = track.segments.get(int(track.active_segment_slot_id))
    active_segment_record = segment_record(active_segment) if active_segment is not None else {}
    all_segment_records = [segment_record(segment) for segment in track.segments.values()]
    metadata.update(
        {
            "persistent_track_id": track.track_id,
            "internal_object_id": track.track_id,
            "persistent_instance_id": int(track.instance_id),
            "local_segment_id": f"{track.track_id}:slot_{int(track.hydra_label_id)}",
            "semantic_segment_id": f"{track.track_id}:slot_{int(track.hydra_label_id)}",
            "local_segment_slot_id": int(track.hydra_label_id),
            "local_segment_event": str(track.last_segment_event),
            "local_segment_match_reason": str(track.last_segment_match_reason),
            "local_segment_match_score": None if track.last_segment_match_score is None else float(track.last_segment_match_score),
            "local_segment_xy_span_m": active_segment_record.get("local_segment_xy_span_m"),
            "local_segment_centroid_3d": active_segment_record.get("centroid_3d"),
            "local_segment_bbox_3d_min": active_segment_record.get("bbox_3d_min"),
            "local_segment_bbox_3d_max": active_segment_record.get("bbox_3d_max"),
            "local_segment_seen_count": active_segment_record.get("seen_count", 0),
            "semantic_segments": all_segment_records,
            "semantic_slot_ids": [int(item.get("hydra_slot_id", 0)) for item in all_segment_records],
            "semantic_segment_count": len(all_segment_records),
            "persistent_track_event": track_event,
            "persistent_track_seen_count": int(track.seen_count),
            "persistent_match_reason": match_reason,
            "persistent_match_score": None if match_score is None else float(match_score),
            "hydra_label_id": int(track.hydra_label_id),
            "hydra_label_name": track.hydra_label_name,
            "semantic_kind": track.semantic_kind,
            "canonical_label": track.canonical_label,
            "semantic_label_source": track.label_source,
            "semantic_label_confidence": float(track.label_confidence),
            "raw_rap_label": track.raw_rap_label,
            "raw_vlm_label": track.raw_vlm_label,
            "mobility_class": track.mobility_class,
            "mobility_confidence": float(track.mobility_confidence),
            "mobility_source": track.mobility_source,
            "object_detail": track.object_detail,
            "slot_state": track.slot_state,
            "labeling_dispatched": bool(track.labeling_dispatched),
            "labeling_completed": bool(track.labeling_completed),
            "labeling_status": str(track.labeling_status),
        }
    )
