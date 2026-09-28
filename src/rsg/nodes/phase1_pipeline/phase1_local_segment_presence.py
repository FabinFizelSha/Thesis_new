"""Per-frame local-segment heartbeat and restored-session presence backfill.

The fuser uses these timestamps for presence confidence. The message is
keyed by semantic slot, not by internal object, so revisiting section B of
a long object resets only section B.
"""

from __future__ import annotations

from typing import Any, Dict, List

from std_msgs.msg import String

from nodes.support.phase1.json_utils import safe_json_dumps
from nodes.support.phase1.persistent_object_tracker import _as_list


class Phase1LocalSegmentPresenceStage:
    """Owns the per-frame active-segment heartbeat and restored-presence backfill."""

    def __init__(self, coordinator: Any, config: Any, logger: Any):
        self.coordinator = coordinator
        self.config = config
        self.logger = logger

    def _publish_active_local_segments(self, frame: Any, track_records: List[Dict[str, Any]], timestamp_sec: float) -> None:
        """Publish the local Hydra slots observed in the current frame.

        The fuser uses these timestamps for presence confidence.  The message is
        keyed by semantic slot, not by internal object, so revisiting section B
        of a long object resets only section B.
        """
        coordinator = self.coordinator
        # No early return on an empty track_records: a resumed session has
        # restored slots to republish before anything has been re-observed,
        # and those frames are exactly when the fuser needs them. The
        # "nothing to send" case is handled by the `if not segments` check
        # once the restored entries have had their chance to contribute.
        segments: List[Dict[str, Any]] = []
        seen_slots = set()
        for record in track_records:
            slot_id = int(record.get("hydra_slot_id", record.get("hydra_label_id", 0)) or 0)
            if slot_id <= 0 or slot_id in seen_slots:
                continue
            seen_slots.add(slot_id)
            all_segments = list(record.get("semantic_segments", []) or [])
            segment_slot_ids = []
            for segment in all_segments:
                try:
                    segment_slot = int(segment.get("hydra_slot_id", segment.get("hydra_label_id", 0)) or 0)
                except Exception:
                    segment_slot = 0
                if segment_slot > 0:
                    segment_slot_ids.append(segment_slot)
            segments.append({
                "persistent_track_id": str(record.get("persistent_track_id", "")),
                "internal_object_id": str(record.get("internal_object_id", record.get("persistent_track_id", ""))),
                "persistent_instance_id": int(record.get("persistent_instance_id", 0) or 0),
                "local_segment_id": str(record.get("local_segment_id", record.get("semantic_segment_id", f"slot_{slot_id}"))),
                "semantic_segment_id": str(record.get("semantic_segment_id", record.get("local_segment_id", f"slot_{slot_id}"))),
                "hydra_slot_id": slot_id,
                "hydra_slot_name": str(record.get("hydra_slot_name", record.get("hydra_label_name", ""))),
                "last_observed_timestamp_sec": float(record.get("last_seen_timestamp_sec", timestamp_sec) or timestamp_sec),
                "centroid_3d": record.get("centroid_3d"),
                "bbox_3d_min": record.get("local_segment_bbox_3d_min", record.get("last_bbox_3d_min", record.get("bbox_3d_min"))),
                "bbox_3d_max": record.get("local_segment_bbox_3d_max", record.get("last_bbox_3d_max", record.get("bbox_3d_max"))),
                "last_bbox_3d_min": record.get("local_segment_bbox_3d_min", record.get("last_bbox_3d_min")),
                "last_bbox_3d_max": record.get("local_segment_bbox_3d_max", record.get("last_bbox_3d_max")),
                "local_segment_xy_span_m": record.get("local_segment_xy_span_m"),
                "track_event": str(record.get("track_event", record.get("persistent_track_event", ""))),
                "local_segment_event": str(record.get("local_segment_event", "")),
                "local_segment_match_reason": str(record.get("local_segment_match_reason", "")),
                "local_segment_match_score": record.get("local_segment_match_score"),
                "canonical_label": str(record.get("canonical_label", "")),
                "semantic_label": str(record.get("semantic_label", "")),
                "semantic_label_source": str(record.get("semantic_label_source", "")),
                "semantic_label_confidence": float(record.get("semantic_label_confidence", 0.0) or 0.0),
                "mobility_class": str(record.get("mobility_class", "unknown") or "unknown"),
                "mobility_confidence": float(record.get("mobility_confidence", 0.0) or 0.0),
                "mobility_source": str(record.get("mobility_source", "none") or "none"),
                "semantic_timestamp_sec": record.get("semantic_timestamp_sec"),
                "first_seen_timestamp_sec": record.get("first_seen_timestamp_sec"),
                "last_seen_timestamp_sec": record.get("last_seen_timestamp_sec"),
                "persistent_track_seen_count": int(record.get("persistent_track_seen_count", 0) or 0),
                "labeling_status": str(record.get("labeling_status", "")),
                "labeling_completed": bool(record.get("labeling_completed", False)),
                "semantic_segments": all_segments,
                "all_semantic_segments": all_segments,
                "semantic_slot_ids": segment_slot_ids,
                "object_identity": {
                    "internal_object_id": str(record.get("internal_object_id", record.get("persistent_track_id", ""))),
                    "persistent_track_id": str(record.get("persistent_track_id", "")),
                    "persistent_instance_id": int(record.get("persistent_instance_id", 0) or 0),
                    "canonical_label": str(record.get("canonical_label", "")),
                    "semantic_label": str(record.get("semantic_label", "")),
                    "semantic_label_source": str(record.get("semantic_label_source", "")),
                    "semantic_label_confidence": float(record.get("semantic_label_confidence", 0.0) or 0.0),
                    "mobility_class": str(record.get("mobility_class", "unknown") or "unknown"),
                    "mobility_confidence": float(record.get("mobility_confidence", 0.0) or 0.0),
                    "mobility_source": str(record.get("mobility_source", "none") or "none"),
                    "semantic_slot_ids": segment_slot_ids,
                    "semantic_segment_count": len(segment_slot_ids),
                },
            })

            # If a resolved object later creates a new local slot, immediately
            # propagate the cached class label to that slot.  This avoids waiting
            # for another RAP/VLM job for a section of the same long object.
            if (
                str(record.get("local_segment_event", "")) == "new_segment"
                and bool(record.get("labeling_completed", False))
                and str(record.get("semantic_label", ""))
            ):
                task = {
                    "persistent_track_id": str(record.get("persistent_track_id", "")),
                    "hydra_slot_id": slot_id,
                    "hydra_slot_name": str(record.get("hydra_slot_name", record.get("hydra_label_name", ""))),
                    "frame_id": frame.rsg_frame_id,
                    "sequence": int(frame.sequence),
                    "timestamp_sec": float(timestamp_sec),
                    "object_metadata": record,
                }
                propagation_event = dict(record)
                propagation_event["semantic_segments"] = [{
                    "hydra_slot_id": slot_id,
                    "hydra_slot_name": str(record.get(
                        "hydra_slot_name",
                        record.get("hydra_label_name", ""),
                    )),
                    "local_segment_id": str(record.get(
                        "local_segment_id",
                        record.get("semantic_segment_id", f"slot_{slot_id}"),
                    )),
                    "semantic_segment_id": str(record.get(
                        "semantic_segment_id",
                        record.get("local_segment_id", f"slot_{slot_id}"),
                    )),
                    "centroid_3d": record.get("centroid_3d"),
                }]
                coordinator.semantic_dispatch._emit_semantic_label_result(
                    propagation_event,
                    task,
                    source="object_label_propagation",
                    finalize_track=False,
                )

        # Restored tracks that have not been re-observed yet still need their
        # slot -> internal_object_id mapping published, because the fuser's
        # presence cache is per-process and starts empty. Without it the fuser
        # cannot tell that two restored nodes belong to one physical object:
        # internal_id_for() returns empty, each node becomes its own "__solo_"
        # group, and their overlap renders as a solid contact edge between two
        # different objects instead of a dotted same-object edge.
        #
        # Published every frame until the track is re-observed, rather than once
        # at startup, so it cannot be lost to a phase1/fuser startup race. The
        # timestamps are the restored (past-shifted) ones, so presence
        # confidence correctly treats these as stale rather than fresh.
        segments.extend(self._restored_presence_segments(seen_slots))

        if not segments:
            return
        payload = {
            "event": "local_segment_observations",
            "frame_id": frame.rsg_frame_id,
            "sequence": int(frame.sequence),
            "timestamp_sec": float(timestamp_sec),
            "segments": segments,
        }
        coordinator._safe_publish(coordinator.active_segments_pub, String(data=safe_json_dumps(payload)))

    def _restored_presence_segments(self, seen_slots: set) -> List[Dict[str, Any]]:
        """Presence entries for restored slots not yet re-observed this session.

        Carries only identity and geometry the fuser needs to group a physical
        object's segments. A *slot* drops out of here once that slot itself is
        re-observed, at which point the ordinary heartbeat covers it. Retiring
        at track granularity instead would strand a multi-segment object's
        other segments, since the heartbeat only ever republishes the segment
        observed in the current frame.
        """
        coordinator = self.coordinator
        pending = getattr(coordinator, "_restored_presence_pending", None)
        if not pending:
            return []

        # A slot observed this frame is now covered by the heartbeat.
        pending -= seen_slots
        if not pending:
            return []

        out: List[Dict[str, Any]] = []
        for track_id, track in coordinator.persistent_tracker._tracks.items():
            for slot_id, segment in track.segments.items():
                slot = int(slot_id)
                if slot <= 0 or slot in seen_slots or slot not in pending:
                    continue
                seen_slots.add(slot)
                out.append({
                    "persistent_track_id": track_id,
                    "internal_object_id": track_id,
                    "persistent_instance_id": int(track.instance_id or 0),
                    "local_segment_id": str(segment.segment_id),
                    "semantic_segment_id": str(segment.segment_id),
                    "hydra_slot_id": slot,
                    "hydra_slot_name": str(segment.hydra_label_name or ""),
                    "last_observed_timestamp_sec": float(segment.last_seen_timestamp_sec or 0.0),
                    "centroid_3d": _as_list(segment.centroid_3d),
                    "bbox_3d_min": _as_list(segment.bbox_3d_min),
                    "bbox_3d_max": _as_list(segment.bbox_3d_max),
                    "last_bbox_3d_min": _as_list(segment.last_bbox_3d_min),
                    "last_bbox_3d_max": _as_list(segment.last_bbox_3d_max),
                    "canonical_label": str(track.canonical_label or ""),
                    "restored_from_previous_session": True,
                })
        return out
