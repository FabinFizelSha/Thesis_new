"""Save and restore PersistentObjectTracker state across runs.

Purpose: the tracker's revisit-association branch (``global_revisit_weight_*``,
``global_revisit_min_score``, ``revisit_min_2d_iou``) selects on

    age_sec = max(0.0, timestamp_sec - track.last_seen_timestamp_sec)
    recent_mode = age_sec <= persistent_continuation_max_age_sec   # 8.0 s

Within a single bag replay every track is re-observed within a few frames, so
that branch effectively never runs. Restoring a previous session's tracks makes
it run for every association.

**Why the timestamps are shifted at save time.** Bag replay timestamps restart
at the *bag's* own start time, which is smaller than a saved
``last_seen_timestamp_sec``. The raw difference is therefore negative for the
whole replay and ``max(0.0, ...)`` clamps it to 0.0 -- restored tracks would be
evaluated in *recent* mode, the exact opposite of the intent. Shifting stored
times one day into the past makes ``age_sec ~= 86400 - bag_duration``, which is
unambiguously revisit.

The same shift also un-breaks two other clamps that would otherwise mis-behave
for restored tracks: the labelling settling window
(``prepare_active_for_labeling``) and the local-segment 2D-fallback staleness
guard, both of which compare against a ``max(0.0, ...)`` age.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from nodes.support.phase1.persistent_object_tracker import (
    PersistentObjectSegment,
    PersistentObjectTrack,
    _as_list,
    _as_xyz,
)

SCHEMA_VERSION = 1

# Timestamp fields shifted into the past on save. Kept explicit rather than
# inferred by name so a future field addition is a deliberate decision.
_TRACK_TIME_FIELDS = (
    "first_seen_timestamp_sec",
    "last_seen_timestamp_sec",
    "semantic_timestamp_sec",
)
_SEGMENT_TIME_FIELDS = (
    "first_seen_timestamp_sec",
    "last_seen_timestamp_sec",
)

# Numpy geometry: stored as plain lists, restored with the tracker's own
# validating converter so a malformed file yields None rather than a bad array.
_TRACK_ARRAY_FIELDS = (
    "centroid_3d",
    "bbox_3d_min",
    "bbox_3d_max",
    "last_bbox_3d_min",
    "last_bbox_3d_max",
)
_SEGMENT_ARRAY_FIELDS = _TRACK_ARRAY_FIELDS


def _json_safe(value: Any) -> Any:
    """Best-effort conversion for the free-form ``metadata`` dict."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _segment_to_dict(segment: PersistentObjectSegment, shift: float) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "segment_id": segment.segment_id,
        "hydra_label_id": int(segment.hydra_label_id),
        "hydra_label_name": segment.hydra_label_name,
        "instance_id": int(segment.instance_id),
        "first_seen_frame_id": segment.first_seen_frame_id,
        "last_seen_frame_id": segment.last_seen_frame_id,
        "first_seen_sequence": int(segment.first_seen_sequence),
        "last_seen_sequence": int(segment.last_seen_sequence),
        "bbox_2d": _json_safe(segment.bbox_2d),
        "seen_count": int(segment.seen_count),
        "closed": bool(segment.closed),
    }
    for name in _SEGMENT_TIME_FIELDS:
        raw = getattr(segment, name)
        data[name] = None if raw is None else float(raw) - shift
    for name in _SEGMENT_ARRAY_FIELDS:
        data[name] = _as_list(getattr(segment, name))
    return data


def _segment_from_dict(data: Dict[str, Any]) -> PersistentObjectSegment:
    kwargs: Dict[str, Any] = {
        "segment_id": str(data.get("segment_id", "")),
        "hydra_label_id": int(data.get("hydra_label_id", 0)),
        "hydra_label_name": str(data.get("hydra_label_name", "")),
        "instance_id": int(data.get("instance_id", 0)),
        "first_seen_frame_id": str(data.get("first_seen_frame_id", "")),
        "last_seen_frame_id": str(data.get("last_seen_frame_id", "")),
        "first_seen_sequence": int(data.get("first_seen_sequence", 0)),
        "last_seen_sequence": int(data.get("last_seen_sequence", 0)),
        "bbox_2d": data.get("bbox_2d"),
        "first_seen_timestamp_sec": float(data.get("first_seen_timestamp_sec") or 0.0),
        "last_seen_timestamp_sec": float(data.get("last_seen_timestamp_sec") or 0.0),
        "centroid_3d": _as_xyz(data.get("centroid_3d")),
        "bbox_3d_min": _as_xyz(data.get("bbox_3d_min")),
        "bbox_3d_max": _as_xyz(data.get("bbox_3d_max")),
        "last_bbox_3d_min": _as_xyz(data.get("last_bbox_3d_min")),
        "last_bbox_3d_max": _as_xyz(data.get("last_bbox_3d_max")),
    }
    segment = PersistentObjectSegment(**kwargs)
    segment.seen_count = int(data.get("seen_count", 1))
    segment.closed = bool(data.get("closed", False))
    return segment


def _track_to_dict(track: PersistentObjectTrack, shift: float) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "track_id": track.track_id,
        "instance_id": int(track.instance_id),
        "hydra_label_id": int(track.hydra_label_id),
        "hydra_label_name": track.hydra_label_name,
        "semantic_kind": track.semantic_kind,
        "canonical_label": track.canonical_label,
        "label_source": track.label_source,
        "label_confidence": float(track.label_confidence),
        "first_seen_frame_id": track.first_seen_frame_id,
        "first_seen_sequence": int(track.first_seen_sequence),
        "last_seen_frame_id": track.last_seen_frame_id,
        "last_seen_sequence": int(track.last_seen_sequence),
        "bbox_volume_m3": None if track.bbox_volume_m3 is None else float(track.bbox_volume_m3),
        "bbox_2d": _json_safe(track.bbox_2d),
        "seen_count": int(track.seen_count),
        "raw_rap_label": track.raw_rap_label,
        "raw_vlm_label": track.raw_vlm_label,
        "mobility_class": track.mobility_class,
        "mobility_confidence": float(track.mobility_confidence),
        "mobility_source": track.mobility_source,
        "metadata": _json_safe(track.metadata),
        "slot_state": track.slot_state,
        "semantic_update_count": int(track.semantic_update_count),
        "semantic_label": track.semantic_label,
        "semantic_label_source": track.semantic_label_source,
        "semantic_label_confidence": float(track.semantic_label_confidence),
        "semantic_hydra_class_id": int(track.semantic_hydra_class_id),
        "semantic_reason": track.semantic_reason,
        "label_evidence": {str(k): float(v) for k, v in (track.label_evidence or {}).items()},
        "label_observations": {str(k): int(v) for k, v in (track.label_observations or {}).items()},
        "labeling_dispatched": bool(track.labeling_dispatched),
        "labeling_completed": bool(track.labeling_completed),
        "labeling_status": track.labeling_status,
        "active_segment_slot_id": int(track.active_segment_slot_id),
        "last_segment_event": track.last_segment_event,
        "last_segment_match_reason": track.last_segment_match_reason,
        "last_segment_match_score": (
            None if track.last_segment_match_score is None else float(track.last_segment_match_score)
        ),
        "segments": {
            str(slot_id): _segment_to_dict(segment, shift)
            for slot_id, segment in track.segments.items()
        },
    }
    for name in _TRACK_TIME_FIELDS:
        raw = getattr(track, name)
        data[name] = None if raw is None else float(raw) - shift
    for name in _TRACK_ARRAY_FIELDS:
        data[name] = _as_list(getattr(track, name))
    return data


def _track_from_dict(data: Dict[str, Any]) -> PersistentObjectTrack:
    track = PersistentObjectTrack(
        track_id=str(data.get("track_id", "")),
        instance_id=int(data.get("instance_id", 0)),
        hydra_label_id=int(data.get("hydra_label_id", 0)),
        hydra_label_name=str(data.get("hydra_label_name", "")),
        semantic_kind=str(data.get("semantic_kind", "slot")),
        canonical_label=str(data.get("canonical_label", "")),
        label_source=str(data.get("label_source", "")),
        label_confidence=float(data.get("label_confidence", 0.0)),
        first_seen_frame_id=str(data.get("first_seen_frame_id", "")),
        first_seen_sequence=int(data.get("first_seen_sequence", 0)),
        first_seen_timestamp_sec=float(data.get("first_seen_timestamp_sec") or 0.0),
        last_seen_frame_id=str(data.get("last_seen_frame_id", "")),
        last_seen_sequence=int(data.get("last_seen_sequence", 0)),
        last_seen_timestamp_sec=float(data.get("last_seen_timestamp_sec") or 0.0),
        centroid_3d=_as_xyz(data.get("centroid_3d")),
        bbox_volume_m3=(
            None if data.get("bbox_volume_m3") is None else float(data["bbox_volume_m3"])
        ),
        bbox_2d=data.get("bbox_2d"),
        bbox_3d_min=_as_xyz(data.get("bbox_3d_min")),
        bbox_3d_max=_as_xyz(data.get("bbox_3d_max")),
        last_bbox_3d_min=_as_xyz(data.get("last_bbox_3d_min")),
        last_bbox_3d_max=_as_xyz(data.get("last_bbox_3d_max")),
    )
    track.seen_count = int(data.get("seen_count", 1))
    track.raw_rap_label = str(data.get("raw_rap_label", ""))
    track.raw_vlm_label = str(data.get("raw_vlm_label", ""))
    track.mobility_class = str(data.get("mobility_class", "unknown"))
    track.mobility_confidence = float(data.get("mobility_confidence", 0.0))
    track.mobility_source = str(data.get("mobility_source", "none"))
    track.metadata = dict(data.get("metadata") or {})
    track.slot_state = str(data.get("slot_state", "active"))
    semantic_ts = data.get("semantic_timestamp_sec")
    track.semantic_timestamp_sec = None if semantic_ts is None else float(semantic_ts)
    track.semantic_update_count = int(data.get("semantic_update_count", 0))
    track.semantic_label = str(data.get("semantic_label", ""))
    track.semantic_label_source = str(data.get("semantic_label_source", ""))
    track.semantic_label_confidence = float(data.get("semantic_label_confidence", 0.0))
    track.semantic_hydra_class_id = int(data.get("semantic_hydra_class_id", 0))
    track.semantic_reason = str(data.get("semantic_reason", ""))
    track.label_evidence = {
        str(k): float(v) for k, v in (data.get("label_evidence") or {}).items()
    }
    track.label_observations = {
        str(k): int(v) for k, v in (data.get("label_observations") or {}).items()
    }
    track.labeling_dispatched = bool(data.get("labeling_dispatched", False))
    track.labeling_completed = bool(data.get("labeling_completed", False))
    track.labeling_status = str(data.get("labeling_status", "collecting"))
    track.active_segment_slot_id = int(data.get("active_segment_slot_id", 0))
    track.last_segment_event = str(data.get("last_segment_event", ""))
    track.last_segment_match_reason = str(data.get("last_segment_match_reason", ""))
    score = data.get("last_segment_match_score")
    track.last_segment_match_score = None if score is None else float(score)
    # JSON stringifies dict keys; segments are keyed by int slot id.
    track.segments = {
        int(slot_id): _segment_from_dict(segment_data)
        for slot_id, segment_data in (data.get("segments") or {}).items()
    }
    return track


def save_tracker_state(
    tracker: Any,
    path: Path,
    time_shift_sec: float,
    logger: Any = None,
    source_run: str = "",
) -> bool:
    """Serialise tracker state, shifting every timestamp into the past.

    Written atomically (``.tmp`` then ``replace``) so an interrupted shutdown
    cannot leave a half-written file that would later load as valid state.

    Takes ``tracker._lock``: on shutdown the tracking thread has been asked to
    stop but is not yet joined, so ``_tracks`` may still be mutating.
    """
    path = Path(path).expanduser()
    shift = float(time_shift_sec)

    with tracker._lock:
        # Carried over by load_tracker_state, so a file that has been through
        # several save/load cycles reports total age rather than just the last
        # increment. Absent on a first run.
        previous_shift = float(getattr(tracker, "_restored_cumulative_time_shift_sec", 0.0))
        tracks = {
            track_id: _track_to_dict(track, shift)
            for track_id, track in tracker._tracks.items()
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_run": source_run,
            "time_shift_sec": shift,
            # Cumulative across save/load cycles, so a state file that has been
            # round-tripped several times is legible rather than mysterious.
            "cumulative_time_shift_sec": previous_shift + shift,
            "track_count": len(tracks),
            "next_track_index": int(tracker._next_track_index),
            "next_slot_index": int(tracker._next_slot_index),
            "next_instance_id": int(tracker._next_instance_id),
            # Not rebuildable from `tracks`: slots are never released, so this
            # is a strict superset of live slot ids. Rebuilding it would let a
            # later run re-issue a retired slot.
            "allocated_slot_ids": sorted(int(v) for v in tracker._allocated_slot_ids),
            "tracks": tracks,
        }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, default=_json_safe)
        temporary_path.replace(path)
    except Exception as exc:  # noqa: BLE001 - shutdown path must not raise
        if logger is not None:
            logger.error(f"Failed to save tracker state to {path}: {exc}")
        return False

    if logger is not None:
        logger.info(
            f"Saved tracker state: {len(payload['tracks'])} tracks, "
            f"timestamps shifted {shift:.0f}s into the past -> {path}"
        )
    return True


def load_tracker_state(tracker: Any, path: Path, logger: Any = None) -> int:
    """Restore tracker state saved by :func:`save_tracker_state`.

    Returns the number of tracks restored; 0 when the file is absent, which is
    the normal first-run case and never an error.

    Must be called before the node subscribes or starts worker threads, so no
    lock contention is possible -- but the lock is taken anyway for consistency
    with every other bulk mutation of ``_tracks``.
    """
    path = Path(path).expanduser()
    if not path.exists():
        if logger is not None:
            logger.info(f"No tracker state to restore at {path} (first run)")
        return 0

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except Exception as exc:  # noqa: BLE001 - a bad file must not block startup
        if logger is not None:
            logger.error(f"Failed to read tracker state from {path}: {exc}")
        return 0

    version = int(payload.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        if logger is not None:
            logger.error(
                f"Tracker state schema {version} != expected {SCHEMA_VERSION}; ignoring {path}"
            )
        return 0

    try:
        restored = {
            str(track_id): _track_from_dict(track_data)
            for track_id, track_data in (payload.get("tracks") or {}).items()
        }
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.error(f"Failed to parse tracker state from {path}: {exc}")
        return 0

    with tracker._lock:
        tracker._tracks = restored
        tracker._allocated_slot_ids = {
            int(v) for v in (payload.get("allocated_slot_ids") or [])
        }
        # Guard against a stale counter silently overwriting a restored track:
        # `_new_track` does `_tracks[track_id] = track`, so a reused id is not
        # an error, it is data loss.
        highest = 0
        for track_id in restored:
            suffix = track_id.rsplit("_", 1)[-1]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        tracker._next_track_index = max(int(payload.get("next_track_index", 1)), highest + 1)
        tracker._next_slot_index = max(1, int(payload.get("next_slot_index", 1)))
        tracker._next_instance_id = max(1, int(payload.get("next_instance_id", 1)))

        # The spatial index is derived, never serialised: its cell size is a
        # function of config, so cached cell keys would be wrong if config
        # changed between runs. Rebuild from the restored geometry instead.
        tracker._spatial_bbox_cells = {}
        tracker._spatial_centroid_cells = {}
        tracker._spatial_bbox_cells_by_track = {}
        tracker._spatial_centroid_cell_by_track = {}
        tracker._spatial_fallback_track_ids = set()
        for track in restored.values():
            tracker._refresh_spatial_index(track)

        # Read back by save_tracker_state so the next save reports total age.
        tracker._restored_cumulative_time_shift_sec = float(
            payload.get("cumulative_time_shift_sec", payload.get("time_shift_sec", 0.0))
        )

    if logger is not None:
        shift = float(getattr(tracker, "_restored_cumulative_time_shift_sec", 0.0))
        logger.info(
            f"Restored {len(restored)} tracks from {path} "
            f"(timestamps {shift:.0f}s in the past; next track id index "
            f"{tracker._next_track_index}, {len(tracker._allocated_slot_ids)} slots reserved)"
        )
    return len(restored)
