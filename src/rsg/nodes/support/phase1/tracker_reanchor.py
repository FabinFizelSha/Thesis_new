"""Loop-closure re-anchoring and duplicate-track merging for PersistentObjectTracker.

When the SLAM back-end folds an accumulated-drift correction into the
``map -> odom`` transform, every cached track/segment coordinate is stale by
that same rigid step. ``reanchor_all`` moves the geometry in place;
``merge_reanchor_duplicates`` folds identities that were split before the
correction and only coincide once the geometry has been moved. Neither runs
in the steady-state association path, so unlike every other tracker_*.py
collaborator, the two externally-called entry points here (``reanchor_all``,
``merge_reanchor_duplicates``) acquire ``tracker._lock`` themselves --
``_reanchor_labels_compatible``/``_merge_track_pair`` do not, since they are
only ever called from within that already-held lock.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import numpy as np

from nodes.support.phase1.tracker_geometry import (
    _aabb_iou_3d,
    _aabb_xy_diagonal,
    _aabb_gap_xy,
    _normalise_label,
    _rigid_aabb,
    _rigid_point,
    _track_sort_key,
)


class TrackerReanchor:
    """Owns rigid re-anchoring and post-reanchor duplicate merging."""

    def __init__(self, tracker: Any):
        self.tracker = tracker

    def reanchor_all(
        self,
        rotation: Any,
        translation: Any,
        *,
        stamp: Optional[float] = None,
    ) -> int:
        """Rigid-transform every cached track and segment by ``p -> R @ p + t``.

        Only geometry moves.  EMA state, observation counts, labels, Hydra
        slots and the shared crop registries are untouched.  ``bbox_volume_m3``
        is invariant under a rigid motion and is left as-is.  Returns the number
        of tracks re-anchored.
        """
        tracker = self.tracker
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        trans = np.asarray(translation, dtype=np.float64).reshape(3)
        with tracker._lock:
            for track in tracker._tracks.values():
                track.centroid_3d = _rigid_point(track.centroid_3d, rot, trans)
                track.bbox_3d_min, track.bbox_3d_max = _rigid_aabb(
                    track.bbox_3d_min, track.bbox_3d_max, rot, trans
                )
                track.last_bbox_3d_min, track.last_bbox_3d_max = _rigid_aabb(
                    track.last_bbox_3d_min, track.last_bbox_3d_max, rot, trans
                )
                for segment in track.segments.values():
                    segment.centroid_3d = _rigid_point(segment.centroid_3d, rot, trans)
                    segment.bbox_3d_min, segment.bbox_3d_max = _rigid_aabb(
                        segment.bbox_3d_min, segment.bbox_3d_max, rot, trans
                    )
                    segment.last_bbox_3d_min, segment.last_bbox_3d_max = _rigid_aabb(
                        segment.last_bbox_3d_min, segment.last_bbox_3d_max, rot, trans
                    )
                tracker.spatial_index._refresh_spatial_index(track)
            count = len(tracker._tracks)
            tracker._last_reanchor = {
                "stamp_sec": float(stamp) if stamp is not None else None,
                "translation": [float(v) for v in trans],
                "rotation": [float(v) for v in rot.reshape(9)],
                "track_count": int(count),
            }
        return count

    def last_reanchor(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the most recent :meth:`reanchor_all` summary."""
        tracker = self.tracker
        with tracker._lock:
            return dict(tracker._last_reanchor) if tracker._last_reanchor else None

    def _reanchor_labels_compatible(self, a: Any, b: Any) -> bool:
        if a.semantic_kind != b.semantic_kind:
            return False
        generic = {"", "unknown", "unknown object", "object", "thing", "stuff", "background"}

        def strong_label(track: Any) -> str:
            for candidate in (
                track.semantic_label,
                track.canonical_label,
                track.raw_vlm_label,
                track.raw_rap_label,
            ):
                name = _normalise_label(candidate)
                if name and name not in generic:
                    return name
            return ""

        label_a = strong_label(a)
        label_b = strong_label(b)
        if label_a and label_b:
            return label_a == label_b
        return True

    def _merge_track_pair(
        self,
        keep: Any,
        drop: Any,
        *,
        iou_3d: float,
        distance_m: float,
        reason: str,
        adopt_drop_geometry: bool,
    ) -> None:
        """Fold ``drop`` into ``keep``.

        ``keep`` always retains the surviving ``track_id``, label evidence and
        (earlier) first-seen.  With ``adopt_drop_geometry`` the survivor takes
        ``drop``'s centroid and last box verbatim -- used for the loop-closure
        case, where ``drop`` is the freshly re-observed, drift-corrected copy
        and ``keep`` is the stale original.  Otherwise the centroid is an
        observation-weighted blend and the global box is the union.
        """
        tracker = self.tracker
        kc = max(1, int(keep.seen_count))
        dc = max(1, int(drop.seen_count))

        union_min = None
        union_max = None
        if keep.bbox_3d_min is not None and drop.bbox_3d_min is not None:
            union_min = np.minimum(keep.bbox_3d_min, drop.bbox_3d_min)
            union_max = np.maximum(keep.bbox_3d_max, drop.bbox_3d_max)

        if adopt_drop_geometry:
            if drop.centroid_3d is not None:
                keep.centroid_3d = np.asarray(drop.centroid_3d, dtype=np.float64).copy()
            if drop.bbox_3d_min is not None:
                keep.bbox_3d_min = np.asarray(drop.bbox_3d_min, dtype=np.float64).copy()
                keep.bbox_3d_max = np.asarray(drop.bbox_3d_max, dtype=np.float64).copy()
            if drop.last_bbox_3d_min is not None:
                keep.last_bbox_3d_min = np.asarray(drop.last_bbox_3d_min, dtype=np.float64).copy()
                keep.last_bbox_3d_max = np.asarray(drop.last_bbox_3d_max, dtype=np.float64).copy()
        else:
            if keep.centroid_3d is not None and drop.centroid_3d is not None:
                keep.centroid_3d = (kc * keep.centroid_3d + dc * drop.centroid_3d) / float(kc + dc)
            elif keep.centroid_3d is None:
                keep.centroid_3d = drop.centroid_3d
            if union_min is not None:
                keep.bbox_3d_min, keep.bbox_3d_max = union_min, union_max
            if drop.last_bbox_3d_min is not None and (
                keep.last_bbox_3d_min is None
                or float(drop.last_seen_timestamp_sec) >= float(keep.last_seen_timestamp_sec)
            ):
                keep.last_bbox_3d_min = drop.last_bbox_3d_min
                keep.last_bbox_3d_max = drop.last_bbox_3d_max

        if keep.bbox_3d_min is not None and keep.bbox_3d_max is not None:
            keep.bbox_volume_m3 = float(
                np.prod(np.maximum(0.0, keep.bbox_3d_max - keep.bbox_3d_min))
            )
        keep.seen_count = kc + dc

        if float(drop.first_seen_timestamp_sec) < float(keep.first_seen_timestamp_sec):
            keep.first_seen_timestamp_sec = drop.first_seen_timestamp_sec
            keep.first_seen_frame_id = drop.first_seen_frame_id
            keep.first_seen_sequence = drop.first_seen_sequence
        if float(drop.last_seen_timestamp_sec) > float(keep.last_seen_timestamp_sec):
            keep.last_seen_timestamp_sec = drop.last_seen_timestamp_sec
            keep.last_seen_frame_id = drop.last_seen_frame_id
            keep.last_seen_sequence = drop.last_seen_sequence

        for key, value in (drop.label_evidence or {}).items():
            keep.label_evidence[key] = keep.label_evidence.get(key, 0.0) + float(value)
        for key, value in (drop.label_observations or {}).items():
            keep.label_observations[key] = keep.label_observations.get(key, 0) + int(value)

        # Hydra slot ids are globally unique, so a non-overlapping ``drop``
        # segment can be re-keyed straight onto the survivor without
        # collision. But ``keep`` and ``drop`` were tracked independently
        # until now, so each may already have built its own local segment
        # covering the same physical patch (e.g. the same ceiling tile seen
        # moments apart before the duplicate was recognised). Re-keying both
        # verbatim would leave two permanently-static, heavily-overlapping
        # segments sitting side by side forever -- geometry is reconciled
        # into the closest touching existing segment instead of adding a
        # new, redundant one. Only genuinely disjoint drop segments keep
        # their own slot; presence continuity matters more than slot economy
        # for those.
        gap_limit = float(getattr(tracker.config, "persistent_local_segment_gap_m", 0.20))
        max_span = float(getattr(tracker.config, "persistent_local_segment_max_xy_span_m", 4.0))
        for slot_id, segment in drop.segments.items():
            absorbing: Optional[Any] = None
            best_gap: Optional[float] = None
            if segment.bbox_3d_min is not None and segment.bbox_3d_max is not None:
                for existing in keep.segments.values():
                    if existing.bbox_3d_min is None or existing.bbox_3d_max is None:
                        continue
                    gap = _aabb_gap_xy(
                        segment.bbox_3d_min, segment.bbox_3d_max,
                        existing.bbox_3d_min, existing.bbox_3d_max,
                    )
                    if gap <= gap_limit and (best_gap is None or gap < best_gap):
                        absorbing, best_gap = existing, gap

            if absorbing is None:
                segment.segment_id = f"{keep.track_id}:slot_{int(segment.hydra_label_id)}"
                keep.segments.setdefault(int(slot_id), segment)
                continue

            absorbing.bbox_3d_min = np.minimum(absorbing.bbox_3d_min, segment.bbox_3d_min)
            absorbing.bbox_3d_max = np.maximum(absorbing.bbox_3d_max, segment.bbox_3d_max)
            if _aabb_xy_diagonal(absorbing.bbox_3d_min, absorbing.bbox_3d_max) > max_span:
                # Same freeze-on-cap rule as live association: a merge can
                # bridge the cap once, but the result must not keep growing.
                absorbing.closed = True
            if float(segment.last_seen_timestamp_sec) >= float(absorbing.last_seen_timestamp_sec):
                absorbing.last_bbox_3d_min = segment.last_bbox_3d_min
                absorbing.last_bbox_3d_max = segment.last_bbox_3d_max
                absorbing.last_seen_frame_id = segment.last_seen_frame_id
                absorbing.last_seen_sequence = segment.last_seen_sequence
                absorbing.last_seen_timestamp_sec = segment.last_seen_timestamp_sec
                if segment.bbox_2d:
                    absorbing.bbox_2d = segment.bbox_2d
            if float(segment.first_seen_timestamp_sec) < float(absorbing.first_seen_timestamp_sec):
                absorbing.first_seen_frame_id = segment.first_seen_frame_id
                absorbing.first_seen_sequence = segment.first_seen_sequence
                absorbing.first_seen_timestamp_sec = segment.first_seen_timestamp_sec
            absorbing.seen_count += int(segment.seen_count)
            # `segment`'s own Hydra slot is retired here: it is dropped from
            # the merged track's bookkeeping so no future observation is
            # ever routed to it again, and its existing DSG node stops
            # growing -- but note this cannot retroactively erase mesh
            # geometry Hydra already committed under that slot before the
            # merge; only the *tracker's* forward association state is
            # reconciled.

        keep.metadata.setdefault("reanchor_merged_from", []).append(
            {
                "track_id": drop.track_id,
                "internal_object_id": drop.track_id,
                "seen_count": int(dc),
                "iou_3d": round(float(iou_3d), 4),
                "distance_m": round(float(distance_m), 4),
                "reason": str(reason),
                "adopted_geometry": bool(adopt_drop_geometry),
                "slot_ids": sorted(int(s) for s in drop.segments),
            }
        )
        tracker.spatial_index._forget_spatial_index(drop.track_id)

    def merge_reanchor_duplicates(
        self,
        *,
        correction_translation_m: float = 0.0,
        now_sec: Optional[float] = None,
        recent_window_sec: float = 5.0,
        distance_slack_m: float = 0.6,
        min_iou_3d: float = 0.30,
        max_centroid_distance_m: Optional[float] = None,
    ) -> int:
        """Fold together track identities that describe one physical object but
        were split before a loop closure.

        Two complementary passes, both conservative and both label-gated:

        * **drift pass** (needs ``now_sec``): a track re-observed within
          ``recent_window_sec`` is folded into an older compatible track whose
          centroid lies within ``|correction| * 1.25 + distance_slack_m``.  A
          rigid re-anchor cannot close this gap because the pair straddles the
          drift; the survivor is the older identity but it adopts the fresh
          (drift-corrected) geometry.
        * **overlap pass**: any remaining track pair that now genuinely overlaps
          (``min_iou_3d``) and sits within ``max_centroid_distance_m`` is folded
          with an observation-weighted blend -- ordinary fragmentation cleanup.

        Runs once per correction, never in the steady-state association path.
        Returns the number of tracks removed.
        """
        tracker = self.tracker
        if max_centroid_distance_m is None:
            max_centroid_distance_m = float(
                getattr(
                    tracker.config,
                    "persistent_global_centroid_pass_m",
                    getattr(tracker.config, "persistent_max_match_distance_m", 1.0),
                )
            )
        drift_radius = max(
            float(distance_slack_m),
            abs(float(correction_translation_m)) * 1.25 + float(distance_slack_m),
        )
        removed = 0
        with tracker._lock:
            def _xy_gap(a: Any, b: Any) -> Optional[float]:
                if a.centroid_3d is None or b.centroid_3d is None:
                    return None
                return float(np.linalg.norm(a.centroid_3d[:2] - b.centroid_3d[:2]))

            gone: Set[str] = set()
            absorbed: Set[str] = set()

            # --- drift pass -------------------------------------------------
            if now_sec is not None:
                recent = sorted(
                    (
                        t
                        for t in tracker._tracks.values()
                        if t.bbox_3d_min is not None
                        and t.centroid_3d is not None
                        and float(now_sec) - float(t.last_seen_timestamp_sec) <= float(recent_window_sec)
                    ),
                    key=lambda t: _track_sort_key(t.track_id),
                )
                for fresh in recent:
                    if fresh.track_id in gone:
                        continue
                    best: Optional[Any] = None
                    best_gap = drift_radius
                    for cand in tracker._tracks.values():
                        if cand.track_id == fresh.track_id or cand.track_id in gone or cand.track_id in absorbed:
                            continue
                        if cand.bbox_3d_min is None or cand.centroid_3d is None:
                            continue
                        if float(now_sec) - float(cand.last_seen_timestamp_sec) <= float(recent_window_sec):
                            continue  # both fresh -> not a stale/fresh pair
                        if float(cand.first_seen_timestamp_sec) > float(fresh.first_seen_timestamp_sec):
                            continue  # survivor must be the older identity
                        if not self._reanchor_labels_compatible(cand, fresh):
                            continue
                        gap = _xy_gap(cand, fresh)
                        if gap is None or gap > best_gap:
                            continue
                        best, best_gap = cand, gap
                    if best is None:
                        continue
                    iou = _aabb_iou_3d(
                        best.bbox_3d_min, best.bbox_3d_max,
                        fresh.bbox_3d_min, fresh.bbox_3d_max,
                    )
                    self._merge_track_pair(
                        best, fresh,
                        iou_3d=iou, distance_m=best_gap,
                        reason="loop_closure_drift", adopt_drop_geometry=True,
                    )
                    gone.add(fresh.track_id)
                    absorbed.add(best.track_id)
                    tracker.spatial_index._refresh_spatial_index(best)
                    removed += 1

            # --- overlap pass --------------------------------------------------
            ordered = sorted(
                (
                    t
                    for t in tracker._tracks.values()
                    if t.track_id not in gone
                    and t.bbox_3d_min is not None
                    and t.bbox_3d_max is not None
                ),
                key=lambda t: (-int(t.seen_count), _track_sort_key(t.track_id)),
            )
            for keep in ordered:
                if keep.track_id in gone:
                    continue
                for cid in tracker.spatial_index._candidate_track_ids(
                    keep.centroid_3d, keep.bbox_3d_min, keep.bbox_3d_max
                ):
                    if cid == keep.track_id or cid in gone or cid in absorbed:
                        continue
                    drop = tracker._tracks.get(cid)
                    if (
                        drop is None
                        or drop.bbox_3d_min is None
                        or drop.bbox_3d_max is None
                        or int(drop.seen_count) > int(keep.seen_count)
                        or not self._reanchor_labels_compatible(keep, drop)
                    ):
                        continue
                    iou = _aabb_iou_3d(
                        keep.bbox_3d_min, keep.bbox_3d_max,
                        drop.bbox_3d_min, drop.bbox_3d_max,
                    )
                    if iou < float(min_iou_3d):
                        continue
                    gap = _xy_gap(keep, drop)
                    if gap is not None and gap > float(max_centroid_distance_m):
                        continue
                    self._merge_track_pair(
                        keep, drop,
                        iou_3d=iou, distance_m=(gap if gap is not None else 0.0),
                        reason="post_reanchor_overlap", adopt_drop_geometry=False,
                    )
                    gone.add(cid)
                    absorbed.add(keep.track_id)
                    removed += 1
                if keep.track_id not in gone:
                    tracker.spatial_index._refresh_spatial_index(keep)

            for cid in gone:
                tracker._tracks.pop(cid, None)
        return removed
