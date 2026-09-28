"""Spatial candidate-pruning index for PersistentObjectTracker.

A coarse grid over track footprints/centroids, used to cut the per-observation
candidate set down before the expensive per-pair evaluation in
``PersistentObjectTracker._find_match``. Every method here assumes the
caller already holds ``tracker._lock`` (all current callers are
``PersistentObjectTracker`` methods that acquire it themselves) -- nothing
in this file acquires or releases it.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple

import numpy as np


class TrackerSpatialIndex:
    """Owns PersistentObjectTracker's spatial grid and candidate lookup."""

    def __init__(self, tracker: Any):
        self.tracker = tracker

    def _spatial_cell_size(self) -> float:
        tracker = self.tracker
        return max(0.25, min(2.0, max(
            float(getattr(tracker.config, "persistent_global_centroid_pass_m", tracker.config.persistent_max_match_distance_m)),
            float(tracker.config.persistent_continuation_gap_m),
            float(tracker.config.persistent_revisit_overlap_gap_m),
        )))

    def _spatial_cells(
        self, bbox_min: np.ndarray, bbox_max: np.ndarray, padding: float = 0.0
    ) -> Optional[Set[Tuple[int, int]]]:
        size = self._spatial_cell_size()
        x0, y0 = np.floor((bbox_min[:2] - padding) / size).astype(int)
        x1, y1 = np.floor((bbox_max[:2] + padding) / size).astype(int)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 256:
            return None
        return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}

    def _forget_spatial_index(self, track_id: str) -> None:
        """Drop every spatial-index entry for one track id."""
        tracker = self.tracker
        for cell in tracker._spatial_bbox_cells_by_track.pop(track_id, set()):
            ids = tracker._spatial_bbox_cells.get(cell)
            if ids is None:
                continue
            ids.discard(track_id)
            if not ids:
                del tracker._spatial_bbox_cells[cell]
        centroid_cell = tracker._spatial_centroid_cell_by_track.pop(track_id, None)
        if centroid_cell is not None:
            ids = tracker._spatial_centroid_cells.get(centroid_cell)
            if ids is not None:
                ids.discard(track_id)
                if not ids:
                    del tracker._spatial_centroid_cells[centroid_cell]
        tracker._spatial_fallback_track_ids.discard(track_id)

    def _refresh_spatial_index(self, track: Any) -> None:
        tracker = self.tracker
        track_id = track.track_id
        self._forget_spatial_index(track_id)

        if track.bbox_3d_min is None or track.bbox_3d_max is None:
            tracker._spatial_fallback_track_ids.add(track_id)
        else:
            cells = self._spatial_cells(track.bbox_3d_min, track.bbox_3d_max)
            if cells is None:
                tracker._spatial_fallback_track_ids.add(track_id)
            else:
                tracker._spatial_bbox_cells_by_track[track_id] = cells
                for cell in cells:
                    tracker._spatial_bbox_cells.setdefault(cell, set()).add(track_id)
        if track.centroid_3d is not None:
            cell = tuple(np.floor(track.centroid_3d[:2] / self._spatial_cell_size()).astype(int))
            tracker._spatial_centroid_cell_by_track[track_id] = cell
            tracker._spatial_centroid_cells.setdefault(cell, set()).add(track_id)

    def _candidate_track_ids(
        self,
        centroid: Optional[np.ndarray],
        bbox_3d_min: Optional[np.ndarray],
        bbox_3d_max: Optional[np.ndarray],
    ) -> List[str]:
        """Return every track that can still pass the exact association gates."""
        tracker = self.tracker
        global_enabled = bool(getattr(tracker.config, "persistent_global_association_enabled", True))
        block_2d = bool(getattr(tracker.config, "persistent_global_block_2d_on_3d_contradiction", True))
        if not global_enabled or not block_2d or bbox_3d_min is None or bbox_3d_max is None:
            return list(tracker._tracks)

        candidate_ids = set(tracker._spatial_fallback_track_ids)
        footprint_cells = self._spatial_cells(
            bbox_3d_min,
            bbox_3d_max,
            max(float(tracker.config.persistent_continuation_gap_m), float(tracker.config.persistent_revisit_overlap_gap_m)),
        )
        if footprint_cells is None:
            return list(tracker._tracks)
        for cell in footprint_cells:
            candidate_ids.update(tracker._spatial_bbox_cells.get(cell, ()))
        if centroid is not None:
            radius = float(getattr(
                tracker.config, "persistent_global_centroid_pass_m", tracker.config.persistent_max_match_distance_m
            ))
            centroid_cells = self._spatial_cells(centroid, centroid, radius)
            if centroid_cells is None:
                return list(tracker._tracks)
            for cell in centroid_cells:
                candidate_ids.update(tracker._spatial_centroid_cells.get(cell, ()))
        return [track_id for track_id in tracker._tracks if track_id in candidate_ids]
