"""Comprehensive tracking quality diagnostics and evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class TrackObservation:
    """Single observation of a track in a frame."""
    frame_id: str
    sequence: int
    centroid_3d: List[float]
    centroid_2d: List[float]
    bbox_volume_m3: float
    mask_area_px: int
    depth_mean_m: float
    mask_iou_3d: Optional[float] = None
    observation_quality_score: float = 1.0
    timestamp_iso: str = ""

    # Computed relative to previous observation
    prev_distance_m: Optional[float] = None
    velocity_mm_per_frame: Optional[float] = None
    volume_ratio: Optional[float] = None
    displacement_3d: Optional[List[float]] = None

    # Association components for Level 1 optimization diagnostics
    global_association_components: Optional[Dict[str, float]] = None
    match_reason: Optional[str] = None


@dataclass
class AssociationDecision:
    """Tracking decision: how a mask was matched to a track."""
    frame_id: str
    sequence: int
    mask_id: int
    mask_area_px: float
    mask_centroid_3d: List[float]

    # Matching result
    matched_track_id: Optional[str] = None
    match_type: str = "new_object"  # new_object, centroid_distance, iou_fallback, continuation
    match_distance_m: Optional[float] = None
    match_iou_3d: Optional[float] = None
    match_score: Optional[float] = None

    # Previous track state at time of match
    prev_track_age_frames: Optional[int] = None
    prev_track_observations: Optional[int] = None
    prev_centroid_3d: Optional[List[float]] = None
    prev_bbox_volume_m3: Optional[float] = None

    reason: str = ""
    timestamp_iso: str = ""


@dataclass
class TrackLifecycle:
    """Complete lifecycle of one persistent track."""
    track_id: str
    birth_frame: int
    death_frame: Optional[int] = None

    observations: List[TrackObservation] = field(default_factory=list)

    def add_observation(self, obs: TrackObservation) -> None:
        self.observations.append(obs)

    def compute_stats(self) -> Dict[str, Any]:
        """Compute aggregate tracking quality metrics."""
        if not self.observations:
            return {}

        obs = self.observations
        duration = obs[-1].sequence - obs[0].sequence + 1

        # Displacements
        displacements = [o.prev_distance_m for o in obs[1:] if o.prev_distance_m is not None]
        velocities = [o.velocity_mm_per_frame for o in obs[1:] if o.velocity_mm_per_frame is not None]
        volume_ratios = [o.volume_ratio for o in obs[1:] if o.volume_ratio is not None]

        stats = {
            "track_id": self.track_id,
            "birth_frame": self.birth_frame,
            "death_frame": self.death_frame,
            "duration_frames": duration,
            "observation_count": len(obs),
            "gaps_count": duration - len(obs),
            "is_active": self.death_frame is None,
        }

        if displacements:
            stats.update({
                "avg_displacement_m": sum(displacements) / len(displacements),
                "max_displacement_m": max(displacements),
                "min_displacement_m": min(displacements),
            })

        if velocities:
            stats.update({
                "avg_velocity_mm_per_frame": sum(velocities) / len(velocities),
                "velocity_stdev": _stdev(velocities),
                "max_velocity_mm_per_frame": max(velocities),
            })

        if volume_ratios:
            stats.update({
                "avg_volume_stability": sum(volume_ratios) / len(volume_ratios),
                "volume_stability_stdev": _stdev(volume_ratios),
            })

        # Depth consistency
        depths = [o.depth_mean_m for o in obs]
        if depths:
            stats.update({
                "avg_depth_m": sum(depths) / len(depths),
                "depth_stdev_m": _stdev(depths),
            })

        return stats


class TrackingQualityRecorder:
    """Record and analyze persistent object tracking quality over time."""

    def __init__(self, enabled: bool, output_dir: str, logger: Any) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.logger = logger

        self.frame_events: List[Dict[str, Any]] = []
        self.track_lifecycles: Dict[str, TrackLifecycle] = {}
        self.association_decisions: List[AssociationDecision] = []

        self._lock = Lock()
        self.frame_count = 0
        self.track_count = 0

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Tracking quality recording enabled. Output: {self.output_dir}")
        else:
            self.logger.info("Tracking quality recording disabled.")

    def log_frame_start(self, frame_id: str, sequence: int, sam_mask_count: int) -> None:
        """Log frame processing start with SAM mask count."""
        if not self.enabled:
            return
        self.frame_count += 1
        with self._lock:
            self.frame_events.append({
                "frame_id": frame_id,
                "sequence": sequence,
                "timestamp_iso": datetime.utcnow().isoformat(),
                "sam_mask_count": sam_mask_count,
                "tracking_decisions": [],
                "tracks_born": [],
                "tracks_died": [],
            })

    def log_association_decision(
        self,
        frame_id: str,
        sequence: int,
        mask_id: int,
        mask_area_px: float,
        mask_centroid_3d: List[float],
        matched_track_id: Optional[str],
        match_type: str,
        match_distance_m: Optional[float] = None,
        match_iou_3d: Optional[float] = None,
        match_score: Optional[float] = None,
        prev_track_age_frames: Optional[int] = None,
        prev_track_observations: Optional[int] = None,
        prev_centroid_3d: Optional[List[float]] = None,
        prev_bbox_volume_m3: Optional[float] = None,
        reason: str = "",
    ) -> None:
        """Log a single mask-to-track association decision."""
        if not self.enabled:
            return

        decision = AssociationDecision(
            frame_id=frame_id,
            sequence=sequence,
            mask_id=mask_id,
            mask_area_px=mask_area_px,
            mask_centroid_3d=mask_centroid_3d,
            matched_track_id=matched_track_id,
            match_type=match_type,
            match_distance_m=match_distance_m,
            match_iou_3d=match_iou_3d,
            match_score=match_score,
            prev_track_age_frames=prev_track_age_frames,
            prev_track_observations=prev_track_observations,
            prev_centroid_3d=prev_centroid_3d,
            prev_bbox_volume_m3=prev_bbox_volume_m3,
            reason=reason,
            timestamp_iso=datetime.utcnow().isoformat(),
        )

        with self._lock:
            self.association_decisions.append(decision)
            # Also add to frame event
            if self.frame_events:
                self.frame_events[-1]["tracking_decisions"].append(asdict(decision))

    def log_track_observation(
        self,
        track_id: str,
        frame_id: str,
        sequence: int,
        centroid_3d: List[float],
        centroid_2d: List[float],
        bbox_volume_m3: float,
        mask_area_px: int,
        depth_mean_m: float,
        mask_iou_3d: Optional[float] = None,
        quality_score: float = 1.0,
        global_association_components: Optional[Dict[str, float]] = None,
        match_reason: Optional[str] = None,
    ) -> None:
        """Log an observation of a track (add to its lifecycle)."""
        if not self.enabled:
            return

        obs = TrackObservation(
            frame_id=frame_id,
            sequence=sequence,
            centroid_3d=centroid_3d,
            centroid_2d=centroid_2d,
            bbox_volume_m3=bbox_volume_m3,
            mask_area_px=mask_area_px,
            depth_mean_m=depth_mean_m,
            mask_iou_3d=mask_iou_3d,
            observation_quality_score=quality_score,
            timestamp_iso=datetime.utcnow().isoformat(),
            global_association_components=global_association_components,
            match_reason=match_reason,
        )

        with self._lock:
            if track_id not in self.track_lifecycles:
                self.track_lifecycles[track_id] = TrackLifecycle(
                    track_id=track_id,
                    birth_frame=sequence
                )

            # Compute relative to previous observation
            lifecycle = self.track_lifecycles[track_id]
            if lifecycle.observations:
                prev_obs = lifecycle.observations[-1]
                obs.displacement_3d = [
                    c1 - c2 for c1, c2 in zip(centroid_3d, prev_obs.centroid_3d)
                ]
                obs.prev_distance_m = (
                    sum(d**2 for d in obs.displacement_3d) ** 0.5
                )
                obs.velocity_mm_per_frame = obs.prev_distance_m * 1000

                if prev_obs.bbox_volume_m3 > 0:
                    obs.volume_ratio = bbox_volume_m3 / prev_obs.bbox_volume_m3

            lifecycle.add_observation(obs)

    def log_track_birth(self, frame_id: str, sequence: int, track_id: str) -> None:
        """Log when a new track is created."""
        if not self.enabled:
            return
        self.track_count += 1
        with self._lock:
            if self.frame_events:
                self.frame_events[-1]["tracks_born"].append(track_id)

    def log_track_death(self, frame_id: str, sequence: int, track_id: str) -> None:
        """Log when a track is terminated."""
        if not self.enabled:
            return
        with self._lock:
            if self.frame_events:
                self.frame_events[-1]["tracks_died"].append(track_id)

            if track_id in self.track_lifecycles:
                self.track_lifecycles[track_id].death_frame = sequence

    def save_snapshots(self, suffix: str = "") -> None:
        """Save all tracked events to JSONL files."""
        if not self.enabled:
            return

        with self._lock:
            frame_events = list(self.frame_events)
            association_decisions = list(self.association_decisions)
            track_lifecycles = dict(self.track_lifecycles)

        if not frame_events and not association_decisions:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Save frame-level events
        try:
            filepath = self.output_dir / f"tracking_frames_{timestamp}{suffix}.jsonl"
            with open(filepath, "w") as f:
                for event in frame_events:
                    f.write(json.dumps(event) + "\n")
            self.logger.info(f"Saved {len(frame_events)} frame events to {filepath}")
        except Exception as exc:
            self.logger.error(f"Failed to save frame events: {exc}")

        # Save association decisions
        try:
            filepath = self.output_dir / f"tracking_associations_{timestamp}{suffix}.jsonl"
            with open(filepath, "w") as f:
                for decision in association_decisions:
                    f.write(json.dumps(asdict(decision)) + "\n")
            self.logger.info(f"Saved {len(association_decisions)} association decisions to {filepath}")
        except Exception as exc:
            self.logger.error(f"Failed to save associations: {exc}")

        # Save track lifecycles
        try:
            filepath = self.output_dir / f"tracking_lifecycles_{timestamp}{suffix}.json"
            lifecycles_data = {
                tid: {
                    **lifecycle.compute_stats(),
                    "observations": [asdict(o) for o in lifecycle.observations],
                } for tid, lifecycle in track_lifecycles.items()
            }
            with open(filepath, "w") as f:
                json.dump(lifecycles_data, f, indent=2)
            self.logger.info(f"Saved {len(track_lifecycles)} track lifecycles to {filepath}")
        except Exception as exc:
            self.logger.error(f"Failed to save track lifecycles: {exc}")

    def generate_report(self, suffix: str = "") -> None:
        """Generate human-readable tracking quality report."""
        if not self.enabled:
            return

        with self._lock:
            track_lifecycles = dict(self.track_lifecycles)
            frame_events = list(self.frame_events)
            association_decisions = list(self.association_decisions)

        if not track_lifecycles:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"tracking_quality_report_{timestamp}{suffix}.md"

        try:
            with open(filepath, "w") as f:
                f.write("# Tracking Quality Report\n\n")
                f.write(f"Generated: {datetime.utcnow().isoformat()}\n")
                f.write(f"Total frames: {len(frame_events)}\n")
                f.write(f"Total unique tracks: {len(track_lifecycles)}\n\n")

                # Summary statistics
                f.write("## Summary Statistics\n\n")

                all_stats = [lifecycle.compute_stats() for lifecycle in track_lifecycles.values()]
                durations = [s.get("duration_frames", 0) for s in all_stats]

                if durations:
                    f.write(f"### Track Continuity\n")
                    f.write(f"- Average track duration: {sum(durations)/len(durations):.1f} frames\n")
                    f.write(f"- Longest track: {max(durations)} frames\n")
                    f.write(f"- Shortest track: {min(durations)} frames\n")
                    f.write(f"- Median track duration: {sorted(durations)[len(durations)//2]} frames\n\n")

                velocities = [s.get("avg_velocity_mm_per_frame", 0) for s in all_stats if "avg_velocity_mm_per_frame" in s]
                if velocities:
                    f.write(f"### Spatial Stability\n")
                    f.write(f"- Avg centroid velocity: {sum(velocities)/len(velocities):.1f} mm/frame\n")
                    f.write(f"- Max velocity: {max(velocities):.1f} mm/frame\n")
                    f.write(f"- Min velocity: {min(velocities):.1f} mm/frame\n\n")

                volume_stabs = [s.get("avg_volume_stability", 1.0) for s in all_stats if "avg_volume_stability" in s]
                if volume_stabs:
                    f.write(f"### Volume Stability\n")
                    f.write(f"- Avg volume stability: {sum(volume_stabs)/len(volume_stabs):.3f}\n")
                    f.write(f"- Min stability: {min(volume_stabs):.3f}\n\n")

                # Association statistics
                if association_decisions:
                    f.write(f"### Association Quality\n")
                    match_types = {}
                    for decision in association_decisions:
                        mt = decision.match_type
                        match_types[mt] = match_types.get(mt, 0) + 1

                    total_assoc = len(association_decisions)
                    for mt, count in sorted(match_types.items(), key=lambda x: x[1], reverse=True):
                        pct = 100 * count / total_assoc
                        f.write(f"- {mt}: {count} ({pct:.1f}%)\n")
                    f.write("\n")

                # Top tracks by duration
                f.write("## Top 10 Longest Tracks\n\n")
                top_tracks = sorted(all_stats, key=lambda s: s.get("duration_frames", 0), reverse=True)[:10]
                for rank, stats in enumerate(top_tracks, 1):
                    tid = stats["track_id"]
                    dur = stats.get("duration_frames", 0)
                    vel = stats.get("avg_velocity_mm_per_frame", 0)
                    f.write(f"{rank}. {tid}: {dur} frames, avg velocity {vel:.1f} mm/frame\n")

                f.write("\n")

            self.logger.info(f"Saved tracking quality report to {filepath}")
        except Exception as exc:
            self.logger.error(f"Failed to generate report: {exc}")


def _stdev(values: List[float]) -> float:
    """Compute standard deviation."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5
