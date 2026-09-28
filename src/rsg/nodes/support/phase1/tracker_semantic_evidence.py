"""Label/mobility evidence accumulation and commit for PersistentObjectTracker.

Every method here assumes the caller already holds ``tracker._lock`` --
nothing in this file acquires or releases it.
"""

from __future__ import annotations

from typing import Any, Tuple

from nodes.support.phase1.tracker_geometry import _normalise_label


class TrackerSemanticEvidence:
    """Owns per-track label evidence accumulation and the final label choice."""

    def __init__(self, tracker: Any):
        self.tracker = tracker

    @staticmethod
    def _source_rank(source: str) -> int:
        return {"pending": 0, "rap": 1, "vlm": 2}.get(str(source), 0)

    def _canonicalise_label(self, raw_label: Any) -> str:
        label = _normalise_label(raw_label)
        if not label or label in {"unknown", "unknown object", "unknown_object"}:
            return ""
        aliases = getattr(self.tracker.config, "persistent_label_aliases", {}) or {}
        return _normalise_label(aliases.get(label, label))

    def _source_weight(self, source: str) -> float:
        if source == "vlm":
            return float(getattr(self.tracker.config, "persistent_vlm_evidence_weight", 1.0))
        if source == "rap":
            return float(getattr(self.tracker.config, "persistent_rap_evidence_weight", 0.70))
        return 0.0

    def _add_evidence(self, track: Any, raw_label: str, source: str, confidence: float) -> None:
        label = self._canonicalise_label(raw_label)
        if not label:
            return
        score = max(0.0, min(1.0, float(confidence))) * max(0.0, self._source_weight(source))
        if score <= 0.0:
            return
        track.label_evidence[label] = float(track.label_evidence.get(label, 0.0) + score)
        track.label_observations[label] = int(track.label_observations.get(label, 0) + 1)

    def _update_semantics(self, track: Any, raw_label: str, source: str, confidence: float) -> None:
        label = self._canonicalise_label(raw_label)
        if not label:
            return
        if source == "rap":
            track.raw_rap_label = label
        elif source == "vlm":
            track.raw_vlm_label = label
        self._add_evidence(track, label, source, confidence)
        candidate_rank = self._source_rank(source)
        current_rank = self._source_rank(track.label_source)
        if candidate_rank > current_rank or (candidate_rank == current_rank and confidence >= track.label_confidence):
            track.canonical_label = label
            track.label_source = source
            track.label_confidence = float(confidence)

    @staticmethod
    def _update_mobility(
        track: Any,
        *,
        mobility_class: Any,
        confidence: float,
        source: str,
    ) -> None:
        """Keep the strongest valid static/dynamic/unknown mobility decision."""
        mobility = str(mobility_class or "unknown").strip().lower()
        if mobility not in {"static", "dynamic", "unknown"}:
            mobility = "unknown"
        score = max(0.0, min(1.0, float(confidence)))
        if mobility == "unknown" and track.mobility_class in {"static", "dynamic"}:
            return
        source_name = str(source)
        current_source = str(track.mobility_source)
        source_rank = 2 if source_name.startswith("vlm") else 1 if source_name.startswith("rap") else 0
        current_rank = 2 if current_source.startswith("vlm") else 1 if current_source.startswith("rap") else 0
        if source_rank > current_rank or (source_rank == current_rank and score >= track.mobility_confidence):
            track.mobility_class = mobility
            track.mobility_confidence = score
            track.mobility_source = str(source)

    def _commit_semantic_label(self, track: Any, timestamp_sec: float, reason: str) -> None:
        semantic_label, final_source, final_confidence, semantic_hydra_class_id = self._choose_semantic_label(track)
        track.slot_state = "semantic_resolved"
        track.semantic_timestamp_sec = float(timestamp_sec)
        track.semantic_update_count += 1
        track.semantic_label = semantic_label
        track.semantic_label_source = final_source
        track.semantic_label_confidence = float(final_confidence)
        track.semantic_hydra_class_id = int(semantic_hydra_class_id)
        track.semantic_reason = str(reason)

    def _choose_semantic_label(self, track: Any) -> Tuple[str, str, float, int]:
        total = float(sum(track.label_evidence.values()))
        if not track.label_evidence:
            return "unclassified_object", "none", 0.0, 0
        best_label, best_score = max(track.label_evidence.items(), key=lambda item: item[1])
        consensus = float(best_score / total) if total > 0.0 else 0.0
        observations = int(track.label_observations.get(best_label, 0))
        enough_observations = track.seen_count >= int(self.tracker.config.semantic_result_min_observations)
        enough_consensus = consensus >= float(self.tracker.config.semantic_result_min_consensus)
        enough_confidence = best_score >= float(self.tracker.config.semantic_result_min_evidence)
        if not (enough_observations and enough_consensus and enough_confidence):
            return "unclassified_object", "insufficient_evidence", consensus, 0
        # The slot remains the physical identity. The selected label is stored
        # only as semantic metadata for the downstream Hydra/RAP fuser.
        class_id = int(track.hydra_label_id) if int(track.hydra_label_id) > 0 else 0
        source = "vlm" if _normalise_label(track.raw_vlm_label) == best_label else "rap"
        # ``consensus`` measures agreement between candidate labels; it is not
        # model confidence. With one observation it is always 1.0 and hid the
        # actual RAP/VLM confidence from the fuser and diagnostics.
        raw_confidence = float(track.label_confidence)
        if _normalise_label(track.canonical_label) != best_label:
            raw_confidence = min(1.0, max(0.0, float(best_score)))
        return best_label, source, raw_confidence, class_id
