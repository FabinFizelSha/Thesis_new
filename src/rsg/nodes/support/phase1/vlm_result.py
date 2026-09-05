"""Validate structured VLM labels and mobility metadata.

The local Qwen server is asked to return one compact JSON object.  This module
keeps response parsing and validation independent from ROS so it can be tested
without starting the Phase 1 node.  The validator is deliberately conservative:
a valid semantic label can be accepted even when mobility is uncertain, but an
invalid or low-confidence mobility decision is normalised to ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


UNKNOWN_LABELS = {
    "",
    "unknown",
    "unknown_object",
    "unclear",
    "none",
    "n/a",
}
# Labels the model itself emits when it looked at the crop and could not
# identify the target.  Kept separate from UNKNOWN_LABELS so a deliberate
# model abstention is recorded as ``VLM_no_result`` rather than collapsing
# into ``unknown_object`` (which stays reserved for parse failures, transport
# errors, and sub-threshold real guesses).
NO_RESULT_LABEL = "VLM_no_result"
NO_RESULT_LABELS = {
    "vlm_no_result",
    "vlm_unknown",
    "no_result",
}
VALID_MOBILITY_CLASSES = {"static", "dynamic", "unknown"}
_MOBILITY_ALIASES = {
    "stationary": "static",
    "fixed": "static",
    "non_dynamic": "static",
    "non-dynamic": "static",
    "movable": "dynamic",
    "mobile": "dynamic",
    "moving": "dynamic",
    "uncertain": "unknown",
    "unsure": "unknown",
}


@dataclass(frozen=True)
class ValidatedVlmResult:
    """Normalised semantic and mobility output from one VLM response."""

    success: bool
    label: str
    label_confidence: float
    mobility_class: str
    mobility_confidence: float
    validation_status: str
    validation_reason: str
    parsed_payload: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation used by Phase 1."""
        return {
            "success": bool(self.success),
            "label": self.label,
            "confidence": float(self.label_confidence),
            "label_confidence": float(self.label_confidence),
            "mobility_class": self.mobility_class,
            "mobility_confidence": float(self.mobility_confidence),
            "validation_status": self.validation_status,
            "validation_reason": self.validation_reason,
            "parsed_payload": dict(self.parsed_payload),
        }


def normalise_label(value: Any) -> str:
    """Convert a model label to the lowercase underscore form used by RSG."""
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = " ".join(text.split())
    if not text:
        return "unknown_object"
    return text.replace(" ", "_")[:80]


def _normalise_mobility(value: Any) -> str:
    """Map mobility aliases to the canonical three-class vocabulary."""
    text = str(value or "unknown").strip().lower().replace(" ", "_")
    text = _MOBILITY_ALIASES.get(text, text)
    return text if text in VALID_MOBILITY_CLASSES else "unknown"


def _normalise_confidence(value: Any) -> Optional[float]:
    """Return a confidence in ``[0, 1]`` while accepting percentage notation."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < confidence <= 100.0:
        confidence /= 100.0
    if not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def _extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extract the first JSON object from plain text or a fenced response."""
    raw = str(text or "").strip()
    if not raw:
        return None, "empty_response"
    fenced = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced).strip()
    decoder = json.JSONDecoder()
    for candidate in (fenced, raw):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, "json_exact"
        except Exception:
            pass
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed, "json_embedded"
    return None, "json_not_found"


def _contains_hint(label: str, hints: Iterable[str]) -> bool:
    """Return whether a normalised label contains one configured phrase."""
    words = f"_{normalise_label(label)}_"
    for hint in hints:
        token = normalise_label(hint)
        if token and f"_{token}_" in words:
            return True
    return False


def infer_mobility_from_label(
    label: str,
    *,
    dynamic_label_hints: Iterable[str],
    static_label_hints: Iterable[str],
) -> str:
    """Infer a conservative mobility class from a validated semantic label.

    Static hints take precedence so broad dynamic tokens cannot incorrectly
    classify objects such as a fixed robot arm. The helper is used only as a
    consistency check or when an older RAP backend cannot return metadata.
    """
    if _contains_hint(label, static_label_hints):
        return "static"
    if _contains_hint(label, dynamic_label_hints):
        return "dynamic"
    return "unknown"


@dataclass(frozen=True)
class ValidatedRiskResult:
    """Normalised output from one Risk VLM response.

    Deliberately narrower than ``ValidatedVlmResult``: there is no
    label/mobility contract here, just a clamped hazard score and a short,
    capped list of hazard strings.
    """

    success: bool
    risk_score: float  # signed, [-1.0, 1.0]: negative = risk-reducing, 0 = neutral, positive = hazard
    risk_factors: Tuple[str, ...]
    validation_status: str
    validation_reason: str
    parsed_payload: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation used by Phase 1."""
        return {
            "success": bool(self.success),
            "risk_score": float(self.risk_score),
            "risk_factors": list(self.risk_factors),
            "validation_status": self.validation_status,
            "validation_reason": self.validation_reason,
            "parsed_payload": dict(self.parsed_payload),
        }


def _normalise_risk_score(value: Any) -> Optional[float]:
    """Return a risk score clamped to ``[-1.0, 1.0]``.

    Unlike confidence values (``_normalise_confidence``, always ``[0, 1]``),
    risk scores are signed: negative means the object actively *reduces*
    risk (safety equipment such as a fire extinguisher, hazard-warning
    signage), 0 is neutral (no hazard, no mitigation), positive is a hazard.
    No percentage-notation handling here -- a model returning e.g. ``-150``
    almost certainly means it ignored the range instruction, and clamping
    straight to ``[-1, 1]`` is simpler and safer than guessing a percentage
    scale for a signed value.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(-1.0, min(1.0, score))


def validate_risk_response(
    text: str,
    *,
    max_risk_factors: int = 5,
    max_risk_factor_length: int = 120,
) -> ValidatedRiskResult:
    """Parse and validate the two-field risk JSON contract: ``risk_score`` +
    ``risk_factors``.

    ``risk_score`` ranges over ``[-1.0, 1.0]``, not ``[0.0, 1.0]``: negative
    means the object actively reduces risk (safety equipment, hazard-warning
    signage), 0 is neutral, positive is a hazard -- see
    ``_normalise_risk_score``. A missing or unparsable ``risk_score`` fails
    the whole result -- it is the one field every downstream consumer
    (fuser metadata, any future risk-based filtering) needs to be
    trustworthy. An empty ``risk_factors`` list is *not* a failure on its
    own: a genuinely neutral object can legitimately have no hazards or
    mitigations to report. ``risk_factors`` is capped to ``max_risk_factors``
    entries, each truncated to ``max_risk_factor_length`` characters, so a
    verbose or malformed model response can never grow the fuser's per-node
    metadata or an RViz label without bound.
    """
    payload, parse_status = _extract_json_object(text)
    if payload is None:
        return ValidatedRiskResult(
            success=False,
            risk_score=0.0,
            risk_factors=(),
            validation_status="rejected",
            validation_reason=parse_status,
            parsed_payload={},
        )

    risk_score = _normalise_risk_score(payload.get("risk_score"))
    if risk_score is None:
        return ValidatedRiskResult(
            success=False,
            risk_score=0.0,
            risk_factors=(),
            validation_status="rejected",
            validation_reason="invalid_risk_score",
            parsed_payload=payload,
        )

    raw_factors = payload.get("risk_factors", [])
    if isinstance(raw_factors, str):
        raw_factors = [raw_factors]
    factors: List[str] = []
    if isinstance(raw_factors, list):
        limit = max(0, int(max_risk_factors))
        for item in raw_factors:
            if len(factors) >= limit:
                break
            text_item = str(item or "").strip()
            if text_item:
                factors.append(text_item[: max(1, int(max_risk_factor_length))])

    # A non-zero score without any stated reason is not trustworthy: the
    # prompt asks the model to always name what's driving a score away from
    # neutral, but real responses have come back with e.g. risk_score=0.1
    # and risk_factors=[] -- a number with no justification behind it.
    # Mandatory, not just prompted: reject rather than publish an
    # unexplained non-zero score. Uses a small epsilon, not a literal
    # `!= 0.0`, purely to tolerate float representation noise (e.g. a value
    # arithmetic left at 1e-17 instead of exactly 0.0) -- it is not a
    # "close enough to neutral" allowance.
    if abs(risk_score) > 1e-9 and not factors:
        return ValidatedRiskResult(
            success=False,
            risk_score=float(risk_score),
            risk_factors=(),
            validation_status="rejected",
            validation_reason="nonzero_score_missing_factors",
            parsed_payload=payload,
        )

    return ValidatedRiskResult(
        success=True,
        risk_score=float(risk_score),
        risk_factors=tuple(factors),
        validation_status="accepted",
        validation_reason=parse_status,
        parsed_payload=payload,
    )


def validate_vlm_response(
    text: str,
    *,
    min_label_confidence: float,
    min_mobility_confidence: float,
    dynamic_label_hints: Iterable[str],
    static_label_hints: Iterable[str],
) -> ValidatedVlmResult:
    """Parse and validate the four-field VLM JSON contract.

    Validation rules:

    * label and both confidence fields must be present and well formed;
    * low-confidence semantic labels become ``unknown_object``;
    * mobility must be ``static``, ``dynamic``, or ``unknown``;
    * low-confidence mobility becomes ``unknown`` without discarding a valid
      object label;
    * obvious static-object labels override a contradictory ``dynamic`` result;
    * configured human, animal, and mobile-robot labels override a contradictory
      ``static`` result after the confidence checks have passed.
    """
    payload, parse_status = _extract_json_object(text)
    if payload is None:
        return ValidatedVlmResult(
            success=False,
            label="unknown_object",
            label_confidence=0.0,
            mobility_class="unknown",
            mobility_confidence=0.0,
            validation_status="rejected",
            validation_reason=parse_status,
            parsed_payload={},
        )

    label = normalise_label(payload.get("label", payload.get("object_label", "unknown_object")))
    label_confidence = _normalise_confidence(
        payload.get("label_confidence", payload.get("confidence"))
    )
    mobility = _normalise_mobility(
        payload.get("mobility_class", payload.get("mobility", "unknown"))
    )
    mobility_confidence = _normalise_confidence(
        payload.get("mobility_confidence", payload.get("dynamic_confidence"))
    )

    reasons = [parse_status]
    if label_confidence is None:
        label_confidence = 0.0
        reasons.append("invalid_label_confidence")
    if mobility_confidence is None:
        mobility_confidence = 0.0
        reasons.append("invalid_mobility_confidence")

    model_abstained = label in NO_RESULT_LABELS
    if model_abstained or label in UNKNOWN_LABELS or label_confidence < float(min_label_confidence):
        reasons.append("model_abstained" if model_abstained else "unknown_or_low_label_confidence")
        return ValidatedVlmResult(
            success=False,
            label=NO_RESULT_LABEL if model_abstained else "unknown_object",
            label_confidence=float(label_confidence),
            mobility_class="unknown",
            mobility_confidence=float(mobility_confidence),
            validation_status="rejected",
            validation_reason=";".join(reasons),
            parsed_payload=payload,
        )

    mobility_is_confident = mobility_confidence >= float(min_mobility_confidence)
    if not mobility_is_confident:
        mobility = "unknown"
        reasons.append("low_mobility_confidence")

    if mobility_is_confident:
        inferred_mobility = infer_mobility_from_label(
            label,
            dynamic_label_hints=dynamic_label_hints,
            static_label_hints=static_label_hints,
        )
        if mobility == "dynamic" and inferred_mobility == "static":
            mobility = "static"
            reasons.append("dynamic_corrected_by_static_hint")
        elif mobility in {"static", "unknown"} and inferred_mobility == "dynamic":
            mobility = "dynamic"
            reasons.append("mobility_corrected_by_dynamic_hint")
        elif mobility == "dynamic" and inferred_mobility == "dynamic":
            reasons.append("dynamic_hint_confirmed")

    return ValidatedVlmResult(
        success=True,
        label=label,
        label_confidence=float(label_confidence),
        mobility_class=mobility,
        mobility_confidence=float(mobility_confidence),
        validation_status="accepted",
        validation_reason=";".join(reasons),
        parsed_payload=payload,
    )
