"""Unit tests for structured VLM semantic and mobility validation."""

from __future__ import annotations

import json
import unittest

from nodes.support.phase1.vlm_result import validate_vlm_response


DYNAMIC_HINTS = [
    "person", "human", "man", "woman", "child", "pedestrian",
    "dog", "cat", "animal", "horse", "bird",
    "mobile_robot", "robot_dog", "agv", "amr", "drone",
]
STATIC_HINTS = [
    "chair", "office_chair", "wheelchair", "table", "desk", "door",
    "curtain", "fan", "robot_arm", "cabinet", "sofa", "bed", "wall",
    "floor", "ceiling",
]


def validate(payload: object):
    """Validate one payload using the package default confidence thresholds."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return validate_vlm_response(
        text,
        min_label_confidence=0.35,
        min_mobility_confidence=0.50,
        dynamic_label_hints=DYNAMIC_HINTS,
        static_label_hints=STATIC_HINTS,
    )


class VlmResultValidationTests(unittest.TestCase):
    """Exercise accepted, corrected, uncertain, and malformed VLM outputs."""

    def test_dynamic_person_is_accepted(self) -> None:
        """Accept a confident human result as dynamic."""
        result = validate({
            "label": "person",
            "label_confidence": 0.98,
            "mobility_class": "dynamic",
            "mobility_confidence": 0.99,
        })
        self.assertTrue(result.success)
        self.assertEqual(result.label, "person")
        self.assertEqual(result.mobility_class, "dynamic")

    def test_static_contradiction_is_corrected(self) -> None:
        """Correct an impossible dynamic office-chair classification."""
        result = validate({
            "label": "office chair",
            "label_confidence": 0.94,
            "mobility_class": "dynamic",
            "mobility_confidence": 0.97,
        })
        self.assertTrue(result.success)
        self.assertEqual(result.label, "office_chair")
        self.assertEqual(result.mobility_class, "static")
        self.assertIn("dynamic_corrected_by_static_hint", result.validation_reason)

    def test_dynamic_contradiction_is_corrected(self) -> None:
        """Correct a confident animal label misclassified as static."""
        result = validate({
            "label": "dog",
            "label_confidence": 0.95,
            "mobility_class": "static",
            "mobility_confidence": 0.96,
        })
        self.assertTrue(result.success)
        self.assertEqual(result.mobility_class, "dynamic")
        self.assertIn("mobility_corrected_by_dynamic_hint", result.validation_reason)

    def test_low_mobility_confidence_preserves_label(self) -> None:
        """Keep the semantic label while downgrading uncertain mobility."""
        result = validate({
            "label": "cabinet",
            "label_confidence": 0.91,
            "mobility_class": "static",
            "mobility_confidence": 0.20,
        })
        self.assertTrue(result.success)
        self.assertEqual(result.label, "cabinet")
        self.assertEqual(result.mobility_class, "unknown")
        self.assertIn("low_mobility_confidence", result.validation_reason)

    def test_percentage_confidences_are_normalised(self) -> None:
        """Normalise confidence percentages to the unit interval."""
        result = validate({
            "label": "mobile robot",
            "label_confidence": 92,
            "mobility_class": "dynamic",
            "mobility_confidence": 95,
        })
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.label_confidence, 0.92)
        self.assertAlmostEqual(result.mobility_confidence, 0.95)

    def test_low_mobility_confidence_is_not_overridden_by_label_hint(self) -> None:
        """Keep mobility unknown when a human result has weak mobility evidence."""
        result = validate({
            "label": "person",
            "label_confidence": 0.98,
            "mobility_class": "dynamic",
            "mobility_confidence": 0.20,
        })
        self.assertTrue(result.success)
        self.assertEqual(result.mobility_class, "unknown")

    def test_malformed_response_is_rejected(self) -> None:
        """Reject a response that does not contain a JSON object."""
        result = validate("this is not JSON")
        self.assertFalse(result.success)
        self.assertEqual(result.label, "unknown_object")
        self.assertEqual(result.mobility_class, "unknown")

    def test_model_abstention_is_recorded_as_vlm_no_result(self) -> None:
        """A deliberate ``VLM_no_result`` abstention keeps that label, not ``unknown_object``."""
        result = validate({
            "label": "VLM_no_result",
            "label_confidence": 0.0,
            "mobility_class": "unknown",
            "mobility_confidence": 0.0,
        })
        self.assertFalse(result.success)
        self.assertEqual(result.label, "VLM_no_result")
        self.assertEqual(result.validation_status, "rejected")
        self.assertIn("model_abstained", result.validation_reason)

    def test_legacy_vlm_unknown_alias_maps_to_no_result(self) -> None:
        """The pre-rename ``VLM_unknown`` token is still treated as an abstention."""
        result = validate({
            "label": "VLM_unknown",
            "label_confidence": 0.0,
            "mobility_class": "unknown",
            "mobility_confidence": 0.0,
        })
        self.assertFalse(result.success)
        self.assertEqual(result.label, "VLM_no_result")

    def test_low_confidence_real_guess_stays_unknown_object(self) -> None:
        """A weak but genuine label is rejected as ``unknown_object``, not an abstention."""
        result = validate({
            "label": "chair",
            "label_confidence": 0.10,
            "mobility_class": "static",
            "mobility_confidence": 0.90,
        })
        self.assertFalse(result.success)
        self.assertEqual(result.label, "unknown_object")
        self.assertIn("unknown_or_low_label_confidence", result.validation_reason)


if __name__ == "__main__":
    unittest.main()
