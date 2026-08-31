"""Generate analysis and report for VLM prompt optimization experiment."""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


class VLMExperimentReport:
    """Generate accuracy metrics and analysis from verification results."""

    def __init__(self, session_dir: Path):
        """Initialize report generator.

        Args:
            session_dir: Session directory containing results
        """
        self.session_dir = Path(session_dir)
        self.verification_file = self.session_dir / "verification_results.csv"
        self.vlm_outputs_file = self.session_dir / "vlm_outputs.jsonl"

    def generate_metrics(self) -> Dict[str, Any]:
        """Generate accuracy metrics from verified results.

        Returns:
            Dictionary with metrics
        """
        results = self._load_verification_results()
        if not results:
            return {"error": "No verification results found"}

        total = len(results)
        verified = sum(1 for r in results if r.get("is_correct") is not None)
        correct = sum(1 for r in results if r.get("is_correct") is True)

        accuracy = (correct / verified * 100) if verified > 0 else 0
        false_positive = sum(1 for r in results if r.get("is_correct") is False and r.get("vlm_confidence", 0) > 0.5)
        false_negative = sum(1 for r in results if r.get("is_correct") is True and r.get("vlm_confidence", 0) <= 0.5)

        # Processing time stats
        proc_times = [r.get("vlm_processing_time_ms", 0) for r in results if r.get("vlm_processing_time_ms")]
        avg_time = sum(proc_times) / len(proc_times) if proc_times else 0
        min_time = min(proc_times) if proc_times else 0
        max_time = max(proc_times) if proc_times else 0

        # Confidence analysis
        confidence_data = defaultdict(lambda: {"count": 0, "correct": 0})
        for r in results:
            if r.get("is_correct") is not None:
                conf_bin = f"{int(r.get('vlm_confidence', 0) * 100)}%"
                confidence_data[conf_bin]["count"] += 1
                if r.get("is_correct"):
                    confidence_data[conf_bin]["correct"] += 1

        return {
            "total_samples": total,
            "verified_samples": verified,
            "correct_predictions": correct,
            "accuracy_percent": round(accuracy, 2),
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "processing_time_ms": {
                "average": round(avg_time, 2),
                "min": round(min_time, 2),
                "max": round(max_time, 2),
            },
            "confidence_analysis": dict(confidence_data),
        }

    def _load_verification_results(self) -> List[Dict[str, Any]]:
        """Load verification results from CSV."""
        results = []
        try:
            with open(self.verification_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("manual_verified"):
                        result = {
                            "object_id": row.get("object_id"),
                            "vlm_class": row.get("vlm_class"),
                            "vlm_confidence": float(row.get("vlm_confidence", 0)),
                            "vlm_processing_time_ms": float(row.get("vlm_processing_time_ms", 0)),
                            "manual_class": row.get("manual_class"),
                            "is_correct": row.get("is_correct", "").lower() == "true",
                        }
                        results.append(result)
        except FileNotFoundError:
            pass
        return results

    def generate_markdown_report(self, prompt_version: str = "v1_production") -> str:
        """Generate markdown report for thesis chapter.

        Args:
            prompt_version: Version of prompt tested

        Returns:
            Markdown formatted report
        """
        metrics = self.generate_metrics()

        if "error" in metrics:
            return f"# VLM Prompt Optimization Report\n\n{metrics['error']}"

        report = f"""# VLM Prompt Optimization: {prompt_version}

## Executive Summary

This experiment evaluates VLM (Vision Language Model) performance for object classification with a specific prompt formulation.

**Prompt Version:** {prompt_version}
**Test Duration:** 300 seconds (wall time)
**RAP Enabled:** No (all classifications to VLM)

## Results

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| Total Samples | {metrics['total_samples']} |
| Verified Samples | {metrics['verified_samples']} |
| Correct Predictions | {metrics['correct_predictions']} |
| **Overall Accuracy** | **{metrics['accuracy_percent']}%** |
| False Positives | {metrics['false_positives']} |
| False Negatives | {metrics['false_negatives']} |

### Processing Performance

| Metric | Value |
|--------|-------|
| Average Processing Time | {metrics['processing_time_ms']['average']} ms |
| Min Processing Time | {metrics['processing_time_ms']['min']} ms |
| Max Processing Time | {metrics['processing_time_ms']['max']} ms |

### Confidence Analysis

The following table shows prediction accuracy stratified by VLM confidence levels:

| Confidence | Count | Correct | Accuracy |
|------------|-------|---------|----------|
"""

        for conf_bin, data in sorted(metrics.get("confidence_analysis", {}).items()):
            count = data["count"]
            correct = data["correct"]
            acc = (correct / count * 100) if count > 0 else 0
            report += f"| {conf_bin} | {count} | {correct} | {acc:.1f}% |\n"

        report += """
## Analysis

### Strengths
- Evaluate based on accuracy and processing time
- Identify high-confidence correct predictions
- Determine failure modes

### Weaknesses
- Point out low-confidence or incorrect predictions
- Identify systematic errors

### Recommendations

For prompt improvements, consider:
1. Cases where model was wrong despite high confidence (False Positives)
2. Cases where model was uncertain on correct objects (False Negatives)
3. Patterns in failure modes

## Appendix: Sample Images

See `crops/` directory for all evaluated crop images.

---

Report generated automatically from VLM Prompt Optimization System
"""
        return report

    def save_report(self, filename: str = "report.md") -> Path:
        """Save markdown report to file."""
        report_file = self.session_dir / filename
        with open(report_file, "w") as f:
            f.write(self.generate_markdown_report())
        return report_file

    def save_metrics_json(self, filename: str = "metrics.json") -> Path:
        """Save metrics as JSON."""
        metrics_file = self.session_dir / filename
        metrics = self.generate_metrics()
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics_file

    def get_summary(self) -> str:
        """Get summary as string for terminal output."""
        metrics = self.generate_metrics()

        if "error" in metrics:
            return metrics["error"]

        return f"""
VLM Experiment Results
═══════════════════════════════════════
Verified Samples: {metrics['verified_samples']}/{metrics['total_samples']}
Accuracy:         {metrics['accuracy_percent']}% ({metrics['correct_predictions']}/{metrics['verified_samples']})

Performance:
  Avg Time:       {metrics['processing_time_ms']['average']} ms
  Min/Max:        {metrics['processing_time_ms']['min']}/{metrics['processing_time_ms']['max']} ms

Errors:
  False Pos:      {metrics['false_positives']}
  False Neg:      {metrics['false_negatives']}
═══════════════════════════════════════
"""
