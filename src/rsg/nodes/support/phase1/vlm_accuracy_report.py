"""Generate accuracy report from manual verification CSV."""

import csv
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


class VLMAccuracyReport:
    """Analyze verification results and generate accuracy metrics."""

    def __init__(self, results_file: Path):
        """Initialize with results CSV file.

        Args:
            results_file: Path to vlm_results.csv
        """
        self.results_file = Path(results_file)
        self.session_dir = self.results_file.parent

    def load_results(self) -> List[Dict[str, Any]]:
        """Load and parse results CSV."""
        results = []
        try:
            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("manual_is_correct"):  # Only verified rows
                        results.append({
                            "object_id": row.get("object_id"),
                            "label": row.get("label"),
                            "label_confidence": float(row.get("label_confidence", 0)),
                            "mobility_class": row.get("mobility_class"),
                            "mobility_confidence": float(row.get("mobility_confidence", 0)),
                            "vlm_processing_time_ms": float(row.get("vlm_processing_time_ms", 0)),
                            "manual_label": row.get("manual_label"),
                            "manual_is_correct": row.get("manual_is_correct").lower() == "true",
                            "manual_notes": row.get("manual_notes", ""),
                        })
        except FileNotFoundError:
            pass
        return results

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate accuracy and performance metrics."""
        results = self.load_results()

        if not results:
            return {"error": "No verified results found"}

        total = len(results)
        correct = sum(1 for r in results if r["manual_is_correct"])
        accuracy = (correct / total * 100) if total > 0 else 0

        # Confidence analysis
        high_conf_correct = sum(1 for r in results if r["label_confidence"] > 0.7 and r["manual_is_correct"])
        high_conf_total = sum(1 for r in results if r["label_confidence"] > 0.7)
        high_conf_acc = (high_conf_correct / high_conf_total * 100) if high_conf_total > 0 else 0

        low_conf_correct = sum(1 for r in results if r["label_confidence"] <= 0.7 and r["manual_is_correct"])
        low_conf_total = sum(1 for r in results if r["label_confidence"] <= 0.7)
        low_conf_acc = (low_conf_correct / low_conf_total * 100) if low_conf_total > 0 else 0

        # Mobility classification
        mobility_correct = sum(1 for r in results if r.get("mobility_class") and r["manual_is_correct"])

        # Processing time
        proc_times = [r["vlm_processing_time_ms"] for r in results]
        avg_time = sum(proc_times) / len(proc_times) if proc_times else 0

        return {
            "total_verified": total,
            "correct": correct,
            "accuracy_percent": round(accuracy, 2),
            "high_confidence_accuracy": round(high_conf_acc, 2),
            "high_confidence_samples": high_conf_total,
            "low_confidence_accuracy": round(low_conf_acc, 2),
            "low_confidence_samples": low_conf_total,
            "mobility_correct": mobility_correct,
            "avg_processing_time_ms": round(avg_time, 2),
            "results": results,
        }

    def print_report(self) -> None:
        """Print human-readable report to console."""
        metrics = self.calculate_metrics()

        if "error" in metrics:
            print(f"❌ {metrics['error']}")
            return

        print(f"""
╔════════════════════════════════════════════════════════════╗
║         VLM TESTING RESULTS SUMMARY                        ║
╚════════════════════════════════════════════════════════════╝

📊 OVERALL ACCURACY
   Verified Samples:  {metrics['total_verified']}
   Correct:           {metrics['correct']}/{metrics['total_verified']}
   Accuracy:          {metrics['accuracy_percent']}%

🎯 CONFIDENCE ANALYSIS
   High Confidence (>70%)
     Samples:         {metrics['high_confidence_samples']}
     Accuracy:        {metrics['high_confidence_accuracy']}%

   Low Confidence (≤70%)
     Samples:         {metrics['low_confidence_samples']}
     Accuracy:        {metrics['low_confidence_accuracy']}%

⚙️  PERFORMANCE
   Avg Processing Time: {metrics['avg_processing_time_ms']} ms

═══════════════════════════════════════════════════════════════

Errors & Failures:
""")
        results = metrics["results"]
        errors = [r for r in results if not r["manual_is_correct"]]
        if errors:
            for err in errors[:10]:  # Show first 10
                print(f"  ❌ {err['object_id']}: VLM={err['label']} (conf={err['label_confidence']:.2f}), "
                      f"Actual={err['manual_label']}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("  ✅ No errors!")

        print(f"\n📁 Session: {self.session_dir}")
        print(f"📄 Results: {self.results_file}\n")

    def save_markdown_report(self, filename: str = "accuracy_report.md") -> Path:
        """Save report as markdown."""
        metrics = self.calculate_metrics()

        if "error" in metrics:
            content = f"# VLM Test Report\n\n{metrics['error']}\n"
        else:
            content = f"""# VLM Test Results

## Summary
- **Total Verified:** {metrics['total_verified']} samples
- **Correct:** {metrics['correct']}
- **Overall Accuracy:** {metrics['accuracy_percent']}%

## Confidence Breakdown
- **High Confidence (>70%):** {metrics['high_confidence_accuracy']}% accuracy ({metrics['high_confidence_samples']} samples)
- **Low Confidence (≤70%):** {metrics['low_confidence_accuracy']}% accuracy ({metrics['low_confidence_samples']} samples)

## Performance
- **Avg Processing Time:** {metrics['avg_processing_time_ms']} ms

## Error Cases
"""
            results = metrics["results"]
            errors = [r for r in results if not r["manual_is_correct"]]
            if errors:
                content += f"\nFound {len(errors)} incorrect predictions:\n\n"
                for err in errors:
                    content += f"- **{err['object_id']}:** VLM predicted `{err['label']}` "
                    content += f"(confidence {err['label_confidence']:.2f}), "
                    content += f"actual was `{err['manual_label']}`\n"
                    if err["manual_notes"]:
                        content += f"  - Notes: {err['manual_notes']}\n"
            else:
                content += "\nNo errors! All predictions correct.\n"

        report_file = self.session_dir / filename
        with open(report_file, "w") as f:
            f.write(content)

        return report_file
