#!/usr/bin/env python3
"""Summarise one Phase 1 Part 5 CSV (contention root-cause investigation).

Forked from optimisation_part4/analyse_timing.py at the start of Part 5.
Covers both of this part's steps -- Step 1 (CPU core affinity) and Step 2
(jetson_clocks/power mode) -- since neither adds a per-frame-varying metric:
affinity is a static per-thread setting applied once at thread start, and
jetson_clocks is a system-wide setting applied before launch. Either step's
effect shows up in the existing sam_inference_ms / geometry_metadata_ms /
frame_assignment_ms / sam_output_queue_wait_ms columns, not a new one. Same
throughput-first framing as Part 4, since this part is still on the
throughput axis (removing contention that was inflating per-frame cost),
not the per-stage-cost axis Parts 1-3 used.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


LEAF_STAGES = (
    "callback_enqueue_delay_ms", "frame_queue_wait_ms",
    "sam_output_queue_wait_ms",
    "image_conversion_delay_ms", "sam_prepare_ms", "sam_inference_ms",
    "sam_restore_ms", "sam_other_ms", "geometry_metadata_ms",
    "geometry_mask_extract_ms", "geometry_depth_gather_ms",
    "geometry_projection_ms", "geometry_stats_ms",
    "frame_assignment_ms", "assignment_candidate_search_ms",
    "assignment_row_init_ms", "assignment_3d_geometry_ms",
    "assignment_centroid_iou_ms", "assignment_scoring_ms",
    "assignment_a2_redundancy_ms", "assignment_a3_nested_ms",
    "assignment_hungarian_ms", "assignment_lock_wait_ms",
    "track_association_ms", "association_lock_wait_ms", "crop_update_ms",
    "run_rap_other_ms", "active_segments_publish_ms", "semantic_dispatch_ms",
    "quality_deferred_release_ms", "label_map_delay_ms", "metadata_delay_ms",
    "result_message_build_delay_ms", "classifier_other_ms",
    "classifier_debug_record_delay_ms", "hydra_depth_filter_ms",
    "hydra_metadata_build_ms", "hydra_build_other_ms",
    "hydra_publish_delay_ms", "unknown_publish_delay_ms",
    "coordinator_other_ms", "pipeline_wait_ms",
)
AGGREGATES = (
    "sent_to_classifier_delay_ms", "sam_delay_ms", "rap_delay_ms",
    "hydra_build_delay_ms", "classifier_delay_ms", "coordinator_delay_ms",
    "total_delay_ms",
)
EXCLUDED_FROM_CANDIDATE = {
    "sam_prepare_ms", "sam_inference_ms", "sam_restore_ms", "sam_other_ms",
    "geometry_mask_extract_ms", "geometry_depth_gather_ms",
    "geometry_projection_ms", "geometry_stats_ms",
    "assignment_candidate_search_ms", "assignment_row_init_ms",
    "assignment_3d_geometry_ms", "assignment_centroid_iou_ms",
    "assignment_scoring_ms",
    "assignment_a2_redundancy_ms", "assignment_a3_nested_ms",
    "assignment_hungarian_ms", "assignment_lock_wait_ms",
    "association_lock_wait_ms",
    "semantic_dispatch_ms", "quality_deferred_release_ms",
    "classifier_debug_record_delay_ms", "pipeline_wait_ms",
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def latest_counts(records: list[dict]) -> dict:
    """Return the last lifecycle row's running received/processed/dropped/hydra_published counts."""
    counted = [row for row in records if row.get("received_count") not in (None, "")]
    if not counted:
        return {}
    last = counted[-1]
    return {
        "received": int(last["received_count"]),
        "processed": int(last["processed_count"]),
        "failed": int(last["failed_count"]),
        "dropped": int(last["dropped_count"]),
        "hydra_published": int(last["hydra_published_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_csv", type=Path)
    args = parser.parse_args()
    with args.trace_csv.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    traces = [row for row in records if row.get("event") == "frame_trace"]
    drop_rows = [row for row in records if str(row.get("event", "")).startswith("dropped")]
    drops = {row.get("sequence") for row in drop_rows if row.get("sequence")}
    failures = [row for row in records if row.get("event") == "failed"]
    if not traces:
        raise SystemExit("No frame_trace rows found. Stop the launch cleanly with Ctrl+C.")

    counts = latest_counts(records)

    summary = []
    for stage in (*LEAF_STAGES, *AGGREGATES):
        values = [float(row[stage]) for row in traces if row.get(stage) not in {None, ""}]
        if values:
            summary.append({
                "stage": stage,
                "kind": "leaf" if stage in LEAF_STAGES else "aggregate",
                "candidate_eligible": stage in LEAF_STAGES and stage not in EXCLUDED_FROM_CANDIDATE,
                "samples": len(values),
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
                "max_ms": max(values),
            })
    summary.sort(key=lambda row: float(row["mean_ms"]), reverse=True)
    candidate = max(
        (row for row in summary if row["candidate_eligible"]),
        key=lambda row: float(row["mean_ms"]),
    )
    output_dir = args.trace_csv.parent
    with (output_dir / "stage_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    with (output_dir / "stage_summary.md").open("w", encoding="utf-8") as stream:
        stream.write("# Phase 1 optimisation Part 5 timing summary\n\n")
        stream.write(
            f"Complete frame traces: {len(traces)}; drops: {len(drops)}; "
            f"failures: {len(failures)}.\n\n"
        )
        stream.write("Asynchronous RAP/VLM inference and retrieval are outside this analysis.\n\n")

        stream.write("## Throughput (the primary question for this part)\n\n")
        stream.write(
            "Part 5 removes cross-thread CPU contention (via core affinity) from "
            "the concurrency model Part 4 introduced, so the number that matters "
            "here is still how many frames got through the 180 s window -- and "
            "whether sam_inference_ms/geometry_metadata_ms/frame_assignment_ms "
            "fall back toward their pre-Part-4 serial-mode values.\n\n"
        )
        if counts:
            ratio = 100.0 * counts["processed"] / counts["received"] if counts["received"] else 0.0
            stream.write(
                "| Received | Processed | Dropped | Failed | Hydra published | Processing ratio |\n"
                "|---:|---:|---:|---:|---:|---:|\n"
                f"| {counts['received']} | {counts['processed']} | {counts['dropped']} | "
                f"{counts['failed']} | {counts['hydra_published']} | {ratio:.2f}% |\n\n"
            )
        else:
            stream.write(
                "No lifecycle rows with running counts were found in this trace "
                "(only present on `dropped_oldest`/`dropped_newest`/`failed` events).\n\n"
            )
        sam_wait = next((row for row in summary if row["stage"] == "sam_output_queue_wait_ms"), None)
        if sam_wait:
            stream.write(
                "`sam_output_queue_wait_ms` -- time a completed SAM stage waits for the "
                "tracking/publish thread to become free. Small relative to "
                "`sam_inference_ms` means the two stages are overlapping well; large "
                "means tracking/publish has become the new bottleneck.\n\n"
                f"Mean {sam_wait['mean_ms']:.3f} ms, median {sam_wait['median_ms']:.3f} ms, "
                f"p95 {sam_wait['p95_ms']:.3f} ms, max {sam_wait['max_ms']:.3f} ms "
                f"(n={sam_wait['samples']}).\n\n"
            )

        stream.write("## Per-stage breakdown (same framing as Parts 1-3, for continuity)\n\n")
        stream.write(
            f"Largest eligible synchronous leaf stage: **{candidate['stage']}** "
            f"(mean {candidate['mean_ms']:.3f} ms, p95 {candidate['p95_ms']:.3f} ms). "
            "This framing answers \"which stage costs the most,\" which is not the "
            "question Part 5 is testing -- see the throughput section above.\n\n"
        )
        stream.write("| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |\n")
        stream.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            stream.write(
                f"| {row['stage']} | {row['kind']} | {row['candidate_eligible']} | "
                f"{row['samples']} | {row['mean_ms']:.3f} | {row['median_ms']:.3f} | "
                f"{row['p95_ms']:.3f} | {row['max_ms']:.3f} |\n"
            )
    print(output_dir / "stage_summary.md")


if __name__ == "__main__":
    main()
