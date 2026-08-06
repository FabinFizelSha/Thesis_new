#!/usr/bin/env python3
"""Summarise one Phase 1 Part 3 (frame assignment) CSV without ranking async RAP/VLM work.

Forked from optimisation_part2/analyse_timing.py at the start of Part 3.
Step 1 added four assignment_* sub-step columns:
assignment_candidate_search_ms, assignment_a2_redundancy_ms,
assignment_a3_nested_ms, assignment_hungarian_ms. Step 1 follow-up ("finer
profiling") added a second, nested level splitting
assignment_candidate_search_ms itself into assignment_row_init_ms,
assignment_3d_geometry_ms, assignment_centroid_iou_ms,
assignment_scoring_ms (accumulated inside _find_match across every
candidate track evaluated). Path B added two lock-wait columns —
assignment_lock_wait_ms (time prepare_frame_assignments() blocks acquiring
PersistentObjectTracker._lock, measured before any of the above sub-steps
run) and association_lock_wait_ms (same, for associate()) — to test
whether async RAP/VLM-thread lock contention explains the severe scattered
tail spikes observed in earlier sessions. All ten summarise like any other
leaf stage but are excluded from "largest bottleneck" candidate selection
because they are sub-components of frame_assignment_ms/track_association_ms,
not independent stages. Two additional non-timing diagnostic columns
(assignment_candidate_count_total, assignment_candidate_count_max) are
written to the raw CSV by the instrumentation but are not stage-summarised
here since they are counts, not durations.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


LEAF_STAGES = (
    "callback_enqueue_delay_ms", "frame_queue_wait_ms",
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
DIAGNOSTIC_COUNT_COLUMNS = (
    "assignment_candidate_count_total", "assignment_candidate_count_max",
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


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
    # Part 3 hypothesis check: does per-observation candidate-track count (and
    # frame_assignment_ms itself) grow across the run as the track registry
    # fills up? Split traces into quartiles by arrival order and report each.
    quartile_rows = []
    n = len(traces)
    if n >= 4:
        quartile_size = n // 4
        for q in range(4):
            start = q * quartile_size
            end = (q + 1) * quartile_size if q < 3 else n
            chunk = traces[start:end]
            row = {"quartile": q + 1, "frames": len(chunk)}
            for stage in ("frame_assignment_ms", *DIAGNOSTIC_COUNT_COLUMNS):
                values = [float(r[stage]) for r in chunk if r.get(stage) not in {None, ""}]
                row[stage] = statistics.fmean(values) if values else 0.0
            quartile_rows.append(row)

    with (output_dir / "stage_summary.md").open("w", encoding="utf-8") as stream:
        stream.write("# Phase 1 optimisation Part 3 timing summary\n\n")
        stream.write(
            f"Complete frame traces: {len(traces)}; drops: {len(drops)}; "
            f"failures: {len(failures)}.\n\n"
        )
        stream.write("Asynchronous RAP/VLM inference and retrieval are outside this analysis.\n\n")
        stream.write(
            f"Largest eligible synchronous leaf stage: **{candidate['stage']}** "
            f"(mean {candidate['mean_ms']:.3f} ms, p95 {candidate['p95_ms']:.3f} ms).\n\n"
        )
        stream.write("| Stage | Kind | Eligible | Samples | Mean ms | Median ms | P95 ms | Max ms |\n")
        stream.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            stream.write(
                f"| {row['stage']} | {row['kind']} | {row['candidate_eligible']} | "
                f"{row['samples']} | {row['mean_ms']:.3f} | {row['median_ms']:.3f} | "
                f"{row['p95_ms']:.3f} | {row['max_ms']:.3f} |\n"
            )
        if quartile_rows:
            stream.write(
                "\n## frame_assignment_ms and candidate-count growth across the run\n\n"
            )
            stream.write(
                "Tests the hypothesis that per-frame assignment cost grows as the "
                "persistent-track registry fills the explored scene, independent of "
                "whole-run mean.\n\n"
            )
            stream.write(
                "| Quartile | Frames | frame_assignment_ms mean | candidate_count_total mean | candidate_count_max mean |\n"
            )
            stream.write("|---:|---:|---:|---:|---:|\n")
            for row in quartile_rows:
                stream.write(
                    f"| {row['quartile']} | {row['frames']} | {row['frame_assignment_ms']:.3f} | "
                    f"{row.get('assignment_candidate_count_total', 0.0):.2f} | "
                    f"{row.get('assignment_candidate_count_max', 0.0):.2f} |\n"
                )
    print(output_dir / "stage_summary.md")


if __name__ == "__main__":
    main()
