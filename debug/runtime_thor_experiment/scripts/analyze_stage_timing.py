#!/usr/bin/env python3
"""Break down where a frame's time actually goes inside Phase 1, per pipeline stage.

No new diagnostics needed: every `frame_trace` row in phase1_timing.csv
already carries ~45 named, mutually-exclusive sub-stage timings covering the
whole path from "frame received" to "published to Hydra"
(_publish_hydra_from_result in phase1.py, unconditional on every frame
whenever phase1.performance.measure_timing / write_timing_csv are on --
already the case in the current setup, unchanged by this script). This just
aggregates and ranks what is already being recorded.

Usage:

    python3 debug/revisit_experiment/analyze_stage_timing.py <phase1_timing.csv> [more.csv ...]
    python3 debug/revisit_experiment/analyze_stage_timing.py --session session_20260909_234618
    python3 debug/revisit_experiment/analyze_stage_timing.py --all-sessions

Writes a ranked CSV (default: stage_timing_breakdown.csv next to the first
input, or debug/revisit_experiment/sessions/stage_timing_breakdown.csv for
--all-sessions) with one row per named stage: mean/median/max milliseconds,
percentage of total_delay_ms, and its parent stage in the timing hierarchy
below -- sorted by mean_ms descending, so the biggest contributor is the
first data row.

The hierarchy (from phase1.py's own subtraction structure, not invented
here -- see _publish_hydra_from_result and run_tracking_publish_stage):

    total_delay_ms                                    (frame received -> published to Hydra)
    +-- sent_to_classifier_delay_ms                    (queued before classification starts)
    +-- sam_output_queue_wait_ms
    +-- classifier_delay_ms
    |   +-- image_conversion_delay_ms
    |   +-- sam_delay_ms
    |   |   +-- sam_prepare_ms
    |   |   +-- sam_inference_ms
    |   |   +-- sam_restore_ms
    |   |   +-- sam_other_ms                           (residual within sam_delay_ms)
    |   +-- rap_delay_ms
    |   |   +-- geometry_metadata_ms
    |   |   |   +-- geometry_mask_extract_ms
    |   |   |   +-- geometry_depth_gather_ms
    |   |   |   +-- geometry_projection_ms
    |   |   |   +-- geometry_stats_ms
    |   |   +-- frame_assignment_ms
    |   |   |   +-- assignment_candidate_search_ms
    |   |   |   +-- assignment_row_init_ms
    |   |   |   +-- assignment_3d_geometry_ms
    |   |   |   +-- assignment_centroid_iou_ms
    |   |   |   +-- assignment_scoring_ms
    |   |   |   +-- assignment_a2_redundancy_ms
    |   |   |   +-- assignment_a3_nested_ms
    |   |   |   +-- assignment_hungarian_ms
    |   |   +-- track_association_ms
    |   |   +-- crop_update_ms
    |   |   +-- active_segments_publish_ms
    |   |   +-- semantic_dispatch_ms
    |   |   +-- quality_deferred_release_ms
    |   |   +-- run_rap_other_ms                       (residual within the run_rap_and_metadata call)
    |   +-- label_map_delay_ms
    |   +-- metadata_delay_ms
    |   +-- result_message_build_delay_ms
    |   +-- classifier_other_ms                         (residual within classifier_delay_ms)
    +-- classifier_debug_record_delay_ms
    +-- coordinator_delay_ms
    |   +-- hydra_build_delay_ms
    |   |   +-- hydra_depth_filter_ms
    |   |   +-- hydra_metadata_build_ms
    |   |   +-- hydra_build_other_ms                    (residual within hydra_build_delay_ms)
    |   +-- hydra_publish_delay_ms
    |   +-- unknown_publish_delay_ms
    |   +-- coordinator_other_ms                         (residual within coordinator_delay_ms)
    +-- pipeline_wait_ms                                 (residual within total_delay_ms)
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSIONS_DIR = REPO / "debug" / "revisit_experiment" / "sessions"

# (stage, parent) -- parent=None for total_delay_ms, the root.
HIERARCHY: list[tuple[str, str | None]] = [
    ("total_delay_ms", None),
    ("sent_to_classifier_delay_ms", "total_delay_ms"),
    ("sam_output_queue_wait_ms", "total_delay_ms"),
    ("classifier_delay_ms", "total_delay_ms"),
    ("image_conversion_delay_ms", "classifier_delay_ms"),
    ("sam_delay_ms", "classifier_delay_ms"),
    ("sam_prepare_ms", "sam_delay_ms"),
    ("sam_inference_ms", "sam_delay_ms"),
    ("sam_restore_ms", "sam_delay_ms"),
    ("sam_other_ms", "sam_delay_ms"),
    ("rap_delay_ms", "classifier_delay_ms"),
    ("geometry_metadata_ms", "rap_delay_ms"),
    ("geometry_mask_extract_ms", "geometry_metadata_ms"),
    ("geometry_depth_gather_ms", "geometry_metadata_ms"),
    ("geometry_projection_ms", "geometry_metadata_ms"),
    ("geometry_stats_ms", "geometry_metadata_ms"),
    ("frame_assignment_ms", "rap_delay_ms"),
    ("assignment_candidate_search_ms", "frame_assignment_ms"),
    ("assignment_row_init_ms", "frame_assignment_ms"),
    ("assignment_3d_geometry_ms", "frame_assignment_ms"),
    ("assignment_centroid_iou_ms", "frame_assignment_ms"),
    ("assignment_scoring_ms", "frame_assignment_ms"),
    ("assignment_a2_redundancy_ms", "frame_assignment_ms"),
    ("assignment_a3_nested_ms", "frame_assignment_ms"),
    ("assignment_hungarian_ms", "frame_assignment_ms"),
    ("track_association_ms", "rap_delay_ms"),
    ("crop_update_ms", "rap_delay_ms"),
    ("active_segments_publish_ms", "rap_delay_ms"),
    ("semantic_dispatch_ms", "rap_delay_ms"),
    ("quality_deferred_release_ms", "rap_delay_ms"),
    ("run_rap_other_ms", "rap_delay_ms"),
    ("label_map_delay_ms", "classifier_delay_ms"),
    ("metadata_delay_ms", "classifier_delay_ms"),
    ("result_message_build_delay_ms", "classifier_delay_ms"),
    ("classifier_other_ms", "classifier_delay_ms"),
    ("classifier_debug_record_delay_ms", "total_delay_ms"),
    ("coordinator_delay_ms", "total_delay_ms"),
    ("hydra_build_delay_ms", "coordinator_delay_ms"),
    ("hydra_depth_filter_ms", "hydra_build_delay_ms"),
    ("hydra_metadata_build_ms", "hydra_build_delay_ms"),
    ("hydra_build_other_ms", "hydra_build_delay_ms"),
    ("hydra_publish_delay_ms", "coordinator_delay_ms"),
    ("unknown_publish_delay_ms", "coordinator_delay_ms"),
    ("coordinator_other_ms", "coordinator_delay_ms"),
    ("pipeline_wait_ms", "total_delay_ms"),
]
STAGE_NAMES = [name for name, _ in HIERARCHY]
PARENT_OF = dict(HIERARCHY)


def _load_frame_trace_rows(csv_paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("event") == "frame_trace":
                    rows.append(row)
    return rows


def _stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return sum(values) / len(values), median, max(values)


def analyze(rows: list[dict]) -> list[dict]:
    """One dict per named stage: mean/median/max ms, % of total, % of parent."""
    per_stage_values: dict[str, list[float]] = {name: [] for name in STAGE_NAMES}
    for row in rows:
        for name in STAGE_NAMES:
            raw = row.get(name)
            if raw:
                try:
                    per_stage_values[name].append(float(raw))
                except ValueError:
                    pass

    mean_total, _, _ = _stats(per_stage_values["total_delay_ms"])
    mean_by_stage = {name: _stats(vals)[0] for name, vals in per_stage_values.items()}

    results = []
    for name in STAGE_NAMES:
        mean_ms, median_ms, max_ms = _stats(per_stage_values[name])
        parent = PARENT_OF[name]
        parent_mean = mean_by_stage.get(parent, 0.0) if parent else mean_total
        results.append({
            "stage": name,
            "parent_stage": parent or "",
            "n_frames": len(per_stage_values[name]),
            "mean_ms": round(mean_ms, 3),
            "median_ms": round(median_ms, 3),
            "max_ms": round(max_ms, 3),
            "pct_of_total_delay": round(100.0 * mean_ms / mean_total, 2) if mean_total > 0 else 0.0,
            "pct_of_parent": round(100.0 * mean_ms / parent_mean, 2) if parent_mean > 0 else 0.0,
        })

    # Root first, then sorted by mean_ms descending -- the biggest single
    # contributor to frame latency is the first data row after the header.
    root = [r for r in results if r["stage"] == "total_delay_ms"]
    rest = sorted((r for r in results if r["stage"] != "total_delay_ms"),
                 key=lambda r: r["mean_ms"], reverse=True)
    return root + rest


def write_csv(results: list[dict], out_path: Path) -> None:
    fieldnames = ["stage", "parent_stage", "n_frames", "mean_ms", "median_ms",
                 "max_ms", "pct_of_total_delay", "pct_of_parent"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="*", help="one or more phase1_timing.csv files")
    parser.add_argument("--session", help="shorthand for sessions/<name>/phase1_timing.csv")
    parser.add_argument("--all-sessions", action="store_true",
                        help="aggregate every sessions/*/phase1_timing.csv together")
    parser.add_argument("--out", help="output CSV path (default: alongside the input)")
    args = parser.parse_args()

    paths: list[Path] = [Path(p) for p in args.csv_files]
    if args.session:
        paths.append(SESSIONS_DIR / args.session / "phase1_timing.csv")
    if args.all_sessions:
        paths.extend(Path(p) for p in sorted(glob.glob(str(SESSIONS_DIR / "session_*" / "phase1_timing.csv"))))

    paths = [p for p in paths if p.exists()]
    if not paths:
        print("No phase1_timing.csv files found -- pass a path, --session <name>, or --all-sessions.",
              file=sys.stderr)
        return 1

    rows = _load_frame_trace_rows(paths)
    if not rows:
        print(f"No event==frame_trace rows found across {len(paths)} file(s).", file=sys.stderr)
        return 1

    results = analyze(rows)

    if args.out:
        out_path = Path(args.out)
    elif args.all_sessions:
        out_path = SESSIONS_DIR / "stage_timing_breakdown.csv"
    else:
        out_path = paths[0].parent / "stage_timing_breakdown.csv"
    write_csv(results, out_path)

    total_mean = results[0]["mean_ms"]
    print(f"{len(rows)} frame(s) across {len(paths)} file(s). "
          f"mean total_delay_ms = {total_mean:.1f}\n")
    print(f"{'stage':32s} {'parent':22s} {'mean_ms':>9s} {'%total':>7s} {'%parent':>8s}")
    for r in results[:15]:
        print(f"{r['stage']:32s} {r['parent_stage']:22s} {r['mean_ms']:9.1f} "
              f"{r['pct_of_total_delay']:6.1f}% {r['pct_of_parent']:7.1f}%")
    print(f"\n({len(results) - 15} more rows in {out_path})" if len(results) > 15 else "")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
