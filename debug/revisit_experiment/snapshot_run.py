#!/usr/bin/env python3
"""Archive one run's diagnostic data for the revisit/FPS experiment.

Phase 1 and Hydra each overwrite their save file on every run
(memory/tracker/phase1_tracker_state.json, memory/hydra/backend/dsg.json),
and phase1's per-frame timing CSV gets a fresh timestamped name every launch
but is never moved anywhere durable. Run this once after each run's shutdown
and before starting the next one, or later data overwrites earlier data and
the comparison this experiment exists to make becomes impossible.

    python3 debug/revisit_experiment/snapshot_run.py --run-label run1

Copies (never moves/deletes the originals):
  - memory/tracker/phase1_tracker_state.json  -> runs/<label>/phase1_tracker_state.json
  - memory/hydra/backend/dsg.json             -> runs/<label>/hydra_dsg.json
  - the newest rsg_object_detection_debug_*.xlsx under
    /home/student/rsg_ros2_ws/debug/ (it is CSV content despite the
    extension -- see phase1_timing_recorder.py)
                                               -> runs/<label>/phase1_timing.csv

Reads (does not copy, already written directly there by monitor_resources.py):
  - runs/<label>/resource_usage.csv

Then prints a summary, writes runs/<label>/summary.json, and appends one row
to runs/summary_all_runs.csv so all N runs can be plotted from a single file.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "debug" / "revisit_experiment" / "runs"
TRACKER_STATE = REPO / "memory" / "tracker" / "phase1_tracker_state.json"
HYDRA_DSG = REPO / "memory" / "hydra" / "backend" / "dsg.json"
PHASE1_TIMING_GLOB = "/home/student/rsg_ros2_ws/debug/rsg_object_detection_debug_*.xlsx"
CHECK_DUPLICATES = REPO / "debug" / "check_duplicate_objects.py"

SUMMARY_CSV_FIELDS = [
    "run_label", "track_count", "allocated_slot_count",
    "hydra_object_node_count", "hydra_duplicate_slot_count",
    "frame_count", "avg_fps", "recorded_span_sec", "frame_drop_count",
    "avg_frame_latency_ms", "median_frame_latency_ms", "max_frame_latency_ms",
    "avg_classifier_delay_ms", "avg_sam_inference_ms",
    "vlm_call_count", "vlm_success_count", "vlm_failed_count",
    "avg_gpu_pct", "peak_gpu_pct", "avg_cpu_pct", "peak_ram_used_mb",
]


def _latest_timing_csv() -> Path | None:
    import glob
    candidates = glob.glob(PHASE1_TIMING_GLOB)
    if not candidates:
        return None
    return Path(max(candidates, key=lambda p: Path(p).stat().st_mtime))


def _copy_tracker_and_hydra(out_dir: Path) -> dict:
    result = {}
    if TRACKER_STATE.exists():
        dest = out_dir / "phase1_tracker_state.json"
        shutil.copy2(TRACKER_STATE, dest)
        result["tracker_state_copied"] = str(dest)
    else:
        print(f"[snapshot_run] WARNING: {TRACKER_STATE} not found -- was session_persistence "
              "enabled and did the run shut down cleanly?", file=sys.stderr)

    if HYDRA_DSG.exists():
        dest = out_dir / "hydra_dsg.json"
        shutil.copy2(HYDRA_DSG, dest)
        result["hydra_dsg_copied"] = str(dest)
    else:
        print(f"[snapshot_run] WARNING: {HYDRA_DSG} not found.", file=sys.stderr)

    return result


def _copy_timing_csv(out_dir: Path) -> Path | None:
    src = _latest_timing_csv()
    if src is None:
        print(f"[snapshot_run] WARNING: no file matching {PHASE1_TIMING_GLOB} -- "
              "was phase1.performance.measure_timing on?", file=sys.stderr)
        return None
    dest = out_dir / "phase1_timing.csv"
    shutil.copy2(src, dest)
    return dest


def _read_tracker_summary(out_dir: Path) -> dict:
    path = out_dir / "phase1_tracker_state.json"
    if not path.exists():
        return {"track_count": None, "allocated_slot_count": None}
    data = json.loads(path.read_text())
    return {
        "track_count": data.get("track_count"),
        "allocated_slot_count": len(data.get("allocated_slot_ids", []) or []),
    }


def _read_hydra_summary(out_dir: Path) -> dict:
    path = out_dir / "hydra_dsg.json"
    if not path.exists():
        return {"hydra_object_node_count": None, "hydra_duplicate_slot_count": None}
    data = json.loads(path.read_text())
    objects = [
        n for n in data.get("nodes", [])
        if n.get("layer") == 2 and n.get("attributes", {}).get("type") == "ObjectNodeAttributes"
    ]
    slots = {}
    for n in objects:
        slot = n["attributes"].get("semantic_label")
        slots[slot] = slots.get(slot, 0) + 1
    duplicate_slots = sum(1 for count in slots.values() if count > 1)

    # Also run the fuser-collapse-predicate replay for a fuller picture in
    # the printed summary (not folded into the CSV row -- see its own output).
    if CHECK_DUPLICATES.exists():
        try:
            proc = subprocess.run(
                [sys.executable, str(CHECK_DUPLICATES), str(path)],
                capture_output=True, text=True, timeout=30,
            )
            print("\n--- debug/check_duplicate_objects.py ---")
            print(proc.stdout.strip())
            if proc.returncode != 0:
                print(proc.stderr.strip(), file=sys.stderr)
        except Exception as exc:  # pragma: no cover -- diagnostic best-effort
            print(f"[snapshot_run] check_duplicate_objects.py failed: {exc}", file=sys.stderr)

    return {
        "hydra_object_node_count": len(objects),
        "hydra_duplicate_slot_count": duplicate_slots,
    }


def _read_timing_summary(out_dir: Path) -> dict:
    path = out_dir / "phase1_timing.csv"
    if not path.exists():
        return {
            "frame_count": None, "avg_fps": None, "recorded_span_sec": None,
            "avg_frame_latency_ms": None, "median_frame_latency_ms": None, "max_frame_latency_ms": None,
            "avg_classifier_delay_ms": None, "avg_sam_inference_ms": None,
            "vlm_call_count": None, "vlm_success_count": None, "vlm_failed_count": None,
            "frame_drop_count": None,
        }

    # The combined CSV interleaves rows from five different call sites,
    # distinguished by the existing `event` column (not something added for
    # this experiment):
    #   frame_trace        -- one per processed frame (_publish_hydra_from_result)
    #   completed          -- one per finished VLM call (reason = vlm_done/vlm_failed)
    #   dropped_oldest/dropped_newest/failed -- frame-lifecycle drops
    #   enqueued / dequeued -- VLM queue admission, not a completed call
    # wall_clock_unix_sec was added on the frame_trace and completed rows
    # specifically for this experiment; older CSVs from before that change
    # simply won't have it, and span/fps come back None for those.
    frame_count = 0
    frame_times = []
    frame_latencies_ms = []      # total_delay_ms: input (frame received) -> published to Hydra
    classifier_delays_ms = []    # subset of that spent in SAM + association ("classifier")
    sam_inference_ms_vals = []   # subset of that spent purely in SAM inference
    vlm_count = 0
    vlm_success = 0
    vlm_failed = 0
    drop_count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event = row.get("event", "")

            if event == "frame_trace":
                frame_count += 1
                # Separate from the count itself: a CSV from before this
                # experiment added wall_clock_unix_sec would otherwise report
                # frame_count correctly but span/fps as None, not zero frames.
                wc = row.get("wall_clock_unix_sec")
                if wc:
                    try:
                        frame_times.append(float(wc))
                    except ValueError:
                        pass
                # total_delay_ms is computed in _publish_hydra_from_result as
                # hydra_publish_complete - cached.received_monotonic -- exactly
                # "time from this frame entering phase1 to phase1 finishing
                # publishing it to Hydra." classifier_delay_ms/sam_inference_ms
                # are the two biggest contributors and explain *why* latency
                # moves, not just that it did.
                for target, key in (
                    (frame_latencies_ms, "total_delay_ms"),
                    (classifier_delays_ms, "classifier_delay_ms"),
                    (sam_inference_ms_vals, "sam_inference_ms"),
                ):
                    value = row.get(key)
                    if value:
                        try:
                            target.append(float(value))
                        except ValueError:
                            pass
            elif event == "completed":
                vlm_count += 1
                reason = str(row.get("reason", "")).lower()
                if "fail" in reason:
                    vlm_failed += 1
                else:
                    vlm_success += 1
            elif event in ("dropped_oldest", "dropped_newest", "failed"):
                drop_count += 1

    if len(frame_times) >= 2:
        span = max(frame_times) - min(frame_times)
        # Use the count of timestamped frames for the rate, not the overall
        # frame_count -- they only differ for a CSV predating
        # wall_clock_unix_sec, where they would otherwise silently understate
        # the row count's contribution to an otherwise-correct span.
        avg_fps = len(frame_times) / span if span > 0 else None
    else:
        span = None
        avg_fps = None

    def _stats(values):
        if not values:
            return None, None, None
        ordered = sorted(values)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        return round(sum(values) / len(values), 2), round(median, 2), round(max(values), 2)

    avg_latency_ms, median_latency_ms, max_latency_ms = _stats(frame_latencies_ms)
    avg_classifier_ms, _, _ = _stats(classifier_delays_ms)
    avg_sam_inference_ms, _, _ = _stats(sam_inference_ms_vals)

    return {
        "frame_count": frame_count,
        "avg_fps": round(avg_fps, 4) if avg_fps is not None else None,
        "recorded_span_sec": round(span, 2) if span is not None else None,
        # Input (frame received by phase1) -> output (published to Hydra).
        "avg_frame_latency_ms": avg_latency_ms,
        "median_frame_latency_ms": median_latency_ms,
        "max_frame_latency_ms": max_latency_ms,
        # Breakdown: how much of that latency is SAM/association vs. pure
        # SAM inference -- explains a latency change, doesn't just report one.
        "avg_classifier_delay_ms": avg_classifier_ms,
        "avg_sam_inference_ms": avg_sam_inference_ms,
        "vlm_call_count": vlm_count,
        "vlm_success_count": vlm_success,
        "vlm_failed_count": vlm_failed,
        "frame_drop_count": drop_count,
    }


def _read_resource_summary(out_dir: Path) -> dict:
    path = out_dir / "resource_usage.csv"
    if not path.exists():
        print(f"[snapshot_run] WARNING: {path} not found -- was monitor_resources.py "
              "run alongside this run?", file=sys.stderr)
        return {"avg_gpu_pct": None, "peak_gpu_pct": None, "avg_cpu_pct": None, "peak_ram_used_mb": None}

    gpu_vals, cpu_vals, ram_vals = [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for target, key in ((gpu_vals, "gpu_pct"), (cpu_vals, "cpu_avg_pct"), (ram_vals, "ram_used_mb")):
                value = row.get(key)
                if value:
                    try:
                        target.append(float(value))
                    except ValueError:
                        pass

    return {
        "avg_gpu_pct": round(sum(gpu_vals) / len(gpu_vals), 2) if gpu_vals else None,
        "peak_gpu_pct": max(gpu_vals) if gpu_vals else None,
        "avg_cpu_pct": round(sum(cpu_vals) / len(cpu_vals), 2) if cpu_vals else None,
        "peak_ram_used_mb": max(ram_vals) if ram_vals else None,
    }


def _append_summary_csv(row: dict) -> Path:
    csv_path = RUNS_DIR / "summary_all_runs.csv"
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_CSV_FIELDS})
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-label", required=True, help="e.g. run1, run2 -- matches monitor_resources.py")
    args = parser.parse_args()

    out_dir = RUNS_DIR / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_tracker_and_hydra(out_dir)
    _copy_timing_csv(out_dir)

    summary = {"run_label": args.run_label}
    summary.update(_read_tracker_summary(out_dir))
    summary.update(_read_hydra_summary(out_dir))
    summary.update(_read_timing_summary(out_dir))
    summary.update(_read_resource_summary(out_dir))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False))
    csv_path = _append_summary_csv(summary)

    print(f"\n=== {args.run_label} summary ===")
    for key in SUMMARY_CSV_FIELDS[1:]:
        print(f"  {key:28s} {summary.get(key)}")
    print(f"\nWrote {out_dir / 'summary.json'}")
    print(f"Appended to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
