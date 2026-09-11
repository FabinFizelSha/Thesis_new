#!/usr/bin/env python3
"""Re-compute a session's metrics over the same *input frame budget* as a
reference session, instead of the same wall-clock window.

Every session plays for the same fixed 300s wall-clock window (see
EXPERIMENT_PLAN.md Section 4), but at different --rate values. A higher rate
covers more of the bag's own timeline in that same 300s, which means a
higher-rate session's later portion is genuinely unseen scene content the
earlier session never reached -- new physical locations with new objects
that need fresh VLM dispatch for reasons that have nothing to do with the
accumulated-memory effect this experiment is trying to isolate. Comparing
two sessions' full 300s windows head-on conflates "more memory" with "more
never-before-seen bag content," in the direction that would erase the effect
being measured (a higher-rate session's extra reach adds VLM load that a
same-frame-budget comparison would not have).

The fix: hold the *input frame budget* constant instead of the wall-clock
window. Given a reference session's total input frame count (frame_count +
frame_drop_count -- every frame that arrived at phase 1, whether processed or
dropped), find how much wall-clock time the session being normalized needed
to reach that same input frame count, using its own empirically observed
input arrival rate (frame_count + frame_drop_count) / recorded_span_sec --
measured directly, not assumed from --rate x native_rate, so it absorbs any
real scheduling/publish jitter instead of assuming a perfectly metronomic bag.
Metrics are then recomputed using only frame_trace/completed rows whose
wall_clock_unix_sec falls within that shorter window.

    python3 debug/fps_scaling_experiment/normalize_session.py --reference run1 --session run2
    python3 debug/fps_scaling_experiment/normalize_session.py --reference run1 --session run3
    ...

Reads sessions/<reference>/summary.json and sessions/<session>/phase1_timing.csv
(both already written by snapshot_session.py -- run that first). Writes
sessions/<session>/summary_normalized.json and appends one row to
sessions/summary_all_sessions_normalized.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSIONS_DIR = REPO / "debug" / "fps_scaling_experiment" / "sessions"

NORMALIZED_CSV_FIELDS = [
    "session", "reference", "target_frame_count",
    "raw_recorded_span_sec", "raw_avg_fps",
    "normalized_window_sec", "normalized_frame_count", "normalized_avg_fps",
    "normalized_avg_frame_latency_ms", "normalized_avg_classifier_delay_ms",
    "normalized_avg_sam_inference_ms",
    "normalized_vlm_call_count", "normalized_vlm_success_count", "normalized_vlm_failed_count",
    "suggested_next_rate", "suggested_next_duration_sec",
]


def _stats(values: list[float]):
    if not values:
        return None, None, None
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return round(sum(values) / len(values), 2), round(median, 2), round(max(values), 2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", required=True,
                         help="session whose (frame_count + frame_drop_count) sets the input "
                              "frame budget every other session gets truncated to match, e.g. run1")
    parser.add_argument("--session", required=True,
                         help="session to re-compute over that same input frame budget, e.g. run2")
    parser.add_argument("--target-multiplier", type=float, default=2.0,
                         help="feed the next session's input at this many times the normalized "
                              "avg_fps (default: 2.0, matches snapshot_session.py's default)")
    parser.add_argument("--native-rate-hz", type=float, default=16.4121,
                         help="bag's native topic rate at --rate 1.0 (default: 16.4121, "
                              "uHumans2_office_s1_00h_ros2's /tesse/left_cam/rgb/image_raw)")
    args = parser.parse_args()

    ref_summary_path = SESSIONS_DIR / args.reference / "summary.json"
    if not ref_summary_path.exists():
        print(f"error: {ref_summary_path} not found -- run snapshot_session.py "
              f"--session {args.reference} first.", file=sys.stderr)
        return 1
    ref_summary = json.loads(ref_summary_path.read_text())
    ref_frame_count = ref_summary.get("frame_count")
    ref_drop_count = ref_summary.get("frame_drop_count")
    if ref_frame_count is None or ref_drop_count is None:
        print(f"error: {ref_summary_path} is missing frame_count/frame_drop_count.", file=sys.stderr)
        return 1
    target_frame_count = ref_frame_count + ref_drop_count

    timing_csv = SESSIONS_DIR / args.session / "phase1_timing.csv"
    if not timing_csv.exists():
        print(f"error: {timing_csv} not found -- run snapshot_session.py "
              f"--session {args.session} first.", file=sys.stderr)
        return 1

    frame_trace_rows = []   # (wall_clock, total_delay_ms, classifier_delay_ms, sam_inference_ms)
    completed_rows = []     # (wall_clock, success: bool)
    drop_count_total = 0
    with timing_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event = row.get("event", "")
            if event == "frame_trace":
                wc = row.get("wall_clock_unix_sec")
                if not wc:
                    continue
                try:
                    wc = float(wc)
                except ValueError:
                    continue

                def _f(key):
                    v = row.get(key)
                    try:
                        return float(v) if v else None
                    except ValueError:
                        return None

                frame_trace_rows.append((wc, _f("total_delay_ms"), _f("classifier_delay_ms"), _f("sam_inference_ms")))
            elif event == "completed":
                wc = row.get("wall_clock_unix_sec")
                if not wc:
                    continue
                try:
                    wc = float(wc)
                except ValueError:
                    continue
                reason = str(row.get("reason", "")).lower()
                completed_rows.append((wc, "fail" not in reason))
            elif event in ("dropped_oldest", "dropped_newest", "failed"):
                drop_count_total += 1

    if len(frame_trace_rows) < 2:
        print(f"error: {timing_csv} has fewer than 2 timestamped frame_trace rows; "
              "cannot compute this session's own input rate.", file=sys.stderr)
        return 1

    frame_trace_rows.sort(key=lambda r: r[0])
    session_frame_count = len(frame_trace_rows)
    session_span = frame_trace_rows[-1][0] - frame_trace_rows[0][0]
    raw_avg_fps = round(session_frame_count / session_span, 4) if session_span > 0 else None

    session_total_input = session_frame_count + drop_count_total
    if session_span <= 0 or session_total_input <= 0:
        print(f"error: {args.session} has zero recorded span or zero total input frames; "
              "cannot compute an empirical input rate.", file=sys.stderr)
        return 1
    empirical_input_rate = session_total_input / session_span

    if target_frame_count > session_total_input:
        print(f"WARNING: {args.session}'s own total input frame count ({session_total_input}) "
              f"is smaller than {args.reference}'s ({target_frame_count}) -- {args.session} never "
              "reached the reference's frame budget even over its full window. Reporting the full "
              "session, not a truncated one.", file=sys.stderr)
        normalized_window_sec = session_span
        cutoff = frame_trace_rows[-1][0]
    else:
        normalized_window_sec = round(target_frame_count / empirical_input_rate, 2)
        cutoff = frame_trace_rows[0][0] + normalized_window_sec

    kept_frames = [r for r in frame_trace_rows if r[0] <= cutoff]
    kept_completed = [r for r in completed_rows if r[0] <= cutoff]

    normalized_frame_count = len(kept_frames)
    if normalized_frame_count >= 2:
        kept_span = kept_frames[-1][0] - kept_frames[0][0]
        normalized_avg_fps = round(normalized_frame_count / kept_span, 4) if kept_span > 0 else None
    else:
        normalized_avg_fps = None

    latency_vals = [r[1] for r in kept_frames if r[1] is not None]
    classifier_vals = [r[2] for r in kept_frames if r[2] is not None]
    sam_vals = [r[3] for r in kept_frames if r[3] is not None]
    avg_latency, _, _ = _stats(latency_vals)
    avg_classifier, _, _ = _stats(classifier_vals)
    avg_sam, _, _ = _stats(sam_vals)

    vlm_success = sum(1 for _, ok in kept_completed if ok)
    vlm_failed = sum(1 for _, ok in kept_completed if not ok)

    # Derived from the *normalized* fps, not the raw one -- the whole point of
    # normalizing is that it's the fair reading, so it's what should drive the
    # next session's parameters. The duration formula simplifies to not need
    # native_rate_hz at all: play at rate R = k*fps/native for exactly the
    # time it takes rate R to deliver target_frame_count frames, i.e.
    # target_frame_count / (R * native) = target_frame_count / (k*fps), so any
    # imprecision in native_rate_hz only affects the suggested *rate*, not the
    # suggested *duration* needed to hit the same frame budget at that rate.
    suggested_next_rate = None
    suggested_next_duration_sec = None
    if normalized_avg_fps:
        suggested_next_rate = round(args.target_multiplier * normalized_avg_fps / args.native_rate_hz, 4)
        suggested_next_duration_sec = round(target_frame_count / (args.target_multiplier * normalized_avg_fps), 1)

    result = {
        "session": args.session,
        "reference": args.reference,
        "target_frame_count": target_frame_count,
        "raw_recorded_span_sec": round(session_span, 2),
        "raw_avg_fps": raw_avg_fps,
        "normalized_window_sec": normalized_window_sec,
        "normalized_frame_count": normalized_frame_count,
        "normalized_avg_fps": normalized_avg_fps,
        "normalized_avg_frame_latency_ms": avg_latency,
        "normalized_avg_classifier_delay_ms": avg_classifier,
        "normalized_avg_sam_inference_ms": avg_sam,
        "normalized_vlm_call_count": len(kept_completed),
        "normalized_vlm_success_count": vlm_success,
        "normalized_vlm_failed_count": vlm_failed,
        "suggested_next_rate": suggested_next_rate,
        "suggested_next_duration_sec": suggested_next_duration_sec,
    }

    out_path = SESSIONS_DIR / args.session / "summary_normalized.json"
    out_path.write_text(json.dumps(result, indent=2))

    csv_path = SESSIONS_DIR / "summary_all_sessions_normalized.csv"
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(result)

    print(f"=== {args.session} normalized to {args.reference}'s input-frame budget "
          f"({target_frame_count} frames) ===")
    print(f"  raw:        {session_frame_count} frames / {session_span:.2f}s  -> {raw_avg_fps} fps")
    print(f"  normalized: {normalized_frame_count} frames / {normalized_window_sec:.2f}s "
          f"-> {normalized_avg_fps} fps")
    for key in NORMALIZED_CSV_FIELDS[8:]:
        print(f"  {key:34s} {result.get(key)}")
    print(f"\nWrote {out_path}")
    print(f"Appended to {csv_path}")
    if suggested_next_rate is not None:
        print(f"\n--> Next session: ros2 bag play ... --rate {suggested_next_rate}, "
              f"for {suggested_next_duration_sec}s (timeout --signal=INT {int(round(suggested_next_duration_sec))})")
        print(f"    ({args.target_multiplier:g} x {normalized_avg_fps:.4f} normalized fps "
              f"/ {args.native_rate_hz:g} Hz native; duration sized to land on the same "
              f"{target_frame_count}-frame budget by construction, not truncated after the fact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
