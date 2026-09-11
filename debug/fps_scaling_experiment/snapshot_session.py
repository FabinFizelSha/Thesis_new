#!/usr/bin/env python3
"""Archive one session's diagnostic data for the memory/FPS scaling experiment,
and compute the bag --rate to use for the *next* session.

This does not run, launch, or monitor anything -- run it by hand once after a
session's clean shutdown and before starting the next one, exactly like
debug/revisit_experiment/snapshot_run.py (whose helpers this reuses directly
rather than re-implementing). See EXPERIMENT_PLAN.md in this folder for the
full procedure this supports.

    python3 debug/fps_scaling_experiment/snapshot_session.py --session calibration
    python3 debug/fps_scaling_experiment/snapshot_session.py --session run1
    python3 debug/fps_scaling_experiment/snapshot_session.py --session run2
    ...

Copies (never moves/deletes the originals -- same sources as snapshot_run.py):
  - memory/tracker/phase1_tracker_state.json  -> sessions/<session>/phase1_tracker_state.json
  - memory/hydra/backend/dsg.json             -> sessions/<session>/hydra_dsg.json
  - the newest rsg_object_detection_debug_*.xlsx under
    /home/student/rsg_ros2_ws/debug/ (CSV content despite the extension)
                                               -> sessions/<session>/phase1_timing.csv

Then prints a summary, writes sessions/<session>/summary.json, appends one row
to sessions/summary_all_sessions.csv, and -- the part specific to this
experiment -- prints the suggested bag --rate for the next session:

    next_rate = target_multiplier * this_session's_avg_fps / native_rate_hz

where native_rate_hz is the bag's own /tesse/left_cam/rgb/image_raw publish
rate at --rate 1.0 (measured once for uHumans2_office_s1_00h_ros2: 8307
messages / 506.1499 s = 16.4121 Hz -- recompute with `ros2 bag info` for a
different bag and pass --native-rate-hz).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSIONS_DIR = REPO / "debug" / "fps_scaling_experiment" / "sessions"

# Reuse debug/revisit_experiment/snapshot_run.py's helpers rather than
# re-implementing the same tracker/hydra/timing-CSV parsing a second time.
sys.path.insert(0, str(REPO / "debug" / "revisit_experiment"))
import snapshot_run as base  # noqa: E402

DEFAULT_NATIVE_RATE_HZ = 16.4121  # uHumans2_office_s1_00h_ros2, see docstring above.
DEFAULT_TARGET_MULTIPLIER = 2.0

SUMMARY_CSV_FIELDS = base.SUMMARY_CSV_FIELDS[:1] + ["suggested_next_rate"] + base.SUMMARY_CSV_FIELDS[1:]
# ^ ["run_label", "suggested_next_rate", "track_count", ...] -- run_label holds
#   the session name here (kept as-is so the underlying dict/row shape from
#   snapshot_run.py's helpers needs no translation).


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", required=True,
                         help="e.g. calibration, run1, run2, run3, run4, run5")
    parser.add_argument("--native-rate-hz", type=float, default=DEFAULT_NATIVE_RATE_HZ,
                         help=f"bag's native topic rate at --rate 1.0 (default: {DEFAULT_NATIVE_RATE_HZ}, "
                              "uHumans2_office_s1_00h_ros2's /tesse/left_cam/rgb/image_raw)")
    parser.add_argument("--target-multiplier", type=float, default=DEFAULT_TARGET_MULTIPLIER,
                         help=f"feed the next session's input at this many times the achieved fps "
                              f"(default: {DEFAULT_TARGET_MULTIPLIER})")
    args = parser.parse_args()

    out_dir = SESSIONS_DIR / args.session
    out_dir.mkdir(parents=True, exist_ok=True)

    base._copy_tracker_and_hydra(out_dir)
    base._copy_timing_csv(out_dir)

    summary = {"run_label": args.session}
    summary.update(base._read_tracker_summary(out_dir))
    summary.update(base._read_hydra_summary(out_dir))
    summary.update(base._read_timing_summary(out_dir))
    summary.update(base._read_resource_summary(out_dir))

    avg_fps = summary.get("avg_fps")
    if avg_fps:
        suggested_next_rate = round(args.target_multiplier * avg_fps / args.native_rate_hz, 4)
    else:
        suggested_next_rate = None
        print("[snapshot_session] WARNING: avg_fps unavailable (no wall_clock_unix_sec-stamped "
              "frame_trace rows, or fewer than 2 of them) -- cannot suggest a next rate. "
              "Was phase1.performance.measure_timing on, and did the session run long enough "
              "to process at least 2 frames?", file=sys.stderr)
    summary["suggested_next_rate"] = suggested_next_rate

    import json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False))
    # _append_summary_csv writes with snapshot_run's own module-level
    # SUMMARY_CSV_FIELDS (a global lookup in *its* namespace, unaffected by a
    # same-named variable here) -- patch it for this call so
    # suggested_next_rate actually lands in the row instead of being silently
    # dropped by DictWriter's extrasaction="raise" default... actually
    # DictWriter uses whatever fieldnames it's given to select columns, so
    # this substitution is what makes the extra column appear at all.
    base.SUMMARY_CSV_FIELDS = SUMMARY_CSV_FIELDS
    csv_path = base._append_summary_csv(summary, csv_path=SESSIONS_DIR / "summary_all_sessions.csv")

    print(f"\n=== {args.session} summary ===")
    for key in SUMMARY_CSV_FIELDS[1:]:
        print(f"  {key:28s} {summary.get(key)}")
    print(f"\nWrote {out_dir / 'summary.json'}")
    print(f"Appended to {csv_path}")
    if suggested_next_rate is not None:
        print(f"\n--> Next session: ros2 bag play ... --rate {suggested_next_rate}")
        print(f"    ({args.target_multiplier:g} x {avg_fps:.4f} fps achieved this session, "
              f"/ {args.native_rate_hz:g} Hz native)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
