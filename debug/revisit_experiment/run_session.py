#!/usr/bin/env python3
"""One command per run: sample resources, detect shutdown, archive -- automatically.

Supersedes running monitor_resources.py and snapshot_run.py by hand. Both
existing failure modes -- forgetting to archive at all, and typing the
literal placeholder "runN" instead of a real label -- are structural: they
require the operator to remember a manual step at the right moment. This
removes the step. Nothing overwrites a previous run's data: every run gets
its own auto-generated, sorted, never-reused session id.

Usage -- start this in its own terminal right after launching the pipeline
(terminal A), before starting the bag:

    python3 debug/revisit_experiment/run_session.py

Prints a session id immediately (e.g. session_20260909_223015) -- note it
down, or read it back later from sessions/summary_all_sessions.csv. Then:

  1. Waits for both hydra_ros_node and rsg_phase1_semantic_coordinator to
     be observed running (confirms it was started after the pipeline, not
     before -- otherwise "both processes are gone" would be true from the
     first check and it would archive nothing, immediately).
  2. Samples tegrastats every second into sessions/<id>/resource_usage.csv,
     same format as monitor_resources.py.
  3. Once both processes are seen alive and then BOTH have exited (i.e. you
     Ctrl+C'd the pipeline in terminal A and it finished its shutdown save),
     stops sampling, waits a few seconds for the save to settle, and
     archives automatically -- no separate snapshot_run.py call needed.
  4. Prints the summary and appends one row to sessions/summary_all_sessions.csv.

Ctrl+C here early (before the pipeline exits) stops sampling and archives
whatever is available at that moment, as a manual override -- the automatic
path above is what removes the need for that.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor_resources import parse_tegrastats_line  # noqa: E402
from snapshot_run import (  # noqa: E402
    SUMMARY_CSV_FIELDS,
    _append_summary_csv,
    _copy_tracker_and_hydra,
    _copy_timing_csv,
    _read_hydra_summary,
    _read_resource_summary,
    _read_timing_summary,
    _read_tracker_summary,
)

REPO = Path(__file__).resolve().parents[2]
SESSIONS_DIR = REPO / "debug" / "revisit_experiment" / "sessions"
SUMMARY_CSV = SESSIONS_DIR / "summary_all_sessions.csv"

PIPELINE_PROCESSES = ["hydra_ros_node", "rsg_phase1_semantic_coordinator"]
STARTUP_TIMEOUT_SEC = 120
SETTLE_SEC = 4  # after both processes exit, before archiving -- lets buffered saves flush


def _ancestor_pids() -> set[int]:
    """This process and its ancestors -- pgrep -f matches full command lines,
    so without this it would match the shell/script that launched it too."""
    import os

    pids: set[int] = set()
    pid = os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat[stat.rindex(")") + 1 :].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def _is_running(pattern: str, ignore: set[int]) -> bool:
    try:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid not in ignore:
            return True
    return False


def _both_running(ignore: set[int]) -> bool:
    return all(_is_running(pattern, ignore) for pattern in PIPELINE_PROCESSES)


def _both_gone(ignore: set[int]) -> bool:
    return all(not _is_running(pattern, ignore) for pattern in PIPELINE_PROCESSES)


def main() -> int:
    session_id = time.strftime("session_%Y%m%d_%H%M%S")
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    resource_csv = session_dir / "resource_usage.csv"

    print(f"[run_session] session id: {session_id}")
    print(f"[run_session] archiving to: {session_dir}")

    ignore = _ancestor_pids()

    print(f"[run_session] waiting up to {STARTUP_TIMEOUT_SEC}s for "
          f"{' and '.join(PIPELINE_PROCESSES)} to be running...")
    start_wait = time.time()
    while not _both_running(ignore):
        if time.time() - start_wait > STARTUP_TIMEOUT_SEC:
            print("[run_session] ERROR: pipeline not detected running -- "
                  "start terminal A first, then this. Exiting without archiving.",
                  file=sys.stderr)
            return 1
        time.sleep(1)
    print("[run_session] pipeline detected. Sampling started -- play the bag now.")

    proc = subprocess.Popen(
        ["tegrastats", "--interval", "1000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    import csv as csv_module

    fieldnames: list[str] = []
    rows_buffer: list[dict] = []
    sample_count = 0
    interrupted = False

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.strip():
                row = parse_tegrastats_line(line, time.time())
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
                rows_buffer.append(row)
                sample_count += 1

                temp_path = resource_csv.with_suffix(".csv.tmp")
                with temp_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv_module.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows_buffer)
                temp_path.replace(resource_csv)

                if sample_count % 15 == 0:
                    gpu = row.get("gpu_pct", "?")
                    print(f"[run_session] t={sample_count}s  gpu={gpu}%  "
                          f"(Ctrl+C to stop early; normally just Ctrl+C terminal A instead)")

            # Checked every ~1s, the same cadence as sampling -- once both
            # pipeline processes have exited, the run is over.
            if _both_gone(ignore):
                print("[run_session] pipeline has exited -- stopping sampling.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[run_session] stopped early by Ctrl+C.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not interrupted:
        print(f"[run_session] waiting {SETTLE_SEC}s for the shutdown save to settle...")
        time.sleep(SETTLE_SEC)

    print("[run_session] archiving...")
    _copy_tracker_and_hydra(session_dir)
    _copy_timing_csv(session_dir)

    summary = {"run_label": session_id}
    summary.update(_read_tracker_summary(session_dir))
    summary.update(_read_hydra_summary(session_dir))
    summary.update(_read_timing_summary(session_dir))
    summary.update(_read_resource_summary(session_dir))

    import json

    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False))
    _append_summary_csv(summary, csv_path=SUMMARY_CSV)

    print(f"\n=== {session_id} summary ===")
    for key in SUMMARY_CSV_FIELDS[1:]:
        print(f"  {key:28s} {summary.get(key)}")
    print(f"\nSession id: {session_id}")
    print(f"Wrote {session_dir / 'summary.json'}")
    print(f"Appended to {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
