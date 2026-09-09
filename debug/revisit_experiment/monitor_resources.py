#!/usr/bin/env python3
"""Sample Jetson CPU/GPU load for one run of the revisit/FPS experiment.

This machine is a Jetson Orin (Tegra), not a discrete-GPU box: `nvidia-smi`
reports no usable utilisation figures on this platform (confirmed --
`memory.total` comes back "N/A"). The real source is `tegrastats`, Jetson's
own sampler, which this script wraps and parses into one CSV row per sample:
wall-clock timestamp, RAM used, per-core CPU%, GR3D_FREQ (the GPU load
percentage), temperatures, and power draw.

Usage -- run this in its own terminal, started right when you launch the
pipeline and stopped (Ctrl+C) right when you stop it:

    python3 debug/revisit_experiment/monitor_resources.py --run-label run1

Writes debug/revisit_experiment/runs/run1/resource_usage.csv, one row per
--interval-ms (default 1000). The file is flushed after every row, so a
Ctrl+C or crash never loses more than the last sample.

The wall-clock timestamp is what lines up this file against
rsg_object_detection_debug_*.csv's own wall_clock_unix_sec column (see
snapshot_run.py) -- both are real system time, not ROS/bag sim time, so they
compare directly regardless of playback rate.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO / "debug" / "revisit_experiment" / "runs"

# Matches "RAM 16579/62828MB" / "SWAP 14/31414MB"
_RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
_SWAP_RE = re.compile(r"SWAP (\d+)/(\d+)MB")
# Matches the bracketed per-core list: "[9%@729,3%@729,...]"
_CPU_BLOCK_RE = re.compile(r"CPU \[([^\]]+)\]")
_CPU_CORE_RE = re.compile(r"(\d+)%@(\d+)")
# GPU load. Field name varies across L4T/Jetpack versions.
_GPU_RE = re.compile(r"(?:GR3D_FREQ|GPU) (\d+)%")
# Temperatures: "gpu@42.718C", "cpu@47.187C" -- word@float C, distinct from
# the per-core CPU list's digit%@digit form above.
_TEMP_RE = re.compile(r"\b(\w+)@([\d.]+)C\b")
# Power rails: "VDD_GPU_SOC 3607mW/3607mW" (instantaneous/average)
_POWER_RE = re.compile(r"(VDD_\w+|VIN_\w+) (\d+)mW/(\d+)mW")


def parse_tegrastats_line(line: str, wall_clock: float) -> dict:
    """Parse one tegrastats line into a flat dict. Missing fields stay blank."""
    row = {"wall_clock_unix_sec": wall_clock, "raw_line": line.strip()}

    ram = _RAM_RE.search(line)
    if ram:
        row["ram_used_mb"] = int(ram.group(1))
        row["ram_total_mb"] = int(ram.group(2))

    swap = _SWAP_RE.search(line)
    if swap:
        row["swap_used_mb"] = int(swap.group(1))

    cpu_block = _CPU_BLOCK_RE.search(line)
    if cpu_block:
        cores = _CPU_CORE_RE.findall(cpu_block.group(1))
        if cores:
            pcts = [int(p) for p, _freq in cores]
            row["cpu_num_cores"] = len(pcts)
            row["cpu_avg_pct"] = sum(pcts) / len(pcts)
            row["cpu_max_pct"] = max(pcts)

    gpu = _GPU_RE.search(line)
    if gpu:
        row["gpu_pct"] = int(gpu.group(1))

    for name, value in _TEMP_RE.findall(line):
        row[f"temp_{name}_c"] = float(value)

    for rail, now, avg in _POWER_RE.findall(line):
        row[f"power_{rail.lower()}_now_mw"] = int(now)
        row[f"power_{rail.lower()}_avg_mw"] = int(avg)

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-label", required=True,
                        help="e.g. run1, run2 -- must match what you pass to snapshot_run.py")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="parent directory; writes <out-dir>/<run-label>/resource_usage.csv")
    parser.add_argument("--interval-ms", type=int, default=1000,
                        help="tegrastats sampling interval")
    parser.add_argument("--duration-sec", type=float, default=None,
                        help="optional auto-stop after this many seconds; default runs until Ctrl+C")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "resource_usage.csv"

    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(args.interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print(f"[monitor_resources] sampling every {args.interval_ms}ms -> {out_path}")
    print("[monitor_resources] Ctrl+C to stop when the run ends.")

    start = time.time()
    written = 0
    fieldnames: list[str] = []
    rows_buffer: list[dict] = []

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line.strip():
                continue
            now = time.time()
            row = parse_tegrastats_line(line, now)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
            rows_buffer.append(row)
            written += 1

            # Rewrite the file each sample rather than append: the column set
            # can only grow (a field absent from the very first line but
            # present later), and CSV needs one fixed header, so a plain
            # append could produce a header mismatched to later rows. At a 1s
            # interval and a five-minute run this is a few hundred rows --
            # cheap to rewrite every time, and it means the file is always
            # valid and complete up to "now" if you Ctrl+C or it crashes.
            temp_path = out_path.with_suffix(".csv.tmp")
            with temp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows_buffer)
            temp_path.replace(out_path)

            if written % 10 == 0:
                elapsed = now - start
                gpu = row.get("gpu_pct", "?")
                cpu = row.get("cpu_avg_pct", "?")
                print(f"[monitor_resources] t={elapsed:6.1f}s  samples={written:4d}  "
                      f"gpu={gpu}%  cpu_avg={cpu}%", flush=True)

            if args.duration_sec is not None and (now - start) >= args.duration_sec:
                break
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"[monitor_resources] done: {written} samples over {time.time() - start:.1f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
