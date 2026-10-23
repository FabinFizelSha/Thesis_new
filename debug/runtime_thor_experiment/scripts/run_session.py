#!/usr/bin/env python3
"""Per-run orchestration for the Thor runtime benchmark (adapted from
debug/revisit_experiment/run_session.py).

Two steps per run:

  1. BEFORE launching the pipeline (terminal A):
         python3 run_session.py prepare --run-id T0_r1 --profile uhumans2
     Creates runs/<run_id>_<timestamp>/, retargets the symlink runs/_current
     (the pipeline YAMLs write phase1_timing.csv / preprocessor_timing.xlsx
     through it), and saves a copy of the effective YAML + launch metadata.

  2. Right after launching the pipeline (terminal C), before playing the bag:
         python3 run_session.py monitor
     Waits for hydra_ros_node + rsg_phase1_semantic_coordinator, then samples
     tegrastats (+ nvidia-smi GPU %) at 1 Hz and runs topic_rate_logger.py.
     Once both processes have exited (Ctrl+C on the launch terminal and its
     shutdown save finished) it archives everything and prints a summary.
"""
from __future__ import annotations

import argparse, csv, glob, json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from monitor_resources import parse_tegrastats_line  # noqa: E402
from snapshot_run import _read_resource_summary, _read_timing_summary  # noqa: E402

REPO = HERE.parents[2]
RUNS = REPO / "debug" / "runtime_thor_experiment" / "runs"
CURRENT = RUNS / "_current"
CONFIGS = {"uhumans2": REPO / "src/rsg/config/rsg_pipeline.yaml",
           "openloris": REPO / "src/rsg/config/rsg_pipeline_openloris.yaml"}
HYDRA_DATASET = {"uhumans2": "uhumans2", "openloris": "openloris"}
PROCS = ["hydra_ros_node", "rsg_phase1_semantic_coordinator"]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ANSI2 = re.compile(r"\[\d+(;\d+)*m")   # colour codes whose ESC byte was lost through tee
VLM_LINE = re.compile(r"print_timing.*(prompt eval time|\beval time|total time)")


def _ancestors():
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            st = Path(f"/proc/{pid}/stat").read_text()
            pid = int(st[st.rindex(")") + 1:].split()[1])
        except Exception:
            break
    return pids


def _running(pattern, ignore):
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return any(int(x) not in ignore for x in r.stdout.split())


def prepare(a):
    run_dir = RUNS / f"{a.run_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)
    if CURRENT.is_symlink() or CURRENT.exists():
        CURRENT.unlink()
    CURRENT.symlink_to(run_dir.name)
    shutil.copy2(CONFIGS[a.profile], run_dir / "effective_pipeline.yaml")
    meta = {"run_id": a.run_id, "profile": a.profile, "bag_rate": a.rate, "notes": a.notes,
            "prepared_at": time.time(), "git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepare] run dir : {run_dir}\n[prepare] _current -> {run_dir.name}")
    print(f"[prepare] launch with:  ... 2>&1 | tee -i {CURRENT}/launch.log")


def monitor(a):
    run_dir = (RUNS / os.readlink(CURRENT)).resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    ignore = _ancestors()
    print(f"[monitor] run {run_dir.name}; waiting for {PROCS}")
    t0 = time.time()
    while not all(_running(p, ignore) for p in PROCS):
        if time.time() - t0 > 180:
            print("[monitor] ERROR: pipeline not detected", file=sys.stderr); return 1
        time.sleep(1)
    start = time.time()
    meta["monitor_started"] = start
    # Topic logger (sourced env is inherited from this shell)
    tl_log = open(run_dir / "topic_rate_logger.stderr", "w")
    tl = subprocess.Popen([sys.executable, str(HERE / "topic_rate_logger.py"), "--out", str(run_dir / "topic_rates.csv")],
                          stderr=tl_log)
    ts = subprocess.Popen(["tegrastats", "--interval", "1000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    (run_dir / "tegrastats_raw.txt").write_text("")
    raw = open(run_dir / "tegrastats_raw.txt", "a")
    rows, fields, n = [], [], 0
    print("[monitor] sampling. Wait 30 s warm-up, then play the bag.")
    try:
        for line in ts.stdout:
            if not line.strip():
                continue
            raw.write(line); raw.flush()
            row = parse_tegrastats_line(line, time.time()); row.pop("raw_line", None)
            for k in row:
                if k not in fields: fields.append(k)
            rows.append(row); n += 1
            if n % 10 == 0:   # periodic rewrite, valid file at any time
                with open(run_dir / "resource_usage.csv", "w", newline="") as h:
                    w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
                if n % 30 == 0:
                    print(f"[monitor] t={n}s gpu={row.get('gpu_pct')}% cpu={row.get('cpu_avg_pct', 0):.0f}%")
            if all(not _running(p, ignore) for p in PROCS):
                print("[monitor] pipeline exited."); break
    except KeyboardInterrupt:
        print("[monitor] interrupted; archiving what exists.")
    finally:
        ts.terminate()
        tl.send_signal(signal.SIGINT)
        try: tl.wait(timeout=10)
        except subprocess.TimeoutExpired: tl.kill()
    with open(run_dir / "resource_usage.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    time.sleep(4)
    archive(run_dir, meta)
    return 0


def archive(run_dir, meta):
    ds = HYDRA_DATASET[meta["profile"]]
    # VLM server log from launch.log
    lg = run_dir / "launch.log"
    if lg.exists():
        lines = [ANSI2.sub("", ANSI.sub("", l)) for l in lg.read_text(errors="replace").splitlines()]
        keep = [l for l in lines if VLM_LINE.search(l)]
        (run_dir / "vlm_server.log").write_text("\n".join(keep) + "\n")
    else:
        print("[archive] WARNING: launch.log missing (was tee used?)")
    # Hydra timers (written at Hydra shutdown to its log dir)
    hd = REPO / "memory" / "hydra" / ds
    out = run_dir / "hydra_timing"; out.mkdir(exist_ok=True)
    if (hd / "timing_stats.csv").exists():
        shutil.copy2(hd / "timing_stats.csv", out)
    if (hd / "timing").is_dir():
        shutil.copytree(hd / "timing", out / "timing", dirs_exist_ok=True)
    # Existence report
    files = {f: (run_dir / f).exists() and (run_dir / f).stat().st_size > 0
             for f in ["phase1_timing.csv", "preprocessor_timing.xlsx", "resource_usage.csv", "topic_rates.csv", "vlm_server.log", "launch.log"]}
    files["hydra_timing_stats.csv"] = (out / "timing_stats.csv").exists()
    # VLM numbers
    vlm = []
    p = run_dir / "phase1_timing.csv"
    if p.exists():
        for r in csv.DictReader(open(p, newline="")):
            if r.get("vlm_delay_ms"):   # VLM result rows have an empty `event` column
                try: vlm.append(float(r["vlm_delay_ms"]))
                except ValueError: pass
    summ = {"run_id": meta["run_id"], "profile": meta["profile"], "files_nonempty": files}
    summ.update(_read_timing_summary(run_dir)); summ.update(_read_resource_summary(run_dir))
    summ["vlm_completed_rows_with_delay"] = len(vlm)
    summ["vlm_delay_ms_mean"] = round(sum(vlm) / len(vlm), 1) if vlm else None
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\n=== {run_dir.name} ===")
    for k in ["frame_count", "avg_fps", "avg_frame_latency_ms", "avg_sam_inference_ms", "vlm_call_count", "vlm_delay_ms_mean"]:
        print(f"  {k:26s} {summ.get(k)}")
    print("  files:", {k: ("ok" if v else "MISSING/EMPTY") for k, v in files.items()})


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--run-id", required=True)
    p.add_argument("--profile", choices=list(CONFIGS), required=True)
    p.add_argument("--rate", type=float, default=1.0); p.add_argument("--notes", default="")
    sub.add_parser("monitor")
    a = ap.parse_args()
    return prepare(a) if a.cmd == "prepare" else monitor(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
