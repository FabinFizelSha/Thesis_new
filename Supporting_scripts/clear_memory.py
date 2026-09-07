#!/usr/bin/env python3
"""Delete the pipeline's cross-run memory so the next run starts from scratch.

Everything that survives a run lives under `memory/` (see memory/README.md).
This deletes it. It does not touch any config: the persistence features stay
enabled, so the run after a clear saves state again, and the run after *that*
reads it back. That is the normal way to get a clean two-run resume test.

    python3 clear_memory.py                # clear everything
    python3 clear_memory.py --dry-run      # show what would go, delete nothing
    python3 clear_memory.py --tracker      # clear only the phase 1 tracker
    python3 clear_memory.py --hydra --rap  # clear those two, keep the tracker

Stores:
  tracker   memory/tracker/   phase 1 object tracker state (tracks, slot ids,
                              labels). Clearing it means run 1 has no revisits.
  hydra     memory/hydra/     Hydra's saved DSG, mesh and deformation graph,
                              plus its logs. Clearing it means run 1 opens on
                              an empty map.
  rap       memory/rap/       RAP visual memory (Chroma store + audit log).

Refuses to run while the pipeline is up, since phase 1 rewrites its state at
shutdown and would undo the clear -- and Chroma holds its directory open.
Override with --force if you know the processes are unrelated.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Hardcoded rather than derived from __file__: this script lives under
# Supporting_scripts/, one level below the workspace root, and every other
# path in this project (rsg_pipeline.yaml's state_path, storage_path, etc.)
# is already hardcoded to this same absolute root rather than computed.
REPO = Path("/home/student/Thesis_new")
MEMORY = REPO / "memory"

# Kept because they are the only tracked things in memory/ -- .gitignore has
# `memory/*` plus `!memory/README.md`.
KEEP = {"README.md", ".gitignore"}

STORES = {
    "tracker": ("memory/tracker", "phase 1 object tracker state"),
    "hydra": ("memory/hydra", "Hydra DSG, mesh and logs"),
    "rap": ("memory/rap", "RAP visual memory (Chroma + audit log)"),
}

# Processes that hold a store's files open, or would rewrite them on exit,
# keyed by the store they affect. Scoped per store so a long-lived Chroma
# server does not block clearing the tracker, which it has nothing to do with.
STORE_PROCESSES = {
    "tracker": ["phase1"],
    "hydra": ["hydra_node"],
    "rap": ["chroma"],
}


def _ancestor_pids():
    """This process and its ancestors, to be ignored when scanning pgrep output.

    `pgrep -f` matches against full command lines, so it happily matches the
    shell that launched this script (the pattern appears in its command line)
    and this script itself. Without this the check reports every store busy.
    """
    pids = set()
    pid = os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            # Fields after the comm field, which is parenthesised and may
            # itself contain spaces: ppid is the second field after it.
            pid = int(stat[stat.rindex(")") + 1:].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def running_processes(selected):
    """Process names blocking a clear of the selected stores, if any."""
    patterns = sorted({p for name in selected for p in STORE_PROCESSES.get(name, [])})
    ignore = _ancestor_pids()
    found = []
    for pattern in patterns:
        try:
            result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
        except FileNotFoundError:
            return []  # no pgrep; skip the check rather than block the clear
        for line in result.stdout.splitlines():
            pid_text, _, cmdline = line.partition(" ")
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if pid in ignore or Path(__file__).name in cmdline:
                continue
            found.append(pattern)
            break
    return found


def describe(path):
    """Return (file count, total bytes) for a file or directory tree."""
    if path.is_file():
        return 1, path.stat().st_size
    count = 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            count += 1
            total += child.stat().st_size
    return count, total


def human(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def targets_for(selected):
    """Selected stores that actually hold data, as (name, path, files, bytes).

    A store left as an empty directory by a previous clear is skipped, so
    re-running reports nothing to do rather than "deleted 0 files".
    """
    found = []
    for name in selected:
        relative, _ = STORES[name]
        path = REPO / relative
        if not path.exists():
            continue
        count, size = describe(path)
        if count:
            found.append((name, path, count, size))
    return found


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name, (_, blurb) in STORES.items():
        parser.add_argument(f"--{name}", action="store_true", help=f"clear {blurb}")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be deleted, delete nothing")
    parser.add_argument("--force", action="store_true",
                        help="clear even while pipeline processes are running")
    args = parser.parse_args()

    # No store flags means all of them: "clear memory" is the common case.
    selected = [name for name in STORES if getattr(args, name)] or list(STORES)

    if not MEMORY.exists():
        print(f"nothing to clear: {MEMORY} does not exist")
        return 0

    if not args.dry_run and not args.force:
        busy = running_processes(selected)
        if busy:
            print(f"refusing to clear: {', '.join(busy)} still running.", file=sys.stderr)
            print("Phase 1 rewrites its state at shutdown and would undo this; "
                  "Chroma holds its directory open.", file=sys.stderr)
            print("Stop the pipeline first, or pass --force.", file=sys.stderr)
            return 1

    found = targets_for(selected)
    if not found:
        print(f"nothing to clear: no data present for {', '.join(selected)}")
        return 0

    verb = "would delete" if args.dry_run else "deleted"
    total_files = 0
    total_bytes = 0
    for name, path, count, size in found:
        total_files += count
        total_bytes += size
        print(f"  {verb} {path.relative_to(REPO)}  ({count} file(s), {human(size)})")
        if args.dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path)
            # Recreate empty: the writers all create their own parents, but an
            # existing directory keeps paths in configs and launch files valid.
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.unlink()

    # Sweep anything in memory/ that is not a known store, so a renamed or
    # newly added store cannot silently survive a "clear everything".
    if set(selected) == set(STORES):
        known = {(REPO / relative).name for relative, _ in STORES.values()}
        for stray in sorted(MEMORY.iterdir()):
            if stray.name in KEEP or stray.name in known:
                continue
            count, size = describe(stray)
            total_files += count
            total_bytes += size
            print(f"  {verb} {stray.relative_to(REPO)}  "
                  f"({count} file(s), {human(size)}) [unrecognised]")
            if args.dry_run:
                continue
            shutil.rmtree(stray) if stray.is_dir() else stray.unlink()

    print(f"\n{verb}: {total_files} file(s), {human(total_bytes)}")
    if not args.dry_run:
        print("Next run starts clean and saves state; the run after it reads that state back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
