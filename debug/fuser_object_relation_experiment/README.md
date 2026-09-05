# Fuser object-relation experiment

Goal: verify the fuser's object-contact edges (amber double-arrows) and
object-segment edges (blue dotted lines) — see `src/rsg/nodes/fuser.cpp`,
`computeObjectContacts()` / `computeObjectSegmentEdges()` /
`buildMainObjectGroups()` — actually match what the underlying geometry says
they should be, on a real bag, not just on hand-built synthetic scenes.

**See `IMPLEMENTATION.md` for the full design writeup** — the main-object
grouping model, the broad-phase/narrow-phase contact algorithm with worked
examples, the segment formation-order chain, rendering, the full parameter
reference, and a chronological log of every bug found and fixed during
development.

The fuser can dump every main object's aggregate bbox, every member's own
bbox, and the contact/segment edges it actually produced, one JSON line per
fusion cycle. `analyze_contact_diagnostics.py` reads that log and
**independently recomputes** the expected edges straight from the raw
geometry — a from-scratch Python reimplementation of the algorithm, not a
call into the C++ — then diffs against what was actually logged. A bug in
the C++ (wrong filter, wrong tie-break, a stale edge, a duplicate) shows up
as a mismatch instead of silently agreeing with itself.

---

## 1. Files

| file | purpose |
|---|---|
| `IMPLEMENTATION.md` | full design writeup: algorithm, examples, params, bug history |
| `analyze_contact_diagnostics.py` | reads a JSONL log, recomputes expected edges, reports mismatches |
| `results/` | one subfolder per recorded run: `contact_diagnostics.jsonl` (raw log) plus whatever `findings.csv`/`contact_edges.csv`/`segment_edges.csv` you've generated from it |
| `README.md` | this file |

## 2. Recording a log

Diagnostics are off by default (no cost on production runs). Enable via
node params, e.g. in the launch file or `ros2 param set`:

```
object_contact_diagnostics_enabled: true
```

That's the only param you need. Leave `object_contact_diagnostics_path`
unset and the fuser creates a fresh timestamped folder under `results/` the
first time it writes a line each run:

```
results/run_20260904_161530/contact_diagnostics.jsonl
```

— so successive runs never clobber each other's logs. The file is opened
once and kept open for the node's lifetime, appending + flushing every
cycle, so the log is usable even if the node is killed mid-run. Set
`object_contact_diagnostics_path` explicitly only if you want one fixed
path instead (e.g. to keep appending to the same file across restarts).

## 3. Running the analysis

```
python3 analyze_contact_diagnostics.py results/run_20260904_161530/contact_diagnostics.jsonl
```

Every node is identified by a readable `1a`/`1b`/`2a`-style id, not a raw
NodeId — main-object number + sub-object letter, assigned in the same
deterministic order the fuser itself groups and sorts them, so `1a`/`1b`
are always segments of the same physical object and `2a` is always a
different one.

For each cycle, prints nothing if it's clean; otherwise prints the sequence
number, timestamp, and every finding:

- `CONTACT missing: A-B expected (...) but not in logged contact_edges` —
  the geometry says A and B should have a contact edge and the fuser didn't
  produce one.
- `CONTACT extra: A-B logged but not expected from raw geometry` — the
  fuser produced an edge the geometry doesn't support.
- `SEGMENT missing` / `SEGMENT extra` — same, for the same-physical-object
  chain.
- `BOTH TYPES: A-B is logged as both a contact edge and a segment edge` —
  should be structurally impossible (contact edges only ever connect
  *different* main-object groups, segment edges only ever connect members
  of the *same* group); if this ever fires, it's the "dotted and solid line
  between the same two nodes" symptom (still open — see IMPLEMENTATION.md
  §"Still open"), and worth an actual repro.
- `STRUCTURAL: node N appears in two different main-object groups` — a
  logging-level inconsistency in `buildMainObjectGroups()`'s own output.

Exit code is `1` if anything was found (so it plugs into a script or CI
check), `0` if every cycle was clean. `--verbose` also prints every matched
edge, not just mismatches. `--max-lines N` stops early on a big log.

`--csv` additionally writes every finding as a CSV row (to open in a
spreadsheet, filter by `finding_type`, etc.) alongside the normal console
output — nothing is written if the log was entirely clean, just the header
row. Its parent directory is created automatically if it doesn't exist yet.
Columns: `line_number, sequence, stamp_sec, finding_type, source, target,
detail`, where `finding_type` is one of `CONTACT_MISSING`, `CONTACT_EXTRA`,
`SEGMENT_MISSING`, `SEGMENT_EXTRA`, `BOTH_TYPES`, `STRUCTURAL`.

Bare `--csv` (no path) defaults to `findings.csv` next to the input log, so
a run's raw JSONL and its derived CSV end up in the same folder:

```
python3 analyze_contact_diagnostics.py results/run_20260904_161530/contact_diagnostics.jsonl --csv
# -> results/run_20260904_161530/findings.csv
```

Or give it an explicit path to write somewhere else:

```
python3 analyze_contact_diagnostics.py results/run_20260904_161530/contact_diagnostics.jsonl --csv /tmp/findings.csv
```

## 4. Exporting the raw data (not just findings)

`--csv` only ever writes *mismatches* — nothing is written for a cycle that
matched expectations, so it's the wrong tool for browsing what the fuser
actually saw. For that, `--export-csv` dumps the logged data itself, one
row per edge observed per cycle, regardless of whether it was correct:

```
python3 analyze_contact_diagnostics.py results/run_20260904_161530/contact_diagnostics.jsonl --export-csv
# -> results/run_20260904_161530/contact_edges.csv
# -> results/run_20260904_161530/segment_edges.csv
```

Bare `--export-csv` writes both files next to the input log; `--export-csv DIR`
writes them into `DIR` instead (created if it doesn't exist). Every row
includes each side's resolved semantic label (`source_label`/`target_label`
— e.g. `floor`, `chair`; blank if phase 1/RAP hadn't labeled that node yet
that cycle, or if the log predates this field), alongside the readable
`1a`/`2a` display id, the raw NodeId, and (for contact edges) the full
geometry: gap, all three IoU values, contact axis, centroid distance.

## 5. Keeping this in sync

`aabb_contact()` / `expected_contact_edges()` / `expected_segment_edges()`
in `analyze_contact_diagnostics.py` are a hand-ported mirror of
`aabbContact()` / `computeObjectContacts()` / `computeObjectSegmentEdges()`
in `src/rsg/nodes/fuser.cpp`. There's no shared source of truth between them
— that's intentional, since the whole point is an independent check — but
it also means: if the C++ algorithm changes, this file needs the matching
update by hand, or every cycle will "fail" against the old rules.
