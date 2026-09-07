# `memory/` — data that survives a run

Everything the pipeline carries from one run into the next lives here. The
contents are machine-local data stores, not source: `.gitignore` keeps this
README and ignores the rest.

```
memory/
  tracker/
    phase1_tracker_state.json    phase 1 object tracker, saved at shutdown
  rap/
    chroma/                      RAP visual memory (Chroma DB)
    phase1_rap_memory.jsonl      audit log of RAP memory additions
```

## `tracker/phase1_tracker_state.json`

Written by `destroy_node`, read at startup, both gated by
`phase1.persistent_tracking.session_persistence.enabled` (default off).

Holds the phase-1 side of the map together with the label attached to each
piece of it:

- every track — 3D box, centroid, volume, observation counts
- every local Hydra segment of every track, with its **slot id**, which is what
  ties a label to a Hydra node
- the semantic label, its confidence and source, the accumulated
  `label_evidence` / `label_observations`, and mobility classification
- `_allocated_slot_ids` and the id counters, so a restored session cannot
  re-issue an id that a restored track already owns

**Timestamps are shifted into the past on save** (`time_shift_sec`, one day by
default). This is required, not cosmetic. Bag replay restarts at the bag's own
start time, which is *earlier* than a saved `last_seen_timestamp_sec`, so the
raw age is negative and the tracker's `max(0.0, …)` clamp turns it into `0.0` —
which selects **recent** association mode. Shifting the stored times back makes
the age ~24 h, which selects **revisit** mode. See
`src/rsg/nodes/support/phase1/tracker_state_store.py`.

Restored tracks keep `labeling_completed`, so they are never re-sent to RAP or
the VLM. Their stored label is instead republished to the fuser once, on the
frame they are re-observed — the fuser's overlay cache is per-process and starts
empty, so without that republication a correctly tracked object would render
unlabeled.

## `rap/`

The Chroma store backing RAP retrieval, plus its audit log. The server is
launched with `--path` pointing here (`rsg_full_stack.launch.py`, overridable
via `RSG_RAP_STORAGE_PATH`); `phase1.rap.storage_path` must match.

## Starting clean

Delete the relevant file or directory. A missing tracker state file is treated
as a normal first run — it is logged and startup continues.
