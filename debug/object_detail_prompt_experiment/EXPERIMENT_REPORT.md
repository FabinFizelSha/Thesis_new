# object_detail Prompt-Addition Experiment — VLM Visual-State Description

**Single source of truth for this experiment. Do not create additional main
documents — append to this file (Changelog, §15) as work happens, exactly
like `debug/prompt_optimisation_experiment/EXPERIMENT_REPORT.md` did for the
label-prompt experiment.**

- **Started:** 2026-09-18
- **Author:** Fabin Fizel Sha
- **Builds on:** `debug/prompt_optimisation_experiment/` (frozen, complete,
  9/9 runs) — that experiment answered *"which prompt best identifies the
  object?"*. This one answers *"can one short, appended section make the same
  frozen prompt also report the object's current visible state, without
  breaking the identification it already does well?"*
- **Status:** V1 ACTIVATED 2026-09-18 — `phase1.vlm.prompt` in
  `rsg_pipeline.yaml` is now `BASELINE_FROZEN` + `V1_basic_addition`
  (byte-verified), `object_detail_experiment.enabled: true`. Awaiting the
  first graded test run (§11/§12 still empty).

---

## 1. Objective

Add a fifth output field, `object_detail`, to the VLM's JSON response: a
short phrase describing the *current visible state* of the object (what is on
it, its condition, whether it looks in use) — as opposed to `label`, which
only names *what it is*.

Constraints set by the thesis supervisor's brief for this feature (see the
pipeline-compatibility work in §2):

1. The existing object-detection prompt — winner of the prior campaign,
   `qwen3_5_4b_q4` × `P3_v6_structural_priority`, 88 % accuracy — **must not
   change**. It is reproduced verbatim in
   `object_detail_prompts_under_test.yaml` as `BASELINE_FROZEN` and every
   version tested here is that text plus one appended section, never an edit
   to it.
2. Start with a very basic addition, then iterate systematically based on
   real results — the same methodology
   `debug/prompt_optimisation_experiment/` used (P1 → P2 → P3, each version
   written *against the previous version's actual failures*, not guessed
   up front).
3. `object_detail` should describe *state*, not identity — the three worked
   examples given were: a table with "monitor and keyboard on table,
   organized properly"; a floor with "water on floor" or "carpet covering
   floor"; a monitor with "display on" / "powered on".

Primary questions:

1. Does a short appended instruction get the model to reliably emit a useful
   `object_detail` value, or does it get ignored / degrade into a generic
   filler phrase?
2. Does adding this field **regress** the frozen prompt's existing 88 %
   label accuracy (e.g. by confusing the model about the JSON schema, or by
   spending its limited attention/tokens on the new field)? This is graded
   with the *same* `error_category` vocabulary the label experiment used
   (§9 there), reused unchanged in the CSV schema here (§8).
3. How many iterations does the addition need before `object_detail` is
   consistently accurate and specific rather than vague or hallucinated?

Out of scope: re-opening the label-prompt choice itself (frozen), re-running
the model sweep (this experiment holds the model fixed at the production
winner, `qwen3_5_4b_q4` — see §6), RAP/Hydra fusion behaviour.

---

## 2. Relationship to prior work

**The label-prompt experiment** (`debug/prompt_optimisation_experiment/`,
frozen 2026-09-05) established the baseline prompt this experiment must not
touch. Read that report's §13 before touching anything here — it explains
*why* the frozen prompt is shaped the way it is (structural-priority
ordering, the format-hardening `OUTPUT RULE` block, the calibrated-confidence
language), all of which V1's addition risks colliding with (see §14).

**The pipeline-compatibility work** (this same development thread, completed
immediately before this experiment folder was created) wired `object_detail`
through the whole pipeline *ahead of* the prompt actually asking for it, using
a dummy placeholder, so that turning the real field on later needs no code
changes — only a prompt change and a config flag flip. Concretely, already in
place and unaffected by anything in this experiment:

- `Phase1VlmResult.msg` carries `object_detail` (string).
- `vlm_result.py`'s `validate_vlm_response` extracts `object_detail` from the
  parsed JSON when present, else falls back to
  `DEFAULT_OBJECT_DETAIL = "This_is_a_sample_sentence"`. Spaces in whatever
  value is used (dummy or real) are normalised to underscores
  (`"a red chair"` → `"a_red_chair"`) so the fuser's RViz text label never
  wraps across whitespace.
- `persistent_object_tracker.py`'s `Track.object_detail` field, set by both
  `apply_vlm_result` (fresh VLM call) and `apply_rap_result` (RAP-only
  recognition — reads a previously-stored `object_detail` back out of RAP
  memory, so an object recognised by RAP alone still shows its last known
  description).
- `rap_memory.py`'s JSONL log and the RAP vector-store `add_image` metadata
  both persist `object_detail`.
- `tracker_state_store.py` persists `object_detail` across a phase 1 restart
  (`save_tracker_state` / `load_tracker_state`), so it survives pipeline
  closure and session resume.
- `fuser.cpp` displays it as an extra line under the object's RViz label
  (`show_object_detail`, default on) and includes it in the `rsg_rap`/`rap`
  diagnostic JSON blocks.

None of that is re-tested here — it was verified with the dummy placeholder
flowing end-to-end. This experiment only concerns the **text asked of the
VLM** and **whether the resulting value is any good**, using the diagnostic
infrastructure described in §7.

---

## 3. Baseline prompt (verbatim — frozen, never edited by this experiment)

Byte-identical to `phase1.vlm.prompt` in `src/rsg/config/rsg_pipeline.yaml`
as of 2026-09-18 (verified programmatically — see
`object_detail_prompts_under_test.yaml`'s `BASELINE_FROZEN` entry, which is
loaded from the same YAML parser and diffed equal). Reproduced in full here
because the thesis chapter for this experiment needs to quote it directly
without depending on the other experiment's document staying unchanged.

```text
OUTPUT RULE — READ THIS FIRST
Your entire reply is ONE JSON object and nothing else. The first character you output is "{" and the last is "}". No preamble, no analysis, no numbered steps, no sentences such as "Based on the visual evidence...", and no markdown code fences. Work through the approach below silently and output only the result.

This is an indoor object-classification task. Identify the object that the cyan boundary encloses.

APPROACH (reason through this silently - do not write any of it)
1. Look only inside the cyan boundary. Everything outside it is background - never name it, however large or obvious. Use the surroundings only to work out what the enclosed region is.
2. Decide what the enclosed region mainly shows.
3. Choose the label using the priority order below.

PRIORITY ORDER
a. Room-defining structure over incidental detail. The large planes that form the room - overhead, underfoot, the vertical sides - and the openings and supports in them take priority over small things mounted on, set into, or resting against them. If the boundary spans many repeated units of one surface (several ceiling tiles or panels, a run of floor, a stretch of wall), name the surface itself ("ceiling", "wall", "floor"), not one unit ("ceiling_tile").
b. BUT distinctive infrastructure that fills the boundary is named, not generalised. If exposed pipes, ductwork, a large air vent or diffuser, a fire hose reel, a sprinkler main or a big fixture dominates the enclosed region, name that thing. Collapsing to "ceiling" or "wall" is only right when the fixture is small or incidental and the surface is what the boundary mostly shows.
c. Furniture over the objects on it - but only with a furniture cue. A freestanding piece of furniture outranks the smaller things on it. A large flat vertical surface with no handle, seam, gap, toe-kick or visible depth is a wall or a pillar, not a cabinet - do not upgrade a plain surface to furniture without a cue.
d. When two readings compete, choose the larger, more dominant, more structural one.

MATERIAL AND AMBIGUITY CUES
- A transparent or translucent panel, a framed glass screen, a glazed partition or a glass-panelled door -> name it as glass (for example "glass_wall", "glass_partition", "glass_door"). Offices often divide space with glass rather than solid wall.
- Judge transparency directly: if the room, objects, people or light behind the panel are visible through it - even dimly, even with a frame, mullions or a faint surface reflection - it is glass; a panel you cannot see through at all is solid (wall, door, cabinet).
- A rectangular panel that reflects the room or shows a mirror image -> mirror. A plain matte white or lightly-marked panel -> whiteboard. If you cannot tell, name the wall it sits on at low confidence.
- Low detail, deep shadow, motion blur or heavy truncation -> do not commit to a specific object; give the structural reading (wall / floor / ceiling) at low confidence, or VLM_no_result if nothing is identifiable.

LABEL FORM
- A concise, singular, lowercase label that best describes what it actually is. There is no fixed list of labels - name whatever it is. Do not describe colour, condition or activity.

CONFIDENCE CALIBRATION for "label_confidence" - report it honestly
- 0.90-1.00: unmistakable; the enclosed region can only be this.
- 0.70-0.90: clear, minor uncertainty.
- 0.45-0.70: plausible, but a competing reading exists - the right band for cabinet-vs-wall and mirror-vs-whiteboard.
- 0.20-0.45: genuinely ambiguous, or a guess.
- 0.0 with label VLM_no_result: incomprehensible crop.
A low confidence on a hard crop is better than a confident wrong answer. Do not default to 0.90+.

Mobility:
- "dynamic": only a human, an animal, or a self-propelled moving object
- "static": any stationary object, including all surfaces, structures, furniture, glass and fixtures
- "unknown": when the label is unknown

OUTPUT - exactly one JSON object, with nothing before it or after it:
{"label": "<label>", "label_confidence": <0-1>, "mobility_class": "<static|dynamic|unknown>", "mobility_confidence": <0-1>}

GOOD - this is the whole reply:
{"label": "wall", "label_confidence": 0.85, "mobility_class": "static", "mobility_confidence": 0.99}

BAD - never reply like any of these:
- Based on the visual evidence within the cyan boundary: 1. Analysis: ... then the JSON
- The object is a wall. {"label": "wall", ...}
- the JSON object wrapped in triple-backtick code fences

WORKED EXAMPLES (illustrating the approach, not a list of allowed labels):

Room-defining structure over incidental detail:
- Boundary spans a panelled ceiling covering many tiles and a recessed light -> {"label": "ceiling", "label_confidence": 0.92, "mobility_class": "static", "mobility_confidence": 0.99}
- Boundary covers a wall area with a whiteboard mounted on part of it -> {"label": "wall", "label_confidence": 0.85, "mobility_class": "static", "mobility_confidence": 0.99}
- Boundary on a wall plane; an armchair sits in front of it, outside the boundary -> {"label": "wall", "label_confidence": 0.86, "mobility_class": "static", "mobility_confidence": 0.99}
- Boundary on a floor area with a rug on it -> {"label": "floor", "label_confidence": 0.9, "mobility_class": "static", "mobility_confidence": 0.99}

Distinctive infrastructure that fills the boundary - name it:
- Exposed pipes or ductwork running across the ceiling, filling the boundary -> {"label": "pipes", "label_confidence": 0.8, "mobility_class": "static", "mobility_confidence": 0.99}
- A large air vent or diffuser that dominates the enclosed region -> {"label": "air_vent", "label_confidence": 0.82, "mobility_class": "static", "mobility_confidence": 0.99}
- A fire hose reel on the wall, filling the boundary -> {"label": "fire_hose_reel", "label_confidence": 0.85, "mobility_class": "static", "mobility_confidence": 0.99}

Glass and reflective surfaces:
- A framed glass partition dividing two areas -> {"label": "glass_wall", "label_confidence": 0.8, "mobility_class": "static", "mobility_confidence": 0.99}
- A glass-panelled door in a frame -> {"label": "glass_door", "label_confidence": 0.82, "mobility_class": "static", "mobility_confidence": 0.99}
- A reflective rectangular panel showing a mirror image of the room -> {"label": "mirror", "label_confidence": 0.75, "mobility_class": "static", "mobility_confidence": 0.99}
- A plain white rectangular panel on a wall, no reflection -> {"label": "whiteboard", "label_confidence": 0.7, "mobility_class": "static", "mobility_confidence": 0.99}

Furniture vs plain surface:
- An office chair filling most of the frame -> {"label": "office_chair", "label_confidence": 0.9, "mobility_class": "static", "mobility_confidence": 0.99}
- A desk seen clearly, with items on top -> {"label": "desk", "label_confidence": 0.9, "mobility_class": "static", "mobility_confidence": 0.99}
- A flat vertical surface with a visible handle, seam and toe-kick -> {"label": "cabinet", "label_confidence": 0.8, "mobility_class": "static", "mobility_confidence": 0.99}
- A flat vertical surface with no handles, seams or depth cues -> {"label": "wall", "label_confidence": 0.5, "mobility_class": "static", "mobility_confidence": 0.95}

Moving agents:
- A person in the space -> {"label": "person", "label_confidence": 0.93, "mobility_class": "dynamic", "mobility_confidence": 0.98}

Unclear:
- A blurred, dark or truncated region with no identifiable object -> {"label": "VLM_no_result", "label_confidence": 0.0, "mobility_class": "unknown", "mobility_confidence": 0.0}

Return only the JSON object - nothing before "{", nothing after "}". No markdown, no explanation, no numbered analysis.
```

---

## 4. object_detail addition — design goal & V1 (basic)

**Design goal.** `object_detail` should capture the object's *current,
instance-specific visible state* — items on/with it, condition, whether it
looks in use — never a restatement of its identity, material or colour
(those are `label`'s job already). The three worked examples given for this
feature:

| Object (`label`) | Wanted `object_detail` |
|---|---|
| table | "monitor and keyboard on table, organized properly" |
| floor | "water on floor" *or* "carpet covering floor" |
| monitor | "display on" / "powered on" |

**V1 — basic addition (this version).** Deliberately minimal: one short
appended section, five worked examples covering the three given cases plus
one explicit "nothing notable" fallback, a word-count cap, and — necessarily
— a restated `OUTPUT`/`GOOD` example line so the JSON schema itself changes
from 4 keys to 5 (see §14 for why this restatement could not be avoided even
in a "minimal" version, and the conflict it knowingly creates with the
frozen prompt's own four-key `OUTPUT`/`GOOD` lines above it).

Full text appended after the baseline's final line (`"Return only the JSON
object..."`):

```text
ADDITIONAL FIELD - object_detail (experimental addition, V1_basic_addition)

Everything above is unchanged: still decide "label", "label_confidence",
"mobility_class" and "mobility_confidence" exactly as instructed.

Now also add a fifth key, "object_detail": a short phrase describing the
object's CURRENT VISIBLE STATE - not its identity, material or colour (the
label already covers that). Describe what is different about this specific
instance right now: items on or with it, its condition, or whether it looks
active or in use.

Examples:
- a table with a monitor and keyboard on it, tidily arranged -> "monitor and keyboard on table, organized properly"
- a floor with a wet patch -> "water on floor"
- a floor covered by a rug -> "carpet covering floor"
- a monitor showing an image -> "display on"
- a monitor with a black, blank screen -> "display off"
- nothing notable to report -> "no notable detail"

Keep object_detail to 10 words or fewer. Do not repeat the label inside it.

OUTPUT - exactly one JSON object, with nothing before it or after it:
{"label": "<label>", "label_confidence": <0-1>, "mobility_class": "<static|dynamic|unknown>", "mobility_confidence": <0-1>, "object_detail": "<short phrase>"}

GOOD - this is the whole reply:
{"label": "table", "label_confidence": 0.9, "mobility_class": "static", "mobility_confidence": 0.99, "object_detail": "monitor and keyboard on table, organized properly"}
```

The complete, ready-to-paste V1 prompt (baseline + this section) lives in
`object_detail_prompts_under_test.yaml` under `V1_basic_addition` and is
currently `active: true` there.

---

## 5. Manual grading rubric for `object_detail` (`manual_object_detail_rating`)

Separate from the label experiment's `error_category` vocabulary (§9 of
`prompt_optimisation_experiment/EXPERIMENT_REPORT.md`), which is **still
used unchanged** in the `error_category` column here to catch label/mobility
regressions. This new vocabulary grades the *new* field only:

| Token | Use when |
|---|---|
| `accurate` | describes a real, specific, visible state usefully |
| `plausible_unverifiable` | reasonable given the crop, but ground truth can't be confirmed from the image alone |
| `generic` | technically not wrong but conveys no information (e.g. always "no notable detail" when something *is* visible) |
| `wrong` | describes a state/detail that is not actually present |
| `identity_repeat` | just restates the label/material/colour instead of a state — a direct instruction violation |
| `missing` | `object_detail` key absent from the parsed JSON (validator fell back to `DEFAULT_OBJECT_DETAIL`) |
| `format_break` | **critical** — including this field broke JSON parsing, or corrupted/dropped `label`/`mobility_class` (a regression on the frozen prompt; if this appears even once, stop and treat as a V1 failure, do not just tally it) |
| `crop_quality` | crop too blurred/truncated/dark to judge — excluded from the object_detail accuracy denominator |
| `other` | anything else — explain in `manual_object_detail_notes` |

`object_detail usefulness rate` = `count(accurate) / count(verified rows
excluding crop_quality)`.

---

## 6. Fixed experimental conditions

Unlike the label-prompt experiment's 3-model × 3-prompt matrix, this
experiment **holds the model fixed** at the production winner and only
varies the appended `object_detail` section (V1, V2, …) — the label-prompt
choice and model choice are both closed questions here.

| Condition | Value | Where |
|---|---|---|
| Model profile | `qwen3_5_4b_q4` (the production winner — unchanged) | `phase1.vlm.active_profile` |
| Dataset | same as the label experiment — `uHumans2_office_s1_00h_ros2` (or current production bag; record whichever is actually used per run in §11) | — |
| RAP | production setting (currently disabled per `update_memory_from_vlm: false` / RAP memory clear) — every crop the VLM sees is a fresh call, so `object_detail` is generated fresh, not read back from a stale RAP entry, while grading | `rsg_pipeline.yaml` |
| max_tokens | **512**, matching the frozen prompt's own R6 setting — watch for truncation once the 5th key adds length to every response; raise (and record in §15) if truncated JSON appears | `phase1.vlm.max_tokens` |
| temperature | `0.0` | `phase1.vlm` |
| Crop format | **png** (lossless) — see §7; distinguishes this experiment's diagnostics from the label experiment's jpg crops | `phase1.vlm.object_detail_experiment.crop_format` |
| result_validation | unchanged (`min_label_confidence: 0.80`, `min_mobility_confidence: 0.70`) — validation does not yet gate on `object_detail` at all; a missing/blank value never rejects the result | `phase1.vlm.result_validation` |
| Sample target | ~30–50 verified objects per version (smaller than the label experiment's 50 — this is a single-model iteration loop, not a matrix; raise if early rows are inconclusive) | verification step |

If any of these changes between versions, record it in §15 and note which
versions it does/doesn't apply to.

---

## 7. Diagnostic infrastructure (crops + CSV) — what was built and why

Reused and extended the **existing** `VLMTestDiagnostics` class
(`src/rsg/nodes/support/phase1/vlm_test_diagnostics.py`) rather than building
a parallel one — it already did exactly what this experiment needs (save the
exact crop the VLM saw + append one CSV row per call), and is already wired
into `phase1.py`'s VLM worker loop. Building a second, separate mechanism
would have meant two divergent logging paths for the same underlying event.
Changes made:

1. **`vlm_test_diagnostics.py`**
   - `CSV_HEADERS` gained `object_detail` (pipeline-filled, straight from the
     VLM result dict) and two human-filled columns, `manual_object_detail_rating`
     and `manual_object_detail_notes` (§5's rubric).
   - `log_vlm_result` now writes `vlm_output.get("object_detail", "")` into
     the new column — no caller change needed, since `phase1.py`'s call site
     already passes the whole VLM result dict, which has carried
     `object_detail` since the pipeline-compatibility work (§2).
   - New `crop_format` constructor parameter (`"jpg"` default, unchanged for
     the label experiment; `"png"` for this one). Crop filenames become
     `obj_NNNNNN_crop.png`; `cv2.imwrite` picks the codec from the extension
     automatically.

2. **`phase1_config.py`** — new config block mirroring the existing
   `vlm_prompt_opt_*` fields exactly, but under its own names so the frozen
   label experiment's config is never touched: `vlm_object_detail_experiment_enabled`,
   `_run_id`, `_prompt_version`, `_output_root`, `_crop_format` (default
   `"png"`), loaded from a new `phase1.vlm.object_detail_experiment` YAML
   block.

3. **`rsg_pipeline.yaml`** — added that `object_detail_experiment:` block
   right after the existing (frozen) `prompt_optimisation:` block, disabled
   by default (`enabled: false`), with `output_root` pointing at
   `debug/object_detail_prompt_experiment/runs`.

4. **`phase1.py`** — the diagnostics-directory selection (previously: "if
   the label experiment is enabled, redirect output there, else use the
   default `VLM-Test-Session/`") now checks `object_detail_experiment`
   **first**, falls back to the (frozen, currently disabled) label
   experiment's flag, and only then the default. Whichever branch is taken
   also decides the `crop_format` passed to `VLMTestDiagnostics` (`png` for
   the object_detail branch, `jpg` for the other two). The two experiment
   toggles are mutually exclusive in practice — both being `true`
   simultaneously has defined behaviour (object_detail wins) but is not a
   configuration anyone should actually use.

Output layout produced by a real run (once `enabled: true`):

```
debug/object_detail_prompt_experiment/runs/<run_id>/session_<YYYYmmdd_HHMMSS>/
    crops/obj_NNNNNN_crop.png     <- exact RGB crop the VLM was given (lossless)
    vlm_results.csv               <- one row per VLM call, schema in §8
```

`runs/*/session_*/` is git-ignored (added to `.gitignore` alongside the
label experiment's equivalent rule) so scratch/smoke sessions don't clutter
the tree; a completed, annotated session is committed deliberately with
`git add -f <session_dir>` and recorded in §11/§15, exactly like the label
experiment's convention.

---

## 8. Data layout & CSV schema

```
debug/object_detail_prompt_experiment/
├── EXPERIMENT_REPORT.md                    <- this file (the only main document)
├── object_detail_prompts_under_test.yaml   <- BASELINE_FROZEN + versioned additions, loadable
├── vlm_results_TEMPLATE.csv                <- header-only template (generated from CSV_HEADERS)
└── runs/
    └── V{n}__{model}__{addition_name}/
        ├── crops/               <- scaffold placeholder (.gitkeep)
        ├── vlm_results.csv      <- scaffold placeholder (header only)
        └── session_<YYYYmmdd_HHMMSS>/   <- one per pipeline start
            ├── crops/           <- every crop PNG for that session
            └── vlm_results.csv  <- one row per VLM call
```

`vlm_results.csv` columns (generated from the actual `CSV_HEADERS` constant
so this table can never silently drift from the code):

| Column | Filled by | Meaning |
|---|---|---|
| `object_id` | pipeline | unique object/track id |
| `crop_filename` | pipeline | PNG in `crops/` |
| `frame_timestamp` | pipeline | bag/wall time of the crop |
| `run_id` | pipeline/manual | e.g. `V1__qwen35_4b__basic_addition` |
| `model_profile` | pipeline/manual | `qwen3_5_4b_q4` (held fixed — see §6) |
| `prompt_version` | pipeline/manual | e.g. `V1_basic_addition` |
| `vlm_label` | pipeline | predicted label (must not regress — grade with `error_category` below) |
| `label_confidence` | pipeline | 0–1 |
| `mobility_class` | pipeline | static \| dynamic \| unknown |
| `mobility_confidence` | pipeline | 0–1 |
| **`object_detail`** | pipeline | **new** — the field under test |
| `vlm_inference_ms` | pipeline | model compute time |
| `end_to_end_ms` | pipeline | client `perf_counter` around the whole HTTP call |
| `success` | pipeline | VLM call + JSON parse ok |
| `validation_status` | pipeline | accepted \| rejected (`result_validation`; does not yet consider `object_detail`) |
| `raw_response` | pipeline | raw model text, trimmed to ~200 chars — the only place to see whether the model actually emitted 5 keys with real spaces before normalisation |
| `manual_label` | **human** | ground truth (regression check — same meaning as the label experiment) |
| `manual_is_correct` | **human** | true \| false \| blank |
| `error_category` | **human** | one token from the label experiment's §9 vocabulary — **watch specifically for `format_break`-style effects on the frozen label**, i.e. did adding a 5th key make the *label* worse, not just judge object_detail in isolation |
| `manual_notes` | **human** | free text |
| **`manual_object_detail_rating`** | **human** | one token from §5 |
| **`manual_object_detail_notes`** | **human** | what's wrong / notable about `object_detail` specifically |

---

## 9. Procedure (per version)

Mirrors `debug/prompt_optimisation_experiment/EXPERIMENT_REPORT.md` §6,
adapted for a single-model iteration loop instead of a model×prompt matrix.

**1. Wire the version in.** In
`object_detail_prompts_under_test.yaml`, set `active: true` on the version to
test (and `false` on the others). Copy its full `template:` text verbatim
into `phase1.vlm.prompt` in `src/rsg/config/rsg_pipeline.yaml`. Rebuild:
```bash
cd ~/Thesis_pipeline_split_lean
colcon build --packages-select rsg --symlink-install
source install/setup.bash
```

**2. Enable the experiment routing.** In `rsg_pipeline.yaml`, under
`phase1.vlm.object_detail_experiment`:
```yaml
enabled: true
run_id: V1__qwen35_4b__basic_addition
prompt_version: V1_basic_addition
```
(`output_root` and `crop_format` are already set correctly by default.)
Confirm `phase1.vlm.prompt_optimisation.enabled` stays `false` — the two
routes are mutually exclusive in practice (§7).

**3. Clear memory** (fresh tracker/RAP state, so nothing from a previous
session leaks into this run's crops or grading):
```bash
python3 ~/Thesis_pipeline_split_lean/clear_memory.py
```

**4. Run** the pipeline against the chosen bag/live feed for long enough to
collect the sample target (§6). Every VLM call auto-logs a PNG crop and a
CSV row under
`debug/object_detail_prompt_experiment/runs/V1__qwen35_4b__basic_addition/session_<timestamp>/`.

**5. Archive**: leave the session where it landed (already under the
experiment's `runs/<run_id>/` — no copy step needed, unlike the label
experiment, since `output_root` already points here). Commit deliberately
with `git add -f <session_dir>` once annotated.

**6. Manually grade every verified row**, filling:
- `manual_label`, `manual_is_correct`, `error_category`, `manual_notes` —
  same meaning as the label experiment (regression check).
- `manual_object_detail_rating`, `manual_object_detail_notes` — §5's rubric.

**7. Fill this version's block in §11** (usefulness rate, error tally,
label-regression check, 3–6 bullet observations) and **§15 Changelog**.

**8. Decide the next version.** Following the label experiment's own
methodology exactly: the next version is written *against this version's
actual failures*, not designed speculatively in advance. Update §4 with the
new version's rationale once §11 has real numbers to react to.

**9. Turn the experiment routing back off** (`object_detail_experiment.enabled:
false`) once done, so normal production runs don't write into the experiment
folder.

---

## 10. Metrics (compute per version)

- **object_detail usefulness rate** = `count(manual_object_detail_rating ==
  accurate) / count(verified rows excluding crop_quality)` (§5).
- **object_detail error tally** = count per rating token (absolute + % of
  verified).
- **Label regression check** = label accuracy on this version's rows, using
  the *same formula* as the frozen experiment (§10 there): `count(manual_is_correct
  == true) / count(verified excluding crop_quality)`. Compare directly
  against the frozen prompt's **88 %** (R6, `qwen3_5_4b_q4` × P3). A material
  drop means the appended section is interfering with the frozen prompt and
  the addition needs rework before anything else.
- **Format-break rate** = `count(error_category == format_break OR
  manual_object_detail_rating == format_break) / count(all VLM calls)`.
  Should be ~0; investigate immediately if not (see §5's note on this
  rating).
- **Latency delta** = median `vlm_inference_ms` this version vs the frozen
  R6 baseline (2942 ms) — a longer prompt + a longer expected response can
  measurably slow inference; worth knowing even though V1 is not chasing
  speed.
- 95 % CI on accuracy at N≈30–50 is wide (roughly ±13–18 pp) — as in the
  label experiment, treat small differences between versions as noise and
  prefer qualitative failure-category shifts as the finer signal.

---

## 11. Run matrix

| Version | Model | Addition | Folder | Status | Verified N | Label accuracy | object_detail usefulness | format_break count |
|---|---|---|---|---|---|---|---|---|
| V1 | qwen35_4b | basic_addition | `runs/V1__qwen35_4b__basic_addition/session_<ts>/` | ◐ activated, awaiting graded run | — | — | — | — |

Add V2+ rows here as they're wired, following the label experiment's
convention (§7 there) exactly.

---

## 12. Per-version results

*(Fill one block per version, once run. Keep the prose to observations the
CSV cannot show on its own — this is exactly the material for the thesis
experiment chapter.)*

### V1 — qwen35_4b × basic_addition — ◐ ACTIVATED, AWAITING GRADED RUN

Activated 2026-09-18 in `rsg_pipeline.yaml` (prompt + `object_detail_experiment`
flag) and rebuilt. An initial ungraded test run already showed the pipeline
producing *something* other than the removed dummy placeholder (see §15) —
this section will be filled once a full session is collected and manually
graded per §9/§5.

---

## 13. Findings & recommendation

*(Fill once at least one version has real, graded data.)*

---

## 14. Deviations & risks

- **The appended `OUTPUT`/`GOOD` restatement is a genuine, foreseeable
  conflict, shipped deliberately in V1.** The frozen baseline already ends
  with an explicit `OUTPUT - exactly one JSON object...` template and a
  `GOOD - this is the whole reply:` example, both showing exactly four keys,
  and the prior experiment's own changelog (R6, format-hardening) showed
  this exact model family follows such literal templates/examples closely.
  V1's appended section repeats both lines with a fifth key. Two plausible
  failure modes to watch for specifically when grading V1: (a) the model
  ignores the appended section and keeps emitting only 4 keys (`object_detail`
  falls back to `DEFAULT_OBJECT_DETAIL` every time — rate this `missing` in
  §5, not `format_break`); (b) the *earlier*, four-key example wins and the
  model produces malformed JSON by trying to satisfy both templates at once
  (`format_break` — critical, see §5/§10). This is not treated as a bug to
  silently avoid by, say, editing the frozen baseline's own `OUTPUT`/`GOOD`
  lines — that would violate constraint 1 in §1. If V1 fails this way, the
  fix belongs in V2 (e.g. moving the restated schema earlier, or a stronger
  "supersedes the schema above" framing) and must be logged in §15 as a
  mistake-and-rework, exactly as instructed for this document.
- **`max_tokens: 512` may be tight once every response carries a 5th key and
  a short phrase.** The frozen prompt's own changelog shows this same model
  needed headroom raised twice (96→256→512) purely from response length
  growing with prompt complexity. Watch the first few `raw_response` values
  in V1 for truncated JSON before trusting a whole session's numbers.
- **Single model, single scene (whatever bag/scene is used per §6) —**
  `object_detail` usefulness will be skewed by whichever object types
  dominate that scene, same caveat as the label experiment's §14.
- **`result_validation` does not gate on `object_detail` at all** — a
  missing, empty, or nonsense `object_detail` never causes the whole VLM
  result to be rejected (by design: this field is advisory/descriptive, not
  safety- or identity-critical the way `label` is). This means
  `validation_status == accepted` rows can still have `manual_object_detail_rating
  == missing` or `wrong` — grade the two independently, don't assume one
  implies the other.
- **RAP-sourced `object_detail` is a separate, already-wired path (§2) not
  exercised by this experiment** — RAP is disabled per the current
  production config (`update_memory_from_vlm: false`), so every crop in a
  V1 session gets a fresh VLM call and a fresh `object_detail`, never a
  value read back from RAP memory. If RAP is re-enabled before running a
  later version, note it in §15 — rows whose `object_detail` came from RAP
  recall rather than this session's VLM call should be excluded from the
  usefulness-rate denominator, since they don't test the prompt at all.

---

## 15. Changelog

| Date | Change |
|---|---|
| 2026-09-18 | **Scaffold created.** Folder tree (`runs/V1__qwen35_4b__basic_addition/{crops/.gitkeep,vlm_results.csv}`), `object_detail_prompts_under_test.yaml` (`BASELINE_FROZEN` verified byte-identical to the live `rsg_pipeline.yaml` prompt via the YAML parser; `V1_basic_addition` = baseline + appended section, `active: true`), `vlm_results_TEMPLATE.csv` (generated from `CSV_HEADERS`, not hand-typed, so it can't drift from the code), this report. Diagnostic infrastructure extended rather than duplicated: `vlm_test_diagnostics.py` gained the `object_detail` column, two manual-grading columns (§5), and a `crop_format` parameter (`png` for this experiment, `jpg` preserved as the label experiment's default so its own frozen sessions are unaffected); `phase1_config.py` gained a parallel `vlm_object_detail_experiment_*` config block (mirrors but does not touch the existing `vlm_prompt_opt_*` fields); `rsg_pipeline.yaml` gained the matching `object_detail_experiment:` block, disabled by default; `phase1.py`'s diagnostics-directory selection now checks it first, ahead of the (frozen) label experiment's own flag. `.gitignore` extended with the matching `runs/*/session_*/` rule. V1 not yet run — the `OUTPUT`/`GOOD`-restatement conflict (§14) is flagged *before* the first run specifically so it's judged as a predicted risk being tested, not a surprise discovered after the fact. |
| 2026-09-18 | **First live test run — dummy placeholder discovered, not a bug in the field itself.** The pipeline-compatibility work (before this experiment folder existed) shipped `DEFAULT_OBJECT_DETAIL = "This_is_a_sample_sentence"` as a fallback for when the prompt doesn't request the field yet. A test run against the still-unmodified baseline prompt (V1 not yet wired at that point) correctly showed that placeholder on every object in RViz — expected, but easy to mistake for a broken feature since it looks identical on every object regardless of what the VLM actually said. **Fix:** `DEFAULT_OBJECT_DETAIL` changed from a placeholder sentence to `""` (empty) in `vlm_result.py` — no fuser.cpp change needed, since `objectDisplayLabel()` already only appends the object_detail line when non-empty (written defensively in the original pipeline-compatibility pass). Effect: an object with no real `object_detail` now simply shows no extra line, instead of a fake one. Pure Python constant, symlinked install, live immediately without a rebuild. |
| 2026-09-18 | **V1 activated.** `phase1.vlm.prompt` in `rsg_pipeline.yaml` replaced with `object_detail_prompts_under_test.yaml`'s `V1_basic_addition` template (byte-verified equal via the YAML parser: `BASELINE_FROZEN` text + the §4 addition, 9109 chars). `phase1.vlm.object_detail_experiment` flipped to `enabled: true`, `run_id: V1__qwen35_4b__basic_addition`, `prompt_version: V1_basic_addition` (`prompt_optimisation.enabled` confirmed still `false` — the two routes stay mutually exclusive per §7). Rebuilt (`colcon build --packages-select rsg --symlink-install`). Memory was **not** cleared as part of this activation — do that (`python3 clear_memory.py`) before the graded run in §9 step 3, so old dummy-era tracker/RAP state doesn't leak into V1's numbers. §11/§12 status updated to "activated, awaiting graded run"; still pending: a full session collected and manually graded end-to-end per §5/§9/§10. |
