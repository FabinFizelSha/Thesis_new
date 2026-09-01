# Prompt Optimisation Experiment — VLM Object Classification (Fresh Run)

**Single source of truth for this experiment. Do not create additional main documents — append to this file.**

- **Started:** 2026-08-31
- **Author:** Fabin Fizel Sha
- **Supersedes:** the frozen 2026-08-30 campaign in
  `debug/VLM-PROMPT-OPTIMIZATION-DOCUMENTATION/` (Qwen3-VL-8B only, uneven
  methodology — see §2).
- **Status:** SCAFFOLD READY — 0 / 9 runs complete.

---

## 1. Objective

Measure how VLM object-classification quality and inference latency vary across
**3 prompt formulations × 3 model profiles** on the RSG Phase 1 pipeline, under a
controlled VLM-only setting, with a consistent methodology (fixed run length,
fixed sample target, mandatory latency logging, manual verification with typed
error categories).

Primary questions:

1. Does the newer **Qwen 3.5** family (4B, 9B) break the ~50 % accuracy ceiling
   and the persistent `ceiling` → `ceiling_light` visual-salience error that
   Qwen3-VL-8B could not escape through prompt engineering alone?
2. Which of the 3 prompts is best **per model** (the best prompt may differ by
   model)?
3. What is the accuracy ↔ latency trade-off across the 3 models (4B Q4 vs 9B Q4
   vs 8B FP16)?

Out of scope: RAP retrieval quality, crop-scoring/quality gate tuning (frozen —
see `debug/CROP_SCORING_DOCUMENTATION/`), Hydra fusion.

---

## 2. Relationship to the prior campaign

The 2026-08-30 campaign tested 6 prompts on **Qwen3-VL-8B-Instruct FP16** and
concluded prompt tuning plateaued at ~46–52 % accuracy, dominated by the
`ceiling_light` error. Its weaknesses, which this experiment fixes:

| Prior weakness | Fix here |
|---|---|
| Non-deterministic crop set — every prompt saw different crops | Same 700 s bag, same pipeline config, same slow replay rate for all 9 runs; sample target fixed at ~50 |
| Tiny/uneven N (30–55), v4 only 30/58 verified | Fixed target **~50 verified samples per run**, verify all or mark `crop_quality` exclusion explicitly |
| Latency logged inconsistently, absent for v4–v6 | **Mandatory** `vlm_inference_ms` + `end_to_end_ms` every row |
| Free-text error notes, not tallyable | Typed `error_category` vocabulary (§9) + free-text `manual_notes` |
| Raw CSVs / crops not archived | **Per-run `crops/` + `vlm_results.csv` committed** under `runs/` |
| v2/v3/v5 prompts never stored verbatim | All 3 prompts frozen in `prompts_under_test.yaml` and §4 below |

Prompts carried forward (chosen 2026-08-31) as a **complexity gradient**, all
re-worked to be **open-vocabulary** (no enumerated label lists — see §4):

- **P1** — from v1_simplified: short, zero-shot.
- **P2** — from v5_examples_based: minimal rules + many worked examples.
- **P3** — from v6_structural_priority: full reasoning strategy + confidence
  calibration + grouped examples.

v2 (marginal), v3 (regressed −9.3 %), v4 (unverifiable) are dropped.

Prior Qwen3-VL-8B numbers, for the source prompts, reference only (**not**
directly comparable — the prompts were re-worded, label lists removed, and the
boundary colour differs; see §4 and §14):

| Source prompt | Prior 8B accuracy | Prior `ceiling_light` errors |
|---|---|---|
| v1_simplified (→ P1) | 46.9 % (23/49) | 9 |
| v5_examples_based (→ P2) | 51.8 % (N not recorded) | 16 |
| v6_structural_priority (→ P3) | 50.0 % (N not recorded) | 14 |

---

## 3. Models under test

One local llama.cpp OpenAI-compatible server on `http://127.0.0.1:8000/v1/chat/completions`.
**Only one model loaded at a time.** Profile is selected in
`src/rsg/config/rsg_pipeline.yaml` → `phase1.vlm.active_profile`, then rebuild.

| Short name | `active_profile` | Weights | Quant | Notes |
|---|---|---|---|---|
| **qwen3vl8b** | `qwen3_vl_8b_f16` | Qwen3-VL-8B-Instruct + FP16 vision projector | FP16 | Prior campaign model. Needs ≥32 GB; do not run on 16 GB Orin. |
| **qwen35_4b** | `qwen3_5_4b_q4` | Qwen3.5-VL-4B-Instruct | Q4_K_M | `~/rsg_models/qwen3_5_4b/Qwen3.5-4B-Q4_K_M.gguf` + `mmproj-BF16.gguf` |
| **qwen35_9b** | `qwen3_5_9b_q4` | Qwen3.5-VL-9B-Instruct | Q4_K_M | `~/rsg_models/qwen3_5_9b/Qwen3.5-9B-Q4_K_M.gguf` + `mmproj-BF16.gguf` |

Shared VLM request settings (do not change between runs): `temperature: 0.0`,
`max_tokens: 96`, `jpeg_quality: 100`, `timeout_sec: 60`.

---

## 4. Prompts under test

Authoritative full text: [`prompts_under_test.yaml`](prompts_under_test.yaml).
The blocks below are the same text; run from the YAML.

**Design of the set:**

- **Complexity gradient** — the one deliberate axis: P1 minimal / zero-shot →
  P2 minimal rules + many worked examples → P3 full reasoning strategy +
  confidence calibration + grouped examples.
- **No enumerated label lists** in any prompt. Every prompt tells the model to
  name whatever the object is with a concise lowercase label and states "there
  is no fixed list of labels". Restricted vocabularies were shown to hurt
  generalisation in the prior campaign; this experiment keeps all three prompts
  open-vocabulary so the comparison is about *strategy*, not *vocabulary*.
- **JSON output** — all three give the exact JSON schema **and** end with
  "Return only the JSON object. No markdown…". Output-format compliance is held
  constant so it is not a confound.
- **Shared core** — identical cyan-boundary reference, identical JSON schema
  string, identical mobility definitions, `VLM_no_result` + 0.0 fallback.
- Examples in P2/P3 are labelled *"not a list of allowed labels"* and both
  include one `dynamic` (person) example so mobility has at least one exemplar.
- P1 is derived from v1_simplified (label enumeration removed, JSON-only line
  added); P2 from v5_examples_based (label list removed, examples broadened);
  P3 from the live v6_structural_priority (structural-label enumeration
  generalised to a description, dynamic example added). None is byte-verbatim.

> **Decision (2026-08-31):** boundary colour is **cyan**, as in the original
> 2026-08-30 campaign. All 3 prompts refer to a "cyan boundary".
>
> **Required pipeline change before run R1:** the live contour is currently
> white. Set it to cyan so the rendered contour matches the prompt wording:
> ```yaml
> # src/rsg/config/rsg_pipeline.yaml  ->  phase1.semantic_crop
> target_contour_rgb: [0, 255, 255]        # was [255, 255, 255]
> ```
> Rebuild `rsg` after the edit. Keep cyan fixed for all 9 runs.

### P1 — simple (short, zero-shot, general)

```text
Classify the single SAM-segmented target in this indoor crop.

The target object is the one outlined by the cyan boundary. Identify only this object. The surrounding area may be used as context to help identify it.

Assume an indoor setting.

Name the object with a concise, singular, lowercase snake_case label that best describes what it actually is. There is no fixed list of labels. Do not describe colour, material, position, condition or activity. Use a confidence above 0.90 only when the identification is unmistakable. If the target is unclear, truncated or unidentifiable, use the label VLM_no_result with confidence 0.0.

Mobility:
- "dynamic": a human, an animal, or a self-propelled robot only
- "static": everything else
- "unknown": when the label is unknown or mobility cannot be determined

Output exactly one JSON object:
{"label": "<label>", "label_confidence": <0-1>, "mobility_class": "<static|dynamic|unknown>", "mobility_confidence": <0-1>}

Return only the JSON object. No markdown, explanations, or extra text.
```

### P2 — example-driven (boundary-focus + generalisation, reworked against R1)

Reworked after R1 (P1 @ 48 %). Targets the three R1 failure clusters: boundary
violations (5), sub-part-instead-of-surface (8), and 0.95-confidence hallucination
on ambiguous crops (subset of 13 `wrong_class`). Still open-vocabulary — no label list.

```text
Identify the object that the cyan boundary encloses in this indoor crop.

WHICH OBJECT TO NAME
- The cyan boundary marks one region of the image. Name only what lies inside that boundary.
- Everything outside the boundary is background. A large, obvious object outside the boundary - a chair, a shelf, a bookcase, a whiteboard, a plant - is NOT the answer. Use the surroundings only to understand what the enclosed region is, never as the label itself.
- First decide what the enclosed region mainly shows, then name that.

HOW TO CHOOSE THE LABEL
- Name the largest, most complete thing the boundary covers. If the boundary encloses only part of a bigger structure, name the bigger structure.
- If the boundary spans many repeated units of one surface - several ceiling tiles or panels, a run of floor, a stretch of wall - name the surface itself ("ceiling", "floor", "wall"), not a single unit ("ceiling_tile").
- If the boundary covers a surface that carries a smaller mounted or attached item - a wall bearing a whiteboard, sign, monitor or vent; a ceiling holding a light or diffuser; a floor with a rug on it - name the surface, not the attached item. Name the attached item only when it clearly fills the boundary on its own.
- When you are torn between a specific object and the surface or structure behind it, choose the surface or structure.
- Use a concise, singular, lowercase label that best fits what you actually see. There is no fixed list of labels - name whatever it is.

CONFIDENCE ("label_confidence") - be honest, do not default to 0.95
- 0.90-1.00: unmistakable; the enclosed region can only be this.
- 0.60-0.85: probable, but a competing reading exists.
- 0.30-0.55: genuinely ambiguous, low detail, or a guess.
- 0.0 with label VLM_no_result: too blurred, dark, or truncated to identify at all.
Do not report 0.90+ on a crop you would not bet on.

Mobility:
- "dynamic": a human, an animal, or a self-propelled robot only
- "static": everything else
- "unknown": when the label is unknown or mobility cannot be determined

Output exactly one JSON object:
{"label": "<label>", "label_confidence": <0-1>, "mobility_class": "<static|dynamic|unknown>", "mobility_confidence": <0-1>}

WORKED EXAMPLES (these show the reasoning, not a list of allowed labels):
- Boundary on a wall plane; an armchair sits in front of it, outside the boundary -> {"label": "wall", ...}
- Boundary on the floor; a shelf unit stands beyond it, outside the boundary -> {"label": "floor", ...}
- Boundary spans a panelled ceiling covering many tiles and a light fixture -> {"label": "ceiling", ...}
- Boundary covers a wall area with a whiteboard mounted on part of it -> {"label": "wall", ...}
- Boundary covers a single whiteboard that fills almost the whole region -> {"label": "whiteboard", ...}
- Boundary on a flat vertical surface with little detail - cabinet front / pillar / wall -> {"label": "wall", "label_confidence": 0.4, ...}
- Boundary shows a reflective rectangular panel on a wall - mirror or whiteboard -> {"label": "wall", "label_confidence": 0.4, ...}
- Recessed light set into a ceiling panel -> {"label": "ceiling", ...}
- Floor with a rug on it -> {"label": "floor", ...}
- Potted plant that fills the region -> {"label": "potted_plant", ...}
- Door and its frame -> {"label": "door", ...}
- Person standing in the space -> {"label": "person", "mobility_class": "dynamic", ...}
- Blurred, dark or cut-off region with no identifiable object -> {"label": "VLM_no_result", ...}

Return only the JSON object. No markdown, explanations, or extra text.
```

(Full example JSON with confidences is in `prompts_under_test.yaml`.)

### P3 — detailed (structural priority + calibration + grouped examples, tuned on R1+R2)

Built on P2 (R2 @ 72 %). Keeps P2's wins (boundary discipline; generalise
repeated surface units; surface over mounted fixture) and adds the three R2
fixes: (1) **distinctive infrastructure that fills the boundary is named, not
generalised** — pipes / vents / ducts → `ceiling` was wrong 4×; (2) **glass
partitions / glass-panelled doors get a glass label** — 5 R2 misses; (3) a
**discriminative cue for the flat-surface cabinet-vs-wall/pillar** case (handle /
seam / toe-kick → furniture; none → wall/pillar) with a wider honest confidence
band. Still open-vocabulary — no label list.

```text
This is an indoor object-classification task. Identify the object that the cyan boundary encloses.

APPROACH
1. Look only inside the cyan boundary. Everything outside it is background - never name it, however large or obvious.
2. Decide what the enclosed region mainly shows.
3. Choose the label using the priority order below.

PRIORITY ORDER
a. Room-defining structure over incidental detail. Large room planes + their openings/supports outrank small things on them. Many repeated units of one surface (ceiling tiles, a run of floor, a stretch of wall) -> name the surface ("ceiling"/"wall"/"floor"), not one unit.
b. BUT distinctive infrastructure that fills the boundary is named, not generalised. Exposed pipes, ductwork, a large vent/diffuser, a fire hose reel, a big fixture dominating the region -> name that thing. Collapse to the surface only when the fixture is small/incidental.
c. Furniture over objects on it - but only with a furniture cue. A flat vertical surface with no handle, seam, gap, toe-kick or visible depth is a wall or pillar, not a cabinet.
d. When readings compete, choose the larger, more dominant, more structural one.

MATERIAL AND AMBIGUITY CUES
- Transparent/translucent panel, framed glass screen, glazed partition, glass-panelled door -> name it as glass ("glass_wall", "glass_partition", "glass_door").
- Judge transparency directly: if the room / objects / people / light behind the panel are visible through it - even dimly, even with a frame or faint reflection - it is glass; a panel you cannot see through at all is solid.
- Reflects the room / shows a mirror image -> mirror. Plain matte white or lightly-marked panel -> whiteboard. Can't tell -> name the wall it sits on, low confidence.
- Low detail / shadow / blur / heavy truncation -> structural reading (wall/floor/ceiling) at low confidence, or VLM_no_result.

LABEL FORM
- Concise, singular, lowercase; no fixed list of labels; no colour/condition/activity.

CONFIDENCE CALIBRATION for "label_confidence" - report it honestly
- 0.90-1.00: unmistakable.
- 0.70-0.90: clear, minor uncertainty.
- 0.45-0.70: plausible, competing reading exists - the right band for cabinet-vs-wall and mirror-vs-whiteboard.
- 0.20-0.45: genuinely ambiguous, or a guess.
- 0.0 with VLM_no_result: incomprehensible crop.
A low confidence on a hard crop beats a confident wrong answer. Do not default to 0.90+.

Mobility:
- "dynamic": only a human, an animal, or a self-propelled moving object
- "static": any stationary object, including all surfaces, structures, furniture, glass and fixtures
- "unknown": when the label is unknown

Output format (required) - exactly one JSON object:
{"label": "<label>", "label_confidence": <0-1>, "mobility_class": "<static|dynamic|unknown>", "mobility_confidence": <0-1>}

WORKED EXAMPLES (illustrating the approach, not a list of allowed labels):

Room-defining structure over incidental detail:
- Panelled ceiling, many tiles + a recessed light -> {"label": "ceiling", ...}
- Wall area with a whiteboard mounted on part of it -> {"label": "wall", ...}
- Wall plane; an armchair in front of it, outside the boundary -> {"label": "wall", ...}
- Floor area with a rug on it -> {"label": "floor", ...}

Distinctive infrastructure that fills the boundary - name it:
- Exposed pipes / ductwork across the ceiling, filling the boundary -> {"label": "pipes", ...}
- A large air vent / diffuser dominating the region -> {"label": "air_vent", ...}
- A fire hose reel on the wall, filling the boundary -> {"label": "fire_hose_reel", ...}

Glass and reflective surfaces:
- A framed glass partition dividing two areas -> {"label": "glass_wall", ...}
- A glass-panelled door in a frame -> {"label": "glass_door", ...}
- A panel showing a mirror image of the room -> {"label": "mirror", ...}
- A plain white panel on a wall, no reflection -> {"label": "whiteboard", ...}

Furniture vs plain surface:
- Office chair filling the frame -> {"label": "office_chair", ...}
- Desk seen clearly, items on top -> {"label": "desk", ...}
- Flat vertical surface with a visible handle, seam and toe-kick -> {"label": "cabinet", ...}
- Flat vertical surface with no handles, seams or depth cues -> {"label": "wall", "label_confidence": 0.5, ...}

Moving agents:
- A person in the space -> {"label": "person", "mobility_class": "dynamic", ...}

Unclear:
- Blurred / dark / truncated, nothing identifiable -> {"label": "VLM_no_result", ...}

Return only the JSON object. No markdown, explanations, or additional text.
```

(Full text with all example JSON is in `prompts_under_test.yaml`.)

---

## 5. Fixed experimental conditions

Identical for all 9 runs. If any of these changes mid-experiment, record it in
§15 Changelog and re-run affected rows.

| Condition | Value | Where |
|---|---|---|
| Dataset | `~/datasets/uhumans2/uHumans2_office_s1_00h_ros2` (ROS 2 bag, converted from `uHumans2_office_s1_00h.bag`) | — |
| RAP | **disabled** (`phase1.rap.enabled: false` or experiment flag) — every crop goes to the VLM | `rsg_pipeline.yaml` |
| VLM | enabled, `mode: qwen_http`, `async: true` | `rsg_pipeline.yaml` |
| Run length | **700 s wall time** per run (hard cap) | manual timer / `timeout 700s` |
| Sample target | **~50 verified objects** per run (stop verifying at 50 good crops; if the run yields <50, note it) | verification step |
| Bag replay rate | `--rate 0.1` (inherited from prior campaign) — 700 s wall ≈ 70 s bag-time. Raise only if <50 objects are produced; if changed, keep the new rate fixed for all 9 runs and note in §15. | `ros2 bag play` |
| Crop context ratio | `0.10` | `phase1.vlm.crop_context_ratio` |
| Context suppression | greyscale + `vlm_context_alpha: 0.12` | `phase1.semantic_crop` |
| Target contour | **cyan `[0,255,255]`**, 2 px (change from live `[255,255,255]` — see §4) | `phase1.semantic_crop` |
| Temperature | `0.0` | `phase1.vlm` |
| max_tokens | `96` | `phase1.vlm` |
| result_validation | unchanged from live (`min_label_confidence: 0.80`, `min_mobility_confidence: 0.70`) | `phase1.vlm.result_validation` |
| RAP memory / visual store | **cleared before every run** | see §6 step 2 |

---

## 6. Procedure (per run)

Repeat for each of the 9 rows in the matrix (§7). `RUN_ID` = e.g. `R1__qwen3vl8b__v1_simplified`.

**1. Select model profile** (only when the model changes — i.e. before R1, R4, R7):
```bash
# edit phase1.vlm.active_profile in src/rsg/config/rsg_pipeline.yaml
cd ~/rsg_ros2_ws
colcon build --packages-select rsg --symlink-install
source install/setup.bash
```
Confirm no other llama-server is running on :8000. Start the server for the
selected profile (profile-aware launcher or `rsg_all` launch). Wait until the
model is fully loaded (first request succeeds).

**2. Clear RAP + visual memory** (before every run):
```bash
rm -f  ~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl
rm -rf ~/rsg_ros2_ws/visual_memory/* && mkdir -p ~/rsg_ros2_ws/visual_memory
rm -rf ~/rsg_ros2_ws/VLM-Test-Session && mkdir -p ~/rsg_ros2_ws/VLM-Test-Session/crops
```

**3. Set the active prompt:** set `active: true` for the target prompt in
`prompts_under_test.yaml` (and `false` for the other two), or paste the prompt
text into `phase1.vlm.prompt`. Rebuild if the prompt lives in the source YAML.

**4. Run (700 s):**
```bash
# terminal A — pipeline (VLM-only, RAP disabled)
ros2 run rsg rsg_phase1.py            # add --vlm-prompt-optimization if using the harness flag

# terminal B — replay, capped at 700 s
timeout 700s ros2 bag play ~/datasets/uhumans2/uHumans2_office_s1_00h_ros2 \
  --rate 0.1 --qos-profile-overrides-path ~/.tf_overrides.yaml
```
Every VLM call auto-logs a crop to `VLM-Test-Session/crops/` and a row to
`VLM-Test-Session/vlm_results.csv` with **both** timing fields (see §8).

**5. Archive raw outputs immediately:**
```bash
DST=~/Thesis_new/debug/prompt_optimisation_experiment/runs/$RUN_ID
cp -r ~/rsg_ros2_ws/VLM-Test-Session/crops/*        $DST/crops/
cp    ~/rsg_ros2_ws/VLM-Test-Session/vlm_results.csv $DST/vlm_results.csv
```

**6. Manual verification** (target ~50 rows). For each verified row fill:
- `manual_label` — ground-truth class (what the object actually is)
- `manual_is_correct` — `true` if `vlm_label` matches `manual_label`, else `false`
- `error_category` — one value from §9 (`ok` when correct)
- `manual_notes` — what specifically is wrong / why (free text)

Rows you skip: leave `manual_is_correct` blank. Rows too blurred/truncated to
judge: `error_category = crop_quality`, excluded from accuracy.

**7. Fill the run’s results block** in §11 (accuracy, error tally, latency
percentiles) and write 3–6 bullet observations.

**8. Update the matrix (§7) status and the Changelog (§15).**

---

## 7. Run matrix

Order: finish all 3 prompts on one model before switching model (minimises
rebuilds / model reloads).

| Run | Model | Prompt | Folder | Status | Verified N | Accuracy | `ceiling_light` errs | median inference ms |
|-----|-------|--------|--------|--------|-----------|----------|----------------------|---------------------|
| R1 | qwen3vl8b | P1 v1_simplified | `runs/R1__qwen3vl8b__v1_simplified/session_20260901_195516/` | ☑ **complete** | 50 | **48 %** | 6 (+2 as `ceiling`) | 6250 |
| R2 | qwen3vl8b | P2 v5_examples_based (reworked) | `runs/R2__qwen3vl8b__v5_examples_based/session_20260901_204248/` | ☑ **complete** | 50 | **72 %** | 2 | 4826 |
| R3 | qwen3vl8b | P3 v6_structural_priority (tuned on R1+R2) | `runs/R3__qwen3vl8b__v6_structural_priority/` | ◐ wired & ready — run pending | — | — | — | — |
| R4 | qwen35_4b | P1 v1_simplified | `runs/R4__qwen35_4b__v1_simplified/` | ☐ not started | — | — | — | — |
| R5 | qwen35_4b | P2 v5_examples_based | `runs/R5__qwen35_4b__v5_examples_based/` | ☐ not started | — | — | — | — |
| R6 | qwen35_4b | P3 v6_structural_priority | `runs/R6__qwen35_4b__v6_structural_priority/` | ☐ not started | — | — | — | — |
| R7 | qwen35_9b | P1 v1_simplified | `runs/R7__qwen35_9b__v1_simplified/` | ☐ not started | — | — | — | — |
| R8 | qwen35_9b | P2 v5_examples_based | `runs/R8__qwen35_9b__v5_examples_based/` | ☐ not started | — | — | — | — |
| R9 | qwen35_9b | P3 v6_structural_priority | `runs/R9__qwen35_9b__v6_structural_priority/` | ☐ not started | — | — | — | — |

“maybe more if needed” — add R10+ rows here for extra prompts/models; keep the
same folder + CSV convention.

---

## 8. Data layout & CSV schema

```
debug/prompt_optimisation_experiment/
├── EXPERIMENT_REPORT.md          <- this file (the only main document)
├── prompts_under_test.yaml       <- the 3 prompts, loadable
├── vlm_results_TEMPLATE.csv      <- header-only template
└── runs/
    └── R{n}__{model}__{prompt}/
        ├── crops/               <- scaffold placeholder (.gitkeep)
        ├── vlm_results.csv      <- scaffold placeholder (header only)
        └── session_<YYYYmmdd_HHMMSS>/   <- one per pipeline start
            ├── crops/           <- every crop JPEG for that session
            └── vlm_results.csv  <- one row per VLM call (schema below)
```

`runs/*/session_*/` is git-ignored so scratch/smoke sessions don't clutter the
tree; a **completed, annotated** session is committed deliberately with
`git add -f <session_dir>` and recorded in §7 / §11.

`vlm_results.csv` columns:

| Column | Filled by | Meaning |
|---|---|---|
| `object_id` | pipeline | unique object/track id |
| `crop_filename` | pipeline | JPEG in `crops/` |
| `frame_timestamp` | pipeline | bag time of the crop |
| `run_id` | pipeline/manual | e.g. `R1__qwen3vl8b__v1_simplified` |
| `model_profile` | pipeline/manual | `qwen3_vl_8b_f16` \| `qwen3_5_4b_q4` \| `qwen3_5_9b_q4` |
| `prompt_version` | pipeline/manual | `P1_v1_simplified` \| `P2_v5_examples_based` \| `P3_v6_structural_priority` |
| `vlm_label` | pipeline | predicted label |
| `label_confidence` | pipeline | 0–1 |
| `mobility_class` | pipeline | static \| dynamic \| unknown |
| `mobility_confidence` | pipeline | 0–1 |
| **`vlm_inference_ms`** | pipeline | **MANDATORY** — model compute time. Use llama.cpp `timings.prompt_ms + timings.predicted_ms` if the server returns them; else blank and rely on `end_to_end_ms`. |
| **`end_to_end_ms`** | pipeline | **MANDATORY** — client `perf_counter` around the whole HTTP call (request build → response parsed) |
| `success` | pipeline | VLM call + JSON parse ok |
| `validation_status` | pipeline | accepted \| rejected (result_validation) |
| `raw_response` | pipeline | raw model text (trim to ~200 chars) |
| `manual_label` | **human** | ground truth |
| `manual_is_correct` | **human** | true \| false \| (blank = not verified) |
| `error_category` | **human** | one token from §9 |
| `manual_notes` | **human** | what is wrong / notable |

---

## 9. Error category vocabulary (`error_category`)

| Token | Use when |
|---|---|
| `ok` | prediction matches ground truth (label correct; mobility also correct) |
| `ceiling_light` | ground truth is `ceiling` but model returned the embedded light/fixture (the prior campaign’s signature failure) |
| `surface_vs_fixture` | other surface↔attachment confusion: `wall`↔whiteboard/sign/AC/monitor, `floor`↔rug/carpet, `door`↔handle, etc. |
| `boundary_violation` | model labelled an object **outside** the cyan contour |
| `unknown_overuse` | model returned `VLM_no_result`/`unknown_object` but the object was clearly identifiable |
| `wrong_class` | plain misclassification not covered above (e.g. `chair`→`desk`) |
| `mobility_wrong` | label correct but `mobility_class` wrong (e.g. person → static) |
| `hallucinated_object` | label names something not present in the crop at all |
| `crop_quality` | crop too blurred/truncated/dark to verify — **excluded from accuracy denominator** |
| `other` | anything else — explain in `manual_notes` |

---

## 10. Metrics (compute per run, then compare)

- **Accuracy** = `count(manual_is_correct == true) / count(verified rows excluding crop_quality)`.
- **Accuracy (label-only)** = same, but ignoring `mobility_wrong` (label correct
  counts as correct). Report both this and strict accuracy.
- **Rejection rate** = `count(validation_status == rejected) / count(all VLM
  calls)`. Report accuracy **including** rejected rows (rejected = wrong) **and
  excluding** them, so the `min_label_confidence: 0.80` gate does not silently
  favour the prompt with more explicit confidence-band wording (P3) over the
  leaner ones (P1, P2).
- **Error tally** = count per `error_category` (absolute + % of verified).
- **Ceiling-light rate** = `count(error_category == ceiling_light) / count(rows whose ground truth is ceiling)`.
- **Confidence calibration** = accuracy split at `label_confidence` ≥ 0.90 vs 0.80–0.90 vs <0.80.
- **Latency** = median, p90, p95, max of `vlm_inference_ms` (and of `end_to_end_ms`); also mean.
- **Throughput** = verified objects produced per 700 s run.
- 95 % CI on accuracy at N≈50 ≈ ±14 pp — treat differences smaller than that as
  not significant; prefer error-category shifts as the finer signal.

---

## 11. Per-run results

> Fill one block per run. Keep the prose to observations that the CSV cannot show.

### R1 — qwen3vl8b × P1 v1_simplified  ✅ COMPLETE
- Run date / wall time: 2026-09-01 · ~700 s wall @ `--rate 0.1` · session `session_20260901_195516`
- Data: `runs/R1__qwen3vl8b__v1_simplified/session_20260901_195516/` (59 crops + `vlm_results.csv`)
- Objects classified: 59 | Verified: 50 (rows 51–59 not annotated) | Excluded (`crop_quality`): 0
- **Accuracy: 24/50 = 48 %** (strict). Label-only accuracy identical — every row is `mobility_class: static` and every mobility call is correct, so mobility contributes no errors.
- Rejection rate: **0/59** — every call `accepted`. `label_confidence` is **0.95 or 0.99 on every single crop**, including the 52 % that are wrong. The confidence field carries no signal, so the `min_label_confidence: 0.80` gate never fires and cannot help. Accuracy is therefore the same with or without rejected rows.
- Latency (`vlm_inference_ms`): median 6250 | p90 6525 | p95 6650 | max 6955 | mean 6119 (min 4903)
- Latency (`end_to_end_ms`): median 6461 | p90 7194 | max 7447 | mean 6428
- Throughput: 59 crops / 700 s ≈ one every ~12 s wall (≈ 6.2 s of that is model compute).
- Error tally (of 26 wrong): `ceiling_light`* 6 | `surface_vs_fixture` 2 | `boundary_violation` 5 | `unknown_overuse` 0 | `wrong_class` 13 | `mobility_wrong` 0 | `other` 0
  - *\*`ceiling_light` bucket used for the same failure mode with a different sub-part token:* all 6 are `ceiling_tile` returned where the contour spans a whole panelled ceiling (ground truth `ceiling`) — rows 1, 7, 9, 14, 32, 43. Of the 9 ground-truth-`ceiling` crops in the run, only 1 (row 33) was labelled `ceiling`; `ceiling_panel` on row 19 was accepted as correct by the annotator. Panelled-ceiling failure from the prior campaign is **fully reproduced**.
- Observations:
  - **Boundary violations (5):** the model names a salient object that is *in the frame but outside the cyan contour* — armchair (rows 2, 6), shelf (35), whiteboard (38), bookshelf (47). The contour target was a wall/floor plane behind/around the named object. P1's "identify only this object … the surrounding area may be used as context" is not strong enough — the model treats the whole crop as fair game.
  - **Sub-part instead of the enclosing surface (6 + 2):** `ceiling_tile` for a panelled ceiling (×6); `whiteboard` for a wall region that merely contains a whiteboard (rows 26, 48). When the contour encloses many repeated instances or a surface-with-a-fixture, the model picks the small nameable thing, not the general term.
  - **Hallucination on low-information crops (subset of the 13 `wrong_class`):** mirror↔whiteboard confusion twice (39, 42), `keyboard` invented from a ceiling panel (22), `refrigerator`/`cabinet` for what the annotator read as a wall/pillar (13, 21, 29), `fireplace` for a cabinet (20), `glass_surface` missed as `shelf` (49). The model commits to a specific object at 0.95 even when the crop is ambiguous rather than backing off to the dominant structure or `VLM_no_result`.
  - **Mobility is a non-issue** for this scene — everything static, all correct.
  - Latency is tight and predictable (~6.1–6.3 s inference, low variance) — not a differentiator to worry about for the 8B FP16 profile.

### R2 — qwen3vl8b × P2 v5_examples_based (reworked)  ✅ COMPLETE
- Run date / wall time: 2026-09-01 · ~700 s wall @ `--rate 0.1` · session `session_20260901_204248`
- Data: `runs/R2__qwen3vl8b__v5_examples_based/session_20260901_204248/` (53 crops + `vlm_results.csv`)
- Objects classified: 53 | Verified: 50 (rows 51–53 not annotated) | Excluded (`crop_quality`): 0
- **Accuracy: 36/50 = 72 %** (strict). Label-only identical — all `static`, every mobility call correct.
- **vs R1 (P1): 48 % → 72 %, +24 pp** — larger than the ±14 pp CI, so a real prompt effect, not noise.
- Rejection rate: **0/53**. Confidence now spreads — 0.85 ×9, 0.92 ×17, 0.95 ×27 (R1 was a flat 0.95/0.99) — but still nothing below the 0.80 gate, so rejection still contributes no signal. The 0.85 rows are a mix of right and wrong, so the spread is not yet a usable reliability signal.
- Latency (`vlm_inference_ms`): median 4826 | p90 5624 | p95 5882 | max 8099 (cold first call) | mean 4997 (min 4438)
- Latency (`end_to_end_ms`): median 5450 | p90 5921 | max 8159 | mean 5522
- **Faster than R1** (median 4826 vs 6250 ms) despite the longer prompt — the more directive instructions yield shorter completions.
- Error tally (of 14 wrong): `ceiling_light`* 2 | `surface_vs_fixture` 0 | `boundary_violation` **0** | `unknown_overuse` 0 | `wrong_class` 12 | `mobility_wrong` 0 | `other` 0
  - *\*ceiling-related: rows 23, 37 — a `rug`/`wall` returned where the ground truth was `ceiling_tile` (orientation confusion). The R1 failure mode (panelled ceiling → `ceiling_tile`) is **gone** — R2 says `ceiling` and is correct on those crops.*
- Observations:
  - **Boundary violations eliminated (5 → 0).** The elaborated "name only what the cyan boundary encloses; a salient object outside it is background, never the answer" section did its job — no armchair/shelf/bookshelf-behind-the-plane mistakes this run.
  - **Panelled-ceiling fixed.** R1: `ceiling_tile` ×6, all wrong. R2: `ceiling` on the same class of crop, marked correct (rows 1, 2, 6, 9, 14, 25, 28, 29, 42, 49 …). ceiling→sub-part rate fell from 8/9 to ~2/13.
  - **Surface-with-fixture fixed.** whiteboard-on-a-wall now returns `wall` (R1 rows 26, 48 were wrong); standalone whiteboards still correctly `whiteboard` (rows 36, 43).
  - **New dominant error — glass partitions (5): rows 20, 21, 32, 44, 46.** The scene has glass partition walls / glass-panelled doors; the model has no example for them and falls back to `wall` / `door` / `shelf` / `ceiling`. The annotator flagged this exact gap in R1 (row 49). Needs a `glass_wall` / `glass_partition` exemplar.
  - **Flat-vertical-surface ambiguity persists (3): rows 22 (`cabinet`, truth wall/pillar), 34 (`curtain`, truth cabinet), 40 (`wall`, truth cabinet).** P2's low-confidence "cabinet/pillar/wall → wall @ 0.4" example did not move the model — it still answered 0.85–0.95 here.
  - **Generalisation rule over-fired on distinctive infrastructure (4): rows 10, 30 (exposed pipes → `ceiling`), 17 (air vent → `ceiling`), 20 (→ `ceiling`).** "Name the surface, not the fixture" is right for a light in a panel but wrong when exposed ductwork/pipes/vents fill the boundary and the annotator wants them named. P3 must thread this: generalise repeated *surface units*, still name distinctive attached infrastructure that dominates the crop.
  - **mirror ↔ whiteboard** still misfires once (row 50) — down from 2 in R1.
  - Mobility remains a non-issue for this scene.

### R3 — qwen3vl8b × P3 v6_structural_priority
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded: —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

### R4 — qwen35_4b × P1 v1_simplified
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded: —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

### R5 — qwen35_4b × P2 v5_examples_based
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded: —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

### R6 — qwen35_4b × P3 v6_structural_priority
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded: —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

### R7 — qwen35_9b × P1 v1_simplified
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded: —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

### R8 — qwen35_9b × P2 v5_examples_based
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded: —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

### R9 — qwen35_9b × P3 v6_structural_priority
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded: —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

---

## 12. Cross-run comparison (fill after all runs)

### 12.1 Accuracy — model × prompt

| Accuracy | P1 v1_simplified | P2 v5_examples_based | P3 v6_structural_priority | best prompt |
|---|---|---|---|---|
| qwen3vl8b | 48 % (24/50) | **72 % (36/50)** | — | P2 so far (+24 pp) |
| qwen35_4b | — | — | — | — |
| qwen35_9b | — | — | — | — |
| best model | — | — | — | — |

### 12.2 `ceiling_light` error rate — model × prompt

| ceiling→sub-part rate | P1 | P2 | P3 |
|---|---|---|---|
| qwen3vl8b | 8/9 (89 %) — 6×`ceiling_tile`, +`keyboard`, +`shelf`; 1×`ceiling` correct | ~2/13 (15 %) — `ceiling` now the default; misses are `rug`/`wall` for `ceiling_tile` crops. But 4 *new* over-generalisations: exposed pipes/vents → `ceiling` | — |
| qwen35_4b | — | — | — |
| qwen35_9b | — | — | — |

### 12.3 Latency — model × prompt (median `vlm_inference_ms`)

| median inf. ms | P1 | P2 | P3 |
|---|---|---|---|
| qwen3vl8b | 6250 (mean 6119, p95 6650) | 4826 (mean 4997, p95 5882) | — |
| qwen35_4b | — | — | — |
| qwen35_9b | — | — | — |

### 12.4 Accuracy ↔ latency

| Model | best accuracy (prompt) | median inf. ms at that prompt | verified objects / 700 s |
|---|---|---|---|
| qwen3vl8b | — | — | — |
| qwen35_4b | — | — | — |
| qwen35_9b | — | — | — |

---

## 13. Findings & recommendation (fill at the end)

- Q1 — does Qwen 3.5 break the ~50 % ceiling / `ceiling_light` error?  **—**
- Q2 — best prompt per model:  **—**
- Q3 — accuracy/latency trade-off; recommended production profile + prompt:  **—**
- Any prompt change to promote into `rsg_pipeline.yaml`:  **—**

---

## 14. Deviations & risks

- **Boundary colour: cyan** (matches the frozen campaign). Requires setting the
  live pipeline contour back to `[0,255,255]` before R1 (§4, §5); it is currently
  white. Verify the rendered crops actually show a cyan contour on the first run.
- **All 3 prompts are re-worked, not verbatim.** Deliberate: label enumerations
  removed (open-vocabulary, for generalisation), JSON-output instruction made
  uniform, one `dynamic` example added to P2 and P3. P3 is therefore no longer
  the byte-exact production prompt — it is the production *strategy* without the
  structural-label list. Prior 8B numbers (§2) are for the source prompts only.
- **P1↔P2↔P3 differ on more than one axis** (examples, and amount of reasoning
  guidance). They are three holistic strategies along a complexity gradient, not
  minimal pairs. Phrase §13 conclusions as "strategy P_x beats P_y on model M",
  never "examples help" / "calibration helps".
- **No `dynamic` exemplar in P1** (it has no examples at all — inherent to the
  gradient, not a fixable confound). P2 and P3 each carry exactly one.
- **Q4 vs FP16**: qwen35_4b/9b are 4-bit; qwen3vl8b is FP16. Model *and* quant
  differ together — a 4B-vs-8B accuracy gap cannot be cleanly attributed to size
  alone. Note this in §13.
- **Bag coverage**: 700 s wall @ `--rate 0.1` ≈ 70 s of the office scene. If a run
  produces <50 objects, raising `--rate` is allowed but must then be held fixed
  for all 9 runs (re-run any already-done rows). Record in §15.
- **Single annotator, single scene** (office_s1). Accuracy is skewed by whatever
  classes dominate this scene (historically: lots of panelled ceiling).
- **One model at a time** on :8000 — never launch two llama-servers.
- `qwen3_vl_8b_f16` needs ≥32 GB; if it will not load, record R1–R3 as blocked
  rather than dropping to the 2B profile mid-experiment.

---

## 15. Changelog

| Date | Change |
|---|---|
| 2026-08-31 | Scaffold created: folder tree, 9 run dirs (`runs/R1..R9`), CSV template, `prompts_under_test.yaml`, this report. Prompts chosen: v1_simplified, v5_examples_based, v6_structural_priority. 0/9 runs done. Depends on `uHumans2_office_s1_00h_ros2` bag conversion (in progress) and Qwen 3.5 4B/9B weights being present under `~/rsg_models/`. |
| 2026-08-31 | Boundary colour set to **cyan** (was drafted as white). All 3 prompts now say "cyan boundary"; live pipeline `target_contour_rgb` must be changed `[255,255,255]` → `[0,255,255]` before R1. |
| 2026-08-31 | Prompts re-worked to a clean design: (a) **complexity gradient** P1 simple → P2 +many examples → P3 detailed strategy+calibration; (b) **all enumerated label lists removed** — every prompt is open-vocabulary ("no fixed list of labels") for generalisation; (c) JSON output format + "return only JSON" line now in **all three**; (d) one `dynamic` (person) example added to P2 and P3; (e) shared JSON schema / mobility defs / `VLM_unknown` fallback unified. P3 is no longer byte-verbatim from `rsg_pipeline.yaml`. §10 now also requires accuracy reported with/without `rejected` rows. |
| 2026-09-01 | Model-abstention label renamed `VLM_unknown` → **`VLM_no_result`** across all three prompts, `rsg_pipeline.yaml` `phase1.vlm.prompt`, and this report. `vlm_result.py` now recognises `vlm_no_result`/`vlm_unknown` as abstention sentinels and records the rejected result as `label="VLM_no_result"` (distinct from `unknown_object`, which stays reserved for parse failures, HTTP/worker errors, and sub-threshold real guesses). Lets CSV review separate "model looked and could not identify" from "no VLM result at all". |
| 2026-09-01 | **R1 complete** (session `session_20260901_195516`, 59 crops, 50 annotated). Accuracy **48 %** (24/50). Dominant failures: `wrong_class` 13 (hallucination on ambiguous crops), `ceiling`→sub-part 6+ (panelled-ceiling failure reproduced), `boundary_violation` 5, `surface_vs_fixture` 2. Confidence is a constant 0.95/0.99 — zero rejection signal. Latency median 6.25 s. §11 R1 + §12.1/12.2/12.3 (qwen3vl8b row) filled. |
| 2026-09-01 | **P2 reworked** against R1 findings: elaborated boundary-focus section (name only what the contour encloses; salient context objects are *not* the target), explicit "multiple instances / surface-with-fixture → name the general enclosing surface" rule, calibrated-confidence table so ambiguous crops stop returning 0.95, and worked examples drawn from real R1 misses (armchair-behind-wall, `ceiling_tile`→`ceiling`, whiteboard-on-wall→`wall`, mirror↔whiteboard, wall/pillar↔cabinet). Still open-vocabulary — no label list. `active: true` moved P1 → P2; `rsg_pipeline.yaml` `phase1.vlm.prompt` + `prompt_optimisation.{run_id,prompt_version}` switched to R2. |
| 2026-09-01 | **R2 complete** (session `session_20260901_204248`, 53 crops, 50 annotated). Accuracy **72 %** (36/50) — **+24 pp over R1**. `boundary_violation` 5 → 0; panelled-ceiling and whiteboard-on-wall fixed. Remaining errors (14): glass partitions 5 (new — no exemplar), flat-surface cabinet↔wall/pillar 3, generalisation over-firing on exposed pipes/vents → `ceiling` 4, `ceiling_tile` inversions 2, mirror↔whiteboard 1. Latency median 4826 ms (faster than R1's 6250). Confidence now spreads 0.85/0.92/0.95 but still no <0.80. §7 / §11 / §12.1–12.3 filled. |
| 2026-09-01 | **P3 updated against R2 findings** (still open-vocabulary). Kept P2's wins; added: (b) "distinctive infrastructure that fills the boundary is named, not generalised" with a pipes/vent/fire-hose example group (fixes R2's 4 pipes/vents → `ceiling`); a glass-surface cue + `glass_wall`/`glass_door`/mirror-vs-whiteboard example group (fixes R2's 5 glass misses); (c) a handle/seam/toe-kick discriminator for cabinet-vs-wall/pillar plus a wider 0.45–0.70 honest-confidence band for it (R2's low-confidence example didn't move the model). §4 P3 text refreshed. `active` still on P2 — P3 not yet wired into `rsg_pipeline.yaml`. |
| 2026-09-01 | P3: added a direct see-through transparency test (room/objects/light visible through the panel → glass; opaque → solid) — R2's glass misses were framed, lightly-reflective partitions read as `wall`. |
| 2026-09-01 | **P3 wired for R3.** `prompts_under_test.yaml` `active` flipped P2 → P3. `rsg_pipeline.yaml` `phase1.vlm.prompt` = P3 template (verified byte-identical), `prompt_optimisation.run_id = R3__qwen3vl8b__v6_structural_priority`, `prompt_version = P3_v6_structural_priority`. Crops + CSV for the next run land in `runs/R3__qwen3vl8b__v6_structural_priority/session_<ts>/`. Needs a phase1 coordinator restart. |
