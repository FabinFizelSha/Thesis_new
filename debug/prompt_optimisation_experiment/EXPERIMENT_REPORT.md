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
  string, identical mobility definitions, `VLM_unknown` + 0.0 fallback.
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

Name the object with a concise, singular, lowercase snake_case label that best describes what it actually is. There is no fixed list of labels. Do not describe colour, material, position, condition or activity. Use a confidence above 0.90 only when the identification is unmistakable. If the target is unclear, truncated or unidentifiable, use the label VLM_unknown with confidence 0.0.

Mobility:
- "dynamic": a human, an animal, or a self-propelled robot only
- "static": everything else
- "unknown": when the label is unknown or mobility cannot be determined

Output exactly one JSON object:
{"label": "<label>", "label_confidence": <0-1>, "mobility_class": "<static|dynamic|unknown>", "mobility_confidence": <0-1>}

Return only the JSON object. No markdown, explanations, or extra text.
```

### P2 — example-driven (simple rules + many worked examples, general)

```text
Classify the main object outlined by the cyan boundary in this indoor crop.

Rules:
- Identify only the object inside the cyan boundary. Ignore anything outside it.
- Name the broadest, largest, most dominant object the boundary covers. When a surface such as a ceiling, wall or floor contains fixtures, attachments or details, name the surface, not the smaller thing on it.
- Use a concise, singular, lowercase label that best fits what you see. There is no fixed list of labels.
- Use a confidence above 0.90 only when the identification is unmistakable.
- If the crop is unclear or truncated, use label VLM_unknown with confidence 0.0.

Mobility:
- "dynamic": a human, an animal, or a self-propelled robot only
- "static": everything else
- "unknown": when the label is unknown

Output exactly one JSON object:
{"label": "<label>", "label_confidence": <0-1>, "mobility_class": "<static|dynamic|unknown>", "mobility_confidence": <0-1>}

Examples (these show the reasoning pattern, not a list of allowed labels):
- Recessed lights set into a ceiling panel -> {"label": "ceiling", ...}
- A ceiling panel with embedded vents and fixtures -> {"label": "ceiling", ...}
- A wall with a whiteboard mounted on it -> {"label": "wall", ...}
- A wall with an air-conditioning unit on it -> {"label": "wall", ...}
- A floor with a rug lying on it -> {"label": "floor", ...}
- A door and its frame -> {"label": "door", ...}
- A window with a view behind it -> {"label": "window", ...}
- An office chair filling most of the frame -> {"label": "office_chair", ...}
- A potted plant on a stand -> {"label": "potted_plant", ...}
- A person standing in the space -> {"label": "person", "mobility_class": "dynamic", ...}
- A blurred or cut-off crop with no identifiable object -> {"label": "VLM_unknown", ...}

Return only the JSON object. No markdown, explanations, or extra text.
```

(Full example JSON with confidences is in `prompts_under_test.yaml` — 11 examples.)

### P3 — detailed (reasoning strategy + confidence calibration + grouped examples, general)

```text
This is an indoor object-classification task. Identify the main object outlined by the cyan boundary.

Approach:
1. Look only inside the cyan boundary. Ignore anything outside it completely.
2. Decide what the largest, most dominant thing inside the boundary is.
3. Prefer the broadest structure over the details on it:
   - The surfaces and structures that form the room itself - the large planes overhead, underfoot and to the sides, together with the openings and supports in them - take priority over anything mounted on, set into, or resting against them.
   - A freestanding piece of furniture takes priority over the smaller objects attached to or placed on it.
   - When two objects compete, choose the larger and more dominant one.
   - When a surface carries fixtures, attachments or markings, name the surface, not the fixture.
4. Give the object a concise, singular, lowercase label that best describes what it actually is. There is no fixed list of labels.

Confidence calibration for "label_confidence":
- 0.90-1.00: unmistakable, no ambiguity
- 0.70-0.90: clear, minor uncertainty
- 0.50-0.70: plausible but competing interpretations
- below 0.50: very uncertain
- 0.0: incomprehensible crop (use label VLM_unknown)

Mobility:
- "dynamic": only a human, an animal, or a self-propelled moving object
- "static": any stationary object, including all surfaces, structures, furniture and fixtures
- "unknown": when the label is unknown

Output format (required) - exactly one JSON object:
{"label": "<label>", "label_confidence": <0-1>, "mobility_class": "<static|dynamic|unknown>", "mobility_confidence": <0-1>}

Worked examples (illustrating the approach, not a list of allowed labels):

Room-defining structures take priority over their details:
- Recessed lights set into a ceiling panel -> {"label": "ceiling", ...}
- A ceiling panel with embedded vents and fixtures -> {"label": "ceiling", ...}
- A wall carrying a whiteboard or a mounted sign -> {"label": "wall", ...}
- A floor with a rug or carpet on it -> {"label": "floor", ...}
- A door and its opening -> {"label": "door", ...}

Furniture takes priority over the objects on it:
- An office chair filling most of the frame -> {"label": "office_chair", ...}
- A desk seen clearly, with items on top -> {"label": "desk", ...}
- A sofa or couch -> {"label": "sofa", ...}

Moving agents:
- A person in the space -> {"label": "person", "mobility_class": "dynamic", ...}

Unclear:
- A blurred, truncated or unidentifiable crop -> {"label": "VLM_unknown", ...}

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
| R1 | qwen3vl8b | P1 v1_simplified | `runs/R1__qwen3vl8b__v1_simplified/` | ☐ not started | — | — | — | — |
| R2 | qwen3vl8b | P2 v5_examples_based | `runs/R2__qwen3vl8b__v5_examples_based/` | ☐ not started | — | — | — | — |
| R3 | qwen3vl8b | P3 v6_structural_priority | `runs/R3__qwen3vl8b__v6_structural_priority/` | ☐ not started | — | — | — | — |
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
        ├── crops/                <- every crop JPEG for that run
        └── vlm_results.csv       <- one row per VLM call (schema below)
```

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
| `unknown_overuse` | model returned `VLM_unknown`/`unknown_object` but the object was clearly identifiable |
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

### R1 — qwen3vl8b × P1 v1_simplified
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded (`crop_quality`): —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

### R2 — qwen3vl8b × P2 v5_examples_based
- Run date / wall time: —
- Objects classified: — | Verified: — | Excluded: —
- **Accuracy: —**
- Latency (`vlm_inference_ms`): median — | p90 — | p95 — | max — | mean —
- Latency (`end_to_end_ms`): median — | p90 — | max —
- Error tally: `ceiling_light` — | `surface_vs_fixture` — | `boundary_violation` — | `unknown_overuse` — | `wrong_class` — | `mobility_wrong` — | `other` —
- Observations:
  - —

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
| qwen3vl8b | — | — | — | — |
| qwen35_4b | — | — | — | — |
| qwen35_9b | — | — | — | — |
| best model | — | — | — | — |

### 12.2 `ceiling_light` error rate — model × prompt

| ceiling→ceiling_light rate | P1 | P2 | P3 |
|---|---|---|---|
| qwen3vl8b | — | — | — |
| qwen35_4b | — | — | — |
| qwen35_9b | — | — | — |

### 12.3 Latency — model × prompt (median `vlm_inference_ms`)

| median inf. ms | P1 | P2 | P3 |
|---|---|---|---|
| qwen3vl8b | — | — | — |
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
