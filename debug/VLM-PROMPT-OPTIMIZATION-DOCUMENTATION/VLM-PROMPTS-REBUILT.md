# VLM Prompt Optimization — Rebuilt Prompt Set (v1–v6)

**Purpose:** Single reference holding the full text of every prompt tested in the
2026-08-30 VLM prompt-optimization campaign, reconstructed where the original
documentation only described the prompt. Intended as the control set for the
Qwen 3.5 re-run.

**Original model for all six runs:** Qwen3-VL-8B-Instruct, FP16 GGUF, llama.cpp
on `http://127.0.0.1:8000/v1/chat/completions`, `temperature: 0.0`,
`max_tokens: 96`.

## Provenance status

| Ver | Provenance | Source | Fidelity |
|-----|-----------|--------|----------|
| v1_simplified        | **VERBATIM**     | `VLM-PROMPT-OPTIMIZATION-EXPERIMENT.md` §3.1 | exact |
| v2_refined           | **RECONSTRUCTED** | `VLM-PROMPT-OPTIMIZATION-EXPERIMENT.md` §3.2 + `VLM-PROMPT-OPTIMIZATION-FINDINGS.md` (bullet descriptions only) | approximate |
| v3_hierarchical      | **RECONSTRUCTED** | `VLM-PROMPT-OPTIMIZATION-FINDINGS.md` "v3_hierarchical Prompt Structure" (structural outline) | close — outline was explicit |
| v4_filtering         | **VERBATIM**     | `VLM-PROMPT-V3-V4-ANALYSIS.md` (`prompt: |` YAML block) | exact |
| v5_examples_based    | **RECONSTRUCTED** | `VLM-V5-PROMPT.md` (structure + example snippets) | approximate |
| v6_structural_priority | **VERBATIM**   | live `src/rsg/config/rsg_pipeline.yaml:331-385` (frozen final prompt) | exact |

> **Caveat — boundary colour:** v1–v5 docs all say "cyan boundary". The live
> pipeline now draws a **white** 2 px target contour
> (`semantic_crop.target_contour_rgb: [255,255,255]`), and context is greyscaled
> and dimmed to 12 % intensity. v6 verbatim still says "cyan boundary". For a
> clean re-run, reconcile the wording with the actual contour colour, or restore
> a cyan contour, and keep it identical across all six prompts.

> **Caveat — reconstructions:** v2, v3, v5 were **not** stored verbatim. The
> per-version archive folders (`VLM-Test-Session-v2_refined-20260830-190300/`
> etc.) referenced in the docs are not in this repo and `~/rsg_ros2_ws` no longer
> exists. The text below reproduces every documented feature of each prompt but
> will not be byte-identical to what Qwen3-VL-8B received. Treat v2/v3/v5
> accuracy comparisons against the Qwen 3.5 run as indicative, not exact.

## Tested results (Qwen3-VL-8B-Instruct FP16) — for reference

| Ver | Accuracy | "ceiling_light" errors | Notes |
|-----|----------|------------------------|-------|
| v1_simplified        | 46.9 % (23/49) | 9  | baseline; over-uses `unknown_object` |
| v2_refined           | 47.5 % (19/40) | 11 | +0.6 %, within noise; hierarchy ignored |
| v3_hierarchical      | 38.2 % (21/55) | 15 | regression −9.3 %; complexity backfired |
| v4_filtering         | 96.7 % on 30/58 verified | 9+ unverified | number is a partial-verification artefact; filters not applied |
| v5_examples_based    | 51.8 %         | 16 | best headline, worst ceiling count |
| v6_structural_priority | 50.0 %       | 14 | frozen / current live prompt |

---

## v1_simplified — VERBATIM

```text
Classify the single SAM-segmented target in this indoor crop.

The object to be classified is identified as surrounded by the cyan coloured boundary. Identify only this object.
The surrounding area can be used to deduce the context which can help in identifying the main object.

The general context of the object is that it will be indoor.

Return exactly one JSON object with these keys:

{"label":"lowercase_object_name","label_confidence":0.0,"mobility_class":"static|dynamic|unknown","mobility_confidence":0.0}

Rules:
- Use concise, singular, lowercase snake_case labels
- Prefer specific labels (office_chair, computer_monitor, desk, cabinet, potted_plant, rug, cup)
- Do not describe colour, material, position, condition or activity
- Ignore objects in darkened context, shadows, reflections, screen content
- If unclear/truncated: return VLM_unknown with 0.0 confidence
- Use confidence >0.90 only for unmistakable targets

Mobility:
- "dynamic": human, animal, self-propelled robot only
- "static": all other objects (doors, furniture, tools, etc.)
- "unknown": when label is unknown or mobility cannot be determined
```

---

## v2_refined — RECONSTRUCTED

Built on the v1 base. Documented changes folded in:
explicit "ignore anything outside the cyan boundary"; an explicit
surface-vs-fixture distinction (`ceiling` ≠ `ceiling_light`, `wall` ≠
`whiteboard`); confidence guidance nudged to reduce `unknown_object` overuse;
worked examples that show surface classification and boundary respect.

```text
Classify the single SAM-segmented target in this indoor crop.

The object to be classified is the one surrounded by the cyan coloured boundary. Identify ONLY this object.
IGNORE any object that lies OUTSIDE the cyan boundary, even if it is large or prominent.
The surrounding area may be used only as context to help identify the target.

The general context of the object is that it will be indoor.

SURFACE vs FIXTURE:
- Classify the SURFACE, not the fixtures mounted on or embedded in it.
- A ceiling that contains recessed lights, vents or fixtures is still "ceiling", not "ceiling_light".
- A wall that carries a whiteboard, sign, screen or air conditioner is still "wall", not "whiteboard" / "sign".
- A floor that has a rug or carpet on it is still "floor", not "rug".
- Only classify the fixture itself when the fixture is the sole object inside the boundary and no host surface is visible.

Return exactly one JSON object with these keys:

{"label":"lowercase_object_name","label_confidence":0.0,"mobility_class":"static|dynamic|unknown","mobility_confidence":0.0}

Rules:
- Use concise, singular, lowercase snake_case labels
- Prefer specific labels (office_chair, computer_monitor, desk, cabinet, potted_plant, rug, cup)
- Do not describe colour, material, position, condition or activity
- Ignore shadows, reflections and screen content
- Commit to a label whenever the target is identifiable; use VLM_unknown only when the crop is genuinely unclear or truncated
- Use confidence >0.90 only for unmistakable targets

Mobility:
- "dynamic": human, animal, self-propelled robot only
- "static": all other objects (doors, furniture, tools, surfaces, etc.)
- "unknown": when label is unknown or mobility cannot be determined

Examples:
- Ceiling panel with embedded lights, target box covers the ceiling → {"label":"ceiling","label_confidence":0.92,"mobility_class":"static","mobility_confidence":0.99}
- Wall with a whiteboard mounted on it, boundary around the wall → {"label":"wall","label_confidence":0.90,"mobility_class":"static","mobility_confidence":0.99}
- Office chair filling the boundary → {"label":"office_chair","label_confidence":0.91,"mobility_class":"static","mobility_confidence":0.99}
- Prominent lamp OUTSIDE the cyan boundary, plain wall inside it → {"label":"wall","label_confidence":0.85,"mobility_class":"static","mobility_confidence":0.99}
- Blurry or truncated crop → {"label":"VLM_unknown","label_confidence":0.0,"mobility_class":"unknown","mobility_confidence":0.0}

Return only the JSON object. No markdown or explanations.
```

---

## v3_hierarchical — RECONSTRUCTED

From the explicit outline in `VLM-PROMPT-OPTIMIZATION-FINDINGS.md`
("v3_hierarchical Prompt Structure"): a CRITICAL hierarchical rule, a three-tier
priority list, and ❌/✅ anti-examples, with the JSON output and mobility rules
carried over from v1/v2. This is the version that regressed to 38.2 %.

```text
Classify the single SAM-segmented target inside the cyan boundary in this indoor crop.
Identify ONLY the object inside the cyan boundary. Ignore everything outside it.

CRITICAL RULE: Use hierarchical classification. Always choose the FIRST tier that applies.

HIERARCHY (pick from the FIRST tier that matches what is inside the boundary):

  Tier 1 - STRUCTURAL SURFACES (highest priority):
    ceiling, wall, floor, door, window, partition, pillar

  Tier 2 - FURNITURE / STRUCTURAL OBJECTS:
    office_chair, desk, sofa, cabinet, wardrobe, dresser

  Tier 3 - FIXTURES / DETAILS (only if nothing in Tier 1 or Tier 2 applies):
    whiteboard, monitor, lamp, air_vent, pipe, plant

ANTI-EXAMPLES (what NOT to do):
  Ceiling with lights          -> do NOT return "ceiling_light"  -> return "ceiling"
  Wall with sign or whiteboard  -> do NOT return "sign"           -> return "wall"
  Floor with rug                -> do NOT return "rug"            -> return "floor"
  Only a rug visible, no floor   -> return "rug"   (correct: no higher tier is present)

Return exactly one JSON object:

{"label":"lowercase_object_name","label_confidence":0.0,"mobility_class":"static|dynamic|unknown","mobility_confidence":0.0}

Confidence:
- >0.90 only for unmistakable targets
- lower values when the tier choice is uncertain
- 0.0 with label VLM_unknown when the crop is unclear or truncated

Mobility:
- "dynamic": human, animal, self-propelled robot only
- "static": all other objects
- "unknown": when the label is unknown

Return only the JSON object. No markdown, no explanation.
```

---

## v4_filtering — VERBATIM

```text
Classify the main object within the cyan boundary.

CRITICAL RULES (in order):
1. Objects OUTSIDE the cyan boundary → IGNORE them completely
2. Apply these filters:
   - If you see "ceiling_light" or "lights in ceiling" → return "ceiling" instead
   - If you see "wall_sign" or "sign on wall" → return "wall" instead
   - If you see "floor_rug" or "rug on floor" → return "floor" instead
3. Classify the BROADEST/PRIMARY category you can identify

Object categories (use preferentially):
ceiling, wall, floor, door, window, partition, pillar,
office_chair, desk, sofa, cabinet, wardrobe, dresser,
whiteboard, monitor, pipe, air_vent, plant

Return exactly one JSON:
{"label":"lowercase_object_name","label_confidence":0.0,"mobility_class":"static|dynamic|unknown","mobility_confidence":0.0}

Confidence:
- 0.90+: Unmistakable primary object
- 0.70-0.90: Clear but less obvious
- <0.70: Only if genuinely uncertain
- 0.0: Only for unknown_object

Mobility:
- "dynamic": human, animal, self-propelled robot only
- "static": everything else
- "unknown": when label is unknown

Examples:
• Ceiling with lights → {"label":"ceiling","label_confidence":0.92,"mobility_class":"static","mobility_confidence":0.99}
• Wall surface → {"label":"wall","label_confidence":0.94,"mobility_class":"static","mobility_confidence":0.99}
• Office chair → {"label":"office_chair","label_confidence":0.91,"mobility_class":"static","mobility_confidence":0.99}
• Ambiguous → {"label":"unknown_object","label_confidence":0.0,"mobility_class":"unknown","mobility_confidence":0.0}

Return only JSON. No markdown or explanations.
```

---

## v5_examples_based — RECONSTRUCTED

From `VLM-V5-PROMPT.md`: simple critical rules (boundary, broadest object,
confidence), a plain label list, the JSON format, then a block of worked
examples that teach the pattern by repetition — 4 ceiling, 3 wall, 1 floor,
2 furniture, 1 unknown. Deliberately **no** "never return X" / "invalid labels"
language; the pattern is carried entirely by the examples.

```text
Classify the main object inside the cyan boundary in this indoor crop.

Rules:
- Identify only the object inside the cyan boundary. Ignore anything outside it.
- Return the broadest / largest object that the boundary covers.
- Use confidence >0.90 only when the target is unmistakable.
- Use label VLM_unknown with confidence 0.0 only when the crop is unclear or truncated.

Labels to choose from (use the closest match; you may use a more specific indoor label if clearly warranted):
ceiling, wall, floor, door, window, partition, pillar,
office_chair, desk, sofa, cabinet, bookshelf, shelf, table,
whiteboard, monitor, lamp, air_vent, pipe, plant, rug, cup

Output exactly one JSON object:
{"label":"lowercase_object_name","label_confidence":0.0,"mobility_class":"static|dynamic|unknown","mobility_confidence":0.0}

Mobility:
- "dynamic": human, animal, self-propelled robot only
- "static": everything else
- "unknown": when label is unknown

Examples:
• Crop with recessed ceiling lights → {"label":"ceiling","label_confidence":0.92,"mobility_class":"static","mobility_confidence":0.99}
• Crop with embedded ceiling fixtures → {"label":"ceiling","label_confidence":0.91,"mobility_class":"static","mobility_confidence":0.99}
• Crop with ceiling and ventilation ducts → {"label":"ceiling","label_confidence":0.90,"mobility_class":"static","mobility_confidence":0.99}
• Crop showing ceiling panel with lights → {"label":"ceiling","label_confidence":0.92,"mobility_class":"static","mobility_confidence":0.99}
• Crop with wall and whiteboard → {"label":"wall","label_confidence":0.91,"mobility_class":"static","mobility_confidence":0.99}
• Crop with wall and mounted sign → {"label":"wall","label_confidence":0.90,"mobility_class":"static","mobility_confidence":0.99}
• Crop showing wall with air conditioner → {"label":"wall","label_confidence":0.89,"mobility_class":"static","mobility_confidence":0.99}
• Crop showing floor with a rug on it → {"label":"floor","label_confidence":0.90,"mobility_class":"static","mobility_confidence":0.99}
• Office chair filling the frame → {"label":"office_chair","label_confidence":0.91,"mobility_class":"static","mobility_confidence":0.99}
• Desk clearly visible → {"label":"desk","label_confidence":0.92,"mobility_class":"static","mobility_confidence":0.99}
• Blurry or truncated crop → {"label":"VLM_unknown","label_confidence":0.0,"mobility_class":"unknown","mobility_confidence":0.0}

Return only the JSON object. No markdown, explanations, or extra text.
```

---

## v6_structural_priority — VERBATIM (current live prompt)

Source: `src/rsg/config/rsg_pipeline.yaml:331-385`. This is the frozen final
prompt and the one currently in production.

```text
This is an INDOOR object classification task. Classify the main object within the cyan boundary.

PRIORITY: Look for common INDOOR STRUCTURAL FEATURES first
These large, primary structures are always present in indoor spaces:
- ceiling (large horizontal surface at top)
- wall (large vertical surface)
- floor (large horizontal surface at bottom)
- door (opening in wall)
- window (opening in wall/structure)
- pillar/column (vertical structural support)
- partition/divider (structural divider)

Classification Strategy:
1. Identify ONLY objects clearly inside the cyan boundary
2. Ignore objects OUTSIDE the cyan boundary completely
3. Classify the BROADEST/PRIMARY/LARGEST object visible
   - Structural features (ceiling, wall, floor) take priority over details/fixtures
   - Furniture takes priority over mounted objects/attachments
   - When a surface contains details/fixtures/attachments, classify the SURFACE not the details
   - When multiple objects compete, choose the larger/more dominant one

Output Format (required):
{"label":"lowercase_object_name","label_confidence":0.0-1.0,"mobility_class":"static|dynamic|unknown","mobility_confidence":0.0-1.0}

Confidence Calibration:
- 0.90+: Unmistakable, high visual salience, no ambiguity
- 0.70-0.90: Clear classification, minor uncertainty
- 0.50-0.70: Plausible but uncertain, competing interpretations
- <0.50: Very uncertain or ambiguous
- 0.0: Unknown or incomprehensible crop

Mobility Classification:
- "dynamic": Only for human, animal, self-propelled moving objects
- "static": All stationary objects (surfaces, furniture, fixtures, structures)
- "unknown": When label is unknown

EXAMPLES (demonstrating the strategy - prioritize structural features):

STRUCTURAL FEATURES (high priority):
• Crop showing recessed lights embedded in ceiling surface → {"label":"ceiling","label_confidence":0.92,"mobility_class":"static","mobility_confidence":0.99}
• Crop showing wall with mounted whiteboard or sign → {"label":"wall","label_confidence":0.89,"mobility_class":"static","mobility_confidence":0.99}
• Crop showing floor with rug or carpet → {"label":"floor","label_confidence":0.90,"mobility_class":"static","mobility_confidence":0.99}
• Crop showing ceiling panel with embedded fixtures and vents → {"label":"ceiling","label_confidence":0.91,"mobility_class":"static","mobility_confidence":0.99}
• Crop showing door frame and opening → {"label":"door","label_confidence":0.93,"mobility_class":"static","mobility_confidence":0.99}

FURNITURE (secondary priority):
• Office chair occupying most of frame → {"label":"office_chair","label_confidence":0.91,"mobility_class":"static","mobility_confidence":0.99}
• Desk clearly visible → {"label":"desk","label_confidence":0.92,"mobility_class":"static","mobility_confidence":0.99}
• Sofa or couch → {"label":"sofa","label_confidence":0.90,"mobility_class":"static","mobility_confidence":0.99}

AMBIGUOUS OR UNCLEAR:
• Blurry, truncated, or unidentifiable crop → {"label":"VLM_unknown","label_confidence":0.0,"mobility_class":"unknown","mobility_confidence":0.0}

Return only the JSON object. No markdown, explanations, or additional text.
```
