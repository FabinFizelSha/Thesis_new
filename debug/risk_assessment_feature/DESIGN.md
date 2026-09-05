# Risk assessment feature — design

## 1. What this is

A second piece of per-object metadata, **Risk**, attached to every Hydra
object node in the fuser's output, alongside the existing label/mobility
metadata. Risk is a hazard assessment — what could go wrong with this
object, and how severe/likely — computed by a second, independent VLM call
(the "Risk VLM"), made only after an object's identity is already known.

Two examples that motivated the schema (from the feature request):

- A chemical bottle at the **edge** of a table: `risk_score ≈ 0.9`,
  `risk_factors = ["bottle falling from table edge", "chemical spillage",
  "toxic fume exposure"]` — falling is probable from an edge position.
- The same bottle in the **middle** of a table: `risk_score ≈ 0.6`,
  `risk_factors = ["chemical spillage", "toxic fume exposure"]` — spillage
  is still possible, but accidental falling is far less likely away from an
  edge, so the overall score is lower even though it's the same object.

The score has to combine *severity* (how bad if something goes wrong) with
*probability* (how likely, given the visible spatial context) — which is why
the Risk VLM is given the object's crop with surrounding context, not a
tightly-cropped image of the object alone, and is told to reason about the
whole scene, not just the object.

## 2. Two trigger points, one shared path

Phase 1 already classifies every track exactly once, via one of two routes:

1. **RAP hit** (`_process_rap_task`, `is_known` branch) — the track is
   resolved from RAP's retrieval memory without ever calling the
   object-detection VLM.
2. **RAP miss → object-detection VLM** (`_vlm_loop`, `msg.success` branch) —
   RAP didn't confidently recognize it, so the object-detection VLM
   classifies it instead.

Risk assessment is dispatched from *both* of these, immediately after
`_emit_semantic_label_result` publishes the classification — never before
(there's nothing to reason about risk from without a label), and never on a
classification failure (`unknown_object` never gets a risk assessment; see
Limitations). Both call sites converge on one shared method,
`_enqueue_risk_task`, so the two paths can't drift out of sync with each
other.

**No risk assessment on failed classification is a deliberate scope
decision, not an oversight.** An unidentified object could still be
hazardous purely from its position, but the two trigger points described in
the original request are both post-classification; adding a third trigger
for `unknown_object` tracks is a natural extension if wanted later, not
something this design precludes.

## 3. Why the crop has to be captured at enqueue time, not looked up later

The object-detection VLM's queue stores *track IDs*, and re-snapshots
whatever the current best crop is when that ID is actually dequeued — useful
there, because a better crop might arrive while the ID is waiting in the
FIFO.

Risk assessment doesn't have that luxury: `_emit_semantic_label_result`
calls `_retire_track_crop(track_id)` immediately after publishing the
classification result, which is also the moment `_enqueue_risk_task` gets
called. By the time the risk worker thread would service a byid lookup, the
crop registry entry is already gone. So the risk queue stores **the crop
array itself**, captured at enqueue time (`task.get("vlm_rgb_crop",
task.get("rgb_crop"))` — the same wide, context-padded crop that was just
used for classification), not a track ID to resolve later. This is why the
risk queue's full-queue behavior is simpler than the VLM queue's too: there's
no "wait for a better crop" concept to defer to, so a full queue just drops
the oldest pending task.

## 4. Why a separate model/server, and what "priority" means here

The Risk VLM is a genuinely different model/server from the object-detection
VLM (`risk_vlm_*` config, entirely independent of `vlm_*`) — its own
endpoint, its own queue (`risk_queue`), its own daemon thread (`_risk_loop`),
its own backend class (`RiskVlmBackend`). A slow or unreachable risk server
can only ever delay or drop risk results; it can never block classification,
tracking, or Hydra publishing, because the two pipelines share nothing at
runtime except (see below) a soft priority hint.

That hint exists because "separate model" doesn't necessarily mean "separate
hardware." On this Jetson deployment specifically, an earlier investigation
this session found NanoSAM, the object-detection VLM, and Hydra's mapping
backend all measurably contending for the same physical GPU even though
they're logically independent processes. If the risk server ends up on that
same GPU, dispatching risk requests while object-detection VLM requests are
in flight would silently slow classification down — the one thing this
feature must never do. So `_risk_loop` checks `vlm_queue.empty()` before
dispatching and backs off (`risk_vlm_yield_backoff_sec`, default 0.2s) while
object-detection has pending work. This is a **priority hint, not a
guarantee** — a risk call already in flight when an object-detection request
arrives still causes some contention. If the risk server is confirmed to run
on genuinely separate hardware, `risk_vlm_yield_to_object_vlm: false` removes
the throttle entirely.

**Why no adaptive tuning is needed**: RAP's hit rate improves as it
accumulates more images over a session (per phase1's own "RAP scheduling:
immediate_async_once_per_track" design), so object-detection VLM traffic
*naturally decreases* as a run goes on — fewer RAP misses need it.
`vlm_queue` sits empty more often on its own as a result, and risk
throughput rises without this loop needing to know anything about RAP's hit
rate directly. The yield rule stays correct as the ratio between the two
kinds of traffic shifts over time.

## 5. Schema and validation

```jsonc
{
  "risk_score": 0.9,           // signed float, clamped to [-1.0, 1.0]
  "risk_factors": ["...", ...] // short strings, count- and length-capped
}
```

**`risk_score` is signed, not just `[0.0, 1.0]`**: negative means the object
actively *reduces* risk (safety equipment -- a fire extinguisher, a safety
rail; hazard-warning signage -- a "Wet Floor" or "High Voltage" sign), 0.0 is
neutral (no hazard, nothing mitigating), positive is a hazard. Clamped by a
dedicated `_normalise_risk_score` (`vlm_result.py`), deliberately separate
from `_normalise_confidence` (used for `label_confidence`/
`mobility_confidence`, always `[0, 1]`) -- reusing that function would have
silently clamped every negative score to 0, discarding exactly the signal
this extension exists to capture. No percentage-notation handling for risk
scores (unlike confidences): a model returning e.g. `-150` almost certainly
ignored the range instruction, and a straight clamp to `[-1, 1]` is simpler
and safer than guessing a percentage scale for a signed value. The fuser's
`handleRiskResult` clamps to the same `[-1.0, 1.0]` range when parsing the
published JSON, independently (no shared code between the Python validator
and the C++ fuser, by the same design principle used everywhere else in this
codebase — see `analyze_contact_diagnostics.py`'s independent-oracle
rationale).

The prompt also now explicitly instructs the model to fold a **smaller
component visible within the main object's crop** into the main object's own
assessment — e.g. a warning sign mounted on a ceiling should affect the
*ceiling's* `risk_score` and be named in its `risk_factors`, not be silently
dropped just because the ceiling is the larger, primary subject. This uses
the existing schema as-is (no new field): `risk_factors` already accepts any
short string, so "hazard warning sign visible" slots in next to genuine
hazards like "pipe leak" without any structural change.

`validate_risk_response` (`vlm_result.py`, next to the existing
`validate_vlm_response`) parses this from the model's raw text. A missing or
unparsable `risk_score` fails the whole result — every downstream consumer
needs that field to be trustworthy. An **empty `risk_factors` list is not a
failure** on its own: a genuinely low-risk object (a cushion on the floor)
can legitimately have no hazards to report. `risk_factors` is capped
(`max_risk_factors`, `max_risk_factor_length`) so a verbose or malformed
model response can never grow the fuser's per-node metadata or an RViz label
without bound. A failed validation (or a transport/HTTP error, or an empty
crop) degrades to "no risk metadata published for that track" — never a
crash, never a retry storm.

**A non-zero score with no stated reason is also rejected.** Real session
data (`debug/risk_assessment_feature/session_20260905_142129/risk_results.csv`)
turned up two accepted results with `risk_score: 0.1` and `risk_factors: []`
(`fire_hose_reel`, `air_vent`) — a number with no justification behind it,
despite the prompt already asking the model to always name what's driving a
score away from neutral. `validate_risk_response` now enforces this as a hard
rule rather than relying on the prompt alone: any response where
`abs(risk_score) > 1e-9` (a small epsilon, only to tolerate float
representation noise — not a "close enough to neutral" allowance) and
`risk_factors` is empty is rejected (`validation_status="rejected"`,
`validation_reason="nonzero_score_missing_factors"`), degrading to "no risk
metadata for that track" exactly like any other validation failure. Exactly
`0.0` with empty factors is still accepted — that's the legitimate
"nothing to report" case. The rejected `risk_score` is preserved on the
`ValidatedRiskResult` (not zeroed out) purely so it still shows up in the
diagnostics CSV for debugging, even though it's never published to the
fuser. This is defense-in-depth: the prompt (below) was also strengthened
with an unmissable "MANDATORY" rule at the very top, in the same spot the
object-detection prompt puts its most critical formatting rule, but the
hard validation check is what actually guarantees the invariant regardless
of whether the model follows instructions.

**Prompt** (`phase1_config.py`'s `risk_vlm_prompt` default, editable via
`phase1.risk_vlm.prompt` in YAML) went through two rounds of revision after
the first real runs, in response to review:

- **Focus on the cyan-boundary object, not the surroundings.** The same
  crop-marking convention used for object-detection (a cyan boundary drawn
  around the target) applies here; the prompt now says explicitly that the
  surroundings are only for judging *position/context*, never the subject of
  the assessment itself.
- **The classification label is generic — inspect the image, don't just read
  the label.** `{label}` comes from an earlier, separate classification step
  (RAP or the object-detection VLM) and carries no condition information: the
  same `"floor"` label covers a bone-dry floor and a freshly-mopped one. The
  prompt now says this explicitly and instructs the model to visually inspect
  condition (wetness, damage, obstruction, wear) rather than pattern-match on
  label text.
- **Examples now match this pipeline's actual object vocabulary.** The
  original three examples (chemical bottle, cushion) were all small tabletop
  items — not representative of what this pipeline actually classifies
  (`floor`/`wall`/`ceiling`/`pipes`/`sofa`, per real earlier fuser runs).
  Added same-label/different-condition pairs (dry vs. wet/polished floor;
  intact vs. corroded/leaking pipes) specifically to reinforce the
  label-is-generic point with domain-relevant cases, alongside the original
  bottle-position example (kept, since it's still the clearest illustration
  of position-driven probability).
- **Output-format hardening**, borrowing the exact proven wording from the
  object-detection prompt's own winning P3 revision (88% accuracy, 50/50
  clean bare JSON in the real experiment — see
  `debug/prompt_optimisation_experiment/EXPERIMENT_REPORT.md` §13): "the
  first character you output is `{` and the last is `}`," plus an explicit
  ban on preamble/analysis/narration sentences. Verified against the live
  server afterward — still returns clean bare JSON with the longer prompt,
  though only checked with synthetic solid-color test crops, not real hazard
  imagery (no annotated risk dataset exists yet, unlike the object-detection
  prompt which had 50 labeled crops per run to score against).
- **Literal output-format examples added** (not just prose description): a
  "VALID OUTPUTS LOOK EXACTLY LIKE THIS" block showing 4 complete example
  JSON completions verbatim, explicitly labeled as format-only so the model
  doesn't anchor on the specific numbers.
- **Signed score + risk-reducing objects + sub-component awareness** (second
  revision round, same day): added the "SCORE MEANING" and "SMALLER
  COMPONENTS WITHIN THE MAIN OBJECT" sections described in §5 above, plus
  matching examples (fire extinguisher on a wall, warning sign on a
  ceiling). This was a real schema change, not just prompt wording — see §5
  for `_normalise_risk_score` and the fuser clamp-range update that had to
  ship alongside it.
- **"MANDATORY: non-zero needs a factor" rule added at the top**, right
  after OUTPUT RULE — the most emphatic spot in the prompt. Prompted by the
  same real-run failures described in §5 (`fire_hose_reel`/`air_vent`
  accepted with `risk_score: 0.1` and no factors): the existing "always name
  them" sentence lived only inside SCORE MEANING and only covered the
  negative/safety-equipment case, so it never applied to small positive
  scores at all. Prompt reinforcement alone isn't relied on for correctness
  here — see the `validate_risk_response` hard check in §5, which is what
  actually rejects a non-zero/no-factors response regardless of whether the
  model reads this instruction.
- **Material-based hazards** (per user feedback that overall sensitivity was
  too low — real hazards like glass and sharp metal edges weren't being
  flagged): a new "MATERIAL-BASED HAZARDS" section, placed right after "THE
  LABEL IS GENERIC" since it's the same underlying instruction (inspect the
  actual object, not the label) applied to material/form instead of surface
  condition. Calls out brittle materials (glass, ceramic, thin acrylic) for
  their inherent breakage-into-sharp-shards risk even while intact, and
  metal objects (railings, shelving, ductwork, sheet-metal) for exposed or
  sharp edges/corners/torn metal, plus the added electrical-shock risk when
  bare/corroded metal is near moisture or exposed wiring. Two new worked
  examples added to EXAMPLES: an intact glass panel (moderate score, purely
  from breakage potential) vs. a visibly cracked one (high score), and a
  metal shelf/railing with a visible sharp/bent edge. These are additive to
  the existing surface-condition and position-based reasoning, not a
  replacement for it — an object can be flagged for both, e.g. a cracked
  glass panel near a walkway edge.

## 6. Delivery: a new topic, matched by slot ID, not track ID

Risk results publish on their own topic (`/rsg/objects/risk_result`, plain
`std_msgs/String` JSON, same pattern as `semantic_label_result` and
`rap_result`) rather than being folded into the existing semantic-label
message: risk arrives later, from an independent async call, and carries an
unrelated payload shape.

A track may own several local Hydra segments (phase 1 splits a long
physical object into multiple slots — see `persistent_object_tracker.py`).
Risk is computed once per *track*, then fanned out to every slot that track
owns, using the same segment-resolution helper
(`_semantic_segments_for_fanout`) that `_emit_semantic_label_result` already
uses for label fan-out — factored out specifically so the two fan-out paths
can't disagree about which slots a track owns.

The fuser (`fuser.cpp`) matches incoming risk results purely by
**`hydra_slot_id`**, mirroring the same key `rsg_presence` already uses —
never by `persistent_track_id`. This matters for correctness: this
session's earlier loop-closure work can merge two track identities into one
survivor, retiring the old track ID while its Hydra slots live on unchanged.
Matching by slot ID means a risk result that happens to arrive after such a
merge still lands on the right node; matching by track ID could silently
drop it. `RiskOverlay` is deliberately simpler than `SemanticOverlay` for
the same reason labels need: there's no centroid-fallback matching or
multi-candidate tie-breaking, because risk is one-shot — a slot either has a
result or it doesn't.

## 7. Limitations

- **Shared-GPU risk despite "separate model."** See §4 — the yield throttle
  is a mitigation, not an isolation guarantee. True isolation requires the
  deployment topology to be confirmed (separate GPU or separate machine).
- **One-shot, first-crop-only.** If the crop at classification time is a
  poor view for judging spatial risk (e.g. the table edge isn't visible from
  that angle), that track's risk assessment is permanently based on that one
  view. This is the explicitly requested lifecycle (one-shot per track, not
  periodic re-evaluation) — not an accident.
- **No risk on failed classification** (§2). An object that never gets
  identified never gets a risk assessment either, even though an
  unidentified object could still be positionally hazardous.
- **No auto-launched risk server.** Unlike `phase1.vlm`, which can spawn its
  own llama.cpp server via a `profiles`/`server` block, `phase1.risk_vlm`
  has no equivalent — the configured endpoint is expected to already be
  running, launched independently. Adding auto-launch support is a natural
  follow-up, not part of this change.
- **Malformed VLM output degrades gracefully**, per §5 — never a crash, just
  "no risk metadata for that track" that cycle.

## 8. Auto-launching the Risk VLM server

Originally the Risk VLM endpoint was expected to already be running,
launched independently (§7's "no auto-launch" limitation). That's since been
addressed: `rsg_risk_vlm_server` (`scripts/rsg_risk_vlm_server`) mirrors
`rsg_vlm_server` (the object-detection VLM's launcher) exactly, but reads
`phase1.risk_vlm.server` directly instead of resolving a `phase1.vlm.profiles`
entry -- risk has no profile-switching system, it's always one dedicated
model/server, so there's nothing to select between. Both scripts share the
same shape: read `server.provider`/`binary`/`model_path`/`mmproj_path`/
`host`/`port`/`gpu_layers`/`context_size`/`extra_args` from YAML, validate
`min_system_memory_gib` against `/proc/meminfo`, build the llama.cpp command
line, and `os.execv` into it (so ROS2 launch tracks the actual llama-server
process directly, no wrapper left behind).

Wired into `rsg_full_stack.launch.py` as a second `ExecuteProcess`
(`risk_vlm_server`, alongside the existing `qwen_vlm_server`), gated by a new
`start_risk_vlm` launch argument (default `true`, passed through
`rsg_all.launch.py` the same way `start_qwen` already is) -- so `ros2 launch
rsg rsg_all.launch.py` now starts both VLM servers automatically, on their
own configured ports (8000 for object-detection, 8100 for risk), the same
way it always has for the object-detection one.

**Model choice**: `phase1.risk_vlm.server` is configured for Qwen3.5-4B
(`/home/student/rsg_models/qwen3_5_4b/`) rather than matching
object-detection's 9B profile -- deliberately smaller, since this process
runs *concurrently* with the already-loaded object-detection model and adds
to total GPU memory/compute rather than replacing it. `min_system_memory_gib:
8` reflects that (vs. 16 for the 9B profile).

**Verified for real**: started `rsg_risk_vlm_server` standalone (not through
ROS2 launch, to avoid disturbing anything else running) with nothing else
using the GPU; it loaded the model and began listening on port 8100 in about
10 seconds. Then ran an actual `RiskVlmBackend.assess()` call against it (a
synthetic test image, not a real hazard scene) and got back a valid,
correctly-parsed JSON response (`risk_score: 0.0, risk_factors: []` for a
plain solid-color test crop -- a sensible answer given the input) in ~1.36s
of model inference time. Server stopped cleanly afterward. This confirms the
full chain end-to-end: config → launch script → real llama.cpp process →
real HTTP call → real model response → validation -- not just the
command-construction dry-run.

## 9. Diagnostics

`RiskVlmDiagnostics` (`risk_vlm_diagnostics.py`) mirrors the existing
`VLMTestDiagnostics` used for the object-detection VLM: every risk call —
successful, failed, or malformed — saves the exact crop it was given plus one
CSV row, to `debug/risk_assessment_feature/session_<timestamp>/`
(`crops/obj_NNNNNN_crop.jpg` + `risk_results.csv`). Logged unconditionally
(not just on success): a failed or malformed response is exactly the case
worth being able to inspect later. Gated behind the same
`phase1.diagnostics.enabled` master switch as every other diagnostic writer
in this node — off by default, zero cost on production runs. CSV columns:
`object_id, crop_filename, track_id, hydra_slot_id, frame_timestamp, source,
label, mobility_class, risk_score, risk_factors, risk_inference_ms,
end_to_end_ms, success, validation_status, raw_response`.

## 10. Verification performed

- `validate_risk_response`: unit-tested directly (normal case, percentage
  notation, missing score, missing factors, malformed JSON, cap
  enforcement, and the `nonzero_score_missing_factors` rejection added in
  §5 — positive non-zero + empty factors rejected, negative non-zero +
  empty factors rejected, non-zero + non-empty factors accepted, exact
  zero + empty factors still accepted, float noise near zero tolerated).
- `RiskVlmBackend`/`DummyRiskVlmBackend`/`make_risk_vlm_backend`: unit-tested
  with the HTTP layer mocked (empty crop, successful call with prompt
  substitution, malformed model response, transport exception) — none of
  these ever raise out of `assess()`.
- `OpenAICompatibleVlmBackend.identify()`: re-verified unchanged after the
  shared-HTTP-helper refactor (behavior-preserving check).
- `Phase1Config.from_yaml`: verified the real `rsg_pipeline.yaml` loads
  correctly with the new `risk_vlm:` block, and that `object_geometry` (the
  section immediately after it) still loads correctly too.
- Dispatch logic in `phase1.py` (`_enqueue_risk_task`, `_risk_loop`,
  `_publish_risk_result`, `_semantic_segments_for_fanout`): exercised
  directly (unbound method calls against a lightweight fake node, no real
  ROS2 node/executor) covering: disabled-flag no-op, missing-crop/missing-
  slot guards, exact-crop capture, multi-segment fan-out with de-duplication,
  drop-oldest behavior on a full queue, the yield-to-object-VLM throttle
  (confirmed it withholds dispatch while `vlm_queue` is non-empty and
  dispatches once it drains), an end-to-end enqueue → assess → publish
  round trip producing the expected JSON payload, and that
  `RiskVlmDiagnostics.log_risk_result` is called for both successful and
  failed assessments.
- `RiskVlmDiagnostics`: unit-tested directly (CSV row content, crop file
  written, disabled-mode is a true no-op that creates no directory).
- `fuser.cpp`: compiles clean after each change (`colcon build
  --packages-select rsg`).
- **Not yet verified**: a real pipeline run with `risk_vlm.enabled: true`
  against a live (or dummy-mode) risk server, confirming `ros2 topic echo
  /rsg/objects/risk_result` produces exactly one message per classified
  track and the corresponding DSG node(s) carry `rsg_risk` metadata.
