# VLM Prompt Optimization Experiment - Thesis Report

**Date Created:** 2026-08-30  
**Experiment Phase:** Baseline v1 (VLM-only, RAP disabled)  
**Author:** Fabin Fizel Sha

---

## 1. Executive Summary

This document records the complete VLM prompt optimization experiment designed to improve object classification accuracy for robotic perception. The experiment uses a systematic approach to test and evaluate different VLM prompts, starting with the current production prompt as a baseline.

**Objective:** Establish baseline VLM performance metrics and iteratively improve prompt formulation through controlled testing and manual verification.

---

## 2. Experimental Setup

### 2.1 System Configuration

**Test Environment:**
- ROS 2 framework running on Jetson Orin
- VLM Backend: Qwen OpenAI-compatible HTTP endpoint
- RAP (Visual Retrieval): **DISABLED** for this test
- All object classifications route through VLM only

**Configuration Changes:**
```yaml
# rsg_pipeline.yaml
rap:
  enabled: false        # Disabled for VLM-only evaluation
vlm:
  enabled: true         # All crops go to VLM
  mode: qwen_http
  active_profile: qwen3_vl_8b_f16
```

### 2.2 Crop Processing

**Context Padding:** 10%  
- SAM bounding box + 10% padding on all sides
- Tight context focusing on the target object

**Boundary Marking:** Cyan contours (0, 255, 255)  
- Applied when crop becomes best (once-off, not per-frame)
- Helps visualize object boundaries for manual verification

**Crop Format:**
- Saved as JPEG
- RGB color space
- Full VLM output logged (label, confidence, mobility_class, mobility_confidence)

### 2.3 Diagnostic Logging

**Automatic Data Capture:**
- Every VLM call is logged to CSV
- Crop images saved for visual inspection
- VLM processing time measured (perf_counter precision)
- Full pipeline output preserved (label confidence, mobility classification)

**Output Location:**
```
/home/student/rsg_ros2_ws/VLM-Test-Session/
├── crops/                  # Crop images (obj_XXXXXX_crop.jpg)
└── vlm_results.csv         # Results (auto-filled VLM predictions)
```

### 2.4 Manual Verification

**Process:**
1. Run test for ~300 seconds (wall time, manual timeout)
2. View crop images from `crops/` folder
3. Fill CSV columns:
   - `manual_label` - Actual object class (ground truth)
   - `manual_is_correct` - true if VLM matched, false if wrong
   - `manual_notes` - Optional observations

**Target Sample Size:** 50 samples minimum (more for higher confidence)

---

## 3. Prompt Engineering

### 3.1 Baseline Prompt (v1_simplified)

**Official Baseline Prompt (Simple & Direct):**

```
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

**Characteristics:**
- **Length:** ~200 words (concise, focused)
- **Specificity:** Clear instructions aligned with system design
- **Output Format:** Structured JSON with label, confidence, mobility_class, mobility_confidence
- **Boundary Reference:** Direct cyan boundary identification (matches our implementation)
- **Context Usage:** Allows context to help identification (practical improvement)
- **Design:** Clean, production-ready specification

**Key Improvements Over Original:**
- ✅ Direct cyan boundary reference (instead of complex grayscale logic)
- ✅ Allows context to aid identification (more realistic)
- ✅ Simpler, more readable instructions
- ✅ Maintains strict JSON output format
- ✅ Keeps confidence thresholds and mobility rules
- ✅ Removes unnecessary architectural surface complexity

### 3.2 Iteration 2: v2_refined (Boundary-Aware with Surface/Fixture Distinction)

**Rationale:** v1_simplified baseline showed 46.9% accuracy with two main failure modes:
1. VLM over-analyzes fine details (ceiling_light instead of ceiling)
2. VLM classifies objects outside the cyan boundary

**Key Improvements in v2_refined:**
- Explicit instruction: "IGNORE any objects... OUTSIDE the cyan boundary"
- Surface vs. Fixture distinction: ceiling ≠ ceiling_light, wall ≠ whiteboard
- Better confidence guidance (reduce unknown_object overuse)
- Examples showing surface classification and boundary respect

**Expected Impact:** Target 60-70% accuracy by addressing boundary confusion and detail over-focus.

### 3.3 Future Prompt Versions (to be tested)

#### v3_improved
```
"You are an object recognition system for robotic manipulation.
Identify the main object in the image. Be specific with object type.
Reply ONLY with: lowercase object name. If unclear, reply: unknown_object"
```

#### v3_structured
```
"Analyze this robot perception crop. Identify the main object.
Provide: 1) object name (lowercase), 2) confidence (0-1), 3) reasoning (one sentence)
Format: {name: STRING, confidence: 0-1, reason: STRING}
Fallback: {name: unknown_object, confidence: 0.0, reason: EXPLAIN}"
```

#### v4_contextual
```
"This is a robot perception crop from an office environment.
Identify the main object considering context clues.
Answer: OBJECT_NAME (lowercase)
Examples: chair, desk, lamp, wall, door, etc.
If unsure: unknown_object"
```

---

## 4. VLM Output Format

**Fixed Structure** (preserved from pipeline):
```json
{
  "label": "tennis_ball",
  "label_confidence": 0.92,
  "mobility_class": "graspable",
  "mobility_confidence": 0.87,
  "success": true,
  "validation_status": "accepted",
  "raw_response": "..."
}
```

**Fields Used for Evaluation:**
- `label` - VLM's predicted object class
- `label_confidence` - VLM's confidence in the label (0-1)
- `mobility_class` - Grasp classification (graspable, static, unknown)
- `mobility_confidence` - Confidence in mobility classification
- `vlm_processing_time_ms` - Inference latency (measured with perf_counter)

---

## 5. Evaluation Methodology

### 5.1 Metrics

**Accuracy:**
- Overall: correct_predictions / verified_samples
- By Confidence: Stratified accuracy (high >70%, low ≤70%)
- False Positives: Incorrect predictions with high confidence
- False Negatives: Missed predictions despite confident wrong answer

**Performance:**
- Average VLM processing time (milliseconds)
- Min/max inference latency
- Percentiles (p50, p95, p99)

**Error Analysis:**
- Common failure modes
- Confidence vs. correctness correlation
- Object-type-specific accuracy (chairs vs. walls vs. small objects)

### 5.2 Statistical Significance

**Sample Size Guidance:**
- **20 samples:** Quick baseline, ~±22% confidence interval (95%)
- **50 samples:** Standard test, ~±14% confidence interval
- **100 samples:** High confidence, ~±10% confidence interval

---

## 6. Test Runs

### Test Run 1: VLM-Only Baseline (v1_simplified)

**Status:** ✅ COMPLETED  
**Date Completed:** 2026-08-30 18:01 UTC  
**Prompt Version:** v1_simplified (clean, direct cyan boundary reference)  
**RAP Enabled:** No (disabled for VLM-only evaluation)  
**Test Duration:** ~300 seconds (wall time)

**System Configuration:**
```yaml
rap:
  enabled: false              # RAP disabled - VLM only
vlm:
  enabled: true
  prompt: [v1_simplified above]
  model: qwen3_vl_8b_f16
crop:
  vlm_crop_context_ratio: 0.10  # 10% padding
  boundary_color: (0, 255, 255) # Cyan
```

**Diagnostic System:**
- ✅ Crops saved to: `/home/student/rsg_ros2_ws/VLM-Test-Session/crops/`
- ✅ Results logged to: `/home/student/rsg_ros2_ws/VLM-Test-Session/vlm_results.csv`
- ✅ VLM processing time tracked (milliseconds)
- ✅ Full JSON output preserved

**Results:**
- **Samples Classified:** 51
- **Samples Verified:** 49/50 (1 unverified)
- **Overall Accuracy:** 46.9% (23/49 correct)
- **Processing Time Avg:** ~5,900 ms (5.9 seconds per crop)

**Error Analysis:**
- **Ceiling light misclassification:** 9 errors (VLM says "ceiling_light" when it should be "ceiling")
  - Root cause: VLM focuses on fine details instead of main surface
- **Unknown object failures:** 12 errors (too conservative with unknown classification)
- **Boundary violations:** 5+ noted (VLM classifying objects outside cyan boundary)

**Key Finding:** v1_simplified's boundary and surface/fixture rules were not followed effectively by the VLM. The model consistently over-focuses on visual details (lights, signs) over broad surface classification.

---

### Test Run 2: v2_refined (Boundary-Aware with Surface/Fixture Distinction)

**Status:** ✅ COMPLETED  
**Date Completed:** 2026-08-30 19:03 UTC  
**Prompt Version:** v2_refined  
**Rationale:** Address ceiling_light confusion with explicit surface/fixture hierarchy

**Results:**
- **Samples Classified:** 41
- **Samples Verified:** 40/41 (1 unverified)
- **Overall Accuracy:** 47.5% (19/40 correct)
- **Processing Time Avg:** ~5,500 ms
- **Change from v1:** +0.6% (marginal, not significant)

**Error Analysis:**
- **Ceiling light misclassification:** 11 errors (WORSENED from 9)
  - Despite explicit "surface vs fixture" instructions, VLM still returns "ceiling_light"
  - User feedback repeated: "VLM focusing on ceiling light inside crop instead of full crop"
- **Unknown object failures:** 10 errors (slight improvement from 12)
- **Same boundary issues persist**

**Critical Finding:** v2_refined's approach to surface/fixture distinction was ineffective. The explicit instructions to "classify SURFACE not FIXTURES" and the hierarchical list did NOT override VLM's tendency to focus on visual details. The model appears unable to prioritize broad categories over fine details despite instructions.

---

### Test Run 3: v3_hierarchical (Hierarchical Classification with Anti-Examples)

**Status:** ✅ COMPLETED (FAILED)  
**Date Completed:** 2026-08-30 20:05 UTC  
**Prompt Version:** v3_hierarchical  
**Approach:** Hierarchical rules with three tiers + explicit anti-examples with ❌/✅

**Results:**
- **Samples Classified:** 55
- **Samples Verified:** 55
- **Overall Accuracy:** 38.2% (21/55 correct)
- **Change from v2:** -9.3% ⚠️ **CRITICAL DETERIORATION**

**Error Analysis:**
- **Ceiling light errors:** 15 (WORSENED from 11 in v2)
- **Unknown object errors:** 6 "unknown → wall" (new failure pattern)
- **Boundary violations:** Still persisting

**Critical Finding:** The hierarchical prompt with anti-examples made performance WORSE, not better. Complex rules confuse the model. The model cannot override visual salience (prominent ceiling lights) with abstract semantic hierarchies, even with explicit anti-examples.

**Lesson:** Simpler prompts outperform complex ones:
- v1 (simple): 46.9% ✅
- v2 (moderate): 47.5% ✅  
- v3 (complex): 38.2% ❌

---

## 7. Thesis Report Structure

This experiment will contribute to the thesis with the following sections:

### 7.1 Introduction
- Problem: Object classification in robotic systems
- Current state: RAP retrieval + VLM fallback
- Gap: Prompt engineering not systematically evaluated
- Contribution: Systematic VLM prompt optimization

### 7.2 Methods
- System architecture (Section 2: Experimental Setup)
- Baseline prompt (Section 3.1: Current Production Prompt)
- Prompt design iterations (Section 3.2: Future Versions)
- Evaluation framework (Section 5: Evaluation Methodology)

### 7.3 Results
- Baseline performance (Test Run 1)
- Accuracy metrics by confidence level
- Processing time analysis
- Failure mode breakdown

### 7.4 Discussion
- Prompt effectiveness across object types
- Trade-offs: accuracy vs. inference latency
- Confidence calibration analysis
- Recommendations for production

### 7.5 Conclusion
- Key findings
- Practical implications
- Future work (model selection, multi-modal context, etc.)

---

## 8. Implementation Notes

### 8.1 Code Components

**Diagnostic System:**
- `vlm_test_diagnostics.py` - Crop saving + output logging
- `vlm_accuracy_report.py` - Metrics calculation + report generation

**Integration Points:**
- `phase1.py` (_vlm_loop, lines ~2540-2570) - Timing measurement + logging call
- `rsg_pipeline.yaml` (line 265) - RAP enabled/disabled flag

### 8.2 Data Preservation

**All test data is preserved:**
- Crop images (never overwritten)
- CSV results (manual_* columns editable)
- Session timestamps (reproducible)

**Between Tests:**
- Move old `VLM-Test-Session/` to `VLM-Test-Session-v1_backup/`
- Create fresh session for next prompt version
- Maintains audit trail for thesis

---

## 8. Comparative Results Summary

| Metric | v1_simplified | v2_refined | v3_hierarchical | Trend |
|--------|---------------|-----------|-----------------|-------|
| **Accuracy** | 46.9% | 47.5% | **38.2%** | ⚠️ Deteriorating |
| **Ceiling_light errors** | 9 | 11 | **15** | ⚠️ Worsening |
| **Unknown errors** | 12 | 10 | Many new patterns | ⚠️ Chaotic |
| **Boundary violations** | 5+ | persisted | Persisted | ⚠️ Unresolved |
| **Processing time (ms)** | 5,900 | 5,500 | ~5,800 | Stable |
| **Samples** | 49 | 40 | 55 | Consistent |

**Critical Finding:** Adding complexity WORSENS performance. Hierarchy + anti-examples confuse the model rather than guide it. **v4_filtering** returns to simplicity with targeted filtering rules instead of abstract hierarchies.

---

## 9. Learning from Failures

### Why v2_refined Failed (Marginal +0.6%)
1. **VLM prioritizes visual features over instructions** - When it sees ceiling_lights prominently, it labels them regardless of "ignore fixtures" guidance
2. **Abstract hierarchies don't override visual salience** - Telling VLM "ceiling > ceiling_light" conceptually doesn't overcome the visual salience of lights
3. Ceiling_light errors: 9 → 11 (+22% worsening)

### Why v3_hierarchical Failed CRITICALLY (-9.3%)
1. **Complexity worsens performance** - Adding three-tier hierarchy actually reduced accuracy from 47.5% to 38.2%
2. **Anti-examples did NOT help** - Using ❌/✅ symbols and explicit "do NOT return X" made things worse
3. **Unknown_object errors spiked** - New failure pattern: many "unknown → wall" misclassifications
4. **Ceiling_light errors worsened again** - 11 → 15 errors

### Critical Principle: Simplicity > Complexity
**Finding:** VLMs perform better with simple, direct instructions than with elaborate rules
- v1 (simple): 46.9% ✅
- v2 (moderate): 47.5% ✅
- v3 (complex): 38.2% ❌

### v4_filtering Strategy (Return to Simplicity + Targeted Filters)
- **Revert to v1/v2 simplicity** - Remove complex hierarchies
- **Add filtering rules** - Specific "If you see X → return Y" for known errors
- **Boundary rule FIRST** - Objects outside boundary have top priority to ignore
- **Emphasize PRIMARY/BROADEST category** - Not "ignore details" but "return the widest category"
- **No abstract reasoning required** - Just concrete decision rules

---

## 10. Future Tests (To Be Scheduled)

- [x] Test Run 1: v1_simplified (46.9% - baseline)
- [x] Test Run 2: v2_refined (47.5% - surface/fixture distinction, minimal improvement)
- [x] Test Run 3: v3_hierarchical (38.2% - FAILED, complexity worsened performance)
- [ ] Test Run 4: v4_filtering (READY - simplicity + targeted filtering rules)
- [ ] Comparative analysis (accuracy vs. latency trade-offs)
- [ ] Final recommendation (best prompt for production)

---

## 10. References & Resources

**Configuration Files:**
- `/home/student/rsg_ros2_ws/rsg_pipeline.yaml` - VLM/RAP settings
- `/home/student/rsg_ros2_ws/src/rsg/nodes/support/phase1/phase1_config.py` - Default prompt

**Testing Guide:**
- `/home/student/rsg_ros2_ws/VLM-TESTING-GUIDE.md` - Quick start workflow

**Code Locations:**
- VLM integration: `src/rsg/nodes/phase1.py` (lines 2538-2575)
- Diagnostics: `src/rsg/nodes/support/phase1/vlm_test_diagnostics.py`
- Reporting: `src/rsg/nodes/support/phase1/vlm_accuracy_report.py`

---

## Appendix A: VLM Processing Time Analysis

**Baseline Performance (Qwen 3 VL 8B F16):**
- Typical inference: 4-6 seconds per crop
- Bottleneck: HTTP round-trip latency + model inference
- Opportunity: Could optimize with batching or faster model variants

**For Thesis:**
- Include processing time as secondary metric (not just accuracy)
- Discuss accuracy-latency trade-off in conclusion
- Consider multi-model strategy for future work

---

## Appendix B: Crop Examples Analysis

**Common Object Types in Test Data:**

From baseline test run (office environment):
- Static objects (walls, floors, ceiling, doors): 97%+ accuracy expected
- Furniture (chairs, sofas, desks): 90%+ accuracy expected
- Small objects (phones, cups, papers): 70-85% accuracy expected
- Challenging: Partially occluded, unusual angles, ambiguous small objects

**For Thesis:**
- Stratify accuracy by object category
- Discuss why certain types are harder
- Recommend prompt improvements targeting weak categories

---

**Document Version:** 1.0 (Baseline v1 setup)  
**Last Updated:** 2026-08-30  
**Next Review:** After Test Run 1 completion


### Test Run 4: v4_filtering (Simplified + Targeted Filtering)

**Status:** ✅ COMPLETED (PARTIAL SUCCESS)  
**Date Completed:** 2026-08-30 21:00 UTC  
**Prompt Version:** v4_filtering  
**Approach:** Return to simplicity with explicit "If X → return Y" filtering rules

**Results (Verified Only):**
- **Samples Verified:** 30/58 (52% complete)
- **Verified Accuracy:** 96.7% (29/30 correct) ⚠️ Only verified samples
- **Unverified samples show:** ceiling_light still appearing in rows 1,2,14,16,19,28,32,41,44

**Critical Finding:** Filtering rules NOT applied by VLM!
- Despite instruction "If ceiling_light → return ceiling"
- VLM still returns "ceiling_light" in 9+ unverified rows
- Wall misclassifications persist (air_conditioner→wall, unknown→wall)
- Boundary violations still present (office_chair→floor)

**Root Cause:** VLM commits to "ceiling_light" classification and does NOT apply post-output filtering. The instruction is read but ignored when visual evidence is strong.

**Lesson:** Filtering rules applied to OUTPUT don't work. Must BLOCK invalid labels BEFORE VLM chooses them.

---

### Test Run 5: v5_explicit_ceiling (MANDATORY Blocking Rules)

**Status:** READY TO TEST  
**Date Started:** [TO BE FILLED]  
**Prompt Version:** v5_explicit_ceiling  
**New Approach:** Block invalid labels BEFORE they can be selected
- Explicit list of INVALID labels (ceiling_light, wall_sign, lamp, etc.)
- MANDATORY RULE: "Never return ceiling_light - always return ceiling"
- Multiple ceiling-specific examples
- Wall rules blocking "wall_sign" outputs

**Expected Change:** Force VLM to select from valid label set only

