# VLM Prompt Optimization - Comprehensive Test Report
**Date:** 2026-08-30  
**Status:** FROZEN - Ready for new model testing  
**Model Tested:** Qwen 3 VL 8B F16  

---

## Executive Summary

Systematic prompt optimization testing with Qwen 3 VL 8B F16 revealed a fundamental limitation: **the model prioritizes visual salience (prominent ceiling lights) over semantic hierarchy (ceiling surface) regardless of prompt design**.

**Key Finding:** No prompt engineering approach successfully eliminated the ceiling_light misclassification error. Accuracy ceiling appears to be ~50% with this model on this task.

---

## Complete Test Results

| Version | Approach | Accuracy | Ceiling Errors | Key Issue |
|---------|----------|----------|----------------|-----------|
| **v1_simplified** | Simple baseline | 46.9% | 9 | Over-focuses on details |
| **v2_refined** | Surface/fixture hierarchy | 47.5% | 11 | Hierarchy ignored |
| **v3_hierarchical** | Complex hierarchy + anti-examples | 38.2% | 15 | Complexity backfired |
| **v4_filtering** | Filtering rules | 96.7%* | 9+ unverified | Rules not applied |
| **v5_strategy** | Instructions-based (no restrictions) | 51.8% | 16 | Examples insufficient |
| **v6_structural_priority** | Explicit structural features | 50.0% | 14 | Salience overrides |

**\* v4's 96.7% was on only 30 verified samples; unverified rows showed 9+ failures**

---

## Detailed Error Analysis

### The Ceiling_Light Problem (Core Issue)

**Persistence Across All Versions:**
- v1: 9 errors (18% of errors)
- v2: 11 errors (WORSENED)
- v3: 15 errors (WORSENED)
- v4: 9+ errors (unverified)
- v5: 16 errors (most common error)
- v6: 14 errors (still dominant)

**Pattern:** VLM consistently returns "ceiling_light" when shown ceiling with embedded lights, regardless of prompt instructions.

**User Feedback (Repeated):**
> "The ceiling is a large panel which contains ceiling lights. The VLM is focusing on the ceiling light inside the crop instead of the full crop"

**Root Cause:** 
- Recessed/embedded lights are visually more salient than the broad ceiling surface
- VLM architecture prioritizes detail detection over semantic hierarchy
- No prompt formulation overrides this visual priority

---

### Other Persistent Issues

#### 1. Wall Misclassifications (Various Versions)
- air_conditioner → wall
- whiteboard → wall (when wall is visible)
- sign → wall
- air_vent → ceiling (misidentifying mounted objects)

**Pattern:** Objects mounted on surfaces get confused with the surface itself or vice versa.

#### 2. Unknown Object Overuse (v1, v2, v3)
- Many valid objects classified as "unknown_object"
- Excessive conservatism with confidence thresholds

#### 3. Boundary Violations (v1, v2, v3)
- Objects outside cyan boundary still being classified
- Office_chair → floor (when chair is outside boundary)
- Boundary emphasis not effective

---

## Prompt Evolution & Lessons Learned

### v1 → v2 (Marginal +0.6%)
**Change:** Added surface/fixture distinction  
**Result:** Ceiling_light errors INCREASED (9 → 11)  
**Lesson:** Abstract hierarchical rules don't override visual salience

### v2 → v3 (FAILED -9.3%)
**Change:** Explicit three-tier hierarchy + ❌/✅ anti-examples  
**Result:** Accuracy dropped to 38.2%  
**Lesson:** Complexity confuses the model; simpler prompts outperform elaborate ones

### v3 → v4 (Recovery)
**Change:** Return to simplicity + filtering rules  
**Result:** 96.7% on verified samples (but 9+ failures in unverified)  
**Lesson:** Filtering rules are read but not applied; VLM commits to "ceiling_light" before considering the filter

### v4 → v5 (Decline -45%)
**Change:** Remove label restrictions; pure instructions + examples  
**Result:** 51.8% accuracy  
**Lesson:** No label list = worse performance (VLM needs some guidance); but restricted lists hurt generalization

### v5 → v6 (No Improvement)
**Change:** Explicit structural feature prioritization  
**Result:** 50.0% accuracy (14 ceiling errors)  
**Lesson:** Even when told to prioritize structural features, VLM still chooses detail objects

---

## Why This Model Hits a Ceiling (~50%)

### Visual Salience Problem
- **Light fixtures** are high-contrast, detailed, visually prominent
- **Ceiling surface** is large but low-contrast, lacks edges
- VLM's visual encoding prioritizes sharp features (lights) over broad surfaces
- No prompt can overcome this architectural bias

### Semantic Understanding Limitation
- VLM doesn't fully grasp the concept "ceiling SURFACE contains ceiling LIGHTS"
- Model treats them as competing labels, not hierarchical
- Classification is output selection, not semantic reasoning

### Prompt Ceiling
- All prompt variations plateaued around 46-52% accuracy
- Best outcome: v1/v2 baseline at ~47%
- Attempts to improve made it worse (v3, v4 unverified)
- Current model appears to have hit its accuracy ceiling on this task

---

## Configuration Final State

### Prompt (v6_structural_priority)
```
- Explicitly mentions INDOOR context
- Lists common structural features (ceiling, wall, floor, door, etc.)
- Prioritizes structural features over details/fixtures
- Provides structured examples
- No label restrictions (allows generalization)
```

### System Configuration
- **Model:** Qwen3-VL-8B-Instruct-GGUF (F16)
- **Endpoint:** http://127.0.0.1:8000/v1/chat/completions
- **Crop Context:** 10% padding (tighter crops)
- **Boundary:** Cyan (0, 255, 255), 2 pixels thick
- **RAP:** Disabled (VLM-only testing)
- **Processing Time:** ~5,000-6,000 ms per crop

---

## Archived Sessions

All test sessions preserved for reference:

1. **VLM-Test-Session-v1_simplified-20260830-180100** (46.9%)
2. **VLM-Test-Session-v2_refined-20260830-190300** (47.5%)
3. **VLM-Test-Session-v3_hierarchical-20260830-200500** (38.2%)
4. **VLM-Test-Session-v4_filtering-20260830-210000** (96.7% verified / 9+ failures)
5. **VLM-Test-Session-v5_strategy-20260830-220000** (51.8%)
6. **VLM-Test-Session-v6_structural-20260830-230000** (50.0%)

---

## Recommendations for Future Testing

### 1. Try Stronger Vision Models
- GPT-4V (better semantic understanding)
- Claude 3.5 Vision (stronger reasoning)
- Gemini 2.0 (improved instruction following)

**Hypothesis:** Larger models with better semantic reasoning might overcome the visual salience problem.

### 2. Architectural Approaches (If Staying with Qwen)
- **Pre-processing:** Blur/darken ceiling lights in crop before sending to VLM
- **Post-processing:** Rule-based correction (if VLM says "ceiling_light", replace with "ceiling")
- **Ensemble:** Multiple crops (original + modified versions) with voting

### 3. System-Level Changes
- Reduce crop context padding below 10% (make lights less prominent relative to surface)
- Increase cyan boundary thickness further (make boundary edge detection more salient)
- Try different wavelength for boundary (not cyan - something that stands out more)

### 4. Task Reformulation
- Two-stage classification: First identify SURFACE TYPE, then fine-grained details
- Confidence thresholding: Require >0.9 confidence for detail objects
- Return multiple candidate labels with confidence distribution

---

## Thesis Contribution

### What We Learned About VLM Prompt Optimization

1. **Complexity Hurts:** Adding layers of rules/hierarchies/anti-examples reduced accuracy, not improved it
2. **Visual Salience Dominates:** No prompt can overcome architectural biases in feature attention
3. **Semantic Hierarchy Doesn't Translate:** Abstract concepts (surface > fixture) don't map to VLM decision-making
4. **Simpler is Better:** v1 baseline (46.9%) performed comparably to optimized attempts
5. **Model Limitations Are Real:** ~50% accuracy appears to be ceiling for Qwen 3 VL 8B on this task

### Practical Implications

For robotic perception systems:
- Don't rely solely on prompt engineering for problematic object types
- Combine VLM with deterministic heuristics (e.g., size-based rules)
- Consider ensemble approaches or model selection based on input characteristics
- Invest in better models if accuracy is critical (try stronger VLMs)

---

## Conclusion

**Qwen 3 VL 8B F16 has hit a performance ceiling (~50%) on ceiling/ceiling_light classification.** The model prioritizes visual salience (detailed light fixtures) over semantic hierarchy (ceiling surface) regardless of prompt design. Six iterations of prompt engineering produced no significant improvement.

**Next steps:** Test with newer, more capable VLM models to determine if stronger semantic understanding can overcome this limitation.

---

**Report Date:** 2026-08-30  
**Status:** FROZEN - Ready for model upgrade  
**Author:** Fabin Fizel Sha  

