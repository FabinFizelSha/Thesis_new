# VLM Prompt Optimization: v3 Failure & v4 Strategy

**Date:** 2026-08-30  
**Status:** v3 failed, v4_filtering ready for testing

---

## v3_hierarchical: Critical Failure Analysis

### Performance Collapse
- **v2 Accuracy:** 47.5% (19/40)
- **v3 Accuracy:** 38.2% (21/55)
- **Change:** -9.3% (WORSE, not better)

### Error Breakdown
| Error Type | v2 | v3 | Change |
|-----------|----|----|--------|
| Ceiling_light errors | 11 | **15** | +36% ⚠️ |
| Unknown object errors | 10 | Many | Complex patterns |
| Unknown → Wall pattern | - | **6** | New failure |

### Root Cause: Complexity Kills Performance

**What went wrong:**
1. Three-tier hierarchy confused the model
2. Anti-examples with ❌/✅ symbols made it worse (not better)
3. Adding 200+ words of rules reduced clarity
4. Visual salience (ceiling lights) still overrides written instructions

**Evidence from crop analysis:**
- User feedback: "VLM is focusing on ceiling light inside crop instead of full crop"
- Despite explicit anti-example: "❌ ceiling_light → Return ceiling"
- VLM still returns "ceiling_light" 36% more often than in v2

### Key Insight: Simpler Prompts Win

**Performance Trend:**
```
v1 (simple)          → 46.9% ✅
v2 (moderate)        → 47.5% ✅ (+0.6%)
v3 (complex)         → 38.2% ❌ (-9.3%)
```

**Principle:** Each added layer of complexity reduces accuracy. VLMs respond better to concise, direct instructions than elaborate hierarchies.

---

## v4_filtering: New Strategy

### Design Philosophy
**Return to simplicity. Add only what is necessary to fix known errors.**

Instead of:
- Complex hierarchies ❌
- Abstract reasoning ❌
- Multiple tiers ❌

Use:
- Simple baseline prompt ✅
- Explicit filtering rules ✅
- Boundary-first logic ✅
- Shortest possible instructions ✅

### v4_filtering Prompt

```yaml
prompt: |
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

### Key Differences from v3

| Aspect | v3_hierarchical | v4_filtering |
|--------|-----------------|--------------|
| **Lines of prompt** | 30+ | ~20 |
| **Hierarchy tiers** | 3 explicit | Implicit in list |
| **Anti-examples** | ❌/✅ symbols (30+ words) | Simple "if-then" rules |
| **Boundary handling** | Mixed in | **CRITICAL RULE #1** |
| **Label specification** | Abstract ranking | Simple preference list |
| **Tone** | Elaborate | Direct |

### Expected Improvements

**Target:** Return to >46% accuracy by removing complexity

**Mechanism:**
1. Filtering rules handle known errors (ceiling_light, wall_sign)
2. Simpler instructions reduce confusion
3. Boundary rule placed first ensures priority
4. "Broadest category" emphasis replaces hierarchy

**Stretch goal:** 50%+ if filtering rules actually work

---

## Session Archive

**v3_hierarchical backed up:**
- Location: `/home/student/rsg_ros2_ws/VLM-Test-Session-v3_hierarchical-20260830-200500/`
- 55 crops, 55 verified samples
- 38.2% accuracy (documented failure)
- Complete CSV with error analysis

---

## Next Steps

1. Update prompt with v4_filtering
2. Clear RAP memory
3. Run 300-second test
4. Verify 50+ samples
5. Compare results against v2 baseline

**Success criteria:** Get back to >46% accuracy (higher than v3's 38.2%)

