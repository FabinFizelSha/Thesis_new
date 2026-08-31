# v5_examples_based Prompt Strategy

**Date:** 2026-08-30  
**Approach:** In-context learning through examples (NOT explicit filters)

---

## Design Philosophy

Instead of telling the model what it cannot do:
```
❌ WRONG: "Never return ceiling_light"
❌ WRONG: "Invalid labels: ceiling_light, wall_sign, ..."
```

Show the model the pattern through multiple examples:
```
✅ RIGHT: 
  • Ceiling with lights → {"label":"ceiling",...}
  • Ceiling panel with lights → {"label":"ceiling",...}  
  • Ceiling with fixtures → {"label":"ceiling",...}
  • Wall with whiteboard → {"label":"wall",...}
  • Wall with sign → {"label":"wall",...}
```

**Why?** VLMs learn through pattern recognition in examples. Multiple examples of "ceiling+lights → ceiling" teaches the model to classify broad categories over details.

---

## v5_examples_based Prompt

**Structure:**
1. Simple critical rules (boundary, broadest object, confidence)
2. Label list (what to use)
3. JSON format
4. **Multiple examples showing the pattern:**
   - 4 ceiling examples (all return "ceiling")
   - 3 wall examples (all return "wall")
   - 1 floor example
   - 2 furniture examples
   - 1 unknown example

**Key ceiling examples:**
```
• "Crop with recessed ceiling lights" → ceiling
• "Crop with embedded ceiling fixtures" → ceiling
• "Crop with ceiling and ventilation ducts" → ceiling
• "Crop showing ceiling panel with lights" → ceiling
```

This pattern teaches: "No matter what fixtures are in/on the surface, classify the SURFACE."

**Wall examples:**
```
• "Crop with wall and whiteboard" → wall
• "Crop with wall and mounted sign" → wall
• "Crop showing wall with air conditioner" → wall
```

Pattern: "Return the surface when it's the main object, even with attachments."

---

## Expected Behavior

By seeing 4 ceiling examples all returning "ceiling", the model learns:
- Broad categories (ceiling) take priority
- Details/fixtures don't change the surface classification
- This is the consistent pattern

**No explicit "never return" or "invalid labels"** - just the learned pattern from examples.

---

## Session Summary

| Version | Approach | Result | Reason |
|---------|----------|--------|--------|
| v1 | Simple baseline | 46.9% | Clean but no detail handling |
| v2 | Abstract hierarchy | 47.5% | Marginal (+0.6%) |
| v3 | Complex hierarchy + anti-examples | 38.2% | Complexity backfired |
| v4 | Filtering rules | 96.7% verified / 9+ failures | Rules not applied by VLM |
| v5 | In-context examples | TESTING | VLM learns pattern, no rigidity |

---

## Why Examples Work Better

1. **No contradiction** - Not telling VLM "don't do X" while showing it X
2. **Pattern learning** - VLMs excel at in-context pattern recognition
3. **Flexible** - Model can adapt if context is ambiguous
4. **Proven** - Few-shot prompting is gold standard for LLMs
5. **Natural** - Aligns with how VLMs actually learn

---

## Ready for Testing

Run v5 with 300-second test. Target: Eliminate ceiling_light errors through learned pattern recognition.

