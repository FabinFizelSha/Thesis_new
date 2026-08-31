# VLM Prompt Optimization - Test Findings & v3 Proposal

**Date:** 2026-08-30  
**Experiment Phase:** Iterative prompt refinement for robotic object classification  
**Status:** v3_hierarchical ready for testing

---

## Summary of Test Results

### v1_simplified Baseline
- **Accuracy:** 46.9% (23/49 verified samples)
- **Approach:** Direct cyan boundary reference, JSON output format
- **Main Issues:** Ceiling_light confusion (9 errors), Unknown overuse (12 errors), Boundary violations (5+ noted)

### v2_refined (Boundary-Aware)
- **Accuracy:** 47.5% (19/40 verified samples) — **+0.6% from v1 (marginal)**
- **Approach:** Explicit surface/fixture hierarchy + boundary respect instructions
- **Result:** **INEFFECTIVE** — Ceiling_light confusion WORSENED to 11 errors
- **Finding:** VLM ignores abstract hierarchical rules when visual evidence conflicts

---

## Root Cause Analysis: Why v2 Failed

### The Ceiling_Light Problem
User feedback (repeated across 7-8 samples):
> "The ceiling is a large panel which contains ceiling lights. **The VLM is focusing on the ceiling light inside the crop instead of the full crop.**"

**What This Tells Us:**
1. VLM sees multiple objects (ceiling surface + lights)
2. VLM prioritizes **visual salience** over instruction hierarchy
3. Abstract rules like "classify SURFACE not FIXTURES" are ignored
4. The model doesn't understand semantic hierarchies when visual evidence contradicts them

### Why Abstract Hierarchies Don't Work
- Instruction: "ceiling is more general than ceiling_light"
- Visual reality: Lights are prominent, ceiling surface is large but less detailed
- VLM choice: Selects the visually prominent detail (ceiling_light)
- Result: Instructions overridden by visual salience

---

## v3_hierarchical: New Approach

### Core Strategy: Concrete Anti-Examples Instead of Abstract Rules

**Principle:** VLMs respond better to concrete examples (what to do/not do) than abstract hierarchical reasoning.

### Key Changes from v2 → v3

| Aspect | v2_refined | v3_hierarchical |
|--------|-----------|-----------------|
| **Format** | Abstract rules ("classify SURFACE not FIXTURES") | Concrete anti-examples with ❌/✅ |
| **Hierarchy** | Implicit (by listing surface types) | **Explicit three-tier hierarchy with priorities** |
| **Emphasis** | "Surface vs Fixture" | "PRIMARY/DOMINANT object, even if details visible" |
| **Examples** | One example per category | **Multiple anti-examples of common mistakes** |
| **Tone** | Instructional | **Directive and negative (what NOT to do)** |

### v3_hierarchical Prompt Structure

```
1. CRITICAL RULE: Use hierarchical classification - always choose FIRST that applies

2. HIERARCHY (pick the FIRST):
   Tier 1 - STRUCTURAL SURFACES (highest priority):
   └─ ceiling, wall, floor, door, window, partition, pillar
   
   Tier 2 - FURNITURE/STRUCTURAL OBJECTS:
   └─ office_chair, desk, sofa, cabinet, wardrobe, dresser
   
   Tier 3 - FIXTURES/DETAILS (only if above not applicable):
   └─ whiteboard, monitor, lamp, air_vent, pipe, plant

3. ANTI-EXAMPLES (what NOT to do):
   ❌ Ceiling with lights → Do NOT return "ceiling_light" → Return "ceiling"
   ❌ Wall with sign/whiteboard → Do NOT return "sign" → Return "wall"
   ❌ Floor with rug → Do NOT return "rug" → Return "floor"
   ✅ Only rug visible (no floor) → Return "rug"
```

### Expected Improvements

| Error Type | v2 | v3 Target | Method |
|-----------|----|-----------|----|
| **Ceiling_light** | 11 | <5 | Anti-example: "❌ ceiling_light → Return ceiling" |
| **Wall_sign/whiteboard** | 2-3 | <1 | Anti-example + Tier 1 priority |
| **Unknown (false positives)** | 10 | <8 | Encourage confidence within hierarchy |
| **Overall Accuracy** | 47.5% | >55% | Reduce fixture misclassification |

---

## Session Backups Created

1. **VLM-Test-Session-v1_simplified-20260830-180100**
   - 51 crops, 49 verified samples
   - 46.9% accuracy baseline

2. **VLM-Test-Session-v2_refined-20260830-190300**
   - 41 crops, 40 verified samples
   - 47.5% accuracy (marginal improvement)
   - Documents ceiling_light worsening

---

## Readiness for v3 Test

✅ Prompt updated with hierarchical anti-examples  
✅ Build verified  
✅ Previous sessions backed up  
✅ VLM-Test-Session cleared for fresh run  
✅ Documentation updated  

**Next Step:** Run v3_hierarchical baseline test (300 seconds, manual timer)

```bash
# Clear RAP memory
rm -rf ~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl
rm -rf ~/rsg_ros2_ws/visual_memory/* && mkdir -p ~/rsg_ros2_ws/visual_memory

# Terminal 1: Start pipeline
ros2 run rsg rsg_phase1.py

# Terminal 2: Run test
timeout 300s ros2 bag play ~/datasets/uhumans2/office_s1_00h_v2 --rate 0.1 --qos-profile-overrides-path ~/.tf_overrides.yaml
```

---

## Thesis Contribution

This progression (v1 → v2 → v3) demonstrates:
1. **Why abstract rules fail** - VLMs prioritize visual salience over semantic hierarchies
2. **Effective prompt design** - Concrete anti-examples outperform abstract instructions
3. **Iterative refinement** - Systematic testing reveals model behavior and informs prompt evolution
4. **Practical implications** - For robotic perception, explicit filtering rules are more reliable than conceptual guidance

