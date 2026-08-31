# Crop Scoring System Documentation

**Complete collection of design, implementation, and test results for the 3-metric weighted additive crop scoring system.**

**Final Configuration:** 2:2:1 weights (Pixel:Sharpness:Margin) with 0.5% hysteresis  
**Status:** ✅ FINALIZED (2026-08-30)  
**Test Session:** `session_20260830_003232_final`

---

## Quick Navigation

### 📋 For Designers / Decision-Makers
Start with: **01_DESIGN_DOCUMENT.md**
- Covers entire design journey from problem to solution
- Explains all trade-offs and design decisions
- Shows test results and validation

### 🔧 For Developers / Implementation
Start with: **SOURCE_CODE_REFERENCE.md**
- Code locations and line numbers
- All formulas with examples
- How to modify parameters
- Debugging tips

### ⚡ For Quick Reference
Start with: **02_QUICK_REFERENCE.md**
- Key decisions summarized
- Configuration guide
- Performance characteristics
- Test results overview

### 📊 For Data Analysis
Start with: **TEST_SESSION_RESULTS.csv**
- All crop scores and metrics
- Track-by-track analysis
- Update history with timestamps

---

## Document Guide

### 01_DESIGN_DOCUMENT.md (Main Document)
**14 sections, 6,500+ words**

1. Executive Summary
2. Problem Statement (why 4-metric multiplicative failed)
3. Design Iteration Process (why additive chosen)
4. Margin Score Evolution (3 iterations to get it right)
5. Weight Optimization (7-config simulation study)
6. Hysteresis Threshold Optimization (0.5% selection)
7. Final Implementation (code structure)
8. Performance Characteristics (1ms per crop)
9. Configuration Guide (all parameters explained)
10. Design Decisions & Trade-offs (reasoning for each)
11. **Validation & Testing (final session results)**
12. Known Limitations & Future Work
13. Usage & Integration
14. Key Takeaways
15. Appendix (Quick reference)

### 02_QUICK_REFERENCE.md
**Summary for future sessions**

- Key design decisions
- Configuration parameters
- Test results summary
- Trade-offs explained

### SOURCE_CODE_REFERENCE.md
**Exact code locations and formulas**

- File paths and line numbers
- All formula implementations
- How to modify parameters
- Debugging tips

### TEST_SESSION_RESULTS.csv
**Final test data**

- All 50+ tracked objects
- Every crop scored with 4 metrics (pixel, sharpness, margin, composite)
- Track-by-track results showing improvements
- Frame numbers and deltas

---

## Key Findings

### Problem
Original 4-metric multiplicative system:
- ❌ Multiplicative penalties devastating
- ❌ Redundant metrics (extent + margin)
- ❌ Scores compressed to 0.3-0.6 range
- ❌ No discrimination between crops

### Solution
3-metric weighted additive system:
- ✅ Independent metrics (no redundancy)
- ✅ Scores spread 0.6-0.95 range
- ✅ Clear discrimination between options
- ✅ Tunable weights for different priorities

### Final Design: 2:2:1 Weights
```
score = (2×pixel + 2×sharpness + 1×margin) / 5

Pixel:      40% of score (object prominence)
Sharpness:  40% of score (image quality)
Margin:     20% of score (framing quality)
```

### Results
**Test Session:** session_20260830_003232_final

| Metric | Value |
|--------|-------|
| Trivial updates rejected | ~70% |
| Track 1 updates | 6 → 3 (removed marginal frames) |
| Track 3 updates | 6 → 2 (removed frame 22) |
| Overall reduction | 44% fewer updates |
| Margin score discrimination | 0.6-0.95 (was 0.94-0.99) |

---

## How to Use

### 1. Understand the System
```
Start: 01_DESIGN_DOCUMENT.md
Read: Sections 1-7 (Problem → Implementation)
Time: ~20 minutes
```

### 2. Modify Configuration
```
File: src/rsg/nodes/support/phase1/tracking_crop_manager.py
Ref: SOURCE_CODE_REFERENCE.md (Constants section)
Changes: Edit weights, thresholds, normalization
Rebuild: colcon build --packages-select rsg
```

### 3. Test Changes
```
Run: ros2 launch rsg phase1_launch.py
Check: debug/best_crop_analysis/crops/session_*/crop_progression_diagnostics.csv
Analyze: TEST_SESSION_RESULTS.csv for baseline comparison
```

### 4. Troubleshoot
```
Problem: Too many trivial updates?
  → Increase HYSTERESIS_MARGIN (0.005 → 0.010)
  
Problem: Margin scores too high?
  → Increase MARGIN_EDGE_PROXIMITY (3 → 5)
  
Problem: Small objects rejected?
  → Increase WEIGHT_PIXEL or decrease PIXEL_NORMALIZATION_BASE
  
Reference: SOURCE_CODE_REFERENCE.md (Debugging section)
```

---

## File Organization

```
CROP_SCORING_DOCUMENTATION/
├── README.md                          (this file)
├── 01_DESIGN_DOCUMENT.md              (complete design journey)
├── 02_QUICK_REFERENCE.md              (quick lookup)
├── SOURCE_CODE_REFERENCE.md           (implementation details)
├── TEST_SESSION_RESULTS.csv           (final test data)
└── [sample crops optional]
```

---

## Key Parameters

**All in:** `src/rsg/nodes/support/phase1/tracking_crop_manager.py` (lines 26-40)

```python
# Normalization
PIXEL_NORMALIZATION_BASE = 100000
SHARPNESS_DIVISOR = 400.0

# Thresholds
SHARPNESS_SKIP_THRESHOLD = 0.1
HYSTERESIS_MARGIN = 0.005

# Metric configuration
DEFAULT_SHARPNESS = 0.5
MARGIN_EDGE_PROXIMITY = 3

# Weights
WEIGHT_PIXEL = 2.0
WEIGHT_SHARPNESS = 2.0
WEIGHT_MARGIN = 1.0
```

---

## Score Interpretation

| Score | Quality | Typical Case |
|-------|---------|--------------|
| 0.95+ | Excellent | Large, sharp, well-framed |
| 0.90-0.95 | Good | Decent on all metrics |
| 0.80-0.90 | Acceptable | One metric weaker |
| 0.70-0.80 | Poor | Multiple issues |
| < 0.70 | Reject | Marginal crop |

**Examples from Test Session:**
- Track 1 best: 0.948 (perfect framing, sharp, good size)
- Track 3 best: 0.938 (tight margin, but large sharp object)
- Rejected update: 0.875 (small improvement from 0.875, < 0.5% gain)

---

## Common Configuration Adjustments

### To Accept Smaller Objects
```python
PIXEL_NORMALIZATION_BASE = 150000  # Raise saturation point
WEIGHT_PIXEL = 2.0                 # Keep pixel importance
```

### To Require Better Framing
```python
WEIGHT_MARGIN = 2.0        # Double margin weight
MARGIN_EDGE_PROXIMITY = 5  # Stricter edge detection
```

### To Be More Selective
```python
HYSTERESIS_MARGIN = 0.010  # 1% improvement (was 0.5%)
WEIGHT_PIXEL = 3.0         # Penalize small objects
```

### To Accept Quality Variations
```python
SHARPNESS_DIVISOR = 600.0      # More forgiving sharpness
HYSTERESIS_MARGIN = 0.002      # Accept 0.2% improvements
```

See **SOURCE_CODE_REFERENCE.md** for detailed examples.

---

## Test Session Details

**Date:** 2026-08-30 00:32:32  
**Path:** `debug/best_crop_analysis/crops/session_20260830_003232_final/`

**Results:**
- 16 tracks analyzed (shown in CSV)
- 44 total best crop updates
- 2-4 updates per track average
- 70% of potential updates filtered by hysteresis

**Best Performing Tracks:**
- rsg_obj_000001: 3 updates (removed trivial improvements)
- rsg_obj_000003: 2 updates (removed frame 22 anomaly)
- rsg_obj_000007: 9 updates (all > 0.5% improvement)

See **TEST_SESSION_RESULTS.csv** for complete data.

---

## For Future Development

### Known Limitations
1. Margin not normalized by object size (small objects naturally score higher)
2. Sharpness saturates at 1.0 (most crops already sharp)
3. No temporal smoothing (frame-by-frame independent)
4. No absolute quality floor

### Proposed Improvements
1. Scale-aware margin: normalize by object bounding box size
2. Temporal EMA smoothing: exponential moving average across frames
3. Absolute quality gates: reject if any metric below threshold
4. Track-specific weights: learn weights per object type

See **01_DESIGN_DOCUMENT.md** Section 11 for full details.

---

## Support

**Questions about design?** → Read 01_DESIGN_DOCUMENT.md sections 1-6  
**Questions about implementation?** → Read SOURCE_CODE_REFERENCE.md  
**Questions about results?** → Analyze TEST_SESSION_RESULTS.csv  
**Questions about configuration?** → Read 02_QUICK_REFERENCE.md  

**All parameters explained:** SOURCE_CODE_REFERENCE.md (Constants section)  
**All formulas included:** SOURCE_CODE_REFERENCE.md (Formula References section)  
**Complete rationale:** 01_DESIGN_DOCUMENT.md (Design Decisions section)

---

**Documentation Version:** 1.0  
**Last Updated:** 2026-08-30  
**Status:** FINALIZED ✅

