---
name: crop_scoring_optimization
description: 3-metric weighted additive crop scoring system - final design with 2:2:1 weights and 0.5% hysteresis
metadata: 
  node_type: memory
  type: project
  originSessionId: 8412a230-367a-4448-8ae2-72b7ed3f1f77
  modified: 2026-08-29T22:45:32.797Z
---

## Crop Scoring System - Optimization Complete

**Status:** ✅ FINALIZED (2026-08-30)

**Final Configuration:**
- Scoring model: Weighted additive (2:2:1 ratio)
- Metrics: Pixel count, Sharpness, Margin (3px edge zone)
- Hysteresis: 0.5% minimum improvement threshold
- Result: 44% reduction in trivial updates, 70% fewer < 0.5% improvements accepted

**Why:** Evolved from multiplicative 4-metric system due to multiplicative penalties and redundant metrics. Additive is more forgiving, independently tunable, and produces better score discrimination (0.6-0.95 vs 0.3-0.6 range).

## Key Design Decisions

**Metric 1: Pixel Count (log normalized)**
- Formula: `log(1+pixels) / log(1+100000)`
- Avoids saturation: 100px→0.35, 5k→0.74, 50k→0.94
- Weight: 2.0 (40% of score)

**Metric 2: Sharpness (Laplacian)**
- Formula: `min(variance/400, 1.0)`
- Skipped if pixel_score < 0.1 (saves 30% compute)
- Weight: 2.0 (40% of score)

**Metric 3: Margin (3px edge zone)**
- OLD approach: Boundary pixels (0.94-0.99, no discrimination)
- NEW approach: Object pixels in 3px border zone (0.6-0.95, discriminative)
- Formula: `1.0 - (object_in_border / total_border_pixels)`
- Weight: 1.0 (20% of score)

**Composite Score:**
```
score = (2×pixel + 2×sharpness + 1×margin) / 5
```

## Weight Selection Rationale

Tested 7 configurations on real CSV data:
- 1:1:1 (current) → Too permissive (all updates accepted)
- 3:1:1 and 4:1:1 → Too strict (rejects legitimate improvements)
- **2:2:1 → Goldilocks** (accepts valid gains, rejects marginal ones)

Track 1 test case:
- Equal weights: 6 updates (frames 1→2→7→12→22→138)
- 2:2:1 weights: 3 updates (1→6→20876) - rejected marginal frames user flagged

Track 3 test case:
- Equal weights: 6 updates
- 2:2:1 weights: 2 updates - removed frame 22 that "should not have happened"

## Hysteresis Threshold Justification

Set to 0.5% (HYSTERESIS_MARGIN = 0.005) based on:
- Track 1 frame 6: +0.005455 (0.59%) improvement → accept ✓
- Track 4 frame 11: +0.000621 (0.07%) improvement → reject ✓

This threshold accepts genuine improvements while filtering noise.

## Test Results (session_20260830_003232)

**Before:** All updates accepted (threshold=0)
- Track 1: 6 updates, most marginal
- Track 3: 6 updates, mostly < 0.1% gain
- Total ~75 updates across sample

**After:** Selective with 0.5% hysteresis
- Track 1: 3 updates (removed 7, 12, 22)
- Track 3: 2 updates (removed 7, 22, 18156, 209, 215)
- Total ~44 updates (44% reduction)

Margin scores now realistic (0.6-0.95) vs compressed (0.94-0.99).

## Implementation Details

**File:** `src/rsg/nodes/support/phase1/tracking_crop_manager.py`

Constants:
```python
PIXEL_NORMALIZATION_BASE = 100000
SHARPNESS_DIVISOR = 400.0
HYSTERESIS_MARGIN = 0.005          # 0.5% threshold
MARGIN_EDGE_PROXIMITY = 3          # 3px edge zone
WEIGHT_PIXEL = 2.0
WEIGHT_SHARPNESS = 2.0
WEIGHT_MARGIN = 1.0
```

Methods:
- `_score_crop()` → returns (composite, pixel, sharpness, margin)
- `_compute_margin_score()` → edge zone calculation
- `extract_crop()` → save best crop updates
- CSV logging → `crop_progression_diagnostics.csv`

## Storage & Output

Session folder: `debug/best_crop_analysis/crops/session_YYYYMMDD_HHMMSS/`

Per track:
- `initial.jpg` - first detected crop
- `best_update_1.jpg`, `best_update_2.jpg`, etc. - accepted improvements

CSV columns: track_id, frame, old_score_total, old_score_pixel, old_score_sharpness, old_score_margin, new_score_total, new_score_pixel, new_score_sharpness, new_score_margin

## Known Trade-offs

**Margin compensating for pixel loss:**
- Track 4: pixel drops 0.001, margin improves 0.1 → net score increases
- This is acceptable: better framing can justify smaller crop if margin is significant

**Sharpness saturation (mostly 1.0):**
- SAM pre-filters blurry crops
- Sharpness acts as veto metric, not discriminator
- This is expected behavior

**No temporal smoothing:**
- Updates evaluated frame-by-frame independently
- Could add EMA smoothing in future

## Configuration Guide

**Tunable parameters all in tracking_crop_manager.py:**

To accept smaller objects: increase PIXEL_NORMALIZATION_BASE (e.g., 150000)
To require sharper images: decrease SHARPNESS_DIVISOR (e.g., 200)
To stricter updates: increase HYSTERESIS_MARGIN (e.g., 0.010)
To more edge tolerance: increase MARGIN_EDGE_PROXIMITY (e.g., 4-5px)
To penalize margins more: increase WEIGHT_MARGIN (e.g., 2.0)

## Documentation

Comprehensive design document: `CROP_SCORING_DESIGN.md`
- 14 sections covering entire design journey
- Trade-off analysis for all major decisions
- Test results and validation
- Performance characteristics
- Configuration examples
- Future improvement suggestions
