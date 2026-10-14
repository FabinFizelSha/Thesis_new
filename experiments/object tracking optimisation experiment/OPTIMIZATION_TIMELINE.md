# Phase 2 Optimization Timeline

## Run 1: Baseline (0.75 revisit_min_score)
**Date**: 2026-08-24
**Config**: Standard weights, high thresholds
**Result**: 86 tracks, 191 crops
**Observation**: Objects 2/8, 1/9, 3/15, 13/14 fragmenting

## Run 2: Lowered Revisit Threshold (0.75 → 0.40)
**Date**: 2026-08-24
**Config**: global_revisit_min_score: 0.40
**Result**: 93 tracks (WORSE - regression)
**Issue**: Problem objects still not merging
**Learning**: Wrong approach - threshold alone insufficient

## Run 3: Increased Image Weight in Revisit (0.00 → 0.35)
**Date**: 2026-08-24
**Config**: global_revisit_weight_image: 0.35
**Result**: 92 tracks (worse than 0.40 alone)
**Issue**: Weight imbalance creating more fragmentation
**Learning**: Need holistic weight rebalancing

## Run 4: Extended Time Window Test (15.0s)
**Date**: 2026-08-25
**Config**: continuation_max_age_sec: 15.0 (reverted)
**Result**: 67-71 tracks (mixed results)
**Issue**: Still fragmenting despite long time window
**Learning**: Time window too long delays revisit logic

## Run 5: Critical Discovery - Recent vs Revisit Modes
**Date**: 2026-08-25 (afternoon)
**Analysis**: 
- Objects 2/8 appear 0.5s apart
- 0.5s << 15.0s → USE RECENT MODE
- We were optimizing REVISIT weights (wrong mode!)
- Recent weights have low image weight (0.25)
**Insight**: MUST optimize RECENT mode for close-in-time objects

## Run 6: Target Recent Mode (0.40 threshold + img weight 0.35)
**Date**: 2026-08-25
**Config**: 
- global_recent_min_score: 0.40
- global_recent_weight_image: 0.35
- continuation_max_age_sec: 15.0
**Result**: 79 tracks (improvement!)
**Progress**: Moving in right direction

## Run 7: Even Lower Recent Threshold (0.30)
**Date**: 2026-08-26
**Config**:
- global_recent_min_score: 0.30 (critical!)
- global_recent_weight_image: 0.35
- continuation_max_age_sec: 8.0 (reduced)
**Result**: 64 tracks (26% improvement!)
**Breakthrough**: Problem objects FINALLY consolidating
**Why it worked**: 
- Lower threshold allows marginal matches
- Shorter time window forces re-eval sooner
- Recent mode emphasis on visual + spatial

## Run 8: BB Overlap Priority (Final Tuning)
**Date**: 2026-08-27
**Config**:
- global_recent_weight_historical: 0.70 (was 0.55)
- global_recent_weight_centroid: 0.15 (was 0.20)
- global_recent_weight_image: 0.45 (was 0.35)
**Result**: 63 tracks (maintained 26% improvement)
**Tuning**: Confirmed BB overlap priority is correct approach
**Weights normalize to**: Historical 54% | Image 35% | Centroid 11%

## Final Configuration
**Frozen**: 2026-08-27 23:30 UTC
**Track reduction**: 86 → 63-64 (26% improvement)
**Test stability**: Consistent 63-65 track range across multiple runs
**Consolidation**: Objects 2/8 merging verified in visual inspection

---

## Key Transitions

### Transition 1: Revisit → Recent Mode Focus
**When**: Run 5 - Critical discovery
**Why**: Realized problem objects use recent mode, not revisit
**Impact**: Changed optimization strategy from revisit-only to recent-focused

### Transition 2: High Threshold → Low Threshold
**When**: Run 6-7 - Lowered thresholds progressively
**Why**: Same-object matches score ~0.40-0.45, need threshold ≤ 0.30
**Impact**: Allowed marginal matches to pass, consolidated fragments

### Transition 3: Long Time Window → Short Time Window
**When**: Run 7 - Reduced continuation_max_age_sec
**Why**: 15s too long, delays revisit re-evaluation
**Impact**: Faster track re-identification on re-appearance

### Transition 4: Centroid Balance → BB Overlap Priority
**When**: Run 8 - Rebalanced weights
**Why**: Distance penalty was over-weighted relative to spatial overlap
**Impact**: Objects from different angles now reliably consolidate

---

## Critical Thresholds for Objects 2/8

| Config | Recent Min Score | Weights (H:C:I) | Result |
|--------|-----------------|-----------------|--------|
| Baseline | 0.50 | 0.55:0.20:0.25 | Separate |
| Run 2 | 0.40 | (revisit) | Separate |
| Run 6 | 0.40 | 0.55:0.20:0.35 | Separate |
| Run 7 | 0.30 | 0.55:0.20:0.35 | **Consolidated** |
| Run 8 | 0.30 | 0.70:0.15:0.45 | **Consolidated** |

**Critical threshold**: 0.30 min_score
**Optimal weight ratio**: 70:15:45 (historical:centroid:image)

---

## Failed Approaches (Learning)

1. **Just lower revisit threshold** ❌
   - Objects 2/8 never reach revisit mode
   - Wrong optimization target

2. **Increase image weight alone** ❌
   - Without threshold reduction, still too low
   - Weight imbalance creates new problems

3. **Extend time window** ❌
   - Delays revisit logic
   - Objects stay in recent mode longer
   - Recent mode weights are the problem

4. **Equal weight distribution** ❌
   - 0.33:0.33:0.33 doesn't work
   - Recent mode needs BB overlap > distance
   - 0.70:0.15:0.45 works better

---

## Successful Approaches (Summary)

1. ✅ **Identify correct mode** (Recent, not Revisit)
2. ✅ **Lower recent threshold** (0.5 → 0.30)
3. ✅ **Increase recent image weight** (0.25 → 0.45)
4. ✅ **Reduce time window** (15.0 → 8.0s)
5. ✅ **Prioritize BB overlap** (historical 0.70)
6. ✅ **Reduce distance penalty** (centroid 0.15)

---

## Lessons for Future Optimization

1. **Understand mode logic first**
   - Don't optimize blindly
   - Track which mode objects use
   - Some objects may need recent, others revisit tuning

2. **Score analysis is critical**
   - Calculate actual scores for problem cases
   - Identify threshold gap (score - threshold)
   - Design fixes to close the gap

3. **Weight ratios matter more than absolute values**
   - 0.70:0.15:0.45 works (normalized: 54:11:35)
   - 0.55:0.20:0.25 doesn't (normalized: 55:20:25)
   - Redistribution more important than individual increases

4. **Time windows create mode boundaries**
   - Short windows force re-evaluation sooner
   - Longer windows stick objects in recent mode
   - 8s is sweet spot for this dataset

5. **Test thoroughly between changes**
   - Run multiple times (RVIZ variability)
   - Check both quantitative (track count) and qualitative (visuals)
   - Verify specific problem cases (objects 2/8) improve

