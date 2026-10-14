# Phase 2: Persistent Object Tracking Weight Optimization Experiment

## Overview

This directory contains a comprehensive report of Phase 2 optimization work for persistent object tracking in the RSG (Robotic Spatial Graph) system. The experiment focused on optimizing tracking association weights to consolidate fragmented object detections caused by SAM segmentation imperfections.

## Contents

### 1. **EXPERIMENT_REPORT.md** (Main Document)
Comprehensive 40+ section report covering:
- Executive summary with key results
- Background on SAM limitations
- 5-phase optimization journey
- Objects 2/8 case study (detailed root cause analysis)
- Technical architecture and implementation details
- Tracking modes (Recent vs Revisit)
- Score components and calculation
- Final frozen configuration
- Results and observations
- Lessons learned
- Future work recommendations

**Key Insight**: The critical breakthrough was identifying that same-object-different-angle consolidation requires optimization of RECENT mode weights (not revisit), because objects appearing < 8 seconds apart use different matching logic.

### 2. **FINAL_CONFIG.yaml**
Frozen configuration with detailed comments explaining every parameter:
- Matching thresholds (min_score values)
- Mode transition logic (continuation_max_age_sec)
- Recent mode weights (0.70, 0.15, 0.45)
- Revisit mode weights (0.60, 0.25, 0.35)
- Spatial overlap gates
- Distance gates
- Global association logic
- Depth padding parameters

Each parameter includes explanation of:
- Current value
- Previous value (for context)
- Why it was changed
- What it controls

### 3. **OPTIMIZATION_TIMELINE.md**
Run-by-run optimization history with:
- Run 1-8 progression showing results at each step
- Key transitions and why they happened
- Failed approaches (what didn't work and why)
- Critical threshold discovery (0.30 min_score)
- Optimal weight ratios (70:15:45)
- Timeline of when each decision was made
- Table showing critical thresholds for objects 2/8

## Results Summary

### Quantitative Improvements
- **Track reduction**: 86 → 64 tracks (26% improvement)
- **Test duration**: 300 seconds (real-world rosbag playback)
- **Crop consolidation**: Objects from multiple angles now merge into single tracks

### Specific Problem Cases Fixed
- **Objects 2/8**: Same crop, different angles → NOW CONSOLIDATED
- **Objects 1/9**: Same crop, different angles → NOW CONSOLIDATED
- **Objects 3/15**: Same crop, different angles → NOW CONSOLIDATED
- **Objects 13/14**: Same crop, different angles → NOW CONSOLIDATED

### Qualitative Results
- **HYDRA visualization**: Visually acceptable spatial mapping
- **Fragmentation**: Significantly reduced
- **False positives**: Minimal (BB overlap priority prevents accidental merges)
- **Computational efficiency**: No degradation

## Key Technical Insights

### 1. Mode-Based Matching
The tracking system uses two matching modes:
- **Recent mode** (objects < 8s since last update): Uses spatial overlap + visual similarity
- **Revisit mode** (objects > 8s dormancy): Uses re-identification logic

Most optimization attempts failed because we were tuning revisit weights while the problem objects used recent mode.

### 2. Critical Parameters

**global_recent_min_score: 0.30**
- Allows marginal same-object-different-angle matches (score ~0.40-0.45)
- Lowered from 0.5
- Critical for objects 2/8 consolidation

**global_recent_weight_historical: 0.70**
- Prioritizes spatial/BB overlap (3D ground footprint)
- Increased from 0.55
- Objects from different angles occupy same space

**global_recent_weight_image: 0.45**
- Visual similarity emphasis
- Increased from 0.25
- Different angles may have variable visual features

**continuation_max_age_sec: 8.0**
- Time window before track transitions to dormancy
- Reduced from 15.0
- Forces re-evaluation sooner if initial match fails

### 3. Weight Balance
- Recent mode normalized: Historical 54% | Image 35% | Centroid 11%
- Prioritizes spatial overlap over distance penalty
- Works better than equal weight distribution (33:33:33)

## How to Use This Report

### For Understanding the Problem
→ Read **EXPERIMENT_REPORT.md** sections 1-2 (Executive Summary, Background)

### For Understanding the Solution
→ Read **EXPERIMENT_REPORT.md** sections 4-5 (Journey, Case Study)

### For Technical Implementation Details
→ Read **EXPERIMENT_REPORT.md** section 6 (Technical Details)

### For Configuration Reference
→ See **FINAL_CONFIG.yaml** (with inline comments)

### For Optimization History
→ Read **OPTIMIZATION_TIMELINE.md** (run-by-run progression)

## Citation

If using these weights or optimization methodology in future work, reference:

```
Phase 2: Persistent Object Tracking Weight Optimization
RSG System - Robotic Spatial Graph
Experiment Date: 2026-08-27
Track Reduction: 26% (86 → 64 tracks)
Configuration: Recent mode weight rebalancing + threshold reduction
```

## Contact & Questions

For questions about specific optimization choices or implementation details, refer to:
- **OPTIMIZATION_TIMELINE.md** for run-by-run decisions
- **EXPERIMENT_REPORT.md** section on Objects 2/8 for root cause analysis
- **FINAL_CONFIG.yaml** for parameter documentation

## Future Optimization Opportunities

1. **Adaptive weights**: Adjust based on object type or tracking confidence
2. **Multi-hypothesis tracking**: Track multiple hypotheses, merge later
3. **Learned scoring**: Use neural network instead of hand-tuned weights
4. **SAM post-processing**: Improve mask consistency across views

See **EXPERIMENT_REPORT.md** Future Work section for detailed recommendations.

