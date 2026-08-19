# SAM Backend Comparison Study - Cleaned Workspace

## Overview

Comprehensive thesis evaluation comparing ViT-B and NanoSAM SAM backends for object segmentation.

**Key Result**: ViT-B shows +12.3% higher F1 score but is 3.4x slower than NanoSAM.

**Phase Status**: Phase 1 ✅ Complete | Phase 2-3 📋 Pending (see EXPERIMENT_CHECKLIST.md)

## Files and Directories

### Documentation (Updated as work progresses)
- `README.md` - This file (quick reference)
- `FINAL_RESULTS_SUMMARY.md` - Complete thesis results with methodology and findings
- `EXPERIMENT_CHECKLIST.md` - **Living progress tracker** (update this as Phase 2-3 work proceeds)
- `PHASE_1_DETAILED_JOURNEY.md` - **Unofficial detailed log** of issues faced, fixes applied, and lessons learned

### Phase 1 Comparison (`phase_1_comparison/`)

#### Core Implementation
- `comparison_runner.py` - Main test orchestrator
- `runners.py` - ViT-B and NanoSAM backend implementations  
- `ground_truth.py` - IoU-based evaluation metrics
- `timing.py` - Nanosecond-precision latency measurement
- `recorder.py` - Metrics storage and CSV export

#### Configuration Files (`configs/`)
- `vitb_strict.yaml` - ViT-B high-precision (16×16 grid, 4000px threshold)
- `vitb_medium.yaml` - ViT-B balanced (6×6 grid, 8000px threshold)
- `vitb_loose.yaml` - ViT-B real-time (3×3 grid, 12000px threshold)
- `nanosam_strict.yaml` - NanoSAM high-precision (identical params to ViT-B)
- `nanosam_medium.yaml` - NanoSAM balanced (identical params to ViT-B)
- `nanosam_loose.yaml` - NanoSAM real-time (identical params to ViT-B)

#### Dataset (`datasets/`)
- `phase1_frames/` - 20 TESSE frames with RGB, depth, and semantic ground truth
- `dataset_metadata.json` - Frame metadata and provenance

#### Results (`results/`)
- `phase1_full_300frames/` - Final comprehensive test results
  - `metrics.csv` - Per-frame metrics for 300 frames × 6 configs = 1800 inferences
  - `summary.json` - Aggregate statistics by backend and complexity level

## Quick Start

### Run Full Comparison (300 frames)
```bash
cd phase_1_comparison
python comparison_runner.py \
  --dataset datasets/phase1_frames \
  --output results/test_run \
  --backends vitb nanosam \
  --levels strict medium loose \
  --max-frames 300
```

### View Results
```bash
cd results/phase1_full_300frames
# metrics.csv - raw per-frame data
# summary.json - aggregated statistics
```

## Key Findings

| Metric | ViT-B | NanoSAM | Winner |
|---|---|---|---|
| **F1 Score** | 0.352 | 0.314 | ViT-B (+12.3%) |
| **Precision** | 0.329 | 0.296 | ViT-B (+11.1%) |
| **Recall** | 0.557 | 0.511 | ViT-B (+9.0%) |
| **Speed** | 0.11 fps | 0.38 fps | NanoSAM (3.4x) |

### By Complexity Level

- **STRICT**: ViT-B +13.1% F1 (3.4x slower)
- **MEDIUM**: ViT-B +22.3% F1 (3.0x slower) ← Largest advantage
- **LOOSE**: ViT-B +2.2% F1 (6.4x slower)

## Methodology

- **Dataset**: 300 frames from TESSE uHumans2 office_s1_00h_v2
- **Evaluation**: IoU-based ground truth matching (≥0.3 threshold)
- **Depth filtering**: 0.3-6m valid range (applied to both masks and ground truth)
- **Total inferences**: 1,800 (300 frames × 6 configurations)
- **Configuration consistency**: Identical parameters per complexity level

## Thesis Conclusion

ViT-B SAM demonstrates superior accuracy (+12.3% F1) but at 3.4× computational cost. NanoSAM excels at real-time performance. Backend selection is a speed-accuracy tradeoff: choose ViT-B for accuracy-critical applications, NanoSAM for real-time deployment.

## Workspace Size

- **Total**: 67 MB (from ~25 GB originally)
- **Breakdown**: 
  - Datasets: 67 MB (20 frames)
  - Results: minimal (metrics CSVs)
  - Code: ~50 KB

---
**Last Updated**: 2026-08-14
**Status**: Thesis-Ready ✅
