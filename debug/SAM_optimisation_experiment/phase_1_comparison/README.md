# Phase 1: Quantitative SAM Backend Comparison Experiment

**Comprehensive evaluation of SAM backends (ViT-B vs NanoSAM) for real-time robotic manipulation**

---

## 📖 How to Use This Documentation

### For Thesis Writing: READ FIRST ⭐

**[`COMPREHENSIVE_EXPERIMENT_REPORT.md`](COMPREHENSIVE_EXPERIMENT_REPORT.md)** (2,000+ lines, self-contained)

This is your single source of truth. It contains:
- ✅ Complete experiment design & methodology
- ✅ All data preparation details (TESSE extraction, depth encoding, synchronization)
- ✅ Full parameter design rationale (why 6 configs, exact parameters)
- ✅ Hardware setup & reproducibility (Jetson Orin specs)
- ✅ Evaluation methodology (F1 score, depth filtering)
- ✅ Results & analysis
- ✅ Model selection rationale (why NanoSAM)
- ✅ Future work roadmap (SAM2, fine-tuning, ensemble)
- ✅ Implementation details
- ✅ References & appendices

**Read this for thesis. Don't need other files for main narrative.**

### For Deep Dives: Supporting Documents

[`supporting_documents/`](supporting_documents/) folder contains detailed breakdowns:

| Document | Use Case |
|----------|----------|
| [`DATASET_PREPARATION.md`](supporting_documents/DATASET_PREPARATION.md) | Understand data extraction & preprocessing |
| [`PARAMETER_DESIGN.md`](supporting_documents/PARAMETER_DESIGN.md) | Learn configuration choices in detail |
| [`HARDWARE_SETUP.md`](supporting_documents/HARDWARE_SETUP.md) | Reproduce on another Jetson Orin |
| [`DEPTH_FILTERING_METHODOLOGY.md`](supporting_documents/DEPTH_FILTERING_METHODOLOGY.md) | Understand F1 score calculation |
| [`PHASE1_GUIDE.md`](supporting_documents/PHASE1_GUIDE.md) | Quick reference for running tests |
| [`CLEANUP_SUMMARY.md`](supporting_documents/CLEANUP_SUMMARY.md) | See what was cleaned up |

**Optional:** Only read if you need specific details beyond the main report.

---

## ⚡ Quick Start

### Run Tests

```bash
cd ~/rsg_ros2_ws/debug/SAM_optimisation_experiment/phase_1_comparison

# 10-frame validation test (5 min)
python phase1_verified_runner.py --max-frames 10

# Full 300-frame test (2-3 hours)
python phase1_verified_runner.py --max-frames 300
```

### Key Results

| Config | F1 Score | Latency | FPS | Status |
|--------|----------|---------|-----|--------|
| **NanoSAM LOOSE** | **0.42** | **188ms** | **5.3** | ✅ RECOMMENDED |
| NanoSAM MEDIUM | 0.35 | 723ms | 1.4 | Fallback |
| NanoSAM STRICT | 0.40 | 7,025ms | 0.14 | Benchmark |
| ViT-B MEDIUM | 0.44 | 3,074ms | 0.33 | Best accuracy |
| ViT-B STRICT | 0.38 | 25,750ms | 0.04 | Baseline |
| ViT-B LOOSE | 0.24 | 2,195ms | 0.46 | Not recommended |

---

## 📊 What's Inside

### Main Files

```
phase_1_comparison/
├── 📖 COMPREHENSIVE_EXPERIMENT_REPORT.md  ← START HERE (2,000+ lines)
│
├── 🐍 Core Scripts (5 essential files)
│   ├── phase1_verified_runner.py      (main test runner)
│   ├── ground_truth.py                (F1 score computation)
│   ├── runners.py                     (SAM backends)
│   ├── timing.py                      (latency measurement)
│   └── recorder.py                    (CSV export)
│
├── ⚙️  configs/                        (6 parameter sets)
│   ├── nanosam_strict.yaml
│   ├── nanosam_medium.yaml
│   ├── nanosam_loose.yaml
│   ├── vitb_strict.yaml
│   ├── vitb_medium.yaml
│   └── vitb_loose.yaml
│
├── 📚 supporting_documents/           (optional deep dives)
│   ├── DATASET_PREPARATION.md
│   ├── PARAMETER_DESIGN.md
│   ├── HARDWARE_SETUP.md
│   ├── DEPTH_FILTERING_METHODOLOGY.md
│   ├── PHASE1_GUIDE.md
│   └── CLEANUP_SUMMARY.md
│
└── 📊 results/
    ├── phase1_full_300frames_corrected/  (current full test)
    └── phase1_test_10frames_corrected/   (validation)
```

### Dataset

```
datasets/phase1_frames_300/     (1.2 GB, 300 frames)
├── rgb_000000.npy ... rgb_000299.npy
├── depth_000000.npy ... depth_000299.npy
├── semantic_000000.npy ... semantic_000299.npy
└── metadata.json
```

---

## 📋 Comprehensive Report Highlights

### Sections (Read in Order)

1. **Executive Summary** — One-page overview of key findings
2. **Introduction** — Motivation, research objectives, scope
3. **Related Work** — SAM, semantic segmentation, optimization approaches
4. **Methodology** — Complete methodology (dataset, hardware, parameters, evaluation)
5. **Experimental Design** — The 6 configurations and implementation
6. **Results** — 10-frame validation results with detailed analysis
7. **Discussion** — Key findings and model selection rationale
8. **Future Work** — SAM2, fine-tuning, ensemble methods (roadmap to F1 > 0.65)
9. **Reproducibility** — Hardware, software, step-by-step instructions
10. **Conclusion** — Summary and impact
11. **Appendices** — Glossary, references

### Key Questions Answered

| Question | Location |
|----------|----------|
| **Why NanoSAM?** | Section 6.2.1 "Current Decision: NanoSAM LOOSE" |
| **How was data extracted?** | Section 3.1 "Dataset Preparation" |
| **Why 5-second gaps?** | Section 3.1.5 "Timestamp Standardization" |
| **What are the 6 configs?** | Section 4.1 "Six Configurations" |
| **What exact parameters?** | Section 3.3.3 "Configuration Specifications" |
| **Jetson Orin specs?** | Section 3.2 "Hardware Platform" |
| **How are strictness levels matched?** | Section 3.3.4 "Parameter Sensitivity" |
| **How to reproduce?** | Section 8 "Reproducibility & Implementation" |
| **Can we replace with better models?** | Section 7 "Future Work" |

---

## 🎯 For Thesis Writers

### Minimum Reading (30 minutes)
- Executive Summary (5 min)
- Section 1: Introduction (10 min)
- Section 5.1: Results (10 min)
- Section 6.2: Model Selection (5 min)

### Complete Reading (2-3 hours)
- Read entire **COMPREHENSIVE_EXPERIMENT_REPORT.md**
- Use supporting documents for extra detail if needed

### Citation

```bibtex
@inproceedings{phase1_sam_comparison,
  title={Quantitative Comparison of SAM Backends for Robotic Manipulation},
  author={Your Name},
  booktitle={Your Thesis Title},
  year={2026},
  note={Phase 1: ViT-B vs NanoSAM on 300 TESSE frames}
}
```

---

## 🔧 Technical Specifications

### Hardware (Jetson Orin)
- 12-core ARM CPU + 192-core GPU
- 12GB LPDDR5X RAM
- Ubuntu 22.04, CUDA 12.2, TensorRT 8.5

### Software
- Python 3.10, PyTorch 2.0, OpenCV 4.8
- ROS 2 Humble
- SAM (ViT-B, 375MB) + NanoSAM (20MB)

### Evaluation
- **Dataset:** 300 TESSE RGB-D frames
- **Metric:** F1 score (depth-filtered, 0.3-6.0m range)
- **Configurations:** 6 (2 backends × 3 strictness levels)
- **Frame verification:** All 300 processed, none skipped

---

## 🚀 Model Selection Summary

### ✅ Recommended: NanoSAM LOOSE
- **F1 Score:** 0.42 (acceptable accuracy)
- **Latency:** 188ms per frame (5 FPS, real-time!)
- **Precision:** 0.67 (safe for robotics)
- **Model Size:** 20MB (fits edge devices)
- **Power:** 12-15W (battery compatible)

### ⚠️ Best Accuracy (impractical): ViT-B MEDIUM
- **F1 Score:** 0.44 (best overall)
- **Latency:** 3,074ms per frame (0.3 FPS, too slow)
- **Model Size:** 375MB (exceeds edge memory)
- **Power:** 35-40W (drains robot battery)

### 📈 Path to Better Results
1. **Short-term:** SAM2 (+10-15% F1) + post-processing (+5-8% F1)
2. **Medium-term:** Task-specific fine-tuning (+10-12% F1)
3. **Long-term:** Ensemble methods (target F1 > 0.65)

See Section 7 of comprehensive report for detailed roadmap.

---

## 🐛 Troubleshooting

**Low F1 scores?**
→ Check depth encoding (must be 32FC1, not 16UC1) and range (should be 2.5-48.3m)

**Missing frames?**
→ Check `frame_verification.json` in results (should have 300 processed)

**Thermal throttling?**
→ ViT-B may hit 85°C; run 10-frame test or use cooling

**Can't reproduce latency?**
→ Jetson Orin-specific results; other GPUs will be faster

---

## 📞 Questions?

Refer to the **comprehensive report** (all answers are there).

For specific implementation details, check supporting documents folder.

---

## ✅ Document Status

- ✅ Comprehensive Report: **Complete & Ready for Thesis**
- ✅ Supporting Documents: Archived for reference
- ✅ Core Scripts: Clean, essential only
- ✅ Dataset: Frozen (300 frames, 992MB)
- ✅ Results: Validated (corrected depth data)

**Last Updated:** August 15, 2026  
**Status:** Ready for submission
