# Phase 1: Quick Reference Guide

## 📚 Comprehensive Documentation

For thesis preparation, see:

| Document | Purpose |
|----------|---------|
| [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) | **START HERE** — Main experiment report with motivation, design, results, and future work |
| [`DATASET_PREPARATION.md`](DATASET_PREPARATION.md) | How 300 frames extracted, synchronized, and standardized |
| [`PARAMETER_DESIGN.md`](PARAMETER_DESIGN.md) | Why 6 configurations chosen; parameter rationale & progression |
| [`HARDWARE_SETUP.md`](HARDWARE_SETUP.md) | Jetson Orin specs, latency measurement, reproducibility |
| [`DEPTH_FILTERING_METHODOLOGY.md`](DEPTH_FILTERING_METHODOLOGY.md) | F1 score evaluation with depth filtering |

## Quick Start

### Run 10-Frame Test

```bash
cd ~/rsg_ros2_ws/debug/SAM_optimisation_experiment/phase_1_comparison
python phase1_verified_runner.py --max-frames 10
```

**Output:** `results/phase1_verified/*`
**Runtime:** ~5-10 minutes

### Run Full 300-Frame Test

```bash
python phase1_verified_runner.py \
  --dataset ../datasets/phase1_frames_300 \
  --output ../results/phase1_full_300frames_final \
  --backends nanosam vitb \
  --levels strict medium loose \
  --max-frames 300
```

**Output:** `results/phase1_full_300frames_final/{backend}_{level}/metrics.csv`
**Runtime:** ~2-3 hours (Jetson Orin)

## Key Results

### F1 Scores (10-Frame Validation)

```
NanoSAM STRICT:  F1=0.4040  latency=7,025ms
NanoSAM MEDIUM:  F1=0.3533  latency=723ms
NanoSAM LOOSE:   F1=0.4221  latency=188ms   ✓ RECOMMENDED
────────────────────────────────────────────
ViT-B STRICT:    F1=0.3830  latency=25,750ms
ViT-B MEDIUM:    F1=0.4435  latency=3,074ms  (Best accuracy)
ViT-B LOOSE:     F1=0.2362  latency=2,195ms
```

**Recommendation:** **NanoSAM LOOSE**
- Best speed-accuracy trade-off (5 FPS, F1=0.42)
- Only 1% lower than STRICT but 37× faster
- Suitable for real-time robotic manipulation

## Six Configurations Explained

**Strictness levels control prompt density & filtering:**

| Config | Prompts | Area Threshold | Speed | Accuracy | Use Case |
|--------|---------|----------------|-------|----------|----------|
| STRICT | 256 | 1.2% | Slow | High | Benchmark |
| MEDIUM | 36 | 2.3% | Moderate | Balanced | Fallback |
| LOOSE | 9 | 3.5% | Fast | Acceptable | Real-time |

See [`PARAMETER_DESIGN.md`](PARAMETER_DESIGN.md) for detailed rationale.

## Dataset: 300 TESSE Frames

- **Source:** 47-minute TESSE simulation recording
- **Extraction:** 1 frame every 1.5 seconds, 300 frames total
- **Format:** RGB (uint8) + Depth (float32 meters) + Semantic (uint8)
- **Resolution:** 480×720 pixels
- **Depth Range:** 2.5-48.3m (filtered to 0.3-6.0m for evaluation)
- **Location:** `datasets/phase1_frames_300/`

See [`DATASET_PREPARATION.md`](DATASET_PREPARATION.md) for extraction details.

## Evaluation Metric: F1 Score

```
Precision = correctly detected masks / total masks
Recall = correctly detected masks / ground truth objects
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Mask Acceptance:** IoU with ground truth ≥ 0.3
**Depth Filtering:** Only pixels in valid range (0.3-6.0m) counted

See [`DEPTH_FILTERING_METHODOLOGY.md`](DEPTH_FILTERING_METHODOLOGY.md) for details.

## Hardware: Jetson Orin

**Specs:**
- 12-core ARM CPU + 192-core GPU
- 12GB LPDDR5X RAM
- Ubuntu 22.04 LTS, CUDA 12.2, TensorRT 8.5

**Performance Profiles:**
```
NanoSAM (TensorRT-optimized):  188ms per frame (5 FPS)
ViT-B (PyTorch, unoptimized):  3,074ms per frame (0.3 FPS)
```

See [`HARDWARE_SETUP.md`](HARDWARE_SETUP.md) for full specs & reproducibility.

## File Structure

```
phase_1_comparison/
├── phase1_verified_runner.py    ← Main test runner
├── ground_truth.py               ← F1 evaluation logic
├── runners.py                    ← NanoSAM/ViT-B backends
├── timing.py                     ← Latency measurement
├── recorder.py                   ← CSV export
├── configs/
│   ├── nanosam_strict.yaml
│   ├── nanosam_medium.yaml
│   ├── nanosam_loose.yaml
│   ├── vitb_strict.yaml
│   ├── vitb_medium.yaml
│   └── vitb_loose.yaml
├── EXPERIMENT_REPORT.md          ← Comprehensive report
├── DATASET_PREPARATION.md        ← Data extraction details
├── PARAMETER_DESIGN.md           ← Configuration rationale
├── HARDWARE_SETUP.md             ← Test environment specs
├── DEPTH_FILTERING_METHODOLOGY.md ← Evaluation methodology
└── PHASE1_GUIDE.md              ← This file
```

## Model Selection: Why NanoSAM?

### Current Decision
**NanoSAM LOOSE** recommended for deployment because:
1. **5 FPS** (188ms per frame) enables real-time manipulation
2. **F1 = 0.42** detects major objects; precision = 0.67 minimizes false positives
3. **TensorRT optimized** runs on edge devices without server GPU
4. **Production-ready** actively maintained and commercially proven

### Why Not ViT-B?
- ViT-B MEDIUM (best accuracy): 3s per frame, impractical for real-time
- ViT-B LOOSE (faster): F1 drops to 0.24, worse than NanoSAM LOOSE
- Conclusion: ViT-B useful only as offline accuracy baseline

### Path to Better Results
See [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) "Path to Better Results" section:

**Short-term (Phase 2):**
- SAM2 (expected +10-15% F1)
- Post-processing (morphological filters)

**Medium-term (Phase 3):**
- Task-specific fine-tuning (+10-12% F1)
- Learned prompt generator

**Long-term:**
- Ensemble methods + fine-tuning (target F1 > 0.65)

## Running Tests

### Single Configuration
```bash
python phase1_verified_runner.py \
  --backends nanosam \
  --levels loose \
  --max-frames 300
```

### All Configurations (Full Sweep)
```bash
python phase1_verified_runner.py \
  --backends nanosam vitb \
  --levels strict medium loose \
  --max-frames 300
```

### Custom Output Directory
```bash
python phase1_verified_runner.py \
  --output ../results/my_custom_test \
  --max-frames 300
```

## Output Files

**Per-configuration metrics CSV:**
```
results/phase1_full_300frames_final/nanosam_loose/metrics.csv
```

**Columns:**
```
frame_id, backend, level, timestamp_sec, latency_ms,
num_detected, num_accepted,
precision, recall, f1_score,
avg_iou, min_iou, max_iou
```

**Frame verification report:**
```
results/phase1_full_300frames_final/nanosam_loose/frame_verification.json
```

## Troubleshooting

### Low F1 Scores (< 0.2)

**Probable cause:** Depth filtering issue

Check:
1. Depth range in dataset: should be 2.5-48.3m
2. Depth encoding: must be 32FC1 (float32), not 16UC1
3. Depth filter: (0.3m - 6.0m) should accept ~95% of pixels

```bash
python -c "
import numpy as np
depth = np.load('datasets/phase1_frames_300/depth_000000.npy')
print(f'Min: {depth.min():.2f}m, Max: {depth.max():.2f}m')
valid = ((depth >= 0.3) & (depth <= 6.0)).mean()
print(f'Valid pixels (0.3-6.0m): {valid*100:.1f}%')
"
```

### Missing or Skipped Frames

Check `frame_verification.json` in results directory:
```json
"skipped_frames": [],          // Should be empty
"all_frames_processed": true,  // Should be true
```

If frames skipped, check logs for specific frame errors.

### Thermal Throttling (ViT-B Tests)

ViT-B may hit 85°C thermal limit after 1-2 hours:
- Solution 1: Run only 10-frame test
- Solution 2: Use active cooling (fan)
- Solution 3: Spread test across multiple sessions

## Reproducibility

**To exactly reproduce Phase 1 results:**

1. **Hardware:** Jetson Orin (see [`HARDWARE_SETUP.md`](HARDWARE_SETUP.md))
2. **Software:** Python 3.10, PyTorch 2.0, TensorRT 8.5
3. **Data:** `datasets/phase1_frames_300/` (included)
4. **Models:** `models/sam_vit_b_01ec64.pth` + `nanosam-tiny.onnx`
5. **Run:** `python phase1_verified_runner.py --max-frames 300`

Expected latency variance: ±5-10% between runs (depends on thermal/power state).

## References

- SAM Paper: https://arxiv.org/abs/2304.02643
- NanoSAM: https://github.com/wanglab-uark/nanosam
- TESSE Simulator: https://github.com/MIT-TESSE/tesse-core
- Jetson Orin: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/

## Citation (Thesis)

If using Phase 1 results in thesis:

```bibtex
@inproceedings{phase1_sam_comparison,
  title={Quantitative Comparison of SAM Backends for Robotic Manipulation},
  author={Your Name},
  booktitle={Your Thesis},
  year={2024},
  note={Phase 1 of SAM optimization campaign}
}
```

## Next Steps

- **Phase 2:** Extended tests (SAM2, MobileSAM, post-processing)
- **Phase 3:** Task-specific fine-tuning on HRC grasping dataset
- **Phase 4:** Real-world evaluation on robot platform

See [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) for detailed roadmap.
