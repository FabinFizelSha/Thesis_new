# VLM Prompt Optimization System

Systematic framework for testing and optimizing VLM prompts for object classification in robotic perception.

## Overview

This system enables:
1. **Controlled experiments** - Run with RAP disabled, VLM-only for 300 seconds
2. **Automatic logging** - Track crops, VLM outputs, and timing
3. **Manual verification** - Web UI + CSV for human validation
4. **Analysis & reporting** - Accuracy metrics and thesis-ready reports

## Architecture

```
VLM-Prompt-Optimization/
├── experiment_20260830_HHMMSS__v1_production/
│   ├── crops/                      # All crop images for verification
│   │   ├── obj_000001_track_1_crop_000001.jpg
│   │   ├── obj_000002_track_2_crop_000002.jpg
│   │   └── ...
│   ├── vlm_outputs.jsonl           # VLM predictions with timing
│   ├── verification_results.csv    # Verification results (editable)
│   ├── verification_ui.html        # Interactive web verification UI
│   ├── metrics.json                # Accuracy metrics
│   └── report.md                   # Markdown thesis report
└── [more experiments...]
```

## Usage

### Step 1: Configure Experiment

Edit `vlm_prompt_optimization.yaml`:
```yaml
experiment:
  duration_sec: 300        # 5 minutes wall time
  description: "Test VLM prompts"

rap_vlm:
  rap_enabled: false       # ← Key: Disable RAP
  vlm_enabled: true

prompts:
  v1_production:
    active: true           # Current prompt to test
```

### Step 2: Run Experiment

Start the system with experiment mode:
```bash
ros2 run rsg rsg_phase1.py --vlm-prompt-optimization
```

The system will:
- Run for ~300 seconds
- Disable RAP, send all crops to VLM
- Log every crop and VLM output
- Create experiment session in `VLM-Prompt-Optimization/`

### Step 3: Manual Verification (Web UI)

1. Open `verification_ui.html` in browser:
```bash
open VLM-Prompt-Optimization/experiment_TIMESTAMP__v1_production/verification_ui.html
```

2. For each sample:
   - View crop image (left)
   - See VLM prediction
   - Enter actual class name
   - Rate your confidence
   - Click ✓ or ✗ to record result

3. Stats update in real-time:
   - Accuracy %
   - Verified count
   - Avg processing time

### Step 4: Manual Verification (CSV Fallback)

Edit `verification_results.csv` if UI doesn't work:
```csv
object_id,manual_verified,manual_class,is_correct,verified_by,notes
000001,true,tennis_ball,true,fabinfizelsha,"Clear green color"
000002,true,tennis_ball,false,fabinfizelsha,"Partial occlusion"
```

Columns:
- `manual_verified`: true/false (mark as verified)
- `manual_class`: Ground truth class name
- `is_correct`: true if VLM matched ground truth
- `verified_by`: Your name
- `notes`: Any observations

### Step 5: Generate Report

```bash
python3 -c "
from pathlib import Path
from nodes.support.phase1.vlm_experiment_report import VLMExperimentReport

session_dir = Path('VLM-Prompt-Optimization/experiment_TIMESTAMP__v1_production')
report = VLMExperimentReport(session_dir)

# Generate markdown report
print(report.generate_markdown_report())

# Save all outputs
report.save_report('report.md')
report.save_metrics_json('metrics.json')
print(report.get_summary())
"
```

Output files:
- `report.md` - Markdown report (for thesis)
- `metrics.json` - Detailed metrics (for analysis)
- Console output - Quick summary

## Logged Metrics

### Per-Crop
- `object_id` - Unique object identifier
- `crop_filename` - Saved crop image
- `vlm_class` - VLM prediction
- `vlm_confidence` - Prediction confidence (0-1)
- `vlm_processing_time_ms` - VLM inference time
- `timestamp` - Frame time

### Per-Prompt
- **Accuracy %** - Correct predictions / verified samples
- **False Positives** - Wrong predictions with high confidence
- **False Negatives** - Missed predictions despite correct class present
- **Processing Time** - Average/min/max VLM inference time
- **Confidence Distribution** - Accuracy by confidence level

## Prompt Versions

Edit `vlm_prompt_optimization.yaml` to test different prompts:

```yaml
prompts:
  v1_production:
    template: |
      You are an object classifier for robotic perception.
      Analyze this image and identify the object.
      Format: {"class": "NAME", "confidence": 0.X}
    active: true

  v2_simple:
    template: |
      What is this object? 
      Answer: {"class": "NAME", "confidence": 0.0-1.0}
    active: false

  v3_detailed:
    template: |
      Identify the object, considering size, shape, color.
      Provide visual features.
      Format: {"class": "NAME", "confidence": 0.X, "features": [...]}
    active: false
```

### To Test Different Prompt

1. Set `active: true` for new prompt in YAML
2. Set `active: false` for others
3. Run experiment again
4. Results go to separate `experiment_TIMESTAMP__v2_simple/` folder

## Output Format

VLM outputs must be JSON with at minimum:
```json
{
  "class": "tennis_ball",
  "confidence": 0.92,
  "reasoning": "Green spherical object characteristic of tennis balls"
}
```

Extended format (optional):
```json
{
  "class": "tennis_ball",
  "confidence": 0.92,
  "features": ["green color", "spherical shape", "fuzzy texture"],
  "ambiguities": ["could be small ball", "could be toy"],
  "robot_action": "pick and place"
}
```

## Thesis Report

The system generates markdown reports suitable for thesis chapter:

```markdown
# VLM Prompt Optimization: v1_production

## Results Summary
- Accuracy: 85.6%
- Verified Samples: 43/50
- Avg Processing Time: 2,340 ms

## Confidence Analysis
| Confidence | Count | Accuracy |
|------------|-------|----------|
| 90%+ | 15 | 93.3% |
| 70-90% | 18 | 83.3% |
| < 70% | 10 | 70.0% |

## False Positive Analysis
3 high-confidence incorrect predictions:
- [Crop images + analysis]

## Recommendations
1. Prompt should emphasize [specific visual features]
2. Consider adding context about scene
3. Handle edge cases like occlusion/scale
```

## Advanced: Batch Testing Multiple Prompts

```bash
# Test all prompts sequentially
for prompt in v1_production v2_simple v3_detailed v4_complex; do
  # Update YAML to activate only this prompt
  sed -i "s/active:.*/active: false/g" vlm_prompt_optimization.yaml
  sed -i "/$prompt:/,/active:/ s/active:.*/active: true/" vlm_prompt_optimization.yaml
  
  # Run experiment
  ros2 run rsg rsg_phase1.py --vlm-prompt-optimization
  sleep 5
  
  # Wait for experiment to complete (300 sec)
  echo "Running $prompt experiment..."
  sleep 320
done

# Generate comparison report
python3 generate_comparison_report.py VLM-Prompt-Optimization/
```

## Files Reference

| File | Purpose |
|------|---------|
| `vlm_prompt_optimization.yaml` | Experiment configuration |
| `vlm_prompt_logger.py` | Logging system (auto-imported) |
| `vlm_verification_ui.py` | Web UI generator (auto-imported) |
| `vlm_experiment_report.py` | Report generation (auto-imported) |

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `duration_sec` | 300 | Wall time for experiment |
| `rap_enabled` | false | Must be false for VLM-only test |
| `sample_size` | 50 | Target for manual verification |
| `tracking_processing_time` | true | Log VLM inference time |
| `vlm_output_format` | "json" | Fixed format required |

## Troubleshooting

**Q: Crops not appearing in verification UI?**
A: Check `crops/` directory exists and has .jpg files. Ensure crop paths are relative to HTML file.

**Q: VLM output format different from expected?**
A: JSON must include `class` and `confidence` fields. Use `vlm_output_format: "json"` in config.

**Q: How to handle multi-run crop differences?**
A: Expected due to non-deterministic cropping. That's why we test 50+ samples - statistical significance from larger N.

**Q: Can I interrupt the experiment mid-run?**
A: Yes, manually stop and re-run verification on existing data. Session data is saved immediately.

## Next Steps

1. ✅ Set up initial experiment with v1_production prompt
2. ✅ Verify ~50 samples with web UI
3. ✅ Generate accuracy report
4. ✅ Analyze failure modes
5. → Design improved prompt (v2, v3, etc.)
6. → Repeat testing with new prompt
7. → Compare results across prompts
8. → Write thesis chapter

---

**Contact:** For issues or suggestions, see the main RSG documentation.
