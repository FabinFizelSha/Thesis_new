# VLM Testing Guide

Simple workflow for testing and evaluating VLM prompt performance.

## Quick Start

### Step 1: Clear RAP Memory (Optional but Recommended)

For VLM-only testing, clear RAP memory to avoid carrying over old retrievals:

```bash
# Clear RAP memory and visual storage (start fresh)
rm -rf ~/rsg_ros2_ws/debug/phase1_rap_memory.jsonl
rm -rf ~/rsg_ros2_ws/visual_memory/*
mkdir -p ~/rsg_ros2_ws/visual_memory
```

This ensures baseline test is truly VLM-only without historical RAP context.

### Step 2: Run Test (Manual Timer)

```bash
# Terminal: Start the system
ros2 run rsg rsg_phase1.py

# Separately: Start a timer
# Let it run for ~300 seconds (5 minutes)
# Ctrl+C to stop
```

All VLM outputs will be automatically logged to:
```
VLM-Test-Session/
├── crops/                 # Crop images
│   ├── obj_000001_crop.jpg
│   └── obj_000002_crop.jpg
└── vlm_results.csv        # Raw outputs (you verify this)
```

### Step 2: Verify Results (Manual)

Open `VLM-Test-Session/vlm_results.csv` in Excel/Google Sheets:

```csv
object_id, crop_filename, label, label_confidence, mobility_class, mobility_confidence, ..., manual_label, manual_is_correct, manual_notes
000001, obj_000001_crop.jpg, tennis_ball, 0.95, graspable, 0.87, ..., [YOU FILL], [true/false], [notes]
000002, obj_000002_crop.jpg, ball, 0.75, graspable, 0.82, ..., [YOU FILL], [true/false], ...
```

For each row:
1. **View crop** - Open image file from `crops/` folder
2. **Fill `manual_label`** - What the object actually is
3. **Fill `manual_is_correct`** - true if VLM was correct, false if wrong
4. **Fill `manual_notes`** - Optional observations

Target: **Verify 50 samples** (or more for higher confidence)

### Step 3: Generate Report

After verification, run:

```bash
python3 -c "
from pathlib import Path
from nodes.support.phase1.vlm_accuracy_report import VLMAccuracyReport

report = VLMAccuracyReport(Path('VLM-Test-Session/vlm_results.csv'))
report.print_report()
report.save_markdown_report()
"
```

**Output:**
- Console: Quick accuracy summary
- `VLM-Test-Session/accuracy_report.md`: Detailed report (for thesis)

## CSV Fields Explained

**VLM Predictions (Auto-logged):**
- `object_id` - Unique ID for this sample
- `crop_filename` - Saved crop image (in `crops/` folder)
- `timestamp` - Frame timestamp
- `label` - VLM's object classification
- `label_confidence` - VLM's confidence in label (0-1)
- `mobility_class` - VLM's grasp classification
- `mobility_confidence` - VLM's confidence in grasp
- `vlm_processing_time_ms` - ⭐ **VLM inference time**
- `success` - Whether VLM succeeded
- `validation_status` - Pipeline validation result
- `raw_response` - Raw VLM response (first 100 chars)

**Manual Verification (You Fill):**
- `manual_label` - **REQUIRED** - Actual object class
- `manual_is_correct` - **REQUIRED** - true if VLM matched, false if wrong
- `manual_notes` - Optional - Why it was wrong, special cases, etc.

## Verification Tips

1. **Keep crops and spreadsheet open side-by-side**
   - Reference the image while filling in the label

2. **Batch similar objects** - Fill in blocks of similar types

3. **Notes examples:**
   - "Partially occluded, but clearly tennis ball"
   - "Model confused with similar colored object"
   - "Very blurry crop, hard to verify"

4. **Target 50 samples** for statistical confidence
   - More is better (100+ for higher confidence)
   - Less is faster to verify (20+ for initial test)

## Report Output

Console output shows:
```
╔════════════════════════════════════════════════════════════╗
║         VLM TESTING RESULTS SUMMARY                        ║
╚════════════════════════════════════════════════════════════╝

📊 OVERALL ACCURACY
   Verified Samples:  50
   Correct:           43/50
   Accuracy:          86.0%

🎯 CONFIDENCE ANALYSIS
   High Confidence (>70%)
     Samples:         35
     Accuracy:        91.4%

   Low Confidence (≤70%)
     Samples:         15
     Accuracy:        73.3%

⚙️  PERFORMANCE
   Avg Processing Time: 2345 ms
```

Markdown report includes error analysis and recommendations.

## File Structure

```
VLM-Test-Session/
├── crops/                          # All crop images
│   ├── obj_000001_crop.jpg
│   ├── obj_000002_crop.jpg
│   └── ...
├── vlm_results.csv                 # Results (edit this)
├── accuracy_report.md              # Generated report
└── [session timestamp]             # Session metadata
```

## Tips for Different Scenarios

**Testing Simple Prompt:**
- Run test for 300-600 sec
- Verify 30-50 samples quickly
- Get baseline metrics

**Iterating on Prompt:**
- Make small prompt changes
- Re-run test (new session folder)
- Compare accuracy metrics
- Track what changed between versions

**Publication/Thesis:**
- Run for extended duration (600-1200 sec)
- Verify 100+ samples for higher confidence
- Generate detailed markdown report
- Include accuracy tables and failure analysis

## Troubleshooting

**Q: No crops in folder?**
A: Make sure system is actually running VLM. Check terminal output for VLM calls.

**Q: CSV has many rows?**
A: Each row is one VLM classification. You only need to verify ~50. Mark others as blank.

**Q: Processing time very high?**
A: Normal if using remote VLM API. Will show actual latency in your system.

**Q: Can't open CSV in Excel?**
A: Use Google Sheets (upload CSV) or a text editor that handles CSV well.

---

**Session saves to:** `/home/student/rsg_ros2_ws/VLM-Test-Session/`

**All crops and results preserved** for later analysis.
