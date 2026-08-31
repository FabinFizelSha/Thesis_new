# VLM Prompt Optimization Documentation

Complete documentation of VLM prompt engineering experiments with Qwen 3 VL 8B F16.

## Files

- **VLM-OPTIMIZATION-FINAL-REPORT.md** - Comprehensive final report with all 6 test iterations, lessons learned, and recommendations
- **VLM-PROMPT-OPTIMIZATION-EXPERIMENT.md** - Detailed experiment design and test run documentation
- **VLM-PROMPT-OPTIMIZATION-README.md** - Original project overview
- **VLM-PROMPT-OPTIMIZATION-FINDINGS.md** - Root cause analysis and v3 proposal
- **VLM-PROMPT-V3-V4-ANALYSIS.md** - Analysis of v3 failure and v4 strategy
- **VLM-V5-PROMPT.md** - v5 prompt strategy documentation
- **VLM-TESTING-GUIDE.md** - Quick start guide for running tests

## Summary

**Status:** Frozen after 6 prompt iterations  
**Best Accuracy:** 47.5% (v2_refined)  
**Model:** Qwen3-VL-8B-Instruct-GGUF (F16)  
**Key Finding:** Model accuracy ceiling at ~50% due to visual salience prioritizing ceiling_lights over ceiling surface

## Test Results

| Version | Approach | Accuracy | Ceiling Errors |
|---------|----------|----------|----------------|
| v1_simplified | Simple baseline | 46.9% | 9 |
| v2_refined | Surface/fixture hierarchy | 47.5% | 11 |
| v3_hierarchical | Complex hierarchy | 38.2% | 15 |
| v4_filtering | Filtering rules | 96.7%* | 9+ |
| v5_strategy | Instructions-based | 51.8% | 16 |
| v6_structural | Structural priority | 50.0% | 14 |

*v4's 96.7% was only on 30 verified samples; unverified rows showed failures

## Recommendations

For improved results:
- Test stronger VLM models (GPT-4V, Claude 3.5 Vision, Gemini 2.0)
- Consider hybrid approaches (VLM + deterministic rules)
- Explore preprocessing/postprocessing alternatives
- Investigate two-stage classification

See **VLM-OPTIMIZATION-FINAL-REPORT.md** for complete analysis and recommendations.
