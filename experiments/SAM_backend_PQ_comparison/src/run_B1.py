"""Experiment B, stage B1: NanoSAM grid size at the deployed pipeline settings (see EXPERIMENT_B_PROPOSAL.md). Stops after B1."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from runner import run_config, RES
FIXED = dict(input_scale=0.85, max_masks=8, min_mask_pixels=3500, mask_threshold=0.80, nms_iou=0.10, point_margin_fraction=0.167, skip_black_points=True, min_prompt_luma=4)
rows = [run_config(f"B1_g{g}", "nanosam", dict(FIXED, points_per_side=g)) for g in (7, 6, 5, 4, 3)]
df = pd.DataFrame(rows)
cols = ["config_id", "points_per_side", "PQ", "SQ", "RQ", "precision", "recall", "TP", "FP", "FN", "masks_per_frame_mean", "latency_median_ms", "latency_p95_ms", "FPS", "encoder_median_ms", "decoder_median_ms", "determinism_frames_different"]
df[cols].to_csv(f"{RES}/comparison_B1.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, ys, t in zip(ax, (["PQ", "SQ", "RQ"], ["FPS"], ["latency_median_ms", "latency_p95_ms"]), ("PQ / SQ / RQ", "FPS", "latency (ms)")):
    for y in ys: a.plot(df.points_per_side, df[y], "o-", label=y)
    a.set_xlabel("points per side"); a.set_title(t); a.invert_xaxis(); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig(f"{RES}/B1_plot.png", dpi=150)
print(df[cols[:14]].round(3).to_string(index=False))
