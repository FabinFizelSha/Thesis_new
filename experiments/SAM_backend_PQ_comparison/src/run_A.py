"""Experiment A only: NanoSAM vs SAM ViT-B at 3x3, 5x5, 7x7 on all 500 frames (base settings)."""
import sys, json, os; sys.path.insert(0, "src")
import pandas as pd
from runner import run_config, RES
rows = []
for g in (3, 5, 7):
    for m, tag in (("nanosam", "nano"), ("sam_vitb", "vitb")):
        rows.append(run_config(f"A_{tag}_{g}", m, dict(points_per_side=g)))
df = pd.DataFrame(rows)
cols = ["config_id", "model", "points_per_side", "PQ", "SQ", "RQ", "precision", "recall", "TP", "FP", "FN", "masks_per_frame_mean", "masks_per_frame_median",
        "latency_median_ms", "latency_p95_ms", "FPS", "encoder_median_ms", "decoder_median_ms", "determinism_frames_different"]
df[cols].to_csv(f"{RES}/comparison.csv", index=False)
g = {(r.model, r.points_per_side): r for r in df.itertuples()}
out = ["grid  nanoPQ  nanoSQ  nanoRQ  nanoLat  vitPQ  vitSQ  vitRQ  vitLat"]
for n in (3, 5, 7):
    a, b = g[("nanosam", n)], g[("sam_vitb", n)]
    out.append(f"g{n}    {a.PQ:.2f}  {a.SQ:.2f}  {a.RQ:.2f}  {a.latency_median_ms:.0f}  {b.PQ:.2f}  {b.SQ:.2f}  {b.RQ:.2f}  {b.latency_median_ms:.0f}")
open(f"{RES}/latex_tables_A.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
