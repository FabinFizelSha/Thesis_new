"""Experiment B stages B2 (mask_threshold) and B3 (nms_iou) at the grid chosen after B1 (4x4). Selection: highest PQ; among values within 0.005 PQ of the best, highest FPS."""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from runner import run_config, RES
G = 4
FIXED = dict(input_scale=0.85, max_masks=8, min_mask_pixels=3500, nms_iou=0.10, point_margin_fraction=0.167, skip_black_points=True, min_prompt_luma=4, points_per_side=G)
log = ["# Experiment B selection log", "", "Rule: highest PQ; among values within 0.005 PQ of the best, the highest FPS.", "",
       f"## B1 (grid): chosen by the user = {G}x{G} (PQ 0.320, 42.2 ms, 23.7 FPS) as a balance between PQ and latency. The rule alone would have picked 7x7 (highest PQ).", ""]

def select(rows, key):
    best = max(r["PQ"] for r in rows)
    cand = [r for r in rows if best - r["PQ"] <= 0.005]
    return max(cand, key=lambda r: r["FPS"]), best, cand

def table(rows, key):
    return ["| " + key + " | PQ | SQ | RQ | median ms | p95 ms | FPS | masks/frame | overlap px frac |", "|---|---|---|---|---|---|---|---|---|"] + \
           [f"| {r[key]} | {r['PQ']:.3f} | {r['SQ']:.3f} | {r['RQ']:.3f} | {r['latency_median_ms']:.1f} | {r['latency_p95_ms']:.1f} | {r['FPS']:.1f} | {r['masks_per_frame_mean']:.2f} | {r.get('overlap_pixel_fraction_mean', float('nan')):.3f} |" for r in rows]

# B2: thresholds 0.7 / 0.9 new; 0.8 reused from B1_g4
b1 = json.load(open(f"{RES}/B1_g{G}/summary.json"))
b2 = {0.8: b1}
for thr in (0.7, 0.9): b2[thr] = run_config(f"B2_thr{thr}_g{G}", "nanosam", dict(FIXED, mask_threshold=thr))
rows2 = [b2[t] for t in (0.7, 0.8, 0.9)]
win2, best2, cand2 = select(rows2, "mask_threshold")
log += ["## B2 (mask_threshold) at grid 4x4, nms_iou 0.10 (0.8 reused from B1_g4)", ""] + table(rows2, "mask_threshold") + ["",
        f"Best PQ {best2:.3f}; within 0.005: {[r['mask_threshold'] for r in cand2]}. **Chosen: mask_threshold = {win2['mask_threshold']}**", ""]
THR = win2["mask_threshold"]

# B3: nms 0 / 0.15 / 0.30 at the chosen threshold; 0.10 recomputed as a reference row (B1/B2 runs lack the overlap metric)
b3 = {n: run_config(f"B3_nms{n}_g{G}_thr{THR}", "nanosam", dict(FIXED, mask_threshold=THR, nms_iou=n)) for n in (0.0, 0.10, 0.15, 0.30)}
rows3 = [b3[n] for n in (0.0, 0.15, 0.30)]
win3, best3, cand3 = select(rows3, "nms_iou")
log += [f"## B3 (nms_iou) at grid 4x4, mask_threshold {THR} (0.10 is a reference row, not part of the selection list)", ""] + table([b3[n] for n in (0.0, 0.10, 0.15, 0.30)], "nms_iou") + ["",
        f"Best PQ among {{0, 0.15, 0.30}} {best3:.3f}; within 0.005: {[r['nms_iou'] for r in cand3]}. **Chosen: nms_iou = {win3['nms_iou']}**", "",
        f"## Final B configuration: grid {G}x{G}, mask_threshold {THR}, nms_iou {win3['nms_iou']}, max_masks 8, min_mask_pixels 3500, input_scale 0.85", ""]
open(f"{RES}/selection_log.md", "w").write("\n".join(log) + "\n")

allrows = [json.load(open(f"{RES}/B1_g{g}/summary.json")) for g in (7, 6, 5, 4, 3)]
allrows = [dict(r, stage="B1") for r in allrows] + [dict(r, stage="B2") for r in (b2[0.7], b2[0.9])] + [dict(r, stage="B3") for r in (b3[n] for n in (0.0, 0.10, 0.15, 0.30))]
cols = ["stage", "config_id", "points_per_side", "mask_threshold", "nms_iou", "PQ", "SQ", "RQ", "precision", "recall", "TP", "FP", "FN", "masks_per_frame_mean", "overlap_pixel_fraction_mean", "latency_median_ms", "latency_p95_ms", "FPS", "encoder_median_ms", "decoder_median_ms", "determinism_frames_different"]
pd.DataFrame(allrows).reindex(columns=cols).to_csv(f"{RES}/comparison_B.csv", index=False)
tex = ["grid  PQ    FPS"] + [f"g{r['points_per_side']}    {r['PQ']:.2f}  {r['FPS']:.1f}" for r in allrows[:5]]
tex += ["", "thr   PQ    FPS"] + [f"{r['mask_threshold']}   {r['PQ']:.2f}  {r['FPS']:.1f}" for r in rows2]
tex += ["", "nms   PQ    FPS"] + [f"{n}   {b3[n]['PQ']:.2f}  {b3[n]['FPS']:.1f}" for n in (0.0, 0.15, 0.30)]
open(f"{RES}/latex_tables_B.txt", "w").write("\n".join(tex) + "\n"); print("\n".join(log)); print("\n".join(tex))
