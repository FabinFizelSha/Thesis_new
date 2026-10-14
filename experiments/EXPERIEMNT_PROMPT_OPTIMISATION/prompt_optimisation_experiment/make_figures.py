#!/usr/bin/env python3
"""Regenerate the comparison figures for the prompt-optimisation chapter.

Reads the committed per-run session CSVs, computes accuracy and median
inference latency, and writes PNGs into ``figures/``.  Run from this
directory:  ``python3 make_figures.py``
"""
import csv
import glob
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# run id -> (model label, prompt label, verified correct, verified N)
# Accuracy is the manually-verified value from EXPERIMENT_REPORT.md section 11
# (the per-run annotation conventions differ, so it is not re-derived here).
# Latency IS re-derived from the session CSVs (machine-written field).
RUNS = {
    "R1": ("qwen3-vl-8b", "P1", 24, 50), "R2": ("qwen3-vl-8b", "P2", 36, 50), "R3": ("qwen3-vl-8b", "P3", 31, 49),
    "R4": ("qwen3.5-4b", "P1", 32, 50),  "R5": ("qwen3.5-4b", "P2", 40, 50),  "R6": ("qwen3.5-4b", "P3", 44, 50),
    "R7": ("qwen3.5-9b", "P1", 32, 50),  "R8": ("qwen3.5-9b", "P2", 34, 50),  "R9": ("qwen3.5-9b", "P3", 34, 50),
}
MODELS = ["qwen3-vl-8b", "qwen3.5-4b", "qwen3.5-9b"]
PROMPTS = ["P1", "P2", "P3"]
MCOLOR = {"qwen3-vl-8b": "#4C72B0", "qwen3.5-4b": "#DD8452", "qwen3.5-9b": "#55A868"}


def find_csv(run_id):
    pat = os.path.join(HERE, "runs", f"{run_id}__*", "session_*", "vlm_results.csv")
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def load(run_id):
    model, prompt, correct, n = RUNS[run_id]
    path = find_csv(run_id)
    inf = []
    if path:
        rows = list(csv.DictReader(open(path)))
        inf = [float(r["vlm_inference_ms"]) for r in rows if r["vlm_inference_ms"].strip()]
    return {
        "n": n,
        "correct": correct,
        "accuracy": 100.0 * correct / n if n else 0.0,
        "lat_median": st.median(inf) if inf else 0.0,
        "lat_mean": st.mean(inf) if inf else 0.0,
    }


DATA = {rid: load(rid) for rid in RUNS}
for rid, d in DATA.items():
    m, p, _, _ = RUNS[rid]
    print(f"{rid} {m:12s} {p}  acc={d['accuracy']:5.1f}%  ({d['correct']}/{d['n']})  "
          f"lat_median={d['lat_median']:.0f}ms  lat_mean={d['lat_mean']:.0f}ms")


def grouped_bar(metric, ylabel, title, fname, fmt, annotate_best=None, legend_loc="upper left"):
    x = np.arange(len(PROMPTS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for i, model in enumerate(MODELS):
        vals = []
        for p in PROMPTS:
            rid = next(r for r, tup in RUNS.items() if tup[0] == model and tup[1] == p)
            vals.append(DATA[rid][metric])
        bars = ax.bar(x + (i - 1) * w, vals, w, label=model, color=MCOLOR[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["P1 — zero-shot", "P2 — example-driven", "P3 — structural-priority"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="model", frameon=False, loc=legend_loc)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    top = max(max(DATA[r][metric] for r in RUNS), 1)
    ax.set_ylim(0, top * 1.18)
    if annotate_best:
        rid = annotate_best
        m, p = RUNS[rid][0], RUNS[rid][1]
        pi = PROMPTS.index(p)
        mi = MODELS.index(m)
        v = DATA[rid][metric]
        ax.annotate("  best  ", xy=(pi + (mi - 1) * w, v), xytext=(0, 30),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=MCOLOR[m], ec="none"),
                    arrowprops=dict(arrowstyle="-", lw=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=150)
    plt.close(fig)
    print("wrote", fname)


grouped_bar("accuracy", "accuracy (%)",
            "Labelling accuracy — model x prompt (n = 50 crops per run, 49 for R3)",
            "accuracy_by_prompt.png", "{:.0f}", annotate_best="R6")

grouped_bar("lat_median", "median inference time (ms)",
            "VLM call time — median inference latency, model x prompt",
            "latency_by_prompt.png", "{:.0f}", legend_loc="upper right")


# scatter: accuracy vs latency, all 9 runs
fig, ax = plt.subplots(figsize=(8.2, 5.2))
mark = {"P1": "o", "P2": "s", "P3": "^"}
for rid, (m, p, _c, _n) in RUNS.items():
    d = DATA[rid]
    ax.scatter(d["lat_median"], d["accuracy"], s=130, marker=mark[p],
               color=MCOLOR[m], edgecolor="black", linewidth=0.6, zorder=3)
    ax.annotate(f"{rid}", (d["lat_median"], d["accuracy"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8)
ax.set_xlabel("median inference time (ms)  —  lower is better")
ax.set_ylabel("accuracy (%)  —  higher is better")
ax.set_title("Accuracy vs. call time (all 9 runs)")
ax.grid(alpha=0.3)
ax.set_axisbelow(True)
# legends
from matplotlib.lines import Line2D
mh = [Line2D([0], [0], marker="o", color="w", markerfacecolor=MCOLOR[m],
             markeredgecolor="black", markersize=10, label=m) for m in MODELS]
ph = [Line2D([0], [0], marker=mark[p], color="w", markerfacecolor="#888",
             markeredgecolor="black", markersize=10, label=p) for p in PROMPTS]
l1 = ax.legend(handles=mh, title="model", loc="lower right", frameon=False)
ax.add_artist(l1)
ax.legend(handles=ph, title="prompt", loc="lower center", frameon=False)
# highlight winner
d6 = DATA["R6"]
ax.annotate("winner: qwen3.5-4b x P3\n88 %, 2.9 s",
            (d6["lat_median"], d6["accuracy"]), xytext=(40, -25),
            textcoords="offset points", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.0))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "accuracy_vs_latency.png"), dpi=150)
plt.close(fig)
print("wrote accuracy_vs_latency.png")
