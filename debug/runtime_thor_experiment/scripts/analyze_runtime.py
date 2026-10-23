#!/usr/bin/env python3
"""Analysis for the Thor runtime benchmark. Writes to ../analysis/.

Wall-clock (wall_clock_unix_sec) throughout. Steady state = frame_trace rows
after the first WARMUP processed frames; values with warm-up are also written.
Usage: analyze_runtime.py [--runs T0_r1 T0_r2 ...] [--profile uhumans2]
"""
from __future__ import annotations
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_stage_timing import HIERARCHY  # copied from debug/revisit_experiment/analyze_stage_timing.py

EXP = HERE.parent
RUNS, OUT = EXP / "runs", EXP / "analysis"
WARMUP = 20

# Mutually exclusive thesis-level groups; they sum to total_delay_ms (see METHOD.md).
GROUPS = {
    "Queueing": ["sent_to_classifier_delay_ms", "sam_output_queue_wait_ms", "pipeline_wait_ms"],
    "Image conversion": ["image_conversion_delay_ms"],
    "Segmentation (NanoSAM)": ["sam_delay_ms"],
    "Geometry (mask to 3D)": ["geometry_metadata_ms"],
    "Association / tracking": ["frame_assignment_ms", "track_association_ms"],
    "Crop scoring and update": ["crop_update_ms"],
    "Semantic dispatch and publishing": ["active_segments_publish_ms", "semantic_dispatch_ms",
                                         "quality_deferred_release_ms", "run_rap_other_ms"],
    "Label map, metadata, message build": ["label_map_delay_ms", "metadata_delay_ms",
                                           "result_message_build_delay_ms", "classifier_other_ms"],
    "Hydra input build and publish (Phase 1)": ["coordinator_delay_ms"],
    "Diagnostics overhead": ["classifier_debug_record_delay_ms"],
}
RENAME = {"assignment_hungarian_ms": "assignment_solve_ms"}   # legacy name; implementation is greedy


def run_dirs(ids):
    out = []
    for d in sorted(RUNS.glob("T*_r*_*")):
        rid = "_".join(d.name.split("_")[:2])
        if ids is None or rid in ids:
            out.append((rid, d))
    return out


def stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return dict(n=0, mean=np.nan, median=np.nan, p5=np.nan, p95=np.nan, max=np.nan, std=np.nan)
    return dict(n=len(s), mean=s.mean(), median=s.median(), p5=s.quantile(.05), p95=s.quantile(.95), max=s.max(), std=s.std())


def load(d):
    t = pd.read_csv(d / "phase1_timing.csv", low_memory=False)
    ft = t[t.event == "frame_trace"].sort_values("wall_clock_unix_sec").reset_index(drop=True)
    return t, ft


def throughput(rid, d, t, ft):
    w0, w1 = ft.wall_clock_unix_sec.iloc[0], ft.wall_clock_unix_sec.iloc[-1]
    span = w1 - w0
    r = dict(run=rid, processed=len(ft), span_s=span, fps=(len(ft) - 1) / span)
    ss = ft.iloc[WARMUP:]
    r["fps_steady"] = (len(ss) - 1) / (ss.wall_clock_unix_sec.iloc[-1] - ss.wall_clock_unix_sec.iloc[0])
    for ev in ("dropped_oldest", "dropped_newest", "failed"):
        r[ev] = int((t.event == ev).sum())
    tr = pd.read_csv(d / "topic_rates.csv")
    def n(topic, a=-np.inf, b=np.inf):
        x = tr[(tr.topic == topic) & tr.wall_clock_unix_sec.between(a, b)]
        return len(x)
    r["offered_to_phase1"] = n("/rsg/preprocessed/frame")
    r["sent_to_hydra_rgb"] = n("/rsg/hydra/rgb/image_raw")
    r["processing_ratio"] = r["processed"] / r["offered_to_phase1"] if r["offered_to_phase1"] else np.nan
    for name, topic in (("hydra_dsg", "/hydra/backend/dsg"), ("fuser_markers", "/rsg/scene_graph/markers"), ("fused_dsg", "/rsg/hydra/fused_dsg")):
        x = tr[tr.topic == topic].wall_clock_unix_sec
        r[f"{name}_msgs"] = len(x)
        r[f"{name}_hz"] = (len(x) - 1) / (x.max() - x.min()) if len(x) > 1 else np.nan
    # Delay Hydra input (semantic image) -> next DSG message; DSG -> next markers message
    def next_delay(a_topic, b_topic):
        a = tr[tr.topic == a_topic].wall_clock_unix_sec.values; b = np.sort(tr[tr.topic == b_topic].wall_clock_unix_sec.values)
        if len(a) == 0 or len(b) == 0: return np.array([])
        idx = np.searchsorted(b, a, side="right"); ok = idx < len(b)
        return b[idx[ok]] - a[ok]
    for nm, (a, b) in {"hydra_in_to_dsg_s": ("/rsg/hydra/semantic/image_raw", "/hydra/backend/dsg"),
                       "dsg_to_markers_s": ("/hydra/backend/dsg", "/rsg/scene_graph/markers")}.items():
        d_ = next_delay(a, b); r[nm + "_median"] = np.median(d_) if len(d_) else np.nan; r[nm + "_p95"] = np.quantile(d_, .95) if len(d_) else np.nan
    return r, (w0, w1)


def resources(rid, d, win):
    rs = pd.read_csv(d / "resource_usage.csv")
    rs = rs[rs.wall_clock_unix_sec.between(*win)]
    cols = {"gpu_pct": "gpu_pct", "cpu_avg_pct": "cpu_avg_pct", "ram_gib": None, "power_total_w": "power_total_w",
            "power_gpu_w": "power_gpu_w", "power_cpu_soc_w": "power_cpu_soc_w", "temp_gpu_c": "temp_gpu_c", "temp_cpu_c": "temp_cpu_c", "temp_tj_c": "temp_tj_c"}
    rs = rs.assign(ram_gib=rs.ram_used_mb / 1024)
    out = dict(run=rid, samples=len(rs))
    for c in cols:
        if c in rs:
            out[f"{c}_mean"] = rs[c].mean(); out[f"{c}_p95"] = rs[c].quantile(.95); out[f"{c}_max"] = rs[c].max()
    core = [c for c in rs.columns if re.fullmatch(r"cpu\d+_pct", c)]
    for c in core: out[f"{c}_mean"] = rs[c].mean()
    return out


def vlm(rid, d, t):
    q = t[t.event == "dequeued"][["candidate_id", "queue_wait_ms"]]
    res = t[t.event.isna() & t.vlm_delay_ms.notna()].copy()
    res["queue_wait_ms"] = res.candidate_id.map(q.drop_duplicates("candidate_id").set_index("candidate_id").queue_wait_ms)
    log = [l for l in (d / "vlm_server.log").read_text().splitlines()]
    rows, cur = [], {}
    for l in log:
        m = re.search(r"task (\d+) \|\s+(prompt eval|eval|total) time =\s+([\d.]+) ms /\s+(\d+) (?:tokens|token)", l)
        if m:
            k = m.group(2); cur[k + "_ms"] = float(m.group(3)); cur[k + "_tok"] = int(m.group(4))
            if k == "total": cur["task"] = int(m.group(1)); rows.append(cur); cur = {}
    srv = pd.DataFrame(rows)
    return res, srv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="*"); a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tp, rsrc, brk, vlm_s, calls_all = [], [], {}, [], []
    pooled = []
    for rid, d in run_dirs(a.runs):
        t, ft = load(d)
        r, win = throughput(rid, d, t, ft); tp.append(r)
        rsrc.append(resources(rid, d, win))
        ss = ft.iloc[WARMUP:]
        r["energy_J_per_frame"] = rsrc[-1]["power_total_w_mean"] / r["fps"]
        rows = []
        for stage, parent in HIERARCHY:
            if stage not in ft: continue
            s_all, s_ss = stats(ft[stage]), stats(ss[stage])
            tot = ss["total_delay_ms"].mean()
            rows.append(dict(run=rid, stage=RENAME.get(stage, stage), parent=RENAME.get(parent, parent), **{f"ss_{k}": v for k, v in s_ss.items()},
                             share_of_total=s_ss["mean"] / tot, withwarm_mean=s_all["mean"]))
        df = pd.DataFrame(rows); df.to_csv(OUT / f"stage_breakdown_{rid}.csv", index=False)
        # children-sum check
        chk = []
        for stage, parent in HIERARCHY:
            kids = [s for s, p in HIERARCHY if p == stage and s in ss]
            if kids and stage in ss:
                diff = (ss[stage] - ss[kids].sum(axis=1)).mean(); chk.append((stage, diff))
        brk[rid] = dict(df=df, chk=chk)
        # groups
        g = {name: ss[cols].sum(axis=1) for name, cols in GROUPS.items()}
        gdf = pd.DataFrame(g); gdf["sum"] = gdf.sum(axis=1); gdf["total"] = ss.total_delay_ms.values
        pooled.append((rid, ss, gdf))
        res, srv = vlm(rid, d, t)
        res["run"] = rid; calls_all.append(res)
        vlm_s.append(dict(run=rid, calls=len(res), status_counts=res.status.value_counts().to_dict(),
                          **{f"vlm_delay_{k}": v for k, v in stats(res.vlm_delay_ms).items() if k != "n"},
                          **{f"queue_wait_{k}": v for k, v in stats(res.queue_wait_ms).items() if k != "n"},
                          **{f"total_age_{k}": v for k, v in stats(res.total_age_ms).items() if k != "n"},
                          srv_calls=len(srv),
                          srv_prompt_ms_mean=srv["prompt eval_ms"].mean(), srv_prompt_tok_mean=srv["prompt eval_tok"].mean(),
                          srv_gen_ms_mean=srv["eval_ms"].mean(), srv_gen_tok_mean=srv["eval_tok"].mean(),
                          srv_total_ms_mean=srv["total_ms"].mean(), srv_busy_share=srv["total_ms"].sum() / 1000 / (res.wall_clock_unix_sec.max() - res.wall_clock_unix_sec.min()), vlm_active_span_s=res.wall_clock_unix_sec.max() - res.wall_clock_unix_sec.min()))
    tpdf = pd.DataFrame(tp); tpdf.to_csv(OUT / "throughput.csv", index=False)
    pd.DataFrame(rsrc).to_csv(OUT / "resources_summary.csv", index=False)
    pd.DataFrame(vlm_s).to_csv(OUT / "vlm_summary.csv", index=False)
    pd.concat(calls_all).to_csv(OUT / "vlm_calls.csv", index=False)
    # groups per run + pooled
    out = []
    for rid, ss, gdf in pooled:
        for name in GROUPS:
            out.append(dict(run=rid, group=name, mean_ms=gdf[name].mean(), median_ms=gdf[name].median(), p95_ms=gdf[name].quantile(.95), share=gdf[name].mean() / gdf.total.mean()))
        out.append(dict(run=rid, group="TOTAL (total_delay_ms)", mean_ms=gdf.total.mean(), median_ms=gdf.total.median(), p95_ms=gdf.total.quantile(.95), share=1.0))
        out.append(dict(run=rid, group="check: groups sum - total", mean_ms=(gdf["sum"] - gdf.total).abs().mean()))
    pd.DataFrame(out).to_csv(OUT / "stage_groups_per_run.csv", index=False)
    allss = pd.concat([p[2] for p in pooled]); pr = []
    for name in GROUPS:
        pr.append(dict(group=name, mean_ms=allss[name].mean(), median_ms=allss[name].median(), p5_ms=allss[name].quantile(.05), p95_ms=allss[name].quantile(.95), std_ms=allss[name].std(), share=allss[name].mean() / allss.total.mean()))
    pr.append(dict(group="TOTAL", mean_ms=allss.total.mean(), median_ms=allss.total.median(), p5_ms=allss.total.quantile(.05), p95_ms=allss.total.quantile(.95), std_ms=allss.total.std(), share=1.0))
    pd.DataFrame(pr).to_csv(OUT / "stage_groups_pooled.csv", index=False)
    pooled_ft = pd.concat([p[1] for p in pooled]); rows = []
    for stage, parent in HIERARCHY:
        if stage in pooled_ft:
            rows.append(dict(stage=RENAME.get(stage, stage), parent=RENAME.get(parent, parent), **stats(pooled_ft[stage]), share_of_total=pooled_ft[stage].mean() / pooled_ft.total_delay_ms.mean()))
    pd.DataFrame(rows).to_csv(OUT / "stage_breakdown_pooled.csv", index=False)
    json.dump({r: b["chk"] for r, b in brk.items()}, open(OUT / "children_sum_check.json", "w"), indent=1)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
