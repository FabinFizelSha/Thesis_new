"""Run one configuration: 10 warm-up frames, then 3 passes over all frames (pass 1 saves + is scored, passes 2-3 time only)."""
import os, sys, json, glob, time, yaml
import numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from proposals import run_frame, BASE
from pq_eval import pq_frame, pq_summary

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DATA = f"{EXP}/data"; RES = f"{EXP}/results"
_cache = {}


def load_frames(limit=None):
    if "frames" not in _cache:
        ids = sorted(os.path.basename(f)[4:-4] for f in glob.glob(f"{DATA}/frames/rgb_*.png"))
        _cache["frames"] = [(i, cv2.imread(f"{DATA}/frames/rgb_{i}.png")[..., ::-1].copy()) for i in ids]
        _cache["gt"] = {}
        for i in ids:
            d = np.load(f"{DATA}/gt/frame_{i}.npz"); _cache["gt"][i] = (d["gt"], d["void"])
    fr = _cache["frames"]
    return fr[:limit] if limit else fr


def get_model(name):
    if name not in _cache:
        from models import make_model
        _cache[name] = make_model(name)
    return _cache[name]


def run_config(cid, model_name, overrides, n_warm=10, passes=3, limit=None, force=False):
    out = f"{RES}/{cid}"
    if os.path.exists(f"{out}/summary.json") and not force:
        return json.load(open(f"{out}/summary.json"))
    cfg = dict(BASE); cfg.update(overrides)
    os.makedirs(f"{out}/pred", exist_ok=True)
    yaml.safe_dump(dict(model=model_name, **cfg), open(f"{out}/config.yaml", "w"))
    frames = load_frames(limit); model = get_model(model_name)
    for _, rgb in frames[:n_warm]: run_frame(model, rgb, cfg)
    lat = {p: [] for p in range(passes)}; enc = {p: [] for p in range(passes)}; dec = {p: [] for p in range(passes)}
    labels1, first20 = {}, {}; nm = []
    for p in range(passes):
        for k, (i, rgb) in enumerate(frames):
            label, sc, t = run_frame(model, rgb, cfg)
            lat[p].append(t["total_ms"]); enc[p].append(t["enc_ms"]); dec[p].append(t["dec_ms"])
            if p == 0:
                np.savez_compressed(f"{out}/pred/frame_{i}.npz", label=label, scores=sc); labels1[i] = label; nm.append(t["n_masks"])
            elif p == 1 and k < 20:
                first20[i] = label
        print(f"[{cid}] pass {p+1}/{passes} median {np.median(lat[p]):.1f} ms", flush=True)
    ndiff = sum(int(not np.array_equal(labels1[i], l)) for i, l in first20.items())
    res, rows = [], []
    for k, (i, _) in enumerate(frames):
        gt, void = _cache["gt"][i]; ious, fp, fn = pq_frame(labels1[i], gt, void); res.append((ious, fp, fn))
        rows.append(dict(frame=i, TP=len(ious), FP=fp, FN=fn, sum_IoU=float(sum(ious)), n_masks=nm[k],
                         **{f"latency_ms_pass{p+1}": lat[p][k] for p in range(passes)},
                         encoder_ms=float(np.median([enc[p][k] for p in range(passes)])), decoder_ms=float(np.median([dec[p][k] for p in range(passes)]))))
    pd.DataFrame(rows).to_csv(f"{out}/per_frame.csv", index=False)
    allt = np.concatenate([lat[p] for p in range(passes)])
    summ = pq_summary(res)
    summ.update(config_id=cid, model=model_name, n_frames=len(frames), masks_per_frame_mean=float(np.mean(nm)), masks_per_frame_median=float(np.median(nm)),
                latency_median_ms=float(np.median(allt)), latency_p95_ms=float(np.percentile(allt, 95)), FPS=float(1000 / np.median(allt)),
                encoder_median_ms=float(np.median(np.concatenate([enc[p] for p in range(passes)]))),
                decoder_median_ms=float(np.median(np.concatenate([dec[p] for p in range(passes)]))),
                determinism_frames_checked=len(first20), determinism_frames_different=ndiff, **{k: v for k, v in cfg.items()})
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1)
    print(f"[{cid}] PQ {summ['PQ']:.3f} SQ {summ['SQ']:.3f} RQ {summ['RQ']:.3f}  {summ['latency_median_ms']:.1f} ms  masks {summ['masks_per_frame_mean']:.1f}  nondeterministic frames {ndiff}/{len(first20)}", flush=True)
    return summ
