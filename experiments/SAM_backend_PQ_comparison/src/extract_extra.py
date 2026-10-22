"""Extract 200 extra frames from the uHumans2 office bag (rosbags, no ROS), interleaved between the 300 thesis frames."""
import json, os, numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
SRC = Path.home() / "Thesis_pipeline_split_lean/experiments/SAM_optimisation_experiment/SAM_optimisation_experiment/phase_1/datasets/phase1_frames_300"
BAG = Path.home() / "datasets/uhumans2/office_s1_00h_v2"
OUT = Path(__file__).resolve().parent.parent / "data" / "extra_src"; OUT.mkdir(parents=True, exist_ok=True)
TOPICS = {"rgb": "/tesse/left_cam/rgb/image_raw", "depth": "/tesse/depth_cam/mono/image_raw", "semantic": "/tesse/seg_cam/rgb/image_raw"}
TOL = 0.1; N_EXTRA = 200
meta = json.load(open(SRC / "metadata.json")); thesis_ts = np.array([f["original_bag_timestamp"] for f in meta["frames"]])

def decode(m, kind):
    if kind == "depth":
        if m.encoding == "32FC1": a = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
        elif m.encoding == "16UC1": a = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
        else: raise ValueError(f"depth encoding {m.encoding}")
        return a
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
    if m.encoding == "rgb8": return (a[..., 2::-1] if kind == "semantic" else a[..., :3]).copy()  # thesis semantic frames store the seg_cam channels reversed (verified on frame 0)
    if m.encoding == "bgr8": return (a[..., :3] if kind == "semantic" else a[..., 2::-1]).copy()
    raise ValueError(f"colour encoding {m.encoding}")

with AnyReader([BAG]) as r:
    conns = {k: [c for c in r.connections if c.topic == t] for k, t in TOPICS.items()}
    # candidate stamps = rgb message times midway between consecutive thesis frames (+0.75 s), within the bag
    t0, t1 = r.start_time / 1e9, r.end_time / 1e9
    cand = thesis_ts + 0.75; cand = cand[(cand > t0 + TOL) & (cand < t1 - TOL)]
    cand = cand[np.linspace(0, len(cand) - 1, N_EXTRA).round().astype(int)]
    targets = [("val", thesis_ts[0])] + [("extra", float(t)) for t in cand]   # frame 0 is re-extracted to validate the decoding
    print(len(cand), "candidates; bag", t0, t1)
    best = {k: [(1e9, None, None) for _ in targets] for k in TOPICS}
    tarr = np.array([t for _, t in targets])
    for k in TOPICS:
        for c, t, raw in r.messages(connections=conns[k]):
            ts = t / 1e9; i = int(np.argmin(np.abs(tarr - ts))); d = abs(tarr[i] - ts)
            if d < best[k][i][0]: best[k][i] = (d, raw, c)
        print(k, "scanned", flush=True)
    frames = []; skipped = []; nid = 300
    for i, (kind, t) in enumerate(targets):
        if any(best[k][i][0] > TOL for k in TOPICS): skipped.append(t); continue
        arrs = {k: decode(r.deserialize(best[k][i][1], best[k][i][2].msgtype), k) for k in TOPICS}
        if kind == "val":
            for k in TOPICS:
                ref = np.load(SRC / f"{k}_000000.npy"); print("validate frame 0", k, ref.shape, arrs[k].shape, "identical:", np.array_equal(ref, arrs[k]), "maxdiff", float(np.abs(ref.astype(float) - arrs[k]).max()))
            continue
        for k in TOPICS: np.save(OUT / f"{k}_{nid:06d}.npy", arrs[k])
        frames.append({"frame_id": nid, "original_bag_timestamp": t, "dt_rgb": best["rgb"][i][0], "dt_depth": best["depth"][i][0], "dt_semantic": best["semantic"][i][0]}); nid += 1
json.dump({"frames": frames, "skipped": skipped}, open(OUT / "extra_metadata.json", "w"), indent=1)
print("extracted", len(frames), "skipped", len(skipped))
