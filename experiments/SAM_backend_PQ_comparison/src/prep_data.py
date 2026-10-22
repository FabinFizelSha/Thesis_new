import sys, json, glob, os, shutil, random, csv
import numpy as np, cv2
from collections import defaultdict
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from pq_eval import gt_instances

SRC = sys.argv[1]; EXTRA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "extra_src"); N_FRAMES = 500; ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MIN_PX = 3500
for d in ["frames", "gt", "class_examples", "gt_overlays"]: os.makedirs(f"{ROOT}/{d}", exist_ok=True)
src_of = {}
for d in (SRC, EXTRA):
    for f in glob.glob(f"{d}/rgb_*.npy"): src_of[os.path.basename(f)[4:-4]] = d
ids = sorted(src_of)
assert len(ids) == N_FRAMES, len(ids)
for i in ids: assert all(os.path.exists(f"{src_of[i]}/{k}_{i}.npy") for k in ("depth", "semantic"))
tm = json.load(open(f"{SRC}/metadata.json")); em = json.load(open(f"{EXTRA}/extra_metadata.json"))
json.dump({"total_frames": N_FRAMES, "note": "frames 0-299: thesis frames (phase1_frames_300); 300-499: extra frames from the same bag, interleaved",
           "frames": [dict(f, source="thesis") for f in tm["frames"]] + [dict(f, source="extra") for f in em["frames"]]},
          open(f"{ROOT}/metadata.json", "w"), indent=1)
for old in glob.glob(f"{ROOT}/gt_overlays/*") + glob.glob(f"{ROOT}/class_examples/*") + glob.glob(f"{ROOT}/gt/*") + glob.glob(f"{ROOT}/frames/*"): os.remove(old)

stats = defaultdict(lambda: [0, 0, 0]); n_obj = []; void_frac = []; depth_rng = [1e9, -1e9]; shapes = set()
pix_total = 0; example = {}
for i in ids:
    rgb = np.load(f"{src_of[i]}/rgb_{i}.npy"); sem = np.load(f"{src_of[i]}/semantic_{i}.npy"); dep = np.load(f"{src_of[i]}/depth_{i}.npy")
    assert rgb.shape == (480, 720, 3) and sem.shape == (480, 720, 3) and dep.shape == (480, 720)
    assert rgb.dtype == np.uint8 and sem.dtype == np.uint8 and dep.dtype == np.float32
    shapes.add((rgb.shape, dep.shape))
    fin = dep[np.isfinite(dep)]; depth_rng = [min(depth_rng[0], fin.min()), max(depth_rng[1], fin.max())]
    cv2.imwrite(f"{ROOT}/frames/rgb_{i}.png", rgb[..., ::-1]); cv2.imwrite(f"{ROOT}/frames/semantic_{i}.png", sem[..., ::-1])
    np.save(f"{ROOT}/frames/depth_{i}.npy", dep)
    gt, void = gt_instances(sem, MIN_PX)
    np.savez_compressed(f"{ROOT}/gt/frame_{i}.npz", gt=gt, void=void)
    n_obj.append(int(gt.max())); void_frac.append(void.mean()); pix_total += void.size
    codes = (sem[..., 0].astype(np.int64) << 16) | (sem[..., 1].astype(np.int64) << 8) | sem[..., 2]
    for c, cnt in zip(*np.unique(codes, return_counts=True)):
        s = stats[int(c)]; s[0] += int(cnt); s[1] += 1
        lab, n = ndimage.label(codes == c, structure=np.ones((3, 3)))
        s[2] += int((np.bincount(lab.ravel())[1:] >= MIN_PX).sum())
        if int(c) not in example or cnt > example[int(c)][0]: example[int(c)] = (int(cnt), i)

rows = []
for c, (px, nf, ni) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
    r, g, b = c >> 16, (c >> 8) & 255, c & 255
    rows.append((r, g, b, px, nf, ni))
    _, i = example[c]
    rgb = cv2.imread(f"{ROOT}/frames/rgb_{i}.png"); sem = cv2.imread(f"{ROOT}/frames/semantic_{i}.png")[..., ::-1]
    m = (sem == (r, g, b)).all(-1)
    out = (rgb * 0.35).astype(np.uint8); out[m] = rgb[m]
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    cv2.putText(out, f"({r},{g},{b}) frame {i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(f"{ROOT}/class_examples/{r}_{g}_{b}.png", out)
with open(f"{ROOT}/class_colours.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["r", "g", "b", "total_pixels", "n_frames", "n_instances_ge_3500px"]); w.writerows(rows)
if not os.path.exists(f"{ROOT}/surface_colours.json"): json.dump([], open(f"{ROOT}/surface_colours.json", "w"))

random.seed(0)
for i in random.sample(ids, 5):
    rgb = cv2.imread(f"{ROOT}/frames/rgb_{i}.png"); d = np.load(f"{ROOT}/gt/frame_{i}.npz"); gt, void = d["gt"], d["void"]
    rng = np.random.default_rng(int(i)); out = rgb.copy()
    for k in range(1, gt.max() + 1):
        m = gt == k; col = rng.integers(40, 255, 3); out[m] = (0.5 * out[m] + 0.5 * col).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out, cnts, -1, tuple(int(x) for x in col), 2)
    out[void] = (0.6 * out[void]).astype(np.uint8)
    cv2.imwrite(f"{ROOT}/gt_overlays/frame_{i}.png", out)

summ = dict(frames=len(ids), shapes=[str(s) for s in shapes], depth_min=float(depth_rng[0]), depth_max=float(depth_rng[1]),
            n_class_colours=len(rows), mean_gt_objects=float(np.mean(n_obj)), median_gt_objects=float(np.median(n_obj)),
            void_fraction=float(np.mean(void_frac)))
json.dump(summ, open(f"{ROOT}/prep_summary.json", "w"), indent=2); print(json.dumps(summ, indent=2))
