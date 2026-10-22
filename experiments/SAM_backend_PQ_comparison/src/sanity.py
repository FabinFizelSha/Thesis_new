import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pq_eval import pq_frame, pq_summary
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gt"); rng = np.random.default_rng(0)
ids = sorted(os.path.basename(f)[6:-4] for f in glob.glob(f"{D}/frame_*.npz")); ids = list(rng.choice(ids, 5, replace=False))
lines, ok = ["# Sanity checks (5 frames: " + ", ".join(ids) + ")", ""], True
def tot(fr): return sum(len(f[0]) for f in fr), sum(f[1] for f in fr), sum(f[2] for f in fr)
for i in ids:
    d = np.load(f"{D}/frame_{i}.npz"); gt, void = d["gt"], d["void"]; n = int(gt.max()); ok_f = True
    base = pq_frame(gt.copy(), gt, void); s = pq_summary([base])
    c1 = abs(s["PQ"] - 1) < 1e-12 and abs(s["SQ"] - 1) < 1e-12 and abs(s["RQ"] - 1) < 1e-12 and tot([base]) == (n, 0, 0)
    perm = rng.permutation(n) + 1; lut = np.concatenate([[0], perm]); c2 = tot([pq_frame(lut[gt].astype(np.int32), gt, void)]) == tot([base])
    areas = np.bincount(gt.ravel())[1:]; big = int(np.argmax(areas)) + 1
    ys, xs = np.nonzero(gt == big); cut = np.quantile(xs, 0.7); sp = gt.copy(); sp[(gt == big) & (np.arange(gt.shape[1])[None, :] > cut)] = n + 1
    t = tot([pq_frame(sp, gt, void)]); c3 = t == (n, 1, 0)
    order = np.argsort(areas); a, b = int(order[-1]) + 1, int(order[0]) + 1; mg = gt.copy(); mg[gt == b] = a
    t4 = tot([pq_frame(mg, gt, void)]); c4 = n >= 2 and t4 == (n - 1, 0, 1)
    ok &= c1 and c2 and c3 and c4
    lines.append(f"- frame {i} ({n} objects): identity PQ=SQ=RQ=1 {'PASS' if c1 else 'FAIL'}; shuffled ids same TP/FP/FN {'PASS' if c2 else 'FAIL'}; split (70/30) -> +1 FP {'PASS' if c3 else 'FAIL'} {t}; merge -> +1 FN {'PASS' if c4 else 'FAIL'} {t4}")
lines += ["", "Overall: " + ("ALL PASS" if ok else "FAILED")]
open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "sanity.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines)); sys.exit(0 if ok else 1)
