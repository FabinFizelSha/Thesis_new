import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_loop import load_odom, DEFAULT_BAG
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

t0, rows = load_odom(DEFAULT_BAG)
b = [r for i, r in enumerate(rows) if i % 40 == 0]   # ~5 Hz
t = np.array([r[0] for r in b]); x = np.array([r[1] for r in b]); y = np.array([r[2] for r in b])

# (first-visit t, revisit t, label, dx/dy text offset for the two annotations)
LOOPS = [
    (23.8, 221.7, "L1", (6, 6), (6, -12)),    # cut bag uHumans2_loop_L1  [15.8, 229.7]
    (239.9, 274.2, "L2", (6, 8), (6, -14)),
    (348.8, 378.0, "L3", (8, 6), (-30, -14)),  # cut bag uHumans2_loop_L3  [325.0, 388.0]
]

fig, ax = plt.subplots(figsize=(9, 8))
sc = ax.scatter(x, y, c=t, cmap="viridis", s=6)
ax.plot(x[0], y[0], "k^", ms=12, label="start (t=0)")
ax.plot(x[-1], y[-1], "ks", ms=10, label=f"end (t={t[-1]:.0f}s)")
for (tj, ti, name, o1, o2) in LOOPS:
    for tt, mk, off in ((tj, "o", o1), (ti, "*", o2)):
        k = int(np.argmin(np.abs(t - tt)))
        ax.plot(x[k], y[k], "r" + mk, ms=16, mfc="none", mew=2)
        tag = f"{name} first {tt:.0f}s" if mk == "o" else f"{name} revisit {tt:.0f}s"
        ax.annotate(tag, (x[k], y[k]), fontsize=8, xytext=off, textcoords="offset points")
ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("uHumans2 office_s1 — /tesse/odom trajectory (colour = bag time)\n"
             "circle = loop first-visit, star = revisit   |   cut bags: L1, L3")
fig.colorbar(sc, label="bag time (s)")
ax.legend(loc="best")
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "trajectory_loops.png"), dpi=140)
print("wrote trajectory_loops.png")

# tighter-radius rescan
def scan(radius, min_gap):
    hits = []
    for i in range(len(b)):
        for j in range(i):
            if b[i][0] - b[j][0] < min_gap: break
            d = math.hypot(b[i][1]-b[j][1], b[i][2]-b[j][2])
            if d <= radius:
                hits.append((b[j][0], b[i][0], d)); break
    w = []
    for (tj, ti, d) in hits:
        if w and ti - w[-1][1] < 5: w[-1] = (min(w[-1][0],tj), max(w[-1][1],ti), min(w[-1][2],d))
        else: w.append((tj, ti, d))
    return w
for R in (0.6, 0.9, 1.2):
    print(f"radius {R} m, gap>25s:")
    for (tj, ti, d) in scan(R, 25.0):
        print(f"   first {tj:6.1f}s  revisit {ti:6.1f}s  gap {ti-tj:5.1f}s  min-dist {d:.2f} m")
