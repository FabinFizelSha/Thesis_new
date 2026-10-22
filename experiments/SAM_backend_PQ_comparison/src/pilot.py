import sys, json; sys.path.insert(0, "src")
from runner import run_config
rows = []
for m, tag in (("nanosam", "nano"), ("sam_vitb", "vitb")):
    for g in (3, 5, 7):
        rows.append(run_config(f"pilot_A_{tag}_{g}", m, dict(points_per_side=g), limit=50))
