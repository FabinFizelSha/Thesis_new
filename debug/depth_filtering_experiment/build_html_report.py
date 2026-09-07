#!/usr/bin/env python3
"""Assemble report.html from stats.json + report.template.html.
Keeps large base64 image payloads out of the conversation entirely."""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATS = json.loads((HERE / "results" / "stats.json").read_text(encoding="utf-8"))
TEMPLATE = (HERE / "report.template.html").read_text(encoding="utf-8")
EXAMPLES_DIR = HERE / "results" / "examples"


def pct(x):
    return f"{x * 100:.1f}%"


def mm(x):
    return f"{x * 1000:.2f} mm"


def cm(x):
    return f"{x * 100:.2f} cm"


agg = STATS["aggregate"]

# --- Coverage section ---------------------------------------------------
coverage_html = f"""
<div class="stat-grid">
  <div class="stat-box warn"><span class="label">Valid before</span><span class="value">{pct(agg['valid_before_frac']['mean'])}</span><span class="sub">mean across {agg['n_frames']} frames &middot; worst {pct(agg['valid_before_frac']['min'])}</span></div>
  <div class="stat-box good"><span class="label">Valid after</span><span class="value">{pct(agg['valid_after_frac']['mean'])}</span><span class="sub">worst frame {pct(agg['valid_after_frac']['min'])}</span></div>
  <div class="stat-box"><span class="label">Newly filled</span><span class="value">{pct(agg['newly_filled_frac']['mean'])}</span><span class="sub">hole-filling's direct contribution</span></div>
  <div class="stat-box"><span class="label">Still invalid</span><span class="value">{pct(agg['still_invalid_frac']['mean'])}</span><span class="sub">holes too large even for aggressive filling</span></div>
</div>
<div class="bar-row" style="margin-top:1rem">
  <div class="bar-row-head"><span class="name">Mean composition &mdash; before</span></div>
  <div class="bar-track">
    <div class="bar-fill before" style="width:{agg['valid_before_frac']['mean']*100:.1f}%">{pct(agg['valid_before_frac']['mean'])} valid</div>
  </div>
  <div class="bar-row-head"><span class="name">Mean composition &mdash; after (measured + filled)</span></div>
  <div class="bar-track">
    <div class="bar-fill before" style="width:{agg['valid_before_frac']['mean']*100:.1f}%">measured</div><div class="bar-fill filled" style="width:{agg['newly_filled_frac']['mean']*100:.1f}%">filled</div>
  </div>
</div>
"""

# --- Roughness section ----------------------------------------------------
def roughness_row(label, tag):
    b = agg[f"roughness_{tag}_before_m"]["mean"]
    a = agg[f"roughness_{tag}_after_same_footprint_m"]["mean"]
    f = agg[f"roughness_{tag}_filled_region_m"]["mean"]
    change_pct = -agg[f"roughness_{tag}_reduction_pct"]
    sign = "+" if change_pct >= 0 else ""
    return f"<tr><td>{label}</td><td>{mm(b)}</td><td>{mm(a)}</td><td>{sign}{change_pct:.0f}%</td><td>{mm(f)}</td></tr>"

roughness_html = f"""
<div class="card">
  <table class="roughness">
    <tr><th>Window</th><th>Before (real)</th><th>After, same pixels</th><th>Change</th><th>Hole-filled territory</th></tr>
    {roughness_row("3&times;3 (fine)", "fine")}
    {roughness_row("9&times;9 (coarse)", "coarse")}
  </table>
</div>
"""

fine_b = agg["roughness_fine_before_m"]["mean"]
fine_a = agg["roughness_fine_after_same_footprint_m"]["mean"]
coarse_b = agg["roughness_coarse_before_m"]["mean"]
coarse_a = agg["roughness_coarse_after_same_footprint_m"]["mean"]
fine_filled = agg["roughness_fine_filled_region_m"]["mean"]
mean_change_cm = agg["mean_abs_change_m"]["mean"] * 100

coarse_filled = agg["roughness_coarse_filled_region_m"]["mean"]
mean_change_ratio_fine = (agg['mean_abs_change_m']['mean'] / fine_a) if fine_a else float("nan")
mean_change_ratio_coarse = (agg['mean_abs_change_m']['mean'] / coarse_a) if coarse_a else float("nan")

roughness_interpretation = f"""
<div class="callout warn">
<strong>Honest reading of this number: roughness on already-good pixels increased at both scales,
and more so at the coarser one.</strong> 3&times;3 roughness rose from {mm(fine_b)} to {mm(fine_a)}
(+{-agg['roughness_fine_reduction_pct']:.0f}%); 9&times;9 roughness rose from {mm(coarse_b)} to
{mm(coarse_a)} (+{-agg['roughness_coarse_reduction_pct']:.0f}%) &mdash; proportionally the larger jump,
which rules out "fine dither only" as the full story. The most likely mechanism: every parameter pushed
to its most aggressive setting (&sect;2) is an <em>edge-preserving</em> filter with a hard threshold
(<code>spatial_smooth_delta</code>, <code>temporal_smooth_delta</code>) gating which neighbors get
blended. At maximum aggressiveness this is known to trade smooth gradients for piecewise-flat
"plateaus" &mdash; individually low-roughness at 3&times;3, but stepping between adjacent plateaus shows
up as elevated roughness at a window wide enough to span more than one of them, which is exactly what a
9&times;9 window is more likely to do than a 3&times;3 one. In absolute terms this is still small: the
coarse-scale increase ({mm(coarse_a)}) is about {mean_change_ratio_coarse:.0f}&times; smaller than the
{cm(agg['mean_abs_change_m']['mean'])} mean (and roughly {agg['max_abs_change_m']['max']/coarse_a:.0f}&times;
smaller than the {cm(agg['max_abs_change_m']['max'])} max) the filter is actually revising already-valid
pixels by &mdash; so real corrective smoothing at a coarser scale is still the dominant effect, this is a
secondary one riding on top of it. It does not reverse the headline result (coverage
{pct(agg['valid_before_frac']['mean'])} &rarr; {pct(agg['valid_after_frac']['mean'])}), but it is a real,
concrete cost of choosing the most aggressive setting on every parameter rather than a moderate one, and
worth weighing if smooth large-scale geometry matters more than maximum coverage for a given use of this
bag. A natural follow-up (not run here) would repeat this same before/after analysis with each filter
parameter at its library default instead of its extreme, to see where the coverage/roughness trade-off
actually bends.<br><br>
Separately, and expectedly, roughness inside hole-filled territory is higher still
({mm(fine_filled)} at 3&times;3, {mm(coarse_filled)} at 9&times;9) than either real-measurement
population &mdash; that territory is extrapolated, not sensed, and <code>nearest_from_around</code>
fill can create patchwork seams at the boundaries of what it fills.
</div>
"""

# --- Change magnitude section ---------------------------------------------
change_html = f"""
<div class="stat-grid">
  <div class="stat-box"><span class="label">Mean abs change (already-valid px)</span><span class="value">{cm(agg['mean_abs_change_m']['mean'])}</span><span class="sub">per frame, on pixels valid both before and after</span></div>
  <div class="stat-box warn"><span class="label">Max abs change (any frame)</span><span class="value">{cm(agg['max_abs_change_m']['max'])}</span><span class="sub">largest single-pixel revision seen</span></div>
  <div class="stat-box"><span class="label">Median mean-change</span><span class="value">{cm(agg['mean_abs_change_m']['median'])}</span><span class="sub">typical frame</span></div>
</div>
"""

# --- Examples gallery -------------------------------------------------------
def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


rows = []
for ex in STATS["examples"]:
    rgb_b64 = b64_file(EXAMPLES_DIR / ex["files"]["rgb"])
    before_b64 = b64_file(EXAMPLES_DIR / ex["files"]["depth_before"])
    after_b64 = b64_file(EXAMPLES_DIR / ex["files"]["depth_after"])
    rows.append(f"""
<div class="example-row">
  <div class="example-head">
    <span class="idx">frame {ex['index']:04d}</span>
    <span class="metrics">
      <span>valid <b class="before">{pct(ex['valid_before_frac'])}</b> &rarr; <b class="after">{pct(ex['valid_after_frac'])}</b></span>
      <span>filled {pct(ex['newly_filled_frac'])}</span>
      <span>still invalid {pct(ex['still_invalid_frac'])}</span>
    </span>
  </div>
  <div class="triptych">
    <figure><img src="data:image/jpeg;base64,{rgb_b64}" alt="RGB frame {ex['index']}"><figcaption>RGB (rectified)</figcaption></figure>
    <figure><img src="data:image/png;base64,{before_b64}" alt="Depth before, frame {ex['index']}"><figcaption>depth &middot; before</figcaption></figure>
    <figure><img src="data:image/png;base64,{after_b64}" alt="Depth after, frame {ex['index']}"><figcaption>depth &middot; after</figcaption></figure>
  </div>
</div>
""")

examples_html = "\n".join(rows)

# --- Assemble ---------------------------------------------------------------
html = TEMPLATE
html = html.replace("__N_FRAMES__", f"{agg['n_frames']:,}")
html = html.replace("__COVERAGE_STATS__", coverage_html)
html = html.replace("__ROUGHNESS_STATS__", roughness_html)
html = html.replace("__ROUGHNESS_INTERPRETATION__", roughness_interpretation)
html = html.replace("__CHANGE_STATS__", change_html)
html = html.replace("__EXAMPLES__", examples_html)

out_path = HERE / "report.html"
out_path.write_text(html, encoding="utf-8")
print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
