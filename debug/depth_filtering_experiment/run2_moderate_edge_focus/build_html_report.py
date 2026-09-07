#!/usr/bin/env python3
"""Assemble report.html from stats.json + report.template.html."""
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATS = json.loads((HERE / "results" / "stats.json").read_text(encoding="utf-8"))
TEMPLATE = (HERE / "report.template.html").read_text(encoding="utf-8")
EXAMPLES_DIR = HERE / "results" / "examples"
RUN1_ARTIFACT_URL = "https://claude.ai/code/artifact/ee9bb6f5-8204-463e-a889-18794c9a2893"

agg = STATS["aggregate"]


def pct(x):
    return f"{x * 100:.1f}%"


def mm(x):
    return f"{x * 1000:.2f} mm"


def cm(x):
    return f"{x * 100:.2f} cm"


# --- Coverage & change ------------------------------------------------------
coverage_html = f"""
<div class="stat-grid">
  <div class="stat-box warn"><span class="label">Valid before</span><span class="value">{pct(agg['valid_before_frac']['mean'])}</span><span class="sub">mean, &le;5m &middot; worst frame {pct(agg['valid_before_frac']['min'])}</span></div>
  <div class="stat-box good"><span class="label">Valid after</span><span class="value">{pct(agg['valid_after_frac']['mean'])}</span><span class="sub">worst frame {pct(agg['valid_after_frac']['min'])}</span></div>
  <div class="stat-box"><span class="label">Newly filled</span><span class="value">{pct(agg['newly_filled_frac']['mean'])}</span><span class="sub">hole-filling's contribution</span></div>
  <div class="stat-box"><span class="label">Still invalid</span><span class="value">{pct(agg['still_invalid_frac']['mean'])}</span><span class="sub">unfillable even so</span></div>
  <div class="stat-box"><span class="label">Mean abs change</span><span class="value">{cm(agg['mean_abs_change_m']['mean'])}</span><span class="sub">on already-valid, in-focus pixels</span></div>
  <div class="stat-box warn"><span class="label">Max abs change</span><span class="value">{cm(agg['max_abs_change_m']['max'])}</span><span class="sub">largest single-pixel revision seen</span></div>
</div>
<p class="note" style="margin-top:0.9rem">For reference, Run 1's aggressive settings reached 97.1% mean coverage (uncapped depth range, different bag) &mdash; this run's {pct(agg['valid_after_frac']['mean'])} is lower partly because coverage is now measured only within &le;5m (a stricter bar; far-range fills that would have counted before don't count here) and partly because moderate settings genuinely fill fewer holes than maxed-out ones. That trade is the point of this run.</p>
"""

# --- Edge sharpness ----------------------------------------------------------
ratio_mean = agg["edge_sharpness_ratio"]["mean"]
ratio_median = agg["edge_sharpness_ratio"]["median"]
ratio_class = "good" if 0.9 <= ratio_mean <= 1.15 else ("warn" if ratio_mean < 0.75 else "")
edge_html = f"""
<div class="stat-grid">
  <div class="stat-box {ratio_class}"><span class="label">Edge sharpness ratio</span><span class="value">{ratio_mean:.3f}</span><span class="sub">mean, after/before depth-gradient at RGB edges</span></div>
  <div class="stat-box"><span class="label">Median ratio</span><span class="value">{ratio_median:.3f}</span><span class="sub">typical frame</span></div>
  <div class="stat-box"><span class="label">Depth gradient, before</span><span class="value">{agg['edge_grad_before_m_per_px']['mean']*100:.2f}</span><span class="sub">cm/px, at RGB-edge locations</span></div>
  <div class="stat-box"><span class="label">Depth gradient, after</span><span class="value">{agg['edge_grad_after_m_per_px']['mean']*100:.2f}</span><span class="sub">cm/px, same locations</span></div>
  <div class="stat-box"><span class="label">Edge pixels evaluated</span><span class="value">{agg['edge_pixel_count']['mean']:.0f}</span><span class="sub">mean per frame</span></div>
</div>
<div class="callout{'' if ratio_mean >= 0.9 else ' warn'}">
<strong>Reading this number:</strong> a ratio of {ratio_mean:.3f} means depth transitions at RGB-detected edges are, on average,
{"about as steep after filtering as before &mdash; edges are being preserved, not blurred, at moderate settings." if ratio_mean >= 0.9 else
 (f"roughly {(1-ratio_mean)*100:.0f}% less steep after filtering &mdash; some edge softening is still happening even at moderate settings, though" + (" markedly less than the flat-region roughness cost seen in Run 1's aggressive run would suggest." if ratio_mean >= 0.75 else " this is a real, visible amount of blur worth weighing against the coverage gained."))}
This is measured on the same bag/settings actually used, not inferred from the flat-region roughness metric &mdash; see the caveats below for what this metric does and doesn't prove.
</div>
"""

# --- Roughness ----------------------------------------------------------------
rb = agg["roughness_before_m"]["mean"]
ra = agg["roughness_after_same_footprint_m"]["mean"]
roughness_html = f"""
<div class="stat-grid">
  <div class="stat-box"><span class="label">Roughness before (3&times;3)</span><span class="value">{mm(rb)}</span><span class="sub">real measurements, &le;5m</span></div>
  <div class="stat-box{' warn' if ra > rb else ' good'}"><span class="label">Roughness after, same pixels</span><span class="value">{mm(ra)}</span><span class="sub">change: {agg['roughness_change_pct']:+.1f}%</span></div>
</div>
<p class="note">For comparison, Run 1's aggressive settings produced a +171% change at this same 3&times;3 scale (on a different bag/range). {"This run's smaller shift is consistent with moderate settings costing less flat-region smoothness, as intended." if abs(agg['roughness_change_pct']) < 171 else "This run's roughness shift is in a similar range to Run 1's aggressive result, which is worth a closer look before concluding moderate settings are strictly gentler here too."}</p>
"""

# --- Examples ------------------------------------------------------------------
def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


rows = []
for ex in STATS["examples"]:
    rgb_b64 = b64_file(EXAMPLES_DIR / ex["files"]["rgb"])
    before_b64 = b64_file(EXAMPLES_DIR / ex["files"]["depth_before"])
    after_b64 = b64_file(EXAMPLES_DIR / ex["files"]["depth_after"])
    ratio_str = f"{ex['edge_sharpness_ratio']:.2f}" if ex['edge_sharpness_ratio'] == ex['edge_sharpness_ratio'] else "n/a"
    rows.append(f"""
<div class="example-row">
  <div class="example-head">
    <span class="idx">frame {ex['index']:04d}</span>
    <span class="metrics">
      <span>valid <b class="before">{pct(ex['valid_before_frac'])}</b> &rarr; <b class="after">{pct(ex['valid_after_frac'])}</b></span>
      <span>filled {pct(ex['newly_filled_frac'])}</span>
      <span>edge ratio <b class="edge">{ratio_str}</b></span>
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

html = TEMPLATE
html = html.replace("__N_FRAMES__", f"{agg['n_frames']:,}")
html = html.replace("__RUN1_URL__", RUN1_ARTIFACT_URL)
html = html.replace("__COVERAGE_STATS__", coverage_html)
html = html.replace("__EDGE_STATS__", edge_html)
html = html.replace("__ROUGHNESS_STATS__", roughness_html)
html = html.replace("__EXAMPLES__", examples_html)

out_path = HERE / "report.html"
out_path.write_text(html, encoding="utf-8")
print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
