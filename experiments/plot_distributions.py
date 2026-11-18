#!/usr/bin/env python3
"""
plot_distributions.py -- Compare response-length distributions across runs.

Numbers-first: a prominent stats table, then two linked, INTERACTIVE panels sharing
one set of show/hide checkboxes:
  * CDF   -- read any percentile off the y-axis; best for comparison (no binning noise).
  * DENSITY -- the distribution shape "plotted separately", overlaid; toggle series on/off
              to de-clutter (this is the "toggle the list of responses on one graph").
High-contrast colourblind-safe palette (Okabe-Ito); each series carries a class ser-<i>
so the checkboxes toggle it in both panels at once. Generation cap drawn as an orange rule.
`--hist` additionally emits per-run small-multiple histograms (static).

Consumes run()'s per-run files ({"lengths":[...], "truncated":N, "config":{"max_tokens":...}})
and the legacy {label:[lengths]} dict.

USAGE (quote globs on Windows):
    python experiments/plot_distributions.py "experiments/lengths/qwen3-*_math_lengths.json"
    python experiments/plot_distributions.py "experiments/lengths/*_code_lengths.json" --out code.html
    python experiments/plot_distributions.py <files...> --linear   # linear-x
    python experiments/plot_distributions.py <files...> --hist     # + per-run histograms
"""
import json, sys, os, math, glob, html

import _bootstrap  # noqa: F401  (repo root on sys.path)
import lengths_io
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#8C564B", "#F0E442"]  # Okabe-Ito

# Loading and descriptive stats are shared (lengths_io) so every reader in the repo accepts the
# same set of length-JSON schemas; the chart code below stays local to this figure.
load = lengths_io.load_many
stats = lengths_io.stats


def make_axis(lo, hi, log):
    if log:
        a, b = math.log10(max(lo, 1)), math.log10(hi); span = (b - a) or 1.0
        pos = lambda x: (math.log10(min(max(x, lo), hi)) - a) / span
        nice = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
        ticks = [t for t in nice if lo <= t <= hi]
    else:
        span = (hi - lo) or 1.0
        pos = lambda x: (min(max(x, lo), hi) - lo) / span
        step = 10000 if hi > 40000 else 5000
        ticks = list(range(0, int(hi) + 1, step))
    return pos, ticks


def tk(t):
    return f"{t/1000:.0f}k" if t >= 1000 else str(t)


def main():
    argv = sys.argv[1:]
    out = "distributions.html"
    if "--out" in argv:
        i = argv.index("--out"); out = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    log = "--linear" not in argv
    show_hist = "--hist" in argv
    args = [a for a in argv if not a.startswith("--")]

    paths = []
    for a in args:
        g = glob.glob(a); paths.extend(sorted(g) if g else [a])
    if not paths:
        sys.exit("usage: plot_distributions.py <lengths.json ...> [--out f.html] [--linear] [--hist]")

    series = load(paths)
    if not series:
        sys.exit("no non-empty length data found")
    S = {lab: stats(L) for lab, L, _, _ in series}
    order = sorted(range(len(series)), key=lambda i: S[series[i][0]]["mean"])
    col = {i: PAL[r % len(PAL)] for r, i in enumerate(order)}

    lo = 500
    hi = max(s["p99"] for s in S.values())
    hi = max(hi, max((cap or 0) for _, _, cap, _ in series)) * 1.03
    pos, ticks = make_axis(lo, hi, log)
    caps = sorted({c for _, _, c, _ in series if c and c <= hi})

    def axes(svg, gx, gy, gw, gh, ylabels):
        for t in ticks:
            x = gx + pos(t) * gw
            svg.append(f'<line x1="{x:.0f}" y1="{gy}" x2="{x:.0f}" y2="{gy+gh}" stroke="currentColor" opacity=".06"/>')
            svg.append(f'<text x="{x:.0f}" y="{gy+gh+15}" font-size="10" text-anchor="middle" opacity=".65">{tk(t)}</text>')
        for cap in caps:
            x = gx + pos(cap) * gw
            svg.append(f'<line x1="{x:.0f}" y1="{gy}" x2="{x:.0f}" y2="{gy+gh}" stroke="#D55E00" stroke-width="1" stroke-dasharray="2 3" opacity=".5"/>')
        svg.append(f'<line x1="{gx}" y1="{gy+gh}" x2="{gx+gw}" y2="{gy+gh}" stroke="currentColor" opacity=".4"/>')
        svg.append(f'<line x1="{gx}" y1="{gy}" x2="{gx}" y2="{gy+gh}" stroke="currentColor" opacity=".4"/>')

    # ---- CDF ----
    W, H = 820, 380; gx, gy, gw, gh = 54, 20, W - 90, H - 52
    cdf = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    for yv, lbl in ((0.5, "p50"), (0.9, "p90"), (0.95, "p95")):
        y = gy + (1 - yv) * gh
        cdf.append(f'<line x1="{gx}" y1="{y:.0f}" x2="{gx+gw}" y2="{y:.0f}" stroke="currentColor" opacity=".12"/>')
        cdf.append(f'<text x="{gx-6}" y="{y+3:.0f}" font-size="10" text-anchor="end" opacity=".55">{lbl}</text>')
    axes(cdf, gx, gy, gw, gh, True)
    for i in order:
        lab, L, cap, cens = series[i]; s = S[lab]; c = col[i]; n = len(L)
        step = max(1, n // 400); pts = []
        for k in range(0, n, step):
            pts.append(f"{gx + pos(L[k])*gw:.1f},{gy + (1-(k+1)/n)*gh:.1f}")
        pts.append(f"{gx + pos(L[-1])*gw:.1f},{gy:.1f}")
        cdf.append(f'<polyline class="ser-{i}" points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="2" opacity=".92"/>')
        xm = gx + pos(s["median"]) * gw
        cdf.append(f'<circle class="ser-{i}" cx="{xm:.0f}" cy="{gy+0.5*gh:.0f}" r="2.6" fill="{c}"/>')
    cdf.append(f'<text x="{gx+gw/2:.0f}" y="{H-4}" font-size="11" text-anchor="middle" opacity=".8">response length (tokens, {"log" if log else "linear"})</text>')
    cdf.append("</svg>")

    # ---- Density (overlaid) ----
    nb = 46
    dens = {}
    dmax = 1
    for i in order:
        L = series[i][1]; b = [0] * nb
        for x in L:
            b[min(nb - 1, int(pos(x) * nb))] += 1
        b = [c / len(L) for c in b]
        dens[i] = b; dmax = max(dmax, max(b))
    W2, H2 = 820, 300; dx, dy, dw, dh = 54, 16, W2 - 90, H2 - 48
    den = [f'<svg viewBox="0 0 {W2} {H2}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    axes(den, dx, dy, dw, dh, False)
    for i in order:
        c = col[i]; b = dens[i]; pts = []
        for j, v in enumerate(b):
            pts.append(f"{dx + (j+0.5)/nb*dw:.1f},{dy + dh - v/dmax*dh:.1f}")
        den.append(f'<polyline class="ser-{i}" points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="2" opacity=".85"/>')
    den.append(f'<text x="{dx+dw/2:.0f}" y="{H2-4}" font-size="11" text-anchor="middle" opacity=".8">response length (tokens, {"log" if log else "linear"})</text>')
    den.append("</svg>")

    # ---- controls (toggle both panels) ----
    chips = []
    for i in order:
        lab = series[i][0]; s = S[lab]; c = col[i]
        chips.append(
            f'<label class="chip"><input type="checkbox" data-ser="{i}" checked>'
            f'<span class="sw" style="background:{c}"></span>{html.escape(lab)} '
            f'<span class="mu">&mu;{s["mean"]/1000:.1f}k</span></label>')
    controls = ('<div class="controls"><button id="all">all</button><button id="none">none</button> '
                + " ".join(chips) + "</div>")

    # ---- table ----
    hdr = ["run", "n", "mean", "median", "cv", "p90", "p95", "p99", "max", "cens%", "cap"]
    trs = []
    for i in order:
        lab, L, cap, cens = series[i]; s = S[lab]; c = col[i]
        cens_cell = f'<td class="{"warn" if cens>0.02 else ""}">{100*cens:.0f}%</td>'
        capcell = f"{cap:,}" if cap else "-"
        trs.append(
            f'<tr><td><span class="sw" style="background:{c}"></span>{html.escape(lab)}</td>'
            f'<td>{s["n"]}</td><td>{s["mean"]:,.0f}</td><td>{s["median"]:,.0f}</td><td>{s["cv"]:.2f}</td>'
            f'<td>{s["p90"]:,.0f}</td><td>{s["p95"]:,.0f}</td><td>{s["p99"]:,.0f}</td><td>{s["max"]:,.0f}</td>'
            f'{cens_cell}<td>{capcell}</td></tr>')
    table = f'<table><tr>{"".join(f"<th>{h}</th>" for h in hdr)}</tr>{"".join(trs)}</table>'

    hist_html = ""
    if show_hist:
        panels = []
        for i in order:
            lab, L, cap, cens = series[i]; c = col[i]; b = dens[i]; bm = max(b) or 1
            PW, PH = 390, 120; hx, hy, hw, hh = 38, 22, PW - 54, PH - 42
            p = [f'<svg viewBox="0 0 {PW} {PH}" width="{PW}" height="{PH}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
            p.append(f'<text x="{PW/2:.0f}" y="12" font-size="11" text-anchor="middle" font-weight="600">{html.escape(lab)}</text>')
            for cap2 in caps:
                if cap2 <= hi:
                    xc = hx + pos(cap2) * hw
                    p.append(f'<line x1="{xc:.0f}" y1="{hy}" x2="{xc:.0f}" y2="{hy+hh}" stroke="#D55E00" stroke-dasharray="2 3" opacity=".5"/>')
            bw = hw / nb
            for j, v in enumerate(b):
                if v:
                    h = v / bm * hh
                    p.append(f'<rect x="{hx+j*bw:.1f}" y="{hy+hh-h:.1f}" width="{bw-0.5:.1f}" height="{h:.1f}" fill="{c}" opacity=".85"/>')
            for t in ticks:
                p.append(f'<text x="{hx+pos(t)*hw:.0f}" y="{hy+hh+11}" font-size="8" text-anchor="middle" opacity=".6">{tk(t)}</text>')
            p.append("</svg>")
            panels.append(f"<div>{''.join(p)}</div>")
        hist_html = ('<h2>Per-run histograms</h2><div class="grid">' + "".join(panels) + "</div>")

    open(out, "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8>
<title>Response-length comparison</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.45}}
@media(prefers-color-scheme:dark){{body{{background:#141414;color:#eee}}}}
h1{{font-size:18px}}h2{{font-size:15px;margin:22px 0 6px}}.cap{{font-size:12.5px;opacity:.72}}
table{{border-collapse:collapse;font-size:13px;margin:6px 0;width:100%}}
th,td{{border-bottom:1px solid #8883;padding:4px 9px;text-align:right;font-variant-numeric:tabular-nums}}
th:first-child,td:first-child{{text-align:left}}th{{opacity:.7}}tr:hover td{{background:#8881}}
td.warn{{color:#D55E00;font-weight:600}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.controls{{margin:10px 0;font-size:12.5px}}.chip{{display:inline-block;margin:2px 8px 2px 0;cursor:pointer;user-select:none}}
.chip .mu{{opacity:.6}}.controls button{{font-size:12px;margin-right:4px;cursor:pointer}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}
</style>
<h1>Response-length comparison &mdash; {len(series)} runs</h1>
<p class=cap>Sorted by mean. Orange rule = generation cap; <b>cens%</b> flagged when a mean is a lower bound. Toggle series with the checkboxes &mdash; they control both panels.</p>
{table}
{controls}
<h2>CDF</h2>
<p class=cap>Percentiles read straight off y (median = p50 crossing; dot marks it). A curve going vertical at the orange cap = mass truncated there.</p>
{''.join(cdf)}
<h2>Density (overlaid)</h2>
<p class=cap>Distribution shape. Uncheck series above to de-clutter.</p>
{''.join(den)}
{hist_html}
<p class=cap>Files: {html.escape(', '.join(os.path.basename(p) for p in paths))}</p>
<script>
function setSer(i,on){{document.querySelectorAll('.ser-'+i).forEach(function(el){{el.style.display=on?'':'none';}});}}
document.querySelectorAll('input[data-ser]').forEach(function(cb){{
  cb.addEventListener('change',function(){{setSer(cb.dataset.ser,cb.checked);}});}});
document.getElementById('all').onclick=function(){{document.querySelectorAll('input[data-ser]').forEach(function(cb){{cb.checked=true;setSer(cb.dataset.ser,true);}});}};
document.getElementById('none').onclick=function(){{document.querySelectorAll('input[data-ser]').forEach(function(cb){{cb.checked=false;setSer(cb.dataset.ser,false);}});}};
</script>
""")
    print(f"wrote {out}  ({len(series)} runs, {'log' if log else 'linear'}-x, hi={hi:,.0f})")


if __name__ == "__main__":
    main()
