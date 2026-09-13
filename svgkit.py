#!/usr/bin/env python3
"""
svgkit.py -- Shared inline-SVG chart primitives for this project's self-contained HTML figures.

No matplotlib anywhere in the repo: every figure is hand-written SVG in one HTML file, so it
opens offline, diffs readably and needs no plotting stack. These primitives were duplicated
across the plotters; they live here now.

Colour/theme convention: charts reference CSS variables (--fg/--mut/--grid) so one page works
in light and dark; PAGE_CSS defines them. PAL is the categorical series palette.

Chart primitives
  grouped_bars    log- (or linear-) scaled grouped bars; None-valued bars render as a dashed
                  "missing" slot whose label comes from annotfn (a floor/fallback, not a gap)
  small_multiples one mini panel per item in a row, each with its OWN zero-based linear axis
                  and optional error whiskers -- for predicted-vs-published across anchors
                  whose absolute scales differ by an order of magnitude
  heatmap         rows x cols matrix; numeric cells shade sequentially, string cells are tags
  signed_hbars    diverging bars around zero (e.g. +/- error %), single-hue magnitude ramp
  tornado         horizontal ranges around a 1.0x baseline, for sensitivity sweeps

Every quantitative primitive takes an axis-title argument (ylab / rowaxis+colaxis / xlab).
Pass it: an unlabelled axis is where readers mis-infer what is being counted.
"""
import html
import math

esc = html.escape

# Font-size bounds for grouped_bars' rotated in-bar notes (see `notefn`): shrink to fit the bar's
# height, give up below the minimum rather than render something illegible.
NOTE_FS_MAX, NOTE_FS_MIN = 7.5, 5.2
# Chars of a font-size-15 title that fit inside the 820-unit viewBox (~7.5 units/char, system-ui).
# Past it the text is simply outside the viewBox and never renders -- silent mid-sentence truncation.
TITLE_CPL, TITLE_LH = 106, 17


def wrap_title(title, cpl=TITLE_CPL):
    """Greedy word-wrap for chart titles -> list of lines. An over-long single word is left to
    overhang (breaking mid-token reads worse than one long line)."""
    lines, cur = [], ""
    for word in str(title).split():
        cand = f"{cur} {word}".strip()
        if cur and len(cand) > cpl:
            lines.append(cur); cur = word
        else:
            cur = cand
    return (lines + [cur]) if cur else [""]

# Categorical series palette (Vega/Tableau-ish; distinguishable in both themes).
PAL = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#EECA3B", "#9D755D"]
# Colourblind-safe alternative (Okabe-Ito), used by the distribution plots.
PAL_OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000",
                 "#8C564B", "#F0E442"]

PAGE_CSS = """
:root{--bg:#fff;--fg:#111;--mut:#777;--grid:#e6e6e6;--line:#ddd;--head:#f4f6f9}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0f0f0f;--fg:#eee;--mut:#999;
--grid:#2a2a2a;--line:#333;--head:#181a1e}}
body{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;margin:0 auto;
padding:22px;max-width:880px}
h1{font-size:20px;margin:0 0 2px}
h2{font-size:15px;margin:26px 0 4px;padding:6px 10px;background:var(--head);border-radius:6px}
h3{font-size:13.5px;margin:20px 0 4px;padding:4px 10px;border-left:3px solid var(--mut);background:var(--head);border-radius:0 4px 4px 0}
p.sub{color:var(--mut);margin:0 0 6px;font-size:12.5px}
.panel{border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:right;padding:3px 7px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
.inf{color:var(--mut);text-align:center}th{color:var(--mut);font-weight:600}
"""


def page(title, body, css=PAGE_CSS):
    """Wrap body HTML in a minimal self-contained document."""
    return f"<title>{esc(title)}</title>\n<style>{css}</style>\n{body}\n"


def panel(fig):
    return f'<div class="panel">{fig}</div>'


def lerp_hex(a, b, t):
    """Interpolate two #rrggbb colours."""
    a, b = a.lstrip("#"), b.lstrip("#")
    v = [round(int(a[i:i+2], 16) + (int(b[i:i+2], 16) - int(a[i:i+2], 16)) * t) for i in (0, 2, 4)]
    return f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}"


def pow_grid(vmin, vmax):
    """Decade gridline values spanning [vmin, vmax]."""
    g, p = [], math.floor(math.log10(vmin))
    while 10 ** p <= vmax * 1.001:
        if 10 ** p >= vmin * 0.999:
            g.append(10 ** p)
        p += 1
    return g or [vmin, vmax]


def nice_ticks(vmax, n=4):
    """~`n` round gridline values from 0 up to >= vmax, for a ZERO-BASED LINEAR axis.
    Steps snap to 1/2/2.5/5 x a power of ten so tick labels stay readable."""
    if not (vmax > 0 and math.isfinite(vmax)):
        return [0.0]
    raw = vmax / max(1, n)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    out, v = [], 0.0
    while v <= vmax * 1.0001:
        out.append(v)
        v += step
    return out


def fmt_si(x):
    """Compact axis/bar label: 1.2M / 3.4k / 57. Good for GPU/node counts; for FLOP-scale values
    (1e18+) use fmt_sci instead -- k/M notation stops being readable well before that range."""
    if not math.isfinite(x):
        return "inf"
    for d, u in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= d:
            return f"{x/d:.1f}{u}"
    return f"{x:.0f}"


def fmt_sci(x):
    """Scientific-notation axis/bar label: 1.2e+21. Pass as grouped_bars(..., labelfn=fmt_sci)
    for FLOP-valued charts, where fmt_si's k/M suffixes would either be unreadable (1e21 has no
    named suffix) or, worse, print a garbled digit string if divided by the wrong scale."""
    if not math.isfinite(x):
        return "inf"
    return f"{x:.1e}"


def grouped_bars(cats, series, valfn, title, w=820, h=300, annotfn=None, labelfn=fmt_si,
                 ylab=None, notefn=None, scale="log", hlines=None, hollowfn=None):
    """Grouped bars, log-scaled by default (`scale="linear"` for a zero-based linear axis).

    `valfn(cat, series_label) -> number or None`. A None value is drawn as a dashed outline
    instead of being dropped, labelled with `annotfn(cat, series_label)` -- so "no bar" can
    still carry information (a fallback value, a reason) rather than reading as missing data.

    ylab   -- y-axis title, drawn rotated at the left edge. Always pass one: a bare number axis
              is exactly where a reader mis-infers *what* is being counted (GPUs? FLOP? seconds?).
    notefn -- optional `notefn(cat, series_label) -> str` drawn rotated INSIDE each drawn bar,
              for a second quantity sharing the category/series structure but not the axis
              (e.g. the training-step count behind a min-GPU bar).
    hollowfn -- optional `hollowfn(cat, series_label) -> str|None`. A truthy return draws that
              bar as a dashed OUTLINE at its real height, labelled with the string rotated
              inside: for a value that is real but fails the chart's own criterion (e.g. the
              GPUs at the fastest achievable wall-clock when the window is missed). Keeps the
              number on the axis, where a None value's full-height box would discard it.
    hlines -- optional [(label, value)] full-width dashed reference lines sharing the bars' own
              axis/units (e.g. published model compute budgets on a compute-FLOP axis), so a
              reader can place the bars against a known scale. Included in the axis's own
              min/max so a line outside the bars' range still pulls the axis to show it, rather
              than being silently clipped.
    """
    vals = [v for c in cats for s in series if (v := valfn(c, s[0])) is not None]
    vals += [v for _, v in hlines] if hlines else []
    # hlines' labels live in a dedicated right margin (not overlaid on the plot) so a bar sitting
    # at the same height never occludes them -- sized to the longest label, not a fixed guess.
    pr = max(12, max((len(str(lab)) for lab, _ in hlines), default=0) * 5.3 + 14) if hlines else 12
    # A wrapped title GROWS the canvas rather than eating the plot, so a three-line title never
    # squashes its bars relative to a one-line one.
    tlines = wrap_title(title)
    extra = TITLE_LH * (len(tlines) - 1)
    h += extra
    pl, pt, pb = (78 if ylab else 60), 40 + extra, 54
    x0, x1, y0, y1 = pl, w - pr, pt, h - pb
    if scale == "linear":
        vmin, vmax = 0.0, ((max(vals) * 1.15) if vals else 1.0)
        grid = nice_ticks(vmax)
        yy = lambda v: y1 - (min(max(v, 0.0), vmax) / vmax) * (y1 - y0)
    else:
        # No 1.0 floor on vmin: that assumed every charted quantity is a count >= 1 (GPUs, steps,
        # tokens, FLOP), which held until required-bandwidth figures came along that legitimately
        # sit below 1 Mbps -- clamping those to the axis floor made two whole decades of the data
        # render as identical zero-height bars.
        pos = [v for v in vals if v > 0]
        vmin, vmax = ((min(pos) / 1.6, max(pos) * 1.4) if pos else (1, 10))
        grid = pow_grid(vmin, vmax)
        glo, ghi = math.log10(vmin), math.log10(vmax)
        yy = lambda v: y1 - (math.log10(max(v, vmin)) - glo) / (ghi - glo) * (y1 - y0)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" font-family="system-ui,sans-serif">']
    out += [f'<text x="12" y="{22 + i * TITLE_LH}" font-size="15" font-weight="600" '
            f'fill="var(--fg)">{esc(line)}</text>' for i, line in enumerate(tlines)]
    if ylab:
        ycen = (y0 + y1) / 2
        out.append(f'<text x="17" y="{ycen:.1f}" font-size="10.5" text-anchor="middle" '
                   f'fill="var(--mut)" transform="rotate(-90 17 {ycen:.1f})">{esc(ylab)}</text>')
    for g in grid:
        if g > vmax:
            continue
        y = yy(g)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="var(--grid)"/>')
        out.append(f'<text x="{x0-6}" y="{y+3:.1f}" font-size="10" text-anchor="end" '
                   f'fill="var(--mut)">{labelfn(g)}</text>')
    # Reference hlines: full-width dashed lines at a caller-given scale (e.g. published model
    # compute budgets), distinct from the plain `grid` lines above -- a visible color (not
    # var(--grid)) and a labelled tag so a reader can place the bars against a known quantity.
    # The line crosses the plot (so its height against the bars is legible) but the LABEL sits in
    # the dedicated right margin computed above, clear of every bar -- putting it inside the plot
    # (even with a background pill) got hidden under whichever bar happened to reach that height.
    for lab, v in (hlines or []):
        if v < vmin or v > vmax:
            continue
        y = yy(v)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="var(--fg)" '
                   f'stroke-width="1" stroke-dasharray="5 3" opacity="0.55"/>')
        out.append(f'<text x="{x1+5:.1f}" y="{y+3:.1f}" font-size="9" text-anchor="start" '
                   f'fill="var(--fg)" opacity="0.85">{esc(str(lab))}</text>')
    gw = (x1 - x0) / len(cats); bw = gw / (len(series) + 1)
    for i, cat in enumerate(cats):
        gx = x0 + i * gw
        for line_i, part in enumerate(str(cat).split("\n")):
            out.append(f'<text x="{gx+gw/2:.1f}" y="{h-pb+15+line_i*11}" font-size="10" '
                       f'text-anchor="middle" fill="var(--fg)">{esc(part)}</text>')
        for j, (slab, col) in enumerate(series):
            bx = gx + bw * (j + 0.5); v = valfn(cat, slab)
            if v is not None:
                by = yy(v)
                # Hollow: real height, dashed outline (the same visual language as the None box),
                # so an off-criterion value stays readable on the axis instead of being discarded.
                hol = hollowfn(cat, slab) if hollowfn else None
                style = (f'fill="none" stroke="{col}" stroke-dasharray="3 3" opacity="0.75"'
                         if hol else f'fill="{col}"')
                out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw*0.9:.1f}" '
                           f'height="{max(0.0, y1-by):.1f}" {style}/>')
                out.append(f'<text x="{bx+bw*0.45:.1f}" y="{by-3:.1f}" font-size="8.5" '
                           f'text-anchor="middle" fill="var(--mut)">{labelfn(v)}</text>')
                # Rotated -90, so the note's text length is bounded by the bar's HEIGHT, not its
                # width. Shrink the font to fit rather than truncating or overflowing (overflowing
                # white text lands on the page background and reads as a rendering fault); below
                # NOTE_FS_MIN it would be unreadable anyway, so drop it instead.
                note = hol or (notefn(cat, slab) if notefn else None)
                if note:
                    room = (y1 - by) - 8
                    # ~0.62 * font-size per char is a decent sans-serif advance-width estimate.
                    fs = min(NOTE_FS_MAX, room / (len(str(note)) * 0.62)) if room > 0 else 0
                    if fs >= NOTE_FS_MIN:
                        ncx, ncy = bx + bw * 0.45, (by + y1) / 2
                        nfill = "var(--mut)" if hol else "#fff"   # no fill to sit on when hollow
                        out.append(f'<text x="{ncx:.1f}" y="{ncy:.1f}" font-size="{fs:.1f}" '
                                   f'text-anchor="middle" fill="{nfill}" '
                                   f'transform="rotate(-90 {ncx:.1f} {ncy:.1f})">{esc(note)}</text>')
            else:
                lab = (annotfn(cat, slab) if annotfn else "n/a")
                cx, cy = bx + bw * 0.45, (y0 + y1) / 2
                out.append(f'<rect x="{bx:.1f}" y="{y0:.1f}" width="{bw*0.9:.1f}" '
                           f'height="{y1-y0:.1f}" fill="none" stroke="{col}" '
                           f'stroke-dasharray="3 3" opacity="0.5"/>')
                out.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="8" text-anchor="middle" '
                           f'fill="var(--mut)" transform="rotate(-90 {cx:.1f} {cy:.1f})">{esc(lab)}</text>')
    lx = x0
    for slab, col in series:
        out.append(f'<rect x="{lx}" y="{h-13}" width="11" height="11" fill="{col}"/>')
        out.append(f'<text x="{lx+15}" y="{h-3}" font-size="11" fill="var(--fg)">{esc(slab)}</text>')
        lx += 34 + 7.5 * len(slab)
    out.append("</svg>")
    return "".join(out)


def small_multiples(items, title, ylab=None, w=820, h=250, labelfn=fmt_si,
                    series=(("predicted", PAL[0]), ("published", PAL[3]))):
    """One mini LINEAR-axis panel per item, laid out in a single horizontal row.

    items = [(label, predicted, published, lo, hi)]; `lo`/`hi` are optional (omit, or pass None,
    for no error whisker) and describe an uncertainty interval on the PREDICTED value.

    Why per-panel axes: these anchors' step times span ~200s to ~1900s. On one shared LOG axis
    every predicted/published pair collapses to visually-identical heights (a 2x error looks
    like a rounding difference), which is actively deceiving; on one shared LINEAR axis the
    largest anchor flattens all the others into slivers. A private zero-based linear axis per
    panel keeps each pair honestly readable. The trade-off is that bar heights are NOT
    comparable across panels -- cross-anchor comparison is what the error-% chart is for.
    """
    pt, pb = 46, 36
    y0, y1 = pt, h - pb
    pw = w / max(1, len(items))
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" font-family="system-ui,sans-serif">',
           f'<text x="12" y="22" font-size="15" font-weight="600" fill="var(--fg)">{esc(title)}</text>']
    if ylab:
        ycen = (y0 + y1) / 2
        out.append(f'<text x="11" y="{ycen:.1f}" font-size="9.5" text-anchor="middle" '
                   f'fill="var(--mut)" transform="rotate(-90 11 {ycen:.1f})">{esc(ylab)}</text>')
    for i, it in enumerate(items):
        lab, pred, pub = it[0], it[1], it[2]
        lo = it[3] if len(it) > 3 else None
        hi = it[4] if len(it) > 4 else None
        px = 18 + i * pw
        ax0, ax1 = px + 26, px + pw - 8
        top = max([v for v in (pred, pub, hi) if v is not None] or [1.0])
        vmax = top * 1.24 or 1.0
        scaley = lambda v: y1 - min(max(v, 0.0) / vmax, 1.0) * (y1 - y0)
        for g in nice_ticks(vmax, n=3):
            if g > vmax:
                continue
            gy = scaley(g)
            out.append(f'<line x1="{ax0:.1f}" y1="{gy:.1f}" x2="{ax1:.1f}" y2="{gy:.1f}" '
                       f'stroke="var(--grid)"/>')
            out.append(f'<text x="{ax0-3:.1f}" y="{gy+3:.1f}" font-size="7.5" text-anchor="end" '
                       f'fill="var(--mut)">{labelfn(g)}</text>')
        bw = (ax1 - ax0) / 2.7
        for j, (v, (slab, col)) in enumerate(zip((pred, pub), series)):
            if v is None:
                continue
            bx = ax0 + 5 + j * (bw + 5)
            by = scaley(v)
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                       f'height="{max(0.0, y1-by):.1f}" fill="{col}"/>')
            out.append(f'<text x="{bx+bw/2:.1f}" y="{by-3:.1f}" font-size="7.5" '
                       f'text-anchor="middle" fill="var(--mut)">{labelfn(v)}</text>')
        if lo is not None and hi is not None:
            cx = ax0 + 5 + bw / 2
            ylo, yhi = scaley(lo), scaley(hi)
            out.append(f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" '
                       f'stroke="var(--fg)" stroke-width="1.2"/>')
            for cy in (yhi, ylo):
                out.append(f'<line x1="{cx-3.5:.1f}" y1="{cy:.1f}" x2="{cx+3.5:.1f}" '
                           f'y2="{cy:.1f}" stroke="var(--fg)" stroke-width="1.2"/>')
        for line_i, part in enumerate(str(lab).split("\n")):
            out.append(f'<text x="{(ax0+ax1)/2:.1f}" y="{y1+13+line_i*10}" font-size="8.5" '
                       f'text-anchor="middle" fill="var(--fg)">{esc(part)}</text>')
    lx = 30
    for slab, col in series:
        out.append(f'<rect x="{lx}" y="{h-12}" width="10" height="10" fill="{col}"/>')
        out.append(f'<text x="{lx+14}" y="{h-3}" font-size="10" fill="var(--fg)">{esc(slab)}</text>')
        lx += 30 + 7 * len(slab)
    out.append("</svg>")
    return "".join(out)


def heatmap(rowlabs, collabs, valfn, title, kind="seq", w=820, rh=30, tags=None,
            rowaxis=None, colaxis=None):
    """rows x cols matrix. `valfn(row, col) -> (value, text)`.

    value numeric  -> sequential shading (kind="seq") or green/red (kind="bin")
    value str      -> a tag cell coloured via `tags` (e.g. {"infeas": "#F58518"})
    value None     -> greyed-out cell (still shows its text, so "n/a"/"--" can say WHY)

    rowaxis/colaxis name the quantity each axis varies, for the same reason grouped_bars takes
    a ylab -- self-describing cell labels ("G=64") still leave the reader guessing what G is.
    """
    tags = tags or {"infeas": "#F58518", "nofit": "#9aa0a6"}
    pl, pt = 168, (60 if not colaxis else 74)
    cw = (w - pl - 14) / len(collabs); h = pt + rh * len(rowlabs) + 12
    grid = [[valfn(r, c) for c in collabs] for r in rowlabs]
    vals = [n for row in grid for (n, _) in row if isinstance(n, (int, float)) and kind == "seq"]
    lo, hi = (math.log10(min(vals)), math.log10(max(vals))) if vals else (0, 1)
    if hi <= lo:
        hi = lo + 1
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" font-family="system-ui,sans-serif">',
           f'<text x="12" y="22" font-size="15" font-weight="600" fill="var(--fg)">{esc(title)}</text>']
    if colaxis:
        out.append(f'<text x="{pl+(w-pl-14)/2:.0f}" y="{pt-24}" font-size="10" '
                   f'text-anchor="middle" fill="var(--mut)">{esc(colaxis)}</text>')
    if rowaxis:
        rcen = pt + rh * len(rowlabs) / 2
        out.append(f'<text x="13" y="{rcen:.1f}" font-size="10" text-anchor="middle" '
                   f'fill="var(--mut)" transform="rotate(-90 13 {rcen:.1f})">{esc(rowaxis)}</text>')
    for j, c in enumerate(collabs):
        out.append(f'<text x="{pl+cw*(j+0.5):.0f}" y="{pt-6}" font-size="10.5" '
                   f'text-anchor="middle" fill="var(--mut)">{esc(str(c))}</text>')
    for i, r in enumerate(rowlabs):
        cy = pt + rh * i
        out.append(f'<text x="{pl-8}" y="{cy+rh*0.62:.0f}" font-size="11" text-anchor="end" '
                   f'fill="var(--fg)">{esc(str(r))}</text>')
        for j, c in enumerate(collabs):
            n, txt = grid[i][j]; cx = pl + cw * j
            if n is None:
                fill, tc = "var(--grid)", "var(--mut)"
            elif isinstance(n, str):
                fill, tc = tags.get(n, "var(--grid)"), "#fff"
            elif kind == "seq":
                t = (math.log10(n) - lo) / (hi - lo)
                fill = lerp_hex("#dce7f5", "#123a6e", t)
                tc = "#fff" if t > 0.55 else "#12263f"
            else:
                fill, tc = ("#54A24B" if n else "#E45756"), "#fff"
            out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw-2:.1f}" '
                       f'height="{rh-2:.1f}" fill="{fill}"/>')
            out.append(f'<text x="{cx+cw/2:.1f}" y="{cy+rh*0.62:.0f}" font-size="9.5" '
                       f'text-anchor="middle" fill="{tc}">{esc(txt)}</text>')
    out.append("</svg>")
    return "".join(out)


def signed_hbars(rows, title, w=820, rh=26, dom=60, xlab=None,
                 ramp=("#c3d4e6", "#0b3d6b")):
    """Diverging bars around 0 for signed percentages. rows = [(label, pct)] or, to draw an
    uncertainty whisker on a row, [(label, pct, lo, hi)] where lo/hi are signed percentages
    bracketing pct (e.g. the MC 90% predictive interval mapped to error %).

    Colour is a SINGLE-HUE magnitude ramp (pale -> deep by |value|), deliberately not a
    green/amber/red scheme: a traffic-light palette reads as a verdict (pass/fail) on each
    anchor, and these residuals are a mix of known modelling gaps and genuine error, so
    "how far, and which direction" is the honest reading. Sign is already carried by which
    side of zero the bar sits on; the ramp only encodes magnitude.
    """
    # 26px (not 12) of extra bottom room when there's an xlab: the +/-% tick figures sit at
    # axis_y+11, so a 12px allowance put the axis title within ~5px of them and they collided.
    h = 46 + rh * len(rows) + (26 if xlab else 0)
    pl, pr = 150, 62; x0, x1 = pl, w - pr; xc = (x0 + x1) / 2
    axis_y = 34 + rh * len(rows) + 6
    xx = lambda v: xc + max(-dom, min(dom, v)) / dom * (x1 - xc)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" font-family="system-ui,sans-serif">',
           f'<text x="12" y="22" font-size="15" font-weight="600" fill="var(--fg)">{esc(title)}</text>',
           f'<line x1="{xc}" y1="34" x2="{xc}" y2="{axis_y:.0f}" stroke="var(--mut)"/>']
    for g in (-50, -25, 0, 25, 50):
        out.append(f'<text x="{xx(g):.0f}" y="{axis_y+11:.0f}" font-size="9" text-anchor="middle" '
                   f'fill="var(--mut)">{g:+d}%</text>')
    if xlab:
        out.append(f'<text x="{xc:.0f}" y="{h-2}" font-size="10" text-anchor="middle" '
                   f'fill="var(--mut)">{esc(xlab)}</text>')
    for i, row in enumerate(rows):
        lab, v = row[0], row[1]
        lo, hi = (row[2], row[3]) if len(row) >= 4 else (None, None)
        cy = 40 + rh * i
        col = lerp_hex(ramp[0], ramp[1], min(1.0, abs(v) / dom))
        a, b = (xc, xx(v)) if v >= 0 else (xx(v), xc)
        out.append(f'<rect x="{a:.1f}" y="{cy:.1f}" width="{max(1,b-a):.1f}" '
                   f'height="{rh*0.62:.1f}" fill="{col}"/>')
        # Uncertainty whisker (MC predictive interval mapped to error %), drawn over the bar so
        # the reader sees the point residual AND the range input uncertainty can move it to.
        if lo is not None and hi is not None:
            yb = cy + rh * 0.31
            xlo_, xhi_ = xx(lo), xx(hi)
            out.append(f'<line x1="{xlo_:.1f}" y1="{yb:.1f}" x2="{xhi_:.1f}" y2="{yb:.1f}" '
                       f'stroke="var(--fg)" stroke-width="1.1"/>')
            for cxx in (xlo_, xhi_):
                out.append(f'<line x1="{cxx:.1f}" y1="{cy+rh*0.12:.1f}" x2="{cxx:.1f}" '
                           f'y2="{cy+rh*0.5:.1f}" stroke="var(--fg)" stroke-width="1.1"/>')
        out.append(f'<text x="{pl-6}" y="{cy+rh*0.5:.0f}" font-size="11" text-anchor="end" '
                   f'fill="var(--fg)">{esc(lab)}</text>')
        out.append(f'<text x="{x1+4}" y="{cy+rh*0.5:.0f}" font-size="10" '
                   f'fill="var(--mut)">{v:+.0f}%</text>')
    out.append("</svg>")
    return "".join(out)


def tornado(rows, title, w=820, h=300):
    """Sensitivity tornado. rows = [(name, lo_mult, hi_mult, swing)], drawn around a 1.0x line."""
    pl, pr, pt, pb = 118, 60, 34, 30
    x0, x1, y0, y1 = pl, w - pr, pt, h - pb
    # LOG x-axis: the values are multipliers around 1.0x, and on a linear axis x0.5 and x2 sit at
    # different distances from the baseline even though they are the same factor in opposite
    # directions -- which visually understates the downside swing. A log axis makes reciprocal
    # multipliers symmetric about the 1.0x line, the honest geometry for a multiplicative sweep.
    # Only finite swings set the axis range; an inf/NaN multiplier (a lever that breaks the fit at
    # x0.5 or x2) is clamped to the axis edge and labelled as a break, not left to blow up xx().
    fin = lambda vs: [v for v in vs if math.isfinite(v) and v > 0]
    dlo = min([0.4] + fin(r[1] for r in rows))
    dhi = max([2.5] + fin(r[2] for r in rows))
    llo, lhi = math.log10(dlo), math.log10(dhi)
    xx = lambda v: x0 + (math.log10(min(max(v, dlo), dhi)) - llo) / (lhi - llo) * (x1 - x0)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" font-family="system-ui,sans-serif">',
           f'<text x="12" y="22" font-size="15" font-weight="600" fill="var(--fg)">{esc(title)}</text>']
    # Multiplicative gridlines at the round factors inside the range.
    for g in (0.25, 0.5, 1.0, 2.0, 4.0):
        if dlo <= g <= dhi:
            gx = xx(g)
            out.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" '
                       f'stroke="{"var(--mut)" if g == 1.0 else "var(--grid)"}" '
                       f'stroke-width="{1.5 if g == 1.0 else 1}"/>')
            out.append(f'<text x="{gx:.1f}" y="{y1+16}" font-size="10" text-anchor="middle" '
                       f'fill="var(--mut)">{("1.0x base" if g == 1.0 else f"{g:g}x")}</text>')
    rh = (y1 - y0) / len(rows)
    for i, (name, lo, hi, sw) in enumerate(rows):
        cy = y0 + i * rh + rh / 2; a, b = xx(min(lo, hi)), xx(max(lo, hi))
        out.append(f'<rect x="{a:.1f}" y="{cy-rh*0.32:.1f}" width="{max(1,b-a):.1f}" '
                   f'height="{rh*0.64:.1f}" fill="{PAL[i%len(PAL)]}" opacity="0.85"/>')
        out.append(f'<text x="{pl-6}" y="{cy+4:.1f}" font-size="11" text-anchor="end" '
                   f'fill="var(--fg)">{esc(name)}</text>')
        swlab = "breaks" if not math.isfinite(sw) else f"{sw:.1f}x"
        out.append(f'<text x="{x1+6}" y="{cy+4:.1f}" font-size="11" '
                   f'fill="var(--mut)">{swlab}</text>')
    out.append("</svg>")
    return "".join(out)
