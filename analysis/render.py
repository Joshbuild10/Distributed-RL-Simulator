"""
render.py -- Draw any Fig with matplotlib in the house style (theme.py). One renderer makes the
paper's PDF/SVG/PNG, the dashboard's inline SVG and the explorer's images.

Everything happens inside the theme's rc context, including saving: matplotlib reads font embedding
and face colours at save time, not draw time.
"""
import io
import math
import numbers
import textwrap

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .spec import FORMATS, tex_sci
from .theme import HEAT, ON, THEMES, rc, series_roles

WIDTH_IN = 6.9                               # two-column page width; ~3.35 for a single column
LABEL_FS = 5.6                               # value labels on bars


def _num(v):
    # numbers.Real, not (int, float): a NaN-free pandas column holds numpy int64, which is neither
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _log_axis(axis, fmt, lo=None, hi=None):
    """Decade ticks; 1-2-5 ticks too when the axis spans under two decades (else it shows one or two)."""
    narrow = lo and hi and hi / lo < 100
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5) if narrow else (1.0,), numticks=12))
    axis.set_minor_formatter(NullFormatter())
    if fmt == "sci":
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{round(math.log10(v))}}}$"
                                               if v > 0 and abs(math.log10(v) - round(math.log10(v))) < 1e-9 else ""))
    else:
        axis.set_major_formatter(FuncFormatter(lambda v, _: FORMATS["rate" if fmt == "rate" else "count"](v)))


def _legend(ax, handles, max_cols=6):
    """One legend row above the axes (more if needed), read left to right then down. matplotlib fills
    columns first, so the handles are reordered to make the rows come out in series order."""
    if not handles:
        return
    rows = math.ceil(len(handles) / max_cols)
    cols = math.ceil(len(handles) / rows)
    order = [r * cols + c for c in range(cols) for r in range(rows) if r * cols + c < len(handles)]
    ax.legend(handles=[handles[i] for i in order], loc="lower left", bbox_to_anchor=(0, 1.0), ncols=cols,
              frameon=False, handlelength=1.1, columnspacing=1.1, borderaxespad=0.2)


def _wrap_ticks(ax, labels, fontsize=7):
    """Wrap category labels that would overrun their slot (mathtext lines are left whole)."""
    f = ax.figure
    slot = f.get_figwidth() * 72 * 0.85 / max(len(labels), 1)
    width = max(6, int(slot / (0.52 * fontsize)))
    return ["\n".join(ln if "$" in ln or len(ln) <= width else textwrap.fill(ln, width, break_on_hyphens=False)
                      for ln in str(lab).split("\n")) for lab in labels]


REF_GAP_PT = 15                              # two 5.8 pt label lines plus a little air


def _ref_labels(ax, refs, t, log):
    """Reference lines, labelled in the right margin. Labels start level with their line; _spread_refs
    nudges close ones apart once the layout (and so the axes height) is final."""
    lo, hi = ax.get_ylim()
    pos = lambda y: ((math.log10(y) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) if log
                     else (y - lo) / (hi - lo))
    ax._refs = []
    for lab, y in sorted(refs, key=lambda r: r[1]):
        ax.axhline(y, color=t["muted"], linewidth=0.6, linestyle=(0, (4, 2)), zorder=1)
        ann = ax.annotate(f"{lab}\n{tex_sci(y)}", (1.0, pos(y)), xycoords="axes fraction", xytext=(4, 0),
                          textcoords="offset points", va="center", fontsize=5.8, color=t["muted"],
                          linespacing=0.95, annotation_clip=False)
        ax._refs.append((ann, pos(y)))


def _spread_refs(ax):
    if not getattr(ax, "_refs", None):
        return
    gap = REF_GAP_PT / (ax.get_window_extent().height * 72 / ax.figure.dpi)
    ys = []
    for _, fy in ax._refs:                      # bottom-up: push each label clear of the one below
        ys.append(max(fy, ys[-1] + gap) if ys else fy)
    shift = max(0.0, ys[-1] - 1.02)             # then pull the stack down if it overran the top
    for (ann, _), fy in zip(ax._refs, ys):
        ann.xy = (1.0, fy - shift)


def _fit_inside(ax, t):
    """A label that doesn't fit inside its bar moves on top of it, joining the label already there."""
    r = ax.figure.canvas.get_renderer()
    for ann, bar, above in getattr(ax, "_inside", []):
        if ann.get_window_extent(r).height <= bar.get_window_extent(r).height - 3:
            continue
        if above is not None:
            above.set_text(f"{above.get_text()}  {ann.get_text()}")
            ann.set_visible(False)
        else:
            ann.xyann = (0, 1.5)
            ann.set_va("bottom")
            ann.set_color(t["muted"])


# --- kinds ---------------------------------------------------------------------------------------
def _bars(fig, ax, t):
    cats = fig.categories()
    series = fig.series_values() if fig.series else [None]
    n = len(series)
    roles = series_roles(n, fig.ordinal)
    w = 0.8 / n
    label = FORMATS[fig.fmt]
    shown, inner = [], []
    ax._inside = []
    for j, s in enumerate(series):
        col, on = t[roles[j]], t[ON[roles[j]]]
        for i, x in enumerate(cats):
            r = fig.cell(x, s) if s is not None else fig.data[fig.data[fig.x] == x].iloc[0]
            xc = i - 0.4 + w * (j + 0.5)
            if r is None:
                continue
            if _num(r["value"]):
                v = r["value"]
                bar = ax.bar(xc, v, w * 0.9, color=col, zorder=2)[0]
                if fig.refs:                         # inside the bar top, clear of the reference lines
                    ax._inside.append((ax.annotate(label(v), (xc, v), xytext=(0, -2), textcoords="offset points",
                                                   ha="center", va="top", fontsize=LABEL_FS, color=on,
                                                   rotation=90, zorder=3), bar, None))
                else:
                    ax.annotate(label(v), (xc, v), xytext=(0, 1.5), textcoords="offset points", ha="center",
                                va="bottom", fontsize=LABEL_FS, color=t["muted"], rotation=90 if n > 2 else 0)
                shown.append(v)
                txt = r.get(fig.inner) if fig.inner else None
                if isinstance(txt, str) and txt:
                    inner.append((xc, v, txt, bar, on))
            elif _num(r["fallback"]):
                fb = r["fallback"]
                box = ax.bar(xc, fb, w * 0.9, facecolor="none", edgecolor=col, linewidth=0.8,
                             linestyle=(0, (2, 1.5)), zorder=2)[0]
                above = None
                if not fig.refs:
                    above = ax.annotate(label(fb), (xc, fb), xytext=(0, 1.5), textcoords="offset points",
                                        ha="center", va="bottom", fontsize=LABEL_FS, color=t["muted"],
                                        rotation=90 if n > 2 else 0)
                ax._inside.append((ax.annotate(r["note"], (xc, fb), xytext=(0, -2), textcoords="offset points",
                                               ha="center", va="top", fontsize=LABEL_FS, color=t["fg"],
                                               rotation=90, fontweight="bold", zorder=3), box, above))
                shown.append(fb)
            else:
                ax.annotate(r["note"] or "n/a", (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=LABEL_FS,
                            color=t["muted"], rotation=90)
    refs = list(fig.refs)
    span = shown + [y for _, y in refs]
    if fig.log:
        ax.set_yscale("log")
        if span:
            ax.set_ylim(min(span) / 2.5, max(span) * (8 if n > 2 else 4))
        _log_axis(ax.yaxis, fig.fmt, *ax.get_ylim())
    else:
        ax.set_ylim(0, max(span) * 1.2 if span else 1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: label(v) if v else "0"))
    lo = ax.get_ylim()[0]
    for xc, v, txt, bar, on in inner:             # text inside each bar, clipped to it
        mid = math.sqrt(lo * v) if fig.log else v / 2
        ax.text(xc, mid, txt, rotation=90, ha="center", va="center", fontsize=5.2,
                color=on, zorder=3).set_clip_path(bar)
    _ref_labels(ax, refs, t, fig.log)
    ax.set_xticks(range(len(cats)), _wrap_ticks(ax, [fig.x_fmt(x) for x in cats]))
    ax.set_xlim(-0.55, len(cats) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(fig.ylabel)
    if fig.xlabel:
        ax.set_xlabel(fig.xlabel)
    if fig.series:
        _legend(ax, [Patch(color=t[roles[j]], label=fig.series_fmt(s)) for j, s in enumerate(series)])


def _heatmap(fig, ax, t):
    rows, cols = fig.series_values(), fig.categories()
    ok = [r["value"] for _, r in fig.data.iterrows() if r["status"] == "ok" and _num(r["value"]) and r["value"] > 0]
    lo, hi = (math.log10(min(ok)), math.log10(max(ok))) if ok else (0, 1)
    fills = {"infeas": "infeas", "nofit": "nofit", "over": "over", "under": "under"}
    for i, rw in enumerate(rows):
        for j, c in enumerate(cols):
            r = fig.cell(c, rw)
            if r is None:
                continue
            if r["status"] == "ok" and _num(r["value"]) and r["value"] > 0:
                k = 0 if hi <= lo else round((math.log10(r["value"]) - lo) / (hi - lo) * (len(HEAT) - 1))
                fill, txt = HEAT[k], ("fg" if k < 3 else "bg")
            else:
                fill = fills.get(r["status"], "nofit")
                txt = "bg" if r["status"] in ("over", "under") else "fg"
            ax.add_patch(Rectangle((j, i), 0.96, 0.92, color=t[fill], linewidth=0))
            ax.text(j + 0.48, i + 0.46, r["text"], ha="center", va="center", fontsize=6.5, color=t[txt])
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.48 for j in range(len(cols))], [fig.x_fmt(c) for c in cols])
    ax.set_yticks([i + 0.46 for i in range(len(rows))], [fig.series_fmt(r) for r in rows])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(fig.xlabel)
    ax.set_ylabel(fig.ylabel)


def _tornado(fig, ax, t):
    levers = fig.categories()
    finite = [v for k in ("lo", "hi") for v in fig.data[k] if _num(v)]
    cap = max(2.0, max(finite) * 1.6)            # where a lever that breaks the run is drawn
    for i, lv in enumerate(levers):
        r = fig.data[fig.data[fig.x] == lv].iloc[0]
        if all(_num(r[k]) and abs(r[k] - 1) < 0.005 for k in ("lo", "hi")):
            ax.annotate("no effect", (1, i), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=LABEL_FS, color=t["muted"], fontstyle="italic")
            continue
        for key, role, dy in (("lo", "s1", -0.18), ("hi", "s2", 0.18)):
            v = r[key]
            vv = min(v, cap) if _num(v) else cap
            ax.barh(i + dy, vv - 1, left=1, height=0.34, color=t[role], zorder=2)
            if _num(v) and abs(v - 1) < 0.005:
                continue                             # unchanged: no bar, no label
            txt = f"{v:.2f}x" if _num(v) and v < cap else "breaks (no-fit)"
            ax.annotate(txt, (vv, i + dy), xytext=(3 if vv >= 1 else -3, 0), textcoords="offset points",
                        ha="left" if vv >= 1 else "right", va="center", fontsize=LABEL_FS, color=t["muted"])
    ax.axvline(1, color=t["fg"], linewidth=0.7)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(finite + [1]) / 1.3, cap * 1.5)
    ax.set_yticks(range(len(levers)), levers)
    ax.set_ylim(len(levers) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel(fig.xlabel)
    _legend(ax, [Patch(color=t["s1"], label=fig.labels[0]), Patch(color=t["s2"], label=fig.labels[1])])


def _hbars(fig, ax, t):
    rows = fig.categories()
    for i, x in enumerate(rows):
        r = fig.data[fig.data[fig.x] == x].iloc[0]
        v = r["value"]
        ax.barh(i, v, height=0.6, color=t["s2"] if v > 0 else t["s1"], zorder=2)
        if _num(r.get("lo")) and _num(r.get("hi")):
            ax.plot([r["lo"], r["hi"]], [i, i], color=t["fg"], linewidth=0.8, zorder=3)
            ax.plot([r["lo"]] * 2 + [None] + [r["hi"]] * 2, [i - .15, i + .15, None, i - .15, i + .15],
                    color=t["fg"], linewidth=0.8, zorder=3)
        ax.annotate(FORMATS[fig.fmt](v), (1.0, i), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6.5, color=t["fg"],
                    fontweight="semibold")
    ax.axvline(0, color=t["fg"], linewidth=0.7)
    ax.set_yticks(range(len(rows)), [fig.x_fmt(x) for x in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel(fig.xlabel)


def _multiples(fig, axes, t, f):
    items = fig.categories()
    axes = list(axes) if len(items) > 1 else [axes]
    roles = ("s1", "s2")
    label = FORMATS[fig.fmt]
    for k, (ax, it) in enumerate(zip(axes, items)):
        r = fig.data[fig.data[fig.x] == it].iloc[0]
        vals = [r["predicted"], r["published"]]
        top = max([v for v in vals if _num(v)] + [r["hi"] if _num(r.get("hi")) else 0] + [1e-9])
        whisker = r["hi"] if _num(r.get("hi")) else None
        for b, (v, role) in enumerate(zip(vals, roles)):
            if _num(v):
                ax.bar(b, v, 0.72, color=t[role], zorder=2)
                ax.annotate(label(v), (b, max(v, whisker) if b == 0 and whisker else v), xytext=(0, 1.5), textcoords="offset points", ha="center",
                            va="bottom", fontsize=5.2, color=t["muted"], rotation=90)
            else:
                ax.annotate("n/a", (b, 0), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=LABEL_FS, color=t["muted"])
        if _num(r.get("lo")) and _num(r.get("hi")):
            ax.plot([0, 0], [r["lo"], r["hi"]], color=t["fg"], linewidth=0.8, zorder=3)
            ax.plot([-.12, .12, None, -.12, .12], [r["lo"], r["lo"], None, r["hi"], r["hi"]],
                    color=t["fg"], linewidth=0.8, zorder=3)
        ax.set_ylim(0, top * 1.32)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([])
        ax.set_title(fig.x_fmt(it), fontsize=7, fontweight="semibold", color=t["fg"])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: label(v) if v else "0"))
        ax.tick_params(axis="y", labelsize=6)
        if k == 0:
            ax.set_ylabel(fig.ylabel)
    f.legend(handles=[Patch(color=t[roles[0]], label=fig.labels[0]), Patch(color=t[roles[1]], label=fig.labels[1])],
             loc="outside upper left", ncols=2, frameon=False, handlelength=1.1)


_KINDS = {"bars": _bars, "heatmap": _heatmap, "tornado": _tornado, "hbars": _hbars}


def _height(fig):
    if fig.height:
        return fig.height
    n = len(fig.categories())
    if fig.kind == "heatmap":
        return 0.28 * len(fig.series_values()) + 1.25
    if fig.kind in ("tornado", "hbars"):
        return 0.27 * n + 0.95
    return 2.35 if fig.kind == "multiples" else 2.9


def _figure(size):
    """A figure on its own Agg canvas: no pyplot state (thread-safe under Streamlit), and a real
    renderer whatever backend the host process has selected."""
    f = Figure(figsize=size, layout="constrained")
    FigureCanvasAgg(f)
    return f


def _draw(fig, theme, title, width, scale=1.0):
    """scale > 1 enlarges all text relative to the chart (a poster) by drawing on a smaller canvas."""
    t = THEMES[theme]
    size = (width / scale, _height(fig) / scale)
    if fig.kind == "multiples":
        f = _figure(size)
        axes = f.subplots(1, len(fig.categories()))
        _multiples(fig, axes, t, f)
    else:
        f = _figure(size)
        ax = f.subplots()
        _KINDS[fig.kind](fig, ax, t)
    if title:
        f.suptitle(f"{fig.id}. {fig.title}", x=0.005, ha="left", fontweight="bold", fontsize=9.5, color=t["fg"])
    if any(getattr(a, "_refs", None) or getattr(a, "_inside", None) for a in f.axes):
        f.get_layout_engine().execute(f)          # these need the final axes size
        for a in f.axes:
            _spread_refs(a)
            _fit_inside(a, t)
    return f


def render(fig, fmt="svg", theme="light", title=True, width=WIDTH_IN, dpi=300):
    """One figure as bytes in `fmt` (svg | pdf | png)."""
    with matplotlib.rc_context(rc(theme)):
        f = _draw(fig, theme, title, width)
        buf = io.BytesIO()
        f.savefig(buf, format=fmt, dpi=dpi)
    return buf.getvalue()


def save(fig, stem, theme="light", width=WIDTH_IN, scale=1.0):
    """stem.pdf / .svg / .png for the paper: no title (it belongs in the caption), drawn once."""
    with matplotlib.rc_context(rc(theme)):
        f = _draw(fig, theme, False, width, scale)
        for ext in ("pdf", "svg"):
            f.savefig(f"{stem}.{ext}")
        f.savefig(f"{stem}.png", dpi=300 * scale)
