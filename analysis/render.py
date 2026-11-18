"""
render.py -- Draw a Fig with matplotlib in one house style. The same Axes become the paper's vector
PDF/SVG and the dashboard's inline SVG, so there is one renderer to maintain.
"""
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# Okabe-Ito: colourblind-safe and distinct in greyscale print
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
FULL_WIDTH_IN = 6.9                          # two-column page width; single column is ~3.35
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e3e3e3", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "svg.fonttype": "none", "pdf.fonttype": 42,   # real text in SVG/PDF
}


def _count(v):
    for d, u in ((1e6, "M"), (1e3, "k")):
        if abs(v) >= d:
            return f"{v/d:.3g}{u}"
    return f"{v:.0f}"


def _sci(v):
    e = math.floor(math.log10(v))
    return f"{v/10**e:.1f}e{e}"


def _pow10(v, _=None):
    e = round(math.log10(v))
    return f"$10^{{{e}}}$" if abs(math.log10(v) - e) < 1e-9 else ""


def draw(fig, title=True, width=FULL_WIDTH_IN, height=2.9):
    """A matplotlib Figure for `fig`: grouped bars, log y by default, hollow bars for fallbacks."""
    with plt.rc_context(STYLE):
        f, ax = plt.subplots(figsize=(width, height), layout="constrained")
        cats, series = fig.categories(), fig.series_values()
        n = len(series)
        w = 0.8 / n
        label = _sci if fig.value_fmt == "sci" else _count
        drawn = []
        for j, s in enumerate(series):
            col = PALETTE[j % len(PALETTE)]
            for i, x in enumerate(cats):
                r = fig.cell(x, s)
                if r is None:
                    continue
                xc = i - 0.4 + w * (j + 0.5)
                if r["value"] == r["value"] and r["value"] is not None:          # plotted value
                    ax.bar(xc, r["value"], w * 0.92, color=col, label=fig.series_fmt(s) if i == 0 else None)
                    ax.annotate(label(r["value"]), (xc, r["value"]), xytext=(0, 1.5), textcoords="offset points",
                                ha="center", va="bottom", fontsize=5.5, color="#444", rotation=90 if n > 2 else 0)
                    drawn.append(r["value"])
                elif r["fallback"] == r["fallback"] and r["fallback"] is not None:   # missed: hollow
                    ax.bar(xc, r["fallback"], w * 0.92, facecolor="none", edgecolor=col, linewidth=0.8,
                           linestyle=(0, (2, 1.5)), label=fig.series_fmt(s) if i == 0 else None)
                    ax.annotate(r["note"], (xc, r["fallback"]), xytext=(0, 1.5), textcoords="offset points",
                                ha="center", va="bottom", fontsize=5.5, color=col, rotation=90, fontweight="bold")
                    drawn.append(r["fallback"])
                else:                                                                # nothing to draw
                    ax.annotate(r["note"] or "n/a", (xc, 0), xycoords=("data", "axes fraction"),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                                fontsize=5.5, color="#888", rotation=90)
        if fig.log:
            ax.set_yscale("log")
            if fig.value_fmt == "sci":
                ax.yaxis.set_major_formatter(FuncFormatter(_pow10))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _count(v)))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
        refs = [(lab, y) for lab, y in fig.refs]
        if drawn:
            lo, hi = min(drawn + [y for _, y in refs]), max(drawn + [y for _, y in refs])
            ax.set_ylim(lo / 2.5 if fig.log else 0, hi * (6 if fig.log else 1.2))
        for lab, y in refs:
            ax.axhline(y, color="#555", linewidth=0.6, linestyle=(0, (4, 2)), zorder=0.5)
            ax.annotate(lab, (1.0, y), xycoords=("axes fraction", "data"), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=6, color="#555")
        ax.set_xticks(range(len(cats)), [fig.x_fmt(x) for x in cats])
        ax.set_xlim(-0.55, len(cats) - 0.45)
        ax.set_ylabel(fig.ylabel)
        if fig.xlabel:
            ax.set_xlabel(fig.xlabel)
        ax.grid(axis="x", visible=False)
        # Legend in its own band above the axes, so it can never sit on a tall bar's label
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=min(n, 4), frameon=False,
                  handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)
        if title:
            f.suptitle(f"{fig.id}. {fig.title}", x=0.01, ha="left", fontweight="bold", fontsize=9)
        return f


def save(fig, stem, **kw):
    """Write stem.pdf / .svg / .png (paper versions: no title, it lives in the caption)."""
    f = draw(fig, title=False, **kw)
    for ext in ("pdf", "svg"):
        f.savefig(f"{stem}.{ext}")
    f.savefig(f"{stem}.png", dpi=300)
    plt.close(f)


def svg(fig, **kw):
    """Inline SVG string (dashboard version: with title)."""
    f = draw(fig, title=True, **kw)
    buf = io.StringIO()
    f.savefig(buf, format="svg")
    plt.close(f)
    return buf.getvalue()[buf.getvalue().index("<svg"):]
