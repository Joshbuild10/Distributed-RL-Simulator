"""
spec.py -- What a figure or table IS, independent of how it is drawn: data (a DataFrame) plus a
plotting spec. Figure functions register here; renderers and the dashboard read the registry.

Kinds and the columns they read:
    bars       x, series -> value | fallback (drawn hollow) | note; optional `inner` text column
    heatmap    series (rows) x x (columns) -> value (shading), text, status (ok|infeas|nofit|over|under)
    tornado    x (lever) -> lo, hi (wall-clock ratios to the base case)
    multiples  x (panel) -> predicted, published, optional lo/hi whisker on predicted
    hbars      x (row) -> value, optional lo/hi whisker
"""
import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class Fig:
    id: str
    title: str
    kind: str
    data: pd.DataFrame
    x: str
    series: str = ""
    ylabel: str = ""
    xlabel: str = ""
    log: bool = True
    fmt: str = "count"                       # count | sci | rate | pct | plain
    refs: tuple = ()                         # [(label, y)] reference lines, labelled in the right margin
    ordinal: bool = False                    # series form an ordered sweep (uses the ordinal palette)
    inner: str = ""                          # bars: column of text written inside each bar
    labels: tuple = ("predicted", "published")   # multiples: the two bars' legend labels
    x_fmt: Callable = str
    series_fmt: Callable = str
    caption: str = ""
    section: str = ""
    height: float = 0.0                      # inches; 0 = the kind's default

    def categories(self):
        col = self.data[self.x]
        return list(col.cat.categories) if hasattr(col, "cat") else list(dict.fromkeys(col))

    def series_values(self):
        col = self.data[self.series]
        return list(col.cat.categories) if hasattr(col, "cat") else list(dict.fromkeys(col))

    def cell(self, x, s):
        hit = self.data[(self.data[self.x] == x) & (self.data[self.series] == s)]
        return None if hit.empty else hit.iloc[0]


@dataclass
class Table:
    id: str
    title: str
    data: pd.DataFrame                       # display-ready strings; index = first column
    caption: str = ""
    section: str = ""


# --- registry ------------------------------------------------------------------------------------
SECTIONS = {
    "1": "Model validation",
    "2": "Feasibility",
    "2.1": "Decentralised vs centralised",
    "3": "Response lengths",
    "A": "Appendix",
}
REGISTRY = {}                                # name -> (section, fn(opts) -> Fig | Table)


def register(name, section):
    def deco(fn):
        REGISTRY[name] = (section, fn)
        return fn
    return deco


# --- number formats -----------------------------------------------------------------------------
def fmt_count(v):
    for d, u in ((1e6, "M"), (1e3, "k")):
        if abs(v) >= d:
            return f"{v / d:.3g}{u}"
    return f"{v:.0f}"


def fmt_sci(v):
    e = math.floor(math.log10(abs(v)))
    return f"{v / 10 ** e:.1f}e{e}"


def fmt_rate(v):
    if abs(v) >= 1e3:
        return fmt_count(v)
    return f"{v:.0f}" if abs(v) >= 10 else (f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}")


def fmt_time(days):
    return f"{days:.0f} d" if days >= 1 else f"{days * 24:.1f} h"


FORMATS = {"count": fmt_count, "sci": fmt_sci, "rate": fmt_rate, "pct": lambda v: f"{v:+.0f}%",
           "plain": lambda v: f"{v:.3g}"}


def tex_sci(v):
    """2.5e25 -> '$2.5{\\times}10^{25}$' (mathtext, typeset identically in paper and dashboard)."""
    e = math.floor(math.log10(v))
    m = f"{v / 10 ** e:.2g}"
    return f"$10^{{{e}}}$" if m == "1" else f"${m}{{\\times}}10^{{{e}}}$"


def stock_label(n):
    return f"{n / 1e6:g}M" if n >= 1e6 else (f"{n / 1e3:g}k" if n >= 1e4 else f"{n:,}")
