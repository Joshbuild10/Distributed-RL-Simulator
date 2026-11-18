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
# What each section shows and how to read it: the dashboard and explorer print this under the heading.
# Figures quoted here are at the default threat model.
SECTION_NOTES = {
    "1": "The single-turn analytic core against published runs. prime-rl is the clean anchor: its MFU and "
         "E[R] are both pinned by its own published numbers, and it matches to +0%. The rest under-predict. "
         "AReaL 14B/32B is the FSDP penalty beyond calibrated MFU, growing with shard degree. INTELLECT-3 is "
         "low MoE MFU plus multi-turn context. INTELLECT-2 is broadcast-bound on a weaker volunteer swarm. "
         "Whiskers are 90% Monte-Carlo intervals over the unpublished inputs (table A4); validation/REPORT.md "
         "reads them.",
    "2": "What each RL budget costs a decentralised actor, and which levers move it. A run that misses its "
         "window is never dropped: a hollow bar is drawn at the GPUs that reach the fastest achievable "
         "wall-clock (the whole stock, split optimally), with that wall-clock written in it. No-fit means the "
         "weights or the trainer replica exceed one node. Past a point more GPUs stop helping: each step "
         "has a serial floor (the weight broadcast over the WAN, or decode streaming the weights from HBM), "
         "so only bigger batches, faster links or more bandwidth per node buy wall-clock.",
    "2.1": "Decentralisation costs serial latency, from two sources. (i) Weights cross the WAN every step: "
           "the trainer's gradient all-reduce between its sites, and the broadcast to inference. (ii) The "
           "number of inference waves, set by the HBM a node pools. A datacentre relaxes both: 72-GPU NVL72 "
           "nodes on a ~1 Tbps fabric, against 16-GPU islands on 1 Gbps. Everything else is held fixed, so "
           "the gap between paired bars is the decentralisation penalty. 2r/2s isolate node size. 2t/2u "
           "isolate trainer placement: at Rahman's 150x gradient compression the all-reduce hides under the "
           "broadcast; below ~2x the broadcast's compression it binds. An MoE pays both transfers on its "
           "total parameters but computes on its active ones, so it sits an order of magnitude lower.",
    "3": "E[R] is not one number. In the measurement campaign it moves ~2x along each of three axes, and "
         "they multiply: model (same prompts), difficulty, and task (code vs math). The task multiplier is "
         "itself family-dependent: V4-pro's code runs 3.4x its math, GLM-4.5-Air's 1.35x, Qwen3's ~1.05x. "
         "Within a family, length tracks the post-training recipe, not size: Qwen3 8B/14B are distilled "
         "and short, while 32B went through reasoning RL and runs long. Within a run the distribution is "
         "heavy-tailed (3e: cv 0.3-1.6, mean above median, p95 ~2-3x the mean). The model carries that "
         "spread exactly through E[R^2] = E[R]^2 (1 + cv^2), so cv = 1.6 inflates the attention and KV "
         "terms 3.6x. Agentic episodes (3c) are the multi-turn regime the single-turn model does not yet "
         "represent. Method and full findings: experiments/RESPONSE_LENGTH_MEASUREMENT.md.",
    "A": "The per-run detail behind the charts. The tables follow the configuration above: their titles name "
         "the model and GPU, and their GPU columns the GPU type.",
}
REGISTRY = {}                                # name -> (section, fn(opts) -> Fig | Table)


def register(name, section):
    def deco(fn):
        REGISTRY[name] = (section, fn)
        return fn
    return deco


# --- number formats -----------------------------------------------------------------------------
def fmt_count(v):
    """1,234 -> '1.23k'; 3 significant figures."""
    if not math.isfinite(v):
        return "inf"
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


def fmt_duration(s):
    """A duration in SECONDS: '23.3 s', '7 min', '2.1 h', '173 d', past 1,000 days '9 y'."""
    if not math.isfinite(s):
        return "inf"
    if s < 120:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{s / 60:.0f} min"
    if s < 86400:
        return f"{s / 3600:.1f} h"
    d = s / 86400
    return f"{d:.0f} d" if d < 1000 else f"{d / 365:.0f} y"


def fmt_ratio(frac):
    """Trainer share of the GPUs as 'trainer:inference', e.g. 0.23 -> '23:77'."""
    return f"{frac * 100:.0f}:{100 - frac * 100:.0f}"


FORMATS = {"count": fmt_count, "sci": fmt_sci, "rate": fmt_rate, "pct": lambda v: f"{v:+.0f}%",
           "share": lambda v: f"{v * 100:.3g}%", "dec2": lambda v: f"{v:.2f}",
           "plain": lambda v: f"{v:.3g}"}


def tex_sci(v):
    """2.5e25 -> '$2.5{\\times}10^{25}$' (mathtext, typeset identically in paper and dashboard)."""
    e = math.floor(math.log10(v))
    m = f"{v / 10 ** e:.2g}"
    return f"$10^{{{e}}}$" if m == "1" else f"${m}{{\\times}}10^{{{e}}}$"


def stock_label(n):
    return f"{n / 1e6:g}M" if n >= 1e6 else (f"{n / 1e3:g}k" if n >= 1e4 else f"{n:,}")
