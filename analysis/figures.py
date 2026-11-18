"""
figures.py -- Every chart as data + a plotting spec, with no drawing code. A figure function runs its
sweep and returns a Fig; renderers (paper, web, table) draw any Fig the same way, so the paper PDF,
the dashboard and the CSV can never disagree.

Options shared by every figure: penalties (bool), batch_mode ('scaled' | 'isocompute' | 'flat').
"""
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from feasibility_core import MODELS
from .scenarios import TARGETS, scenario
from .sweep import sweep, min_gpus, max_compute

STOCKS = (128, 1_024, 10_000, 100_000, 1_000_000)
WINDOWS = (30, 90, 180, 720)


@dataclass
class Fig:
    id: str
    title: str
    data: pd.DataFrame
    x: str                                   # column giving the x categories
    series: str                              # column giving the grouped series
    ylabel: str
    log: bool = True
    xlabel: str = ""
    value_fmt: str = "count"                 # count | sci: how bar labels and the axis print
    refs: tuple = ()                         # [(label, y)] horizontal reference lines
    x_fmt: Callable = str
    series_fmt: Callable = str
    caption: str = ""

    def categories(self):
        return list(self.data[self.x].cat.categories)

    def series_values(self):
        return list(self.data[self.series].cat.categories)

    def cell(self, x, s) -> Optional[pd.Series]:
        hit = self.data[(self.data[self.x] == x) & (self.data[self.series] == s)]
        return None if hit.empty else hit.iloc[0]


FIGURES = {}


def figure(name):
    def register(fn):
        FIGURES[name] = fn
        return fn
    return register


def _sci_tex(v):
    """2.5e25 -> '$2.5\\times10^{25}$' (mathtext, so paper and dashboard typeset it the same)."""
    e = int(f"{v:e}".split("e")[1])
    m = f"{v / 10**e:g}"
    return f"$10^{{{e}}}$" if m == "1" else f"${m}\\times10^{{{e}}}$"


def _target_label(t):
    return f"{t}\n{_sci_tex(TARGETS[t])} FLOP"


def _stock_label(n):
    return f"{n/1e6:g}M" if n >= 1e6 else (f"{n/1e3:g}k" if n >= 1e4 else f"{n:,}")


def _target_refs():
    return tuple((t, c) for t, c in sorted(TARGETS.items(), key=lambda kv: kv[1]))


@figure("frontier")
def frontier(penalties=False, batch_mode="scaled"):
    data = sweep({"target": list(TARGETS), "window_d": list(WINDOWS)},
                 lambda p: min_gpus(scenario("decentralised", target=p["target"], batch_mode=batch_mode,
                                             penalties=penalties), p["window_d"]))
    return Fig("2a", "Minimum GPUs to finish each RL compute budget", data, x="target", series="window_d",
               ylabel="GPUs to finish in window", x_fmt=_target_label, series_fmt=lambda w: f"{w} days",
               caption="Llama-3.1-405B, decentralised threat model. Hollow bars miss the window: drawn at "
                       "the GPUs that reach the fastest achievable wall-clock, labelled with it.")


@figure("dec_vs_cen")
def dec_vs_cen(penalties=False, batch_mode="scaled"):
    data = sweep({"stock": list(STOCKS), "setup": ["decentralised", "centralised"]},
                 lambda p: max_compute(scenario(p["setup"], penalties=penalties), p["stock"], 180))
    return Fig("2o", "Maximum RL compute in 180 days: decentralised vs centralised", data, x="stock",
               series="setup", ylabel="RL compute (FLOP)", xlabel="GPU stock", value_fmt="sci",
               refs=_target_refs(), x_fmt=_stock_label, series_fmt=str.capitalize,
               caption="Centralised: 72-GPU NVL72 nodes, 1 Tbps fabric, up to 4-node tensor parallelism. "
                       "Dashed lines: estimated RL budgets of published models.")


@figure("trainer_placement")
def trainer_placement(penalties=False, batch_mode="scaled", model="Llama3-405B"):
    setups = {"Co-located trainer": ("colocated_trainer",),
              "Distributed, 150x gradients": ("grad_150x",),
              "Distributed, 16x gradients": ("grad_16x",)}
    data = sweep({"stock": list(STOCKS), "trainer": list(setups)},
                 lambda p: max_compute(scenario("decentralised", *setups[p["trainer"]],
                                                model=MODELS[model], penalties=penalties), p["stock"], 180))
    return Fig("2t", f"Trainer placement: maximum RL compute in 180 days ({MODELS[model].name})", data,
               x="stock", series="trainer", ylabel="RL compute (FLOP)", xlabel="GPU stock", value_fmt="sci",
               refs=_target_refs(), x_fmt=_stock_label,
               caption="Distributed trainers all-reduce gradients every step over the 1 Gbps WAN.")


def table(fig):
    """The figure's data as a pivot: x down, series across, the plotted value or its hollow/note.
    A DataFrame, so .to_csv() / .to_latex() export it as-is."""
    num = (lambda v: f"{v:.2e}") if fig.value_fmt == "sci" else (lambda v: f"{v:,.0f}")

    def cell(r):
        if pd.notna(r["value"]):
            return num(r["value"])
        if pd.notna(r["fallback"]):
            return f"({num(r['fallback'])}, {r['note']})"
        return r["note"] if isinstance(r["note"], str) else ""
    t = (fig.data.assign(cell=fig.data.apply(cell, axis=1))
         .pivot(index=fig.x, columns=fig.series, values="cell"))
    t.index = [str(x) for x in t.index]                 # raw values: chart formatting stays in the chart
    t.columns = [fig.series_fmt(c) for c in t.columns]
    t.index.name = fig.xlabel or fig.x
    return t
