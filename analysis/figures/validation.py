"""
Section 1 -- the analytic core against published runs: step times, residuals, and the trainer MFU the
model would need to match each one.
"""
from functools import lru_cache

import pandas as pd

from model import simulate
from validation.presets import (areal_1_5b, areal_7b, areal_14b, areal_32b, intellect2, intellect3,
                                primerl_paper)
from validation.solvers import solve_for_mfu
from validation.uncertainty import mc_step_time
from ..spec import Fig, register


@lru_cache(maxsize=1)
def anchors():
    """(label, scenario, point prediction, published step time, Monte-Carlo summary) per anchor.
    Validation presets carry their own logged configs, so run-wide options do not apply here."""
    presets = [("INTELLECT-2\nshort", intellect2("short")), ("INTELLECT-2\nlong", intellect2("long")),
               ("INTELLECT-3", intellect3()), ("prime-rl", primerl_paper()), ("AReaL\n1.5B", areal_1_5b()),
               ("AReaL\n7B", areal_7b()), ("AReaL\n14B", areal_14b()), ("AReaL\n32B", areal_32b())]
    out = []
    for lab, sc in presets:
        pub = (sc.published or {}).get("t_step")
        if pub:
            out.append((lab, sc, simulate(sc).t_step, pub, mc_step_time(sc)))
    return tuple(out)


def _cat(df, col):
    df[col] = pd.Categorical(df[col], categories=list(dict.fromkeys(df[col])), ordered=True)
    return df


@register("validation_step", "1")
def validation_step(opts):
    data = _cat(pd.DataFrame([dict(anchor=lab, predicted=mc["mean"], published=pub, lo=mc["p5"], hi=mc["p95"])
                              for lab, _, _, pub, mc in anchors()]), "anchor")
    return Fig("1a", "Predicted vs published step time (Monte-Carlo mean, whisker = 90% interval)", "multiples",
               data, x="anchor", ylabel="step time (s)", labels=("predicted (MC mean)", "published"),
               caption="Each panel has its own linear axis: published step times span ~200-1,900 s.")


@register("validation_error", "1")
def validation_error(opts):
    rows = [dict(anchor=lab.replace("\n", " "), value=(pt - pub) / pub * 100,
                 lo=(mc["p5"] - pub) / pub * 100, hi=(mc["p95"] - pub) / pub * 100)
            for lab, _, pt, pub, mc in anchors()]
    return Fig("1b", "Prediction error: point estimate (bar) with 90% Monte-Carlo interval (whisker)", "hbars",
               _cat(pd.DataFrame(rows), "anchor"), x="anchor", fmt="pct", xlabel="error vs published step time",
               caption="A whisker that crosses zero is a gap input uncertainty alone can explain.")


@register("implied_mfu", "1")
def implied_mfu(opts):
    rows = []
    for lab, sc, _, pub, _ in sorted(anchors(), key=lambda a: a[1].train_hw.n_shard):
        d = solve_for_mfu(sc, pub)
        if d and d["implied_mfu"] is not None:
            rows.append(dict(anchor=f"{lab.replace(chr(10), ' ')}\n{sc.train_hw.n_shard}-shard",
                             value=d["implied_mfu"], fallback=float("nan"), note=None,
                             flag="*" if d["flipped"] else ""))
    data = _cat(pd.DataFrame(rows), "anchor")
    return Fig("1c", "Trainer MFU the model needs to match each published step time", "bars", data, x="anchor",
               log=False, fmt="plain", ylabel="implied trainer MFU", inner="flag",
               caption="Ordered by FSDP shard degree: the decline is the signature of the unmodelled sharding "
                       "penalty. prime-rl, whose MFU is published (0.385), recovers its own input.")
