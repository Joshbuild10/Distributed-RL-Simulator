"""
Section 3 -- the response-length measurement campaign behind E[R], read from data/lengths/.
"""
import os

import pandas as pd

import lengths_io
from ..spec import Fig, register

LENGTHS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "data", "lengths")
DOMAINS = ["math", "code", "logic", "science"]
DOMAIN_MODELS = [("deepseek-v4-flash", "V4-flash"), ("deepseek-v4-pro-0813", "V4-pro"), ("glm45air", "GLM-4.5-Air")]
QWEN = [("qwen3-8b", "8B"), ("qwen3-14b", "14B"), ("qwen3-30b-a3b", "30B-A3B"), ("qwen3-32b", "32B*"),
        ("qwen3-235b-a22b", "235B-A22B")]
AGENTIC = {"tau2": "tau2 (tool)", "swesmith": "SWE-smith", "nebius": "SWE-rebench", "ccbench": "CC-Bench"}
CENSOR_WARN = 0.10                        # at or above this censored share the mean is a lower bound


def path(model, domain):
    return os.path.join(LENGTHS_DIR, f"{model}_{domain}_lengths.json")


def _frame(rows, axes):
    df = pd.DataFrame(rows)
    for col, cats in axes.items():
        df[col] = pd.Categorical(df[col], categories=cats, ordered=True)
    return df


def _len_rows(models, domains):
    rows = []
    for mdl, lab in models:
        for d in domains:
            m = lengths_io.mean_length(path(mdl, d))
            rows.append(dict(model=lab, domain=d, value=m if m is not None else float("nan"),
                             fallback=float("nan"), note=None if m is not None else "n/a"))
    return rows


@register("lengths_domain", "3")
def lengths_domain(opts):
    data = _frame(_len_rows(DOMAIN_MODELS, DOMAINS), {"model": [l for _, l in DOMAIN_MODELS], "domain": DOMAINS})
    return Fig("3a", "Measured mean response length by task domain", "bars", data, x="model", series="domain",
               ylabel="mean response length (tokens)", caption="Code > math > logic and science.")


@register("lengths_size", "3")
def lengths_size(opts):
    data = _frame(_len_rows(QWEN, ["math", "code"]), {"model": [l for _, l in QWEN], "domain": ["math", "code"]})
    return Fig("3b", "Size scaling: Qwen3 family mean response length", "bars", data, x="model", series="domain",
               ylabel="mean response length (tokens)",
               caption="*Qwen3-32B is ~40% censored at its generation cap, so its mean is a lower bound.")


def _stats(model, domain):
    """Summary stats of one measured series, or None if it wasn't measured."""
    s = lengths_io.load_stats(path(model, domain)) if os.path.exists(path(model, domain)) else {}
    return next(iter(s.values()), None)


@register("lengths_spread", "3")
def lengths_spread(opts):
    models = DOMAIN_MODELS + QWEN
    rows = []
    for mdl, lab in models:
        for d in DOMAINS:
            s = _stats(mdl, d)
            if s is None and mdl in dict(QWEN) and d not in ("math", "code"):
                continue
            rows.append(dict(model=lab, domain=d, value=s["cv"] if s else float("nan"), fallback=float("nan"),
                             note=None if s else "n/a"))
    data = _frame(rows, {"model": [l for _, l in models], "domain": DOMAINS})
    return Fig("3e", "Within-run spread: coefficient of variation of response length", "bars", data, x="model",
               series="domain", log=False, fmt="dec2", ylabel="cv (standard deviation / mean)",
               caption="The model takes E[R^2] = E[R]^2 (1 + cv^2), so cv = 1 doubles the attention and KV terms "
                       "and cv = 1.6 multiplies them 3.6x; the default threat model uses cv = 0.5. Censored runs "
                       "understate their cv (table 3d).")


@register("lengths_agentic", "3")
def lengths_agentic(opts):
    means = {n: lengths_io.agentic_means(os.path.join(LENGTHS_DIR, f"agentic_{n}.json")) for n in AGENTIC}
    order = sorted(AGENTIC, key=lambda n: means[n].get("total") or 0)
    parts = {"effective context": "total", "generated": "gen", "tool output": "tool"}
    rows = [dict(env=AGENTIC[n], part=p, value=means[n].get(k) or float("nan"), fallback=float("nan"), note=None)
            for n in order for p, k in parts.items()]
    data = _frame(rows, {"env": [AGENTIC[n] for n in order], "part": list(parts)})
    return Fig("3c", "Agentic effective context per episode", "bars", data, x="env", series="part",
               ylabel="tokens", caption="Single-turn mean response lengths are ~5-40k tokens for contrast; the "
                                        "model is single-turn (gap A-i).")
