"""
sweep.py -- One engine for every chart and table: expand axes into points, evaluate each once
(cached), return a tidy DataFrame (one row per point). Renderers only ever see frames, never configs.

    rows = sweep({"target": TARGETS, "window_d": [90, 180]},
                 lambda p: min_gpus(scenario("decentralised", target=p["target"]), p["window_d"]))

Every metric returns the same columns, so any renderer can draw any metric:
    value     the number the chart plots; None when the point misses its criterion
    fallback  what to draw hollow instead (e.g. GPUs at the fastest achievable wall-clock)
    note      short text for a hollow or missing bar ("230 d", "no-fit")
plus metric-specific columns (wall_d, bottleneck, eta, ...) for tables.
"""
import itertools

import pandas as pd

from feasibility_core import (evaluate, max_compute_operating_point, steps_and_c_rl,
                              required_bcast_mbps)

DAY = 86400
_cache = {}


def _cached(fn, cfg, *args):
    """fn(cfg, *args), computed once per (fn, config, args). FeasConfig's repr is a complete,
    deterministic description of it, so it doubles as the key."""
    key = (fn.__name__, repr(cfg), args)
    if key not in _cache:
        _cache[key] = fn(cfg, *args)
    return _cache[key]


def cache_info():
    return len(_cache)


def sweep(axes, point_fn):
    """Cartesian product over `axes` ({name: iterable}); one row per point: its axis values plus
    point_fn(point). Axis columns keep their sweep order as ordered categoricals."""
    names = list(axes)
    df = pd.DataFrame([{**point, **point_fn(point)}
                       for point in (dict(zip(names, vals)) for vals in itertools.product(*axes.values()))])
    for name, vals in axes.items():
        df[name] = pd.Categorical(df[name], categories=list(dict.fromkeys(vals)), ordered=True)
    return df


# --- metrics ------------------------------------------------------------------------------------
def min_gpus(cfg, window_d):
    """Minimum GPUs to finish cfg's target in window_d. Missing the window keeps the GPUs that reach
    the fastest achievable wall-clock as a hollow fallback, labelled with that wall-clock."""
    ev = _cached(evaluate, cfg, window_d * DAY)
    if not ev["runnable"]:
        return dict(value=None, fallback=None, note="no-fit", feasible=False, runnable=False)
    return dict(value=ev["gpus"] if ev["feasible"] else None,
                fallback=None if ev["feasible"] else ev["gpus"],
                note=None if ev["feasible"] else f"{ev['wall_d']:.0f} d",
                feasible=ev["feasible"], runnable=True, gpus=ev["gpus"], wall_d=ev["wall_d"],
                train_frac=ev["frac"], bottleneck=ev["bottleneck"], eta=ev["eta"])


def max_compute(cfg, stock, window_d):
    """Max RL compute a stock can spend in window_d at its fastest split (net of failure downtime)."""
    op = _cached(max_compute_operating_point, cfg, stock)
    if op is None:
        return dict(value=None, fallback=None, note="no-fit", runnable=False)
    steps, c_rl = steps_and_c_rl(op, window_d * DAY)
    return dict(value=c_rl, fallback=None, note=None, runnable=True, gpus_used=op["gpus"],
                steps=steps, t_step_s=op["t_step"], bottleneck=op["bottleneck"],
                train_frac=op["frac"], eta=op["eta"], quality_adjusted=c_rl * op["eta"])


def required_link(cfg, window_d):
    """Slowest link x broadcast compression (Mbps at 1x) that keeps the WAN off the critical path."""
    mbps, ev = _cached(required_bcast_mbps, cfg, window_d * DAY)
    return dict(value=mbps, fallback=None, note=None if mbps else "unbounded",
                runnable=ev["runnable"], bottleneck=ev["bottleneck"])
