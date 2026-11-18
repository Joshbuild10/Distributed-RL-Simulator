"""
sweep.py -- One engine for every chart and table: expand axes into points, evaluate each once
(cached), return a tidy DataFrame with one row per point. Renderers only ever see frames.

    df = sweep({"target": TARGETS, "window_d": [90, 180]},
               lambda p: min_gpus(scenario(opts, target=p["target"]), p["window_d"]))

Every metric returns the same three plotting columns, so any renderer can draw any metric:
    value     the number plotted; NaN when the point misses its criterion
    fallback  drawn hollow instead (e.g. the GPUs that reach the fastest achievable wall-clock)
    note      short text for a hollow or missing bar ("173 d", "no-fit", "unbounded")
plus metric-specific columns (wall_d, bottleneck, eta, ...) that tables use.
"""
import itertools
import math

import pandas as pd

from feasibility_core import (evaluate, max_compute_operating_point, steps_and_c_rl,
                              required_bcast_mbps)

DAY = 86400
NAN = float("nan")
_cache = {}


def cached(fn, cfg, *args):
    """fn(cfg, *args) computed once per (fn, config, args). FeasConfig's repr is a complete,
    deterministic description of it, so it doubles as the key."""
    key = (fn.__name__, repr(cfg), args)
    if key not in _cache:
        _cache[key] = fn(cfg, *args)
    return _cache[key]


def cache_size():
    return len(_cache)


def sweep(axes, point_fn):
    """Cartesian product over `axes` ({name: iterable}); one row per point: its axis values plus
    point_fn(point). Axis columns become ordered categoricals in sweep order."""
    names = list(axes)
    rows = [{**p, **point_fn(p)} for p in (dict(zip(names, v)) for v in itertools.product(*axes.values()))]
    df = pd.DataFrame(rows)
    for name, vals in axes.items():
        df[name] = pd.Categorical(df[name], categories=list(dict.fromkeys(vals)), ordered=True)
    return df


def _plot(value=NAN, fallback=NAN, note=None):
    return dict(value=value, fallback=fallback, note=note)


# --- metrics ------------------------------------------------------------------------------------
def min_gpus(cfg, window_d):
    """Minimum GPUs to finish cfg's target in window_d. A miss keeps the GPUs that reach the fastest
    achievable wall-clock as a hollow fallback, labelled with that wall-clock."""
    if cfg is None:
        return {**_plot(note="no-fit"), "status": "nofit"}
    ev = cached(evaluate, cfg, window_d * DAY)
    if not ev["runnable"]:
        return {**_plot(note="no-fit"), "status": "nofit"}
    r = ev["r"]
    common = dict(gpus=ev["gpus"], wall_d=ev["wall_d"], train_frac=ev["frac"], bottleneck=ev["bottleneck"],
                  steps=ev["steps"], cost=ev["cost"], pct_stock=ev["pct_stock"], eta=ev["eta"],
                  t_step=r.t_step, t_update=r.t_update, t_rollout=r.stages["rollout+verify"], t_bc=r.t_bc)
    if ev["feasible"]:
        return {**_plot(value=ev["gpus"]), "status": "ok", **common}
    return {**_plot(fallback=ev["gpus"], note=f"{ev['wall_d']:.0f} d"), "status": "infeas", **common}


def max_compute(cfg, stock, window_d):
    """Max RL compute a stock can spend in window_d at its fastest split, net of failure downtime."""
    op = cached(max_compute_operating_point, cfg, stock)
    if op is None:
        return {**_plot(note="no-fit"), "status": "nofit"}
    steps, c_rl = steps_and_c_rl(op, window_d * DAY)
    return {**_plot(value=c_rl), "status": "ok", "gpus_used": op["gpus"], "steps": steps,
            "t_step": op["t_step"], "t_update": op["t_update"], "t_rollout": op["t_rollout"],
            "t_bc": op["t_broadcast"], "mfu": op["mfu_hw"], "bottleneck": op["bottleneck"],
            "train_frac": op["frac"], "eta": op["eta"], "quality_adjusted": c_rl * op["eta"]}


def required_link(cfg, window_d):
    """Slowest link x broadcast compression (Mbps at 1x) that keeps the WAN off the critical path."""
    mbps, ev = cached(required_bcast_mbps, cfg, window_d * DAY)
    if mbps is None or not math.isfinite(mbps):
        return {**_plot(note="unbounded" if ev["runnable"] else "no-fit"), "status": "nofit"}
    return {**_plot(value=mbps), "status": "ok", "bottleneck": ev["bottleneck"]}
