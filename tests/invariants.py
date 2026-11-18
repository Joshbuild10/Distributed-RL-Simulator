#!/usr/bin/env python3
"""
invariants.py -- regression gate for the feasibility solver (pytest runs it; so can you).

Two jobs:
  * INVARIANTS -- properties that must hold at every operating point, swept over a config grid:
      allocated GPUs <= stock; feasible => runnable, memory fits, finite wall; feasible => the
      wall-clock is inside the window; no-fit only when even a full node can't hold the model.
  * GOLDEN     -- a snapshot of headline numbers, so a refactor that changes them says so.
      `--write` records the current values; a plain run compares against the record.

    python -m tests.invariants              # check invariants + golden
    python -m tests.invariants --write      # re-record golden (after an INTENDED change)
    python -m tests.invariants --penalties  # also sweep with Rahman's penalties on
"""
import argparse
import json
import math
import os
import sys

from feasibility import FeasConfig, MODELS, evaluate, gpus_of, model_targets, optimal_split, with_cfg
from model import simulate
from specs import PenaltySpec

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")
DAY = 86400


WINDOWS_D = (30, 180)
FULL = dict(models=("Llama3-405B", "Qwen3-235B-A22B", "Kimi-K3"), gpus=("H100", "B200", "RTX4090"),
            nodes=(16, 32, 72, 128), ks=(0, 1, 2, 4), stocks=(256, 100_000, 1_500_000))
QUICK = dict(models=("Llama3-405B", "Kimi-K3"), gpus=("H100", "RTX4090"), nodes=(16, 128), ks=(0, 1),
             stocks=(256, 1_500_000))


def grid(penalty=None, models=FULL["models"], gpus=FULL["gpus"], nodes=FULL["nodes"], ks=FULL["ks"],
         stocks=FULL["stocks"]):
    """Config points to sweep: model x gpu x node size x staleness x stock (864 with both windows)."""
    penalty = penalty or PenaltySpec()
    for m in models:
        for gpu in gpus:
            for node in nodes:
                for k in ks:
                    for stock in stocks:
                        yield FeasConfig(target_c_rl=2.5e25, model=MODELS[m], gpu=gpu,
                                         gpus_per_node=node, sync_interval=k, stock_gpus=stock,
                                         penalty=penalty)


def check(cfg, wd):
    """Every invariant at one operating point; returns the failures (empty = pass)."""
    fails = []
    ev = evaluate(cfg, wd * DAY)
    tag = (f"{cfg.model.name} {cfg.gpu} node={cfg.gpus_per_node} k={cfg.sync_interval:g} "
           f"stock={cfg.stock_gpus} {wd}d")
    if ev["gpus"] > cfg.stock_gpus:
        fails.append(f"{tag}: allocated {ev['gpus']:,} > stock {cfg.stock_gpus:,}")
    if ev["feasible"]:
        if not ev["runnable"]:
            fails.append(f"{tag}: feasible but not runnable")
        elif not math.isfinite(ev["wall_s"]):
            fails.append(f"{tag}: feasible but wall is not finite")
        elif not ev["r"].mem.fits:
            fails.append(f"{tag}: feasible but trainer memory does not fit")
        elif ev["wall_s"] > wd * DAY * (1 + 1e-6):
            fails.append(f"{tag}: feasible but wall {ev['wall_d']:.1f}d > {wd}d")
    # no-fit must mean the hardware genuinely cannot hold the model at its most permissive
    # shape -- a full serving group for inference, a max-size replica node for the trainer.
    # Anything else is the solver's sizing search bailing early.
    if not ev["runnable"]:
        full = simulate(cfg.build(cfg.train_gpus_per_node, cfg.gpus_per_node,
                                  max(1, cfg.inf_tp_max)))
        if full.ro.model_fits and full.mem.fits:
            fails.append(f"{tag}: no-fit, yet a full node holds both the weights "
                         f"and the trainer replica")
    return fails


def invariants(penalty=None, **axes):
    points = [(cfg, wd) for cfg in grid(penalty, **axes) for wd in WINDOWS_D]
    return len(points), [f for cfg, wd in points for f in check(cfg, wd)]


def headline():
    """Numbers a refactor must not move silently."""
    base = FeasConfig(target_c_rl=2.5e25)
    out = {}
    for name, c_rl in model_targets().items():
        ev = evaluate(with_cfg(base, target_c_rl=c_rl), 90 * DAY)
        out[f"min_cluster_90d/{name}"] = ev["gpus"] if ev["feasible"] else None
        out[f"wall_d/{name}"] = round(ev["wall_d"], 2) if ev["runnable"] else None
    for stock in (1024, 100_000, 1_500_000):
        tt, nt, ni, r, frac = optimal_split(with_cfg(base, stock_gpus=stock), stock)
        out[f"optimal_split/{stock}"] = [round(tt, 1), nt, ni, round(frac, 4), gpus_of(r)]
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m tests.invariants")
    ap.add_argument("--write", action="store_true", help="re-record golden.json")
    ap.add_argument("--penalties", action="store_true", help="also sweep with penalties on")
    a = ap.parse_args()

    runs = [("penalties off", None)]
    if a.penalties:
        runs += [("penalties on (async inference)", PenaltySpec(enabled=True)),
                 ("penalties on (relay inference)", PenaltySpec(enabled=True, inf_mode="relay"))]
    fails = []
    for tag, pspec in runs:
        n, f = invariants(pspec)
        fails += f
        print(f"invariants, {tag}: {n} operating points, {len(f)} failures")
        for x in f[:10]:
            print("  FAIL", x)
        if len(f) > 10:
            print(f"  ... {len(f)-10} more")

    now = headline()
    if a.write:
        json.dump(now, open(GOLDEN, "w"), indent=1, sort_keys=True)
        print(f"wrote {GOLDEN} ({len(now)} values)")
        return 1 if fails else 0
    if not os.path.exists(GOLDEN):
        print("no golden.json -- run with --write first")
        return 1 if fails else 0
    was = json.load(open(GOLDEN))
    drift = [(k, was.get(k), now[k]) for k in sorted(now) if was.get(k) != now[k]]
    drift += [(k, was[k], "<gone>") for k in sorted(was) if k not in now]
    print(f"golden: {len(now)} values, {len(drift)} changed")
    for k, o, v in drift:
        print(f"  DRIFT {k}: {o} -> {v}")
    return 1 if (fails or drift) else 0


if __name__ == "__main__":
    sys.exit(main())
