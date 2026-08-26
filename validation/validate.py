#!/usr/bin/env python3
"""
validate.py -- Validate the analytic runtime model against PUBLISHED distributed-RL runs.

Reports every published scenario (predicted vs published step time, throughput, MFU,
broadcast) and then inverse-calibrates: given a published outcome, solve for the input the
paper did not disclose (usually E[R], or the swarm's effective capacity).

The model implements the rollout -> verify -> update -> broadcast loop of the prime-rl /
INTELLECT architecture. Equations follow the project's term-splitting (equivalent to, but
arranged differently from, the standard forms).

Run:  python -m validation.validate            (from the repo root)
      python validation/validate.py
      ... --sweep                              (adds response-length sensitivity tables)

Conventions
-----------
* FLOPs/s and bandwidth are specified PER NODE (a node = whatever group of GPUs
  holds one inference replica, or one FSDP data-parallel shard group).
* P_active_layers excludes embeddings and LM head; those are tracked separately,
  because for MoE they are a large fraction of "active params" and behave
  differently (head runs on generated tokens in decode, all tokens in training).
* Causal masking => leading coefficient 2 (not 4) on prefill/training attention.
  Decode attention keeps coefficient 4 because the sum over growing context
  already counts only causal pairs.
* Throughput is reported STAGE-LOCAL (rate while that stage is running) alongside the
  whole-step "achieved" rate; see reporting.CALIBRATION_METRICS for which one each
  published figure is compared against.
* All times in seconds, all memory in bytes, all compute in FLOPs.

Repo layout
-----------
* specs.py                -- dataclasses (ModelSpec, RLSpec, AlgoSpec, HWSpec, NetSpec,
                             VerifySpec, Scenario)
* model.py                -- the analytic core (training/rollout/verify/broadcast terms,
                             simulate(), scenario-tweak helpers)
* validation/presets.py   -- hardware/model constants and the published scenarios
* validation/solvers.py   -- inverse calibration against published numbers
* validation/reporting.py -- fmt()/report()/sweep_response_len()
* feasibility.py          -- feasibility core (FeasConfig, evaluate, min_cluster)
* feasibility_tables.py   -- feasibility sweeps/tables/CSV writers
"""

import argparse
import os
import sys
from typing import Optional, Tuple

# Repo root on sys.path so `specs`/`model` resolve when this file is run directly as a
# script (no-op under `python -m validation.validate`, where the root is already sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimResult, Scenario, simulate, with_response_len, with_capacity_mult
from validation.presets import (intellect2, intellect3, primerl_paper, primerl_deepdive,
                                areal_1_5b, areal_7b, areal_14b, areal_32b)
from validation.solvers import solve_for_capacity, solve_for_response_len, implied_mfu_report
from validation.reporting import fmt, report, sweep_response_len
from validation.uncertainty import print_mc_report


def _capacity_calibration(sc: Scenario, target_gen: float) -> Tuple[Optional[float], Optional[SimResult]]:
    """Solve for the inference-pool capacity multiplier that reproduces target_gen
    (a rollout+verify time). Returns (multiplier or None, result recalibrated at
    that multiplier or None)."""
    mult = solve_for_capacity(sc, target_gen)
    recal = simulate(with_capacity_mult(sc, mult)) if mult else None
    return mult, recal


def _response_len_calibration(sc: Scenario, target_step: float) -> Tuple[Optional[float], Optional[SimResult]]:
    """Solve for the mean response length that reproduces target_step (a step time).
    Returns (E[R] or None, result at that E[R] or None)."""
    R = solve_for_response_len(sc, target_step)
    r = simulate(with_response_len(sc, R)) if R else None
    return R, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="add sensitivity tables")
    args = ap.parse_args()

    scenarios = [intellect2("short"), intellect2("long"), intellect3(),
                 primerl_paper(), primerl_deepdive(),
                 areal_1_5b(), areal_7b(), areal_14b(), areal_32b()]
    for s in scenarios:
        report(simulate(s))

    print("=" * 78)
    print("  INVERSE CALIBRATION (solve for the least-constrained input)")
    print("=" * 78)

    for tag, tgt in (("short", 22 * 60), ("long", 21 * 60)):
        sc = intellect2(tag)
        base = simulate(sc)
        mult, recal = _capacity_calibration(sc, tgt - base.t_verify)
        print(f"\n  INTELLECT-2 TARGET-{tag.upper()}: published step {tgt/60:.0f} min.")
        print(f"    Model with 8 x (8xH100) swarm nodes: rollout+verify = "
              f"{fmt(base.stages['rollout+verify'],'s')}  (broadcast "
              f"{fmt(base.t_bc,'s')}, update {fmt(base.t_update,'s')})")
        if mult:
            print(f"    Implied swarm capacity multiplier = {mult:.3f} "
                  f"({1/mult:.1f}x weaker than assumed)")
            print(f"    -> calibrated: T_step {fmt(recal.t_step,'s')}, bottleneck "
                  f"{recal.bottleneck}, inf:train GPU-time {recal.gputime_ratio:.1f}x "
                  f"(published ~4.5x)")

    pr = primerl_paper()
    bp = simulate(pr)
    print(f"\n  prime-rl reference run: published step {22.9:.1f} min, trainer 11.3K tok/s,"
          f" inference 14.4K tok/s, MFU 38.5%.")
    print(f"    Model at E[R]={pr.rl.response_len_mean:,.0f}: T_step "
          f"{fmt(bp.t_step,'s')}, trainer {bp.tok_s_train:,.0f} tok/s, inference "
          f"{bp.tok_s_inf:,.0f} tok/s, MFU {bp.mfu_model*100:.1f}%")
    Rp, rp = _response_len_calibration(pr, 22.9 * 60)
    if Rp:
        print(f"    E[R] reproducing the 22.9-min step = {Rp:,.0f} tok "
              f"(cap 16,384 -> {'INSIDE' if Rp < 16384 else 'ABOVE'} the cap)")
        print(f"    -> there: trainer {rp.tok_s_train:,.0f} tok/s (pub 11,300), "
              f"inference {rp.tok_s_inf:,.0f} tok/s (pub 14,400), bottleneck "
              f"{rp.bottleneck}")

    i3 = intellect3()
    b3 = simulate(i3)
    print(f"\n  INTELLECT-3: published step 1500 s. Model gives "
          f"{fmt(b3.t_step,'s')} at E[R]={i3.rl.response_len_mean:,.0f}.")
    R, r3 = _response_len_calibration(i3, 1500.0)
    if R:
        print(f"    E[R] reproducing 1500 s = {R:,.0f} tokens "
              f"(max context 65,536 -> plausible for this run)")
        print(f"    -> at that length: bottleneck {r3.bottleneck}, attention "
              f"{r3.tc.attn_frac*100:.0f}% of layer FLOPs, inf:train GPU-time "
              f"{r3.gputime_ratio:.1f}x vs published 1:3 node split")
    mult3, _ = _capacity_calibration(i3, 1500.0 - b3.t_verify)
    if mult3:
        print(f"    (alternative: holding E[R]=32k, an inference capacity multiplier of "
              f"{mult3:.2f} also lands on 1500 s)")

    print("\n" + "=" * 78)
    print("  AReaL (arXiv 2505.24298): disaggregated async, 4-point scale check")
    print("=" * 78)
    print("  E[R] not published -> validate via the E[R] that reproduces the published step.")
    print(f"  {'run':<24}{'model t_step':>16}{'published':>14}{'err':>7}{'inv E[R]':>11}")
    for make in (areal_1_5b, areal_7b, areal_14b, areal_32b):
        sc = make()
        r = simulate(sc)
        R = solve_for_response_len(sc, sc.published["t_step"])
        err = (r.t_step - sc.published["t_step"]) / sc.published["t_step"] * 100
        tag = sc.name.replace("AReaL ", "").split("(")[0].strip()
        cap = sc.rl.max_response_len
        rstr = f"{R:,.0f}" + ("" if (R and R < cap) else "!") if R else "n/a"
        print(f"  {tag:<24}{fmt(r.t_step,'s'):>16}{fmt(sc.published['t_step'],'s'):>14}"
              f"{err:>+6.0f}%{rstr:>11}")
    print("  (inv E[R] under the 32,768-cap = plausible; '!' = above cap)")

    # Predictive intervals. The inverse-calibration blocks above ask "what input value would
    # reproduce the published number"; this asks the complementary question -- given honest
    # uncertainty on the inputs nobody published, is the published number inside our predictive
    # range at all? Text counterpart of dashboard chart 1a's error bars.
    # Explicit short labels, not derived from sc.name: splitting on " (" collapsed
    # TARGET-SHORT/TARGET-LONG to one indistinguishable "INTELLECT-2" and left "prime-rl
    # reference run" wide enough to break the column alignment.
    anchors_for_mc = [("INTELLECT-2 short", intellect2("short")),
                      ("INTELLECT-2 long", intellect2("long")),
                      ("INTELLECT-3", intellect3()),
                      ("prime-rl", primerl_paper()),
                      ("AReaL-1.5B", areal_1_5b()), ("AReaL-7B", areal_7b()),
                      ("AReaL-14B", areal_14b()), ("AReaL-32B", areal_32b())]
    print_mc_report(anchors_for_mc)

    # Complementary to the MC: instead of "is the gap inside our uncertainty", ask "what single
    # efficiency would close it". For update-bound anchors this is an implied trainer MFU, and its
    # decline with shard degree is the concrete, quantified form of the FSDP-penalty residual.
    implied_mfu_report(anchors_for_mc)

    if args.sweep:
        print("\n" + "=" * 78)
        print("  SENSITIVITY")
        print("=" * 78)
        sweep_response_len(intellect2("short"), [1000, 2500, 5000, 10000, 20000])
        sweep_response_len(intellect3(), [4000, 8000, 16000, 32000, 64000])
        sweep_response_len(primerl_paper(), [3000, 6000, 9000, 12000, 15000])
        sweep_response_len(primerl_deepdive(), [2000, 4000, 8000, 16000, 30000])
        print()


if __name__ == "__main__":
    main()
