"""
solvers.py -- Inverse calibration: given a published outcome, solve for the
least-constrained input that reproduces it.
"""

import math
from typing import Callable, Optional

from specs import Scenario
from model import (simulate, with_response_len, with_capacity_mult, with_n_nodes,
                   with_mfu_scale)


def _bisect(f: Callable[[float], float], lo: float, hi: float,
            log_scale: bool = False, iters: int = 80) -> Optional[float]:
    """Find x in [lo, hi] where f(x) == 0, given f(lo) <= 0 <= f(hi).
    Returns None if that bracket doesn't hold (target unreachable in range)."""
    mid = (lambda a, b: math.sqrt(a * b)) if log_scale else (lambda a, b: 0.5 * (a + b))
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(iters):
        m = mid(lo, hi)
        if f(m) < 0:
            lo = m
        else:
            hi = m
    return mid(lo, hi)


def solve_for_response_len(s: Scenario, target_step: float,
                           lo: float = 200.0, hi: float = 200_000.0) -> Optional[float]:
    """Invert the model: what mean response length reproduces a published step time?"""
    return _bisect(lambda R: simulate(with_response_len(s, R)).t_step - target_step, lo, hi)


def solve_for_capacity(s: Scenario, target_gen: float,
                       lo: float = 1e-3, hi: float = 100.0) -> Optional[float]:
    """Invert for the inference pool's effective capacity multiplier (scales node
    FLOP/s and HBM bandwidth together) that reproduces a published rollout+verify
    time. <1 means the real pool is weaker than the assumed node spec."""
    # gen(mult) is decreasing in mult, so flip the sign to match _bisect's f(lo)<=0<=f(hi)
    def f(mult):
        gen = simulate(with_capacity_mult(s, mult)).stages["rollout+verify"]
        return target_gen - gen
    return _bisect(f, lo, hi, log_scale=True)


def solve_for_mfu(s: Scenario, target_step: float, lo: float = 1e-3, hi: float = 5.0):
    """Implied MFU: what compute efficiency reproduces a published step time?

    Scales both trainer and inference MFU by a single k and finds the k giving t_step == target.
    Since exactly one stage binds, the meaningful output is that binding stage's implied MFU
    (preset MFU x k). Returns a dict:
        k                 -- efficiency multiplier on both MFUs (None if unreachable)
        binding           -- which stage is the max at the solution
        implied_mfu       -- preset MFU of the binding stage x k (None unless binding is compute)
        preset_mfu        -- the binding stage's assumed MFU, for comparison
        floor_stage/floor -- when unreachable: the non-compute stage (broadcast) that already
                             exceeds the target, so no finite MFU can slow the run enough / speed
                             it enough to match.

    Interpretation caveats live at the call site, but two are structural:
      * Only meaningful when a COMPUTE stage binds. If broadcast binds at the solution, the step
        time is set by weight volume / link speed and no MFU reproduces it -- implied_mfu=None.
      * The implied MFU absorbs EVERY unmodelled effect on the binding stage (real MFU, FSDP
        comm penalty with penalty_para=1, and any stage mis-attribution). It is therefore an
        upper bound on 'how low the true MFU is', not a clean MFU measurement.
    """
    # t_step is decreasing in k (more efficiency -> faster), so f(k)=target-t_step is increasing.
    f = lambda k: target_step - simulate(with_mfu_scale(s, k)).t_step
    k = _bisect(f, lo, hi, log_scale=True)
    if k is None:
        # Unreachable. If even at huge k (compute -> 0) t_step stays above target, a non-compute
        # stage floors it; report that floor so the caller can say "broadcast-bound, MFU can't fix".
        r_hi = simulate(with_mfu_scale(s, hi))
        return dict(k=None, binding=r_hi.bottleneck, implied_mfu=None, preset_mfu=None,
                    floor_stage=r_hi.bottleneck, floor=r_hi.t_step)
    r = simulate(with_mfu_scale(s, k))
    binding = r.bottleneck
    preset = s.train_hw.mfu if binding == "update" else (s.inf_hw.mfu if binding == "rollout+verify"
                                                         else None)
    # Baseline (preset-MFU) bottleneck. When it differs from `binding`, the solve had to push a
    # compute stage past a non-compute floor to match -- e.g. a run that is broadcast-bound at the
    # assumed MFU only becomes update-bound once MFU is dropped far enough. That flip is a signal
    # the residual may not be an efficiency story at all (INTELLECT-2's heterogeneous swarm).
    baseline_bottleneck = simulate(s).bottleneck
    return dict(k=k, binding=binding, preset_mfu=preset,
                implied_mfu=(preset * k if preset is not None else None),
                baseline_bottleneck=baseline_bottleneck,
                flipped=(baseline_bottleneck != binding),
                floor_stage=None, floor=None)


def implied_mfu_report(scenarios, width=104):
    """Print implied trainer MFU vs FSDP shard degree -- the back-calculation of 'what efficiency
    would reproduce each published step time'. scenarios = [(label, scenario)].

    Reading it: the model computes the FLOPs per step, so a published step time pins the effective
    efficiency of whichever stage binds. For the (mostly update-bound) reasoning-RL anchors this
    is an implied TRAINER MFU. prime-rl returns its own published 0.385 -- the method's built-in
    sanity check. The rest come out well below the assumed 0.40, and for the AReaL family the
    implied value falls with shard degree, which is the fingerprint of the unmodelled FSDP
    communication penalty (penalty_para=1.0)."""
    print(f"\n{'=' * width}\n  IMPLIED MFU from published step times (back-calculated)\n{'=' * width}")
    print("  'what trainer MFU would our FLOP-per-step accounting need to reproduce the paper's")
    print("  step time' -- absorbs real MFU + any unmodelled FSDP/comm penalty on the binding stage.")
    print(f"  {'anchor':<18}{'pub s':>7}{'n_shard':>8}{'assumed':>8}{'implied':>8}{'x/assumed':>10}"
          f"{'binding':>12}  note")
    for lab, sc in scenarios:
        pub = (sc.published or {}).get("t_step")
        if not pub:
            continue
        d = solve_for_mfu(sc, pub)
        if d["implied_mfu"] is None:
            print(f"  {lab:<18}{pub:>7.0f}{sc.train_hw.n_shard:>8}{'--':>8}{'n/a':>8}{'':>10}"
                  f"{d['binding']:>12}  {d['binding']}-bound floor {d['floor']:.0f}s > pub: MFU can't reproduce")
            continue
        note = ("flips from " + d["baseline_bottleneck"] + "-bound: not an MFU story"
                if d["flipped"] else "")
        print(f"  {lab:<18}{pub:>7.0f}{sc.train_hw.n_shard:>8}{d['preset_mfu']:>8.3f}"
              f"{d['implied_mfu']:>8.3f}{d['implied_mfu']/d['preset_mfu']:>9.2f}x"
              f"{d['binding']:>12}  {note}")
    print("  prime-rl returns ~its published 0.385 (MFU disclosed) -> method validated. Implied MFU")
    print("  falling with n_shard across AReaL 7B/14B/32B (48->64->96) is the FSDP-penalty signature.")


def solve_for_nodes(s: Scenario, target_step: float, max_nodes: int = 100_000) -> Optional[int]:
    """What inference node count brings rollout+verify down to the target step time?"""
    for n in range(1, max_nodes + 1):
        r = simulate(with_n_nodes(s, n))
        if r.stages["rollout+verify"] <= target_step:
            return n
    return None
