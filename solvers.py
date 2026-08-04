"""
solvers.py -- Inverse calibration: given a published outcome, solve for the
least-constrained input that reproduces it.
"""

import math
from typing import Callable, Optional

from specs import Scenario
from model import simulate, with_response_len, with_capacity_mult, with_n_nodes


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


def solve_for_nodes(s: Scenario, target_step: float, max_nodes: int = 100_000) -> Optional[int]:
    """What inference node count brings rollout+verify down to the target step time?"""
    for n in range(1, max_nodes + 1):
        r = simulate(with_n_nodes(s, n))
        if r.stages["rollout+verify"] <= target_step:
            return n
    return None
