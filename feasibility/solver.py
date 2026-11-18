"""
solver.py -- Sizing and evaluation: the smallest cluster that finishes a run in a window, the fastest
run on a fixed fleet, the maximum compute a stock can spend, and the WAN link a run needs.
"""
import math

import penalties as pen
from model import simulate
from .config import with_cfg
from .hardware import GPU_STEP, GPUS, effective_node_gpus

# ---------------------------------------------------------------------------
# Core quantities
# ---------------------------------------------------------------------------
def per_step_flop(cfg):
    """Total FLOP per RL step (train + rollout); node-independent, so cheap to probe once."""
    r = simulate(cfg.build(64, 512))
    return r.tc.c_update + r.ro.c_rollout_total


def n_steps(cfg):
    return cfg.target_c_rl / per_step_flop(cfg)


def gpus_of(r):
    """Total GPUs allocated in a simulate result (asymmetric train/inference node sizes)."""
    s = r.scenario
    return s.train_hw.n_nodes * s.train_hw.gpus_per_node + s.inf_hw.n_nodes * s.inf_hw.gpus_per_node


def total_time(cfg, n_train_gpus, n_inf_gpus, tp=1):
    """Wall-clock to finish; inf if the config can't run:
      - inference weights don't fit a serving group (ro.model_fits) -> simulate() NaNs;
      - a replica + resident activation doesn't fit ONE trainer node (mem.fits) -> would need cross-node
        model parallelism, which this model disallows, so the config is inadmissible."""
    try:
        r = simulate(cfg.build(n_train_gpus, n_inf_gpus, tp))
    except (ValueError, ZeroDivisionError, OverflowError):
        return float("inf"), None
    if (r is None or not math.isfinite(r.t_step) or r.t_step <= 0
            or not r.ro.model_fits or not r.mem.fits):
        return float("inf"), r
    return n_steps(cfg) * r.t_step / r.goodput, r


# ONE sizing routine underpins BOTH solvers, so they can never disagree: given a per-step time
# budget tsb, size each stage to the smallest hardware that fits it. Rollout and update overlap
# (t_step = max of stages), so the two are sized INDEPENDENTLY against tsb -- which is what admits
# update-bound splits, where the update only has to beat tsb rather than beat rollout.
#   min_cluster(window)   = min_gpus_for_tsb(window / n_steps)         -- min GPUs to hit a window
#   optimal_split(budget) = smallest tsb whose min_gpus_for_tsb fits   -- min wall within a budget
# TSB_TOL is the relative convergence tolerance for that outer time bisection.
TSB_TOL = 1e-4


def _pack(tt, r):
    return (tt, r.scenario.train_hw.n_nodes, r.scenario.inf_hw.n_nodes, r,
            (gpus_of(r) - r.scenario.inf_hw.n_nodes * r.scenario.inf_hw.gpus_per_node) / gpus_of(r))


def _tp_candidates(cfg):
    """Serving-group sizes the solver may use, smallest first: powers of two up to inf_tp_max."""
    cap, out, t = max(1, cfg.inf_tp_max), [], 1
    while t <= cap:
        out.append(t); t *= 2
    if out[-1] != cap:
        out.append(cap)
    return out


def _min_gpus_at_tp(cfg, roll_b, upd_b, stock, tp, tsb):
    """min_gpus_for_tsb at ONE serving-group size, against per-stage budgets. See that docstring."""
    step = GPU_STEP
    per_gpu_flops = GPUS[cfg.gpu]["flops"] * cfg.mfu_train
    r16 = lambda x: -(-int(math.ceil(x)) // step) * step   # round up to a whole GPU_STEP
    grp = tp * cfg.gpus_per_node                           # a group is tp whole nodes
    inf_min = step if tp == 1 else grp

    def probe(n_inf):                                      # stage times here are trainer-independent
        r = simulate(cfg.build(cfg.train_gpus_per_node, n_inf, tp))
        if not r.ro.model_fits:
            return None
        pgh = r.scenario.train_hw.node_hbm / r.scenario.train_hw.gpus_per_node
        return dict(roll=r.stages["rollout+verify"],
                    mem_floor=r16(r.mem.total / pgh), c_update=r.tc.c_update)

    # build() sizes a single node from the GPU count, so the smallest probe can be a node too small
    # to hold the weights even when a full one would. Walk the floor up instead of calling the whole
    # config unrepresentable -- model_fits only improves with node size, and is flat once the node is
    # full (tp > 1 always builds full nodes, so there is nothing to walk).
    p0 = probe(inf_min)
    while p0 is None and inf_min < cfg.gpus_per_node:
        inf_min = min(inf_min + step, cfg.gpus_per_node)
        p0 = probe(inf_min)
    if p0 is None:
        return None
    max_inf = ((stock - p0["mem_floor"]) // step) * step   # leave room for at least a memory-floor trainer
    if max_inf < inf_min:
        return None
    pmax = probe(max_inf)
    if pmax is None or pmax["roll"] > roll_b:
        return None                                        # even max inference can't get rollout under budget
    if p0["roll"] <= roll_b:
        n_inf = inf_min
    else:
        lo, hi = inf_min, max_inf                          # lo fails, hi fits
        while hi - lo > step:
            mid = max(lo + step, (((lo + hi) // 2) // step) * step)
            pm = probe(mid)
            if pm is not None and pm["roll"] <= roll_b:
                hi = mid
            else:
                lo = mid
        n_inf = hi
    p = probe(n_inf)
    n_train = max(p["mem_floor"], r16(p["c_update"] / (per_gpu_flops * upd_b)))
    if n_train + n_inf > stock:
        return None
    # Re-check the BUILT allocation, not the requested one: build() rounds the trainer up to whole
    # replica nodes and inference into whole tp-groups, so both can overshoot the raw n_train+n_inf
    # test above. total_time also returns a finite-r/infinite-tt result when memory doesn't fit.
    tt, r = total_time(cfg, n_train, n_inf, tp)
    if r is None or not math.isfinite(tt) or gpus_of(r) > stock:
        return None
    # The stages were sized separately; the step they compose must still fit. Penalties make this
    # bind: the broadcast tail grows with the inference node count chosen above, and replica sync
    # adds to the update. The inference count is already the smallest that meets its budget, so a
    # miss here has no cheaper fix at this tp and split.
    if r.t_step > tsb * (1 + 1e-9):
        return None
    return dict(n_train=n_train, n_inf=n_inf, tp=tp, gpus=gpus_of(r), tt=tt, r=r)


# How a step composes decides how its time budget is divided between the stages:
#   k >= 1 (off-policy, overlapped) -- t_step = max(stages), so rollout and update EACH get the whole
#           budget and the broadcast just has to fit under it too (model.py already divides the
#           per-step broadcast by k, so a larger k only relaxes that check).
#   k = 0  (on-policy, serial)      -- t_step = SUM(stages), so the budget must be PARTITIONED. The
#           broadcast is GPU-independent and comes off the top; the remainder is swept between the two
#           compute stages, since which of them is cheaper to speed up varies by config.
SERIAL_SPLITS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def min_gpus_for_tsb(cfg, tsb, budget=None):
    """Smallest (trainer, inference) GPU counts whose step fits the per-step budget tsb.
    Inference: min GPUs with rollout+verify <= its budget (bisection; rollout falls with inference).
    Trainer:   min GPUs with update <= its budget, floored at the replica's one-node memory fit.
    Tries every allowed serving-group size (and, when serial, every budget split) and keeps the
    cheapest, so tp only rises when a bigger group actually buys hardware. -> dict(n_train, n_inf,
    tp, gpus, tt, r), or None if tsb is unreachable inside `budget` (default cfg.stock_gpus)."""
    if not (tsb > 0 and math.isfinite(tsb)):
        return None
    stock = cfg.stock_gpus if budget is None else budget
    bc = simulate(cfg.build(cfg.train_gpus_per_node, cfg.gpus_per_node)).t_bc   # GPU-independent
    if cfg.sync_interval >= 1:
        if bc > tsb:
            return None                                    # broadcast floor alone busts the budget
        splits = [(tsb, tsb)]
    else:
        rem = tsb - bc
        if rem <= 0:
            return None
        splits = [(a * rem, (1 - a) * rem) for a in SERIAL_SPLITS]
    best = None
    for tp in _tp_candidates(cfg):
        for roll_b, upd_b in splits:
            sol = _min_gpus_at_tp(cfg, roll_b, upd_b, stock, tp, tsb)
            if sol is not None and (best is None or sol["gpus"] < best["gpus"]):
                best = sol
    return best


def optimal_split(cfg, total_gpus):
    """Min-wall allocation fitting total_gpus: binary-search the smallest per-step budget tsb whose
    min_gpus_for_tsb() still fits. Searching via tsb covers the whole (trainer, inference) space,
    update-bound splits included. -> (time, n_train_nodes, n_inf_nodes, r, train_frac)."""
    steps = n_steps(cfg)
    if not (math.isfinite(steps) and steps > 0):
        return (float("inf"), 1, 1, None, 0.0)
    # Upper bound: one full serving group at the largest allowed tp -- the most permissive shape the
    # solver may build, so it settles "is this representable at all", and its per-step time is
    # reachable by definition. (A GPU_STEP-node seed instead reported no-fit whenever the weights
    # needed a full node.)
    tp0 = max(1, cfg.inf_tp_max)
    seed = simulate(cfg.build(cfg.train_gpus_per_node, cfg.gpus_per_node * tp0, tp0))
    if not seed.ro.model_fits:
        return (float("inf"), 1, 1, None, 0.0)
    hi = seed.t_step
    # The seed is an allocation, not a budget the sizing routine can necessarily reproduce: under a
    # serial (k=0) composition the budget is partitioned over SERIAL_SPLITS, which cannot land on an
    # exact stage boundary. Relax hi until it IS reproducible rather than declaring the config
    # unaffordable -- the bisection below then refines back down. No-op when k >= 1.
    best, relax = None, 0
    while best is None and relax < 8:
        best = min_gpus_for_tsb(cfg, hi, budget=total_gpus)
        if best is None:
            hi *= 2; relax += 1
    if best is None:
        return (float("inf"), 1, 1, None, 0.0)             # can't even afford the cheapest cluster
    lo = 0.0
    while hi - lo > max(hi * TSB_TOL, 1e-9):
        mid = 0.5 * (lo + hi)
        sol = min_gpus_for_tsb(cfg, mid, budget=total_gpus)
        if sol is not None:
            hi, best = mid, sol
        else:
            lo = mid
    return _pack(best["tt"], best["r"])


def min_cluster(cfg, target_s):
    """Smallest total GPU count finishing within target_s, capped by the stock -- i.e. the minimum
    hardware whose every stage fits the window's per-step budget. Returns dict or None (infeasible
    even at the full stock)."""
    steps = n_steps(cfg)
    if not (math.isfinite(steps) and steps > 0):
        return None
    # Goodput stretches the wall-clock, so the per-step budget is window * u / N_steps. u depends a
    # little on the allocation (failure rate scales with GPUs), so re-solve until the run fits.
    tsb = target_s / steps
    for _ in range(4):
        sol = min_gpus_for_tsb(cfg, tsb)
        if sol is None or sol["tt"] <= target_s * (1 + 1e-9):
            break
        tsb *= target_s / sol["tt"]
    if sol is None or sol["tt"] > target_s * (1 + 1e-9):
        return None
    r, g = sol["r"], sol["gpus"]
    return dict(gpus=g, nt=r.scenario.train_hw.n_nodes, ni=r.scenario.inf_hw.n_nodes,
                tt=sol["tt"], r=r,
                frac=(g - r.scenario.inf_hw.n_nodes * r.scenario.inf_hw.gpus_per_node) / g)


def min_wall_full_stock(cfg):
    """Fastest achievable wall-clock: throw the whole stock at it (optimal split).
    Below the window -> a stock/window limit; above -> the latency wall."""
    tt, nt, ni, r, f = optimal_split(cfg, cfg.stock_gpus)
    return dict(gpus=gpus_of(r) if r else cfg.stock_gpus, nt=nt, ni=ni, tt=tt, r=r, frac=f)


def evaluate(cfg, window_s):
    """THE unifier. One operating point per (cfg, window), ALWAYS populated:
        feasible in window -> the MIN cluster that hits it;
        infeasible         -> the FULL stock at optimal split = the fastest achievable wall-clock
                              ('days to expend the FLOPs') and the optimal trainer:rollout split;
        weights don't fit  -> runnable=False (no single-node inference; genuinely unrepresentable).
    So downstream tables never drop the wall-clock/split just because the window is missed."""
    sol = min_cluster(cfg, window_s)
    feasible = sol is not None
    if not feasible:
        sol = min_wall_full_stock(cfg)
    gpus = sol["gpus"]
    tt, r = sol["tt"], sol["r"]
    runnable = math.isfinite(tt) and r is not None
    cost = gpus * tt / 3600 * cfg.gpu_hr_usd if runnable else float("inf")
    return dict(
        feasible=feasible, runnable=runnable, window_s=window_s,
        gpus=gpus, N=sol["nt"] + sol["ni"], nt=sol["nt"], ni=sol["ni"], frac=sol["frac"],
        wall_s=tt, wall_d=(tt / 86400 if runnable else float("inf")),
        cost=cost, bottleneck=(r.bottleneck if r else "no-fit"),
        pct_stock=100 * gpus / cfg.stock_gpus, r=r,
        # Quality factor (penalties.py): what the compute is worth vs a co-located run. Reported
        # alongside, never used to size -- quality-adjusted compute is a separate presentation.
        eta=(r.eta if r else 1.0), goodput=(r.goodput if r else 1.0),
        # Training steps to spend cfg.target_c_rl at this RL config -- a property of the batch
        # size/model/target, NOT of the cluster or window (per_step_flop is node-count-independent).
        steps=n_steps(cfg) if runnable else float("nan"),
    )


# Link speed at which the weight broadcast is definitely NOT a step bottleneck (~1 Pbit/s -- for any
# model here t_bc collapses to the fixed latency floor). required_bcast_mbps solves the operating
# point AT this speed so the metric is decoupled from the config's own link and compression.
BCAST_FREE_MBPS = 1e9


def required_bcast_mbps(cfg, window_s):
    """Minimum weight-sync link x compression (Mbps at 1x) for the broadcast to stay hidden -- not
    become the step-time bottleneck -- at this target/window. -> (mbps_or_None, ev).

    Solved with the broadcast effectively FREE (BCAST_FREE_MBPS), so evaluate() picks the allocation
    and its stage times without the current link or compression influencing them. Otherwise the
    metric is circular: a slow link slackens the trainer and would report a LOWER requirement, and
    the same coupling made it drift with the compression knob.

    The broadcast overlaps the whole step (>=1-step off-policy: concurrent with the next step's
    generation AND update), so under t_step = max(gen, update, bcast) it is hidden iff it fits under
    the SLOWER compute stage, max(rollout+verify, update). k>=2 gives it k steps.

    Returned at 1x and compression-independent -- uncompressed volume over the time budget -- so a
    real compression factor simply divides it. None when the model doesn't fit, or when even a free
    link's latency floor exceeds the budget: no finite bandwidth suffices, which is a real answer."""
    ev = evaluate(with_cfg(cfg, wan_mbps=BCAST_FREE_MBPS), window_s)
    if not ev["runnable"]:
        return None, ev
    r = ev["r"]
    t_compute = max(r.t_update, r.stages["rollout+verify"])   # broadcast hides behind the slower stage
    lat = r.scenario.net.latency
    k = max(r.scenario.algo.sync_interval, 1)                 # k>=2 amortises the broadcast over k steps
    # Penalties put a fan-out tail f on (volume / bw + latency); hidden iff f*(V/bw + lat) <= k*t_compute.
    p = r.scenario.penalty
    f = pen.straggler_factor(r.scenario.inf_hw.n_nodes, p.inf_mode) if p.enabled else 1.0
    budget = k * t_compute / f - lat
    if not (math.isfinite(t_compute) and budget > 0):
        return None, ev
    raw_weight_bytes = cfg.b_weights_inf * cfg.model.p_total  # uncompressed broadcast volume (1x)
    need = raw_weight_bytes * 8 / 1e6 / budget
    # A distributed trainer also all-reduces gradients every step (up + down, no k amortisation) over
    # the same link; the requirement is whichever transfer needs more.
    hw = r.scenario.train_hw
    n_rep = hw.n_nodes // max(1, math.ceil(hw.n_shard / hw.gpus_per_node))
    if cfg.train_distributed and n_rep > 1:
        fs = pen.straggler_factor(n_rep, p.mode) if p.enabled else 1.0
        sync_budget = t_compute / fs - lat
        if sync_budget <= 0:
            return None, ev
        # Its own compression, so convert to the broadcast's 1x basis: the link x broadcast compression
        need = max(need, 2 * r.scenario.algo.b_grads * cfg.model.p_total * cfg.compression
                   / cfg.grad_compression * 8 / 1e6 / sync_budget)
    return need, ev


def max_compute_operating_point(cfg, stock=None):
    """The window-INDEPENDENT max-throughput operating point: split, T_step (+ the per-stage
    breakdown), achieved whole-cluster MFU, bottleneck, per-step FLOP. `stock` overrides
    cfg.stock_gpus. None if the model doesn't fit. Computed once per (cfg, stock) so a caller
    sweeping windows never re-runs optimal_split.

    The MFU here is derived, not echoed: r.mfu_model/r.mfu_hw only see the trainer's own t_update,
    so they are identical at every stock size. This divides actual FLOPs (train + rollout) by
    T_step x the combined peak of every allocated GPU, so a pool idling on the other stage's
    bottleneck drags it down -- it moves with the split (and craters when broadcast dominates)."""
    c = cfg if stock is None else with_cfg(cfg, stock_gpus=stock)
    eff = effective_node_gpus(c.gpus_per_node, c.stock_gpus)
    if eff != c.gpus_per_node:
        c = with_cfg(c, gpus_per_node=eff)
    sol = optimal_split(c, c.stock_gpus)   # total_time ~monotone in GPUs: whole stock, split optimally
    if sol is None or not math.isfinite(sol[0]) or sol[3] is None:
        return None
    tt, nt, ni, r, frac = sol
    # Per-step time is r.t_step; optimal_split's `tt` is the whole-run wall-clock (n_steps x t_step).
    pf = per_step_flop(c)   # node-count-independent == r.tc.c_update + r.ro.c_rollout_total
    s = r.scenario
    total_peak = s.train_hw.n_nodes * s.train_hw.flops + s.inf_hw.n_nodes * s.inf_hw.flops
    # model-FLOPs convention: strip the recompute coefficient off the training term only (inference
    # is forward-only) -- mirrors model.py's own model_flops.
    model_flops_step = (3.0 / (3 + s.algo.recomp_act + s.algo.recomp_old)) * r.tc.c_update + r.ro.c_rollout_total
    return dict(gpus=gpus_of(r), n_total=nt + ni, nt=nt, ni=ni, frac=frac,
                t_step=r.t_step, t_update=r.t_update, t_rollout=r.stages["rollout+verify"],
                t_broadcast=r.t_bc, bottleneck=r.bottleneck,
                mfu_model=model_flops_step / (r.t_step * total_peak),
                mfu_hw=pf / (r.t_step * total_peak),
                per_step_flop=pf, goodput=r.goodput, eta=r.eta)


def steps_and_c_rl(op, window_s):
    """Window-dependent numbers from an operating point: (steps, max achievable C_RL). Pure
    arithmetic, so sweeping many windows over one op is free. Goodput removes failure downtime;
    multiply by op["eta"] separately for quality-adjusted compute."""
    steps = window_s * op.get("goodput", 1.0) / op["t_step"]
    return steps, op["per_step_flop"] * steps
