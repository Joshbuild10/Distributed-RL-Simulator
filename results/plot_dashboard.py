#!/usr/bin/env python3
"""plot_dashboard.py -- ONE self-contained HTML dashboard (inline SVG, no matplotlib) over
every result in the project, in three strands:

  1. VALIDATION   -- model.simulate vs published step times for the calibration anchors.
  2. FEASIBILITY  -- the feasibility.py sweeps: A1 frontier, max-compute-by-stock, sensitivity
                     tornado, cross-model, GPU family, required broadcast bandwidth, total
                     rollout batch, batch-config compare, HBM elasticity, policy bars, E[R].
  3. LENGTHS      -- the measurement campaign: domain multiplier, size scaling, agentic
                     effective context (read from results/lengths/*.json).

Recomputes the model/feasibility numbers live (always matches the code) and reads the length
JSONs from disk. Usage: python results/plot_dashboard.py [out.html]
"""
import sys, os, math
import _bootstrap  # noqa: F401  (puts the repo root + experiments/ on sys.path)
import lengths_io
from svgkit import (PAL, esc, grouped_bars, heatmap, signed_hbars, small_multiples, tornado,
                    panel, page, fmt_sci)
from validation.presets import (intellect2, intellect3, primerl_paper,
                     areal_1_5b, areal_7b, areal_14b, areal_32b)
from validation.uncertainty import mc_step_time, MC_DRAWS, MFU_INF_RANGE, ER_PCT
from validation.solvers import solve_for_mfu
from model import simulate
from feasibility_core import (FeasConfig, MODELS, POLICY_THRESHOLDS, GPUS, HIGH_BATCH,
                              BASELINE_BATCH, BATCH_CAP, BATCH_SWEEP, HBM_MULTS, HIGH_END_GPUS,
                              batch_split, all_targets, model_targets, evaluate,
                              smallest_fitting_gpu, total_time, optimal_split, per_step_flop,
                              required_bcast_mbps, config_lines, node_gpus_tpp_cap,
                              tpp_per_chip, node_gpus_power_cap, effective_node_gpus,
                              with_cfg as _with, split_for as _split,
                              fmt_time as _ft, fmt_num as _fn, fmt_ratio as _tr, fmt_rate)

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/dashboard.html"
# The 24 small measured-length summary JSONs (~144K) that section 3 reads, copied here from
# experiments/lengths/ (which also holds ~140MB of raw per-response .traces.jsonl that nothing
# reads back -- measurement exhaust, not a dashboard input) so results/ is self-contained and
# git-trackable: clone the repo, run this script, get the same dashboard, section 3 included.
LEN = "results/lengths"
base = FeasConfig(target_c_rl=2.5e25)
GPN = base.gpus_per_node


# ------------------------------------------------------------------ 1. VALIDATION
# Labels split over two lines ("AReaL\n32B") so each small-multiple panel's ~100px of width can
# carry the anchor name without rotating it.
CAL_SC = [("INTELLECT-2\nshort", intellect2("short")), ("INTELLECT-2\nlong", intellect2("long")),
          ("INTELLECT-3", intellect3()), ("prime-rl", primerl_paper()),
          ("AReaL\n1.5B", areal_1_5b()), ("AReaL\n7B", areal_7b()),
          ("AReaL\n14B", areal_14b()), ("AReaL\n32B", areal_32b())]
CAL = []
for lab, sc in CAL_SC:
    r = simulate(sc); pub = (sc.published or {}).get("t_step")
    if pub:
        CAL.append((lab, r.t_step, pub, (r.t_step - pub) / pub * 100))

# Monte-Carlo predictive intervals: trainer/inference MFU (never published by any anchor) and
# E[R] (measured on proxy checkpoints) are genuinely uncertain, so a bare residual can't
# distinguish "modelling gap" from "input uncertainty". Seeded per anchor -- unseeded draws would
# change the HTML on every regeneration. See validation/uncertainty.py for the priors and the
# per-anchor E[R] provenance tiers.
MC = {lab: mc_step_time(sc) for lab, sc in CAL_SC}

# One panel per anchor, each with its own zero-based LINEAR axis. The old single log-scaled
# grouped bar chart spanned ~200s..1900s of published step time, which compressed every
# predicted/published pair into near-identical heights -- a 2x miss read as a rounding error.
# Bar = MC MEAN (not the deterministic point value), whiskers = p5..p95.
cal_bars = small_multiples([(lab, MC[lab]["mean"], pub, MC[lab]["p5"], MC[lab]["p95"])
                            for lab, mdl, pub, err in CAL],
                           "1a. Validation -- MC mean predicted vs published step time, whiskers = 90% "
                           "predictive interval",
                           ylab="step time (s)")
# 1b bar stays on the DETERMINISTIC point prediction on purpose: that error % is the number quoted
# throughout the project's calibration history (prime-rl ~0%, AReaL-32B -72%, ...), and rebasing
# it on the MC mean would silently make every previously-recorded figure incomparable. The whisker
# adds the MC 90% interval mapped to error % -- so each row shows the point residual AND the range
# input uncertainty can move it to. A whisker that fails to cross 0 = a residual uncertainty can't
# explain (the same "explained?" verdict as the MC appendix, shown geometrically).
cal_err = signed_hbars([(l.replace("\n", " "), e,
                         (MC[l]["p5"] - p) / p * 100, (MC[l]["p95"] - p) / p * 100)
                        for l, m, p, e in CAL],
                       "1b. Validation error -- point prediction (bar) with 90% MC interval (whisker)",
                       xlab="prediction error vs published (%)  |  whisker crosses 0 = gap explainable by input uncertainty")

# 1c. Implied MFU back-calculated from each published step time (validation/solvers.solve_for_mfu):
# what trainer efficiency our FLOP-per-step accounting would need to reproduce the paper. Ordered
# by FSDP shard degree so the decline -- the fingerprint of the unmodelled sharding penalty
# (penalty_para=1.0) -- is visible left to right. prime-rl (8-shard, MFU published at 0.385)
# returns ~its own input, which is the method's sanity check.
_IMPL = []
for lab, sc in sorted(CAL_SC, key=lambda kv: kv[1].train_hw.n_shard):
    pub = (sc.published or {}).get("t_step")
    d = solve_for_mfu(sc, pub) if pub else None
    if d and d["implied_mfu"] is not None:
        _IMPL.append((lab.replace("\n", " "), d["implied_mfu"], sc.train_hw.n_shard, d["flipped"]))
cal_mfu = grouped_bars([f"{lab}\n{ns}-shard" for lab, mfu, ns, fl in _IMPL], [("implied MFU", PAL[2])],
                       lambda cat, s: next(mfu for lab, mfu, ns, fl in _IMPL
                                           if f"{lab}\n{ns}-shard" == cat),
                       f"1c. Implied trainer MFU to match published step time (assumed {CAL_SC[0][1].train_hw.mfu:g})",
                       ylab="implied trainer MFU", scale="linear", labelfn=fmt_rate,
                       notefn=lambda cat, s: ("*" if next(fl for lab, mfu, ns, fl in _IMPL
                                                          if f"{lab}\n{ns}-shard" == cat) else ""))


# ------------------------------------------------------------------ 2. FEASIBILITY
# fcell/fbar: one evaluate() -> a heatmap cell or a bar value. Infeasible-in-window is NEVER
# dropped: cells show the full-stock wall-clock FLOOR (days to expend the FLOPs), bars annotate
# the dashed marker with that floor. no-fit = weights exceed a single node.
def fcell(cfg, window_s):
    ev = evaluate(cfg, window_s)
    if not ev["runnable"]:
        return ("nofit", "no-fit")
    if ev["feasible"]:
        return (ev["gpus"], _fn(ev["gpus"]))
    return ("infeas", _ft(ev["wall_s"]) + "!")      # ! = misses window; shown value = floor days


def fbar(cfg, window_s):
    ev = evaluate(cfg, window_s)
    return ev["gpus"] if (ev["runnable"] and ev["feasible"]) else None


def ffloor(cfg, window_s):
    ev = evaluate(cfg, window_s)
    return "no-fit" if not ev["runnable"] else (_ft(ev["wall_s"]) + "!")


MODEL_NAME = base.model.name
WIN = [30, 90, 180, 720]
GROK3_C_RL = 2.5e25
MT = model_targets()                     # {name: C_RL}, real published runs only (no thresholds)

# Chart categories carry the FLOP estimate on a second line: the model name alone doesn't tell a
# governance reader what scale of run it was, and that scale is the whole independent variable.
# (grouped_bars splits category labels on "\n"; heatmap row labels are single-line, so those get
# the same information space-separated.)
def _tlab(name):
    return f"{name}\n{MT[name]:.1e}"           # bar-chart category (2 lines)
def _hmlab(name):
    return f"{name} {MT[name]:.1e}"            # heatmap row label (1 line)
_LAB2NAME = {**{_tlab(n): n for n in MT}, **{_hmlab(n): n for n in MT}}
def _tcfg(label, **ov):
    """Config for whichever target a chart label (either label style) refers to."""
    return _with(base, target_c_rl=MT[_LAB2NAME[label]], **ov)

frontier = grouped_bars([_tlab(n) for n in MT], [(f"{d}d", PAL[i]) for i, d in enumerate(WIN)],
                        lambda cat, s: fbar(_tcfg(cat), int(s[:-1]) * 86400),
                        f"2a. Feasibility frontier -- min GPUs to finish, by RL compute budget and window "
                        f"({MODEL_NAME})",
                        ylab="min GPUs to finish in window",
                        annotfn=lambda cat, s: ffloor(_tcfg(cat), int(s[:-1]) * 86400))

MG = {n: smallest_fitting_gpu(m, GPN) for n, m in MODELS.items()}
MSHORT = {"Llama3-8B": "L3-8B", "Llama3-70B": "L3-70B", "Llama3-405B": "L3-405B",
          "Qwen3-30B-A3B": "Q3-30B/A3", "Qwen3-235B-A22B": "Q3-235B/A22", "Kimi-K3": "K3-2.4T"}
S2F = {v: k for k, v in MSHORT.items()}
def _mcfg(cat):
    n = S2F[cat]; return _with(base, model=MODELS[n], gpu=MG[n], target_c_rl=GROK3_C_RL) if MG[n] else None


def _msteps(cat, _series):
    """Training-step count written sideways inside each cross-model bar. Without it the chart is
    genuinely counterintuitive: SMALL models need MORE GPUs at fixed C_RL, which only makes sense
    once you can see they are running orders of magnitude more serial steps to spend the same
    FLOPs (small model -> less compute per step -> more steps -> deeper into the latency wall).

    Rotated inside a bar, the text length is bounded by the bar's HEIGHT, so svgkit shrinks the
    note's font to fit and drops it only if that would make it illegible -- which is what lets the
    " steps" unit ride along on the label instead of living in the title."""
    cfg = _mcfg(cat)
    return f"{_fn(GROK3_C_RL / per_step_flop(cfg))} steps" if cfg else ""


crossmodel = grouped_bars([MSHORT[n] for n in MODELS], [("90d", PAL[1]), ("180d", PAL[2])],
                          lambda cat, s: (fbar(_mcfg(cat), int(s[:-1]) * 86400) if _mcfg(cat) else None),
                          f"2d. Cross-model -- min GPUs at fixed C_RL={GROK3_C_RL:.1e} "
                          f"(smallest fitting node; K3 needs multi-node)",
                          ylab="min GPUs to finish in window", notefn=_msteps,
                          annotfn=lambda cat, s: (ffloor(_mcfg(cat), int(s[:-1]) * 86400) if _mcfg(cat) else "no-fit"))

GFAM = list(HIGH_END_GPUS)
gpu_hm = heatmap([_hmlab(n) for n in MT], GFAM,
                 lambda r, g: fcell(_tcfg(r, gpu=g), 180 * 86400),
                 "2e. GPU family x RL compute budget (180d) -- GPUs (feasible) or floor-days! (infeasible)",
                 rowaxis="RL compute budget (FLOP)", colaxis="GPU family (HBM bandwidth rises left to right)")

# 2f. Same heatmap as 2e but with each GPU's node sized down to a BIS export cap (node <= N
# H100-equiv of TPP and HBM), so it reads directly against 2e's uncapped node. Column labels carry
# the capped GPUs/node (the CCC-definition node size). High-FLOP/high-HBM parts get forced into
# small nodes (B300/GB300/MI355X -> 4 GPUs, HBM-bound), which worsens inference concurrency and
# raises the min cluster -- and can push a model off single-node inference (no-fit) outright.
# Mirrors feasibility.export_cap_table.
H100_EQUIV = 16
CAP_NODE = {g: node_gpus_tpp_cap(g, H100_EQUIV) for g in GFAM}
_CAPCOL = {f"{g} {CAP_NODE[g]}/nd": g for g in GFAM}   # column label -> GPU key
gpucap_hm = heatmap([_hmlab(n) for n in MT], list(_CAPCOL),
                    lambda r, c: fcell(_tcfg(r, gpu=_CAPCOL[c],
                                             gpus_per_node=CAP_NODE[_CAPCOL[c]]), 180 * 86400),
                    f"2f. Same, under a {H100_EQUIV}-H100 export cap: node shrunk to fit TPP+HBM "
                    f"(GPUs/node in column) -- compare to 2e",
                    rowaxis="RL compute budget (FLOP)",
                    colaxis=f"GPU family, capped GPUs/node under the {H100_EQUIV}-H100 TPP+HBM cap")

# --- 2f. REPLACES the old WAN x compression grid, which mostly restated A1's feasibility answer at
# 16 (bandwidth, compression) pairs. This asks the question a monitor or an evader actually asks:
# what LINK does the weight broadcast need so the network is never the bottleneck?
# Reported as the ACTUAL wire rate, which scales as 1/compression -- so the compression factor is
# named in the axis label, since the same run needs 50x less link at 50x compression.
BW_MODELS = [n for n in MODELS if MG[n]]


def _bwcfg(cat, tlabel):
    n = S2F[cat]
    return _with(base, model=MODELS[n], gpu=MG[n], target_c_rl=MT[_LAB2NAME[tlabel]])


bwfig = grouped_bars([MSHORT[n] for n in BW_MODELS],
                     [(_tlab(n), PAL[i]) for i, n in enumerate(MT)],
                     lambda cat, s: required_bcast_mbps(_bwcfg(cat, s), 180 * 86400)[0],
                     "2g. Weight-sync bandwidth required to avoid communication bottleneck"
                     "(Min GPUs for 2.5e25 in 180d)",
                     ylab="Link Mbps (Bandwidth x compression)",
                     labelfn=fmt_rate, annotfn=lambda cat, s: "unbounded")

# --- 2g. Total rollout batch vs min cluster. Replaces a B x n grid: at fixed total batch the two
# axes are near-substitutes in this model (both only cut serial steps; n is marginally cheaper via
# shared prefill), so the grid was largely redundant along its diagonals and the PRODUCT is the
# variable that actually moves the answer. B x n per point is set even-ish by batch_split().
def _bcfg(catlabel, tlabel):
    total = LABEL_BATCH[catlabel]
    b, n = batch_split(total)
    return _tcfg(tlabel, prompts_per_batch=b, responses_per_prompt=n)


BATCH_LABEL = {t: f"{_fn(t)}\n{batch_split(t)[0]}x{batch_split(t)[1]}" for t in BATCH_SWEEP}
LABEL_BATCH = {v: k for k, v in BATCH_LABEL.items()}
bg_fig = grouped_bars([BATCH_LABEL[t] for t in BATCH_SWEEP],
                      [(_tlab(n), PAL[i]) for i, n in enumerate(MT)],
                      lambda cat, s: fbar(_bcfg(cat, s), 180 * 86400),
                      f"2h. Batch size vs min cluster (2.5e24 in 180d). Large batches mean fewer serial steps",
                      ylab="min GPUs to finish in 180d",
                      annotfn=lambda cat, s: ffloor(_bcfg(cat, s), 180 * 86400))

# --- 2g. min GPUs to reach each target: baseline batch vs high-end batch (sizes read from the
# BASELINE_BATCH/HIGH_BATCH constants, not hardcoded, so this title can't go stale if those are
# retuned). Model targets only -- the batch size is an engineering choice a specific lab made for
# a specific run, so pairing it with governance reporting thresholds compared a decision with a
# regulation.
BC_WIN = [90, 180, 720]
def _bccfg(cat, cfgname):
    return _tcfg(cat, **(HIGH_BATCH if cfgname == "high" else BASELINE_BATCH))
_bc_lab = lambda b: f"{b['prompts_per_batch']}x{b['responses_per_prompt']}"
batchcompare = grouped_bars([_tlab(n) for n in MT],
                            [(f"{d}d base", PAL[0]) for d in BC_WIN] + [(f"{d}d high", PAL[1]) for d in BC_WIN],
                            lambda cat, s: (lambda d, cn: fbar(_bccfg(cat, cn), int(d[:-1]) * 86400))(*s.split()),
                            f"2i. Min GPUs per RL compute budget. Base batch ({_bc_lab(BASELINE_BATCH)}) "
                            f"vs Large batch ({_bc_lab(HIGH_BATCH)})",
                            ylab="min GPUs to finish in window",
                            annotfn=lambda cat, s: (lambda d, cn: ffloor(_bccfg(cat, cn), int(d[:-1]) * 86400))(*s.split()))

# evaluate(), not raw min_cluster(): always returns an operating point (min cluster if feasible,
# else the full-stock floor) so this doesn't break if 2.5e25/180d becomes infeasible under whatever
# FeasConfig defaults are currently live (WAN/batch/etc. get tuned directly in feasibility_core.py).
cfgR = _with(base, target_c_rl=2.5e25); solR = evaluate(cfgR, 180 * 86400)
N, base_t = solR["N"], solR["wall_s"]
LEV = {"WAN bandwidth": lambda c, k: _with(c, wan_mbps=c.wan_mbps * k),
       "compression": lambda c, k: _with(c, compression=c.compression * k),
       "HBM bandwidth": lambda c, k: _with(c, inf_bw_mult=c.inf_bw_mult * k),
       "HBM capacity": lambda c, k: _with(c, hbm_mult=c.hbm_mult * k),
       "inference FLOP": lambda c, k: _with(c, mfu_inf=c.mfu_inf * k),
       "training FLOP": lambda c, k: _with(c, mfu_train=c.mfu_train * k),
       "E[R] length": lambda c, k: _with(c, er=c.er * k), "node count": None}
torn = []
for name, fn in LEV.items():
    if name == "node count":
        lo = total_time(cfgR, max(1, int(solR["nt"] * .5)), max(1, int(solR["ni"] * .5)))[0]
        hi = total_time(cfgR, solR["nt"] * 2, solR["ni"] * 2)[0]
    else:
        lo = total_time(fn(cfgR, .5), *_split(fn(cfgR, .5), N))[0]
        hi = total_time(fn(cfgR, 2.), *_split(fn(cfgR, 2.), N))[0]
    torn.append((name, lo / base_t, hi / base_t, max(lo, hi) / min(lo, hi)))
torn.sort(key=lambda x: -x[3])
tfig = tornado(torn, f"2c. Sensitivity tornado -- wall-clock x-swing per lever "
                     f"({MODEL_NAME}, {GROK3_C_RL:.1e}, 180d op. point)")

# --- 2i. HBM bandwidth on its own axis rather than only a x0.5/x2 tornado bar. It earns the
# space twice over: it is the binding constraint in this rollout-bound regime (it beats every
# network lever in the tornado above), and memory bandwidth is the lever export controls actually
# move -- it is what the cut-down export parts sacrifice. So this is a directly policy-relevant
# elasticity, not just another sensitivity row.
hbmfig = grouped_bars([f"HBM x{k:g}" for k in HBM_MULTS],
                      [(_tlab(n), PAL[i]) for i, n in enumerate(MT)],
                      lambda cat, s: fbar(_tcfg(s, inf_bw_mult=base.inf_bw_mult * float(cat[5:])),
                                          180 * 86400),
                      "2j. HBM bandwidth vs min cluster (180d)",
                      ylab="min GPUs to finish in 180d",
                      annotfn=lambda cat, s: ffloor(_tcfg(s, inf_bw_mult=base.inf_bw_mult * float(cat[5:])),
                                                    180 * 86400))

bars = sorted(POLICY_THRESHOLDS.items(), key=lambda kv: kv[1])
pol_hm = heatmap([_hmlab(n) for n in MT], [f"{n}\n{v:.0e}" for n, v in bars],
                 lambda r, c: (1 if MT[_LAB2NAME[r]] >= float(c.split("\n")[1]) else 0,
                               "OVER" if MT[_LAB2NAME[r]] >= float(c.split("\n")[1]) else "under"),
                 "2k. RL compute budget vs FLOP-threshold",
                 kind="bin", rh=26, rowaxis="RL compute budget (FLOP)",
                 colaxis="governance reporting threshold (total-run FLOP)")

# --- 2b. inverse of 2a: given a FIXED GPU stock (swept, not the target), what's the MAXIMUM RL
# compute achievable at the optimal split. Mirrors feasibility.py's max_compute_by_stock table
# (same optimal_split/per_step_flop calls) so the CLI table and this chart always agree.
STOCK_SWEEP = (16, 128, 1024, 10_000, 100_000, 1_000_000)
STOCK_LABEL = {s: f"{_fn(s)} GPUs" for s in STOCK_SWEEP}
LABEL_STOCK = {v: k for k, v in STOCK_LABEL.items()}


def max_op_at_stock(cfg, stock):
    """Same computation as feasibility.py's max_compute_operating_point(): the window-INDEPENDENT
    operating point at a given GPU stock -- split, T_step (+ T_update/T_rollout+verify/T_broadcast
    breakdown), the ACHIEVED overall MFU of the whole allocated cluster (both conventions),
    bottleneck, per-step FLOP. None if the model doesn't fit a single node. Computed once per
    stock and reused for every window (cheap arithmetic after).

    MFU here is GENUINELY DERIVED, not an echoed input: r.mfu_model/r.mfu_hw (model.py) reduce
    algebraically to (mfu_train input) x a fixed constant -- identical at every stock size since
    they only look at the trainer's own t_update, never whether either pool idles waiting on the
    other. This instead divides total actual FLOPs (train+rollout) by T_step x the COMBINED peak
    FLOPs/s of every allocated GPU, so an idling pool correctly drags it down."""
    # Cap the node to the fleet (shared helper, same rule as feasibility.max_compute_operating_point
    # so the CLI table and this chart still agree): a node can't exceed the stock and disaggregation
    # needs >=2 nodes, so at small stock the configured node shrinks -- otherwise a stock < 2 nodes
    # silently over-allocated whole configured-size nodes (stock=16 with a 720-node -> 1,440 GPUs).
    eff = effective_node_gpus(cfg.gpus_per_node, stock)
    c = _with(cfg, stock_gpus=stock, gpus_per_node=eff)
    n_total = stock // eff
    sol = optimal_split(c, n_total)
    if sol is None or not math.isfinite(sol[0]) or sol[3] is None:
        return None
    tt, nt, ni, r, frac = sol
    # BUG (fixed): `tt` is total_time() for cfg's OWN target_c_rl at this split, NOT the per-step
    # time -- using it here made every T_step off by n_steps(cfg) (see feasibility.py's identical
    # fix for the full explanation). The real per-step time is r.t_step.
    pf = per_step_flop(c)
    s = r.scenario
    train_peak = s.train_hw.n_nodes * s.train_hw.flops
    inf_peak = s.inf_hw.n_nodes * s.inf_hw.flops
    total_peak = train_peak + inf_peak
    model_flops_step = (3.0 / (3 + s.algo.recomp_act + s.algo.recomp_old)) * r.tc.c_update + r.ro.c_rollout_total
    return dict(n_total=n_total, gpus=n_total * eff, nt=nt, ni=ni, frac=frac,
                t_step=r.t_step, t_update=r.t_update, t_rollout=r.stages["rollout+verify"],
                t_broadcast=r.t_bc, bottleneck=r.bottleneck,
                mfu_model=model_flops_step / (r.t_step * total_peak),
                mfu_hw=pf / (r.t_step * total_peak),
                per_step_flop=pf)


def max_c_rl_at_stock(cfg, stock, window_s):
    """Bar-chart value: max achievable C_RL at a stock/window. Thin wrapper over
    max_op_at_stock() so the chart and the appendix table below always agree."""
    op = max_op_at_stock(cfg, stock)
    return None if op is None else op["per_step_flop"] * window_s / op["t_step"]


stockfig = grouped_bars([STOCK_LABEL[s] for s in STOCK_SWEEP], [(f"{d}d", PAL[i]) for i, d in enumerate(WIN)],
                        lambda cat, s: max_c_rl_at_stock(base, LABEL_STOCK[cat], int(s[:-1]) * 86400),
                        f"2b. Max achievable RL compute vs GPU stock -- optimal split per "
                        f"stock size ({MODEL_NAME})",
                        ylab="max achievable RL compute (FLOP)",
                        annotfn=lambda cat, s: "no-fit", labelfn=fmt_sci,
                        # Same axis/units as the bars (RL compute FLOP) -- gives a reader a sense
                        # of scale against the real published-model targets used elsewhere in the
                        # dashboard (cross-model, GPU-family, policy-threshold panels), without
                        # needing to flip to another chart to look the numbers up.
                        hlines=[(_hmlab(n), c) for n, c in sorted(MT.items(), key=lambda kv: kv[1])])

# ============================================================================================
# 2n-2q -- DECENTRALISED vs CENTRALISED-DATACENTRE (the core framing; its own sub-section below).
# The serial step-latency bottlenecks of distributed RL are (i) broadcasting fresh weights
# trainer->inference over the WAN, (ii) the number of inference "waves", set by how much aggregate
# HBM (capacity for concurrency + bandwidth for weight load) a node pools, and (iii) MoE weight
# sparsity. A centralised datacentre relaxes exactly these: a large NVLink domain fuses many GPUs
# into ONE node (pooled bandwidth/memory -> fewer, fatter inference waves) and an intra-DC fabric
# gives ~terabit trainer->inference links. Both scenarios below hold everything else at `base`
# (Llama-405B, H100, 180d) and change ONLY those centralisation levers, so the gap between paired
# bars IS the decentralisation penalty. RETUNE FREELY: edit the dicts (keys are FeasConfig fields),
# nothing else. 405B is dense, so bottleneck (iii) is out of scope here by design. Four charts:
# a primary pair at DC_COMP (16x) sync -- 2n min-GPUs-by-budget, 2o max-compute-by-stock -- and a
# companion pair at lighter 4x sync -- 2p, 2q -- where the WAN/broadcast wall shows (at 16x the
# broadcast is small enough that the WAN is inert and the penalty is purely inference-wave/node-
# size; drop to 4x and the 1 Gbps decentralised link becomes the binding constraint).
DC_COMP = 16.0                                                        # primary weight-sync compression (4-bit int4 x 4x sparsify/DiLoCo)
DEC = dict(gpus_per_node=16, wan_mbps=1_000, compression=DC_COMP)      # small islands, 1 Gbps internet
CEN = dict(gpus_per_node=72, wan_mbps=1_000_000, compression=DC_COMP)  # NVL72 domain, 1 Tbps intra-DC fabric
DEC_INT4 = {**DEC, "compression": 4.0}                                 # companion pair: lighter 4x weight sync,
CEN_INT4 = {**CEN, "compression": 4.0}                                 #   which lets the WAN/broadcast wall show
DC_WIN_S = 180 * 86400
DC_SERIES = [("decentralised", PAL[3]), ("centralised", PAL[2])]


def _dccfg(scen, crl, int4=False):
    ov = ((CEN_INT4 if int4 else CEN) if scen == "centralised" else (DEC_INT4 if int4 else DEC))
    return _with(base, target_c_rl=crl, **ov)


# "find the best node size": sweep the centralised node at the reference target, report the true
# optimum against the real NVL72 (72) we use. Bigger nodes cut inference waves but coarsen the
# allocation granularity (>=2 nodes), so min GPUs is U-shaped in node size -- there is a genuine best.
def _cen_node_sweep(crl=GROK3_C_RL, nodes=(16, 32, 48, 72, 96, 144, 256, 576)):
    out = []
    for nd in nodes:
        ev = evaluate(_with(base, target_c_rl=crl, gpus_per_node=nd,
                            wan_mbps=CEN["wan_mbps"], compression=CEN["compression"]), DC_WIN_S)
        out.append((nd, ev["gpus"] if ev["feasible"] else None))
    feas = [(nd, g) for nd, g in out if g]
    return (min(feas, key=lambda x: x[1]) if feas else None), out


_CEN_BEST, _CEN_SWEEP = _cen_node_sweep()
_cen_at72 = next((g for nd, g in _CEN_SWEEP if nd == CEN["gpus_per_node"]), None)
_cen_note = (f"NVL72 uses {_cen_at72:,} GPUs; swept optimum node={_CEN_BEST[0]} -> {_CEN_BEST[1]:,} "
             f"({100*(_cen_at72/_CEN_BEST[1]-1):+.1f}% vs best)"
             if (_CEN_BEST and _cen_at72) else "node sweep n/a")

dc_frontier = grouped_bars([_tlab(n) for n in MT], DC_SERIES,
                           lambda cat, s: fbar(_dccfg(s, MT[_LAB2NAME[cat]]), DC_WIN_S),
                           f"2n. Decentralised vs centralised -- min GPUs to finish (Llama-405B, 180d, {DC_COMP:g}x sync). cf. 2a",
                           ylab="min GPUs to finish in 180d",
                           annotfn=lambda cat, s: ffloor(_dccfg(s, MT[_LAB2NAME[cat]]), DC_WIN_S))

dc_stockfig = grouped_bars([STOCK_LABEL[s] for s in STOCK_SWEEP], DC_SERIES,
                           lambda cat, s: max_c_rl_at_stock(_with(base, **(CEN if s == "centralised" else DEC)),
                                                            LABEL_STOCK[cat], DC_WIN_S),
                           f"2o. Decentralised vs centralised -- max achievable RL compute vs GPU stock "
                           f"(Llama-405B, 180d, {DC_COMP:g}x sync). cf. 2b",
                           ylab="max achievable RL compute (FLOP)",
                           annotfn=lambda cat, s: "no-fit", labelfn=fmt_sci,
                           hlines=[(_hmlab(n), c) for n, c in sorted(MT.items(), key=lambda kv: kv[1])])

dc_frontier_int4 = grouped_bars([_tlab(n) for n in MT], DC_SERIES,
                                lambda cat, s: fbar(_dccfg(s, MT[_LAB2NAME[cat]], int4=True), DC_WIN_S),
                                f"2p. Same, min GPUs at lighter 4x sync -- exposes the WAN/broadcast wall "
                                f"(decentralised flips from wave- to broadcast-bound at frontier scale). cf. 2n",
                                ylab="min GPUs to finish in 180d",
                                annotfn=lambda cat, s: ffloor(_dccfg(s, MT[_LAB2NAME[cat]], int4=True), DC_WIN_S))

dc_stockfig_int4 = grouped_bars([STOCK_LABEL[s] for s in STOCK_SWEEP], DC_SERIES,
                                lambda cat, s: max_c_rl_at_stock(_with(base, **(CEN_INT4 if s == "centralised" else DEC_INT4)),
                                                                 LABEL_STOCK[cat], DC_WIN_S),
                                f"2q. Same, max compute vs stock at 4x sync -- decentralised compute PLATEAUS as the "
                                f"broadcast wall caps it; centralised (terabit) keeps scaling. cf. 2o",
                                ylab="max achievable RL compute (FLOP)",
                                annotfn=lambda cat, s: "no-fit", labelfn=fmt_sci,
                                hlines=[(_hmlab(n), c) for n, c in sorted(MT.items(), key=lambda kv: kv[1])])

# --- 2k: effect of mean response length E[R] on feasibility, grouped by RL compute budget (not
# swept at one fixed target) -- mirrors feasibility.py's er_table(). Standardised on 180d (matching
# the other single-window panels) since this already has two swept axes; a third would be unreadable.
ER_POINTS = [(1_000, "1k"), (10_000, "10k"), (50_000, "50k"), (100_000, "100k"), (500_000, "500k")]
ER_LABEL_TOK = {lbl: er for er, lbl in ER_POINTS}
er_gpu_fig = grouped_bars([_tlab(n) for n in MT], [(lbl, PAL[i]) for i, (_, lbl) in enumerate(ER_POINTS)],
                          lambda cat, s: fbar(_tcfg(cat, er=ER_LABEL_TOK[s]), 180 * 86400),
                          "2l. Response length vs min cluster, by RL compute budget (180d)",
                          ylab="min GPUs to finish in 180d",
                          annotfn=lambda cat, s: ffloor(_tcfg(cat, er=ER_LABEL_TOK[s]), 180 * 86400))

# 2l. FULLY 4-bit inference side on B300, vs BF16 inference. 4-bit touches every inference term:
#   compute        inf_flop_mult=4     FP4 tensor cores, B300 2.25 -> 9 PFLOP/s dense
#   weights        b_weights_inf=0.5   4-bit (vs BF16 2 B/param): 4x less decode weight-load HBM
#                                      traffic, 4x less HBM weight storage (-> more KV room ->
#                                      concurrency), and 4x smaller raw weight-broadcast volume
#   KV cache       b_kv=0.5            4x smaller KV footprint + traffic
# No inf_bw_mult hack: physical HBM bandwidth is unchanged, 4-bit simply moves less data, which
# b_weights_inf/b_kv model directly. Trainer stays BF16 (4-bit training is not standard).
# Small-multiples like 1a: each target its own linear axis, since min-GPUs spans 32 to thousands.
Q4_GPU = "B300"
def _q4pair(name):
    b = _with(base, gpu=Q4_GPU, target_c_rl=MT[name])
    q = _with(base, gpu=Q4_GPU, target_c_rl=MT[name],
              inf_flop_mult=4, b_weights_inf=0.5, b_kv=0.5)
    gb = fbar(b, 180 * 86400)
    gq = fbar(q, 180 * 86400)
    lab = name + ("" if (gb or gq) else "\n(infeasible)")
    return (lab, gb, gq)
quant_fig = small_multiples([_q4pair(n) for n in MT],
                            f"2m. Fully 4-bit inference on {Q4_GPU} (compute + weights + KV) vs BF16 "
                            f"-- min GPUs, 180d",
                            ylab="min GPUs to finish in 180d",
                            series=(("BF16 inference", PAL[3]), ("4-bit inference", PAL[2])))

# Power-envelope cap (appendix table, built in the BODY below): a per-SITE power budget of 1 MW /
# 5 MW bounds the TOTAL GPUs at a site to floor(MW / board_W). Distinct from the CCC node cap (2f)
# -- that caps the NODE by TPP/HBM; this caps the SITE by power, i.e. how many GPUs you can run
# under a detectability tier. per_pow[gpu] = (board_W, gpus@1MW, gpus@5MW).
POWER_CAPS_MW = (1, 5)
per_pow = {g: (GPUS[g]["watts"], *[node_gpus_power_cap(g, mw * 1000) for mw in POWER_CAPS_MW])
           for g in GFAM}


# ------------------------------------------------------------------ 3. LENGTHS
def meanlen(model, dom):
    """Mean measured E[R] for one (model, domain), or None so the bar renders as missing."""
    return lengths_io.mean_length(f"{LEN}/{model}_{dom}_lengths.json")


DOM = ["math", "code", "logic", "science"]
DMODELS = [("deepseek-v4-flash", "V4-flash"), ("deepseek-v4-pro-0813", "V4-pro"), ("glm45air", "GLM-4.5-Air")]
domfig = grouped_bars([lab for _, lab in DMODELS], [(d, PAL[i]) for i, d in enumerate(DOM)],
                      lambda cat, s: meanlen(dict((lab, mdl) for mdl, lab in DMODELS)[cat], s),
                      "3a. Domain multiplier -- measured E[R] by task. Code > math > logic/science",
                      ylab="mean response length E[R] (tokens)")

QW = [("qwen3-8b", "8B"), ("qwen3-14b", "14B"), ("qwen3-30b-a3b", "30B-A3B"),
      ("qwen3-32b", "32B*"), ("qwen3-235b-a22b", "235B-A22B")]
sizefig = grouped_bars([lab for _, lab in QW], [("math", PAL[0]), ("code", PAL[2])],
                       lambda cat, s: meanlen(dict((lab, mdl) for mdl, lab in QW)[cat], s),
                       "3b. Size scaling -- Qwen3 family E[R] (math/code). *32B ~40% censored = lower bound",
                       ylab="mean response length E[R] (tokens)")


AG = {n: lengths_io.agentic_means(f"{LEN}/agentic_{n}.json")
      for n in ("tau2", "swesmith", "nebius", "ccbench")}
AGN = {"tau2": "tau2 (tool)", "swesmith": "SWE-smith", "nebius": "SWE-rebench", "ccbench": "CC-Bench"}
agorder = sorted(AG, key=lambda n: AG[n].get("total") or 0)
agfig = grouped_bars([AGN[n] for n in agorder],
                     [("effective ctx", PAL[3]), ("generated", PAL[0]), ("tool out", PAL[4])],
                     lambda cat, s: (lambda d: d.get({"effective ctx": "total", "generated": "gen", "tool out": "tool"}[s]))(
                         AG[[n for n in agorder if AGN[n] == cat][0]]),
                     "3c. Agentic effective context -- single-turn E[R] is ~5-40k for contrast",
                     ylab="tokens (log)")


# ------------------------------------------------------------------ A1 table (all targets, via evaluate)
trows = []
for n, tc in all_targets().items():
    for d in WIN:
        ev = evaluate(_with(base, target_c_rl=tc), d * 86400)
        if not ev["runnable"]:
            trows.append(f"<tr><td>{n}</td><td>{tc:.1e}</td><td>{d}d</td>"
                         f"<td colspan='12' class='inf'>no single-node fit</td></tr>")
            continue
        feas = "yes" if ev["feasible"] else "NO*"
        gpus = f"{ev['gpus']:,}" + ("" if ev["feasible"] else "*")
        cost = ("$" + _fn(ev["cost"])) if ev["feasible"] else "&mdash;"
        r = ev["r"]
        trows.append(f"<tr><td>{n}</td><td>{tc:.1e}</td><td>{d}d</td><td>{feas}</td><td>{gpus}</td>"
                     f"<td>{ev['pct_stock']:.2f}%</td><td>{_tr(ev['frac'])}</td><td>{_fn(ev['steps'])}</td>"
                     f"<td>{_ft(ev['wall_s'])}</td><td>{cost}</td><td>{ev['bottleneck']}</td>"
                     f"<td>{_ft(r.t_step)}</td><td>{_ft(r.t_update)}</td>"
                     f"<td>{_ft(r.stages['rollout+verify'])}</td><td>{_ft(r.t_bc)}</td></tr>")


# ------------------------------------------------------------------ Appendix: MC uncertainty
# The provenance behind chart 1a's whiskers. "point" uses each preset's data-grounded per-config
# trainer MFU (not a flat 0.40), so it and the MC mode agree; E[R] band is ±ER_PCT (prime-rl pinned).
mc_trows = []
for lab, mdl, pub, err in CAL:
    mc = MC[lab]
    erb = "pinned" if mc["er_pct"] is None else f"&plusmn;{mc['er_pct']:.0%}"
    verdict = ("&mdash;" if mc["inside"] is None else
               ("<b>yes</b>" if mc["inside"] else "no"))
    mc_trows.append(
        f"<tr><td>{lab.replace(chr(10), ' ')}</td><td>{pub:,.0f}</td><td>{mdl:,.0f}</td>"
        f"<td>{mc['mean']:,.0f}</td><td>{mc['p5']:,.0f} &ndash; {mc['p95']:,.0f}</td>"
        f"<td>{mc['spread']:.2f}&times;</td><td>{erb}</td><td>{mc['bottleneck']}</td>"
        f"<td>{verdict}</td><td style='text-align:left'>{esc(mc['basis'])}</td></tr>")


# ------------------------------------------------------------------ Appendix: E[R] sensitivity
# Full per-(target, E[R]) detail behind chart 2j, at the same fixed 180d window -- mirrors
# feasibility.py's er_table() column-for-column (same 15 columns as the A1 appendix above, with
# E[R] swapped in for window) so the CLI table and this page always agree.
er_trows = []
for n, tc in all_targets().items():
    for _, lbl in ER_POINTS:
        ev = evaluate(_with(base, target_c_rl=tc, er=ER_LABEL_TOK[lbl]), 180 * 86400)
        if not ev["runnable"]:
            er_trows.append(f"<tr><td>{n}</td><td>{tc:.1e}</td><td>{lbl}</td>"
                            f"<td colspan='12' class='inf'>no single-node fit</td></tr>")
            continue
        feas = "yes" if ev["feasible"] else "NO*"
        gpus = f"{ev['gpus']:,}" + ("" if ev["feasible"] else "*")
        cost = ("$" + _fn(ev["cost"])) if ev["feasible"] else "&mdash;"
        r = ev["r"]
        er_trows.append(f"<tr><td>{n}</td><td>{tc:.1e}</td><td>{lbl}</td><td>{feas}</td><td>{gpus}</td>"
                        f"<td>{ev['pct_stock']:.2f}%</td><td>{_tr(ev['frac'])}</td><td>{_fn(ev['steps'])}</td>"
                        f"<td>{_ft(ev['wall_s'])}</td><td>{cost}</td><td>{ev['bottleneck']}</td>"
                        f"<td>{_ft(r.t_step)}</td><td>{_ft(r.t_update)}</td>"
                        f"<td>{_ft(r.stages['rollout+verify'])}</td><td>{_ft(r.t_bc)}</td></tr>")


# ------------------------------------------------------------------ Appendix: max-compute by stock
# Same operating point as chart 2i, but with the metrics a bar can't show: split, full per-stage
# time breakdown, MFU, bottleneck, training steps -- one row per (stock, window), mirroring
# feasibility.py's max_compute_by_stock() text table so the CLI and this page always agree.
stock_trows = []
for s in STOCK_SWEEP:
    op = max_op_at_stock(base, s)
    if op is None:
        stock_trows.append(f"<tr><td>{STOCK_LABEL[s]}</td>"
                           f"<td colspan='10' class='inf'>no-fit: weights exceed a single node</td></tr>")
        continue
    for d in WIN:
        steps = d * 86400 / op["t_step"]
        c_rl = op["per_step_flop"] * steps
        stock_trows.append(f"<tr><td>{STOCK_LABEL[s]}</td><td>{d}d</td><td>{_tr(op['frac'])}</td>"
                           f"<td>{_ft(op['t_step'])}</td><td>{_ft(op['t_update'])}</td>"
                           f"<td>{_ft(op['t_rollout'])}</td><td>{_ft(op['t_broadcast'])}</td>"
                           f"<td>{op['mfu_hw']*100:.0f}%</td><td>{op['bottleneck']}</td>"
                           f"<td>{_fn(steps)}</td><td>{c_rl:.2e}</td></tr>")

config_block = "\n".join(esc(line) for line in config_lines(base))

BODY = f"""<h1>Distributed-RL simulator &mdash; full results dashboard</h1>
<p class="sub">Recomputed live from the model + feasibility code; lengths read from results/lengths/. Bars log-scaled unless noted. Regenerate: <code>python results/plot_dashboard.py</code></p>

<h2>Configuration</h2>
<p class="sub">The FeasConfig behind every chart/table in section 2 below (section 1 uses each published preset's own logged config, in validation/presets.py; section 3 reads external measurement files).</p>
<div class="panel"><pre style="white-space:pre-wrap;font-size:12px;margin:0;font-family:inherit">{config_block}</pre></div>

<h2>1 &middot; Model validation (single-turn analytic core vs published runs)</h2>
<p class="sub">prime-rl is the clean self-pinned anchor (~0%). AReaL residuals are the known FSDP/MFU efficiency gap at high shard; INTELLECT-3 is MoE/multi-turn. See project memo.</p>
{panel(cal_bars)}{panel(cal_err)}{panel(cal_mfu)}

<h2>2 &middot; Feasibility &amp; sensitivity ({MODEL_NAME}, native MoE decode)</h2>
<p class="sub">Infeasible-in-window is never dropped: heatmap cells and dashed bars show the full-stock wall-clock FLOOR (days to expend the FLOPs, marked "!"); no-fit = weights exceed one node.</p>
{panel(frontier)}{panel(stockfig)}{panel(tfig)}{panel(crossmodel)}{panel(gpu_hm)}{panel(gpucap_hm)}{panel(bwfig)}{panel(bg_fig)}{panel(batchcompare)}{panel(hbmfig)}{panel(pol_hm)}{panel(er_gpu_fig)}{panel(quant_fig)}

<h3>2.1 &middot; Decentralised vs centralised datacentre &mdash; the serial-latency penalty (Llama-405B, 180d)</h3>
<p class="sub">The downside of distributed RL is serial step latency: (i) broadcasting fresh weights trainer&rarr;inference over the WAN, and (ii) the number of inference "waves", set by how much aggregate HBM (concurrency + weight-load bandwidth) a node pools. A centralised datacentre relaxes both &mdash; a large NVLink domain fuses GPUs into one fat node ({CEN['gpus_per_node']}/node here, the GB200 NVL72 rack) and an intra-DC fabric gives ~terabit trainer&rarr;inference links ({int(CEN['wan_mbps']/1000):,} Gbps) &mdash; vs small {DEC['gpus_per_node']}-GPU islands on {int(DEC['wan_mbps']):,} Mbps internet. Everything else is held at <code>base</code>, so the gap between paired bars is the decentralisation penalty. <b>Best centralised node size:</b> {_cen_note}. <b>Primary pair (2n, 2o) at {DC_COMP:g}&times; sync:</b> the broadcast is small enough that the WAN is inert, so the penalty is purely the inference-wave/node-size effect &mdash; ~2&times; the GPUs at frontier scale (though at small budgets decentralised is actually <i>cheaper</i>: centralised's 2&times;NVL72 floor is 144 GPUs). <b>Companion pair (2p, 2q) at lighter 4&times; sync:</b> the broadcast term grows 4&times; and the 1&nbsp;Gbps decentralised link becomes binding &mdash; decentralised flips from wave- to broadcast-bound (2p) and its achievable compute <i>plateaus</i> as the broadcast wall caps it, while centralised (terabit) keeps scaling (2q).</p>
{panel(dc_frontier)}{panel(dc_stockfig)}{panel(dc_frontier_int4)}{panel(dc_stockfig_int4)}

<h2>3 &middot; Response-length campaign (measured E[R])</h2>
<p class="sub">Feeds the runtime model's per-(model,task) E[R]. Agentic effective-context is the multi-turn regime the single-turn model doesn't yet represent.</p>
{panel(domfig)}{panel(sizefig)}{panel(agfig)}

<h2>Appendix &middot; Monte-Carlo predictive intervals (chart 1a whiskers)</h2>
<p class="sub">n={MC_DRAWS:,} draws/anchor, seeded so the page is reproducible. Priors: trainer MFU ~ per-config Triangular whose <b>mode is each preset's own data-grounded trainer MFU</b> (= 0.85&times; the Ultra-Scale Playbook / huggingface-nanotron node-matched RL-realistic best MFU; 0.85 corroborated independently by prime-rl and AReaL-7B) &mdash; so the deterministic "point" and the prior mode are the same number, and the point already reflects the calibrated MFU rather than a flat 0.40. Anchored on nanotron, not the step-time-implied MFU (which would be circular), so it sits above the implied MFU for high-shard anchors and the gap is the FSDP residual. Inference MFU ~ U{MFU_INF_RANGE}, sampled independently (physically they correlate, so this errs wide &mdash; the conservative direction). E[R] ~ Triangular &plusmn;{int(ER_PCT*100)}% about the measured value (a simple "the proxy measurement could be this far off the run's true E[R]" band; prime-rl pinned, its E[R] inverse-solved from its own throughput). Within-run length spread already enters the model exactly via E[R&sup2;]=E[R]&sup2;(1+cv&sup2;) (full second moment, tail included &mdash; not a 1-SD slice), so it is not resampled. <b>"explained?"</b> = does the published step time fall inside the 90% interval, i.e. can input uncertainty alone account for the gap? A <b>p95/p5 of ~1.00</b> means the anchor is broadcast-bound: its step time is set by weight volume and link speed, which neither MFU nor E[R] touches.</p>
<div class="panel"><table>
<tr><th>anchor</th><th>published</th><th>point</th><th>MC mean</th><th>90% interval</th><th>p95/p5</th><th>E[R] band</th><th>bottleneck</th><th>explained?</th><th style="text-align:left">E[R] provenance</th></tr>
{''.join(mc_trows)}
</table></div>

<h2>Appendix &middot; Node/site caps by GPU family</h2>
<p class="sub">Two independent physical caps on how a site can be built. <b>Export node cap</b> (BIS CCC): GPUs/node so the node stays under {H100_EQUIV} H100-equivalents of both TPP (compute = flops&times;16&nbsp;bit) and HBM (1,280&nbsp;GB) &mdash; whichever binds; this is the node size used in chart 2f. <b>Power envelope</b>: GPUs a site can run under a 1&nbsp;MW / 5&nbsp;MW draw (floor(MW / board&nbsp;W)) &mdash; a detectability-tier / site cap on total GPUs, separate from the node cap.</p>
<div class="panel"><table>
<tr><th>GPU</th><th>dense BF16 TFLOP/s</th><th>TPP/chip (TFLOP-bit/s)</th><th>HBM GB</th><th>board W</th><th>export node cap (GPUs/node)</th><th>GPUs @1MW</th><th>GPUs @5MW</th></tr>
{''.join(f"<tr><td>{g}</td><td>{GPUS[g]['flops']/1e12:.0f}</td><td>{tpp_per_chip(g)/1e12:,.0f}</td><td>{GPUS[g]['hbm_gb']}</td><td>{GPUS[g]['watts']}</td><td>{CAP_NODE[g]}</td><td>{per_pow[g][1]:,}</td><td>{per_pow[g][2]:,}</td></tr>" for g in GFAM)}
</table></div>

<h2>Appendix &middot; A1 sweep table</h2>
<div class="panel"><table>
<tr><th>target</th><th>C_RL</th><th>window</th><th>feas</th><th>GPUs</th><th>%stock</th><th>tr:inf</th><th>steps</th><th>wall</th><th>cost</th><th>bottleneck</th><th>T_step</th><th>T_update</th><th>T_rollout</th><th>T_bcast</th></tr>
{''.join(trows)}
</table></div>

<h2>Appendix &middot; E[R] sensitivity (by compute budget)</h2>
<p class="sub">Full per-(compute budget, response length) detail behind chart 2k, at the same fixed 180d window -- same 15 columns as the A1 appendix above with E[R] swapped in for window.</p>
<div class="panel"><table>
<tr><th>target</th><th>C_RL</th><th>E[R]</th><th>feas</th><th>GPUs</th><th>%stock</th><th>tr:inf</th><th>steps</th><th>wall</th><th>cost</th><th>bottleneck</th><th>T_step</th><th>T_update</th><th>T_rollout</th><th>T_bcast</th></tr>
{''.join(er_trows)}
</table></div>

<h2>Appendix &middot; Max achievable RL compute (by stock)</h2>
<p class="sub">Same operating point as chart 2b (one row group per stock size), with the metrics a bar can't show: split, full per-stage time breakdown (T_step total / T_update / T_rollout+verify / T_broadcast), ACHIEVED overall-cluster MFU (hw-FLOPs/8N convention -- matches published RL-training MFU figures, e.g. prime-rl's 38%; divides actual FLOPs done by T_step &times; combined train+inference peak FLOPs/s, so an idling pool drags it down; NOT an echo of the mfu_train/mfu_inf config inputs), bottleneck, training steps.</p>
<div class="panel"><table>
<tr><th>stock</th><th>window</th><th>tr:inf</th><th>T_step</th><th>T_update</th><th>T_rollout</th><th>T_bcast</th><th>MFU</th><th>bottleneck</th><th>steps</th><th>max C_RL</th></tr>
{''.join(stock_trows)}
</table></div>
"""
DOC = page("Distributed-RL: full results dashboard", BODY)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(DOC)
print(f"wrote {OUT}  ({DOC.count('<svg')} charts, "
      f"{len(mc_trows) + len(trows) + len(er_trows) + len(stock_trows)} table rows)")
