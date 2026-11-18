"""
Tables: the per-run detail a chart can't carry. Each is a DataFrame of display strings, so the dashboard
shows it and the CLI writes it as CSV unchanged.
"""
import math

import pandas as pd

from feasibility import (FACILITY_PUE, GPUS, HIGH_END_GPUS, MODELS, min_wall_full_stock,
                         model_targets, n_steps, node_gpus_power_cap, node_gpus_tpp_cap, sites_power,
                         smallest_fitting_gpu, tpp_per_chip, with_cfg)
from validation.uncertainty import ER_PCT, MC_DRAWS, MFU_INF_RANGE
from ..scenarios import BATCH_MODES, TARGETS, scenario
from ..spec import Table, fmt_count, fmt_duration, fmt_ratio, register
from ..sweep import max_compute, min_gpus
from .feasibility import ER_POINTS, H100_EQUIV, STOCKS, WINDOWS
from .lengths import CENSOR_WARN, DOMAIN_MODELS, DOMAINS, QWEN, _stats
from .validation import anchors


def _on(opts):
    """' (Llama-3.1-405B on H100)': the main config a table was run on, for its title."""
    cfg = scenario(opts)
    return f" ({cfg.model.name} on {cfg.gpu})"


def _gpu_cols(df, opts):
    """Name GPU-count columns after the configured GPU: 'GPUs' -> 'H100s'."""
    g = scenario(opts).gpu
    return df.rename(columns={"GPUs": f"{g}s", "GPUs used": f"{g}s used"})


def _run_cols(ev):
    """The shared per-run columns of the A1 and E[R] tables."""
    if ev["status"] == "nofit":
        return {"feasible": "no-fit"}
    feas = ev["status"] == "ok"
    return {"feasible": "yes" if feas else "NO*", "GPUs": f"{ev['gpus']:,}" + ("" if feas else "*"),
            "% stock": f"{ev['pct_stock']:.2f}%", "tr:inf": fmt_ratio(ev["train_frac"]),
            "steps": fmt_count(ev["steps"]), "wall": fmt_duration(ev["wall_d"] * 86400),
            "cost": "$" + fmt_count(ev["cost"]) if feas else "-", "bottleneck": ev["bottleneck"],
            "T_step": fmt_duration(ev["t_step"]), "T_update": fmt_duration(ev["t_update"]),
            "T_rollout": fmt_duration(ev["t_rollout"]), "T_bcast": fmt_duration(ev["t_bc"])}


@register("table_runs", "A")
def table_runs(opts):
    rows = []
    for t, c in TARGETS.items():
        cfg = scenario(opts, target=t)
        for w in WINDOWS:
            rows.append({"target": t, "C_RL": f"{c:.1e}",
                         "batch": fmt_count(cfg.prompts_per_batch * cfg.responses_per_prompt),
                         "window": f"{w} d", **_run_cols(min_gpus(cfg, w))})
    return Table("A1", "Every target x window: the minimum cluster or the full-stock floor" + _on(opts),
                 _gpu_cols(pd.DataFrame(rows).fillna("").set_index("target"), opts),
                 caption="* misses the window: GPUs and wall-clock are the full stock's fastest run. "
                         + BATCH_MODES[opts.batch_mode])


@register("table_response_length", "A")
def table_response_length(opts):
    rows = [{"target": t, "C_RL": f"{c:.1e}", "E[R]": fmt_count(er),
             **_run_cols(min_gpus(scenario(opts, target=t, er=er), 180))}
            for t, c in TARGETS.items() for er in ER_POINTS]
    return Table("A2", "Response length sensitivity, every target (180 days)" + _on(opts),
                 _gpu_cols(pd.DataFrame(rows).fillna("").set_index("target"), opts), caption="The detail behind chart 2l.")


@register("table_max_compute", "A")
def table_max_compute(opts):
    rows = []
    for s in STOCKS:
        for w in WINDOWS:
            op = max_compute(scenario(opts), s, w)
            if op["status"] == "nofit":
                rows.append({"stock": f"{s:,}", "window": f"{w} d", "GPUs used": "no-fit"})
                continue
            rows.append({"stock": f"{s:,}", "GPUs used": f"{fmt_count(op['gpus_used'])} ({op['gpus_used'] / s:.1%})",
                         "window": f"{w} d", "tr:inf": fmt_ratio(op["train_frac"]),
                         "T_step": fmt_duration(op["t_step"]), "T_update": fmt_duration(op["t_update"]),
                         "T_rollout": fmt_duration(op["t_rollout"]), "T_bcast": fmt_duration(op["t_bc"]),
                         "MFU": f"{op['mfu']:.0%}", "bottleneck": op["bottleneck"], "steps": fmt_count(op["steps"]),
                         "max C_RL": f"{op['value']:.2e}"})
    return Table("A3", "Maximum RL compute by stock: the operating point behind chart 2b" + _on(opts),
                 _gpu_cols(pd.DataFrame(rows).fillna("").set_index("stock"), opts),
                 caption="GPUs used is what the fastest split allocates (whole nodes, so a remainder can idle). "
                         "MFU is achieved over the whole allocated cluster, so an idle pool lowers it.")


@register("table_monte_carlo", "A")
def table_monte_carlo(opts):
    rows = []
    for lab, _, pt, pub, mc in anchors():
        rows.append({"anchor": lab.replace("\n", " "), "published": f"{pub:,.0f}", "point": f"{pt:,.0f}",
                     "MC mean": f"{mc['mean']:,.0f}", "90% interval": f"{mc['p5']:,.0f} - {mc['p95']:,.0f}",
                     "p95/p5": f"{mc['spread']:.2f}x",
                     "E[R] band": "pinned" if mc["er_pct"] is None else f"+-{mc['er_pct']:.0%}",
                     "bottleneck": mc["bottleneck"],
                     "explained?": "-" if mc["inside"] is None else ("yes" if mc["inside"] else "no"),
                     "E[R] provenance": mc["basis"]})
    return Table("A4", "Monte-Carlo predictive intervals behind chart 1a", pd.DataFrame(rows).set_index("anchor"),
                 caption=f"{MC_DRAWS:,} seeded draws per anchor. Trainer MFU: per-config triangular, mode = the "
                         f"preset's data-grounded MFU. Inference MFU ~ U{MFU_INF_RANGE}. E[R] triangular "
                         f"+-{ER_PCT:.0%} (prime-rl pinned). 'Explained?' = the published step time falls inside "
                         f"the 90% interval.")


@register("table_caps", "A")
def table_caps(opts):
    rows = []
    for g in list(HIGH_END_GPUS) + ["Cerebras-CS4"]:
        rows.append({"GPU": g, "dense BF16 TFLOP/s": f"{GPUS[g]['flops'] / 1e12:,.0f}",
                     "TPP/chip (TFLOP-bit/s)": f"{tpp_per_chip(g) / 1e12:,.0f}",
                     "HBM GB": f"{GPUS[g]['hbm_gb']}" + (" (SRAM)" if g == "Cerebras-CS4" else ""),
                     "board W": f"{GPUS[g]['watts']:,}",
                     f"node cap ({H100_EQUIV} H100-eq)": f"{node_gpus_tpp_cap(g, H100_EQUIV)}",
                     **{f"node @{kw} kW": f"{node_gpus_power_cap(g, kw)}" for kw in NODE_KW},
                     "GPUs @1 MW": f"{node_gpus_power_cap(g, 1000):,}", "GPUs @5 MW": f"{node_gpus_power_cap(g, 5000):,}"})
    return Table("A5", "Node and site caps by GPU family", pd.DataFrame(rows).set_index("GPU"),
                 caption=f"Export node cap: GPUs per node under {H100_EQUIV} H100-equivalents of TPP and HBM. Power: "
                         f"GPUs a node holds at {' / '.join(map(str, NODE_KW))} kW (one dense server, a rack, an NVL72-class "
                         f"rack) and a site runs at 1 / 5 MW, all facility draw (board W x {FACILITY_PUE:g}). Cerebras CS-4 "
                         f"is per WSE-3 wafer; its 44 GB is on-wafer SRAM, so it can't hold a frontier model alone.")


NODE_KW = (10, 40, 132)
SITE_MW = (1, 10, 100)


@register("table_sites", "A")
def table_sites(opts, window_d=90):
    rows = []
    for t, c in TARGETS.items():
        cfg = scenario(opts, target=t)
        ev = min_gpus(cfg, window_d)
        if ev["status"] == "nofit":
            rows.append({"target": t, "feasible": "no-fit"})
            continue
        _, mw = sites_power(cfg, ev["gpus"])
        rows.append({"target": t, "C_RL": f"{c:.1e}", "feasible": "yes" if ev["status"] == "ok" else "NO*",
                     "GPUs": f"{ev['gpus']:,}", "wall": fmt_duration(ev["wall_d"] * 86400), "total MW": f"{mw:,.1f}",
                     **{f"sites @{c} MW": f"{math.ceil(sites_power(with_cfg(cfg, site_power_mw=c), ev['gpus'])[0]):,}"
                        for c in SITE_MW}})
    return Table("A6", f"Sites needed under a per-site power cap ({window_d} days)" + _on(opts),
                 _gpu_cols(pd.DataFrame(rows).fillna("").set_index("target"), opts),
                 caption=f"A site must stay under the cap to avoid the detection tier, so more sites means more "
                         f"exposure. Facility draw: board W x {FACILITY_PUE:g}. * misses the window: the full stock's "
                         f"fastest run.")


@register("table_latency_wall", "A")
def table_latency_wall(opts):
    rows = []
    for t, c in model_targets(top=3).items():
        cfg = scenario(opts, target=t)
        for mult in (1, 4, 16):
            p = cfg.prompts_per_batch * mult
            run = with_cfg(cfg, prompts_per_batch=p)
            sol = min_wall_full_stock(run)
            rows.append({"target": t, "C_RL": f"{c:.1e}",
                         "batch": f"{fmt_count(p * cfg.responses_per_prompt)} ({p}x{cfg.responses_per_prompt})",
                         "steps": fmt_count(n_steps(run)), "per-step floor": fmt_duration(sol["r"].t_step) if sol["r"] else "no-fit",
                         "tr:inf": fmt_ratio(sol["frac"]), "min wall @stock": fmt_duration(sol["tt"])})
    return Table("A7", "The latency wall: fastest run on the whole stock as the batch grows" + _on(opts),
                 pd.DataFrame(rows).set_index("target"),
                 caption="The three largest published targets with x1 / x4 / x16 prompts per batch. A bigger batch "
                         "cuts the serial step count; the per-step floor is what remains.")


@register("table_max_compute_by_model", "A")
def table_max_compute_by_model(opts, windows=(90, 180, 720)):
    base, rows = scenario(opts), []
    for n, m in MODELS.items():
        gpu = smallest_fitting_gpu(m, base.gpus_per_node, base.b_weights_inf)
        if gpu is None:
            rows.append({"model": n, "GPU": "multi-node inference (not modelled)"})
            continue
        cfg = scenario(opts, model=m, gpu=gpu)
        ops = {w: max_compute(cfg, cfg.stock_gpus, w) for w in windows}
        op = ops[windows[0]]
        if op["status"] == "nofit":
            rows.append({"model": n, "GPU": gpu, "tr:inf": "no-fit"})
            continue
        rows.append({"model": n, "GPU": gpu, "GPUs used": fmt_count(op["gpus_used"]), "tr:inf": fmt_ratio(op["train_frac"]),
                     "T_step": fmt_duration(op["t_step"]), "T_update": fmt_duration(op["t_update"]),
                     "T_rollout": fmt_duration(op["t_rollout"]), "T_bcast": fmt_duration(op["t_bc"]), "MFU": f"{op['mfu']:.0%}",
                     "bottleneck": op["bottleneck"], **{f"max C_RL {w} d": f"{ops[w]['value']:.2e}" for w in windows}})
    return Table("A8", f"Maximum RL compute by model ({fmt_count(base.stock_gpus)}-GPU stock)",
                 pd.DataFrame(rows).fillna("").set_index("model"),
                 caption="Each model on the smallest GPU family whose node holds its inference weights, so GPU "
                         "counts differ by tier; max C_RL per window is the comparable column. MFU is achieved "
                         "over the allocated cluster.")


@register("table_censoring", "3")
def table_censoring(opts):
    rows = []
    for mdl, lab in DOMAIN_MODELS + QWEN:
        for d in DOMAINS:
            s = _stats(mdl, d)
            if s is None:
                continue
            frac, cap = s["censored_frac"], s["cap"]
            rows.append({"model": lab, "domain": d, "n": s["n"], "mean E[R]": f"{s['mean']:,.0f}",
                         "median": f"{s['median']:,.0f}", "cv": f"{s['cv']:.2f}", "p95": f"{s['p95']:,.0f}",
                         "censored": f"{frac:.0%}", "cap": f"{cap:,}" if cap else "-",
                         "status": "lower bound" if frac >= CENSOR_WARN else "usable"})
    return Table("3d", "Measured response-length distributions and their censoring",
                 pd.DataFrame(rows).set_index("model"),
                 caption=f"Generation was capped, so a censored sample's mean (and cv) is a lower bound. Rows at "
                         f"or above {CENSOR_WARN:.0%} censored should be used as bounds, not means.")
