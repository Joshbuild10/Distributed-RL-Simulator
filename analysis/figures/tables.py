"""
Tables: the per-run detail a chart can't carry. Each is a DataFrame of display strings, so the dashboard
shows it and the CLI writes it as CSV unchanged.
"""
import pandas as pd

import lengths_io
from feasibility_core import (FACILITY_PUE, GPUS, HIGH_END_GPUS, fmt_num, fmt_ratio, fmt_time,
                              node_gpus_power_cap, node_gpus_tpp_cap, tpp_per_chip)
from validation.uncertainty import ER_PCT, MC_DRAWS, MFU_INF_RANGE
from ..scenarios import BATCH_MODES, TARGETS, scenario
from ..spec import Table, register
from ..sweep import max_compute, min_gpus
from .feasibility import ER_POINTS, H100_EQUIV, STOCKS, WINDOWS
from .lengths import CENSOR_WARN, DOMAIN_MODELS, DOMAINS, QWEN, path
from .validation import anchors


def _run_cols(ev):
    """The shared per-run columns of the A1 and E[R] tables."""
    if ev["status"] == "nofit":
        return {"feasible": "no-fit"}
    feas = ev["status"] == "ok"
    return {"feasible": "yes" if feas else "NO*", "GPUs": f"{ev['gpus']:,}" + ("" if feas else "*"),
            "% stock": f"{ev['pct_stock']:.2f}%", "tr:inf": fmt_ratio(ev["train_frac"]),
            "steps": fmt_num(ev["steps"]), "wall": fmt_time(ev["wall_d"] * 86400),
            "cost": "$" + fmt_num(ev["cost"]) if feas else "-", "bottleneck": ev["bottleneck"],
            "T_step": fmt_time(ev["t_step"]), "T_update": fmt_time(ev["t_update"]),
            "T_rollout": fmt_time(ev["t_rollout"]), "T_bcast": fmt_time(ev["t_bc"])}


@register("table_runs", "A")
def table_runs(opts):
    rows = []
    for t, c in TARGETS.items():
        cfg = scenario(opts, target=t)
        for w in WINDOWS:
            rows.append({"target": t, "C_RL": f"{c:.1e}",
                         "batch": fmt_num(cfg.prompts_per_batch * cfg.responses_per_prompt),
                         "window": f"{w} d", **_run_cols(min_gpus(cfg, w))})
    return Table("A1", "Every target x window: the minimum cluster or the full-stock floor",
                 pd.DataFrame(rows).fillna("").set_index("target"),
                 caption="* misses the window: GPUs and wall-clock are the full stock's fastest run. "
                         + BATCH_MODES[opts.batch_mode])


@register("table_response_length", "A")
def table_response_length(opts):
    rows = [{"target": t, "C_RL": f"{c:.1e}", "E[R]": fmt_num(er),
             **_run_cols(min_gpus(scenario(opts, target=t, er=er), 180))}
            for t, c in TARGETS.items() for er in ER_POINTS]
    return Table("A2", "Response length sensitivity, every target (180 days)",
                 pd.DataFrame(rows).fillna("").set_index("target"), caption="The detail behind chart 2l.")


@register("table_max_compute", "A")
def table_max_compute(opts):
    rows = []
    for s in STOCKS:
        for w in WINDOWS:
            op = max_compute(scenario(opts), s, w)
            if op["status"] == "nofit":
                rows.append({"stock": f"{s:,}", "window": f"{w} d", "GPUs used": "no-fit"})
                continue
            rows.append({"stock": f"{s:,}", "GPUs used": f"{fmt_num(op['gpus_used'])} ({op['gpus_used'] / s:.1%})",
                         "window": f"{w} d", "tr:inf": fmt_ratio(op["train_frac"]),
                         "T_step": fmt_time(op["t_step"]), "T_update": fmt_time(op["t_update"]),
                         "T_rollout": fmt_time(op["t_rollout"]), "T_bcast": fmt_time(op["t_bc"]),
                         "MFU": f"{op['mfu']:.0%}", "bottleneck": op["bottleneck"], "steps": fmt_num(op["steps"]),
                         "max C_RL": f"{op['value']:.2e}"})
    return Table("A3", "Maximum RL compute by stock: the operating point behind chart 2b",
                 pd.DataFrame(rows).fillna("").set_index("stock"),
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
                     "GPUs @1 MW": f"{node_gpus_power_cap(g, 1000):,}", "GPUs @5 MW": f"{node_gpus_power_cap(g, 5000):,}"})
    return Table("A5", "Node and site caps by GPU family", pd.DataFrame(rows).set_index("GPU"),
                 caption=f"Export node cap: GPUs per node under {H100_EQUIV} H100-equivalents of TPP and HBM. Power: "
                         f"GPUs a site runs at 1 / 5 MW of facility draw (board W x {FACILITY_PUE:g}). Cerebras CS-4 "
                         f"is per WSE-3 wafer; its 44 GB is on-wafer SRAM, so it can't hold a frontier model alone.")


@register("table_censoring", "3")
def table_censoring(opts):
    rows = []
    for mdl, lab in DOMAIN_MODELS + QWEN:
        for d in DOMAINS:
            mean = lengths_io.mean_length(path(mdl, d))
            if mean is None:
                continue
            frac, cap, _ = lengths_io.meta(path(mdl, d))
            rows.append({"model": lab, "domain": d, "mean E[R]": f"{mean:,.0f}", "censored": f"{frac:.0%}",
                         "cap": f"{cap:,}" if cap else "-", "status": "lower bound" if frac >= CENSOR_WARN else "usable"})
    return Table("3d", "Right-censoring of the measured lengths", pd.DataFrame(rows).set_index("model"),
                 caption=f"Generation was capped, so a censored sample's mean is a lower bound. Rows at or above "
                         f"{CENSOR_WARN:.0%} censored should be used as bounds, not means.")
