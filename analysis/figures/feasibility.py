"""
Section 2 -- feasibility of the decentralised threat model: what each RL budget costs, what a stock
can do, and which levers move it.
"""
from feasibility import (BASELINE_BATCH, BATCH_SWEEP, HBM_MULTS, HIGH_BATCH, HIGH_END_GPUS,
                         MODELS, POLICY_THRESHOLDS, batch_split, node_gpus_tpp_cap, optimal_split,
                         per_step_flop, smallest_fitting_gpu, total_time, with_cfg)
from ..scenarios import BATCH_MODES, REFERENCE_C_RL, TARGETS, scenario
from ..spec import Fig, fmt_count, register, stock_label, tex_sci
from ..sweep import max_compute, min_gpus, required_link, sweep

WINDOWS = (30, 90, 180, 720)
STOCKS = (64, 128, 1_024, 10_000, 100_000, 1_000_000)
ER_POINTS = (1_000, 10_000, 50_000, 100_000, 500_000)
SENS_GPUS = 100_000
H100_EQUIV = 16
MSHORT = {"Llama3-8B": "L3-8B", "Llama3-70B": "L3-70B", "Llama3-405B": "L3-405B", "Qwen3-30B-A3B": "Q3-30B/A3",
          "Qwen3-235B-A22B": "Q3-235B/A22", "Kimi-K3": "K3-2.4T", "Llama4-402B-A17B": "L4-402B/A17"}


def target_label(t):
    return f"{t}\n{tex_sci(TARGETS[t])}"


def target_refs():
    return tuple((t, c) for t, c in sorted(TARGETS.items(), key=lambda kv: kv[1]))


def days(w):
    return f"{w} days"


def _window_fig(fig_id, title, opts, cfg_fn, caption=""):
    data = sweep({"target": list(TARGETS), "window_d": list(WINDOWS)},
                 lambda p: min_gpus(cfg_fn(p["target"]), p["window_d"]))
    return Fig(fig_id, title, "bars", data, x="target", series="window_d", ordinal=True,
               ylabel="GPUs to finish in window", x_fmt=target_label, series_fmt=days, caption=caption)


@register("frontier", "2")
def frontier(opts):
    return _window_fig("2a", "Minimum GPUs to finish each RL compute budget", opts,
                       lambda t: scenario(opts, target=t),
                       caption=f"{scenario(opts).model.name}, decentralised threat model. Hollow bars miss the "
                               f"window: drawn at the GPUs that reach the fastest achievable wall-clock, which is "
                               f"written inside. " + BATCH_MODES[opts.batch_mode])


@register("max_compute", "2")
def max_compute_by_stock(opts):
    data = sweep({"stock": list(STOCKS), "window_d": list(WINDOWS)},
                 lambda p: max_compute(scenario(opts), p["stock"], p["window_d"]))
    return Fig("2b", "Maximum RL compute a GPU stock can spend", "bars", data, x="stock", series="window_d",
               ordinal=True, ylabel="RL compute (FLOP)", xlabel="GPU stock", fmt="sci", refs=target_refs(),
               x_fmt=stock_label, series_fmt=days,
               caption="Fastest trainer:inference split of each stock. Dashed lines: estimated RL budgets of "
                       "published models.")


@register("tornado", "2")
def tornado(opts):
    import pandas as pd
    cfg = scenario(opts, target_c_rl=REFERENCE_C_RL)
    tt, _, _, r, _ = optimal_split(cfg, SENS_GPUS)
    s = r.scenario
    n_tr, n_inf = s.train_hw.n_nodes * s.train_hw.gpus_per_node, s.inf_hw.n_nodes * s.inf_hw.gpus_per_node
    levers = {"WAN bandwidth": "wan_mbps", "compression": "compression", "HBM bandwidth": "inf_bw_mult",
              "HBM capacity": "hbm_mult", "inference FLOP": "mfu_inf", "training FLOP": "mfu_train",
              "E[R] length": "er"}
    rows = []
    for name, fld in levers.items():
        lo = total_time(with_cfg(cfg, **{fld: getattr(cfg, fld) * 0.5}), n_tr, n_inf)[0]
        hi = total_time(with_cfg(cfg, **{fld: getattr(cfg, fld) * 2.0}), n_tr, n_inf)[0]
        rows.append(dict(lever=name, lo=lo / tt, hi=hi / tt))
    rows.append(dict(lever="GPU budget", lo=total_time(cfg, n_tr * .5, n_inf * .5)[0] / tt,
                     hi=total_time(cfg, n_tr * 2, n_inf * 2)[0] / tt))
    data = pd.DataFrame(rows)
    data["swing"] = data[["lo", "hi"]].max(axis=1) / data[["lo", "hi"]].min(axis=1)
    data = data.sort_values("swing", ascending=False)
    data["lever"] = pd.Categorical(data["lever"], categories=list(data["lever"]), ordered=True)
    return Fig("2c", f"Sensitivity: wall-clock change per lever ({tex_sci(REFERENCE_C_RL)} FLOP, "
                     f"{n_tr + n_inf:,}-GPU cluster held fixed, base {tt / 86400:.0f} d)",
               "tornado", data, x="lever", xlabel="wall-clock relative to base", labels=("lever x0.5", "lever x2"),
               caption="Each physics lever halved and doubled with the cluster's trainer and inference GPU counts "
                       "held fixed; GPU budget scales both counts.")


@register("cross_model", "2")
def cross_model(opts):
    base = scenario(opts)
    fit = {n: smallest_fitting_gpu(m, base.gpus_per_node, base.b_weights_inf) for n, m in MODELS.items()}

    def cfg(n):
        return scenario(opts, model=MODELS[n], gpu=fit[n], target_c_rl=REFERENCE_C_RL) if fit[n] else None

    def point(p):
        c = cfg(p["model"])
        out = min_gpus(c, p["window_d"])
        out["steps_note"] = f"{fmt_count(REFERENCE_C_RL / per_step_flop(c))} steps" if c else ""
        return out
    data = sweep({"model": list(MODELS), "window_d": [90, 180]}, point)
    return Fig("2d", f"Cross-model: minimum GPUs at {tex_sci(REFERENCE_C_RL)} FLOP (smallest node family "
                     f"that holds the weights)", "bars", data, x="model", series="window_d", ordinal=False,
               ylabel="GPUs to finish in window", x_fmt=lambda m: MSHORT.get(m, m), series_fmt=days, inner="steps_note",
               caption="Smaller models need more GPUs at fixed compute: less compute per step means more serial "
                       "steps, deeper into the latency wall (step counts inside the bars).")


def _heat_cell(ev):
    if ev["status"] == "nofit":
        return dict(text="no-fit")
    if ev["status"] == "infeas":
        return dict(text=ev["note"] + "!")
    return dict(text=fmt_count(ev["value"]))


def _gpu_heatmap(fig_id, title, opts, node_fn, col_fmt, caption):
    families = list(HIGH_END_GPUS)
    data = sweep({"gpu": families, "target": list(TARGETS)},
                 lambda p: (lambda ev: {**ev, **_heat_cell(ev)})(
                     min_gpus(scenario(opts, target=p["target"], gpu=p["gpu"], **node_fn(p["gpu"])), 180)))
    return Fig(fig_id, title, "heatmap", data, x="gpu", series="target", x_fmt=col_fmt,
               series_fmt=lambda t: f"{t}  {tex_sci(TARGETS[t])}", xlabel="GPU family (HBM bandwidth rises left to right)",
               ylabel="RL budget (FLOP)", caption=caption)


@register("gpu_family", "2")
def gpu_family(opts):
    return _gpu_heatmap("2e", "GPU family x RL budget: minimum GPUs in 180 days", opts, lambda g: {}, str,
                        caption="Cells: GPUs to finish, or the fastest achievable wall-clock ('!') when 180 days "
                           "is out of reach. Nodes of 16 GPUs throughout.")


@register("gpu_family_capped", "2")
def gpu_family_capped(opts):
    cap = {g: node_gpus_tpp_cap(g, H100_EQUIV) for g in HIGH_END_GPUS}
    return _gpu_heatmap("2f", f"Same, under a {H100_EQUIV}-H100-equivalent export cap per node", opts,
                        lambda g: dict(gpus_per_node=cap[g]), lambda g: f"{g}\n{cap[g]}/node",
                        caption="Each node shrunk to stay under 16 H100-equivalents of TPP and HBM, so "
                           "high-FLOP or high-HBM parts get small nodes.")


@register("required_link", "2")
def required_link_fig(opts):
    base = scenario(opts)
    fit = {n: smallest_fitting_gpu(m, base.gpus_per_node, base.b_weights_inf) for n, m in MODELS.items()}
    models = [n for n in MODELS if fit[n]]
    data = sweep({"model": models, "target": list(TARGETS)},
                 lambda p: required_link(scenario(opts, target=p["target"], batch=False, model=MODELS[p["model"]],
                                             gpu=fit[p["model"]]), 180))
    return Fig("2g", "WAN link needed so communication never bottlenecks (180 days)", "bars", data, x="model",
               series="target", fmt="rate", ylabel="link x compression (Mbps)", x_fmt=lambda m: MSHORT.get(m, m),
               series_fmt=lambda t: t,
               caption="The slower of the weight broadcast and the trainer's gradient all-reduce, quoted at the "
                       "broadcast's 1x so a real compression factor divides it.")


@register("batch_size", "2")
def batch_size(opts):
    def label(total):
        b, n = batch_split(total)
        return f"{fmt_count(total)}\n{b}x{n}"
    data = sweep({"batch": list(BATCH_SWEEP), "target": list(TARGETS)},
                 lambda p: min_gpus(scenario(opts, target=p["target"],
                                        **dict(zip(("prompts_per_batch", "responses_per_prompt"),
                                                   batch_split(p["batch"])))), 180))
    return Fig("2h", "Rollout batch vs minimum GPUs (180 days)", "bars", data, x="batch", series="target",
               ylabel="GPUs to finish in 180 days", xlabel="rollouts per step (prompts x responses)",
               x_fmt=label, caption="Larger batches mean fewer serial steps for the same compute.")


@register("batch_compare", "2")
def batch_compare(opts):
    sizes = {"base": BASELINE_BATCH, "large": HIGH_BATCH}
    data = sweep({"target": list(TARGETS), "setting": [f"{w} days, {k}" for w in (90, 180, 720) for k in sizes]},
                 lambda p: min_gpus(scenario(opts, target=p["target"], **sizes[p["setting"].split(", ")[1]]),
                               int(p["setting"].split(" ")[0])))
    b, h = BASELINE_BATCH, HIGH_BATCH
    return Fig("2i", f"Minimum GPUs: base batch ({b['prompts_per_batch']}x{b['responses_per_prompt']}) vs large "
                     f"({h['prompts_per_batch']}x{h['responses_per_prompt']})", "bars", data, x="target",
               series="setting", ylabel="GPUs to finish in window", x_fmt=target_label)


@register("hbm_bandwidth", "2")
def hbm_bandwidth(opts):
    data = sweep({"mult": list(HBM_MULTS), "target": list(TARGETS)},
                 lambda p: min_gpus(scenario(opts, target=p["target"], inf_bw_mult=p["mult"]), 180))
    return Fig("2j", "HBM bandwidth vs minimum GPUs (180 days)", "bars", data, x="mult", series="target",
               ylabel="GPUs to finish in 180 days", xlabel="inference HBM bandwidth (x H100)",
               x_fmt=lambda m: f"x{m:g}",
               caption="Memory bandwidth is the lever export controls move, and the binding one when rollout "
                       "is the bottleneck.")


@register("policy_thresholds", "2")
def policy_thresholds(opts):
    bars = sorted(POLICY_THRESHOLDS.items(), key=lambda kv: kv[1])
    data = sweep({"threshold": [n for n, _ in bars], "target": list(TARGETS)},
                 lambda p: (lambda over: dict(value=float("nan"), status="over" if over else "under",
                                         text="OVER" if over else "under"))(
                     TARGETS[p["target"]] >= POLICY_THRESHOLDS[p["threshold"]]))
    return Fig("2k", "RL budgets against governance FLOP thresholds", "heatmap", data, x="threshold",
               series="target", x_fmt=lambda n: f"{n}\n{tex_sci(POLICY_THRESHOLDS[n])}",
               series_fmt=lambda t: f"{t}  {tex_sci(TARGETS[t])}", xlabel="reporting threshold (total-run FLOP)",
               ylabel="RL budget (FLOP)")


@register("response_length", "2")
def response_length(opts):
    data = sweep({"target": list(TARGETS), "er": list(ER_POINTS)},
                 lambda p: min_gpus(scenario(opts, target=p["target"], er=p["er"]), 180))
    return Fig("2l", "Mean response length vs minimum GPUs (180 days)", "bars", data, x="target", series="er",
               ordinal=False, ylabel="GPUs to finish in 180 days", x_fmt=target_label,
               series_fmt=lambda e: f"E[R] {fmt_count(e)}")


@register("fp4_inference", "2")
def fp4_inference(opts):
    gpu = "B300"
    variants = {"BF16 inference": (), "4-bit inference": ("fp4_inference",)}
    data = sweep({"target": list(TARGETS), "inference": list(variants)},
                 lambda p: min_gpus(scenario(opts, *variants[p["inference"]], target=p["target"], batch=False,
                                             gpu=gpu), 180))
    return Fig("2m", f"Fully 4-bit inference on {gpu} (compute, weights and KV) vs BF16: minimum GPUs, 180 days",
               "bars", data, x="target", series="inference", ylabel="GPUs to finish in 180 days",
               x_fmt=target_label, caption="Trainer stays BF16. Hollow bars miss the window, as in 2a.")
