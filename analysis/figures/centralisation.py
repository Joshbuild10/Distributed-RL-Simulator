"""
Section 2.1 -- decentralised vs a centralised datacentre: what the WAN, the node size and the trainer's
placement cost, holding everything else at the threat model.
"""
from feasibility_core import MODELS
from ..scenarios import TARGETS, scenario
from ..spec import Fig, register, stock_label
from ..sweep import max_compute, min_gpus, sweep
from .feasibility import days, target_label, target_refs

STOCKS = (128, 1_024, 10_000, 100_000, 1_000_000)     # 64 GPUs cannot form two NVL72 nodes
NODE_SIZES = (16, 32, 64, 72, 128)
NODE_STOCK = 1_500_000
SETUPS = ("decentralised", "centralised")


def _node_label(n):
    return f"{n}\n(NVL72)" if n == 72 else str(n)


def _setups_min(fig_id, title, opts, extra=(), caption=""):
    data = sweep({"target": list(TARGETS), "setup": list(SETUPS)},
                 lambda p: min_gpus(scenario(opts, p["setup"], *extra, target=p["target"]), 180))
    return Fig(fig_id, title, "bars", data, x="target", series="setup", ylabel="GPUs to finish in 180 days",
               x_fmt=target_label, series_fmt=str.capitalize, caption=caption)


def _setups_max(fig_id, title, opts, extra=(), caption=""):
    data = sweep({"stock": list(STOCKS), "setup": list(SETUPS)},
                 lambda p: max_compute(scenario(opts, p["setup"], *extra), p["stock"], 180))
    return Fig(fig_id, title, "bars", data, x="stock", series="setup", ylabel="RL compute (FLOP)",
               xlabel="GPU stock", fmt="sci", refs=target_refs(), x_fmt=stock_label, series_fmt=str.capitalize,
               caption=caption)


CEN_NOTE = ("Centralised: 72-GPU NVL72 nodes, 1 Tbps fabric, up to 4-node tensor parallelism. "
            "Decentralised: 16-GPU islands on 1 Gbps.")


@register("dec_vs_cen_min", "2.1")
def dec_vs_cen_min(opts):
    return _setups_min("2n", "Decentralised vs centralised: minimum GPUs to finish (180 days)", opts,
                       caption=CEN_NOTE)


@register("dec_vs_cen_max", "2.1")
def dec_vs_cen_max(opts):
    return _setups_max("2o", "Decentralised vs centralised: maximum RL compute in 180 days", opts,
                       caption=CEN_NOTE + " The decentralised ceiling is the 1 Gbps weight broadcast.")


@register("dec_vs_cen_min_4x", "2.1")
def dec_vs_cen_min_4x(opts):
    return _setups_min("2p", "Same at 4x broadcast compression: minimum GPUs", opts, extra=("broadcast_4x",),
                       caption="A 4x larger broadcast brings the WAN wall to far smaller budgets.")


@register("dec_vs_cen_max_4x", "2.1")
def dec_vs_cen_max_4x(opts):
    return _setups_max("2q", "Same at 4x broadcast compression: maximum RL compute", opts, extra=("broadcast_4x",),
                       caption="Decentralised compute plateaus lower; the terabit fabric keeps scaling.")


@register("node_size_max", "2.1")
def node_size_max(opts):
    data = sweep({"node": list(NODE_SIZES), "window_d": [180]},
                 lambda p: max_compute(scenario(opts, "centralised_fabric", gpus_per_node=p["node"]), NODE_STOCK, 180))
    return Fig("2r", f"Inference node size vs maximum RL compute ({stock_label(NODE_STOCK)}-GPU stock, 180 days)",
               "bars", data, x="node", series="window_d", ordinal=True, ylabel="RL compute (FLOP)",
               xlabel="GPUs per NVLink domain", fmt="sci", refs=target_refs(), x_fmt=_node_label, series_fmt=days,
               caption="Link held at the centralised fabric to isolate node size: a bigger domain streams the "
                       "weights faster per sequence, so compute rises monotonically.")


@register("node_size_min", "2.1")
def node_size_min(opts):
    data = sweep({"target": list(TARGETS), "node": list(NODE_SIZES)},
                 lambda p: min_gpus(scenario(opts, "centralised_fabric", target=p["target"],
                                             gpus_per_node=p["node"]), 180))
    return Fig("2s", "Inference node size vs minimum GPUs, per RL budget (180 days)", "bars", data, x="target",
               series="node", ordinal=False, ylabel="GPUs to finish in 180 days", x_fmt=target_label,
               series_fmt=lambda n: f"{n}/node" + (" (NVL72)" if n == 72 else ""),
               caption="Node size barely matters below ~1e24 FLOP and dominates at frontier scale.")


def _trainer(fig_id, opts, model):
    setups = {"Co-located trainer": ("colocated_trainer",), "Distributed, 150x gradients": ("grad_150x",),
              "Distributed, 16x gradients": ("grad_16x",)}
    data = sweep({"stock": list(STOCKS), "trainer": list(setups)},
                 lambda p: max_compute(scenario(opts, "decentralised", *setups[p["trainer"]], model=MODELS[model]),
                                       p["stock"], 180))
    m = MODELS[model]
    sync = {c: 2 * 2 * m.p_total / c / 125e6 + 0.5 for c in (150, 16)}
    return Fig(fig_id, f"Trainer placement: maximum RL compute in 180 days ({m.name})", "bars", data, x="stock",
               series="trainer", ylabel="RL compute (FLOP)", xlabel="GPU stock", fmt="sci", refs=target_refs(),
               x_fmt=stock_label,
               caption=f"Distributed trainers all-reduce gradients every step over the 1 Gbps WAN: "
                       f"{sync[150]:.0f} s at 150x, {sync[16]:.0f} s at 16x.")


@register("trainer_dense", "2.1")
def trainer_dense(opts):
    return _trainer("2t", opts, "Llama3-405B")


@register("trainer_moe", "2.1")
def trainer_moe(opts):
    return _trainer("2u", opts, "Qwen3-235B-A22B")
