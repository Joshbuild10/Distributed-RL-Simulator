"""
scenarios.py -- The threat model and its named variants, in one place.

A scenario is a FeasConfig built in a fixed order, so precedence is never ambiguous:

    threat model  <-  Options.over (CLI / explorer knobs)  <-  variants  <-  target batch  <-  overrides

Chart-specific variants win over global knobs: a decentralised-vs-centralised chart must set the
link speed whatever the explorer's slider says.
"""
from dataclasses import dataclass

from feasibility_core import (FeasConfig, MODELS, model_targets, target_batch, target_batch_iso, batch_split,
                              with_cfg)
from specs import PenaltySpec

TARGETS = model_targets()                     # {name: C_RL}, published runs only
REFERENCE_C_RL = 2.5e25                       # Grok-3 scale: the default when a chart has no target axis

# Threat model (report, Methodology): Llama-3.1-405B on H100, 16-GPU inference islands under the Scher
# et al. cap, 128-GPU trainer sites, 1 Gbps WAN, 16x weight broadcast, 150x gradients -- FeasConfig's
# defaults. Variants are the levers the paper compares.
VARIANTS = {
    "decentralised":      dict(gpus_per_node=16, wan_mbps=1_000, compression=16.0, inf_tp_max=1),
    "centralised":        dict(gpus_per_node=72, wan_mbps=1_000_000, compression=16.0, inf_tp_max=4),
    "centralised_fabric": dict(wan_mbps=1_000_000, compression=16.0),   # isolates node size from the WAN
    "broadcast_4x":       dict(compression=4.0),
    "colocated_trainer":  dict(train_distributed=False),
    "grad_150x":          dict(grad_compression=150.0),
    "grad_16x":           dict(grad_compression=16.0),
    "fp4_inference":      dict(inf_flop_mult=4, b_weights_inf=0.5, b_kv=0.5),
}


@dataclass(frozen=True)
class Options:
    """Run-wide settings every figure receives."""
    penalties: bool = False
    batch_mode: str = "scaled"                # scaled | isocompute | flat
    over: tuple = ()                          # FeasConfig overrides as ((field, value), ...); hashable, so
                                              #   `model` is given by MODELS key, not ModelSpec

    def with_over(self, **kw):
        return Options(self.penalties, self.batch_mode, tuple({**dict(self.over), **kw}.items()))


BATCH_MODES = {
    "scaled": "Batch scales with the budget: 256 to 65,536 in x4 square rungs across the five targets.",
    "isocompute": "Batch follows the IsoCompute law (arXiv 2603.12151): 128 problems per batch, "
                  "rollouts per problem rising 8 to 512.",
    "flat": "One batch (128 x 512) for every target.",
}


def batch_for(target, mode="scaled"):
    """Per-target batch overrides under `mode`; {} for flat mode or a non-target."""
    if mode == "flat" or target not in TARGETS:
        return {}
    bp, rp = target_batch_iso(target) if mode == "isocompute" else batch_split(target_batch(target))
    return dict(prompts_per_batch=bp, responses_per_prompt=rp)


def scenario(opts, *variants, target=None, batch=True, **over):
    """FeasConfig for `variants` at `target` (a TARGETS name; sets C_RL and, if `batch`, its batch
    under opts.batch_mode), with `over` applied last."""
    merged = dict(opts.over)
    for v in variants:
        merged.update(VARIANTS[v])
    if target is not None:
        merged["target_c_rl"] = TARGETS[target]
        if batch:
            merged.update(batch_for(target, opts.batch_mode))
    merged.update(over)
    if isinstance(merged.get("model"), str):
        merged["model"] = MODELS[merged["model"]]
    cfg = FeasConfig(target_c_rl=REFERENCE_C_RL, penalty=PenaltySpec(enabled=opts.penalties))
    return with_cfg(cfg, **merged)
