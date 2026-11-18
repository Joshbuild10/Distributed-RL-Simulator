"""
scenarios.py -- The threat model and its named variants, in one place.

A scenario is a FeasConfig. Variants are plain override dicts stacked on `base()`, so every chart and
table names the scenario it ran instead of rebuilding a config by hand.
"""
from feasibility_core import (FeasConfig, model_targets, target_batch, target_batch_iso, batch_split,
                              with_cfg)
from specs import PenaltySpec

TARGETS = model_targets()                     # {name: C_RL}, published runs only

# Threat model (report, Methodology): Llama-3.1-405B on H100, 16-GPU inference islands under the
# Scher et al. cap, 128-GPU trainer sites, 1 Gbps WAN, 16x weight broadcast, 150x gradients.
VARIANTS = {
    "decentralised":     dict(gpus_per_node=16, wan_mbps=1_000, compression=16.0, inf_tp_max=1),
    "centralised":       dict(gpus_per_node=72, wan_mbps=1_000_000, compression=16.0, inf_tp_max=4),
    "broadcast_4x":      dict(compression=4.0),
    "colocated_trainer": dict(train_distributed=False),
    "grad_150x":         dict(grad_compression=150.0),
    "grad_16x":          dict(grad_compression=16.0),
}


def batch_for(target, mode="scaled"):
    """Per-target batch overrides: 'scaled' (x4 per budget rung), 'isocompute' (arXiv 2603.12151), 'flat'."""
    if mode == "flat" or target not in TARGETS:
        return {}
    if mode == "isocompute":
        bp, rp = target_batch_iso(target)
    else:
        bp, rp = batch_split(target_batch(target))
    return dict(prompts_per_batch=bp, responses_per_prompt=rp)


def scenario(*variants, target=None, batch_mode="scaled", penalties=False, **over):
    """FeasConfig for the named variants stacked left to right, at `target` (a TARGETS name, which also
    sets its batch under `batch_mode`), with any FeasConfig field overridden last."""
    cfg = FeasConfig(target_c_rl=TARGETS.get(target, 2.5e25), penalty=PenaltySpec(enabled=penalties))
    merged = {}
    for v in variants:
        merged.update(VARIANTS[v])
    if target is not None:
        merged.update(batch_for(target, batch_mode))
    merged.update(over)
    return with_cfg(cfg, **merged) if merged else cfg
