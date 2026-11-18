"""
config.py -- FeasConfig: every knob of the feasibility study, and how it builds a model Scenario from
trainer and inference GPU counts.
"""
from dataclasses import dataclass, field, replace

from specs import ModelSpec, RLSpec, AlgoSpec, HWSpec, NetSpec, VerifySpec, PenaltySpec, Scenario
from .hardware import FACILITY_PUE, GPU_STEP, node_hw
from .models import DEFAULT_MODEL, MODELS

# ---------------------------------------------------------------------------
# Simulation Config
# ---------------------------------------------------------------------------
@dataclass
class FeasConfig:
    target_c_rl: float

    # Hardware
    gpu: str = "H100"
    gpus_per_node: int = 16             # Inference node
    train_gpus_per_node: int = 128      # Update node (holds a full model replica). Single nodes are resized DOWN to the min GPUs that fit the replica
    wan_mbps: float = 1_000.0
    compression: float = 16.0           # weight-sync compression.
    train_distributed: bool = True      # Trainer nodes at separate sites: every step all-reduces gradients over the
                                        #   same link as the broadcast. False = co-located (sync free)
    grad_compression: float = 150.0     # Gradient compression on that all-reduce: Rahman's default (4-bit + top-k
                                        #   sparsification with error feedback; Covenant-72B ran 146x)
    sync_interval: float = 1            # Off-policy staleness degree k
    inf_tp_max: int = 1                 # Largest tensor-parallel serving group the fabric allows (nodes/group).

    # RL run config (per-step compute)
    model: ModelSpec = field(default_factory=lambda: MODELS[DEFAULT_MODEL])
    prompts_per_batch: int = 128
    responses_per_prompt: int = 512     # High batch size to allow
    prompt_len: int = 1000
    er: float = 20_000.0                # E[R]. Long-context default, anchored on the measured-length campaign
    er_cv: float = 0.5
    omega: float = 1.5                  # Oversampling/filtering degree
    b_optimiser: float = 4.0            # Muon optimiser
    mfu_train: float = 0.40
    mfu_inf: float = 0.3
    inf_bw_mult: float = 1.0           # HBM-bandwidth multiplier (sensitivity knob)
    inf_flop_mult: float = 1.0         # Inference compute-throughput multiplier. e.g. 4-bit inference on FP4 tensor cores is inf_flop_mult=4 (4x dense throughput).
    b_weights_inf: float = 2.0         # inference-side weight precision, BYTES/param (BF16=2, 4-bit=0.5).
    b_kv: float = 2.0                  # KV-cache precision, BYTES (BF16=2, 4-bit KV=0.5): KV memory footprint + traffic.
    hbm_mult: float = 1.0              # HBM-capacity multiplier (sensitivity knob: concurrency / model-fit)

    # Infrastructure
    gpu_hr_usd: float = 2.0
    stock_gpus: int = 1_500_000
    site_power_mw: float = 5.0         # per-site FACILITY power cap, MW (detectability tier)

    # Rahman's distributed-training penalties (penalties.py). Off by default.
    penalty: PenaltySpec = field(default_factory=PenaltySpec)

    def build(self, n_train_gpus, n_inf_gpus, tp=1):
        # Trainer: whole train_gpus_per_node-GPU replica nodes, or ONE right-sized node (<= that) when
        # a single node suffices. n_shard = the node's own GPUs -> each node holds a full replica.
        tnode_max = self.train_gpus_per_node
        n_train_gpus = max(1, int(n_train_gpus))
        if n_train_gpus <= tnode_max:
            train_node, n_train_nodes = n_train_gpus, 1
        else:
            n_train_nodes = -(-n_train_gpus // tnode_max); train_node = tnode_max
        # Inference: whole gpus_per_node nodes, or ONE downsized node (>= GPU_STEP, <= gpus_per_node)
        # when a single node suffices -- so a big centralised (e.g. NVL72) node isn't forced to spend
        # all 72 GPUs when fewer keep rollout off the critical path.
        # With tp > 1 the pool is allocated in whole tp-node serving groups instead.
        inode_max = self.gpus_per_node
        tp = max(1, int(tp))
        n_inf_gpus = max(GPU_STEP, int(n_inf_gpus))
        if tp > 1:
            inode = inode_max
            n_inf_nodes = max(tp, (n_inf_gpus // (tp * inode_max)) * tp)
        elif n_inf_gpus <= inode_max:
            inode, n_inf_nodes = n_inf_gpus, 1
        else:
            n_inf_nodes = -(-n_inf_gpus // inode_max); inode = inode_max
        train_nh = node_hw(self.gpu, train_node); train_nh["node_hbm"] *= self.hbm_mult
        inf_nh = node_hw(self.gpu, inode)
        inf_nh["node_bw"] *= self.inf_bw_mult
        inf_nh["node_flops"] *= self.inf_flop_mult
        inf_nh["node_hbm"] *= self.hbm_mult
        return Scenario(
            name=f"{self.gpu} train {n_train_nodes * train_node:,}/inf {n_inf_nodes * inode:,}",
            model=self.model,
            rl=RLSpec(prompts_per_batch=self.prompts_per_batch,
                      responses_per_prompt=self.responses_per_prompt, prompt_len=self.prompt_len,
                      response_len_mean=self.er, response_len_cv=self.er_cv,
                      max_response_len=int(self.er * 2), n_steps_override=1,
                      oversample_override=self.omega, prefix_caching=True),
            algo=AlgoSpec(b_optimiser=self.b_optimiser, recomp_act=1, recomp_old=0, opt_steps=1,
                          compression_ratio=self.compression,
                          grad_compression=self.grad_compression if self.train_distributed else 1.0,
                          sync_interval=self.sync_interval,   # k>=1 => overlapped (max); k=0 => serial (sum)
                          b_weights_inf=self.b_weights_inf, b_kv=self.b_kv,
                          act_site_parallel=True),   # replica node: activation SP-sharded within the node
            train_hw=HWSpec("trainer", n_nodes=n_train_nodes, gpus_per_node=train_node,
                            mfu=self.mfu_train, penalty_para=1.0, n_shard=train_node, **train_nh),
            inf_hw=HWSpec("inference", n_nodes=n_inf_nodes, gpus_per_node=inode,
                          mfu=self.mfu_inf, bw_eff=0.85, inf_tp=tp, **inf_nh),
            net=NetSpec(bandwidth=self.wan_mbps * 1e6 / 8, latency=0.5,
                        train_bandwidth=self.wan_mbps * 1e6 / 8 if self.train_distributed else None,
                        train_latency=0.5 if self.train_distributed else 0.0),
            verify=VerifySpec(mode="rule", seconds_per_rollout=0.5, n_verifier_workers=10000),
            penalty=self.penalty,
        )


def with_cfg(cfg, **kw):
    """Copy of cfg with fields overridden (dataclasses.replace, shorter at call sites)."""
    return replace(cfg, **kw)


def config_lines(cfg):
    """Full, human-readable dump of every FeasConfig field that affects the results, as
    'key: value' strings -- for embedding at the top of CLI output, as a CSV comment header, or
    in a dashboard panel, so any output artifact on its own carries the exact settings that
    produced it (PROJECT_CONTEXT's reproducibility requirement: every number traceable to a
    logged config). `_cfg_hdr` (feasibility.py) is the one-line compact version used in table
    titles; this is the full version for a standalone config block."""
    return [
        f"target_c_rl: {cfg.target_c_rl:.3e} FLOP",
        f"model: {cfg.model.name}",
        f"gpu: {cfg.gpu} (inference {cfg.gpus_per_node} GPUs/node, "
        f"update {cfg.train_gpus_per_node} GPUs/node, full replica per node)",
        f"inf_tp_max: {cfg.inf_tp_max} node(s)/serving group"
        f"{' (tensor-parallel allowed; solver picks the smallest that wins)' if cfg.inf_tp_max > 1 else ' (no cross-node inference sharding)'}",
        f"wan_mbps: {cfg.wan_mbps:g}",
        f"compression: {cfg.compression:g}x",
        f"trainer: {f'distributed -- gradient all-reduce every step over the same link at {cfg.grad_compression:g}x' if cfg.train_distributed else 'co-located (gradient sync free)'}",
        f"sync_interval (off-policy k): {cfg.sync_interval:g} "
        f"({'on-policy, serial/sum' if cfg.sync_interval < 1 else f'{cfg.sync_interval:g}-step off-policy, overlapped/max, broadcast /{max(cfg.sync_interval,1):g}'})",
        f"batch: {cfg.prompts_per_batch} prompts x {cfg.responses_per_prompt} responses "
        f"= {cfg.prompts_per_batch * cfg.responses_per_prompt:,} rollouts/step",
        f"prompt_len: {cfg.prompt_len:,} tok",
        f"er (E[R]): {cfg.er:,.0f} tok (cv={cfg.er_cv:g})",
        f"omega (oversample): {cfg.omega:g}",
        f"b_optimiser: {cfg.b_optimiser:g} bytes/param",
        f"mfu_train: {cfg.mfu_train:g}  mfu_inf: {cfg.mfu_inf:g}",
        f"inf_bw_mult: {cfg.inf_bw_mult:g}  inf_flop_mult: {cfg.inf_flop_mult:g}  hbm_mult: {cfg.hbm_mult:g}",
        f"b_weights_inf: {cfg.b_weights_inf:g} B/param  b_kv: {cfg.b_kv:g} B",
        f"gpu_hr_usd: ${cfg.gpu_hr_usd:g}",
        f"stock_gpus: {cfg.stock_gpus:,}",
        f"site_power_mw: {cfg.site_power_mw:g} (facility draw, board TDP x {FACILITY_PUE:g})",
        penalty_line(cfg.penalty),
    ]


def penalty_line(p):
    """One config line for the Rahman penalty toggle."""
    if not p.enabled:
        return "penalties: off (penalty-free roofline)"
    return (f"penalties: ON (Rahman) -- trainer mode {p.mode}, inference mode {p.inf_mode}, "
            f"H={p.h_inner or 'auto'}, "
            f"failures {p.failure_rate:g}/GPU-h, quality '{p.scenario}' (eta reported, never sizes)")
