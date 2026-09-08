#!/usr/bin/env python3
"""
feasibility_core.py -- Config, solver and formatting primitives for the distributed-RL
feasibility study. NO printing lives here; the sweeps/tables/CSVs are in feasibility.py.

Split this way so you can tweak parameters and reason about the solver without scrolling past
fifteen table renderers, and so other tools (results/plot_dashboard.py) can import the core
without pulling in presentation code.

Question the study answers: for an RL run of a given COMPUTE SCALE (anchored to known models or
to FLOP thresholds), what is the MINIMUM cluster (node count + optimal trainer:inference split)
that completes it inside a target wall-clock window, on a given GPU family and internet speed,
subject to a finite compute stock and a per-site power (detectability) cap -- and what does it
cost.

Everything is a knob (see FeasConfig): target C_RL, GPU/server family, WAN speed, weight-sync
compression, RL hyperparams (batch, E[R], omega), economics, stock, per-site power.

RESULT MODEL: every table goes through evaluate(cfg, window), which ALWAYS returns one operating
point -- so a scenario that misses the window is never just "infeasible". When feasible: the min
cluster that hits the window. When infeasible: the full stock at optimal split, i.e. the FASTEST
ACHIEVABLE wall-clock ("days to expend the FLOPs") and the optimal trainer:rollout ratio at that
floor. `no-fit` = model weights exceed a single node (no multi-node inference TP modelled ->
genuinely unrepresentable, not just slow).

KNOWN FIRST-CUT BIASES (documented; do not read absolutes as final):
  * MoE decode-memory: handled in model.py (mem_weights_decode) -- decode bandwidth reads the
    active path (p_active_layers + head); HBM CAPACITY still holds all experts (p_total);
    KV-cache traffic stays dense (all layers). inf_bw_mult is a pure HBM-bandwidth knob.
  * penalty_para=1.0 (A-ii): INTER-node throughput loss under sharding, held at 1 -- the update runs
    on co-located replica nodes (fast intra-node interconnect) and at these bandwidths the inter-node
    loss is negligible; it would NOT hold for a WAN-sharded learner (unmodelled term). Broadcast to
    the remote INFERENCE sites IS charged (net.*).
  * Trainer node model: each update node holds a FULL model REPLICA (intra-node shard only, no cross-
    node model parallelism); multiple trainer nodes are DATA-PARALLEL replicas. Nodes are up to
    train_gpus_per_node (128) GPUs; a SINGLE trainer node is right-sized DOWN (rightsize_trainer) to
    the min GPUs that (a) fit the replica + resident activation in ONE node and (b) keep update off
    the critical path. The solver ENFORCES the full mem.fits (replica + activation <= one node's HBM),
    so a node too small for the replica is no-fit (would need the disallowed cross-node parallelism).
    act_site_parallel=True: activation is sequence/tensor-parallel sharded WITHIN the node (does not
    scale with node size); the data-parallel per-GPU path (act_site_parallel=False) is kept for the
    calibrated validation presets. Not yet modelled (would only tighten): largest gathered FSDP unit,
    collective/workspace buffers, master-copy dtype.
  * Off-policy staleness (A-iii): `sync_interval` (k) now models the DiLoCo/async_level relaxation
    -- k=0 on-policy (serial), k=1 one-step off-policy (overlapped, default), k>=2 broadcasts every
    k steps (per-step broadcast /k). Its BENEFIT only: the sample-efficiency cost of staleness is
    not modelled, so large k reads as a free win. `compression` is now purely a quantization factor.
  * Single-node inference only: weights must fit one node (no inference TP across nodes).
Time windows are ours; anchor them against Epoch's distributed-training feasibility model when
writing.
"""
import math
from dataclasses import dataclass, field, replace

from specs import ModelSpec, RLSpec, AlgoSpec, HWSpec, NetSpec, VerifySpec, Scenario
from model import simulate

# ---------------------------------------------------------------------------
# GPU / server-family presets  (per-GPU bf16 dense peak; HBM GB; HBM BW TB/s; board W)
# ---------------------------------------------------------------------------
# The Blackwell-Ultra parts keep roughly B200/GB200 dense BF16 throughput
# Their improvements are concentrated in FP4 and memory, which this BF16 roofline excludes
# They move results mainly via HBM capacity/bandwidth.
GPUS = {
    "A100":    dict(flops=312e12,  hbm_gb=80,  bw_tbs=2.03, watts=400),
    "H100":    dict(flops=989e12,  hbm_gb=80,  bw_tbs=3.35, watts=700),
    "H200":    dict(flops=989e12,  hbm_gb=141, bw_tbs=4.80, watts=700),
    "B200":    dict(flops=2250e12, hbm_gb=192, bw_tbs=8.00, watts=1000),
    "B300":    dict(flops=2250e12, hbm_gb=288, bw_tbs=8.00, watts=1100),  # Est: Blackwell Ultra HGX
    "GB200":   dict(flops=2500e12, hbm_gb=186, bw_tbs=8.00, watts=1200),  # Per-GPU in an NVL rack
    "GB300":   dict(flops=2500e12, hbm_gb=288, bw_tbs=8.00, watts=1400),  # Est, per-GPU
#    "MI300X":  dict(flops=1307e12, hbm_gb=192, bw_tbs=5.30, watts=750),   # EST: AMD, see note above
    "MI355X":  dict(flops=2500e12, hbm_gb=288, bw_tbs=8.2, watts=1400),
    "RTX4090": dict(flops=165e12,  hbm_gb=24,  bw_tbs=1.01, watts=450),   # non-flagship / consumer P2P
    "Cerebras-CS4": dict(flops=125e15, hbm_gb=44, bw_tbs=14333.0, watts=43333), # Cerebras CS-4, as reported by semianalysis.
}

# The high-end families, for the GPU-family x target sweeps. Ordered by HBM bandwidth then
# capacity, so a table reading left-to-right tracks the axis that actually binds rollout.
HIGH_END_GPUS = ("A100", "H100", "H200", "B200", "B300", "GB200", "GB300", "MI355X")
GB, TB = 1e9, 1e12

# Granularity (and minimum) for right-sizing BOTH pools: trainer replica nodes and single inference
# nodes downsize in whole GPU_STEP chunks (16 = a good-topology unit), never to an arbitrary count.
# Bump this if a coarser/finer allocation grain is wanted.
GPU_STEP = 16


def node_hw(gpu, g=8):
    s = GPUS[gpu]
    return dict(node_flops=g * s["flops"], node_hbm=g * s["hbm_gb"] * GB, node_bw=g * s["bw_tbs"] * TB)


# ---------------------------------------------------------------------------
# Export-control node caps (US BIS "Total Processing Performance").
# ---------------------------------------------------------------------------
# TPP = 2 x (non-sparse MacTOPS) x (bit length of the multiply input). We store dense BF16 FLOP/s
# ('flops'), and 1 MAC = 2 FLOP, so MacTOPS = flops/2 and TPP = 2 x (flops/2) x bits = flops x bits.
# At the reference BF16 precision (16-bit) this makes per-chip TPP just flops x 16 -- so an H100 is
# 989 TFLOP/s x 16 = 15,824 TFLOP-bit/s, and the "16 H100" cap is 253,184 TFLOP-bit/s. NOTE: at a
# uniform BF16 rating the TPP cap is numerically a pure FLOP-equivalent cap; TPP only diverges from
# FLOP-equivalents once chips are rated at different precisions (e.g. an FP4-heavy part), which this
# BF16-roofline model does not track -- so read these caps as "N H100-equivalents of dense compute".
TPP_BITS = 16                          # reference precision for the TPP rating (BF16)


def tpp_per_chip(gpu):
    """Per-chip TPP in FLOP-bit/s (BF16 rating). Divide by 1e12 for TFLOP-bit/s."""
    return GPUS[gpu]["flops"] * TPP_BITS


def node_gpus_tpp_cap(gpu, h100_equiv=16, cap_by_memory=True):
    """Max chips per node such that the node does not exceed `h100_equiv` H100s of TPP (compute)
    -- and, when cap_by_memory, also not their HBM capacity (16 H100 = 1,280 GB). The node is the
    tighter of the two, since 'compute/memory of N H100s' means neither may be exceeded. >=1."""
    n_tpp = int(h100_equiv * tpp_per_chip("H100") // tpp_per_chip(gpu))
    if not cap_by_memory:
        return max(1, n_tpp)
    n_hbm = int(h100_equiv * GPUS["H100"]["hbm_gb"] // GPUS[gpu]["hbm_gb"])
    return max(1, min(n_tpp, n_hbm))


def node_gpus_power_cap(gpu, node_kw):
    """Max chips per node under a per-node power budget (kW), from board TDP. >=1."""
    return max(1, int(node_kw * 1000 // GPUS[gpu]["watts"]))


def effective_node_gpus(gpus_per_node, stock_gpus):
    """Node size actually usable at a given fleet size: min(configured, stock//2), floored at 1.

    Two hard limits a by-stock sweep must respect, both otherwise silently violated when the
    configured node is large (e.g. an NVL72/720 node against a 16-GPU fleet):
      - a node cannot be bigger than the whole fleet;
      - disaggregated RL needs >=2 nodes (>=1 trainer + >=1 inference), which optimal_split floors
        each side to -- so a node is at most HALF the stock, else optimal_split over-allocates whole
        configured-size nodes and reports a fleet far larger than `stock` (the old bug: stock=16 with
        a 720-node silently simulated 2x720=1,440 GPUs, so 16/128/1024 all returned the same value).
    When the configured node already fits (stock >= 2*node) this is a no-op, so it only reshapes the
    small-fleet end of a sweep -- there it correctly reports the small-node physics (or no-fit, when
    even stock//2 GPUs can't hold the model for single-node inference) instead of a fiction."""
    return max(1, min(gpus_per_node, stock_gpus // 2))


# ---------------------------------------------------------------------------
# Target compute-scale presets  (total RL FLOP; see data/reasoning_models.csv for provenance)
# ---------------------------------------------------------------------------
# Names carry NO "(est)" suffix: every figure here except DeepSeek-R1-Zero's is an estimate, the
# per-entry provenance note below is the record of how each was derived, and the writeup
# disclaims the estimation method in prose. Repeating "(est)" in the key just made every chart
# category and table row 6 characters longer for a governance audience that is being told the
# provenance anyway.
MODEL_TARGETS = {          # (C_RL, provenance-note)
    "DeepSeek-R1-Zero":     (2e22, "FLOP calculations"),
    "o1":       (1.0e23, "ESTIMATE: rough bracket, absolute n/d"),
    "o3":       (1.0e24, "ESTIMATE: 10x o1 disclosed ratio, absolute n/d"),
    "Grok-3-Reasoning":   (2.5e25, "SPECULATIVE ESTIMATE, inferred from published chart"),
    "Grok-4":   (2.5e26, "SPECULATIVE ESTIMATE, inferred from published chart"),
}
THRESHOLD_TARGETS = {"1e24": 1e24, "1e25": 1e25, "1e26": 1e26}

# Named governance thresholds = total-training-run FLOP reporting/registration bars. Point for the
# writeup: RL stages (1.3e23..6e23) sit 2-3 OOM BELOW these, so an RL run reproducing a frontier
# reasoning model is invisible to FLOP-threshold governance -- the threshold binds on pre-training.
POLICY_THRESHOLDS = {
    "MIRI-monitored":   1e22, 
    "EU GPAI":          1e23,   # EU AI Act GPAI baseline
    "MIRI-strict":      1e24,
    "EU systemic":      1e25,   # EU AI Act systemic-risk tier
    "SB53 / dual-use":  1e26,   # CA SB 53 covered-model / US dual-use foundation model
}


def all_targets():
    """Model-anchored + FLOP-threshold-anchored targets, merged into one {label: C_RL} dict."""
    t = {k: v[0] for k, v in MODEL_TARGETS.items()}
    t.update({f"C={k}": v for k, v in THRESHOLD_TARGETS.items()})
    return t


def model_targets(top=None):
    """Just the model-anchored targets as {label: C_RL} -- MODEL_TARGETS itself stores
    (C_RL, provenance) tuples, which every caller was unpacking by hand. `top=n` keeps only the
    n largest by C_RL (the latency-wall/batch tables only make their point at large targets).

    Use this, not all_targets(), for anything framed around REAL PUBLISHED RUNS: the "C=1e24"
    style rows all_targets() adds are governance reporting thresholds, not models, and mixing
    them into a per-model comparison reads as if they were five more systems."""
    t = {k: v[0] for k, v in MODEL_TARGETS.items()}
    if top is None:
        return t
    return dict(sorted(t.items(), key=lambda kv: kv[1])[-top:])


# --- Experiment: per-target batch that scales with compute budget --------------------------------
# A single FIXED batch across all targets leaves the smallest budgets finishing in a handful of steps
# (at 65,536 batch, DeepSeek-R1-Zero's 2e22 spends its whole budget in ~3 steps), where serial step
# latency -- the whole point of the study -- is irrelevant. Scaling the batch with the budget instead
# gives every target a comparable, SIZEABLE step count (~750-37,000), so the latency wall actually
# binds across the ladder. Smallest target = TARGET_BATCH_BASE, x4 per step up the sorted budget
# ladder (256 -> 1,024 -> 4,096 -> 16,384 -> 65,536 for the five model targets). This is also
# physically defensible: critical batch size grows with training scale. Callers opt in explicitly
# (the dashboard's USE_TARGET_BATCH flag); model_targets()/the solver are unchanged.
TARGET_BATCH_BASE = 256

def target_batch(name):
    """Total rollout batch assigned to model target `name`, scaling x4 with each rung up the budget
    ladder (TARGET_BATCH_BASE for the smallest). None for an unknown name. Split into a square
    prompts x responses via batch_split() at the call site."""
    order = [n for n, _ in sorted(MODEL_TARGETS.items(), key=lambda kv: kv[1][0])]
    return TARGET_BATCH_BASE * 4 ** order.index(name) if name in order else None


# --- IsoCompute law (arXiv 2603.12151), an alternative to the x4 heuristic above ------------------
# That paper decomposes RL sampling compute as C = Bp * n * M (problems-per-batch x rollouts-per-
# problem x update-steps) and finds: (i) Bp is mainly a STABILITY knob, marginal within a moderate
# range -> hold it FIXED at a small-stable value; (ii) the compute-sensitive lever is n, which rises
# with budget along a sigmoid in log-log and SATURATES (~512 easy set, ~128-256 hard); (iii) the mix
# shifts from problem-heavy (low budget) to rollout-heavy (high budget). So instead of a square
# Bp x n that grows unboundedly, this fixes Bp and scales n up a saturating rung ladder, capped at
# the easy-set ceiling. ISO_N_BY_RUNG is a STYLIZED read of the paper's Figure-7 shape (powers of
# two, saturating at 512), NOT their raw fitted curve -- the exact n*(C) values weren't extractable.
ISO_BP = 128                                 # fixed problems/batch: mid of the paper's {32..1024} sweep, "stable"
ISO_N_BY_RUNG = (8, 32, 128, 256, 512)       # rollouts/problem up the budget ladder; saturating, cap 512 (easy set)

def target_batch_iso(name):
    """(prompts_per_batch, responses_per_prompt) for model target `name` under the IsoCompute law:
    fixed Bp = ISO_BP, rollouts-per-problem n from the saturating ISO_N_BY_RUNG ladder. Total Bp*n
    stays <= 65,536 (the paper's hardware cap). None for an unknown name."""
    order = [n for n, _ in sorted(MODEL_TARGETS.items(), key=lambda kv: kv[1][0])]
    return (ISO_BP, ISO_N_BY_RUNG[order.index(name)]) if name in order else None


# ---------------------------------------------------------------------------
# Model list for the cross-model feasibility sweep + the default target model.
# CAVEAT: Kimi-K3 internals (d_model/layers/heads/vocab) are ESTIMATES (post-cutoff, DeepSeek-V3/Kimi-K2-scaled)
# Its p_total/active (2.4T/104B) are given. If K3 uses MLA the GQA KV term over-counts KV.
# ---------------------------------------------------------------------------
MODELS = {
    "Llama3-8B":       ModelSpec(name="Llama-3.1-8B", is_moe=False, p_total=8.03e9, p_active=7.0e9,
                                 d_model=4096, n_layers=32, n_q_heads=32, n_kv_heads=8, head_dim=128, vocab=128256),
    "Llama3-70B":      ModelSpec(name="Llama-3.1-70B", is_moe=False, p_total=70.6e9, p_active=68.5e9,
                                 d_model=8192, n_layers=80, n_q_heads=64, n_kv_heads=8, head_dim=128, vocab=128256),
    "Llama3-405B":     ModelSpec(name="Llama-3.1-405B", is_moe=False, p_total=405e9, p_active=401e9,
                                 d_model=16384, n_layers=126, n_q_heads=128, n_kv_heads=8, head_dim=128, vocab=128256),
    "Qwen3-30B-A3B":   ModelSpec(name="Qwen3-30B-A3B (MoE)", is_moe=True, p_total=30.5e9, p_active=3.3e9,
                                 d_model=2048, n_layers=48, n_q_heads=32, n_kv_heads=4, head_dim=128, vocab=151936),
    "Qwen3-235B-A22B": ModelSpec(name="Qwen3-235B-A22B (MoE)", is_moe=True, p_total=235e9, p_active=22e9,
                                 d_model=4096, n_layers=94, n_q_heads=64, n_kv_heads=4, head_dim=128, vocab=151936),
    "Kimi-K3":         ModelSpec(name="Kimi-K3 (MoE, EST arch)", is_moe=True, p_total=2.4e12, p_active=104e9,
                                 d_model=7168, n_layers=90, n_q_heads=128, n_kv_heads=8, head_dim=128, vocab=163840),
    "Llama4-402B-A17B": ModelSpec(name="Llama4-402B-A17B", is_moe=True, p_total=400e9, p_active=17e9,
                                     d_model=5120, n_layers=48, n_q_heads=40, n_kv_heads=8, head_dim=128, vocab=202048),
}
# DEFAULT feasibility model = Qwen3-235B-A22B (an open, frontier-relevant MoE we have measured E[R] for).
DEFAULT_MODEL = "Llama3-405B"


def smallest_fitting_gpu(m, gpus_per_node, families=("H100", "H200", "B200")):
    """Smallest-HBM node (in order) whose HBM holds the bf16 weights; None -> needs multi-node
    inference TP, which this single-node-inference model does NOT represent."""
    w = 2.0 * m.p_total
    fit = [(g, node_hw(g, gpus_per_node)["node_hbm"]) for g in families]
    fit = [(g, hbm) for g, hbm in fit if hbm >= w]
    return min(fit, key=lambda x: x[1])[0] if fit else None


# Batch levers. 128 prompts x G=512 = 65,536 total rollout batch.
HIGH_BATCH = dict(prompts_per_batch=128, responses_per_prompt=512)
BASELINE_BATCH = dict(prompts_per_batch=32, responses_per_prompt=16)


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
    wan_mbps: float = 1000.0
    compression: float = 16.0           # weight-sync compression.
    sync_interval: float = 1            # Off-policy staleness degree k

    # RL run config (per-step compute)
    model: ModelSpec = field(default_factory=lambda: MODELS[DEFAULT_MODEL])
    prompts_per_batch: int = 128
    responses_per_prompt: int = 512     # High batch size to allow
    prompt_len: int = 1000
    er: float = 20_000.0                # E[R]. Slightly conservative long-context default for Qwen3-235B
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
    site_power_mw: float = 5.0         # per-site cap (detectability tier)
    fine_split: bool = False           # Precision of optimal_split grid: coarse 13-point default, or every 1%

    def build(self, n_train_gpus, n_inf_gpus):
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
        inode_max = self.gpus_per_node
        n_inf_gpus = max(GPU_STEP, int(n_inf_gpus))
        if n_inf_gpus <= inode_max:
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
                          sync_interval=self.sync_interval,   # k>=1 => overlapped (max); k=0 => serial (sum)
                          b_weights_inf=self.b_weights_inf, b_kv=self.b_kv,
                          act_site_parallel=True),   # replica node: activation SP-sharded within the node
            train_hw=HWSpec("trainer", n_nodes=n_train_nodes, gpus_per_node=train_node,
                            mfu=self.mfu_train, penalty_para=1.0, n_shard=train_node, **train_nh),
            inf_hw=HWSpec("inference", n_nodes=n_inf_nodes, gpus_per_node=inode,
                          mfu=self.mfu_inf, bw_eff=0.85, **inf_nh),
            net=NetSpec(bandwidth=self.wan_mbps * 1e6 / 8, latency=0.5),
            verify=VerifySpec(mode="rule", seconds_per_rollout=0.5, n_verifier_workers=10000),
        )


def with_cfg(cfg, **kw):
    """Copy of cfg with fields overridden (dataclasses.replace, shorter at call sites)."""
    return replace(cfg, **kw)


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


def total_time(cfg, n_train_gpus, n_inf_gpus):
    """Wall-clock to finish; inf if the config can't run:
      - inference weights don't fit a node (ro.model_fits) -> can't shard inference, simulate() NaNs;
      - a replica + resident activation doesn't fit ONE trainer node (mem.fits) -> would need cross-node
        model parallelism, which this model disallows, so the config is inadmissible."""
    try:
        r = simulate(cfg.build(n_train_gpus, n_inf_gpus))
    except (ValueError, ZeroDivisionError, OverflowError):
        return float("inf"), None
    if (r is None or not math.isfinite(r.t_step) or r.t_step <= 0
            or not r.ro.model_fits or not r.mem.fits):
        return float("inf"), r
    return n_steps(cfg) * r.t_step, r


def rightsize_trainer(cfg, n_inf_gpus):
    """Min trainer GPUs for this inference allocation: enough to (a) hold a full replica in ONE node
    and (b) keep update off the critical path (t_update <= max(rollout+verify, broadcast)). Rollout
    and broadcast don't depend on trainer size, so probe once. Returns (n_train_gpus, probe_result)."""
    r = simulate(cfg.build(cfg.train_gpus_per_node, n_inf_gpus))
    per_gpu_hbm = r.scenario.train_hw.node_hbm / r.scenario.train_hw.gpus_per_node
    mem_floor = math.ceil(r.mem.total / per_gpu_hbm)              # replica + resident activation, one node
    t_other = max(r.stages["rollout+verify"], r.t_bc)
    per_gpu_flops = GPUS[cfg.gpu]["flops"] * cfg.mfu_train        # trainer per-GPU effective FLOP/s
    if t_other > 0 and math.isfinite(t_other) and per_gpu_flops > 0:
        upd_floor = math.ceil(r.tc.c_update / (per_gpu_flops * t_other))
    else:
        upd_floor = mem_floor
    need = max(mem_floor, upd_floor)
    return -(-need // GPU_STEP) * GPU_STEP, r            # round up to a whole GPU_STEP


# The solver searches INFERENCE GPUs directly (trainer right-sized to each via rightsize_trainer).
# Two monotone facts make every search an exact bisection AND keep the reported GPU count self-
# consistent (re-deriving at that count reproduces the same wall):
#   - total(n_inf) = trainer(n_inf) + n_inf is INCREASING in n_inf (more inference -> smaller rollout
#     -> larger update floor -> larger right-sized trainer), so "max inference within a GPU budget"
#     is one bisection;
#   - wall(n_inf) is NON-INCREASING in n_inf (more inference -> faster rollout, until the broadcast
#     floor), so "min inference to hit a window" is another. The trainer is always kept just off the
#     critical path, so the trainer:inference balance falls out of these two rather than a frac grid.
def _alloc(cfg, n_inf_gpus):
    """(wall-clock, result, total_gpus) for an inference allocation with the trainer right-sized."""
    tg, _ = rightsize_trainer(cfg, n_inf_gpus)
    tt, r = total_time(cfg, tg, n_inf_gpus)
    return tt, r, (gpus_of(r) if r is not None else float("inf"))


def _pack(tt, r):
    return (tt, r.scenario.train_hw.n_nodes, r.scenario.inf_hw.n_nodes, r,
            (gpus_of(r) - r.scenario.inf_hw.n_nodes * r.scenario.inf_hw.gpus_per_node) / gpus_of(r))


def optimal_split(cfg, total_gpus, fracs=None):
    """Min-wall allocation whose trainer+inference fits total_gpus: the LARGEST inference the budget
    affords (trainer right-sized on top), found by bisection since total(n_inf) is monotone. Returns
    (time, n_train_nodes, n_inf_nodes, r, train_frac). `fracs` is accepted but unused (exact search)."""
    step = GPU_STEP
    if _alloc(cfg, step)[2] > total_gpus:            # can't afford the smallest inference node + its trainer
        return (float("inf"), 1, 1, None, 0.0)
    lo, hi = step, (total_gpus // step) * step        # inference can't exceed the budget
    while hi - lo > step:                             # largest n_inf with total(n_inf) <= budget
        mid = max(lo + step, (((lo + hi) // 2) // step) * step)
        if _alloc(cfg, mid)[2] <= total_gpus:
            lo = mid
        else:
            hi = mid
    tt, r, _ = _alloc(cfg, lo)
    return _pack(tt, r) if r is not None else (float("inf"), 1, 1, None, 0.0)


def min_cluster(cfg, target_s):
    """Smallest total GPU count finishing within target_s, capped by the stock. Sizes EACH stage to
    the per-step time budget tsb = target_s / n_steps INDEPENDENTLY (rollout and update overlap, so
    t_step = max of stages): min inference with rollout+verify <= tsb, and the SMALLEST trainer with
    update <= tsb (floored at the replica's memory fit). So a loose window lets the trainer downsize
    to the memory floor instead of needlessly matching a fast rollout -- the same downsizing in the
    centralised (big-inference-node) case as the decentralised one. Returns dict or None (infeasible
    even at the full stock)."""
    step = GPU_STEP
    steps = n_steps(cfg)
    if not (math.isfinite(steps) and steps > 0):
        return None
    tsb = target_s / steps                              # per-step time budget
    per_gpu_flops = GPUS[cfg.gpu]["flops"] * cfg.mfu_train
    stock = cfg.stock_gpus
    r16 = lambda x: -(-int(math.ceil(x)) // step) * step  # round up to a whole GPU_STEP

    def probe(n_inf):                                    # stage times at this inference (trainer-independent)
        r = simulate(cfg.build(cfg.train_gpus_per_node, n_inf))
        if not r.ro.model_fits:
            return None
        pgh = r.scenario.train_hw.node_hbm / r.scenario.train_hw.gpus_per_node
        return dict(roll=r.stages["rollout+verify"], bc=r.t_bc,
                    mem_floor=r16(r.mem.total / pgh), c_update=r.tc.c_update)

    p0 = probe(step)
    if p0 is None or p0["bc"] > tsb:                     # broadcast wall: no GPU count can beat it
        return None
    max_inf = ((stock - p0["mem_floor"]) // step) * step  # leave room for at least the memory-floor trainer
    if max_inf < step or probe(max_inf)["roll"] > tsb:
        return None                                     # even max inference can't get rollout under budget
    if p0["roll"] <= tsb:
        n_inf = step
    else:
        lo, hi = step, max_inf                           # lo fails, hi fits; rollout decreases with n_inf
        while hi - lo > step:
            mid = max(lo + step, (((lo + hi) // 2) // step) * step)
            if probe(mid)["roll"] <= tsb:
                hi = mid
            else:
                lo = mid
        n_inf = hi
    p = probe(n_inf)
    upd_floor = r16(p["c_update"] / (per_gpu_flops * tsb)) if tsb > 0 else p["mem_floor"]
    n_train = max(p["mem_floor"], upd_floor)             # smallest trainer with update <= tsb, replica fits
    if n_train + n_inf > stock:
        return None
    tt, r = total_time(cfg, n_train, n_inf)
    if r is None:
        return None
    g = gpus_of(r)
    return dict(gpus=g, nt=r.scenario.train_hw.n_nodes, ni=r.scenario.inf_hw.n_nodes, tt=tt, r=r,
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
        # Training steps to spend cfg.target_c_rl at this RL config -- a property of the batch
        # size/model/target, NOT of the cluster or window (per_step_flop is node-count-independent).
        steps=n_steps(cfg) if runnable else float("nan"),
    )


def cost_of(cfg, sol):
    """Legacy helper (min_cluster-style dict). Prefer evaluate() for new code."""
    gpus = sol["gpus"]
    gpu_hrs = gpus * sol["tt"] / 3600
    return gpus, gpu_hrs, gpu_hrs * cfg.gpu_hr_usd


# Link speed at which the weight broadcast is definitely NOT a step bottleneck (~1 Pbit/s -- for any
# model here t_bc collapses to the fixed latency floor). required_bcast_mbps solves the operating
# point AT this speed so the metric is decoupled from the config's own link and compression.
BCAST_FREE_MBPS = 1e9


def required_bcast_mbps(cfg, window_s):
    """Minimum weight-sync link*compression (Mbps, at 1x compression) for the broadcast to stay
    hidden -- i.e. NOT become the step-time bottleneck -- at this target/window. -> (mbps_or_None, ev).

    Solved with the broadcast effectively FREE (BCAST_FREE_MBPS): evaluate() picks the min-hardware
    allocation and its compute-stage times WITHOUT the current link or compression influencing them
    (otherwise the metric is circular -- a slow link slackens the trainer via rightsize_trainer, which
    would report a LOWER requirement -- and drifts with the compression knob via the same operating-
    point coupling). The requirement is then set purely by the compute stages the broadcast overlaps.

    Broadcast overlaps the WHOLE step (>=1-step off-policy: concurrent with the next step's generation
    AND update), so per model.py's t_step = max(gen, update, bcast) it is hidden iff it fits under the
    SLOWER compute stage: t_compute = max(rollout+verify, update). k>=2 off-policy gives it k steps.

    Returned at 1x (link*compression), compression-INDEPENDENT: the uncompressed weight volume over
    the time budget, so a real compression factor simply DIVIDES it (int4-at-4x needs a quarter).

    None when the model doesn't fit a node, or when even a free link's fixed latency floor already
    exceeds the compute budget -- there no finite bandwidth suffices ('unbounded'), a real answer."""
    ev = evaluate(with_cfg(cfg, wan_mbps=BCAST_FREE_MBPS), window_s)
    if not ev["runnable"]:
        return None, ev
    r = ev["r"]
    t_compute = max(r.t_update, r.stages["rollout+verify"])   # broadcast hides behind the slower stage
    lat = r.scenario.net.latency
    k = max(r.scenario.algo.sync_interval, 1)                 # k>=2 amortises the broadcast over k steps
    budget = k * t_compute - lat
    if not (math.isfinite(t_compute) and budget > 0):
        return None, ev
    raw_weight_bytes = cfg.b_weights_inf * cfg.model.p_total  # uncompressed broadcast volume (1x)
    return raw_weight_bytes * 8 / 1e6 / budget, ev


# arXiv 2603.12151's largest EMPIRICALLY TESTED rollout batch. No longer a hard bound on the
# sweep below (which deliberately runs one point past it) -- it marks where the runtime model
# stops having calibration to stand on, so presentation layers should flag points above it as
# extrapolation rather than silently omitting or silently showing them.
BATCH_CAP = 65_536

# Total rollout batch (B questions x n samples/question), x4 per step so six points span three
# orders of magnitude. All exponents are even, so batch_split() gives clean square B x n splits.
BATCH_SWEEP = (256, 1024, 4096, 16_384, 65_536, 262_144)


def batch_split(total):
    """Split a power-of-two total rollout batch into (B questions, n samples per question) as
    evenly as powers of two allow: B = 2**ceil(e/2), n = 2**floor(e/2) for total = 2**e.

    Even-ish is the right default because at fixed total batch the two axes are near-substitutes
    for wall-clock in this model -- both just cut the serial step count, and n is only marginally
    cheaper per rollout via shared prefill. That substitutability is exactly why a full B x n grid
    is mostly redundant along its diagonals, and why the honest presentation sweeps the PRODUCT.
    (Where the axes stop being substitutes is learning dynamics, which this runtime model does not
    represent -- that needs the sigmoid fits from arXiv 2603.12151.)"""
    e = round(math.log2(total))
    return 2 ** ((e + 1) // 2), 2 ** (e // 2)

# HBM-bandwidth multipliers for the dedicated memory-bandwidth sweep (CLI table + dashboard panel).
HBM_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def sites_power(cfg, gpus):
    w = GPUS[cfg.gpu]["watts"]
    gpus_per_site = cfg.site_power_mw * 1e6 / w
    return gpus / gpus_per_site, gpus * w / 1e6   # (n_sites at the cap, total MW)


def split_for(cfg, total_gpus):
    """The optimal (n_train_nodes, n_inf_nodes) at a fixed total GPU budget."""
    _, nt, ni, _, _ = optimal_split(cfg, total_gpus)
    return nt, ni


# ---------------------------------------------------------------------------
# Formatting (shared by feasibility.py tables and results/plot_dashboard.py)
# ---------------------------------------------------------------------------
def fmt_time(s):
    """Single shared time formatter -- every table/CSV-adjacent display and the dashboard (as
    `_ft`) route through this, so one fix here corrects every printed figure at once.
    Values under 120s used to round to whole minutes (`f"{s/60:.0f} min"`), which silently
    destroyed precision exactly where the max-compute tables' T_update/T_broadcast often land at
    large GPU stocks: 23.3s printed as "0 min" (implying ~0), 38.1s printed as "1 min" (implying
    60s, +57% off). Below 120s now prints seconds directly instead."""
    if not math.isfinite(s):  return "inf"
    if s < 120:    return f"{s:.1f} s"
    if s < 3600:   return f"{s/60:.0f} min"
    if s < 86400:  return f"{s/3600:.1f} h"
    if s < 3.15e7: return f"{s/86400:.0f} d"
    return f"{s/3.15e7:.1f} yr"


def fmt_num(x):
    if not math.isfinite(x):  return "inf"
    for d, u in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= d:
            return f"{x/d:.1f}{u}"
    return f"{x:.0f}"


def fmt_rate(x):
    """Like fmt_num, but keeps significant digits BELOW 1. fmt_num floors to "%.0f" under 1e3, so
    every value under 0.5 printed as a flat "0" -- fine for the counts it was written for (GPUs,
    steps, dollars, all >= 1) but wrong for required-bandwidth figures, where a small model's
    weight sync genuinely needs a fraction of an Mbps and "0" reads as "none/free"."""
    if not math.isfinite(x):  return "inf"
    for d, u in ((1e6, "M"), (1e3, "k")):
        if abs(x) >= d:
            return f"{x/d:.1f}{u}"
    if abs(x) >= 10:  return f"{x:.0f}"
    if abs(x) >= 1:   return f"{x:.1f}"
    return f"{x:.2f}"


def fmt_ratio(frac):
    """Trainer:inference (rollout) node ratio as a compact 'NN:MM'."""
    return f"{frac*100:.0f}:{100-frac*100:.0f}"


def cell_text(ev):
    """Compact cell used by matrix-style tables:
        feasible   -> 'GPUs/wall/tr:inf'
        infeasible -> 'wall!/tr:inf'   (! = misses window; wall = full-stock floor = days-to-expend)
        no-fit     -> 'no-fit'"""
    if not ev["runnable"]:
        return "no-fit"
    if ev["feasible"]:
        return f"{fmt_num(ev['gpus'])}/{fmt_time(ev['wall_s'])}/{fmt_ratio(ev['frac'])}"
    return f"{fmt_time(ev['wall_s'])}!/{fmt_ratio(ev['frac'])}"


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
        f"wan_mbps: {cfg.wan_mbps:g}",
        f"compression: {cfg.compression:g}x",
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
        f"site_power_mw: {cfg.site_power_mw:g}",
        f"split_search: {'fine (1% grid)' if cfg.fine_split else 'coarse (13-point grid)'}",
    ]
