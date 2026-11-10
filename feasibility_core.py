#!/usr/bin/env python3
"""
feasibility_core.py -- Config, solver and formatting primitives for the distributed-RL
feasibility study. NO printing lives here; the sweeps/tables/CSVs are in feasibility.py, and
results/plot_dashboard.py imports this module for the same numbers.

The question: for an RL run of a given COMPUTE SCALE (anchored to known models or to FLOP
thresholds), what is the minimum cluster -- node count + trainer:inference split -- that finishes
it inside a target wall-clock window, on a given GPU family and internet speed, subject to a
finite compute stock and a per-site power (detectability) cap, and what does it cost. Every input
is a knob: see FeasConfig.

RESULT MODEL: every table goes through evaluate(cfg, window), which ALWAYS returns one operating
point, so a scenario that misses the window is never just "infeasible":
    feasible   -> the minimum cluster that hits the window
    infeasible -> the full stock at optimal split = the fastest achievable wall-clock ("days to
                  expend the FLOPs") and the trainer:rollout ratio at that floor
    no-fit     -> the weights exceed one serving group, or the replica exceeds one trainer node;
                  either needs parallelism this model does not represent, so it is unrepresentable
                  rather than slow.

Modelling assumptions and their known biases are catalogued ONCE in SIMULATOR_MAP.md ("Deliberate
deviations" and "Known gaps" A-i..A-x) -- read that before quoting an absolute number. The two
that shape this file's solver: each trainer node holds a FULL model replica (intra-node sharding
only; multiple nodes are data-parallel replicas), and inference weights must fit ONE serving group
of inf_tp nodes. Both are enforced, not assumed -- total_time() rejects a config that violates
either. Time windows are ours; anchor them against Epoch's distributed-training feasibility model
when writing.
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

# Facility power multiplier on board TDP: hosts, networking, power conversion and cooling. Every
# power figure in the model is a FACILITY draw, so a "1 MW site" means 1 MW at the meter, not 1 MW
# of GPU board power. Set to 1.0 to quote raw board watts instead.
FACILITY_PUE = 1.6

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
# TPP = 2 x (non-sparse MacTOPS) x (multiply-input bit length). With 1 MAC = 2 FLOP that collapses
# to flops x bits, so per-chip TPP at BF16 is just flops x 16: an H100 is 15,824 TFLOP-bit/s and the
# "16 H100" cap is 253,184. At a uniform BF16 rating this is numerically a pure FLOP-equivalent cap
# -- TPP only diverges once chips are rated at different precisions (an FP4-heavy part), which this
# BF16 roofline does not track. Read the caps as "N H100-equivalents of dense compute".
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


def node_gpus_power_cap(gpu, node_kw, pue=FACILITY_PUE):
    """Max chips per node under a per-node FACILITY power budget (kW). Board TDP x pue. >=1."""
    return max(1, int(node_kw * 1000 // (GPUS[gpu]["watts"] * pue)))


def effective_node_gpus(gpus_per_node, stock_gpus):
    """Node size usable at a given fleet size. Disaggregation needs >=2 nodes, so a node is at most
    half the stock -- otherwise a by-stock sweep allocates whole configured-size nodes and reports a
    fleet larger than `stock`. No-op once stock >= 2 x node, so it only reshapes a sweep's small-
    fleet end (where the honest answer is the small-node physics, or no-fit)."""
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
# writeup: the disclosed RL stages (MODEL_TARGETS, 2e22 up) sit below these, so an RL run reproducing a frontier
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
# One FIXED batch across all targets leaves the smallest budgets finishing in ~3 steps, where serial
# step latency -- the whole point of the study -- cannot bind. Scaling the batch with the budget
# (TARGET_BATCH_BASE, x4 per rung: 256 -> 65,536 across the five targets) gives every target a
# sizeable step count, and is defensible anyway since critical batch size grows with scale.
# Callers opt in explicitly; model_targets() and the solver are unchanged.
TARGET_BATCH_BASE = 256

def target_batch(name):
    """Total rollout batch assigned to model target `name`, scaling x4 with each rung up the budget
    ladder (TARGET_BATCH_BASE for the smallest). None for an unknown name. Split into a square
    prompts x responses via batch_split() at the call site."""
    order = [n for n, _ in sorted(MODEL_TARGETS.items(), key=lambda kv: kv[1][0])]
    return TARGET_BATCH_BASE * 4 ** order.index(name) if name in order else None


# --- IsoCompute law (arXiv 2603.12151), an alternative to the x4 heuristic above ------------------
# That paper decomposes sampling compute as C = Bp * n * M and finds Bp mainly a stability knob
# (hold it fixed) while n is the compute-sensitive lever, rising along a log-log sigmoid that
# saturates (~512 easy set, ~128-256 hard). So: fix Bp, walk n up a saturating ladder.
# ISO_N_BY_RUNG is a STYLIZED read of their Figure 7 (powers of two, saturating at 512), not their
# fitted curve -- the exact n*(C) values were not extractable.
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
# DEFAULT feasibility model: dense Llama-3.1-405B -- open weights, frontier-scale, and the arch
# every published figure in the writeup is quoted at. Switch to Qwen3-235B-A22B for the MoE path
# (an open MoE with a measured E[R]).
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
    wan_mbps: float = 1_000.0
    compression: float = 16.0           # weight-sync compression.
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
                          sync_interval=self.sync_interval,   # k>=1 => overlapped (max); k=0 => serial (sum)
                          b_weights_inf=self.b_weights_inf, b_kv=self.b_kv,
                          act_site_parallel=True),   # replica node: activation SP-sharded within the node
            train_hw=HWSpec("trainer", n_nodes=n_train_nodes, gpus_per_node=train_node,
                            mfu=self.mfu_train, penalty_para=1.0, n_shard=train_node, **train_nh),
            inf_hw=HWSpec("inference", n_nodes=n_inf_nodes, gpus_per_node=inode,
                          mfu=self.mfu_inf, bw_eff=0.85, inf_tp=tp, **inf_nh),
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
    return n_steps(cfg) * r.t_step, r


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


def _min_gpus_at_tp(cfg, roll_b, upd_b, stock, tp):
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
            sol = _min_gpus_at_tp(cfg, roll_b, upd_b, stock, tp)
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
    sol = min_gpus_for_tsb(cfg, target_s / steps)
    if sol is None:
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
    budget = k * t_compute - lat
    if not (math.isfinite(t_compute) and budget > 0):
        return None, ev
    raw_weight_bytes = cfg.b_weights_inf * cfg.model.p_total  # uncompressed broadcast volume (1x)
    return raw_weight_bytes * 8 / 1e6 / budget, ev


# arXiv 2603.12151's largest empirically tested rollout batch. Not a bound on the sweep below
# (which runs one point past it) -- it marks where calibration runs out, so presentation layers
# flag points above it as extrapolation rather than showing them unmarked.
BATCH_CAP = 65_536

# Total rollout batch (B questions x n samples/question), x4 per step so six points span three
# orders of magnitude. All exponents are even, so batch_split() gives clean square B x n splits.
BATCH_SWEEP = (256, 1024, 4096, 16_384, 65_536, 262_144)


def batch_split(total):
    """Split a power-of-two total rollout batch into (B questions, n samples per question) as
    evenly as powers of two allow: B = 2**ceil(e/2), n = 2**floor(e/2) for total = 2**e.

    Even-ish because at fixed total batch the two axes are near-substitutes for wall-clock here:
    both cut the serial step count, and n is only marginally cheaper via shared prefill. That is
    why a full B x n grid is redundant along its diagonals and the presentation sweeps the PRODUCT.
    They stop being substitutes in learning dynamics, which this runtime model does not cover."""
    e = round(math.log2(total))
    return 2 ** ((e + 1) // 2), 2 ** (e // 2)

# HBM-bandwidth multipliers for the dedicated memory-bandwidth sweep (CLI table + dashboard panel).
HBM_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)


def sites_power(cfg, gpus, pue=FACILITY_PUE):
    """(sites at the per-site cap, total facility MW). Both sides are facility draw: board TDP x pue."""
    w = GPUS[cfg.gpu]["watts"] * pue
    gpus_per_site = cfg.site_power_mw * 1e6 / w
    return gpus / gpus_per_site, gpus * w / 1e6


def split_for(cfg, total_gpus):
    """The optimal (n_train_nodes, n_inf_nodes) at a fixed total GPU budget."""
    _, nt, ni, _, _ = optimal_split(cfg, total_gpus)
    return nt, ni


def max_achievable_split(cfg):
    """(T_step, nt, ni, r, frac) at the full stock, minimising step time. total_time is ~monotone
    non-increasing in node count, so the max-throughput point is always "whole stock, split
    optimally", never a subset."""
    return optimal_split(cfg, cfg.stock_gpus)


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
    sol = max_achievable_split(c)
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
                per_step_flop=pf)


def steps_and_c_rl(op, window_s):
    """Window-dependent numbers from an operating point: (steps, max achievable C_RL). Pure
    arithmetic, so sweeping many windows over one op is free."""
    steps = window_s / op["t_step"]
    return steps, op["per_step_flop"] * steps


# ---------------------------------------------------------------------------
# Formatting (shared by feasibility.py tables and results/plot_dashboard.py)
# ---------------------------------------------------------------------------
def fmt_time(s):
    """Single shared time formatter -- every table, CSV and the dashboard (as `_ft`) route through
    this, so one fix corrects every printed figure. Sub-120s prints SECONDS, not whole minutes:
    the max-compute tables' T_update/T_broadcast land there at large stock, where "0 min" for
    23.3s reads as zero."""
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
    """Like fmt_num, but keeps significant digits below 1 -- required-bandwidth figures genuinely
    land at a fraction of an Mbps, where fmt_num's "0" would read as free."""
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
        f"inf_tp_max: {cfg.inf_tp_max} node(s)/serving group"
        f"{' (tensor-parallel allowed; solver picks the smallest that wins)' if cfg.inf_tp_max > 1 else ' (no cross-node inference sharding)'}",
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
        f"site_power_mw: {cfg.site_power_mw:g} (facility draw, board TDP x {FACILITY_PUE:g})",
    ]
