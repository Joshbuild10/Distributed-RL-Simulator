"""
model.py -- Analytic core: rollout -> verify -> update -> broadcast.

Equations use this project's term-splitting: equivalent to the standard forms, arranged
differently. Unit conventions (seconds, bytes, FLOP, per-node hardware) are in SIMULATOR_MAP.md.
"""

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional

import penalties as pen
from specs import Scenario


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
# Typed in place of plain dicts so a stale/renamed field is a lookup error at
# the call site immediately, rather than a KeyError three files away.

@dataclass
class TrainingCompute:
    tok_batch: float
    c_layers: float
    c_attn: float
    c_update: float
    attn_frac: float


@dataclass
class TrainingMemory:
    weights: float
    grads: float
    optimiser: float
    activations: float
    act_per_gpu: float
    total: float
    grad_accum: float
    fits: bool
    headroom: float


@dataclass
class RolloutTerms:
    kv_tok: float
    mem_weights_inf: float
    mem_weights_decode: float
    kv_peak: float
    kv_expected: float
    seq_par: int
    seqs_per_wave: float
    model_fits: bool
    seq_fits: bool
    t_dec_comp: float
    t_dec_mem: float
    t_decode: float
    t_inf_attn: float
    t_prefill: float
    t_env: float
    oversample_ratio: float
    n_waves: float
    t_rollout: float
    c_rollout_total: float
    decode_bound: str
    f_tail: float = 1.0      # straggler tail on each wave (1 without penalties)


@dataclass
class TrainerPenalty:
    pp_stages: int           # nodes one replica is pipelined over (1 = fits a node)
    n_groups: int            # replicas that sync with each other
    t_comp: float            # update compute, incl. pipeline bubble + activation handoffs
    t_sync: float            # one replica sync, incl. its straggler tail
    h: int                   # local steps between syncs
    t_update: float
    eta_h: float
    eta_comp: float
    eta_act: float
    replica_loss: float      # Rahman's replica loss multiplier; reported, never folded into eta (1 at H = 1)


@dataclass
class SimResult:
    tok_s_train: float
    tok_s_inf: float
    mfu_model: float
    mfu_hw: float
    scenario: Scenario
    tc: TrainingCompute
    mem: TrainingMemory
    ro: RolloutTerms
    t_update: float
    t_verify: float
    vol_bc: float
    t_bc: float
    stages: Dict[str, float]
    t_step: float
    mode: str
    bottleneck: str
    n_steps: int
    t_total: float
    policy_age: int          # steps of policy staleness = sync_interval (k)
    flop_ratio: float
    gputime_ratio: float
    efficiency: Dict[str, float]
    tok_s_inf_achieved: float
    pen: Optional[TrainerPenalty] = None
    t_sync: float = 0.0                             # gradient all-reduce per step (0 = single replica / co-located)
    eta: float = 1.0                                # quality factor: compute the run is worth / compute done
    eta_parts: Optional[Dict[str, float]] = None
    goodput: float = 1.0                            # useful fraction of GPU-hours after failures


# ---------------------------------------------------------------------------
# Scenario-tweak helpers
# ---------------------------------------------------------------------------
# Shorthand for the nested dataclasses.replace() calls solvers/reporting/main
# all need to probe the model at a different response length, capacity, etc.

def with_response_len(s: Scenario, response_len_mean: float) -> Scenario:
    return replace(s, rl=replace(s.rl, response_len_mean=response_len_mean))


def with_capacity_mult(s: Scenario, capacity_mult: float) -> Scenario:
    return replace(s, inf_hw=replace(s.inf_hw, capacity_mult=capacity_mult))


def with_n_nodes(s: Scenario, n_nodes: int) -> Scenario:
    return replace(s, inf_hw=replace(s.inf_hw, n_nodes=n_nodes))


def with_mfu_scale(s: Scenario, k: float) -> Scenario:
    """Scale BOTH trainer and inference MFU by k. Used by the implied-MFU inverse solve: the
    published step time constrains whichever compute stage binds, so scaling both and reading off
    the binding one recovers 'what efficiency would this run need', without having to know the
    stage split in advance."""
    return replace(s, train_hw=replace(s.train_hw, mfu=s.train_hw.mfu * k),
                   inf_hw=replace(s.inf_hw, mfu=s.inf_hw.mfu * k))


# ---------------------------------------------------------------------------
# Compute terms
# ---------------------------------------------------------------------------

def training_compute(s: Scenario) -> TrainingCompute:
    m, rl, a = s.model, s.rl, s.algo

    # Total number of tokens processed in the batch
    tok_batch = rl.size_batch * rl.context_len

    # FLOPs for a forward pass through the network, excluding self-attention
    c_layers = 2.0 * m.p_active * tok_batch

    # Causal self-attention over the full context: 4 * B * T^2 * N * H * L
    if a.include_attention:
        # Calculates E[T] / expected context length using E[(P+R)^2] = P^2 + 2*P*E[R] + E[R^2].
        e_T = rl.prompt_len ** 2 + 2 * rl.prompt_len * rl.response_len_mean + rl.er2
        # Total compute used in self-attention on the given batch
        c_attn = a.causal_coef * rl.size_batch * e_T * m.attn_coef
    else: # Self-attention compute excluded
        c_attn = 0.0

    # Multiplier for the extra compute used in backpropagation, or activation recomputations.
    coef = 3 + a.recomp_act + a.recomp_old
    # Compute per training step in the layers (embedding/unembedding FLOPs treated as negligible)
    c_body = coef * (c_layers + c_attn)

    return TrainingCompute(tok_batch=tok_batch, c_layers=c_layers, c_attn=c_attn, c_update=c_body,
                            attn_frac=c_attn / max(c_layers + c_attn, 1.0))


def training_time(s: Scenario, c_update: float) -> float:
    hw = s.train_hw
    # Effective FLOPs/s during training. Accounts for efficiency losses due to node config
    flops_eff = hw.flops * hw.mfu * s.algo.tput_mult_train * hw.penalty_para * hw.n_nodes
    return c_update / flops_eff


def gradient_sync_time(s: Scenario, n_groups: int, bw: float, latency: float) -> float:
    """One gradient all-reduce across n_groups replicas: each sends and receives its compressed
    gradients. 0 for a single replica."""
    if n_groups <= 1:
        return 0.0
    vol = s.algo.b_grads * s.model.p_total / s.algo.grad_compression
    return 2 * vol / bw + latency


def trainer_penalties(s: Scenario, tc: TrainingCompute, mem: TrainingMemory) -> TrainerPenalty:
    """Rahman's trainer throughput + quality terms on top of the ideal update time."""
    p, hw, m, a = s.penalty, s.train_hw, s.model, s.algo
    inf = float("inf")

    # A replica too big for one node is pipelined over S nodes. Cross-node FSDP (n_shard > gpus/node,
    # validation presets only) also spans several nodes. Replicas are groups of both.
    stages = max(1, math.ceil(mem.total / hw.node_hbm))
    shard_nodes = max(1, math.ceil(hw.n_shard / hw.gpus_per_node))
    n_groups = hw.n_nodes // (stages * shard_nodes)
    if n_groups < 1:
        return TrainerPenalty(stages, 0, inf, inf, 1, inf, 1.0, 1.0, 1.0, 1.0)

    node_flops = hw.flops * hw.mfu * a.tput_mult_train * hw.penalty_para
    t_rep = tc.c_update / n_groups / (shard_nodes * node_flops)    # one replica's update on one stage slot
    fabric = hw.gpus_per_node * p.nic_bw_per_gpu
    if stages == 1:
        t_comp = t_rep
    else:
        # GPipe: (M + S - 1) ticks, each one stage on one micro-batch plus its activation handoff
        mb = p.micro_batches
        act = tc.tok_batch / n_groups * m.d_model * a.b_weights_train / p.pp_compression
        t_handoff = 2 * act / mb / (p.pp_bw or fabric)
        t_comp = (mb + stages - 1) * (t_rep / (stages * mb)
                                      + (t_handoff + p.pp_latency) * pen.straggler_factor(stages, p.mode))

    # Replicas exchange full gradients (up + down) over the trainer link, waiting on the slowest
    n = s.net
    bw, lat = (n.train_bandwidth, n.train_latency) if n.train_bandwidth else (fabric, p.fabric_latency)
    t_sync = gradient_sync_time(s, n_groups, bw, lat) * pen.straggler_factor(n_groups, p.mode)
    h = p.h_inner or max(1, math.ceil(t_sync / t_comp))
    t_update = max(t_comp, t_sync / h) if p.streaming else t_comp + t_sync / h

    syncs = n_groups > 1
    return TrainerPenalty(
        pp_stages=stages, n_groups=n_groups, t_comp=t_comp, t_sync=t_sync, h=h, t_update=t_update,
        eta_h=pen.eta_h(h, m.p_total) if syncs else 1.0,
        eta_comp=pen.compression_quality(a.grad_compression, p.scenario, p.error_feedback) if syncs else 1.0,
        eta_act=pen.activation_compression_quality(p.pp_compression, stages, p.scenario),
        replica_loss=pen.replica_loss_multiplier(n_groups, m.p_total, h))


def training_memory(s: Scenario) -> TrainingMemory:
    m, a, hw = s.model, s.algo, s.train_hw
    params = m.p_total

    # Per-node persistent state (weights + grads + optimiser). Sharded across the FSDP group
    # (n_shard = GPUs per group): per-GPU = state/n_shard, per-node = that x gpus_per_node.
    world_gpus = hw.n_nodes * hw.gpus_per_node
    shard_gpus = hw.n_shard
    if not (1 <= shard_gpus <= world_gpus) or world_gpus % shard_gpus:
        raise ValueError(f"n_shard={shard_gpus} invalid for {world_gpus}-GPU world "
                         f"(need 1 <= n_shard <= world and n_shard | world).")
    node_state_scale = hw.gpus_per_node / shard_gpus

    weights = a.b_weights_train * params * node_state_scale
    grads = a.b_grads * params * node_state_scale
    opt_states = a.b_optimiser * params * node_state_scale

    recomp_coef = 1 if a.recomp_act == 1 else 20

    # Activation of the grad-accumulation micro-batch. act_site_parallel: one micro-batch sharded
    # across the site (per-node, independent of node size). Else data-parallel: one per GPU.
    toks_per_micro_per_gpu = a.seqs_per_micro_per_gpu * s.rl.context_len
    act_per_gpu = 2.0 * recomp_coef * toks_per_micro_per_gpu * m.d_model * m.n_layers
    if a.act_site_parallel:
        act = act_per_gpu
        resident_per_node = a.seqs_per_micro_per_gpu
    else:
        act = act_per_gpu * hw.gpus_per_node
        resident_per_node = a.seqs_per_micro_per_gpu * hw.gpus_per_node

    mem_total = weights + grads + opt_states + act

    grad_accum = s.rl.size_batch / max(hw.n_nodes * resident_per_node, 1) / max(a.opt_steps, 1)

    return TrainingMemory(weights=weights, grads=grads, optimiser=opt_states, activations=act,
                           act_per_gpu=act_per_gpu, total=mem_total, grad_accum=grad_accum,
                           fits=mem_total <= hw.node_hbm, headroom=hw.node_hbm - mem_total)


def rollout_terms(s: Scenario) -> RolloutTerms:
    m, rl, a, hw, v = s.model, s.rl, s.algo, s.inf_hw, s.verify

    # Calculate memory requirements for model.

    # All params need to fit in HBM so memory is based on p_total.
    mem_weights_inf = a.b_weights_inf * m.p_total

    # Each decode token reads only the active weights.
    # Note that multiple experts can be activated over generation and may increase memory requirements
    # Clamped at total params to avoid double counting the head when p_active + p_head > p_total
    mem_weights_decode = min(a.b_weights_inf * (m.p_active + m.p_head),
                             a.b_weights_inf * m.p_total)

    # A group is inf_tp nodes holding one shared copy of the weights/KV cache. Only viable with fast interconnect between nodes.
    # NOTE: To add tensor parallel degree automatically inferred
    tp = max(1, hw.inf_tp)
    free_mem = tp * hw.node_hbm - mem_weights_inf
    model_fits = free_mem > 0 # If model doesn't fit, some form of sharding needs to be employed during inference

    # Effective FLOPs and bandwidth once hardware utilisation is taken into account
    bw_eff = hw.bw * hw.bw_eff
    flops_eff = hw.flops * a.tput_mult_inf * hw.mfu
    grp_bw, grp_flops = tp * bw_eff, tp * flops_eff   # pooled across the group

    # KV cache memory per token, and read/write volume over a sequence
    kv_per_tok = a.b_kv * 2.0 * m.n_layers * m.n_kv_heads * m.head_dim
    vol_kv = kv_per_tok * (rl.prompt_len * rl.response_len_mean + rl.er2 / 2.0)

    # Calculates the number of concurrent sequences per node under continuous batching.
    # This is based on either:
    #   "peak"     -- KV cache allocated for a max-context sequence (giving a conservative lower bound on concurrency)
    #   "expected" -- KV cache allocated for the mean in-flight footprint P + E[R]/2 (optimistic scheduling).
    max_ctx = rl.max_response_len or rl.context_len
    kv_peak = kv_per_tok * max_ctx
    kv_expected = kv_per_tok * (rl.prompt_len + rl.response_len_mean / 2.0)
    kv_provision = kv_peak if a.kv_provisioning == "peak" else kv_expected
    seq_fits = free_mem >= kv_peak          # A max-length sequence must still fit in memory
    seq_par = math.floor(free_mem / kv_provision) if model_fits and kv_provision > 0 else 0

    # Compute for prefill per response (Divided by the group if it's prefix-cached)
    c_pre_layers = 2.0 * m.p_active * rl.prompt_len
    # Optionally adds attention compute
    if a.include_attention:
        # Total compute used in self-attention on the given batch
        c_pre_attn = a.causal_coef * (rl.prompt_len ** 2) * m.attn_coef
    else:
        c_pre_attn = 0.0

    # Total compute used in prefill
    c_prefill = c_pre_layers + c_pre_attn
    if rl.prefix_caching:
        c_prefill /= rl.responses_per_prompt

    # Compute for decode per response
    c_dec_layers = 2.0 * m.p_active * rl.response_len_mean
    # Sum the compute in the decode steps of attention over the growing context:
    #   4 * N*H*L * sum_t (P + t)  = 4*N*H*L*(P*E[R] + E[R^2]/2)
    if a.include_attention:
        c_dec_attn = 4.0 * m.attn_coef * (rl.prompt_len * rl.response_len_mean + rl.er2 / 2.0)
    else:
        c_dec_attn = 0.0
    c_decode = c_dec_layers + c_dec_attn


    # Oversampled rollouts are generated alongside the batch, so they fill spare groups before adding waves.
    oversample_ratio = rl.oversample_ratio()
    n_gen = rl.size_batch * oversample_ratio

    # A "wave" is a continuously-batched set of sequences with rollouts generation.
    # To distribute the workload evenly across waves, we calculate the average sequences per wave
    # We allocate at least one sequence per group of connected nodes. With extras left idling.
    n_groups = max(1, hw.n_nodes // tp)
    active_groups = min(n_groups, max(1, math.ceil(n_gen))) if seq_par else 0

    n_waves = math.ceil(n_gen / max(seq_par * active_groups, 1)) if seq_par else float("inf")
    seqs_per_wave = n_gen / (active_groups * n_waves) if seq_par else float("inf")

    # Time calculations (per wave)
    t_dec_comp = (c_decode * seqs_per_wave) / grp_flops if seq_par else float("inf")
    t_dec_mem = (mem_weights_decode * rl.response_len_mean) / grp_bw if bw_eff else float("inf")
    t_decode = max(t_dec_comp, t_dec_mem) # Overall decode step time

    # KV shards over heads first (<= n_kv_heads) then over the batch, so tp keeps helping past the head
    # count at the cost of per-layer AllToAlls (not modelled). Batch shards cap at sequences in flight.
    kv_shard = min(tp, m.n_kv_heads * max(1.0, seqs_per_wave))
    t_inf_attn = (vol_kv * seqs_per_wave) / (bw_eff * kv_shard) if bw_eff else float("inf")
    t_prefill = (c_prefill * seqs_per_wave) / grp_flops if seq_par else float("inf")
    t_env = v.t_env_per_rollout

    # Overall step time (oversampling already counted in the waves). With penalties, each wave waits
    # on its slowest serving group.
    f_tail = pen.straggler_factor(active_groups, s.penalty.inf_mode) if s.penalty.enabled else 1.0
    t_rollout = (t_decode + t_inf_attn + t_prefill + t_env) * n_waves * f_tail

    # Total rollout FLOPs executed (to calculate compute-ratios). c_prefill is already per-response --
    # divided by the group above when prefix-cached -- so each response counts its prefill once either
    # way, and the total is invariant to how a fixed batch is grouped into prompts x responses.
    c_rollout_total = (c_prefill + c_decode) * rl.size_batch * oversample_ratio

    return RolloutTerms(
        kv_tok=kv_per_tok, mem_weights_inf=mem_weights_inf, mem_weights_decode=mem_weights_decode,
        kv_peak=kv_peak, kv_expected=kv_expected,
        seq_par=seq_par, seqs_per_wave=seqs_per_wave, model_fits=model_fits, seq_fits=seq_fits,
        t_dec_comp=t_dec_comp,
        t_dec_mem=t_dec_mem, t_decode=t_decode, t_inf_attn=t_inf_attn,
        t_prefill=t_prefill, t_env=t_env, oversample_ratio=oversample_ratio, n_waves=n_waves,
        t_rollout=t_rollout, c_rollout_total=c_rollout_total,
        decode_bound="memory" if t_dec_mem >= t_dec_comp else "compute", f_tail=f_tail)


# Returns the time used for verifying and scoring answers
def verify_time(s: Scenario, tok_batch: float, oversample_ratio: float) -> float:
    v, rl = s.verify, s.rl

    # Rule based verification
    if v.mode == "rule":
        return (rl.size_batch * v.seconds_per_rollout * oversample_ratio) / max(v.n_verifier_workers, 1)
    
    # Model based verification. (Only counts prefill time, generation/response assumed to be short)
    c = 2.0 * v.p_verifier * tok_batch * oversample_ratio
    return c / max(v.verifier_flops * v.verifier_mfu, 1.0)


# Returns the per-step broadcast time (assumes uniform upload/download speeds)
# Effects of the off-policy staleness (a.sync_interval = k):
#   For k=0 (on-policy) and k=1 (one-step off-policy), the full weights are broadcasted every step
#   For k>=2 the full weight volume is broadcast once every k steps, so its per-step contribution is 1/k of a full sync.
def broadcast_time(s: Scenario) -> tuple:
    m, a, n = s.model, s.algo, s.net
    vol = (a.b_weights_inf * m.p_total) / a.compression_ratio
    k = max(a.sync_interval, 1)   # k=0 and k=1 both broadcast every step (divisor 1); k>=2 amortises
    # With penalties, the fan-out waits on the slowest receiving inference node
    p = s.penalty
    f_tail = pen.straggler_factor(s.inf_hw.n_nodes, p.inf_mode) if p.enabled else 1.0
    return vol, (vol / n.bandwidth + n.latency) * f_tail / k


# Returns the number of training steps either as given, or given the number of propmts in the dataset and batch size.
def n_steps(s: Scenario) -> int:
    if s.rl.n_steps_override:
        return s.rl.n_steps_override
    if s.rl.n_prompts_total:
        return math.ceil(s.rl.n_prompts_total / s.rl.prompts_per_batch) * s.rl.n_epochs
    return 1


# ---------------------------------------------------------------------------
# Top-level simulation
# ---------------------------------------------------------------------------

def simulate(s: Scenario) -> SimResult:
    # Extracts computation times for the scenario
    tc = training_compute(s)
    mem = training_memory(s)
    p = s.penalty
    if p.enabled:
        for md in (p.mode, p.inf_mode):
            if md not in pen.MODES:
                raise ValueError(f"penalty mode {md!r} not in {pen.MODES}")
        tp = trainer_penalties(s, tc, mem)
        t_update = tp.t_update
        t_sync = tp.t_sync
    else:
        tp, t_update = None, training_time(s, tc.c_update)
        # A distributed trainer all-reduces gradients across sites every step (H = 1), overlapped
        # with compute, so the update can't finish before the sync does
        t_sync = 0.0
        if s.net.train_bandwidth:
            hw = s.train_hw
            n_rep = hw.n_nodes // max(1, math.ceil(hw.n_shard / hw.gpus_per_node))
            t_sync = gradient_sync_time(s, n_rep, s.net.train_bandwidth, s.net.train_latency)
            t_update = max(t_update, t_sync)
    ro = rollout_terms(s)
    t_verify = verify_time(s, tc.tok_batch, ro.oversample_ratio)
    vol_bc, t_bc = broadcast_time(s)


    gen_stage = ro.t_rollout + t_verify
    stages = {"rollout+verify": gen_stage, "update": t_update, "broadcast": t_bc}

    # Step composition is set by the off-policy staleness degree k = algo.sync_interval:
    #   k=0  on-policy -> the next batch needs the just-updated weights, so stages run SERIALLY
    #        (t_step = sum). k>=1 off-policy -> inference runs >=1 step behind on stale weights, so
    #        update+broadcast OVERLAP generation (t_step = max). (Supersedes in_flight_updates.)
    if s.algo.sync_interval >= 1:
        t_step = max(stages.values())
        mode = f"overlapped ({s.algo.sync_interval:g}-step off-policy, max of stages)"
    else:
        t_step = gen_stage + t_update + t_bc
        mode = "serial (on-policy, sum of stages)"

    # Calculates the longest (bottleneck) step
    bottleneck = max(stages, key=stages.get)
    steps = n_steps(s)
    
    # Policy age in steps
    policy_age = int(max(0.0, s.algo.sync_interval))

    # Compute ratios: true FLOPs vs GPU-time
    flop_ratio = ro.c_rollout_total / tc.c_update
    gputime_inf = ro.t_rollout * s.inf_hw.n_nodes * s.inf_hw.gpus_per_node
    gputime_train = t_update * s.train_hw.n_nodes * s.train_hw.gpus_per_node
    gputime_ratio = gputime_inf / max(gputime_train, 1e-9)

    # Throughput + implied MFU, for comparison against published tok/s and MFU.
    # MFU convention: MODEL FLOPs (6N, excluding rematerialisation) over hardware peak.
    model_flops = (3.0 / (3 + s.algo.recomp_act + s.algo.recomp_old)) * tc.c_update
    peak_train = s.train_hw.n_nodes * s.train_hw.flops
    # STAGE-LOCAL (not whole-step): tokens in the training batch over the trainer's own active
    # time, i.e. the rate while the trainer is actually working. This is the convention papers
    # usually quote for "trainer tok/s" (prime-rl's 11.3K matches it), and the inference side
    # mirrors it with tok_s_inf (peak) below.
    tok_s_train = tc.tok_batch / max(t_update, 1e-9)
    # Inference throughput, two conventions:
    #   PEAK     -- tokens / t_rollout: the pool's flat-out rate while actively generating.
    #   ACHIEVED -- tokens / t_step: the realised rate over the whole step. When the run is
    #     NOT rollout-bound the pool idles waiting for the trainer, so achieved < peak; this is
    #     the apples-to-apples match to a paper's reported "inference tok/s" (total gen / wall-clock).
    gen_tokens = s.rl.size_batch * s.rl.response_len_mean * ro.oversample_ratio
    tok_s_inf = gen_tokens / max(ro.t_rollout, 1e-9)
    tok_s_inf_achieved = gen_tokens / max(t_step, 1e-9)
    mfu_model = model_flops / max(t_update * peak_train, 1e-9)
    mfu_hw = tc.c_update / max(t_update * peak_train, 1e-9)

    # Penalties: goodput stretches the run; eta discounts what the compute is worth. Mitigation only
    # costs quality when replicas actually sync. A failed trainer node stalls training (Rahman's modes);
    # a failed inference node only loses its in-flight sequences, so inference keeps just the
    # node-down fraction rather than whole-cluster restart/checkpoint semantics.
    eta, eta_parts, goodput = 1.0, None, 1.0
    if tp:
        state = (s.algo.b_weights_train + s.algo.b_grads + s.algo.b_optimiser) * s.model.p_total
        train_gpus = s.train_hw.n_nodes * s.train_hw.gpus_per_node
        u_train, eta_mit = pen.reliability(train_gpus, steps * t_step, state, p.mode,
                                           p.failure_rate, p.recovery_s)
        u_inf = math.exp(-p.failure_rate * p.recovery_s / 3600.0)
        goodput = u_train * u_inf
        eta_parts = dict(h=tp.eta_h, comp=tp.eta_comp, act=tp.eta_act,
                         mit=eta_mit if tp.n_groups > 1 else 1.0)
        eta = math.prod(eta_parts.values())

    return SimResult(
        tok_s_train=tok_s_train, tok_s_inf=tok_s_inf,
        mfu_model=mfu_model, mfu_hw=mfu_hw,
        scenario=s, tc=tc, mem=mem, ro=ro, t_update=t_update, t_verify=t_verify,
        vol_bc=vol_bc, t_bc=t_bc, stages=stages, t_step=t_step, mode=mode,
        bottleneck=bottleneck, n_steps=steps, t_total=steps * t_step / max(goodput, 1e-12),
        policy_age=policy_age, flop_ratio=flop_ratio,
        gputime_ratio=gputime_ratio,
        efficiency={k: v / t_step for k, v in stages.items()},
        tok_s_inf_achieved=tok_s_inf_achieved,
        pen=tp, t_sync=t_sync, eta=eta, eta_parts=eta_parts, goodput=goodput,
    )
