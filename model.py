"""
model.py -- Analytic core: rollout -> verify -> update -> broadcast.

Equations follow the user's term-splitting (equivalent to, but arranged
differently from, the standard forms). See drl_model.py module docstring
for the conventions these functions assume.
"""

import math
from dataclasses import dataclass, replace
from typing import Dict

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
    kv_peak: float
    kv_expected: float
    seq_par: int
    seqs_per_wave: float
    model_fits: bool
    seq_fits: bool
    c_prefill: float
    c_decode: float
    c_dec_attn: float
    vol_kv: float
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


@dataclass
class SimResult:
    n_train_gpus: int
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
    staleness: int
    flop_ratio: float
    gputime_ratio: float
    efficiency: Dict[str, float]
    tok_s_inf_achieved: float


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


# ---------------------------------------------------------------------------
# Compute terms
# ---------------------------------------------------------------------------

def training_compute(s: Scenario) -> TrainingCompute:
    m, rl, a = s.model, s.rl, s.algo

    # Total number of tokens processed in the batch
    tok_batch = rl.size_batch * rl.context_len

    # FLOPs for a forward pass through the network, excluding self-attention
    c_layers = 2.0 * m.p_active_layers * tok_batch

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


def training_memory(s: Scenario) -> TrainingMemory:
    m, a, hw = s.model, s.algo, s.train_hw
    params = m.p_total

    # Calculates memory per node to stores the weights, gradients and optimiser states
    weights = a.b_weights_train * params / hw.n_shard
    grads = a.b_grads * params / hw.n_shard
    opt_states = a.b_optimiser * params / hw.n_shard

    recomp_coef = 1 if a.recomp_act == 1 else 20

    # Activation memory is set by the gradient-accumulation micro-batch resident on one
    # GPU, not by the whole rollout batch.
    toks_per_micro_per_gpu = a.seqs_per_micro_per_gpu * s.rl.context_len
    act_per_gpu = 2.0 * recomp_coef * toks_per_micro_per_gpu * m.d_model * m.n_layers
    act = act_per_gpu * hw.gpus_per_node

    mem_total = weights + grads + opt_states + act

    n_gpus = hw.n_nodes * hw.gpus_per_node

    grad_accum = s.rl.size_batch / max(n_gpus * a.seqs_per_micro_per_gpu, 1) / max(a.opt_steps, 1)

    return TrainingMemory(weights=weights, grads=grads, optimiser=opt_states, activations=act,
                           act_per_gpu=act_per_gpu, total=mem_total, grad_accum=grad_accum,
                           fits=mem_total <= hw.node_hbm, headroom=hw.node_hbm - mem_total)


def rollout_terms(s: Scenario) -> RolloutTerms:
    m, rl, a, hw, v = s.model, s.rl, s.algo, s.inf_hw, s.verify
    # Calculate memory requirements for model
    mem_weights_inf = a.b_weights_inf * m.p_total
    free_mem = hw.node_hbm - mem_weights_inf
    model_fits = free_mem > 0 # If model doesn't fit, some form of sharding needs to be employed during inference

    # Effective FLOPs and bandwidth once hardware utilisation is taken into account
    bw_eff = hw.bw * hw.bw_eff
    flops_eff = hw.flops * a.tput_mult_inf * hw.mfu

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
    c_pre_layers = 2.0 * m.p_active_layers * rl.prompt_len
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
    c_dec_layers = 2.0 * m.p_active_layers * rl.response_len_mean
    # Sum the compute in the decode steps of attention over the growing context:
    #   4 * N*H*L * sum_t (P + t)  = 4*N*H*L*(P*E[R] + E[R^2]/2)
    if a.include_attention:
        c_dec_attn = 4.0 * m.attn_coef * (rl.prompt_len * rl.response_len_mean + rl.er2 / 2.0)
    else:
        c_dec_attn = 0.0
    c_decode = c_dec_layers + c_dec_attn


    # A "wave" is a continuously-batched set of sequences with rollouts generation.
    # To distribute the workload evenly across waves, we calculate the average sequences per wave
    n_waves = math.ceil(rl.size_batch / max(seq_par * hw.n_nodes, 1)) if seq_par else float("inf")
    seqs_per_wave = rl.size_batch / (hw.n_nodes * n_waves) if seq_par else float("inf")

    # Time calculations (per wave)
    t_dec_comp = (c_decode * seqs_per_wave) / flops_eff if seq_par else float("inf")
    t_dec_mem = (mem_weights_inf * rl.response_len_mean) / bw_eff if bw_eff else float("inf")
    t_decode = max(t_dec_comp, t_dec_mem) # Overall decode step time

    t_inf_attn = (vol_kv * seqs_per_wave) / bw_eff if bw_eff else float("inf")
    t_prefill = (c_prefill * seqs_per_wave) / flops_eff if seq_par else float("inf")
    t_env = v.t_env_per_rollout

    oversample_ratio = rl.oversample_ratio()

    # Overall step time
    t_rollout = (t_decode + t_inf_attn + t_prefill + t_env) * n_waves * oversample_ratio

    # Total rollout FLOPs executed (to calculate compute-ratios)
    if rl.prefix_caching:
        c_rollout_total = (c_decode * rl.size_batch + c_prefill * rl.size_batch) * oversample_ratio
    else:
        c_rollout_total = (c_prefill * rl.responses_per_prompt + c_decode) * rl.size_batch * oversample_ratio

    return RolloutTerms(
        kv_tok=kv_per_tok, mem_weights_inf=mem_weights_inf, kv_peak=kv_peak, kv_expected=kv_expected,
        seq_par=seq_par, seqs_per_wave=seqs_per_wave, model_fits=model_fits, seq_fits=seq_fits,
        c_prefill=c_prefill, c_decode=c_decode,
        c_dec_attn=c_dec_attn, vol_kv=vol_kv, t_dec_comp=t_dec_comp,
        t_dec_mem=t_dec_mem, t_decode=t_decode, t_inf_attn=t_inf_attn,
        t_prefill=t_prefill, t_env=t_env, oversample_ratio=oversample_ratio, n_waves=n_waves,
        t_rollout=t_rollout, c_rollout_total=c_rollout_total,
        decode_bound="memory" if t_dec_mem >= t_dec_comp else "compute")


# Returns the time used for verifying and scoring answers
def verify_time(s: Scenario, tok_batch: float, oversample_ratio: float) -> float:
    v, rl = s.verify, s.rl

    # Rule based verification
    if v.mode == "rule":
        return (rl.size_batch * v.seconds_per_rollout * oversample_ratio) / max(v.n_verifier_workers, 1)
    
    # Model based verification. (Only counts prefill time, generation/response assumed to be short)
    c = 2.0 * v.p_verifier * tok_batch * oversample_ratio
    return c / max(v.verifier_flops * v.verifier_mfu, 1.0)


# Returns the time used for weight broadcasting (assumes uniform upload/download speeds)
def broadcast_time(s: Scenario) -> tuple:
    m, a, n = s.model, s.algo, s.net
    vol = (a.b_weights_inf * m.p_total) / a.compression_ratio
    return vol, vol / n.bandwidth + n.latency


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
    t_update = training_time(s, tc.c_update)
    mem = training_memory(s)
    ro = rollout_terms(s)
    t_verify = verify_time(s, tc.tok_batch, ro.oversample_ratio)
    vol_bc, t_bc = broadcast_time(s)


    gen_stage = ro.t_rollout + t_verify
    stages = {"rollout+verify": gen_stage, "update": t_update, "broadcast": t_bc}

    # Calculates step time depending on if in-flight weight updates are conducted (overlapping times), or not
    if s.algo.in_flight_updates:
        t_step = max(stages.values())
        mode = "overlapped (max of stages)"
    else:
        t_step = gen_stage + t_update + t_bc
        mode = "serial (sum of stages)"

    # Calculates the longest (bottleneck) step
    bottleneck = max(stages, key=stages.get)
    steps = n_steps(s)
    
    # Calculates staleness
    staleness = math.ceil((t_bc + gen_stage) / t_step) if t_step > 0 else 0

    # Compute ratios: true FLOPs vs GPU-time
    flop_ratio = ro.c_rollout_total / tc.c_update
    gputime_inf = ro.t_rollout * s.inf_hw.n_nodes * s.inf_hw.gpus_per_node
    gputime_train = t_update * s.train_hw.n_nodes * s.train_hw.gpus_per_node
    gputime_ratio = gputime_inf / max(gputime_train, 1e-9)

    # Throughput + implied MFU, for comparison against published tok/s and MFU.
    # MFU convention: MODEL FLOPs (6N, excluding rematerialisation) over hardware peak.
    n_train_gpus = s.train_hw.n_nodes * s.train_hw.gpus_per_node
    model_flops = (3.0 / (3 + s.algo.recomp_act + s.algo.recomp_old)) * tc.c_update
    peak_train = s.train_hw.n_nodes * s.train_hw.flops
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

    return SimResult(
        n_train_gpus=n_train_gpus, tok_s_train=tok_s_train, tok_s_inf=tok_s_inf,
        mfu_model=mfu_model, mfu_hw=mfu_hw,
        scenario=s, tc=tc, mem=mem, ro=ro, t_update=t_update, t_verify=t_verify,
        vol_bc=vol_bc, t_bc=t_bc, stages=stages, t_step=t_step, mode=mode,
        bottleneck=bottleneck, n_steps=steps, t_total=steps * t_step,
        staleness=staleness, flop_ratio=flop_ratio,
        gputime_ratio=gputime_ratio,
        efficiency={k: v / t_step for k, v in stages.items()},
        tok_s_inf_achieved=tok_s_inf_achieved,
    )
