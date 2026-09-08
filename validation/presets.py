"""
presets.py -- Hardware/model constants and the published scenarios
(INTELLECT-2, INTELLECT-3, prime-rl reference run, prime-rl DeepDive).
"""

from dataclasses import replace

from specs import ModelSpec, RLSpec, AlgoSpec, HWSpec, NetSpec, VerifySpec, Scenario

GB = 1e9
TB = 1e12

# --- hardware constants (dense bf16 peak, no sparsity) ---
H100_NODE = dict(node_flops=8 * 989e12, node_hbm=8 * 80 * GB, node_bw=8 * 3.35 * TB)
H200_NODE = dict(node_flops=8 * 989e12, node_hbm=8 * 141 * GB, node_bw=8 * 4.8 * TB)
# H800 = H100 compute (989 TF bf16) + HBM (80 GB @ 3.35 TB/s); only interconnect is cut.
H800_NODE = dict(node_flops=8 * 989e12, node_hbm=8 * 80 * GB, node_bw=8 * 3.35 * TB)

# --- Models ---
# For INTELLECT-2's runs. 
# Architecture description: https://huggingface.co/Qwen/Qwen2.5-32B, https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json
QWQ32B = ModelSpec(  # QwQ-32B == Qwen2.5-32B backbone
    name="QwQ-32B (dense)", is_moe=False,
    p_total=32.5e9, p_active=31.0e9,
    d_model=5120, n_layers=64, n_q_heads=40, n_kv_heads=8, head_dim=128,
    vocab=152064, tied_embeddings=False)

# For INTELLECT-3
# Architecture description: https://arxiv.org/abs/2508.06471, https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json 
GLM45_AIR = ModelSpec(  # INTELLECT-3 base: 106B total / 12B active
    name="GLM-4.5-Air (MoE 106B/12B)", is_moe=True,
    p_total=106e9, p_active=12e9,
    d_model=4096, n_layers=46, n_q_heads=96, n_kv_heads=8, head_dim=128,
    vocab=151552, tied_embeddings=False)

# For prime-rl
# Architecture description: https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json
QWEN3_4B = ModelSpec(
    name="Qwen3-4B-Instruct (dense)", is_moe=False,
    p_total=4.0e9, p_active=3.6e9,
    d_model=2560, n_layers=36, n_q_heads=32, n_kv_heads=8, head_dim=128,
    vocab=151936, tied_embeddings=True)


def intellect2(target: str = "short") -> Scenario:
    """INTELLECT-2 (https://arxiv.org/abs/2505.07291). Decentralised inference swarm + small trusted trainer.

    Training setup: 4096 samples = 256 prompts x 16 responses per prompt, 8 optimizer steps in batches of 512,
    32K max seq.
    Logprobs are recomputed on the trainer, FSDP2 + activation recomputation, two-step asynchrony.
    62 GB of weights broadcast in ~14 min (~590 Mb/s).
    Step times: (SHORT) ~22 min. 22 min for rollouts, 22 min training, 
    (LONG) ~21 min. 29 min for rollouts, 21 min training.
    Reported training:inference compute ~4-4.5x. (FLOPs or wall-clock?)
    """
    if target == "short":
        rmean, cv, steps, tstep = 2500.0, 0.5, 200, 22 * 60
    else:
        rmean, cv, steps, tstep = 6000.0, 0.5, 350, 21 * 60

    # https://huggingface.co/datasets/PrimeIntellect/INTELLECT-2-RL-Dataset. Mean prompt length
    return Scenario(
        name=f"INTELLECT-2 (TARGET-{target.upper()})",
        model=QWQ32B,
        rl=RLSpec(prompts_per_batch=256, responses_per_prompt=16,
                  prompt_len=150, response_len_mean=rmean, response_len_cv=cv,
                  max_response_len=32000 - 150, n_steps_override=steps, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=8.0, recomp_act=1, recomp_old=1, opt_steps=8,
                      seqs_per_micro_per_gpu=1.0),
        train_hw=HWSpec("4x8 H100 (node count assumed)", n_nodes=4, mfu=0.40,
                        penalty_para=1, n_shard=32, **H100_NODE),
        inf_hw=HWSpec("decentralised swarm", n_nodes=8, mfu=0.25, bw_eff=0.85,
                      capacity_mult=1.0, **H100_NODE),
        net=NetSpec(bandwidth=590e6 / 8, latency=0.1),   # 590 Mb/s -> bytes/s
        verify=VerifySpec(mode="rule", seconds_per_rollout=0.02,
                          n_verifier_workers=1, t_env_per_rollout=0.0),
        notes="Decentralised, heterogeneous swarm; TOPLOC verification; SHARDCAST WAN broadcast.",
        published=dict(t_step=tstep, t_broadcast=14 * 60, vol_broadcast=62 * GB,
                       inf_train_gputime_ratio=4.5),
    )


def intellect3() -> Scenario:
    """INTELLECT-3 (https://arxiv.org/abs/2512.16144 ). Centralised 512xH200 cluster, prime-rl.

    Published anchors: 106B/12B MoE on GLM-4.5-Air; 256 prompts x 16 rollouts;
    max context 65k; 60 nodes x 8 H200 split 16 train / 44 infer (~1:3);
    step time ~1500 s with in-flight weight updates (>2x without); Muon lr 1e-6;
    FSDP degree 32 with FULL activation checkpointing + CPU activation offload;
    no expert parallelism; logprobs taken directly from vLLM (no trainer recompute);
    max_off_policy_steps = 8; online difficulty filtering; 400G IB (>=160 GB/s).
    """
    return Scenario(
        name="INTELLECT-3 (prime-rl, 512x H200)",
        model=GLM45_AIR,
        # === z-ai/glm-4.5-air on INTELLECT-3-RL:code @ T=1.0, cap=49,152 ===
        # samples 256 | censored at cap: 2 (0.8%)
        # mean 22,961  [95% CI 19,665 - 26,192]  (+/-14%, cluster-robust)
        # median 23,063 | std 10,724 | cv 0.47
        # p50 23,170 | p90 36,133 | p95 38,542 | p99 49,152 | max 53,641
        # variance: within-prompt sd 5,758 | between-prompt sd 9,201 | ICC 0.72
        # --> presets.py:  response_len_mean=22,961, response_len_cv=0.47, max_response_len=53,641
        rl=RLSpec(prompts_per_batch=256, responses_per_prompt=16,
                  prompt_len=1000, response_len_mean=23000.0, response_len_cv=0.5,
                  max_response_len=65000 - 1000, n_steps_override=600,
                  oversample_override=1.0, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=4.0, recomp_act=1, recomp_old=0, # Muon optimiser
                      opt_steps=1, seqs_per_micro_per_gpu=1.0),
        train_hw=HWSpec("16x8 H200", n_nodes=16, mfu=0.17, penalty_para=1,  # MoE @16nd, nanotron 70-80B x0.6
                        n_shard=32, **H200_NODE),
        inf_hw=HWSpec("44x8 H200 (vLLM, TP=8)", n_nodes=44, mfu=0.25,
                      bw_eff=0.85, **H200_NODE),
        net=NetSpec(bandwidth=160e9, latency=5.0),   # NCCL over 400G IB
        verify=VerifySpec(mode="rule", seconds_per_rollout=2.0,
                          n_verifier_workers=4000,
                          t_env_per_rollout=0.0),
        notes=("Centralised cluster -- broadcast is NCCL/IB, not WAN. Mixed environments "
               "includes multi-turn SWE (<=200 turns) and DeepDive search, which the model does NOT represent."),
        published=dict(t_step=1500.0, node_split="16 train / 44 infer (1:3)"),
    )


def primerl_paper() -> Scenario:
    """prime-rl reference run (https://openreview.net/forum?id=yk3ICpEbv8 openreview yk3ICpEbv8).

    The best-anchored of the three: single-turn math, so this model's single-turn
    assumption is exactly valid, and throughput/MFU are published as well as step time.

    [published] DeepSeek-R1-Distill-Qwen-32B (Qwen2.5-32B backbone).
    24 H200 GPUs = 8 trainer (1 node) + 16 inference (DP=4, TP=4 across 2 nodes, i.e. FOUR TP=4 replicas)
    128 prompts x 16 rollouts = 2048/step; 16,384 max context;
    160 steps, async_level=1, activation checkpointing
    Results: trainer 11.3K +/-1K tok/s, inference 14.4K +/-1.3K tok/s, peak trainer
    MFU 38.46%, step 22.9 +/- 3.4 min, 64 h total / 1,536 GPU-hours.
    """
    # one inference "node" in this model = one TP=4 vLLM replica (4 GPUs)
    INF_REPLICA = dict(node_flops=4 * 989e12, node_hbm=4 * 141 * GB, node_bw=4 * 4.8 * TB)
    
    return Scenario(
        name="prime-rl reference run (R1-Distill-Qwen-32B, 24x H200)",
        model=QWQ32B,   # Qwen2.5-32B
        # E[R] and oversample_ratio calculated from the published throughputs:
        #   trainer 11.3K tok/s * 1374 s / 2048 = 7,570 = P + E[R]
        #   infer  14.4K tok/s * 1374 s / (2048*E[R]) = oversample_ratio = 1.35
        
        #   Estimated with Qwen/Qwen3-32B on Skywork-OR1 math (prime-rl data, filtered) @ T=1.0, cap=32,768 ===
        #   samples 256 | censored at cap: 0 (0.0%)
        #   mean 7,536  [95% CI 6,720 - 8,311]  (+/-11%, cluster-robust)
        #   median 7,637 | std 2,543 | cv 0.34
        #   p50 7,651 | p90 10,737 | p95 11,372 | p99 12,152 | max 12,942
        #   variance: within-prompt sd 1,207 | between-prompt sd 2,276 | ICC 0.78
        #   --> presets.py:  response_len_mean=7,536, response_len_cv=0.34, max_response_len=12,942
        rl=RLSpec(prompts_per_batch=128, responses_per_prompt=16,
                  prompt_len=100, response_len_mean=7470.0, response_len_cv=0.35,
                  max_response_len=16384 - 100, n_steps_override=160,
                  oversample_override=1.35, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=8.0, recomp_act=1, recomp_old=0, opt_steps=1,
                      seqs_per_micro_per_gpu=1.0),
        train_hw=HWSpec("1x8 H200 trainer", n_nodes=1, gpus_per_node=8, mfu=0.3846,
                        penalty_para=1.00, n_shard=8, **H200_NODE),
        inf_hw=HWSpec("4x TP=4 vLLM replicas (16 H200)", n_nodes=4, gpus_per_node=4,
                      mfu=0.25, bw_eff=0.85, **INF_REPLICA),
        net=NetSpec(bandwidth=1e9, latency=0.2),   # "decent direct Ethernet"
        verify=VerifySpec(mode="rule", seconds_per_rollout=0.1,
                          n_verifier_workers=64, t_env_per_rollout=0.0),
        notes="Single-turn math dataset https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data",
        published=dict(t_step=22.9 * 60, tok_s_train=11300.0, tok_s_inf=14400.0,
                       mfu_train=0.3846, t_total=64 * 3600),
    )


def primerl_deepdive() -> Scenario:
    """prime-rl DeepDive validation run (INTELLECT-3 report, S3.1.5).
    Qwen3-4B-Instruct-2507, 122 RL steps, group size 16, total batch 512
    (=> 32 prompts x 16). Multi-turn web-search agent.
    No published step time or node count, so this scenario is a prediction."""
    return Scenario(
        name="prime-rl DeepDive (Qwen3-4B, agentic search)",
        model=QWEN3_4B,
        rl=RLSpec(prompts_per_batch=32, responses_per_prompt=16,
                  prompt_len=1000, response_len_mean=8000.0, response_len_cv=0.5,
                  max_response_len=32768, n_steps_override=122,
                  success_rate=0.25, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=4.0, recomp_act=1, recomp_old=0, opt_steps=1,
                      seqs_per_micro_per_gpu=1.0),
        train_hw=HWSpec("1x8 H200", n_nodes=1, mfu=0.40, penalty_para=1,
                        n_shard=8, **H200_NODE),
        inf_hw=HWSpec("1x8 H200", n_nodes=1, mfu=0.25, bw_eff=0.85, **H200_NODE),
        net=NetSpec(bandwidth=160e9, latency=0.1),
        verify=VerifySpec(mode="rule", seconds_per_rollout=0.0,
                          n_verifier_workers=1, t_env_per_rollout=8.0),   # search/click tool round-trips
        notes=("Multi-turn search agent: t_env stands in crudely for tool latency. "
               "Single-turn compute model under-counts re-prefill across turns."),
        published=None,
    )


# ---------------------------------------------------------------------------
# AReaL (arXiv 2505.24298) -- disaggregated ASYNC RL, a 4-point scale anchor.
# ---------------------------------------------------------------------------
# Publishes total training hours AND PPO step counts, so per-step wall-clock is
# recoverable, across R1-Distill-Qwen at four sizes on one H800 setup. This is the
# model's home regime (disaggregated async).
# Architectures: DeepSeek-R1-Distill-Qwen2.5-{1.5B,7B,14B,32B}.
# CAVEAT: AReaL does not publish the average response length E[R]; the E[R] here is
# assumed, so each run is really validated by whether the E[R] that reproduces the
# published step time comes out plausible (< the 32K cap) -- see drl_model.py.

R1_QWEN_1_5B = ModelSpec(
    name="R1-Distill-Qwen-1.5B", is_moe=False,
    p_total=1.78e9, p_active=1.31e9,
    d_model=1536, n_layers=28, n_q_heads=12, n_kv_heads=2, head_dim=128,
    vocab=151936, tied_embeddings=False)

R1_QWEN_7B = ModelSpec(          # Qwen2.5-Math-7B backbone
    name="R1-Distill-Qwen-7B", is_moe=False,
    p_total=7.61e9, p_active=6.52e9,
    d_model=3584, n_layers=28, n_q_heads=28, n_kv_heads=4, head_dim=128,
    vocab=152064, tied_embeddings=False)

R1_QWEN_14B = ModelSpec(
    name="R1-Distill-Qwen-14B", is_moe=False,
    p_total=14.77e9, p_active=13.2e9,
    d_model=5120, n_layers=48, n_q_heads=40, n_kv_heads=8, head_dim=128,
    vocab=152064, tied_embeddings=False)

R1_QWEN_32B = replace(QWQ32B, name="R1-Distill-Qwen-32B")   # Qwen2.5-32B backbone

# Presets for the range of runs in the AReaL paper https://arxiv.org/pdf/2505.24298
def _areal(model: ModelSpec, n_nodes_total: int, steps: int, total_hours: float,
           er_mean: float, task: str, mfu_train: float, inf_tp: int = 8, er_cv: float = 0.5) -> Scenario:
    """One AReaL run. 1/4 of nodes train, 3/4 infer (async, in-flight).
    512 prompts x 16 responses per prompt = 8192 batch size. Max prompt 1024; 32K max context length;
    4 PPO minibatches; H800 nodes; per-step = total_hours*3600/steps.

    mfu_train is the per-config trainer MFU derived from the Ultra-Scale Playbook node-matched
    RL-like (FSDP) frontier -- see validation/uncertainty.py mfu_train_tri(), which centres its
    Monte-Carlo prior on exactly this value. It is NOT the flat 0.40 nominal: a data-grounded MFU
    is a better deterministic input, so the point prediction reflects it too."""

    train_nodes = round(n_nodes_total * 0.25)
    inf_nodes = n_nodes_total - train_nodes
    inf_gpus = inf_nodes * 8
    per_step = total_hours * 3600 / steps
    inf_node = dict(node_flops=inf_tp * 989e12, node_hbm=inf_tp * 80 * GB, node_bw=inf_tp * 3.35 * TB)

    return Scenario(
        name=f"AReaL {model.name} ({task}, {n_nodes_total} H800 nodes)",
        model=model,
        rl=RLSpec(prompts_per_batch=512, responses_per_prompt=16,
                  prompt_len=1000, response_len_mean=er_mean, response_len_cv=er_cv,
                  max_response_len=32000 - 1000, n_steps_override=steps,
                  oversample_override=1.0, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=8.0, recomp_act=1, recomp_old=0, opt_steps=4,
                      seqs_per_micro_per_gpu=1.0),
        train_hw=HWSpec(f"{train_nodes}x8 H800 (FSDP)", n_nodes=train_nodes,
                        gpus_per_node=8, mfu=mfu_train, penalty_para=1,
                        n_shard=train_nodes * 8, **H800_NODE),
        inf_hw=HWSpec(f"{inf_gpus} H800 (TP={inf_tp})", n_nodes=inf_gpus // inf_tp,
                      gpus_per_node=inf_tp, mfu=0.25, bw_eff=0.85, **inf_node),
        net=NetSpec(bandwidth=400e9, latency=0.1),   # 3.2 Tbps RoCE, centralised
        verify=VerifySpec(mode="rule", seconds_per_rollout=0.05, n_verifier_workers=512),
        notes=f"AReaL async disaggregated PPO; {task}. E[R]={er_mean:,.0f} from measured {task} reasoning traces.",
        published=dict(t_step=per_step, t_total=total_hours * 60 * 60),
    )



def areal_1_5b(): return _areal(R1_QWEN_1_5B, 16, 250, 14.8, 10000.0,  inf_tp=1, task="math", er_cv=0.8, mfu_train=0.31)
# 7B: Mean response measured at 10,125, cv=0.77
#  === R1-Distill-Qwen-7B on DeepScaleR (math) @ T=1.0, cap=32768 ===
#  samples 256  | truncated at cap: 6 (2%)
#  mean 10,125 | median 7,040 | std 7,747 | cv 0.77
#  p50 7,069 | p90 21,434 | p95 27,032 | p99 32,768 | max 32,768
def areal_7b():   return _areal(R1_QWEN_7B,   24, 250, 25.4, 10000.0, task="math", er_cv=0.80, mfu_train=0.34)
# === deepseek-ai/DeepSeek-R1-Distill-Qwen-14B on DeepCoder pooled (code) @ T=1.0, cap=32,768 ===
#   samples 512 | censored at cap: 0 (0.0%)
#   mean 10,376  [95% CI 9,339 - 11,395]  (+/-10%, cluster-robust)
#   median 10,616 | std 4,638 | cv 0.45
#   p50 10,636 | p90 16,374 | p95 17,448 | p99 19,800 | max 22,583
#   variance: within-prompt sd 2,207 | between-prompt sd 4,113 | ICC 0.78
def areal_14b():  return _areal(R1_QWEN_14B,  32,  80, 21.9, 10500.0, task="code", er_cv=0.45, mfu_train=0.32)
# === Qwen/Qwen3-32B on DeepCoder pooled (code) @ T=1.0, cap=32,768 ===
# Not exact model match.
#  samples 256 | censored at cap: 0 (0.0%)
#  mean 8,137  [95% CI 7,465 - 8,752]  (+/-8%, cluster-robust)
#  median 7,858 | std 2,285 | cv 0.28
#  p50 7,866 | p90 11,190 | p95 11,863 | p99 13,785 | max 14,896
#  variance: within-prompt sd 1,450 | between-prompt sd 1,796 | ICC 0.61
def areal_32b():  return _areal(R1_QWEN_32B,  48,  60, 31.1, 8000.0, task="code", er_cv=0.30, mfu_train=0.28)
