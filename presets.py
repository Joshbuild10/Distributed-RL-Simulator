"""
presets.py -- Hardware/model constants and the published scenarios
(INTELLECT-2, INTELLECT-3, prime-rl reference run, prime-rl DeepDive).
"""

from specs import ModelSpec, RLSpec, AlgoSpec, HWSpec, NetSpec, VerifySpec, Scenario

GB = 1e9
TB = 1e12

# --- hardware constants (dense bf16 peak, no sparsity) ---
H100_NODE = dict(node_flops=8 * 989e12, node_hbm=8 * 80 * GB, node_bw=8 * 3.35 * TB)
H200_NODE = dict(node_flops=8 * 989e12, node_hbm=8 * 141 * GB, node_bw=8 * 4.8 * TB)

# --- models ---
QWQ32B = ModelSpec(  # QwQ-32B == Qwen2.5-32B backbone
    name="QwQ-32B (dense)", is_moe=False,
    p_total=31.2e9, p_active_layers=31.2e9,
    d_model=5120, n_layers=64, n_q_heads=40, n_kv_heads=8, head_dim=128,
    vocab=152064, tied_embeddings=False)

GLM45_AIR = ModelSpec(  # INTELLECT-3 base: 106B total / 12B active
    name="GLM-4.5-Air (MoE 106B/12B)", is_moe=True,
    p_total=106e9, p_active_layers=12e9,
    d_model=4096, n_layers=46, n_q_heads=96, n_kv_heads=8, head_dim=128,
    vocab=151552, tied_embeddings=False)

QWEN3_4B = ModelSpec(
    name="Qwen3-4B-Instruct (dense)", is_moe=False,
    p_total=3.6e9, p_active_layers=3.6e9,
    d_model=2560, n_layers=36, n_q_heads=32, n_kv_heads=8, head_dim=128,
    vocab=151936, tied_embeddings=False)


def intellect2(target: str = "short") -> Scenario:
    """INTELLECT-2 (arXiv 2505.07291). Decentralised swarm + small trusted trainer.

    Published anchors: 4096 samples = 256 prompts x 16; 8 optimizer steps @ batch 512;
    32K max seq; 62 GB broadcast in ~14 min (~590 Mb/s); step time ~22 min (SHORT)
    / ~21 min (LONG); inference:training compute ~4-4.5x; logprobs recomputed on the
    trainer; FSDP2 + activation recomputation; two-step asynchrony.
    """
    if target == "short":
        rmean, cv, steps, tstep = 2500.0, 0.6, 200, 22 * 60
    else:
        rmean, cv, steps, tstep = 6000.0, 0.6, 350, 21 * 60

    return Scenario(
        name=f"INTELLECT-2 (TARGET-{target.upper()})",
        model=QWQ32B,
        rl=RLSpec(prompts_per_batch=256, responses_per_prompt=16,
                  prompt_len=500, response_len_mean=rmean, response_len_cv=cv,
                  max_response_len=32768 - 500, n_steps_override=steps,
                  success_rate=0.30, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=8.0, recomp_act=1, recomp_old=1, opt_steps=8,
                      seqs_per_micro_per_gpu=1.0, in_flight_updates=True),
        train_hw=HWSpec("4x8 H100 (node count assumed)", n_nodes=4, mfu=0.40,
                        penalty_para=0.85, n_shard=32, **H100_NODE),
        inf_hw=HWSpec("decentralised swarm", n_nodes=8, mfu=0.35, bw_eff=0.85,
                      capacity_mult=1.0, **H100_NODE),
        net=NetSpec(bandwidth=590e6 / 8, latency=0.0),   # 590 Mb/s -> bytes/s
        verify=VerifySpec(mode="rule", seconds_per_rollout=0.02,
                          n_verifier_workers=1, t_env_per_rollout=0.0),
        notes="Decentralised, heterogeneous swarm; TOPLOC verification; SHARDCAST WAN broadcast.",
        published=dict(t_step=tstep, t_broadcast=14 * 60, vol_broadcast=62 * GB,
                       inf_train_gputime_ratio=4.5),
    )


def intellect3() -> Scenario:
    """INTELLECT-3 (arXiv 2512.16144). Centralised 512xH200 cluster, prime-rl.

    Published anchors: 106B/12B MoE on GLM-4.5-Air; 256 prompts x 16 rollouts;
    max context 65,536; 60 nodes x 8 H200 split 16 train / 44 infer (~1:3);
    step time ~1500 s with in-flight weight updates (>2x without); Muon lr 1e-6;
    FSDP degree 32 with FULL activation checkpointing + CPU activation offload;
    no expert parallelism; logprobs taken directly from vLLM (no trainer recompute);
    max_off_policy_steps = 8; online difficulty filtering; 400G IB (>=160 GB/s).
    """
    return Scenario(
        name="INTELLECT-3 (prime-rl, 512x H200)",
        model=GLM45_AIR,
        rl=RLSpec(prompts_per_batch=256, responses_per_prompt=16,
                  prompt_len=1500, response_len_mean=32000.0, response_len_cv=0.5,
                  max_response_len=65536 - 1500, n_steps_override=300,
                  oversample_override=2.0, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=4.0,          # Muon: lighter state than Adam
                      recomp_act=1,             # full activation checkpointing
                      recomp_old=0,             # vLLM logprobs used directly
                      opt_steps=1, seqs_per_micro_per_gpu=1.0,
                      in_flight_updates=True),
        train_hw=HWSpec("16x8 H200", n_nodes=16, mfu=0.40, penalty_para=0.85,
                        n_shard=32, **H200_NODE),
        inf_hw=HWSpec("44x8 H200 (vLLM, TP=8)", n_nodes=44, mfu=0.35,
                      bw_eff=0.85, **H200_NODE),
        net=NetSpec(bandwidth=160e9, latency=5.0),   # NCCL over 400G IB
        verify=VerifySpec(mode="rule", seconds_per_rollout=2.0,
                          n_verifier_workers=4000,   # >4000 concurrent sandboxes
                          t_env_per_rollout=0.0),
        notes=("Centralised cluster -- broadcast is NCCL/IB, not WAN. Mixed environments "
               "incl. multi-turn SWE (<=200 turns) and DeepDive search, which this "
               "single-turn model does NOT represent."),
        published=dict(t_step=1500.0, node_split="16 train / 44 infer (1:3)"),
    )


def primerl_paper() -> Scenario:
    """prime-rl reference run (Senghaas et al., NeurIPS 2025, openreview yk3ICpEbv8).

    The best-anchored of the three: single-turn math, so this model's single-turn
    assumption is exactly valid, and throughput/MFU are published as well as step time.

    [published] DeepSeek-R1-Distill-Qwen-32B (Qwen2.5-32B backbone); 24 H200 GPUs =
    8 for the trainer (1 node) + 16 for inference (DP=4, TP=4 across 2 nodes, i.e. FOUR
    TP=4 replicas); 128 prompts x 16 rollouts = 2048/step; 16,384 max context;
    micro-batch-size 1; 160 steps; lr 1e-6; async_level=1; activation checkpointing
    (--model.ac); logprobs taken from vLLM, NOT recomputed (so recomp_old=0); AIPO
    token-level objective with delta=8; skywork-math with solve-rate filter
    [0.001, 0.999] (i.e. almost no filtering, so oversample_ratio ~ 1).
    Results: trainer 11.3K +/-1K tok/s, inference 14.4K +/-1.3K tok/s, peak trainer
    MFU 38.46%, step 22.9 +/- 3.4 min, 64 h total / 1,536 GPU-hours.
    """
    # one inference "node" in this model = one TP=4 vLLM replica (4 GPUs)
    INF_REPLICA = dict(node_flops=4 * 989e12, node_hbm=4 * 141 * GB, node_bw=4 * 4.8 * TB)
    return Scenario(
        name="prime-rl reference run (R1-Distill-Qwen-32B, 24x H200)",
        model=QWQ32B,   # identical Qwen2.5-32B architecture
        rl=RLSpec(prompts_per_batch=128, responses_per_prompt=16,
                  prompt_len=400, response_len_mean=7170.0, response_len_cv=0.5,
                  max_response_len=16384 - 400, n_steps_override=160,
                  # E[R] and oversample_ratio triangulated from the two published throughputs:
                  #   trainer 11.3K tok/s * 1374 s / 2048 = 7,570 = P + E[R]
                  #   infer  14.4K tok/s * 1374 s / (2048*E[R]) = oversample_ratio = 1.35
                  oversample_override=1.35, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=8.0, recomp_act=1, recomp_old=0, opt_steps=1,
                      seqs_per_micro_per_gpu=1.0, in_flight_updates=True),
        train_hw=HWSpec("1x8 H200 trainer", n_nodes=1, gpus_per_node=8, mfu=0.40,
                        penalty_para=0.95, n_shard=8, **H200_NODE),
        inf_hw=HWSpec("4x TP=4 vLLM replicas (16 H200)", n_nodes=4, gpus_per_node=4,
                      mfu=0.35, bw_eff=0.85, **INF_REPLICA),
        net=NetSpec(bandwidth=12.5e9, latency=2.0),   # "decent direct Ethernet"
        verify=VerifySpec(mode="rule", seconds_per_rollout=0.1,
                          n_verifier_workers=64, t_env_per_rollout=0.0),
        notes="Single-turn math -- the one scenario where this model's assumptions fully hold.",
        published=dict(t_step=22.9 * 60, tok_s_train=11300.0, tok_s_inf=14400.0,
                       mfu_train=0.3846, t_total=64 * 3600),
    )


def primerl_deepdive() -> Scenario:
    """prime-rl DeepDive validation run (INTELLECT-3 report, S3.1.5).
    Qwen3-4B-Instruct-2507, 122 RL steps, group size 16, total batch 512
    (=> 32 prompts x 16). Multi-turn web-search agent. No published step time or
    node count, so this scenario is a PREDICTION, not a calibration point."""
    return Scenario(
        name="prime-rl DeepDive (Qwen3-4B, agentic search)",
        model=QWEN3_4B,
        rl=RLSpec(prompts_per_batch=32, responses_per_prompt=16,
                  prompt_len=1000, response_len_mean=8000.0, response_len_cv=0.7,
                  max_response_len=32768, n_steps_override=122,
                  success_rate=0.25, prefix_caching=True),
        algo=AlgoSpec(b_optimiser=4.0, recomp_act=1, recomp_old=0, opt_steps=1,
                      seqs_per_micro_per_gpu=1.0, in_flight_updates=True),
        train_hw=HWSpec("1x8 H200", n_nodes=1, mfu=0.40, penalty_para=0.95,
                        n_shard=8, **H200_NODE),
        inf_hw=HWSpec("1x8 H200", n_nodes=1, mfu=0.35, bw_eff=0.85, **H200_NODE),
        net=NetSpec(bandwidth=160e9, latency=2.0),
        verify=VerifySpec(mode="rule", seconds_per_rollout=0.0,
                          n_verifier_workers=1,
                          t_env_per_rollout=8.0),   # search/click tool round-trips
        notes=("Multi-turn search agent: t_env stands in crudely for tool latency. "
               "Single-turn compute model under-counts re-prefill across turns."),
        published=None,
    )
