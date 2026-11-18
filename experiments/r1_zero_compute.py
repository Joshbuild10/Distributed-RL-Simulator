#!/usr/bin/env python3
"""
r1_zero_compute.py -- DeepSeek-R1-Zero: total RL training FLOPs (via this repo's own
compute-accounting core), raw GPU FLOPs available over the disclosed run, and the
implied MFU. Source: arXiv 2501.12948 (fetched + extracted directly from the PDF,
2026-08-16) + HF deepseek-ai/DeepSeek-V3-Base/config.json.

DISCLOSED (quoted/paraphrased from the paper; page/section noted):
  - GRPO. "For each question, we sample 16 outputs with a maximum length of 32,768
    tokens before the 8.2k step and 65,536 tokens afterward... training continuing
    for a total of 10,400 steps, corresponding to 1.6 training epochs. Each training
    step consists of 32 unique questions, resulting in a training batch size of 512."
    (Sec 2.1) -> G=16, 32 questions/step, size_batch=512, n_steps=10,400.
  - "For the training of DeepSeek-R1-Zero, we employed 64*8 H800 GPUs, and the
    process required approximately 198 hours." (Appendix B.4.4) -> 512 GPUs, 198h.
  - Table 7: 101K H800 GPU-hours for R1-Zero. Check: 512*198 = 101,376 -- matches.
  - Base model DeepSeek-V3-Base: 671B total / 37B activated params, MoE (paper
    abstract + arXiv 2412.19437). Architecture (HF config.json, deepseek-ai/
    DeepSeek-V3-Base): hidden_size=7168, num_hidden_layers=61, num_attention_heads=128,
    vocab_size=129280, tie_word_embeddings=false. Uses MLA (Multi-head Latent
    Attention): qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
    kv_lora_rank=512.

NOT DISCLOSED (exposed as explicit swept/assumed inputs, never fabricated as fact):
  - Average response length E[R] during training. Figure 1(b) shows a RISING curve
    on a 0-20,000-token y-axis with no numeric data points given in text -- only the
    axis range is readable from the extracted PDF. SWEPT below across a bracket
    spanning that axis, in place of inventing a single number.
  - Mean prompt length for R1-Zero's math/code/STEM prompt set -- not stated.
    Assumed 500 tok (order-of-magnitude consistent with this repo's other presets:
    INTELLECT-2 150, prime-rl 100, AReaL max-prompt 1024); swept 200-1000.
  - RL-stage training precision. DeepSeek-V3 PRETRAINING is disclosed as
    substantially FP8; the RL stage's precision is not stated. MFU below uses this
    repo's standard BF16-peak convention (989 TFLOP/s/GPU, presets.py H800_NODE).
    If the RL stage in fact runs FP8 (~2x BF16 peak), implied MFU roughly HALVES.
    Flagged, not resolved.
  - Reference-model KL forward pass. GRPO's objective (paper Figure 3/Eq 1) penalises
    KL against a frozen reference model, updated every 400 steps -- an EXTRA
    forward-only pass over the batch not captured by the "3+recomp_act+recomp_old"
    coefficient below. Roughly +1 forward-equivalent -> if included, c_update rises
    ~20-25% and implied MFU falls by the same fraction. Reported as a bracket, not
    baked in.

ARCHITECTURE CAVEAT -- MLA, not plain GQA (do not silently normalise away):
  This repo's ModelSpec/model.py assume standard MHA/GQA, where the QK^T and score x V
  matmuls share one symmetric head_dim H (model.py's c_attn: causal_coef(2 or 4) x
  T^2 x n_q_heads x H x n_layers, i.e. the non-causal base 4H = 2H(QK) + 2H(AV)).
  DeepSeek-V3's MLA is ASYMMETRIC: QK^T operates on qk_nope_head_dim+qk_rope_head_dim
  = 128+64 = 192; score x V operates on v_head_dim = 128. Matching the model's own
  FLOP convention (2H_qk + 2H_v vs the symmetric 4H) requires an EFFECTIVE
  head_dim = (192+128)/2 = 160, used ONLY for the attention-FLOP terms below.
  This is EXACT for the FLOP tally computed here. It is NOT a faithful KV-cache size
  (MLA's real cache is one compressed 512+64=576-value latent per token per layer,
  not per-head K,V) -- n_kv_heads/head_dim as set below are placeholders reused only
  for the attn_coef property; no KV-cache, seq_par or timing output is computed or
  reported by this script, only FLOP TALLIES (training_compute/rollout_terms), so
  that mismatch does not contaminate anything reported here. A faithful MLA KV-cache
  mode would be a real specs.py/model.py extension -- out of scope for this script.

METHOD: pure FLOP-tally, not a timing simulation. total FLOPs = n_steps *
(c_update + c_rollout_total), using model.py's training_compute()/rollout_terms()
directly (no t_step/broadcast/verify timing -- those need system details, e.g.
serial vs overlapped execution, colocation of train/inference pools, that the paper
does not disclose, and are irrelevant to a FLOP count). raw GPU FLOPs = n_gpus x
peak_flops/s x wall_clock_s. implied MFU = total FLOPs / raw GPU FLOPs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from specs import ModelSpec, RLSpec, AlgoSpec, Scenario, HWSpec, NetSpec, VerifySpec
from model import training_compute, rollout_terms

GB, TB = 1e9, 1e12

# --- disclosed hardware / run scale ---
N_GPUS = 512                 # 64 nodes x 8 H800
WALL_HOURS = 198.0
WALL_S = WALL_HOURS * 3600
GPU_PEAK_BF16 = 989e12        # FLOP/s per H800 (BF16; matches this repo's H800_NODE convention)
N_STEPS = 10_400
G = 16
SIZE_BATCH = 32 * G           # 512 rollouts/step (Sec 2.1)

# --- MLA-effective head_dim derivation (see module docstring) ---
QK_HEAD_DIM = 128 + 64        # qk_nope_head_dim + qk_rope_head_dim (HF config.json)
V_HEAD_DIM = 128              # v_head_dim (HF config.json)
HEAD_DIM_EFF = (QK_HEAD_DIM + V_HEAD_DIM) / 2   # = 160

DEEPSEEK_V3 = ModelSpec(
    name="DeepSeek-V3-Base (MoE 671B/37B, MLA)", is_moe=True,
    p_total=671e9, p_active=37e9,          # disclosed; repo convention treats "active" as active-layer params
    d_model=7168, n_layers=61, n_q_heads=128,     # HF config.json: hidden_size, num_hidden_layers, num_attention_heads
    n_kv_heads=1,                                  # PLACEHOLDER -- unused (see MLA caveat above; no KV-cache computed)
    head_dim=HEAD_DIM_EFF,                         # MLA-effective, attention-FLOP terms only
    vocab=129280, tied_embeddings=False,           # HF config.json
)


def build(er_mean, er_cv=0.5, prompt_len=500.0, max_response_len=32000.0,
          recomp_act=1, recomp_old=0, prompts_per_batch=32, responses_per_prompt=G):
    """Minimal Scenario just to exercise training_compute()/rollout_terms() -- hardware/net/verify
    fields are placeholders (unused by the FLOP-tally functions)."""
    return Scenario(
        name="R1-Zero FLOP tally",
        model=DEEPSEEK_V3,
        rl=RLSpec(prompts_per_batch=prompts_per_batch, responses_per_prompt=responses_per_prompt,
                  prompt_len=prompt_len, response_len_mean=er_mean, response_len_cv=er_cv,
                  max_response_len=max_response_len, prefix_caching=True,
                  oversample_override=1.0),
        algo=AlgoSpec(b_optimiser=8.0, recomp_act=recomp_act, recomp_old=recomp_old, opt_steps=1),
        train_hw=HWSpec("512xH800", n_nodes=64, gpus_per_node=8, mfu=0.40,
                        node_flops=8 * GPU_PEAK_BF16, node_hbm=8 * 80 * GB, node_bw=8 * 3.35 * TB),
        inf_hw=HWSpec("512xH800", n_nodes=64, gpus_per_node=8, mfu=0.10,
                      node_flops=8 * GPU_PEAK_BF16, node_hbm=8 * 80 * GB, node_bw=8 * 3.35 * TB),
        net=NetSpec(bandwidth=1e15, latency=0.0),   # colocated, no real WAN broadcast -- placeholder
        verify=VerifySpec(mode="rule", seconds_per_rollout=0.0, n_verifier_workers=1),
    )


def total_flops(er_mean, n_steps=N_STEPS, **kw):
    s = build(er_mean, **kw)
    tc = training_compute(s)
    ro = rollout_terms(s)
    c_update = tc.c_update
    c_rollout = ro.c_rollout_total
    per_step = c_update + c_rollout
    return per_step * n_steps, per_step, c_update, c_rollout, tc.attn_frac


# ---------------------------------------------------------------------------
# Cross-check vs Epoch AI's estimate (epoch.ai/gradient-updates/what-went-into-training-deepseek-r1)
# ---------------------------------------------------------------------------
# Epoch's formula: N*B*G*L*P_active*8 + N*(K-1)*B*G*L*P_active*6, with N*K=8,000 total
# gradient steps, K=1 (so the second term vanishes), L=4,000, B=1,024 questions/step, G=64,
# P_active=37B. Result: 6.1e23 FLOP. Epoch's own writeup states B=1,024 and G=64 come from
# "prior work" (an earlier DeepSeek RL paper), NOT from the R1-Zero paper's own text.
#
# k_term selects the backward-term multiplier:
#   "k_minus_1" -- the formula AS PUBLISHED. At K=1 (one inner epoch, matching R1-Zero's
#     disclosed "trained for only a single inner epoch") this is IDENTICALLY ZERO, so the
#     published 6.1e23 is generation-only compute, no backward pass at all (see epoch_comparison()).
#   "k"         -- correct's TYPO-FIX hypothesis: (K-1) was almost certainly meant to be K, so a
#     single inner epoch (K=1) still counts ONE full forward+backward training pass (standard
#     6ND convention) over the generated batch, instead of zero. Generation term is unaffected.
def epoch_formula_flops(n_k=8000, k=1, L=4000.0, b=1024, g=64, p_active=37e9, k_term="k_minus_1"):
    train_mult = k if k_term == "k" else (k - 1)
    return n_k * b * g * L * p_active * 8 + n_k * train_mult * b * g * L * p_active * 6


def epoch_comparison():
    print("\n" + "=" * 92)
    print("  CROSS-CHECK vs EPOCH AI (epoch.ai/gradient-updates/what-went-into-training-deepseek-r1)")
    print("=" * 92)
    epoch_flops = epoch_formula_flops()
    print(f"  Epoch's own formula, evaluated as stated: {epoch_flops:.3e} FLOP "
          f"(their reported result: 6.1e23 -- matches)")
    print("  Epoch's assumed hyperparameters: B=1,024 questions/step, G=64, N*K=8,000 steps, L=4,000")
    print("  -- their writeup states B=1,024/G=64 are 'prior work' (an earlier DeepSeek RL paper),")
    print("     NOT read from the R1-Zero paper. The R1-Zero paper's OWN Sec 2.1 text states:")
    print("     32 questions/step x G=16 = 512 rollouts/step, 10,400 steps.")

    disclosed_tot, *_ = total_flops(4000.0, n_steps=N_STEPS,
                                     prompts_per_batch=32, responses_per_prompt=G)
    print(f"\n  MY core, DISCLOSED hyperparams (32x16=512/step, 10,400 steps), L=4,000: "
          f"{disclosed_tot:.3e} FLOP")

    epoch_hp_tot, *_ = total_flops(4000.0, n_steps=8000,
                                    prompts_per_batch=1024, responses_per_prompt=64)
    print(f"  MY core, EPOCH's hyperparams  (1024x64=65,536/step, 8,000 steps), L=4,000: "
          f"{epoch_hp_tot:.3e} FLOP")

    vol_ratio = (8000 * 1024 * 64) / (N_STEPS * SIZE_BATCH)
    print(f"\n  batch-volume ratio (steps x rollouts/step): Epoch {8000*1024*64:,} vs "
          f"disclosed {N_STEPS*SIZE_BATCH:,} = {vol_ratio:.1f}x")
    print(f"  ratio of totals at matched L=4,000 (Epoch-hyperparams/disclosed-hyperparams): "
          f"{epoch_hp_tot/disclosed_tot:.1f}x  -- tracks the volume ratio almost exactly:")
    print(f"     this batch/group assumption is ~ALL of the gap between this repo's FLOP tally")
    print(f"     ({disclosed_tot:.1e}-7.5e22 across the E[R] sweep above) and Epoch's 6.1e23.")
    print(f"  residual ({epoch_hp_tot/epoch_flops:.2f}x, holding hyperparams equal): a convention")
    print(f"     difference -- Epoch's '8 FLOP/param/token' covers generation+training combined in")
    print(f"     one term; this repo tallies them separately (~8x training at recomp_act=1, PLUS")
    print(f"     ~2x rollout generation on top, plus attention) -- closer to ~10x combined than 8x.")

    print("\n" + "-" * 92)
    print("  correct'S CORRECTION: disclosed batch (32x16=512) + disclosed N=10,400 steps +")
    print("  E[R]=8,000 + *K not *(K-1) (typo fix: K=1 should count one train pass, not zero)")
    print("-" * 92)
    L_correct, N_correct, B_correct, G_correct = 8000.0, N_STEPS, 32, G
    gen_term = N_correct * B_correct * G_correct * L_correct * 37e9 * 8
    train_term_orig = N_correct * (1 - 1) * B_correct * G_correct * L_correct * 37e9 * 6   # (K-1), K=1 -> 0
    train_term_fixed = N_correct * 1 * B_correct * G_correct * L_correct * 37e9 * 6        # *K, K=1
    corrected = epoch_formula_flops(n_k=N_correct, k=1, L=L_correct, b=B_correct, g=G_correct, k_term="k")
    print(f"  generation term (x8, unaffected by the typo fix): {gen_term:.3e}")
    print(f"  training term as PUBLISHED (x(K-1)=x0):            {train_term_orig:.3e}")
    print(f"  training term with the *K fix (x1):                {train_term_fixed:.3e}")
    print(f"  TOTAL (Epoch's formula, fully corrected):          {corrected:.3e} FLOP")
    print(f"    vs Epoch's published 6.1e23:  {corrected/6.1e23:.4f}x")
    core_tot, *_ = total_flops(L_correct, n_steps=N_correct, prompts_per_batch=B_correct, responses_per_prompt=G_correct)
    print(f"    vs this repo's own core-model tally at the same inputs: {core_tot:.3e} "
          f"({corrected/core_tot:.2f}x)")


def main():
    raw_gpu_flops = N_GPUS * GPU_PEAK_BF16 * WALL_S
    print("=" * 92)
    print("  DeepSeek-R1-Zero -- RL training compute (arXiv 2501.12948)")
    print("=" * 92)
    print(f"  disclosed: {N_GPUS} H800 GPUs x {WALL_HOURS:.0f}h = {N_GPUS*WALL_HOURS:,.0f} GPU-hours "
          f"(Table 7: 101K -- match)")
    print(f"  raw GPU FLOPs available = {N_GPUS} x {GPU_PEAK_BF16:.2e} FLOP/s x {WALL_S:,.0f}s "
          f"= {raw_gpu_flops:.3e} FLOP\n")

    print(f"  {'E[R] (tok)':>11}{'total FLOPs':>14}{'C_update':>13}{'C_rollout':>13}"
          f"{'roll:train':>11}{'attn%':>7}{'implied MFU':>13}")
    for er in (2000, 5000, 8000, 12000, 16000, 20000):
        tot, per_step, c_up, c_ro, attn_frac = total_flops(er)
        mfu = tot / raw_gpu_flops
        print(f"  {er:>11,}{tot:>14.3e}{c_up*N_STEPS:>13.3e}{c_ro*N_STEPS:>13.3e}"
              f"{c_ro/c_up:>10.2f}x{attn_frac*100:>6.1f}%{mfu:>12.2%}")

    print("\n  (roll:train = rollout-generation FLOPs / training-update FLOPs, per step;")
    print("   attn% = dot-product attention's share of training layer FLOPs at that E[R].)")

    # inverse-solve: what flat E[R] makes implied MFU land at a chosen reference value
    # (bisection, mirroring solvers.py's convention)
    def mfu_at(er):
        tot, *_ = total_flops(er)
        return tot / raw_gpu_flops

    for target_mfu in (0.05, 0.10, 0.20, 0.30):
        lo, hi = 100.0, 200_000.0
        if mfu_at(lo) > target_mfu or mfu_at(hi) < target_mfu:
            continue
        for _ in range(60):
            mid = (lo + hi) / 2
            if mfu_at(mid) < target_mfu:
                lo = mid
            else:
                hi = mid
        print(f"  E[R] implying MFU={target_mfu:.0%}: {mid:,.0f} tok")

    print("\n  SENSITIVITY -- prompt_len (at E[R]=8,000):")
    for pl in (200, 500, 1000):
        tot, *_ = total_flops(8000, prompt_len=pl)
        print(f"    prompt_len={pl:>5}: total FLOPs {tot:.3e}, implied MFU {tot/raw_gpu_flops:.2%}")

    print("\n  SENSITIVITY -- recomp_old (reference-model / old-policy logprob recompute), at E[R]=8,000:")
    for ro_flag, label in ((0, "reused from generation (assumed default)"), (1, "recomputed on trainer (+1x fwd)")):
        tot, *_ = total_flops(8000, recomp_old=ro_flag)
        print(f"    recomp_old={ro_flag} ({label}): total FLOPs {tot:.3e}, implied MFU {tot/raw_gpu_flops:.2%}")
    print("  (KL-vs-reference-model forward pass, Sec 2.1/Fig 3, is a SEPARATE extra ~+1x fwd cost")
    print("   not modelled by recomp_old at all -- see module docstring. Not included above.)")

    print("\n  CAVEATS: MFU on BF16-peak convention (989 TFLOP/s/GPU); if the RL stage runs FP8,")
    print("  peak roughly doubles and implied MFU roughly HALVES. attn head_dim is an MLA-effective")
    print("  approximation (see docstring) -- exact for this FLOP tally, not a KV-cache model.")
    print("  E[R] is UNMEASURED for R1-Zero specifically (Figure 1b gives no numeric data points) --")
    print("  read the sweep as a bracket, and the inverse-solved values as 'what E[R] WOULD have to")
    print("  be for MFU to reach X', not as a claim about the true value.")

    epoch_comparison()


if __name__ == "__main__":
    main()
