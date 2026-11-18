#!/usr/bin/env python3
"""
tulu3_sft_dpo_compute.py -- Tulu 3 8B / 70B / 405B: SFT and DPO training FLOPs and
implied MFU, alongside the RLVR figure (from tulu3_rlvr_compute.py) so the full
post-training pipeline is comparable on ONE convention. Answers: is SFT/DPO MFU
similar to RLVR's? (Short answer: all three phases are single-digit % -- see below.)
Source: arXiv 2411.15124 (PDF, 2026-08-17); dataset sizes + sequence lengths MEASURED
from the released allenai/tulu-3-* mixtures (2026-08-17).

CONVENTION -- MODEL FLOPs, 6ND (standard for compute-accounting / governance tables;
Epoch/Chinchilla): forward=2ND, backward=4ND, train=6ND per token, N=total params,
attention terms negligible at these context lengths (<1%, so 6*N*tokens is exact
enough). Activation recomputation is NOT in the numerator (it is hardware overhead,
not model FLOPs). NOTE: the earlier tulu3_rlvr_compute.py reported RLVR on the 8ND
(recompute-included, "HFU-style") convention; the RLVR row here is recomputed on 6ND
for a clean apples-to-apples comparison, so its MFU is ~0.75x the figure in that
script. The RL FLOP *count* difference is the same 0.75x factor.

DISCLOSED per-phase wall-clock (Sec 4.3 / 5.4.2 / 6.3.1):
  SFT:  8B  32 GPUs x  6h =   192 GPU-hrs | 70B 64 GPUs x 50h = 3,200 | 405B NOT DISCLOSED
  DPO:  8B   8 GPUs x 10h =    80 GPU-hrs | 70B 64 GPUs x 19h = 1,216 | 405B NOT DISCLOSED
  RLVR: 8B   8 GPUs x 65h =   520 GPU-hrs | 70B 48 GPUs x 60h = 2,880 | 405B 256x46h=11,776
  (405B SFT/DPO give only hyperparameters, Table 34 -- so 405B SFT/DPO MFU is NOT
  derivable; FLOPs are, and an ideal-MFU lower bound on hours is shown.)

DATASETS -- MEASURED (QwQ/Qwen tokenizer as a Llama-3 proxy, ~within ~10%; random
offsets across the full released mixtures):
  SFT  allenai/tulu-3-sft-mixture : 939,343 examples, mean 314 tok (prompt+completion,
       median 239, p90 589; 0% exceed the 4,096 cap). Epochs=2 (Table 11/34). Same mix
       at all three sizes. -> SFT tokens = 939,343 * 314 * 2 = 5.90e8 (~0.59B).
  DPO  preference mixtures (chosen+rejected tokens/pair, 1 epoch):
       8B  allenai/llama-3.1-tulu-3-8b-preference-mixture : 272,898 pairs, 1,256 tok/pair
       70B allenai/llama-3.1-tulu-3-70b-preference-mixture: 337,186 pairs, 1,585 tok/pair
       405B allenai/llama-3.1-tulu-3-405b-preference-mixture: 360,924 pairs, 1,775 tok/pair
       (DPO max_seq=2,048 per sequence caps the tail -- medians are ~900-1,150/pair, so
       truncation trims ~5-10% of tokens at 70B/405B; treated as a minor downward
       uncertainty, not modelled per-sequence.)

FLOP MODEL:
  SFT  = 6 * N * (examples * mean_len * epochs)                    [pure fwd+bwd]
  DPO  = 8 * N * (pairs * pair_tokens)                             [policy 6ND fwd+bwd
         over chosen+rejected, 1 epoch, + reference 2ND forward, CACHED once (Tulu 3
         pre-computes+caches reference logprobs, Sec 5.4.2 -> 1 pass, not per-epoch)]
  RLVR = N_ep * [2*N*E[R] (gen) + 2*N*ctx (ref) + K*6*N*ctx (policy update)]
         + value-model term (8B value net at all sizes; see tulu3_rlvr_compute.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GPU_PEAK = 989e12   # H100 BF16 dense FLOP/s
V_PARAMS = 8.03e9   # 8B value model, used at ALL sizes (disclosed 405B; confirmed 70B)

MODELS = {  # name -> (N_total, sft_gpu_h, dpo_gpu_h, rlvr_gpu_h,  dpo_pairs, dpo_pairtok,
            #          rlvr_er, rlvr_prompt, rlvr_Nep, rlvr_K)
    "8B":   (8.03e9,  192, 80,  520,   272_898, 1256,  400, 482, 50_000,  4),
    "70B":  (70.6e9,  3200, 1216, 2880, 337_186, 1585,  300, 482, 50_000,  4),
    "405B": (405.9e9, None, None, 11776, 360_924, 1775, 400, 651, 185_600, 1),
}
SFT_EXAMPLES, SFT_MEANLEN, SFT_EPOCHS = 939_343, 314, 2
SFT_MAXSEQ = 4096


def raw(gpu_h):
    return None if gpu_h is None else gpu_h * 3600 * GPU_PEAK


def sft_flops(N):
    return 6 * N * (SFT_EXAMPLES * SFT_MEANLEN * SFT_EPOCHS)


def dpo_flops(N, pairs, pairtok):
    return 8 * N * pairs * pairtok            # 6 (policy fwd+bwd) + 2 (cached reference fwd)


def rlvr_flops(N, er, prompt, n_ep, k):
    ctx = prompt + er
    policy = n_ep * (2 * N * er + 2 * N * ctx + k * 6 * N * ctx)   # gen + ref + K*update
    value = n_ep * (k * 6 * V_PARAMS * ctx + 2 * V_PARAMS * ctx)   # value update + scoring
    return policy + value


def mfu(flops, r):
    return None if (r is None or flops is None) else flops / r


def fmt(x):
    return "n/d" if x is None else f"{x:.2e}"


def pct(x):
    return "  n/d " if x is None else f"{x*100:5.2f}%"


def main():
    print("=" * 104)
    print("  Tulu 3 post-training compute -- SFT / DPO / RLVR, MODEL FLOPs (6ND), implied MFU")
    print("=" * 104)
    print(f"  SFT dataset: {SFT_EXAMPLES:,} ex x {SFT_MEANLEN} tok x {SFT_EPOCHS} ep = "
          f"{SFT_EXAMPLES*SFT_MEANLEN*SFT_EPOCHS:.2e} tokens (measured; ~0.59B -- TINY)\n")
    hdr = f"  {'model':<6}{'phase':<6}{'GPU-hrs':>9}{'raw GPU FLOPs':>15}{'model FLOPs':>14}{'MFU':>8}{'note':>26}"
    print(hdr)
    for name, m in MODELS.items():
        N, sft_h, dpo_h, rl_h, pairs, pairtok, er, pr, nep, k = m
        rows = [
            ("SFT",  sft_h,  sft_flops(N),                         f"{SFT_EXAMPLES//1000}k ex, 2 ep"),
            ("DPO",  dpo_h,  dpo_flops(N, pairs, pairtok),         f"{pairs//1000}k pairs, {pairtok} tok"),
            ("RLVR", rl_h,   rlvr_flops(N, er, pr, nep, k),        f"N={nep//1000}k, E[R]={er}, K={k}"),
        ]
        for phase, gh, fl, note in rows:
            r = raw(gh)
            print(f"  {name:<6}{phase:<6}{(str(gh) if gh else 'n/d'):>9}{fmt(r):>15}{fmt(fl):>14}"
                  f"{pct(mfu(fl, r)):>8}{note:>26}")
        print()

    # padding / overhead diagnostic for SFT (why is SFT MFU so low?)
    print("-" * 104)
    print("  WHY SFT/DPO MFU IS LOW -- the datasets are small, so the reported GPU-hours are NOT")
    print("  FLOP-bound. SFT useful MFU vs HFU-if-padded-to-max_seq (4096/314 = 13x padding ceiling):")
    for name, m in MODELS.items():
        N, sft_h = m[0], m[1]
        if sft_h is None:
            print(f"    {name:<5} SFT: hours n/d"); continue
        r = raw(sft_h); u = sft_flops(N) / r
        hfu = min(1.0, u * SFT_MAXSEQ / SFT_MEANLEN)
        print(f"    {name:<5} SFT: useful MFU {u*100:.1f}%  |  HFU if padded-to-4096 ~{hfu*100:.0f}% "
              f"-> the run is padding-/overhead-bound, not compute-bound")

    # 405B SFT/DPO: hours not disclosed -> ideal-MFU lower bound on wall-clock
    print("-" * 104)
    N405 = MODELS["405B"][0]
    for phase, fl in (("SFT", sft_flops(N405)),
                      ("DPO", dpo_flops(N405, MODELS['405B'][4], MODELS['405B'][5]))):
        ideal_h = fl / (256 * GPU_PEAK * 0.40) / 3600          # if 40% MFU on 256 GPUs
        print(f"  405B {phase}: model FLOPs {fl:.2e}; wall-clock NOT disclosed. At an (optimistic) 40% "
              f"MFU on 256 GPUs -> >= {ideal_h*256:,.0f} GPU-hrs; realistically 5-15x that at the")
        print(f"           ~2-4% MFU the 70B phase shows.")

    print("=" * 104)
    print("  ANSWER: NOT the ~30-45% of compute-bound pretraining SFT. All three post-training phases")
    print("  sit at SINGLE-DIGIT MFU -- DPO (~6-7%) > SFT (~2-4%) > RLVR (~1%, 6ND) -- because the")
    print("  datasets are tiny (<1B tokens) relative to the GPU-hours spent, so the runs are")
    print("  overhead/padding-bound (SFT/DPO) or generation-bound (RLVR), not FLOP-bound. GOVERNANCE")
    print("  IMPLICATION: for these phases, GPU-hours x peak x (assumed 40% MFU) OVERSTATES the")
    print("  algorithmic compute ~10x; use the dataset-derived 6ND FLOPs instead.")


if __name__ == "__main__":
    main()
