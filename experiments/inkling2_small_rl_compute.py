#!/usr/bin/env python3
"""
inkling2_small_rl_compute.py -- Thinking Machines "Inkling 2 Small" main RL run: RL FLOPs
via this repo's compute core, compared against the lab's own "over 4 x 10^23 FLOPs".
Source: https://thinkingmachines.ai/news/inkling-2-small/ (fetched 2026-11-06).

DISCLOSED ("Scale of the main RL run" / "Stable training" tables, Training section):
  Model: MoE, 594B total / 17B active. Layers, heads, KV heads, expert count, vocab,
         attention type: NOT reported.
  RL:    1,075 steps | 17M+ rollouts | 3T+ tokens | 9 days | 2,048 GPUs (type not stated
         for RL; pretraining was GB300 NVL72, and RL runs "NVFP4 MoE training and sampling",
         a Blackwell-only format -> GB300 assumed) | compute "Over 4 x 10^23 FLOPs"
         (figure axes: "cumulative learner and sampler FLOPs"; method not described).
         Rollout batch 16K (17M/1,075 = 15.8K -> rollouts, not prompts). Median staleness
         17 updates. Total tokens/rollout median 43K, p99 1.5M. Rollout duration median
         8 min, p99 4 h. Context 256K with compaction. Sampler:trainer 5:3. Prefix-cache
         hit rate 93%+ (DRAM KV offload). Async StableReinforce in Tinker.
  Pretraining: H1 8.4T tok in 68 h on GB300 NVL72 (GPU count NOT given), H2 3.8T,
         H3 ~320B (incl. replay), H4 ~34B -> ~12.55T training tokens.

MAPPING MULTI-TURN AGENTIC RL ONTO THE SINGLE-TURN CORE (exact for FLOP totals):
  Each rollout is one trajectory of L total tokens (prompt + generated + tool output,
  interleaved). Under perfect prefix caching every token is forwarded exactly once by
  the sampler (as prefill if environment/prompt, as decode if generated), and once
  fwd+bwd by the learner. So set P = (1-g)L, R = gL with P+R = L: the model's
  c_prefill + c_decode = 2*N*L linear regardless of g, and its attention terms
  2P^2 + 4(PR + R^2/2) = 2(P+R)^2 = 2L^2 -- the interleaving order does not change the
  causal-pair count. So the generated fraction g does NOT affect FLOPs, only timing.
  What the single-turn core does NOT capture, added here explicitly:
   (1) compaction: training/sampling contexts are capped at W=256K, so the attention
       context per token is kappa = E[L*min(L,W)]/E[L], not E[L^2]/E[L];
   (2) cache misses: 93% hit => 7% of the REQUESTED prefill (sum of context over turns)
       is recomputed; measured agentic traces (experiments/lengths/agentic_ccbench.json)
       give sum(ctx_t)/L = 26-31, so recomputed prefill can exceed the single pass;
   (3) attention architecture is undisclosed and dominates the attention term -- bracketed.

CONVENTION: model FLOPs (learner 6ND fwd+bwd, sampler 2ND fwd), activation recompute
excluded (hardware overhead, shown separately). N = 17B active.
"""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from specs import ModelSpec, RLSpec, AlgoSpec, HWSpec, NetSpec, VerifySpec, Scenario
from model import training_compute, rollout_terms

# ---------------- disclosed ----------------
N_ACTIVE, N_TOTAL = 17e9, 594e9
STEPS, BATCH = 1075, 16_000                 # 16K rollouts/step x 1,075 = 17.2M rollouts
T_TOKENS = 3e12                             # "3T+" total rollout tokens (lower bound)
DAYS, GPUS = 9, 2048
SAMPLER_FRAC = 5 / 8                        # 5:3 sampler:trainer
W_CTX = 256_000                             # context window, compaction beyond
MED_L, P99_L = 43_000, 1_500_000
CACHE_HIT = 0.93
LAB_FLOPS = 4e23
# Pretraining phases: (name, tokens, context, wall-clock hours, provenance). Only H1 has a
# duration; NO phase discloses a GPU count, so pretraining MFU is only derivable as a
# function of an assumed GPU count.
PT_PHASES = [
    ("H1", 8.4e12,  None,      68,   "8.4T tok, 68 h on GB300 NVL72 racks; context + GPU count NOT stated"),
    ("H2", 3.8e12,  512_000,   None, "3.8T tok of newer data at 512K context; duration NOT stated"),
    ("H3", 0.32e12, 1_000_000, None, "~320B tok reasoning/agent trajectories at 1M, incl. replayed data"),
    ("H4", 0.034e12, 1_000_000, None, "~34B tok top agent trajectories (from Kimi K3 / earlier Inkling)"),
]

# ---------------- assumed (flagged) ----------------
# GB300 (Blackwell Ultra) per-GPU DENSE peaks, from the GB300 NVL72 datasheet / 72.
PEAK = {"BF16": 2.5e15, "FP8": 5.0e15, "NVFP4": 15.0e15}
# Share of the LINEAR (parameter-GEMM) FLOPs actually executed in NVFP4. "Native NVFP4" MoE
# training/sampling puts expert + projection GEMMs in FP4; the LM head/embeddings (~d_model x
# vocab ~ 1.1B of the 17B active, ~7%), router and any layers kept in BF16 for stability are not.
F_FP4 = {"all-FP4": 1.0, "central": 0.9, "conservative": 0.8}
G_FRAC = 0.11        # generated share of trajectory tokens (ccbench measured) -- timing only


def eff_peak(f_fp4, f_attn=0.0, attn_prec="BF16"):
    """Precision-matched peak FLOP/s per GPU for a FLOP mix: of the linear FLOPs a share f_fp4
    runs at the NVFP4 peak and the rest at BF16; attention matmuls (share f_attn of ALL FLOPs)
    run at attn_prec. Ideal time/FLOP = sum(share_i / peak_i) (Amdahl), so a 10% BF16 slice
    costs as much time as the 90% in FP4 -- the effective peak sits well below 15 PFLOP/s."""
    lin = f_fp4 / PEAK["NVFP4"] + (1 - f_fp4) / PEAK["BF16"]
    return 1.0 / ((1 - f_attn) * lin + f_attn / PEAK[attn_prec])
# attn_coef = (query heads) x (effective head dim) x (layers with quadratic attention)
ARCH = {
    "dense MLA-like (61L x 64H x 160)":        dict(attn_coef=61 * 64 * 160, sparse_k=None),
    "hybrid 3:1 linear:full (1/4 layers)":     dict(attn_coef=61 * 64 * 160 / 4, sparse_k=None),
    "sparse top-k=2048 (DSA-style)":           dict(attn_coef=61 * 64 * 160, sparse_k=2048),
}

E_L = T_TOKENS / (STEPS * BATCH)            # 174K mean tokens/rollout (from the totals)
WALL_S = DAYS * 86400
RAW_GPU_S = GPUS * WALL_S


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def kappa_lognormal(mu, sig, W):
    """E[L*min(L,W)] / E[L] for L ~ lognormal(mu, sig): attention context per token with
    compaction segments of length W (a trajectory L>W costs ~ (L/W) segments x W^2)."""
    lnW = math.log(W)
    e_l = math.exp(mu + sig ** 2 / 2)
    e_l2_low = math.exp(2 * mu + 2 * sig ** 2) * Phi((lnW - mu - 2 * sig ** 2) / sig)
    e_l_high = e_l * (1 - Phi((lnW - mu - sig ** 2) / sig))
    return (e_l2_low + W * e_l_high) / e_l


def tail_fits():
    mu = math.log(MED_L)
    sig_mean = math.sqrt(2 * math.log(E_L / MED_L))          # matches median + 3T/17.2M mean
    sig_p99 = math.log(P99_L / MED_L) / 2.3263                # matches median + p99
    return {"fit median+mean (sig=%.2f)" % sig_mean: kappa_lognormal(mu, sig_mean, W_CTX),
            "fit median+p99  (sig=%.2f)" % sig_p99: kappa_lognormal(mu, sig_p99, W_CTX)}


def scenario(attn=False, cv=0.0):
    """Single-turn mapping: P=(1-g)L, R=gL, one rollout per 'prompt', no group prefix share."""
    m = ModelSpec(name="Inkling 2 Small (594B/17B MoE, arch n/d)", is_moe=True,
                  p_total=N_TOTAL, p_active=N_ACTIVE, d_model=7168, n_layers=61,
                  n_q_heads=64, n_kv_heads=8, head_dim=160, vocab=160_000)   # attn arch placeholder
    rack = dict(node_flops=72 * PEAK["BF16"], node_hbm=72 * 288e9, node_bw=72 * 8e12, gpus_per_node=72)
    return Scenario(
        name="Inkling 2 Small RL",
        model=m,
        rl=RLSpec(prompts_per_batch=BATCH, responses_per_prompt=1,
                  prompt_len=(1 - G_FRAC) * E_L, response_len_mean=G_FRAC * E_L,
                  response_len_cv=cv, max_response_len=W_CTX, prefix_caching=False,
                  n_steps_override=STEPS, oversample_override=1.0),
        algo=AlgoSpec(b_weights_inf=0.5625, b_kv=1.0, recomp_act=0, recomp_old=0,
                      include_attention=attn, sync_interval=17),
        train_hw=HWSpec("GB300 NVL72 (trainer)", n_nodes=round(GPUS * (1 - SAMPLER_FRAC) / 72),
                        n_shard=72, **rack),
        inf_hw=HWSpec("GB300 NVL72 (sampler)", n_nodes=round(GPUS * SAMPLER_FRAC / 72), **rack),
        net=NetSpec(bandwidth=50e9), verify=VerifySpec(mode="rule"))


def main():
    s = scenario()
    learner = training_compute(s).c_update * STEPS            # 6ND over all trajectory tokens
    sampler = rollout_terms(s).c_rollout_total * STEPS          # 2ND, every token once
    base = learner + sampler

    print("=" * 100)
    print("  Inkling 2 Small -- main RL run compute (repo core, multi-turn mapped to single-turn)")
    print("=" * 100)
    print(f"  disclosed: {STEPS:,} steps x {BATCH:,} rollouts, {T_TOKENS:.0e} tokens -> mean "
          f"{E_L:,.0f} tok/rollout (median {MED_L:,}, p99 {P99_L:,}); {GPUS:,} GPUs x {DAYS} d = "
          f"{RAW_GPU_S/3600:,.0f} GPU-h")
    print(f"\n  [1] LINEAR (no attention, perfect caching):")
    print(f"      learner 6ND {learner:.3e} + sampler 2ND {sampler:.3e} = {base:.3e} FLOP")
    print(f"      lab reports 'over 4e23'  ->  ratio {base/LAB_FLOPS:.3f}.  Their figure == 8*N_active*T.")

    print(f"\n  [2] + ATTENTION (compaction-capped at {W_CTX//1000}K; arch undisclosed -> bracketed)")
    kaps = tail_fits()
    for fit, kap in kaps.items():
        print(f"      tail {fit}: attention context/token kappa = {kap:,.0f} tok")
    print(f"      {'architecture':<40}{'attn FLOPs':>13}{'total':>12}{'x lab':>8}")
    attn_by_arch = {}
    for arch, a in ARCH.items():
        vals = []
        for kap in kaps.values():
            k_eff = 2 * a["sparse_k"] if a["sparse_k"] else kap   # sparse: each token sees k keys
            vals.append(8 * a["attn_coef"] * T_TOKENS * k_eff)    # 3x learner fwd+bwd + 1x sampler
        lo, hi = min(vals), max(vals)
        attn_by_arch[arch] = (lo, hi)
        print(f"      {arch:<40}{lo:>6.1e}-{hi:<6.1e}{(base+lo):>6.1e}-{(base+hi):<6.1e}"
              f"{(base+lo)/LAB_FLOPS:>4.1f}-{(base+hi)/LAB_FLOPS:.1f}")

    # validation: feed the dense/compaction e_T through the model's own attention formula
    a = ARCH["dense MLA-like (61L x 64H x 160)"]
    kap = list(kaps.values())[0]
    target_eT = kap * E_L
    cv = math.sqrt(max(target_eT - E_L ** 2, 0)) / (G_FRAC * E_L)
    sv = scenario(attn=True, cv=cv)
    tc, ro = training_compute(sv), rollout_terms(sv)
    model_attn = (tc.c_update - learner / STEPS) * STEPS + (ro.c_rollout_total - sampler / STEPS) * STEPS
    print(f"      check: model's own c_attn path (cv={cv:.2f} to reproduce E[T^2]) gives "
          f"{model_attn:.3e} vs analytic {8*a['attn_coef']*T_TOKENS*kap:.3e}")

    # cache-miss re-prefill from measured agentic traces
    print(f"\n  [3] + PREFILL RECOMPUTE from cache misses ({100*(1-CACHE_HIT):.0f}% of requested prefill)")
    tr = [x for x in json.load(open(os.path.join(os.path.dirname(__file__), "lengths",
                                                  "agentic_ccbench.json"))) if x.get("cum_in") and x.get("total")]
    ratios = {"median traj": sorted(x["cum_in"] / x["total"] for x in tr)[len(tr) // 2],
              "token-weighted": sum(x["cum_in"] for x in tr) / sum(x["total"] for x in tr)}
    miss = {}
    for k, r in ratios.items():
        extra_frac = max(0.0, (1 - CACHE_HIT) * r - (1 - G_FRAC))   # beyond the single prefill pass
        miss[k] = 2 * N_ACTIVE * T_TOKENS * extra_frac
        print(f"      sum(ctx_t)/L = {r:.1f} ({k}): recomputed {100*(1-CACHE_HIT)*r:.0f}% of L vs "
              f"{100*(1-G_FRAC):.0f}% single-pass -> +{miss[k]:.2e} FLOP")
    print("      (0 if '93%' counts only prefix lookups rather than requested tokens; upper bound applies "
          "coding-like turn counts to all tasks)")

    # totals
    mid_attn = attn_by_arch["hybrid 3:1 linear:full (1/4 layers)"]
    lo_tot = base                                                        # sparse attn, no misses
    hi_tot = base + attn_by_arch["dense MLA-like (61L x 64H x 160)"][1] + max(miss.values())
    print(f"\n  RANGE: {lo_tot:.1e} (lab-equivalent: sparse attn, no misses) to {hi_tot:.1e} "
          f"(dense attn, heavy tail, misses)  = {lo_tot/LAB_FLOPS:.1f}-{hi_tot/LAB_FLOPS:.1f}x the lab's figure")
    print(f"  activation recompute (if full checkpointing) adds {learner/3:.1e} HARDWARE FLOP to the learner "
          f"-- not model FLOPs.")

    # MFU -- precision-matched
    print(f"\n  [4] MFU, PRECISION-MATCHED (GB300 assumed; dense peaks/GPU BF16 2.5P, FP8 5P, NVFP4 15P)")
    print(f"      MFU = achieved model FLOP/s per GPU / peak of the precision the run COMPUTES in. Against")
    print(f"      the BF16 peak an NVFP4 run gets credit for its precision choice (6x), not its utilisation;")
    print(f"      for cross-hardware comparison use the absolute FLOP/s column instead.")
    for nm, f in F_FP4.items():
        print(f"        effective linear-FLOP peak, {nm:<12} (f_FP4={f:.1f}): {eff_peak(f)/1e15:5.1f} PFLOP/s/GPU")
    pools = {"whole cluster": (base, GPUS),
             "learner pool (3/8)": (learner, GPUS * (1 - SAMPLER_FRAC)),
             "sampler pool (5/8)": (sampler, GPUS * SAMPLER_FRAC)}
    print(f"      {'pool':<20}{'GPUs':>6}{'FLOP/s/GPU':>12}" + "".join(f"{'MFU f='+str(f):>12}" for f in F_FP4.values()))
    for name, (fl, g) in pools.items():
        rate = fl / (g * WALL_S)
        print(f"      {name:<20}{g:>6.0f}{rate:>12.2e}" + "".join(f"{rate/eff_peak(f):>11.1%} " for f in F_FP4.values()))
    hyb_attn = sum(mid_attn) / 2
    hyb = base + hyb_attn
    fa = hyb_attn / hyb
    print(f"      counting hybrid-attention FLOPs too ({fa:.0%} of all FLOPs, attention matmuls in BF16 / FP8),"
          f" f_FP4=0.9:")
    for prec in ("BF16", "FP8"):
        print(f"        whole cluster {hyb/(GPUS*WALL_S):.2e} FLOP/s/GPU vs peak {eff_peak(0.9, fa, prec)/1e15:.1f}P "
              f"-> MFU {hyb/(GPUS*WALL_S)/eff_peak(0.9, fa, prec):.1%}  (attention in {prec})")
    print(f"      FLOP ratio sampler:learner = {sampler/learner:.2f}  vs  GPU ratio 5:3 = {5/3:.2f}"
          f"  -> sampler does {(learner/(GPUS*(1-SAMPLER_FRAC)))/(sampler/(GPUS*SAMPLER_FRAC)):.1f}x "
          f"fewer FLOP per GPU-second")

    # pretraining -- step-by-step, every input tagged
    print(f"\n  [5] PRETRAINING FLOPs + MFU, step by step  [D]=disclosed  [A]=assumed  [=]=derived")
    print(f"      FLOPs_linear = 6 x N_active x tokens, N_active = 17e9 [D]  (same convention as the lab's RL "
          f"figure; replayed tokens still cost compute)")
    print(f"      {'phase':<6}{'tokens [D]':>11}{'context [D]':>13}{'hours [D]':>11}{'6ND [=]':>11}{'share':>7}   provenance")
    pt_lin = {ph: 6 * N_ACTIVE * tok for ph, tok, _, _, _ in PT_PHASES}
    pt_all = sum(pt_lin.values())
    for ph, tok, ctx, hrs, prov in PT_PHASES:
        print(f"      {ph:<6}{tok:>11.3g}{(f'{ctx//1000}K' if ctx else 'n/d'):>13}{(str(hrs) if hrs else 'n/d'):>11}"
              f"{pt_lin[ph]:>11.2e}{pt_lin[ph]/pt_all:>7.0%}   {prov}")
    tok_all = sum(t for _, t, _, _, _ in PT_PHASES)
    print(f"      {'total':<6}{tok_all:>11.4g}{'':>13}{'':>11}{pt_all:>11.2e}")

    print(f"\n      Attention (EXCLUDED above, as in the lab's RL figure). Per token, forward attention / linear")
    print(f"      = S_eff x attn_coef / N, S_eff = attention span per token (= packed length without document")
    print(f"      masking; << that with masking; 2k for top-k sparse). Upper bounds (no masking):")
    spans = [("H1", 8_000, "[A] ctx n/d"), ("H1", 32_000, "[A] ctx n/d"), ("H2", 512_000, "[D] ctx"),
             ("H3/H4", 1_000_000, "[D] ctx")]
    print(f"      {'phase':<7}{'S_eff':>9}  " + "".join(f"{lab:>16}" for lab in ("dense", "hybrid 3:1", "sparse k=2048")))
    for ph, S, tag in spans:
        cells = ""
        for a in ARCH.values():
            s_eff = 2 * a["sparse_k"] if a["sparse_k"] else S
            cells += f"{s_eff * a['attn_coef'] / N_ACTIVE:>15.2f}x"
        print(f"      {ph:<7}{S//1000:>7}K {tag[:3]}" + cells)
    h2_dense = pt_lin["H2"] * 512_000 * ARCH["dense MLA-like (61L x 64H x 160)"]["attn_coef"] / N_ACTIVE
    print(f"      -> dense full attention at 512K with no masking would add {h2_dense:.1e} FLOP to H2 alone "
          f"({h2_dense/pt_all:.0f}x all linear pretraining):")
    print(f"         implausible for 3.8T tokens, so the architecture is sparse/hybrid and/or documents are masked.")

    h1 = PT_PHASES[0]
    fl_h1, hrs_h1 = pt_lin["H1"], h1[3]
    print(f"\n      H1 MFU = FLOPs_H1 / (G x peak_eff x {hrs_h1} h x 3600).  G (GPU count) NOT disclosed -> tabulated:")
    print(f"        H1 FLOPs_linear = 6 x 17e9 x 8.4e12 = {fl_h1:.3e}  [=]")
    print(f"        throughput     = 8.4e12 tok / {hrs_h1*3600:,} s = {h1[1]/(hrs_h1*3600):.3e} tok/s cluster-wide  [=]")
    print(f"        {'G (racks of 72)':<18}{'tok/s/GPU':>10}{'FLOP/s/GPU':>12}" + "".join(f"{'MFU f='+str(f):>12}" for f in F_FP4.values()))
    for G in (576, 1152, 2048, 4608):
        rate = fl_h1 / (G * hrs_h1 * 3600)
        lab = f"{G} ({G/72:.0f})" if G % 72 == 0 else f"{G} (=RL count)"
        mfus = "".join((f"{rate/eff_peak(f):>11.1%} " if rate < eff_peak(f) else f"{'>100% (!)':>12}") for f in F_FP4.values())
        print(f"        {lab:<18}{h1[1]/(hrs_h1*3600)/G:>10,.0f}{rate:>12.2e}" + mfus)
    print(f"        inverse -- G needed at a given precision-matched MFU (f_FP4=0.9, peak {eff_peak(0.9)/1e15:.0f}P):")
    for m in (0.2, 0.3, 0.4):
        G = fl_h1 / (m * eff_peak(0.9) * hrs_h1 * 3600)
        print(f"          MFU {m:.0%}: {G:,.0f} GPUs ({G/72:.0f} NVL72 racks)")
    learner_rate = learner / (GPUS * (1 - SAMPLER_FRAC) * WALL_S)
    print(f"      RL learner pool runs {learner_rate:.2e} FLOP/s/GPU = {learner_rate/eff_peak(0.9):.1%} MFU (f=0.9); "
          f"H1 matches that MFU only if G ~ {fl_h1/(learner_rate*hrs_h1*3600):,.0f}.")
    print(f"\n      RL / pretraining (linear/linear) = {base:.2e} / {pt_all:.2e} = {base/pt_all:.0%}"
          f"  (cf. R1-Zero ~0.7% = 2e22/~3e24 per joshbuild.substack.com/p/deepseek-r1-zeros-rl-training-used;"
          f" Tulu-3 << 1%).")
    print(f"      Attention inflates BOTH sides (RL kappa ~210K; H2-H4 at 512K-1M), so the ratio can move either way.")

    # staleness consistency
    t_step = WALL_S / STEPS
    print(f"\n  [6] CONSISTENCY: step time {t_step:,.0f} s. Median rollout 8 min = {480/t_step:.1f} steps; "
          f"p99 4 h = {14400/t_step:.0f} steps")
    print(f"      -> median staleness 17 matches the age of the hour-long trajectories that carry most tokens "
          f"(mean {E_L/1000:.0f}K >> median 43K), i.e. it is token-weighted, not per-rollout.")


if __name__ == "__main__":
    main()
