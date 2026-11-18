#!/usr/bin/env python3
"""
tulu3_rlvr_compute.py -- Tulu 3 8B / 70B / 405B RLVR (PPO): total RL training FLOPs,
raw GPU FLOPs over the disclosed run, and implied MFU. Value-model FLOPs EXCLUDED from
the headline per instruction (see VALUE-MODEL CAVEAT -- this is rigorous only at 405B).
Source: arXiv 2411.15124 (fetched + extracted directly from the PDF, 2026-08-17);
prompt lengths MEASURED from the actual allenai/RLVR-* datasets (2026-08-17).

DISCLOSED PER-MODEL CONFIG (section noted):
  8B:   Table 21 (RLVR column). batch=224, K=4. Hardware: 8 H100, 65h = 520 GPU-hours
        (Sec 6.3.1). Fig 23 "Final RLVR Runs" x-axis runs 0-50K episodes with the final
        8B checkpoint chosen ~45K in -> N=50,000 is the run that the 65h corresponds to
        (Table 21's "Total Episodes 100,000" is the config NOMINAL, not what the final
        run reached). Fig 23 GSM8K response-length curve settles ~400 tok -> E[R]=400.
        Prompt pool: GSM8K+MATH+IF-verifiable = 29,946 (Table 22).
  70B:  Table 21 + Sec 6.4 override: LR 1e-7, batch=640, response length 2,048, episodes
        (nominal) 400,000, beta=0.07. K=4 (inherits RLVR default). Fig 23 final 70B
        checkpoint ~25K in, run shown to 50K -> N=50,000 (same reasoning as 8B; the
        400,000 is nominal). Fig 23 70B response length ~300 tok -> E[R]=300. Hardware:
        60h x 48 GPUs = 2,880 GPU-hours (Sec 6.3.1). Prompt pool: same blended 29,946.
  405B: "Hyperparameters 405B RLVR" table (Sec 8): LR 1e-7, batch=1,856, K=1 (NOT 4),
        response length target 1,024. "For Tulu 3 405B RLVR we only used the MATH train
        set" -> prompt pool = MATH train 7,500 (Table 22), GSM8K/IFEval dropped. SETTLED
        (correct, 2026-08-17): 100 steps = 185,600 episodes (Fig 25's x-axis), NOT the
        text's "75 steps" -- the disclosed ~550s+25s+1500s=2,075s/step figure is a
        SERIAL sum, but this is an ASYNC setup (Sec 6.3.1's own framing: inference and
        training overlap, training GPUs don't idle waiting for rollouts) -- so real
        step time is shorter than the serial sum. 46h*3600/100 steps = 1,656s/step,
        which is a sensible ASYNC overlap of the three disclosed stages (~1,660s) --
        internally consistent, so 100 steps is the anchor, not 75. Fig 25 MATH response
        length ~400 tok -> E[R]=400. Hardware: 46h x 256 GPUs = 11,776 GPU-hours.

BATCH SIZE CANCELS: total FLOPs = n_batches x per-batch, n_batches = N_episodes/batch,
per-batch FLOPs ~ batch. So total FLOPs depends only on N_episodes (total sequences),
E[R], K and model -- NOT on batch size. Batch only sets the update-step count.

PROMPT LENGTHS -- MEASURED (not assumed), QwQ/Qwen tokenizer as a Llama-3 proxy
(~within ~10%), from the ACTUAL datasets Tulu 3 released:
  allenai/RLVR-GSM   mean 772 tok  (8-shot CoT prompt: 8 worked Q/A examples + question)
  allenai/RLVR-MATH  mean 651 tok  (8-shot CoT prompt)
  allenai/RLVR-IFeval mean 252 tok (zero-shot instruction prompts)
  -> 8B/70B blended by Table 22 counts (GSM 7,473 / MATH 7,500 / IF 14,973):
       (7473*772 + 7500*651 + 14973*252)/29946 = 482 tok
  -> 405B (MATH-only): 651 tok
  The few-shot prefix is real prefill/train context; loss is on response tokens only,
  but the fwd+bwd pass still processes the full P+R sequence (standard PPO), so P counts.

VALUE-MODEL CAVEAT -- SETTLED (correct, 2026-08-17): the 8B value model ("to reduce
computational cost", Sec 6.3.1, disclosed explicitly for 405B) was ALSO used for 70B,
not a size-matched 70B value net. So the value-model correction is:
    8B:   value model = 8B = SAME size as policy -> adds ~+95% (~1.9x total).
    70B:  value model = 8B (small vs 70B policy) -> adds only ~+12% -- "pretty
          negligible" (correct), consistent with 405B's own "~+2%, negligible" case.
    405B: value model = 8B (disclosed) -> adds ~+2% -> negligible.
  Recommended headline: value-INCLUDED at 8B (real compute, ~1.9x); value-EXCLUDED
  (or included, barely matters) at 70B/405B.

METHOD (pure FLOP tally via this repo's training_compute()/rollout_terms()):
  per batch: rollout_policy (autoregressive gen) + reference_scoring (1 forward, KL
             penalty) + K x policy_update (fwd+bwd). Value terms excluded (see above).
  total = n_batches x per-batch ; raw = n_gpus x 989e12 x wall_h x 3600 ; MFU = total/raw.
  6ND cross-check printed alongside (6 x N_params x trained_tokens x K + 2ND generation).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from specs import ModelSpec, RLSpec, AlgoSpec, Scenario, HWSpec, NetSpec, VerifySpec
from model import training_compute, rollout_terms

GB, TB = 1e9, 1e12
GPU_PEAK_BF16 = 989e12   # FLOP/s per H100 (BF16 dense; matches this repo's H100_NODE convention)

BLEND_PROMPT = 482.0     # 8B/70B blended prompt length (measured, Table-22-weighted; see docstring)
MATH_PROMPT = 651.0      # 405B MATH-only prompt length (measured)


def _llama_spec(name, p_total, d_model, n_layers, n_q_heads, n_kv_heads, vocab):
    return ModelSpec(name=name, is_moe=False, p_total=p_total,
                     p_active=p_total - 2 * d_model * vocab,   # untied embed+head excluded
                     d_model=d_model, n_layers=n_layers, n_q_heads=n_q_heads,
                     n_kv_heads=n_kv_heads, head_dim=d_model // n_q_heads, vocab=vocab,
                     tied_embeddings=False)


LLAMA3_8B = _llama_spec("Llama-3.1-8B", 8.03e9, 4096, 32, 32, 8, 128256)
LLAMA3_70B = _llama_spec("Llama-3.1-70B", 70.6e9, 8192, 80, 64, 8, 128256)
LLAMA3_405B = _llama_spec("Llama-3.1-405B", 405.9e9, 16384, 126, 128, 8, 128256)

# name -> config. er/N are the ANCHOR (headline); er_sweep/n_sweep for sensitivity.
CONFIGS = {
    "8B":  dict(model=LLAMA3_8B, n_gpus=8, wall_h=65.0, batch=224, k=4,
                prompt=BLEND_PROMPT, er=400, n_ep=50_000, value_params=8.03e9, value_note="8B (disclosed, = policy)",
                er_sweep=(200, 400, 600, 1000), n_sweep=(50_000, 100_000)),
    "70B": dict(model=LLAMA3_70B, n_gpus=48, wall_h=60.0, batch=640, k=4,
                prompt=BLEND_PROMPT, er=300, n_ep=50_000, value_params=8.03e9, value_note="8B (correct: same as 405B's, not size-matched -- negligible)",
                er_sweep=(200, 300, 600, 1000), n_sweep=(50_000, 100_000, 400_000)),
    "405B": dict(model=LLAMA3_405B, n_gpus=256, wall_h=46.0, batch=1856, k=1,
                 prompt=MATH_PROMPT, er=400, n_ep=185_600, value_params=8.03e9, value_note="8B (disclosed)",
                 er_sweep=(300, 400, 450), n_sweep=(185_600,)),
}


def build(model, batch, response_len, prompt_len, recomp_act=1):
    return Scenario(
        name=f"{model.name} RLVR", model=model,
        rl=RLSpec(prompts_per_batch=batch, responses_per_prompt=1, prompt_len=prompt_len,
                  response_len_mean=response_len, response_len_cv=0.5,
                  max_response_len=max(2048.0, response_len), prefix_caching=False,
                  oversample_override=1.0),
        algo=AlgoSpec(b_optimiser=8.0, recomp_act=recomp_act, recomp_old=0, opt_steps=1),
        train_hw=HWSpec("H100s", n_nodes=1, gpus_per_node=8, mfu=0.40,
                        node_flops=8 * GPU_PEAK_BF16, node_hbm=8 * 80 * GB, node_bw=8 * 3.35 * TB),
        inf_hw=HWSpec("H100s", n_nodes=1, gpus_per_node=8, mfu=0.10,
                      node_flops=8 * GPU_PEAK_BF16, node_hbm=8 * 80 * GB, node_bw=8 * 3.35 * TB),
        net=NetSpec(bandwidth=1e15, latency=0.0), verify=VerifySpec(mode="rule"))


def per_seq(model, response_len, prompt_len, k, recomp_act=1):
    """Per-EPISODE (one prompt-response pair) FLOPs; batch cancels, so probe at batch=1."""
    s = build(model, 1, response_len, prompt_len, recomp_act)
    tc = training_compute(s)
    ro = rollout_terms(s)
    fwd_only = tc.c_update / (3 + recomp_act)              # one teacher-forced forward pass
    return dict(rollout_policy=ro.c_rollout_total, reference_scoring=fwd_only,
                policy_update=k * tc.c_update, policy_update_1x=tc.c_update, attn_frac=tc.attn_frac)


def totals(cfg, response_len=None, n_ep=None):
    model, k = cfg["model"], cfg["k"]
    response_len = cfg["er"] if response_len is None else response_len
    n_ep = cfg["n_ep"] if n_ep is None else n_ep
    ps = per_seq(model, response_len, cfg["prompt"], k)
    t = {key: v * n_ep for key, v in ps.items() if key not in ("attn_frac", "policy_update_1x")}
    t["total"] = t["rollout_policy"] + t["reference_scoring"] + t["policy_update"]
    # value-model contribution (ratio-scaled: mirrors policy update + one scoring pass)
    ratio = (cfg["value_params"] - 2 * model.d_model * model.vocab) / model.p_active_layers
    t["value_flops"] = ratio * (t["policy_update"] + t["reference_scoring"])
    raw = cfg["n_gpus"] * GPU_PEAK_BF16 * cfg["wall_h"] * 3600
    return t, raw, ps["attn_frac"]


def sixnd_check(cfg, response_len, n_ep):
    """Independent 6ND estimate: 6*N_active*trained_tokens*K (fwd+bwd) + 2ND generation."""
    m = cfg["model"]; ctx = cfg["prompt"] + response_len
    train = 6 * m.p_active_layers * n_ep * ctx * cfg["k"]
    gen = 2 * m.p_active_layers * n_ep * response_len
    return train + gen


def fmt(x): return f"{x:.3e}"


def report(tag, cfg):
    t, raw, attn = totals(cfg)
    gpu_h = cfg["n_gpus"] * cfg["wall_h"]
    print("=" * 98)
    print(f"  Tulu 3 {tag} RLVR (PPO) -- {cfg['model'].p_total/1e9:.1f}B | anchor E[R]={cfg['er']}, "
          f"N={cfg['n_ep']:,} episodes, K={cfg['k']}, prompt~{cfg['prompt']:.0f}")
    print("=" * 98)
    print(f"  DISCLOSED     : {cfg['n_gpus']} H100 x {cfg['wall_h']:.0f}h = {gpu_h:,.0f} GPU-hours")
    print(f"  RAW GPU FLOPs : {cfg['n_gpus']} x {GPU_PEAK_BF16:.2e} x {cfg['wall_h']*3600:,.0f}s = {fmt(raw)} FLOP")
    print(f"  RL FLOPs (value EXCLUDED) : {fmt(t['total'])}   ->   implied MFU {t['total']/raw:.2%}")
    six = sixnd_check(cfg, cfg["er"], cfg["n_ep"])
    print(f"    6ND cross-check: {fmt(six)}  (ratio to tally {six/t['total']:.2f}x; ~1 = consistent)")
    n_steps = cfg["n_ep"] / cfg["batch"]
    print(f"    update steps = N/batch = {cfg['n_ep']:,}/{cfg['batch']} = {n_steps:,.0f}")
    print(f"    components: rollout {100*t['rollout_policy']/t['total']:.0f}% | ref-score "
          f"{100*t['reference_scoring']/t['total']:.0f}% | policy-update {100*t['policy_update']/t['total']:.0f}%"
          f" | attn {attn*100:.1f}% of layer FLOPs")
    incl = t["total"] + t["value_flops"]
    print(f"  + value model ({cfg['value_note']}): +{fmt(t['value_flops'])} -> "
          f"RL FLOPs INCL value {fmt(incl)}, MFU {incl/raw:.2%}  ({incl/t['total']:.2f}x)")
    # E[R] sweep
    print(f"  E[R] sweep (value-excl RL FLOPs / MFU) @ N={cfg['n_ep']:,}:")
    for er in cfg["er_sweep"]:
        te, _, _ = totals(cfg, response_len=er)
        print(f"     E[R]={er:>5,}: {fmt(te['total'])} / {te['total']/raw:.2%}")
    # N sweep
    print(f"  N sweep (value-excl RL FLOPs / MFU) @ E[R]={cfg['er']}:")
    for n in cfg["n_sweep"]:
        tn, _, _ = totals(cfg, n_ep=n)
        star = "  <- disclosed target" if (tag != "405B" and n >= 100_000) else ("  <- settled anchor (100 steps, async)" if n == 185_600 else "")
        print(f"     N={n:>8,}: {fmt(tn['total'])} / {tn['total']/raw:.2%}{star}")
    print()


def main():
    for tag, cfg in CONFIGS.items():
        report(tag, cfg)
    print("=" * 98)
    print("  CROSS-SIZE SUMMARY (headline anchors)")
    print("=" * 98)
    print(f"  {'model':<7}{'GPU-hrs':>9}{'raw GPU FLOPs':>15}{'RL FLOPs(no val)':>18}{'MFU':>7}"
          f"{'RL FLOPs(+val)':>16}{'MFU+val':>9}")
    for tag, cfg in CONFIGS.items():
        t, raw, _ = totals(cfg)
        gpu_h = cfg["n_gpus"] * cfg["wall_h"]
        incl = t["total"] + t["value_flops"]
        print(f"  {tag:<7}{gpu_h:>9,.0f}{fmt(raw):>15}{fmt(t['total']):>18}{t['total']/raw:>6.2%}"
              f"{fmt(incl):>16}{incl/raw:>8.2%}")
    print("\n  405B settled at N=185,600 (100 steps, Fig 25): the disclosed ~550+25+1500=2,075s/step")
    print("  is a SERIAL sum, but Sec 6.3.1 describes an ASYNC setup (overlap, not sum) -- 46h/100")
    print("  steps = 1,656s/step, a sensible async overlap of the three stages, so 100 steps (not the")
    print("  text's 75) is internally consistent and used throughout. 8B/70B N=50,000 is the run Fig 23")
    print("  shows (not Table 21's 100K/400K nominal). Value model = 8B for ALL THREE sizes (disclosed")
    print("  at 405B, confirmed also used at 70B) -- material only at 8B (~+95%, same-size); ~+12% at")
    print("  70B and ~+2% at 405B, both genuinely negligible as instructed.")


if __name__ == "__main__":
    main()
