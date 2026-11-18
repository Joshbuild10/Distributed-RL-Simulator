# Oversampling ratio ω — measurement

**Goal.** Pin the oversampling ratio ω that inflates rollout+verify compute (but not the
update) in the runtime model, replacing the guessed `RLSpec.oversample_override` /
`success_rate`. ω moves the inference:training ratio — the number the governance argument
turns on — so it must be measured, not assumed.

**Definition.** RL discards a group of G binary-reward samples iff all G agree (zero advantage),
so the accepted fraction is `f(p) = 1 − [p^G + (1−p)^G]` and `ω = 1/f`. Evaluating f at a single
average p is wrong — f is strongly convex and real datasets are bimodal (a mass of trivial
prompts at p≈1 and impossible ones at p≈0, both discarded). The correct quantity averages over
the prompt difficulty distribution: **ω = 1 / E_p[f(p)]**, computed here from datasets that
publish a per-prompt solve rate. Keyless, free, seconds to run (no generation).

Two ω are reported:
- **ω (full)** — over the full prompt pool.
- **ω (filtered)** — over live prompts only (0<p<1), i.e. after online difficulty filtering
  removes the always-solved/always-failed mass. This is what a *filtering* run actually pays on
  rollouts (INTELLECT-3; prime-rl's [0.001,0.999] band).

**Method / reproduce.** `experiments/measure_oversampling.py`, per-prompt solve-rate columns via
the keyless HF datasets-server.
```bash
python experiments/measure_oversampling.py --group 16 --n 400   # -> oversampling.json
```

## Results (G = 16, n = 400 prompts/source)

| source | mean p | always-0/1 | E[f] | **ω (full)** | ω (filtered) | naive 1/f(p̄) |
|---|---|---|---|---|---|---|
| INTELLECT-3-RL · math | 0.27 | 50% | 0.48 | **2.08** | 1.05 | 1.01 |
| INTELLECT-3-RL · code | 0.10 | 78% | 0.21 | **4.73** | 1.05 | 1.24 |
| INTELLECT-3-RL · logic | 0.20 | 40% | 0.57 | **1.74** | 1.05 | 1.03 |
| INTELLECT-3-RL · science | 0.34 | 25% | 0.67 | **1.48** | 1.11 | 1.00 |
| INTELLECT-2-RL | 0.30 | 73% | 0.26 | **3.78** | 1.02 | 1.00 |
| Skywork/prime-rl · 1B policy | 0.45 | 38% | 0.55 | **1.81** | 1.12 | 1.00 |
| Skywork/prime-rl · 7B policy | 0.62 | 52% | 0.42 | **2.36** | 1.14 | 1.00 |
| Skywork/prime-rl · 32B policy | 0.68 | 63% | 0.32 | **3.14** | 1.15 | 1.00 |

## Findings

1. **The oversampling cost *is* the always-0/1 mass.** After difficulty filtering (0<p<1), ω
   collapses to **~1.05–1.15 across every source** — zero-advantage discarding among *live*
   prompts is nearly negligible at G=16. Clean dichotomy for the runtime model:
   - **No filtering** → pay full ω (1.5–4.7×), dominated by wasted dead-prompt rollouts.
   - **With filtering** → pay ~1.1× on rollouts, **but** pay the difficulty-*discovery* generation
     separately (you must roll out to learn a prompt is dead, unless using precomputed solve
     rates / a curriculum). The dead fraction (25–78%) is paid **either way**, as oversampling or
     as filtering passes — the model should charge it somewhere.

2. **ω rises with policy strength on a fixed dataset** (Skywork: 1B 1.81 → 7B 2.36 → 32B 3.14).
   A stronger policy trivially-solves more prompts (mean p 0.45→0.68, always-0/1 38%→63%), so
   *more* groups become zero-advantage. Better policy ⇒ more oversampling waste unless actively
   curriculum-managed — which is why frontier RL leans on online difficulty filtering. This also
   means the evader's rollout-overhead depends on the (policy, dataset-difficulty) match, not just
   the dataset.

3. **The naive single-p form is wrong.** `naive = 1/f(mean p)` stays ~1.0 everywhere while the true
   ω is 1.5–4.7× — so the model's `success_rate` path (one average p) under-counts ω by that factor.
   Use ω directly, not a single success rate.

**prime-rl cross-check.** Its filtered ω on the 32B policy column (~1.15) sits just below the
throughput-triangulated 1.35; the gap is dynamic-sampling margin beyond pure zero-advantage.
Defensible band **1.15–1.35**; the preset's 1.35 is a fine upper edge.

## Caveats

- Solve rates are measured with reference policies (Qwen3-4B for INTELLECT-3, R1-distill-{1,7,32}B
  for Skywork/prime-rl), **not** necessarily the exact run policy — a stronger real policy shifts p
  up and (per finding 2) ω up. Treat as policy-conditioned, not universal.
- G-sensitivity not swept here (G=16, the runs' value). f is convex in p and steep in G near the
  tails, so ω would change under a different G.
- **Not computable keyless** (publish problem/answer only, no per-prompt solve rate): DeepScaleR-Preview,
  DeepCoder-Preview, DAPO-Math-17k, and the *original* Skywork-OR1-RL-Data (only PrimeIntellect's
  prime-rl-format fork adds solve rates). ω there needs generation.

## Preset implications

- **INTELLECT-3** scenario (`oversample_override=1.0` currently) uses difficulty filtering, but the
  discovery compute is paid regardless → carry the **full per-domain ω** (math 2.08 / code 4.73 /
  logic 1.74 / science 1.48). Consistent with the earlier calibration where ω=2.08 closed the I3
  math step-time gap −65% → −28%.
- **prime-rl** scenario: keep ω ≈ 1.35 (upper edge of the 1.15–1.35 band).
- **Frontier scenarios (V4 / GLM / Qwen)**: set ω from the closest-matching domain row, and note it
  rises with policy strength.

## Provenance
`derived` from third-party published per-prompt solve rates. Script: `experiments/measure_oversampling.py`.
Data: `experiments/oversampling.json`. Datasets: `PrimeIntellect/INTELLECT-3-RL`,
`PrimeIntellect/INTELLECT-2-RL-Dataset`, `PrimeIntellect/Skywork-OR1-RL-Data-v1-math-prime-rl-format`.
