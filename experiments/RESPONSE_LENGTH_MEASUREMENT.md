# Measuring E[R]: the response-length distribution in RL rollouts

Why E[R] had to be measured, how, what came out across every run so far, and what it
changed in the runtime model. Companion docs: `AGENTIC_TRACE_MEASUREMENT.md` (multi-turn
effective context) and `OVERSAMPLING_MEASUREMENT.md` (ω).

## 1. Why this measurement exists

`RLSpec.response_len_mean` (E[R]) and `response_len_cv` were the model's largest free
parameters, and they enter almost every term:

| term | scaling |
|---|---|
| decode compute, trainer tokens/batch | ∝ E[R] |
| attention (train + decode), KV traffic | ∝ E[R²] = E[R]²(1+cv²) |
| KV footprint / concurrency `seq_par` | ∝ context, tail-sensitive |

Because nearly everything scales with E[R], an inverse solve for E[R] can absorb *any*
under-prediction and still look plausible — so a step-time match tuned by choosing E[R]
was partly circular. **Measuring E[R] independently converts a fitted parameter into a
fixed input, so the remaining error becomes attributable** (E[R] error vs genuine model error).

## 2. Method

**(a) Dataset proxy — keyless, free.** Tokenise reasoning traces published in HF datasets
with the Qwen2.5/QwQ tokenizer. Real traces on real tasks, but SFT/distillation outputs from
a *teacher*, not live RL rollouts. `measure_response_lengths.py dataset --hf <ds> --col <col>`.

**(b) Live generation — needs an endpoint.** Sample G completions/prompt from the *actual
policy*, at the run's temperature and cap, on the run's *actual dataset*. Length read from the
server's `usage.completion_tokens` (exact; `max`'d with `reasoning_tokens` so closed models that
hide CoT still yield the count). `rl_length_probe.py` core + per-run drivers; `measure_domain_sweep.py`
holds a model fixed and sweeps domains. Frontier models via OpenRouter / DeepSeek (DeepSeek-R1/V4
return `reasoning_content`, so full distributions without self-hosting).

Reporting always includes: **bootstrap 95% CI on the mean** (run-to-run spread is large, §4 F2);
**censored fraction** (samples at the cap → mean is a lower bound, a bracket is printed);
**within/between-prompt variance (ICC)** (buy-more-prompts vs -samples, and the correlation the
tail term needs). Streaming is on by default (long generations otherwise get dropped, biasing the
mean low). Trace capture (`--save-traces`) reads reasoning under both `reasoning_content` *and*
`reasoning` (OpenRouter renamed it — earlier traces silently lost the text; length stats were
unaffected as they come from `usage`).

## 3. Results

### 3.1 Dataset proxies (teacher traces)

| source | model · task | n | mean | median | cv | p95 | max |
|---|---|---|---|---|---|---|---|
| OpenR1-Math-220k | R1 671B · math | 506 | 5,795 | 4,552 | 0.69 | 15,031 | 18,684 |
| codeforces-cots | R1 671B · code | 240 | 12,696 | 13,332 | 0.52 | 22,637 | 24,200 |
| QwQ-LongCoT-130K | QwQ-32B · math | 450 | 2,280 | 1,316 | 1.09 | 7,957 | 13,674 |
| s1K-1.1 | Gemini-thinking · math | 1000 | 4,692 | 4,970 | 0.31 | 6,682 | 7,995 |
| s1K-1.1 | DeepSeek-R1 · math | 1000 | 9,292 | 8,974 | 0.49 | 16,525 | 26,238 |

### 3.2 Validation-anchor policy sampling (tied to the calibration runs)

| run · model · task | n | mean | median | cv | cens | cap |
|---|---|---|---|---|---|---|
| prime-rl · R1-Distill-32B · Skywork math (filtered) | 256 | **7,536** | 7,637 | 0.34 | 0% | 32,768 |
| AReaL-7B · R1-Distill-7B · DeepScaleR math (T=1.0) | 256 | 10,125 | 7,040 | 0.77 | 2% | 32,768 |
| AReaL-7B · same (T=0.6) | 768 | 9,347 | 5,698 | 0.92 | 6% | 32,768 |
| AReaL-14B · R1-Distill-14B · DeepCoder code | 512 | 10,376 | 10,616 | 0.45 | 0% | 32,768 |
| AReaL-32B · Qwen3-32B(proxy) · DeepCoder code | 256 | 8,137 | 7,858 | 0.28 | 0% | 32,768 |

### 3.3 Frontier single-turn, cross-domain (INTELLECT-3-RL prompts, T=1.0, thinking on)

Mean E[R] / cv; **bold** = censored (lower bound).

| model | math | code | logic | science |
|---|---|---|---|---|
| DeepSeek-V4-Pro | 11,553 / 0.86 | **39,180** / 0.62 (10% @131k) | 4,338 / 1.30 | 8,163 / 1.39 |
| DeepSeek-V4-Flash | 12,312 / 1.10 (2%) | **22,553** / 0.84 (6% @65k) | 4,097 / 1.61 | 5,361 / 1.61 |
| GLM-4.5-Air (I3 base) | 17,016 / 0.54 | 22,961 / 0.47 | — | — |

n = 128 (256 for science / GLM). GLM-4.5-Air is INTELLECT-3's actual policy — its code ~23k / math
~17k anchor the I3 preset (guessed 24k) as roughly right.

### 3.4 Qwen3 family — size scaling (INTELLECT-3-RL prompts, thinking on, seed 0, n≈128)

| model | params (act.) | math mean / cv | code mean / cv | note |
|---|---|---|---|---|
| qwen3-8b | 8.2B dense | 15,359 / 0.51 | 16,417 / 0.40 | |
| qwen3-14b | 14.8B dense | 15,270 / 0.52 | 17,272 / 0.39 | |
| qwen3-32b | 32.8B dense | **33,693** / 0.41 | **38,386** / 0.30 | 38–42% censored at a ~47,103 provider wall; code run partial (n=60) |
| qwen3-30b-a3b | 3.3B (MoE 30B) | 16,574 / 0.58 | 17,232 / 0.42 | |
| qwen3-235b-a22b | 22B (MoE 235B) | 14,275 / 0.43 | 16,294 / 0.39 | shortest on math |

*(Multi-turn / agentic effective context — ~50k median, ~19k generated — is in `AGENTIC_TRACE_MEASUREMENT.md`.)*
*(A tiny n=4 DeepSeek-R1 functionality-test run exists in `lengths/` but is too small to use.)*

## 4. Findings

**F1. E[R] is not one number — it moves ~2× along each of three axes, and they multiply.**
Controlled (s1K-1.1, *same 1000 prompts*): Gemini 4,692 vs DeepSeek-R1 9,292 → **~2× from the model**.
R1 on OpenR1-Math 5,795 vs harder s1K 9,292 → **~1.6× from difficulty**. Math vs code → **~2× from task**.
No global preset can be right across the 6k–39k range now measured.

**F2. The domain multiplier is large but itself model-family-dependent.** Relative to its own math,
V4-Pro's code is **3.4×** (39k vs 11.5k), GLM's is 1.35×, but Qwen3's is only ~1.05× (code≈math). So
"code ≫ math" is a property of some families, not a law — the multiplier must be measured per family,
not transplanted.

**F3. Within a family, response length tracks the post-training recipe, not parameter count.**
Qwen3 dense 8B≈14B (~15k) and even the 235B MoE (~14k, the *shortest* on math) sit flat; only the 32B
jumps (≥34k). The Qwen3 report (arXiv 2505.09388) explains it: **8B/14B are strong-to-weak distillations
of the flagships, while 32B/235B went through full reasoning-RL** (which lengthens CoT); thinking-mode
fusion + higher-capability efficiency then compress the 235B back down. Distilled → short, RL'd → long,
independent of scale. Governance-relevant: you can't infer an evader's E[R] from model size alone.

**F4. Frontier policies reason far longer than the distill anchors — on code especially.**
Code E[R] climbs 10.4k (R1-Distill-14B) → 16–17k (Qwen3) → 23k (GLM) → 39k+ (V4-Pro). This raises
feasibility per-step compute well above the distill-based presets — cutting toward *more* expensive RL
than earlier numbers implied. Math is noisier and non-monotonic (V4-Pro math 11.5k < Qwen3 15k).

**F5. Distributions are heavy-tailed; cv is a real term.** Measured cv spans 0.28–1.61, mean > median
throughout, p95 ≈ 2–3× mean. Since attention/KV scale with E[R²]=E[R]²(1+cv²), cv=1.6 inflates them 3.6×.
Frontier models (V4) show the highest cv (0.9–1.6, bimodal); Qwen3 thinking is more length-consistent
(0.3–0.6). Peak-provisioning/straggler terms should key off a high quantile, not the mean.

**F6. Long tails are mostly genuine reasoning, not repetition loops — with one class of exception.**
Trace loop-check (zlib compression + unique-line ratio): p90–p99 samples are healthy (zlib ~3–4×, 82–100%
unique). The *cap-hitting extreme* can degenerate — one V4-Pro 131k sample was a pure verification loop
(zlib 7.3×, 43% unique). Qwen3-32B's censored ~47k mass is **cleared** (zlib 4.3–6×, 99–100% unique) — its
length is real, only provider-capped. Implication: right-censor the extreme tail for capability, but for
*compute* accounting the loop tokens are still paid.

**F7. Measuring E[R] resolved the calibration split it was designed to test.**
- **prime-rl now measured (7,536) ≈ its triangulated 7,470** → the +1% anchor is real, not circular. Best anchor.
- **AReaL-7B**: guessed 8k → measured 10.1k closed the gap −35% → −15% (η≈1.18). Mostly E[R].
- **AReaL-14B**: proxy 13k → measured **shorter** (10.4k), which *widens* the gap → the large-model gaps are
  **genuine efficiency error** (large-shard/TP penalty), not E[R]. This was the hypothesis under test.
- **GLM/INTELLECT-3**: measured 17k math / 23k code ≈ the 24k guess → I3's −65% gap is **not** E[R]; it's
  MoE-TP + multi-turn (see agentic doc).

**F8. Sampling noise is large; proxies set scale, not value.** Two identical AReaL-7B T=0.6 runs gave 8,714
vs 10,613 (22% swing) — hence the CIs. Proxy vs live disagreed in *direction* (code proxy 21% high, math
proxy 43% low), so teacher traces bracket the scale but can't substitute for the policy.

## 5. Consolidated E[R] / cv for presets

Use per (model-class, task). Frontier rows from §3.3–3.4; distill/anchor rows from §3.2.

| regime | math E[R]/cv | code E[R]/cv | source |
|---|---|---|---|
| distill (R1-Distill 7–14B) | ~10,000 / 0.8 | ~10,400 / 0.45 | AReaL live |
| prime-rl (R1-Distill-32B, filtered math) | 7,536 / 0.34 | — | live, = anchor |
| INTELLECT-3 (GLM-4.5-Air) | 17,016 / 0.54 | 22,961 / 0.47 | live, on-policy |
| frontier dense (Qwen3-32B) | ≥33,693 / 0.41 | ≥38,386 / 0.30 | live, provider-capped |
| frontier MoE (V4-Pro) | 11,553 / 0.86 | ≥39,180 / 0.62 | live, code censored |
| frontier MoE (V4-Flash) | 12,312 / 1.10 | ≥22,553 / 0.84 | live, both censored |

logic/science (V4): ~4k / ~5–8k, cv 1.3–1.6 — much shorter, higher variance.

## 6. Still open

- **RL lengthening over training.** All measurements are of a *released checkpoint*; responses lengthen as
  RL proceeds, so a run-average E[R] exceeds the initial-policy E[R]. Sampling successive public checkpoints
  would bound this — the main remaining fidelity gap.
- **Provider-capped 32B** — qwen3-32b clamps at ~47k, so its true mean is unmeasurable at that provider; it's
  a lower bound and a clear outlier, not worth chasing for one point.
- **Closed frontier anchor** (Qwen3.x-Max / Kimi K3-Max) — a small counts-only ceiling check for the
  scaling-story, low leverage for the feasibility spine (evaders reproduce open weights). Optional.
- **Sampling temperature**. Generally T=1 for tested data.

## 7. Caveats

1. Live measurements are of the **released checkpoint**, not mid-RL policies (see §6).
2. **Censoring:** V4/Flash code and math, and the whole qwen3-32b row, are cap-censored → those means are
   lower bounds (brackets in the raw output).
3. Proxy traces (§3.1) come from **teacher** models (R1 671B), which run longer than the small distills.
4. Cross-*dataset* proxy rows confound model × difficulty; only the s1K pair is a controlled model comparison.
5. Frontier cross-domain rows are on INTELLECT-3-RL prompts — a fixed prompt set across models (controlled),
   but off-policy for everything except GLM-4.5-Air.
6. Sample sizes are modest (n≈128–512 live); CIs are reported for exactly this reason.
