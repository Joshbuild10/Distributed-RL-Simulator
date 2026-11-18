# Agentic trajectory length — first-order measurement

**Goal.** Pin the *effective context* and *generated-token* magnitudes of multi-turn / agentic
RL, which the single-turn response-length probes (`measure_domain_sweep.py`) cannot reach.
This is the regime that dominates feasibility per-step compute and that the runtime model's
single-turn assumption most under-represents (cf. INTELLECT-3's SWE mix, the −69% gap).

**Method.** Tokenise *released* agent trajectories — no API, no agent harness, no cost.
`experiments/measure_agentic_traces.py` pulls trajectories via the keyless HF datasets-server
and tokenises them with the Qwen2.5/QwQ tokenizer. Two metrics, which differ and are both needed:

- **Effective context** = total tokens across *all* messages (system + user + assistant + tool).
  The context the model carries by end-of-trajectory → drives the **quadratic attention / KV**
  terms and tends to flip a run update-bound.
- **Generated** = assistant-authored tokens only (content + tool-call arguments) → drives
  **decode** compute (what the inference pool actually generates).

Sources (two independent agent stacks, as a cross-check):

| tag | dataset | scaffold / policy | format |
|---|---|---|---|
| nebius | `nebius/SWE-rebench-openhands-trajectories` | OpenHands + Qwen3-Coder-480B | structured role-tagged messages → full gen/tool/turn split |
| swesmith | `SWE-bench/SWE-smith-trajectories` (`tool` split) | SWE-agent + Claude 3.7 Sonnet | single serialized string → total only |

**Reproduce.**
```bash
python experiments/measure_agentic_traces.py --source nebius   --n 120 --seed 0 --out experiments/lengths/agentic_nebius.json
python experiments/measure_agentic_traces.py --source swesmith --n 120 --seed 0 --split tool --out experiments/lengths/agentic_swesmith.json
```
Raw console → `experiments/lengths/agentic_trace_log.txt`; per-trajectory arrays → the two `agentic_*.json`.

## Results (n = 120 each, seed 0)

**nebius / SWE-rebench (OpenHands + Qwen3-Coder-480B):**

| metric | mean | median | cv | p90 | p95 | max |
|---|---|---|---|---|---|---|
| effective context | 54,308 | 50,220 | 0.30 | 79,270 | 84,728 | 106,699 |
| generated | 19,277 | 17,896 | 0.35 | 29,632 | 32,780 | 42,288 |
| tool output | 31,382 | 28,706 | 0.37 | 48,406 | 52,421 | 69,269 |
| turns | 66 | 63 | 0.28 | 100 | 100 (cap) | 100 |

resolved (n=53): effective-ctx mean **49,686**  ·  unresolved (n=67): **57,963**

**swesmith / SWE-smith (SWE-agent + Claude 3.7), `tool` split:**

| metric | mean | median | cv | p90 | p95 | max |
|---|---|---|---|---|---|---|
| effective context | 37,018 | 28,767 | 0.65 | 73,353 | 94,419 | 144,293 |

resolved (n=60): mean **30,420**  ·  unresolved (n=60): **43,616**

## Findings

1. **Agentic effective context is ~30–50k median (p95 ~85–94k, max ~107–144k)** — roughly an
   order of magnitude above single-turn logic/science (~2–4k) and ~1.3–2× single-turn code
   (~25–39k). Two independent agent stacks agree on the magnitude. This is the input the
   runtime model's quadratic terms should use for agentic environments.

2. **Tool output dominates the context (58%: 31k of 54k on nebius); the model *generates*
   only ~19k.** In agentic RL most of the context is environment-authored, not model-authored,
   so **decode volume is decoupled from context size** — the inference pool prefills/attends over
   ~54k but only decodes ~19k. Single-turn accounting conflates these; the multi-turn extension
   must split them: **prefill/attention ∝ effective context, decode ∝ generated**. This is the
   cleaner, more defensible structural finding than any single agentic "E[R]".

3. **Failed trajectories run longer than successful ones** on both stacks (nebius 58.0k vs 49.7k,
   +17%; swesmith 43.6k vs 30.4k, +43%). Since RL rollouts include failures (and oversampling
   generates many), the RL-weighted mean is biased *upward* relative to these (curated, often
   resolved-heavy) SFT samples.

4. **Turn-cap censoring** — nebius trajectories cap at 100 turns (p90 = p95 = 100), compressing
   its spread (cv 0.30). swesmith has no visible turn cap → longer tail (max 144k) and higher
   cv (0.65). Real agentic RL with larger turn budgets would push the tail out further.

## Caveats (state next to any number)

- **Off-policy.** These are Claude-3.7 / Qwen3-Coder-480B trajectories, not the policy being
  modelled. Length is scaffold- and policy-dependent, so treat as magnitude, not policy-exact E[R].
- **SFT-curated / resolved-biased.** Released trace sets are filtered for quality/success, so the
  failed-loop long tail an RL run actually pays for is under-represented → **lower-ish** first-order
  estimate, not a ceiling.
- **swesmith is a serialized string** (role tags + JSON scaffolding included) → its effective-ctx
  slightly over-counts pure text and has no gen/tool split. Trust nebius for the clean number;
  swesmith corroborates magnitude and the resolved<unresolved ordering.
- **Scaffold spread.** OpenHands (nebius) accumulates more tool output than SWE-agent (swesmith),
  which explains most of the 50k-vs-29k median gap — a reminder that the harness, not just the
  task, sets effective context.

## Implications

- **Runtime model (multi-turn extension):** for agentic environments, size the prefill/attention/KV
  terms on **effective context ~50k** (not the single-turn E[R]) and the decode term on
  **generated ~19k**. This is the mechanism behind INTELLECT-3's SWE gap being context-growth, not
  re-prefill (consistent with the earlier crude accounting).
- **Feasibility / governance:** agentic RL shifts the compute profile toward **prefill/attention over
  large environment-authored contexts** and away from pure decode — the inference pool is
  provisioned for context volume, not generation volume. Worth carrying into the trainer:inference
  split and the "which workload is more identifying" argument.

## Cross-environment extension (2026-08-16)

The SWE numbers above are one environment type. Adding tool-use and agentic-coding shows
**"agentic" is not one length** — effective context spans ~1.8 orders of magnitude by env type.
One representative released trace set per type (browser/web omitted: no clean, ungated public
trajectory dump exists — WebArena etc. are environments, not released traces).

| env type | source | eff-ctx median | eff-ctx mean | cv | generated (med) | turns (med) | n |
|---|---|---|---|---|---|---|---|
| tool-use | `Jarrodbarnes/tau2-sft-v4-dataset` | 1,847 | 2,044 | 0.46 | 1,021 | 1 | 120 |
| SWE | `SWE-bench/SWE-smith-trajectories` | 28,767 | 37,018 | 0.65 | — | — | 120 |
| SWE | `nebius/SWE-rebench-openhands-trajectories` | 50,220 | 54,308 | 0.30 | 17,896 | 63 | 120 |
| agentic-coding | `zai-org/CC-Bench-trajectories` | 122,911 | 203,916 | 1.07 | 11,324\* | 63 | 100 |

\* CC-Bench ships its own (provider-tokenizer) counts; "generated" here is its cumulative
output. CC-Bench also reports **cumulative input 2.71M median** → **prefill:decode ≈ 274:1**
(no-prefix-cache).

**Findings**

5. **Agentic ≠ one length.** Median effective context: **tool-use ~1.8k → SWE ~30–50k →
   agentic-coding ~123k** (mean 204k, p95 750k, max 1.05M at 441 turns). The runtime model needs
   a **per-environment-type range**, not a single agentic E[R].
6. **Generation stays modest everywhere (~1–23k); context is what explodes.** Even agentic-coding
   only *generates* ~11k while carrying ~123k of context. Reinforces the SWE finding: **decode ∝
   generated (bounded ~10–20k), prefill/attention ∝ effective context (env-dependent, up to 100k+).**
7. **Agentic-coding is prefill-dominated (~274:1 cumulative in:out).** Prefix caching is the
   decisive modelling lever: with caching, per-step prefill collapses toward the effective context
   (~123k); without, cumulative prefill is ~2.7M. The `prefix_caching` toggle + prefill/decode split
   dominate the agentic compute estimate.
8. **Heavy tail (cv 1.07, p95 750k).** A minority of very-long agentic-coding sessions set the
   peak-provisioning / KV memory — size those on p95/max, not the mean.

**Extension caveats.** tau2 rows are short SFT prefixes (turns median 1) → ~2k is a *lower bound*;
tool-use is unambiguously the short end regardless. CC-Bench effective-ctx is QwQ-tokenised from a
serialized string (slight over-count, like swesmith); its in/out counts use the provider tokenizer
(the 274:1 *ratio* is tokenizer-robust, the absolute token counts are not directly comparable to the
QwQ-tokenised columns). The CC-Bench fetch is rate-limited — the n=100 here uses 429 backoff; an
earlier n=12 run was 429-biased and is superseded.

## Provenance
`derived` (our tokenisation of third-party released trajectories). Datasets:
`nebius/SWE-rebench-openhands-trajectories`, `SWE-bench/SWE-smith-trajectories`,
`Jarrodbarnes/tau2-sft-v4-dataset`, `zai-org/CC-Bench-trajectories`. Script:
`experiments/measure_agentic_traces.py`. Log: `experiments/lengths/agentic_trace_log.txt`;
per-trajectory arrays: `experiments/lengths/agentic_{nebius,swesmith,tau2,ccbench}.json`.
Next fidelity step: on-policy trajectories from the actual modelled policy under a real agent loop
(the deferred harness), a full-trajectory tool-use source (tau-bench `traj` split, round-grouped),
and a higher/curriculum turn budget to characterise the uncensored tail.
