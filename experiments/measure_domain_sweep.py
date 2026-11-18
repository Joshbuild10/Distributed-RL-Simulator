#!/usr/bin/env python3
"""
measure_domain_sweep.py -- One model, many domains.

Measures the response-length distribution of a FIXED reasoning model across a panel
of task domains, holding sampling constant so the only thing varying is the domain.
This pins the DOMAIN MULTIPLIER on E[R] (math vs code vs logic vs science) that the
runtime model's per-(model, task) response_len_mean input needs, and -- for a
frontier policy like DeepSeek-R1 -- gives a frontier-scale E[R] reference per domain.

Default panel = the four PrimeIntellect/INTELLECT-3-RL splits (math 21,161 /
code 8,579 / logic 11,647 / science 29,307), the real INTELLECT-3 RL mix, all a
plain `question` column. Reuses rl_length_probe (streaming, cluster-robust CI,
censoring bracket, per-split resume).

Works against any OpenAI-compatible endpoint:
  * DeepSeek-R1 (primary, full distributions -- no self-hosting):
      DeepSeek API   base https://api.deepseek.com/v1   model deepseek-reasoner
      OpenRouter     base https://openrouter.ai/api/v1  model deepseek/deepseek-r1
      Nscale         base $NSCALE_BASE_URL               model <provider's R1 id>
    R1 returns reasoning_content, so the full length distribution is measured, not
    just a token count.
  * Closed frontier (tier split -- COUNT only, CoT text hidden):
      OpenRouter with --reasoning-effort high, model openai/gpt-5, openai/o3,
      anthropic/claude-*, google/gemini-2.5-pro, on the SAME prompts.

KEY (first match wins): OPENROUTER_API_KEY, DEEPSEEK_API_KEY, NSCALE_API_KEY,
                        NSCALE_SERVICE_TOKEN, OPENAI_API_KEY

RUN (always --selftest first: one tiny request, validates model id + URL + key):
    python experiments/measure_domain_sweep.py --model deepseek/deepseek-v4-flash \
        --base-url https://openrouter.ai/api/v1 --reasoning-effort high --selftest

    python experiments/measure_domain_sweep.py --model deepseek/deepseek-v4-flash \
        --base-url https://openrouter.ai/api/v1 --reasoning-effort high \
        --splits math,code,logic --prompts 32 --group 4 --temperature 1.0 --resume

    # science is the long tail -- run it alone with a higher cap
    python experiments/measure_domain_sweep.py --model deepseek/deepseek-v4-flash \
        --base-url https://openrouter.ai/api/v1 --reasoning-effort high \
        --splits science --prompts 32 --group 4 --max-tokens 131072 --resume

Default --max-tokens is 65,536 (raised from the shared 32,768 default -- this is a
fresh probe, not replicating a run's real training cap the way the AReaL/prime-rl
scenario scripts do, so there's no reason to truncate early; the provider bills
actual tokens generated, not the cap, so raising it costs nothing unless a
generation actually runs that long -- which is exactly what would otherwise be
invisibly censored). Add --save-traces to also keep the raw reasoning/content text
(appended to <out>.traces.jsonl) for later qualitative inspection (repetition
loops, bimodality, distribution-shape fitting) without re-generating.

Cost note (DeepSeek V4 pricing, confirmed on OpenRouter, 2026-08): Flash
$0.06426/$0.1285 per 1M in/out tokens, Pro $0.348/$0.696 -- at this campaign's
scale (a few hundred samples/domain, 10-30k tok each) total cost is single-digit
to low-double-digit dollars even on Pro. Start small (--prompts 8 --group 2) to
confirm it works, then scale --prompts up freely; the CI/ICC print tells you
whether to buy more prompts or more samples/prompt. High ICC (~0.8 seen
elsewhere in this project) means effective n ~= n_prompts.
"""

import argparse
import _bootstrap  # noqa: F401  (repo root on sys.path)
import lengths_io
import os

from rl_length_probe import (fetch_prompts, normalize_base, resolve_key, run,
                             add_common_args)

OUTDIR = "experiments/lengths"
DATASET = "PrimeIntellect/INTELLECT-3-RL"
SPLIT_SIZES = {"math": 21161, "code": 8579, "logic": 11647, "science": 29307}
SUFFIXES = {
    "math": "\n\nReason step by step, and put your final answer within \\boxed{}.",
    "code": ("\n\nWrite a complete, correct Python solution. Reason through the "
             "approach, then give the final code in a single ```python code block."),
    "logic": "\n\nReason step by step, then give your final answer clearly.",
    "science": "\n\nReason step by step, then give your final answer clearly.",
}


def domain_table(tag, splits):
    """Re-read the per-split JSONs run() wrote and print the domain multiplier
    (E[R] relative to math), so a single sweep reports the cross-domain ratio."""
    rows = []
    for sp in splits:
        path = f"{OUTDIR}/{tag}_{sp}_lengths.json"
        st = next(iter(lengths_io.load_stats(path).values()), None)
        if not st:
            continue
        rows.append((sp, st["mean"], st["n"], st["censored_frac"]))
    if not rows:
        print("\n  (no per-split results to summarise)")
        return
    base = next((m for s, m, _, _ in rows if s == "math"), rows[0][1])
    print("\n" + "=" * 60)
    print(f"  DOMAIN MULTIPLIER  --  {tag}")
    print("=" * 60)
    print(f"  {'domain':<10}{'E[R]':>11}{'x math':>9}{'n':>7}{'cens%':>7}")
    for sp, m, n, fc in rows:
        flag = "  <- censored, E[R] is a lower bound" if fc > 0.02 else ""
        print(f"  {sp:<10}{m:>11,.0f}{m/base:>8.2f}x{n:>7}{100*fc:>6.0f}%{flag}")
    print("  (x math = E[R]_domain / E[R]_math; the multiplier the runtime model needs)")


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "OPENROUTER_BASE_URL", default_prompts=64, default_group=4)
    ap.add_argument("--splits", default="math,code,logic,science",
                    help="comma-separated INTELLECT-3-RL configs to sweep")
    ap.add_argument("--tag", default=None,
                    help="output-filename tag (default: derived from the model id)")
    # Raise the shared 32,768 default: this is a fresh probe (no real run's cap to
    # match), the provider bills actual tokens not the cap, and a low cap would just
    # re-create the truncation bias this measurement exists to remove.
    ap.set_defaults(max_tokens=65536)
    cfg = ap.parse_args()
    cfg.key = resolve_key("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "NSCALE_API_KEY",
                          "NSCALE_SERVICE_TOKEN", "OPENAI_API_KEY")
    cfg.base_url = normalize_base(cfg.base_url)
    tag = cfg.tag or cfg.model.split("/")[-1].replace(":", "_")
    splits = [s.strip() for s in cfg.splits.split(",") if s.strip() in SPLIT_SIZES]

    os.makedirs(OUTDIR, exist_ok=True)

    if cfg.selftest:
        cfg.suffix = ""
        run(cfg, [], "selftest", f"{OUTDIR}/{tag}_selftest.json")
        return

    for sp in splits:
        cfg.suffix = SUFFIXES[sp]
        out_path = f"{OUTDIR}/{tag}_{sp}_lengths.json"
        print(f"\n{'#'*60}\n### DOMAIN: {sp}  ->  {out_path}\n{'#'*60}")
        prompts = fetch_prompts(DATASET, sp, "train", "question", cfg.prompts,
                                seed=cfg.seed, rows_pool=SPLIT_SIZES[sp])
        run(cfg, prompts, f"{cfg.model} on INTELLECT-3-RL:{sp}", out_path)

    domain_table(tag, splits)


if __name__ == "__main__":
    main()
