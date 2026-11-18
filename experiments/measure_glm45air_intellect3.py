#!/usr/bin/env python3
"""
measure_glm45air_intellect3.py -- Measure the RL rollout response-length distribution
for GLM-4.5-Air on INTELLECT-3's actual RL dataset.

Dataset: PrimeIntellect/INTELLECT-3-RL (confirmed on HF), which has FOUR splits:
math (21,161), code (8,579), logic (11,647), science (29,307) -- all a plain
`question` column, no chat template. This is the real INTELLECT-3 training mix,
which is why this measurement matters more than the others: the current preset
assumes E[R]=24,000 as a pure "frontier long-context" guess with zero empirical
grounding, and the model's biggest unexplained gap (INTELLECT-3, -65% at that
assumption) sits on this exact number.

INTELLECT-3 trains on all four splits together; this script defaults to sampling
math+code (the two the model's single-turn assumption actually covers -- logic/science
prompts may still be useful contrasts but are not what the presets.py scenario claims
to represent). Run --split separately for each if you want the full breakdown.

Endpoint: defaults to OpenRouter now. Set the key (PowerShell: $env:OPENROUTER_API_KEY="sk-or-...").
Override the base URL with $env:OPENROUTER_BASE_URL or --base-url; a Nscale key
still works as a fallback (--base-url https://inference.api.nscale.com/v1).

GLM-4.5-Air is a hybrid thinking model: if the selftest returns a trivially small
token count, thinking is off -- add --reasoning-effort high to force the CoT (and
so the reasoning_content the length measurement reads).

RUN:
    python experiments/measure_glm45air_intellect3.py --model "z-ai/glm-4.5-air" --selftest

    python experiments/measure_glm45air_intellect3.py \
        --model "z-ai/glm-4.5-air" --split math \
        --prompts 64 --group 8 --temperature 1.0 --max-tokens 49152 \
        --reasoning-effort high --out glm45air_math_lengths.json

    python experiments/measure_glm45air_intellect3.py \
        --model "z-ai/glm-4.5-air" --split code \
        --prompts 64 --group 8 --temperature 1.0 --max-tokens 49152 \
        --reasoning-effort high --out glm45air_code_lengths.json

Notes:
  * max-tokens default 49152 (~48K), inside the run's 65,536 context (minus prompt) --
    IMPORTANT: raise this to the actual cap if Nscale's serving config allows more;
    INTELLECT-3's real gap is suspected to be at very long context, so a low cap here
    would just reproduce the same truncation problem the model's over-optimism has.
  * If code is a much shorter split than math (plausible), that alone would explain
    part of why the flat E[R]=24,000 guess is wrong in one direction for one task and
    the other direction for the other -- report them separately, don't just average.
  * GLM-4.5-Air is MoE; if the endpoint doesn't support long enough context reliably,
    fall back to --max-tokens 16384 and treat the result as a lower-context proxy
    (note the caveat explicitly if you do).
"""

import argparse
import os
from rl_length_probe import fetch_prompts, normalize_base, resolve_key, run, add_common_args

DATASET = "PrimeIntellect/INTELLECT-3-RL"
SPLIT_SIZES = {"math": 21161, "code": 8579, "logic": 11647, "science": 29307}
SUFFIXES = {
    "math": "\n\nReason step by step, and put your final answer within \\boxed{}.",
    "code": ("\n\nWrite a complete, correct Python solution. Reason through the "
             "approach, then give the final code in a single ```python code block."),
    "logic": "\n\nReason step by step, then give your final answer clearly.",
    "science": "\n\nReason step by step, then give your final answer clearly.",
}


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "OPENROUTER_BASE_URL", default_prompts=64, default_group=8)
    ap.add_argument("--split", choices=list(SPLIT_SIZES), default="math",
                    help="INTELLECT-3-RL config to sample from")
    # raise the shared 32768 default: INTELLECT-3 runs at 65,536 context and the
    # suspected gap is AT long context, so a low cap would just re-create the
    # truncation bias we are trying to measure away. Default endpoint = OpenRouter.
    ap.set_defaults(max_tokens=49152,
                    base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    ap.add_argument("--out", default=None)
    cfg = ap.parse_args()
    cfg.key = resolve_key("OPENROUTER_API_KEY", "NSCALE_API_KEY",
                          "NSCALE_SERVICE_TOKEN", "OPENAI_API_KEY")
    cfg.base_url = normalize_base(cfg.base_url)
    cfg.suffix = SUFFIXES[cfg.split]
    out_path = cfg.out or f"glm45air_{cfg.split}_lengths.json"

    if cfg.selftest:
        run(cfg, [], "selftest", out_path); return

    print(f"fetching {cfg.prompts} INTELLECT-3-RL '{cfg.split}' prompts ...")
    prompts = fetch_prompts(DATASET, cfg.split, "train", "question", cfg.prompts,
                            seed=cfg.seed, rows_pool=SPLIT_SIZES[cfg.split])
    label = f"{cfg.model} on INTELLECT-3-RL:{cfg.split}"
    run(cfg, prompts, label, out_path)


if __name__ == "__main__":
    main()
