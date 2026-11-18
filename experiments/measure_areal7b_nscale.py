#!/usr/bin/env python3
"""
measure_areal7b_nscale.py -- Measure the RL rollout response-length distribution for
R1-Distill-Qwen-7B on the math dataset AReaL trained on (DeepScaleR), via an
OpenAI-compatible endpoint. This pins the E[R] that the AReaL-7B scenario in
validation/presets.py otherwise only assumes -- and settled whether its -35% gap was E[R]
or real model error (answer: mostly E[R]; measured ~10.1k at T=1.0 closed it to ~-15%).

PORTED to the shared rl_length_probe core (was a standalone pre-core implementation with its
own endpoint/retry/summarise code and NO streaming). Streaming matters: a non-streamed long
completion can exceed the gateway's response timeout, and because it is the LONG ones that get
dropped, the surviving mean is biased low. The shared core also brings the cluster-robust CI,
the censoring bracket, --resume and the categorised failure tally.

SET YOUR KEY AS AN ENV VAR -- never paste it into the file:
    # Windows PowerShell:
    $env:NSCALE_API_KEY="sk-..."; $env:NSCALE_BASE_URL="https://inference.api.nscale.com/v1"
    # bash:
    export NSCALE_API_KEY=sk-...   NSCALE_BASE_URL=https://inference.api.nscale.com/v1

RUN (always --selftest first: one tiny request, validates model id + URL + key):
    python experiments/measure_areal7b_nscale.py --model "<exact id>" --selftest

    python experiments/measure_areal7b_nscale.py --model "<exact id>" \
        --prompts 64 --group 8 --temperature 1.0 --max-tokens 32768 --resume

Notes on faithfulness to the AReaL run:
  * AReaL sampled at T=1.0 -- that is the run-faithful setting and the one the preset uses.
    T=0.6 is R1-Distill's own card recommendation; two T=0.6 replicates came out 8,714 and
    10,613, i.e. real run-to-run sampling spread, so report E[R] as a band not a point.
  * AReaL used G=16 rollouts/prompt; --group 16 reproduces the within-prompt spread but costs
    2x. ICC is ~0.8 here, so effective n ~= n_prompts: buy PROMPTS before samples/prompt.
  * --max-tokens should be the run's real cap (32,768). If the printed censored share is
    non-trivial the mean is a LOWER BOUND; the summary prints the bracket.
"""
import argparse

from rl_length_probe import fetch_prompts, normalize_base, resolve_key, run, add_common_args

DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
ROWS_POOL = 40315          # DeepScaleR train rows
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "NSCALE_BASE_URL", default_prompts=64, default_group=8)
    ap.add_argument("--out", default="lengths/areal7b_lengths.json")
    cfg = ap.parse_args()
    cfg.key = resolve_key("NSCALE_API_KEY", "NSCALE_SERVICE_TOKEN", "OPENAI_API_KEY")
    cfg.base_url = normalize_base(cfg.base_url)
    cfg.suffix = MATH_SUFFIX

    if cfg.selftest:
        run(cfg, [], "selftest", cfg.out)
        return

    print(f"fetching {cfg.prompts} DeepScaleR math prompts ...")
    prompts = fetch_prompts(DATASET, "default", "train", "problem", cfg.prompts,
                            seed=cfg.seed, rows_pool=ROWS_POOL)
    run(cfg, prompts, f"{cfg.model} on DeepScaleR (math)", cfg.out)


if __name__ == "__main__":
    main()
