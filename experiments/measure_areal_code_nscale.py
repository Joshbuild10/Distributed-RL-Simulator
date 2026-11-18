#!/usr/bin/env python3
"""
measure_areal_code_nscale.py -- Measure the RL rollout response-length distribution
for R1-Distill-Qwen-14B or -32B on the CODE dataset AReaL actually trained on.

AReaL's own paper (arXiv 2505.24298, Appendix A.2): "For code training, we used the
dataset released by DeepCoder" -- agentica-org/DeepCoder-Preview-Dataset. Its TRAIN
splits are taco (7,436), primeintellect (16,252), and lcbv5 (599); codeforces is
test-only so excluded. Default samples from all three, weighted by their relative
size, matching DeepCoder's actual training mix (not just one arbitrary subset).

This is the highest-value remaining measurement: AReaL 14B/32B currently under-predict
step time by 31%/53% using a generic codeforces-cots proxy for E[R] (~13k). If the
live-measured E[R] on the REAL training data comes in near that, the gap is genuine
model/efficiency error (large-shard training penalty); if it's much higher, it was an
E[R] problem all along -- exactly what happened when AReaL-7B was measured on math.

SET YOUR KEY:  export NSCALE_API_KEY=sk-...   (or NSCALE_SERVICE_TOKEN)
              export NSCALE_BASE_URL=https://inference.api.nscale.com/v1

RUN:
    # cheap smoke test first:
    python experiments/measure_areal_code_nscale.py --model "<id>" --selftest

    python experiments/measure_areal_code_nscale.py \
        --model "<exact R1-Distill-Qwen-14B id on Nscale>" \
        --prompts 64 --group 8 --temperature 1.0 --max-tokens 32768 --concurrency 8 \
        --out areal14b_code_lengths.json

    python experiments/measure_areal_code_nscale.py \
        --model "<exact R1-Distill-Qwen-32B id on Nscale>" \
        --prompts 64 --group 8 --temperature 1.0 --max-tokens 32768 --concurrency 8 \
        --out areal32b_code_lengths.json

Notes:
  * temperature 1.0 default: AReaL's own reported sampling temperature (per Josh's
    correction on the math run) -- use the SAME regime for code for consistency.
  * max-tokens 32768 default, matching AReaL's context cap. Watch the truncation %
    the script reports; raise if it's above ~5%.
  * The three DeepCoder subsets differ a lot (LiveCodeBench-style vs TACO vs
    PrimeIntellect-style problems) -- use --config to isolate one if you want to see
    the per-subset spread instead of the pooled mean.
"""

import argparse
from rl_length_probe import fetch_prompts, normalize_base, resolve_key, run, add_common_args

DATASET = "agentica-org/DeepCoder-Preview-Dataset"
# (config, split, weight) -- weight = relative size of each TRAIN split in DeepCoder
SPLITS = [("taco", "train", 7436), ("primeintellect", "train", 16252), ("lcbv5", "train", 599)]
CODE_SUFFIX = ("\n\nWrite a complete, correct Python solution. Reason through the "
               "approach, then give the final code in a single ```python code block.")


def fetch_pooled_prompts(n, seed, config_filter=None):
    splits = [s for s in SPLITS if config_filter is None or s[0] == config_filter]
    total_w = sum(w for _, _, w in splits)
    prompts = []
    for cfg, split, w in splits:
        n_this = max(1, round(n * w / total_w))
        starter = "starter_code" if cfg == "lcbv5" else None
        prompts += fetch_prompts(DATASET, cfg, split, "problem", n_this, seed=seed,
                                 rows_pool=w, starter_code_col=starter)
        seed += 1
    return prompts[:n] if len(prompts) > n else prompts


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "NSCALE_BASE_URL", default_prompts=64, default_group=8)
    ap.add_argument("--config", choices=[s[0] for s in SPLITS], default=None,
                    help="restrict to one DeepCoder subset instead of the pooled mix")
    ap.add_argument("--out", default="areal_code_lengths.json")
    cfg = ap.parse_args()
    cfg.key = resolve_key("NSCALE_API_KEY", "NSCALE_SERVICE_TOKEN", "OPENAI_API_KEY")
    cfg.base_url = normalize_base(cfg.base_url)
    cfg.suffix = CODE_SUFFIX

    if cfg.selftest:
        run(cfg, [], "selftest", cfg.out); return

    print(f"fetching {cfg.prompts} DeepCoder problems"
          f"{' (' + cfg.config + ' only)' if cfg.config else ' (pooled: taco+primeintellect+lcbv5)'} ...")
    prompts = fetch_pooled_prompts(cfg.prompts, cfg.seed, cfg.config)
    label = f"{cfg.model} on DeepCoder {cfg.config or 'pooled'} (code)"
    run(cfg, prompts, label, cfg.out)


if __name__ == "__main__":
    main()
