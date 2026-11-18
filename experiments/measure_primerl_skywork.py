#!/usr/bin/env python3
"""
measure_primerl_skywork.py -- Measure the RL rollout response-length distribution for
R1-Distill-Qwen-32B on prime-rl's ACTUAL training data.

WHY THIS ONE MATTERS MOST
-------------------------
prime-rl is the runtime model's only clean anchor: step time +1%, trainer tok/s -1%,
total runtime -4%. But its E[R]=7,470 was never measured -- it was TRIANGULATED from
the paper's own published throughputs:
    trainer 11.3K tok/s * 1374 s / 2048 = 7,570 = P + E[R]
That makes the +1% partly circular: E[R] was chosen so the throughput matches, so of
course the step time does too. An independent measurement is the only way to tell
whether the model is actually right here, or whether two errors cancel.

Three outcomes, all informative:
  * measured ~7.5k  -> the anchor is real and independent. The model's dense/
    single-node envelope is genuinely validated, and the 1.5-2.5x gaps elsewhere
    are definitely efficiency, not accounting.
  * measured much higher -> the model was under-predicting here too and the
    triangulated E[R] hid it. The "one clean anchor" claim collapses and every
    downstream eta estimate shifts.
  * measured much lower -> the model over-predicts rollout and something else
    (e.g. the trainer term) is compensating.

DATASET
-------
PrimeIntellect/Skywork-OR1-RL-Data-v1-math-prime-rl-format (105,055 rows) -- the
prime-rl-formatted Skywork-OR1 math set, with a plain-string `prompt` column.

prime-rl filtered this to solve rate [0.001, 0.999], i.e. it DROPPED problems the
policy always solves and always fails. The dataset publishes per-prompt solve rates
measured with the exact policy prime-rl used (solve_rate_qwen_r1_distill_32b), so
this script replicates that filter by default (--no-filter to disable). This matters:
always-solved problems are short and always-failed ones run to the cap, so sampling
the unfiltered pool would measure a materially different distribution than the one
the run actually trained on.

SET YOUR KEY:
    export NSCALE_API_KEY=sk-...        # or NSCALE_SERVICE_TOKEN
    export NSCALE_BASE_URL=https://inference.api.nscale.com/v1

RUN:
    python experiments/measure_primerl_skywork.py --model "<32B id>" --selftest

    python experiments/measure_primerl_skywork.py \
        --model "<exact R1-Distill-Qwen-32B id on Nscale>" \
        --prompts 64 --group 8 --temperature 0.7 --max-tokens 16384 \
        --concurrency 8 --out primerl_skywork_lengths.json --resume

NOTES
-----
  * --max-tokens 16384 matches the run's published max context. Unlike the AReaL
    runs, DO NOT raise it: the whole point is to measure the distribution under the
    same cap the real run imposed, since truncation is part of that run's E[R].
    (Expect a non-trivial censored %, and read the reported bracket, not the mean.)
  * temperature: prime-rl's sampling temperature is not stated in the paper as far
    as I could confirm -- 1.0 is the default here for consistency with AReaL. If you
    find the real value, re-run; T is the single biggest lever on E[R] after the model.
  * G=16 in the real run (128 prompts x 16). --group 8 halves cost and still gives a
    tight mean; use --group 16 if you also want the true within-group spread.
"""

import argparse, json, urllib.parse, urllib.request, time, random, sys
from rl_length_probe import normalize_base, resolve_key, run, add_common_args, DS_ROWS

DATASET = "PrimeIntellect/Skywork-OR1-RL-Data-v1-math-prime-rl-format"
ROWS_TOTAL = 105055
SOLVE_COL = "solve_rate_qwen_r1_distill_32b"      # the exact policy prime-rl trained
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def fetch_filtered_prompts(n, seed=0, lo=0.001, hi=0.999, apply_filter=True, page=100):
    """Sample prompts, replicating prime-rl's solve-rate filter [0.001, 0.999].

    Returns (prompts, n_seen, n_kept) so the script can report how much the filter
    actually removed -- that fraction is itself a useful number (it is the mass that
    drives the oversampling ratio in the runtime model)."""
    random.seed(seed)
    hi_off = max(1, ROWS_TOTAL - page)
    offs = sorted(random.sample(range(0, hi_off), max(1, (n * 3) // page + 2)))
    prompts, seen, kept = [], 0, 0
    for o in offs:
        url = (f"{DS_ROWS}?dataset={urllib.parse.quote(DATASET)}"
               f"&config=default&split=train&offset={o}&length={page}")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    rows = json.load(r)["rows"]
                break
            except Exception:
                time.sleep(4 * (attempt + 1))
        else:
            continue
        for it in rows:
            row = it["row"]
            p = row.get("prompt")
            if not isinstance(p, str) or not p:
                continue
            seen += 1
            if apply_filter:
                try:
                    sr = float(row.get(SOLVE_COL))
                except (TypeError, ValueError):
                    continue                      # no solve rate -> can't confirm it passed
                if not (lo <= sr <= hi):
                    continue
            kept += 1
            prompts.append(p)
        time.sleep(0.5)
        if len(prompts) >= n:
            break
    return prompts[:n], seen, kept


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "NSCALE_BASE_URL", default_prompts=64, default_group=8)
    ap.set_defaults(max_tokens=16384)             # the run's real cap -- do not raise
    ap.add_argument("--no-filter", action="store_true",
                    help="skip prime-rl's [0.001,0.999] solve-rate filter")
    ap.add_argument("--out", default="primerl_skywork_lengths.json")
    cfg = ap.parse_args()
    cfg.key = resolve_key("NSCALE_API_KEY", "NSCALE_SERVICE_TOKEN", "OPENAI_API_KEY")
    cfg.base_url = normalize_base(cfg.base_url)
    cfg.suffix = MATH_SUFFIX

    if cfg.selftest:
        run(cfg, [], "selftest", cfg.out); return

    print(f"fetching {cfg.prompts} Skywork-OR1 math prompts"
          f"{'' if cfg.no_filter else ' with solve-rate filter [0.001, 0.999]'} ...")
    prompts, seen, kept = fetch_filtered_prompts(cfg.prompts, cfg.seed,
                                                 apply_filter=not cfg.no_filter)
    if not prompts:
        sys.exit("no prompts fetched -- check the dataset is reachable")
    if not cfg.no_filter and seen:
        print(f"  filter kept {kept}/{seen} rows ({100*kept/seen:.0f}%) -- the dropped "
              f"{100*(1-kept/seen):.0f}% are always-solved/always-failed")
    label = (f"{cfg.model} on Skywork-OR1 math (prime-rl data"
             f"{'' if cfg.no_filter else ', filtered'})")
    run(cfg, prompts, label, cfg.out)


if __name__ == "__main__":
    main()
