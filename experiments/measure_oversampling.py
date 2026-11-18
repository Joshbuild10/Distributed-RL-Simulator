#!/usr/bin/env python3
"""
measure_oversampling.py -- Measure the oversampling ratio omega from PUBLISHED
per-prompt solve rates. Keyless, free, seconds to run: no generation needed.

Why: RLSpec.oversample_override / success_rate is a free parameter in the runtime
model, currently guessed. omega inflates rollout+verify but not the update, so it
moves the inference:training ratio -- the number the governance argument turns on.

The model derives omega from zero-advantage filtering: a group of G binary-reward
samples is discarded iff all G agree, so
    accepted fraction f(p) = 1 - [p^G + (1-p)^G],   omega = 1/f
Evaluating that at a SINGLE average p is wrong -- f is strongly convex in p, and
real datasets are bimodal (many trivial prompts at p~1, many impossible at p~0),
both of which are discarded. The right quantity averages over the prompt
distribution:  omega = 1 / E_p[f(p)]  -- which is what this computes, using
datasets that publish a per-prompt pass rate.

Two omega are reported:
  * OMEGA      -- 1/E[f(p)] over the FULL prompt pool (no filtering).
  * omega_filt -- 1/E[f(p)] over only the LIVE prompts (0<p<1), i.e. after online
    difficulty filtering strips the always-solved/always-failed mass. This is what a
    run that filters (INTELLECT-3; prime-rl's [0.001,0.999] band) ACTUALLY pays, and
    it should land near prime-rl's throughput-triangulated ~1.35 on the 32B column.

Sources used (all have real measured per-prompt solve rates):
  * PrimeIntellect/INTELLECT-3-RL -- avg@8/avg@16 columns per split (math, code,
    logic, science). INTELLECT-3's ACTUAL RL data.
  * PrimeIntellect/INTELLECT-2-RL-Dataset -- solve_rate_qwen_r1_distill_7b.
  * PrimeIntellect/Skywork-OR1-RL-Data-v1-math-prime-rl-format -- prime-rl's ACTUAL
    math data, with solve_rate for three policy strengths (1B/7B/32B distill). The
    32B column is the policy the prime-rl reference run used; the three together show
    omega falling as the policy strengthens (higher p -> fewer discarded groups).

NOT usable (checked 2026-08: publish problem/answer only, no per-prompt solve rate,
so omega needs generation, not free): agentica-org/DeepScaleR-Preview-Dataset,
agentica-org/DeepCoder-Preview-Dataset, BytedTsinghua-SIA/DAPO-Math-17k,
Skywork/Skywork-OR1-RL-Data (the ORIGINAL; only the prime-rl-format fork adds rates).

RUN:
    python experiments/measure_oversampling.py                 # all sources
    python experiments/measure_oversampling.py --group 16      # set G
"""

import argparse, json, urllib.parse, urllib.request, time, random

DS_ROWS = "https://datasets-server.huggingface.co/rows"

SOURCES = [
    ("INTELLECT-3-RL:math",    "PrimeIntellect/INTELLECT-3-RL", "math",    "avg@8_qwen3_4b_thinking_2507", 21161),
    ("INTELLECT-3-RL:code",    "PrimeIntellect/INTELLECT-3-RL", "code",    "avg@8_qwen3_4b_instruct_2507", 8579),
    ("INTELLECT-3-RL:logic",   "PrimeIntellect/INTELLECT-3-RL", "logic",   "avg@16_qwen3_4b_instruct_2507", 11647),
    ("INTELLECT-3-RL:science", "PrimeIntellect/INTELLECT-3-RL", "science", "avg@16_qwen3_4b_instruct_2507", 29307),
    ("INTELLECT-2-RL",         "PrimeIntellect/INTELLECT-2-RL-Dataset", "default",
     "solve_rate_qwen_r1_distill_7b", 284741),
    ("Skywork-primerl:pol=1B",  "PrimeIntellect/Skywork-OR1-RL-Data-v1-math-prime-rl-format",
     "default", "solve_rate_qwen_r1_distill_1b", 105055),
    ("Skywork-primerl:pol=7B",  "PrimeIntellect/Skywork-OR1-RL-Data-v1-math-prime-rl-format",
     "default", "solve_rate_qwen_r1_distill_7b", 105055),
    ("Skywork-primerl:pol=32B", "PrimeIntellect/Skywork-OR1-RL-Data-v1-math-prime-rl-format",
     "default", "solve_rate_qwen_r1_distill_32b", 105055),
]


def fetch_rates(dataset, config, col, pool, n, seed=0, page=100):
    random.seed(seed)
    offs = sorted(random.sample(range(0, max(1, pool - page)), max(1, n // page)))
    out = []
    for o in offs:
        url = (f"{DS_ROWS}?dataset={urllib.parse.quote(dataset)}"
               f"&config={config}&split=train&offset={o}&length={page}")
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
            v = it["row"].get(col)
            try:
                p = float(v)
            except (TypeError, ValueError):
                continue
            if 0.0 <= p <= 1.0:
                out.append(p)
        time.sleep(0.6)
        if len(out) >= n:
            break
    return out[:n]


def omega_stats(ps, G):
    """Correct omega = 1/E[f(p)] (full pool), the post-difficulty-filter omega over
    live prompts (0<p<1), and the naive 1/f(E[p]) the single-p model would use."""
    f = lambda p: 1.0 - (p ** G + (1.0 - p) ** G)
    ef = sum(f(p) for p in ps) / len(ps)
    live = [p for p in ps if 0.0 < p < 1.0]
    ef_live = (sum(f(p) for p in live) / len(live)) if live else float("nan")
    pbar = sum(ps) / len(ps)
    dead = 1.0 - len(live) / len(ps)
    return dict(n=len(ps), pbar=pbar, dead=dead, ef=ef,
                omega=1.0 / max(ef, 1e-6),
                omega_filt=(1.0 / max(ef_live, 1e-6)) if live else float("nan"),
                omega_naive=1.0 / max(f(pbar), 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=16, help="G, responses per prompt")
    ap.add_argument("--n", type=int, default=1000, help="prompts sampled per source")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"Oversampling from zero-advantage filtering at G={a.group}")
    print(f"  omega = 1/E_p[f(p)],  f(p) = 1 - [p^G + (1-p)^G]\n")
    print(f"  {'source':<26}{'n':>6}{'mean p':>8}{'always0/1':>11}{'E[f]':>8}"
          f"{'OMEGA':>8}{'w_filt':>8}{'naive':>8}")
    results = {}
    for label, ds, cfg, col, pool in SOURCES:
        ps = fetch_rates(ds, cfg, col, pool, a.n, a.seed)
        if not ps:
            print(f"  {label:<26}  (no data / column missing)"); continue
        s = omega_stats(ps, a.group)
        results[label] = s
        print(f"  {label:<26}{s['n']:>6}{s['pbar']:>8.2f}{100*s['dead']:>10.0f}%"
              f"{s['ef']:>8.2f}{s['omega']:>8.2f}{s['omega_filt']:>8.2f}{s['omega_naive']:>8.2f}")

    print("\nReading:")
    print("  OMEGA  = correct value over the FULL pool, averaged over the real difficulty dist.")
    print("  w_filt = OMEGA after online difficulty filtering removes always-0/1 prompts (0<p<1).")
    print("           This is what a filtering run (INTELLECT-3, prime-rl [0.001,0.999]) pays;")
    print("           the prime-rl 32B row should sit near its throughput-triangulated ~1.35.")
    print("  naive  = 1/f(mean p), what you get plugging one average success_rate into")
    print("           RLSpec.success_rate. Where OMEGA and naive differ, the naive form is WRONG.")
    print("  always0/1 = share of prompts every sample gets right or wrong -> pure waste,")
    print("           and exactly what online difficulty filtering is designed to remove.")
    json.dump(results, open("oversampling.json", "w"), indent=1)
    print("\nsaved -> oversampling.json")


if __name__ == "__main__":
    main()
