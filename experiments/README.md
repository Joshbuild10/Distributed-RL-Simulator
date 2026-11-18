# Measurement experiments

Pin the runtime model's free parameters (E[R], cv, omega) against real data instead of guesses.
All scripts are stdlib-only. Prompts come from the HF datasets-server (keyless);
generation needs an OpenAI-compatible endpoint.

```bash
export NSCALE_API_KEY=...            # or NSCALE_SERVICE_TOKEN
export NSCALE_BASE_URL=https://inference.api.nscale.com/v1
```

PowerShell:

```powershell
$env:NSCALE_API_KEY="..."
$env:NSCALE_BASE_URL="https://inference.api.nscale.com/v1"
```

Always `--selftest` first (one tiny request; validates model id + URL + key before you spend money).

## 1. AReaL 14B / 32B — code, on DeepCoder (AReaL's real code data)

Highest value: 14B/32B currently under-predict step time by 31%/53% using a
*proxy* E[R]. Settles whether that gap is E[R] or genuine efficiency error.
Dataset: `agentica-org/DeepCoder-Preview-Dataset`, pooled train splits
taco (7,436) + primeintellect (16,252) + lcbv5 (599), weighted by size.

```bash
python measure_areal_code_nscale.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --selftest

python measure_areal_code_nscale.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --prompts 64 --group 8 --temperature 1.0 --max-tokens 32768 \
  --concurrency 8 --out areal14b_code_lengths.json --resume

python measure_areal_code_nscale.py --model "<32B id>" \
  --prompts 64 --group 8 --temperature 1.0 --max-tokens 32768 \
  --concurrency 8 --out areal32b_code_lengths.json --resume
```

`--config taco|primeintellect|lcbv5` isolates one subset if you want the spread.

## 2. GLM-4.5-Air — INTELLECT-3's real RL mix

INTELLECT-3's preset E[R]=24,000 is a pure guess and it carries the model's
largest gap. Dataset: `PrimeIntellect/INTELLECT-3-RL` (math 21,161 / code 8,579 /
logic 11,647 / science 29,307; column `question`). Run splits **separately** —
they differ a lot and averaging hides it.

```bash
python measure_glm45air_intellect3.py --model "<GLM-4.5-Air id>" --selftest

python measure_glm45air_intellect3.py --model "<id>" --split math \
  --prompts 48 --group 8 --temperature 1.0 --max-tokens 49152 --resume
python measure_glm45air_intellect3.py --model "<id>" --split code \
  --prompts 48 --group 8 --temperature 1.0 --max-tokens 49152 --resume
```

Keep `--max-tokens` high (INTELLECT-3 runs at 65,536 context). A low cap
re-creates the truncation bias this measurement exists to remove.

## 3. prime-rl main run — R1-Distill-Qwen-32B on Skywork-OR1

The model's only clean anchor (+1% step time) — but its E[R]=7,470 was *triangulated
from the paper's own published throughput*, never measured, which makes that match
partly circular. This is the independent check.

Dataset: `PrimeIntellect/Skywork-OR1-RL-Data-v1-math-prime-rl-format` (105,055 rows,
plain-string `prompt`). Replicates prime-rl's solve-rate filter [0.001, 0.999] using
`solve_rate_qwen_r1_distill_32b` — the exact policy the run used. The filter keeps
only ~33% of rows, and the dropped always-solved/always-failed problems have very
different lengths, so filtering materially changes the measured distribution.

```bash
python measure_primerl_skywork.py --model "<32B id>" --selftest

python measure_primerl_skywork.py --model "<32B id>" \
  --prompts 64 --group 8 --temperature 1.0 --max-tokens 16384 \
  --concurrency 8 --out primerl_skywork_lengths.json --resume
```

Unlike the AReaL runs, **do not raise `--max-tokens`**: 16,384 is the run's real cap,
and truncation at that cap is part of the E[R] the real run experienced.
`--no-filter` measures the unfiltered pool for contrast.

## 4. Oversampling omega — free, no API

```bash
python measure_oversampling.py --group 16 --n 1000
```

Uses per-prompt solve rates published in the datasets. Computes
`omega = 1/E_p[f(p)]` — the model's `success_rate` path uses `1/f(mean p)`,
which is wrong under bimodal difficulty (measured: real omega 1.5–4.7x vs naive ~1.0).

## 5. Baselines from existing traces (no API)

```bash
python measure_response_lengths.py dataset --hf open-r1/OpenR1-Math-220k --col generations --n 500
python plot_distributions.py lengths.json --out distributions.html
```

## Resuming an interrupted run

These runs are slow and paid, so every generation script can pick up where it stopped.

**How it works.** The script writes the `--out` JSON every 20 completions (not just at
the end), storing `lengths`, `truncated`, `by_prompt` and the config. With `--resume`
it loads that file, counts how many samples are already there, and **skips that many
jobs** from the front of the queue, running only the remainder.

**To resume:** re-run the *exact same command* with `--resume` added, keeping
`--prompts`, `--group`, `--seed` and `--out` identical:

```bash
# original (died partway)
python measure_areal_code_nscale.py --model "<14B id>" \
  --prompts 64 --group 8 --temperature 1.0 --out areal14b_code_lengths.json

# continue it
python measure_areal_code_nscale.py --model "<14B id>" \
  --prompts 64 --group 8 --temperature 1.0 --out areal14b_code_lengths.json --resume
```

It prints `resuming: N samples already in <file>` and then `(N done, M to go)`.
When everything is finished it prints `nothing left to do` and just re-summarises.

**Rules:**
* `--seed` must match, or you resume against a *different* prompt set and silently
  pool two samples.
* `--prompts`/`--group` must match; they define the job queue length.
* Changing `--temperature`, `--model` or `--max-tokens` and resuming would mix
  incompatible samples — start a new `--out` file instead.
* To *extend* an existing measurement (same config, more data), don't use `--resume`:
  run with a different `--seed` into a new `--out`, then merge the `lengths` arrays.
* Safe to run `--resume` on a completed file: it no-ops and re-prints the summary.

Caveat: resume counts *successful* samples, so jobs that failed outright are retried
as part of the remainder — the prompt-to-sample mapping stays approximate for the
resumed portion (fine for the aggregate distribution, slightly noisy for ICC).

## Troubleshooting failing requests

The generation scripts (via `rl_length_probe.py`) **stream by default** and print a
categorised failure tally, e.g. `fails: timeout:12 http_429:3`. Diagnose by the top
category:

* **`timeout` / `http_504` / `urlerror`** — long generations being dropped. This is
  the common one for big models at high `--max-tokens`. Streaming (default) fixes most
  of it; if it persists, lower `--concurrency` (try 4) — a serverless endpoint has
  finite capacity and 8 simultaneous multi-minute generations can overwhelm it.
* **`http_429`** — rate/concurrency limited. Lower `--concurrency`, raise `--backoff`.
* **`http_4xx`** (400/404/422) — usually a bad `--model` id or `--max-tokens` above the
  model's limit. The script prints the first error body per category; read it.

Why streaming matters: a non-streamed request returns the whole completion at once, so
a 10k–32k-token reasoning generation can exceed the gateway's fixed response timeout
and get cut — and because it's the *long* ones that get cut, the surviving mean is
biased low. Streaming keeps the connection alive chunk-by-chunk, so `--timeout` becomes
the max gap *between* chunks, not the total generation time. Use `--no-stream` only to
test whether streaming is the fix.

(The standalone `measure_areal7b_nscale.py` predates the shared core and does not
stream — the 7B was fast enough that it didn't need to. Prefer the `rl_length_probe`
based scripts for the larger models.)

## Reading the output

* **95% CI on the mean** — two nominally identical 7B runs gave 8,714 and 10,613.
  If the CI is wider than ±15%, you cannot distinguish a real 20–50% model gap
  from noise: raise `--prompts`.
* **censored %** — samples that hit the cap are recorded *at* the cap, so the mean
  is a **lower bound**. Above ~2% the script prints a bracket; raise `--max-tokens`.
* **ICC** — high (>0.5) means prompt difficulty dominates, so buy more *prompts*;
  low means sampling noise dominates, so buy more *samples per prompt*.
* Temperature: use **1.0** — AReaL's actual sampling temperature.

Note: this directory is in `.gitignore`, so these scripts are untracked. Un-ignore
the `.py`/`.md` files (and keep ignoring `*.json` / `.qwen_tokenizer.json`) if you
want them in the repo.
