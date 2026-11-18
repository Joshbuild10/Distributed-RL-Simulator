#!/usr/bin/env python3
"""
measure_response_lengths.py -- Measure the E[R] (response-length) distribution that
feeds RLSpec.response_len_mean / response_len_cv, so those stop being guesses.

Two modes:

  DATASET  (keyless, runs anywhere): tokenize the generations already published in a
           HF reasoning dataset and report the length distribution. This is what was
           used to get the math ~6k / code ~12.7k anchors. It measures real
           reasoning-model traces on the real tasks, but they are SFT/distillation
           traces, not live RL rollouts (RL lengthens responses over training).

    python experiments/measure_response_lengths.py dataset \
        --hf open-r1/OpenR1-Math-220k --col generations --n 500

  GENERATE (needs an OpenAI-compatible endpoint: local vLLM, OpenRouter, Together,
           DeepInfra, ...): sample completions from the ACTUAL policy at RL sampling
           settings on the ACTUAL prompts and measure. This is the high-fidelity
           version -- point it at the real checkpoint to pin the RL-lengthened E[R].

    python experiments/measure_response_lengths.py generate \
        --base-url http://localhost:8000/v1 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --prompt-hf PrimeIntellect/INTELLECT-2-RL-Dataset --prompt-col prompt \
        --n 128 --group 4 --temperature 0.7 --max-tokens 16384
        # set OPENAI_API_KEY for hosted endpoints; local vLLM ignores it.

Both modes tokenize with the Qwen2.5/QwQ tokenizer (downloaded once to a cache) and
emit an RLSpec fragment you can paste into presets.py.
"""

import argparse, json, os, statistics as st, urllib.request, urllib.parse, random

TOKENIZER_URL = "https://huggingface.co/Qwen/QwQ-32B/resolve/main/tokenizer.json"
CACHE = os.path.join(os.path.dirname(__file__), ".qwen_tokenizer.json")
ROWS = "https://datasets-server.huggingface.co/rows"


def get_tokenizer(path=None):
    from tokenizers import Tokenizer
    p = path or CACHE
    if not os.path.exists(p):
        print(f"downloading tokenizer -> {p}")
        urllib.request.urlretrieve(TOKENIZER_URL, p)
    return Tokenizer.from_file(p)


def hf_rows(dataset, config, split, offset, length):
    url = (f"{ROWS}?dataset={urllib.parse.quote(dataset)}"
           f"&config={config}&split={split}&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.load(r)["rows"]


def summarize(lengths, label):
    L = sorted(lengths); n = len(L)
    if n == 0:
        print(f"  {label}: no samples"); return None
    mean = sum(L) / n; sd = st.pstdev(L); q = lambda p: L[min(n - 1, int(p * n))]
    print(f"\n== {label}  (n={n}) ==")
    print(f"  mean {mean:,.0f} | median {st.median(L):,.0f} | std {sd:,.0f} | cv {sd/mean:.2f}")
    print(f"  p50 {q(.5):,.0f} | p90 {q(.9):,.0f} | p95 {q(.95):,.0f} | p99 {q(.99):,.0f} | max {L[-1]:,.0f}")
    print(f"  -> RLSpec(response_len_mean={mean:,.0f}, response_len_cv={sd/mean:.2f}, "
          f"max_response_len={L[-1]:,.0f})")
    return dict(mean=mean, cv=sd / mean, p95=q(.95), n=n)


def run_dataset(a):
    tok = get_tokenizer(a.tokenizer)
    random.seed(a.seed)
    n_calls = max(1, a.n // a.batch)
    offs = sorted(random.sample(range(0, max(1, a.rows_total - a.batch)), min(n_calls, 64)))
    lengths = []
    for off in offs:
        try:
            rows = hf_rows(a.hf, a.config, a.split, off, a.batch)
        except Exception as e:
            print("  skip offset", off, e); continue
        for it in rows:
            val = it["row"].get(a.col)
            gens = val if isinstance(val, list) else [val]
            for g in gens:
                if isinstance(g, str) and g:
                    lengths.append(len(tok.encode(g).ids))
        if len(lengths) >= a.n:
            break
    summarize(lengths, f"{a.hf}:{a.col}")


def run_generate(a):
    tok = get_tokenizer(a.tokenizer)
    key = os.environ.get("OPENAI_API_KEY", "none")
    # gather prompts
    prompts = []
    if a.prompt_hf:
        random.seed(a.seed)
        offs = sorted(random.sample(range(0, 200000), max(1, a.n // 50)))
        for off in offs:
            for it in hf_rows(a.prompt_hf, a.config, a.split, off, 50):
                p = it["row"].get(a.prompt_col)
                if isinstance(p, str) and p:
                    prompts.append(p)
                if len(prompts) >= a.n:
                    break
            if len(prompts) >= a.n:
                break
    elif a.prompt_file:
        prompts = [l.strip() for l in open(a.prompt_file) if l.strip()][:a.n]
    else:
        raise SystemExit("generate mode needs --prompt-hf or --prompt-file")

    lengths = []
    for i, p in enumerate(prompts):
        for _ in range(a.group):                       # G samples per prompt, like RL rollouts
            body = json.dumps({"model": a.model,
                               "messages": [{"role": "user", "content": p}],
                               "temperature": a.temperature, "top_p": a.top_p,
                               "max_tokens": a.max_tokens}).encode()
            req = urllib.request.Request(a.base_url.rstrip("/") + "/chat/completions",
                                         data=body, headers={"Content-Type": "application/json",
                                                             "Authorization": f"Bearer {key}"})
            try:
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    d = json.load(r)
                m = d["choices"][0]["message"]
                text = (m.get("content") or "") + (m.get("reasoning_content") or "")
                # prefer server token count if present, else tokenize locally
                usage = d.get("usage") or {}
                lengths.append(usage.get("completion_tokens") or len(tok.encode(text).ids))
            except Exception as e:
                print(f"  gen fail [{i}]:", e)
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(prompts)} prompts, {len(lengths)} samples")
    summarize(lengths, f"{a.model} @ T={a.temperature}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("dataset")
    d.add_argument("--hf", required=True); d.add_argument("--col", required=True)
    d.add_argument("--config", default="default"); d.add_argument("--split", default="train")
    d.add_argument("--n", type=int, default=500); d.add_argument("--batch", type=int, default=40)
    d.add_argument("--rows-total", type=int, default=90000)
    d.add_argument("--tokenizer", default=None); d.add_argument("--seed", type=int, default=7)

    g = sub.add_parser("generate")
    g.add_argument("--base-url", required=True); g.add_argument("--model", required=True)
    g.add_argument("--prompt-hf", default=None); g.add_argument("--prompt-col", default="prompt")
    g.add_argument("--prompt-file", default=None)
    g.add_argument("--config", default="default"); g.add_argument("--split", default="train")
    g.add_argument("--n", type=int, default=64); g.add_argument("--group", type=int, default=4)
    g.add_argument("--temperature", type=float, default=0.7); g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--max-tokens", type=int, default=32768); g.add_argument("--timeout", type=int, default=1200)
    g.add_argument("--tokenizer", default=None); g.add_argument("--seed", type=int, default=7)

    a = ap.parse_args()
    (run_dataset if a.mode == "dataset" else run_generate)(a)


if __name__ == "__main__":
    main()
