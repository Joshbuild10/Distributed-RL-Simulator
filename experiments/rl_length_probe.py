#!/usr/bin/env python3
"""
rl_length_probe.py -- Shared core for measuring RL rollout response-length
distributions against a live OpenAI-compatible endpoint. Used by
measure_areal7b_nscale.py, measure_areal_code_nscale.py and
measure_glm45air_intellect3.py; not meant to be run directly.

Design: stdlib only. Prompts pulled from HF datasets-server (keyless). Response
length read from the endpoint's own usage.completion_tokens (exact -- no local
tokenizer needed). Concurrent requests with retry/backoff on 429/5xx.

Reporting choices that matter for this project:
  * BOOTSTRAP CI on the mean. Two nominally identical T=0.6 runs of the 7B gave
    E[R] 8,714 (n=512) and 10,613 (n=256) -- a 22% swing that is pure sampling
    noise. A point estimate without a CI is not decision-grade here, because the
    whole question is whether a model-vs-published gap of 15-50% is real.
  * WITHIN- vs BETWEEN-PROMPT variance. Response length is strongly
    prompt-dependent (hard problem -> all G samples long). That decomposition
    tells you whether to buy more prompts or more samples/prompt, and it is the
    correlation the static-batching tail penalty in the runtime model needs.
  * CENSORING. Samples that hit max_tokens are recorded AT the cap, so the mean
    is a LOWER BOUND when truncation is non-trivial. We report the censored
    fraction and a Kaplan-Meier-style lower/upper bracket rather than one number.
  * RESUME. These runs are slow and paid; --resume continues an interrupted file
    instead of paying twice.
"""

import json, os, sys, time, socket, collections, urllib.request, urllib.parse, urllib.error
import statistics as st
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_TRACE_LOCK = threading.Lock()


def _append_trace(path, obj):
    """Append one JSON line under a lock -- concurrent completions from the
    ThreadPoolExecutor would otherwise interleave writes to the same file."""
    line = json.dumps(obj, ensure_ascii=False)
    with _TRACE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

DS_ROWS = "https://datasets-server.huggingface.co/rows"


# ---------------------------------------------------------------------------
# Prompt sourcing
# ---------------------------------------------------------------------------

def fetch_prompts(dataset, config, split, column, n, seed=0, rows_pool=40000,
                  starter_code_col=None, extra_cols=None):
    """Pull n prompts from an HF dataset via the keyless datasets-server.

    starter_code_col: appended when non-empty (LiveCodeBench-style rows separate the
    problem statement from the function signature).
    extra_cols: also return these columns (e.g. published per-prompt solve rates),
    in which case the return is a list of (prompt, {col: val}) instead of a list of str.
    """
    random.seed(seed)
    page = 15
    # never request an offset past the end of the split
    hi = max(1, rows_pool - page)
    k = max(1, n // page + 1)
    offs = sorted(random.sample(range(0, hi), min(k, hi)))
    out = []
    for o in offs:
        url = (f"{DS_ROWS}?dataset={urllib.parse.quote(dataset)}"
               f"&config={config}&split={split}&offset={o}&length={page}")
        for attempt in range(5):
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
            p = row.get(column)
            if not isinstance(p, str) or not p:
                continue
            if starter_code_col and row.get(starter_code_col):
                p = p + "\n\n" + row[starter_code_col]
            out.append((p, {c: row.get(c) for c in extra_cols}) if extra_cols else p)
        if len(out) >= n:
            break
    return out[:n]


# ---------------------------------------------------------------------------
# Endpoint plumbing
# ---------------------------------------------------------------------------

def normalize_base(u):
    """Tolerate the common base-URL mistakes: trailing slash, an appended
    /chat/completions, or a missing /v1."""
    u = u.strip().rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")].rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def resolve_key(*env_names):
    for name in env_names:
        v = os.environ.get(name)
        if v:
            return v
    sys.exit(f"Set one of {env_names} as an env var; do not hardcode the key.")


def _tokens_from_usage(usage):
    """completion_tokens, taking max with reasoning_tokens (some providers split them)."""
    ct = usage.get("completion_tokens")
    if ct is None:
        return None
    det = usage.get("completion_tokens_details") or {}
    return max(int(ct), int(det.get("reasoning_tokens") or 0))


def _reasoning_field(d):
    """Reasoning text lives under different keys depending on provider/route:
    'reasoning_content' (DeepSeek-direct, vLLM OpenAI-compat) or 'reasoning'
    (OpenRouter's current unified field -- they renamed FROM reasoning_content
    TO reasoning; the old name is legacy). Silently checking only one drops the
    text entirely on whichever route uses the other name, while usage-based
    token counts stay correct (they come from the server's own accounting, not
    from re-summing this text) -- which is exactly what happened here: content
    captured fine, reasoning always empty, on OpenRouter's newer field name."""
    return d.get("reasoning_content") or d.get("reasoning") or ""


def _parse_json(d, want_text=False):
    ch = d["choices"][0]
    ntok = _tokens_from_usage(d.get("usage") or {})
    m = ch.get("message", {})
    content = m.get("content") or ""
    reasoning = _reasoning_field(m)
    chars = len(content) + len(reasoning)
    if want_text:
        return ntok, ch.get("finish_reason"), chars, reasoning, content
    return ntok, ch.get("finish_reason"), chars, None, None


def _read_stream(r, want_text=False):
    """Parse an SSE stream. Streaming keeps the connection alive with incremental
    chunks, so a long generation never trips the gateway's single-response timeout --
    the fix for the 'long completions fail, short ones succeed' bias.

    want_text=False (default) only counts chars, so memory stays O(1) per request
    regardless of run size -- the common case, unmodified from before this option existed."""
    ntok, fin, chars = None, None, 0
    reasoning_parts, content_parts = ([], []) if want_text else (None, None)
    for raw in r:                       # urllib response is line-iterable
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if obj.get("usage"):
            t = _tokens_from_usage(obj["usage"])
            if t is not None:
                ntok = t
        for c0 in (obj.get("choices") or []):
            delta = c0.get("delta") or {}
            dc, dr = delta.get("content") or "", _reasoning_field(delta)
            chars += len(dc) + len(dr)
            if want_text:
                if dc: content_parts.append(dc)
                if dr: reasoning_parts.append(dr)
            if c0.get("finish_reason"):
                fin = c0["finish_reason"]
    if want_text:
        return ntok, fin, chars, "".join(reasoning_parts), "".join(content_parts)
    return ntok, fin, chars, None, None


def one_completion(cfg, prompt, want_text=False):
    """One chat completion. Returns a dict:
        {"ok": True,  "tok": int, "finish": str, "reasoning": str?, "content": str?}
        {"ok": False, "err": str, "detail": str?}   # categorised so failures are visible

    Streams by default (cfg.stream) so long reasoning generations don't hit the
    gateway's non-streaming response timeout. Token count comes from the server's
    usage (exact); chars//4 only as a last-resort fallback. want_text=True additionally
    returns the raw reasoning/content text (see --save-traces in add_common_args).
    """
    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt + cfg.suffix}],
        "temperature": cfg.temperature, "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
    }
    # Closed models hide the CoT text but return its token COUNT; OpenRouter/OpenAI
    # gate that behind a reasoning field. Additive: absent unless the caller sets it,
    # so DeepSeek-R1 (returns reasoning_content unprompted) and the existing scripts
    # are unaffected.
    eff = getattr(cfg, "reasoning_effort", None)
    if eff:
        payload["reasoning"] = {"effort": eff}
    extra = getattr(cfg, "extra_body", None)
    if extra:
        payload.update(json.loads(extra) if isinstance(extra, str) else extra)
    if getattr(cfg, "enable_thinking", False):
        # setdefault so an explicit --extra-body chat_template_kwargs isn't clobbered
        payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
    if getattr(cfg, "stream", True):
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    body = json.dumps(payload).encode()
    last = "unknown"
    for attempt in range(cfg.retries):
        try:
            req = urllib.request.Request(
                cfg.base_url + "/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.key}"})
            with urllib.request.urlopen(req, timeout=cfg.timeout) as r:
                if payload.get("stream"):
                    ntok, fin, chars, reasoning, content = _read_stream(r, want_text)
                else:
                    ntok, fin, chars, reasoning, content = _parse_json(json.load(r), want_text)
            if ntok is None:
                ntok = max(1, chars // 4)
            out = {"ok": True, "tok": int(ntok), "finish": fin}
            if want_text:
                out["reasoning"], out["content"] = reasoning, content
            return out
        except urllib.error.HTTPError as e:
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529):
                last = f"http_{e.code}"
                time.sleep(cfg.backoff * (attempt + 1)); continue
            try:
                detail = e.read()[:300].decode("utf-8", "replace")
            except Exception:
                detail = ""
            return {"ok": False, "err": f"http_{e.code}", "detail": detail}
        except (socket.timeout, TimeoutError):
            last = "timeout"
            time.sleep(cfg.backoff); continue
        except urllib.error.URLError as e:
            last = f"urlerror:{getattr(e, 'reason', e)}"[:50]
            time.sleep(cfg.backoff * (attempt + 1)); continue
        except Exception as e:
            last = f"{type(e).__name__}:{e}"[:50]
            time.sleep(cfg.backoff * (attempt + 1)); continue
    return {"ok": False, "err": f"{last}_exhausted"}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(values, iters=2000, alpha=0.05, seed=0):
    """i.i.d. percentile bootstrap CI for the mean. Valid only if samples are
    independent -- NOT true when several samples share a prompt (see the cluster
    version below), so this is used only when by_prompt is unavailable."""
    if len(values) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters)])


def bootstrap_ci_clustered(groups, iters=2000, alpha=0.05, seed=0):
    """Cluster (block) bootstrap: resample PROMPTS with replacement, not individual
    samples. With high ICC the G samples of one prompt are near-duplicates, so the
    i.i.d. bootstrap treats ~n independent draws when there are really ~n_prompts,
    understating the CI by ~sqrt(1+(G-1)*ICC). This is the honest interval."""
    k = len(groups)
    if k < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        pool_sum = pool_n = 0
        for _ in range(k):
            g = groups[rng.randrange(k)]
            pool_sum += sum(g); pool_n += len(g)
        means.append(pool_sum / pool_n)
    means.sort()
    return (means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters)])


def variance_decomposition(by_prompt):
    """Random-effects ANOVA split of total variance into within- and between-prompt
    variance COMPONENTS. by_prompt: {prompt_idx: [lengths]}. Returns
    (within_sd, between_sd, icc). Uses the unbiased group-size factor m0 for
    unbalanced groups; between-var = (MS_between - MS_within)/m0 (clamped >=0)."""
    groups = [v for v in by_prompt.values() if len(v) >= 2]
    k = len(groups)
    if k < 2:
        return (float("nan"),) * 3
    ns = [len(g) for g in groups]
    N = sum(ns)
    gmeans = [sum(g) / n for g, n in zip(groups, ns)]
    grand = sum(sum(g) for g in groups) / N
    ss_within = sum(sum((x - m) ** 2 for x in g) for g, m in zip(groups, gmeans))
    ss_between = sum(n * (m - grand) ** 2 for n, m in zip(ns, gmeans))
    ms_within = ss_within / max(1, N - k)
    ms_between = ss_between / max(1, k - 1)
    m0 = (N - sum(n * n for n in ns) / N) / max(1, k - 1)     # ~mean group size
    var_within = ms_within
    var_between = max(0.0, (ms_between - ms_within) / max(m0, 1e-9))
    denom = var_between + var_within
    icc = var_between / denom if denom > 0 else float("nan")
    return var_within ** 0.5, var_between ** 0.5, icc


def summarize(lengths, truncated, cfg, label, by_prompt=None):
    L = sorted(lengths); n = len(L)
    if not n:
        print("no samples collected"); return
    mean = sum(L) / n; sd = st.pstdev(L); q = lambda p: L[min(n - 1, int(p * n))]
    frac_trunc = truncated / n

    # Use the cluster-robust CI when we know the prompt grouping (samples of one
    # prompt are correlated); fall back to i.i.d. only when we don't.
    groups = [v for v in by_prompt.values() if v] if by_prompt else []
    clustered = len(groups) >= 3
    lo, hi = (bootstrap_ci_clustered(groups, seed=cfg.seed) if clustered
              else bootstrap_ci(L, seed=cfg.seed))

    print(f"\n=== {label} @ T={cfg.temperature}, cap={cfg.max_tokens:,} ===")
    print(f"  samples {n} | censored at cap: {truncated} ({100*frac_trunc:.1f}%)")
    print(f"  mean {mean:,.0f}  [95% CI {lo:,.0f} - {hi:,.0f}]  (+/-{100*(hi-lo)/2/mean:.0f}%"
          f"{', cluster-robust' if clustered else ''})")
    print(f"  median {st.median(L):,.0f} | std {sd:,.0f} | cv {sd/mean:.2f}")
    print(f"  p50 {q(.5):,.0f} | p90 {q(.9):,.0f} | p95 {q(.95):,.0f} | p99 {q(.99):,.0f} | max {L[-1]:,.0f}")

    if by_prompt:
        w, b, icc = variance_decomposition(by_prompt)
        if w == w:  # not nan
            print(f"  variance: within-prompt sd {w:,.0f} | between-prompt sd {b:,.0f} "
                  f"| ICC {icc:.2f}")
            n_prompts = len([v for v in by_prompt.values() if len(v) >= 2])
            print(f"    -> {'prompt choice' if icc > 0.5 else 'sampling noise'} dominates "
                  f"({n_prompts} prompts x ~{n//max(len(groups),1)}); "
                  f"buy more {'prompts' if icc > 0.5 else 'samples per prompt'}")

    print(f"\n  --> presets.py:  response_len_mean={mean:,.0f}, response_len_cv={sd/mean:.2f}, "
          f"max_response_len={L[-1]:,.0f}")
    if frac_trunc > 0.02:
        # censored samples sit AT the cap, so the true mean is higher.
        naive_upper = (sum(x for x in L if x < cfg.max_tokens) +
                       truncated * cfg.max_tokens * 1.5) / n
        print(f"  !! {100*frac_trunc:.1f}% censored at {cfg.max_tokens:,}: the mean above is a LOWER "
              f"BOUND.\n     True mean is between {mean:,.0f} and ~{naive_upper:,.0f} "
              f"(if censored runs average 1.5x the cap).\n     Raise --max-tokens to remove the bias.")
    if (hi - lo) / 2 / mean > 0.15:
        print(f"  !! CI is +/-{100*(hi-lo)/2/mean:.0f}% of the mean -- too wide to distinguish a "
              f"20-50% model gap.\n     Increase --prompts (and --group) before drawing conclusions.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def add_common_args(ap, default_base_url_env, default_prompts=48, default_group=8):
    ap.add_argument("--base-url", default=os.environ.get(default_base_url_env,
                    "https://inference.api.nscale.com/v1"))
    ap.add_argument("--model", required=True, help="exact model id from the provider's console")
    ap.add_argument("--prompts", type=int, default=default_prompts)
    ap.add_argument("--group", type=int, default=default_group,
                    help="completions per prompt (match the real run's G if known)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=900,
                    help="per-read socket timeout (s). With streaming this is the max GAP "
                         "between chunks, not the total gen time, so it can stay modest.")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=int, default=6)
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                    help="stream the response (default on; --no-stream to disable). "
                         "Streaming avoids the gateway dropping long non-streamed completions.")
    ap.add_argument("--reasoning-effort", default=None,
                    help="low/medium/high. Sends {'reasoning':{'effort':...}} (OpenRouter/OpenAI "
                         "convention) so closed models (GPT-5/o3/Claude/Gemini) return their "
                         "reasoning-token COUNT. Not needed for DeepSeek-R1 (returns reasoning_content).")
    ap.add_argument("--extra-body", default=None,
                    help="JSON merged verbatim into the request body for provider-specific fields, "
                         "e.g. '{\"provider\":{\"order\":[\"deepseek\"]}}' to pin an OpenRouter provider.")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="inject chat_template_kwargs={'enable_thinking':true} -- the lever for "
                         "HYBRID Qwen3 (the original series) served via vLLM/OpenRouter. NOT needed "
                         "for the dedicated *-Thinking-2507 variants (thinking always on), and "
                         "unsupported by *-Instruct-2507 (never thinks). Always confirm via the "
                         "selftest tok count that thinking actually engaged.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted --out file instead of restarting")
    ap.add_argument("--selftest", action="store_true",
                    help="send ONE tiny request, report the result, and exit")
    ap.add_argument("--save-traces", action="store_true",
                    help="append the raw reasoning/content text of every completion to "
                         "<out>.traces.jsonl (one JSON object per line: idx, tok, finish, "
                         "reasoning, content). Off by default -- only the token COUNT is kept "
                         "otherwise, which is all the length statistics need. Turn on to enable "
                         "later qualitative checks (repetition loops, bimodal length causes, "
                         "distribution-shape fitting) without re-paying for generation. Appends "
                         "as it goes (not held in memory), so --resume plays back the surviving "
                         "lines fine -- just don't remove the file between resumes.")
    return ap


def run(cfg, prompts, label, out_path):
    print(f"endpoint: {cfg.base_url}/chat/completions | model: {cfg.model} "
          f"| token: ****{cfg.key[-4:]}")
    if cfg.selftest:
        saved = cfg.max_tokens
        cfg.max_tokens = min(64, saved)
        res = one_completion(cfg, "2+2=? Answer with just the number.")
        cfg.max_tokens = saved
        if res.get("ok"):
            print(f"selftest OK: {res['tok']} completion tokens, finish_reason={res['finish']!r}"
                  f"  (stream={'on' if cfg.stream else 'off'})")
        else:
            print(f"selftest FAILED: {res.get('err')} {res.get('detail','')}\n"
                  f"  check --model id, base URL and key.", file=sys.stderr)
            sys.exit(1)
        return

    lengths, truncated, by_prompt = [], 0, {}
    if cfg.resume and os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            lengths = list(prev.get("lengths", []))
            truncated = int(prev.get("truncated", 0))
            by_prompt = {int(k): v for k, v in (prev.get("by_prompt") or {}).items()}
            print(f"resuming: {len(lengths)} samples already in {out_path}")
        except Exception as e:
            print(f"could not resume ({e}); starting fresh")

    already = len(lengths)
    jobs = [(i, p) for i, p in enumerate(prompts) for _ in range(cfg.group)][already:]
    if not jobs:
        print("nothing left to do"); summarize(lengths, truncated, cfg, label, by_prompt); return
    print(f"{len(prompts)} prompts x {cfg.group} = {len(prompts)*cfg.group} samples "
          f"({already} done, {len(jobs)} to go) at concurrency {cfg.concurrency}")

    save_traces = getattr(cfg, "save_traces", False)
    trace_path = out_path.rsplit(".json", 1)[0] + ".traces.jsonl" if save_traces else None
    if trace_path:
        print(f"saving raw traces -> {trace_path} (appended, not held in memory)")

    def save():
        json.dump({"lengths": lengths, "truncated": truncated,
                   "by_prompt": by_prompt,
                   "config": {k: getattr(cfg, k) for k in
                              ("model", "temperature", "top_p", "max_tokens", "group", "prompts")}},
                  open(out_path, "w"))

    done = 0; t0 = time.time(); fails = collections.Counter(); shown = set()
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
        futs = {ex.submit(one_completion, cfg, p, save_traces): idx for idx, p in jobs}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"ok": False, "err": f"future:{type(e).__name__}"}
            done += 1
            if res.get("ok"):
                lengths.append(res["tok"])
                by_prompt.setdefault(idx, []).append(res["tok"])
                if res.get("finish") == "length":
                    truncated += 1
                if trace_path:
                    _append_trace(trace_path, {"idx": idx, "tok": res["tok"],
                                               "finish": res.get("finish"),
                                               "reasoning": res.get("reasoning", ""),
                                               "content": res.get("content", "")})
            else:
                reason = res.get("err", "none")
                fails[reason] += 1
                if reason not in shown:            # surface the first body per error kind
                    shown.add(reason)
                    if res.get("detail"):
                        print(f"    [{reason}] {res['detail'][:200]}", file=sys.stderr)
            if done % 20 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9) / 60
                fail_str = " ".join(f"{k}:{v}" for k, v in fails.most_common(4))
                print(f"  {done}/{len(jobs)} ({len(lengths)} ok) {rate:.2f}/s ETA {eta:.0f} min"
                      + (f"  | fails: {fail_str}" if fails else ""))
                save()
    save()
    if fails:
        print(f"\n  FAILURES ({sum(fails.values())} total): "
              + ", ".join(f"{k}={v}" for k, v in fails.most_common()))
        top = fails.most_common(1)[0][0]
        if top.startswith(("timeout", "http_504", "urlerror")):
            print("  -> looks like long generations being dropped. Ensure --stream is on "
                  "(default), lower --concurrency, or the endpoint has a hard response cap.")
        elif top.startswith("http_429"):
            print("  -> rate/concurrency limited. Lower --concurrency and/or raise --backoff.")
        elif top.startswith("http_4"):
            print("  -> client error (bad model id? max_tokens above the model's limit?). "
                  "Check the [detail] line above.")
    summarize(lengths, truncated, cfg, label, by_prompt)
    print(f"\nraw saved to {out_path}   (plot: python plot_distributions.py <merged json>)")
