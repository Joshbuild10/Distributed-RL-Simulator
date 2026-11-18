#!/usr/bin/env python3
"""
measure_agentic_traces.py -- First-order response-length estimates for AGENTIC / multi-turn
RL, from RELEASED trajectory datasets. No API, no agent harness, no cost -- tokenizes
public SWE-agent trajectories via the HF datasets-server (keyless).

Why: the single-turn probes (measure_domain_sweep) cannot capture the regime that
dominates feasibility per-step compute -- multi-turn agents accumulate prior turns +
tool outputs into an effective context far above single-turn E[R]. This gives the
first-order magnitude of that (is it 30k / 100k / 300k?) so the runtime model's
quadratic attention/KV terms have a grounded agentic input.

Two metrics matter and they differ:
  * GENERATED tokens  = sum of assistant-authored tokens (content + tool-call args)
      -> drives DECODE compute (what the inference pool actually generates).
  * EFFECTIVE CONTEXT = total tokens across ALL messages (system+user+assistant+tool)
      -> the context the model carries by end-of-trajectory; drives the quadratic
         attention / KV terms and flips runs update-bound.
Also reports turns and splits by `resolved` (failed trajectories tend to run longer).

SOURCES (--source), one representative released trace set per environment type:
  nebius   SWE / code-repair    nebius/SWE-rebench-openhands-trajectories (OpenHands +
           Qwen3-Coder-480B; role-tagged messages -> full gen/tool/turn breakdown)
  swesmith SWE / code-repair    SWE-bench/SWE-smith-trajectories (SWE-agent + Claude 3.7;
           serialized string -> total-length only, cross-check)
  tau2     tool-use             Jarrodbarnes/tau2-sft-v4-dataset (tau2-bench retail/airline,
           Qwen3-235B-Thinking; `prompt` = role/content/name message list)
  ccbench  agentic-coding / CLI zai-org/CC-Bench-trajectories (Claude-Code style; ships its
           own token counts -> also yields the cumulative prefill:decode ratio)
Browser/web: no clean, ungated public trace dataset found (WebArena etc. are environments,
not released trajectory dumps) -- left as a known gap rather than a bad proxy.

RUN:
    python experiments/measure_agentic_traces.py --source nebius   --n 120
    python experiments/measure_agentic_traces.py --source tau2     --n 120
    python experiments/measure_agentic_traces.py --source ccbench  --n 100

CAVEATS (state these next to any number): released traces are (a) OFF-POLICY (Claude /
Qwen3-Coder, not the model you're modelling), (b) SFT-CURATED (often filtered to
resolved/clean runs), so the failed-loop long tail an RL run actually pays for is
under-represented -> treat as a lower-ish first-order magnitude, not a policy-exact E[R].
"""
import argparse, json, random, statistics as st, time, urllib.error
from measure_response_lengths import get_tokenizer, hf_rows


def _fetch_retry(hf, config, split, off, page, tries=6):
    """hf_rows with backoff on 429/5xx -- the big-row datasets (CC-Bench, 166k-char
    rows) trip the datasets-server rate limit, and the plain fetcher just drops those
    offsets, biasing the sample toward whatever got through."""
    for t in range(tries):
        try:
            return hf_rows(hf, config, split, off, page)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(3 * (t + 1)); continue
            raise
        except Exception:
            time.sleep(3 * (t + 1))
    return None

SOURCES = {
    # SWE / code-repair
    "nebius":   dict(hf="nebius/SWE-rebench-openhands-trajectories", config="default",
                     split="train", col="trajectory", fmt="messages", resolved="resolved",
                     env="SWE", n_rows=8000),
    "swesmith": dict(hf="SWE-bench/SWE-smith-trajectories", config="default",
                     split="tool", col="messages", fmt="string", resolved="resolved",
                     env="SWE", n_rows=5000),
    # tool-use / function-calling (customer-service domains: retail, airline)
    "tau2":     dict(hf="Jarrodbarnes/tau2-sft-v4-dataset", config="default",
                     split="train", col="prompt", fmt="messages", resolved=None,
                     env="tool-use", n_rows=219, page=6),
    # agentic coding / CLI (Claude-Code style; ships its own token counts)
    "ccbench":  dict(hf="zai-org/CC-Bench-trajectories", config="default",
                     split="train", col="trajectory", fmt="precomputed", resolved=None,
                     env="agentic-coding", n_rows=150, page=2, sleep=1.5),
}


def _toklen(tok, s):
    return len(tok.encode(s).ids) if s else 0


def _text_of(content):
    """content may be a str or a list of parts ({type,text} etc.)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""


def analyze_messages(tok, traj):
    gen = tool = other = turns = 0
    for m in traj:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        n = _toklen(tok, _text_of(m.get("content")))
        tc = m.get("tool_calls")
        n_tc = _toklen(tok, json.dumps(tc)) if tc else 0
        if role == "assistant":
            gen += n + n_tc; turns += 1
        elif role == "tool":
            tool += n
        else:
            other += n
    return dict(total=gen + tool + other, gen=gen, tool=tool, other=other, turns=turns)


def summarize(vals, label):
    v = sorted(x for x in vals if x is not None)
    n = len(v)
    if not n or sum(v) == 0:
        print(f"  {label:<16} (all zero / no data -- not present in this schema)"); return
    m = sum(v) / n; sd = st.pstdev(v); q = lambda p: v[min(n - 1, int(p * n))]
    cv = sd / m if m else 0.0
    print(f"  {label:<16} mean {m:>9,.0f} | median {st.median(v):>9,.0f} | cv {cv:>4.2f} "
          f"| p90 {q(.9):>9,.0f} | p95 {q(.95):>9,.0f} | max {v[-1]:>9,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="nebius")
    ap.add_argument("--n", type=int, default=120, help="number of trajectories to sample")
    ap.add_argument("--split", default=None, help="override the source's default split")
    ap.add_argument("--rows-total", type=int, default=None,
                    help="upper bound on random offset (default: per-source; keep <= dataset size)")
    ap.add_argument("--page", type=int, default=None, help="rows per datasets-server call (default: per-source)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=None,
                    help="seconds between datasets-server calls (default: per-source; raise if 429s)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default=None, help="optional JSON dump of the per-trajectory arrays")
    a = ap.parse_args()
    src = SOURCES[a.source]
    split = a.split or src["split"]
    rows_total = a.rows_total or src.get("n_rows", 8000)
    page = a.page or src.get("page", 4)
    tok = get_tokenizer(a.tokenizer)

    random.seed(a.seed)
    hi = max(1, rows_total - page)
    offs = sorted(random.sample(range(0, hi), min(a.n // page + 1, hi)))

    recs = []
    sleep = a.sleep if a.sleep is not None else src.get("sleep", 0.0)
    for off in offs:
        rows = _fetch_retry(src["hf"], src["config"], split, off, page)
        if rows is None:
            print(f"  skip offset {off}: retries exhausted (429/5xx)"); continue
        if sleep:
            time.sleep(sleep)
        for it in rows:
            row = it["row"]
            rk = src.get("resolved")
            resolved = row.get(rk) if rk else None
            fmt = src["fmt"]
            if fmt == "messages":
                traj = row.get(src["col"])
                if not isinstance(traj, list):
                    continue
                r = analyze_messages(tok, traj)
            elif fmt == "precomputed":
                # dataset ships its own token counts (provider tokenizer). Tokenise the
                # serialized trajectory with OUR tokenizer for a comparable EFFECTIVE CTX,
                # and keep the provider's CUMULATIVE input/output for the prefill:decode ratio.
                s = row.get(src["col"])
                r = dict(total=_toklen(tok, s) if isinstance(s, str) else None,
                         gen=None, tool=None, other=None, turns=row.get("assistant_messages"))
                r["cum_in"] = row.get("total_input_tokens")
                r["cum_out"] = row.get("total_output_tokens")
            else:  # string format: total only
                s = row.get(src["col"])
                if not isinstance(s, str) or not s:
                    continue
                r = dict(total=_toklen(tok, s), gen=None, tool=None, other=None, turns=None)
            r["resolved"] = bool(resolved) if resolved is not None else None
            recs.append(r)
        if len(recs) >= a.n:
            break
    recs = recs[:a.n]
    if not recs:
        raise SystemExit("no trajectories collected -- check --source/--split/--rows-total")

    print(f"\n=== {src['hf']} : {split}  |  env={src.get('env','?')}  |  {len(recs)} trajectories ===")
    summarize([r["total"] for r in recs], "EFFECTIVE CTX")
    if any(r.get("gen") is not None for r in recs):
        summarize([r["gen"] for r in recs], "generated")
        summarize([r["tool"] for r in recs], "tool output")
    if any(r.get("turns") is not None for r in recs):
        summarize([r["turns"] for r in recs], "turns")
    cin = [r["cum_in"] for r in recs if r.get("cum_in") is not None]
    cout = [r["cum_out"] for r in recs if r.get("cum_out") is not None]
    if cin and cout:
        summarize(cin, "cum input*")
        summarize(cout, "cum output*")
        mi = sum(cin) / len(cin); mo = sum(cout) / len(cout)
        print(f"  * provider tokenizer, CUMULATIVE over turns. prefill:decode ~{mi/max(mo,1):.0f}:1 "
              f"(no-prefix-cache upper bound; caching collapses prefill toward effective-ctx).")
    res = [r for r in recs if r["resolved"] is True]
    unres = [r for r in recs if r["resolved"] is False]
    if res and unres:
        print(f"  resolved n={len(res)}  effective-ctx mean {sum(r['total'] for r in res)/len(res):,.0f}"
              f"   |   unresolved n={len(unres)}  mean {sum(r['total'] for r in unres)/len(unres):,.0f}")
    # single-turn comparison anchor
    st_med = st.median([r["total"] for r in recs])
    print(f"\n  --> agentic effective context is ~{st_med/1e3:.0f}k median tokens; compare to single-turn "
          f"E[R] ~5-40k. Feed EFFECTIVE CTX into the quadratic (attn/KV) terms, GENERATED into decode.")
    if a.out:
        json.dump(recs, open(a.out, "w"))
        print(f"  raw per-trajectory records -> {a.out}")


if __name__ == "__main__":
    main()
