#!/usr/bin/env python3
"""
lengths_io.py -- ONE schema-tolerant reader + descriptive statistics for every
response-length artefact this project produces.

Written because five places had grown their own loader/stats and they disagreed on schema:
a file written by an older script is a bare list or a {name: [lengths]} map, while
rl_length_probe writes {lengths, truncated, by_prompt, config}. Readers built for one shape
crash or silently skip the others.

Schemas accepted by load_series():
  1. {"lengths": [...], "truncated": n, "by_prompt": {...}, "config": {...}}   probe output
  2. {"name": [...], "name2": [...]}                                          multi-series
  3. [...]                                                                    bare list
  4. [{"total": .., "gen": .., "tool": .., "turns": ..}, ...]                 agentic trajectories
     (use agentic_records() for these -- load_series() reads the `total` field)

NOTE: rl_length_probe.summarize() deliberately keeps its own printing, bootstrap CI and
variance decomposition -- that is the live (paid) measurement path and is not refactored here.
This module is for READING artefacts back: plotting, dashboards, cross-run tables.
"""
import json
import os
import statistics as st

AGENTIC_KEYS = ("total", "gen", "tool", "turns")


def _clean(values):
    return sorted(float(x) for x in values if isinstance(x, (int, float)) and x)


def load_series(path):
    """-> {series_label: sorted[lengths]}. Empty dict if the file is missing/unreadable/empty.
    Single-series files are labelled from the filename (minus _lengths/.json)."""
    if not os.path.exists(path):
        return {}
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}
    label = os.path.basename(path).replace("_lengths.json", "").replace(".json", "")
    if isinstance(d, list):
        if d and isinstance(d[0], dict):                       # agentic trajectory records
            vals = _clean([r.get("total") for r in d if isinstance(r, dict)])
            return {label: vals} if vals else {}
        vals = _clean(d)
        return {label: vals} if vals else {}
    if isinstance(d, dict):
        if "lengths" in d:
            vals = _clean(d.get("lengths") or [])
            return {label: vals} if vals else {}
        out = {}
        for k, v in d.items():
            if isinstance(v, list):
                vals = _clean(v)
                if vals:
                    out[k] = vals
        return out
    return {}


def meta(path):
    """Censoring/config metadata for a probe-schema file: (truncated_frac, cap, model)."""
    if not os.path.exists(path):
        return (0.0, None, None)
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return (0.0, None, None)
    if not isinstance(d, dict) or "lengths" not in d:
        return (0.0, None, None)
    n = len([x for x in (d.get("lengths") or []) if x]) or 1
    cfg = d.get("config") or {}
    return ((d.get("truncated", 0) or 0) / n, cfg.get("max_tokens"), cfg.get("model"))


def stats(lengths, truncated_frac=0.0, cap=None):
    """Descriptive stats on an (unsorted is fine) list of lengths. `censored_frac` carries the
    right-censoring share so callers can flag the mean as a LOWER BOUND."""
    L = sorted(lengths)
    n = len(L)
    if not n:
        return None
    m = sum(L) / n
    sd = st.pstdev(L)
    q = lambda p: L[min(n - 1, int(p * n))]
    return dict(n=n, mean=m, median=st.median(L), sd=sd, cv=(sd / m if m else 0.0),
                p5=q(.05), p50=q(.5), p90=q(.9), p95=q(.95), p99=q(.99), max=L[-1],
                censored_frac=truncated_frac, cap=cap)


def load_stats(path):
    """-> {series_label: stats-dict}, with censoring metadata attached where available."""
    tf, cap, _ = meta(path)
    return {k: stats(v, tf, cap) for k, v in load_series(path).items()}


def load_many(paths):
    """Several files -> [(label, sorted_lengths, cap, censored_frac)], the shape the plotters use."""
    out = []
    for p in paths:
        tf, cap, _ = meta(p)
        for label, vals in load_series(p).items():
            out.append((label, vals, cap, tf))
    return out


def mean_length(path, series=None):
    """Mean of one series (the only one, or `series` by name). None if absent -- convenient for
    table/chart cells that must tolerate a missing measurement."""
    s = load_series(path)
    if not s:
        return None
    vals = s.get(series) if series else next(iter(s.values()))
    return (sum(vals) / len(vals)) if vals else None


def agentic_records(path):
    """Per-trajectory agentic records as-is. Use for the gen/tool/turns breakdown that
    load_series() flattens away (effective context = `total`)."""
    if not os.path.exists(path):
        return []
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    return [r for r in d if isinstance(r, dict)] if isinstance(d, list) else []


def agentic_means(path, keys=AGENTIC_KEYS):
    """{key: mean} over agentic trajectory records; missing keys map to None."""
    recs = agentic_records(path)
    out = {}
    for k in keys:
        vals = [r[k] for r in recs if isinstance(r.get(k), (int, float))]
        out[k] = (sum(vals) / len(vals)) if vals else None
    return out
