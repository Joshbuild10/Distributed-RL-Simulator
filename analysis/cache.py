"""
cache.py -- Skip work an earlier build already did. Two layers, both under .cache/ (gitignored):

    results   every solver call sweep.cached makes, keyed by a hash of the model's source and data,
              so editing the model invalidates them and editing a chart doesn't;
    outputs   a figure's paper files and dashboard card, keyed by its data, its spec and the render
              code, so a figure whose numbers and styling are unchanged isn't redrawn.

Delete .cache/ (or pass --fresh) to rebuild from scratch.
"""
import dataclasses
import glob
import hashlib
import json
import os
import pickle
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, ".cache")
MODEL_SRC = ("model.py", "specs.py", "penalties.py", "lengths_io.py", "feasibility/*.py", "validation/*.py",
             "analysis/scenarios.py", "analysis/sweep.py", "data/lengths/*.json")
RENDER_SRC = ("analysis/render.py", "analysis/theme.py", "analysis/web.py", "analysis/spec.py")


def _digest(patterns):
    h = hashlib.sha256()
    for p in sorted(f for pat in patterns for f in glob.glob(os.path.join(ROOT, pat))):
        h.update(os.path.relpath(p, ROOT).encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


MODEL_HASH = _digest(MODEL_SRC)
RENDER_HASH = _digest(RENDER_SRC)
_RESULTS = os.path.join(DIR, "results", MODEL_HASH)


# --- results -------------------------------------------------------------------------------------
def load_results():
    """Every shard written under the current model hash, merged."""
    store = {}
    for p in glob.glob(os.path.join(_RESULTS, "*.pkl")):
        try:
            with open(p, "rb") as fh:
                store.update(pickle.load(fh))
        except Exception:                      # a torn shard from an interrupted build: ignore it
            pass
    return store


def save_results(new):
    """Write `new` as one shard (one file per writer, so parallel workers never clash)."""
    if new:
        os.makedirs(_RESULTS, exist_ok=True)
        tmp = os.path.join(_RESULTS, f"{uuid.uuid4().hex}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(new, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, tmp[:-4] + ".pkl")


# --- outputs -------------------------------------------------------------------------------------
def fig_key(fig, *extra):
    """Hash of everything a figure's rendering depends on: its data, every spec field (callables
    applied to the labels they format), the render code and `extra` (theme, scale, ...)."""
    h = hashlib.sha256(RENDER_HASH.encode())
    h.update(fig.data.to_csv().encode())
    for f in dataclasses.fields(fig):
        v = getattr(fig, f.name)
        if f.name == "data":
            continue
        if f.name == "x_fmt":
            v = [v(x) for x in fig.categories()]
        elif f.name == "series_fmt":
            v = [v(s) for s in fig.series_values()] if fig.series else None
        elif callable(v):
            raise TypeError(f"fig_key: no rule for callable spec field {f.name!r}")
        h.update(f"{f.name}={v!r};".encode())
    h.update(repr(extra).encode())
    return h.hexdigest()


def _stamp_path(out, fid):
    return os.path.join(DIR, "outputs", hashlib.sha1(os.path.abspath(out).encode()).hexdigest()[:12], f"{fid}.json")


def get_output(out, fid, key, files):
    """The cached dashboard card if `key` matches and every output file still exists, else None."""
    try:
        with open(_stamp_path(out, fid), encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError):
        return None
    return st["card"] if st.get("key") == key and all(os.path.exists(f) for f in files) else None


def put_output(out, fid, key, card):
    p = _stamp_path(out, fid)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"key": key, "card": card}, fh)
