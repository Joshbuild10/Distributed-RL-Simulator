"""
build.py -- Write every figure and table: paper exports, CSVs and the dashboard. Items are independent,
so they build in parallel worker processes; PDF font embedding dominates the cost (~3 s a figure).
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

from feasibility_core import config_lines
from . import figures  # noqa: F401  (registers everything)
from . import render, web
from .scenarios import BATCH_MODES, TARGETS, batch_for, scenario
from .spec import REGISTRY, Fig


def build_one(name, opts, out, theme="light", paper=True):
    """Build `name`; write its paper exports and CSV; return (section, id, title, dashboard card)."""
    t0 = time.time()
    sec, fn = REGISTRY[name]
    it = fn(opts)
    if isinstance(it, Fig):
        if paper:
            render.save(it, os.path.join(out, "figures", it.id), theme=theme)
        it.data.to_csv(os.path.join(out, "tables", f"{it.id}.csv"), index=False)
    else:
        it.data.to_csv(os.path.join(out, "tables", f"{it.id}.csv"))
    return sec, it.id, it.title, web.card(it), name, time.time() - t0


def build(names, opts, out, theme="light", paper=True, workers=None):
    for sub in ("figures", "tables"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    t0 = time.time()
    workers = workers or min(len(names), os.cpu_count() or 1)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            done = list(pool.map(build_one, names, *zip(*[(opts, out, theme, paper)] * len(names))))
    else:
        done = [build_one(n, opts, out, theme, paper) for n in names]
    for sec, fid, _, _, name, dt in done:
        print(f"  {fid:<4} {name:<24} {dt:5.1f}s")
    lines = config_lines(scenario(opts))
    lines.append(f"batch mode: {opts.batch_mode} -- {BATCH_MODES[opts.batch_mode]}")
    if opts.batch_mode != "flat":
        lines.append("  per target: " + ", ".join(
            f"{t} {b['prompts_per_batch']}x{b['responses_per_prompt']}"
            for t, b in ((t, batch_for(t, opts.batch_mode)) for t in TARGETS)))
    with open(os.path.join(out, "dashboard.html"), "w", encoding="utf-8") as fh:
        fh.write(web.page([d[:4] for d in done], lines))
    print(f"wrote {out}  ({len(done)} items, {workers} workers, {time.time() - t0:.0f}s)")
