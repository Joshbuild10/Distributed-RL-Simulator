"""
build.py -- Write every figure and table: paper exports, CSVs and the dashboard. Items are independent,
so they build in parallel worker processes. Solver results and unchanged figures come from .cache/
(cache.py), so a rebuild only redraws what changed; PDF font embedding dominates a redraw (~3 s).
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

from feasibility import config_lines
from . import cache, figures, render, sweep, web  # noqa: F401  (figures registers everything)
from .scenarios import BATCH_MODES, TARGETS, batch_for, scenario
from .spec import REGISTRY, Fig


def build_one(name, opts, out, theme="light", paper=True, font_scale=1.0):
    """Build `name`; write its paper exports and CSV; return (section, id, title, card, name, seconds, redrawn)."""
    t0 = time.time()
    sec, fn = REGISTRY[name]
    it = fn(opts)
    sweep.flush()
    drawn = None                                       # tables are always rewritten; figures may be skipped
    if isinstance(it, Fig):
        drawn = False
        stem = os.path.join(out, "figures", it.id)
        files = [f"{stem}.{e}" for e in ("pdf", "svg", "png")] if paper else []
        key = cache.fig_key(it, theme, font_scale, paper)
        card = cache.get_output(out, it.id, key, files)
        if card is None:
            drawn = True
            svg = None
            if paper:
                render.save(it, stem, theme=theme, scale=font_scale)
                if theme == "light" and font_scale == 1:      # the paper SVG is the dashboard's, recoloured
                    with open(f"{stem}.svg", encoding="utf-8") as fh:
                        svg = fh.read()
            card = web.card(it, svg=svg)
            cache.put_output(out, it.id, key, card)
        it.data.to_csv(os.path.join(out, "tables", f"{it.id}.csv"), index=False)
    else:
        card = web.card(it)
        it.data.to_csv(os.path.join(out, "tables", f"{it.id}.csv"))
    return sec, it.id, it.title, card, name, time.time() - t0, drawn


def build(names, opts, out, theme="light", paper=True, workers=None, font_scale=1.0):
    for sub in ("figures", "tables"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    t0 = time.time()
    workers = workers or min(len(names), os.cpu_count() or 1)
    args = (opts, out, theme, paper, font_scale)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            done = list(pool.map(build_one, names, *zip(*[args] * len(names))))
    else:
        done = [build_one(n, *args) for n in names]
    for sec, fid, _, _, name, dt, drawn in done:
        print(f"  {fid:<4} {name:<28} {dt:5.1f}s{'  (unchanged)' if drawn is False else ''}")
    lines = config_lines(scenario(opts))
    lines.append(f"batch mode: {opts.batch_mode} -- {BATCH_MODES[opts.batch_mode]}")
    if opts.batch_mode != "flat":
        lines.append("  per target: " + ", ".join(
            f"{t} {b['prompts_per_batch']}x{b['responses_per_prompt']}"
            for t, b in ((t, batch_for(t, opts.batch_mode)) for t in TARGETS)))
    with open(os.path.join(out, "dashboard.html"), "w", encoding="utf-8") as fh:
        fh.write(web.page([d[:4] for d in done], lines))
    redrawn = sum(bool(d[6]) for d in done)
    print(f"wrote {out}  ({len(done)} items, {redrawn} redrawn, {workers} workers, {time.time() - t0:.0f}s)")
