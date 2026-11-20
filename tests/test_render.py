"""Every figure renders under contrasting configs with no exception, no overlapping text and no text
off the figure. Catches the label-layout regressions that only show under non-default settings."""
import itertools

import matplotlib
import pytest

from analysis import figures  # noqa: F401  (registers everything)
from analysis import render
from analysis.scenarios import Options
from analysis.spec import REGISTRY, Fig
from analysis.theme import rc

CONFIGS = {
    "default": Options(),
    "moe_broadcast_bound": Options(over=(("model", "Llama4-402B-A17B"),)),
    "mla_h200_penalties": Options(True, "scaled", (("gpu", "H200"), ("model", "DeepSeek-V3"))),
    "slow_link_small_stock": Options(False, "flat", (("wan_mbps", 100.0), ("stock_gpus", 10_000))),
    "long_responses": Options(False, "isocompute", (("er", 80_000.0), ("omega", 4.0))),
}
FIGS = [n for n, (_, fn) in REGISTRY.items() if not n.startswith("table_")]


def _texts(f):
    """(label, window bbox) of every drawn text: tick labels outside the axis range are skipped, and a
    legend label reported twice (by its container) counts once."""
    r = f.canvas.get_renderer()
    ticks_all, ticks_drawn = set(), set()
    for ax in f.axes:
        for axis in (ax.xaxis, ax.yaxis):
            ticks_all.update(id(t) for t in axis.get_ticklabels() + axis.get_ticklabels(minor=True))
            for tk in axis._update_ticks():
                ticks_drawn.update((id(tk.label1), id(tk.label2)))
    out, seen = [], set()
    for t in f.findobj(lambda a: hasattr(a, "get_text") and hasattr(a, "get_window_extent")):
        if id(t) in seen or (id(t) in ticks_all and id(t) not in ticks_drawn):
            continue
        seen.add(id(t))
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width >= 1 and bb.height >= 1:
            out.append((str(t.get_text()).replace("\n", " "), bb))
    return out


def layout_problems(f):
    texts, fb = _texts(f), f.bbox
    off = [s for s, b in texts if b.x0 < fb.x0 - 2 or b.x1 > fb.x1 + 2 or b.y0 < fb.y0 - 2 or b.y1 > fb.y1 + 2]
    over = []
    for (s1, a), (s2, b) in itertools.combinations(texts, 2):
        if s1 == s2 and abs(a.x0 - b.x0) < 3 and abs(a.y0 - b.y0) < 3:
            continue
        w, h = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
        if w > 1.5 and h > 1.5 and w * h > 0.05 * min(a.width * a.height, b.width * b.height):
            over.append(f"{s1!r} overlaps {s2!r}")
    return over + [f"{s!r} off the figure" for s in off]


@pytest.mark.slow
@pytest.mark.parametrize("config", list(CONFIGS))
def test_figures_render_cleanly(config):
    problems = []
    for name in FIGS:
        it = REGISTRY[name][1](CONFIGS[config])
        assert isinstance(it, Fig)
        theme = ("light", "dark")[list(CONFIGS).index(config) % 2]   # layout is theme-independent
        with matplotlib.rc_context(rc(theme)):
            f = render._draw(it, theme, False, render.WIDTH_IN)
            f.canvas.draw()
            problems += [f"{it.id} {theme}: {p}" for p in layout_problems(f)]
    assert not problems, "\n".join(problems[:30])


def test_detector_catches_overlap():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    f = Figure(figsize=(3, 2))
    FigureCanvasAgg(f)
    ax = f.subplots()
    ax.text(0.5, 0.5, "label A")
    ax.text(0.52, 0.5, "label B")
    ax.text(1.4, 0.5, "off the page")
    f.canvas.draw()
    assert len(layout_problems(f)) == 2
