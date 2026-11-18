"""
python -m analysis [FIGURE ...] [--penalties] [--batch-mode scaled|isocompute|flat] [--out DIR]

Builds the named figures (default: all) and writes, under DIR (default analysis/out):
    paper/<id>.pdf .svg .png    publication versions (no title; it belongs in the caption)
    tables/<id>.csv             the data behind each figure
    dashboard.html              every figure with its caption, data and download links
"""
import argparse
import os
import time

from feasibility_core import config_lines
from . import render, web
from .figures import FIGURES, table
from .scenarios import scenario
from .sweep import cache_info


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", choices=[[], *FIGURES], metavar="FIGURE",
                    help=f"any of: {', '.join(FIGURES)} (default: all)")
    ap.add_argument("--penalties", action="store_true", help="apply Rahman's distributed-training penalties")
    ap.add_argument("--batch-mode", choices=("scaled", "isocompute", "flat"), default="scaled")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    a = ap.parse_args(argv)

    os.makedirs(os.path.join(a.out, "paper"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    figs = []
    for name in a.figures or FIGURES:
        t0 = time.time()
        f = FIGURES[name](penalties=a.penalties, batch_mode=a.batch_mode)
        render.save(f, os.path.join(a.out, "paper", f.id))
        table(f).to_csv(os.path.join(a.out, "tables", f"{f.id}.csv"))
        figs.append(f)
        print(f"  {f.id:<4} {name:<20} {time.time() - t0:5.1f}s")
    lines = config_lines(scenario("decentralised", penalties=a.penalties))
    lines.append(f"batch mode: {a.batch_mode} (per-budget charts)")
    with open(os.path.join(a.out, "dashboard.html"), "w", encoding="utf-8") as fh:
        fh.write(web.page(figs, lines))
    print(f"wrote {a.out}  ({len(figs)} figures, {cache_info()} cached evaluations)")


if __name__ == "__main__":
    main()
