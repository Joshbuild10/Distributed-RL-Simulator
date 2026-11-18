"""
python -m analysis [build] [NAME ...] [--theme light|dark] [--penalties] [--batch-mode MODE] [--out DIR]
python -m analysis explore          interactive explorer (Streamlit)
python -m analysis list             every figure and table name

`build` (the default) writes under DIR (default results/):
    dashboard.html              every figure and table, light/dark toggle, self-contained
    figures/<id>.pdf .svg .png  paper versions: no title (it belongs in the caption)
    tables/<id>.csv             the data behind each figure, and each table
"""
import argparse
import os
import subprocess
import sys

from .build import build
from .scenarios import BATCH_MODES, Options
from .spec import REGISTRY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="build", choices=("build", "explore", "list"))
    ap.add_argument("names", nargs="*", help="figure/table names (default: all; see `list`)")
    ap.add_argument("--theme", choices=("light", "dark"), default="light", help="theme of the paper exports")
    ap.add_argument("--penalties", action="store_true", help="apply Rahman's distributed-training penalties")
    ap.add_argument("--batch-mode", choices=tuple(BATCH_MODES), default="scaled")
    ap.add_argument("--no-paper", action="store_true", help="dashboard and CSVs only (skips PDF/SVG/PNG)")
    ap.add_argument("--workers", type=int, default=None, help="parallel build processes (default: one per CPU)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args(argv)
    if a.command == "list":
        for name, (sec, _) in REGISTRY.items():
            print(f"  {sec:<4} {name}")
        return
    if a.command == "explore":
        app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", app], cwd=ROOT))
    unknown = [n for n in a.names if n not in REGISTRY]
    if unknown:
        ap.error(f"unknown name(s) {unknown}; see `python -m analysis list`")
    build(a.names or list(REGISTRY), Options(a.penalties, a.batch_mode), a.out, a.theme,
          paper=not a.no_paper, workers=a.workers)


if __name__ == "__main__":
    main()
