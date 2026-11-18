"""
python -m analysis [build] [NAME ...]   write figures, tables and the dashboard (default: everything)
python -m analysis show NAME ...        print figures' data / tables to the terminal
python -m analysis list                 every figure and table name
python -m analysis explore              interactive explorer (Streamlit)

Options (any command): --penalties, --batch-mode MODE, --set FIELD=VALUE (repeatable; any FeasConfig
field, e.g. --set wan_mbps=10000 --set model=Llama3-70B --set train_distributed=false).

`build` writes under --out (default results/):
    dashboard.html              every figure and table, light/dark toggle, self-contained
    figures/<id>.pdf .svg .png  paper versions: no title (it belongs in the caption)
    tables/<id>.csv             the data behind each figure, and each table
"""
import argparse
import dataclasses
import os
import subprocess
import sys

from feasibility import FeasConfig

from . import figures  # noqa: F401  (registers everything)
from .scenarios import BATCH_MODES, Options
from .spec import REGISTRY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = {f.name for f in dataclasses.fields(FeasConfig)}


def _value(s):
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _override(s):
    k, sep, v = s.partition("=")
    if not sep or k not in FIELDS:
        raise argparse.ArgumentTypeError(f"expected FIELD=VALUE with FIELD a FeasConfig field, got {s!r}")
    return k, _value(v)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="build", choices=("build", "show", "list", "explore"))
    ap.add_argument("names", nargs="*", help="figure/table names (default: all; see `list`)")
    ap.add_argument("--penalties", action="store_true", help="apply Rahman's distributed-training penalties")
    ap.add_argument("--batch-mode", choices=tuple(BATCH_MODES), default="scaled")
    ap.add_argument("--set", dest="over", type=_override, action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--theme", choices=("light", "dark"), default="light", help="theme of the paper exports")
    ap.add_argument("--font-scale", type=float, default=1.0, help="scale every font, e.g. 1.5 for a poster")
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
    opts = Options(a.penalties, a.batch_mode, tuple(sorted(dict(a.over).items())))
    if a.command == "show":
        import pandas as pd
        pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
        for n in a.names or ap.error("show needs at least one name"):
            it = REGISTRY[n][1](opts)
            print(f"\n{it.id}  {it.title}\n{it.data.to_string()}\n" + (f"({it.caption})\n" if it.caption else ""))
        return
    from .build import build
    build(a.names or list(REGISTRY), opts, a.out, a.theme, paper=not a.no_paper, workers=a.workers,
          font_scale=a.font_scale)


if __name__ == "__main__":
    main()
