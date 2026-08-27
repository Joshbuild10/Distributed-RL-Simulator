"""
Put the repo root AND experiments/ on sys.path, so scripts in this directory can import
`specs` / `model` / `feasibility_core` (repo root) as well as `svgkit` / `lengths_io`
(experiments/ -- plot_dashboard.py still bare-imports those; they weren't moved here since
other experiments/ scripts also depend on them). Every script here is run directly
(`python results/foo.py`), so sys.path[0] is this directory and a plain `import _bootstrap`
always resolves.

Import for its side effect, before any repo-root/experiments import:

    import _bootstrap  # noqa: F401
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
