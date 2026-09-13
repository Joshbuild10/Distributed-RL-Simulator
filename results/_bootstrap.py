"""
Put the repo root on sys.path so scripts in this directory can import `specs` / `model` /
`feasibility_core` / `svgkit` / `lengths_io`. Every script here is run directly
(`python results/foo.py`), so sys.path[0] is this directory and a plain `import _bootstrap`
always resolves. experiments/ is appended too, but only so the gitignored measurement
scripts stay importable -- nothing in results/ needs it.

Import for its side effect, before any repo-root import:

    import _bootstrap  # noqa: F401
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
