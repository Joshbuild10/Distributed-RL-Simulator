"""
Put the repo root on sys.path so `specs` / `model` / `feasibility` import cleanly from
scripts in this directory. Every script here is run directly (`python experiments/foo.py`), so
sys.path[0] is this directory and a plain `import _bootstrap` always resolves.

Import for its side effect, before any repo-root import:

    import _bootstrap  # noqa: F401
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
