"""The analysis layer: every figure and table builds, the CLI parses, and the committed results/
tables are what the current code produces (so the dashboard can always be regenerated)."""
import os

import pytest

from analysis import figures  # noqa: F401  (registers everything)
from analysis.__main__ import _override, main
from analysis.scenarios import Options, scenario
from analysis.spec import REGISTRY, Fig
from analysis.theme import THEMES, series_roles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(ROOT, "results", "tables")


def test_override_parsing():
    assert _override("wan_mbps=10000") == ("wan_mbps", 10000)
    assert _override("train_distributed=false") == ("train_distributed", False)
    assert _override("model=Llama3-70B") == ("model", "Llama3-70B")
    with pytest.raises(Exception):
        _override("not_a_field=1")


def test_set_reaches_the_config():
    cfg = scenario(Options(over=(("wan_mbps", 10_000.0), ("model", "Llama3-70B"))))
    assert cfg.wan_mbps == 10_000 and cfg.model.name.startswith("Llama")


def test_show_prints_a_table(capsys):
    main(["show", "table_caps"])
    assert "H100" in capsys.readouterr().out


def test_dark_theme_has_no_grey_only_series():
    for n in (1, 2, 3):
        assert "s0" not in series_roles(n, ordinal=True)
        assert "s0" not in series_roles(n)
    assert set(THEMES["dark"]) == set(THEMES["light"])


@pytest.mark.slow
@pytest.mark.parametrize("name", list(REGISTRY))
def test_results_reproduce(name):
    """results/tables/<id>.csv is exactly what the code produces now."""
    it = REGISTRY[name][1](Options())
    path = os.path.join(TABLES, f"{it.id}.csv")
    assert os.path.exists(path), f"{path} missing: run python -m analysis build"
    fresh = it.data.to_csv(index=not isinstance(it, Fig))
    with open(path, encoding="utf-8", newline="") as fh:
        assert fresh.replace("\r\n", "\n") == fh.read().replace("\r\n", "\n"), \
            f"{it.id} differs from results/: rebuild with python -m analysis build {name}"


def test_result_cache_is_thread_safe(monkeypatch):
    """The explorer flushes the result cache while other script threads are still filling it."""
    import threading
    from analysis import cache, sweep
    import time
    saved = []

    def save(new):                               # walk the dict as pickling real results does, yielding
        n = 0
        for _ in new:
            n += 1
            time.sleep(0)
        saved.append(n)
    monkeypatch.setattr(cache, "save_results", save)
    monkeypatch.setattr(sweep, "_cache", {})
    monkeypatch.setattr(sweep, "_new", {})
    errors = []

    def fill(k):
        try:
            for i in range(20_000):
                sweep.cached(lambda c, *a: c, f"{k}-{i}")
        except Exception as e:                   # pragma: no cover - the failure being tested for
            errors.append(e)

    fillers = [threading.Thread(target=fill, args=(k,)) for k in range(4)]
    for t in fillers:
        t.start()
    try:
        while any(t.is_alive() for t in fillers):   # flush continuously while the others write
            sweep.flush()
    except Exception as e:                       # pragma: no cover
        errors.append(e)
    for t in fillers:
        t.join()
    sweep.flush()
    assert not errors, errors[:1]
    assert sum(saved) == 80_000
