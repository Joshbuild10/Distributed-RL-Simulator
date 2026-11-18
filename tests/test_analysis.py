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
