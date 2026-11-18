"""The feasibility solver: invariants at every operating point, and the headline numbers."""
import json

import pytest

from specs import PenaltySpec
from tests import invariants as inv

QUICK = [(cfg, wd) for cfg in inv.grid(**inv.QUICK) for wd in inv.WINDOWS_D]
PENALTIES = {"off": None, "async": PenaltySpec(enabled=True), "relay": PenaltySpec(enabled=True, inf_mode="relay")}


@pytest.mark.parametrize("cfg,wd", QUICK, ids=lambda v: None if isinstance(v, int) else
                         f"{v.model.name}-{v.gpu}-n{v.gpus_per_node}-k{v.sync_interval:g}-s{v.stock_gpus}")
def test_invariants_quick(cfg, wd):
    assert inv.check(cfg, wd) == []


@pytest.mark.slow
@pytest.mark.parametrize("penalty", list(PENALTIES), ids=str)
def test_invariants_full(penalty):
    n, fails = inv.invariants(PENALTIES[penalty])
    assert n == 864
    assert fails == [], "\n".join(fails[:20])


def test_golden():
    with open(inv.GOLDEN) as fh:
        was = json.load(fh)
    now = inv.headline()
    assert now == was, "headline numbers moved; if intended: python -m tests.invariants --write"
