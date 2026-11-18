import pytest


def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", help="also run the slow tests (full gate, results)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        return
    skip = pytest.mark.skip(reason="slow: run with --full")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)
