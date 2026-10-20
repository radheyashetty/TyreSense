import os
import glob
import pytest


def pytest_collection_modifyitems(config, items):
    """Skip data, slow, network tests on plain pytest; run when -m is specified."""
    mark_expr = config.getoption("-m")
    if not mark_expr:
        skip_marker = pytest.mark.skip(reason="skipped: marked data/slow/network; run with -m <marker>")
        for item in items:
            if any(m in item.keywords for m in ["data", "slow", "network"]):
                item.add_marker(skip_marker)
