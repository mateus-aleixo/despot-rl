"""Shared fixtures.

Most of the simulator reads the game's balance tables, which are extracted from
a local copy of the game into data/ and never committed. The suite is split on
that line: everything checkable without the tables runs anywhere, CI included,
and the tests that need them skip when data/ is empty.
"""
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
# tools/ is a folder of scripts, not a package; the Rijndael tests import from it.
sys.path.insert(0, str(ROOT / "tools"))

HAS_DATA = (ROOT / "data" / "extracted" / "gamedata" / "metadata.txt").exists()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_data: needs the extracted game tables in data/ (skips without them)")


def pytest_collection_modifyitems(config, items):
    if HAS_DATA:
        return
    skip = pytest.mark.skip(reason="needs the extracted game tables in data/; see the README's Data section")
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def core():
    """The Rust core, or a skip when it has not been built.

    CI sets DESPOT_REQUIRE_CORE so that a missing build fails the run instead,
    and the parity tests cannot pass by not running.
    """
    from sim import fast
    try:
        return fast.load()
    except fast.CoreUnavailable as exc:
        if os.environ.get("DESPOT_REQUIRE_CORE"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
