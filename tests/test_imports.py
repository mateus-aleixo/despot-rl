"""Every module imports without the game data.

The tables are read when a ruleset is loaded, never at import time; this keeps
it that way, and catches a broken import anywhere in `sim/` or `rl/`.
"""
import importlib
import pkgutil

import pytest

import rl
import sim

MODULES = sorted(f"{pkg.__name__}.{m.name}"
                 for pkg in (sim, rl) for m in pkgutil.iter_modules(pkg.__path__))


@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)
