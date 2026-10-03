"""The Rust core's RNG against CPython's, bit for bit (core/src/rng.rs).

Crit and evasion draw random numbers, so the core can only be diffed exactly
against the Python oracle if both consume the same stream. These probes call
the core's exported test hooks and compare against `random.Random`.
"""
import ctypes
import random

import pytest

from sim import fast

# Zero, small, and seeds that need two 32-bit words in `init_by_array`.
SEEDS = [0, 1, 7, 42, 2**31 - 1, 2**32 - 1, 2**32, 2**32 + 1, 2**63 + 12345, 2**64 - 1]


def rng_probe(lib, seed: int, n: int) -> list[float]:
    lib.despot_rng_probe.restype = None
    lib.despot_rng_probe.argtypes = [ctypes.c_uint64, ctypes.c_int32,
                                     ctypes.POINTER(ctypes.c_double)]
    out = (ctypes.c_double * n)()
    lib.despot_rng_probe(seed, n, out)
    return list(out)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_matches_cpython(core, seed):
    # 2000 draws is 4000 words: past three regenerations of the 624-word state.
    py = random.Random(seed)
    assert rng_probe(core, seed, 2000) == [py.random() for _ in range(2000)]


@pytest.mark.parametrize("seed", SEEDS)
def test_choice_matches_cpython(core, seed):
    # `choice` rejects and redraws, so how many words it eats depends on the
    # values drawn; mixing lengths checks the two streams stay in step.
    lens = [1, 2, 3, 5, 6, 7, 10, 17, 100, 1000, 2**20 + 3] * 30
    py = random.Random(seed)
    assert fast.choice_probe(seed, lens) == [py.choice(range(n)) for n in lens]
