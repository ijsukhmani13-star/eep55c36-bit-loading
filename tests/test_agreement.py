"""Checks specific to the modification: identical output to the baseline.

The hypothesis is that the heap changes *speed only*. These tests check exact
equality (not approximate) of bits, energy and remaining budget on many cases,
including the tie-heavy flat channel and saturation.
"""
import numpy as np
import pytest

from src.baseline import allocate_scan
from src.data import make_flat_case, make_loading_case
from src.modified import allocate_heap, heap_allocate, heap_setup


def assert_identical(a, b):
    np.testing.assert_array_equal(a.bits, b.bits)
    assert a.energy_used == b.energy_used        # exact: same operations, same order
    assert a.remaining == b.remaining
    assert a.increments == b.increments


@pytest.mark.parametrize("seed", range(0, 20))            # development seeds only
@pytest.mark.parametrize("tones", [16, 64, 256])
@pytest.mark.parametrize("which", ["g_true", "g_hat"])
def test_heap_identical_to_scan_on_synthetic_channels(seed, tones, which):
    g = getattr(make_loading_case(seed, tones=tones), which)
    budget = 40.0 * tones
    assert_identical(allocate_scan(g, budget, gap=9.55, bmax=15),
                     allocate_heap(g, budget, gap=9.55, bmax=15))


@pytest.mark.parametrize("budget", [0.0, 1.0, 7.5, 64.0, 1e3, 1e9])
def test_flat_channel_ties_resolved_identically(budget):
    g = make_flat_case(tones=32, value=1.0)
    scan = allocate_scan(g, budget, gap=1.0, bmax=6)
    heap = allocate_heap(g, budget, gap=1.0, bmax=6)
    assert_identical(scan, heap)
    # tie rule: bits are non-increasing with tone index on a flat channel
    assert np.all(np.diff(scan.bits) <= 0)


def test_ablation_selector_identical_to_scan():
    g = make_loading_case(3, tones=128).g_true
    assert_identical(allocate_scan(g, 3000.0, gap=9.55, bmax=15),
                     allocate_heap(g, 3000.0, gap=9.55, bmax=15, selector="list-scan"))


def test_separate_setup_and_allocate_match_one_call():
    g = make_loading_case(4, tones=64).g_true
    state = heap_setup(g, gap=9.55, bmax=15)
    assert len(state["heap"]) == 64
    assert_identical(heap_allocate(state, 2000.0),
                     allocate_heap(g, 2000.0, gap=9.55, bmax=15))


def test_unknown_selector_rejected():
    with pytest.raises(ValueError):
        allocate_heap([1.0], 1.0, selector="quicksort")
