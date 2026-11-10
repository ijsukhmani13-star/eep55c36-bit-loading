"""Correctness checks for the bit-loading allocators.

Every check is run for each method in ``METHODS`` so the baseline and the
modification are held to exactly the same contract.

Run from the repository root:  python -m pytest -q
"""
import itertools

import numpy as np
import pytest

from src.baseline import allocate_scan
from src.metrics import allocation_energy
from src.modified import allocate_heap, allocate_list_scan

METHODS = [allocate_scan, allocate_heap, allocate_list_scan]
METHOD_IDS = [m.__name__ for m in METHODS]


def brute_force(g, energy_budget, gap, bmax):
    """Exhaustive search over every capped allocation (tiny cases only).

    Written independently of the greedy code. Returns
    (max_total_bits, min_energy_among_allocations_with_that_many_bits).
    """
    best_bits, best_energy = -1, np.inf
    for b in itertools.product(range(bmax + 1), repeat=len(g)):
        energy = allocation_energy(b, g, gap)
        if energy <= energy_budget * (1 + 1e-12) + 1e-12:
            total = sum(b)
            if total > best_bits or (total == best_bits and energy < best_energy):
                best_bits, best_energy = total, energy
    return best_bits, best_energy


# ------------------------------------------------------------ hand-computable examples
@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_two_equal_tones_budget_3(allocate):
    # g=[1,1], Gamma=1: first-bit costs are 1 and 1.
    # step 1: tie -> tone 0 (first index), remaining 2, tone-0 cost -> 2
    # step 2: tone 1 is cheapest (1 < 2), remaining 1, tone-1 cost -> 2
    # step 3: cheapest is 2 > 1 -> stop.  Result [1, 1], energy 2.
    # NOTE: the course guide quotes [2, 1] "using energy 3" for this case, but
    # e(2)+e(1) = 3+1 = 4 > 3, so [2, 1] is infeasible at E=3 (see docs/progress.md).
    result = allocate([1.0, 1.0], 3.0, gap=1.0, bmax=2)
    assert result.bits.tolist() == [1, 1]
    assert result.energy_used == pytest.approx(2.0, abs=1e-12)
    assert result.remaining == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_two_equal_tones_budget_4_gives_guide_allocation(allocate):
    # With E=4 the third increment (cost 2, tie -> tone 0) fits exactly: [2, 1].
    result = allocate([1.0, 1.0], 4.0, gap=1.0, bmax=2)
    assert result.bits.tolist() == [2, 1]
    assert result.energy_used == pytest.approx(4.0, abs=1e-12)
    assert result.remaining == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_stronger_tone_is_loaded_first(allocate):
    # g=[1,4], Gamma=1: costs tone0 1,2,4..., tone1 0.25,0.5,1,2...
    # E=1.75: tone1 0.25 (rem 1.5), tone1 0.5 (rem 1.0), then tone0 and tone1 both
    # cost 1 -> tie -> tone 0 (rem 0) -> [1, 2].
    result = allocate([1.0, 4.0], 1.75, gap=1.0, bmax=5)
    assert result.bits.tolist() == [1, 2]
    # E=1.5 stops after the two cheap tone-1 bits -> [0, 2].
    assert allocate([1.0, 4.0], 1.5, gap=1.0, bmax=5).bits.tolist() == [0, 2]


# ------------------------------------------------------------------ edge / limiting cases
@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_zero_budget_gives_zero_bits(allocate):
    result = allocate([0.5, 2.0, 10.0], 0.0, gap=1.0, bmax=8)
    assert result.bits.tolist() == [0, 0, 0]
    assert result.energy_used == 0.0


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_budget_below_cheapest_bit(allocate):
    result = allocate([1.0, 2.0], 0.49, gap=1.0, bmax=8)   # cheapest bit costs 0.5
    assert result.bits.sum() == 0


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_zero_cap_gives_zero_bits(allocate):
    result = allocate([1.0, 2.0], 100.0, gap=1.0, bmax=0)
    assert result.bits.sum() == 0


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_huge_budget_saturates_every_tone_at_cap(allocate):
    result = allocate([0.1, 1.0, 10.0], 1e12, gap=1.0, bmax=6)
    assert result.bits.tolist() == [6, 6, 6]


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_no_tones(allocate):
    result = allocate([], 10.0, gap=1.0, bmax=4)
    assert result.bits.shape == (0,)
    assert result.increments == 0


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize("bad", [
    dict(g=[1.0, 0.0], energy_budget=1.0),          # zero gain
    dict(g=[1.0, -2.0], energy_budget=1.0),         # negative gain
    dict(g=[1.0, np.nan], energy_budget=1.0),       # NaN gain
    dict(g=[1.0, 2.0], energy_budget=-1.0),         # negative budget
    dict(g=[1.0, 2.0], energy_budget=np.inf),       # infinite budget
    dict(g=[[1.0, 2.0]], energy_budget=1.0),        # wrong shape
    dict(g=[1.0, 2.0], energy_budget=1.0, gap=0.0),
    dict(g=[1.0, 2.0], energy_budget=1.0, bmax=2.5),
    dict(g=[1.0, 2.0], energy_budget=1.0, bmax=-1),
])
def test_invalid_inputs_raise(allocate, bad):
    with pytest.raises(ValueError):
        allocate(**bad)


# ------------------------------------------------------------------- properties
@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize("seed", range(40))
def test_matches_exhaustive_search_on_tiny_cases(allocate, seed):
    """Greedy total bits equal the true optimum, with the minimum energy for that count."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 4))
    bmax = int(rng.integers(1, 4))
    g = rng.uniform(0.2, 5.0, size=n)
    gap = float(rng.uniform(1.0, 3.0))
    budget = float(rng.uniform(0.0, 3.0 * n * gap))
    result = allocate(g, budget, gap=gap, bmax=bmax)
    best_bits, best_energy = brute_force(g, budget, gap, bmax)
    assert result.bits.sum() == best_bits
    assert allocation_energy(result.bits, g, gap) == pytest.approx(best_energy, rel=1e-12)


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
@pytest.mark.parametrize("seed", range(10))
def test_feasibility_shape_and_finiteness(allocate, seed):
    rng = np.random.default_rng(1000 + seed)
    g = rng.exponential(10.0, size=200) + 1e-6
    budget = float(rng.uniform(0, 5000))
    result = allocate(g, budget, gap=9.55, bmax=15)
    assert result.bits.shape == (200,)
    assert result.bits.dtype == np.int64
    assert np.all((result.bits >= 0) & (result.bits <= 15))
    assert result.increments == int(result.bits.sum())
    assert np.isfinite(result.energy_used) and np.isfinite(result.remaining)
    # energy-budget feasibility, and the tracked energy agrees with the closed form
    assert result.energy_used <= budget * (1 + 1e-12)
    assert allocation_energy(result.bits, g, 9.55) == pytest.approx(result.energy_used,
                                                                    rel=1e-9)
    assert result.energy_used + result.remaining == pytest.approx(budget, rel=1e-9)


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_more_budget_never_gives_fewer_bits(allocate):
    rng = np.random.default_rng(7)
    g = rng.exponential(5.0, size=50) + 1e-3
    totals = [allocate(g, e, gap=2.0, bmax=10).bits.sum() for e in np.linspace(0, 500, 26)]
    assert all(a <= b for a, b in zip(totals, totals[1:]))


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_stronger_tones_never_get_fewer_bits(allocate):
    rng = np.random.default_rng(3)
    g = rng.exponential(5.0, size=100) + 1e-3
    bits = allocate(g, 800.0, gap=3.0, bmax=12).bits
    order = np.argsort(g)                        # weakest -> strongest (ties impossible here)
    assert np.all(np.diff(bits[order]) >= 0)


@pytest.mark.parametrize("allocate", METHODS, ids=METHOD_IDS)
def test_input_is_not_mutated(allocate):
    g = np.array([3.0, 1.0, 2.0])
    copy = g.copy()
    allocate(g, 10.0, gap=1.0, bmax=4)
    np.testing.assert_array_equal(g, copy)
