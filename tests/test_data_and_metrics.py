"""Checks for the synthetic data generator and the metric helpers."""
import math

import numpy as np
import pytest

from src.data import (DEV_SEEDS, TEST_SEEDS, VAL_SEEDS, make_flat_case,
                      make_loading_case)
from src.metrics import (allocation_energy, budget_violation,
                         geometric_speedup_ci95, mean_ci95)


def test_case_shapes_positive_and_finite():
    case = make_loading_case(seed=5, tones=128)
    for g in (case.g_true, case.g_hat):
        assert g.shape == (128,)
        assert np.all(np.isfinite(g)) and np.all(g > 0)


def test_same_seed_reproduces_and_different_seeds_differ():
    a, b, c = make_loading_case(1), make_loading_case(1), make_loading_case(2)
    np.testing.assert_array_equal(a.g_true, b.g_true)
    np.testing.assert_array_equal(a.g_hat, b.g_hat)
    assert not np.allclose(a.g_true, c.g_true)


def test_mean_channel_gain_matches_model():
    # Tap powers sum to 1, so E|h_k|^2 = 1 and E[g] = 1 / noise_var = 10.
    g = np.concatenate([make_loading_case(s, tones=64).g_true for s in range(200)])
    assert g.mean() == pytest.approx(10.0, rel=0.05)


def test_seed_sets_are_disjoint():
    dev, val, test = set(DEV_SEEDS), set(VAL_SEEDS), set(TEST_SEEDS)
    assert not (dev & val or dev & test or val & test)
    assert len(test) >= 20


def test_flat_case():
    g = make_flat_case(tones=16, value=4.0)
    assert g.shape == (16,) and np.all(g == 4.0)


def test_allocation_energy_hand_value():
    # gap=2: tone0 g=1, 2 bits -> 2*(4-1)=6 ; tone1 g=4, 1 bit -> 0.5*(2-1)=0.5
    assert allocation_energy([2, 1], [1.0, 4.0], 2.0) == pytest.approx(6.5)


def test_budget_violation():
    assert budget_violation([1, 1], [1.0, 1.0], 1.0, 2.0) == 0.0
    assert budget_violation([2, 1], [1.0, 1.0], 1.0, 3.0) == pytest.approx(1 / 3)


def test_mean_ci95_hand_value():
    # mean 2, sample sd 1, n=3 -> half width 4.303 * 1 / sqrt(3)
    mean, low, high = mean_ci95([1.0, 2.0, 3.0])
    half = 4.303 / math.sqrt(3)
    assert (mean, low, high) == pytest.approx((2.0, 2.0 - half, 2.0 + half))


def test_geometric_speedup_of_identical_times_is_one():
    s, lo, hi = geometric_speedup_ci95([1.0, 2.0, 4.0], [1.0, 2.0, 4.0])
    assert s == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
