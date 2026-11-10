"""Shared input validation and result container for both allocation methods.

Keeping these in one place guarantees that the baseline and the modified method
accept exactly the same inputs and report results in the same format.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Allocation:
    """Result of a bit-loading run.

    Attributes
    ----------
    bits : np.ndarray, shape (N,), dtype int64
        Integer bits b_i assigned to each tone i.
    energy_used : float
        Sum of the marginal energies actually spent, in the same normalised units
        as the budget. Equals sum_i (gap / g_i) * (2**b_i - 1) up to round-off.
    remaining : float
        energy_budget - energy_used, as tracked during the greedy loop.
    increments : int
        Number of one-bit increments made (= bits.sum()).
    """

    bits: np.ndarray
    energy_used: float
    remaining: float
    increments: int


def validate_inputs(g, energy_budget, gap, bmax):
    """Check and convert the bit-loading inputs.

    Parameters
    ----------
    g : array_like, shape (N,)
        Channel-to-noise ratio per unit energy, g_i = |h_i|^2 / sigma_i^2. Must be
        finite and strictly positive.
    energy_budget : float
        Total energy E_max >= 0 (finite).
    gap : float
        SNR gap Gamma > 0 (Gamma >= 1 in practice; values below 1 are allowed only
        so that the hand-checkable examples in the tests can use Gamma = 1).
    bmax : int
        Per-tone cap on bits, integer >= 0.

    Returns
    -------
    g : np.ndarray, shape (N,), float64
        A *new* array (the caller's input is never modified).
    energy_budget, gap : float
    bmax : int

    Raises
    ------
    ValueError
        If any input violates the conditions above.
    """
    g = np.array(g, dtype=float, copy=True)
    if g.ndim != 1:
        raise ValueError("g must be a one-dimensional array of tones")
    if not np.all(np.isfinite(g)) or np.any(g <= 0):
        raise ValueError("g must be finite and strictly positive")
    energy_budget = float(energy_budget)
    if not np.isfinite(energy_budget) or energy_budget < 0:
        raise ValueError("energy_budget must be finite and non-negative")
    gap = float(gap)
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("gap must be finite and positive")
    if int(bmax) != bmax or bmax < 0:
        raise ValueError("bmax must be a non-negative integer")
    return g, energy_budget, gap, int(bmax)
