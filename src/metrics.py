"""Quality, feasibility and statistics helpers.

Quality metrics measure the *allocation* (engineering task quality); timing is
handled in ``run_experiments.py``.
"""
import math

import numpy as np

# Two-sided 97.5% quantiles of Student's t distribution, for 95% confidence intervals.
# Values from standard tables; used instead of SciPy to keep dependencies minimal.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
         15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
         21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
         27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
         120: 1.980}


def allocation_energy(bits, g, gap):
    """Energy needed by an allocation, sum_i (gap / g_i) * (2**b_i - 1).

    Parameters
    ----------
    bits : array_like of int, shape (N,)
    g : array_like, shape (N,)
    gap : float

    Returns
    -------
    float
    """
    bits = np.asarray(bits, dtype=float)
    g = np.asarray(g, dtype=float)
    return float(np.sum(gap / g * (np.exp2(bits) - 1.0)))


def budget_violation(bits, g, gap, energy_budget):
    """Relative overspend max(0, E(b) - E_max) / E_max (0 means feasible).

    With the *true* g this measures whether an allocation planned with estimated g
    would really fit the budget. Returns 0.0 for a zero budget with zero energy.
    """
    used = allocation_energy(bits, g, gap)
    if energy_budget == 0:
        return 0.0 if used == 0 else math.inf
    return max(0.0, used - energy_budget) / energy_budget


def t_quantile_975(dof):
    """Student-t 97.5% quantile for ``dof`` degrees of freedom (table lookup)."""
    if dof < 1:
        raise ValueError("need at least 2 paired cases for a confidence interval")
    keys = sorted(_T975)
    for k in keys:
        if dof <= k:
            return _T975[k]
    return 1.960


def mean_ci95(values):
    """Mean and 95% t-interval of independent paired-case values.

    Parameters
    ----------
    values : array_like, shape (n,), n >= 2
        One value per independent case (e.g. log speed-up of a channel realisation).

    Returns
    -------
    (mean, low, high) : tuple of float
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    mean = float(x.mean())
    half = t_quantile_975(n - 1) * float(x.std(ddof=1)) / math.sqrt(n)
    return mean, mean - half, mean + half


def geometric_speedup_ci95(baseline_times, modified_times):
    """Geometric-mean speed-up (baseline / modified) with a 95% CI.

    The interval is computed on log ratios of *paired* cases and exponentiated, so a
    value above 1 means the modified method is faster.
    """
    b = np.asarray(baseline_times, dtype=float)
    m = np.asarray(modified_times, dtype=float)
    mean, low, high = mean_ci95(np.log(b / m))
    return math.exp(mean), math.exp(low), math.exp(high)
