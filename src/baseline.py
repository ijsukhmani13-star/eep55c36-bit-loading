"""Baseline: greedy marginal-energy bit loading with a linear scan.

Model (gap approximation, see report Sec. 2)
--------------------------------------------
For tone i with channel-to-noise ratio g_i and gap Gamma, carrying b_i bits needs

    e_i(b_i) = (Gamma / g_i) * (2**b_i - 1).

The extra energy for one more bit is

    Delta_i(b_i) = e_i(b_i + 1) - e_i(b_i) = (Gamma / g_i) * 2**b_i,

so each tone's marginal cost doubles after every bit it receives.

Algorithm (reproduced core of Campello 1999)
--------------------------------------------
Start with b = 0. Repeatedly pick the tone with the smallest marginal cost among tones
below the cap (``argmin`` -> first index wins ties). If that cost exceeds the remaining
energy, stop; otherwise spend it and give that tone one bit.

Cost: each selection scans all N tones, so the total work is Theta(N * B), where
B = total bits allocated (B <= N * bmax).
"""
import numpy as np

from .common import Allocation, validate_inputs


def allocate_scan(g, energy_budget, gap=1.0, bmax=15):
    """Greedy discrete bit loading using a full scan for every increment.

    Parameters
    ----------
    g : array_like, shape (N,)
        Channel-to-noise ratio per unit energy for each tone (> 0).
    energy_budget : float
        Total energy E_max (>= 0), normalised units.
    gap : float, default 1.0
        SNR gap Gamma (> 0).
    bmax : int, default 15
        Maximum bits per tone.

    Returns
    -------
    Allocation
        ``bits`` has shape (N,) and dtype int64. The input ``g`` is not modified.
    """
    g, remaining, gap, bmax = validate_inputs(g, energy_budget, gap, bmax)
    n_tones = g.size
    bits = np.zeros(n_tones, dtype=np.int64)
    cost = gap / g                      # Delta_i(0): cost of each tone's first bit
    spent = 0.0
    increments = 0

    while n_tones > 0:
        eligible_cost = np.where(bits < bmax, cost, np.inf)
        i = int(np.argmin(eligible_cost))   # first index breaks ties
        step = eligible_cost[i]
        if step > remaining:                # also true when every tone is capped (inf)
            break
        remaining -= step
        spent += step
        bits[i] += 1
        cost[i] *= 2.0                      # Delta_i(b+1) = 2 * Delta_i(b), exact in binary
        increments += 1

    return Allocation(bits=bits, energy_used=spent, remaining=remaining,
                      increments=increments)
