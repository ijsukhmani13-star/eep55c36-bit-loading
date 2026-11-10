"""Modification: greedy bit loading with priority-queue (binary heap) selection.

What changes, exactly
---------------------
Only the *selection step* of the baseline changes. The baseline finds the cheapest
next bit by scanning all N tones every time (``np.argmin``, O(N) per increment).
Here the next-bit costs live in a binary min-heap of ``(cost, tone_index)`` pairs
(Python ``heapq``), so

* the cheapest affordable increment is read from the heap top in O(1);
* after buying it, that tone's entry is replaced by ``(2 * cost, i)`` and sifted
  down in O(log N) (``heapreplace``); a tone that reaches ``bmax`` is popped.

Everything else is unchanged: the energy model, the stopping rule
(stop when the cheapest eligible cost exceeds the remaining energy), the order of
floating-point operations, and the tie rule. Python compares tuples
lexicographically, so among equal costs the smallest tone index is on top -- the
same "first index wins" rule as ``np.argmin``. The allocation is therefore
expected to be *bit-for-bit identical* to the baseline.

Cost: O(N) to build the heap (``heapify``) + O(B log N) for B increments,
versus Theta(N * B) for the baseline scan.

Attribution: using a priority queue for repeated minimum selection is standard
(e.g. Cormen et al., *Introduction to Algorithms*, ch. 6); the course guide lists it
as the suggested change for this track. The implementation here is my own.

Ablation (``selector="list-scan"``)
-----------------------------------
The heap version is written with Python lists and floats, while the baseline uses
NumPy arrays. To separate the effect of the *data structure* from the effect of the
*implementation style*, ``selector="list-scan"`` keeps every other decision of this
module fixed (same lists, same loop, same arithmetic) but switches off only the
heap, selecting by a linear Python scan instead.
"""
import heapq

import numpy as np

from .common import Allocation, validate_inputs

SELECTORS = ("heap", "list-scan")


def heap_setup(g, gap=1.0, bmax=15):
    """Reusable setup: validate inputs and build the heap of first-bit costs.

    Parameters
    ----------
    g : array_like, shape (N,)
    gap : float
    bmax : int

    Returns
    -------
    state : dict
        ``heap`` (list of (cost, index) tuples in heap order), ``bits`` (list of int,
        length N), ``bmax`` (int). Consumed (mutated) by :func:`heap_allocate`.
    """
    g, _, gap, bmax = validate_inputs(g, 0.0, gap, bmax)
    costs = (gap / g).tolist()                 # identical float64 values to the baseline
    heap = [(c, i) for i, c in enumerate(costs)] if bmax > 0 else []
    heapq.heapify(heap)                        # O(N)
    return {"heap": heap, "bits": [0] * len(costs), "bmax": bmax}


def heap_allocate(state, energy_budget):
    """Run the greedy loop on a prepared heap state (mutates ``state``).

    Returns
    -------
    Allocation
    """
    energy_budget = float(energy_budget)
    if not np.isfinite(energy_budget) or energy_budget < 0:
        raise ValueError("energy_budget must be finite and non-negative")
    heap, bits, bmax = state["heap"], state["bits"], state["bmax"]
    remaining, spent, increments = energy_budget, 0.0, 0

    while heap:
        cost, i = heap[0]                      # cheapest eligible increment
        if cost > remaining:
            break
        remaining -= cost
        spent += cost
        bits[i] += 1
        increments += 1
        if bits[i] < bmax:
            heapq.heapreplace(heap, (cost * 2.0, i))   # O(log N)
        else:
            heapq.heappop(heap)                        # tone reached the cap

    return Allocation(bits=np.array(bits, dtype=np.int64), energy_used=spent,
                      remaining=remaining, increments=increments)


def _list_scan_allocate(g, energy_budget, gap, bmax):
    """Ablation: same Python-list implementation, heap switched off (linear scan)."""
    g, remaining, gap, bmax = validate_inputs(g, energy_budget, gap, bmax)
    costs = (gap / g).tolist()
    n = len(costs)
    bits = [0] * n
    spent, increments = 0.0, 0
    inf = float("inf")
    while n > 0:
        best_i, best_c = -1, inf
        for i in range(n):                     # strict '<' keeps the first index on ties
            if bits[i] < bmax and costs[i] < best_c:
                best_i, best_c = i, costs[i]
        if best_c > remaining:                 # inf when all tones are capped
            break
        remaining -= best_c
        spent += best_c
        bits[best_i] += 1
        costs[best_i] = best_c * 2.0
        increments += 1
    return Allocation(bits=np.array(bits, dtype=np.int64), energy_used=spent,
                      remaining=remaining, increments=increments)


def allocate_heap(g, energy_budget, gap=1.0, bmax=15, selector="heap"):
    """Greedy discrete bit loading with priority-queue selection.

    Same signature and contract as :func:`src.baseline.allocate_scan`, plus
    ``selector`` for the ablation.

    Parameters
    ----------
    g : array_like, shape (N,)
    energy_budget : float
    gap : float, default 1.0
    bmax : int, default 15
    selector : {"heap", "list-scan"}, default "heap"
        "heap" is the modification; "list-scan" is the ablation (change switched off).

    Returns
    -------
    Allocation
        ``bits`` has shape (N,) and dtype int64. The input ``g`` is not modified.
    """
    if selector == "heap":
        validate_inputs(g, energy_budget, gap, bmax)   # same errors as the baseline
        return heap_allocate(heap_setup(g, gap, bmax), energy_budget)
    if selector == "list-scan":
        return _list_scan_allocate(g, energy_budget, gap, bmax)
    raise ValueError(f"selector must be one of {SELECTORS}")


def allocate_list_scan(g, energy_budget, gap=1.0, bmax=15):
    """Convenience wrapper for the ablation (``allocate_heap(..., selector="list-scan")``)."""
    return allocate_heap(g, energy_budget, gap, bmax, selector="list-scan")
