# Progress log

Short record of decisions, problems and next steps, newest entries at the bottom.

## 1. Plan

- **Track chosen:** D — Data transmission: discrete bit loading
  (J. Campello, "Practical Bit Loading for DMT", IEEE ICC 1999, DOI 10.1109/ICC.1999.765384).
- **Reproduction scope:** only the discrete marginal-energy (greedy) allocation core under the
  gap approximation, with a per-tone cap `bmax`. The paper's initialisation and acceleration
  steps (e.g. starting from a rounded continuous solution, bit-removal "tightening") are **not**
  reproduced.
- **Baseline:** start from zero bits and repeatedly scan all tones (`numpy.argmin`) for the
  cheapest affordable next bit; first index wins ties.
- **Planned change:** keep the next-bit costs in a binary min-heap (priority queue) keyed by
  `(cost, tone index)`, so each selection costs O(log N) instead of O(N).
- **Hypothesis:** the heap returns the *identical* allocation (same tie rule, same floating-point
  operations) and its runtime advantage grows with the number of tones N.
- **Primary metric (chosen in advance):** paired per-case speed-up = baseline time / modified
  time (larger is better), summarised as a geometric mean with a 95% confidence interval.
  Secondary: exact allocation agreement, total bits, energy used, budget violation.

Next steps: data generator, baseline, checks.

## 2. Baseline implemented and verified

- `src/baseline.py` implements the scan version exactly as described in the guide
  (`np.where` to mask capped tones, `np.argmin` so the first index wins ties).
- Checks in `tests/` (run with `python -m pytest`): hand-computed examples, zero budget,
  zero cap, saturation, empty input, invalid inputs, exhaustive search on 40 random tiny
  cases, feasibility/shape/finiteness, monotonicity in budget and in g, and no input mutation.
- **Discrepancy found in the guide's sanity check.** The guide says g=[1,1], Gamma=1,
  E=3 gives [2,1] "using energy 3". Under the stated model e(b)=(Gamma/g)(2^b-1),
  [2,1] needs 3+1 = 4 > 3, so it is infeasible. Both the greedy rule and exhaustive search
  give [1,1] (energy 2) at E=3, and [2,1] (energy 4) at E=4. Both cases are tested.
- My own first hand calculation for g=[1,4] was also wrong (I forgot the tie at cost 1);
  the failing test caught it, and I corrected the *expectation*, after re-deriving it by hand.

Baseline is tagged `baseline-verified` before any change is introduced.

## 3. Modification: priority-queue selection

- `src/modified.py` changes **only the selection step**: a `heapq` min-heap of
  `(cost, tone_index)` replaces the `argmin` scan. Tuple ordering keeps the
  first-index tie rule. Capped tones are popped; others are re-inserted with cost x2.
- Setup (`heap_setup`, O(N) heapify) and allocation (`heap_allocate`, O(B log N)) are
  separate functions so their costs can be timed separately.
- **Ablation:** `selector="list-scan"` keeps the same Python-list loop and arithmetic but
  switches the heap off (linear Python scan). This separates the data-structure effect
  from the NumPy-vs-pure-Python effect.
- **Prediction:** helps most for large N and many increments; may help least (or hurt) for
  small N, where heap bookkeeping in Python competes with one vectorised NumPy `argmin`.
  The NumPy scan's per-increment cost should grow ~linearly in N, the heap's ~log N.
- Checks: all 13 contract tests now run for all three methods; `tests/test_agreement.py`
  asserts *exact* equality of bits, energy and remaining budget on 120 synthetic cases
  (dev seeds), tie-heavy flat channels and saturation. All pass.
