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
