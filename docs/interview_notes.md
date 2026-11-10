# Interview notes (5 minutes)

Backup if the live demo fails: `results/final/fig2_main_comparison.png`, `fig4_stress_budget.png` and
`summary.csv` are committed at tag `v1.0-submission`.

## 1. Two-minute explanation (with Fig. 2)

1. **Problem.** In DMT, each tone *i* has a channel-to-noise ratio g_i. A tone carrying b bits needs energy
   e_i(b) = (Γ/g_i)(2^b − 1). The task is to maximise the total bits Σb_i subject to Σe_i ≤ E_max and integer b_i ≤ bmax.
2. **Baseline (Campello's greedy core).** One more bit on tone i costs Δ_i = (Γ/g_i)·2^{b_i}, which doubles each time.
   Repeatedly buy the cheapest affordable bit (`np.argmin`, ties go to the lowest index) and stop when the cheapest is unaffordable.
   It is optimal for this separable model because each tone's costs increase.
3. **My change.** Only the selection step changes: a min-heap of (cost, index) replaces the full scan. The cost per bit
   drops from O(N) to O(log N), with O(N) setup. The tuple ordering keeps the same tie rule, so the output should be
   identical.
4. **Finding.** The allocations were identical in all 1186 runs. The heap is about 7× faster at N = 64–1024 and 12× at 4096. It is no faster
   for tiny budgets (ρ = 0.01: 0.87×), because the heap build is not repaid. The speed-up did **not** grow from 64 to 1024 as I
   predicted, because NumPy's `argmin` is vectorised and call overhead dominates. The ablation (pure-Python scan versus heap)
   shows the true data-structure effect: 10×, 30×, 99×.

## 2. One-minute demo

```bash
python -m pytest                                        # 348 passed
python run_experiments.py --config configs/smoke.json   # ~2 s
```
- Inputs: `configs/smoke.json` (sizes, ρ candidates, Γ = 9.8 dB, bmax = 15, timing repeats).
- Seeds: validation 100–104, test from 1000 (defined in `src/data.py`).
- Outputs: `results/smoke/` (CSV files and fig1–fig5). The final results are in `results/final/`.

## 3. Equation → code map

| Maths | Code |
|---|---|
| Δ_i(0) = Γ/g_i | `cost = gap / g` (baseline.py); `costs = (gap / g).tolist()` (modified.py) |
| skip tones at the cap | `np.where(bits < bmax, cost, np.inf)` / `heappop` when `bits[i] == bmax` |
| pick the cheapest, ties to the lowest index | `np.argmin(...)` / `heap[0]` with `(cost, i)` tuples |
| stop if Δ > remaining energy | `if step > remaining: break` / `if cost > remaining: break` |
| Δ_i(b+1) = 2Δ_i(b) | `cost[i] *= 2.0` / `heapreplace(heap, (cost * 2.0, i))` |
| e(b) = Γ/g (2^b − 1) (check) | `metrics.allocation_energy` |
| 95% CI of the speed-up | `metrics.geometric_speedup_ci95` (t-interval on paired log ratios) |

## 4. Likely questions and short answers

- **What does it optimise, and what do the parameters mean?** It maximises total bits under an energy budget. Γ is the SNR gap (how far a practical
  modulation and code is from capacity; 9.8 dB ≈ uncoded QAM at symbol-error rate 1e-7). bmax is the per-tone modulation cap. ρ is the energy per tone,
  with E_max = ρN.
- **Why is greedy optimal?** Each tone's marginal costs strictly increase, and you cannot buy bit k without bits 1…k−1. So the
  cheapest set of B increments is always the B globally cheapest affordable ones. If the next cheapest does not fit,
  nothing with B+1 bits fits. This **fails** with coupled tones (crosstalk), unequal bit steps, or another objective.
- **How did you show the baseline is correct?** With hand-computed cases, exhaustive search on 40 tiny random cases (same total bits and minimum
  energy), limits (zero budget, cap 0, saturation, empty input), and properties (feasibility, monotonicity in budget and in g, no input
  mutation). I also found the guide's example ([2,1] at E = 3) is infeasible, because it needs energy 4.
- **What exactly changed, and what does it cost?** Only the selection data structure. The extra cost is O(N) heap setup and O(N) memory for tuples.
  It loses when B (the total number of increments) is small compared with N.
- **How did you prevent unfair tuning or leakage?** ρ was chosen on validation seeds only, by a rule fixed beforehand. Test seeds were used only for the
  reported results. Both methods get identical g, budget, Γ and bmax, and the same timing harness, warm-up and repeats. There is no extra information.
- **Why a geometric mean?** Speed-ups are ratios, so averaging log ratios is symmetric, and pairing removes channel-to-channel variation.
  Each case is an independent channel, so the cases (not samples or tones) are the units for the CI.
- **What if I change the noise, size, seed or budget?** The allocation stays identical (it's the same algorithm). Smaller budgets (fewer bits)
  reduce the speed-up, which is below 1 at ρ = 0.01. Larger N eventually increases it. Noise changes g and hence B, but not agreement.
- **Least convincing result?** The scaling claim. The speed-up is flat from 64 to 1024, and timing comes from one shared VM. Also, ρ = 0.01 has a CI that
  includes 1, so I say "no faster", not "slower".
- **Where does the method fail?** With tiny budgets (speed). For both methods, planning on estimated g overspends the true energy (+9% with 20 pilots),
  because noise inflates |ĥ|² and greedy picks the over-estimated tones. That is model validity, not code correctness.
- **Reproduce a figure from the submitted commit?** `git checkout v1.0-submission`, then
  `python run_experiments.py --config configs/final.json --figures-only` redraws from the committed CSVs. Without `--figures-only`
  it regenerates everything.
