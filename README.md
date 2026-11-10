# Discrete bit loading for DMT: linear scan vs priority queue

**EEP55C36 Computational Methods and Algorithms — final project, Track D (Data transmission).**
Author: Inderjeet Singh Sukhmani · Trinity College Dublin · November 2026

## What this project does

In discrete multitone (DMT) transmission, *bit loading* decides how many bits each tone carries under a
total energy budget. This repository:

1. **Reproduces** the discrete marginal-energy (greedy) allocation core of
   J. Campello, *"Practical Bit Loading for DMT"*, IEEE ICC 1999, DOI
   [10.1109/ICC.1999.765384](https://doi.org/10.1109/ICC.1999.765384). The **baseline** scans all tones
   with `numpy.argmin` for every one-bit increment.
2. **Changes one thing.** It selects the next increment from a **binary min-heap (priority queue)**
   instead. The allocation stays identical, but each step costs O(log N) instead of O(N).
3. **Compares** both methods fairly. It uses paired synthetic test cases at 3 problem sizes and 2 information
   conditions, plus stress cases, an ablation and 95% confidence intervals.

**Main finding.** The allocations are identical in 100% of 1186 runs. The heap is **13–14× faster** at
N = 64–1024 and **16.7× faster** at N = 4096. The advantage **disappears for tiny budgets**: at ρ = 0.01 the speed-up is
1.16×, 95% CI [0.96, 1.40]. The hypothesis "the speed-up grows with N" is only partly supported, because NumPy
vectorisation hides the O(N) scan cost until N ≈ 1000. Full write-up: [report/report.pdf](report/report.pdf).

Scope and simplifications: only the greedy core with a per-tone cap `bmax` is reproduced. The paper's
initialisation and acceleration steps are not. All data are **synthetic**, and no physical error-rate claims are made.

## Where things are

| What | Where |
|---|---|
| Baseline (scan) | [`src/baseline.py`](src/baseline.py), `allocate_scan` |
| Modification (heap) and ablation | [`src/modified.py`](src/modified.py), `allocate_heap`, `heap_setup`, `heap_allocate`, `allocate_list_scan` |
| Shared input checks and result type | [`src/common.py`](src/common.py) |
| Synthetic data generator and seed ranges | [`src/data.py`](src/data.py) |
| Metrics and 95% CIs | [`src/metrics.py`](src/metrics.py) |
| Figures (drawn only from CSV) | [`src/figures.py`](src/figures.py) |
| Experiment pipeline | [`run_experiments.py`](run_experiments.py) |
| Experiment settings | [`configs/smoke.json`](configs/smoke.json), [`configs/final.json`](configs/final.json) |
| Checks | [`tests/`](tests) |
| Submitted results | [`results/final/`](results/final) (`summary.csv`, `raw_runs.csv`, `validation.csv`, `environment.json`, `fig*.png`) |
| Report (PDF and LaTeX source) | [`report/report.pdf`](report/report.pdf), [`report/report.tex`](report/report.tex) |
| Development log | [`docs/progress.md`](docs/progress.md) |
| Screenshots of the final run (backup evidence) | [`docs/screenshots/`](docs/screenshots) |
| Interview / demo notes | [`docs/interview_notes.md`](docs/interview_notes.md) |

## Tested environment

- **Submitted results:** Windows 11 laptop, Python 3.13, run from the VS Code PowerShell terminal.
  The exact CPU, OS and versions are recorded in `results/final/environment.json`.
  Terminal screenshots of that run: [`docs/screenshots/`](docs/screenshots).
- Also tested on Ubuntu 24.04 (x86_64, 2-vCPU cloud VM, Python 3.13.18). Identical allocations, smaller speed-ups (~7×).
- NumPy 2.5.3, matplotlib 3.11.2, pytest 9.1.1 (pinned in `requirements.txt`).
- No SciPy, pandas or GPU needed. Any Python ≥ 3.10 with these packages should work. macOS was not tested, but no
  OS-specific code is used (only `pathlib` paths).

## Install, check and run

```bash
git clone https://github.com/ijsukhmani13-star/eep55c36-bit-loading.git
cd eep55c36-bit-loading
git checkout v1.0-submission
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m pytest                                          # checks: 348 passed, ~3 s
python run_experiments.py --config configs/smoke.json     # quick run, ~10 s
python run_experiments.py --config configs/final.json     # full results, ~3-4 min
python run_experiments.py --config configs/final.json --figures-only   # redraw figures from saved CSVs
```

All commands are run from the repository root. Output folders are created automatically.

### What you should see

| Command | Time (tested machine) | Output |
|---|---|---|
| `python -m pytest` | ~3 s | `348 passed` |
| smoke run | ~10 s | `results/smoke/` with `validation.csv`, `raw_runs.csv`, `summary.csv`, `environment.json`, `fig1`–`fig5` PNGs. Progress lines end with `finished in … s`. |
| final run | ~3–4 min | Same files in `results/final/`. They overwrite the committed results, which you can then compare with `git diff --stat results/final`. |

**Expected tolerance.** Allocations, bit totals, energies, agreement rates and calibration overspends are deterministic
and should match exactly (same seeds, same float operations). **Runtimes and speed-ups will differ on another
machine**, and also between runs on a shared machine. Expect the same picture: a large, roughly constant speed-up at moderate N (7× on the cloud VM, 14× on my laptop),
and little or no gain at ρ = 0.01. `raw_runs.csv` will therefore differ only in the timing columns.

## How data, seeds, parameters and test cases were chosen

- **Data.** These are synthetic 8-tap multipath channels from the course guide (`make_loading_case`). `g_true = |h|²/σ²`. `g_hat` is
  estimated from 20 averaged unit pilots and 20 independent noise-only observations. Both methods always receive the
  *same* `g` (either true or estimated), budget, gap and cap.
- **Seeds** (fixed in `src/data.py` before testing): development 0–19 (debugging and the agreement tests),
  validation 100–119, test 1000–1029. The seed sets are disjoint. Each case is a whole independent channel realisation.
- **Fixed model settings:** gap Γ = 9.8 dB (uncoded QAM at symbol-error rate ≈ 1e-7), `bmax` = 15, noise variance 0.1.
- **Chosen on validation only:** the energy per tone ρ in `E_max = ρN`. Rule: mean bits per tone closest to 5, a
  mid-range load. Candidates 10, 20, 40, 80, 160 gave 40 (see `results/final/validation.csv`).
- **Final test:** N ∈ {64, 256, 1024} × {true, calibrated} × 30 paired cases. The stress cases are a budget sweep
  (ρ = 0.01–40), a 4/20/100-pilot calibration test and a perfectly flat (all-ties) channel. There is also a scaling run up to N = 4096
  and the list-scan **ablation** (heap switched off, all else fixed).
- **Primary metric (declared in advance):** the paired speed-up `t_scan / t_heap` (larger is better), reported as a geometric mean with
  a 95% t-interval over independent cases.
- **Timing:** `time.perf_counter`, garbage collector off, 1 warm-up run, median of 5 repeats. Heap setup is included in its
  total and also reported separately.

## Limitations

- Synthetic channels, the gap approximation, and independent tones (no crosstalk). The greedy optimality proof needs
  separable costs.
- Timings come from one laptop, where background load adds noise. The heap's advantage depends on interpreted-Python versus NumPy overheads, not only on
  asymptotics.
- Planning on estimated `g` overspends the *true* energy (+8.9% on average with 20 pilots). This is a model and estimation
  issue that both methods share, and it is not a bug.
- Only the greedy core is reproduced. Campello's faster initialisation and the bit-removal steps are not.

## Interface

There is no graphical interface. The figures in `results/final/` are the demonstration, and `--figures-only` redraws them
from the saved CSVs. For a live demo, run the smoke config.

## Citations and assistance declaration

- J. Campello, "Practical bit loading for DMT," Proc. IEEE ICC, vol. 2, pp. 801–805, 1999.
- T. H. Cormen et al., *Introduction to Algorithms*, 3rd ed., MIT Press, 2009, ch. 6 (priority queues).
- The course mathematical guide (Track D) by D. Ramsook. The data generator follows its listing.

The paper PDF is **not** included (copyright).

**Assistance:** this project was developed with substantial help from a generative-AI assistant (Claude,
Anthropic), used under the module's AI policy. No external code or datasets were used. See the
statement in the report.
