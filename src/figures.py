"""Figures for the report, generated ONLY from the saved CSV files.

Each method keeps the same colour and marker in every figure (colour is never the
only cue: markers and line styles differ too). No numbers are edited by hand.
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .baseline import allocate_scan  # noqa: E402
from .data import make_loading_case  # noqa: E402
from .modified import allocate_heap  # noqa: E402

STYLE = {  # fixed categorical order: scan, heap, ablation
    "scan": dict(color="#2a78d6", marker="o", ls="-", label="baseline: NumPy scan"),
    "heap": dict(color="#eb6834", marker="s", ls="--", label="modified: heap"),
    "list-scan": dict(color="#1baf7a", marker="^", ls=":",
                      label="ablation: heap off (Python scan)"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "lines.markersize": 5, "legend.frameon": False,
})


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    return float(x) if x not in ("", None) else np.nan


def _rows(summary, **match):
    return [r for r in summary if all(r[k] == str(v) for k, v in match.items())]


def fig_example(cfg, out):
    """Representative case: channel gain and identical allocations (test seed)."""
    ex, m = cfg["example"], cfg["model"]
    rho = next(float(r["rho"]) for r in _read(out / "validation.csv") if r["selected"] == "1")
    case = make_loading_case(ex["seed"], tones=ex["tones"], taps=m["taps"],
                             noise_var=m["noise_var"], pilots=m["pilots"])
    budget = rho * ex["tones"]
    scan = allocate_scan(case.g_true, budget, m["gap"], m["bmax"])
    heap = allocate_heap(case.g_true, budget, m["gap"], m["bmax"])
    k = np.arange(ex["tones"])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 3.9), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.2]})
    a1.plot(k, 10 * np.log10(case.g_true), color=INK, lw=1.2)
    a1.set_ylabel("g (dB)")
    a1.set_title(f"Representative test case: N = {ex['tones']} tones, seed {ex['seed']}, "
                 f"E_max = {budget:g}, Γ = {m['gap_db']} dB, b_max = {m['bmax']}", loc="left")
    a2.bar(k, scan.bits, width=0.8, color=STYLE["scan"]["color"], label=STYLE["scan"]["label"])
    a2.plot(k, heap.bits, ls="none", marker="s", ms=3.2, color=STYLE["heap"]["color"],
            label=STYLE["heap"]["label"])
    a2.set_ylabel("bits per tone")
    a2.set_xlabel("tone index i")
    same = "identical" if np.array_equal(scan.bits, heap.bits) else "DIFFERENT"
    a2.set_ylim(0, scan.bits.max() * 1.75)
    a2.text(0.99, 0.97, f"total {scan.bits.sum()} bits, energy used {scan.energy_used:.1f}\n"
            f"allocations {same}", transform=a2.transAxes, ha="right", va="top",
            color=INK, fontsize=8)
    a2.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(out / "fig1_example.png")
    plt.close(fig)


def fig_main(summary, cfg, out):
    """Overall comparison: runtime per N and paired speed-up with 95% CI."""
    tones = cfg["main"]["tones"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for method, col in (("scan", "scan_median_ms"), ("heap", "heap_median_ms"),
                        ("list-scan", "list_scan_median_ms")):
        y = [_f(_rows(summary, stage="main", tones=n, info="true")[0].get(col, ""))
             for n in tones]
        s = STYLE[method]
        a1.plot(tones, y, color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"])
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xlabel("number of tones N")
    a1.set_ylabel("median runtime per allocation (ms)")
    a1.set_title("(a) Runtime, true g (setup included)", loc="left")
    a1.legend(loc="upper left")

    width = 0.18
    for j, info in enumerate(cfg["main"]["info"]):
        r = [_rows(summary, stage="main", tones=n, info=info)[0] for n in tones]
        sp = np.array([_f(x["speedup_scan_over_heap"]) for x in r])
        lo = np.array([_f(x["speedup_ci_low"]) for x in r])
        hi = np.array([_f(x["speedup_ci_high"]) for x in r])
        x = np.arange(len(tones)) + (j - 0.5) * 2 * width
        a2.bar(x, sp, width=width * 1.8, color=STYLE["heap"]["color"],
               alpha=1.0 if info == "true" else 0.55, hatch=None if info == "true" else "///",
               edgecolor="white", linewidth=0.5, label=f"{info} g")
        a2.errorbar(x, sp, yerr=[sp - lo, hi - sp], fmt="none", ecolor=INK, capsize=3, lw=1)
        for xi, v in zip(x, sp):
            a2.text(xi, v + 0.55, f"{v:.1f}×", ha="center", va="bottom", fontsize=7.5,
                    color=INK)
    a2.axhline(1.0, color=MUTED, lw=1, ls="--")
    a2.set_xticks(np.arange(len(tones)), [f"N={n}" for n in tones])
    a2.set_ylabel("speed-up = t_scan / t_heap")
    n_cases = _rows(summary, stage="main", tones=tones[0], info="true")[0]["n_cases"]
    a2.set_title(f"(b) Paired speed-up ± 95% CI ({n_cases} cases/bar)", loc="left")
    a2.set_ylim(0, max(10, a2.get_ylim()[1] * 1.3))
    a2.legend(loc="upper left", ncol=2)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig2_main_comparison.png")
    plt.close(fig)


def fig_scaling(raw, out):
    """Complexity check: time per one-bit increment versus N."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    rows = [r for r in raw if r["stage"] == "scaling"]
    tones = sorted({int(r["tones"]) for r in rows})
    for method in ("scan", "heap", "list-scan"):
        med, n_ok = [], []
        for n in tones:
            v = [_f(r["t_total_s"]) / max(_f(r["increments"]), 1) * 1e6 for r in rows
                 if r["method"] == method and int(r["tones"]) == n]
            if v:
                med.append(np.median(v))
                n_ok.append(n)
        s = STYLE[method]
        ax.plot(n_ok, med, color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of tones N")
    ax.set_ylabel("µs per one-bit increment (median)")
    ax.set_title("Measured cost per one-bit increment (true g)", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig3_scaling.png")
    plt.close(fig)


def fig_budget(summary, out):
    """Stress / failure case: tiny budgets, where O(N) heap setup is not repaid."""
    r = sorted(_rows(summary, stage="budget"), key=lambda x: _f(x["rho"]))
    rho = np.array([_f(x["rho"]) for x in r])
    sp = np.array([_f(x["speedup_scan_over_heap"]) for x in r])
    lo = np.array([_f(x["speedup_ci_low"]) for x in r])
    hi = np.array([_f(x["speedup_ci_high"]) for x in r])
    inc = np.array([_f(x["increments_mean"]) for x in r])
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.fill_between(rho, lo, hi, color=STYLE["heap"]["color"], alpha=0.18, lw=0)
    ax.plot(rho, sp, color=STYLE["heap"]["color"], marker="s", label="geo. mean speed-up, 95% CI")
    ax.axhline(1, color=MUTED, ls="--", lw=1)
    ax.text(rho[-1], 0.93, "below 1: heap slower", color=MUTED, fontsize=7.5,
            ha="right", va="top")
    for x, y, b in zip(rho, sp, inc):
        ax.annotate(f"B≈{b:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 7) if y >= 1 else (14, -10), ha="center", fontsize=6.8,
                    color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("energy per tone ρ  (E_max = ρN)")
    ax.set_ylabel("speed-up t_scan / t_heap")
    ticks = [0.5, 1, 2, 4, 8]
    ax.set_yticks(ticks, [f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0.5, 10)
    ax.set_title(f"Stress: budget sweep, N = {r[0]['tones']}, {r[0]['n_cases']} cases per ρ",
                 loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig4_stress_budget.png")
    plt.close(fig)


def fig_calibration(raw, out):
    """Stress / model validity: planning with estimated g can overspend true energy."""
    rows = [r for r in raw if r["stage"] == "calibration" and r["method"] == "scan"]
    pilots = sorted({int(r["pilots"]) for r in rows})
    data = [[(_f(r["true_energy"]) / _f(r["budget"]) - 1) * 100 for r in rows
             if int(r["pilots"]) == p] for p in pilots]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.boxplot(data, tick_labels=[str(p) for p in pilots], widths=0.45,
               medianprops=dict(color=STYLE["scan"]["color"], lw=1.6),
               boxprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=INK, lw=0.9),
               capprops=dict(color=INK, lw=0.9),
               flierprops=dict(marker="o", ms=3, mfc="none", mec=MUTED))
    rng = np.random.default_rng(0)  # jitter for display only
    for i, d in enumerate(data, start=1):
        ax.plot(i + rng.uniform(-0.12, 0.12, len(d)), d, ls="none", marker="o", ms=2.6,
                color=STYLE["scan"]["color"], alpha=0.5)
    ax.axhline(0, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("pilot / noise-only observations used to estimate g")
    ax.set_ylabel("true energy vs budget (%)")
    ax.set_title("Stress: allocation planned on ĝ, checked on true g\n"
                 "(identical for scan and heap; >0 = over budget)", loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig5_stress_calibration.png")
    plt.close(fig)


def make_all_figures(cfg, out):
    out = Path(out)
    summary = _read(out / "summary.csv")
    raw = _read(out / "raw_runs.csv")
    fig_example(cfg, out)
    fig_main(summary, cfg, out)
    fig_scaling(raw, out)
    fig_budget(summary, out)
    fig_calibration(raw, out)
    with open(out / "figures.json", "w", encoding="utf-8") as fh:
        json.dump({"fig1_example.png": "representative example",
                   "fig2_main_comparison.png": "overall performance comparison",
                   "fig3_scaling.png": "complexity / scaling check",
                   "fig4_stress_budget.png": "stress: failure case (tiny budget)",
                   "fig5_stress_calibration.png": "stress: estimated-g model validity"},
                  fh, indent=2)
