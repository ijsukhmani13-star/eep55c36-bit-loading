"""Run the full pipeline: data -> both methods (+ ablation) -> metrics -> CSV -> figures.

Usage (from the repository root)
--------------------------------
    python run_experiments.py --config configs/smoke.json   # ~10-20 s install check
    python run_experiments.py --config configs/final.json   # ~2-3 min, report results
    python run_experiments.py --config configs/final.json --figures-only

Outputs go to ``results/<config name>/`` (created automatically):
    validation.csv         rho candidates evaluated on validation seeds, and the choice
    raw_runs.csv           one row per (stage, setting, seed, method)
    summary.csv            paired speed-ups with 95% CIs, agreement, quality per setting
    environment.json       hardware, OS, Python/NumPy versions, git commit, config
    fig*.png               figures, generated from the CSV files only

Stages
------
validation   choose energy-per-tone rho on VALIDATION seeds (never test seeds)
main         3 tone counts x 2 information conditions x n test seeds (primary result)
scaling      runtime per increment versus N (complexity check)
budget       stress: sweep rho at fixed N (where the heap's O(N) setup can dominate)
calibration  stress: allocation planned with estimated g, energy checked with true g
ties         stress: perfectly flat channel (every selection is a tie)
"""
import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.baseline import allocate_scan                      # noqa: E402
from src.data import TEST_SEEDS, VAL_SEEDS, make_flat_case, make_loading_case  # noqa: E402
from src.metrics import (allocation_energy, budget_violation,  # noqa: E402
                         geometric_speedup_ci95, mean_ci95)
from src.modified import allocate_list_scan, heap_allocate, heap_setup  # noqa: E402

RAW_FIELDS = ["stage", "tones", "info", "rho", "pilots", "seed", "method",
              "increments", "total_bits", "budget", "energy_used", "true_energy",
              "true_violation", "agrees_with_scan", "t_setup_s", "t_alloc_s", "t_total_s"]


# ----------------------------------------------------------------------------- helpers
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["model"]["gap"] = 10 ** (cfg["model"]["gap_db"] / 10)
    return cfg


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment(cfg):
    import matplotlib
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "os": platform.platform(),
            "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "git_commit": git_commit(),
            "timer": "time.perf_counter, garbage collector disabled while timing",
            "config": cfg}


def time_call(fn, warmup, repeats, prepare=None):
    """Median wall time (s) of ``fn(prepare())`` over ``repeats`` runs after ``warmup``.

    ``prepare`` (untimed) builds a fresh argument for each run, e.g. a new heap that
    ``fn`` will consume. Returns (median_seconds, last_result).
    """
    times, result = [], None
    for k in range(warmup + repeats):
        arg = prepare() if prepare else None
        gc.disable()
        start = time.perf_counter()
        result = fn(arg) if prepare else fn()
        elapsed = time.perf_counter() - start
        gc.enable()
        if k >= warmup:
            times.append(elapsed)
    return statistics.median(times), result


def run_methods(g_plan, budget, cfg, g_true=None, timing=True, include_list_scan=True):
    """Run scan, heap (and the ablation) on identical inputs; return result rows.

    g_plan : gains the allocator is allowed to see (true or calibrated estimate).
    g_true : true gains, used only *after* allocation to measure real energy.
    """
    m, t = cfg["model"], cfg["timing"]
    gap, bmax = m["gap"], m["bmax"]
    warm, reps = (t["warmup"], t["repeats"]) if timing else (0, 1)
    g_true = g_plan if g_true is None else g_true
    rows = []

    t_scan, scan = time_call(lambda: allocate_scan(g_plan, budget, gap, bmax), warm, reps)
    t_setup, _ = time_call(lambda: heap_setup(g_plan, gap, bmax), warm, reps)
    t_alloc, heap = time_call(lambda st: heap_allocate(st, budget), warm, reps,
                              prepare=lambda: heap_setup(g_plan, gap, bmax))
    t_heap, _ = time_call(lambda: heap_allocate(heap_setup(g_plan, gap, bmax), budget),
                          warm, reps)
    results = [("scan", scan, math.nan, math.nan, t_scan),
               ("heap", heap, t_setup, t_alloc, t_heap)]
    if include_list_scan:
        t_ls, ls = time_call(lambda: allocate_list_scan(g_plan, budget, gap, bmax),
                             warm, reps)
        results.append(("list-scan", ls, math.nan, math.nan, t_ls))

    for name, res, ts, ta, tt in results:
        agrees = (np.array_equal(res.bits, scan.bits) and res.energy_used == scan.energy_used)
        true_e = allocation_energy(res.bits, g_true, gap)
        rows.append({"method": name, "increments": res.increments,
                     "total_bits": int(res.bits.sum()), "budget": budget,
                     "energy_used": res.energy_used, "true_energy": true_e,
                     "true_violation": budget_violation(res.bits, g_true, gap, budget),
                     "agrees_with_scan": int(agrees), "t_setup_s": ts, "t_alloc_s": ta,
                     "t_total_s": tt})
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def progress(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------ stages
def stage_validation(cfg, out):
    """Pick rho on validation seeds: mean bits/tone closest to the declared target."""
    v, m = cfg["validation"], cfg["model"]
    seeds = list(VAL_SEEDS)[: v["n_seeds"]]
    rows = []
    for rho in v["rho_candidates"]:
        per_tone = []
        for s in seeds:
            case = make_loading_case(s, tones=v["tones"], taps=m["taps"],
                                     noise_var=m["noise_var"], pilots=m["pilots"])
            res = allocate_scan(case.g_hat, rho * v["tones"], m["gap"], m["bmax"])
            per_tone.append(res.bits.mean())
        rows.append({"rho": rho, "mean_bits_per_tone": float(np.mean(per_tone)),
                     "distance_to_target": abs(float(np.mean(per_tone)) -
                                               v["target_bits_per_tone"])})
    best = min(rows, key=lambda r: r["distance_to_target"])
    for r in rows:
        r["selected"] = int(r is best)
    write_csv(out / "validation.csv", rows,
              ["rho", "mean_bits_per_tone", "distance_to_target", "selected"])
    progress(f"validation: selected rho = {best['rho']} "
             f"({best['mean_bits_per_tone']:.2f} bits/tone on validation seeds)")
    return best["rho"]


def stage_main(cfg, rho, raw):
    c, m = cfg["main"], cfg["model"]
    for tones in c["tones"]:
        for s in list(TEST_SEEDS)[: c["n_seeds"]]:
            case = make_loading_case(s, tones=tones, taps=m["taps"],
                                     noise_var=m["noise_var"], pilots=m["pilots"])
            for info in c["info"]:
                g_plan = case.g_true if info == "true" else case.g_hat
                for row in run_methods(g_plan, rho * tones, cfg, g_true=case.g_true):
                    raw.append({"stage": "main", "tones": tones, "info": info, "rho": rho,
                                "pilots": m["pilots"], "seed": s, **row})
        progress(f"main: N={tones} done")


def stage_scaling(cfg, rho, raw):
    c, m = cfg["scaling"], cfg["model"]
    for tones in c["tones"]:
        for s in list(TEST_SEEDS)[: c["n_seeds"]]:
            case = make_loading_case(s, tones=tones, taps=m["taps"],
                                     noise_var=m["noise_var"], pilots=m["pilots"])
            rows = run_methods(case.g_true, rho * tones, cfg,
                               include_list_scan=tones <= c["list_scan_max_tones"])
            for row in rows:
                raw.append({"stage": "scaling", "tones": tones, "info": "true", "rho": rho,
                            "pilots": m["pilots"], "seed": s, **row})
        progress(f"scaling: N={tones} done")


def stage_budget(cfg, raw):
    c, m = cfg["budget_stress"], cfg["model"]
    for r in c["rhos"]:
        for s in list(TEST_SEEDS)[: c["n_seeds"]]:
            case = make_loading_case(s, tones=c["tones"], taps=m["taps"],
                                     noise_var=m["noise_var"], pilots=m["pilots"])
            for row in run_methods(case.g_true, r * c["tones"], cfg, include_list_scan=False):
                raw.append({"stage": "budget", "tones": c["tones"], "info": "true", "rho": r,
                            "pilots": m["pilots"], "seed": s, **row})
    progress("budget stress done")


def stage_calibration(cfg, rho, raw):
    c, m = cfg["calibration_stress"], cfg["model"]
    for p in c["pilots"]:
        for s in list(TEST_SEEDS)[: c["n_seeds"]]:
            case = make_loading_case(s, tones=c["tones"], taps=m["taps"],
                                     noise_var=m["noise_var"], pilots=p)
            for row in run_methods(case.g_hat, rho * c["tones"], cfg, g_true=case.g_true,
                                   timing=False, include_list_scan=False):
                raw.append({"stage": "calibration", "tones": c["tones"], "info": "calibrated",
                            "rho": rho, "pilots": p, "seed": s, **row})
    progress("calibration stress done")


def stage_ties(cfg, rho, raw):
    c, m = cfg["ties_stress"], cfg["model"]
    for tones in c["tones"]:
        g = make_flat_case(tones, c["g_value"])
        for row in run_methods(g, rho * tones, cfg, include_list_scan=tones <= 1024):
            raw.append({"stage": "ties", "tones": tones, "info": "true", "rho": rho,
                        "pilots": "", "seed": "", **row})
    progress("ties stress done")


# ---------------------------------------------------------------------------- summary
def read_raw(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarise(raw_rows):
    """Paired statistics per setting. Pairs are matched on (stage, setting, seed)."""
    groups = {}
    for r in raw_rows:
        key = (r["stage"], r["tones"], r["info"], r["rho"], r["pilots"])
        groups.setdefault(key, {}).setdefault(r["method"], {})[r["seed"]] = r
    out = []
    for (stage, tones, info, rho, pilots), by_method in groups.items():
        scan, heap = by_method["scan"], by_method["heap"]
        seeds = sorted(set(scan) & set(heap))
        row = {"stage": stage, "tones": tones, "info": info, "rho": rho, "pilots": pilots,
               "n_cases": len(seeds)}
        ts = np.array([float(scan[s]["t_total_s"]) for s in seeds])
        th = np.array([float(heap[s]["t_total_s"]) for s in seeds])
        row["scan_median_ms"] = float(np.median(ts)) * 1e3
        row["heap_median_ms"] = float(np.median(th)) * 1e3
        row["heap_setup_median_ms"] = float(np.median(
            [float(heap[s]["t_setup_s"]) for s in seeds])) * 1e3
        row["increments_mean"] = float(np.mean([float(scan[s]["increments"]) for s in seeds]))
        if len(seeds) >= 2 and stage != "calibration":
            row["speedup_scan_over_heap"], row["speedup_ci_low"], row["speedup_ci_high"] = \
                geometric_speedup_ci95(ts, th)
        if "list-scan" in by_method and len(seeds) >= 2 and stage != "calibration":
            tl = np.array([float(by_method["list-scan"][s]["t_total_s"]) for s in seeds])
            row["list_scan_median_ms"] = float(np.median(tl)) * 1e3
            (row["ablation_speedup_listscan_over_heap"], row["ablation_ci_low"],
             row["ablation_ci_high"]) = geometric_speedup_ci95(tl, th)
        row["agreement_rate"] = float(np.mean(
            [int(m[s]["agrees_with_scan"]) for m in by_method.values() for s in seeds]))
        bits = [float(scan[s]["total_bits"]) for s in seeds]
        row["total_bits_mean"] = float(np.mean(bits))
        viol = np.array([float(scan[s]["true_violation"]) for s in seeds])
        row["true_violation_mean_pct"] = float(viol.mean()) * 100
        row["true_violation_max_pct"] = float(viol.max()) * 100
        row["frac_cases_over_budget"] = float(np.mean(viol > 0))
        if len(seeds) >= 2:
            _, lo, hi = mean_ci95(viol * 100)
            row["true_violation_ci_low_pct"], row["true_violation_ci_high_pct"] = lo, hi
        out.append(row)
    fields = []
    for r in out:
        fields += [k for k in r if k not in fields]
    return out, fields


# ------------------------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="path to a JSON config")
    parser.add_argument("--figures-only", action="store_true",
                        help="regenerate figures/summary from an existing raw_runs.csv")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = ROOT / "results" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    if not args.figures_only:
        progress(f"config {args.config} -> outputs in {out.relative_to(ROOT)}/")
        with open(out / "environment.json", "w", encoding="utf-8") as fh:
            json.dump(environment(cfg), fh, indent=2)
        rho = stage_validation(cfg, out)
        raw = []
        stage_main(cfg, rho, raw)
        stage_scaling(cfg, rho, raw)
        stage_budget(cfg, raw)
        stage_calibration(cfg, rho, raw)
        stage_ties(cfg, rho, raw)
        write_csv(out / "raw_runs.csv", raw, RAW_FIELDS)

    raw = read_raw(out / "raw_runs.csv")
    summary, fields = summarise(raw)
    write_csv(out / "summary.csv", summary, fields)

    from src.figures import make_all_figures
    make_all_figures(cfg, out)
    progress(f"finished in {time.perf_counter() - start:.1f} s; "
             f"see {out.relative_to(ROOT)}/summary.csv and fig*.png")


if __name__ == "__main__":
    main()
