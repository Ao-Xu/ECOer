"""
Statistical significance tests:
  Wilcoxon signed-rank test comparing ECOer vs each baseline on each metric.
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def wilcoxon_ecoe_vs_baselines(
    ecoe_per_instance: dict,       # {metric: list[float]}  raw per-instance for ECOer
    baseline_per_instance: dict,   # {baseline: {metric: list[float]}}
    metrics: list = None,
    alternative: str = "two-sided",
) -> dict:
    """
    Run scipy.stats.wilcoxon for each (baseline, metric) pair.

    Returns
    -------
    nested dict: {baseline: {metric: {'statistic', 'p_value', 'n'}}}
    """
    if metrics is None:
        metrics = ["l1", "l2", "sparsity"]

    results = {}
    for baseline, base_data in baseline_per_instance.items():
        results[baseline] = {}
        for metric in metrics:
            ecoe_vals  = np.array(ecoe_per_instance.get(metric, []))
            base_vals  = np.array(base_data.get(metric, []))

            # Align lengths
            n = min(len(ecoe_vals), len(base_vals))
            if n < 10:
                results[baseline][metric] = {"statistic": float("nan"),
                                             "p_value": float("nan"), "n": n}
                continue
            ecoe_v = ecoe_vals[:n]
            base_v = base_vals[:n]
            diff = ecoe_v - base_v
            if np.all(diff == 0):
                results[baseline][metric] = {"statistic": 0.0, "p_value": 1.0, "n": n}
                continue
            try:
                stat, pval = stats.wilcoxon(ecoe_v, base_v, alternative=alternative)
            except Exception:
                stat, pval = float("nan"), float("nan")
            results[baseline][metric] = {
                "statistic": float(stat),
                "p_value":   float(pval),
                "n":         n,
            }
    return results


def format_significance_table(results: dict) -> pd.DataFrame:
    """
    Returns a DataFrame with baselines as rows, metrics as columns.
    Cells show p-value with significance markers: * p<0.05, ** p<0.01, *** p<0.001
    """
    baselines = list(results.keys())
    if not baselines:
        return pd.DataFrame()
    metrics = list(results[baselines[0]].keys())

    def fmt(r):
        p = r.get("p_value", float("nan"))
        if np.isnan(p):
            return "n/a"
        stars = ""
        if p < 0.001: stars = "***"
        elif p < 0.01: stars = "**"
        elif p < 0.05: stars = "*"
        return f"{p:.4f}{stars}"

    data = {}
    for metric in metrics:
        data[metric] = [fmt(results[b].get(metric, {})) for b in baselines]

    df = pd.DataFrame(data, index=[config.BASELINE_DISPLAY.get(b, b) for b in baselines])
    return df


def save_stats_results(results: dict, path: str) -> None:
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
