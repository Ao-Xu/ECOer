"""
Experiment 5: Wilcoxon signed-rank statistical significance tests.
Uses per-instance result arrays collected in Exp 2/3.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config
from src.stats import wilcoxon_ecoe_vs_baselines, format_significance_table, save_stats_results
from src.plotting import plot_exp5_stats

RESULTS_DIR_IN  = os.path.join(config.RESULTS_DIR, "exp2_proximity")
RESULTS_DIR_OUT = os.path.join(config.RESULTS_DIR, "exp5_stats")
os.makedirs(RESULTS_DIR_OUT, exist_ok=True)


def _extract_raw_arrays(method_results: dict, metric_key: str) -> list:
    """Extract the raw per-instance array from evaluate_all output."""
    key = f"_{metric_key}_raw"
    return method_results.get(key, [])


def run() -> None:
    print("[Exp 5] Loading Exp 2 results ...")

    # Collect per-instance arrays across all datasets × classifiers (knn5 only for main table)
    ecoe_raw   = {"l1": [], "l2": [], "sparsity": []}
    base_raw   = {b: {"l1": [], "l2": [], "sparsity": []} for b in config.BASELINES}
    method_map = {
        "ECOer": "ECOer",
        **{config.BASELINE_DISPLAY[b]: b for b in config.BASELINES},
    }

    for ds_name in config.DATASETS:
        cache = os.path.join(RESULTS_DIR_IN, f"{ds_name}_knn5.json")
        if not os.path.exists(cache):
            print(f"  [{ds_name}] Exp 2 results not found — skipping")
            continue
        with open(cache) as f:
            ds_results = json.load(f)

        # ECOer raw arrays
        for metric in ["l1", "l2", "sparsity"]:
            arr = _extract_raw_arrays(ds_results.get("ECOer", {}), metric)
            ecoe_raw[metric].extend(arr)

        # Baseline raw arrays
        for disp_name, b_key in method_map.items():
            if disp_name == "ECOer":
                continue
            for metric in ["l1", "l2", "sparsity"]:
                arr = _extract_raw_arrays(ds_results.get(disp_name, {}), metric)
                base_raw[b_key][metric].extend(arr)

    print("[Exp 5] Running Wilcoxon tests ...")
    stats_results = wilcoxon_ecoe_vs_baselines(
        ecoe_raw, base_raw,
        metrics=["l1", "l2", "sparsity"],
    )

    out_path = os.path.join(RESULTS_DIR_OUT, "wilcoxon_results.json")
    save_stats_results(stats_results, out_path)

    table = format_significance_table(stats_results)
    print("\n── Statistical significance (ECOer vs baselines) ──")
    print(table.to_string())
    table.to_csv(os.path.join(RESULTS_DIR_OUT, "wilcoxon_table.csv"))

    print("\n[Exp 5] Generating heatmap ...")
    plot_exp5_stats(stats_results)
    print("[Exp 5] Done.")


if __name__ == "__main__":
    run()
