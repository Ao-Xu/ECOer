"""
Experiment 3: Comprehensive quality evaluation — DP, IM, Sparsity.
Single-CF and multi-CF (5) settings.
Reuses cached surrogates from Exp 2.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config
from src.preprocessing import load_processed
from src.classifiers import get_or_train_classifier
from src.r2snn import get_or_train_r2snn
from src.ecoe_optimizer import generate_counterfactuals_batch
from src.baselines import run_dice, run_face, run_growing_spheres, run_revise, run_wach, run_dpmdce
from src.metrics import evaluate_all
from src.plotting import plot_exp3_quality

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp3_quality")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _run_multi_cf(ds_name, clf_name, data, clf, model, Gamma, n_cfs, device):
    """Generate n_cfs CFs per instance by running ECOer with different seeds."""
    seed = config.SEED
    results = {}

    # ECOer: average over n_cfs runs with different seeds
    all_cfs = []
    for cf_i in range(n_cfs):
        cfs = generate_counterfactuals_batch(
            data["X_test"], model, Gamma, clf=clf,
            n_instances=config.N_TEST_INSTANCES, seed=seed + cf_i * 17, device=device
        )
        all_cfs.extend(cfs)
    results["ECOer"] = evaluate_all(all_cfs, data["X_train"], data["y_train"],
                                    data["cov_matrix"], clf)

    # Baselines (single CF per instance is sufficient for DP/IM quality metrics)
    baseline_fns = {
        "DiCE":           lambda: run_dice(data["X_train"], data["y_train"],
                                           data["X_test"], clf,
                                           n_instances=config.N_TEST_INSTANCES,
                                           n_cfs=n_cfs, seed=seed),
        "FACE":           lambda: run_face(data["X_train"], data["y_train"],
                                           data["X_test"], clf,
                                           n_instances=config.N_TEST_INSTANCES, seed=seed),
        "GrowingSpheres": lambda: run_growing_spheres(data["X_test"], clf,
                                                       n_instances=config.N_TEST_INSTANCES,
                                                       seed=seed),
        "Revise":         lambda: run_revise(data["X_train"], data["y_train"],
                                             data["X_test"], clf,
                                             n_instances=config.N_TEST_INSTANCES, seed=seed),
        "WACH":           lambda: run_wach(data["X_test"], clf, model,
                                           n_instances=config.N_TEST_INSTANCES,
                                           seed=seed, device=device),
        "DPMDCE":         lambda: run_dpmdce(data["X_test"], clf, model, Gamma,
                                              n_instances=config.N_TEST_INSTANCES,
                                              seed=seed, device=device),
    }
    for method_name, fn in baseline_fns.items():
        cfs = fn()
        results[method_name] = evaluate_all(cfs, data["X_train"], data["y_train"],
                                             data["cov_matrix"], clf)
    return results


def run() -> None:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[Exp 3] Device: {device}")

    all_1cf = {}
    all_5cf = {}

    for ds_name in config.DATASETS:
        all_1cf[ds_name] = {}
        all_5cf[ds_name] = {}
        data = load_processed(ds_name)

        for clf_name in config.CLASSIFIERS:
            for n_cfs, res_dict in [(1, all_1cf), (5, all_5cf)]:
                cache = os.path.join(RESULTS_DIR, f"{ds_name}_{clf_name}_{n_cfs}cf.json")
                if os.path.exists(cache):
                    print(f"  [{ds_name}/{clf_name}/{n_cfs}cf] cached")
                    with open(cache) as f:
                        res_dict[ds_name][clf_name] = json.load(f)
                    continue

                print(f"  [{ds_name}/{clf_name}/{n_cfs}cf] running ...")
                clf   = get_or_train_classifier(ds_name, clf_name,
                                                data["X_train"], data["y_train"])
                model, Gamma = get_or_train_r2snn(ds_name, clf_name, clf,
                                                   data["X_train"], device=device)
                method_results = _run_multi_cf(ds_name, clf_name, data, clf,
                                               model, Gamma, n_cfs, device)
                res_dict[ds_name][clf_name] = method_results
                with open(cache, "w") as f:
                    json.dump(method_results, f, indent=2,
                              default=lambda x: float(x) if hasattr(x, '__float__') else x)

    # ── Figures (primary classifier = KNN) ──
    print("[Exp 3] Generating quality figures ...")
    for clf_name in config.CLASSIFIERS:
        res_1cf_plot = {ds: all_1cf[ds].get(clf_name, {}) for ds in config.DATASETS}
        res_5cf_plot = {ds: all_5cf[ds].get(clf_name, {}) for ds in config.DATASETS}
        suffix = "" if clf_name == "knn5" else f"_{clf_name}"
        fname = f"exp2_quality_2x2_dp_im{suffix}.png"
        plot_exp3_quality(res_1cf_plot, res_5cf_plot, filename=fname)

    print("[Exp 3] Done.")


if __name__ == "__main__":
    run()
