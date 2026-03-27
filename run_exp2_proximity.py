"""
Experiment 2: Proximity metrics (ℓ₁, ℓ₂) — ECOer vs all baselines.
Runs on all 6 datasets × 3 classifiers.
Saves per-instance arrays for later statistical tests (Exp 5).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config
from src.preprocessing import load_processed
from src.classifiers import get_or_train_classifier
from src.r2snn import get_or_train_r2snn
from src.ecoe_optimizer import generate_counterfactuals_batch
from src.baselines import (run_dice, run_face, run_growing_spheres,
                            run_revise, run_wach, run_dpmdce)
from src.metrics import evaluate_all
from src.plotting import plot_exp2_proximity

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp2_proximity")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _run_all_methods(ds_name, clf_name, data, clf, model, Gamma, device):
    """Run ECOer + all 6 baselines on N_TEST_INSTANCES test points."""
    results = {}
    seed = config.SEED

    # ECOer
    print(f"    [ECOer] ...")
    t0 = time.time()
    ecoe_cfs = generate_counterfactuals_batch(
        data["X_test"], model, Gamma, clf=clf,
        n_instances=config.N_TEST_INSTANCES, seed=seed, device=device
    )
    results["ECOer"] = evaluate_all(ecoe_cfs, data["X_train"], data["y_train"],
                                    data["cov_matrix"], clf)
    print(f"    [ECOer] l1={results['ECOer']['l1_mean']:.4f}  ({time.time()-t0:.1f}s)")

    # DiCE
    print(f"    [DiCE] ...")
    t0 = time.time()
    dice_cfs = run_dice(data["X_train"], data["y_train"], data["X_test"], clf,
                        n_instances=config.N_TEST_INSTANCES, seed=seed)
    results["DiCE"] = evaluate_all(dice_cfs, data["X_train"], data["y_train"],
                                   data["cov_matrix"], clf)
    print(f"    [DiCE] l1={results['DiCE']['l1_mean']:.4f}  ({time.time()-t0:.1f}s)")

    # FACE
    print(f"    [FACE] ...")
    t0 = time.time()
    face_cfs = run_face(data["X_train"], data["y_train"], data["X_test"], clf,
                        n_instances=config.N_TEST_INSTANCES, seed=seed)
    results["FACE"] = evaluate_all(face_cfs, data["X_train"], data["y_train"],
                                   data["cov_matrix"], clf)
    print(f"    [FACE] l1={results['FACE']['l1_mean']:.4f}  ({time.time()-t0:.1f}s)")

    # GrowingSpheres
    print(f"    [GrowingSpheres] ...")
    t0 = time.time()
    gs_cfs = run_growing_spheres(data["X_test"], clf,
                                 n_instances=config.N_TEST_INSTANCES, seed=seed)
    results["GrowingSpheres"] = evaluate_all(gs_cfs, data["X_train"], data["y_train"],
                                             data["cov_matrix"], clf)
    print(f"    [GrowingSpheres] l1={results['GrowingSpheres']['l1_mean']:.4f}  ({time.time()-t0:.1f}s)")

    # Revise
    print(f"    [Revise] ...")
    t0 = time.time()
    rev_cfs = run_revise(data["X_train"], data["y_train"], data["X_test"], clf,
                         n_instances=config.N_TEST_INSTANCES, seed=seed)
    results["Revise"] = evaluate_all(rev_cfs, data["X_train"], data["y_train"],
                                     data["cov_matrix"], clf)
    print(f"    [Revise] l1={results['Revise']['l1_mean']:.4f}  ({time.time()-t0:.1f}s)")

    # WACH
    print(f"    [WACH] ...")
    t0 = time.time()
    wach_cfs = run_wach(data["X_test"], clf, model,
                        n_instances=config.N_TEST_INSTANCES, seed=seed, device=device)
    results["WACH"] = evaluate_all(wach_cfs, data["X_train"], data["y_train"],
                                   data["cov_matrix"], clf)
    print(f"    [WACH] l1={results['WACH']['l1_mean']:.4f}  ({time.time()-t0:.1f}s)")

    # DPMDCE
    print(f"    [DPMDCE] ...")
    t0 = time.time()
    dp_cfs = run_dpmdce(data["X_test"], clf, model, Gamma,
                        n_instances=config.N_TEST_INSTANCES, seed=seed, device=device)
    results["DPMDCE"] = evaluate_all(dp_cfs, data["X_train"], data["y_train"],
                                     data["cov_matrix"], clf)
    print(f"    [DPMDCE] l1={results['DPMDCE']['l1_mean']:.4f}  ({time.time()-t0:.1f}s)")

    return results


def run() -> None:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[Exp 2] Device: {device}")

    all_results = {}  # {ds_name: {clf_name: {method: metrics}}}

    for ds_name in config.DATASETS:
        all_results[ds_name] = {}
        data = load_processed(ds_name)

        for clf_name in config.CLASSIFIERS:
            cache_path = os.path.join(RESULTS_DIR, f"{ds_name}_{clf_name}.json")
            if os.path.exists(cache_path):
                print(f"  [{ds_name}/{clf_name}] cached — loading")
                with open(cache_path) as f:
                    all_results[ds_name][clf_name] = json.load(f)
                continue

            print(f"  [{ds_name}/{clf_name}] running ...")
            clf   = get_or_train_classifier(ds_name, clf_name,
                                            data["X_train"], data["y_train"])
            model, Gamma = get_or_train_r2snn(
                ds_name, clf_name, clf, data["X_train"],
                m=config.R2SNN_HIDDEN, device=device
            )
            method_results = _run_all_methods(
                ds_name, clf_name, data, clf, model, Gamma, device
            )
            all_results[ds_name][clf_name] = method_results

            with open(cache_path, "w") as f:
                json.dump(method_results, f, indent=2,
                          default=lambda x: float(x) if hasattr(x, '__float__') else x)
            print(f"  [{ds_name}/{clf_name}] done")

    # ── Generate proximity figure for the primary classifier (KNN) ──
    print("[Exp 2] Generating proximity figures ...")
    for clf_name in config.CLASSIFIERS:
        # Build {ds: {method_display: metrics}} for this clf
        res_for_plot = {}
        for ds_name in config.DATASETS:
            res_for_plot[ds_name] = {}
            for method, metrics in all_results[ds_name].get(clf_name, {}).items():
                res_for_plot[ds_name][method] = metrics

        suffix = "" if clf_name == "knn5" else f"_{clf_name}"
        fname = f"exp1_proximity_plot_1x2_std{suffix}.png"
        from src.plotting import plot_exp2_proximity
        plot_exp2_proximity(res_for_plot, filename=fname)

    print("[Exp 2] Done.")


if __name__ == "__main__":
    run()
