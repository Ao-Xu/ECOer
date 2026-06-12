"""
Experiment 4: Ablation studies on Adult dataset.
  (a) Classifier choice: KNN k=5/10, Random Forest, SVM
  (b) Architecture (m) + convex relaxation
  (c) Reconstruction-regularization components
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config
from src.preprocessing import load_processed
from src.classifiers import get_or_train_classifier, train_classifier
from src.r2snn import get_or_train_r2snn, train_r2snn, build_elm_reconstruction
from src.ecoe_optimizer import generate_counterfactuals_batch
from src.baselines import run_dice, run_wach, run_dpmdce
from src.metrics import evaluate_all
from src.plotting import plot_exp4_classifiers, plot_exp4_components
from sklearn.neighbors import KNeighborsClassifier

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp4_ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)

DS_NAME = "adult"   # All ablation on Adult dataset


# ──────────────────────────────────────────────────────────────────────────────
# (a) Classifier choice ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation_classifiers(data, device):
    cache = os.path.join(RESULTS_DIR, "ablation_classifiers.json")
    if os.path.exists(cache):
        print("  [4a] cached")
        with open(cache) as f:
            return json.load(f)

    clf_variants = {
        "KNN k=5":  KNeighborsClassifier(n_neighbors=5),
        "KNN k=10": KNeighborsClassifier(n_neighbors=10),
    }
    # Add RF and SVM via standard names
    for name in ["rf", "svm"]:
        clf_variants[config.CLF_DISPLAY[name]] = get_or_train_classifier(
            DS_NAME, name, data["X_train"], data["y_train"]
        )

    results = {}
    for variant_label, clf in clf_variants.items():
        print(f"    [4a {variant_label}] ...")
        if not hasattr(clf, "predict"):
            clf.fit(data["X_train"], data["y_train"])

        model, Gamma = get_or_train_r2snn(
            DS_NAME, variant_label.replace(" ", "_").lower()[:15],
            clf, data["X_train"], device=device
        )

        ecoe_cfs = generate_counterfactuals_batch(
            data["X_test"], model, Gamma, clf=clf,
            n_instances=config.N_TEST_INSTANCES, device=device
        )
        dice_cfs = run_dice(data["X_train"], data["y_train"],
                            data["X_test"], clf,
                            n_instances=config.N_TEST_INSTANCES)

        results[variant_label] = {
            "ECOer": evaluate_all(ecoe_cfs, data["X_train"], data["y_train"],
                                  data["cov_matrix"], clf),
            "DiCE":  evaluate_all(dice_cfs, data["X_train"], data["y_train"],
                                  data["cov_matrix"], clf),
        }

    with open(cache, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else x)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# (b) Architecture (m) sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation_arch(data, clf, device):
    cache = os.path.join(RESULTS_DIR, "ablation_arch.json")
    if os.path.exists(cache):
        print("  [4b] cached")
        with open(cache) as f:
            return json.load(f)

    m_values = [20, 30, 40, 50]
    arch_results = {}

    for m in m_values:
        print(f"    [4b m={m}] ...")
        t0 = time.time()
        model, Gamma = get_or_train_r2snn(
            DS_NAME, "knn5", clf, data["X_train"], m=m, device=device
        )
        ecoe_cfs = generate_counterfactuals_batch(
            data["X_test"], model, Gamma, clf=clf,
            n_instances=config.N_TEST_INSTANCES, device=device
        )
        metrics = evaluate_all(ecoe_cfs, data["X_train"], data["y_train"],
                               data["cov_matrix"], clf)
        metrics["runtime_mean"] = float(np.mean([r["runtime"] for r in ecoe_cfs
                                                  if r is not None]))
        arch_results[str(m)] = metrics
        print(f"    [4b m={m}] l1={metrics['l1_mean']:.4f}  ({time.time()-t0:.1f}s)")

    with open(cache, "w") as f:
        json.dump(arch_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else x)
    return arch_results


# ──────────────────────────────────────────────────────────────────────────────
# (c) Reconstruction components ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation_reconstruction(data, clf, model, Gamma, device):
    """
    4 variants:
      proximity_only : WACH (x-space, no feature space)
      +validity      : DPMDCE (feature space, term I only)
      +manifold      : feature space, term II only (beta=1, lambda=0)
      full_ECOer     : complete ECOer
    """
    cache = os.path.join(RESULTS_DIR, "ablation_reconstruction.json")
    if os.path.exists(cache):
        print("  [4c] cached")
        with open(cache) as f:
            return json.load(f)

    results = {}

    # proximity_only = WACH
    print("    [4c proximity_only=WACH] ...")
    wach_cfs = run_wach(data["X_test"], clf, model,
                        n_instances=config.N_TEST_INSTANCES, device=device)
    results["Proximity\nOnly"] = evaluate_all(wach_cfs, data["X_train"], data["y_train"],
                                               data["cov_matrix"], clf)

    # +validity = DPMDCE (term I only, no Psi)
    print("    [4c +validity=DPMDCE] ...")
    dp_cfs = run_dpmdce(data["X_test"], clf, model, Gamma,
                         n_instances=config.N_TEST_INSTANCES, device=device)
    results["+Validity"] = evaluate_all(dp_cfs, data["X_train"], data["y_train"],
                                        data["cov_matrix"], clf)

    # +manifold: reconstruction regularizer only.
    # Backward-compatible aliases lambda1/lambda2 are used here to zero out term I.
    print("    [4c +manifold] ...")
    manifold_cfs = generate_counterfactuals_batch(
        data["X_test"], model, Gamma, clf=clf,
        n_instances=config.N_TEST_INSTANCES, device=device,
        lambda1=0.0, lambda2=0.0, beta=1.0
    )
    results["+Manifold"] = evaluate_all(manifold_cfs, data["X_train"], data["y_train"],
                                        data["cov_matrix"], clf)

    # Full ECOer
    print("    [4c full ECOer] ...")
    ecoe_cfs = generate_counterfactuals_batch(
        data["X_test"], model, Gamma, clf=clf,
        n_instances=config.N_TEST_INSTANCES, device=device
    )
    results["Full\nECOer"] = evaluate_all(ecoe_cfs, data["X_train"], data["y_train"],
                                          data["cov_matrix"], clf)

    with open(cache, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else x)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run() -> None:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[Exp 4] Device: {device}")
    data = load_processed(DS_NAME)
    clf  = get_or_train_classifier(DS_NAME, "knn5",
                                   data["X_train"], data["y_train"])

    # (a) classifier choice
    print("[Exp 4a] Classifier choice ablation ...")
    clf_results = run_ablation_classifiers(data, device)

    # (b) architecture sweep
    print("[Exp 4b] Architecture sweep ...")
    arch_results = run_ablation_arch(data, clf, device)
    # Convert str keys to int for plotting
    arch_results_int = {int(k): v for k, v in arch_results.items()}

    # (c) reconstruction components
    print("[Exp 4c] Reconstruction components ...")
    model, Gamma = get_or_train_r2snn(DS_NAME, "knn5", clf,
                                       data["X_train"], device=device)
    reconstruction_results = run_ablation_reconstruction(data, clf, model, Gamma, device)

    # ── Figures ──
    print("[Exp 4] Generating figures ...")

    # Reshape clf_results for plotting: {clf_variant: {method: metrics}}
    # already in correct shape
    plot_exp4_classifiers(clf_results)

    # Arch sweep has convex vs non-convex; here we only show arch size
    # (convex relaxation runtime comparison baked into arch sweep)
    plot_exp4_components(arch_results_int, reconstruction_results)

    print("[Exp 4] Done.")


if __name__ == "__main__":
    run()
