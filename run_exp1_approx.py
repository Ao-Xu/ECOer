"""
Experiment 1: R2SNN approximation quality vs. SingleReLU.
Reproduces comparison.png for all 6 datasets (KNN target classifier).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import config
from src.preprocessing import load_processed
from src.classifiers import get_or_train_classifier
from src.r2snn import eval_r2snn_vs_single_relu
from src.plotting import plot_exp1_approx

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp1_approx")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run() -> None:
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[Exp 1] Device: {device}")
    results_all = {}

    for ds_name in config.DATASETS:
        cache_path = os.path.join(RESULTS_DIR, f"{ds_name}_knn5.json")
        if os.path.exists(cache_path):
            print(f"  [{ds_name}] loading cached results")
            with open(cache_path) as f:
                raw = json.load(f)
            # Rebuild numpy arrays from lists
            res = {k: np.array(v) if isinstance(v, list) else v
                   for k, v in raw.items()}
            results_all[ds_name] = res
            continue

        print(f"  [{ds_name}] running R2SNN sweep ...")
        data = load_processed(ds_name)
        clf  = get_or_train_classifier(ds_name, "knn5",
                                       data["X_train"], data["y_train"])
        # Cap training data at 8000 samples for sweep speed (Exp1 only)
        X_tr = data["X_train"]
        if len(X_tr) > 8000:
            rng = np.random.default_rng(config.SEED)
            idx = rng.choice(len(X_tr), 8000, replace=False)
            X_tr = X_tr[idx]
        res  = eval_r2snn_vs_single_relu(
            clf, X_tr, data["X_test"],
            m_values=config.M_VALUES, n_seeds=config.N_SEEDS, device=device
        )
        # Save (convert arrays to lists for JSON)
        to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v
                   for k, v in res.items()}
        with open(cache_path, "w") as f:
            json.dump(to_save, f)
        results_all[ds_name] = res
        print(f"  [{ds_name}] done")

    print("[Exp 1] Generating comparison.png ...")
    plot_exp1_approx(results_all, m_values=config.M_VALUES)
    print("[Exp 1] Done.")


if __name__ == "__main__":
    run()
