"""
Exp 7: Numerical stability of the ELM reconstruction map.

The experiment evaluates ridge-regularized reconstruction maps
Gamma_eta = X^T E (E^T E + eta I)^{-1}, where E contains hidden activations.
The mapping remains linear, so convexity arguments that rely on linear gamma
are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.classifiers import get_or_train_classifier
from src.ecoe_optimizer import generate_counterfactuals_batch
from src.metrics import evaluate_all
from src.preprocessing import load_processed
from src.r2snn import train_r2snn


RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp7_stability")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _hidden_features(model, X_data: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_data.astype(np.float32)).to(device)
        feats = model.get_features(X_t).cpu().numpy()
    return feats.astype(np.float64)


def build_ridge_reconstruction(E: np.ndarray, X_data: np.ndarray, eta: float) -> np.ndarray:
    # Solve min_G ||E G^T - X||_F^2 + eta ||G||_F^2, returning Gamma=(d,m).
    m = E.shape[1]
    A = E.T @ E + eta * np.eye(m)
    B = E.T @ X_data.astype(np.float64)
    try:
        Gamma_T = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        Gamma_T = np.linalg.pinv(A) @ B
    return Gamma_T.T.astype(np.float32)


def condition_number(E: np.ndarray, eta: float) -> float:
    m = E.shape[1]
    A = E.T @ E + eta * np.eye(m)
    return float(np.linalg.cond(A))


def _write_markdown_table(path: str, rows: List[Dict], columns: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n")


def run_stability(
    datasets: List[str],
    clf_name: str,
    etas: List[float],
    n_instances: int,
    m: int,
    epochs: int,
    n_uniform: int,
    n_boundary: int,
    device: str,
) -> Dict:
    all_results = {}
    rows = []

    for ds_name in datasets:
        print(f"[Exp7] {ds_name}/{clf_name}")
        data = load_processed(ds_name)
        clf = get_or_train_classifier(ds_name, clf_name, data["X_train"], data["y_train"])
        model = train_r2snn(
            clf,
            data["X_train"],
            m=m,
            epochs=epochs,
            n_uniform=n_uniform,
            n_boundary=n_boundary,
            device=device,
            seed=config.SEED,
        )
        E = _hidden_features(model, data["X_train"], device)
        all_results[ds_name] = {}

        for eta in etas:
            print(f"  eta={eta:g}")
            gamma_start = time.perf_counter()
            Gamma = build_ridge_reconstruction(E, data["X_train"], eta)
            gamma_sec = time.perf_counter() - gamma_start
            cond = condition_number(E, eta)
            cfs = generate_counterfactuals_batch(
                data["X_test"],
                model,
                Gamma,
                clf=clf,
                n_instances=n_instances,
                seed=config.SEED,
                device=device,
            )
            metrics = evaluate_all(cfs, data["X_train"], data["y_train"], data["cov_matrix"], clf)
            all_results[ds_name][str(eta)] = {
                "condition_number": cond,
                "gamma_sec": gamma_sec,
                "metrics": metrics,
            }
            rows.append({
                "dataset": ds_name,
                "eta": eta,
                "cond": f"{cond:.3e}",
                "gamma_s": round(gamma_sec, 4),
                "online_ms": round(metrics.get("runtime_mean", float("nan")) * 1000, 3),
                "validity": round(metrics.get("validity_rate", float("nan")), 3),
                "l1": round(metrics.get("l1_mean", float("nan")), 4),
                "dp": round(metrics.get("dp", float("nan")), 4),
                "im": round(metrics.get("im_mean", float("nan")), 4),
            })

    return {"results": all_results, "summary_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "heloc"])
    parser.add_argument("--clf", default="knn5", choices=config.CLASSIFIERS)
    parser.add_argument("--etas", nargs="+", type=float, default=[0.0, 1e-8, 1e-6, 1e-4, 1e-2])
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-uniform", type=int, default=1000)
    parser.add_argument("--n-boundary", type=int, default=300)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"[Exp7] device={device}")

    out = run_stability(
        args.datasets, args.clf, args.etas, args.n_instances, args.m, args.epochs,
        args.n_uniform, args.n_boundary, device
    )
    payload = {
        "stability": out["results"],
        "summary": out["summary_rows"],
        "settings": vars(args),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "stability_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    _write_markdown_table(
        os.path.join(RESULTS_DIR, "stability_summary.md"),
        out["summary_rows"],
        ["dataset", "eta", "cond", "gamma_s", "online_ms", "validity", "l1", "dp", "im"],
    )
    print(f"[Exp7] wrote {json_path}")


if __name__ == "__main__":
    main()
