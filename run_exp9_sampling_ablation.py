"""Exp 9: sampling/training-pipeline ablation for the R2SNN surrogate.

The original codebase implements uniform sampling, boundary-aware resampling,
and geometry-aware regularization in a single surrogate-training routine.  This
script isolates those components by switching them on progressively.
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
from src.r2snn import build_elm_reconstruction, train_r2snn


RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp9_sampling_ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _agreement(model, clf, X: np.ndarray, device: str) -> float:
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.from_numpy(X.astype(np.float32)).to(device)).argmax(dim=1).cpu().numpy()
    pred_c = clf.predict(X)
    return float(np.mean(pred_s == pred_c))


def _write_markdown(path: str, rows: List[Dict], columns: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n")


def run(datasets, clf_name, n_instances, epochs, m, device):
    variants = {
        "Uniform only": {"n_uniform": 1000, "n_boundary": 0, "zeta1": 0.0, "zeta2": 0.0},
        "Boundary-aware": {"n_uniform": 1000, "n_boundary": 300, "zeta1": 0.0, "zeta2": 0.0},
        "Boundary+consistency": {"n_uniform": 1000, "n_boundary": 300, "zeta1": 0.0, "zeta2": config.R2SNN_ZETA2},
        "Full geometry-aware": {"n_uniform": 1000, "n_boundary": 300, "zeta1": config.R2SNN_ZETA1, "zeta2": config.R2SNN_ZETA2},
    }
    raw = {}
    rows = []
    for ds_name in datasets:
        print(f"[Exp9/sampling] {ds_name}/{clf_name}")
        data = load_processed(ds_name)
        clf = get_or_train_classifier(ds_name, clf_name, data["X_train"], data["y_train"])
        raw[ds_name] = {}
        for name, params in variants.items():
            print(f"  {name}")
            t0 = time.perf_counter()
            model = train_r2snn(
                clf, data["X_train"], m=m, epochs=epochs,
                n_uniform=params["n_uniform"], n_boundary=params["n_boundary"],
                zeta1=params["zeta1"], zeta2=params["zeta2"],
                device=device, seed=config.SEED
            )
            train_sec = time.perf_counter() - t0
            Gamma = build_elm_reconstruction(model, data["X_train"], device=device)
            agree = _agreement(model, clf, data["X_test"], device)
            cfs = generate_counterfactuals_batch(
                data["X_test"], model, Gamma, clf=clf, n_instances=n_instances,
                seed=config.SEED, device=device
            )
            metrics = evaluate_all(cfs, data["X_train"], data["y_train"], data["cov_matrix"], clf)
            raw[ds_name][name] = {
                "params": params,
                "train_sec": train_sec,
                "agreement": agree,
                "metrics": metrics,
            }
            rows.append({
                "dataset": ds_name,
                "variant": name,
                "agreement": round(agree, 4),
                "train_s": round(train_sec, 3),
                "online_ms": round(metrics.get("runtime_mean", float("nan")) * 1000, 3),
                "validity": round(metrics.get("validity_rate", float("nan")), 3),
                "l1": round(metrics.get("l1_mean", float("nan")), 4),
                "dp": round(metrics.get("dp", float("nan")), 4),
                "im": round(metrics.get("im_mean", float("nan")), 4),
            })
    return raw, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "heloc", "compas"])
    parser.add_argument("--clf", default="knn5", choices=config.CLASSIFIERS)
    parser.add_argument("--n-instances", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    raw, rows = run(args.datasets, args.clf, args.n_instances, args.epochs, args.m, device)
    out = {"raw": raw, "summary_rows": rows, "settings": vars(args), "torch": {"version": torch.__version__, "device": device, "cuda_available": torch.cuda.is_available()}}
    json_path = os.path.join(RESULTS_DIR, "sampling_ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    _write_markdown(
        os.path.join(RESULTS_DIR, "sampling_ablation_summary.md"),
        rows,
        ["dataset", "variant", "agreement", "train_s", "online_ms", "validity", "l1", "dp", "im"],
    )
    print(f"[Exp9] wrote {json_path}")


if __name__ == "__main__":
    main()
