"""Exp 10: small-scale tree-specific RF sanity baseline.

This is not an exact MIP implementation such as OCEAN/FOCUS.  It is a
transparent tree-specific feasibility baseline: for each query, choose a
training instance that the random forest predicts as the target class and whose
RF leaf-code is closest to the query, breaking ties by input-space L1 distance.
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


RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp10_rf_specialized")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def rf_leaf_target_nn(X_test, X_train, rf, n_instances, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_test), size=min(n_instances, len(X_test)), replace=False)
    train_pred = rf.predict(X_train)
    train_leaf = rf.apply(X_train)
    results = []
    for i in idx:
        x = X_test[i]
        y = int(rf.predict(x.reshape(1, -1))[0])
        target = 1 - y
        cand_idx = np.where(train_pred == target)[0]
        t0 = time.perf_counter()
        if len(cand_idx) == 0:
            results.append({"x_cf": x.copy(), "x_in": x, "valid": False, "runtime": time.perf_counter() - t0, "steps": 0})
            continue
        leaf = rf.apply(x.reshape(1, -1))[0]
        cand_leaf = train_leaf[cand_idx]
        leaf_dist = np.mean(cand_leaf != leaf, axis=1)
        l1 = np.sum(np.abs(X_train[cand_idx] - x), axis=1)
        # Lexicographic preference: nearby RF path first, then input proximity.
        best_local = np.lexsort((l1, leaf_dist))[0]
        best = cand_idx[best_local]
        x_cf = X_train[best].astype(np.float32)
        valid = bool(rf.predict(x_cf.reshape(1, -1))[0] != y)
        results.append({
            "x_cf": x_cf,
            "x_in": x,
            "valid": valid,
            "runtime": time.perf_counter() - t0,
            "steps": int(leaf_dist[best_local] * rf.n_estimators),
        })
    return results


def _write_markdown(path: str, rows: List[Dict], columns: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n")


def run(datasets, n_instances, epochs, n_uniform, n_boundary, m, device):
    raw = {}
    rows = []
    for ds_name in datasets:
        print(f"[Exp10/RF] {ds_name}")
        data = load_processed(ds_name)
        rf = get_or_train_classifier(ds_name, "rf", data["X_train"], data["y_train"])

        model = train_r2snn(
            rf, data["X_train"], m=m, epochs=epochs,
            n_uniform=n_uniform, n_boundary=n_boundary,
            device=device, seed=config.SEED
        )
        Gamma = build_elm_reconstruction(model, data["X_train"], device=device)

        ecoe_cfs = generate_counterfactuals_batch(
            data["X_test"], model, Gamma, clf=rf, n_instances=n_instances,
            seed=config.SEED, device=device
        )
        leaf_cfs = rf_leaf_target_nn(data["X_test"], data["X_train"], rf, n_instances, config.SEED)

        methods = {
            "ECOer-RF": ecoe_cfs,
            "RF-LeafTargetNN": leaf_cfs,
        }
        raw[ds_name] = {}
        for method, cfs in methods.items():
            metrics = evaluate_all(cfs, data["X_train"], data["y_train"], data["cov_matrix"], rf)
            raw[ds_name][method] = metrics
            rows.append({
                "dataset": ds_name,
                "method": method,
                "online_ms": round(metrics.get("runtime_mean", float("nan")) * 1000, 3),
                "validity": round(metrics.get("validity_rate", float("nan")), 3),
                "l1": round(metrics.get("l1_mean", float("nan")), 4),
                "l2": round(metrics.get("l2_mean", float("nan")), 4),
                "dp": round(metrics.get("dp", float("nan")), 4),
                "im": round(metrics.get("im_mean", float("nan")), 4),
            })
    return raw, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "german_credit", "compas"])
    parser.add_argument("--n-instances", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-uniform", type=int, default=1000)
    parser.add_argument("--n-boundary", type=int, default=300)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    raw, rows = run(args.datasets, args.n_instances, args.epochs, args.n_uniform, args.n_boundary, args.m, device)
    out = {"raw": raw, "summary_rows": rows, "settings": vars(args), "torch": {"version": torch.__version__, "device": device, "cuda_available": torch.cuda.is_available()}}
    json_path = os.path.join(RESULTS_DIR, "rf_specialized_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    _write_markdown(
        os.path.join(RESULTS_DIR, "rf_specialized_summary.md"),
        rows,
        ["dataset", "method", "online_ms", "validity", "l1", "l2", "dp", "im"],
    )
    print(f"[Exp10] wrote {json_path}")


if __name__ == "__main__":
    main()
