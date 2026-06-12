"""
Exp 6: Runtime and online generation latency.

This script separates one-time offline costs (R2SNN fitting and Gamma
construction) from online counterfactual generation latency. It also measures
the sensitivity of ECOer to the hidden width m.
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
from src.baselines import (
    run_dice,
    run_dpmdce,
    run_face,
    run_growing_spheres,
    run_revise,
    run_wach,
)
from src.classifiers import get_or_train_classifier
from src.ecoe_optimizer import generate_counterfactuals_batch
from src.metrics import evaluate_all
from src.preprocessing import load_processed
from src.r2snn import build_elm_reconstruction, train_r2snn


RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp6_runtime")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _mean_runtime_ms(metrics: Dict) -> float:
    val = metrics.get("runtime_mean", float("nan"))
    return float(val) * 1000.0 if val == val else float("nan")


def _write_markdown_table(path: str, rows: List[Dict], columns: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n")


def _memory_mb(n_rows: int, d: int, m: int) -> Dict[str, float]:
    """Estimate reconstruction-map memory footprint for float32 Sigma/Gamma."""
    sigma_mb = n_rows * m * 4 / (1024 ** 2)
    gamma_mb = d * m * 4 / (1024 ** 2)
    normal_mb = m * m * 8 / (1024 ** 2)
    return {
        "sigma_mb": sigma_mb,
        "gamma_mb": gamma_mb,
        "normal_mb": normal_mb,
        "total_mb": sigma_mb + gamma_mb + normal_mb,
    }


def run_runtime_benchmark(
    datasets: List[str],
    clf_name: str,
    n_instances: int,
    m: int,
    epochs: int,
    n_uniform: int,
    n_boundary: int,
    device: str,
) -> Dict:
    results = {}
    summary_rows = []

    for ds_name in datasets:
        print(f"[Exp6/runtime] {ds_name}/{clf_name}")
        data = load_processed(ds_name)
        clf = get_or_train_classifier(ds_name, clf_name, data["X_train"], data["y_train"])

        fit_start = time.perf_counter()
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
        surrogate_fit_sec = time.perf_counter() - fit_start

        gamma_start = time.perf_counter()
        Gamma = build_elm_reconstruction(model, data["X_train"], device=device)
        gamma_sec = time.perf_counter() - gamma_start
        mem = _memory_mb(len(data["X_train"]), data["X_train"].shape[1], m)

        method_results = {}

        print("  ECOer")
        ecoe = generate_counterfactuals_batch(
            data["X_test"],
            model,
            Gamma,
            clf=clf,
            n_instances=n_instances,
            seed=config.SEED,
            device=device,
        )
        method_results["ECOer"] = evaluate_all(
            ecoe, data["X_train"], data["y_train"], data["cov_matrix"], clf
        )

        baseline_calls = {
            "DiCE": lambda: run_dice(
                data["X_train"], data["y_train"], data["X_test"], clf,
                n_instances=n_instances, seed=config.SEED
            ),
            "FACE": lambda: run_face(
                data["X_train"], data["y_train"], data["X_test"], clf,
                n_instances=n_instances, seed=config.SEED
            ),
            "GrowingSpheres": lambda: run_growing_spheres(
                data["X_test"], clf, n_instances=n_instances, seed=config.SEED
            ),
            "Revise": lambda: run_revise(
                data["X_train"], data["y_train"], data["X_test"], clf,
                n_instances=n_instances, seed=config.SEED, device=device
            ),
            "WACH": lambda: run_wach(
                data["X_test"], clf, model, n_instances=n_instances,
                seed=config.SEED, device=device
            ),
            "DPMDCE": lambda: run_dpmdce(
                data["X_test"], clf, model, Gamma, n_instances=n_instances,
                seed=config.SEED, device=device
            ),
        }

        for method, fn in baseline_calls.items():
            print(f"  {method}")
            try:
                cfs = fn()
                method_results[method] = evaluate_all(
                    cfs, data["X_train"], data["y_train"], data["cov_matrix"], clf
                )
            except Exception as exc:
                method_results[method] = {"error": repr(exc)}

        results[ds_name] = {
            "offline": {
                "surrogate_fit_sec": surrogate_fit_sec,
                "gamma_construction_sec": gamma_sec,
                "m": m,
                "epochs": epochs,
                "n_instances": n_instances,
                "device": device,
                "memory_mb": mem,
            },
            "methods": method_results,
        }

        for method, metrics in method_results.items():
            summary_rows.append({
                "dataset": ds_name,
                "method": method,
                "offline_fit_s": round(surrogate_fit_sec, 3) if method == "ECOer" else "-",
                "gamma_s": round(gamma_sec, 3) if method in ("ECOer", "DPMDCE") else "-",
                "recon_mem_mb": round(mem["total_mb"], 3) if method in ("ECOer", "DPMDCE") else "-",
                "online_ms": round(_mean_runtime_ms(metrics), 3) if "error" not in metrics else "ERR",
                "total_s": round(metrics.get("runtime_mean", float("nan")) * n_instances, 3) if "error" not in metrics else "ERR",
                "validity": round(metrics.get("validity_rate", float("nan")), 3) if "error" not in metrics else "ERR",
                "l1": round(metrics.get("l1_mean", float("nan")), 4) if "error" not in metrics else "ERR",
            })

    return {"results": results, "summary_rows": summary_rows}


def run_cached_online_summary(datasets: List[str], clf_name: str) -> Dict:
    """Build the runtime table from existing Exp. 2 per-method runtime caches."""
    exp2_dir = os.path.join(config.RESULTS_DIR, "exp2_proximity")
    results = {}
    summary_rows = []
    for ds_name in datasets:
        path = os.path.join(exp2_dir, f"{ds_name}_{clf_name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing cached Exp. 2 result: {path}")
        with open(path, "r", encoding="utf-8") as f:
            method_results = json.load(f)
        results[ds_name] = {"methods": method_results}
        for method, metrics in method_results.items():
            summary_rows.append({
                "dataset": ds_name,
                "method": method,
                "offline_fit_s": "cached",
                "gamma_s": "cached" if method in ("ECOer", "DPMDCE") else "-",
                "recon_mem_mb": "cached" if method in ("ECOer", "DPMDCE") else "-",
                "online_ms": round(_mean_runtime_ms(metrics), 3),
                "total_s": round(metrics.get("runtime_mean", float("nan")) * metrics.get("n_valid", 0), 3),
                "validity": round(metrics.get("validity_rate", float("nan")), 3),
                "l1": round(metrics.get("l1_mean", float("nan")), 4),
            })
    return {"results": results, "summary_rows": summary_rows}


def run_m_sensitivity(
    dataset: str,
    clf_name: str,
    m_values: List[int],
    n_instances: int,
    epochs: int,
    n_uniform: int,
    n_boundary: int,
    device: str,
) -> Dict:
    data = load_processed(dataset)
    clf = get_or_train_classifier(dataset, clf_name, data["X_train"], data["y_train"])
    rows = []
    raw = {}

    for m in m_values:
        print(f"[Exp6/m] {dataset}/{clf_name}/m={m}")
        fit_start = time.perf_counter()
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
        fit_sec = time.perf_counter() - fit_start

        gamma_start = time.perf_counter()
        Gamma = build_elm_reconstruction(model, data["X_train"], device=device)
        gamma_sec = time.perf_counter() - gamma_start
        mem = _memory_mb(len(data["X_train"]), data["X_train"].shape[1], m)

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
        raw[str(m)] = {"fit_sec": fit_sec, "gamma_sec": gamma_sec, "memory_mb": mem, "metrics": metrics}
        rows.append({
            "m": m,
            "fit_s": round(fit_sec, 3),
            "gamma_s": round(gamma_sec, 3),
            "recon_mem_mb": round(mem["total_mb"], 3),
            "online_ms": round(_mean_runtime_ms(metrics), 3),
            "total_s": round(metrics.get("runtime_mean", float("nan")) * n_instances, 3),
            "validity": round(metrics.get("validity_rate", float("nan")), 3),
            "l1": round(metrics.get("l1_mean", float("nan")), 4),
        })

    return {"raw": raw, "summary_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "heloc", "german_credit", "compas"])
    parser.add_argument("--clf", default="knn5", choices=config.CLASSIFIERS)
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-uniform", type=int, default=1000)
    parser.add_argument("--n-boundary", type=int, default=300)
    parser.add_argument("--m-values", nargs="+", type=int, default=[10, 20, 30, 50, 100])
    parser.add_argument("--m-dataset", default="adult")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rerun-methods", action="store_true",
                        help="Rerun ECOer and all baselines instead of using cached Exp. 2 online runtimes.")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"[Exp6] device={device}")

    if args.rerun_methods:
        runtime = run_runtime_benchmark(
            args.datasets, args.clf, args.n_instances, args.m, args.epochs,
            args.n_uniform, args.n_boundary, device
        )
    else:
        print("[Exp6] using cached Exp. 2 online runtime metrics; pass --rerun-methods to rerun baselines")
        runtime = run_cached_online_summary(args.datasets, args.clf)
    sensitivity = run_m_sensitivity(
        args.m_dataset, args.clf, args.m_values, args.n_instances, args.epochs,
        args.n_uniform, args.n_boundary, device
    )

    out = {
        "runtime": runtime["results"],
        "runtime_summary": runtime["summary_rows"],
        "m_sensitivity": sensitivity["raw"],
        "m_sensitivity_summary": sensitivity["summary_rows"],
        "settings": vars(args),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
        },
    }
    json_path = os.path.join(RESULTS_DIR, "runtime_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)

    _write_markdown_table(
        os.path.join(RESULTS_DIR, "runtime_summary.md"),
        runtime["summary_rows"],
        ["dataset", "method", "offline_fit_s", "gamma_s", "recon_mem_mb", "online_ms", "total_s", "validity", "l1"],
    )
    _write_markdown_table(
        os.path.join(RESULTS_DIR, "m_sensitivity_summary.md"),
        sensitivity["summary_rows"],
        ["m", "fit_s", "gamma_s", "recon_mem_mb", "online_ms", "total_s", "validity", "l1"],
    )
    print(f"[Exp6] wrote {json_path}")


if __name__ == "__main__":
    main()
