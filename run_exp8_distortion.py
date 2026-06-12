"""Exp 8: direct metric-distortion diagnostic.

This experiment compares ECOer with and without the reconstruction regularizer
Psi.  The diagnostic measures the mismatch between hidden-space displacement
and decoded input-space displacement:

    Delta_gamma(e; x) = | ||gamma(e)-x||_2 - ||e-p(x)||_2 |.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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


RESULTS_DIR = os.path.join(config.RESULTS_DIR, "exp8_distortion")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _hidden(model, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X.astype(np.float32)).to(device)
        return model.get_features(xt).cpu().numpy()


def _distortion_stats(cfs: List[Dict], model, Gamma: np.ndarray, device: str) -> Dict:
    valid = [r for r in cfs if r is not None and r.get("valid", False)]
    if not valid:
        return {"n_valid": 0}

    x_in = np.asarray([r["x_in"] for r in valid], dtype=np.float32)
    x_cf = np.asarray([r["x_cf"] for r in valid], dtype=np.float32)
    e_cf = np.asarray([r["e_cf"] for r in valid], dtype=np.float32)
    e_in = _hidden(model, x_in, device)
    x_dec = e_cf @ Gamma.T

    dx = np.linalg.norm(x_dec - x_in, axis=1)
    de = np.linalg.norm(e_cf - e_in, axis=1)
    delta = np.abs(dx - de)
    ratio = dx / (de + 1e-8)
    rel_delta = delta / (dx + de + 1e-8)
    corr = float(np.corrcoef(dx, de)[0, 1]) if len(dx) > 1 and np.std(dx) > 0 and np.std(de) > 0 else float("nan")

    return {
        "n_valid": len(valid),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "ratio_mean": float(ratio.mean()),
        "ratio_std": float(ratio.std()),
        "relative_delta_mean": float(rel_delta.mean()),
        "relative_delta_std": float(rel_delta.std()),
        "input_dist_mean": float(dx.mean()),
        "hidden_dist_mean": float(de.mean()),
        "decoded_cf_gap_mean": float(np.linalg.norm(x_dec - x_cf, axis=1).mean()),
        "input_hidden_corr": corr,
    }


def _write_markdown(path: str, rows: List[Dict], columns: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |\n")


def run(datasets, clf_name, n_instances, epochs, n_uniform, n_boundary, m, device):
    raw = {}
    rows = []
    for ds_name in datasets:
        print(f"[Exp8/distortion] {ds_name}/{clf_name}")
        data = load_processed(ds_name)
        clf = get_or_train_classifier(ds_name, clf_name, data["X_train"], data["y_train"])
        model = train_r2snn(
            clf, data["X_train"], m=m, epochs=epochs, n_uniform=n_uniform,
            n_boundary=n_boundary, device=device, seed=config.SEED
        )
        Gamma = build_elm_reconstruction(model, data["X_train"], device=device)

        variants = {
            "ECOer": {"beta": config.BETA},
            "ECOer w/o Psi": {"beta": 0.0},
        }
        raw[ds_name] = {}
        for name, kwargs in variants.items():
            cfs = generate_counterfactuals_batch(
                data["X_test"], model, Gamma, clf=clf, n_instances=n_instances,
                seed=config.SEED, device=device, beta=kwargs["beta"]
            )
            metrics = evaluate_all(cfs, data["X_train"], data["y_train"], data["cov_matrix"], clf)
            dist = _distortion_stats(cfs, model, Gamma, device)
            raw[ds_name][name] = {"metrics": metrics, "distortion": dist}
            rows.append({
                "dataset": ds_name,
                "variant": name,
                "delta": round(dist.get("delta_mean", float("nan")), 4),
                "rel_delta": round(dist.get("relative_delta_mean", float("nan")), 4),
                "ratio": round(dist.get("ratio_mean", float("nan")), 4),
                "corr": round(dist.get("input_hidden_corr", float("nan")), 4),
                "validity": round(metrics.get("validity_rate", float("nan")), 3),
                "l1": round(metrics.get("l1_mean", float("nan")), 4),
                "dp": round(metrics.get("dp", float("nan")), 4),
                "im": round(metrics.get("im_mean", float("nan")), 4),
            })
    return raw, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["adult", "heloc", "german_credit", "compas"])
    parser.add_argument("--clf", default="knn5", choices=config.CLASSIFIERS)
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
    raw, rows = run(
        args.datasets, args.clf, args.n_instances, args.epochs,
        args.n_uniform, args.n_boundary, args.m, device
    )
    out = {"raw": raw, "summary_rows": rows, "settings": vars(args), "torch": {"version": torch.__version__, "device": device, "cuda_available": torch.cuda.is_available()}}
    json_path = os.path.join(RESULTS_DIR, "distortion_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_json_default)
    _write_markdown(
        os.path.join(RESULTS_DIR, "distortion_summary.md"),
        rows,
        ["dataset", "variant", "delta", "rel_delta", "ratio", "corr", "validity", "l1", "dp", "im"],
    )
    print(f"[Exp8] wrote {json_path}")


if __name__ == "__main__":
    main()
