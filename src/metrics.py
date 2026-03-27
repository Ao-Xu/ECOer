"""
Evaluation metrics for counterfactual explanations:
  ℓ₁, ℓ₂, Sparsity, DP (Discriminative Power), IM (Implausibility/Mahalanobis)
"""
import sys
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def l1_distance(x_cf: np.ndarray, x_in: np.ndarray) -> float:
    return float(np.sum(np.abs(x_cf - x_in)))


def l2_distance(x_cf: np.ndarray, x_in: np.ndarray) -> float:
    return float(np.linalg.norm(x_cf - x_in))


def sparsity(x_cf: np.ndarray, x_in: np.ndarray, threshold: float = 1e-4) -> int:
    """Number of features with |x_cf[i] - x_in[i]| > threshold."""
    return int(np.sum(np.abs(x_cf - x_in) > threshold))


def discriminative_power(
    x_cfs: np.ndarray,      # (N, d) valid counterfactuals
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_class: int = 1,
) -> float:
    """
    DP: Train 1-NN on X_train; evaluate how well it classifies x_cfs as target_class.
    DP = accuracy of 1-NN when evaluated on x_cfs with assumed label = target_class.
    Higher DP → CFs are more representative of the target class.
    """
    if len(x_cfs) == 0:
        return float("nan")
    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(X_train, y_train)
    preds = nn.predict(x_cfs)
    return float(np.mean(preds == target_class))


def implausibility(
    x_cfs: np.ndarray,     # (N, d)
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_class: int = 1,
    cov_matrix: np.ndarray = None,
) -> float:
    """
    IM: Mean Mahalanobis distance from each x_cf to the centroid of target-class
    training instances. Lower IM → more plausible (closer to real data manifold).
    """
    if len(x_cfs) == 0:
        return float("nan")
    target_mask = y_train == target_class
    X_target = X_train[target_mask]
    if len(X_target) == 0:
        return float("nan")

    centroid = X_target.mean(axis=0)

    if cov_matrix is None:
        cov_matrix = np.cov(X_target.T)

    # Regularise covariance for numerical stability
    d = cov_matrix.shape[0]
    cov_reg = cov_matrix + 1e-5 * np.eye(d)
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(d)

    distances = []
    for x_cf in x_cfs:
        diff = x_cf - centroid
        mah = float(np.sqrt(max(0.0, diff @ cov_inv @ diff)))
        distances.append(mah)
    return float(np.mean(distances))


def evaluate_all(
    cf_results: list,       # list of result dicts from ECOer/baseline
    X_train: np.ndarray,
    y_train: np.ndarray,
    cov_matrix: np.ndarray,
    clf=None,
    target_class: int = 1,
) -> dict:
    """
    Aggregate all metrics over valid counterfactuals.

    Parameters
    ----------
    cf_results : list of {'x_cf', 'x_in', 'valid', 'runtime', 'steps'}

    Returns
    -------
    dict with mean/std for ℓ₁, ℓ₂, sparsity, DP, IM, runtime, + validity_rate
    """
    valid = [r for r in cf_results if r is not None and r.get("valid", False)]
    validity_rate = len(valid) / max(len(cf_results), 1)

    if not valid:
        nan = float("nan")
        return {
            "l1_mean": nan, "l1_std": nan,
            "l2_mean": nan, "l2_std": nan,
            "sparsity_mean": nan, "sparsity_std": nan,
            "dp": nan,
            "im_mean": nan, "im_std": nan,
            "runtime_mean": nan, "runtime_std": nan,
            "validity_rate": 0.0,
            "n_valid": 0,
        }

    x_cfs = np.array([r["x_cf"] for r in valid])
    x_ins = np.array([r["x_in"] for r in valid])
    runtimes = np.array([r["runtime"] for r in valid])

    l1s = np.array([l1_distance(cf, inp) for cf, inp in zip(x_cfs, x_ins)])
    l2s = np.array([l2_distance(cf, inp) for cf, inp in zip(x_cfs, x_ins)])
    sps = np.array([sparsity(cf, inp) for cf, inp in zip(x_cfs, x_ins)])

    dp = discriminative_power(x_cfs, X_train, y_train, target_class)
    im = implausibility(x_cfs, X_train, y_train, target_class, cov_matrix)

    return {
        "l1_mean":       float(l1s.mean()),
        "l1_std":        float(l1s.std()),
        "l2_mean":       float(l2s.mean()),
        "l2_std":        float(l2s.std()),
        "sparsity_mean": float(sps.mean()),
        "sparsity_std":  float(sps.std()),
        "dp":            dp,
        "im_mean":       float(np.nanmean([im])),  # already scalar
        "im_std":        0.0,  # IM is aggregated globally
        "runtime_mean":  float(runtimes.mean()),
        "runtime_std":   float(runtimes.std()),
        "validity_rate": validity_rate,
        "n_valid":       len(valid),
        # Store raw arrays for statistical tests
        "_l1_raw":       l1s.tolist(),
        "_l2_raw":       l2s.tolist(),
        "_sparsity_raw": sps.tolist(),
        "_runtime_raw":  runtimes.tolist(),
    }
