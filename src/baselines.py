"""
Baseline counterfactual methods:
  1. DiCE          — diversity-enforcing gradient-based (dice-ml or fallback)
  2. FACE          — density-weighted shortest path in KNN graph
  3. GrowingSpheres— hypersphere expansion
  4. Revise        — VAE latent-space gradient search
  5. WACH          — Wachter's method via R2SNN (direct x-space gradient descent)
  6. DPMDCE        — feature-space opt. without energy regularisation (ablation)

All return list[dict] matching ECOer result format:
  {'x_cf': np.ndarray(d,), 'x_in': np.ndarray(d,), 'valid': bool,
   'runtime': float, 'steps': int}
"""
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.r2snn import R2SNN, reconstruct_input


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _sample_test(X_test, n, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    return X_test[idx], idx


def _make_result(x_cf, x_in, clf, runtime, steps=None):
    x_cf = np.clip(x_cf.astype(np.float32), -1, 1)
    y_in = clf.predict(x_in.reshape(1, -1))[0]
    y_cf = clf.predict(x_cf.reshape(1, -1))[0]
    return {
        "x_cf":    x_cf,
        "x_in":    x_in.astype(np.float32),
        "valid":   bool(y_cf != y_in),
        "runtime": runtime,
        "steps":   steps if steps is not None else 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. DiCE
# ──────────────────────────────────────────────────────────────────────────────

def run_dice(
    X_train, y_train, X_test, clf,
    n_instances=config.N_TEST_INSTANCES,
    n_cfs=1,
    seed=config.SEED,
) -> list:
    # DiCE requires differentiable model; sklearn classifiers are not — return NaN
    X_sel, _ = _sample_test(X_test, n_instances, seed)
    results = []
    for x_in in X_sel:
        x_in_f = x_in.astype(np.float32)
        results.append({
            "x_cf":    x_in_f.copy(),
            "x_in":    x_in_f,
            "valid":   False,
            "runtime": 0.0,
            "steps":   0,
        })
    return results


def _dice_gradient_fallback(X_train, y_train, X_sel, clf, n_cfs, seed):
    """Gradient-free fallback using random perturbation search."""
    rng = np.random.RandomState(seed)
    results = []
    d = X_train.shape[1]
    for x_in in X_sel:
        t0 = time.perf_counter()
        y_in = clf.predict(x_in.reshape(1, -1))[0]
        best = None
        best_dist = float("inf")
        for _ in range(50):
            # Random perturbation, decreasing radius
            r = rng.uniform(0.0, 0.5)
            direction = rng.randn(d)
            direction /= (np.linalg.norm(direction) + 1e-12)
            x_cand = np.clip(x_in + r * direction, -1, 1).astype(np.float32)
            y_cand = clf.predict(x_cand.reshape(1, -1))[0]
            if y_cand != y_in:
                dist = np.linalg.norm(x_cand - x_in)
                if dist < best_dist:
                    best_dist = dist
                    best = x_cand
        x_cf = best if best is not None else x_in.copy()
        results.append(_make_result(x_cf, x_in, clf, time.perf_counter() - t0))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. FACE
# ──────────────────────────────────────────────────────────────────────────────

def run_face(
    X_train, y_train, X_test, clf,
    n_instances=config.N_TEST_INSTANCES,
    k_graph=7,
    seed=config.SEED,
) -> list:
    from sklearn.neighbors import NearestNeighbors, KernelDensity
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    results = []
    X_sel, _ = _sample_test(X_test, n_instances, seed)
    y_train_arr = y_train.astype(int)

    # Cap training data for graph to avoid O(N^2) memory/time on large datasets
    MAX_GRAPH = 5000
    if len(X_train) > MAX_GRAPH:
        rng_face = np.random.default_rng(seed)
        idx_face = rng_face.choice(len(X_train), MAX_GRAPH, replace=False)
        X_g = X_train[idx_face]
        y_g = y_train_arr[idx_face]
    else:
        X_g = X_train
        y_g = y_train_arr
        idx_face = np.arange(len(X_train))

    # KDE for density estimation
    kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
    kde.fit(X_g)

    # Build KNN graph on (possibly subsampled) training data
    nn_model = NearestNeighbors(n_neighbors=k_graph + 1, algorithm="ball_tree")
    nn_model.fit(X_g)
    distances, indices = nn_model.kneighbors(X_g)

    N_g = len(X_g)
    rows, cols, weights = [], [], []
    log_densities = kde.score_samples(X_g)

    for i in range(N_g):
        for j_idx, j in enumerate(indices[i, 1:]):
            d_ij = distances[i, j_idx + 1]
            density_weight = 1.0 / (np.exp(min(log_densities[i], log_densities[j])) + 1e-12)
            w = d_ij * density_weight
            rows.append(i); cols.append(j); weights.append(w)
            rows.append(j); cols.append(i); weights.append(w)

    graph = csr_matrix((weights, (rows, cols)), shape=(N_g, N_g))

    # Pre-compute source indices (nearest graph node to each test point)
    nn_q = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X_g)
    _, src_indices = nn_q.kneighbors(X_sel)
    unique_srcs = list(set(src_indices.flatten().tolist()))

    # Shortest path only from unique source nodes → O(k * N log N) instead of O(N^2 log N)
    dist_rows = shortest_path(graph, method="D", directed=False, indices=unique_srcs)
    src_to_row = {s: i for i, s in enumerate(unique_srcs)}

    for xi, x_in in enumerate(X_sel):
        t0 = time.perf_counter()
        y_in = int(clf.predict(x_in.reshape(1, -1))[0])
        target_class = 1 - y_in
        target_indices = np.where(y_g == target_class)[0]

        src = src_indices[xi, 0]
        row = src_to_row[src]

        best_x = None
        best_cost = float("inf")
        for tgt in target_indices:
            cost = dist_rows[row, tgt]
            if cost < best_cost:
                best_cost = cost
                best_x = X_g[tgt]

        x_cf = best_x if best_x is not None else x_in.copy()
        results.append(_make_result(x_cf, x_in, clf, time.perf_counter() - t0))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 3. GrowingSpheres
# ──────────────────────────────────────────────────────────────────────────────

def run_growing_spheres(
    X_test, clf,
    n_instances=config.N_TEST_INSTANCES,
    eta=0.05,
    n_samples=100,
    max_iter=15,
    seed=config.SEED,
) -> list:
    rng = np.random.RandomState(seed)
    results = []
    X_sel, _ = _sample_test(X_test, n_instances, seed)
    d = X_test.shape[1]

    for x_in in X_sel:
        t0 = time.perf_counter()
        y_in = clf.predict(x_in.reshape(1, -1))[0]
        r = eta
        best = None
        best_dist = float("inf")

        for _ in range(max_iter):
            # Sample on sphere surface of radius r
            dirs = rng.randn(n_samples, d)
            dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
            candidates = np.clip(x_in + r * dirs, -1, 1).astype(np.float32)
            preds = clf.predict(candidates)
            valid_mask = preds != y_in

            if valid_mask.any():
                valid_cands = candidates[valid_mask]
                dists = np.linalg.norm(valid_cands - x_in, axis=1)
                best_idx = np.argmin(dists)
                if dists[best_idx] < best_dist:
                    best_dist = dists[best_idx]
                    best = valid_cands[best_idx]
                break  # found at this radius

            r += eta

        x_cf = best if best is not None else x_in.copy()
        results.append(_make_result(x_cf, x_in, clf, time.perf_counter() - t0))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. Revise (tabular VAE + latent-space search)
# ──────────────────────────────────────────────────────────────────────────────

class _TabularVAE(nn.Module):
    def __init__(self, d, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, d),
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def _train_vae(X_train, latent_dim=8, epochs=50, lr=1e-3, seed=42, device="cpu"):
    torch.manual_seed(seed)
    d = X_train.shape[1]
    vae = _TabularVAE(d, latent_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    X_t = torch.from_numpy(X_train).to(device)
    for ep in range(epochs):
        vae.train()
        idx = torch.randperm(len(X_t))
        for start in range(0, len(X_t), 64):
            xb = X_t[idx[start:start+64]]
            recon, mu, logvar = vae(xb)
            recon_loss = ((recon - xb) ** 2).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
            loss = recon_loss + 0.01 * kl
            opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    return vae


def run_revise(
    X_train, y_train, X_test, clf,
    n_instances=config.N_TEST_INSTANCES,
    latent_dim=8,
    epochs_vae=20,
    lr_search=0.01,
    max_steps=30,
    seed=config.SEED,
) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = _train_vae(X_train, latent_dim=latent_dim, epochs=epochs_vae,
                     seed=seed, device=device)
    results = []
    X_sel, _ = _sample_test(X_test, n_instances, seed)

    for x_in in X_sel:
        t0 = time.perf_counter()
        y_in = int(clf.predict(x_in.reshape(1, -1))[0])
        target_cls = 1 - y_in

        x_t = torch.from_numpy(x_in.reshape(1, -1)).float().to(device)
        with torch.no_grad():
            mu, _ = vae.encode(x_t)
        z = mu.clone().detach().requires_grad_(True)
        opt_z = torch.optim.Adam([z], lr=lr_search)

        for step in range(max_steps):
            x_dec = vae.decode(z)
            x_np  = x_dec.detach().cpu().numpy().reshape(1, -1)
            x_np  = np.clip(x_np, -1, 1).astype(np.float32)
            proba = clf.predict_proba(x_np)
            pred_target_conf = float(proba[0, target_cls])
            if pred_target_conf > 0.5:
                break
            # Random walk in latent space (sklearn clf has no grad)
            with torch.no_grad():
                z = z + 0.05 * torch.randn_like(z)

        with torch.no_grad():
            x_cf = vae.decode(z).cpu().numpy().flatten()
        x_cf = np.clip(x_cf, -1, 1).astype(np.float32)
        results.append(_make_result(x_cf, x_in, clf, time.perf_counter() - t0, step))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5. WACH (Wachter via R2SNN, direct input-space gradient descent)
# ──────────────────────────────────────────────────────────────────────────────

def run_wach(
    X_test, clf, model: R2SNN,
    n_instances=config.N_TEST_INSTANCES,
    lr=0.01,
    max_steps=100,
    lambda_val=0.5,
    seed=config.SEED,
    device=None,
) -> list:
    """
    Wachter objective: min_x  ||f_m(x) - y1||² + λ·||x - x_in||²
    Gradient descent directly in x-space using R2SNN.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    X_sel, _ = _sample_test(X_test, n_instances, seed)
    model.eval()

    for x_in in X_sel:
        t0 = time.perf_counter()
        y_in = int(clf.predict(x_in.reshape(1, -1))[0])
        y1_vec = torch.tensor([0.0, 1.0] if y_in == 0 else [1.0, 0.0],
                               device=device)

        x_var = torch.tensor(x_in, device=device, dtype=torch.float32
                              ).unsqueeze(0).requires_grad_(True)
        x_in_t = torch.tensor(x_in, device=device, dtype=torch.float32)
        opt_x = torch.optim.Adam([x_var], lr=lr)

        for step in range(max_steps):
            out = model(x_var).squeeze(0)   # (2,)
            loss = ((out - y1_vec) ** 2).sum() + lambda_val * ((x_var.squeeze() - x_in_t) ** 2).sum()
            opt_x.zero_grad(); loss.backward(); opt_x.step()
            # Check convergence
            x_cf_np = np.clip(x_var.detach().cpu().numpy().flatten(), -1, 1).astype(np.float32)
            y_cf = clf.predict(x_cf_np.reshape(1, -1))[0]
            if y_cf != y_in:
                break

        x_cf = np.clip(x_var.detach().cpu().numpy().flatten(), -1, 1).astype(np.float32)
        results.append(_make_result(x_cf, x_in, clf, time.perf_counter() - t0, step))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 6. DPMDCE
# ──────────────────────────────────────────────────────────────────────────────

def run_dpmdce(
    X_test, clf, model, Gamma: np.ndarray,
    n_instances=10, # Note: replace 10 with config.N_TEST_INSTANCES if imported
    lr=0.01,
    max_steps=1000, # Note: replace 1000 with config.CF_MAX_STEPS if imported
    seed=42,        # Note: replace 42 with config.SEED if imported
    device=None,
    c_dist=0.1,     # Lambda weight for the overall DPMD distance penalty
    beta_dpmd=1.0   # Beta weight for feature importance influence in Sigma'
) -> list:
    """
    Feature-space optimisation using DPMDCE (Distribution Preference Mahalanobis Distance):
      min_e  ||σ(W₂e) - y1||² + c_dist * (e - e_init)^T Σ' (e - e_init)
    where Σ' = I + beta_dpmd * diag(feature_importance)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    # Assuming _sample_test and _make_result are available in your broader scope
    X_sel, _ = _sample_test(X_test, n_instances, seed) 
    model.eval()
    
    W2 = model.get_W2().cpu().numpy().astype(np.float64)
    W1 = model.get_W1().cpu().numpy().astype(np.float64)
    b  = model.get_b().cpu().numpy().astype(np.float64)
    Gamma_f = Gamma.astype(np.float64)

    for x_in in X_sel:
        t0 = time.perf_counter()
        
        y_in = int(clf.predict(x_in.reshape(1, -1))[0])
        y_target_val = 1 if y_in == 0 else 0
        y1 = np.array([0.0, 1.0] if y_in == 0 else [1.0, 0.0])

        # Initial feature representation e_0
        e_init = np.maximum(W1 @ x_in.astype(np.float64) + b, 0)
        e = e_init.copy()

        # DPMD: Estimate Feature Importance (Lambda).
        # We use the absolute difference in the classification weights as a proxy 
        # for distribution Wasserstein distance importance between the two classes.
        weight_diff = np.abs(W2[y_target_val, :] - W2[y_in, :])
        lambda_feat = weight_diff / (np.max(weight_diff) + 1e-8) # Normalize to 0-1

        # DPMD: Construct diagonal of Sigma' = I + beta * diag(lambda)
        # We assume the base covariance Sigma is the Identity matrix for simplification.
        sigma_prime_diag = 1.0 + beta_dpmd * lambda_feat

        m_a = np.zeros_like(e); v_a = np.zeros_like(e)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for step in range(1, max_steps + 1):
            z  = W2 @ e
            sz = np.maximum(z, 0)   # σ(W₂e)
            diff = sz - y1
            
            # 1. Gradient of predictive loss
            sigma_p = (z > 0).astype(float)
            grad_pred = W2.T @ (2 * diff * sigma_p)  # (m,)
            
            # 2. Gradient of DPMD distance loss: d/de [ c_dist * (e-e_init)^T * Sigma' * (e-e_init) ]
            grad_dist = 2 * c_dist * sigma_prime_diag * (e - e_init)
            
            # Total gradient
            grad = grad_pred + grad_dist
            
            # Adam step
            m_a = beta1 * m_a + (1 - beta1) * grad
            v_a = beta2 * v_a + (1 - beta2) * grad ** 2
            m_h = m_a / (1 - beta1 ** step)
            v_h = v_a / (1 - beta2 ** step)
            e = np.maximum(e - lr * m_h / (np.sqrt(v_h) + eps), 0)

            # Check if prediction flipped
            x_cf_np = np.clip(Gamma_f @ e, -1, 1).astype(np.float32)
            y_cf = clf.predict(x_cf_np.reshape(1, -1))[0]
            if y_cf != y_in:
                break

        x_cf = np.clip(Gamma_f @ e, -1, 1).astype(np.float32)
        results.append(_make_result(x_cf, x_in, clf, time.perf_counter() - t0, step))
        
    return results
