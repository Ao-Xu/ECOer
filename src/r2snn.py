"""
R2SNN: ReLU-ReLU Single-hidden-layer Neural Network
  f_m(x) = σ(W₂ · σ(W₁x + b))
  W₁ ∈ R^{m×d}, b ∈ R^m, W₂ ∈ R^{2×m}

Also contains:
  - train_r2snn()         : training with L_approx + R_grad + R_cons
  - build_elm_reconstruction() : ELM pseudoinverse mapping Γ: E→X
  - SingleReLU            : ablation baseline (single ReLU activation)
  - eval_r2snn_vs_single_relu() : Exp 1 comparison
"""
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SURR_DIR = os.path.join(config.MODELS_DIR, "surrogates")
os.makedirs(SURR_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Network architectures
# ──────────────────────────────────────────────────────────────────────────────

class R2SNN(nn.Module):
    """
    Dual-ReLU shallow network: f_m(x) = ReLU(W₂ · ReLU(W₁x + b))
    Output shape: (N, 2)  — soft probability-like scores for 2 classes.
    """
    def __init__(self, d: int, m: int = 30):
        super().__init__()
        self.d = d
        self.m = m
        self.W1 = nn.Linear(d, m, bias=True)
        self.W2 = nn.Linear(m, 2, bias=False)
        # He initialisation
        nn.init.kaiming_normal_(self.W1.weight, nonlinearity="relu")
        nn.init.zeros_(self.W1.bias)
        nn.init.kaiming_normal_(self.W2.weight, nonlinearity="relu")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden activations: σ(W₁x + b), shape (N, m)."""
        return torch.relu(self.W1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """f_m(x), shape (N, 2)."""
        return torch.relu(self.W2(torch.relu(self.W1(x))))

    def get_W2(self) -> torch.Tensor:
        return self.W2.weight.detach()   # (2, m)

    def get_W1(self) -> torch.Tensor:
        return self.W1.weight.detach()   # (m, d)

    def get_b(self) -> torch.Tensor:
        return self.W1.bias.detach()     # (m,)


class SingleReLU(nn.Module):
    """
    f(x) = W₂ · ReLU(W₁x + b)  — single ReLU (linear output).
    Used as ablation baseline in Exp 1.
    """
    def __init__(self, d: int, m: int = 30):
        super().__init__()
        self.d = d
        self.m = m
        self.W1 = nn.Linear(d, m, bias=True)
        self.W2 = nn.Linear(m, 2, bias=False)
        nn.init.kaiming_normal_(self.W1.weight, nonlinearity="relu")
        nn.init.zeros_(self.W1.bias)
        nn.init.xavier_normal_(self.W2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(torch.relu(self.W1(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _build_training_data(
    clf,
    X_train: np.ndarray,
    n_uniform: int,
    n_boundary: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build augmented training dataset for R2SNN:
      - n_uniform random points in [-1,1]^d
      - n_boundary boundary-augmented points (Gaussian perturbation, σ=0.01)
    Returns (X_aug, F_aug) where F_aug = clf.predict_proba(X_aug).
    """
    rng = np.random.RandomState(seed)
    d = X_train.shape[1]

    # Uniform samples in [-1, 1]^d
    X_unif = rng.uniform(-1, 1, size=(n_uniform, d)).astype(np.float32)

    # Identify approximate boundary points from training set
    proba = clf.predict_proba(X_train)
    uncertainty = np.abs(proba[:, 0] - proba[:, 1])  # close to 0 → near boundary
    boundary_mask = uncertainty < 0.2
    X_boundary_base = X_train[boundary_mask]

    if len(X_boundary_base) == 0:
        # Fallback: use most uncertain 20% of training data
        idx = np.argsort(uncertainty)[:max(1, len(uncertainty) // 5)]
        X_boundary_base = X_train[idx]

    # Perturb boundary points
    idx = rng.choice(len(X_boundary_base), size=n_boundary, replace=True)
    noise = rng.normal(0, 0.01, size=(n_boundary, d)).astype(np.float32)
    X_bnd = np.clip(X_boundary_base[idx] + noise, -1, 1)

    X_aug = np.vstack([X_train, X_unif, X_bnd])
    F_aug = clf.predict_proba(X_aug).astype(np.float32)
    return X_aug, F_aug


def _grad_penalty(model: R2SNN, X_batch: torch.Tensor, gamma: float, zeta1: float) -> torch.Tensor:
    """
    R_grad = ζ₁ · E[ReLU(||J_fm(x)||_F - γ)²]
    Computes Jacobian of f_m w.r.t. x on a small subset.
    """
    # Use a subset of batch for efficiency (every 5th point)
    X_sub = X_batch[::5].requires_grad_(True)
    out = model(X_sub)           # (N_sub, 2)
    # Sum outputs for scalar gradient
    grads = []
    for j in range(2):
        g = torch.autograd.grad(
            out[:, j].sum(), X_sub,
            create_graph=True, retain_graph=True
        )[0]  # (N_sub, d)
        grads.append(g)
    # Frobenius norm of Jacobian ≈ sqrt(sum of squared grad norms)
    J_norm = torch.sqrt(sum((g ** 2).sum(dim=1) for g in grads) + 1e-8)  # (N_sub,)
    penalty = zeta1 * torch.mean(torch.relu(J_norm - gamma) ** 2)
    return penalty


def _consistency_term(
    out: torch.Tensor,
    y0_vec: torch.Tensor,
    y1_vec: torch.Tensor,
    delta_c: float,
    zeta2: float,
) -> torch.Tensor:
    """
    R_cons = -(ζ₂/(N·δ_c)) Σ log(exp(-δ_c||f_m(x)-y0||²) + exp(-δ_c||f_m(x)-y1||²))
    """
    d0 = ((out - y0_vec) ** 2).sum(dim=1)  # (N,)
    d1 = ((out - y1_vec) ** 2).sum(dim=1)
    log_sum = torch.log(torch.exp(-delta_c * d0) + torch.exp(-delta_c * d1) + 1e-12)
    return -zeta2 / delta_c * log_sum.mean()


def train_r2snn(
    clf,
    X_train: np.ndarray,
    m: int = config.R2SNN_HIDDEN,
    n_uniform: int = config.R2SNN_N_UNIFORM,
    n_boundary: int = config.R2SNN_N_BOUNDARY,
    epochs: int = config.R2SNN_EPOCHS,
    batch_size: int = config.R2SNN_BATCH,
    lr: float = config.R2SNN_LR,
    zeta1: float = config.R2SNN_ZETA1,
    zeta2: float = config.R2SNN_ZETA2,
    gamma_clip: float = config.R2SNN_GAMMA_CLIP,
    tau_clip: float = config.R2SNN_TAU_CLIP,
    delta_c: float = config.R2SNN_DELTA_C,
    patience: int = config.R2SNN_PATIENCE,
    seed: int = config.SEED,
    device: str = None,
    use_single_relu: bool = False,
) -> nn.Module:
    """
    Train R2SNN (or SingleReLU if use_single_relu=True) to approximate clf.
    Returns trained model in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    d = X_train.shape[1]
    model = SingleReLU(d, m) if use_single_relu else R2SNN(d, m)
    model = model.to(device)

    # Build augmented training set
    X_aug, F_aug = _build_training_data(clf, X_train, n_uniform, n_boundary, seed)
    N = len(X_aug)

    X_t = torch.from_numpy(X_aug).to(device)
    F_t = torch.from_numpy(F_aug).to(device)

    y0_vec = torch.tensor([1.0, 0.0], device=device)
    y1_vec = torch.tensor([0.0, 1.0], device=device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = X_t[batch_idx].requires_grad_(True)
            fb = F_t[batch_idx]

            out = model(xb)

            # L_approx: MSE against soft clf output
            l_approx = ((out - fb) ** 2).mean()

            # R_grad: gradient penalty (skip for SingleReLU to save compute)
            if not use_single_relu and zeta1 > 0:
                l_grad = _grad_penalty(model, xb.detach().requires_grad_(True), gamma_clip, zeta1)
            else:
                l_grad = torch.tensor(0.0, device=device)

            # R_cons: consistency / entropy term
            l_cons = _consistency_term(out, y0_vec, y1_vec, delta_c, zeta2)

            loss = l_approx + l_grad + l_cons

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tau_clip)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# ELM reconstruction mapping  Γ = lstsq(Σ, X)
# ──────────────────────────────────────────────────────────────────────────────

def build_elm_reconstruction(
    model: R2SNN,
    X_data: np.ndarray,
    device: str = None,
) -> np.ndarray:
    """
    Compute Γ such that γ(e) = e @ Γ.T ≈ x.
    Γ shape: (d, m)  — so x ≈ e @ Γ.T  (equivalently x = Γ @ e for single vector).

    Uses least-squares: Σ @ Γ.T = X  →  Γ.T = lstsq(Σ, X)  →  Γ = result.T
    Σ: (N, m), X: (N, d)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_data).to(device)
        Sigma = model.get_features(X_t).cpu().numpy()  # (N, m)

    # Solve Σ @ Γ.T = X  in least-squares sense
    Gamma_T, _, _, _ = np.linalg.lstsq(Sigma, X_data, rcond=None)  # (m, d)
    Gamma = Gamma_T.T.astype(np.float32)  # (d, m)
    return Gamma


def reconstruct_input(e: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """
    x* = Γ @ e  for e shape (m,)  → x shape (d,)
    or  x* = e @ Γ.T  for e shape (N, m)  → x shape (N, d)
    """
    if e.ndim == 1:
        return Gamma @ e
    return e @ Gamma.T


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def surr_path(dataset_name: str, clf_name: str, m: int = config.R2SNN_HIDDEN) -> str:
    return os.path.join(SURR_DIR, f"{dataset_name}_{clf_name}_m{m}.pt")


def save_r2snn(model: nn.Module, dataset_name: str, clf_name: str,
               Gamma: np.ndarray, m: int = config.R2SNN_HIDDEN) -> None:
    path = surr_path(dataset_name, clf_name, m)
    torch.save({"state_dict": model.state_dict(),
                "d": model.d, "m": model.m,
                "Gamma": Gamma,
                "type": "SingleReLU" if isinstance(model, SingleReLU) else "R2SNN"},
               path)


def load_r2snn(dataset_name: str, clf_name: str,
               m: int = config.R2SNN_HIDDEN,
               device: str = None) -> tuple:
    """Returns (model, Gamma)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = surr_path(dataset_name, clf_name, m)
    if not os.path.exists(path):
        raise FileNotFoundError(f"R2SNN not found: {path}")
    ckpt = torch.load(path, map_location=device)
    ModelClass = SingleReLU if ckpt.get("type") == "SingleReLU" else R2SNN
    model = ModelClass(ckpt["d"], ckpt["m"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["Gamma"]


def get_or_train_r2snn(
    dataset_name: str,
    clf_name: str,
    clf,
    X_train: np.ndarray,
    m: int = config.R2SNN_HIDDEN,
    seed: int = config.SEED,
    device: str = None,
    use_single_relu: bool = False,
) -> tuple:
    """Returns (model, Gamma), training and saving if not cached."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    path = surr_path(dataset_name, clf_name, m)
    # Check cache (differentiate R2SNN vs SingleReLU by filename)
    if use_single_relu:
        path = path.replace(".pt", "_srelu.pt")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        ModelClass = SingleReLU if use_single_relu else R2SNN
        model = ModelClass(ckpt["d"], ckpt["m"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt["Gamma"]

    # Cap training data to keep runtime tractable (surrogate quality unaffected)
    if len(X_train) > 10000:
        rng_cap = np.random.default_rng(seed)
        idx_cap = rng_cap.choice(len(X_train), 10000, replace=False)
        X_train_fit = X_train[idx_cap]
    else:
        X_train_fit = X_train
    model = train_r2snn(clf, X_train_fit, m=m, seed=seed, device=device,
                        use_single_relu=use_single_relu)
    Gamma = build_elm_reconstruction(model, X_train_fit, device=device)
    # Save
    torch.save({"state_dict": model.state_dict(),
                "d": model.d, "m": model.m,
                "Gamma": Gamma,
                "type": "SingleReLU" if use_single_relu else "R2SNN"},
               path)
    return model, Gamma


# ──────────────────────────────────────────────────────────────────────────────
# Exp 1: R2SNN vs SingleReLU approximation comparison
# ──────────────────────────────────────────────────────────────────────────────

def _compute_approx_errors(
    model_proba: np.ndarray,
    clf_proba: np.ndarray,
    clf_labels: np.ndarray,
    model_labels: np.ndarray,
) -> dict:
    diff = model_proba - clf_proba
    linf = float(np.abs(diff).max())
    l2   = float(np.sqrt((diff ** 2).sum(axis=1)).mean())
    l1   = float(np.abs(diff).sum(axis=1).mean())
    acc_diff = float(np.mean(clf_labels != model_labels))
    return {"linf": linf, "l2": l2, "l1": l1, "acc_diff": acc_diff}


def eval_r2snn_vs_single_relu(
    clf,
    X_train: np.ndarray,
    X_test: np.ndarray,
    m_values: list = None,
    n_seeds: int = config.N_SEEDS,
    device: str = None,
) -> dict:
    """
    Exp 1: Train R2SNN and SingleReLU for each m in m_values,
    over n_seeds random seeds. Evaluate on X_test.

    Returns dict with arrays of shape (len(m_values), n_seeds) for each metric.
    """
    if m_values is None:
        m_values = config.M_VALUES
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    clf_proba  = clf.predict_proba(X_test).astype(np.float32)
    clf_labels = clf.predict(X_test)

    results = {
        "m_values": m_values,
        "r2snn_linf": np.zeros((len(m_values), n_seeds)),
        "r2snn_l2":   np.zeros((len(m_values), n_seeds)),
        "r2snn_l1":   np.zeros((len(m_values), n_seeds)),
        "r2snn_acc_diff": np.zeros((len(m_values), n_seeds)),
        "srelu_linf": np.zeros((len(m_values), n_seeds)),
        "srelu_l2":   np.zeros((len(m_values), n_seeds)),
        "srelu_l1":   np.zeros((len(m_values), n_seeds)),
        "srelu_acc_diff": np.zeros((len(m_values), n_seeds)),
    }

    for mi, m in enumerate(m_values):
        for si in range(n_seeds):
            seed_i = config.SEED + si * 100

            # R2SNN
            r2snn = train_r2snn(clf, X_train, m=m, seed=seed_i, device=device,
                                 epochs=300)  # shorter for sweep
            with torch.no_grad():
                X_t = torch.from_numpy(X_test).to(device)
                r2_proba = r2snn(X_t).cpu().numpy()
            r2_labels = np.argmax(r2_proba, axis=1)
            e = _compute_approx_errors(r2_proba, clf_proba, clf_labels, r2_labels)
            results["r2snn_linf"][mi, si] = e["linf"]
            results["r2snn_l2"][mi, si]   = e["l2"]
            results["r2snn_l1"][mi, si]   = e["l1"]
            results["r2snn_acc_diff"][mi, si] = e["acc_diff"]

            # SingleReLU
            srelu = train_r2snn(clf, X_train, m=m, seed=seed_i, device=device,
                                epochs=300, use_single_relu=True)
            with torch.no_grad():
                sr_proba = srelu(X_t).cpu().numpy()
            sr_labels = np.argmax(sr_proba, axis=1)
            e2 = _compute_approx_errors(sr_proba, clf_proba, clf_labels, sr_labels)
            results["srelu_linf"][mi, si] = e2["linf"]
            results["srelu_l2"][mi, si]   = e2["l2"]
            results["srelu_l1"][mi, si]   = e2["l1"]
            results["srelu_acc_diff"][mi, si] = e2["acc_diff"]

    return results
