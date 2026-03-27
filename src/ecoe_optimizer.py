"""
ECOer Algorithm 1: Counterfactual generation via energy-regularized
convex optimization in feature space.

Objective (Eq. main):
  L̃(e) = I_c(e) · (||σ(W₂e)||² - 2y₁ᵀW₂e) + β · d(Γe, x_in)

Algorithm:
  1. Initialize e* = ReLU(W₁ x_in + b)
  2. While I_c(e*) == λ₂ and steps < max_steps:
       gradient step on L̃(e*)
       steps++
  3. x* = Γ @ e*
"""
import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.r2snn import R2SNN, reconstruct_input


def _indicator(
    e: np.ndarray,
    W2: np.ndarray,      # (2, m)
    lambda1: float,
    lambda2: float,
    y_target: np.ndarray,   # target class vector, e.g. [0,1] or [1,0]
    y_source: np.ndarray,   # source class vector
) -> float:
    """
    Return λ₁ if ||c(e) - y_target|| <= ||c(e) - y_source|| (converged to target)
    Return λ₂ otherwise (still in source class, keep optimising).
    Loop condition: while I_c == λ₂ → terminates when I_c == λ₁ (target reached).
    """
    ce = np.maximum(W2 @ e, 0)   # σ(W₂e), shape (2,)
    d_target = np.linalg.norm(ce - y_target)
    d_source = np.linalg.norm(ce - y_source)
    return lambda1 if d_target <= d_source else lambda2


def _psi_and_grad(
    e: np.ndarray,
    Gamma: np.ndarray,    # (d, m)
    x_in: np.ndarray,     # (d,)
    dist: str = "l2",
) -> tuple[float, np.ndarray]:
    """
    Compute Ψ(e) = d(Γe, x_in) and its gradient w.r.t. e.

    Returns (psi_value, grad_e)
    """
    x_rec = Gamma @ e           # (d,)
    diff = x_rec - x_in         # (d,)
    if dist == "l1":
        psi = np.sum(np.abs(diff))
        # Subgradient of L1
        grad_diff = np.sign(diff)  # (d,)
        grad_e = Gamma.T @ grad_diff   # (m,)
    else:  # l2
        psi = np.linalg.norm(diff)
        if psi > 1e-10:
            grad_diff = diff / psi
        else:
            grad_diff = diff
        grad_e = Gamma.T @ grad_diff   # (m,)
    return psi, grad_e


def _objective_and_grad(
    e: np.ndarray,
    W2: np.ndarray,    # (2, m)
    Gamma: np.ndarray, # (d, m)
    x_in: np.ndarray,
    lambda1: float,
    lambda2: float,
    beta: float,
    dist: str,
    y_target: np.ndarray,
    y_source: np.ndarray,
) -> tuple:
    """
    Returns (L̃(e), ∇_e L̃(e), I_c_value).

    Term I:  I_c · (||σ(W₂e)||² - 2y_targetᵀW₂e)
    Term II: β · Ψ(e)
    """
    I_c = _indicator(e, W2, lambda1, lambda2, y_target, y_source)

    # σ(W₂e)
    z  = W2 @ e         # (2,)
    sz = np.maximum(z, 0)   # σ(W₂e)

    # Term I value: ||σ(W₂e)||² - 2 y_targetᵀ W₂e
    term1 = I_c * (np.dot(sz, sz) - 2 * np.dot(y_target, z))

    # Gradient of Term I w.r.t. e
    sigma_prime  = (z > 0).astype(float)
    dterm1_dz    = 2 * sz * sigma_prime - 2 * y_target
    grad_term1   = I_c * (W2.T @ dterm1_dz)  # (m,)

    # Term II
    psi, grad_psi = _psi_and_grad(e, Gamma, x_in, dist)
    term2      = beta * psi
    grad_term2 = beta * grad_psi

    return term1 + term2, grad_term1 + grad_term2, I_c


def generate_counterfactual_ecoe(
    x_in: np.ndarray,
    model: R2SNN,
    Gamma: np.ndarray,
    clf=None,                        # true classifier to determine y_in; if None use R2SNN
    lambda1: float = config.LAMBDA1,
    lambda2: float = config.LAMBDA2,
    beta: float = config.BETA,
    lr: float = config.CF_LR,
    max_steps: int = config.CF_MAX_STEPS,
    dist: str = "l2",
    device: str = None,
) -> dict:
    """
    Algorithm 1: ECOer counterfactual generation.

    Parameters
    ----------
    x_in  : (d,) input instance (float32, in [-1,1]^d)
    model : trained R2SNN
    Gamma : (d, m) ELM reconstruction matrix
    clf   : true classifier (used to determine x_in's label); uses R2SNN if None

    Returns
    -------
    dict with keys: x_cf, e_cf, steps, converged, runtime
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    model.eval()
    W2 = model.get_W2().cpu().numpy()   # (2, m)
    W1 = model.get_W1().cpu().numpy()   # (m, d)
    b  = model.get_b().cpu().numpy()    # (m,)

    # Determine source class and target class
    if clf is not None:
        y_in = int(clf.predict(x_in.reshape(1, -1))[0])
    else:
        # Use R2SNN prediction
        import torch as _t
        with _t.no_grad():
            out = model(_t.from_numpy(x_in).float().unsqueeze(0).to(device))
        y_in = int(out.argmax(dim=1).item())

    y_target = np.array([1.0, 0.0] if y_in == 1 else [0.0, 1.0])  # flip label
    y_source = np.array([0.0, 1.0] if y_in == 1 else [1.0, 0.0])

    # Step 1: initialise e* = ReLU(W₁ x_in + b)
    e = np.maximum(W1 @ x_in.astype(np.float64) + b, 0)
    Gamma_f = Gamma.astype(np.float64)
    W2_f    = W2.astype(np.float64)
    x_in_f  = x_in.astype(np.float64)
    y_target_f = y_target.astype(np.float64)
    y_source_f = y_source.astype(np.float64)

    # Adam optimiser state
    m_adam = np.zeros_like(e)
    v_adam = np.zeros_like(e)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    steps     = 0
    I_c       = _indicator(e, W2_f, lambda1, lambda2, y_target_f, y_source_f)
    converged = False

    while I_c == lambda2 and steps < max_steps:
        _, grad, I_c = _objective_and_grad(
            e, W2_f, Gamma_f, x_in_f, lambda1, lambda2, beta, dist,
            y_target_f, y_source_f
        )
        steps += 1
        m_adam = beta1 * m_adam + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
        m_hat  = m_adam / (1 - beta1 ** steps)
        v_hat  = v_adam / (1 - beta2 ** steps)
        e = e - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        e = np.maximum(e, 0)   # ReLU activations are non-negative

        I_c = _indicator(e, W2_f, lambda1, lambda2, y_target_f, y_source_f)

    converged = (I_c == lambda1)   # λ₁ = reached target class in R2SNN

    # Phase 2: if converged in R2SNN but true clf not flipped,
    # continue minimising Ψ(e) while keeping I_c = λ₁ (target region)
    # to push x* further across the true decision boundary.
    if converged and clf is not None:
        x_cf_test = np.clip(Gamma_f @ e, -1, 1).astype(np.float32)
        y_cf_test = clf.predict(x_cf_test.reshape(1, -1))[0]
        if y_cf_test == y_in:
            # Extra gradient steps on Ψ only (term II), staying in target region
            for extra in range(max_steps):
                psi_val, grad_psi = _psi_and_grad(e, Gamma_f, x_in_f, dist)
                # Negate gradient: move AWAY from x_in (deeper into target class)
                grad = -grad_psi   # maximise distance = push into target class
                steps += 1
                m_adam = beta1 * m_adam + (1 - beta1) * grad
                v_adam = beta2 * v_adam + (1 - beta2) * grad ** 2
                m_hat  = m_adam / (1 - beta1 ** steps)
                v_hat  = v_adam / (1 - beta2 ** steps)
                e_new  = e - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
                e_new  = np.maximum(e_new, 0)
                # Stay in target class region (reject step if leaves)
                I_check = _indicator(e_new, W2_f, lambda1, lambda2,
                                     y_target_f, y_source_f)
                if I_check == lambda1:
                    e = e_new
                x_cf_test = np.clip(Gamma_f @ e, -1, 1).astype(np.float32)
                y_cf_test = clf.predict(x_cf_test.reshape(1, -1))[0]
                if y_cf_test != y_in:
                    break

    x_cf = np.clip(Gamma_f @ e, -1, 1).astype(np.float32)

    return {
        "x_cf":      x_cf,
        "e_cf":      e.astype(np.float32),
        "steps":     steps,
        "converged": converged,
        "runtime":   time.perf_counter() - t0,
    }


def generate_counterfactuals_batch(
    X_test: np.ndarray,
    model: R2SNN,
    Gamma: np.ndarray,
    clf=None,
    n_instances: int = config.N_TEST_INSTANCES,
    seed: int = config.SEED,
    device: str = None,
    **ecoe_kwargs,
) -> list:
    """
    Run ECOer for each of the first n_instances rows in X_test.
    Returns list of result dicts (one per instance).
    """
    rng = np.random.RandomState(seed)
    N = min(n_instances, len(X_test))
    # Randomly sample n_instances test points
    idx = rng.choice(len(X_test), size=N, replace=False)
    results = []
    for i in idx:
        res = generate_counterfactual_ecoe(
            X_test[i], model, Gamma, clf=clf, device=device, **ecoe_kwargs
        )
        # Validity check
        y_in = clf.predict(X_test[i:i+1])[0] if clf is not None else None
        y_cf = clf.predict(res["x_cf"].reshape(1, -1))[0] if clf is not None else None
        res["valid"] = bool(y_cf != y_in) if clf is not None else res["converged"]
        res["x_in"] = X_test[i]
        results.append(res)
    return results
