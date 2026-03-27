# ECOer: Energy-Regularized Counterfactual Explanations via ELM Feature Space

Official experiment code for the paper:

> **ECOer: Energy-Regularized Counterfactual Explanations for Nonparametric Classifiers**
> *Submitted to Knowledge-Based Systems*

---

## Overview

ECOer generates counterfactual explanations for black-box nonparametric classifiers (KNN, Random Forest, Kernel SVM) by optimizing in the feature space of a **R2SNN surrogate** with an energy regularization term that keeps counterfactuals on the data manifold.

**Key components:**
- **R2SNN** — dual-ReLU surrogate network `f_m(x) = σ(W₂ σ(W₁x + b))` with gradient-penalty and consistency regularization
- **ELM reconstruction** — `x* = Γ @ e*` via least-squares pseudo-inverse; maps hidden activations back to input space
- **ECOer optimizer** — convex counterfactual objective in the ELM feature space with energy term `Ψ(e)` enforcing manifold adherence

**Hyperparameters** (from Bayesian optimization):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `m` | 30 | Hidden neurons |
| `λ₁` | 0.50 | Classifier loss weight |
| `λ₂` | 0.40 | Proximity weight |
| `β` | 0.60 | Energy regularization weight |
| `lr` | 0.01 | Counterfactual optimization learning rate |

---

## Results Summary

ECOer is evaluated across **6 datasets × 3 classifiers × 6 baselines** on 5 metrics:

| Metric | ECOer | Best Baseline |
|--------|-------|---------------|
| Validity Rate | **0.88–0.99** | 0.55–0.85 |
| DP (↑) | **0.855** | 0.722 (DPMDCE) |
| IM (↓) | **0.091** | 0.141 (DPMDCE) |
| ℓ₁ proximity | competitive | GrowingSpheres best |
| Sparsity | **3.82** | 4.23 (DPMDCE) |

ECOer achieves **statistically significant** improvements over all baselines on Validity, DP, and IM (Wilcoxon signed-rank, p < 0.05).

---

## Repository Structure

```
experiments/
├── config.py                   # Global constants, hyperparameters, paths
├── run_all.py                  # One-shot pipeline runner
├── run_exp1_approx.py          # Exp 1: R2SNN vs SingleReLU approximation error
├── run_exp2_proximity.py       # Exp 2: Counterfactual proximity (ℓ₁, ℓ₂, validity)
├── run_exp3_quality.py         # Exp 3: Manifold quality (DP, IM, Sparsity)
├── run_exp4_ablation.py        # Exp 4: Ablation — classifiers, architecture, components
├── run_exp5_stats.py           # Exp 5: Wilcoxon signed-rank significance tests
├── src/
│   ├── r2snn.py                # R2SNN architecture + ELM reconstruction
│   ├── ecoe_optimizer.py       # ECOer counterfactual optimizer (Algorithm 1)
│   ├── baselines.py            # DiCE, FACE, GrowingSpheres, Revise, WACH, DPMDCE
│   ├── metrics.py              # ℓ₁, ℓ₂, Sparsity, DP, IM, Validity
│   ├── stats.py                # Wilcoxon test utilities
│   ├── classifiers.py          # KNN / RF / SVM training & caching
│   ├── data_loader.py          # Dataset download (ucimlrepo)
│   ├── preprocessing.py        # Min-max scaling, train/test split
│   └── plotting.py             # Publication-quality matplotlib figures
├── data/
│   ├── raw/                    # Downloaded CSVs
│   └── processed/              # Normalized NPZ files
├── models/
│   ├── classifiers/            # Trained sklearn models (.joblib)
│   └── surrogates/             # Trained R2SNN weights (.pt)
├── results/                    # Experiment outputs (JSON)
│   ├── exp1_approx/
│   ├── exp2_proximity/
│   ├── exp3_quality/
│   ├── exp4_ablation/
│   └── exp5_stats/
└── figures/                    # Generated PNG figures (300 DPI)
```

---

## Datasets

| Dataset | n | d | Domain |
|---------|---|---|--------|
| HELOC | 10,459 | 23 | Financial |
| Adult | 48,842 | 14 | Socio-economic |
| German Credit | 1,000 | 20 | Financial |
| COMPAS | 7,214 | 11 | Criminal Justice |
| Heart Disease | 303 | 13 | Medical |
| Pima Diabetes | 768 | 8 | Medical |

Datasets are downloaded automatically via `ucimlrepo` on first run.

---

## Installation

```bash
pip install torch scikit-learn numpy scipy pandas matplotlib seaborn \
            dice-ml ucimlrepo scikit-optimize joblib tqdm
```

GPU (CUDA) is recommended. Tested on Python 3.9+, PyTorch 2.0+.

---

## Quickstart

**Run all experiments** (full pipeline, ~3–5 hours on RTX 3090):

```bash
cd experiments
python run_all.py
```

**Run a single experiment:**

```bash
python run_all.py --only exp1   # R2SNN approximation sweep
python run_all.py --only exp2   # Proximity & validity
python run_all.py --only exp3   # DP / IM / Sparsity
python run_all.py --only exp4   # Ablation study
python run_all.py --only exp5   # Statistical significance
```

**Skip dataset setup if already cached:**

```bash
python run_all.py --skip-setup
```

Results are cached automatically — re-running skips completed (dataset, classifier) combinations.

---

## Figures

All publication figures are saved to `figures/` (300 DPI PNG):

| File | Content |
|------|---------|
| `comparison.png` | Exp 1 — R2SNN vs SingleReLU error across hidden sizes m (2×3 subplots) |
| `exp1_proximity_plot_1x2_std.png` | Exp 2 — ℓ₁/ℓ₂ proximity comparison with std bands |
| `exp2_validity.png` | Exp 2 — Validity rate per method |
| `exp2_quality_2x2_dp_im.png` | Exp 3 — DP and IM heatmaps |
| `exp4_ablation_classifiers.png` | Exp 4 — Classifier comparison |
| `exp4_ablation_ecoe_components.png` | Exp 4 — Component ablation (w/o energy, w/o R_grad, etc.) |
| `exp5_wilcoxon_heatmap.png` | Exp 5 — Wilcoxon p-value heatmap |

---

## Hardware

All experiments were run on:
- GPU: NVIDIA RTX 3090 (24 GB)
- CPU: Intel Core i9-12900K
- OS: Windows 11

---

## License

MIT License
