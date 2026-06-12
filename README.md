# ECOer Experiments

Experiment code for the revised manuscript:

> **Convex-Relaxed Counterfactual Optimization for Non-differentiable Classifiers via Neural Surrogates**
> Submitted to *Neural Networks*.

## Overview

ECOer generates counterfactual explanations for black-box non-differentiable or
non-smooth classifiers, including KNN, random forests, and kernel SVMs. It fits a
ReLU-ReLU single-hidden-layer neural network (R2SNN) surrogate, constructs a
ridge-regularized linear reconstruction map, and optimizes a region-wise relaxed
counterfactual objective in the surrogate feature space.

Key components:

- **R2SNN surrogate**: dual-ReLU shallow network trained with approximation,
  gradient-penalty, and consistency terms.
- **Ridge reconstruction**: `x_cf = Gamma_eta @ e_cf`, where `Gamma_eta` solves a
  ridge-regularized least-squares reconstruction problem.
- **Metric-preserving reconstruction regularizer**: `Psi(e) = d(Gamma_eta e,
  x_in)`, used to reduce latent-input metric distortion during feature-space
  counterfactual search.

Default counterfactual hyperparameters:

| Parameter | Value | Description |
| --- | ---: | --- |
| `m` | 30 | Hidden neurons |
| `lambda_pre` | 0.50 | Source-side / pre-crossing regional weight |
| `lambda_post` | 0.40 | Target-side / post-crossing regional weight |
| `beta` | 0.60 | Reconstruction-regularization weight |
| `eta` | 1e-4 | Ridge parameter for `Gamma_eta` |
| `lr` | 0.01 | Counterfactual optimization learning rate |

The old aliases `lambda1` and `lambda2` are still accepted by the optimizer for
backward compatibility with earlier scripts. They map to `lambda_post` and
`lambda_pre`, respectively.

## Repository Structure

```text
experiments/
  config.py                     Global constants, hyperparameters, paths
  run_all.py                    One-shot pipeline runner
  run_exp1_approx.py            Exp. 1: R2SNN vs SingleReLU approximation
  run_exp2_proximity.py         Exp. 2: Proximity and validity
  run_exp3_quality.py           Exp. 3: DP, IM, sparsity
  run_exp4_ablation.py          Exp. 4: Classifier/architecture/component ablations
  run_exp5_stats.py             Exp. 5: Wilcoxon signed-rank tests
  run_exp6_runtime.py           Exp. 6: Runtime, hidden-width, memory
  run_exp7_stability.py         Exp. 7: Ridge reconstruction stability
  run_exp8_distortion.py        Exp. 8: Direct metric-distortion diagnostic
  run_exp9_sampling_ablation.py Exp. 9: Sampling/training ablation
  run_exp10_rf_specialized.py   Exp. 10: RF-specific sanity baseline
  ecoe/                         Minimal reusable ECOer API
  src/                          Experiment implementation
  data/                         Raw and processed datasets
  models/                       Cached classifiers and surrogates
  results/                      Experiment outputs
  figures/                      Generated PNG figures
```

## Minimal Algorithm API

The revised repository provides a compact `ecoe` module that contains only the
core ECOer components, separated from experiment orchestration, plotting, and
benchmark code.

```python
from ecoe import ECOerExplainer

explainer = ECOerExplainer(m=30)
explainer.fit(X_train, predictor)  # predictor exposes predict_proba/predict
result = explainer.explain(x_query)
x_cf = result["x_cf"]
```

Lower-level API:

```python
from ecoe import fit_surrogate, build_reconstruction, generate_counterfactual

model = fit_surrogate(predictor, X_train, m=30)
Gamma = build_reconstruction(model, X_train, eta=1e-4)
result = generate_counterfactual(x_query, model, Gamma, clf=predictor)
```

## Running Experiments

Install the Python dependencies:

```bash
pip install torch scikit-learn numpy scipy pandas matplotlib seaborn \
            dice-ml ucimlrepo scikit-optimize joblib tqdm
```

Run the full pipeline:

```bash
cd experiments
python run_all.py
```

Run selected experiments:

```bash
python run_all.py --only exp1
python run_all.py --only exp2
python run_all.py --only exp3
python run_all.py --only exp4
python run_all.py --only exp5
python run_all.py --only exp6
python run_all.py --only exp7
python run_all.py --only exp8
python run_all.py --only exp9
python run_all.py --only exp10
```

Results are cached under `results/`. Cached models are stored under `models/`.

## Revision Experiments

The reviewer-response revision added the following experiments:

- Runtime and hidden-width sensitivity (`run_exp6_runtime.py`).
- Ridge reconstruction stability (`run_exp7_stability.py`).
- Direct metric-distortion diagnostic (`run_exp8_distortion.py`).
- Sampling/training-pipeline ablation (`run_exp9_sampling_ablation.py`).
- RF-specific sanity baseline (`run_exp10_rf_specialized.py`).

The corresponding manuscript figures are generated in `submission_extracted/`
for inclusion in the revised LaTeX source.

## Citation

```bibtex
@article{ecoe2026,
  title   = {Convex-Relaxed Counterfactual Optimization for Non-differentiable Classifiers via Neural Surrogates},
  journal = {Neural Networks},
  year    = {2026},
  note    = {Under review}
}
```
