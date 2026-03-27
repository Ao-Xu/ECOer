"""
Publication-quality figure generation for ECOer paper.
All figures are saved to experiments/figures/ AND copied to the LaTeX folder.
"""
import sys
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.FIGURES_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_figure(fig, filename: str) -> None:
    """Save to figures/ and copy to LATEX_DIR."""
    fig_path = os.path.join(config.FIGURES_DIR, filename)
    fig.savefig(fig_path, dpi=config.FIG_DPI, bbox_inches="tight")
    latex_path = os.path.join(config.LATEX_DIR, filename)
    if os.path.isdir(config.LATEX_DIR):
        shutil.copy2(fig_path, latex_path)
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def _method_order():
    return ["ECOer"] + [config.BASELINE_DISPLAY[b] for b in config.BASELINES]


def _colors():
    return {k: v for k, v in config.METHOD_COLORS.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Exp 1: R2SNN vs SingleReLU approximation (replaces comparison.png)
# ──────────────────────────────────────────────────────────────────────────────

def plot_exp1_approx(
    results_by_dataset: dict,    # {ds_name: eval_r2snn_vs_single_relu output}
    m_values: list = None,
    filename: str = "comparison.png",
) -> None:
    if m_values is None:
        m_values = config.M_VALUES

    datasets = list(results_by_dataset.keys())
    n_ds = len(datasets)

    metrics = [
        ("r2snn_linf", "srelu_linf", r"$L_\infty$-Error"),
        ("r2snn_l2",   "srelu_l2",   r"$L_2$-Error"),
        ("r2snn_l1",   "srelu_l1",   r"$L_1$-Error"),
        ("r2snn_acc_diff", "srelu_acc_diff", "Accuracy Difference"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax_i, (r2_key, sr_key, metric_label) in enumerate(metrics):
        ax = axes[ax_i]
        for ds_i, ds_name in enumerate(datasets):
            res = results_by_dataset[ds_name]
            color_r2  = plt.cm.tab10(ds_i * 2)
            color_sr  = plt.cm.tab10(ds_i * 2 + 1)
            ds_label  = config.DATASET_DISPLAY.get(ds_name, ds_name)

            r2_mean = res[r2_key].mean(axis=1)
            r2_std  = res[r2_key].std(axis=1)
            sr_mean = res[sr_key].mean(axis=1)
            sr_std  = res[sr_key].std(axis=1)

            ax.plot(m_values, r2_mean, "-o", color=color_r2,
                    label=f"{ds_label} R2SNN", linewidth=1.5, markersize=4)
            ax.fill_between(m_values, r2_mean - r2_std, r2_mean + r2_std,
                            alpha=0.15, color=color_r2)
            ax.plot(m_values, sr_mean, "--s", color=color_sr,
                    label=f"{ds_label} SingleReLU", linewidth=1.5, markersize=4)
            ax.fill_between(m_values, sr_mean - sr_std, sr_mean + sr_std,
                            alpha=0.15, color=color_sr)

        ax.set_xlabel("Number of neurons $m$", fontsize=config.FIG_FONT)
        ax.set_ylabel(metric_label, fontsize=config.FIG_FONT)
        ax.set_title(metric_label, fontsize=config.FIG_FONT + 1, fontweight="bold")
        ax.tick_params(labelsize=config.FIG_FONT - 1)
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=config.FIG_FONT - 2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("R2SNN vs Single-ReLU approximation performance",
                 fontsize=config.FIG_FONT + 2, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_figure(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Exp 2: Proximity metrics — ℓ₁, ℓ₂ (replaces exp1_proximity_plot_1x2_std.png)
# ──────────────────────────────────────────────────────────────────────────────

def plot_exp2_proximity(
    results: dict,        # {ds_name: {method_display: {'l1_mean','l1_std','l2_mean','l2_std'}}}
    method_order: list = None,
    filename: str = "exp1_proximity_plot_1x2_std.png",
) -> None:
    if method_order is None:
        method_order = _method_order()

    datasets = list(results.keys())
    ds_display = [config.DATASET_DISPLAY.get(d, d) for d in datasets]
    n_ds  = len(datasets)
    n_met = len(method_order)
    x     = np.arange(n_ds)
    width = 0.8 / n_met
    colors = _colors()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col_i, (metric_key, metric_label) in enumerate(
        [("l1_mean", r"$\ell_1$ Distance"), ("l2_mean", r"$\ell_2$ Distance")]
    ):
        ax = axes[col_i]
        std_key = metric_key.replace("mean", "std")
        for mi, method in enumerate(method_order):
            vals = []
            stds = []
            for ds in datasets:
                v = results[ds].get(method, {}).get(metric_key, float("nan"))
                s = results[ds].get(method, {}).get(std_key, 0.0)
                vals.append(v)
                stds.append(s)
            offset = (mi - n_met / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=method,
                          color=colors.get(method, f"C{mi}"), alpha=0.85)
            ax.errorbar(x + offset, vals, yerr=stds,
                        fmt="none", color="gray", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_display, fontsize=config.FIG_FONT - 1)
        ax.set_ylabel(metric_label, fontsize=config.FIG_FONT)
        ax.set_title(metric_label, fontsize=config.FIG_FONT + 1, fontweight="bold")
        ax.tick_params(axis="y", labelsize=config.FIG_FONT - 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_met,
               fontsize=config.FIG_FONT - 1, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Counterfactual proximity comparison", fontsize=config.FIG_FONT + 2,
                 fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Exp 3: Quality metrics — DP, IM (replaces exp2_quality_2x2_dp_im.png)
# ──────────────────────────────────────────────────────────────────────────────

def plot_exp3_quality(
    results_1cf: dict,   # {ds: {method: {dp, im_mean, im_std}}}
    results_5cf: dict,
    method_order: list = None,
    filename: str = "exp2_quality_2x2_dp_im.png",
) -> None:
    if method_order is None:
        method_order = _method_order()

    datasets = list(results_1cf.keys())
    ds_display = [config.DATASET_DISPLAY.get(d, d) for d in datasets]
    n_ds  = len(datasets)
    n_met = len(method_order)
    x     = np.arange(n_ds)
    width = 0.8 / n_met
    colors = _colors()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    configs = [
        (axes[0, 0], results_1cf, "dp",      "DP",  r"Discriminative Power (#CF=1)"),
        (axes[0, 1], results_5cf, "dp",      "DP",  r"Discriminative Power (#CF=5)"),
        (axes[1, 0], results_1cf, "im_mean", "IM",  r"Implausibility (#CF=1)"),
        (axes[1, 1], results_5cf, "im_mean", "IM",  r"Implausibility (#CF=5)"),
    ]

    for ax, res_dict, mkey, _, title in configs:
        std_key = "im_std" if mkey == "im_mean" else None
        for mi, method in enumerate(method_order):
            vals = [res_dict[ds].get(method, {}).get(mkey, float("nan")) for ds in datasets]
            stds = ([res_dict[ds].get(method, {}).get(std_key, 0.0) for ds in datasets]
                    if std_key else [0.0] * n_ds)
            offset = (mi - n_met / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=method,
                   color=colors.get(method, f"C{mi}"), alpha=0.85)
            ax.errorbar(x + offset, vals, yerr=stds,
                        fmt="none", color="gray", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_display, fontsize=config.FIG_FONT - 1)
        ax.set_title(title, fontsize=config.FIG_FONT + 1, fontweight="bold")
        ax.tick_params(labelsize=config.FIG_FONT - 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_met,
               fontsize=config.FIG_FONT - 1, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Comprehensive quality evaluation", fontsize=config.FIG_FONT + 2,
                 fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Exp 4a: Ablation — classifier choice (replaces exp4_ablation_classifiers.png)
# ──────────────────────────────────────────────────────────────────────────────

def plot_exp4_classifiers(
    results: dict,   # {clf_variant_label: {method: {l1_mean,...}}}
    filename: str = "exp4_ablation_classifiers.png",
) -> None:
    clf_variants = list(results.keys())
    n_clf = len(clf_variants)
    methods = ["ECOer", "DiCE"]
    x = np.arange(n_clf)
    width = 0.35
    colors = _colors()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metric_configs = [
        ("l1_mean", "l1_std",  r"$\ell_1$ Distance"),
        ("l2_mean", "l2_std",  r"$\ell_2$ Distance"),
        ("dp",      None,      "Discriminative Power"),
        ("im_mean", "im_std",  "Implausibility"),
    ]

    for ax, (mkey, skey, mlabel) in zip(axes, metric_configs):
        for mi, method in enumerate(methods):
            vals = [results[v].get(method, {}).get(mkey, float("nan")) for v in clf_variants]
            stds = ([results[v].get(method, {}).get(skey, 0.0) for v in clf_variants]
                    if skey else [0.0] * n_clf)
            offset = (mi - 0.5) * width
            ax.bar(x + offset, vals, width, label=method,
                   color=colors.get(method, f"C{mi}"), alpha=0.85)
            ax.errorbar(x + offset, vals, yerr=stds,
                        fmt="none", color="gray", capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(clf_variants, rotation=20, fontsize=config.FIG_FONT - 2)
        ax.set_title(mlabel, fontsize=config.FIG_FONT, fontweight="bold")
        ax.tick_params(labelsize=config.FIG_FONT - 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=config.FIG_FONT, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Ablation: Effect of nonparametric classifier choice",
                 fontsize=config.FIG_FONT + 1, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Exp 4b: Ablation — arch size + energy components (replaces exp4_ablation_ecoe_components.png)
# ──────────────────────────────────────────────────────────────────────────────

def plot_exp4_components(
    arch_results: dict,    # {m: {'l1_mean','l2_mean','dp','im_mean','runtime_mean'}}
    energy_results: dict,  # {variant_label: {'l1_mean','l2_mean','dp','im_mean'}}
    filename: str = "exp4_ablation_ecoe_components.png",
) -> None:
    m_values = sorted(arch_results.keys())
    energy_variants = list(energy_results.keys())

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # Top row: architecture sweep (m)
    arch_metrics = [
        ("l1_mean", r"$\ell_1$ Distance"),
        ("l2_mean", r"$\ell_2$ Distance"),
        ("dp",      "Discriminative Power"),
        ("runtime_mean", "Runtime (s)"),
    ]
    for ax, (mkey, mlabel) in zip(axes[0], arch_metrics):
        vals = [arch_results[m].get(mkey, float("nan")) for m in m_values]
        stds = [arch_results[m].get(mkey.replace("mean", "std"), 0.0) for m in m_values]
        ax.plot(m_values, vals, "-o", color=config.METHOD_COLORS["ECOer"],
                linewidth=2, markersize=6)
        ax.fill_between(m_values,
                        np.array(vals) - np.array(stds),
                        np.array(vals) + np.array(stds),
                        alpha=0.2, color=config.METHOD_COLORS["ECOer"])
        ax.set_xlabel("Hidden neurons $m$", fontsize=config.FIG_FONT - 1)
        ax.set_title(mlabel, fontsize=config.FIG_FONT, fontweight="bold")
        ax.tick_params(labelsize=config.FIG_FONT - 1)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Bottom row: energy component ablation
    x = np.arange(len(energy_variants))
    e_colors = [plt.cm.Set2(i) for i in range(len(energy_variants))]
    for ax, (mkey, mlabel) in zip(axes[1], [
        ("l1_mean", r"$\ell_1$"), ("l2_mean", r"$\ell_2$"),
        ("dp", "DP"), ("im_mean", "IM"),
    ]):
        vals = [energy_results[v].get(mkey, float("nan")) for v in energy_variants]
        stds = [energy_results[v].get(mkey.replace("mean", "std"), 0.0) for v in energy_variants]
        bars = ax.bar(x, vals, color=e_colors, alpha=0.85)
        ax.errorbar(x, vals, yerr=stds, fmt="none", color="gray", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(energy_variants, rotation=25, fontsize=config.FIG_FONT - 2)
        ax.set_title(mlabel, fontsize=config.FIG_FONT, fontweight="bold")
        ax.tick_params(labelsize=config.FIG_FONT - 1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0, 0].set_ylabel("Architecture sweep", fontsize=config.FIG_FONT)
    axes[1, 0].set_ylabel("Energy components", fontsize=config.FIG_FONT)
    fig.suptitle("Ablation: Architecture size and energy function components",
                 fontsize=config.FIG_FONT + 2, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# Exp 5: Wilcoxon p-value heatmap (new figure)
# ──────────────────────────────────────────────────────────────────────────────

def plot_exp5_stats(
    stats_results: dict,   # {baseline: {metric: {p_value,...}}}
    filename: str = "exp5_wilcoxon_heatmap.png",
) -> None:
    import seaborn as sns
    baselines = list(stats_results.keys())
    if not baselines:
        return
    metrics   = list(stats_results[baselines[0]].keys())

    pvals = np.array([
        [stats_results[b][m]["p_value"] for m in metrics]
        for b in baselines
    ])

    baseline_labels = [config.BASELINE_DISPLAY.get(b, b) for b in baselines]

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 1.5), max(4, len(baselines) * 0.9)))
    mask = np.isnan(pvals)
    # Use log10 of p-value for colour scale
    log_pvals = np.where(mask, 0, -np.log10(np.clip(pvals, 1e-10, 1.0)))

    im = ax.imshow(log_pvals, cmap="RdYlGn", vmin=0, vmax=4, aspect="auto")

    for i in range(len(baselines)):
        for j in range(len(metrics)):
            p = pvals[i, j]
            if np.isnan(p):
                text = "n/a"
            else:
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                text = f"{p:.3f}{stars}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=config.FIG_FONT - 2,
                    color="black" if log_pvals[i, j] < 2.5 else "white")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=config.FIG_FONT)
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels(baseline_labels, fontsize=config.FIG_FONT)
    ax.set_xlabel("Metric", fontsize=config.FIG_FONT + 1)
    ax.set_ylabel("Baseline", fontsize=config.FIG_FONT + 1)
    ax.set_title("Wilcoxon signed-rank test: ECOer vs. Baselines\n"
                 r"($-\log_{10}(p)$; * p<0.05, ** p<0.01, *** p<0.001)",
                 fontsize=config.FIG_FONT + 1, fontweight="bold")
    plt.colorbar(im, ax=ax, label=r"$-\log_{10}(p)$")
    plt.tight_layout()
    _save_figure(fig, filename)
