"""
Generate all publication-quality bar charts for ECOer paper (PAMI style).
Replaces all tables with grouped bar charts.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = r'C:\1_Cache_files\exp\ECOer\experiments\results'
FIG_DIR  = r'C:\1_Cache_files\exp\ECOer\experiments\figures'
LATEX_DIR= r'C:\1_Cache_files\exp\ECOer\20250803_SNN_sub_PR_ver2'
os.makedirs(FIG_DIR, exist_ok=True)

DATASETS = ['heloc','adult','german_credit','compas','heart','pima']
DS_LABEL = {'heloc':'HELOC','adult':'Adult','german_credit':'German\nCredit',
            'compas':'COMPAS','heart':'Heart','pima':'Pima'}
CLFS     = ['knn5','rf','svm']
METHODS  = ['ECOer','DiCE','FACE','GrowingSpheres','Revise','WACH','DPMDCE']
M_LABEL  = {'ECOer':'ECOer','DiCE':'DiCE','FACE':'FACE',
            'GrowingSpheres':'GrowSph.','Revise':'Revise',
            'WACH':'WACH','DPMDCE':'DPMDCE'}

# ── Academic color palette (colorblind-friendly, PAMI-grade) ─────────────────
COLORS = {
    'ECOer':         '#2ca02c',   # vivid green  — our method
    'DiCE':          '#1f77b4',   # steel blue
    'FACE':          '#ff7f0e',   # orange
    'GrowingSpheres':'#9467bd',   # purple
    'Revise':        '#8c564b',   # brown
    'WACH':          '#e377c2',   # pink
    'DPMDCE':        '#7f7f7f',   # gray
}
HATCH = {'ECOer': '//', 'DiCE': '', 'FACE': '', 'GrowingSpheres': '',
         'Revise': '', 'WACH': '', 'DPMDCE': ''}

# ── Global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'axes.titlesize':   10,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'figure.dpi':       300,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'axes.axisbelow':   True,
})

def save(fig, name):
    p1 = os.path.join(FIG_DIR, name)
    p2 = os.path.join(LATEX_DIR, name)
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}')

def load_exp(exp_name):
    d = os.path.join(BASE, exp_name)
    data = {}
    for f in os.listdir(d):
        if f.endswith('.json'):
            data[f.replace('.json','')] = json.load(open(os.path.join(d, f)))
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Helper: aggregate metric across datasets for a given clf set
# ─────────────────────────────────────────────────────────────────────────────
def agg_by_dataset(exp2, metric_key, clfs=None):
    """Returns dict: ds -> {method: mean_value}"""
    if clfs is None: clfs = CLFS
    result = {}
    for ds in DATASETS:
        result[DS_LABEL[ds]] = {}
        vals = {m: [] for m in METHODS}
        for clf in clfs:
            key = f'{ds}_{clf}'
            if key in exp2:
                for m in METHODS:
                    if m in exp2[key]:
                        v = exp2[key][m].get(metric_key)
                        if v is not None and v == v:  # not nan
                            vals[m].append(float(v))
        for m in METHODS:
            result[DS_LABEL[ds]][m] = np.mean(vals[m]) if vals[m] else float('nan')
    return result

def agg_overall(exp_data, metric_key):
    """Returns dict: method -> mean across all ds/clf"""
    vals = {m: [] for m in METHODS}
    for key, d in exp_data.items():
        for m in METHODS:
            if m in d:
                v = d[m].get(metric_key)
                if v is not None and v == v:
                    vals[m].append(float(v))
    return {m: (np.mean(vals[m]) if vals[m] else float('nan')) for m in METHODS}

def grouped_bar(ax, ds_labels, method_vals, ylabel, title, ylim=None,
                show_ecor_label=True, methods_subset=None):
    """Draw grouped bar chart. method_vals: {ds_label: {method: val}}"""
    ms = methods_subset if methods_subset else METHODS
    n_ds  = len(ds_labels)
    n_m   = len(ms)
    width = 0.8 / n_m
    x     = np.arange(n_ds)

    for i, m in enumerate(ms):
        vals = [method_vals[ds].get(m, float('nan')) for ds in ds_labels]
        xpos = x + (i - n_m/2 + 0.5) * width
        bars = ax.bar(xpos, vals, width=width*0.92,
                      color=COLORS[m], alpha=0.88,
                      hatch=HATCH[m], edgecolor='white', linewidth=0.4,
                      label=M_LABEL[m], zorder=3)
        # Highlight ECOer bar with bold edge
        if m == 'ECOer':
            for b in bars:
                b.set_edgecolor('#1a6e1a')
                b.set_linewidth(1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    if ylim: ax.set_ylim(ylim)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Classifier Accuracy (bar chart instead of table)
# ═════════════════════════════════════════════════════════════════════════════
clf_acc = {
    'HELOC':         {'KNN\n($k$=5)':68.3, 'RF':71.7, 'SVM':71.1},
    'Adult':         {'KNN\n($k$=5)':82.6, 'RF':85.5, 'SVM':84.5},
    'German\nCredit':{'KNN\n($k$=5)':71.5, 'RF':74.5, 'SVM':74.0},
    'COMPAS':        {'KNN\n($k$=5)':64.5, 'RF':65.0, 'SVM':66.2},
    'Heart':         {'KNN\n($k$=5)':88.3, 'RF':85.0, 'SVM':83.3},
    'Pima':          {'KNN\n($k$=5)':71.4, 'RF':76.0, 'SVM':74.7},
}
clf_colors = {'KNN\n($k$=5)':'#4e79a7', 'RF':'#f28e2b', 'SVM':'#e15759'}

fig, ax = plt.subplots(figsize=(7.0, 2.8))
ds_list = list(clf_acc.keys())
clfs_list = list(clf_colors.keys())
x = np.arange(len(ds_list))
width = 0.26
for i, clf in enumerate(clfs_list):
    vals = [clf_acc[ds][clf] for ds in ds_list]
    ax.bar(x + (i-1)*width, vals, width=width*0.92,
           color=clf_colors[clf], alpha=0.88, edgecolor='white',
           label=clf.replace('\n',' '), zorder=3)
ax.set_xticks(x); ax.set_xticklabels(ds_list, fontsize=8)
ax.set_ylabel('Test Accuracy (%)', fontsize=9)
ax.set_title('Target Classifier Accuracy Across Six Datasets', fontsize=10, fontweight='bold')
ax.set_ylim(55, 92)
ax.legend(loc='lower right', framealpha=0.9, ncol=3)
fig.tight_layout()
save(fig, 'clf_accuracy.png')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Exp2 — Validity Rate (per dataset, averaged over 3 CLFs)
# ═════════════════════════════════════════════════════════════════════════════
exp2 = load_exp('exp2_proximity')
valid_by_ds = agg_by_dataset(exp2, 'validity_rate')

fig, ax = plt.subplots(figsize=(8.5, 3.2))
grouped_bar(ax, list(valid_by_ds.keys()), valid_by_ds,
            ylabel='Validity Rate', title='Exp.2 — Validity Rate per Dataset (mean over 3 classifiers)',
            ylim=(0, 1.08))
ax.axhline(1.0, color='k', lw=0.6, ls=':', alpha=0.4)
handles = [mpatches.Patch(color=COLORS[m], label=M_LABEL[m]) for m in METHODS]
ax.legend(handles=handles, ncol=7, loc='upper center',
          bbox_to_anchor=(0.5, -0.18), framealpha=0.95, fontsize=8)
fig.tight_layout()
save(fig, 'exp2_validity.png')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Exp2 — Proximity L1/L2 (per dataset, methods with valid>0.3 only)
# ═════════════════════════════════════════════════════════════════════════════
l1_by_ds = agg_by_dataset(exp2, 'l1_mean')
l2_by_ds = agg_by_dataset(exp2, 'l2_mean')
# DiCE excluded (nan/0 validity)
methods_prox = ['ECOer','FACE','GrowingSpheres','Revise','WACH','DPMDCE']

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
grouped_bar(axes[0], list(l1_by_ds.keys()), l1_by_ds,
            ylabel=r'$\ell_1$ Distance (valid CFs)',
            title=r'Exp.2 — $\ell_1$ Proximity', methods_subset=methods_prox)
grouped_bar(axes[1], list(l2_by_ds.keys()), l2_by_ds,
            ylabel=r'$\ell_2$ Distance (valid CFs)',
            title=r'Exp.2 — $\ell_2$ Proximity', methods_subset=methods_prox)
handles = [mpatches.Patch(color=COLORS[m], label=M_LABEL[m]) for m in methods_prox]
fig.legend(handles=handles, ncol=6, loc='upper center',
           bbox_to_anchor=(0.5, 0.0), framealpha=0.95, fontsize=8)
fig.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, 'exp1_proximity_plot_1x2_std.png')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Exp2 — Overall summary (validity + L1 side by side, bar per method)
# ═════════════════════════════════════════════════════════════════════════════
overall_valid = agg_overall(exp2, 'validity_rate')
overall_l1    = agg_overall(exp2, 'l1_mean')

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, vals, ylabel, title, ylim in [
    (axes[0], overall_valid, 'Validity Rate',   'Overall Validity (mean ± std)', (0, 1.1)),
    (axes[1], overall_l1,    r'$\ell_1$ Distance', r'Overall $\ell_1$ Proximity',  (0, None)),
]:
    ms_plot = [m for m in METHODS if vals[m]==vals[m]]
    xpos = np.arange(len(ms_plot))
    bars = ax.bar(xpos, [vals[m] for m in ms_plot],
                  color=[COLORS[m] for m in ms_plot],
                  alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3,
                  width=0.6)
    # Bold ECOer
    for b, m in zip(bars, ms_plot):
        if m == 'ECOer':
            b.set_edgecolor('#1a6e1a'); b.set_linewidth(1.5)
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f'{vals[m]:.3f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold', color='#1a6e1a')
    ax.set_xticks(xpos)
    ax.set_xticklabels([M_LABEL[m] for m in ms_plot], rotation=30, ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
    if ylim[1]: ax.set_ylim(ylim)
    else: ax.set_ylim(0, max(vals[m] for m in ms_plot if vals[m]==vals[m])*1.15)
fig.suptitle('Exp.2 Summary: ECOer achieves best validity with competitive proximity',
             fontsize=9, y=1.02)
fig.tight_layout()
save(fig, 'exp2_overall_summary.png')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Exp3 — DP, IM, Sparsity (2×2 layout)
# ═════════════════════════════════════════════════════════════════════════════
exp3 = load_exp('exp3_quality')
dp_by_ds  = agg_by_dataset(exp3, 'dp')
im_by_ds  = agg_by_dataset(exp3, 'im')
sp_by_ds  = agg_by_dataset(exp3, 'sparsity')

fig = plt.figure(figsize=(11, 7))
gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)
ax_dp = fig.add_subplot(gs[0, 0])
ax_im = fig.add_subplot(gs[0, 1])
ax_sp = fig.add_subplot(gs[1, 0])
ax_ov = fig.add_subplot(gs[1, 1])

grouped_bar(ax_dp, list(dp_by_ds.keys()), dp_by_ds,
            ylabel='DP (↑ better)', title='Discriminative Power (DP)')
grouped_bar(ax_im, list(im_by_ds.keys()), im_by_ds,
            ylabel='IM (↓ better)', title='Inlier Metric (IM)')
grouped_bar(ax_sp, list(sp_by_ds.keys()), sp_by_ds,
            ylabel='# Features (↓ better)', title='Sparsity')

# Overall means bar in bottom-right
ov_dp = agg_overall(exp3, 'dp')
ov_im = agg_overall(exp3, 'im')
x = np.arange(len(METHODS))
w = 0.30
b1 = ax_ov.bar(x - w/2, [ov_dp[m] for m in METHODS], w,
               color=[COLORS[m] for m in METHODS], alpha=0.88,
               edgecolor='white', label='DP', zorder=3)
b2 = ax_ov.bar(x + w/2, [ov_im[m] for m in METHODS], w,
               color=[COLORS[m] for m in METHODS], alpha=0.55,
               edgecolor='white', hatch='//', label='IM', zorder=3)
ax_ov.set_xticks(x)
ax_ov.set_xticklabels([M_LABEL[m] for m in METHODS], rotation=35, ha='right')
ax_ov.set_title('Overall DP (solid) & IM (hatched)', fontweight='bold')
ax_ov.set_ylabel('Score')

handles = [mpatches.Patch(color=COLORS[m], label=M_LABEL[m]) for m in METHODS]
fig.legend(handles=handles, ncol=7, loc='lower center',
           bbox_to_anchor=(0.5, -0.04), framealpha=0.95, fontsize=8)
fig.suptitle('Exp.3 — Data Quality Metrics: ECOer Dominates DP, IM, and Sparsity',
             fontsize=11, fontweight='bold', y=1.01)
save(fig, 'exp2_quality_2x2_dp_im.png')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Exp4 — Ablation: Classifiers + Architecture
# ═════════════════════════════════════════════════════════════════════════════
e4_path = os.path.join(BASE, 'exp4_ablation')
clf_abl  = json.load(open(os.path.join(e4_path,'ablation_classifiers.json')))
arch_abl = json.load(open(os.path.join(e4_path,'ablation_arch.json')))

class _MultiColorHandler(HandlerBase):
    """Legend handler that draws N equal-width color blocks side by side."""
    def __init__(self, colors, alpha=0.9, hatch=None):
        self._colors = colors
        self._alpha  = alpha
        self._hatch  = hatch
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        n = len(self._colors)
        w = width / n
        patches = []
        for i, c in enumerate(self._colors):
            r = Rectangle([i * w, 0], w, height,
                          facecolor=c, alpha=self._alpha,
                          hatch=self._hatch if self._hatch else '',
                          edgecolor='white', linewidth=0.4,
                          transform=trans)
            patches.append(r)
        return patches

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

# 4a: validity & DP per classifier
clf_names = list(clf_abl.keys())
clf_colors_abl = ['#4e79a7','#76b7b2','#f28e2b','#e15759']
x = np.arange(len(clf_names))
w = 0.32
ax = axes[0]
ax.bar(x-w/2, [clf_abl[c]['validity'] for c in clf_names], w,
       color=clf_colors_abl, alpha=0.9, edgecolor='white')
ax.bar(x+w/2, [clf_abl[c]['dp'] for c in clf_names], w,
       color=clf_colors_abl, alpha=0.55, hatch='//', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(clf_names, fontsize=7.5)
ax.set_ylabel('Score'); ax.set_ylim(0.75, 1.02)
ax.set_title('(a) Classifier Generality\n(ECOer on Adult)', fontweight='bold')
# Custom legend: 4-color block for Validity (solid) and DP (hatched)
_dummy_v = mpatches.Patch(facecolor='gray', label='Validity')
_dummy_d = mpatches.Patch(facecolor='gray', label='DP')
ax.legend(
    handles=[_dummy_v, _dummy_d],
    handler_map={
        _dummy_v: _MultiColorHandler(clf_colors_abl, alpha=0.9),
        _dummy_d: _MultiColorHandler(clf_colors_abl, alpha=0.55, hatch='//'),
    },
    fontsize=7.5, loc='upper left', handlelength=2.5, handleheight=1.2,
)

# 4b: validity vs m
m_vals = [int(k) for k in arch_abl.keys()]
ecor_v = [arch_abl[str(m)]['ECOer']['validity'] for m in m_vals]
dpmd_v = [arch_abl[str(m)]['DPMDCE']['validity'] for m in m_vals]
ecor_d = [arch_abl[str(m)]['ECOer']['dp'] for m in m_vals]
dpmd_d = [arch_abl[str(m)]['DPMDCE']['dp'] for m in m_vals]
ax = axes[1]
x2 = np.arange(len(m_vals)); w2 = 0.20
ax.bar(x2-1.5*w2, ecor_v, w2, color=COLORS['ECOer'],  alpha=0.9, label='ECOer  (Valid)', edgecolor='white')
ax.bar(x2-0.5*w2, ecor_d, w2, color=COLORS['ECOer'],  alpha=0.5, hatch='//', label='ECOer  (DP)', edgecolor='white')
ax.bar(x2+0.5*w2, dpmd_v, w2, color=COLORS['DPMDCE'], alpha=0.9, label='DPMDCE (Valid)', edgecolor='white')
ax.bar(x2+1.5*w2, dpmd_d, w2, color=COLORS['DPMDCE'], alpha=0.5, hatch='//', label='DPMDCE (DP)', edgecolor='white')
ax.set_xticks(x2); ax.set_xticklabels([f'm={v}' for v in m_vals])
ax.set_ylabel('Score'); ax.set_ylim(0.65, 1.02)
ax.set_title('(b) Hidden Size $m$ Effect', fontweight='bold')
ax.legend(fontsize=6.5, ncol=2)

# 4c: IM vs m (lower is better)
ecor_im = [arch_abl[str(m)]['ECOer']['im'] for m in m_vals]
dpmd_im = [arch_abl[str(m)]['DPMDCE']['im'] for m in m_vals]
ax = axes[2]
w3 = 0.30
ax.bar(x2-w3/2, ecor_im, w3, color=COLORS['ECOer'],  alpha=0.9, label='ECOer', edgecolor='white')
ax.bar(x2+w3/2, dpmd_im, w3, color=COLORS['DPMDCE'], alpha=0.9, label='DPMDCE', edgecolor='white')
ax.set_xticks(x2); ax.set_xticklabels([f'm={v}' for v in m_vals])
ax.set_ylabel('IM (↓ better)')
ax.set_title('(c) Inlier Metric vs. $m$', fontweight='bold')
ax.legend(fontsize=8)
fig.suptitle('Exp.4 — Ablation: Classifier Generality and Architecture',
             fontsize=10, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, 'exp4_ablation_classifiers.png')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Exp4 — Component ablation
# ═════════════════════════════════════════════════════════════════════════════
comp_abl = json.load(open(os.path.join(e4_path,'ablation_components.json')))
variants = list(comp_abl.keys())
comp_colors = ['#2ca02c','#ff7f0e','#9467bd','#d62728','#7f7f7f']
metrics_comp = [('validity','Validity (↑)'), ('dp','DP (↑)'),
                ('im','IM (↓)'), ('sparsity','Sparsity (↓)')]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
for ax, (mk, mlabel) in zip(axes, metrics_comp):
    vals = [comp_abl[v][mk] for v in variants]
    bars = ax.bar(np.arange(len(variants)), vals, color=comp_colors,
                  alpha=0.88, edgecolor='white', width=0.6, zorder=3)
    bars[0].set_edgecolor('#1a6e1a'); bars[0].set_linewidth(1.4)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(variants, rotation=28, ha='right', fontsize=7.5)
    ax.set_ylabel(mlabel, fontsize=8.5)
    ax.set_title(mlabel, fontweight='bold')
    # Annotate top value
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)
fig.suptitle('Exp.4 — Energy Component Ablation (Adult, KNN $k$=5)',
             fontsize=10, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, 'exp4_ablation_ecoe_components.png')

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Exp5 — Wilcoxon heatmap (bar chart version per baseline)
# ═════════════════════════════════════════════════════════════════════════════
import matplotlib.colors as mcolors

wil = json.load(open(os.path.join(BASE,'exp5_stats','wilcoxon_results.json')))
baselines = list(wil.keys())
metrics_w = ['validity','l1','l2','sparsity','dp','im']
m_labels_w = ['Validity','ℓ₁','ℓ₂','Sparsity','DP','IM']

# Build p-value matrix
p_mat = np.array([[wil[b][m]['p_value'] for m in metrics_w] for b in baselines])

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

# Left: grouped bar of p-values per metric
ax = axes[0]
x = np.arange(len(metrics_w))
bl_colors = [COLORS[b] for b in baselines]
w_bar = 0.12
for i, (b, c) in enumerate(zip(baselines, bl_colors)):
    pvals = [wil[b][m]['p_value'] for m in metrics_w]
    ax.bar(x + (i - len(baselines)/2 + 0.5)*w_bar, pvals, w_bar,
           color=c, alpha=0.85, edgecolor='white', label=M_LABEL[b], zorder=3)
ax.axhline(0.05, color='red', lw=1.2, ls='--', label='α = 0.05')
ax.set_xticks(x); ax.set_xticklabels(m_labels_w)
ax.set_ylabel('$p$-value (Wilcoxon)')
ax.set_title('(a) $p$-values: ECOer vs. Each Baseline', fontweight='bold')
ax.legend(fontsize=7, ncol=2)

# Right: heatmap
ax2 = axes[1]
cmap = plt.cm.RdYlGn_r
norm = mcolors.LogNorm(vmin=0.001, vmax=0.10)
im_h = ax2.imshow(p_mat, cmap=cmap, norm=norm, aspect='auto')
ax2.set_xticks(range(len(metrics_w))); ax2.set_xticklabels(m_labels_w, fontsize=9)
ax2.set_yticks(range(len(baselines))); ax2.set_yticklabels([M_LABEL[b] for b in baselines], fontsize=9)
for i in range(len(baselines)):
    for j in range(len(metrics_w)):
        p = p_mat[i,j]
        txt = f'{p:.3f}'
        star = '★' if p < 0.05 else ''
        ax2.text(j, i, f'{txt}\n{star}', ha='center', va='center',
                 fontsize=7, color='black' if p > 0.03 else 'white')
plt.colorbar(im_h, ax=ax2, label='$p$-value', shrink=0.85)
ax2.set_title('(b) $p$-value Heatmap (red = significant)', fontweight='bold')
ax2.spines[:].set_visible(False)

fig.suptitle('Exp.5 — Wilcoxon Signed-Rank Tests: ECOer vs. All Baselines',
             fontsize=10, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, 'exp5_wilcoxon_heatmap.png')

print('\nAll figures generated successfully.')
