"""Global configuration for ECOer experiments."""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
LATEX_DIR  = os.path.join(BASE_DIR, "..", "20250803_SNN_sub_PR_ver2")

# ── Datasets ───────────────────────────────────────────────────────────────────
DATASETS = ["heloc", "adult", "german_credit", "compas", "heart", "pima"]
DATASET_DISPLAY = {
    "heloc":          "HELOC",
    "adult":          "Adult",
    "german_credit":  "German Credit",
    "compas":         "COMPAS",
    "heart":          "Heart",
    "pima":           "Pima",
}

# ── Classifiers ────────────────────────────────────────────────────────────────
CLASSIFIERS = ["knn5", "rf", "svm"]
CLF_DISPLAY = {"knn5": "KNN (k=5)", "rf": "Random Forest", "svm": "Kernel SVM"}

# ── Baselines ──────────────────────────────────────────────────────────────────
BASELINES = ["dice", "face", "growing_spheres", "revise", "wach", "dpmdce"]
BASELINE_DISPLAY = {
    "dice":            "DiCE",
    "face":            "FACE",
    "growing_spheres": "GrowingSpheres",
    "revise":          "Revise",
    "wach":            "WACH",
    "dpmdce":          "DPMDCE",
}

# ── R2SNN / ECOer hyperparameters ──────────────────────────────────────────────
R2SNN_HIDDEN    = 30        # default hidden neurons m
R2SNN_EPOCHS    = 600
R2SNN_LR        = 1e-3
R2SNN_BATCH     = 512
R2SNN_PATIENCE  = 20
R2SNN_N_UNIFORM = 10000     # uniform samples in X
R2SNN_N_BOUNDARY = 1000     # boundary-augmentation points
R2SNN_ZETA1     = 0.1       # gradient-penalty weight
R2SNN_ZETA2     = 0.1       # consistency-term weight
R2SNN_GAMMA_CLIP = 1.0      # gradient magnitude threshold for R_grad
R2SNN_TAU_CLIP  = 1.0       # gradient clipping for AdamW
R2SNN_DELTA_C   = 10.0      # sharpness for R_cons

# ECOer counterfactual optimiser
LAMBDA1    = 0.50
LAMBDA2    = 0.40
BETA       = 0.60
CF_LR      = 0.01
CF_MAX_STEPS = 200

# ── Experiment settings ────────────────────────────────────────────────────────
N_TEST_INSTANCES = 100      # per dataset
N_CV_FOLDS       = 5
N_SEEDS          = 1        # for R2SNN approximation std estimation
SEED             = 42
M_VALUES         = [10, 20, 30, 40, 50]   # for Exp 1 arch sweep

# ── Plotting ───────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "ECOer":          "#2ca02c",   # green
    "DiCE":           "#1f77b4",
    "FACE":           "#ff7f0e",
    "GrowingSpheres": "#9467bd",
    "Revise":         "#8c564b",
    "WACH":           "#e377c2",
    "DPMDCE":         "#7f7f7f",
    "R2SNN":          "#2ca02c",
    "SingleReLU":     "#d62728",
}
FIG_DPI    = 300
FIG_FONT   = 11
