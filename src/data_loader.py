"""
Dataset loading for ECOer experiments.
Supports: heloc, adult, german_credit, compas, heart, pima

Download strategy (in order of priority):
  1. Look for CSV in data/raw/<name>.csv  (manually placed)
  2. Try ucimlrepo for UCI datasets
  3. Fall back to direct URL download (sklearn fetch_openml / pandas read_csv)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RAW_DIR = os.path.join(config.DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)


def load_dataset(name: str) -> tuple[pd.DataFrame, str]:
    """
    Load a named dataset.

    Returns
    -------
    df : pd.DataFrame
        Raw dataframe with a column named 'target' (int 0/1).
    label_col : str
        Always 'target'.
    """
    loaders = {
        "heloc":         _load_heloc,
        "adult":         _load_adult,
        "german_credit": _load_german_credit,
        "compas":        _load_compas,
        "heart":         _load_heart,
        "pima":          _load_pima,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders)}")
    return loaders[name](), "target"


# ──────────────────────────────────────────────────────────────────────────────
# Individual loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_heloc() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "heloc.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Standardise target column
        if "RiskPerformance" in df.columns:
            df["target"] = (df["RiskPerformance"] == "Good").astype(int)
            df.drop(columns=["RiskPerformance"], inplace=True)
        elif "target" not in df.columns:
            raise ValueError("HELOC CSV has no 'RiskPerformance' or 'target' column.")
        # Replace sentinel -9 with NaN then fill median
        df.replace(-9, np.nan, inplace=True)
        for col in df.columns:
            if col != "target":
                df[col].fillna(df[col].median(), inplace=True)
        return df
    # Fallback: try fetch_openml
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="HELOC", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        rp = df["RiskPerformance"].astype(str).str.strip()
        # openml encodes Good=1, Bad=0 (or string "Good"/"Bad")
        if rp.isin(["Good", "Bad"]).any():
            df["target"] = (rp == "Good").astype(int)
        else:
            df["target"] = rp.astype(int)  # '0'/'1' categorical
        df.drop(columns=["RiskPerformance"], inplace=True)
        df.replace(-9, np.nan, inplace=True)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].median(), inplace=True)
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        raise RuntimeError(
            f"HELOC dataset not found at {path}. "
            "Download from https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc "
            f"and place as {path}. Original error: {e}"
        )


def _load_adult() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "adult.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "income" in df.columns:
            df["target"] = df["income"].str.strip().isin([">50K", ">50K."]).astype(int)
            df.drop(columns=["income"], inplace=True)
        return df
    try:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        X = adult.data.features.copy()
        y = adult.data.targets.copy()
        X.columns = [c.replace(" ", "_") for c in X.columns]
        df = X.copy()
        income_col = y.columns[0]
        df["target"] = y[income_col].str.strip().isin([">50K", ">50K."]).astype(int)
        df.to_csv(path, index=False)
        return df
    except Exception:
        pass
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
        df = data.frame.copy()
        df["target"] = (df["class"].str.strip() == ">50K").astype(int)
        df.drop(columns=["class"], inplace=True)
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        raise RuntimeError(f"Adult dataset unavailable. Error: {e}")


def _load_german_credit() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "german_credit.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "target" not in df.columns:
            # Last column is the label (1=good, 2=bad)
            last = df.columns[-1]
            df["target"] = (df[last] == 1).astype(int)
            df.drop(columns=[last], inplace=True)
        return df
    try:
        from ucimlrepo import fetch_ucirepo
        gc = fetch_ucirepo(id=144)
        X = gc.data.features.copy()
        y = gc.data.targets.copy()
        df = X.copy()
        label_col = y.columns[0]
        df["target"] = (y[label_col] == 1).astype(int)
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        raise RuntimeError(f"German Credit dataset unavailable. Error: {e}")


def _load_compas() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "compas.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "target" not in df.columns:
            if "two_year_recid" in df.columns:
                df["target"] = df["two_year_recid"].astype(int)
                df.drop(columns=["two_year_recid"], inplace=True)
        return df
    # Try ProPublica raw URL
    try:
        url = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
               "master/compas-scores-two-years.csv")
        df = pd.read_csv(url)
        # Keep columns used in original paper
        keep = ["age", "c_charge_degree", "race", "age_cat", "score_text",
                "sex", "priors_count", "days_b_screening_arrest",
                "decile_score", "is_recid", "two_year_recid", "c_jail_in",
                "c_jail_out"]
        available = [c for c in keep if c in df.columns]
        df = df[available].copy()
        df = df[df["days_b_screening_arrest"] <= 30].copy()
        df = df[df["days_b_screening_arrest"] >= -30].copy()
        df = df[df["is_recid"] != -1].copy()
        df = df[df["c_charge_degree"] != "O"].copy()
        df["target"] = df["two_year_recid"].astype(int)
        drop_cols = ["two_year_recid", "is_recid", "c_jail_in", "c_jail_out"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        raise RuntimeError(
            f"COMPAS dataset not found at {path}. "
            "Place compas.csv (with 'two_year_recid' column) there. "
            f"Error: {e}"
        )


def _load_heart() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "heart.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "target" not in df.columns:
            last = df.columns[-1]
            df["target"] = (df[last] > 0).astype(int)
            df.drop(columns=[last], inplace=True)
        return df
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        X = heart.data.features.copy()
        y = heart.data.targets.copy()
        df = X.copy()
        label_col = y.columns[0]
        df["target"] = (y[label_col] > 0).astype(int)
        # Drop rows with '?' missing values
        df.replace("?", np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df.astype(float)
        df["target"] = df["target"].astype(int)
        df.to_csv(path, index=False)
        return df
    except Exception:
        pass
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="heart-disease", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        label_col = data.target_names[0] if hasattr(data, "target_names") else df.columns[-1]
        df["target"] = (df[label_col].astype(float) > 0).astype(int)
        if label_col != "target":
            df.drop(columns=[label_col], inplace=True)
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        raise RuntimeError(f"Heart Disease dataset unavailable. Error: {e}")


def _load_pima() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "pima.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "target" not in df.columns:
            if "Outcome" in df.columns:
                df["target"] = df["Outcome"].astype(int)
                df.drop(columns=["Outcome"], inplace=True)
        return df
    try:
        url = ("https://raw.githubusercontent.com/jbrownlee/"
               "Datasets/master/pima-indians-diabetes.data.csv")
        cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
        df = pd.read_csv(url, header=None, names=cols)
        df["target"] = df["Outcome"].astype(int)
        df.drop(columns=["Outcome"], inplace=True)
        df.to_csv(path, index=False)
        return df
    except Exception:
        pass
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
        label_col = df.columns[-1]
        df["target"] = (df[label_col].astype(str).str.strip() == "tested_positive").astype(int)
        df.drop(columns=[label_col], inplace=True)
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        raise RuntimeError(f"Pima Diabetes dataset unavailable. Error: {e}")
