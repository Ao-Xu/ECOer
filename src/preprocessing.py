"""
Preprocessing: encode categoricals, normalise to [-1, 1], train/test split.
Saves/loads processed splits as compressed NPZ files.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PROC_DIR = os.path.join(config.DATA_DIR, "processed")
os.makedirs(PROC_DIR, exist_ok=True)


def preprocess(
    df: pd.DataFrame,
    dataset_name: str,
    test_size: float = 0.2,
    seed: int = config.SEED,
) -> dict:
    """
    Encode, normalise, split.

    Returns a dict with:
        X_train, X_test  : np.ndarray float32 in [-1, 1]^d
        y_train, y_test  : np.ndarray int
        feature_names    : list[str]
        scaler           : fitted MinMaxScaler (feature_range=(-1,1))
        cov_matrix       : np.ndarray (d, d) covariance of X_train for Mahalanobis
        d                : int (feature dimension)
    """
    df = df.copy()
    y = df.pop("target").values.astype(int)

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    df = df.apply(pd.to_numeric, errors="coerce")
    df.fillna(df.median(), inplace=True)
    X = df.values.astype(np.float32)
    feature_names = df.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Scale to [-1, 1] based on training set statistics
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # Covariance matrix of training data (for Mahalanobis distance)
    cov_matrix = np.cov(X_train.T).astype(np.float32)

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "feature_names": feature_names,
        "scaler":        scaler,
        "cov_matrix":    cov_matrix,
        "d":             X_train.shape[1],
    }


def save_processed(data: dict, dataset_name: str) -> None:
    path = os.path.join(PROC_DIR, f"{dataset_name}.npz")
    np.savez_compressed(
        path,
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=data["y_train"],
        y_test=data["y_test"],
        cov_matrix=data["cov_matrix"],
        feature_names=np.array(data["feature_names"]),
    )
    # Save scaler separately using joblib
    import joblib
    joblib.dump(data["scaler"], os.path.join(PROC_DIR, f"{dataset_name}_scaler.joblib"))


def load_processed(dataset_name: str) -> dict:
    path = os.path.join(PROC_DIR, f"{dataset_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found for '{dataset_name}'. Run preprocessing first."
        )
    npz = np.load(path, allow_pickle=True)
    import joblib
    scaler = joblib.load(os.path.join(PROC_DIR, f"{dataset_name}_scaler.joblib"))
    data = {
        "X_train":       npz["X_train"],
        "X_test":        npz["X_test"],
        "y_train":       npz["y_train"],
        "y_test":        npz["y_test"],
        "cov_matrix":    npz["cov_matrix"],
        "feature_names": npz["feature_names"].tolist(),
        "scaler":        scaler,
        "d":             npz["X_train"].shape[1],
    }
    return data


def setup_all_datasets() -> None:
    """Load raw data, preprocess, and save all 6 datasets."""
    import importlib
    dl = importlib.import_module("src.data_loader")

    for name in config.DATASETS:
        proc_path = os.path.join(PROC_DIR, f"{name}.npz")
        if os.path.exists(proc_path):
            print(f"  [{name}] already processed — skipping")
            continue
        print(f"  [{name}] loading and preprocessing ...")
        try:
            df, _ = dl.load_dataset(name)
            data = preprocess(df, name)
            save_processed(data, name)
            print(f"  [{name}] done — X_train {data['X_train'].shape}, d={data['d']}")
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")
