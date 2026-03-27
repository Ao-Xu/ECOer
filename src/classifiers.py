"""
Target classifier training and persistence.
Supports: knn5 (KNN k=5), rf (Random Forest), svm (Kernel SVM).
"""
import os
import sys
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CLF_DIR = os.path.join(config.MODELS_DIR, "classifiers")
os.makedirs(CLF_DIR, exist_ok=True)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    clf_name: str,
    seed: int = config.SEED,
):
    """
    Train and return a fitted sklearn classifier.

    clf_name options:
        'knn5' → KNeighborsClassifier(n_neighbors=5)
        'rf'   → RandomForestClassifier(n_estimators=100)
        'svm'  → SVC(kernel='rbf', probability=True)
    """
    if clf_name == "knn5":
        clf = KNeighborsClassifier(n_neighbors=5, weights="uniform", n_jobs=-1)
    elif clf_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        )
    elif clf_name == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier '{clf_name}'")
    clf.fit(X_train, y_train)
    return clf


def eval_classifier(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1":       float(f1_score(y_test, y_pred, zero_division=0)),
    }


def clf_path(dataset_name: str, clf_name: str) -> str:
    return os.path.join(CLF_DIR, f"{dataset_name}_{clf_name}.joblib")


def save_classifier(clf, dataset_name: str, clf_name: str) -> None:
    joblib.dump(clf, clf_path(dataset_name, clf_name))


def load_classifier(dataset_name: str, clf_name: str):
    p = clf_path(dataset_name, clf_name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Classifier not found: {p}")
    return joblib.load(p)


def get_or_train_classifier(
    dataset_name: str,
    clf_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = config.SEED,
):
    """Load cached classifier or train and save a new one."""
    p = clf_path(dataset_name, clf_name)
    if os.path.exists(p):
        return joblib.load(p)
    clf = train_classifier(X_train, y_train, clf_name, seed)
    save_classifier(clf, dataset_name, clf_name)
    return clf


def setup_all_classifiers(data_map: dict) -> None:
    """
    data_map: {dataset_name: processed_data_dict}
    Trains and saves all 6 × 3 classifiers (skips existing ones).
    """
    for ds_name, data in data_map.items():
        for clf_name in config.CLASSIFIERS:
            p = clf_path(ds_name, clf_name)
            if os.path.exists(p):
                clf = joblib.load(p)
                metrics = eval_classifier(clf, data["X_test"], data["y_test"])
                print(f"  [{ds_name}/{clf_name}] cached — acc={metrics['accuracy']:.3f}")
                continue
            print(f"  [{ds_name}/{clf_name}] training ...")
            clf = train_classifier(data["X_train"], data["y_train"], clf_name)
            save_classifier(clf, ds_name, clf_name)
            metrics = eval_classifier(clf, data["X_test"], data["y_test"])
            print(f"  [{ds_name}/{clf_name}] done — acc={metrics['accuracy']:.3f}")
