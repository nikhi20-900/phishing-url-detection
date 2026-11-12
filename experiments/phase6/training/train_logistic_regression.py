"""
Train Candidate Model E3: Logistic Regression with Validated Benign Augmentation.

Preserves the 18 production features, scales features with StandardScaler fit
strictly on training data, and trains a LogisticRegression estimator.
Saves candidate artifact to models/phase6_candidates/logistic_regression_augmented.joblib.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]


def train_candidate_lr(
    train_csv_path: Path,
    output_model_path: Path,
    c_val: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Logistic Regression candidate model E3 with pipeline scaling."""
    print("Training Phase 6 Candidate E3: Logistic Regression with Validated Augmentation...")

    # Invariant guard: Never write to production model
    prod_path = REPO_ROOT / "models" / "random_forest_model.joblib"
    if output_model_path.resolve() == prod_path.resolve():
        raise ValueError("CRITICAL INVARIANT VIOLATION: Cannot overwrite production model!")

    train_df = pd.read_csv(train_csv_path)
    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"].values.astype(int)

    train_sha = hashlib.sha256(train_csv_path.read_bytes()).hexdigest().lower()

    # Preprocessing strictly fit only on training data inside Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c_val, max_iter=max_iter, random_state=random_state)),
    ])
    pipe.fit(X_train, y_train)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    classes = pipe.named_steps["clf"].classes_
    artifact = {
        "model": pipe,
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "phishing_index": int(list(classes).index(0)),
        "legitimate_index": int(list(classes).index(1)),
        "model_id": "E3",
        "model_name": "Logistic Regression with Validated Benign Augmentation",
        "model_family": "LogisticRegression",
        "hyperparameters": {
            "C": c_val,
            "max_iter": max_iter,
            "random_state": random_state,
            "scaler": "StandardScaler",
        },
        "training_metadata": {
            "train_samples": len(train_df),
            "train_phishing": int((y_train == 0).sum()),
            "train_legitimate": int((y_train == 1).sum()),
            "train_pilot_count": int((train_df["source_dataset"] == "phase6_validated_pilot").sum()),
            "train_sha256": train_sha,
            "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    joblib.dump(artifact, output_model_path)
    model_sha = hashlib.sha256(output_model_path.read_bytes()).hexdigest().lower()
    print(f"[E3 TRAINED] Saved candidate to {output_model_path.relative_to(REPO_ROOT)} (SHA-256: {model_sha[:16]}...)")

    return artifact


if __name__ == "__main__":
    train_csv = REPO_ROOT / "experiments" / "phase6" / "datasets" / "splits" / "train.csv"
    out_model = REPO_ROOT / "models" / "phase6_candidates" / "logistic_regression_augmented.joblib"
    train_candidate_lr(train_csv, out_model)
