"""
Train Candidate Model E1: Random Forest with Validated Benign Augmentation.

Preserves the 18 production features and trains a new candidate RandomForestClassifier
on the domain-separated training split (PhiUSIIL train + Validated Pilot train).
Saves candidate artifact to models/phase6_candidates/rf_augmented.joblib.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features import FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]


def train_candidate_rf(
    train_csv_path: Path,
    output_model_path: Path,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Train Random Forest candidate model E1."""
    print("Training Phase 6 Candidate E1: Random Forest with Validated Augmentation...")
    
    # Invariant guard: Never write to production model
    prod_path = REPO_ROOT / "models" / "random_forest_model.joblib"
    if output_model_path.resolve() == prod_path.resolve():
        raise ValueError("CRITICAL INVARIANT VIOLATION: Cannot overwrite production model!")

    train_df = pd.read_csv(train_csv_path)
    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"].values.astype(int)

    train_sha = hashlib.sha256(train_csv_path.read_bytes()).hexdigest().lower()

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    clf.fit(X_train, y_train)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "classes": clf.classes_,
        "phishing_index": int(list(clf.classes_).index(0)),
        "legitimate_index": int(list(clf.classes_).index(1)),
        "model_id": "E1",
        "model_name": "Random Forest with Validated Benign Augmentation",
        "model_family": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": n_jobs,
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
    print(f"[E1 TRAINED] Saved candidate to {output_model_path.relative_to(REPO_ROOT)} (SHA-256: {model_sha[:16]}...)")
    
    return artifact


if __name__ == "__main__":
    train_csv = REPO_ROOT / "experiments" / "phase6" / "datasets" / "splits" / "train.csv"
    out_model = REPO_ROOT / "models" / "phase6_candidates" / "rf_augmented.joblib"
    train_candidate_rf(train_csv, out_model)
