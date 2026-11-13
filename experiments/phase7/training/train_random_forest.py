"""
Train Candidate Model E4: Targeted Augmented Random Forest for Phase 7 (Milestone 3).

Follows strict experimental governance:
1. Trains ONLY on `experiments/phase7/datasets/splits/train.csv`.
2. Preserves the exact 18 production features from `src/features.py`.
3. Pre-registers all hyperparameters (n_estimators=100, random_state=42, n_jobs=-1).
4. Stores candidate model in `models/phase7_candidates/rf_augmented_e4.joblib`.
5. Strictly prevents overwriting or modifying the frozen production model (`models/random_forest_model.joblib`).
6. Completely isolates `internal_test.csv` (never accessed during training).

Outputs:
- models/phase7_candidates/rf_augmented_e4.joblib
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

EXPECTED_TRAIN_SHA256 = "9297458d03841fbe9c32bae7c4ad323e1b3f0b221c0c755f912350808d0c6892"
EXPECTED_FROZEN_MODEL_SHA256 = "95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def train_candidate_e4(
    train_csv_path: Path,
    output_model_path: Path,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Train Phase 7 Candidate E4 Random Forest on train.csv."""
    print("=" * 70)
    print("Executing Phase 7 M3: Candidate Model Training (E4 Random Forest)...")
    print("=" * 70)

    # Invariant guard 1: Never overwrite production model
    prod_path = REPO_ROOT / "models" / "random_forest_model.joblib"
    if output_model_path.resolve() == prod_path.resolve():
        raise ValueError("CRITICAL INVARIANT VIOLATION: Cannot overwrite production model!")

    # Invariant guard 2: Verify production model remains frozen before training
    prod_sha = compute_file_sha256(prod_path)
    if prod_sha != EXPECTED_FROZEN_MODEL_SHA256:
        raise ValueError(f"CRITICAL INVARIANT VIOLATION: Production model SHA-256 mismatch!\nExpected: {EXPECTED_FROZEN_MODEL_SHA256}\nActual:   {prod_sha}")
    print(f"  [PASS] Production model frozen verified ({prod_sha[:16]}...).")

    # Invariant guard 3: Verify train.csv hash
    actual_train_sha = compute_file_sha256(train_csv_path)
    if actual_train_sha != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"CRITICAL INVARIANT VIOLATION: train.csv SHA-256 mismatch!\nExpected: {EXPECTED_TRAIN_SHA256}\nActual:   {actual_train_sha}")
    print(f"  [PASS] train.csv SHA-256 verified ({actual_train_sha[:16]}...).")

    # Load training data
    print(f"Loading training data from {train_csv_path.relative_to(REPO_ROOT)}...")
    train_df = pd.read_csv(train_csv_path)
    X_train = train_df[FEATURE_NAMES]
    y_train = train_df["label"].values.astype(int)

    n_phish = int((y_train == 0).sum())
    n_legit = int((y_train == 1).sum())
    n_benign_exp = int((train_df["source_dataset"] == "phase7_m1_2_2_benign").sum())

    print(f"Training dataset: {len(train_df)} samples (Phishing: {n_phish}, Legitimate: {n_legit}, Benign Expansion: {n_benign_exp})")

    # Fit Random Forest classifier
    print(f"Fitting RandomForestClassifier(n_estimators={n_estimators}, random_state={random_state}, n_jobs={n_jobs})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    clf.fit(X_train, y_train)
    print("  [PASS] Model fitting complete.")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "classes": clf.classes_,
        "phishing_index": int(list(clf.classes_).index(0)),
        "legitimate_index": int(list(clf.classes_).index(1)),
        "model_id": "E4",
        "model_name": "Phase 7 Targeted Augmented Random Forest",
        "model_family": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": n_jobs,
        },
        "training_metadata": {
            "train_samples": len(train_df),
            "train_phishing": n_phish,
            "train_legitimate": n_legit,
            "train_benign_expansion_count": n_benign_exp,
            "train_sha256": actual_train_sha,
            "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    joblib.dump(artifact, output_model_path)
    model_sha = compute_file_sha256(output_model_path)
    artifact["model_sha256"] = model_sha

    print(f"[E4 TRAINED] Saved candidate to {output_model_path.relative_to(REPO_ROOT)} (SHA-256: {model_sha[:16]}...)")

    # Invariant guard 4: Post-training check on production model
    prod_sha_after = compute_file_sha256(prod_path)
    if prod_sha_after != EXPECTED_FROZEN_MODEL_SHA256:
        raise ValueError("CRITICAL INVARIANT VIOLATION: Production model was altered during training!")
    print(f"  [PASS] Production model remains frozen post-training ({prod_sha_after[:16]}...).")

    return artifact


if __name__ == "__main__":
    train_csv = REPO_ROOT / "experiments" / "phase7" / "datasets" / "splits" / "train.csv"
    out_model = REPO_ROOT / "models" / "phase7_candidates" / "rf_augmented_e4.joblib"
    train_candidate_e4(train_csv, out_model)
