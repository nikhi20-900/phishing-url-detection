"""
Tests for Phase 6 Milestone 4: Experimental Model Training.
"""

from pathlib import Path
import joblib
import pytest

from experiments.phase6.verify_m0_contract import (
    REPO_ROOT,
    EXPECTED_HASHES,
    compute_sha256,
)
from src.features import FEATURE_NAMES


def test_production_model_strictly_frozen():
    """Verify that models/random_forest_model.joblib has not been altered during training."""
    prod_path = REPO_ROOT / "models" / "random_forest_model.joblib"
    assert compute_sha256(prod_path) == EXPECTED_HASHES["frozen_model"]["expected_sha256"]


def test_candidate_artifacts_exist_and_isolated():
    """Verify that candidate artifacts E1, E2, E3 exist in models/phase6_candidates/."""
    candidates_dir = REPO_ROOT / "models" / "phase6_candidates"
    expected_files = [
        "rf_augmented.joblib",
        "xgboost_augmented.joblib",
        "logistic_regression_augmented.joblib",
    ]
    for cfile in expected_files:
        cpath = candidates_dir / cfile
        assert cpath.exists(), f"Candidate artifact {cfile} missing!"

        # Verify not identical to production model
        assert compute_sha256(cpath) != EXPECTED_HASHES["frozen_model"]["expected_sha256"]


def test_candidate_model_loading_and_prediction():
    """Verify that candidate models load and predict using the 18 production features."""
    from src.features import extract_url_features
    import pandas as pd

    sample_url = "https://www.harvard.edu/academics/undergraduate-study/"
    feats = extract_url_features(sample_url)
    X_sample = pd.DataFrame([feats], columns=FEATURE_NAMES)

    candidates = [
        ("E1", "rf_augmented.joblib"),
        ("E2", "xgboost_augmented.joblib"),
        ("E3", "logistic_regression_augmented.joblib"),
    ]

    for mid, filename in candidates:
        if filename == "xgboost_augmented.joblib":
            pytest.importorskip("xgboost")
        artifact = joblib.load(REPO_ROOT / "models" / "phase6_candidates" / filename)
        assert isinstance(artifact, dict)
        assert "model" in artifact
        assert artifact["model_id"] == mid
        assert artifact["feature_names"] == FEATURE_NAMES

        model = artifact["model"]
        probs = model.predict_proba(X_sample)
        assert probs.shape == (1, 2)
        assert 0.0 <= probs[0, 0] <= 1.0
        assert 0.0 <= probs[0, 1] <= 1.0
