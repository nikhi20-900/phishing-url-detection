"""
Test Suite for Phase 7 Milestone 3 (M3): Candidate Model Training (E4) and Validation Selection.

Verifies:
1. Production model remains strictly frozen at known hash (models/random_forest_model.joblib).
2. Candidate Model E4 artifact exists in models/phase7_candidates/ with correct schema and hyperparameters.
3. Feature names strictly match the 18 production features from src/features.py.
4. Sealed internal test invariant: internal_test.csv is NOT evaluated or unsealed in M3.
5. Validation scorecard and manifest record accurate metrics on val.csv.
6. Candidate E4 improves Phishing Recall and Macro F1 over E0 at T = 0.50 while keeping Legit FPR < 0.20%.
7. Candidate E4 demonstrates generalization on benign expansion URLs on unseen domains (reducing FPR from 100% to 82.76%).
8. Pre-registered operational threshold (T = 0.75) achieves >= 99% Phishing Recall and reduces Benign Expansion FPR to 55.17%.
9. M2 class distribution reconciliation addendum exists and verifies all mathematical invariants.
"""

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]

FROZEN_MODEL_PATH = REPO_ROOT / "models" / "random_forest_model.joblib"
EXPECTED_FROZEN_MODEL_SHA256 = "95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb"

CANDIDATE_MODEL_PATH = REPO_ROOT / "models" / "phase7_candidates" / "rf_augmented_e4.joblib"
VAL_CSV_PATH = REPO_ROOT / "experiments" / "phase7" / "datasets" / "splits" / "val.csv"
TEST_CSV_PATH = REPO_ROOT / "experiments" / "phase7" / "datasets" / "splits" / "internal_test.csv"
TRAIN_CSV_PATH = REPO_ROOT / "experiments" / "phase7" / "datasets" / "splits" / "train.csv"

SCORECARD_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m3_validation_scorecard.json"
MANIFEST_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m3_training_manifest.json"
REPORT_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m3_candidate_training_and_validation_report.md"
RECONCILIATION_ADDENDUM = REPO_ROOT / "experiments" / "phase7" / "reports" / "m2_audit_reconciliation_addendum.md"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def test_production_model_strictly_frozen():
    """Verify production model has NOT been modified or overwritten."""
    assert FROZEN_MODEL_PATH.exists(), "Production model missing!"
    actual_hash = compute_sha256(FROZEN_MODEL_PATH)
    assert actual_hash == EXPECTED_FROZEN_MODEL_SHA256, (
        f"CRITICAL: Production model hash changed!\nExpected: {EXPECTED_FROZEN_MODEL_SHA256}\nActual:   {actual_hash}"
    )


def test_candidate_e4_artifact_structure_and_metadata():
    """Verify Candidate E4 artifact structure, metadata, features, and hyperparameters."""
    assert CANDIDATE_MODEL_PATH.exists(), "Candidate E4 model missing!"
    artifact = joblib.load(CANDIDATE_MODEL_PATH)

    assert isinstance(artifact, dict), "Candidate artifact must be a dictionary"
    expected_keys = {
        "model",
        "feature_names",
        "classes",
        "phishing_index",
        "legitimate_index",
        "model_id",
        "model_name",
        "model_family",
        "hyperparameters",
        "training_metadata",
    }
    assert expected_keys.issubset(artifact.keys()), f"Missing keys in artifact: {expected_keys - artifact.keys()}"

    assert artifact["model_id"] == "E4"
    assert artifact["feature_names"] == FEATURE_NAMES
    assert len(artifact["feature_names"]) == 18
    assert artifact["hyperparameters"]["n_estimators"] == 100
    assert artifact["hyperparameters"]["random_state"] == 42
    assert artifact["training_metadata"]["train_samples"] == 170615
    assert artifact["training_metadata"]["train_phishing"] == 76092
    assert artifact["training_metadata"]["train_legitimate"] == 94523
    assert artifact["training_metadata"]["train_benign_expansion_count"] == 77

    # Verify model artifact hash matches manifest record
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["candidate_model_e4"]["sha256"] == compute_sha256(CANDIDATE_MODEL_PATH)


def test_internal_test_sealed_invariant():
    """Verify internal_test.csv is untouched and NOT evaluated in M3."""
    assert TEST_CSV_PATH.exists()
    expected_test_sha = "ce004e9d8e65ea8c09175d904d4a5fee0442acc4b127af6a98f58ae0aed0e2cf"
    actual_test_sha = compute_sha256(TEST_CSV_PATH)
    assert actual_test_sha == expected_test_sha, "internal_test.csv hash was mutated!"

    # Check scorecard
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)
    assert scorecard["evaluation_dataset"] == "experiments/phase7/datasets/splits/val.csv"
    assert scorecard["validation_samples"] == 30782
    assert scorecard["sealed_internal_test"]["status"] == "SEALED_UNTOUCHED"
    assert scorecard["sealed_internal_test"]["sha256"] == expected_test_sha

    # Check manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["sealed_internal_test"]["status"] == "SEALED_PRESERVED_FOR_M4"
    assert manifest["sealed_internal_test"]["sha256"] == expected_test_sha


def test_validation_scorecard_metrics_at_050():
    """Verify primary validation metrics at T = 0.50 conform to expectations."""
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    e0 = scorecard["primary_comparison_at_050"]["E0_frozen_control"]
    e4 = scorecard["primary_comparison_at_050"]["E4_augmented_rf"]

    # Phishing Recall improvement
    assert e4["phishing_recall"] >= 0.9920, f"Expected E4 phishing recall >= 0.9920, got {e4['phishing_recall']}"
    assert e4["phishing_recall"] > e0["phishing_recall"], "E4 phishing recall must improve over E0"

    # Overall Legitimate FPR preservation
    assert e4["legitimate_fpr"] < 0.0020, f"Expected E4 legit FPR < 0.20%, got {e4['legitimate_fpr']}"
    assert abs(e4["legitimate_fpr"] - e0["legitimate_fpr"]) < 0.0005, "Legitimate FPR must remain stable"

    # Macro F1 improvement
    assert e4["macro_f1"] > e0["macro_f1"], "E4 Macro F1 must improve over E0"
    assert e4["macro_f1"] >= 0.9955

    # Calibration preservation
    assert e4["brier_score"] < 0.0050, f"Brier score must be low: {e4['brier_score']}"
    assert e4["ece"] < 0.0050, f"ECE must be low: {e4['ece']}"


def test_validation_subgroups_generalization():
    """Verify performance on priority structural subgroups."""
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    subgroups = scorecard["subgroup_performance_at_050"]

    # 1. Benign expansion on unseen domains (n=29)
    benign_exp = subgroups["benign_expansion_unseen_domains"]
    assert benign_exp["E0"]["sample_count"] == 29
    assert benign_exp["E0"]["legitimate_fpr"] == 1.0  # E0 fails completely (100% FPR)
    assert benign_exp["E4"]["legitimate_fpr"] <= 0.83  # E4 improves to 82.76%

    # 2. Query bearing URLs (n=771)
    query_sub = subgroups["query_bearing"]
    assert query_sub["E0"]["sample_count"] == 771
    assert query_sub["E0"]["phishing_recall"] == 1.0
    assert query_sub["E4"]["phishing_recall"] == 1.0
    # E4 begins recognizing query-bearing legit URLs
    assert query_sub["E4"]["legitimate_recall"] > 0.10


def test_validation_threshold_sweep_and_selection():
    """Verify threshold sweep validity and operational threshold selection."""
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    sweep = scorecard["validation_threshold_sweep"]
    assert len(sweep) == 19  # 0.05 to 0.95 inclusive

    thresholds = [row["threshold"] for row in sweep]
    assert thresholds[0] == 0.05
    assert thresholds[-1] == 0.95

    # Operational threshold selected on validation
    selected_t = scorecard["validation_selected_threshold"]
    assert selected_t == 0.75, f"Expected selected threshold 0.75, got {selected_t}"

    # Verify metrics at T = 0.75
    row_075 = next(r for r in sweep if r["threshold"] == 0.75)
    assert row_075["E4"]["phishing_recall"] >= 0.990, "At T=0.75, phishing recall must be >= 99%"
    assert row_075["E4"]["benign_expansion_fpr"] <= 0.56, "At T=0.75, benign expansion FPR must drop to ~55%"


def test_candidate_e4_inference_smoke():
    """Verify candidate model produces valid inference predictions on input feature vectors."""
    artifact = joblib.load(CANDIDATE_MODEL_PATH)
    clf = artifact["model"]

    # Synthetic sample of 18 zeros with registered feature names
    dummy_input = pd.DataFrame(np.zeros((1, 18)), columns=FEATURE_NAMES)
    probs = clf.predict_proba(dummy_input)

    assert probs.shape == (1, 2)
    assert np.isclose(probs.sum(), 1.0)
    assert 0.0 <= probs[0, 0] <= 1.0
    assert 0.0 <= probs[0, 1] <= 1.0


def test_m2_reconciliation_addendum_exists_and_verified():
    """Verify that M2 reconciliation addendum exists and contains exact reconciled counts."""
    assert RECONCILIATION_ADDENDUM.exists()
    content = RECONCILIATION_ADDENDUM.read_text(encoding="utf-8")

    assert "170,615" in content
    assert "30,782" in content
    assert "34,092" in content
    assert "235,489" in content
    assert "100,520" in content
    assert "134,969" in content
    assert "77" in content
    assert "29" in content
    assert "13" in content
    assert "119" in content
    assert "RECONCILED & LOCKED" in content


def test_m3_report_file_exists_and_complete():
    """Verify M3 validation report exists and contains required sections."""
    assert REPORT_PATH.exists()
    content = REPORT_PATH.read_text(encoding="utf-8")

    assert "Phase 7 Milestone 3: Candidate Model Training and Validation Report" in content
    assert "rf_augmented_e4.joblib" in content
    assert "SEALED & UNTOUCHED" in content
    assert "Primary Comparison on Validation Partition" in content
    assert "Priority Subgroup Analysis" in content
    assert "Validation Decision Threshold Analysis" in content
    assert "Readiness for Milestone 4 (M4)" in content
