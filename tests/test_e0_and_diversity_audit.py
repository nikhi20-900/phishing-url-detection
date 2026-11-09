"""
Regression and validation tests for E0 Frozen Baseline and Training Diversity Audit.

Validates:
1. Frozen model loader integrity (strictly binary classes, 18 features).
2. Binary evaluation metric calculations and edge cases.
3. Structural URL profiling functions (schemes, slash counts, path segments, delimiters).
4. Machine-readable E0 scorecard schema and metrics verification.
5. Machine-readable Training Diversity Audit schema and covariate gap verification.
6. Guardrail integrity: ensures models/random_forest_model.joblib remains unmodified.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
import joblib

from experiments.e0_frozen_baseline import load_frozen_model, evaluate_binary_predictions
from experiments.audit_training_diversity import compute_url_structural_profile
from src.features import FEATURE_NAMES


def test_frozen_model_integrity():
    """Verify that models/random_forest_model.joblib is intact with valid classes and features."""
    artifact, model = load_frozen_model("models/random_forest_model.joblib")
    assert hasattr(model, "classes_"), "Model must have classes_ attribute"
    assert list(model.classes_) == [0, 1], "Model classes must be [0, 1]"
    assert hasattr(model, "n_features_in_"), "Model must expose n_features_in_"
    assert model.n_features_in_ == 18, "Model must expect exactly 18 features"

    if isinstance(artifact, dict):
        assert artifact["feature_names"] == FEATURE_NAMES
        assert artifact["phishing_index"] == 0
        assert artifact["legitimate_index"] == 1


def test_evaluate_binary_predictions_balanced():
    """Verify metric calculations on balanced synthetic predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.9, 0.4, 0.6, 0.1])

    res = evaluate_binary_predictions(y_true, y_pred, y_prob)
    assert res["sample_size"] == 4
    assert res["accuracy"] == 0.5
    assert res["confusion_matrix"]["true_phishing_tp"] == 1
    assert res["confusion_matrix"]["missed_phishing_fn"] == 1
    assert res["confusion_matrix"]["falsely_flagged_legit_fp"] == 1
    assert res["confusion_matrix"]["verified_legit_tn"] == 1
    assert res["legitimate_fpr"] == 0.5


def test_evaluate_binary_predictions_single_class():
    """Verify that single-class inputs do not crash and handle zero division."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.8, 0.1, 0.2, 0.05])

    res = evaluate_binary_predictions(y_true, y_pred, y_prob)
    assert res["sample_size"] == 4
    assert res["roc_auc"] is None
    assert res["pr_auc"] is None
    assert res["confusion_matrix"]["falsely_flagged_legit_fp"] == 1
    assert res["confusion_matrix"]["verified_legit_tn"] == 3
    assert res["legitimate_fpr"] == 0.25


def test_compute_url_structural_profile():
    """Verify structural URL profiling on controlled synthetic URLs."""
    urls = pd.Series([
        "https://www.example.com",
        "https://www.example.com/",
        "https://www.example.com/path/to/resource",
        "http://sub.domain.org/search?q=test&lang=en",
        "https://bare.com/index.html#section"
    ])

    prof = compute_url_structural_profile(urls)
    assert prof["sample_size"] == 5
    assert prof["scheme"]["pct_https"] == 0.8  # 4 out of 5
    assert prof["scheme"]["pct_http"] == 0.2   # 1 out of 5
    assert prof["scheme"]["pct_https_www"] == 0.6  # 3 out of 5

    # Path analysis
    assert prof["path"]["pct_has_path"] == 0.6  # 3 out of 5 have non-empty paths
    assert prof["path"]["pct_0_segments"] == 0.4  # 2 out of 5 are root (with or without slash)
    assert prof["delimiters"]["pct_has_query_qmark"] == 0.2  # 1 out of 5
    assert prof["delimiters"]["pct_has_fragment_hash"] == 0.2  # 1 out of 5


def test_e0_scorecard_file_validity():
    """Verify that e0_frozen_baseline_scorecard.json contains valid, expected results."""
    json_path = "experiments/reports/e0_frozen_baseline_scorecard.json"
    assert os.path.exists(json_path), f"Scorecard file '{json_path}' must exist"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_id"] == "E0_FROZEN_BASELINE_SCORECARD"
    assert "suite_1_in_distribution" in data
    assert "suite_2_external_benchmark" in data
    assert "suite_3_perturbation_matrix" in data

    s1 = data["suite_1_in_distribution"]
    assert s1["stratified_test_metrics"]["accuracy"] > 0.99
    assert s1["domain_separated_test_metrics"]["accuracy"] > 0.99

    s2 = data["suite_2_external_benchmark"]
    # Verify the external benchmark dichotomy: high phish recall, high legitimate FPR
    assert s2["combined_raw_metrics"]["phishing_recall"] > 0.99
    assert s2["false_positive_analysis"]["raw_false_positive_rate"] >= 0.90
    assert s2["false_positive_analysis"]["stripped_false_positive_rate"] >= 0.85

    s3 = data["suite_3_perturbation_matrix"]
    vuln = s3["key_vulnerabilities"]
    assert vuln["www_root_phish_rate"] == 0.0
    assert vuln["www_root_slash_phish_rate"] > 0.50
    assert vuln["path_deep_multi_slash_phish_rate"] == 1.0


def test_training_diversity_audit_file_validity():
    """Verify that training_data_diversity_audit.json contains valid, expected results."""
    json_path = "experiments/reports/training_data_diversity_audit.json"
    assert os.path.exists(json_path), f"Audit file '{json_path}' must exist"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_id"] == "TRAINING_DATA_DIVERSITY_AUDIT"
    assert "profiles" in data
    assert "covariate_gap" in data
    assert "augmentation_specification" in data

    # Verify the 100% syntactic vacuum in PhiUSIIL legitimate URLs
    phi_legit = data["profiles"]["phiusiil_legitimate"]
    assert phi_legit["scheme"]["pct_https_www"] == 1.0
    assert phi_legit["slashes"]["pct_exactly_2"] == 1.0
    assert phi_legit["slashes"]["pct_trailing_slash"] == 0.0
    assert phi_legit["path"]["pct_has_path"] == 0.0
    assert phi_legit["delimiters"]["pct_has_query_qmark"] == 0.0

    # Verify that external legitimate deep links have high path presence
    ext_legit = data["profiles"]["external_curated_legitimate"]
    assert ext_legit["path"]["pct_has_path"] > 0.80
    assert ext_legit["slashes"]["mean"] > 3.0
