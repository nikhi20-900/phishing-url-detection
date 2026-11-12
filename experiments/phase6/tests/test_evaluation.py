"""
Tests for Phase 6 Milestones 5, 6, 7: Evaluation, External Confirmation, and Statistics.
"""

import json
from pathlib import Path
import pytest

from experiments.phase6.verify_m0_contract import (
    REPO_ROOT,
    EXPECTED_HASHES,
    compute_sha256,
)


def test_baseline_results_artifact():
    """Verify that M3 baseline results artifact exists and contains valid metrics."""
    res_path = REPO_ROOT / "experiments" / "phase6" / "reports" / "baseline_results.json"
    assert res_path.exists(), "baseline_results.json missing"
    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["milestone"] == "M3"
    assert data["status"] == "APPROVED"
    assert "evaluation_results" in data
    m = data["evaluation_results"]["classification_metrics"]
    assert 0.95 <= m["phishing_recall"] <= 1.0
    assert 0.95 <= m["legitimate_recall"] <= 1.0


def test_candidate_comparison_artifact():
    """Verify that M5 candidate comparison artifact exists and contains all 4 models."""
    res_path = REPO_ROOT / "experiments" / "phase6" / "reports" / "candidate_comparison.json"
    assert res_path.exists(), "candidate_comparison.json missing"
    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["milestone"] == "M5"
    assert "candidate_results" in data
    cands = data["candidate_results"]
    for mid in ["E0", "E1", "E2", "E3"]:
        assert mid in cands, f"Model {mid} missing from candidate comparison"
        assert "fixed_threshold_0_50" in cands[mid]
        assert "validation_tuning" in cands[mid]


def test_confirmatory_external_evaluation_artifact():
    """Verify that M6 external confirmatory evaluation preserves the benchmark hash."""
    res_path = REPO_ROOT / "experiments" / "phase6" / "reports" / "confirmatory_external_evaluation.json"
    assert res_path.exists(), "confirmatory_external_evaluation.json missing"
    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["milestone"] == "M6"
    assert data["benchmark_sha256"] == EXPECTED_HASHES["phase4_benchmark"]["expected_sha256"]
    assert data["benchmark_sample_count"] == 1047

    comp = data["comparison_results"]
    assert "E0" in comp and "E1" in comp
    # E0 reproduced historical 94% Curated FPR
    e0_cur_fpr = comp["E0"]["source_slices"]["Curated-Authoritative-Web-Corpus"]["legitimate_fpr"]
    assert e0_cur_fpr == 0.9400


def test_statistical_analysis_artifact():
    """Verify that M7 statistical analysis contains McNemar and Bootstrap tests."""
    res_path = REPO_ROOT / "experiments" / "phase6" / "reports" / "statistical_analysis.json"
    assert res_path.exists(), "statistical_analysis.json missing"
    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["milestone"] == "M7"
    comp = data["candidate_statistical_comparisons"]
    assert "E1" in comp
    mcnemar = comp["E1"]["mcnemar_test_vs_E0"]
    assert "statistic" in mcnemar
    assert "p_value" in mcnemar
    assert mcnemar["statistically_significant_p05"] is True
