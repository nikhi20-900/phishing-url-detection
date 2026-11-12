"""
Tests for Phase 7 Milestone 0 (M0): Research Freeze, Artifact Integrity, and Governance Contract.
"""

import copy
import hashlib
import json
from pathlib import Path
import pytest
import yaml

from experiments.phase7.verify_m0_contract import (
    REPO_ROOT,
    EXPECTED_HASHES,
    REQUIRED_HISTORICAL_REPORTS,
    compute_sha256,
    verify_protocol_governance,
    verify_frozen_model_prediction,
    run_m0_verification,
)


def test_frozen_production_model_hash_invariant():
    """Verify that models/random_forest_model.joblib matches the immutable Phase 3 reference hash."""
    model_path = REPO_ROOT / "models" / "random_forest_model.joblib"
    actual_hash = compute_sha256(model_path)
    assert actual_hash == EXPECTED_HASHES["frozen_production_model"]["expected_sha256"]


def test_phiusiil_source_hash_invariant():
    """Verify that data/PhiUSIIL_Phishing_URL_Dataset.csv matches the immutable source hash."""
    phi_path = REPO_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    actual_hash = compute_sha256(phi_path)
    assert actual_hash == EXPECTED_HASHES["phiusiil_source"]["expected_sha256"]


def test_phase4_benchmark_hash_and_quarantine_status():
    """Verify that Phase 4 external benchmark hash matches and role is quarantined as historical."""
    bench_path = REPO_ROOT / "data" / "external" / "external_validation_benchmark.csv"
    actual_hash = compute_sha256(bench_path)
    assert actual_hash == EXPECTED_HASHES["phase4_benchmark"]["expected_sha256"]
    assert EXPECTED_HASHES["phase4_benchmark"]["role"] == "historical_comparison_only_not_untouched_confirmatory"


def test_phase5_pilot_hashes_invariant():
    """Verify that Phase 5 approved pilot corpus and review queue match expected hashes."""
    pilot_path = REPO_ROOT / "experiments" / "generalization" / "datasets" / "approved_pilot_corpus.csv"
    queue_path = REPO_ROOT / "experiments" / "generalization" / "datasets" / "pilot_review_queue.csv"
    assert compute_sha256(pilot_path) == EXPECTED_HASHES["phase5_approved_pilot"]["expected_sha256"]
    assert compute_sha256(queue_path) == EXPECTED_HASHES["phase5_review_queue"]["expected_sha256"]


def test_phase6_artifacts_hashes_invariant():
    """Verify that Phase 6 validated corpus and candidate models match their immutable hashes."""
    p6_corpus = REPO_ROOT / "experiments" / "phase6" / "datasets" / "validated_benign_corpus.csv"
    rf_aug = REPO_ROOT / "models" / "phase6_candidates" / "rf_augmented.joblib"
    lr_aug = REPO_ROOT / "models" / "phase6_candidates" / "logistic_regression_augmented.joblib"
    xgb_aug = REPO_ROOT / "models" / "phase6_candidates" / "xgboost_augmented.joblib"

    assert compute_sha256(p6_corpus) == EXPECTED_HASHES["phase6_validated_benign"]["expected_sha256"]
    assert compute_sha256(rf_aug) == EXPECTED_HASHES["phase6_rf_augmented"]["expected_sha256"]
    assert compute_sha256(lr_aug) == EXPECTED_HASHES["phase6_lr_augmented"]["expected_sha256"]
    assert compute_sha256(xgb_aug) == EXPECTED_HASHES["phase6_xgb_augmented"]["expected_sha256"]


def test_historical_reports_presence_and_integrity():
    """Verify all historical evaluation reports exist and are non-empty."""
    for rpath in REQUIRED_HISTORICAL_REPORTS:
        assert rpath.exists(), f"Missing historical report: {rpath}"
        assert rpath.stat().st_size > 0, f"Empty historical report: {rpath}"


def test_m0_verification_execution():
    """Verify that M0 verification executes successfully and generates approved manifest and report."""
    manifest = run_m0_verification()
    assert manifest["phase"] == 7
    assert manifest["milestone"] == "M0"
    assert manifest["status"] == "APPROVED"
    assert manifest["exit_criteria_met"] is True

    # Assert artifact hashes
    for name, item in manifest["artifact_hashes"].items():
        assert item["matches"] is True, f"Hash check failed for {name}"

    # Assert governance checks
    for check, res in manifest["governance_checks"].items():
        assert res is True, f"Governance check failed for {check}"

    # Verify manifest and report files exist on disk
    manifest_file = REPO_ROOT / "experiments" / "phase7" / "reports" / "experiment_manifest.json"
    report_file = REPO_ROOT / "experiments" / "phase7" / "reports" / "m0_verification_report.md"
    assert manifest_file.exists()
    assert report_file.exists()
    assert manifest_file.stat().st_size > 0
    assert report_file.stat().st_size > 0


def test_hash_tampering_detection(monkeypatch, tmp_path):
    """Verify that any file modification or hash mismatch immediately raises ValueError."""
    fake_file = tmp_path / "tampered_model.joblib"
    fake_file.write_text("tampered model binary data")

    import experiments.phase7.verify_m0_contract as m0_mod

    tampered_hashes = copy.deepcopy(m0_mod.EXPECTED_HASHES)
    tampered_hashes["frozen_production_model"]["path"] = fake_file

    monkeypatch.setattr(m0_mod, "EXPECTED_HASHES", tampered_hashes)

    with pytest.raises(ValueError, match="FATAL INVARIANT VIOLATION"):
        m0_mod.run_m0_verification()


def test_candidate_models_isolation_invariant():
    """Verify that candidate models cannot overwrite models/random_forest_model.joblib."""
    exp_cfg_path = REPO_ROOT / "experiments" / "phase7" / "configs" / "experiment.yaml"
    with open(exp_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prod_path = "models/random_forest_model.joblib"
    models_cfg = cfg.get("models", {})

    for mid, mspec in models_cfg.items():
        if mid != "E0":
            assert mspec.get("path") != prod_path, f"Candidate {mid} targets production model path!"
            if "output_dir" in mspec:
                assert mspec["output_dir"].startswith("models/phase7_candidates"), (
                    f"Candidate {mid} must output to models/phase7_candidates/"
                )


def test_quarantine_governance_enforcement():
    """Verify that violating benchmark quarantine or test reusability raises ValueError."""
    exp_cfg_path = REPO_ROOT / "experiments" / "phase7" / "configs" / "experiment.yaml"
    split_cfg_path = REPO_ROOT / "experiments" / "phase7" / "configs" / "split_config.yaml"

    with open(exp_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(split_cfg_path, "r", encoding="utf-8") as f:
        scfg = yaml.safe_load(f)

    # Test Phase 4 benchmark breach
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["external_confirmation"]["phase4_benchmark_is_untouched"] = True
    with pytest.raises(ValueError, match="Phase 4 benchmark must NOT be reused as fresh confirmatory benchmark"):
        verify_protocol_governance(bad_cfg, scfg)

    # Test Phase 6 test reusability breach
    bad_cfg2 = copy.deepcopy(cfg)
    bad_cfg2["splitting"]["phase6_internal_test_reusable"] = True
    with pytest.raises(ValueError, match="Phase 6 internal test set must NOT be reused as fresh internal test"):
        verify_protocol_governance(bad_cfg2, scfg)


def test_label_mapping_consistency():
    """Verify that label convention is strictly 0=Phishing and 1=Legitimate."""
    exp_cfg_path = REPO_ROOT / "experiments" / "phase7" / "configs" / "experiment.yaml"
    with open(exp_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    labels = cfg["labels"]
    phishing_label = labels.get(0) or labels.get("0")
    legit_label = labels.get(1) or labels.get("1")
    assert phishing_label == "Phishing"
    assert legit_label == "Legitimate"


def test_frozen_model_prediction_reproducibility():
    """Verify that the frozen control produces exact expected classes on known test URLs."""
    result = verify_frozen_model_prediction()
    assert result["status"] == "PASSED"
    for v in result["verifications"]:
        assert v["matches_expected"] is True
