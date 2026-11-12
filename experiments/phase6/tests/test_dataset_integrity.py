"""
Tests for Phase 6 Dataset Integrity, Artifact Invariants, and M0 Protocol Contract.
"""

import copy
import hashlib
import json
from pathlib import Path
import pytest
import yaml

from experiments.phase6.verify_m0_contract import (
    REPO_ROOT,
    EXPECTED_HASHES,
    compute_sha256,
    run_m0_verification,
)


def test_frozen_model_hash_matches_invariant():
    """Verify that models/random_forest_model.joblib matches the non-negotiable hash."""
    model_path = REPO_ROOT / "models" / "random_forest_model.joblib"
    actual_hash = compute_sha256(model_path)
    assert (
        actual_hash == EXPECTED_HASHES["frozen_model"]["expected_sha256"]
    ), f"Frozen model hash mismatch: {actual_hash}"


def test_external_benchmark_hash_matches_invariant():
    """Verify that Phase 4 external benchmark matches the quarantined hash."""
    bench_path = REPO_ROOT / "data" / "external" / "external_validation_benchmark.csv"
    actual_hash = compute_sha256(bench_path)
    assert (
        actual_hash == EXPECTED_HASHES["phase4_benchmark"]["expected_sha256"]
    ), f"External benchmark hash mismatch: {actual_hash}"


def test_phiusiil_source_hash_matches_invariant():
    """Verify that PhiUSIIL source dataset matches the immutable hash."""
    phiusiil_path = REPO_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    actual_hash = compute_sha256(phiusiil_path)
    assert (
        actual_hash == EXPECTED_HASHES["phiusiil_source"]["expected_sha256"]
    ), f"PhiUSIIL source hash mismatch: {actual_hash}"


def test_phase5_pilot_hashes_match_invariants():
    """Verify that Phase 5 approved pilot corpus and review queue match expected hashes."""
    pilot_path = REPO_ROOT / "experiments" / "generalization" / "datasets" / "approved_pilot_corpus.csv"
    queue_path = REPO_ROOT / "experiments" / "generalization" / "datasets" / "pilot_review_queue.csv"
    assert compute_sha256(pilot_path) == EXPECTED_HASHES["phase5_approved_pilot"]["expected_sha256"]
    assert compute_sha256(queue_path) == EXPECTED_HASHES["phase5_review_queue"]["expected_sha256"]


def test_m0_verification_execution():
    """Verify that M0 verification runs and produces a valid approved manifest."""
    manifest = run_m0_verification()
    assert manifest["milestone"] == "M0"
    assert manifest["status"] == "APPROVED"
    assert manifest["exit_criteria_met"] is True
    assert manifest["artifact_hashes"]["frozen_model"]["matches"] is True
    assert manifest["artifact_hashes"]["phase4_benchmark"]["matches"] is True
    assert manifest["artifact_hashes"]["phiusiil_source"]["matches"] is True


def test_hash_mismatch_detection(monkeypatch, tmp_path):
    """Verify that any hash tampering or mismatch immediately raises ValueError."""
    fake_file = tmp_path / "tampered.csv"
    fake_file.write_text("tampered content")

    import experiments.phase6.verify_m0_contract as m0_mod

    tampered_hashes = copy.deepcopy(m0_mod.EXPECTED_HASHES)
    tampered_hashes["frozen_model"]["path"] = fake_file

    monkeypatch.setattr(m0_mod, "EXPECTED_HASHES", tampered_hashes)

    with pytest.raises(ValueError, match="FATAL INVARIANT VIOLATION"):
        m0_mod.run_m0_verification()


def test_candidate_models_isolation():
    """Verify that candidate model paths never overwrite production models/random_forest_model.joblib."""
    exp_config_path = REPO_ROOT / "experiments" / "phase6" / "configs" / "experiment.yaml"
    with open(exp_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    prod_model_path = "models/random_forest_model.joblib"
    models_cfg = config["models"]

    for model_id, mspec in models_cfg.items():
        if mspec.get("retrained", False):
            cand_path = mspec["artifact_path"]
            assert cand_path != prod_model_path, f"Candidate {model_id} targets production model path!"
            assert cand_path.startswith("models/phase6_candidates/"), (
                f"Candidate {model_id} must be stored in models/phase6_candidates/"
            )


def test_label_mapping_consistency():
    """Verify that label convention is strictly 0=Phishing and 1=Legitimate."""
    exp_config_path = REPO_ROOT / "experiments" / "phase6" / "configs" / "experiment.yaml"
    with open(exp_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    labels = config["label_mapping"]
    assert labels[0] == "Phishing"
    assert labels[1] == "Legitimate"
