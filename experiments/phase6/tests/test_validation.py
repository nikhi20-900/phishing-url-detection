"""
Tests for Phase 6 Milestone 1: Hybrid Page-Level Authenticity Validation.
"""

import json
from pathlib import Path
import pytest

from experiments.phase6.validation.review_schema import (
    ValidationRecord,
    ValidationStatus,
)
from experiments.phase6.validation.validate_candidates import (
    REPO_ROOT,
    run_page_authenticity_validation,
    compute_file_sha256,
)


def test_validation_record_schema_and_eligibility():
    """Verify that only verified_page status is training eligible."""
    rec = ValidationRecord(
        candidate_id="cand_1",
        original_url="https://example.com/deep/page",
        normalized_url="https://example.com/deep/page",
        registered_domain="example.com",
        validation_status=ValidationStatus.VERIFIED_PAGE.value,
        evidence_references=["ref_1"],
        automated_observations={"http_status": 200},
        redirect_chain=["https://example.com/deep/page"],
        reviewer="auditor-1",
        review_timestamp="2026-11-12T00:00:00Z",
        review_decision="approved",
        legitimacy_rationale="Official institutional directory with verified ownership.",
    )
    assert rec.is_training_eligible() is True


def test_inaccessible_or_ambiguous_cannot_be_approved():
    """Verify that inaccessible or ambiguous records cannot have review_decision == 'approved'."""
    with pytest.raises(ValueError, match="Cannot approve"):
        ValidationRecord(
            candidate_id="cand_ambiguous",
            original_url="https://example.com/unknown",
            normalized_url="https://example.com/unknown",
            registered_domain="example.com",
            validation_status=ValidationStatus.AMBIGUOUS.value,
            evidence_references=[],
            automated_observations={},
            redirect_chain=[],
            reviewer="auditor-1",
            review_timestamp="2026-11-12T00:00:00Z",
            review_decision="approved",
            legitimacy_rationale="Insufficient evidence",
        )


def test_risk_flags_require_second_review():
    """Verify that risk-flagged candidates require an independent second reviewer to be eligible."""
    # Without second review -> raises ValueError during post_init if verified_page
    with pytest.raises(ValueError, match="requires independent second reviewer"):
        ValidationRecord(
            candidate_id="cand_flagged",
            original_url="http://insecure.example.com/page",
            normalized_url="http://insecure.example.com/page",
            registered_domain="example.com",
            validation_status=ValidationStatus.VERIFIED_PAGE.value,
            evidence_references=[],
            automated_observations={},
            redirect_chain=[],
            reviewer="auditor-1",
            review_timestamp="2026-11-12T00:00:00Z",
            review_decision="approved",
            legitimacy_rationale="Insecure legacy protocol documentation.",
            has_risk_flags=True,
            risk_flags="insecure_http",
            second_reviewer=None,
            second_review_decision=None,
        )

    # With second review -> passes and is eligible
    rec_approved = ValidationRecord(
        candidate_id="cand_flagged",
        original_url="http://insecure.example.com/page",
        normalized_url="http://insecure.example.com/page",
        registered_domain="example.com",
        validation_status=ValidationStatus.VERIFIED_PAGE.value,
        evidence_references=[],
        automated_observations={},
        redirect_chain=[],
        reviewer="auditor-1",
        review_timestamp="2026-11-12T00:00:00Z",
        review_decision="approved",
        legitimacy_rationale="Insecure legacy protocol documentation.",
        has_risk_flags=True,
        risk_flags="insecure_http",
        second_reviewer="auditor-senior-2",
        second_review_decision="approved",
    )
    assert rec_approved.is_training_eligible() is True


def test_m1_validation_workflow_outputs():
    """Verify that M1 validation workflow outputs valid artifacts and manifest."""
    manifest_path = REPO_ROOT / "experiments" / "phase6" / "datasets" / "manifests" / "validation_manifest.json"
    assert manifest_path.exists(), "Validation manifest missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["milestone"] == "M1"
    assert manifest["exit_criteria"]["status"] == "APPROVED"
    assert manifest["validated_benign_corpus"]["sample_count"] == 602
    assert manifest["validated_benign_corpus"]["unique_registered_domains"] == 98

    corpus_path = REPO_ROOT / manifest["validated_benign_corpus"]["path"]
    assert corpus_path.exists()
    assert compute_file_sha256(corpus_path) == manifest["validated_benign_corpus"]["sha256"]
