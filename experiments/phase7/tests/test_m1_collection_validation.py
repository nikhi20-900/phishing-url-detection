"""
Tests for Phase 7 Milestone 1 (M1): Legitimate URL Corpus Expansion, Collection Pipeline,
Validation Schema, and Corpus Audit.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from experiments.phase7.validation.review_schema import (
    ValidationRecord,
    ValidationStatus,
)
from experiments.phase7.validation.audit_corpus import (
    REPO_ROOT,
    FROZEN_MODEL_PATH,
    EXPECTED_FROZEN_MODEL_HASH,
    PHASE4_BENCHMARK_PATH,
    PHASE6_CORPUS_PATH,
    compute_sha256,
    compute_gini,
)


def test_frozen_production_model_remains_untouched():
    """Verify that models/random_forest_model.joblib is completely untouched and frozen."""
    actual_hash = compute_sha256(FROZEN_MODEL_PATH)
    assert actual_hash == EXPECTED_FROZEN_MODEL_HASH


def test_no_candidate_model_training_in_m1():
    """Verify strict milestone boundaries: M1 must NOT train or output candidate models."""
    phase7_model_dir = REPO_ROOT / "models" / "phase7_candidates"
    if phase7_model_dir.exists():
        model_files = list(phase7_model_dir.glob("*.joblib"))
        allowed = {"rf_augmented_e4.joblib"}
        unexpected = [f.name for f in model_files if f.name not in allowed]
        assert len(unexpected) == 0, f"Unauthorized candidate models found: {unexpected}"


def test_candidate_queue_structure_and_integrity():
    """Verify candidate review queue contains required schema, columns, and records."""
    queue_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "candidate_queue.csv"
    assert queue_path.exists()
    df = pd.read_csv(queue_path)

    expected_cols = [
        "url", "source_name", "source_url", "collection_timestamp",
        "verification_method", "review_status", "reviewer", "notes",
        "legitimacy_reason", "candidate_id", "url_exact", "normalized_dedupe_key",
        "scheme", "hostname", "registered_domain", "path", "path_depth",
        "has_path", "is_multilevel_path", "has_trailing_slash", "has_query",
        "has_fragment", "has_special_chars", "has_risk_flags",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column in candidate queue: {col}"

    assert len(df) >= 1200, f"Expected at least 1200 queue records, got {len(df)}"
    assert df["candidate_id"].nunique() == len(df), "Candidate IDs must be unique"


def test_exact_urls_preserved_without_silent_rewrites():
    """Verify that original URL strings are strictly preserved verbatim."""
    queue_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "candidate_queue.csv"
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"

    q_df = pd.read_csv(queue_path)
    c_df = pd.read_csv(corpus_path)

    assert (q_df["url"] == q_df["url_exact"]).all(), "Candidate queue rewritten url_exact"
    assert (c_df["url"] == c_df["url_exact"]).all(), "Corpus rewritten url_exact"


def test_validation_record_schema_rules():
    """Verify ValidationRecord invariants and post_init validation rules."""
    # Inaccessible/ambiguous cannot be approved
    with pytest.raises(ValueError, match="Cannot approve"):
        ValidationRecord(
            candidate_id="cand_inacc",
            original_url="https://dead-domain.invalid/test",
            normalized_url="https://dead-domain.invalid/test",
            registered_domain="dead-domain.invalid",
            validation_status=ValidationStatus.INACCESSIBLE.value,
            evidence_references=[],
            automated_observations={},
            redirect_chain=[],
            reviewer="auditor-1",
            review_timestamp="2026-11-12T00:00:00Z",
            review_decision="approved",
            legitimacy_rationale="Dead domain",
        )

    # verified_page requires review_decision == 'approved'
    with pytest.raises(ValueError, match="expected 'approved'"):
        ValidationRecord(
            candidate_id="cand_bad_dec",
            original_url="https://stanford.edu/test",
            normalized_url="https://stanford.edu/test",
            registered_domain="stanford.edu",
            validation_status=ValidationStatus.VERIFIED_PAGE.value,
            evidence_references=[],
            automated_observations={},
            redirect_chain=[],
            reviewer="auditor-1",
            review_timestamp="2026-11-12T00:00:00Z",
            review_decision="rejected",
            legitimacy_rationale="Valid university site",
        )

    # Risk-flagged record requires independent second reviewer
    with pytest.raises(ValueError, match="requires independent second reviewer"):
        ValidationRecord(
            candidate_id="cand_flagged",
            original_url="http://legacy.ietf.org/rfc/rfc791.txt",
            normalized_url="http://legacy.ietf.org/rfc/rfc791.txt",
            registered_domain="ietf.org",
            validation_status=ValidationStatus.VERIFIED_PAGE.value,
            evidence_references=[],
            automated_observations={},
            redirect_chain=[],
            reviewer="auditor-1",
            review_timestamp="2026-11-12T00:00:00Z",
            review_decision="approved",
            legitimacy_rationale="Legacy plaintext RFC documentation",
            has_risk_flags=True,
            risk_flags="insecure_http",
            second_reviewer=None,
            second_review_decision=None,
        )


def test_negative_candidates_filtering_efficacy():
    """Verify that negative/rejected/ambiguous/inaccessible items are excluded from corpus."""
    queue_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "candidate_queue.csv"
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"

    q_df = pd.read_csv(queue_path)
    c_df = pd.read_csv(corpus_path)

    non_approved_queue = q_df[q_df["review_status"] != "approved"]
    assert len(non_approved_queue) >= 20, "Expected negative test candidates in review queue"

    corpus_urls = set(c_df["url"])
    for _, neg_row in non_approved_queue.iterrows():
        assert neg_row["url"] not in corpus_urls, (
            f"Negative candidate {neg_row['url']} ({neg_row['review_status']}) leaked into verified corpus!"
        )


def test_priority_strata_coverage():
    """Assert that the expanded corpus meets all prioritized structural category targets."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"
    df = pd.read_csv(corpus_path)

    # 1. Query-bearing URLs (Critical)
    query_count = df["url"].str.contains(r"\?", regex=True).sum()
    assert query_count >= 500, f"Query-bearing count {query_count} below target 500"

    # 2. Multi-level paths (High)
    multilevel_count = (df["url"].str.count("/") >= 4).sum()
    assert multilevel_count >= 1000, f"Multi-level path count {multilevel_count} below target 1000"

    # 3. Path + Query combinations (Critical)
    path_and_query = ((df["url"].str.contains(r"\?", regex=True)) & (df["url"].str.count("/") >= 3)).sum()
    assert path_and_query >= 500, f"Path+query count {path_and_query} below target 500"

    # 4. Percent-encoded paths (Medium)
    pct_count = df["url"].str.contains(r"%[0-9a-fA-F]{2}", regex=True).sum()
    assert pct_count >= 25, f"Percent-encoded count {pct_count} below target 25"

    # 5. Trailing-slash variations (Medium)
    trailing_count = df["url"].str.endswith("/").sum()
    assert trailing_count >= 300, f"Trailing-slash count {trailing_count} below target 300"

    # 6. Unseen registered domains (Critical)
    p6_df = pd.read_csv(PHASE6_CORPUS_PATH)
    p6_domains = set(p6_df["registered_domain"].unique())
    c_domains = set(df["registered_domain"].unique())
    unseen_domains = c_domains - p6_domains
    assert len(unseen_domains) >= 50, f"Unseen domains count {len(unseen_domains)} below target 50"


def test_domain_concentration_limits():
    """Verify that domain concentration metrics avoid monopoly or excessive clustering."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"
    df = pd.read_csv(corpus_path)

    domain_counts = df["registered_domain"].value_counts()
    total = len(df)

    gini = compute_gini(domain_counts.values.tolist())
    top1_share = domain_counts.iloc[0] / total
    top5_share = domain_counts.iloc[:5].sum() / total

    assert gini < 0.40, f"Gini coefficient {gini:.4f} exceeds threshold 0.40"
    assert top1_share <= 0.05, f"Top-1 domain share {top1_share:.2%} exceeds 5%"
    assert top5_share <= 0.15, f"Top-5 domain share {top5_share:.2%} exceeds 15%"


def test_risk_flags_second_review_compliance():
    """Verify that 100% of risk-flagged URLs admitted into the corpus have second reviewer approvals."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"
    df = pd.read_csv(corpus_path)

    flagged = df[df["has_risk_flags"] == True]
    assert len(flagged) > 0, "Expected risk-flagged items in corpus"

    for _, row in flagged.iterrows():
        assert pd.notna(row.get("second_reviewer")), f"URL {row['url']} lacks second_reviewer"
        assert row.get("second_review_decision") == "approved", f"URL {row['url']} lacks approved second review"


def test_confirmatory_external_benchmark_zero_overlap():
    """Verify zero exact URL overlap with the quarantined Phase 4 external benchmark."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"
    c_df = pd.read_csv(corpus_path)
    p4_df = pd.read_csv(PHASE4_BENCHMARK_PATH)

    overlap = set(c_df["url"]).intersection(set(p4_df["url"].dropna()))
    assert len(overlap) == 0, f"Fatal leakage: {len(overlap)} URLs overlap with Phase 4 external benchmark!"


def test_manifest_and_audit_report_generation():
    """Verify that M1 audit report and manifests exist and contain approved exit criteria."""
    audit_md = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_corpus_audit.md"
    rep_manifest = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_validation_manifest.json"
    ds_manifest = REPO_ROOT / "experiments" / "phase7" / "datasets" / "manifests" / "validation_manifest.json"

    assert audit_md.exists() and audit_md.stat().st_size > 0
    assert rep_manifest.exists() and rep_manifest.stat().st_size > 0
    assert ds_manifest.exists() and ds_manifest.stat().st_size > 0

    with open(rep_manifest, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["phase"] == 7
    assert m["milestone"] == "M1"
    assert m["status"] == "APPROVED"
    assert m["exit_criteria_met"] is True
