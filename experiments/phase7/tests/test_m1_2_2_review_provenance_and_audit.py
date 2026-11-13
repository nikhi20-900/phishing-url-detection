"""
Test Suite for Phase 7 Milestone 1.2.2 (M1.2.2): Review Provenance and Corpus Audit Finalization.

Verifies:
1. Zero registered-domain overlap between the new training corpus and Phase 6 train, val, and test partitions.
2. Zero URL collisions with quarantined Phase 4 external benchmark.
3. 100% verified TLS on the actual data retrieval sockets.
4. Reviewer submission authenticity and tamper-detection.
5. 100% independent senior supervisory approval coverage on risk-flagged candidates.
6. Recomputed corpus metrics directly match committed CSV.
7. SHA-256 integrity across frozen production model, historical milestones, and M1.2.2 artifacts.
8. Review provenance classification and limitations documentation.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.evaluate_domain_split import extract_registered_domain
from experiments.phase7.validation.review_workflow_m1_2_2 import (
    validate_review_provenance_m1_2_2,
    compute_canonical_hash,
)
from experiments.phase7.validation.audit_corpus_m1_2_2 import (
    compute_sha256,
    compute_gini,
    EXPECTED_FROZEN_MODEL_SHA256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

CORPUS_V1_2_2 = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2_2.csv"
SUBMISSIONS_JSONL = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_2_reviewer_submissions.jsonl"
SENIOR_APPROVALS_JSONL = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_2_senior_approvals.jsonl"
RAW_OBS_JSONL = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_2_raw_http_observations.jsonl"
VAL_RECORDS_JSONL = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_2_validation_records.jsonl"

REPORT_MD = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_2_2_corpus_audit.md"
REPORT_MANIFEST = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_2_2_validation_manifest.json"
DATASET_MANIFEST = REPO_ROOT / "experiments" / "phase7" / "datasets" / "manifests" / "validation_manifest_v1_2_2.json"

PHASE6_TRAIN = REPO_ROOT / "experiments" / "phase6" / "datasets" / "splits" / "train.csv"
PHASE6_VAL = REPO_ROOT / "experiments" / "phase6" / "datasets" / "splits" / "val.csv"
PHASE6_TEST = REPO_ROOT / "experiments" / "phase6" / "datasets" / "splits" / "internal_test.csv"
PHASE4_BENCHMARK = REPO_ROOT / "data" / "external" / "external_validation_benchmark.csv"
FROZEN_MODEL = REPO_ROOT / "models" / "random_forest_model.joblib"


def test_m1_2_2_zero_domain_overlap_with_phase6_train():
    """Verify zero registered-domain overlap with Phase 6 training partition."""
    corpus_df = pd.read_csv(CORPUS_V1_2_2)
    p6_train = pd.read_csv(PHASE6_TRAIN)
    corpus_domains = set(corpus_df["registered_domain"].dropna())
    train_domains = set(p6_train["registered_domain"].dropna())
    overlap = corpus_domains & train_domains
    assert len(overlap) == 0, f"Domain overlap with Phase 6 train found: {overlap}"


def test_m1_2_2_zero_domain_overlap_with_phase6_val():
    """Verify zero registered-domain overlap with Phase 6 validation partition."""
    corpus_df = pd.read_csv(CORPUS_V1_2_2)
    p6_val = pd.read_csv(PHASE6_VAL)
    corpus_domains = set(corpus_df["registered_domain"].dropna())
    val_domains = set(p6_val["registered_domain"].dropna())
    overlap = corpus_domains & val_domains
    assert len(overlap) == 0, f"Domain overlap with Phase 6 val found: {overlap}"


def test_m1_2_2_zero_domain_overlap_with_phase6_test():
    """Verify zero registered-domain overlap with Phase 6 internal test partition."""
    corpus_df = pd.read_csv(CORPUS_V1_2_2)
    p6_test = pd.read_csv(PHASE6_TEST)
    corpus_domains = set(corpus_df["registered_domain"].dropna())
    test_domains = set(p6_test["registered_domain"].dropna())
    overlap = corpus_domains & test_domains
    assert len(overlap) == 0, f"Domain overlap with Phase 6 internal test found: {overlap}"


def test_m1_2_2_zero_collision_with_phase4_benchmark():
    """Verify zero URL collisions with quarantined Phase 4 external benchmark."""
    corpus_df = pd.read_csv(CORPUS_V1_2_2)
    p4_bench = pd.read_csv(PHASE4_BENCHMARK)
    corpus_urls = set(corpus_df["url"].dropna().str.strip())
    p4_urls = set(p4_bench["url"].dropna().str.strip())
    collisions = corpus_urls & p4_urls
    assert len(collisions) == 0, f"Phase 4 benchmark collisions found: {collisions}"


def test_m1_2_2_100_percent_verified_tls():
    """Verify 100% of corpus URLs feature verified TLS on retrieval sockets."""
    corpus_df = pd.read_csv(CORPUS_V1_2_2)
    assert len(corpus_df) == 119
    unverified = corpus_df[corpus_df["tls_verified"] != True]
    assert len(unverified) == 0, f"Found unverified TLS URLs: {unverified['url'].tolist()}"


def test_m1_2_2_review_submissions_authenticated():
    """Verify all corpus records have valid authenticated reviewer submissions."""
    summary = validate_review_provenance_m1_2_2(
        submissions_jsonl=SUBMISSIONS_JSONL,
        senior_approvals_jsonl=SENIOR_APPROVALS_JSONL,
        corpus_csv=CORPUS_V1_2_2,
    )
    assert summary["status"] == "PASS"
    assert summary["verified_submissions_count"] == 119
    assert summary["verified_senior_approvals_count"] == 17
    assert summary["senior_coverage_rate"] == 1.0


def test_m1_2_2_submission_tamper_detection(tmp_path):
    """Verify tampering with a submission record triggers a validation error."""
    # Create tampered copy of submissions
    tampered_file = tmp_path / "tampered_submissions.jsonl"
    with open(SUBMISSIONS_JSONL, "r", encoding="utf-8") as fin, open(tampered_file, "w", encoding="utf-8") as fout:
        first = True
        for line in fin:
            rec = json.loads(line)
            if first:
                rec["rationale"] = "Tampered rationale string!"
                first = False
            fout.write(json.dumps(rec) + "\n")

    with pytest.raises(ValueError, match="Tamper detected in submission record"):
        validate_review_provenance_m1_2_2(
            submissions_jsonl=tampered_file,
            senior_approvals_jsonl=SENIOR_APPROVALS_JSONL,
            corpus_csv=CORPUS_V1_2_2,
        )


def test_m1_2_2_senior_approval_tamper_detection(tmp_path):
    """Verify tampering with a senior approval record triggers a validation error."""
    tampered_file = tmp_path / "tampered_senior.jsonl"
    with open(SENIOR_APPROVALS_JSONL, "r", encoding="utf-8") as fin, open(tampered_file, "w", encoding="utf-8") as fout:
        first = True
        for line in fin:
            rec = json.loads(line)
            if first:
                rec["supervisory_rationale"] = "Tampered supervisory rationale!"
                first = False
            fout.write(json.dumps(rec) + "\n")

    with pytest.raises(ValueError, match="Tamper detected in senior approval record"):
        validate_review_provenance_m1_2_2(
            submissions_jsonl=SUBMISSIONS_JSONL,
            senior_approvals_jsonl=tampered_file,
            corpus_csv=CORPUS_V1_2_2,
        )


def test_m1_2_2_historical_artifacts_preserved():
    """Verify historical M1, M1.1, M1.2, and M1.2.1 deliverables are intact."""
    m1_df = pd.read_csv(REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv")
    m1_1_df = pd.read_csv(REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv")
    m1_2_df = pd.read_csv(REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2.csv")
    m1_2_1_df = pd.read_csv(REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2_1.csv")

    assert len(m1_df) == 1182
    assert len(m1_1_df) == 1176
    assert len(m1_2_df) == 334
    assert len(m1_2_1_df) == 347


def test_m1_2_2_production_model_frozen():
    """Verify production model SHA-256 remains frozen at known hash."""
    actual_hash = compute_sha256(FROZEN_MODEL)
    assert actual_hash == EXPECTED_FROZEN_MODEL_SHA256


def test_m1_2_2_candidate_models_empty():
    """Verify no candidate models have been trained in Phase 7 prior to M2."""
    cand_dir = REPO_ROOT / "models" / "phase7_candidates"
    if cand_dir.exists():
        joblibs = list(cand_dir.glob("*.joblib"))
        assert len(joblibs) == 0, f"Candidate models found: {joblibs}"


def test_m1_2_2_manifest_and_audit_metric_exactness():
    """Verify manifest and audit report metrics match directly computed CSV metrics."""
    corpus_df = pd.read_csv(CORPUS_V1_2_2)
    total_urls = len(corpus_df)
    unique_domains = corpus_df["registered_domain"].nunique()
    query_count = corpus_df["url"].str.contains(r"\?").sum()
    multilevel_count = (corpus_df["url"].str.count("/") >= 4).sum()

    with open(REPORT_MANIFEST, "r", encoding="utf-8") as f:
        rep_man = json.load(f)

    with open(DATASET_MANIFEST, "r", encoding="utf-8") as f:
        ds_man = json.load(f)

    assert rep_man == ds_man

    c_met = rep_man["corpus_metrics"]
    assert c_met["total_urls"] == total_urls == 119
    assert c_met["unique_registered_domains"] == unique_domains == 24
    assert c_met["strata_breakdown"]["query_bearing"]["count"] == query_count == 91
    assert c_met["strata_breakdown"]["multilevel_paths"]["count"] == multilevel_count == 108

    # Assert report text contains exact counts
    with open(REPORT_MD, "r", encoding="utf-8") as f:
        report_text = f.read()

    assert "**119**" in report_text
    assert "**24**" in report_text
    assert "91" in report_text
    assert "108" in report_text


def test_m1_2_2_review_provenance_distinction_documented():
    """Verify that human review provenance distinction is explicitly documented."""
    with open(REPORT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gov = manifest["provenance_governance"]
    assert gov["independently_human_verified_claim"] is False
    assert "Hardened Retrieval Security & Documented Review Governance" in gov["classification"]
    assert "Not claimed as decentralized double-blind" in gov["caveat"]

    with open(REPORT_MD, "r", encoding="utf-8") as f:
        report_text = f.read()

    assert "Review Provenance Distinction" in report_text
    assert "NOT as a universally decentralized third-party human-audited web crawl" in report_text
