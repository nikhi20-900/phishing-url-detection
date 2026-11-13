"""
Tests for Phase 7 Milestone 1.2 (M1.2): Evidence Provenance Correction, Authentic Live HTTP Observations,
Zero Synthetic Evidence Invariants, Independent Human Review Ledger, and Substantiated Corpus v1.2.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from experiments.phase7.validation.audit_corpus_m1_2 import (
    REPO_ROOT,
    FROZEN_MODEL_PATH,
    EXPECTED_FROZEN_MODEL_HASH,
    PHASE4_BENCHMARK_PATH,
    PHASE6_CORPUS_PATH,
    compute_sha256,
    compute_gini,
)
from experiments.phase7.validation.retrieve_live_evidence_m1_2 import is_disallowed_ip


def test_frozen_production_model_remains_untouched():
    """Verify that models/random_forest_model.joblib is completely untouched and frozen."""
    actual_hash = compute_sha256(FROZEN_MODEL_PATH)
    assert actual_hash == EXPECTED_FROZEN_MODEL_HASH, (
        f"Production model was modified! Expected {EXPECTED_FROZEN_MODEL_HASH}, got {actual_hash}"
    )


def test_no_candidate_model_training_in_m1_2():
    """Verify strict milestone boundaries: M1.2 must NOT train or output candidate models."""
    phase7_model_dir = REPO_ROOT / "models" / "phase7_candidates"
    if phase7_model_dir.exists():
        model_files = list(phase7_model_dir.glob("*.joblib"))
        allowed = {"rf_augmented_e4.joblib"}
        unexpected = [f.name for f in model_files if f.name not in allowed]
        assert len(unexpected) == 0, f"Unauthorized candidate models found: {unexpected}"


def test_historical_m1_and_m1_1_artifacts_preserved_unmodified():
    """Verify that prior M1 and M1.1 deliverables were preserved untouched."""
    m1_corpus = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"
    m1_1_corpus = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"

    assert m1_corpus.exists()
    assert m1_1_corpus.exists()

    assert len(pd.read_csv(m1_corpus)) == 1182
    assert len(pd.read_csv(m1_1_corpus)) == 1176


def test_zero_synthetic_evidence_in_observations():
    """Verify all 1,207 observation records are authentic and non-synthetic."""
    raw_obs_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_raw_http_observations.jsonl"
    assert raw_obs_path.exists()

    observations = []
    with open(raw_obs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    assert len(observations) == 1207

    for obs in observations:
        assert obs.get("is_synthetic") is False, f"Synthetic observation detected: {obs['candidate_id']}"
        assert "retrieval_timestamp_utc" in obs
        assert obs["retrieval_timestamp_utc"].endswith("+00:00") or obs["retrieval_timestamp_utc"].endswith("Z")

        # Non-empty responses must have authentic SHA-256
        if obs["content_length_bytes"] > 0:
            assert obs["response_sha256"] is not None
            assert len(obs["response_sha256"]) == 64


def test_observer_fails_adversarially_on_synthetic_evidence():
    """Test that the validation pipeline rejects synthetic observation records."""
    from experiments.phase7.validation.revalidate_corpus_m1_2 import revalidate_corpus_m1_2
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        fake_obs_file = tmp_dir / "fake_obs.jsonl"
        fake_prim_file = tmp_dir / "fake_prim.jsonl"
        fake_sen_file = tmp_dir / "fake_sen.jsonl"
        fake_rec_file = tmp_dir / "fake_rec.jsonl"
        fake_corp_file = tmp_dir / "fake_corp.csv"

        # Write a synthetic observation
        with open(fake_obs_file, "w") as f:
            f.write(json.dumps({
                "candidate_id": "test_synthetic",
                "is_synthetic": True,  # SYNTHETIC!
            }) + "\n")

        with open(fake_prim_file, "w") as f:
            f.write(json.dumps({
                "candidate_id": "test_synthetic",
                "decision": "approved",
                "rationale": "Valid looking",
            }) + "\n")

        with open(fake_sen_file, "w") as f:
            pass

        # Candidate queue stub
        stub_queue = tmp_dir / "stub_q.csv"
        pd.DataFrame([{
            "candidate_id": "test_synthetic",
            "url": "https://example.com/test",
            "url_exact": "https://example.com/test",
            "registered_domain": "example.com",
            "source_name": "Test",
            "collection_timestamp": "2026-11-12T00:00:00Z",
            "has_risk_flags": False,
            "risk_flags": None,
        }]).to_csv(stub_queue, index=False)

        # Must raise ValueError on synthetic evidence
        with pytest.raises(ValueError, match="is_synthetic == False"):
            revalidate_corpus_m1_2(
                stub_queue, fake_obs_file, fake_prim_file, fake_sen_file, fake_rec_file, fake_corp_file
            )


def test_independent_human_review_ledger_integrity():
    """Verify primary and senior human reviews have independently supplied decisions and rationales."""
    prim_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_human_reviews.jsonl"
    sen_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_senior_reviews.jsonl"

    assert prim_path.exists()
    assert sen_path.exists()

    primary_reviews = []
    with open(prim_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                primary_reviews.append(json.loads(line))

    assert len(primary_reviews) == 1207

    allowed_decisions = {"approved", "unsubstantiated_inaccessible", "rejected", "flagged_for_second_review", "inaccessible"}
    reviewers = set()
    for pr in primary_reviews:
        assert pr["decision"] in allowed_decisions, f"Invalid decision: {pr['decision']}"
        assert len(pr["rationale"].strip()) >= 15, f"Trivial rationale in {pr['candidate_id']}"
        assert pr["reviewer_id"].startswith("auditor-")
        reviewers.add(pr["reviewer_id"])

    assert len(reviewers) >= 4, f"Expected multi-reviewer roster, got {len(reviewers)}"

    senior_reviews = []
    with open(sen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                senior_reviews.append(json.loads(line))

    assert len(senior_reviews) >= 20
    for sr in senior_reviews:
        assert sr["senior_reviewer_id"].startswith("auditor-senior-")
        assert sr["second_decision"] in ("approved", "unsubstantiated", "rejected")
        assert len(sr["second_rationale"].strip()) >= 20


def test_ssrf_and_private_ip_blocking():
    """Verify private IP addresses are identified and blocked from connection."""
    assert is_disallowed_ip("127.0.0.1") is True
    assert is_disallowed_ip("10.0.0.1") is True
    assert is_disallowed_ip("172.16.0.5") is True
    assert is_disallowed_ip("192.168.1.1") is True
    assert is_disallowed_ip("169.254.169.254") is True
    assert is_disallowed_ip("8.8.8.8") is False

    # Check that private IP candidate in observations was blocked
    raw_obs_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_raw_http_observations.jsonl"
    with open(raw_obs_path, "r", encoding="utf-8") as f:
        priv_obs = [json.loads(line) for line in f if "192.168.1.1" in line]

    assert len(priv_obs) == 1
    assert priv_obs[0]["is_private_ip"] is True
    assert priv_obs[0]["is_accessible"] is False


def test_substantiated_corpus_v1_2_invariants():
    """Verify substantiated corpus v1.2 properties and complete exclusion of invalid URLs."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2.csv"
    assert corpus_path.exists()
    df = pd.read_csv(corpus_path)

    # 334 substantiated URLs
    assert len(df) == 334
    assert df["registered_domain"].nunique() >= 80

    # Every URL is exact
    assert (df["url"] == df["url_exact"]).all()

    # Zero exact collisions with Phase 4 external benchmark
    p4_df = pd.read_csv(PHASE4_BENCHMARK_PATH)
    p4_urls = set(p4_df["url"].dropna().str.strip())
    v1_2_urls = set(df["url"].dropna().str.strip())
    assert len(v1_2_urls.intersection(p4_urls)) == 0

    # Excluded URLs check: None of the dynamic/interactive wiki/tag URLs are present
    excluded_patterns = [
        "Special:RandomInCategory",
        "Special:Search?search=cryptographic",
        "Special:NewItem",
        "Special:RecentChanges",
        "veaction=edit",
        "ubuntu.com/blog/tag/open-source/?page=1&sort=date",
    ]
    for pattern in excluded_patterns:
        matches = df[df["url"].str.contains(pattern, regex=False)]
        assert len(matches) == 0, f"Excluded pattern '{pattern}' leaked into corpus v1.2"


def test_domain_concentration_limits_in_v1_2():
    """Verify that substantiated corpus v1.2 satisfies anti-concentration limits."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2.csv"
    df = pd.read_csv(corpus_path)

    counts = df["registered_domain"].value_counts()
    total = len(df)

    top1_share = counts.iloc[0] / total
    top5_share = counts.iloc[:5].sum() / total
    top10_share = counts.iloc[:10].sum() / total

    assert top1_share <= 0.05, f"Top 1 domain share exceeded 5%: {top1_share:.2%}"
    assert top5_share <= 0.15, f"Top 5 domain share exceeded 15%: {top5_share:.2%}"
    assert top10_share <= 0.27, f"Top 10 domain share exceeded 27%: {top10_share:.2%}"

    gini = compute_gini(counts.values.tolist())
    assert gini < 0.40, f"Gini coefficient exceeded 0.40: {gini:.4f}"

    shares = counts.values / total
    eff_domains = 1.0 / (shares ** 2).sum()
    assert eff_domains >= 50.0, f"Effective domain count below 50: {eff_domains:.1f}"


def test_priority_strata_coverage_in_v1_2():
    """Verify priority strata coverage in substantiated corpus v1.2."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2.csv"
    df = pd.read_csv(corpus_path)

    # Query-bearing URLs >= 150
    query_count = df["url"].str.contains(r"\?", regex=True).sum()
    assert query_count >= 150, f"Expected >= 150 query-bearing URLs, got {query_count}"

    # Multi-level paths >= 250
    multilevel_count = (df["url"].str.count("/") >= 4).sum()
    assert multilevel_count >= 250, f"Expected >= 250 multi-level paths, got {multilevel_count}"

    # Path + Query combinations >= 150
    path_query_count = ((df["url"].str.count("/") >= 3) & df["url"].str.contains(r"\?", regex=True)).sum()
    assert path_query_count >= 150, f"Expected >= 150 path+query URLs, got {path_query_count}"

    # Trailing-slash variations >= 100
    slash_count = df["has_trailing_slash"].sum() if "has_trailing_slash" in df.columns else df["url"].str.contains(r"/(?:\?|$)", regex=True).sum()
    assert slash_count >= 100, f"Expected >= 100 trailing-slash URLs, got {slash_count}"


def test_audit_report_and_manifest_validity():
    """Verify that M1.2 audit report and cryptographic manifest exist and match files."""
    report_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_2_corpus_audit.md"
    manifest_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_2_validation_manifest.json"
    ds_manifest_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "manifests" / "validation_manifest_v1_2.json"

    assert report_path.exists()
    assert manifest_path.exists()
    assert ds_manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["milestone"] == "M1.2"
    assert manifest["status"] == "APPROVED"
    assert manifest["exit_criteria_met"] is True
    assert manifest["evidence_provenance"]["zero_synthetic_evidence"] is True
    assert manifest["summary"]["substantiated_training_eligible_corpus_v1_2"] == 334

    # Check cryptographic hashes
    for key, item in manifest["artifact_hashes"].items():
        artifact_path = REPO_ROOT / item["path"]
        assert artifact_path.exists(), f"Artifact {item['path']} missing"
        actual_hash = compute_sha256(artifact_path)
        assert actual_hash == item["sha256"], f"Hash mismatch for {key}: expected {item['sha256']}, got {actual_hash}"
