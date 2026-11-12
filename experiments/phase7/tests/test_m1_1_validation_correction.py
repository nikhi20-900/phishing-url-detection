"""
Tests for Phase 7 Milestone 1.1 (M1.1): Validation Correction, Controlled HTTP Observations,
Separation of Concerns, Senior Auditor Review Records, and Substantiated Benign Corpus v1.1.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from experiments.phase7.validation.audit_corpus_m1_1 import (
    REPO_ROOT,
    FROZEN_MODEL_PATH,
    EXPECTED_FROZEN_MODEL_HASH,
    PHASE4_BENCHMARK_PATH,
    PHASE6_CORPUS_PATH,
    compute_sha256,
    compute_gini,
)
from experiments.phase7.validation.observe_http_evidence import is_disallowed_ip


def test_frozen_production_model_remains_untouched():
    """Verify that models/random_forest_model.joblib is completely untouched and frozen."""
    actual_hash = compute_sha256(FROZEN_MODEL_PATH)
    assert actual_hash == EXPECTED_FROZEN_MODEL_HASH, (
        f"Production model was modified! Expected {EXPECTED_FROZEN_MODEL_HASH}, got {actual_hash}"
    )


def test_no_candidate_model_training_in_m1_1():
    """Verify strict milestone boundaries: M1.1 must NOT train or output candidate models."""
    phase7_model_dir = REPO_ROOT / "models" / "phase7_candidates"
    if phase7_model_dir.exists():
        model_files = list(phase7_model_dir.glob("*.joblib"))
        assert len(model_files) == 0, f"Candidate models trained prematurely in M1.1: {model_files}"


def test_m1_historical_artifacts_are_preserved_unmodified():
    """Verify that original Phase 7 M1 artifacts were not overwritten or modified."""
    m1_corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"
    m1_audit_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_corpus_audit.md"
    m1_manifest_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_validation_manifest.json"

    assert m1_corpus_path.exists(), "Original M1 corpus file missing"
    assert m1_audit_path.exists(), "Original M1 audit report missing"
    assert m1_manifest_path.exists(), "Original M1 manifest missing"

    m1_df = pd.read_csv(m1_corpus_path)
    assert len(m1_df) == 1182, f"Original M1 corpus modified: expected 1182 rows, got {len(m1_df)}"


def test_private_ip_guard_blocks_ssrf():
    """Verify that private and loopback IP addresses are recognized and blocked."""
    assert is_disallowed_ip("127.0.0.1") is True
    assert is_disallowed_ip("10.0.0.1") is True
    assert is_disallowed_ip("172.16.0.5") is True
    assert is_disallowed_ip("192.168.1.1") is True
    assert is_disallowed_ip("169.254.169.254") is True
    assert is_disallowed_ip("8.8.8.8") is False
    assert is_disallowed_ip("151.101.1.69") is False


def test_raw_http_observations_schema_and_separation():
    """Verify raw HTTP observations file exists, is valid JSONL, and contains no human review fields."""
    raw_obs_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_1_raw_http_observations.jsonl"
    assert raw_obs_path.exists(), "m1_1_raw_http_observations.jsonl missing"

    observations = []
    with open(raw_obs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                observations.append(json.loads(line))

    assert len(observations) == 1207, f"Expected 1207 raw observation records, got {len(observations)}"

    human_fields = {"reviewer", "review_decision", "legitimacy_rationale", "second_reviewer", "second_review_status", "primary_reviewer_id"}
    for obs in observations:
        assert "candidate_id" in obs
        assert "target_url" in obs
        assert "connection_attempted" in obs
        assert "resolved_ip" in obs
        assert "http_status_code" in obs
        # Separation of concerns check:
        for hf in human_fields:
            assert hf not in obs, f"Observation record leaked human review field '{hf}': {obs['candidate_id']}"

    # Verify RFC 1918 test entry in observations is flagged as private IP and blocked from connection
    private_ip_obs = [obs for obs in observations if "192.168.1.1" in obs["target_url"]]
    assert len(private_ip_obs) == 1
    assert private_ip_obs[0]["is_private_ip"] is True
    assert private_ip_obs[0]["connection_attempted"] is False


def test_human_review_decisions_schema_and_genuineness():
    """Verify human review decisions are stored separately with genuine rationales and senior reviewer records."""
    dec_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_1_human_review_decisions.jsonl"
    assert dec_path.exists(), "m1_1_human_review_decisions.jsonl missing"

    decisions = []
    with open(dec_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                decisions.append(json.loads(line))

    assert len(decisions) == 1207, f"Expected 1207 human decision records, got {len(decisions)}"

    # Check distinct rationales and reviewers
    reviewers = {d["primary_reviewer_id"] for d in decisions}
    assert len(reviewers) >= 4, f"Expected at least 4 primary reviewers, got {len(reviewers)}"

    # Check senior reviewer assignment on flagged/second-reviewed items
    second_reviews = [d for d in decisions if d.get("second_reviewer_id") is not None]
    assert len(second_reviews) >= 28, f"Expected at least 28 second-reviewed decisions, got {len(second_reviews)}"
    for sr in second_reviews:
        assert sr["second_reviewer_id"].startswith("auditor-senior-")
        assert sr["second_review_decision"] in ("approved", "unsubstantiated")
        assert len(sr["second_review_rationale"]) > 10, "Senior review rationale must be explanatory"


def test_unsubstantiated_candidates_excluded_from_corpus_v1_1():
    """Verify that all 6 dynamic/interactive/unsubstantiated candidate URLs are excluded from v1.1."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"
    assert corpus_path.exists(), "validated_benign_corpus_v1_1.csv missing"
    df = pd.read_csv(corpus_path)

    # Exactly 1176 validated URLs
    assert len(df) == 1176, f"Expected 1176 URLs in corpus v1.1, got {len(df)}"

    excluded_urls = [
        "https://en.wikipedia.org/wiki/Special:RandomInCategory?category=Data_structures",
        "https://en.wikipedia.org/wiki/Special:Search?search=cryptographic%20hash%20function&ns0=1",
        "https://www.wikidata.org/wiki/Special:NewItem?lang=en",
        "https://www.wikidata.org/wiki/Special:RecentChanges?limit=50&namespace=0",
        "https://en.wikipedia.org/wiki/Information_retrieval?veaction=edit",
        "https://ubuntu.com/blog/tag/open-source/?page=1&sort=date",
    ]
    for url in excluded_urls:
        matches = df[df["url"] == url]
        assert len(matches) == 0, f"Corpus v1.1 illegally contains unsubstantiated candidate '{url}'"


def test_all_negative_queue_items_excluded_from_corpus_v1_1():
    """Verify that none of the 25 negative queue items exist in corpus v1.1."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"
    queue_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "candidate_queue.csv"

    c_df = pd.read_csv(corpus_path)
    q_df = pd.read_csv(queue_path)

    # Negative queue items are the 25 control candidates with rejected/ambiguous/inaccessible status
    neg_cands = set(q_df[q_df["review_status"].isin(["rejected", "ambiguous", "inaccessible"])]["candidate_id"])
    assert len(neg_cands) == 25, f"Expected 25 negative queue items, found {len(neg_cands)}"

    c_cand_ids = set(c_df["candidate_id"])
    leakage = neg_cands.intersection(c_cand_ids)
    assert len(leakage) == 0, f"Negative queue candidates leaked into corpus v1.1: {leakage}"


def test_no_exact_collision_with_phase4_benchmark():
    """Verify complete independence: 0 exact URL collisions with external benchmark."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"
    c_df = pd.read_csv(corpus_path)
    p4_df = pd.read_csv(PHASE4_BENCHMARK_PATH)

    p4_urls = set(p4_df["url"].dropna().str.strip())
    v1_1_urls = set(c_df["url"].dropna().str.strip())

    collisions = v1_1_urls.intersection(p4_urls)
    assert len(collisions) == 0, f"Collisions found between corpus v1.1 and Phase 4 benchmark: {collisions}"


def test_url_exactness_preserved_in_v1_1():
    """Verify that original URLs are preserved verbatim without silent transformations."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"
    df = pd.read_csv(corpus_path)
    assert (df["url"] == df["url_exact"]).all(), "Corpus v1.1 contains modified url_exact values"


def test_priority_strata_coverage_requirements():
    """Verify that substantiated corpus v1.1 fulfills all priority strata requirements."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"
    df = pd.read_csv(corpus_path)
    phase6_df = pd.read_csv(PHASE6_CORPUS_PATH)

    # 1. Query-bearing: >= 500 URLs
    query_count = df["has_query"].sum()
    assert query_count >= 500, f"Expected >= 500 query-bearing URLs, got {query_count}"

    # 2. Multi-level paths: >= 1000 URLs
    multilevel_count = df["is_multilevel_path"].sum()
    assert multilevel_count >= 1000, f"Expected >= 1000 multi-level paths, got {multilevel_count}"

    # 3. Path + Query combination: >= 500 URLs
    path_query_count = (df["has_path"] & df["has_query"]).sum()
    assert path_query_count >= 500, f"Expected >= 500 path+query URLs, got {path_query_count}"

    # 4. Percent-encoded paths: >= 25 URLs
    encoded_count = df["url"].str.contains(r"%[0-9a-fA-F]{2}", regex=True).sum()
    assert encoded_count >= 25, f"Expected >= 25 percent-encoded URLs, got {encoded_count}"

    # 5. Trailing-slash variations: >= 300 URLs
    slash_count = df["has_trailing_slash"].sum()
    assert slash_count >= 300, f"Expected >= 300 trailing-slash URLs, got {slash_count}"

    # 6. Unseen registered domains vs Phase 6: >= 50
    corpus_domains = set(df["registered_domain"].unique())
    phase6_domains = set(phase6_df["registered_domain"].unique())
    unseen_domains = corpus_domains - phase6_domains
    assert len(unseen_domains) >= 50, f"Expected >= 50 unseen domains, got {len(unseen_domains)}"


def test_domain_concentration_and_diversity_limits():
    """Verify that substantiated corpus v1.1 satisfies anti-concentration requirements."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"
    df = pd.read_csv(corpus_path)

    domain_counts = df["registered_domain"].value_counts()
    total_urls = len(df)
    assert len(domain_counts) >= 120, f"Expected >= 120 domains, got {len(domain_counts)}"

    top1_share = domain_counts.iloc[0] / total_urls
    top5_share = domain_counts.iloc[:5].sum() / total_urls
    top10_share = domain_counts.iloc[:10].sum() / total_urls

    assert top1_share <= 0.05, f"Top-1 domain share exceeded 5%: {top1_share:.2%}"
    assert top5_share <= 0.15, f"Top-5 domain share exceeded 15%: {top5_share:.2%}"
    assert top10_share <= 0.25, f"Top-10 domain share exceeded 25%: {top10_share:.2%}"

    domain_shares = domain_counts.values / total_urls
    hhi = float((domain_shares ** 2).sum())
    effective_domains = 1.0 / hhi
    assert effective_domains >= 100.0, f"Effective domain count below 100: {effective_domains:.1f}"

    gini = compute_gini(domain_counts.values.tolist())
    assert gini < 0.40, f"Gini coefficient exceeded 0.40: {gini:.4f}"


def test_audit_report_and_manifest_validity():
    """Verify that M1.1 audit report and cryptographic manifest exist and are consistent."""
    report_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_1_corpus_audit.md"
    manifest_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_1_validation_manifest.json"
    ds_manifest_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "manifests" / "validation_manifest_v1_1.json"

    assert report_path.exists(), "m1_1_corpus_audit.md missing"
    assert manifest_path.exists(), "m1_1_validation_manifest.json missing"
    assert ds_manifest_path.exists(), "validation_manifest_v1_1.json missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["milestone"] == "M1.1"
    assert manifest["status"] == "APPROVED"
    assert manifest["exit_criteria_met"] is True
    assert manifest["summary"]["substantiated_training_eligible_corpus_v1_1"] == 1176
    assert manifest["revalidation_details"]["excluded_from_proposed_m1"] == 6
    assert manifest["revalidation_details"]["excluded_negative_audit_items"] == 25

    # Check that artifact hashes in manifest match actual files
    for key, item in manifest["artifact_hashes"].items():
        artifact_path = REPO_ROOT / item["path"]
        assert artifact_path.exists(), f"Artifact {item['path']} does not exist"
        actual_hash = compute_sha256(artifact_path)
        assert actual_hash == item["sha256"], f"Hash mismatch for {key}: expected {item['sha256']}, got {actual_hash}"
