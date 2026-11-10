"""Tests for the Phase 5 legitimate pilot corpus audit module.

Includes:
1. Unit tests for concentration metrics calculation (HHI, entropy, Gini).
2. Live pilot dataset audit tests across all 5 scientific dimensions.
3. 10 adversarial synthetic audit tests (P0.4):
   - Metadata tampering without changing URLs.
   - Exact benchmark collisions.
   - Normalized benchmark collisions.
   - Missing approved reviewer.
   - Missing approved verification method.
   - Missing approved legitimacy rationale.
   - Unsorted domain counter insertion order.
   - Empty and malformed audit inputs.
   - Feature distribution / baseline criterion failures.
   - Deliberately failing audit CLI returning non-zero exit code.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from experiments.generalization.audit.audit_pilot_corpus import (
    audit_domain_diversity,
    audit_provenance_and_reproducibility,
    audit_quarantine_overlap,
    audit_risk_flags_and_verification,
    audit_structural_feature_coverage,
    compute_concentration_metrics,
    compute_review_queue_hash,
    generate_audit_manifest,
    main,
    validate_baseline_artifact,
)


@pytest.fixture
def pilot_datasets_paths() -> dict[str, Path]:
    base = Path("experiments/generalization/datasets")
    return {
        "raw": base / "pilot_candidates_raw.csv",
        "queue": base / "pilot_review_queue.csv",
        "approved": base / "approved_pilot_corpus.csv",
    }


# ---------------------------------------------------------------------------
# 1. Baseline and Live Pilot Tests
# ---------------------------------------------------------------------------

def test_concentration_metrics_calculation() -> None:
    counts = Counter({"domain-a.org": 50, "domain-b.org": 30, "domain-c.org": 20})
    metrics = compute_concentration_metrics(counts, 100)
    assert metrics["unique_registered_domains"] == 3
    assert metrics["total_urls"] == 100
    assert metrics["top1_share"] == 0.50
    assert metrics["top5_share"] == 1.00
    # HHI = 50^2 + 30^2 + 20^2 = 2500 + 900 + 400 = 3800
    assert metrics["hhi_index"] == 3800.0


def test_audit_domain_diversity_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    result = audit_domain_diversity(approved_df, queue_df)
    assert result["status"] == "PASS"
    assert result["metrics"]["unique_registered_domains"] == 98
    assert result["reconciliation"]["candidate_queue_unique_domains"] == 101
    assert result["reconciliation"]["rejected_unique_domains_count"] == 3
    assert result["metrics"]["top1_share"] < 0.10
    assert result["metrics"]["hhi_index"] < 1500


def test_audit_risk_flags_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    result = audit_risk_flags_and_verification(queue_df)
    assert result["status"] == "PASS"
    assert result["flagged_candidates_count"] == 6
    assert result["approved_rationale_coverage"] == 1.0
    assert len(result["rejected_cases_audited"]) == 3


def test_audit_structural_coverage_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    result = audit_structural_feature_coverage(approved_df)
    assert result["status"] == "PASS"
    dist = result["pilot_corpus_distribution"]
    assert dist["path_depth_ge_2_rate"] > 0.80
    assert dist["NoOfOtherSpecialCharsInURL_pct_ge_2"] > 0.80
    assert dist["SpacialCharRatioInURL_pct_ge_0_0785"] > 0.60


def test_audit_quarantine_overlap_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    result = audit_quarantine_overlap(approved_df)
    assert result["status"] == "PASS"
    assert result["exact_collisions_with_benchmark"] == 0
    assert result["normalized_collisions_with_benchmark"] == 0
    assert result["quarantine_guarantee_maintained"] is True


def test_audit_provenance_reproducibility_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    result = audit_provenance_and_reproducibility(pilot_datasets_paths["raw"], queue_df, approved_df)
    assert result["status"] == "PASS"
    assert result["missing_metadata_errors_count"] == 0
    assert result["invalid_metadata_errors_count"] == 0
    assert result["deterministic_reproducibility"]["deterministic_match"] is True


def test_audit_pilot_corpus_cli_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"
    manifest_out = tmp_path / "manifest.json"

    test_args = [
        "audit_pilot_corpus.py",
        "--json-output",
        str(json_out),
        "--md-output",
        str(md_out),
        "--manifest-output",
        str(manifest_out),
    ]
    monkeypatch.setattr("sys.argv", test_args)
    exit_code = main()
    assert exit_code == 0
    assert json_out.exists()
    assert md_out.exists()
    assert manifest_out.exists()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["overall_summary"]["overall_status"] == "PASS"
    assert data["overall_summary"]["approved_candidates"] == 602
    assert data["audit_1_domain_diversity"]["metrics"]["unique_registered_domains"] == 98


# ---------------------------------------------------------------------------
# 2. P0 Adversarial Audit Tests (10 Mandatory Test Scenarios)
# ---------------------------------------------------------------------------

def test_adversarial_1_metadata_tampering_without_url_change_fails_audit(
    pilot_datasets_paths: dict[str, Path], tmp_path: Path
) -> None:
    """Tampering with reviewer or verification_method while keeping URLs intact must fail the audit."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])

    # Tamper with verification_method on one row without changing any URL
    tampered_queue = queue_df.copy()
    tampered_queue.loc[0, "verification_method"] = "tampered_unvetted_script"

    result = audit_provenance_and_reproducibility(pilot_datasets_paths["raw"], tampered_queue, approved_df)
    assert result["status"] == "FAIL"
    assert result["deterministic_reproducibility"]["deterministic_match"] is False
    assert result["pass_criteria"]["deterministic_reingestion_match"] is False


def test_adversarial_2_exact_benchmark_collision_fails_audit() -> None:
    """Inserting an exact URL from external_validation_benchmark.csv must trigger FAIL."""
    colliding_df = pd.DataFrame([{
        "url_exact": "https://www.mit.edu/",  # Known legitimate URL in benchmark
        "registered_domain": "mit.edu",
    }])
    result = audit_quarantine_overlap(colliding_df)
    assert result["status"] == "FAIL"
    assert result["exact_collisions_with_benchmark"] >= 1
    assert result["pass_criteria"]["zero_exact_benchmark_collisions"] is False


def test_adversarial_3_normalized_benchmark_collision_fails_audit() -> None:
    """Inserting a URL that canonicalizes to a benchmark URL must trigger FAIL."""
    # Benchmark contains "https://www.mit.edu/". A URL with uppercase host, default port, or fragment normalizes to it.
    colliding_df = pd.DataFrame([{
        "url_exact": "HTTPS://WWW.MIT.EDU:443/#section-1",
        "registered_domain": "mit.edu",
    }])
    result = audit_quarantine_overlap(colliding_df)
    assert result["status"] == "FAIL"
    assert result["normalized_collisions_with_benchmark"] >= 1
    assert result["pass_criteria"]["zero_normalized_benchmark_collisions"] is False


def test_adversarial_4_missing_approved_reviewer_fails_audit(
    pilot_datasets_paths: dict[str, Path]
) -> None:
    """An approved candidate with an empty reviewer must trigger FAIL."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])

    corrupted_queue = queue_df.copy()
    corrupted_queue.loc[0, "reviewer"] = ""

    result = audit_provenance_and_reproducibility(pilot_datasets_paths["raw"], corrupted_queue, approved_df)
    assert result["status"] == "FAIL"
    assert result["missing_metadata_errors_count"] >= 1
    assert any("missing 'reviewer'" in err for err in result["missing_metadata_errors"])


def test_adversarial_5_missing_approved_verification_method_fails_audit(
    pilot_datasets_paths: dict[str, Path]
) -> None:
    """An approved candidate with an empty verification_method must trigger FAIL."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])

    corrupted_queue = queue_df.copy()
    corrupted_queue.loc[0, "verification_method"] = "   "

    result = audit_provenance_and_reproducibility(pilot_datasets_paths["raw"], corrupted_queue, approved_df)
    assert result["status"] == "FAIL"
    assert result["missing_metadata_errors_count"] >= 1
    assert any("missing 'verification_method'" in err for err in result["missing_metadata_errors"])


def test_adversarial_6_missing_approved_legitimacy_rationale_fails_audit(
    pilot_datasets_paths: dict[str, Path]
) -> None:
    """An approved candidate with an empty legitimacy_reason must trigger FAIL."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])

    corrupted_queue = queue_df.copy()
    corrupted_queue.loc[0, "legitimacy_reason"] = ""

    result = audit_provenance_and_reproducibility(pilot_datasets_paths["raw"], corrupted_queue, approved_df)
    assert result["status"] == "FAIL"
    assert result["missing_metadata_errors_count"] >= 1
    assert any("missing 'legitimacy_reason'" in err for err in result["missing_metadata_errors"])


def test_adversarial_7_unsorted_domain_counter_independence() -> None:
    """Domain concentration metrics must remain invariant under Counter insertion order."""
    # Counter 1: Ascending insertion order (rare first, dominant last)
    counter_asc = Counter()
    counter_asc["rare.org"] = 1
    counter_asc["medium.org"] = 10
    counter_asc["dominant.org"] = 89

    # Counter 2: Descending insertion order (dominant first, rare last)
    counter_desc = Counter()
    counter_desc["dominant.org"] = 89
    counter_desc["medium.org"] = 10
    counter_desc["rare.org"] = 1

    m_asc = compute_concentration_metrics(counter_asc, 100)
    m_desc = compute_concentration_metrics(counter_desc, 100)

    assert m_asc["top1_domain"] == "dominant.org"
    assert m_desc["top1_domain"] == "dominant.org"
    assert m_asc["top1_share"] == 0.89
    assert m_desc["top1_share"] == 0.89
    assert m_asc["top5_share"] == 1.00
    assert m_desc["top5_share"] == 1.00
    assert m_asc["hhi_index"] == m_desc["hhi_index"]
    assert m_asc["shannon_entropy_bits"] == m_desc["shannon_entropy_bits"]
    assert m_asc["gini_coefficient"] == m_desc["gini_coefficient"]


def test_adversarial_8_empty_and_malformed_inputs_handled_safely() -> None:
    """Empty dataframes and zero totals must fail cleanly without crashing."""
    empty_df = pd.DataFrame()

    m_empty = compute_concentration_metrics(Counter(), 0)
    assert m_empty["total_urls"] == 0
    assert m_empty["hhi_index"] == 0.0

    r1 = audit_domain_diversity(empty_df)
    assert r1["status"] == "FAIL"

    r2 = audit_risk_flags_and_verification(empty_df)
    assert r2["status"] == "FAIL"

    r3 = audit_structural_feature_coverage(empty_df)
    assert r3["status"] == "FAIL"

    r4 = audit_quarantine_overlap(empty_df)
    assert r4["status"] == "FAIL"

    raw_fake = Path("nonexistent.csv")
    r5 = audit_provenance_and_reproducibility(raw_fake, empty_df)
    assert r5["status"] == "FAIL"


def test_adversarial_9_corrupted_feature_distribution_fails_audit() -> None:
    """An approved corpus that fails deep link criteria must return FAIL in structural coverage."""
    # Corpus with only bare root URLs (0 path depth, 0 special characters)
    bare_df = pd.DataFrame([
        {
            "url_exact": "https://www.example.org/",
            "is_multilevel_path": False,
            "has_path": False,
            "path_depth": 0,
            "has_trailing_slash": False,
            "has_query": False,
        }
    ])
    result = audit_structural_feature_coverage(bare_df)
    assert result["status"] == "FAIL"
    assert result["pass_criteria"]["deep_link_coverage_gt_80pct"] is False


def test_adversarial_10_deliberately_failing_audit_cli_returns_nonzero_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pilot_datasets_paths: dict[str, Path]
) -> None:
    """A corrupted review queue must cause the CLI main() to return exit code 1."""
    corrupted_queue_path = tmp_path / "corrupted_queue.csv"
    q_df = pd.read_csv(pilot_datasets_paths["queue"])
    # Corrupt review queue by injecting a benchmark collision URL
    q_df.loc[0, "url_exact"] = "https://www.mit.edu/"
    q_df.to_csv(corrupted_queue_path, index=False)

    corrupted_approved_path = tmp_path / "corrupted_approved.csv"
    app_df = pd.read_csv(pilot_datasets_paths["approved"])
    app_df.loc[0, "url_exact"] = "https://www.mit.edu/"
    app_df.to_csv(corrupted_approved_path, index=False)

    test_args = [
        "audit_pilot_corpus.py",
        "--review-queue",
        str(corrupted_queue_path),
        "--approved-corpus",
        str(corrupted_approved_path),
        "--json-output",
        str(tmp_path / "out.json"),
        "--md-output",
        str(tmp_path / "out.md"),
        "--manifest-output",
        str(tmp_path / "out_manifest.json"),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    exit_code = main()
    assert exit_code == 1


# ---------------------------------------------------------------------------
# 3. P1 Versioned Baseline & Comprehensive Audit Manifest Tests
# ---------------------------------------------------------------------------

def test_baseline_artifact_valid() -> None:
    """The live PhiUSIIL legitimate baseline artifact must validate cleanly."""
    baseline_path = Path("experiments/generalization/audit/baselines/phiusiil_legitimate_baseline.json")
    assert baseline_path.exists(), f"Baseline artifact not found at {baseline_path}"
    valid, err, data = validate_baseline_artifact(baseline_path)
    assert valid is True, f"Baseline validation failed: {err}"
    assert err == ""
    assert data["baseline_schema_version"] == "1.0.0"
    assert data["sample_count"] == 134850
    assert len(data["feature_definitions"]) == 18
    assert "structural_indicators" in data["baseline_statistics"]


def test_baseline_artifact_missing_fails(pilot_datasets_paths: dict[str, Path], tmp_path: Path) -> None:
    """Audit 3 must fail cleanly if the baseline artifact is missing."""
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    missing_path = tmp_path / "nonexistent_baseline.json"
    result = audit_structural_feature_coverage(approved_df, baseline_path=missing_path)
    assert result["status"] == "FAIL"
    assert result["baseline_validation_status"] == "FAIL"
    assert "not found" in result["baseline_validation_error"]
    assert result["pass_criteria"]["baseline_artifact_valid"] is False


def test_baseline_artifact_malformed_json_fails(pilot_datasets_paths: dict[str, Path], tmp_path: Path) -> None:
    """Audit 3 must fail cleanly if the baseline artifact contains broken JSON."""
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    corrupted_path = tmp_path / "corrupted_baseline.json"
    corrupted_path.write_text("{ broken json, not valid ...", encoding="utf-8")
    result = audit_structural_feature_coverage(approved_df, baseline_path=corrupted_path)
    assert result["status"] == "FAIL"
    assert result["baseline_validation_status"] == "FAIL"
    assert "malformed JSON" in result["baseline_validation_error"]
    assert result["pass_criteria"]["baseline_artifact_valid"] is False


def test_baseline_artifact_missing_keys_or_incompatible_fails(tmp_path: Path) -> None:
    """Validating an artifact with missing top-level keys or bad features must fail."""
    valid_path = Path("experiments/generalization/audit/baselines/phiusiil_legitimate_baseline.json")
    base_data = json.loads(valid_path.read_text(encoding="utf-8"))

    # Case A: Missing required top-level key
    data_missing_key = dict(base_data)
    del data_missing_key["feature_definitions"]
    p_missing = tmp_path / "missing_key.json"
    p_missing.write_text(json.dumps(data_missing_key), encoding="utf-8")
    valid, err, _ = validate_baseline_artifact(p_missing)
    assert valid is False
    assert "Missing required baseline key" in err

    # Case B: Incompatible sample count (negative or zero)
    data_bad_sample = dict(base_data)
    data_bad_sample["sample_count"] = -5
    p_sample = tmp_path / "bad_sample.json"
    p_sample.write_text(json.dumps(data_bad_sample), encoding="utf-8")
    valid, err, _ = validate_baseline_artifact(p_sample)
    assert valid is False
    assert "Invalid sample_count" in err

    # Case C: Missing model feature in feature_definitions
    data_missing_feat = dict(base_data)
    feats = dict(base_data["feature_definitions"])
    del feats["URLLength"]
    data_missing_feat["feature_definitions"] = feats
    p_feat = tmp_path / "bad_feat.json"
    p_feat.write_text(json.dumps(data_missing_feat), encoding="utf-8")
    valid, err, _ = validate_baseline_artifact(p_feat)
    assert valid is False
    assert "missing model features" in err


def test_baseline_artifact_invalid_rates_fails(tmp_path: Path) -> None:
    """Validating an artifact with invalid rate values (> 1.0 or < 0.0) must fail."""
    valid_path = Path("experiments/generalization/audit/baselines/phiusiil_legitimate_baseline.json")
    base_data = json.loads(valid_path.read_text(encoding="utf-8"))

    data_bad_rate = dict(base_data)
    data_bad_rate["baseline_statistics"] = dict(base_data["baseline_statistics"])
    data_bad_rate["baseline_statistics"]["structural_indicators"] = dict(
        base_data["baseline_statistics"]["structural_indicators"]
    )
    data_bad_rate["baseline_statistics"]["structural_indicators"]["has_path_rate"] = 1.5
    p_bad = tmp_path / "bad_rate.json"
    p_bad.write_text(json.dumps(data_bad_rate), encoding="utf-8")

    valid, err, _ = validate_baseline_artifact(p_bad)
    assert valid is False
    assert "rate must be between 0.0 and 1.0" in err


def test_audit_manifest_structure_and_completeness() -> None:
    """The generated audit manifest must contain all required P1.2 sections and valid hashes."""
    import hashlib

    manifest_path = Path("experiments/generalization/reports/audit_manifest.json")
    assert manifest_path.exists(), f"Audit manifest missing at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Required top-level fields
    assert "manifest_version" in manifest
    assert "execution_timestamp_utc" in manifest
    assert "audit_overall_status" in manifest
    assert manifest["audit_overall_status"] == "PASS"

    # Environment
    assert "environment" in manifest
    env = manifest["environment"]
    assert "git_commit" in env
    assert "audited_source_commit" in env
    assert "git_branch" in env
    assert "python_version" in env
    assert "package_versions" in env
    assert "pandas" in env["package_versions"]

    # Git metadata
    assert "git_metadata" in manifest
    git_meta = manifest["git_metadata"]
    assert "audited_source_commit" in git_meta
    assert "working_tree_clean_at_audit" in git_meta
    assert "manifest_identity_policy" in git_meta

    # Pipeline versions
    assert "pipeline_versions" in manifest
    p_vers = manifest["pipeline_versions"]
    assert "audit_script_version" in p_vers
    assert "review_queue_hash_version" in p_vers
    assert "normalization_policy_version" in p_vers
    assert "feature_extraction_version" in p_vers

    # Inputs
    assert "inputs" in manifest
    inputs = manifest["inputs"]
    for req_input in ["raw_candidates", "review_queue", "approved_corpus", "baseline_artifact", "quarantined_benchmark"]:
        assert req_input in inputs
        assert inputs[req_input]["exists"] is True
        assert len(inputs[req_input]["sha256"]) == 64
        # Verify SHA-256 against actual file on disk
        actual_path = Path(inputs[req_input]["path"])
        assert actual_path.exists()
        assert hashlib.sha256(actual_path.read_bytes()).hexdigest() == inputs[req_input]["sha256"]

    assert "review_queue" in inputs
    assert "dataset_hash" in inputs["review_queue"]
    assert len(inputs["review_queue"]["dataset_hash"]) == 64

    # Outputs, reproducibility, limitations
    assert "outputs" in manifest
    assert "reproducibility" in manifest
    assert "methodological_limitations" in manifest
    assert "dataset_integrity_vs_ground_truth" in manifest["methodological_limitations"]
    assert "reputation_bias_warning" in manifest["methodological_limitations"]
    assert "distribution_coverage_scope" in manifest["methodological_limitations"]
    assert "hash_serialization_qualification" in manifest["methodological_limitations"]


def test_deterministic_reproducibility_across_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the audit twice with the same inputs produces identical metrics, tables, and dataset hashes."""
    run1_json = tmp_path / "run1.json"
    run1_md = tmp_path / "run1.md"
    run1_manifest = tmp_path / "run1_manifest.json"

    run2_json = tmp_path / "run2.json"
    run2_md = tmp_path / "run2.md"
    run2_manifest = tmp_path / "run2_manifest.json"

    monkeypatch.setattr("sys.argv", [
        "audit_pilot_corpus.py",
        "--json-output", str(run1_json),
        "--md-output", str(run1_md),
        "--manifest-output", str(run1_manifest),
    ])
    code1 = main()
    assert code1 == 0

    monkeypatch.setattr("sys.argv", [
        "audit_pilot_corpus.py",
        "--json-output", str(run2_json),
        "--md-output", str(run2_md),
        "--manifest-output", str(run2_manifest),
    ])
    code2 = main()
    assert code2 == 0

    d1 = json.loads(run1_json.read_text(encoding="utf-8"))
    d2 = json.loads(run2_json.read_text(encoding="utf-8"))

    # Audit findings, counts, shares, entropy, Gini, and review queue hash must be identical
    assert d1["audit_1_domain_diversity"] == d2["audit_1_domain_diversity"]
    assert d1["audit_2_risk_flags_and_verification"] == d2["audit_2_risk_flags_and_verification"]
    assert d1["audit_3_structural_feature_coverage"] == d2["audit_3_structural_feature_coverage"]
    assert d1["audit_4_quarantine_overlap"] == d2["audit_4_quarantine_overlap"]
    assert d1["audit_5_provenance_reproducibility"] == d2["audit_5_provenance_reproducibility"]
    assert d1["overall_summary"] == d2["overall_summary"]

    # Manifest inputs and reproducibility metadata must match
    m1 = json.loads(run1_manifest.read_text(encoding="utf-8"))
    m2 = json.loads(run2_manifest.read_text(encoding="utf-8"))
    assert m1["inputs"] == m2["inputs"]
    assert m1["pipeline_versions"] == m2["pipeline_versions"]
    assert m1["audit_overall_status"] == m2["audit_overall_status"]


def test_adversarial_delimiter_injection(pilot_datasets_paths: dict[str, Path]) -> None:
    """Inserting delimiters (|, commas, quotes, newlines) must not collide or break canonical JSON."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    h_clean = compute_review_queue_hash(queue_df)

    # Injected queue with pipe, quotes, commas in legitimacy_reason
    q_injected = queue_df.copy()
    q_injected.loc[0, "legitimacy_reason"] = "Standard RFC | candidate, key='val'\nsecondary text"
    h_injected = compute_review_queue_hash(q_injected)

    assert h_clean != h_injected
    assert len(h_injected) == 64


def test_adversarial_null_sentinel_collision(pilot_datasets_paths: dict[str, Path]) -> None:
    """A literal string '__NULL__' must produce a strictly different hash from an actual null/NaN."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])

    q_nan = queue_df.copy()
    q_nan.loc[0, "notes"] = None
    h_null = compute_review_queue_hash(q_nan)

    q_str = queue_df.copy()
    q_str.loc[0, "notes"] = "__NULL__"
    h_sentinel = compute_review_queue_hash(q_str)

    assert h_null != h_sentinel


def test_adversarial_whitespace_changes(pilot_datasets_paths: dict[str, Path]) -> None:
    """Changing whitespace in strings must alter the canonical hash (exactness preservation)."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    h_orig = compute_review_queue_hash(queue_df)

    q_ws = queue_df.copy()
    q_ws.loc[0, "notes"] = " " + str(q_ws.loc[0, "notes"])
    h_ws = compute_review_queue_hash(q_ws)

    assert h_orig != h_ws


def test_adversarial_duplicate_rows_rejected(pilot_datasets_paths: dict[str, Path]) -> None:
    """Injecting a duplicate candidate_id or row must raise ValueError and fail the audit."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])

    q_dup = pd.concat([queue_df, queue_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate candidate_id"):
        compute_review_queue_hash(q_dup)

    result = audit_provenance_and_reproducibility(pilot_datasets_paths["raw"], q_dup, approved_df)
    assert result["status"] == "FAIL"
    assert any("Duplicate candidate_id" in err for err in result["invalid_metadata_errors"])


def test_adversarial_type_changes(pilot_datasets_paths: dict[str, Path]) -> None:
    """Changing field type (e.g., boolean to string or int to string) must alter the hash."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    h_orig = compute_review_queue_hash(queue_df)

    # Convert a boolean column to string "true" / "false"
    q_type = queue_df.copy()
    q_type["syntax_valid"] = q_type["syntax_valid"].astype(str)
    h_type = compute_review_queue_hash(q_type)

    assert h_orig != h_type


def test_adversarial_missing_and_unexpected_columns(pilot_datasets_paths: dict[str, Path]) -> None:
    """Missing canonical columns or unexpected extra columns must raise ValueError."""
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])

    # Missing column
    q_missing = queue_df.drop(columns=["candidate_id"])
    with pytest.raises(ValueError, match="Missing canonical columns"):
        compute_review_queue_hash(q_missing)

    # Unexpected extra column
    q_unexpected = queue_df.copy()
    q_unexpected["malicious_extra_col"] = 1
    with pytest.raises(ValueError, match="Unexpected columns"):
        compute_review_queue_hash(q_unexpected)


def test_baseline_internal_consistency_reconciliation(tmp_path: Path) -> None:
    """A baseline with contradictory URLLength_mean or NoOfOtherSpecialCharsInURL_mean must fail."""
    valid_path = Path("experiments/generalization/audit/baselines/phiusiil_legitimate_baseline.json")
    base_data = json.loads(valid_path.read_text(encoding="utf-8"))

    # Contradictory URLLength_mean
    bad_data = json.loads(json.dumps(base_data))
    bad_data["baseline_statistics"]["structural_indicators"]["URLLength_mean"] = 30.64
    p_bad = tmp_path / "bad_consistency.json"
    p_bad.write_text(json.dumps(bad_data), encoding="utf-8")

    valid, err, _ = validate_baseline_artifact(p_bad)
    assert valid is False
    assert "Contradictory URLLength_mean" in err


def test_baseline_sha256_format_and_raw_provenance(tmp_path: Path) -> None:
    """Malformed dataset SHA-256 or mismatched raw dataset must fail validation."""
    valid_path = Path("experiments/generalization/audit/baselines/phiusiil_legitimate_baseline.json")
    base_data = json.loads(valid_path.read_text(encoding="utf-8"))

    # Malformed SHA-256 (not 64 hex chars)
    bad_sha_data = json.loads(json.dumps(base_data))
    bad_sha_data["dataset_sha256"] = "invalid_sha256"
    p_sha = tmp_path / "bad_sha.json"
    p_sha.write_text(json.dumps(bad_sha_data), encoding="utf-8")
    valid, err, _ = validate_baseline_artifact(p_sha)
    assert valid is False
    assert "Invalid or malformed dataset_sha256 format" in err

    # Raw dataset mismatch
    fake_raw = tmp_path / "fake_phiusiil.csv"
    fake_raw.write_text("corrupted content", encoding="utf-8")
    valid, err, _ = validate_baseline_artifact(valid_path, raw_dataset_path=fake_raw)
    assert valid is False
    assert "Raw dataset SHA-256 mismatch" in err
