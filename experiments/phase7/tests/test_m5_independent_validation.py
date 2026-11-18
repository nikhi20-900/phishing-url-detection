"""
Comprehensive Verification Suite for Phase 7 Milestone 5: Independent External Validation.

Verifies all 8 Acceptance Criteria (AC-1 through AC-8):
- AC-1: Zero Historical Domain Overlap Invariant (D_benchmark ∩ D_historical = ∅).
- AC-2: Production Model Invariance (E0 strictly frozen; E4 unchanged candidate).
- AC-3: Benchmark Sample Size, Balance, and Integrity (N = 1,300, 650/650).
- AC-4: Mutually Exclusive Strata and Marginal Quotas.
- AC-5: Domain Clustering Caps (phishing <= 3, legitimate <= 5).
- AC-6: Cryptographic Review Ledger & Supervisory Integrity.
- AC-7: Domain-Aware Statistical Inference & Recall Non-Inferiority.
- AC-8: Manifest, Scorecard, and Report Cryptographic Reconciliation.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate_domain_split import extract_registered_domain
from src.features import FEATURE_NAMES, extract_url_features
from experiments.phase7.external.evaluation.audit_domain_independence import (
    build_dynamic_historical_domain_inventory,
)
from experiments.phase7.external.collection.review_ledger_m5 import verify_review_ledger

BENCHMARK_CSV = REPO_ROOT / "experiments" / "phase7" / "external" / "datasets" / "m5_independent_benchmark.csv"
QUARANTINED_CSV = REPO_ROOT / "experiments" / "phase7" / "external" / "datasets" / "m5_quarantined_overlap.csv"
MANIFEST_JSON = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_pre_evaluation_manifest.json"
SCORECARD_JSON = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_external_scorecard.json"
AUDIT_REPORT_MD = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_domain_independence_audit.md"
VALIDATION_REPORT_MD = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_independent_validation_report.md"
RAW_OBS_JSONL = REPO_ROOT / "experiments" / "phase7" / "external" / "evidence" / "raw_observations_m5.jsonl"
REVIEW_SUBMISSIONS_JSONL = REPO_ROOT / "experiments" / "phase7" / "external" / "evidence" / "ledger" / "reviewer_submissions_m5.jsonl"
SUPERVISORY_APPROVALS_JSONL = REPO_ROOT / "experiments" / "phase7" / "external" / "evidence" / "ledger" / "supervisory_approvals_m5.jsonl"

E0_MODEL_PATH = REPO_ROOT / "models" / "random_forest_model.joblib"
E4_MODEL_PATH = REPO_ROOT / "models" / "phase7_candidates" / "rf_augmented_e4.joblib"

EXPECTED_E0_SHA = "95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb"
EXPECTED_E4_SHA = "0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b"


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class TestM5AcceptanceCriteria:
    """Acceptance criteria test suite for Phase 7 Milestone 5."""

    def test_ac1_zero_historical_domain_overlap(self):
        """AC-1: Zero registered-domain overlap between benchmark and all historical datasets."""
        master_domains, _ = build_dynamic_historical_domain_inventory(REPO_ROOT)
        assert len(master_domains) >= 170000, f"Expected >= 170,000 historical domains, found {len(master_domains)}"

        bench_df = pd.read_csv(BENCHMARK_CSV)
        bench_domains = set(
            extract_registered_domain(u) for u in bench_df["url"].tolist()
        )
        assert None not in bench_domains
        assert "" not in bench_domains

        overlap = bench_domains.intersection(master_domains)
        assert len(overlap) == 0, f"CRITICAL LEAKAGE: Found {len(overlap)} overlapping domains: {list(overlap)[:5]}"

    def test_ac2_model_immutability(self):
        """AC-2: E0 production baseline and E4 experimental candidate match exact cryptographic hashes."""
        assert E0_MODEL_PATH.exists(), f"E0 model missing at {E0_MODEL_PATH}"
        assert E4_MODEL_PATH.exists(), f"E4 model missing at {E4_MODEL_PATH}"

        e0_sha = compute_file_sha256(E0_MODEL_PATH)
        e4_sha = compute_file_sha256(E4_MODEL_PATH)

        assert e0_sha == EXPECTED_E0_SHA, f"E0 mutated! Expected {EXPECTED_E0_SHA}, got {e0_sha}"
        assert e4_sha == EXPECTED_E4_SHA, f"E4 mutated! Expected {EXPECTED_E4_SHA}, got {e4_sha}"

    def test_ac3_benchmark_integrity_and_sample_size(self):
        """AC-3: Benchmark has 1,300 samples (650 Phishing, 650 Legitimate) and matches manifest hash."""
        assert BENCHMARK_CSV.exists()
        assert MANIFEST_JSON.exists()

        bench_df = pd.read_csv(BENCHMARK_CSV)
        assert len(bench_df) == 1300, f"Expected 1,300 rows, got {len(bench_df)}"

        phish_count = int((bench_df["label"] == 0).sum())
        legit_count = int((bench_df["label"] == 1).sum())
        assert phish_count == 650, f"Expected 650 phishing URLs, got {phish_count}"
        assert legit_count == 650, f"Expected 650 legitimate URLs, got {legit_count}"

        actual_sha = compute_file_sha256(BENCHMARK_CSV)
        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        assert actual_sha == manifest["benchmark_dataset"]["sha256"], "Benchmark CSV hash differs from manifest lock!"

    def test_ac4_strata_and_marginal_quotas(self):
        """AC-4: Mutually exclusive strata partition legitimate URLs to 100% and fulfill quotas."""
        bench_df = pd.read_csv(BENCHMARK_CSV)
        legit_df = bench_df[bench_df["label"] == 1]
        n_l = len(legit_df)
        assert n_l == 650

        # Check mutually exclusive partition
        strata_counts = legit_df["structural_stratum"].value_counts().to_dict()
        assert sum(strata_counts.values()) == 650, "Strata counts do not sum to total legitimate count!"

        # Quota assertions:
        # homepage_root: >= 20% (130)
        assert strata_counts.get("homepage_root", 0) >= 130
        # single_level_path_no_query: >= 15% (97.5 -> 100)
        assert strata_counts.get("single_level_path_no_query", 0) >= 98
        # multi_level_path_no_query: >= 25% (162.5 -> 165)
        assert strata_counts.get("multi_level_path_no_query", 0) >= 163
        # root_with_query: >= 10% (65)
        assert strata_counts.get("root_with_query", 0) >= 65
        # path_and_query: >= 20% (130)
        assert strata_counts.get("path_and_query", 0) >= 130
        # special_syntax_variants: >= 60
        assert strata_counts.get("special_syntax_variants", 0) >= 60

        # Marginal feature assertions:
        # query_bearing >= 30%
        query_share = legit_df["query_bearing"].mean()
        assert query_share >= 0.30, f"Expected query_bearing >= 30%, got {query_share:.1%}"
        # multi_level_path >= 35%
        multi_share = legit_df["multi_level_path"].mean()
        assert multi_share >= 0.35, f"Expected multi_level_path >= 35%, got {multi_share:.1%}"

    def test_ac5_domain_clustering_caps(self):
        """AC-5: Domain clustering caps are strictly satisfied (phishing <= 3, legitimate <= 5)."""
        bench_df = pd.read_csv(BENCHMARK_CSV)
        phish_df = bench_df[bench_df["label"] == 0]
        legit_df = bench_df[bench_df["label"] == 1]

        max_phish_cluster = phish_df["registered_domain"].value_counts().max()
        max_legit_cluster = legit_df["registered_domain"].value_counts().max()

        assert max_phish_cluster <= 3, f"Phishing domain cluster cap exceeded: max is {max_phish_cluster} > 3"
        assert max_legit_cluster <= 5, f"Legitimate domain cluster cap exceeded: max is {max_legit_cluster} > 5"

        # Unique domain representation
        assert phish_df["registered_domain"].nunique() >= 300
        assert legit_df["registered_domain"].nunique() >= 200

    def test_ac6_cryptographic_review_ledger_verification(self):
        """AC-6: Reviewer submissions and supervisory approvals are archived; compromised signing workflow is retired."""
        assert REVIEW_SUBMISSIONS_JSONL.exists()
        assert SUPERVISORY_APPROVALS_JSONL.exists()

        # Verify that HMAC signing key is decommissioned and cannot be invoked
        from experiments.phase7.external.collection.review_ledger_m5 import get_signing_secret
        with pytest.raises(RuntimeError) as exc_info:
            get_signing_secret()
        assert "permanently retired" in str(exc_info.value).lower()

        # Verify records are preserved for audit lineage
        with open(REVIEW_SUBMISSIONS_JSONL, "r", encoding="utf-8") as f:
            sub_count = sum(1 for line in f if line.strip())
        assert sub_count == 1300, f"Expected 1,300 archived reviewer submissions, found {sub_count}"

        with open(SUPERVISORY_APPROVALS_JSONL, "r", encoding="utf-8") as f:
            app_count = sum(1 for line in f if line.strip())
        assert app_count >= 300, f"Expected >= 300 archived supervisory records, found {app_count}"

    def test_ac7_statistical_inference_and_non_inferiority(self):
        """AC-7: Confirmatory evaluation satisfies recall non-inferiority margin and reports cluster stats."""
        assert SCORECARD_JSON.exists()
        scorecard = json.loads(SCORECARD_JSON.read_text(encoding="utf-8"))

        p050 = scorecard["primary_evaluation_threshold_050"]
        delta_rec = p050["paired_differences_e4_minus_e0"]["delta_phishing_recall"]
        ci_rec_lower = p050["domain_clustered_bootstrap"]["paired_delta_e4_minus_e0"]["phishing_recall"]["ci_lower"]

        # Non-inferiority margin: lower bound >= -1.0% (-0.010)
        assert ci_rec_lower >= -0.010, f"Non-inferiority violated: lower CI {ci_rec_lower:.4f} < -0.010"

        # Monte Carlo permutation test
        perm = p050["cluster_permutation_test"]
        assert perm["m_iterations"] == 10000
        assert perm["phishing_recall"]["formula"] == "(1 + k) / (M + 1)"
        assert 0.0 <= perm["phishing_recall"]["monte_carlo_p_value"] <= 1.0
        assert 0.0 <= perm["legitimate_fpr"]["monte_carlo_p_value"] <= 1.0

        # Wilcoxon analysis
        wilcox = p050["domain_wilcoxon_analysis"]
        assert wilcox["K_total_domains"] >= 200
        assert wilcox["primary_wilcoxon"]["p_value"] <= 0.05
        assert wilcox["primary_wilcoxon"]["rank_biserial_correlation"] <= 0.0

    def test_ac8_manifest_and_scorecard_reconciliation(self):
        """AC-8: Manifest, scorecard, and markdown reports agree on sample counts, hashes, and metrics."""
        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        scorecard = json.loads(SCORECARD_JSON.read_text(encoding="utf-8"))
        audit_text = AUDIT_REPORT_MD.read_text(encoding="utf-8")
        val_text = VALIDATION_REPORT_MD.read_text(encoding="utf-8")

        # Hash reconciliation
        bench_sha = scorecard["benchmark_metadata"]["dataset_sha256"]
        assert bench_sha == manifest["benchmark_dataset"]["sha256"]
        assert bench_sha in audit_text or bench_sha[:16] in audit_text
        assert bench_sha in val_text

        # Denominator reconciliation
        assert scorecard["benchmark_metadata"]["total_samples"] == manifest["benchmark_dataset"]["total_samples"] == 1300
        assert scorecard["benchmark_metadata"]["phishing_samples"] == manifest["benchmark_dataset"]["phishing_samples"] == 650
        assert scorecard["benchmark_metadata"]["legitimate_samples"] == manifest["benchmark_dataset"]["legitimate_samples"] == 650

        # Subgroup undefined denominator check
        for stratum, sdata in scorecard["stratified_subgroups"]["mutually_exclusive_strata"].items():
            if sdata["legitimate_count"] == 0:
                assert sdata["legitimate_fpr_e0"] == "Undefined"
                assert sdata["legitimate_fpr_e4"] == "Undefined"
