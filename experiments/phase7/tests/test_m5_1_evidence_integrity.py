"""
Comprehensive Verification Suite for Phase 7 Milestone 5.1 (M5.1):
Evidence Integrity Audit and Benchmark Qualification.

Verifies all 10 Acceptance Criteria (AC-1 through AC-10):
- AC-1: Security Quarantine and Decommissioning of Compromised HMAC Key.
- AC-2: Fail-Closed Domain Collision Invariant (D_benchmark ∩ D_historical = ∅).
- AC-3: Authentic Live Web Retrieval (Socket network evidence; zero synthetic fallback).
- AC-4: Threat Feed Integrity and Upstream Snapshot Lock (PhishTank verified+online).
- AC-5: Model Immutability & Zero Retraining (E0 and E4 hashes invariant).
- AC-6: Exact Strata Quotas & Marginal Structural Quotas.
- AC-7: Strict Domain Clustering Caps (phishing <= 3, legitimate <= 5, total >= 800 domains).
- AC-8: Domain-Aware Statistical Inference & Non-Inferiority Verification.
- AC-9: Cryptographic Manifest, Scorecard, and Report Reconciliation.
- AC-10: State 3 Statistical Rationale & Stale Wording Regression Guard.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

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
from experiments.phase7.external.collection.ingest_legitimate_sources import (
    DETERMINISTIC_MATRIX,
)
from experiments.phase7.external.collection.ingest_phishing_feeds import (
    EXPECTED_RAW_PHISHTANK_SHA256,
    RAW_PHISHTANK_CACHE,
    extract_structural_stratum,
)
from experiments.phase7.external.evaluation.verify_raw_observations_integrity import (
    validate_raw_observations_integrity,
)

BENCHMARK_CSV = REPO_ROOT / "experiments" / "phase7" / "external" / "datasets" / "m5_1_qualified_benchmark.csv"
QUARANTINED_CSV = REPO_ROOT / "experiments" / "phase7" / "external" / "datasets" / "m5_1_quarantined_overlap.csv"
RAW_OBS_JSONL = REPO_ROOT / "experiments" / "phase7" / "external" / "evidence" / "raw_observations_m5_1.jsonl"
MANIFEST_JSON = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_1_pre_evaluation_manifest.json"
SCORECARD_JSON = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_1_external_scorecard.json"
AUDIT_REPORT_MD = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_1_domain_independence_audit.md"
EVIDENCE_AUDIT_MD = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_1_evidence_integrity_audit.md"

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


class TestM51EvidenceIntegrity:
    """Acceptance criteria test suite for Phase 7 Milestone 5.1."""

    def test_ac1_security_quarantine_of_signing_secret(self):
        """AC-1: Compromised secret key is untracked from Git, ignored, and decommissioned in review ledger."""
        key_rel_path = "experiments/phase7/external/configs/m5_audit_signing_secret.key"
        
        # 1. Assert key is NOT tracked in git index
        res = subprocess.run(
            ["git", "ls-files", key_rel_path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert res.stdout.strip() == "", f"Security leak: {key_rel_path} is still tracked in git!"

        # 2. Assert .gitignore covers key quarantine
        gitignore_content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "m5_audit_signing_secret.key" in gitignore_content or "*.key" in gitignore_content

        # 3. Assert authorized_reviewers.json contains security erratum
        auth_path = REPO_ROOT / "experiments" / "phase7" / "external" / "configs" / "authorized_reviewers.json"
        auth_data = json.loads(auth_path.read_text(encoding="utf-8"))
        assert "security_erratum" in auth_data
        assert "permanently compromised" in auth_data["security_erratum"].lower()

        # 4. Assert HMAC signing workflow is permanently retired
        assert auth_data.get("workflow_status") == "PERMANENTLY_RETIRED_AND_DECOMMISSIONED"
        from experiments.phase7.external.collection.review_ledger_m5 import get_signing_secret
        with pytest.raises(RuntimeError) as exc_info:
            get_signing_secret()
        assert "permanently retired" in str(exc_info.value).lower()

    def test_ac2_fail_closed_domain_collision_invariant(self):
        """AC-2: Dynamic historical inventory is fail-closed, and qualified benchmark has zero historical overlap."""
        # 1. Verify fail-closed behavior on missing file
        with pytest.raises(FileNotFoundError):
            build_dynamic_historical_domain_inventory(
                repo_root=Path("/nonexistent_directory_for_test")
            )

        # 2. Build live dynamic inventory
        master_domains, _ = build_dynamic_historical_domain_inventory(REPO_ROOT)
        assert len(master_domains) >= 170000, f"Expected >= 170,000 historical domains, got {len(master_domains)}"

        # 3. Assert zero overlap between benchmark and master_domains
        bench_df = pd.read_csv(BENCHMARK_CSV)
        bench_domains = set(
            extract_registered_domain(u) for u in bench_df["url"].tolist()
        )
        assert None not in bench_domains
        assert "" not in bench_domains

        overlap_hist = bench_domains.intersection(master_domains)
        assert len(overlap_hist) == 0, f"Historical domain leakage! {len(overlap_hist)} collisions: {list(overlap_hist)[:5]}"

        # 4. Assert zero overlap between phishing and legitimate partitions
        phish_doms = set(extract_registered_domain(u) for u in bench_df[bench_df["label"] == 0]["url"])
        legit_doms = set(extract_registered_domain(u) for u in bench_df[bench_df["label"] == 1]["url"])
        overlap_partitions = phish_doms.intersection(legit_doms)
        assert len(overlap_partitions) == 0, f"Phish/Legit domain collision! {list(overlap_partitions)}"

    def test_ac3_authentic_live_web_retrieval_and_socket_evidence(self):
        """AC-3: Legitimate benchmark is backed by real socket observations with zero synthetic records and 100% verified TLS."""
        assert RAW_OBS_JSONL.exists()
        
        with open(RAW_OBS_JSONL, "r", encoding="utf-8") as f:
            obs_rows = [json.loads(line) for line in f]

        assert len(obs_rows) == 1300, f"Expected 1,300 observations, got {len(obs_rows)}"
        
        # 1. Assert strict 1:1 join and cryptographic URL hash validation between benchmark CSV and observation log
        bench_df = pd.read_csv(BENCHMARK_CSV)
        assert len(bench_df) == 1300
        obs_by_url = {o["url"]: o for o in obs_rows}
        assert len(obs_by_url) == 1300, "Duplicate observation URLs detected!"

        for _, row in bench_df.iterrows():
            url = row["url"]
            assert url in obs_by_url, f"Benchmark URL {url} missing from raw observation log!"
            obs = obs_by_url[url]
            expected_url_sha = hashlib.sha256(url.encode("utf-8")).hexdigest()
            assert obs["url_sha256"] == expected_url_sha, f"URL hash mismatch for {url}"
            assert obs["ground_truth_label"] == row["label"], f"Label mismatch for {url}"
            assert obs["registered_domain"] == row["registered_domain"], f"Domain mismatch for {url}"

        # 2. Assert 100% verified TLS for all 650 legitimate observations
        legit_obs = [o for o in obs_rows if o["ground_truth_label"] == 1]
        assert len(legit_obs) == 650, f"Expected 650 legitimate observations, got {len(legit_obs)}"

        for o in legit_obs:
            assert o["is_synthetic"] is False, f"Found synthetic observation: {o['url']}"
            assert o["url"].startswith("https://"), f"Non-HTTPS legitimate URL: {o['url']}"
            ev = o["observation_evidence"]
            assert ev["tls_verified"] is True, f"Unverified TLS on legitimate URL: {o['url']}"
            assert ev["is_accessible"] is True, f"URL not marked accessible: {o['url']}"
            assert ev["resolved_ip"] is not None, f"Missing resolved IP: {o['url']}"
            assert 200 <= ev["http_status_code"] < 400, f"Invalid HTTP status: {ev['http_status_code']}"
            assert ev["response_sha256"] is not None, f"Missing response body hash: {o['url']}"
            assert len(ev["response_sha256"]) == 64

        # 3. Execute complete independent standalone raw observations validation
        validate_raw_observations_integrity()

    def test_ac4_threat_feed_integrity_and_snapshot_hash(self):
        """AC-4: Phishing candidate ingest requires verified==yes, online==yes, and locks snapshot hash."""
        assert RAW_PHISHTANK_CACHE.exists()
        actual_hash = compute_file_sha256(RAW_PHISHTANK_CACHE)
        assert actual_hash == EXPECTED_RAW_PHISHTANK_SHA256, (
            f"Upstream snapshot mutated! Expected {EXPECTED_RAW_PHISHTANK_SHA256}, got {actual_hash}"
        )

        with open(RAW_OBS_JSONL, "r", encoding="utf-8") as f:
            obs_rows = [json.loads(line) for line in f]

        phish_obs = [o for o in obs_rows if o["ground_truth_label"] == 0]
        assert len(phish_obs) == 650

        for o in phish_obs:
            ev = o["observation_evidence"]
            assert ev["verification_status"].lower() == "yes"
            assert ev["online_status"].lower() == "yes"
            assert ev["upstream_snapshot_sha256"] == EXPECTED_RAW_PHISHTANK_SHA256
            assert o["is_synthetic"] is False

    def test_ac5_model_immutability(self):
        """AC-5: E0 production baseline and E4 experimental candidate match exact cryptographic hashes."""
        assert E0_MODEL_PATH.exists()
        assert E4_MODEL_PATH.exists()

        assert compute_file_sha256(E0_MODEL_PATH) == EXPECTED_E0_SHA
        assert compute_file_sha256(E4_MODEL_PATH) == EXPECTED_E4_SHA

    def test_ac6_exact_strata_quotas_and_marginal_targets(self):
        """AC-6: Legitimate URLs strictly satisfy DETERMINISTIC_MATRIX stratum quotas and marginal targets."""
        bench_df = pd.read_csv(BENCHMARK_CSV)
        assert len(bench_df) == 1300, f"Expected exactly 1,300 benchmark rows, got {len(bench_df)}"
        phish_df = bench_df[bench_df["label"] == 0]
        legit_df = bench_df[bench_df["label"] == 1]
        assert len(phish_df) == 650, f"Expected exactly 650 phishing URLs, got {len(phish_df)}"
        assert len(legit_df) == 650, f"Expected exactly 650 legitimate URLs, got {len(legit_df)}"

        # Check total stratum quotas across legitimate URLs
        expected_strata_totals = {
            "homepage_root": 130,
            "single_level_path_no_query": 100,
            "multi_level_path_no_query": 165,
            "root_with_query": 65,
            "path_and_query": 130,
            "special_syntax_variants": 60,
        }

        actual_strata_totals = legit_df["structural_stratum"].value_counts().to_dict()
        for stratum, expected_cnt in expected_strata_totals.items():
            actual_cnt = actual_strata_totals.get(stratum, 0)
            assert actual_cnt == expected_cnt, (
                f"Stratum '{stratum}' quota mismatch: expected {expected_cnt}, got {actual_cnt}"
            )

        # Marginal feature targets
        query_bearing_pct = (legit_df["query_bearing"] == True).mean()
        multi_level_pct = (legit_df["multi_level_path"] == True).mean()
        assert query_bearing_pct >= 0.30, f"Query-bearing pct {query_bearing_pct:.2%} < 30%"
        assert multi_level_pct >= 0.35, f"Multi-level path pct {multi_level_pct:.2%} < 35%"

    def test_ac7_domain_clustering_caps(self):
        """AC-7: Phishing max <= 3/domain, Legitimate max <= 5/domain, exactly 810 unique domains (650 phish, 160 legit)."""
        bench_df = pd.read_csv(BENCHMARK_CSV)
        
        phish_counts = bench_df[bench_df["label"] == 0]["registered_domain"].value_counts()
        legit_counts = bench_df[bench_df["label"] == 1]["registered_domain"].value_counts()

        # Domain clustering caps
        assert phish_counts.max() <= 3, f"Phishing domain cap violated: {phish_counts.max()} > 3"
        assert legit_counts.max() <= 5, f"Legitimate domain cap violated: {legit_counts.max()} > 5"

        # Exact domain diversity assertions
        assert len(phish_counts) == 650, f"Expected 650 unique phishing domains, got {len(phish_counts)}"
        assert len(legit_counts) == 160, f"Expected 160 unique legitimate domains, got {len(legit_counts)}"
        total_unique_domains = bench_df["registered_domain"].nunique()
        assert total_unique_domains == 810, f"Expected exactly 810 unique domains, got {total_unique_domains}"

    def test_ac8_domain_aware_statistical_inference(self):
        """AC-8: Domain-clustered bootstrap, Monte Carlo permutation, and recall non-inferiority."""
        assert SCORECARD_JSON.exists()
        sc = json.loads(SCORECARD_JSON.read_text(encoding="utf-8"))

        diff_050 = sc["comparative_inference_t_050"]
        boot_050 = diff_050["domain_clustered_bootstrap"]["paired_delta_e4_minus_e0"]
        perm_050 = diff_050["monte_carlo_permutation"]

        # 1. Verify Monte Carlo permutation test formula
        assert perm_050["m_iterations"] == 10000
        assert perm_050["phishing_recall"]["formula"] == "(1 + k) / (M + 1)"

        # 2. Verify domain-clustered bootstrap resamples
        assert diff_050["domain_clustered_bootstrap"]["n_bootstrap"] == 1000

        # 3. Verify Non-Inferiority at T=0.50 (Recall delta CI lower bound >= -1.0%)
        ci_lower_rec = boot_050["phishing_recall"]["ci_lower"]
        assert ci_lower_rec >= -0.010, f"Non-inferiority failed: lower CI bound {ci_lower_rec:.4%} < -1.0%"

    def test_ac9_manifest_and_scorecard_reconciliation(self):
        """AC-9: Cryptographic manifest, scorecard, and report reconciliation; decision matrix adherence."""
        assert MANIFEST_JSON.exists()
        assert SCORECARD_JSON.exists()
        assert AUDIT_REPORT_MD.exists()
        assert EVIDENCE_AUDIT_MD.exists()

        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        sc = json.loads(SCORECARD_JSON.read_text(encoding="utf-8"))

        actual_bench_sha = compute_file_sha256(BENCHMARK_CSV)
        actual_obs_sha = compute_file_sha256(RAW_OBS_JSONL)

        assert manifest["benchmark_dataset"]["sha256"] == actual_bench_sha
        assert sc["benchmark_dataset"]["sha256"] == actual_bench_sha
        assert manifest["benchmark_dataset"]["evidence_observations"]["sha256"] == actual_obs_sha
        assert manifest["models"]["frozen_production_control_e0"]["sha256"] == EXPECTED_E0_SHA
        assert manifest["models"]["candidate_model_e4"]["sha256"] == EXPECTED_E4_SHA

        # Verify Decision Matrix adherence: E4 at T=0.75 must NOT be assigned State 2
        decision_075 = sc["comparative_inference_t_075"]["decision_matrix_evaluation"]
        assert decision_075["state"] != "State 2", "Ad-hoc State 2 improperly assigned!"
        assert "State 2 (Empirical Boundary)" not in decision_075["state"], "Ad-hoc empirical State 2 improperly assigned!"
        assert "State 3" in decision_075["state"] or "FAIL" in decision_075["state"], (
            f"Expected State 3 or FAIL due to operational FPR ceiling breach, got {decision_075['state']}"
        )
        assert "significantly degrades" not in decision_075["description"].lower(), (
            "State 3 must not claim significant degradation when Delta_FPR is negative and CI spans zero!"
        )
        assert "massively exceeds" in decision_075["description"].lower()

    def test_ac10_state_3_statistical_rationale_and_stale_wording_guard(self):
        """AC-10: State 3 must not claim significant degradation; verifies artifacts and statistical rationale."""
        assert SCORECARD_JSON.exists()
        assert EVIDENCE_AUDIT_MD.exists()

        sc = json.loads(SCORECARD_JSON.read_text(encoding="utf-8"))
        audit_md = EVIDENCE_AUDIT_MD.read_text(encoding="utf-8")

        # 1. Assert stale wording is absent across both artifacts
        stale_phrase = "significantly degrades false alarm rate relative to control"
        assert stale_phrase not in sc["comparative_inference_t_075"]["decision_matrix_evaluation"]["description"].lower()
        assert stale_phrase not in audit_md.lower()

        # 2. Assert correct statistical justification is present in scorecard
        desc = sc["comparative_inference_t_075"]["decision_matrix_evaluation"]["description"]
        assert "massively exceeds the operational ceiling of 0.10%" in desc
        assert "Delta_FPR = -1.5385%" in desc
        assert "permutation p = 0.2837" in desc
        assert "failing to certify statistical non-inferiority against zero degradation" in desc

        # 3. Assert correct statistical justification is present in audit report
        assert "massively exceeds the operational ceiling of 0.10%" in audit_md
        assert "Delta_FPR = -1.5385%" in audit_md
        assert "permutation p = 0.2837" in audit_md

        # 4. Assert statistical invariants: E4 point estimate is lower than E0, but CI spans zero
        delta_fpr = sc["comparative_inference_t_075"]["delta_fpr_e4_minus_e0"]
        ci_upper = sc["comparative_inference_t_075"]["domain_clustered_bootstrap"]["paired_delta_e4_minus_e0"]["legitimate_fpr"]["ci_upper"]
        ci_lower = sc["comparative_inference_t_075"]["domain_clustered_bootstrap"]["paired_delta_e4_minus_e0"]["legitimate_fpr"]["ci_lower"]
        perm_p = sc["comparative_inference_t_075"]["monte_carlo_permutation"]["legitimate_fpr"]["monte_carlo_p_value"]

        assert delta_fpr < 0, f"Expected negative Delta_FPR (point estimate improvement), got {delta_fpr}"
        assert ci_lower < 0 < ci_upper, f"Confidence interval must span zero: [{ci_lower}, {ci_upper}]"
        assert perm_p > 0.05, f"Permutation p-value must be non-significant: {perm_p}"
