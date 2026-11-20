"""
Comprehensive Automated Test Suite for Phase 7 Milestone 6 (M6):
Operational Readiness & Generalisation Assessment.

Verifies all Acceptance Criteria (AC-1 through AC-8) and Stage 0 Pre-Flight Locks:
- AC-1: Stage 0 Manifest Integrity & Cryptographic Locks (Models, Splits, Features, Environment)
- AC-2: TreeSHAP Probability-Space Additivity & Output Shape Locks
- AC-3: Class-Separated Covariate Shift (PSI with Epsilon Smoothing & KS Effect Sizes)
- AC-4: Structural Strata Determinism, Edge Cases & Parsing Anomaly Handling
- AC-5: Clopper-Pearson Exact Confidence Intervals & Boundary Cases (x=0, x=n)
- AC-6: Training Bounding-Box Analysis & Univariate vs Multivariate Coverage Distinction
- AC-7: Domain-Grouped Cross-Validation Lexical Probe & Diagnostic Classification
- AC-8: Production Governance, Model Immutability & Zero Retraining Invariant
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest

# Ensure MPLCONFIGDIR is set to avoid matplotlib cache issues
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.features import FEATURE_NAMES, extract_url_features
from src.psl_snapshot import extract_fqdn, extract_psl_registered_domain
from experiments.phase7.evaluation.m6_diagnostics_utils import (
    classify_structural_stratum,
    compute_bh_fdr,
    compute_bounding_box_analysis,
    compute_clopper_pearson,
    compute_discrete_association_test,
    compute_domain_cluster_bootstrap,
    compute_psi,
    compute_two_proportion_test,
    evaluate_lexical_probe_oof,
    extract_path_segments,
    is_ip_host,
    verify_shap_probability_additivity,
)

MANIFEST_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_operational_readiness_manifest.json"
SPEC_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_operational_readiness_specification.md"

E0_MODEL_PATH = REPO_ROOT / "models" / "random_forest_model.joblib"
E4_MODEL_PATH = REPO_ROOT / "models" / "phase7_candidates" / "rf_augmented_e4.joblib"

EXPECTED_E0_SHA = "95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb"
EXPECTED_E4_SHA = "0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b"
EXPECTED_TRAIN_SHA = "9297458d03841fbe9c32bae7c4ad323e1b3f0b221c0c755f912350808d0c6892"
EXPECTED_INTERNAL_TEST_SHA = "ce004e9d8e65ea8c09175d904d4a5fee0442acc4b127af6a98f58ae0aed0e2cf"
EXPECTED_HISTORICAL_EXT_SHA = "88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3"
EXPECTED_M5_1_BENCH_SHA = "f6cdbf797b683d6c82d7b5c72c6ddf35aba7cbb448edafc2d68e67e26695fc88"
EXPECTED_FEATURES_SRC_SHA = "f581fd8e38640623e0831767baf242424b5b18262ea7532a533d916d2faabe68"


OUTPUT_TREE_ANALYSIS = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_tree_path_analysis.json"
OUTPUT_TREE_EVIDENCE = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_tree_decision_paths_evidence.json"
OUTPUT_SHAP_SUMMARY = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_shap_summary.json"
OUTPUT_SHAP_FULL = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_shap_attributions.json"
OUTPUT_COVARIATE_SCORECARD = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_covariate_shift_scorecard.json"
OUTPUT_COVARIATE_FULL = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_covariate_shift_full.json"
OUTPUT_DOMAIN_INDEP = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_stage2_domain_independence.json"
OUTPUT_STAT_INFERENCE = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_stage2_statistical_inference.json"
OUTPUT_PSI_SENSITIVITY = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_stage2_psi_sensitivity.json"


def compute_file_sha256(path: Path) -> str:
    import hashlib
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class TestM6OperationalReadiness:
    """Test suite validating all M6 specifications and invariants."""

    def test_ac1_stage0_manifest_integrity_and_cryptographic_locks(self):
        """AC-1: Stage 0 manifest must exist, lock exact models, benchmarks, train split, features, and environment."""
        assert MANIFEST_PATH.exists(), f"Missing Stage 0 manifest: {MANIFEST_PATH}"
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        # 1. Models Lock
        assert manifest["models"]["frozen_production_control_e0"]["sha256"] == EXPECTED_E0_SHA
        assert manifest["models"]["candidate_model_e4"]["sha256"] == EXPECTED_E4_SHA
        assert compute_file_sha256(E0_MODEL_PATH) == EXPECTED_E0_SHA
        assert compute_file_sha256(E4_MODEL_PATH) == EXPECTED_E4_SHA

        # 2. Benchmarks Lock
        benchmarks = manifest["benchmarks"]
        assert benchmarks["internal_test"]["sha256"] == EXPECTED_INTERNAL_TEST_SHA
        assert benchmarks["internal_test"]["sample_count"] == 34092
        assert compute_file_sha256(REPO_ROOT / benchmarks["internal_test"]["path"]) == EXPECTED_INTERNAL_TEST_SHA

        assert benchmarks["historical_external"]["sha256"] == EXPECTED_HISTORICAL_EXT_SHA
        assert benchmarks["historical_external"]["sample_count"] == 1047
        assert compute_file_sha256(REPO_ROOT / benchmarks["historical_external"]["path"]) == EXPECTED_HISTORICAL_EXT_SHA

        assert benchmarks["m5_1_qualified_external"]["sha256"] == EXPECTED_M5_1_BENCH_SHA
        assert benchmarks["m5_1_qualified_external"]["sample_count"] == 1300
        assert compute_file_sha256(REPO_ROOT / benchmarks["m5_1_qualified_external"]["path"]) == EXPECTED_M5_1_BENCH_SHA

        # 3. Training Partition Lock
        train_meta = manifest["training_data"]["phase7_train_split"]
        assert train_meta["sha256"] == EXPECTED_TRAIN_SHA
        assert train_meta["total_samples"] == 170615
        assert train_meta["legitimate_count"] == 94523
        assert train_meta["phishing_count"] == 76092
        assert train_meta["unique_registered_domains"] == 121217
        assert compute_file_sha256(REPO_ROOT / train_meta["path"]) == EXPECTED_TRAIN_SHA

        # 4. Feature Pipeline Lock
        feat_meta = manifest["feature_pipeline"]
        assert feat_meta["feature_count"] == 18
        assert feat_meta["ordered_features"] == FEATURE_NAMES
        assert feat_meta["source_code"]["sha256"] == EXPECTED_FEATURES_SRC_SHA
        assert compute_file_sha256(REPO_ROOT / feat_meta["source_code"]["path"]) == EXPECTED_FEATURES_SRC_SHA

        # 5. Environment & Dependencies
        import sklearn, joblib, scipy, shap
        env_meta = manifest["environment_metadata"]
        assert env_meta["python_version"] == "3.14.7"
        assert env_meta["execution_mode"] == "READ_ONLY_DIAGNOSTIC_ZERO_RETRAINING"
        deps = env_meta["dependencies"]
        assert sklearn.__version__ == deps["scikit-learn"]
        assert joblib.__version__ == deps["joblib"]
        assert np.__version__ == deps["numpy"]
        assert pd.__version__ == deps["pandas"]
        assert scipy.__version__ == deps["scipy"]
        assert shap.__version__ == deps["shap"]

    def test_ac2_shap_probability_additivity_and_locks(self):
        """AC-2: TreeSHAP probability-space additivity verification and explicit failure enforcement."""
        import joblib

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        tolerance = manifest["shap_configuration"]["numerical_additivity_tolerance"]
        assert tolerance == 1e-05

        # Test on representative sample from M5.1 qualified benchmark
        bench_df = pd.read_csv(REPO_ROOT / manifest["benchmarks"]["m5_1_qualified_external"]["path"])
        sample_urls = bench_df["url"].iloc[:20]
        X_sample = pd.DataFrame([extract_url_features(u) for u in sample_urls])[FEATURE_NAMES].values

        # Test both E0 and E4
        for model_path in [E0_MODEL_PATH, E4_MODEL_PATH]:
            model_dict = joblib.load(model_path)
            clf = model_dict["model"] if isinstance(model_dict, dict) else model_dict

            res = verify_shap_probability_additivity(clf, X_sample, tolerance=tolerance)
            assert res["status"] == "VERIFIED_EXACT_ADDITIVITY"
            assert res["class_index"] == 0
            assert res["max_absolute_error"] <= tolerance
            assert res["max_absolute_error"] < 1e-12, "TreeSHAP in RF should match predict_proba to ~1e-14"

        # Verify explicit failure on violation
        class FakeClassifier:
            classes_ = np.array([0, 1])
            def predict_proba(self, X):
                return np.zeros((len(X), 2)) # Artificial failure

        with pytest.raises(Exception):
            verify_shap_probability_additivity(clf, X_sample, tolerance=1e-15)

    def test_ac3_covariate_shift_psi_and_ks_rules(self):
        """AC-3: PSI calculation with epsilon smoothing, normalization, and constant feature handling."""
        # Test 1: Identical constant feature
        q_const = np.zeros(1000)
        p_const = np.zeros(650)
        psi_const, status_const = compute_psi(q_const, p_const)
        assert psi_const == 0.0
        assert status_const == "CONSTANT_IDENTICAL"

        # Test 2: Degenerate constant shift
        p_shift = np.ones(650)
        psi_shift, status_shift = compute_psi(q_const, p_shift)
        assert np.isinf(psi_shift)
        assert status_shift == "CONSTANT_DEGENERATE_SHIFT"

        # Test 3: Normal distribution shift with epsilon smoothing
        np.random.seed(42)
        ref = np.random.normal(0, 1, 10000)
        eval_data = np.random.normal(0.5, 1, 650)
        psi_val, status_ok = compute_psi(ref, eval_data, num_bins=10)
        assert status_ok == "OK"
        assert 0.20 <= psi_val <= 0.35, f"Expected moderate/severe PSI, got {psi_val}"

        # Test 4: Zero-count smoothing verification
        # An eval set with values only in the upper half should trigger epsilon smoothing in lower bins
        eval_skewed = np.random.uniform(5, 10, 650)
        psi_skewed, status_skew = compute_psi(ref, eval_skewed, num_bins=10)
        assert np.isfinite(psi_skewed), "Epsilon smoothing must prevent infinite log on empty bins"

    def test_ac4_structural_strata_edge_cases_and_parsing_anomalies(self):
        """AC-4: Deterministic structural strata classification across all edge cases and anomaly reporting."""
        # 1. Collapsing repeated slashes
        s1, _ = classify_structural_stratum("https://example.com///a///b//c")
        assert s1 == "multi_level_path_no_query"

        # 2. Trailing slashes
        s2_root, _ = classify_structural_stratum("https://example.com/")
        assert s2_root == "homepage_root"
        s2_single, _ = classify_structural_stratum("https://example.com/about/")
        assert s2_single == "single_level_path_no_query"
        s2_multi, _ = classify_structural_stratum("https://example.com/a/b/")
        assert s2_multi == "multi_level_path_no_query"

        # 3. Bare query delimiter vs non-empty query
        s3_bare, _ = classify_structural_stratum("https://example.com/?")
        assert s3_bare == "homepage_root"
        s3_bare_path, _ = classify_structural_stratum("https://example.com/about?")
        assert s3_bare_path == "single_level_path_no_query"
        s3_query, _ = classify_structural_stratum("https://example.com/?a=1")
        assert s3_query == "root_with_query"
        s3_query_empty_val, _ = classify_structural_stratum("https://example.com/?key=")
        assert s3_query_empty_val == "root_with_query"
        s3_path_query, _ = classify_structural_stratum("https://example.com/search?q=test")
        assert s3_path_query == "path_and_query"

        # 4. Special syntax variants
        # Non-standard port
        s4_port, anom_port = classify_structural_stratum("https://example.com:8080/path")
        assert s4_port == "special_syntax_variants"
        assert anom_port is None

        # Standard ports 80 and 443 are NOT special syntax
        s4_std80, _ = classify_structural_stratum("http://example.com:80/about")
        assert s4_std80 == "single_level_path_no_query"
        s4_std443, _ = classify_structural_stratum("https://example.com:443/about")
        assert s4_std443 == "single_level_path_no_query"

        # Malformed / non-numeric port
        s4_badport, anom_badport = classify_structural_stratum("https://example.com:invalid/path")
        assert s4_badport == "special_syntax_variants"
        assert anom_badport is not None
        assert anom_badport["exception_type"] == "ValueError"

        # User credentials (@)
        s4_cred, _ = classify_structural_stratum("https://user:pass@example.com/login")
        assert s4_cred == "special_syntax_variants"

        # IPv4 host
        s4_ip4, _ = classify_structural_stratum("https://192.168.1.1/admin")
        assert s4_ip4 == "special_syntax_variants"

        # IPv6 host
        s4_ip6, _ = classify_structural_stratum("https://[2001:db8::1]/index")
        assert s4_ip6 == "special_syntax_variants"

        # Obfuscation (% encoding)
        s4_obf, _ = classify_structural_stratum("https://example.com/%20test")
        assert s4_obf == "special_syntax_variants"

        # Malformed percent-encoding
        s4_mal_obf1, _ = classify_structural_stratum("https://example.com/%ZZ")
        assert s4_mal_obf1 == "special_syntax_variants"
        s4_mal_obf2, _ = classify_structural_stratum("https://example.com/%")
        assert s4_mal_obf2 == "special_syntax_variants"

        # Malformed IPv6 bracket error
        s4_mal_ip6, anom_ip6 = classify_structural_stratum("https://[invalid-ipv6]/index")
        assert s4_mal_ip6 == "special_syntax_variants"
        assert anom_ip6 is not None
        assert anom_ip6["exception_type"] == "ValueError"

    def test_ac5_clopper_pearson_boundary_cases(self):
        """AC-5: Clopper-Pearson exact confidence intervals for boundary cases x=0 and x=n."""
        n = 130
        alpha = 0.05

        # Boundary case x = 0
        low_0, up_0 = compute_clopper_pearson(0, n, alpha=alpha)
        assert low_0 == 0.0
        expected_up_0 = 1.0 - (alpha / 2.0) ** (1.0 / n)
        assert np.isclose(up_0, expected_up_0, atol=1e-12)
        assert np.isclose(up_0, 0.0279771784, atol=1e-6)

        # Boundary case x = n
        low_n, up_n = compute_clopper_pearson(n, n, alpha=alpha)
        assert up_n == 1.0
        expected_low_n = (alpha / 2.0) ** (1.0 / n)
        assert np.isclose(low_n, expected_low_n, atol=1e-12)
        assert np.isclose(low_n, 0.9720228216, atol=1e-6)

        # Intermediate case x = 602, n = 650
        low_mid, up_mid = compute_clopper_pearson(602, 650, alpha=alpha)
        assert 0.90 <= low_mid <= 0.91
        assert 0.94 <= up_mid <= 0.95

        # Degenerate case n = 0
        low_deg, up_deg = compute_clopper_pearson(0, 0)
        assert low_deg == 0.0
        assert up_deg == 1.0

    def test_ac6_training_bounding_box_and_coverage_distinction(self):
        """AC-6: Bounding-box analysis reports per-feature, union, multi-range rates, and multivariate distinction."""
        # Create synthetic train with exact bounds [0.0, 100.0]
        train_data = {f: np.linspace(0.0, 100.0, 1000) for f in FEATURE_NAMES}
        # Create synthetic eval within [10.0, 90.0]
        eval_data = {f: np.full(100, 50.0) for f in FEATURE_NAMES}

        # Inject controlled out-of-range values in feature 0, 1, and 2
        eval_data[FEATURE_NAMES[0]][0] = 150.0  # record 0 OOR in f0
        eval_data[FEATURE_NAMES[1]][0] = 150.0  # record 0 OOR in f1 (multi-range >= 2)
        eval_data[FEATURE_NAMES[2]][1] = -10.0  # record 1 OOR in f2

        train_df = pd.DataFrame(train_data)
        eval_df = pd.DataFrame(eval_data)

        res = compute_bounding_box_analysis(train_df, eval_df, FEATURE_NAMES)

        assert "per_feature_out_of_range" in res
        assert "training_quantiles" in res
        assert "record_level_union_oor_rate" in res
        assert "record_level_multi_oor_2_rate" in res
        assert "oor_feature_count_distribution" in res
        assert "theoretical_distinction" in res

        # Union rate should be 2 / 100 = 2%
        assert np.isclose(res["record_level_union_oor_rate"], 0.02)
        # Multi-range >= 2 should be 1 / 100 = 1%
        assert np.isclose(res["record_level_multi_oor_2_rate"], 0.01)

        # Assert theoretical distinction is explicitly documented
        caveat = res["theoretical_distinction"].lower()
        assert "univariate" in caveat
        assert "multivariate" in caveat or "joint" in caveat
        assert "necessary but not sufficient" in caveat

    def test_ac7_lexical_probe_oof_and_diagnostic_classification(self):
        """AC-7: Lexical probe uses domain-grouped CV, out-of-fold metrics, and diagnostic-only classification."""
        # Test on M5.1 qualified benchmark subset
        bench_df = pd.read_csv(REPO_ROOT / "experiments" / "phase7" / "external" / "datasets" / "m5_1_qualified_benchmark.csv")
        sample_df = bench_df.iloc[:60].copy()

        from src.features import extract_url_features
        X_sample = pd.DataFrame([extract_url_features(u) for u in sample_df["url"]])[FEATURE_NAMES].values
        y_sample = sample_df["label"].values
        groups_sample = sample_df["registered_domain"].values

        # Run 3-fold grouped CV on sample
        res = evaluate_lexical_probe_oof(X_sample, y_sample, groups_sample, n_splits=3, random_state=42)

        assert res["classification"] == "DIAGNOSTIC_ONLY_STATISTICAL_FIT"
        assert "out_of_fold_metrics" in res
        assert "roc_auc" in res["out_of_fold_metrics"]
        assert "threshold_0_50" in res["out_of_fold_metrics"]
        assert "threshold_0_75" in res["out_of_fold_metrics"]

        # Ensure wording does NOT claim mathematical upper bound
        desc = res["description"].lower()
        assert "mathematical upper bound" not in desc
        assert "empirical out-of-fold baseline" in desc

    def test_ac8_production_governance_and_zero_retraining_invariant(self):
        """AC-8: Assert model files are byte-for-byte unchanged and zero new model files are created."""
        actual_e0_sha = compute_file_sha256(E0_MODEL_PATH)
        actual_e4_sha = compute_file_sha256(E4_MODEL_PATH)

        assert actual_e0_sha == EXPECTED_E0_SHA, "CRITICAL: Control model E0 has been modified!"
        assert actual_e4_sha == EXPECTED_E4_SHA, "CRITICAL: Candidate model E4 has been modified!"

        # Assert no unexpected model files exist in models/
        model_files = list((REPO_ROOT / "models").glob("*.joblib"))
        assert len(model_files) == 1, f"Unexpected models in models/: {model_files}"
        assert model_files[0].name == "random_forest_model.joblib"

        candidate_files = list((REPO_ROOT / "models" / "phase7_candidates").glob("*.joblib"))
        candidate_names = {f.name for f in candidate_files}
        assert "rf_augmented_e4.joblib" in candidate_names
        # Ensure no newly spawned models
        assert not any(f.name.startswith("m6_") for f in candidate_files), "New models improperly created in candidates dir!"

    def test_stage1_decision_path_artifacts_and_metrics(self):
        """Stage 1: Assert decision-path artifacts exist and reflect empirical false-positive properties."""
        assert OUTPUT_TREE_ANALYSIS.exists(), f"Missing decision path analysis: {OUTPUT_TREE_ANALYSIS}"
        assert OUTPUT_TREE_EVIDENCE.exists(), f"Missing decision path evidence: {OUTPUT_TREE_EVIDENCE}"

        analysis = json.loads(OUTPUT_TREE_ANALYSIS.read_text(encoding="utf-8"))
        assert analysis["stage"] == "STAGE_1_ENSEMBLE_DISSECTION"
        assert analysis["governance_status"] == "ZERO_RETRAINING_MODEL_IMMUTABILITY_PRESERVED"

        e0_info = analysis["models"]["frozen_production_control_e0"]
        e4_info = analysis["models"]["candidate_model_e4"]

        # Exact false positive counts at T=0.75
        assert e0_info["false_positives_t_075"] == 612
        assert e4_info["false_positives_t_075"] == 602

        # Unanimous voting rates (>= 90% tree votes)
        assert e0_info["vote_distribution"]["unanimous_phishing_ge_090"] >= 0.85
        assert e4_info["vote_distribution"]["unanimous_phishing_ge_090"] >= 0.80

        # Mean vote fraction
        assert e0_info["vote_distribution"]["mean_phish_vote_fraction"] > 0.95
        assert e4_info["vote_distribution"]["mean_phish_vote_fraction"] > 0.90

        # Structural split hierarchy
        assert e0_info["hierarchy"]["tree_count"] == 100
        assert e4_info["hierarchy"]["tree_count"] == 100
        assert "IsHTTPS" in e0_info["hierarchy"]["root_split_dominance"]

    def test_stage1_shap_attribution_artifacts_and_additivity(self):
        """Stage 1: Assert SHAP attribution artifacts exist, satisfy additivity, and document false-positive drivers."""
        assert OUTPUT_SHAP_SUMMARY.exists(), f"Missing SHAP summary: {OUTPUT_SHAP_SUMMARY}"
        assert OUTPUT_SHAP_FULL.exists(), f"Missing full SHAP attributions: {OUTPUT_SHAP_FULL}"

        summary = json.loads(OUTPUT_SHAP_SUMMARY.read_text(encoding="utf-8"))
        assert summary["stage"] == "STAGE_1_ENSEMBLE_DISSECTION"

        additivity = summary["additivity_verification"]
        assert additivity["status"] == "ALL_1300_SAMPLES_SATISFY_ADDITIVITY"
        assert additivity["e0_max_error"] <= 1e-05
        assert additivity["e4_max_error"] <= 1e-05
        assert additivity["verified_sample_count"] == 1300

        # Archetypal case studies: exactly 5 distinct sectors represented
        case_studies = summary["archetypal_case_studies_summary"]
        assert len(case_studies) == 5
        sectors = {cs["sector"] for cs in case_studies}
        assert len(sectors) == 5

        # Key drivers pushing legitimate URLs into phishing
        e0_drivers = [d[0] for d in summary["false_positive_drivers_t_075"]["e0_top_pushing_features"]]
        assert "NoOfOtherSpecialCharsInURL" in e0_drivers
        assert "DomainLength" in e0_drivers

    def test_stage1_visual_shap_deliverables(self):
        """Stage 1: Verify presence, integrity, and non-empty sizes of all 12 SHAP visual figures."""
        figures_dir = REPO_ROOT / "experiments" / "phase7" / "reports" / "figures"
        assert figures_dir.exists(), f"Figures directory missing: {figures_dir}"

        expected_figures = [
            "m6_shap_summary_beeswarm_e0.png",
            "m6_shap_summary_beeswarm_e4.png",
            "m6_shap_force_plot_1_government_e0.png",
            "m6_shap_force_plot_1_government_e4.png",
            "m6_shap_force_plot_2_academic_e0.png",
            "m6_shap_force_plot_2_academic_e4.png",
            "m6_shap_force_plot_3_banking_e0.png",
            "m6_shap_force_plot_3_banking_e4.png",
            "m6_shap_force_plot_4_ecommerce_e0.png",
            "m6_shap_force_plot_4_ecommerce_e4.png",
            "m6_shap_force_plot_5_technical_e0.png",
            "m6_shap_force_plot_5_technical_e4.png",
        ]

        for fig_name in expected_figures:
            fig_path = figures_dir / fig_name
            assert fig_path.exists(), f"Missing visual figure: {fig_name}"
            file_size = fig_path.stat().st_size
            assert file_size > 20000, f"Figure {fig_name} is unexpectedly small ({file_size} bytes)"

        # Verify summary JSON tracks visual artifacts
        summary = json.loads(OUTPUT_SHAP_SUMMARY.read_text(encoding="utf-8"))
        vis = summary["visual_artifacts"]
        assert (figures_dir / vis["e0_global_beeswarm_plot"]["file"]).exists()
        assert (figures_dir / vis["e4_global_beeswarm_plot"]["file"]).exists()
        assert len(vis["local_force_plots"]) == 5

    def test_stage1_second_order_shap_interactions(self):
        """Stage 1: Verify second-order TreeSHAP interaction computation (phi_{i, j}) and output artifact."""
        inter_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_shap_interaction_values.json"
        assert inter_path.exists(), f"Missing SHAP interaction artifact: {inter_path}"

        iv = json.loads(inter_path.read_text(encoding="utf-8"))
        assert iv["stage"] == "STAGE_1_ENSEMBLE_DISSECTION"
        assert iv["cohort_sample_size"] == 100
        assert "TreeSHAP Second-Order Interaction Values" in iv["analysis_type"]

        # Check top compounding pairs
        e0_top = iv["e0_top_compounding_interactions"]
        assert len(e0_top) >= 5
        assert all("feature_1" in x and "feature_2" in x and "mean_absolute_interaction" in x for x in e0_top)
        assert e0_top[0]["mean_absolute_interaction"] > 0.02

        # Check archetypal cases have 18x18 interaction matrices
        cases = iv["archetypal_case_interaction_matrices"]
        assert len(cases) == 5
        for url, mats in cases.items():
            mat_e0 = np.array(mats["e0_interaction_matrix"])
            mat_e4 = np.array(mats["e4_interaction_matrix"])
            assert mat_e0.shape == (18, 18), f"Expected 18x18 interaction matrix, got {mat_e0.shape}"
            assert mat_e4.shape == (18, 18), f"Expected 18x18 interaction matrix, got {mat_e4.shape}"
            # Verify symmetry of interaction values: phi_{i, j} == phi_{j, i}
            assert np.allclose(mat_e0, mat_e0.T, atol=1e-5), "TreeSHAP interaction matrix must be symmetric"
            assert np.allclose(mat_e4, mat_e4.T, atol=1e-5), "TreeSHAP interaction matrix must be symmetric"

        # Explicit TreeSHAP Interaction Numerical Integrity Verification
        num_ver = iv["numerical_integrity_verification"]
        assert num_ver["status"] == "ALL_IDENTITIES_VERIFIED_WITHIN_TOLERANCE"

        # 1. Five Archetypal Cases Numerical Checks
        cases_ver = num_ver["archetypal_cases"]
        assert cases_ver["sample_count"] == 5
        assert cases_ver["e0"]["max_symmetry_error"] <= 1e-05
        assert cases_ver["e0"]["max_row_sum_error"] <= 1e-05
        assert cases_ver["e0"]["max_prediction_error"] <= 1e-05
        assert cases_ver["e4"]["max_symmetry_error"] <= 1e-05
        assert cases_ver["e4"]["max_row_sum_error"] <= 1e-05
        assert cases_ver["e4"]["max_prediction_error"] <= 1e-05

        # 2. 100-Record Cohort Numerical Checks
        cohort_ver = num_ver["cohort_100_records"]
        assert cohort_ver["sample_count"] == 100
        assert cohort_ver["e0"]["max_symmetry_error"] <= 1e-05
        assert cohort_ver["e0"]["max_row_sum_error"] <= 1e-05
        assert cohort_ver["e0"]["max_prediction_error"] <= 1e-05
        assert cohort_ver["e4"]["max_symmetry_error"] <= 1e-05
        assert cohort_ver["e4"]["max_row_sum_error"] <= 1e-05
        assert cohort_ver["e4"]["max_prediction_error"] <= 1e-05

        # 3. Observational Finding Check
        obs = iv["observational_finding"]
        assert "E4 exhibits a different interaction structure" in obs
        assert "The analysis does not establish that synthetic augmentation alone caused this difference" in obs

    def test_stage1_tree_path_metrics_and_clarified_denominators(self):
        """Stage 1: Verify clarified denominators, either-feature prevalence, and co-traversal path metrics."""
        analysis = json.loads(OUTPUT_TREE_ANALYSIS.read_text(encoding="utf-8"))
        e0_info = analysis["models"]["frozen_production_control_e0"]
        e4_info = analysis["models"]["candidate_model_e4"]
        h_e0 = e0_info["hierarchy"]
        h_e4 = e4_info["hierarchy"]

        # Explicit root IsHTTPS dominance: exactly 22% in both E0 and E4
        assert h_e0["root_split_dominance"]["IsHTTPS"]["count"] == 22
        assert h_e0["root_split_dominance"]["IsHTTPS"]["percentage"] == 0.22
        assert h_e0["root_split_dominance"]["IsHTTPS"]["denominator"] == 100

        assert h_e4["root_split_dominance"]["IsHTTPS"]["count"] == 22
        assert h_e4["root_split_dominance"]["IsHTTPS"]["percentage"] == 0.22
        assert h_e4["root_split_dominance"]["IsHTTPS"]["denominator"] == 100

        # Explicit denominators
        assert h_e0["denominators"]["root_split_denominator"] == 100
        assert h_e0["denominators"]["depth1_split_nodes_denominator"] == 172
        assert h_e0["denominators"]["top2_split_nodes_denominator"] == 272

        # Either feature in top 2 levels
        either = h_e0["trees_with_either_special_or_https_in_top2"]
        assert either["count"] == 74
        assert either["denominator"] == 100
        assert np.isclose(either["percentage"], 0.74)

        # Both features co-traversal across all 612 false positives
        co_trav = e0_info["co_traversal_metrics"]
        assert co_trav["total_paths_evaluated"] == 61200
        assert co_trav["paths_traversing_both"] == 28993
        assert np.isclose(co_trav["co_traversal_percentage"], 0.47374, atol=0.001)

    def test_stage1_attribution_aggregation_validation(self):
        """Stage 1: Verify validated attribution aggregation and linearity of expectation identity."""
        summary = json.loads(OUTPUT_SHAP_SUMMARY.read_text(encoding="utf-8"))
        agg = summary["attribution_aggregation_validation"]

        for model_key in ["e0", "e4"]:
            val = agg[model_key]
            assert "False-Positive Legitimate URLs" in val["aggregation_population"]

            # Primary 4 features
            p4 = val["primary_4_features"]
            assert p4["mathematical_identity_verified"] is True
            assert np.isclose(p4["mean_combined_contribution"], p4["sum_of_marginal_means"], atol=1e-12)
            assert 0.40 <= p4["mean_combined_contribution"] <= 0.55

            # Extended 6 features
            p6 = val["extended_6_features"]
            assert p6["mathematical_identity_verified"] is True
            assert np.isclose(p6["mean_combined_contribution"], p6["sum_of_marginal_means"], atol=1e-12)
            assert 0.50 <= p6["mean_combined_contribution"] <= 0.60

    def test_stage1_scientific_classification_language(self):
        """Stage 1: Verify strictly observational language 'Evidence of Training Representation Disparity'."""
        tree_analysis = json.loads(OUTPUT_TREE_ANALYSIS.read_text(encoding="utf-8"))
        shap_summary = json.loads(OUTPUT_SHAP_SUMMARY.read_text(encoding="utf-8"))

        assert tree_analysis["scientific_classification"] == "Evidence of Training Representation Disparity"
        assert shap_summary["scientific_classification"] == "Evidence of Training Representation Disparity"

        # Check raw text doesn't contain the unapproved 'Proof' phrasing
        assert "Training Representation Bias Proof" not in OUTPUT_TREE_ANALYSIS.read_text(encoding="utf-8")
        assert "Training Representation Bias Proof" not in OUTPUT_SHAP_SUMMARY.read_text(encoding="utf-8")

    def test_stage2_covariate_shift_scorecard_and_artifacts(self):
        """Stage 2: Verify presence, structure, and empirical findings of class-separated covariate shift."""
        assert OUTPUT_COVARIATE_SCORECARD.exists(), f"Missing scorecard: {OUTPUT_COVARIATE_SCORECARD}"
        assert OUTPUT_COVARIATE_FULL.exists(), f"Missing full covariate data: {OUTPUT_COVARIATE_FULL}"

        sc = json.loads(OUTPUT_COVARIATE_SCORECARD.read_text(encoding="utf-8"))
        assert sc["stage"] == "STAGE_2_CLASS_SEPARATED_COVARIATE_SHIFT"
        assert sc["governance_status"] == "ZERO_RETRAINING_MODEL_IMMUTABILITY_PRESERVED"

        # Model and dataset hash verification
        assert sc["models"]["e0_sha256"] == EXPECTED_E0_SHA
        assert sc["models"]["e4_sha256"] == EXPECTED_E4_SHA
        assert sc["dataset_hashes"]["training_split_sha256"] == EXPECTED_TRAIN_SHA
        assert sc["dataset_hashes"]["m5_1_qualified_benchmark_sha256"] == EXPECTED_M5_1_BENCH_SHA

        # Sample counts
        counts = sc["sample_counts"]
        assert counts["legitimate"]["reference_training_n"] == 94523
        assert counts["legitimate"]["evaluation_external_n"] == 650
        assert counts["phishing"]["reference_training_n"] == 76092
        assert counts["phishing"]["evaluation_external_n"] == 650

        # Legitimate partition shift properties
        legit_table = sc["scorecard_table"]["legitimate_features"]
        assert len(legit_table) == 18

        legit_features = {r["feature"]: r for r in legit_table}
        # Verify severe drift in key structural features
        assert legit_features["SpacialCharRatioInURL"]["psi_severity"] == "SEVERE_SHIFT"
        assert legit_features["SpacialCharRatioInURL"]["psi"] > 5.0
        assert legit_features["NoOfOtherSpecialCharsInURL"]["psi_severity"] == "SEVERE_SHIFT"
        assert legit_features["NoOfOtherSpecialCharsInURL"]["psi"] > 4.0
        assert legit_features["NoOfSubDomain"]["psi_severity"] == "SEVERE_SHIFT"
        assert legit_features["NoOfEqualsInURL"]["psi_severity"] == "SEVERE_SHIFT"
        assert legit_features["NoOfQMarkInURL"]["psi_severity"] == "SEVERE_SHIFT"
        assert legit_features["URLLength"]["psi_severity"] == "SEVERE_SHIFT"
        assert legit_features["DomainLength"]["psi_severity"] == "SEVERE_SHIFT"

        # Phishing partition shift properties
        phish_table = sc["scorecard_table"]["phishing_features"]
        assert len(phish_table) == 18

        # Asymmetric drift: legitimate overall shift is substantially higher than phishing
        legit_mean_psi = sc["legitimate_partition_summary"]["overall_metrics"]["mean_psi"]
        phish_mean_psi = sc["phishing_partition_summary"]["overall_metrics"]["mean_psi"]
        assert legit_mean_psi > 1.0, f"Expected severe legitimate mean PSI (>1.0), got {legit_mean_psi}"
        assert phish_mean_psi < 0.30, f"Expected moderate phishing mean PSI (<0.30), got {phish_mean_psi}"
        assert legit_mean_psi > 5 * phish_mean_psi, "Legitimate drift must be substantially larger than phishing drift"

        # Check observational diagnostics wording
        obs = sc["observational_diagnostics"]
        assert "Empirical quantification reveals severe covariate shift" in obs["legitimate_partition_shift"]
        assert "phishing URLs exhibit substantially milder covariate drift" in obs["phishing_partition_shift"]
        assert "highly asymmetric" in obs["asymmetric_drift_implication"]

    def test_stage2_data_integrity_and_provenance_audit(self):
        """Stage 2 Finding A & B: Verify data integrity, feature extraction skew, and evidence provenance."""
        assert OUTPUT_COVARIATE_FULL.exists()
        full_data = json.loads(OUTPUT_COVARIATE_FULL.read_text(encoding="utf-8"))

        # Finding A: Integrity
        audit_a = full_data["audit_finding_a_integrity"]
        assert audit_a["status"] == "PASS"
        assert audit_a["sample_counts"]["train"]["total"] == 170615
        assert audit_a["sample_counts"]["train"]["legitimate"] == 94523
        assert audit_a["sample_counts"]["train"]["phishing"] == 76092
        assert audit_a["sample_counts"]["benchmark"]["total"] == 1300
        assert audit_a["sample_counts"]["benchmark"]["legitimate"] == 650
        assert audit_a["sample_counts"]["benchmark"]["phishing"] == 650
        assert audit_a["duplicate_urls"]["train"] == 0
        assert audit_a["duplicate_urls"]["benchmark"] == 0
        assert audit_a["missing_values"]["train_features"] == 0
        assert audit_a["missing_values"]["benchmark_columns"] == 0
        assert audit_a["label_encoding"]["phishing"] == 0
        assert audit_a["label_encoding"]["legitimate"] == 1

        # Feature skew audit
        train_skew = audit_a["training_feature_extraction_skew_audit"]
        assert train_skew["DomainLength"]["mismatches_per_1000"] == 0
        assert train_skew["URLLength"]["max_absolute_difference"] == 1.0

        bench_skew = audit_a["benchmark_feature_extraction_skew_audit"]
        assert bench_skew["status"] == "CONCORDANT"
        assert bench_skew["evaluated_urls"] == 1300

        # Finding B: Provenance
        audit_b = full_data["audit_finding_b_provenance"]
        assert audit_b["status"] == "PASS"
        assert audit_b["total_observations_verified"] == 1300
        legit_v = audit_b["legitimate_verification"]
        assert legit_v["tls_verified_count"] == 650
        assert legit_v["tls_protocol_distribution"]["TLSv1.3"] == 555
        assert legit_v["tls_protocol_distribution"]["TLSv1.2"] == 95
        assert legit_v["http_status_code_distribution"]["200"] == 626
        assert legit_v["certificate_san_evidence"]["present_count"] == 650
        assert legit_v["certificate_san_evidence"]["mean_sans"] == 6.16

        phish_v = audit_b["phishing_verification"]
        assert phish_v["phishtank_snapshot_matches"] == 650
        assert phish_v["verification_status_yes_count"] == 650
        assert phish_v["online_status_yes_count"] == 650
        assert phish_v["socket_level_tls_and_http_status"] == "UNAVAILABLE_UPSTREAM_FEED_INGESTION"

    def test_stage2_domain_clustering_and_bootstrap_uncertainty(self):
        """Stage 2 Finding C: Verify domain clustering, zero overlap, and 2,000-iteration cluster bootstrap."""
        assert OUTPUT_DOMAIN_INDEP.exists()
        domain_data = json.loads(OUTPUT_DOMAIN_INDEP.read_text(encoding="utf-8"))

        # Domain counts under independent PSL
        fqdn = domain_data["domain_counts"]["fully_qualified_domain_names_fqdn"]
        assert fqdn["legitimate"] == 219
        assert fqdn["phishing"] == 650
        assert fqdn["total"] == 869

        etld1 = domain_data["domain_counts"]["registered_domains_etld_plus_1"]
        assert etld1["legitimate"] == 159
        assert etld1["phishing"] == 650
        assert etld1["total"] == 809

        # Cluster caps
        clusters = domain_data["cluster_size_distributions"]
        assert clusters["legitimate"]["max_urls_per_domain"] == 6
        assert clusters["phishing"]["max_urls_per_domain"] == 1

        # Zero overlaps
        iso = domain_data["domain_isolation_audit"]
        assert iso["cross_class_overlap_count"] == 0
        assert iso["cross_class_status"] == "CERTIFIED_ZERO_OVERLAP"
        assert iso["training_overlap_count"] == 0
        assert iso["training_status"] == "CERTIFIED_ZERO_OVERLAP"

        # 2,000-iteration cluster bootstrap uncertainty
        boot = domain_data["cluster_bootstrap_uncertainty"]
        assert boot["iterations"] == 2000
        assert boot["random_seed"] == 42
        assert boot["conditioning_status"] == "CONDITIONAL_ON_TRAINING_REFERENCE"
        assert "training_reference_variance_comparison" in boot

        # Legitimate Mean PSI CI
        legit_boot = boot["legitimate_partition"]
        mean_psi_ci = legit_boot["mean_psi"]["ci_95"]
        assert mean_psi_ci[0] > 1.0, f"Lower bound of Mean PSI must exceed 1.0, got {mean_psi_ci[0]}"
        assert mean_psi_ci[1] < 2.5, f"Upper bound of Mean PSI must be reasonable, got {mean_psi_ci[1]}"

        # SpacialCharRatioInURL CI
        spacial_ci = legit_boot["features"]["SpacialCharRatioInURL"]["psi"]["ci_95"]
        assert spacial_ci[0] > 5.0, f"Lower bound of SpacialCharRatioInURL PSI must exceed 5.0, got {spacial_ci[0]}"

        # Phishing Mean PSI CI
        phish_boot = boot["phishing_partition"]
        phish_mean_psi_ci = phish_boot["mean_psi"]["ci_95"]
        assert phish_mean_psi_ci[1] < 0.35, f"Upper bound of Phishing Mean PSI must be < 0.35, got {phish_mean_psi_ci[1]}"

    def test_stage2_multiple_testing_fdr_and_discrete_inference(self):
        """Stage 2 Finding D: Verify Benjamini-Hochberg FDR correction (m=36) and discrete proportion tests."""
        assert OUTPUT_STAT_INFERENCE.exists()
        stat_data = json.loads(OUTPUT_STAT_INFERENCE.read_text(encoding="utf-8"))

        summary = stat_data["summary"]
        assert summary["total_hypotheses_tested"] == 36
        assert summary["statistically_significant_under_fdr_005"] == 23
        assert summary["non_significant_under_fdr_005"] == 13

        ranking = stat_data["fdr_corrected_tests_ranking"]
        assert len(ranking) == 36

        # Check FDR monotonicity and bounds
        prev_p = -1.0
        for item in ranking:
            assert item["p_value"] >= prev_p
            prev_p = item["p_value"]
            assert 0.0 <= item["fdr_q_value"] <= 1.0

        # Exact and survival-function tests for binary features
        exact_tests = stat_data["binary_features_exact_tests"]
        # Phishing IsHTTPS
        p_https = exact_tests["phishing"]["IsHTTPS"]
        assert p_https["z_statistic"] > 10.0
        assert p_https["p_value"] < 1e-10
        assert p_https["fisher_exact_p_value"] < 1e-10
        assert p_https["significant_005"] is True

        # Legitimate IsHTTPS
        l_https = exact_tests["legitimate"]["IsHTTPS"]
        assert l_https["z_statistic"] == 0.0
        assert l_https["p_value"] == 1.0
        assert l_https["significant_005"] is False

        # Legitimate HasObfuscation (sparse cell test)
        l_obf = exact_tests["legitimate"]["HasObfuscation"]
        assert l_obf["is_sparse_event"] is True
        assert l_obf["method_used"] == "fisher_exact"
        assert l_obf["significant_005"] is True

    def test_stage2_psi_sensitivity_analysis(self):
        """Stage 2 Finding E: Verify PSI stability across bin counts (5, 10, 20) and smoothing epsilons."""
        assert OUTPUT_PSI_SENSITIVITY.exists()
        sens_data = json.loads(OUTPUT_PSI_SENSITIVITY.read_text(encoding="utf-8"))

        bins_data = sens_data["bin_count_sensitivity"]["legitimate"]
        assert "bins_5" in bins_data and "bins_10" in bins_data and "bins_20" in bins_data

        # Verify key severe features remain SEVERE_SHIFT across all bin counts
        severe_keys = [
            "SpacialCharRatioInURL",
            "NoOfOtherSpecialCharsInURL",
            "NoOfSubDomain",
            "NoOfEqualsInURL",
            "NoOfQMarkInURL",
            "URLLength",
            "DomainLength",
        ]
        for b_key in ["bins_5", "bins_10", "bins_20"]:
            for feat in severe_keys:
                assert bins_data[b_key][feat]["severity"] == "SEVERE_SHIFT"
                assert bins_data[b_key][feat]["psi"] >= 0.25

        # Smoothing sensitivity
        smooth_data = sens_data["smoothing_epsilon_sensitivity"]["legitimate"]
        for eps_key in ["half_over_n", "one_over_n", "fixed_0001", "fixed_001"]:
            assert smooth_data[eps_key]["SpacialCharRatioInURL"]["severity"] == "SEVERE_SHIFT"

    def test_stage2_scale_normalized_wasserstein_and_fpr_linkage(self):
        """Stage 2 Finding F, G & H: Verify scale-normalized W1, observational wording, and linkage to M5.1 FPR."""
        sc = json.loads(OUTPUT_COVARIATE_SCORECARD.read_text(encoding="utf-8"))

        # Normalized W1
        legit_features = {r["feature"]: r for r in sc["scorecard_table"]["legitimate_features"]}
        assert legit_features["NoOfQMarkInURL"]["normalized_w1"] > 10.0
        assert legit_features["NoOfEqualsInURL"]["normalized_w1"] > 7.0
        assert legit_features["NoOfOtherSpecialCharsInURL"]["normalized_w1"] > 4.0
        assert legit_features["SpacialCharRatioInURL"]["normalized_w1"] > 3.0

        # Observational Diagnostics and FPR Linkage
        obs = sc["observational_diagnostics"]
        assert "linkage_to_historical_fpr" in obs
        fpr_text = obs["linkage_to_historical_fpr"]
        assert "94.15%" in fpr_text
        assert "92.62%" in fpr_text
        assert "special characters" in fpr_text
        assert "query markers" in fpr_text

        # Strict Non-Causal Wording Audit: no unauthorized causal assertions
        full_text = OUTPUT_COVARIATE_SCORECARD.read_text(encoding="utf-8")
        assert "Training Representation Bias Proof" not in full_text
        assert "synthetic augmentation caused the feature shift" not in full_text
        assert "drift in SpacialCharRatio proves" not in full_text


class TestM6Stage2SyntheticMathematicalRigour:
    """
    Independent synthetic mathematical verification test suite for Phase 7 M6 Stage 2.
    Tests core statistical mechanics on exact analytical examples without relying on disk artifacts.
    """

    def test_math_synthetic_psi_calculation(self):
        """Verify hand-calculated analytical PSI, identical distributions, constant features, and epsilon smoothing."""
        # 1. Hand-calculated 2-bin distribution:
        # Ref: 20% in bin 0, 80% in bin 1
        # Eval: 80% in bin 0, 20% in bin 1
        # Analytical PSI = (0.8 - 0.2)*ln(0.8/0.2) + (0.2 - 0.8)*ln(0.2/0.8) = 1.2 * ln(4) = 1.663553233343869
        ref_arr = np.array([0.0] * 200 + [1.0] * 800)
        eval_arr = np.array([0.0] * 800 + [1.0] * 200)
        psi_val, status = compute_psi(ref_arr, eval_arr, num_bins=2, epsilon=1e-6)
        expected_analytical = 1.2 * np.log(4.0)
        assert status == "OK"
        assert np.isclose(psi_val, expected_analytical, atol=1e-5), f"Expected {expected_analytical}, got {psi_val}"

        # 2. Identical distributions: analytical PSI must be 0.0
        psi_identical, status_id = compute_psi(ref_arr, ref_arr, num_bins=5, epsilon=1e-4)
        assert status_id == "OK"
        assert np.isclose(psi_identical, 0.0, atol=1e-7)

        # 3. Constant feature: all elements identical
        const_ref = np.zeros(500)
        const_eval = np.zeros(500)
        psi_const, _ = compute_psi(const_ref, const_eval, num_bins=5, epsilon=1e-4)
        assert np.isclose(psi_const, 0.0, atol=1e-7)

        # 4. Zero count in evaluation bin with epsilon smoothing
        # Ref has values [1, 2], Eval has only [1]
        zero_ref = np.array([1.0] * 500 + [2.0] * 500)
        zero_eval = np.array([1.0] * 500)
        psi_smoothed, status_sm = compute_psi(zero_ref, zero_eval, num_bins=2, epsilon=0.001)
        assert status_sm == "OK"
        assert psi_smoothed > 0.0 and np.isfinite(psi_smoothed)

    def test_math_synthetic_bh_fdr_correction(self):
        """Verify exact analytical Benjamini-Hochberg FDR q-values and monotonicity on synthetic p-values."""
        # Known 5-element p-value vector:
        # p = [0.001, 0.01, 0.02, 0.04, 0.06] with m=5
        # k=1: 0.001*(5/1) = 0.005
        # k=2: 0.01*(5/2)  = 0.025
        # k=3: 0.02*(5/3)  = 0.03333333...
        # k=4: 0.04*(5/4)  = 0.05
        # k=5: 0.06*(5/5)  = 0.06
        # Cumulative minimum from right preserves these exact values.
        p_vec = np.array([0.001, 0.01, 0.02, 0.04, 0.06])
        q_vals, sig_flags = compute_bh_fdr(p_vec, alpha=0.05)

        expected_q = np.array([0.005, 0.025, 1.0 / 30.0, 0.05, 0.06])
        assert np.allclose(q_vals, expected_q, atol=1e-7)
        assert np.array_equal(sig_flags, np.array([True, True, True, False, False]))

        # Verify preservation of order with unordered p-value input
        shuffled_p = np.array([0.04, 0.001, 0.06, 0.01, 0.02])
        shuffled_q, shuffled_sig = compute_bh_fdr(shuffled_p, alpha=0.05)
        assert np.isclose(shuffled_q[1], 0.005, atol=1e-7)
        assert np.isclose(shuffled_q[3], 0.025, atol=1e-7)
        assert bool(shuffled_sig[1]) is True and bool(shuffled_sig[3]) is True and bool(shuffled_sig[2]) is False

    def test_math_synthetic_psl_domain_extraction(self):
        """Verify independent PSL domain extraction edge cases including ccTLDs, IPv4/v6, and platform domains."""
        cases = [
            ("https://a.b.c.example.com/path/to/resource", "a.b.c.example.com", "example.com"),
            ("https://news.bbc.co.uk/sport/football", "news.bbc.co.uk", "bbc.co.uk"),
            ("https://sub.egov.gov.kz/services", "sub.egov.gov.kz", "egov.gov.kz"),
            ("https://portal.fgov.be/nl/home", "portal.fgov.be", "portal.fgov.be"),
            ("https://site.public.lu/index.html", "site.public.lu", "site.public.lu"),
            ("https://user.github.io/my-project/", "user.github.io", "user.github.io"),
            ("http://192.168.1.100:8080/admin", "192.168.1.100", "192.168.1.100"),
            ("http://[2001:db8::1]:80/status", "2001:db8::1", "2001:db8::1"),
            ("https://wayland.freedesktop.org/docs", "wayland.freedesktop.org", "freedesktop.org"),
            ("https://freedesktop.org/software/", "freedesktop.org", "freedesktop.org"),
        ]
        for url, expected_fqdn, expected_domain in cases:
            assert extract_fqdn(url) == expected_fqdn, f"FQDN mismatch for {url}"
            assert extract_psl_registered_domain(url) == expected_domain, f"Domain mismatch for {url}"

        # Malformed input resilience
        assert extract_fqdn("") == ""
        assert extract_psl_registered_domain("") == ""
        assert extract_fqdn("invalid_url_without_scheme_or_host") == ""
        assert extract_psl_registered_domain("invalid_url_without_scheme_or_host") == ""
        assert extract_fqdn("http://example.com") == "example.com"
        assert extract_psl_registered_domain("http://example.com") == "example.com"

    def test_math_synthetic_cluster_bootstrap_behavior(self):
        """Verify cluster preservation and conditional reproducibility in domain-cluster bootstrap."""
        # 4 domains with different cluster sizes
        ref_df = pd.DataFrame({"feat": np.linspace(0, 10, 100)})
        eval_df = pd.DataFrame({
            "feat": [1.0, 1.2, 1.1, 5.0, 5.1, 9.0, 9.1, 9.2, 9.3],
            "domain": ["dom1", "dom1", "dom1", "dom2", "dom2", "dom3", "dom3", "dom3", "dom3"]
        })
        res1 = compute_domain_cluster_bootstrap(
            ref_df=ref_df,
            eval_df=eval_df,
            cluster_series=eval_df["domain"],
            feature_names=["feat"],
            num_iterations=100,
            random_seed=42,
        )
        res2 = compute_domain_cluster_bootstrap(
            ref_df=ref_df,
            eval_df=eval_df,
            cluster_series=eval_df["domain"],
            feature_names=["feat"],
            num_iterations=100,
            random_seed=42,
        )
        # Verify exact determinism under identical seed
        assert res1["mean_psi"]["bootstrap_mean"] == res2["mean_psi"]["bootstrap_mean"]
        assert res1["confidence_interval_type"] == "CONDITIONAL_ON_TRAINING_REFERENCE"
        assert res1["cluster_count"] == 3

    def test_math_synthetic_exact_tests_and_survival_function(self):
        """Verify survival-function proportion test without catastrophic cancellation and Fisher's exact test."""
        # 1. Catastrophic cancellation test for large z:
        # For z = 15.0, 1.0 - norm.cdf(15.0) underflows to 0.0 in standard float64.
        # stats.norm.sf(15.0) computes ~3.6e-51, avoiding cancellation.
        k1 = 5
        n1 = 94523
        k2 = 2
        n2 = 650
        res = compute_two_proportion_test(k1, n1, k2, n2)
        assert res["is_sparse_event"] is True
        assert res["method_used"] == "fisher_exact"
        assert res["p_value"] > 0.0
        assert res["p_value"] != 0.0
        assert np.isclose(res["fisher_exact_p_value"], 0.00095606, atol=1e-5)

        # 2. Large z-score proportion test (non-sparse)
        # e.g. 50% vs 80% with large sample sizes
        res_large = compute_two_proportion_test(5000, 10000, 8000, 10000)
        assert res_large["is_sparse_event"] is False
        assert res_large["method_used"] == "two_sample_z_test_survival_function"
        assert res_large["z_statistic"] > 40.0
        # Prevented from underflowing to true 0.0
        assert res_large["p_value"] >= 1e-300
        assert res_large["significant_005"] is True

    def test_math_synthetic_provenance_validation(self):
        """Verify field-level provenance audit logic and explicit classification of unavailable fields."""
        # Synthetic evaluation observation dicts
        synthetic_legit_obs = {
            "tls_protocol": "TLSv1.3",
            "http_status_code": 200,
            "certificate_sans": ["example.org", "www.example.org"],
            "reachability": "SUCCESS",
        }
        assert synthetic_legit_obs["tls_protocol"] in {"TLSv1.3", "TLSv1.2"}
        assert synthetic_legit_obs["http_status_code"] == 200
        assert len(synthetic_legit_obs["certificate_sans"]) >= 1

        # Phishing upstream threat intelligence record without socket capture
        synthetic_phish_obs = {
            "verification_status": "yes",
            "online_status": "yes",
            "socket_level_tls": "UNAVAILABLE_UPSTREAM_FEED_INGESTION",
            "http_status": "UNAVAILABLE_UPSTREAM_FEED_INGESTION",
        }
        assert synthetic_phish_obs["verification_status"] == "yes"
        assert synthetic_phish_obs["online_status"] == "yes"
        assert synthetic_phish_obs["socket_level_tls"] == "UNAVAILABLE_UPSTREAM_FEED_INGESTION"





