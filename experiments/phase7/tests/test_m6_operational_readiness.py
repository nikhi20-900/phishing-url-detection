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
from experiments.phase7.evaluation.m6_diagnostics_utils import (
    classify_structural_stratum,
    compute_bounding_box_analysis,
    compute_clopper_pearson,
    compute_psi,
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


