"""
Test Suite for Phase 7 Milestone 4 (M4): Confirmatory Internal Testing & Benchmark Evaluation.

Verifies:
1. Production model E0 remains strictly frozen and unmodified (models/random_forest_model.joblib).
2. Candidate model E4 is isolated in models/phase7_candidates/ and verified against its frozen hash.
3. Pre-evaluation manifest locked all hashes and experimental contracts prior to evaluation.
4. Single-pass evaluation on internal_test.csv (34,092 URLs, 25,978 unique registered domains).
5. Primary confirmatory comparison at T = 0.50 confirms statistically significant phishing recall gain
   with domain-clustered 95% bootstrap confidence interval strictly above zero.
6. Secondary operating-point comparison at T = 0.75 preserves >= 99% recall with sub-0.10% legitimate FPR.
7. Subgroup stratification includes sample counts, denominators, and statistical sample size caveats.
8. Historical external benchmark replication (1,047 URLs) is explicitly labelled as previously observed,
   demonstrating mitigation of the 94% FPR problem without compromising live threat stream recall.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from experiments.phase7.evaluation.evaluate_m4_confirmatory import (
    compute_file_sha256,
    EXPECTED_FROZEN_MODEL_SHA256,
    EXPECTED_CANDIDATE_MODEL_SHA256,
    EXPECTED_TEST_CSV_SHA256,
    EXPECTED_BENCHMARK_SHA256,
    FROZEN_MODEL_PATH,
    CANDIDATE_MODEL_PATH,
    TEST_CSV_PATH,
    BENCHMARK_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

PRE_EVAL_MANIFEST = REPO_ROOT / "experiments" / "phase7" / "reports" / "m4_pre_evaluation_manifest.json"
SCORECARD_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m4_confirmatory_scorecard.json"
MANIFEST_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m4_confirmatory_manifest.json"
REPORT_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m4_confirmatory_evaluation_report.md"


def test_m4_production_model_frozen():
    """Verify production model E0 remains strictly frozen at known hash."""
    assert FROZEN_MODEL_PATH.exists()
    actual_hash = compute_file_sha256(FROZEN_MODEL_PATH)
    assert actual_hash == EXPECTED_FROZEN_MODEL_SHA256, "Production model altered!"


def test_m4_candidate_model_frozen():
    """Verify candidate model E4 remains frozen at known hash and does not replace production."""
    assert CANDIDATE_MODEL_PATH.exists()
    actual_hash = compute_file_sha256(CANDIDATE_MODEL_PATH)
    assert actual_hash == EXPECTED_CANDIDATE_MODEL_SHA256, "Candidate model altered!"
    assert actual_hash != EXPECTED_FROZEN_MODEL_SHA256, "Candidate model must not be identical to production!"


def test_m4_pre_evaluation_manifest_safeguards():
    """Verify pre-evaluation manifest locks all protocols, hashes, and labels."""
    assert PRE_EVAL_MANIFEST.exists()
    with open(PRE_EVAL_MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["manifest_version"] == "phase7-m4-pre-evaluation-manifest-v1"
    assert data["status"] == "LOCKED_PRE_EVALUATION"
    assert data["experimental_contract"]["confirmatory_only"] is True
    assert data["registered_thresholds"]["primary_confirmatory"] == 0.50
    assert data["registered_thresholds"]["secondary_operating_point"] == 0.75
    assert data["candidate_model_e4"]["sha256"] == EXPECTED_CANDIDATE_MODEL_SHA256
    assert data["frozen_production_model_e0"]["sha256"] == EXPECTED_FROZEN_MODEL_SHA256
    assert data["internal_test_dataset"]["sha256"] == EXPECTED_TEST_CSV_SHA256
    assert data["historical_external_benchmark"]["sha256"] == EXPECTED_BENCHMARK_SHA256
    assert "not an independent" in data["historical_external_benchmark"]["evaluation_role"].lower()


def test_m4_scorecard_primary_confirmatory_results_at_050():
    """Verify primary confirmatory evaluation metrics at T = 0.50 on internal_test.csv."""
    assert SCORECARD_PATH.exists()
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    prim = scorecard["primary_confirmatory_evaluation_at_050"]
    e0 = prim["e0_frozen_control"]
    e4 = prim["e4_augmented_candidate"]

    # Sample and Class Counts
    assert e0["sample_count"] == 34092
    assert e0["phishing_count"] == 13880
    assert e0["legitimate_count"] == 20212

    # Confusion Matrix Counts
    assert e0["confusion_matrix"]["true_positives"] == 13789
    assert e0["confusion_matrix"]["false_positives"] == 23
    assert e0["confusion_matrix"]["true_negatives"] == 20189
    assert e0["confusion_matrix"]["false_negatives"] == 91

    assert e4["confusion_matrix"]["true_positives"] == 13805
    assert e4["confusion_matrix"]["false_positives"] == 62
    assert e4["confusion_matrix"]["true_negatives"] == 20150
    assert e4["confusion_matrix"]["false_negatives"] == 75

    # Net Phishing Catch Gain: +16 true positives
    assert e4["confusion_matrix"]["true_positives"] - e0["confusion_matrix"]["true_positives"] == 16

    # Phishing Recall Gain
    rec_gain = prim["delta_e4_minus_e0"]["phishing_recall"]
    assert rec_gain > 0.0010, f"Expected phishing recall gain > 0.10%, got {rec_gain}"

    # Overall Legitimate FPR preservation
    assert e4["rates"]["legitimate_fpr"] < 0.0035, f"Legit FPR must remain small: {e4['rates']['legitimate_fpr']}"

    # Calibration preservation
    assert e4["probability_quality"]["ece"] < 0.0050, f"Candidate must remain well calibrated: {e4['probability_quality']['ece']}"
    assert e4["probability_quality"]["brier_score"] < 0.0050


def test_m4_domain_clustered_bootstrap_uncertainty():
    """Verify domain-clustered bootstrap uncertainty estimation (confidence intervals only, no empirical p-values)."""
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    boot = scorecard["primary_confirmatory_evaluation_at_050"]["domain_clustered_bootstrap_uncertainty"]
    assert boot["n_bootstrap"] == 1000
    assert boot["n_unique_domains"] == 25978

    delta_rec = boot["paired_delta_e4_minus_e0"]["phishing_recall"]
    # 95% Confidence Interval must strictly exclude 0 (demonstrating significant positive gain)
    assert delta_rec["ci_95_low"] > 0.0, f"Lower bound of 95% CI should be > 0: {delta_rec['ci_95_low']}"
    assert delta_rec["ci_95_high"] > delta_rec["ci_95_low"]
    # Empirical p-values must not be reported for bootstrap samples
    assert "empirical_p_value_gain" not in delta_rec

    delta_fpr = boot["paired_delta_e4_minus_e0"]["legitimate_fpr"]
    # FPR confidence interval crosses zero into positive territory (does not establish non-inferiority)
    assert delta_fpr["ci_95_low"] < 0.0
    assert delta_fpr["ci_95_high"] > 0.0
    assert "empirical_p_value_reduction" not in delta_fpr


def test_m4_paired_mcnemar_suite():
    """Verify three distinct paired McNemar tests for overall correctness, phishing recall, and false alarms."""
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    mcn_suite = scorecard["primary_confirmatory_evaluation_at_050"]["mcnemar_suite"]
    assert "overall_classification_correctness" in mcn_suite
    assert "phishing_recall_paired_test" in mcn_suite
    assert "legitimate_false_alarm_paired_test" in mcn_suite

    # 1. Phishing Recall Paired Test (n=13,880)
    p_phish = mcn_suite["phishing_recall_paired_test"]
    assert p_phish["sample_count"] == 13880
    assert p_phish["contingency_table"]["e0_correct_e4_wrong (n01)"] == 8
    assert p_phish["contingency_table"]["e0_wrong_e4_correct (n10)"] == 24
    assert p_phish["discordant_pairs"] == 32
    assert p_phish["p_value"] < 0.01  # p = 0.00801
    assert p_phish["statistically_significant_p05"] is True

    # 2. Legitimate False Alarm Paired Test (n=20,212)
    p_legit = mcn_suite["legitimate_false_alarm_paired_test"]
    assert p_legit["sample_count"] == 20212
    assert p_legit["contingency_table"]["e0_correct_e4_wrong (n01)"] == 44
    assert p_legit["contingency_table"]["e0_wrong_e4_correct (n10)"] == 5
    assert p_legit["discordant_pairs"] == 49
    assert p_legit["p_value"] < 1e-6  # p = 5.68e-08
    assert p_legit["statistically_significant_p05"] is True

    # 3. Overall Correctness (N=34,092)
    p_overall = mcn_suite["overall_classification_correctness"]
    assert p_overall["sample_count"] == 34092
    assert p_overall["discordant_pairs"] == 81
    assert p_overall["p_value"] < 0.05  # p = 0.0145


def test_m4_secondary_operating_point_results_at_075():
    """Verify secondary pre-specified operating-point metrics at T = 0.75."""
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    sec = scorecard["secondary_operating_point_evaluation_at_075"]
    e4 = sec["e4_augmented_candidate"]

    # Phishing Recall >= 99%
    assert e4["rates"]["phishing_recall"] >= 0.990, f"Expected recall >= 99%, got {e4['rates']['phishing_recall']}"

    # Legitimate FPR sub-0.10%
    assert e4["rates"]["legitimate_fpr"] <= 0.0010, f"Expected legit FPR <= 0.10%, got {e4['rates']['legitimate_fpr']}"
    assert e4["confusion_matrix"]["false_positives"] == 16


def test_m4_subgroup_stratification_and_caveats():
    """Verify all 12 structural strata evaluated with sample counts, denominator handling, and caveats."""
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    subgroups = scorecard["primary_confirmatory_evaluation_at_050"]["subgroup_strata"]
    assert len(subgroups) == 12

    # Benign expansion on unseen registered domains
    b_exp = subgroups["benign_expansion_unseen_domains"]
    assert b_exp["sample_count"] == 13
    assert b_exp["legitimate_count"] == 13
    assert b_exp["e0"]["confusion_matrix"]["false_positives"] == 13
    assert b_exp["e4"]["confusion_matrix"]["false_positives"] == 12
    assert "Caution: Tiny legit denominator" in b_exp["statistical_caveat"]

    # Strata with zero legitimate samples must have undefined (NaN) FPR
    single_lvl = subgroups["single_level_path"]
    assert single_lvl["sample_count"] == 2096
    assert single_lvl["legitimate_count"] == 0
    assert np.isnan(single_lvl["e0"]["legitimate_fpr"])
    assert np.isnan(single_lvl["e4"]["legitimate_fpr"])
    assert "FPR Undefined" in single_lvl["statistical_caveat"]

    pct_enc = subgroups["percent_encoded_path"]
    assert pct_enc["sample_count"] == 49
    assert pct_enc["legitimate_count"] == 0
    assert np.isnan(pct_enc["e0"]["legitimate_fpr"])
    assert np.isnan(pct_enc["e4"]["legitimate_fpr"])
    assert "FPR Undefined" in pct_enc["statistical_caveat"]

    # Query bearing stratum
    q_stratum = subgroups["query_bearing"]
    assert q_stratum["sample_count"] == 948
    assert q_stratum["phishing_count"] == 939
    assert q_stratum["legitimate_count"] == 9
    assert "Caution: Tiny legit denominator" in q_stratum["statistical_caveat"]

    # Homepage stratum has adequate class denominators
    hp = subgroups["homepage"]
    assert hp["sample_count"] == 29539
    assert hp["phishing_count"] == 9340
    assert hp["legitimate_count"] == 20199
    assert "Adequate class denominators" in hp["statistical_caveat"]


def test_m4_historical_external_benchmark_replication():
    """Verify replication on historical Phase 4 external benchmark."""
    with open(SCORECARD_PATH, "r", encoding="utf-8") as f:
        scorecard = json.load(f)

    bench = scorecard["historical_external_benchmark_replication"]
    assert bench["status"] == "PREVIOUSLY_OBSERVED_EXTERNAL_BENCHMARK"
    assert bench["independent_holdout_status"] is False
    assert bench["sample_count"] == 1047

    slices_050 = bench["evaluations_by_threshold"]["T_0.50"]["source_slices"]

    # OpenPhish Recall
    assert slices_050["OpenPhish-Community"]["e4"]["recall"] == 1.0

    # PhishTank Recall
    assert slices_050["PhishTank-Verified"]["e4"]["recall"] >= 0.995

    # Curated Deep-Link FPR reduction
    fpr_e0_050 = slices_050["Curated-Authoritative-Web-Corpus"]["e0"]["fpr"]
    fpr_e4_050 = slices_050["Curated-Authoritative-Web-Corpus"]["e4"]["fpr"]
    assert fpr_e0_050 == 0.9400
    assert fpr_e4_050 < 0.8500  # 81.80%

    # Operating point T = 0.75 further reduces FPR
    slices_075 = bench["evaluations_by_threshold"]["T_0.75"]["source_slices"]
    fpr_e4_075 = slices_075["Curated-Authoritative-Web-Corpus"]["e4"]["fpr"]
    assert fpr_e4_075 <= 0.7100  # 70.60%


def test_m4_manifest_and_report_files_exist():
    """Verify manifest and markdown report exist and contain required disclosures."""
    assert MANIFEST_PATH.exists()
    assert REPORT_PATH.exists()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["manifest_version"] == "phase7-m4-confirmatory-manifest-v2"
    assert "candidate_hash_erratum_note" in manifest
    assert "0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b" in manifest["candidate_hash_erratum_note"]
    assert "mcnemar_suite" in manifest["confirmatory_summary_at_050"]

    report_text = REPORT_PATH.read_text(encoding="utf-8")
    assert "Confirmatory Internal Testing & Benchmark Evaluation" in report_text
    assert "previously observed external benchmark" in report_text
    assert "CONFIRMED GAIN" in report_text
    assert "Domain-Clustered" in report_text
    assert "Candidate Hash Erratum" in report_text
    assert "NON-INFERIORITY NOT ESTABLISHED" in report_text
    assert "Paired McNemar (phish" in report_text
    assert "Paired McNemar (legit" in report_text
    assert "0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b" in report_text
