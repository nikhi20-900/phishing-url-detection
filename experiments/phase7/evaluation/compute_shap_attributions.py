"""
Stage 1: TreeSHAP Attribution Computation and Local Explanation Analysis.
Phase 7 Milestone 6: Operational Readiness & Generalisation Assessment.

Computes TreeSHAP feature attributions strictly in probability space for the
Phishing class (class index 0, P(phishing) in [0, 1]) for all 1,300 benchmark URLs
across both frozen models (E0 and E4).

Verifies strict probability-space additivity:
    phi_0 + sum(phi_i) == predict_proba[:, 0]  (tolerance <= 1e-5)
Fails fast with explicit AssertionError if any record violates additivity.

Delivers:
1. Global mean absolute SHAP value rankings (|phi_i|).
2. Class-separated distributions (True Positives, False Positives, True Negatives, False Negatives).
3. 5 archetypal institutional error case studies across diverse sectors.
4. Second-order feature attribution covariance matrices.
5. Cryptographic pre-flight and post-flight model immutability verification.
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

# Ensure MPLCONFIGDIR is set to writable temporary directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.features import FEATURE_NAMES, extract_url_features

MANIFEST_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_operational_readiness_manifest.json"
BENCHMARK_PATH = REPO_ROOT / "experiments" / "phase7" / "external" / "datasets" / "m5_1_qualified_benchmark.csv"

E0_MODEL_PATH = REPO_ROOT / "models" / "random_forest_model.joblib"
E4_MODEL_PATH = REPO_ROOT / "models" / "phase7_candidates" / "rf_augmented_e4.joblib"

EXPECTED_E0_SHA = "95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb"
EXPECTED_E4_SHA = "0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b"
EXPECTED_BENCHMARK_SHA = "f6cdbf797b683d6c82d7b5c72c6ddf35aba7cbb448edafc2d68e67e26695fc88"

OUTPUT_SHAP_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_shap_attributions.json"
OUTPUT_SUMMARY_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_shap_summary.json"


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def verify_models_integrity(stage_label: str) -> Dict[str, str]:
    """Verify cryptographic hashes of E0 and E4 models. Fail fast on mismatch."""
    e0_sha = compute_sha256(E0_MODEL_PATH)
    e4_sha = compute_sha256(E4_MODEL_PATH)

    assert e0_sha == EXPECTED_E0_SHA, (
        f"[{stage_label}] CRITICAL INVARIANT VIOLATION: E0 hash mismatch! "
        f"Expected {EXPECTED_E0_SHA}, got {e0_sha}"
    )
    assert e4_sha == EXPECTED_E4_SHA, (
        f"[{stage_label}] CRITICAL INVARIANT VIOLATION: E4 hash mismatch! "
        f"Expected {EXPECTED_E4_SHA}, got {e4_sha}"
    )
    return {"e0_sha256": e0_sha, "e4_sha256": e4_sha}


def compute_model_shap(
    clf,
    X: np.ndarray,
    tolerance: float = 1e-05,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Compute TreeSHAP values strictly in probability space for Phishing class (index 0).
    Verifies probability-space additivity for all records against clf.predict_proba.
    Returns: (shap_values_class0, base_value, max_error, mean_error)
    """
    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    raw_shap = explainer.shap_values(X)

    if isinstance(raw_shap, list):
        shap_0 = raw_shap[0]
        base_0 = float(explainer.expected_value[0])
    elif len(np.shape(raw_shap)) == 3:
        shap_0 = raw_shap[:, :, 0]
        base_0 = float(explainer.expected_value[0])
    else:
        shap_0 = raw_shap
        base_0 = float(explainer.expected_value)

    # Additivity Verification against true predict_proba
    prob_true = clf.predict_proba(X)[:, 0]  # Phishing class probability
    reconstructed = base_0 + np.sum(shap_0, axis=1)
    diff = np.abs(reconstructed - prob_true)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))

    assert max_err <= tolerance, (
        f"TreeSHAP probability additivity violated: max error {max_err:.2e} > tolerance {tolerance:.2e}"
    )

    return shap_0, base_0, max_err, mean_err


def compile_stratified_attributions(
    shap_matrix: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Compile class-stratified attribution profiles (True Positives, False Positives,
    True Negatives, False Negatives) at the operational threshold.
    """
    # In labels: 0 = Phishing, 1 = Legitimate
    # Model predicts Phishing (0) if probs >= threshold
    tp_mask = (labels == 0) & (probs >= threshold)
    fp_mask = (labels == 1) & (probs >= threshold)
    tn_mask = (labels == 1) & (probs < threshold)
    fn_mask = (labels == 0) & (probs < threshold)

    subgroups = {
        "true_positives": tp_mask,
        "false_positives": fp_mask,
        "true_negatives": tn_mask,
        "false_negatives": fn_mask,
    }

    results = {}
    for name, mask in subgroups.items():
        count = int(np.sum(mask))
        if count == 0:
            results[name] = {"count": 0, "mean_shap": {}, "mean_abs_shap": {}}
            continue

        sub_shap = shap_matrix[mask]
        mean_s = np.mean(sub_shap, axis=0)
        mean_abs_s = np.mean(np.abs(sub_shap), axis=0)

        results[name] = {
            "count": count,
            "mean_shap": {f: float(mean_s[i]) for i, f in enumerate(FEATURE_NAMES)},
            "mean_abs_shap": {f: float(mean_abs_s[i]) for i, f in enumerate(FEATURE_NAMES)},
            "top_positive_contributors": sorted(
                [(f, float(mean_s[i])) for i, f in enumerate(FEATURE_NAMES)],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            "top_negative_contributors": sorted(
                [(f, float(mean_s[i])) for i, f in enumerate(FEATURE_NAMES)],
                key=lambda x: x[1],
            )[:5],
        }

    return results


def run_shap_computation() -> Dict[str, Any]:
    start_time = time.time()
    print("=" * 80)
    print("PHASE 7 MILESTONE 6 — STAGE 1: TREESHAP ATTRIBUTION COMPUTATION")
    print("=" * 80)

    # 1. Pre-flight verification
    pre_hashes = verify_models_integrity("PRE-FLIGHT")
    print(f"[PASS] Pre-flight model hashes verified:")
    print(f"       E0: {pre_hashes['e0_sha256']}")
    print(f"       E4: {pre_hashes['e4_sha256']}")

    # 2. Benchmark load
    bench_sha = compute_sha256(BENCHMARK_PATH)
    assert bench_sha == EXPECTED_BENCHMARK_SHA
    bench_df = pd.read_csv(BENCHMARK_PATH)
    print(f"[PASS] Benchmark loaded: {len(bench_df)} URLs, SHA {bench_sha[:16]}...")

    # 3. Feature extraction
    print("Extracting 18 lexical features for all 1,300 benchmark URLs...")
    feature_rows = [extract_url_features(u) for u in bench_df["url"]]
    X_df = pd.DataFrame(feature_rows)[FEATURE_NAMES]
    X = X_df.values
    y = bench_df["label"].values
    urls = bench_df["url"].tolist()
    sectors = bench_df["sector"].tolist()

    # 4. Load models read-only
    e0_dict = joblib.load(E0_MODEL_PATH)
    e0_clf = e0_dict["model"] if isinstance(e0_dict, dict) else e0_dict

    e4_dict = joblib.load(E4_MODEL_PATH)
    e4_clf = e4_dict["model"] if isinstance(e4_dict, dict) else e4_dict

    # 5. Model predictions
    probs_e0 = e0_clf.predict_proba(X)[:, 0]
    probs_e4 = e4_clf.predict_proba(X)[:, 0]

    # 6. Compute TreeSHAP for E0
    print("Computing TreeSHAP for E0 on all 1,300 samples (Class 0: Phishing)...")
    shap_e0, base_e0, max_err_e0, mean_err_e0 = compute_model_shap(e0_clf, X, tolerance=1e-05)
    print(f"[PASS] E0 TreeSHAP Additivity Verified: Max Error = {max_err_e0:.2e} <= 1e-05")

    # 7. Compute TreeSHAP for E4
    print("Computing TreeSHAP for E4 on all 1,300 samples (Class 0: Phishing)...")
    shap_e4, base_e4, max_err_e4, mean_err_e4 = compute_model_shap(e4_clf, X, tolerance=1e-05)
    print(f"[PASS] E4 TreeSHAP Additivity Verified: Max Error = {max_err_e4:.2e} <= 1e-05")

    # 8. Global rankings
    global_abs_e0 = np.mean(np.abs(shap_e0), axis=0)
    global_abs_e4 = np.mean(np.abs(shap_e4), axis=0)

    global_ranking_e0 = sorted(
        [(f, float(global_abs_e0[i])) for i, f in enumerate(FEATURE_NAMES)],
        key=lambda x: x[1],
        reverse=True,
    )
    global_ranking_e4 = sorted(
        [(f, float(global_abs_e4[i])) for i, f in enumerate(FEATURE_NAMES)],
        key=lambda x: x[1],
        reverse=True,
    )

    # 9. Stratified profiles
    strat_e0_075 = compile_stratified_attributions(shap_e0, probs_e0, y, threshold=0.75)
    strat_e4_075 = compile_stratified_attributions(shap_e4, probs_e4, y, threshold=0.75)

    strat_e0_050 = compile_stratified_attributions(shap_e0, probs_e0, y, threshold=0.50)
    strat_e4_050 = compile_stratified_attributions(shap_e4, probs_e4, y, threshold=0.50)

    # 10. Archetypal Institutional Error Case Studies (5 diverse sectors)
    archetypal_targets = [
        ("government_public_stats", "https://belgium.be/"),
        ("academic_research", "https://www.edpsciences.org/en"),
        ("banking_finance", "https://apps.sfc.hk/edistributionWeb/gateway/EN/news-and-announcements/news/doc?refNo=26PR194"),
        ("ecommerce_retail", "https://www.farfetch.com/"),
        ("technical_documentation", "https://www.openbsd.org/?lang=en"),
    ]

    case_studies = []
    for sector_target, url_target in archetypal_targets:
        # Locate sample index
        matching_indices = [i for i, u in enumerate(urls) if u == url_target]
        if not matching_indices:
            # Fallback to first URL in that sector
            matching_indices = [i for i, (u, s) in enumerate(zip(urls, sectors)) if s == sector_target and y[i] == 1]

        idx = matching_indices[0]
        actual_url = urls[idx]
        actual_sector = sectors[idx]
        x_vals = {f: float(X[idx, i]) for i, f in enumerate(FEATURE_NAMES)}

        shap_vals_e0 = {f: float(shap_e0[idx, i]) for i, f in enumerate(FEATURE_NAMES)}
        shap_vals_e4 = {f: float(shap_e4[idx, i]) for i, f in enumerate(FEATURE_NAMES)}

        top_pos_e0 = sorted(shap_vals_e0.items(), key=lambda x: x[1], reverse=True)[:3]
        top_neg_e0 = sorted(shap_vals_e0.items(), key=lambda x: x[1])[:3]

        top_pos_e4 = sorted(shap_vals_e4.items(), key=lambda x: x[1], reverse=True)[:3]
        top_neg_e4 = sorted(shap_vals_e4.items(), key=lambda x: x[1])[:3]

        case_studies.append({
            "url": actual_url,
            "sector": actual_sector,
            "true_label": int(y[idx]),
            "feature_values": x_vals,
            "e0_evaluation": {
                "phishing_probability": float(probs_e0[idx]),
                "base_value_phi0": base_e0,
                "reconstructed_probability": float(base_e0 + sum(shap_vals_e0.values())),
                "shap_attributions": shap_vals_e0,
                "top_positive_contributors": top_pos_e0,
                "top_negative_contributors": top_neg_e0,
            },
            "e4_evaluation": {
                "phishing_probability": float(probs_e4[idx]),
                "base_value_phi0": base_e4,
                "reconstructed_probability": float(base_e4 + sum(shap_vals_e4.values())),
                "shap_attributions": shap_vals_e4,
                "top_positive_contributors": top_pos_e4,
                "top_negative_contributors": top_neg_e4,
            },
        })

    # 11. Pairwise feature covariance in SHAP space
    cov_e0 = np.cov(shap_e0, rowvar=False)
    corr_e0 = np.corrcoef(shap_e0, rowvar=False)

    top_correlations = []
    for i in range(len(FEATURE_NAMES)):
        for j in range(i + 1, len(FEATURE_NAMES)):
            if not np.isnan(corr_e0[i, j]):
                top_correlations.append((
                    FEATURE_NAMES[i],
                    FEATURE_NAMES[j],
                    float(corr_e0[i, j]),
                    float(cov_e0[i, j]),
                ))
    top_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    # 12. Post-flight verification
    post_hashes = verify_models_integrity("POST-FLIGHT")
    print(f"[PASS] Post-flight model hashes strictly verified unchanged:")
    print(f"       E0: {post_hashes['e0_sha256']}")
    print(f"       E4: {post_hashes['e4_sha256']}")

    execution_time = time.time() - start_time
    created_at = datetime.now(timezone.utc).isoformat()

    # Build Summary Payload
    summary_payload = {
        "stage": "STAGE_1_ENSEMBLE_DISSECTION",
        "milestone": "Phase 7 — M6",
        "analysis_type": "TreeSHAP Global and Local Probability Attribution Analysis",
        "created_at_utc": created_at,
        "execution_duration_seconds": execution_time,
        "governance_status": "ZERO_RETRAINING_MODEL_IMMUTABILITY_PRESERVED",
        "models": post_hashes,
        "benchmark_sha256": bench_sha,
        "explainer_parameters": {
            "explainer_type": "TreeExplainer",
            "feature_perturbation": "tree_path_dependent",
            "model_output": "raw_probability",
            "class_index": 0,
            "target_class": "Phishing",
            "numerical_tolerance": 1e-05,
        },
        "additivity_verification": {
            "e0_max_error": max_err_e0,
            "e0_mean_error": mean_err_e0,
            "e4_max_error": max_err_e4,
            "e4_mean_error": mean_err_e4,
            "verified_sample_count": len(bench_df),
            "status": "ALL_1300_SAMPLES_SATISFY_ADDITIVITY",
        },
        "base_values": {
            "e0_expected_value_phi0": base_e0,
            "e4_expected_value_phi0": base_e4,
        },
        "global_importance_rankings": {
            "e0_top5": global_ranking_e0[:5],
            "e4_top5": global_ranking_e4[:5],
        },
        "false_positive_drivers_t_075": {
            "e0_top_pushing_features": strat_e0_075["false_positives"]["top_positive_contributors"],
            "e4_top_pushing_features": strat_e4_075["false_positives"]["top_positive_contributors"],
        },
        "archetypal_case_studies_summary": [
            {
                "sector": cs["sector"],
                "url": cs["url"],
                "e0_phish_prob": cs["e0_evaluation"]["phishing_probability"],
                "e4_phish_prob": cs["e4_evaluation"]["phishing_probability"],
                "top_driver_e0": cs["e0_evaluation"]["top_positive_contributors"][0],
            }
            for cs in case_studies
        ],
        "top_shap_correlations": [
            {"feat1": c[0], "feat2": c[1], "pearson_r": c[2]}
            for c in top_correlations[:5]
        ],
    }

    # Build Full Data Payload
    full_payload = {
        "summary": summary_payload,
        "stratified_profiles_t_075": {
            "e0": strat_e0_075,
            "e4": strat_e4_075,
        },
        "stratified_profiles_t_050": {
            "e0": strat_e0_050,
            "e4": strat_e4_050,
        },
        "case_studies": case_studies,
        "global_rankings_full": {
            "e0": global_ranking_e0,
            "e4": global_ranking_e4,
        },
    }

    OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    OUTPUT_SHAP_PATH.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")

    summary_sha = compute_sha256(OUTPUT_SUMMARY_PATH)
    shap_sha = compute_sha256(OUTPUT_SHAP_PATH)

    print(f"[PASS] SHAP summary written to: {OUTPUT_SUMMARY_PATH.name} (SHA: {summary_sha[:16]}...)")
    print(f"[PASS] Full SHAP attributions written to: {OUTPUT_SHAP_PATH.name} (SHA: {shap_sha[:16]}...)")
    print("=" * 80)
    print("STAGE 1 TREESHAP ATTRIBUTION COMPUTATION COMPLETE.")
    print("=" * 80)

    return {
        "status": "SUCCESS",
        "summary_sha256": summary_sha,
        "shap_sha256": shap_sha,
        "e0_max_additivity_error": max_err_e0,
        "e4_max_additivity_error": max_err_e4,
        "execution_time_seconds": execution_time,
    }


if __name__ == "__main__":
    run_shap_computation()
