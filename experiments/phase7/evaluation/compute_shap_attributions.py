"""
Stage 1: TreeSHAP Attribution Computation, Visual Force Plots & Second-Order Interaction Analysis.
Phase 7 Milestone 6: Operational Readiness & Generalisation Assessment.

Computes TreeSHAP feature attributions strictly in probability space for the
Phishing class (class index 0, P(phishing) in [0, 1]) for all 1,300 benchmark URLs
across both frozen models (E0 and E4).

Verifies strict probability-space additivity:
    phi_0 + sum(phi_i) == predict_proba[:, 0]  (tolerance <= 1e-5)
Fails fast with explicit AssertionError if any record violates additivity.

Delivers:
1. Global summary beeswarm plots (PNG) for E0 and E4.
2. Local force plots (PNG) for 5 representative institutional URLs across diverse sectors for E0 and E4.
3. True second-order TreeSHAP interaction values (18x18 matrices) for archetypal cases and stratified FP cohort.
4. Validated per-record attribution aggregation statistics (linearity of expectation verified).
5. Observational reporting language ("Evidence of Training Representation Disparity").
6. Cryptographic pre-flight and post-flight model immutability verification.
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure MPLCONFIGDIR is set to writable temporary directory before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
import shap

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

FIGURES_DIR = REPO_ROOT / "experiments" / "phase7" / "reports" / "figures"
OUTPUT_SHAP_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_shap_attributions.json"
OUTPUT_SUMMARY_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_shap_summary.json"
OUTPUT_INTERACTION_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_shap_interaction_values.json"


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
) -> Tuple[shap.TreeExplainer, np.ndarray, float, float, float]:
    """
    Compute TreeSHAP values strictly in probability space for Phishing class (index 0).
    Verifies probability-space additivity for all records against clf.predict_proba.
    Returns: (explainer, shap_values_class0, base_value, max_error, mean_error)
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

    return explainer, shap_0, base_0, max_err, mean_err


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


def validate_attribution_aggregation(
    shap_matrix: np.ndarray,
    fp_mask: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Validate that combined feature contributions (+50% to +55%) are computed using
    compatible per-record linear summations, documenting the aggregation population
    and verifying linearity of expectation.
    """
    fp_shap = shap_matrix[fp_mask]
    n_fp = len(fp_shap)

    primary_4_names = ["NoOfOtherSpecialCharsInURL", "DomainLength", "SpacialCharRatioInURL", "NoOfSubDomain"]
    primary_4_indices = [feature_names.index(f) for f in primary_4_names]

    extended_6_names = primary_4_names + ["LetterRatioInURL", "URLLength"]
    extended_6_indices = [feature_names.index(f) for f in extended_6_names]

    # Per-record summations
    per_record_primary4 = np.sum(fp_shap[:, primary_4_indices], axis=1)
    per_record_extended6 = np.sum(fp_shap[:, extended_6_indices], axis=1)

    # Sum of marginal means
    sum_means_primary4 = sum(np.mean(fp_shap[:, idx]) for idx in primary_4_indices)
    sum_means_extended6 = sum(np.mean(fp_shap[:, idx]) for idx in extended_6_indices)

    assert np.isclose(np.mean(per_record_primary4), sum_means_primary4), "Linearity of expectation violated!"
    assert np.isclose(np.mean(per_record_extended6), sum_means_extended6), "Linearity of expectation violated!"

    return {
        "aggregation_population": f"False-Positive Legitimate URLs at T=0.75 (N = {n_fp})",
        "methodology": (
            "Per-record additive combination: Delta_S(x_k) = sum_{j in S} phi_{k, j}. "
            "Because expectation is linear, the population mean combined contribution "
            "identically equals the sum of the population mean marginal SHAP values: "
            "E[Delta_S] = sum_{j in S} E[phi_j]."
        ),
        "primary_4_features": {
            "features": primary_4_names,
            "mean_combined_contribution": float(np.mean(per_record_primary4)),
            "median_combined_contribution": float(np.median(per_record_primary4)),
            "std_combined_contribution": float(np.std(per_record_primary4)),
            "min_combined_contribution": float(np.min(per_record_primary4)),
            "max_combined_contribution": float(np.max(per_record_primary4)),
            "sum_of_marginal_means": float(sum_means_primary4),
            "mathematical_identity_verified": True,
        },
        "extended_6_features": {
            "features": extended_6_names,
            "mean_combined_contribution": float(np.mean(per_record_extended6)),
            "median_combined_contribution": float(np.median(per_record_extended6)),
            "std_combined_contribution": float(np.std(per_record_extended6)),
            "min_combined_contribution": float(np.min(per_record_extended6)),
            "max_combined_contribution": float(np.max(per_record_extended6)),
            "sum_of_marginal_means": float(sum_means_extended6),
            "mathematical_identity_verified": True,
        },
    }


def generate_beeswarm_plot(
    shap_vals: np.ndarray,
    X: np.ndarray,
    model_name: str,
    output_png: Path,
):
    """Generate and save global summary beeswarm plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, feature_names=FEATURE_NAMES, show=False)
    plt.title(f"TreeSHAP Global Summary Beeswarm ({model_name} — Class 0: Phishing)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


def generate_force_plot_image(
    base_val: float,
    shap_sample: np.ndarray,
    x_sample: np.ndarray,
    model_name: str,
    url: str,
    output_png: Path,
):
    """Generate and save local force plot as a standalone high-resolution image."""
    plt.figure(figsize=(14, 3))
    shap.force_plot(
        base_val,
        shap_sample,
        x_sample,
        feature_names=FEATURE_NAMES,
        matplotlib=True,
        show=False,
    )
    # Shorten URL for title display if necessary
    disp_url = (url[:70] + "...") if len(url) > 70 else url
    plt.title(f"TreeSHAP Force Plot ({model_name}) — {disp_url}", fontsize=11, pad=15)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


def run_shap_computation() -> Dict[str, Any]:
    start_time = time.time()
    print("=" * 80)
    print("PHASE 7 MILESTONE 6 — STAGE 1: TREESHAP ATTRIBUTIONS & INTERACTION ANALYSIS")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

    # 6. Compute TreeSHAP for E0 and E4
    print("Computing TreeSHAP for E0 on all 1,300 samples (Class 0: Phishing)...")
    explainer_e0, shap_e0, base_e0, max_err_e0, mean_err_e0 = compute_model_shap(e0_clf, X, tolerance=1e-05)
    print(f"[PASS] E0 TreeSHAP Additivity Verified: Max Error = {max_err_e0:.2e} <= 1e-05")

    print("Computing TreeSHAP for E4 on all 1,300 samples (Class 0: Phishing)...")
    explainer_e4, shap_e4, base_e4, max_err_e4, mean_err_e4 = compute_model_shap(e4_clf, X, tolerance=1e-05)
    print(f"[PASS] E4 TreeSHAP Additivity Verified: Max Error = {max_err_e4:.2e} <= 1e-05")

    # 7. Generate Visual Deliverable 1: Global Beeswarm Plots
    print("Generating Global SHAP Beeswarm plots...")
    beeswarm_e0_path = FIGURES_DIR / "m6_shap_summary_beeswarm_e0.png"
    beeswarm_e4_path = FIGURES_DIR / "m6_shap_summary_beeswarm_e4.png"
    generate_beeswarm_plot(shap_e0, X, "E0 (Production Control)", beeswarm_e0_path)
    generate_beeswarm_plot(shap_e4, X, "E4 (Candidate)", beeswarm_e4_path)
    print(f"[PASS] Beeswarm plots saved to {beeswarm_e0_path.name} and {beeswarm_e4_path.name}")

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

    # 10. Validated Attribution Aggregation Statistics
    legit_mask = (y == 1)
    e0_fp_mask = legit_mask & (probs_e0 >= 0.75)
    e4_fp_mask = legit_mask & (probs_e4 >= 0.75)

    agg_validation_e0 = validate_attribution_aggregation(shap_e0, e0_fp_mask, FEATURE_NAMES)
    agg_validation_e4 = validate_attribution_aggregation(shap_e4, e4_fp_mask, FEATURE_NAMES)
    print(f"[PASS] Attribution aggregation validated: Linear sum of means identically matches per-record mean.")

    # 11. Archetypal Institutional Error Case Studies & Force Plots
    print("Generating Local Force Plots for 5 representative institutional URLs...")
    archetypal_targets = [
        ("government_public_stats", "https://belgium.be/", "1_government"),
        ("academic_research", "https://www.edpsciences.org/en", "2_academic"),
        ("banking_finance", "https://apps.sfc.hk/edistributionWeb/gateway/EN/news-and-announcements/news/doc?refNo=26PR194", "3_banking"),
        ("ecommerce_retail", "https://www.farfetch.com/", "4_ecommerce"),
        ("technical_documentation", "https://www.openbsd.org/?lang=en", "5_technical"),
    ]

    case_studies = []
    visual_force_artifacts = []

    for sector_target, url_target, slug in archetypal_targets:
        matching_indices = [i for i, u in enumerate(urls) if u == url_target]
        if not matching_indices:
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

        # Generate PNG force plots for both models
        force_e0_file = FIGURES_DIR / f"m6_shap_force_plot_{slug}_e0.png"
        force_e4_file = FIGURES_DIR / f"m6_shap_force_plot_{slug}_e4.png"

        generate_force_plot_image(base_e0, shap_e0[idx], X[idx], "E0 Control", actual_url, force_e0_file)
        generate_force_plot_image(base_e4, shap_e4[idx], X[idx], "E4 Candidate", actual_url, force_e4_file)

        visual_force_artifacts.append({
            "sector": actual_sector,
            "url": actual_url,
            "true_label": int(y[idx]),
            "e0_phish_probability": float(probs_e0[idx]),
            "e4_phish_probability": float(probs_e4[idx]),
            "e0_force_plot_image": force_e0_file.name,
            "e4_force_plot_image": force_e4_file.name,
            "e0_force_plot_sha256": compute_sha256(force_e0_file),
            "e4_force_plot_sha256": compute_sha256(force_e4_file),
        })

        case_studies.append({
            "url": actual_url,
            "sector": actual_sector,
            "true_label": int(y[idx]),
            "feature_values": x_vals,
            "e0_evaluation": {
                "phishing_probability": float(probs_e0[idx]),
                "base_value_phi0": base_e0,
                "reconstructed_probability": float(base_e0 + sum(shap_vals_e0.values())),
                "force_plot_image": force_e0_file.name,
                "shap_attributions": shap_vals_e0,
                "top_positive_contributors": top_pos_e0,
                "top_negative_contributors": top_neg_e0,
            },
            "e4_evaluation": {
                "phishing_probability": float(probs_e4[idx]),
                "base_value_phi0": base_e4,
                "reconstructed_probability": float(base_e4 + sum(shap_vals_e4.values())),
                "force_plot_image": force_e4_file.name,
                "shap_attributions": shap_vals_e4,
                "top_positive_contributors": top_pos_e4,
                "top_negative_contributors": top_neg_e4,
            },
        })

    # 12. Deliverable 3: Second-Order TreeSHAP Interaction Analysis
    print("Computing second-order TreeSHAP interaction values (phi_{i, j})...")
    # A. Compute exact 18x18 interaction matrices for all 5 archetypal case studies
    archetypal_indices = [
        matching_indices[0]
        for sector_target, url_target, slug in archetypal_targets
        for matching_indices in [[i for i, u in enumerate(urls) if u == url_target] or [i for i, (u, s) in enumerate(zip(urls, sectors)) if s == sector_target and y[i] == 1]]
    ]
    inter_e0_cases = explainer_e0.shap_interaction_values(X[archetypal_indices])
    inter_e4_cases = explainer_e4.shap_interaction_values(X[archetypal_indices])

    # Extract class 0 (Phishing) interactions
    inter_e0_cases_0 = inter_e0_cases[:, :, :, 0] if len(np.shape(inter_e0_cases)) == 4 else inter_e0_cases
    inter_e4_cases_0 = inter_e4_cases[:, :, :, 0] if len(np.shape(inter_e4_cases)) == 4 else inter_e4_cases

    # B. Compute second-order interaction values for a representative stratified cohort of 100 false positives
    # 20 samples from each of the 5 institutional sectors
    fp_sector_indices = []
    for sector_name in ["government_public_stats", "academic_research", "banking_finance", "ecommerce_retail", "technical_documentation"]:
        sec_fp = [i for i in np.where(e0_fp_mask)[0] if sectors[i] == sector_name]
        fp_sector_indices.extend(sec_fp[:20])

    print(f"Computing interactions for representative cohort of {len(fp_sector_indices)} stratified false-positive URLs...")
    inter_e0_cohort = explainer_e0.shap_interaction_values(X[fp_sector_indices])
    inter_e4_cohort = explainer_e4.shap_interaction_values(X[fp_sector_indices])

    inter_e0_cohort_0 = inter_e0_cohort[:, :, :, 0] if len(np.shape(inter_e0_cohort)) == 4 else inter_e0_cohort
    inter_e4_cohort_0 = inter_e4_cohort[:, :, :, 0] if len(np.shape(inter_e4_cohort)) == 4 else inter_e4_cohort

    # Mean absolute interaction for all 153 feature pairs (off-diagonal i < j)
    def compute_top_interaction_pairs(inter_matrix: np.ndarray) -> List[Dict[str, Any]]:
        mean_abs = np.mean(np.abs(inter_matrix), axis=0)
        mean_signed = np.mean(inter_matrix, axis=0)
        pair_list = []
        for i in range(len(FEATURE_NAMES)):
            for j in range(i + 1, len(FEATURE_NAMES)):
                pair_list.append({
                    "feature_1": FEATURE_NAMES[i],
                    "feature_2": FEATURE_NAMES[j],
                    "mean_absolute_interaction": float(mean_abs[i, j]),
                    "mean_signed_interaction": float(mean_signed[i, j]),
                })
        pair_list.sort(key=lambda x: x["mean_absolute_interaction"], reverse=True)
        return pair_list

    top_interactions_e0 = compute_top_interaction_pairs(inter_e0_cohort_0)
    top_interactions_e4 = compute_top_interaction_pairs(inter_e4_cohort_0)

    # C. Explicit Numerical Integrity Verification of TreeSHAP Interactions
    # 1. Verification on the 5 Archetypal Cases
    sym_cases_e0 = [float(np.max(np.abs(inter_e0_cases_0[k] - inter_e0_cases_0[k].T))) for k in range(len(case_studies))]
    row_cases_e0 = [float(np.max(np.abs(np.sum(inter_e0_cases_0[k], axis=1) - shap_e0[archetypal_indices[k]]))) for k in range(len(case_studies))]
    pred_cases_e0 = [float(abs((base_e0 + np.sum(inter_e0_cases_0[k])) - probs_e0[archetypal_indices[k]])) for k in range(len(case_studies))]

    sym_cases_e4 = [float(np.max(np.abs(inter_e4_cases_0[k] - inter_e4_cases_0[k].T))) for k in range(len(case_studies))]
    row_cases_e4 = [float(np.max(np.abs(np.sum(inter_e4_cases_0[k], axis=1) - shap_e4[archetypal_indices[k]]))) for k in range(len(case_studies))]
    pred_cases_e4 = [float(abs((base_e4 + np.sum(inter_e4_cases_0[k])) - probs_e4[archetypal_indices[k]])) for k in range(len(case_studies))]

    cases_verification = {
        "sample_count": len(case_studies),
        "tolerance": 1e-05,
        "e0": {
            "max_symmetry_error": max(sym_cases_e0),
            "max_row_sum_error": max(row_cases_e0),
            "max_prediction_error": max(pred_cases_e0),
            "status": "PASS",
        },
        "e4": {
            "max_symmetry_error": max(sym_cases_e4),
            "max_row_sum_error": max(row_cases_e4),
            "max_prediction_error": max(pred_cases_e4),
            "status": "PASS",
        },
        "individual_case_results": [
            {
                "url": case_studies[k]["url"],
                "sector": case_studies[k]["sector"],
                "e0_symmetry_error": sym_cases_e0[k],
                "e0_row_sum_error": row_cases_e0[k],
                "e0_prediction_error": pred_cases_e0[k],
                "e4_symmetry_error": sym_cases_e4[k],
                "e4_row_sum_error": row_cases_e4[k],
                "e4_prediction_error": pred_cases_e4[k],
            }
            for k in range(len(case_studies))
        ],
    }

    # 2. Verification on the 100-Record Stratified Cohort
    probs_cohort_e0 = probs_e0[fp_sector_indices]
    probs_cohort_e4 = probs_e4[fp_sector_indices]

    sym_cohort_e0 = float(np.max(np.abs(inter_e0_cohort_0 - np.swapaxes(inter_e0_cohort_0, 1, 2))))
    row_cohort_e0 = float(np.max(np.abs(np.sum(inter_e0_cohort_0, axis=2) - shap_e0[fp_sector_indices])))
    pred_cohort_e0 = float(np.max(np.abs(base_e0 + np.sum(inter_e0_cohort_0, axis=(1, 2)) - probs_cohort_e0)))

    sym_cohort_e4 = float(np.max(np.abs(inter_e4_cohort_0 - np.swapaxes(inter_e4_cohort_0, 1, 2))))
    row_cohort_e4 = float(np.max(np.abs(np.sum(inter_e4_cohort_0, axis=2) - shap_e4[fp_sector_indices])))
    pred_cohort_e4 = float(np.max(np.abs(base_e4 + np.sum(inter_e4_cohort_0, axis=(1, 2)) - probs_cohort_e4)))

    cohort_verification = {
        "sample_count": len(fp_sector_indices),
        "tolerance": 1e-05,
        "e0": {
            "max_symmetry_error": sym_cohort_e0,
            "max_row_sum_error": row_cohort_e0,
            "max_prediction_error": pred_cohort_e0,
            "status": "PASS",
        },
        "e4": {
            "max_symmetry_error": sym_cohort_e4,
            "max_row_sum_error": row_cohort_e4,
            "max_prediction_error": pred_cohort_e4,
            "status": "PASS",
        },
    }

    # Fail fast assertion checks
    assert cases_verification["e0"]["max_symmetry_error"] <= 1e-05, "E0 case symmetry violation"
    assert cases_verification["e0"]["max_row_sum_error"] <= 1e-05, "E0 case row-sum violation"
    assert cases_verification["e0"]["max_prediction_error"] <= 1e-05, "E0 case prediction reconstruction violation"
    assert cases_verification["e4"]["max_symmetry_error"] <= 1e-05, "E4 case symmetry violation"
    assert cases_verification["e4"]["max_row_sum_error"] <= 1e-05, "E4 case row-sum violation"
    assert cases_verification["e4"]["max_prediction_error"] <= 1e-05, "E4 case prediction reconstruction violation"
    assert cohort_verification["e0"]["max_symmetry_error"] <= 1e-05, "E0 cohort symmetry violation"
    assert cohort_verification["e0"]["max_row_sum_error"] <= 1e-05, "E0 cohort row-sum violation"
    assert cohort_verification["e0"]["max_prediction_error"] <= 1e-05, "E0 cohort prediction reconstruction violation"
    assert cohort_verification["e4"]["max_symmetry_error"] <= 1e-05, "E4 cohort symmetry violation"
    assert cohort_verification["e4"]["max_row_sum_error"] <= 1e-05, "E4 cohort row-sum violation"
    assert cohort_verification["e4"]["max_prediction_error"] <= 1e-05, "E4 cohort prediction reconstruction violation"

    print(f"[PASS] TreeSHAP interaction numerical integrity verified (Max Cohort Error: E0={pred_cohort_e0:.2e}, E4={pred_cohort_e4:.2e} <= 1e-05)")

    # Save dedicated interaction values artifact
    interaction_payload = {
        "stage": "STAGE_1_ENSEMBLE_DISSECTION",
        "milestone": "Phase 7 — M6",
        "analysis_type": "TreeSHAP Second-Order Interaction Values (phi_{i, j})",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "Exact second-order TreeSHAP interaction computation decomposing prediction space into "
            "main effects (diagonal phi_{i, i}) and pairwise non-linear compounding effects (off-diagonal phi_{i, j}). "
            "Evaluated strictly on Phishing class probability space (class index 0)."
        ),
        "cohort_sample_size": len(fp_sector_indices),
        "numerical_integrity_verification": {
            "archetypal_cases": cases_verification,
            "cohort_100_records": cohort_verification,
            "status": "ALL_IDENTITIES_VERIFIED_WITHIN_TOLERANCE",
        },
        "observational_finding": (
            "E4 exhibits a different interaction structure, with DomainLength appearing prominently among the largest "
            "interaction pairs. This is an observed difference between the frozen ensembles and is consistent with a change "
            "in learned feature interactions. The analysis does not establish that synthetic augmentation alone caused this difference."
        ),
        "e0_top_compounding_interactions": top_interactions_e0[:15],
        "e4_top_compounding_interactions": top_interactions_e4[:15],
        "archetypal_case_interaction_matrices": {
            case_studies[k]["url"]: {
                "e0_interaction_matrix": inter_e0_cases_0[k].tolist(),
                "e4_interaction_matrix": inter_e4_cases_0[k].tolist(),
            }
            for k in range(len(case_studies))
        },
    }
    OUTPUT_INTERACTION_PATH.write_text(json.dumps(interaction_payload, indent=2), encoding="utf-8")
    interaction_sha = compute_sha256(OUTPUT_INTERACTION_PATH)
    print(f"[PASS] Second-order interaction values saved to {OUTPUT_INTERACTION_PATH.name} (SHA: {interaction_sha[:16]}...)")

    # 13. Post-flight verification
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
        "visual_artifacts": {
            "e0_global_beeswarm_plot": {
                "file": beeswarm_e0_path.name,
                "sha256": compute_sha256(beeswarm_e0_path),
            },
            "e4_global_beeswarm_plot": {
                "file": beeswarm_e4_path.name,
                "sha256": compute_sha256(beeswarm_e4_path),
            },
            "local_force_plots": visual_force_artifacts,
        },
        "attribution_aggregation_validation": {
            "e0": agg_validation_e0,
            "e4": agg_validation_e4,
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
                "true_label": cs["true_label"],
                "e0_phish_prob": cs["e0_evaluation"]["phishing_probability"],
                "e4_phish_prob": cs["e4_evaluation"]["phishing_probability"],
                "top_driver_e0": cs["e0_evaluation"]["top_positive_contributors"][0],
                "force_plot_e0": cs["e0_evaluation"]["force_plot_image"],
                "force_plot_e4": cs["e4_evaluation"]["force_plot_image"],
            }
            for cs in case_studies
        ],
        "top_second_order_interactions": {
            "e0_top5": top_interactions_e0[:5],
            "e4_top5": top_interactions_e4[:5],
        },
        "interaction_numerical_integrity": {
            "archetypal_cases": cases_verification,
            "cohort_100_records": cohort_verification,
            "status": "ALL_IDENTITIES_VERIFIED_WITHIN_TOLERANCE",
        },
        "interaction_observational_finding": (
            "E4 exhibits a different interaction structure, with DomainLength appearing prominently among the largest "
            "interaction pairs. This is an observed difference between the frozen ensembles and is consistent with a change "
            "in learned feature interactions. The analysis does not establish that synthetic augmentation alone caused this difference."
        ),
        "scientific_classification": "Evidence of Training Representation Disparity",
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
    print("STAGE 1 TREESHAP ATTRIBUTION & INTERACTION ANALYSIS COMPLETE.")
    print("=" * 80)

    return {
        "status": "SUCCESS",
        "summary_sha256": summary_sha,
        "shap_sha256": shap_sha,
        "interaction_sha256": interaction_sha,
        "e0_max_additivity_error": max_err_e0,
        "e4_max_additivity_error": max_err_e4,
        "execution_time_seconds": execution_time,
    }


if __name__ == "__main__":
    run_shap_computation()
