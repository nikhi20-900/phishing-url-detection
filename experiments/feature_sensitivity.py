"""
Feature Sensitivity & Threshold Analysis (Phase 4 Experimentation).

Measures model response curves and identifies critical decision thresholds
in the Random Forest ensemble when individual URL characteristics change:
1. Synthetic 1D Feature Sweeps: Varying individual features while holding other
   features at authoritative baseline values to discover exact forest split thresholds.
2. Pairwise Perturbation Delta Analysis: Measuring feature deltas and probability
   transitions from canonical roots (https://www.domain.tld) to perturbed variants.

Outputs:
- experiments/reports/feature_sensitivity_results.json
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_url_features, FEATURE_NAMES
from experiments.deep_link_diagnostics import load_frozen_model


def run_1d_feature_sweeps(
    model,
    phish_idx: int,
    baseline_url: str = "https://www.stanford.edu"
) -> Dict[str, Any]:
    """
    Perform 1D sensitivity sweeps across key lexical features holding other features constant.
    Identifies the exact threshold where model probability crosses 0.5.
    """
    base_feat = extract_url_features(baseline_url)
    sweep_results = {}

    # Define feature sweep ranges
    sweeps = {
        "NoOfOtherSpecialCharsInURL": [1, 2, 3, 4, 5, 6, 7, 8],
        "SpacialCharRatioInURL": [round(x, 3) for x in np.linspace(0.02, 0.25, 24)],
        "URLLength": list(range(15, 85, 3)),
        "DomainLength": list(range(8, 40, 2)),
        "NoOfSubDomain": [0, 1, 2, 3, 4],
        "NoOfLettersInURL": list(range(4, 50, 2)),
        "LetterRatioInURL": [round(x, 3) for x in np.linspace(0.1, 0.9, 17)],
        "NoOfDegitsInURL": [0, 1, 2, 4, 6, 8, 10, 15],
        "DegitRatioInURL": [round(x, 3) for x in np.linspace(0.0, 0.4, 9)],
        "NoOfEqualsInURL": [0, 1, 2, 3, 4],
        "NoOfQMarkInURL": [0, 1, 2],
        "NoOfAmpersandInURL": [0, 1, 2, 3],
        "IsHTTPS": [0, 1]
    }

    for feat_name, sweep_vals in sweeps.items():
        curve = []
        critical_threshold = None

        for val in sweep_vals:
            test_feat = dict(base_feat)
            test_feat[feat_name] = val
            test_df = pd.DataFrame([test_feat], columns=FEATURE_NAMES)
            p_phish = float(model.predict_proba(test_df)[0][phish_idx])
            curve.append({"feature_value": val, "prob_phishing": round(p_phish, 4)})

            if critical_threshold is None and p_phish >= 0.5:
                critical_threshold = val

        sweep_results[feat_name] = {
            "baseline_value": base_feat[feat_name],
            "baseline_prob_phishing": float(model.predict_proba(pd.DataFrame([base_feat], columns=FEATURE_NAMES))[0][phish_idx]),
            "critical_threshold_cross_0_5": critical_threshold,
            "response_curve": curve
        }

    return sweep_results


def analyze_pairwise_deltas(
    perturbation_csv_path: str = "experiments/reports/url_perturbation_results.csv"
) -> Dict[str, Any]:
    """
    Compute feature and probability deltas between canonical www_root and key perturbed variants.
    """
    if not os.path.exists(perturbation_csv_path):
        raise FileNotFoundError(f"Perturbation results not found at '{perturbation_csv_path}'. Run deep_link_diagnostics.py first.")

    df = pd.read_csv(perturbation_csv_path)

    # Filter to canonical base
    base_df = df[df["variant_key"] == "www_root"].set_index("domain")

    comparisons = [
        ("www_root_slash", "Trailing Slash Addition on Root"),
        ("bare_root", "Stripping www. Prefix"),
        ("path_single_medium", "Single Path Segment (/research)"),
        ("path_single_medium_slash", "Single Path with Trailing Slash (/research/)"),
        ("path_two_levels", "Two-Level Directory Path (/research/projects)"),
        ("query_single_param", "Single Query Parameter (?q=cybersecurity)"),
        ("http_www_root", "Unencrypted HTTP Protocol (http://www.domain)")
    ]

    pairwise_results = {}

    for var_key, desc in comparisons:
        var_df = df[df["variant_key"] == var_key].set_index("domain")
        common_domains = sorted(list(set(base_df.index) & set(var_df.index)))

        deltas = []
        for dom in common_domains:
            b_row = base_df.loc[dom]
            v_row = var_df.loc[dom]

            d_p = float(v_row["prob_phishing"] - b_row["prob_phishing"])
            feat_deltas = {
                f"delta_{f}": float(v_row[f] - b_row[f]) for f in FEATURE_NAMES
            }
            deltas.append({
                "domain": dom,
                "base_url": b_row["url"],
                "perturbed_url": v_row["url"],
                "base_prob_phishing": float(b_row["prob_phishing"]),
                "perturbed_prob_phishing": float(v_row["prob_phishing"]),
                "delta_prob_phishing": round(d_p, 4),
                "base_prediction": b_row["predicted_label"],
                "perturbed_prediction": v_row["predicted_label"],
                **feat_deltas
            })

        df_d = pd.DataFrame(deltas)
        pairwise_results[var_key] = {
            "description": desc,
            "mean_delta_prob_phishing": round(float(df_d["delta_prob_phishing"].mean()), 4),
            "flip_to_phishing_rate": round(float((df_d["perturbed_prediction"] == "Phishing").mean()), 4),
            "primary_feature_shifts": {
                f: round(float(df_d[f"delta_{f}"].mean()), 4)
                for f in FEATURE_NAMES if abs(df_d[f"delta_{f}"].mean()) > 1e-4
            },
            "sample_cases": deltas[:3]
        }

    return pairwise_results


def run_sensitivity_analysis(
    model_path: str = "models/random_forest_model.joblib",
    perturbation_csv: str = "experiments/reports/url_perturbation_results.csv",
    output_dir: str = "experiments/reports"
) -> Dict[str, Any]:
    """Execute complete sensitivity and response curve analysis."""
    model, phish_idx, legit_idx = load_frozen_model(model_path)

    print("Running 1D feature sensitivity sweeps on baseline URL (https://www.stanford.edu)...")
    sweeps = run_1d_feature_sweeps(model, phish_idx, baseline_url="https://www.stanford.edu")

    print("Analyzing pairwise transition deltas from perturbation experiments...")
    pairwise = analyze_pairwise_deltas(perturbation_csv)

    results = {
        "baseline_test_url": "https://www.stanford.edu",
        "feature_sweeps_1d": sweeps,
        "pairwise_perturbation_deltas": pairwise
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "feature_sensitivity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSensitivity analysis complete. Results saved to '{out_path}'.")

    # Print summary highlights
    print("\n--- CRITICAL 1D DECISION THRESHOLDS (Baseline: https://www.stanford.edu) ---")
    for feat, info in sweeps.items():
        thresh = info["critical_threshold_cross_0_5"]
        base = info["baseline_value"]
        print(f"  {feat:<28} | Baseline: {str(base):<6} | Flips to Phishing at: {str(thresh)}")

    print("\n--- PAIRWISE PERTURBATION IMPACT SUMMARY ---")
    for var, p_info in pairwise.items():
        print(f"  {var:<26} | Mean ΔP(Phish): {p_info['mean_delta_prob_phishing']:>+7.4f} | Flip Rate: {p_info['flip_to_phishing_rate']:>6.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Feature Sensitivity & Threshold Sweep")
    parser.add_argument("--model-path", default="models/random_forest_model.joblib", help="Path to frozen model artifact")
    parser.add_argument("--perturbation-csv", default="experiments/reports/url_perturbation_results.csv", help="Perturbation CSV path")
    parser.add_argument("--output-dir", default="experiments/reports", help="Output directory")
    args = parser.parse_args()

    run_sensitivity_analysis(model_path=args.model_path, perturbation_csv=args.perturbation_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
