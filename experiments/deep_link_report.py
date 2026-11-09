"""
Deep-Link Report & Visualization Generator (Phase 4 Experimentation).

Produces publication-grade diagnostic visualizations, SHAP attribution comparisons,
and statistical summary tables from perturbation and sensitivity experiment data:
1. Probability Shifts by Perturbation Type (Bar Plot)
2. Special Character Threshold Response Curve (Line Plot)
3. SHAP Attribution Waterfall / Comparison Plot (Root vs Deep Link Pairs)
4. Markdown-formatted summary tables for empirical documentation.

Outputs:
- experiments/reports/probability_shifts_by_perturbation.png
- experiments/reports/special_char_sensitivity_curve.png
- experiments/reports/shap_attribution_comparison.png
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_url_features, FEATURE_NAMES
from src.predict import load_model_and_explainer, compute_shap_explanation, get_class_probabilities


def plot_probability_shifts(
    summary_data: list,
    output_path: str = "experiments/reports/probability_shifts_by_perturbation.png"
):
    """
    Generate horizontal bar plot of mean P(Phish) across all perturbation variants.
    """
    df = pd.DataFrame(summary_data)
    df = df.sort_values(by="mean_prob_phishing", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))

    # Color palette based on threat severity
    colors = []
    for p in df["mean_prob_phishing"]:
        if p < 0.2:
            colors.append("#2ca02c")  # Safe green
        elif p < 0.5:
            colors.append("#1f77b4")  # Moderate blue
        elif p < 0.8:
            colors.append("#ff7f0e")  # Warning orange
        else:
            colors.append("#d62728")  # Threat red

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df["mean_prob_phishing"], color=colors, alpha=0.85, edgecolor="black", height=0.7)

    # Threshold reference line
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.50)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["variant_key"], fontsize=9)
    ax.set_xlabel("Mean Model Phishing Probability P(Phish)", fontsize=11, fontweight="bold")
    ax.set_title("Model Vulnerability Surface: Mean Phishing Score by URL Perturbation Variant\n(N=726 Perturbations across 22 Distinct Domains)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10)

    # Add data labels
    for bar, val in zip(bars, df["mean_prob_phishing"]):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha="left", fontsize=8, color="#222222")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: '{output_path}'")


def plot_sensitivity_curves(
    sweeps_data: dict,
    output_path: str = "experiments/reports/special_char_sensitivity_curve.png"
):
    """
    Plot 1D response curves for critical features (NoOfOtherSpecialCharsInURL & SpacialCharRatioInURL).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. NoOfOtherSpecialCharsInURL Curve
    curve1 = sweeps_data["NoOfOtherSpecialCharsInURL"]["response_curve"]
    x1 = [c["feature_value"] for c in curve1]
    y1 = [c["prob_phishing"] for c in curve1]

    ax1.plot(x1, y1, marker="o", linewidth=2.5, color="#d62728", label="P(Phishing)")
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
    ax1.axvline(1, color="green", linestyle=":", linewidth=1.5, label="Baseline (stanford.edu, 1 dot)")
    ax1.axvline(2, color="red", linestyle=":", linewidth=1.5, label="Critical Cliff (stanford.edu/, 1 dot + 1 slash)")

    ax1.set_xlabel("NoOfOtherSpecialCharsInURL (Count of non-alphanumeric, non-query chars)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Phishing Probability P(Phish)", fontsize=10, fontweight="bold")
    ax1.set_title("Step Jump: Special Character Count Cliff", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    # 2. SpacialCharRatioInURL Curve
    curve2 = sweeps_data["SpacialCharRatioInURL"]["response_curve"]
    x2 = [c["feature_value"] for c in curve2]
    y2 = [c["prob_phishing"] for c in curve2]

    ax2.plot(x2, y2, marker="s", linewidth=2.5, color="#1f77b4", label="P(Phishing)")
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
    ax2.axvline(0.042, color="green", linestyle=":", linewidth=1.5, label="Baseline (0.042)")
    ax2.axvline(0.100, color="red", linestyle=":", linewidth=1.5, label="Critical Cross (0.100)")

    ax2.set_xlabel("SpacialCharRatioInURL (Density of special chars relative to length)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Phishing Probability P(Phish)", fontsize=10, fontweight="bold")
    ax2.set_title("Continuous Response: Special Character Ratio Curve", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: '{output_path}'")


def plot_shap_comparisons(
    model,
    explainer,
    phish_idx: int,
    output_path: str = "experiments/reports/shap_attribution_comparison.png"
):
    """
    Generate side-by-side SHAP attribution comparison for Stanford root vs slash,
    and MIT root vs deep link.
    """
    pairs = [
        ("Pair 1: Trailing Slash Impact",
         "https://www.stanford.edu",
         "https://www.stanford.edu/"),
        ("Pair 2: Deep Link Path Impact",
         "https://www.mit.edu/",
         "https://www.mit.edu/research")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for row_idx, (title, u1, u2) in enumerate(pairs):
        for col_idx, u in enumerate([u1, u2]):
            ax = axes[row_idx, col_idx]

            feat_df = pd.DataFrame([extract_url_features(u)], columns=FEATURE_NAMES)
            probs = get_class_probabilities(model, feat_df)
            p_val = probs["phishing_probability"]

            shap_res = compute_shap_explanation(explainer, feat_df, phish_idx, expected_phishing_prob=p_val)
            vals = np.array(shap_res["shap_values"]).flatten()

            # Top 6 features by absolute magnitude
            top_idx = np.argsort(np.abs(vals))[::-1][:6]
            top_feats = [FEATURE_NAMES[i] for i in top_idx]
            top_vals = [vals[i] for i in top_idx]
            top_raw = [feat_df.iloc[0, i] for i in top_idx]

            bar_colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_vals]
            y_pos = np.arange(len(top_feats))

            ax.barh(y_pos, top_vals, color=bar_colors, alpha=0.85, edgecolor="black", height=0.6)
            ax.axvline(0, color="black", linestyle="-", linewidth=0.8)

            labels = [f"{f} = {val}" for f, val in zip(top_feats, top_raw)]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()

            verdict = "PHISHING" if p_val >= 0.5 else "LEGITIMATE"
            ax.set_title(f"{u}\nP(Phish) = {p_val:.4f} [{verdict}] (Base: {shap_res['base_value']:.4f})", fontsize=10, fontweight="bold")
            ax.set_xlabel("SHAP Impact on Phishing Probability", fontsize=9)
            ax.grid(axis="x", linestyle=":", alpha=0.6)

    fig.suptitle("Forensic Explainable AI (SHAP) Deconstruction: Root Domains vs Perturbed Variants", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: '{output_path}'")


def run_reporting(
    perturbation_json: str = "experiments/reports/url_perturbation_results.json",
    sensitivity_json: str = "experiments/reports/feature_sensitivity_results.json",
    model_path: str = "models/random_forest_model.joblib",
    output_dir: str = "experiments/reports"
):
    """Orchestrate all plot and table generation."""
    print("Loading experiment artifacts...")
    p_data = json.load(open(perturbation_json))
    s_data = json.load(open(sensitivity_json))

    model, explainer = load_model_and_explainer(model_path)
    classes = list(model.classes_)
    phish_idx = classes.index(0)

    print("Generating Figure 1: Probability Shifts by Perturbation...")
    plot_probability_shifts(p_data["variant_summary"], os.path.join(output_dir, "probability_shifts_by_perturbation.png"))

    print("Generating Figure 2: Sensitivity Curves...")
    plot_sensitivity_curves(s_data["feature_sweeps_1d"], os.path.join(output_dir, "special_char_sensitivity_curve.png"))

    print("Generating Figure 3: SHAP Attribution Comparisons...")
    plot_shap_comparisons(model, explainer, phish_idx, os.path.join(output_dir, "shap_attribution_comparison.png"))

    print("\nAll publication plots successfully generated in:", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Experiment Reporting & Visualization")
    parser.add_argument("--perturbation-json", default="experiments/reports/url_perturbation_results.json")
    parser.add_argument("--sensitivity-json", default="experiments/reports/feature_sensitivity_results.json")
    parser.add_argument("--model-path", default="models/random_forest_model.joblib")
    parser.add_argument("--output-dir", default="experiments/reports")
    args = parser.parse_args()

    run_reporting(
        perturbation_json=args.perturbation_json,
        sensitivity_json=args.sensitivity_json,
        model_path=args.model_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
