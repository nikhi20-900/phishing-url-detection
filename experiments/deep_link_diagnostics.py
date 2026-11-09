"""
Deep-Link & URL Perturbation Diagnostics (Phase 4 Experimentation).

Performs controlled, systematic perturbation testing on the frozen Random Forest
model to isolate the exact structural features causing false positives on legitimate URLs
and analyze model stability across URL variations:
- Protocol schemes (http vs https)
- Hostname variations (bare domain, www prefix, multi-level subdomains)
- Trailing slash presence on root domains
- Path depth (single-level, multi-level directory paths)
- Path character composition (alphabetic, hyphens, digits, file extensions)
- Query strings (single parameter, multiple parameters, root query)
- URL fragments (#section)

Outputs:
- experiments/reports/url_perturbation_results.csv
- experiments/reports/url_perturbation_results.json
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_url_features, FEATURE_NAMES, parse_url

# Standard seed domains representing authoritative institutions and known threat roots
SEED_DOMAINS = [
    # Top Academic Institutions (.edu, .ac.uk)
    {"domain": "stanford.edu", "category": "Academic", "subdomain": "cs"},
    {"domain": "harvard.edu", "category": "Academic", "subdomain": "seas"},
    {"domain": "mit.edu", "category": "Academic", "subdomain": "news"},
    {"domain": "berkeley.edu", "category": "Academic", "subdomain": "eecs"},
    {"domain": "cmu.edu", "category": "Academic", "subdomain": "cs"},
    {"domain": "ox.ac.uk", "category": "Academic", "subdomain": "ox"},
    {"domain": "cam.ac.uk", "category": "Academic", "subdomain": "cl"},

    # Governmental & Standards Bodies
    {"domain": "gov.uk", "category": "Government", "subdomain": "www"},
    {"domain": "cisa.gov", "category": "Government", "subdomain": "www"},
    {"domain": "nasa.gov", "category": "Government", "subdomain": "www"},
    {"domain": "w3.org", "category": "Standards", "subdomain": "www"},
    {"domain": "ietf.org", "category": "Standards", "subdomain": "www"},

    # Open Source & Tech Infrastructure
    {"domain": "python.org", "category": "OpenSource", "subdomain": "docs"},
    {"domain": "apache.org", "category": "OpenSource", "subdomain": "projects"},
    {"domain": "kernel.org", "category": "OpenSource", "subdomain": "git"},
    {"domain": "github.com", "category": "TechPlatform", "subdomain": "api"},

    # Scientific & Reference Platforms
    {"domain": "wikipedia.org", "category": "Reference", "subdomain": "en"},
    {"domain": "nature.com", "category": "Reference", "subdomain": "www"},
    {"domain": "arxiv.org", "category": "Reference", "subdomain": "export"},

    # Control Phishing Roots (Bare Root Threat Samples from Phase 3/4)
    {"domain": "gonoconlatarjetaverde.com.mx", "category": "ThreatRoot", "subdomain": "www"},
    {"domain": "mistoreosqv-jp.top", "category": "ThreatRoot", "subdomain": "www"},
    {"domain": "omahanewsbulletin.com", "category": "ThreatRoot", "subdomain": "www"}
]

# Systematic perturbation matrix definitions
PERTURBATION_TEMPLATES = [
    # Protocol & Hostname Baselines
    ("bare_root", "https://{domain}"),
    ("bare_root_slash", "https://{domain}/"),
    ("www_root", "https://www.{domain}"),
    ("www_root_slash", "https://www.{domain}/"),
    ("http_bare_root", "http://{domain}"),
    ("http_bare_root_slash", "http://{domain}/"),
    ("http_www_root", "http://www.{domain}"),
    ("http_www_root_slash", "http://www.{domain}/"),
    ("subdomain_root", "https://{subdomain}.{domain}"),
    ("subdomain_root_slash", "https://{subdomain}.{domain}/"),

    # Path Depths (Without Query)
    ("path_single_short", "https://www.{domain}/info"),
    ("path_single_short_slash", "https://www.{domain}/info/"),
    ("path_single_medium", "https://www.{domain}/research"),
    ("path_single_medium_slash", "https://www.{domain}/research/"),
    ("path_two_levels", "https://www.{domain}/research/projects"),
    ("path_two_levels_slash", "https://www.{domain}/research/projects/"),
    ("path_three_levels", "https://www.{domain}/dept/research/ai"),
    ("path_three_levels_slash", "https://www.{domain}/dept/research/ai/"),
    ("path_deep_five_levels", "https://www.{domain}/a/b/c/d/e"),

    # Path Character Composition
    ("path_hyphenated", "https://www.{domain}/research-and-development"),
    ("path_underscore", "https://www.{domain}/research_and_development"),
    ("path_with_digits", "https://www.{domain}/annual-report-2026"),
    ("path_file_html", "https://www.{domain}/index.html"),
    ("path_file_pdf", "https://www.{domain}/whitepaper.pdf"),
    ("path_upper_case", "https://www.{domain}/Research/Projects"),

    # Query Parameters
    ("query_on_root", "https://www.{domain}/?ref=home"),
    ("query_single_param", "https://www.{domain}/search?q=cybersecurity"),
    ("query_two_params", "https://www.{domain}/search?q=cybersecurity&page=1"),
    ("query_three_params", "https://www.{domain}/search?q=cybersecurity&page=1&lang=en"),
    ("query_complex", "https://www.{domain}/view?doc=annual2026&format=pdf&highlight=true"),

    # Fragment / Anchor Delimiters
    ("fragment_root", "https://www.{domain}/#overview"),
    ("fragment_path", "https://www.{domain}/about#history"),
    ("query_and_fragment", "https://www.{domain}/search?q=test#results")
]


def load_frozen_model(model_path: str = "models/random_forest_model.joblib"):
    """Load model artifact and dynamically resolve estimator and class indices."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at '{model_path}'.")

    artifact = joblib.load(model_path)
    model = artifact["model"] if isinstance(artifact, dict) else artifact

    if not hasattr(model, "classes_"):
        raise ValueError("Loaded model object has no 'classes_' attribute.")

    classes = list(model.classes_)
    phish_idx = classes.index(0)
    legit_idx = classes.index(1)

    return model, phish_idx, legit_idx


def generate_perturbations(seed_domains: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Generate all combinatorial URL perturbations from seed domains."""
    if seed_domains is None:
        seed_domains = SEED_DOMAINS

    records = []
    for entry in seed_domains:
        domain = entry["domain"]
        cat = entry["category"]
        sub = entry.get("subdomain", "dept")

        for key, template in PERTURBATION_TEMPLATES:
            # Format URL template
            url = template.format(domain=domain, subdomain=sub)
            records.append({
                "domain": domain,
                "domain_category": cat,
                "variant_key": key,
                "url": url
            })

    return records


def run_diagnostics(
    model_path: str = "models/random_forest_model.joblib",
    output_dir: str = "experiments/reports"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute end-to-end controlled perturbation diagnostics.
    """
    model, phish_idx, legit_idx = load_frozen_model(model_path)
    url_records = generate_perturbations()

    print(f"Generated {len(url_records)} URL perturbations across {len(SEED_DOMAINS)} seed domains.")
    print("Extracting 18 features and computing model predictions...")

    results = []
    for rec in url_records:
        u = rec["url"]
        feat = extract_url_features(u)
        feat_df = pd.DataFrame([feat], columns=FEATURE_NAMES)
        probs = model.predict_proba(feat_df)[0]

        prob_p = float(probs[phish_idx])
        prob_l = float(probs[legit_idx])
        pred_label = "Phishing" if prob_p >= 0.5 else "Legitimate"

        row = {
            "domain": rec["domain"],
            "domain_category": rec["domain_category"],
            "variant_key": rec["variant_key"],
            "url": u,
            "predicted_label": pred_label,
            "prob_phishing": prob_p,
            "prob_legitimate": prob_l,
            **feat
        }
        results.append(row)

    df_results = pd.DataFrame(results)

    # Compute aggregation by variant type
    variant_summary = df_results.groupby("variant_key").agg(
        total_urls=("url", "count"),
        phishing_rate=("predicted_label", lambda s: float((s == "Phishing").mean())),
        mean_prob_phishing=("prob_phishing", "mean"),
        std_prob_phishing=("prob_phishing", "std"),
        mean_url_length=("URLLength", "mean"),
        mean_special_chars=("NoOfOtherSpecialCharsInURL", "mean"),
        mean_special_ratio=("SpacialCharRatioInURL", "mean"),
        mean_letter_ratio=("LetterRatioInURL", "mean")
    ).reset_index()

    # Reorder by mean_prob_phishing ascending
    variant_summary = variant_summary.sort_values(by="mean_prob_phishing").reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "url_perturbation_results.csv")
    json_path = os.path.join(output_dir, "url_perturbation_results.json")

    df_results.to_csv(csv_path, index=False)
    summary_dict = {
        "total_perturbations": len(df_results),
        "total_seed_domains": len(SEED_DOMAINS),
        "variant_summary": variant_summary.to_dict(orient="records")
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nResults saved to '{csv_path}' and '{json_path}'.")
    return df_results, summary_dict


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Deep Link URL Perturbation Diagnostics")
    parser.add_argument("--model-path", default="models/random_forest_model.joblib", help="Path to frozen model artifact")
    parser.add_argument("--output-dir", default="experiments/reports", help="Output directory for reports")
    args = parser.parse_args()

    run_diagnostics(model_path=args.model_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
