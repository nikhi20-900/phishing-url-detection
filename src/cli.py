"""
Command-line interface for Phishing URL Analysis.
Usage:
    python3 src/cli.py "https://www.google.com"
    python3 src/cli.py "http://paypal-security-update.com/login.php"
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import analyze_url

def main():
    parser = argparse.ArgumentParser(description="Analyze a URL for phishing using Random Forest + SHAP.")
    parser.add_argument("url", type=str, help="URL to analyze")
    args = parser.parse_args()

    print(f"\n🔍 Analyzing URL: {args.url}...\n")
    result = analyze_url(args.url)

    status_icon = "🚨 PHISHING" if result["is_phishing"] else "✅ LEGITIMATE"
    print("=" * 60)
    print(f"VERDICT     : {status_icon}")
    print(f"CONFIDENCE  : {result['confidence']:.1%}")
    print(f"RISK SCORES : Phishing: {result['prob_phishing']:.1%} | Legitimate: {result['prob_legitimate']:.1%}")
    print("=" * 60)

    print("\n🎯 TOP 5 INFLUENCING FEATURES (SHAP):")
    for idx, f in enumerate(result["top_5_features"], 1):
        sign = "+" if f["shap_value"] > 0 else ""
        print(f"  {idx}. {f['feature']} = {f['value']}")
        print(f"     Impact: {f['direction']} ({sign}{f['shap_value']:.4f})")
        print(f"     Note  : {f['interpretation']}\n")

    print("📊 18 EXTRACTED FEATURES:")
    for k, v in result["features"].items():
        print(f"  {k:28}: {v}")
    print()

if __name__ == "__main__":
    main()
