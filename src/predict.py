"""
Prediction and Explainable AI (SHAP) inference module.
Uses the trained Random Forest model to predict Phishing vs Legitimate
and compute SHAP contributions for transparent cybersecurity decisions.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.features import (
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    FEATURE_CATEGORIES,
    extract_url_features,
    extract_features_dataframe
)

_MODEL_CACHE = None
_EXPLAINER_CACHE = None

def load_model_and_explainer(model_path="models/random_forest_model.joblib"):
    """Load model and initialize SHAP TreeExplainer with memoization."""
    global _MODEL_CACHE, _EXPLAINER_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(model_path):
            dataset_path = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
            if os.path.exists(dataset_path):
                from src.train import train_and_save_model
                model_dir = os.path.dirname(model_path) or "models"
                train_and_save_model(data_path=dataset_path, model_dir=model_dir)
            else:
                raise FileNotFoundError(f"Model artifact not found at {model_path}. Run src/train.py first.")
        artifact = joblib.load(model_path)
        _MODEL_CACHE = artifact["model"]
        _EXPLAINER_CACHE = shap.TreeExplainer(_MODEL_CACHE)
    return _MODEL_CACHE, _EXPLAINER_CACHE

def interpret_feature_cybersecurity(feature_name: str, value: float, shap_val: float) -> str:
    """Generate a contextual cybersecurity interpretation for a given feature impact."""
    pushes_phishing = shap_val > 0

    if feature_name == "IsHTTPS":
        if value == 0:
            return "Unencrypted HTTP transport is a common hallmark of credential harvesting and deceptive links."
        return "HTTPS is active, establishing encrypted transport standard for genuine domains."

    if feature_name == "IsDomainIP":
        if value == 1:
            return "Direct IP addressing in domain bypasses DNS registration to evade domain reputation filters."
        return "Uses a standard registered domain name rather than a raw IP address."

    if feature_name == "HasObfuscation" or feature_name == "NoOfObfuscatedChar":
        if value > 0:
            return "Percent-encoded hex characters (%xx) detected, commonly used to hide malicious keywords or paths."
        return "Clean URL without hex percent-encoding obfuscation."

    if feature_name == "NoOfSubDomain":
        if value >= 3:
            return f"High subdomain depth ({int(value)} levels) frequently conceals target brands inside subdomains."
        elif value == 0:
            return "No subdomains present; direct apex or basic domain structure."
        return f"{int(value)} subdomain level(s), typical for standard domain routing."

    if feature_name == "URLLength":
        if pushes_phishing:
            return f"Extended URL length ({int(value)} chars) often harbors embedded tracking parameters or payload strings."
        return f"Compact URL length ({int(value)} chars) consistent with concise legitimate destinations."

    if feature_name == "DomainLength":
        if pushes_phishing:
            return f"Long domain name ({int(value)} chars) may mimic legitimate brand names via typo-squatting."
        return f"Standard domain length ({int(value)} chars)."

    if feature_name == "NoOfOtherSpecialCharsInURL":
        if pushes_phishing:
            return f"Unusually high punctuation/special characters ({int(value)}) typical of complex phishing redirects."
        return f"Minimal special characters ({int(value)}) aligns with clean site architecture."

    if feature_name == "SpacialCharRatioInURL":
        if pushes_phishing:
            return f"Special character ratio ({value:.1%}) is elevated, indicating punctuation noise or redirection syntax."
        return f"Special character ratio ({value:.1%}) within healthy bounds."

    if feature_name == "NoOfDegitsInURL" or feature_name == "DegitRatioInURL":
        if pushes_phishing:
            return f"Elevated digit count ({int(value)}) in URL, often seen in auto-generated phishing domains."
        return f"Low digit count ({int(value)}), typical for authentic human-readable brand names."

    if feature_name == "LetterRatioInURL":
        return f"Alphabetic composition ({value:.1%}) analyzed against benign lexical profiles."

    if feature_name in ["NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL"]:
        if pushes_phishing:
            return f"Heavy query parameter syntax indicates dynamic phishing parameter passing."
        return f"Low query parameter activity ({int(value)})."

    if feature_name == "TLDLength":
        if pushes_phishing:
            return f"Non-standard TLD length ({int(value)}) frequently associated with newer or suspicious TLDs."
        return f"Standard TLD length ({int(value)}) matching common top-level registries."

    return f"Feature value {value} contributes {shap_val:+.3f} toward the final risk assessment."

def analyze_url(url: str, model_path="models/random_forest_model.joblib"):
    """
    Perform end-to-end analysis on a URL:
    1. Feature extraction (18 features)
    2. Prediction & probabilities
    3. SHAP local explanation
    4. Top 5 features influencing the prediction
    """
    model, explainer = load_model_and_explainer(model_path)
    
    # 1. Feature extraction
    features_dict = extract_url_features(url)
    features_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)
    
    # 2. Prediction (0 = Phishing, 1 = Legitimate)
    pred_class = int(model.predict(features_df)[0])
    probabilities = model.predict_proba(features_df)[0]
    
    prob_phishing = float(probabilities[0])
    prob_legitimate = float(probabilities[1])
    
    label = "Phishing" if pred_class == 0 else "Legitimate"
    confidence = prob_phishing if pred_class == 0 else prob_legitimate

    # 3. SHAP local explanation
    # explainer(features_df) returns shape (1, 18, 2)
    shap_res = explainer(features_df)
    
    # We focus on the Phishing class (index 0):
    # positive SHAP -> increases phishing risk
    # negative SHAP -> supports legitimate
    shap_phishing = shap_res.values[0, :, 0]
    base_phishing = float(shap_res.base_values[0, 0])

    # 4. Feature impacts
    impacts = []
    for name, val, sv in zip(FEATURE_NAMES, features_df.iloc[0], shap_phishing):
        sv_float = float(sv)
        impact_dir = "Increases Phishing Risk" if sv_float > 0 else "Supports Legitimate"
        cyber_note = interpret_feature_cybersecurity(name, float(val), sv_float)
        
        impacts.append({
            "feature": name,
            "category": FEATURE_CATEGORIES.get(name, "General"),
            "description": FEATURE_DESCRIPTIONS.get(name, ""),
            "value": float(val),
            "shap_value": sv_float,
            "abs_shap": abs(sv_float),
            "direction": impact_dir,
            "interpretation": cyber_note
        })

    # Sort by absolute SHAP value to find top drivers
    impacts_sorted = sorted(impacts, key=lambda x: x["abs_shap"], reverse=True)
    top_5_features = impacts_sorted[:5]

    return {
        "url": url,
        "prediction": label,
        "is_phishing": pred_class == 0,
        "confidence": confidence,
        "prob_phishing": prob_phishing,
        "prob_legitimate": prob_legitimate,
        "features": features_dict,
        "features_df": features_df,
        "base_value_phishing": base_phishing,
        "all_impacts": impacts_sorted,
        "top_5_features": top_5_features
    }

def create_shap_plot(analysis_result, max_display=10):
    """
    Create a clean matplotlib horizontal bar plot for SHAP contributions.
    """
    impacts = analysis_result["all_impacts"][:max_display]
    impacts.reverse()  # Top feature on top
    
    features = [f"{i['feature']} ({i['value']:.3g}" if isinstance(i['value'], float) and not i['value'].is_integer() 
                else f"{i['feature']} ({int(i['value'])})" for i in impacts]
    shap_values = [i["shap_value"] for i in impacts]
    colors = ["#ef4444" if sv > 0 else "#10b981" for sv in shap_values]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("#0e1726")
    ax.set_facecolor("#0e1726")

    bars = ax.barh(features, shap_values, color=colors, height=0.55, edgecolor="none", zorder=3)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.2, zorder=2)

    # Style axes
    ax.tick_params(axis="y", colors="#e2e8f0", labelsize=9)
    ax.tick_params(axis="x", colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(axis="x", color="#1e293b", linestyle=":", zorder=1)
    ax.set_xlabel("SHAP Impact on Phishing Probability (Red = Risk, Green = Safe)", color="#94a3b8", fontsize=9, labelpad=8)
    ax.set_title("Feature Contribution Breakdown (SHAP)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    return fig
