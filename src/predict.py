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
_LOADED_PATH = None

def clear_model_cache():
    """Clear memoized model and explainer cache."""
    global _MODEL_CACHE, _EXPLAINER_CACHE, _LOADED_PATH
    _MODEL_CACHE = None
    _EXPLAINER_CACHE = None
    _LOADED_PATH = None

def load_model_and_explainer(model_path="models/random_forest_model.joblib"):
    """
    Load model and initialize SHAP TreeExplainer with memoization and validation.

    Validates:
    - Model artifact exists and contains 'model' (or is a valid bare estimator).
    - Model has 'classes_' attribute with exactly binary classes {0, 1}.
    - Artifact feature names and their order match FEATURE_NAMES.
    - Handles older or incomplete artifacts with clear, descriptive ValueError.
    """
    global _MODEL_CACHE, _EXPLAINER_CACHE, _LOADED_PATH
    if _MODEL_CACHE is None or _LOADED_PATH != model_path:
        if not os.path.exists(model_path):
            dataset_path = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
            if os.path.exists(dataset_path):
                from src.train import train_and_save_model
                model_dir = os.path.dirname(model_path) or "models"
                train_and_save_model(data_path=dataset_path, model_dir=model_dir)
            else:
                raise FileNotFoundError(f"Model artifact not found at {model_path}. Run src/train.py first.")
        try:
            artifact = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model artifact at '{model_path}': {e}") from e

        if not isinstance(artifact, dict) and not hasattr(artifact, "predict_proba"):
            raise ValueError(
                f"Invalid model artifact format at '{model_path}': "
                f"expected a dictionary or estimator, got {type(artifact).__name__}."
            )

        if isinstance(artifact, dict):
            if "model" not in artifact:
                raise ValueError(
                    f"Incomplete model artifact at '{model_path}': missing required 'model' key."
                )
            model = artifact["model"]

            # Feature schema validation: check 'feature_names' key if present
            if "feature_names" in artifact:
                artifact_features = list(artifact["feature_names"])
                if artifact_features != FEATURE_NAMES:
                    raise ValueError(
                        f"Feature mismatch between artifact and inference schema:\n"
                        f"Artifact features: {artifact_features}\n"
                        f"Expected features: {FEATURE_NAMES}"
                    )
            elif hasattr(model, "feature_names_in_"):
                if list(model.feature_names_in_) != FEATURE_NAMES:
                    raise ValueError(
                        f"Feature mismatch between model feature_names_in_ and inference schema:\n"
                        f"Model features: {list(model.feature_names_in_)}\n"
                        f"Expected features: {FEATURE_NAMES}"
                    )
            else:
                raise ValueError(
                    f"Incomplete model artifact at '{model_path}': missing 'feature_names' metadata."
                )

            # If artifact contains explicit classes or phishing_index, verify consistency with model.classes_
            if "classes" in artifact and hasattr(model, "classes_"):
                if list(artifact["classes"]) != list(model.classes_):
                    raise ValueError(
                        f"Inconsistent model artifact: artifact 'classes' {artifact['classes']} "
                        f"does not match model.classes_ {list(model.classes_)}."
                    )
        else:
            # Bare model artifact (older legacy format)
            model = artifact
            if hasattr(model, "feature_names_in_"):
                if list(model.feature_names_in_) != FEATURE_NAMES:
                    raise ValueError(
                        f"Feature mismatch between bare model and inference schema:\n"
                        f"Found: {list(model.feature_names_in_)}\n"
                        f"Expected: {FEATURE_NAMES}"
                    )
            else:
                raise ValueError(
                    f"Incomplete bare model artifact at '{model_path}': missing 'feature_names_in_' metadata."
                )

        if not hasattr(model, "classes_"):
            raise ValueError(
                f"Incomplete model artifact at '{model_path}': loaded model does not expose 'classes_' attribute."
            )

        classes = list(model.classes_)
        if set(classes) != {0, 1} or len(classes) != 2:
            raise ValueError(
                f"Invalid model class configuration in loaded model: expected binary classes {{0, 1}}, got {classes}."
            )

        _MODEL_CACHE = model
        _EXPLAINER_CACHE = shap.TreeExplainer(_MODEL_CACHE)
        _LOADED_PATH = model_path
    return _MODEL_CACHE, _EXPLAINER_CACHE

def interpret_feature_cybersecurity(feature_name: str, value: float, shap_val: float) -> str:
    """Generate a scientifically defensible interpretation for a given feature impact."""
    pushes_phishing = shap_val > 0

    if feature_name == "IsHTTPS":
        if value == 0:
            return "Unencrypted HTTP transport contributed toward the phishing classification, as absence of TLS is prevalent among deceptive links in the training data."
        return "HTTPS is active, establishing encrypted transport standard for genuine domains."

    if feature_name == "IsDomainIP":
        if value == 1:
            return "Direct IP host addressing contributed toward the phishing classification based on patterns learned from the training data."
        return "Uses a standard registered domain name rather than a raw IP address."

    if feature_name in ["HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio"]:
        if value > 0:
            return "Percent-encoded hex characters (%xx) contributed toward the phishing classification based on patterns learned from the training data."
        return "Clean URL without hex percent-encoding obfuscation."

    if feature_name == "NoOfSubDomain":
        if pushes_phishing:
            return f"Subdomain structure ({int(value)} level(s)) contributed toward the phishing classification based on patterns learned from the training data."
        return f"{int(value)} subdomain level(s), typical for standard domain routing."

    if feature_name == "URLLength":
        if pushes_phishing:
            return f"URL length ({int(value)} chars) contributed toward the phishing classification based on patterns learned from the training data."
        return f"Compact URL length ({int(value)} chars) consistent with concise legitimate destinations."

    if feature_name == "DomainLength":
        if pushes_phishing:
            return f"Domain length ({int(value)} chars) contributed toward the phishing classification based on patterns learned from the training data."
        return f"Standard domain length ({int(value)} chars) aligned with typical legitimate domain patterns in the training data."

    if feature_name == "NoOfOtherSpecialCharsInURL":
        if pushes_phishing:
            return f"Special character count ({int(value)}) contributed toward the phishing classification based on patterns learned from the training data."
        return f"Minimal special characters ({int(value)}) aligns with clean site architecture."

    if feature_name == "SpacialCharRatioInURL":
        if pushes_phishing:
            return f"Special character ratio ({value:.1%}) contributed toward the phishing classification based on patterns learned from the training data."
        return f"Special character ratio ({value:.1%}) within expected baseline bounds."

    if feature_name in ["NoOfDegitsInURL", "DegitRatioInURL"]:
        if pushes_phishing:
            return f"Digit composition ({int(value)} digits) contributed toward the phishing classification based on patterns learned from the training data."
        return f"Low digit count ({int(value)}), typical for authentic brand names."

    if feature_name == "LetterRatioInURL":
        return f"Alphabetic composition ({value:.1%}) contributed toward the classification based on lexical distributions in the training data."

    if feature_name in ["NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL"]:
        if pushes_phishing:
            return f"Query parameter syntax ({int(value)}) contributed toward the phishing classification based on patterns learned from the training data."
        return f"Low query parameter activity ({int(value)})."

    if feature_name == "TLDLength":
        if pushes_phishing:
            return f"TLD length ({int(value)}) contributed toward the phishing classification based on patterns learned from the training data."
        return f"Standard TLD length ({int(value)}) matching common top-level registries."

    return f"This feature contributed {shap_val:+.3f} toward the classification based on patterns learned from the training data."

def get_class_probabilities(model, features_df: pd.DataFrame) -> dict:
    """
    Dynamically extract class probabilities from the model based on model.classes_.

    Validates that:
    - model.classes_ contains exactly the expected binary classes {0, 1},
      where 0 = Phishing and 1 = Legitimate.
    - features_df columns and their order match FEATURE_NAMES.

    Returns:
        dict: {
            "phishing_probability": float,
            "legitimate_probability": float,
            "phishing_index": int,
            "legitimate_index": int
        }
    Raises:
        ValueError: If model does not have classes_ or if classes_ != {0, 1},
                    or if feature columns do not match FEATURE_NAMES.
    """
    if not hasattr(model, "classes_"):
        raise ValueError("Model does not expose 'classes_' attribute.")

    classes = list(model.classes_)
    if set(classes) != {0, 1} or len(classes) != 2:
        raise ValueError(
            f"Invalid model class configuration: expected binary classes {{0, 1}}, got {classes}."
        )

    if isinstance(features_df, pd.DataFrame):
        if list(features_df.columns) != FEATURE_NAMES:
            raise ValueError(
                f"Feature columns do not match expected FEATURE_NAMES order: "
                f"{list(features_df.columns)} vs {FEATURE_NAMES}."
            )

    probabilities = model.predict_proba(features_df)[0]
    phishing_index = classes.index(0)
    legitimate_index = classes.index(1)

    prob_phishing = float(probabilities[phishing_index])
    prob_legitimate = float(probabilities[legitimate_index])

    return {
        "phishing_probability": prob_phishing,
        "legitimate_probability": prob_legitimate,
        "phishing_index": phishing_index,
        "legitimate_index": legitimate_index
    }

def compute_shap_explanation(
    explainer,
    features_df: pd.DataFrame,
    phishing_index: int,
    expected_phishing_prob: float = None,
    tolerance: float = 1e-3
) -> dict:
    """
    Compute local SHAP explanation specifically aligned with the Phishing class.

    Verifies:
    - Supported output shapes and clear errors for ambiguous/unsupported shapes.
    - Class alignment targeting the phishing index:
      * For 3D outputs (multi-class: samples x features x classes), extracts the phishing slice
        and validates against expected model probability.
      * For 2D outputs (samples x features), independently verifies whether the values explain
        Phishing or Legitimate via metadata and/or additive consistency against expected output.
        Inverts Legitimate explanations to target Phishing, and rejects ambiguous/unverified formats.
    - Feature count and column ordering matching FEATURE_NAMES.
    - Additive consistency: base_value + sum(shap_values) == expected model output for phishing.
    - Finite numerical values (no NaN or Inf).

    Returns:
        dict: {
            "shap_values": np.ndarray,
            "base_value": float,
            "target_class": "Phishing",
            "target_class_index": int,
            "explanation_target": str,
            "verified": bool
        }
    Raises:
        ValueError: On ambiguous output shapes, non-finite values, dimension mismatches,
                    or failure to verify that the explanation matches the phishing model output.
    """
    if isinstance(features_df, pd.DataFrame):
        if list(features_df.columns) != FEATURE_NAMES:
            raise ValueError(
                f"SHAP input columns do not match FEATURE_NAMES order: "
                f"{list(features_df.columns)} vs {FEATURE_NAMES}."
            )

    shap_res = explainer(features_df)
    vals = getattr(shap_res, "values", shap_res)
    base_vals = getattr(shap_res, "base_values", getattr(explainer, "expected_value", 0.0))

    # Support list of arrays from legacy shap_values (e.g. [class_0_vals, class_1_vals])
    if isinstance(vals, (list, tuple)):
        if len(vals) != 2:
            raise ValueError(f"Unsupported SHAP list output: expected 2 class outputs, got {len(vals)}.")
        if phishing_index >= len(vals):
            raise ValueError(f"Phishing index {phishing_index} out of bounds for SHAP output list of length {len(vals)}.")
        vals = np.stack(vals, axis=-1)
        if isinstance(base_vals, (list, tuple)):
            base_vals = np.array(base_vals)

    # Support (classes, samples, features) orientation from stacked list
    if vals.ndim == 3 and vals.shape[0] == 2 and vals.shape[2] == len(FEATURE_NAMES):
        vals = np.moveaxis(vals, 0, -1)

    if vals.ndim == 3:
        if vals.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"SHAP feature dimension mismatch: expected {len(FEATURE_NAMES)} features, got {vals.shape[1]}."
            )
        if vals.shape[2] != 2:
            raise ValueError(
                f"Unsupported SHAP class dimensionality: expected binary output with 2 classes, got {vals.shape[2]}."
            )
        if phishing_index not in (0, 1) or phishing_index >= vals.shape[2]:
            raise ValueError(
                f"Phishing class index {phishing_index} out of bounds for SHAP output with {vals.shape[2]} classes."
            )
        shap_phishing = vals[0, :, phishing_index]

        if np.ndim(base_vals) == 2:
            base_phishing = float(base_vals[0, phishing_index])
        elif np.ndim(base_vals) == 1:
            base_phishing = float(base_vals[phishing_index])
        else:
            base_phishing = float(base_vals)

        # Verify that explanation matches expected model output using floating-point tolerance
        if expected_phishing_prob is not None:
            reconstructed = base_phishing + float(np.sum(shap_phishing))
            if not np.isclose(reconstructed, expected_phishing_prob, atol=tolerance, rtol=1e-3):
                raise ValueError(
                    f"SHAP explanation does not match expected model output: "
                    f"reconstructed phishing probability {reconstructed:.5f} != expected {expected_phishing_prob:.5f} "
                    f"(tolerance {tolerance})."
                )

    elif vals.ndim == 2:
        if vals.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"SHAP feature dimension mismatch: expected {len(FEATURE_NAMES)} features, got {vals.shape[1]}."
            )

        raw_shap = vals[0, :]
        raw_base = float(np.ravel(base_vals)[0]) if np.size(base_vals) > 0 else 0.0
        reconstructed_prob = raw_base + float(np.sum(raw_shap))

        # Check metadata on explanation/explainer
        output_names = getattr(shap_res, "output_names", None)
        target_class = getattr(shap_res, "target_class", None)
        target_class_index = getattr(shap_res, "target_class_index", None)

        is_phishing_by_meta = False
        is_legit_by_meta = False

        if target_class_index is not None:
            is_phishing_by_meta = (target_class_index == phishing_index)
            is_legit_by_meta = (target_class_index == (1 - phishing_index))
        elif target_class is not None:
            t_str = str(target_class).strip().lower()
            if t_str in ("phishing", "phish", "0"):
                is_phishing_by_meta = True
            elif t_str in ("legitimate", "legit", "1"):
                is_legit_by_meta = True
        elif output_names is not None:
            names = [str(n).strip().lower() for n in np.ravel(output_names)]
            if len(names) == 1:
                name = names[0]
                if name in ("phishing", "phish", "0", "output 0", f"output {phishing_index}"):
                    is_phishing_by_meta = True
                elif name in ("legitimate", "legit", "1", "output 1", f"output {1 - phishing_index}"):
                    is_legit_by_meta = True

        # Check additive match with expected probabilities if provided
        is_phishing_by_prob = False
        is_legit_by_prob = False

        if expected_phishing_prob is not None:
            expected_legit_prob = 1.0 - expected_phishing_prob
            matches_phish = bool(np.isclose(reconstructed_prob, expected_phishing_prob, atol=tolerance, rtol=1e-3))
            matches_legit = bool(np.isclose(reconstructed_prob, expected_legit_prob, atol=tolerance, rtol=1e-3))

            if np.isclose(expected_phishing_prob, 0.5, atol=tolerance, rtol=1e-3):
                # When prob ~ 0.5, probability matching alone is ambiguous without metadata
                if is_phishing_by_meta:
                    is_phishing_by_prob = matches_phish
                elif is_legit_by_meta:
                    is_legit_by_prob = matches_legit
                else:
                    raise ValueError(
                        "Ambiguous 2D SHAP output: expected probability is near 0.5 and no metadata "
                        "identifies whether the 2D output targets Phishing or Legitimate."
                    )
            else:
                is_phishing_by_prob = matches_phish
                is_legit_by_prob = matches_legit

        # Resolve verified target class
        if is_phishing_by_meta or is_phishing_by_prob:
            if expected_phishing_prob is not None and not np.isclose(reconstructed_prob, expected_phishing_prob, atol=tolerance, rtol=1e-3):
                raise ValueError(
                    f"2D SHAP explanation identified as Phishing does not match expected model output: "
                    f"reconstructed {reconstructed_prob:.5f} vs expected {expected_phishing_prob:.5f}."
                )
            shap_phishing = raw_shap
            base_phishing = raw_base

        elif is_legit_by_meta or is_legit_by_prob:
            expected_legit_prob = 1.0 - expected_phishing_prob if expected_phishing_prob is not None else None
            if expected_legit_prob is not None and not np.isclose(reconstructed_prob, expected_legit_prob, atol=tolerance, rtol=1e-3):
                raise ValueError(
                    f"2D SHAP explanation identified as Legitimate does not match expected model output: "
                    f"reconstructed {reconstructed_prob:.5f} vs expected {expected_legit_prob:.5f}."
                )
            # Invert Legitimate attributions to target Phishing
            shap_phishing = -raw_shap
            base_phishing = 1.0 - raw_base

        else:
            raise ValueError(
                "Unsupported or ambiguous 2D SHAP output: unable to independently verify that the "
                "2D explanation corresponds to the phishing class. Provide class metadata or expected "
                "model output matching."
            )

    else:
        raise ValueError(f"Unsupported SHAP values dimensionality: {vals.ndim}D array with shape {vals.shape}.")

    if not np.all(np.isfinite(shap_phishing)):
        raise ValueError("SHAP explanation returned non-finite feature attribution values (NaN or Inf).")
    if not np.isfinite(base_phishing):
        raise ValueError(f"SHAP explanation returned non-finite base value: {base_phishing}.")

    return {
        "shap_values": shap_phishing,
        "base_value": base_phishing,
        "target_class": "Phishing",
        "target_class_index": phishing_index,
        "explanation_target": "Phishing risk contribution (negative values support Legitimate)",
        "verified": True
    }

def analyze_url(url: str, model_path="models/random_forest_model.joblib", threshold: float = 0.5):
    """
    Perform end-to-end analysis on a URL:
    1. Feature extraction (18 features)
    2. Prediction & probabilities dynamically aligned with model.classes_
    3. SHAP local explanation independently verified against phishing output
    4. Top 5 features influencing the prediction

    Parameters:
        url (str): Target URL string to analyze.
        model_path (str): Filepath to the serialized model artifact.
        threshold (float): Decision threshold for phishing classification (default: 0.5).
                           A URL is flagged as Phishing if prob_phishing >= threshold.
                           Model probabilities are statistical estimates, not absolute guarantees of safety.
    """
    model, explainer = load_model_and_explainer(model_path)

    # 1. Feature extraction
    features_dict = extract_url_features(url)
    features_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)

    # 2. Prediction & Probability alignment dynamically derived from model.classes_
    class_probs = get_class_probabilities(model, features_df)
    prob_phishing = class_probs["phishing_probability"]
    prob_legitimate = class_probs["legitimate_probability"]
    phishing_index = class_probs["phishing_index"]

    # Decision rule: Phishing if prob_phishing >= threshold (default 0.5)
    is_phishing = prob_phishing >= threshold
    label = "Phishing" if is_phishing else "Legitimate"
    confidence = prob_phishing if is_phishing else prob_legitimate

    # 3. SHAP local explanation (dynamically targeting phishing_index and verified against prob_phishing)
    shap_info = compute_shap_explanation(
        explainer,
        features_df,
        phishing_index=phishing_index,
        expected_phishing_prob=prob_phishing
    )
    shap_phishing = shap_info["shap_values"]
    base_phishing = shap_info["base_value"]

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
        "is_phishing": is_phishing,
        "confidence": confidence,
        "prob_phishing": prob_phishing,
        "prob_legitimate": prob_legitimate,
        "threshold": threshold,
        "features": features_dict,
        "features_df": features_df,
        "base_value_phishing": base_phishing,
        "target_class": shap_info["target_class"],
        "target_class_index": phishing_index,
        "explanation_target": shap_info["explanation_target"],
        "explanation_verified": shap_info.get("verified", False),
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
