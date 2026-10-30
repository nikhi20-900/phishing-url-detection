"""
Regression protection test suite (Phase 2 Task 2.4).

Guarantees that future changes cannot silently break:
1. Ground truth mapping: 0 = Phishing, 1 = Legitimate.
2. The exact names and ordering of the 18 URL features.
3. Dynamic probability-column selection from model.classes_.
4. SHAP class alignment targeting the Phishing class.
5. Compatibility between saved model artifacts and inference runtime.
"""

import unittest
import numpy as np
import pandas as pd
import joblib

from src.features import FEATURE_NAMES, extract_url_features
from src.predict import (
    load_model_and_explainer,
    get_class_probabilities,
    compute_shap_explanation,
    analyze_url,
    clear_model_cache
)

EXPECTED_18_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]

class MockCustomClassesModel:
    """Mock classifier with configurable classes_ array and probability distribution."""
    def __init__(self, classes, prob_col0, prob_col1):
        self.classes_ = np.array(classes)
        self.prob_distribution = np.array([[prob_col0, prob_col1]])

    def predict_proba(self, X):
        return self.prob_distribution

class TestRegressionProtection(unittest.TestCase):
    """Test suite guarding critical correctness invariants against regression."""

    def setUp(self):
        clear_model_cache()

    def tearDown(self):
        clear_model_cache()

    # --- 1. Invariant: 0 = Phishing, 1 = Legitimate ---
    def test_class_label_mapping_contract(self):
        """Verify that 0 is unambiguously treated as Phishing and 1 as Legitimate."""
        # Standard model: index 0 is class 0 (Phishing) with 0.85 prob
        std_model = MockCustomClassesModel(classes=[0, 1], prob_col0=0.85, prob_col1=0.15)
        dummy_df = pd.DataFrame([{f: 1.0 for f in FEATURE_NAMES}], columns=FEATURE_NAMES)
        res_std = get_class_probabilities(std_model, dummy_df)

        self.assertEqual(res_std["phishing_index"], 0)
        self.assertEqual(res_std["legitimate_index"], 1)
        self.assertAlmostEqual(res_std["phishing_probability"], 0.85)
        self.assertAlmostEqual(res_std["legitimate_probability"], 0.15)

        # Inverted model: index 0 is class 1 (Legitimate) with 0.15 prob, index 1 is class 0 (Phishing) with 0.85 prob
        inv_model = MockCustomClassesModel(classes=[1, 0], prob_col0=0.15, prob_col1=0.85)
        res_inv = get_class_probabilities(inv_model, dummy_df)

        self.assertEqual(res_inv["phishing_index"], 1)
        self.assertEqual(res_inv["legitimate_index"], 0)
        self.assertAlmostEqual(res_inv["phishing_probability"], 0.85)
        self.assertAlmostEqual(res_inv["legitimate_probability"], 0.15)

    # --- 2. Invariant: Exact Order of the 18 Features ---
    def test_18_feature_names_and_exact_order(self):
        """Ensure FEATURE_NAMES sequence cannot change or drop features."""
        self.assertEqual(len(FEATURE_NAMES), 18)
        self.assertEqual(FEATURE_NAMES, EXPECTED_18_FEATURES)

        # Verify extract_url_features preserves exact key order
        feat_dict = extract_url_features("https://www.example.com")
        self.assertEqual(list(feat_dict.keys()), EXPECTED_18_FEATURES)

    # --- 3. Invariant: Dynamic Probability Column Selection ---
    def test_dynamic_probability_column_selection(self):
        """Confirm that predict_proba column selection is NEVER hardcoded to index 0."""
        dummy_df = pd.DataFrame([{f: 1.0 for f in FEATURE_NAMES}], columns=FEATURE_NAMES)

        # If classes are [1, 0], column 0 is Legitimate, column 1 is Phishing
        mock_rev = MockCustomClassesModel(classes=[1, 0], prob_col0=0.92, prob_col1=0.08)
        res = get_class_probabilities(mock_rev, dummy_df)

        # Must correctly report phishing_probability as 0.08 (column 1), NOT 0.92 (column 0)
        self.assertAlmostEqual(res["phishing_probability"], 0.08)
        self.assertAlmostEqual(res["legitimate_probability"], 0.92)

    # --- 4. Invariant: SHAP Class Alignment Targeting Phishing ---
    def test_shap_class_alignment_always_targets_phishing(self):
        """Ensure SHAP explanations target Phishing regardless of class ordering."""
        feats_df = pd.DataFrame([extract_url_features("https://www.google.com")], columns=FEATURE_NAMES)
        model, explainer = load_model_and_explainer()

        shap_info = compute_shap_explanation(explainer, feats_df, phishing_index=0)
        self.assertEqual(shap_info["target_class"], "Phishing")
        self.assertEqual(shap_info["target_class_index"], 0)
        self.assertTrue(shap_info["verified"])

        # Additive check: base_value + sum(shap) == prob_phishing
        probs = model.predict_proba(feats_df)[0]
        reconstructed = shap_info["base_value"] + float(np.sum(shap_info["shap_values"]))
        self.assertAlmostEqual(reconstructed, float(probs[0]), places=4)

    # --- 5. Invariant: Artifact and Inference Contract ---
    def test_saved_model_artifact_contract(self):
        """Validate that the persisted production model artifact satisfies all contracts."""
        artifact = joblib.load("models/random_forest_model.joblib")

        # Must be dict with model and feature_names
        self.assertIsInstance(artifact, dict)
        self.assertIn("model", artifact)
        self.assertIn("feature_names", artifact)

        # Feature order must strictly match
        self.assertEqual(list(artifact["feature_names"]), EXPECTED_18_FEATURES)

        # Model classes must be [0, 1]
        model = artifact["model"]
        self.assertTrue(hasattr(model, "classes_"))
        self.assertEqual(list(model.classes_), [0, 1])

        # Load through inference loader
        loaded_model, loaded_explainer = load_model_and_explainer("models/random_forest_model.joblib")
        self.assertEqual(list(loaded_model.classes_), [0, 1])
        self.assertIsNotNone(loaded_explainer)

if __name__ == "__main__":
    unittest.main()
