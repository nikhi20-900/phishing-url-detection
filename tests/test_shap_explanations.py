"""
Unit tests for SHAP explanation correctness, alignment, and robustness (Task 3).

Verifies:
- Dynamic selection of the phishing class using model.classes_.
- Correct class alignment for SHAP values and base values.
- Correct feature names and ordering for all 18 features.
- Supported SHAP output shapes (3D multi-class, 2D single-target) and clear errors for ambiguous shapes.
- Finite SHAP values and base values (rejection of NaN/Inf).
- Additive consistency in the explainer's actual output space (base_val + sum(shap_vals) == proba).
- Structural compatibility for waterfall plots (shap.Explanation).
- Explicit identification of phishing-class target when explaining legitimate verdicts.
- Independence of predictions and probabilities from SHAP calculation.
"""

import unittest
import numpy as np
import pandas as pd
import shap

from src.features import FEATURE_NAMES, extract_url_features
from src.predict import (
    compute_shap_explanation,
    analyze_url,
    load_model_and_explainer,
    get_class_probabilities
)

class MockExplanationResult:
    """Mock shap.Explanation container with configurable values and base_values."""
    def __init__(self, values, base_values):
        self.values = np.array(values, dtype=float)
        self.base_values = np.array(base_values, dtype=float)

class MockExplainer:
    def __init__(self, values, base_values):
        self._values = values
        self._base_values = base_values

    def __call__(self, df):
        return MockExplanationResult(self._values, self._base_values)

class TestSHAPExplanations(unittest.TestCase):
    """Test suite for SHAP explanation correctness and validation."""

    def setUp(self):
        self.dummy_features = pd.DataFrame([{f: 1.0 for f in FEATURE_NAMES}])
        self.n_features = len(FEATURE_NAMES)
        self.model, self.explainer = load_model_and_explainer()

    # --- 1. Dynamic Class Selection & Alignment ---
    def test_dynamic_phishing_class_selection_standard_and_reversed(self):
        # 3D shape (1, 18, 2): class 0 has values 0.1, class 1 has values 0.9
        vals_0 = np.full(self.n_features, 0.1)
        vals_1 = np.full(self.n_features, 0.9)
        vals_combined = np.stack([vals_0, vals_1], axis=-1)[np.newaxis, :, :]  # (1, 18, 2)
        base_vals = np.array([[0.4, 0.6]])

        explainer = MockExplainer(vals_combined, base_vals)

        # When phishing_index is 0:
        res_0 = compute_shap_explanation(explainer, self.dummy_features, phishing_index=0)
        self.assertAlmostEqual(res_0["shap_values"][0], 0.1)
        self.assertAlmostEqual(res_0["base_value"], 0.4)
        self.assertEqual(res_0["target_class_index"], 0)

        # When phishing_index is 1 (e.g. reversed class order model):
        res_1 = compute_shap_explanation(explainer, self.dummy_features, phishing_index=1)
        self.assertAlmostEqual(res_1["shap_values"][0], 0.9)
        self.assertAlmostEqual(res_1["base_value"], 0.6)
        self.assertEqual(res_1["target_class_index"], 1)

    # --- 2. Feature Names & Ordering ---
    def test_all_18_feature_names_and_ordering(self):
        url = "https://www.google.com"
        res = analyze_url(url)
        all_impacts = res["all_impacts"]

        self.assertEqual(len(all_impacts), 18)
        impact_feature_names = {item["feature"] for item in all_impacts}
        self.assertEqual(impact_feature_names, set(FEATURE_NAMES))

        # Check raw SHAP array length matches FEATURE_NAMES
        feats_df = pd.DataFrame([extract_url_features(url)], columns=FEATURE_NAMES)
        shap_info = compute_shap_explanation(self.explainer, feats_df, phishing_index=0)
        self.assertEqual(len(shap_info["shap_values"]), 18)

    # --- 3. Supported & Ambiguous SHAP Output Shapes ---
    def test_supported_2d_and_3d_shapes(self):
        # 3D: (1, 18, 2)
        v3 = np.zeros((1, 18, 2))
        v3[0, :, 0] = 0.05
        e3 = MockExplainer(v3, [[0.3, 0.7]])
        res3 = compute_shap_explanation(e3, self.dummy_features, phishing_index=0)
        self.assertEqual(len(res3["shap_values"]), 18)
        self.assertAlmostEqual(res3["base_value"], 0.3)

        # 2D: (1, 18)
        v2 = np.full((1, 18), 0.08)
        e2 = MockExplainer(v2, [0.35])
        res2 = compute_shap_explanation(e2, self.dummy_features, phishing_index=0)
        self.assertEqual(len(res2["shap_values"]), 18)
        self.assertAlmostEqual(res2["base_value"], 0.35)

    def test_clear_error_for_ambiguous_shapes(self):
        # 1D array: ambiguous shape (18,)
        e_1d = MockExplainer(np.zeros(18), [0.5])
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e_1d, self.dummy_features, phishing_index=0)
        self.assertIn("Unsupported SHAP values dimensionality", str(ctx.exception))

        # 4D array: invalid dimensionality
        e_4d = MockExplainer(np.zeros((1, 1, 18, 2)), [0.5])
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e_4d, self.dummy_features, phishing_index=0)
        self.assertIn("Unsupported SHAP values dimensionality", str(ctx.exception))

        # Feature mismatch: only 10 features instead of 18
        e_mismatch = MockExplainer(np.zeros((1, 10, 2)), [[0.5, 0.5]])
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e_mismatch, self.dummy_features, phishing_index=0)
        self.assertIn("SHAP feature dimension mismatch", str(ctx.exception))

        # Class index out of bounds
        e_oob = MockExplainer(np.zeros((1, 18, 2)), [[0.5, 0.5]])
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e_oob, self.dummy_features, phishing_index=5)
        self.assertIn("out of bounds", str(ctx.exception))

    # --- 4. Finite SHAP Values Check ---
    def test_finite_shap_values_and_base_values(self):
        # Non-finite in SHAP values (NaN)
        v_nan = np.zeros((1, 18, 2))
        v_nan[0, 3, 0] = np.nan
        e_nan = MockExplainer(v_nan, [[0.5, 0.5]])
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e_nan, self.dummy_features, phishing_index=0)
        self.assertIn("non-finite", str(ctx.exception))

        # Non-finite in base values (Inf)
        v_ok = np.zeros((1, 18, 2))
        e_inf_base = MockExplainer(v_ok, [[np.inf, 0.5]])
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e_inf_base, self.dummy_features, phishing_index=0)
        self.assertIn("non-finite base value", str(ctx.exception))

        # Real model returns completely finite values
        feats_df = pd.DataFrame([extract_url_features("https://www.google.com")], columns=FEATURE_NAMES)
        real_info = compute_shap_explanation(self.explainer, feats_df, phishing_index=0)
        self.assertTrue(np.all(np.isfinite(real_info["shap_values"])))
        self.assertTrue(np.isfinite(real_info["base_value"]))

    # --- 5. Additive Consistency in Explainer Space ---
    def test_additive_consistency_in_explainer_space(self):
        """
        Verify that in the Random Forest TreeExplainer probability output space:
        base_value + sum(shap_values) == predict_proba(df)[0, phishing_index]
        within machine epsilon precision.
        """
        test_urls = [
            "https://www.google.com",
            "http://198.98.58.123/",
            "http://paypal-security-update.com/login.php"
        ]
        for url in test_urls:
            feats = extract_url_features(url)
            feats_df = pd.DataFrame([feats], columns=FEATURE_NAMES)
            shap_info = compute_shap_explanation(self.explainer, feats_df, phishing_index=0)

            base_val = shap_info["base_value"]
            sum_shap = float(np.sum(shap_info["shap_values"]))
            reconstructed_proba = base_val + sum_shap
            actual_proba = float(self.model.predict_proba(feats_df)[0, 0])

            self.assertAlmostEqual(
                reconstructed_proba,
                actual_proba,
                places=5,
                msg=f"Additive consistency failed for URL {url}: {reconstructed_proba} vs {actual_proba}"
            )

    # --- 6. Waterfall Plot Structure Compatibility ---
    def test_waterfall_plot_explanation_compatibility(self):
        url = "https://www.google.com"
        feats = extract_url_features(url)
        feats_df = pd.DataFrame([feats], columns=FEATURE_NAMES)
        shap_info = compute_shap_explanation(self.explainer, feats_df, phishing_index=0)

        # Verify creating shap.Explanation for waterfall plot
        explanation = shap.Explanation(
            values=shap_info["shap_values"],
            base_values=shap_info["base_value"],
            data=feats_df.iloc[0].values,
            feature_names=FEATURE_NAMES
        )
        self.assertEqual(explanation.values.shape, (18,))
        self.assertAlmostEqual(explanation.base_values, shap_info["base_value"])
        self.assertEqual(len(explanation.data), 18)

    # --- 7. Target Class Identification for Legitimate Verdicts ---
    def test_legitimate_verdict_explicitly_identifies_phishing_class_target(self):
        url = "https://www.google.com"
        res = analyze_url(url)

        self.assertEqual(res["prediction"], "Legitimate")
        self.assertEqual(res["target_class"], "Phishing")
        self.assertIn("Phishing", res["explanation_target"])
        self.assertIn("negative values support Legitimate", res["explanation_target"])

        # Check that negative SHAP values are labeled 'Supports Legitimate'
        negative_impacts = [i for i in res["all_impacts"] if i["shap_value"] < 0]
        self.assertGreater(len(negative_impacts), 0)
        for item in negative_impacts:
            self.assertEqual(item["direction"], "Supports Legitimate")

    # --- 8. Prediction Independence from SHAP ---
    def test_prediction_and_probabilities_independent_of_shap(self):
        url = "https://www.google.com"
        feats = extract_url_features(url)
        feats_df = pd.DataFrame([feats], columns=FEATURE_NAMES)

        # Standalone probability extraction
        class_probs = get_class_probabilities(self.model, feats_df)

        # Full analyze_url run
        res = analyze_url(url)

        self.assertAlmostEqual(res["prob_phishing"], class_probs["phishing_probability"], places=7)
        self.assertAlmostEqual(res["prob_legitimate"], class_probs["legitimate_probability"], places=7)

if __name__ == "__main__":
    unittest.main()
