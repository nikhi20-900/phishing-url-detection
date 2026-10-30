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
    """Mock shap.Explanation container with configurable values, base_values, and class metadata."""
    def __init__(self, values, base_values, output_names=None, target_class=None, target_class_index=None):
        self.values = values if isinstance(values, list) else np.array(values, dtype=float)
        self.base_values = np.array(base_values, dtype=float)
        self.output_names = output_names
        self.target_class = target_class
        self.target_class_index = target_class_index

class MockExplainer:
    def __init__(self, values, base_values, output_names=None, target_class=None, target_class_index=None):
        self._values = values
        self._base_values = base_values
        self._output_names = output_names
        self._target_class = target_class
        self._target_class_index = target_class_index

    def __call__(self, df):
        return MockExplanationResult(
            self._values,
            self._base_values,
            output_names=self._output_names,
            target_class=self._target_class,
            target_class_index=self._target_class_index
        )

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
    def test_supported_3d_shape_targeting_phishing(self):
        # 3D: (1, 18, 2) targeting phishing_index=0
        v3 = np.zeros((1, 18, 2))
        v3[0, :, 0] = 0.05
        # Expected phishing probability: base (0.3) + sum (18 * 0.05 = 0.90) = 1.20
        # Let's make sum realistic: base=0.10, 18*0.01 = 0.18 -> prob = 0.28
        v3_real = np.zeros((1, 18, 2))
        v3_real[0, :, 0] = 0.01
        e3 = MockExplainer(v3_real, [[0.10, 0.90]])
        res3 = compute_shap_explanation(
            e3, self.dummy_features, phishing_index=0, expected_phishing_prob=0.28
        )
        self.assertEqual(len(res3["shap_values"]), 18)
        self.assertAlmostEqual(res3["base_value"], 0.10)
        self.assertTrue(res3["verified"])

    def test_supported_2d_shape_with_phishing_metadata(self):
        # 2D: (1, 18) with explicit phishing metadata (output_names=['Phishing'])
        v2 = np.full((1, 18), 0.02)
        e2 = MockExplainer(v2, [0.35], output_names=["Phishing"])
        res2 = compute_shap_explanation(e2, self.dummy_features, phishing_index=0)
        self.assertEqual(len(res2["shap_values"]), 18)
        self.assertAlmostEqual(res2["base_value"], 0.35)
        self.assertAlmostEqual(res2["shap_values"][0], 0.02)
        self.assertTrue(res2["verified"])

    def test_supported_2d_shape_verified_via_probability_matching(self):
        # 2D: (1, 18) without metadata, but verified against model's expected phishing probability
        v2 = np.zeros((1, 18))
        v2[0, 0] = 0.40
        # base = 0.30, sum = 0.40 -> prob_phishing = 0.70
        e2 = MockExplainer(v2, [0.30])
        res2 = compute_shap_explanation(
            e2, self.dummy_features, phishing_index=0, expected_phishing_prob=0.70
        )
        self.assertEqual(len(res2["shap_values"]), 18)
        self.assertAlmostEqual(res2["base_value"], 0.30)
        self.assertAlmostEqual(res2["shap_values"][0], 0.40)
        self.assertTrue(res2["verified"])

    def test_supported_2d_shape_explaining_legitimate_inverted_to_phishing(self):
        # 2D: (1, 18) explains Legitimate (e.g. output_names=['Legitimate'])
        # base_legit = 0.70, sum_legit = +0.18 -> prob_legit = 0.88 (prob_phish = 0.12)
        v2 = np.full((1, 18), 0.01)
        e2 = MockExplainer(v2, [0.70], output_names=["Legitimate"])
        res = compute_shap_explanation(
            e2, self.dummy_features, phishing_index=0, expected_phishing_prob=0.12
        )
        # Should be inverted to target phishing: base = 1 - 0.70 = 0.30, shap = -0.01
        self.assertAlmostEqual(res["base_value"], 0.30)
        self.assertAlmostEqual(res["shap_values"][0], -0.01)
        # Additive sum: 0.30 + 18 * (-0.01) = 0.12 == prob_phishing
        reconstructed = res["base_value"] + float(np.sum(res["shap_values"]))
        self.assertAlmostEqual(reconstructed, 0.12)
        self.assertTrue(res["verified"])

    def test_unverified_ambiguous_2d_shape_rejected(self):
        # 2D: (1, 18) with NO metadata and NO expected probability must be rejected
        v2 = np.full((1, 18), 0.05)
        e2 = MockExplainer(v2, [0.35])
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e2, self.dummy_features, phishing_index=0)
        self.assertIn("ambiguous 2d shap output", str(ctx.exception).lower())

    def test_mismatching_2d_shape_rejected(self):
        # 2D output claim does not match expected model probability
        v2 = np.zeros((1, 18))
        e2 = MockExplainer(v2, [0.20], output_names=["Phishing"])
        # base (0.20) + sum (0) = 0.20 != expected (0.85)
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e2, self.dummy_features, phishing_index=0, expected_phishing_prob=0.85)
        self.assertIn("does not match expected model output", str(ctx.exception))

    def test_mismatching_3d_shape_rejected(self):
        # 3D output slice sum + base does not match expected model probability
        v3 = np.zeros((1, 18, 2))
        e3 = MockExplainer(v3, [[0.20, 0.80]])
        # class 0: base 0.20 + sum 0.0 != expected 0.95
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e3, self.dummy_features, phishing_index=0, expected_phishing_prob=0.95)
        self.assertIn("does not match expected model output", str(ctx.exception))

    def test_ambiguous_2d_shape_near_half_probability_rejected_without_metadata(self):
        # Near 0.5 probability (e.g. 0.50), prob matching is symmetric/ambiguous without metadata
        v2 = np.zeros((1, 18))
        e2 = MockExplainer(v2, [0.50])
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(e2, self.dummy_features, phishing_index=0, expected_phishing_prob=0.50)
        self.assertIn("Ambiguous 2D SHAP output", str(ctx.exception))

    def test_legacy_list_of_arrays_supported(self):
        # Older SHAP shap_values returned [array_0, array_1]
        v_0 = np.full((1, 18), 0.01)
        v_1 = np.full((1, 18), -0.01)
        e_list = MockExplainer([v_0, v_1], [0.40, 0.60])
        res = compute_shap_explanation(e_list, self.dummy_features, phishing_index=0)
        self.assertEqual(len(res["shap_values"]), 18)
        self.assertAlmostEqual(res["base_value"], 0.40)
        self.assertAlmostEqual(res["shap_values"][0], 0.01)

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

    # --- 5. Additive Consistency in Explainer Space (Numerical Tolerance) ---
    def test_additive_consistency_legitimate_predictions(self):
        """
        Verify that for legitimate URL predictions, SHAP attributions satisfy
        additive consistency with an appropriate numerical floating-point tolerance:
        base_value + sum(shap_values) ≈ prob_phishing (within atol=1e-4, rtol=1e-4).
        """
        legitimate_urls = [
            "https://www.google.com",
            "https://www.wikipedia.org",
            "https://www.python.org"
        ]
        for url in legitimate_urls:
            res = analyze_url(url)
            self.assertEqual(res["prediction"], "Legitimate")
            self.assertLess(res["prob_phishing"], 0.20)

            base_val = res["base_value_phishing"]
            sum_shap = float(np.sum([i["shap_value"] for i in res["all_impacts"]]))
            reconstructed_proba = base_val + sum_shap
            actual_proba = res["prob_phishing"]

            # Verify with appropriate numerical tolerance, not literal equality
            self.assertTrue(
                np.isclose(reconstructed_proba, actual_proba, atol=1e-4, rtol=1e-4),
                msg=f"Legitimate additive consistency failed for {url}: reconstructed {reconstructed_proba:.6f} vs actual {actual_proba:.6f}"
            )

    def test_additive_consistency_phishing_predictions(self):
        """
        Verify that for phishing URL predictions, SHAP attributions satisfy
        additive consistency with an appropriate numerical floating-point tolerance:
        base_value + sum(shap_values) ≈ prob_phishing (within atol=1e-4, rtol=1e-4).
        """
        phishing_urls = [
            "http://198.98.58.123/",
            "http://paypal-security-update.com/login.php",
            "http://account-verification-alert.com/signin"
        ]
        for url in phishing_urls:
            res = analyze_url(url)
            self.assertEqual(res["prediction"], "Phishing")
            self.assertGreater(res["prob_phishing"], 0.80)

            base_val = res["base_value_phishing"]
            sum_shap = float(np.sum([i["shap_value"] for i in res["all_impacts"]]))
            reconstructed_proba = base_val + sum_shap
            actual_proba = res["prob_phishing"]

            # Verify with appropriate numerical tolerance, not literal equality
            self.assertTrue(
                np.isclose(reconstructed_proba, actual_proba, atol=1e-4, rtol=1e-4),
                msg=f"Phishing additive consistency failed for {url}: reconstructed {reconstructed_proba:.6f} vs actual {actual_proba:.6f}"
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

    # --- 9. Explanation Verification Flag ---
    def test_analysis_result_includes_explanation_verified(self):
        url = "https://www.google.com"
        res = analyze_url(url)
        self.assertIn("explanation_verified", res)
        self.assertTrue(res["explanation_verified"])
        self.assertTrue(res["base_value_phishing"] > 0)
        self.assertEqual(res["target_class"], "Phishing")

if __name__ == "__main__":
    unittest.main()
