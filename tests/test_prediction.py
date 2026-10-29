"""
Unit tests for prediction correctness and probability alignment (Step 3).

Verifies:
- Label-to-verdict mapping: 0 = Phishing, 1 = Legitimate.
- Dynamic probability alignment based on model.classes_ (not hardcoded column indices).
- Handling of reversed class order in model.classes_ ([1, 0]).
- Probabilities summing to approximately 1.0.
- Decision threshold behavior (at, above, and below threshold; custom thresholds).
- Strict rejection of invalid/unexpected model classes.
- Real model artifact verification.
- Consistency between interfaces (CLI and Streamlit both consume analyze_url).
"""

import unittest
import numpy as np
import pandas as pd
from src.predict import (
    get_class_probabilities,
    analyze_url,
    load_model_and_explainer
)
from src.features import FEATURE_NAMES

class MockModel:
    """Mock model with configurable classes_ and predict_proba for deterministic testing."""
    def __init__(self, classes, proba_return):
        self.classes_ = np.array(classes)
        self.proba_return = np.array(proba_return)

    def predict_proba(self, X):
        return self.proba_return

    def predict(self, X):
        # argmax on proba
        max_idx = np.argmax(self.proba_return[0])
        return np.array([self.classes_[max_idx]])

class TestPredictionCorrectness(unittest.TestCase):
    """Test suite ensuring predictions and probabilities are strictly aligned."""

    def setUp(self):
        # Dummy single-row feature vector
        self.dummy_features = pd.DataFrame([{f: 0 for f in FEATURE_NAMES}])

    # --- 1. Dynamic Class Probability Extraction & Column Alignment ---
    def test_probability_alignment_standard_classes(self):
        # Model with standard classes [0, 1]
        # Column 0 is Phishing (0.85), Column 1 is Legitimate (0.15)
        mock = MockModel(classes=[0, 1], proba_return=[[0.85, 0.15]])
        res = get_class_probabilities(mock, self.dummy_features)

        self.assertAlmostEqual(res["phishing_probability"], 0.85)
        self.assertAlmostEqual(res["legitimate_probability"], 0.15)
        self.assertEqual(res["phishing_index"], 0)
        self.assertEqual(res["legitimate_index"], 1)

    def test_probability_alignment_reversed_classes(self):
        # Critical test: model with reversed classes [1, 0]
        # Column 0 is Legitimate (0.20), Column 1 is Phishing (0.80)
        mock = MockModel(classes=[1, 0], proba_return=[[0.20, 0.80]])
        res = get_class_probabilities(mock, self.dummy_features)

        # Must correctly identify index 1 as Phishing and index 0 as Legitimate
        self.assertAlmostEqual(res["phishing_probability"], 0.80)
        self.assertAlmostEqual(res["legitimate_probability"], 0.20)
        self.assertEqual(res["phishing_index"], 1)
        self.assertEqual(res["legitimate_index"], 0)

    def test_probabilities_sum_to_one(self):
        mock = MockModel(classes=[0, 1], proba_return=[[0.643, 0.357]])
        res = get_class_probabilities(mock, self.dummy_features)
        total = res["phishing_probability"] + res["legitimate_probability"]
        self.assertAlmostEqual(total, 1.0, places=5)

    # --- 2. Invalid Model Classes Rejection ---
    def test_invalid_classes_non_zero_one(self):
        # Classes are [1, 2] instead of {0, 1}
        mock = MockModel(classes=[1, 2], proba_return=[[0.5, 0.5]])
        with self.assertRaises(ValueError) as ctx:
            get_class_probabilities(mock, self.dummy_features)
        self.assertIn("Invalid model class configuration", str(ctx.exception))

    def test_invalid_classes_strings(self):
        # Classes are ['phishing', 'legitimate']
        mock = MockModel(classes=["phishing", "legitimate"], proba_return=[[0.7, 0.3]])
        with self.assertRaises(ValueError) as ctx:
            get_class_probabilities(mock, self.dummy_features)
        self.assertIn("Invalid model class configuration", str(ctx.exception))

    def test_invalid_classes_multiclass(self):
        # Classes are [0, 1, 2] (multi-class)
        mock = MockModel(classes=[0, 1, 2], proba_return=[[0.3, 0.4, 0.3]])
        with self.assertRaises(ValueError) as ctx:
            get_class_probabilities(mock, self.dummy_features)
        self.assertIn("Invalid model class configuration", str(ctx.exception))

    def test_model_missing_classes_attribute(self):
        class NoClassesModel:
            def predict_proba(self, X):
                return [[0.5, 0.5]]
        with self.assertRaises(ValueError) as ctx:
            get_class_probabilities(NoClassesModel(), self.dummy_features)
        self.assertIn("classes_", str(ctx.exception))

    # --- 3. Decision Threshold Tests ---
    def test_threshold_default_boundary(self):
        # Test exact boundary at default threshold 0.5
        # If prob_phishing == 0.5 -> Phishing (flagged)
        url = "https://www.google.com"
        # We test threshold logic through analyze_url parameter
        res_at = analyze_url(url, threshold=0.0) # threshold 0 -> always phish
        self.assertTrue(res_at["is_phishing"])
        self.assertEqual(res_at["prediction"], "Phishing")

        res_high = analyze_url(url, threshold=1.0) # threshold 1 -> only phish if 100%
        # Google should be legitimate at threshold 1.0
        self.assertFalse(res_high["is_phishing"])
        self.assertEqual(res_high["prediction"], "Legitimate")

    def test_threshold_confidence_assignment(self):
        # For a Legitimate prediction, confidence should match prob_legitimate
        res = analyze_url("https://www.google.com", threshold=0.5)
        self.assertFalse(res["is_phishing"])
        self.assertEqual(res["prediction"], "Legitimate")
        self.assertEqual(res["confidence"], res["prob_legitimate"])

        # For an IP Phishing URL, confidence should match prob_phishing
        res_phish = analyze_url("http://198.98.58.123/", threshold=0.5)
        self.assertTrue(res_phish["is_phishing"])
        self.assertEqual(res_phish["prediction"], "Phishing")
        self.assertEqual(res_phish["confidence"], res_phish["prob_phishing"])

    # --- 4. Real Model Artifact Verification ---
    def test_real_model_classes_and_mapping(self):
        model, _ = load_model_and_explainer()
        self.assertTrue(hasattr(model, "classes_"))
        self.assertEqual(list(model.classes_), [0, 1],
                         "Trained model artifact must have classes [0, 1]")

    def test_real_model_legitimate_prediction(self):
        res = analyze_url("https://www.google.com")
        self.assertEqual(res["prediction"], "Legitimate")
        self.assertFalse(res["is_phishing"])
        self.assertGreater(res["prob_legitimate"], 0.90)
        self.assertLess(res["prob_phishing"], 0.10)
        self.assertAlmostEqual(res["prob_phishing"] + res["prob_legitimate"], 1.0, places=5)

    def test_real_model_phishing_prediction(self):
        res = analyze_url("http://198.98.58.123/")
        self.assertEqual(res["prediction"], "Phishing")
        self.assertTrue(res["is_phishing"])
        self.assertGreater(res["prob_phishing"], 0.90)
        self.assertLess(res["prob_legitimate"], 0.10)
        self.assertAlmostEqual(res["prob_phishing"] + res["prob_legitimate"], 1.0, places=5)

    # --- 5. Interface Consistency (CLI and Web Agreement) ---
    def test_interface_pipeline_shared_result(self):
        # Both CLI (src/cli.py) and Streamlit (app/main.py) call analyze_url(url)
        # Verify that multiple consecutive invocations produce deterministic identical results
        url = "https://www.southbankmosaics.com"
        run1 = analyze_url(url)
        run2 = analyze_url(url)

        self.assertEqual(run1["prediction"], run2["prediction"])
        self.assertEqual(run1["is_phishing"], run2["is_phishing"])
        self.assertAlmostEqual(run1["prob_phishing"], run2["prob_phishing"], places=7)
        self.assertAlmostEqual(run1["prob_legitimate"], run2["prob_legitimate"], places=7)

if __name__ == "__main__":
    unittest.main()
