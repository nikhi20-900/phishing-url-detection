"""
Tests for system error handling and defensive recovery (Phase 2 Task 2.2).

Verifies:
- Missing and corrupt model artifacts fail safely with descriptive errors.
- Unsupported URL schemes and malformed hostnames are rejected with clear messages.
- CLI and Streamlit errors are understandable and do not leak internal tracebacks.
- Explanation failure NEVER alters or inverts the model's prediction or probabilities.
"""

import unittest
import tempfile
import os
import sys
import subprocess
import joblib
import pandas as pd
import numpy as np

from src.features import extract_url_features, FEATURE_NAMES
from src.predict import (
    load_model_and_explainer,
    clear_model_cache,
    analyze_url,
    create_shap_plot
)

class MockFailingExplainer:
    """Explainer that intentionally raises an exception to simulate runtime SHAP failure."""
    def __init__(self, error_message="Simulated SHAP computation failure"):
        self.error_message = error_message

    def __call__(self, df):
        raise RuntimeError(self.error_message)

class TestErrorHandling(unittest.TestCase):
    """Test suite for error handling, defensive bounds, and explanation isolation."""

    def setUp(self):
        clear_model_cache()

    def tearDown(self):
        clear_model_cache()

    # --- 1. Missing and Corrupt Model Artifacts ---
    def test_missing_model_artifact_raises_file_not_found(self):
        with self.assertRaises((FileNotFoundError, ValueError)) as ctx:
            load_model_and_explainer("models/non_existent_model_artifact_999.joblib")
        self.assertIn("not found", str(ctx.exception).lower())

    def test_corrupted_empty_file_artifact_raises_clear_error(self):
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            tmp.write(b"")  # 0 bytes empty file
            tmp.flush()
            with self.assertRaises(ValueError) as ctx:
                load_model_and_explainer(tmp.name)
            self.assertIn("failed to load model artifact", str(ctx.exception).lower())

    def test_corrupted_truncated_bytes_artifact_raises_clear_error(self):
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            tmp.write(b"\x80\x04\x95\x12\x00\x00")  # Incomplete pickle byte stream
            tmp.flush()
            with self.assertRaises(ValueError) as ctx:
                load_model_and_explainer(tmp.name)
            self.assertIn("failed to load model artifact", str(ctx.exception).lower())

    def test_invalid_type_artifact_raises_clear_error(self):
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            joblib.dump(["just", "a", "list", "not", "model"], tmp.name)
            with self.assertRaises(ValueError) as ctx:
                load_model_and_explainer(tmp.name)
            self.assertIn("invalid model artifact format", str(ctx.exception).lower())

    # --- 2. Unsupported URL Schemes and Malformed Hostnames ---
    def test_unsupported_schemes_raise_clear_error(self):
        unsupported = [
            ("ftp://files.example.com/test", "ftp"),
            ("javascript:void(0)", "javascript"),
            ("mailto:security@company.com", "mailto"),
            ("data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==", "data"),
            ("file:///etc/passwd", "file"),
            ("ws://echo.websocket.events", "ws")
        ]
        for url, scheme in unsupported:
            with self.assertRaises(ValueError) as ctx:
                extract_url_features(url)
            err_msg = str(ctx.exception)
            self.assertIn("Unsupported scheme", err_msg)
            self.assertIn(scheme, err_msg)

    def test_malformed_hostnames_raise_clear_error(self):
        malformed = [
            "http://../path",
            "http://..double-dot.com",
            "http://trailing-dot-label..com",
            "http://invalid_port.com:70000",
            "http://negative_port.com:-80",
            "http://letters_in_port.com:80a",
            "http://space in host.com",
            "http://256.0.0.1"
        ]
        for url in malformed:
            with self.assertRaises(ValueError):
                extract_url_features(url)

    # --- 3. CLI Clean Error Formatting (No Leaked Tracebacks) ---
    def test_cli_clean_error_on_invalid_scheme(self):
        cmd = [sys.executable, os.path.join("src", "cli.py"), "ftp://test.org"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Error: Unsupported scheme 'ftp'", proc.stderr)
        self.assertNotIn("Traceback (most recent call last)", proc.stderr)

    def test_cli_clean_error_on_invalid_port(self):
        cmd = [sys.executable, os.path.join("src", "cli.py"), "http://example.com:99999"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Error: Invalid URL port", proc.stderr)
        self.assertNotIn("Traceback (most recent call last)", proc.stderr)

    # --- 4. Explanation Failure NEVER Alters Prediction ---
    def test_explanation_failure_does_not_alter_legitimate_prediction(self):
        url = "https://www.google.com"

        # Baseline analysis with normal explainer
        baseline = analyze_url(url)
        self.assertEqual(baseline["prediction"], "Legitimate")
        self.assertFalse(baseline["is_phishing"])
        self.assertTrue(baseline["explanation_verified"])
        expected_prob_phish = baseline["prob_phishing"]
        expected_prob_legit = baseline["prob_legitimate"]
        expected_conf = baseline["confidence"]

        # Run with mock failing explainer
        import src.predict
        real_load = src.predict.load_model_and_explainer
        try:
            model, _ = real_load()
            failing_explainer = MockFailingExplainer("TreeExplainer internal error")
            src.predict.load_model_and_explainer = lambda path=None: (model, failing_explainer)

            # Analyze should not raise, and should preserve all prediction outputs
            res = analyze_url(url)

            # Prediction must be identical
            self.assertEqual(res["prediction"], "Legitimate")
            self.assertFalse(res["is_phishing"])
            self.assertAlmostEqual(res["prob_phishing"], expected_prob_phish, places=7)
            self.assertAlmostEqual(res["prob_legitimate"], expected_prob_legit, places=7)
            self.assertAlmostEqual(res["confidence"], expected_conf, places=7)

            # Explanation should be marked unverified
            self.assertFalse(res["explanation_verified"])
            self.assertIn("TreeExplainer internal error", res["explanation_error"])
            self.assertEqual(res["top_5_features"], [])
            self.assertEqual(res["all_impacts"], [])

            # Plot generation should be safe and return placeholder without exception
            fig = create_shap_plot(res)
            self.assertIsNotNone(fig)

        finally:
            src.predict.load_model_and_explainer = real_load
            clear_model_cache()

    def test_explanation_failure_does_not_alter_phishing_prediction(self):
        url = "http://198.98.58.123/"

        # Baseline analysis with normal explainer
        baseline = analyze_url(url)
        self.assertEqual(baseline["prediction"], "Phishing")
        self.assertTrue(baseline["is_phishing"])
        self.assertTrue(baseline["explanation_verified"])
        expected_prob_phish = baseline["prob_phishing"]
        expected_prob_legit = baseline["prob_legitimate"]

        import src.predict
        real_load = src.predict.load_model_and_explainer
        try:
            model, _ = real_load()
            failing_explainer = MockFailingExplainer("Corrupted attribution matrix")
            src.predict.load_model_and_explainer = lambda path=None: (model, failing_explainer)

            res = analyze_url(url)

            # Prediction must remain PHISHING with identical probabilities
            self.assertEqual(res["prediction"], "Phishing")
            self.assertTrue(res["is_phishing"])
            self.assertAlmostEqual(res["prob_phishing"], expected_prob_phish, places=7)
            self.assertAlmostEqual(res["prob_legitimate"], expected_prob_legit, places=7)

            self.assertFalse(res["explanation_verified"])
            self.assertIn("Corrupted attribution matrix", res["explanation_error"])
            self.assertEqual(res["top_5_features"], [])
            self.assertEqual(res["all_impacts"], [])

        finally:
            src.predict.load_model_and_explainer = real_load
            clear_model_cache()

if __name__ == "__main__":
    unittest.main()
