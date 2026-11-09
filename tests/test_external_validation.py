"""
Unit tests for external validation and error diagnostic module (Phase 4).

Verifies:
- Robust multi-source feed parsing (OpenPhish, PhishTank, Curated Legitimate).
- Benchmark integrity checking (schema, nulls, invalid labels, label conflicts).
- URL preprocessing and exclusion tracking (malformed URLs, missing hostnames).
- Overlap calculation (exact URL, FQDN hostname, public-suffix registered domain).
- Single-class and two-class evaluation metric computation (graceful None for ROC-AUC on single-class slices).
- Reproducible deterministic execution without network access.
"""

import os
import json
import re
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.external_validation import (
    load_openphish_feed,
    load_phishtank_feed,
    load_curated_legitimate,
    validate_benchmark_integrity,
    preprocess_and_validate_urls,
    compute_overlap_metrics,
    evaluate_predictions,
    evaluate_dataset,
    build_external_benchmark,
    run_external_validation
)
from src.features import extract_url_features, FEATURE_NAMES


class TestExternalValidation(unittest.TestCase):
    """Test suite for external validation feed loading, validation, and metrics."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # 1. Feed Parsing & Ingestion Tests
    # -----------------------------------------------------------------------

    def test_load_openphish_feed(self):
        sample_feed = (
            "# OpenPhish Community Feed Snapshot\n"
            "\n"
            "https://evil-phish.com/login\n"
            "http://bank-update.xyz/verify.php\n"
            "# Trailing comment\n"
        )
        file_path = os.path.join(self.temp_dir.name, "openphish.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sample_feed)

        df = load_openphish_feed(file_path)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df["label"]), [0, 0])
        self.assertEqual(df.iloc[0]["url"], "https://evil-phish.com/login")
        self.assertEqual(df.iloc[0]["source"], "OpenPhish-Community")
        self.assertEqual(df.iloc[0]["source_id"], "OPENPHISH-0001")
        self.assertEqual(df.iloc[1]["source_id"], "OPENPHISH-0002")

    def test_load_phishtank_feed(self):
        sample_csv = (
            "phish_id,url,phish_detail_url,submission_time,verified,verification_time,online,target\n"
            "12345,https://phish1.com/login,http://detail/1,2026-10-30T10:00:00Z,yes,2026-10-30T10:05:00Z,yes,PayPal\n"
            "12346,https://phish2.com/unverified,http://detail/2,2026-10-30T10:01:00Z,no,2026-10-30T10:06:00Z,yes,Apple\n"
            "12347,https://phish3.com/verify,http://detail/3,2026-10-30T10:02:00Z,true,2026-10-30T10:07:00Z,yes,Amazon\n"
        )
        file_path = os.path.join(self.temp_dir.name, "phishtank.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sample_csv)

        df = load_phishtank_feed(file_path)
        # Should filter out unverified phish2 (verified=no)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df["label"]), [0, 0])
        self.assertEqual(df.iloc[0]["url"], "https://phish1.com/login")
        self.assertEqual(df.iloc[0]["source_id"], "PHISHTANK-12345")
        self.assertIn("PayPal", df.iloc[0]["notes"])
        self.assertEqual(df.iloc[1]["source_id"], "PHISHTANK-12347")

    def test_load_curated_legitimate(self):
        sample_csv = (
            "url,label,source,source_id,collection_timestamp,verification_method,notes\n"
            "https://www.mit.edu/,1,Curated-Authoritative,LEGIT-0001,2026-10-31T00:00:00Z,institutional,Official portal\n"
            "https://www.gov.uk/browse,1,Curated-Authoritative,LEGIT-0002,2026-10-31T00:00:00Z,institutional,Official gov\n"
        )
        file_path = os.path.join(self.temp_dir.name, "legit.csv")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(sample_csv)

        df = load_curated_legitimate(file_path)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df["label"]), [1, 1])

    # -----------------------------------------------------------------------
    # 2. Benchmark Integrity Checking Tests
    # -----------------------------------------------------------------------

    def test_validate_benchmark_integrity_valid(self):
        df = pd.DataFrame({
            "url": ["https://example.com", "https://phish.xyz"],
            "label": [1, 0],
            "source": ["Curated", "OpenPhish"],
            "source_id": ["C-1", "O-1"],
            "collection_timestamp": ["2026-10-31T00:00:00Z", "2026-10-31T00:00:00Z"],
            "verification_method": ["authoritative", "feed"],
            "notes": ["note 1", "note 2"]
        })
        valid, errors = validate_benchmark_integrity(df)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_validate_benchmark_integrity_missing_columns(self):
        df = pd.DataFrame({
            "url": ["https://example.com"],
            "label": [1]
        })
        valid, errors = validate_benchmark_integrity(df)
        self.assertFalse(valid)
        self.assertTrue(any("Missing required columns" in e for e in errors))

    def test_validate_benchmark_integrity_null_and_empty_urls(self):
        df = pd.DataFrame({
            "url": ["https://example.com", None, "   "],
            "label": [1, 0, 1],
            "source": ["A", "B", "C"],
            "source_id": ["1", "2", "3"],
            "collection_timestamp": ["T", "T", "T"],
            "verification_method": ["V", "V", "V"],
            "notes": ["N", "N", "N"]
        })
        valid, errors = validate_benchmark_integrity(df)
        self.assertFalse(valid)
        self.assertTrue(any("null URL entries" in e for e in errors))
        self.assertTrue(any("empty URL strings" in e for e in errors))

    def test_validate_benchmark_integrity_invalid_labels(self):
        df = pd.DataFrame({
            "url": ["https://example.com", "https://other.com"],
            "label": [1, 99],
            "source": ["A", "B"],
            "source_id": ["1", "2"],
            "collection_timestamp": ["T", "T"],
            "verification_method": ["V", "V"],
            "notes": ["N", "N"]
        })
        valid, errors = validate_benchmark_integrity(df)
        self.assertFalse(valid)
        self.assertTrue(any("Invalid label values" in e for e in errors))

    def test_validate_benchmark_integrity_label_conflicts(self):
        df = pd.DataFrame({
            "url": ["https://conflicting.com/page", "https://conflicting.com/page"],
            "label": [0, 1],
            "source": ["OpenPhish", "Curated"],
            "source_id": ["O-1", "C-1"],
            "collection_timestamp": ["T", "T"],
            "verification_method": ["V", "V"],
            "notes": ["N", "N"]
        })
        valid, errors = validate_benchmark_integrity(df)
        self.assertFalse(valid)
        self.assertTrue(any("conflicting class labels" in e for e in errors))

    # -----------------------------------------------------------------------
    # 3. URL Preprocessing & Exclusion Tracking
    # -----------------------------------------------------------------------

    def test_preprocess_and_validate_urls(self):
        df = pd.DataFrame({
            "url": [
                "https://valid.com/path",
                "http://another-valid.org:8080/index.html",
                "://missing-scheme-and-corrupt",
                "http://"  # Empty hostname
            ],
            "label": [1, 0, 0, 1]
        })
        clean_df, excluded_df = preprocess_and_validate_urls(df, url_col="url")
        self.assertEqual(len(clean_df), 2)
        self.assertEqual(len(excluded_df), 2)
        self.assertIn("exclusion_reason", excluded_df.columns)
        self.assertTrue(any("hostname" in r.lower() or "error" in r.lower() for r in excluded_df["exclusion_reason"]))

    # -----------------------------------------------------------------------
    # 4. Overlap Audit Tests
    # -----------------------------------------------------------------------

    def test_compute_overlap_metrics(self):
        # Create a mock reference dataset
        phi_mock = pd.DataFrame({
            "URL": [
                "https://ref-domain.com/page1",
                "https://shared-host.com/home",
                "http://overlap-url.com/exact"
            ],
            "label": [0, 1, 0]
        })
        phi_file = os.path.join(self.temp_dir.name, "mock_phi.csv")
        phi_mock.to_csv(phi_file, index=False)

        # External dataset with known overlaps
        external_mock = pd.DataFrame({
            "url": [
                "http://overlap-url.com/exact",       # Exact URL overlap + host + reg_domain
                "https://shared-host.com/different",  # Host + reg_domain overlap (not exact URL)
                "https://sub.ref-domain.com/login",   # Reg_domain overlap only
                "https://completely-new.org/about"    # Zero overlap
            ],
            "label": [0, 1, 0, 1]
        })

        results = compute_overlap_metrics(external_mock, phi_path=phi_file, url_col="url")
        total_stats = results["external_benchmark_total"]

        # Exact URL overlap: 1 / 4 (25.0%)
        self.assertEqual(total_stats["exact_url_overlap_count"], 1)
        self.assertAlmostEqual(total_stats["exact_url_overlap_rate"], 0.25)

        # Hostname overlap: 2 unique hosts overlap ('overlap-url.com', 'shared-host.com') out of 4
        self.assertEqual(total_stats["hostname_overlap_count"], 2)

        # Reg domain overlap: 3 unique reg domains overlap ('overlap-url.com', 'shared-host.com', 'ref-domain.com') out of 4
        self.assertEqual(total_stats["registered_domain_overlap_count"], 3)

    # -----------------------------------------------------------------------
    # 5. Prediction Evaluation Engine Tests
    # -----------------------------------------------------------------------

    def test_evaluate_predictions_two_classes(self):
        # 2 Phishing (0), 2 Legitimate (1)
        # Predictions: TP=2, FN=0, FP=1, TN=1
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1])
        y_prob_phish = np.array([0.9, 0.8, 0.7, 0.1])

        m = evaluate_predictions(y_true, y_pred, y_prob_phish)
        self.assertEqual(m["sample_size"], 4)
        self.assertEqual(m["accuracy"], 0.75)
        self.assertEqual(m["phishing_recall"], 1.0)
        self.assertEqual(m["phishing_precision"], 2.0 / 3.0)
        self.assertEqual(m["legitimate_recall"], 0.5)
        self.assertEqual(m["legitimate_precision"], 1.0)
        self.assertIsNotNone(m["roc_auc"])
        self.assertIsNotNone(m["pr_auc"])
        self.assertIsNotNone(m["log_loss"])
        self.assertEqual(m["confusion_matrix"]["true_phishing_tp"], 2)
        self.assertEqual(m["confusion_matrix"]["missed_phishing_fn"], 0)
        self.assertEqual(m["confusion_matrix"]["falsely_flagged_legit_fp"], 1)
        self.assertEqual(m["confusion_matrix"]["verified_legit_tn"], 1)

    def test_evaluate_predictions_single_class_phishing(self):
        # All phishing (0)
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 1])  # 2 caught, 1 missed
        y_prob_phish = np.array([0.95, 0.85, 0.20])

        m = evaluate_predictions(y_true, y_pred, y_prob_phish)
        self.assertEqual(m["sample_size"], 3)
        self.assertAlmostEqual(m["accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(m["phishing_recall"], 2.0 / 3.0)
        self.assertEqual(m["phishing_precision"], 1.0)
        # In single-class slice, ROC-AUC and PR-AUC must be None (not raise ValueError)
        self.assertIsNone(m["roc_auc"])
        self.assertIsNone(m["pr_auc"])
        self.assertIsNone(m["log_loss"])
        self.assertEqual(m["confusion_matrix"]["true_phishing_tp"], 2)
        self.assertEqual(m["confusion_matrix"]["missed_phishing_fn"], 1)

    def test_evaluate_predictions_single_class_legitimate(self):
        # All legitimate (1)
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 1])  # 2 false alarms, 1 true negative
        y_prob_phish = np.array([0.95, 0.85, 0.05])

        m = evaluate_predictions(y_true, y_pred, y_prob_phish)
        self.assertEqual(m["sample_size"], 3)
        self.assertAlmostEqual(m["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(m["legitimate_recall"], 1.0 / 3.0)
        self.assertEqual(m["legitimate_precision"], 1.0)
        self.assertIsNone(m["roc_auc"])
        self.assertIsNone(m["pr_auc"])
        self.assertIsNone(m["log_loss"])
        self.assertEqual(m["confusion_matrix"]["falsely_flagged_legit_fp"], 2)
        self.assertEqual(m["confusion_matrix"]["verified_legit_tn"], 1)


# ---------------------------------------------------------------------------
# 6. Duplicate / Provenance Integrity Tests (audit additions)
# ---------------------------------------------------------------------------

class TestBenchmarkIntegrityDuplicates(unittest.TestCase):
    """Exact duplicate URLs (even with identical labels) and duplicate source_ids
    must be flagged: duplicates double-count samples and silently distort metrics."""

    def test_duplicate_url_same_label_flagged(self):
        df = pd.DataFrame({
            "url": ["https://example.com", "https://example.com "],  # trailing whitespace variant
            "label": [0, 0],
            "source": ["OpenPhish", "OpenPhish"],
            "source_id": ["O-1", "O-2"],
            "collection_timestamp": ["T", "T"],
            "verification_method": ["V", "V"],
            "notes": ["N", "N"]
        })
        valid, errors = validate_benchmark_integrity(df)
        self.assertFalse(valid)
        self.assertTrue(any("duplicate URL entries" in e for e in errors))

    def test_unique_urls_still_valid(self):
        df = pd.DataFrame({
            "url": ["https://example.com", "https://other.com"],
            "label": [0, 1],
            "source": ["O", "L"],
            "source_id": ["O-1", "L-1"],
            "collection_timestamp": ["T", "T"],
            "verification_method": ["V", "V"],
            "notes": ["N", "N"]
        })
        valid, errors = validate_benchmark_integrity(df)
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_duplicate_source_id_flagged(self):
        df = pd.DataFrame({
            "url": ["https://a.com", "https://b.com"],
            "label": [0, 1],
            "source": ["O", "L"],
            "source_id": ["DUP-1", "DUP-1"],
            "collection_timestamp": ["T", "T"],
            "verification_method": ["V", "V"],
            "notes": ["N", "N"]
        })
        valid, errors = validate_benchmark_integrity(df)
        self.assertFalse(valid)
        self.assertTrue(any("duplicate source_id entries" in e for e in errors))


# ---------------------------------------------------------------------------
# 7. evaluate_dataset Guard & Threshold-Semantics Tests (audit additions)
# ---------------------------------------------------------------------------

class _StubModel:
    """Deterministic binary model stub aligned with classes_=[0, 1] ordering."""

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X_features):
        # Long URLs -> phishing (0), short URLs -> legitimate (1)
        # Column order follows classes_ = [0, 1]: col 0 = phishing, col 1 = legitimate.
        probs = []
        for url_len in X_features["URLLength"]:
            prob_phish = 0.9 if url_len > 60 else 0.1
            probs.append([prob_phish, 1.0 - prob_phish])
        return np.array(probs)


class TestEvaluateDatasetGuards(unittest.TestCase):
    """Label/schema/input validation for the evaluation engine."""

    def setUp(self):
        self.model = _StubModel()

    def test_empty_dataset_raises(self):
        df = pd.DataFrame({"url": [], "label": []})
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(df, self.model)
        self.assertIn("empty", str(ctx.exception))

    def test_missing_url_column_raises(self):
        df = pd.DataFrame({"label": [0, 1]})
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(df, self.model)
        self.assertIn("URL column", str(ctx.exception))

    def test_missing_label_column_raises(self):
        df = pd.DataFrame({"url": ["https://example.com"]})
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(df, self.model)
        self.assertIn("label column", str(ctx.exception))

    def test_invalid_labels_raise(self):
        df = pd.DataFrame({"url": ["https://a.com"], "label": [7]})
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(df, self.model)
        self.assertIn("invalid label values", str(ctx.exception).lower())

    def test_threshold_semantics_correct_positive_class(self):
        # prob_phish >= 0.5 must predict 0 (phishing); prob_phish < 0.5 predicts 1.
        phishing_url = f"https://{'x' * 60}.{'y' * 55}.example.org/login"  # > 60 chars, valid labels
        legit_url = "https://ab.example.org"                               # < 30 chars
        df = pd.DataFrame({"url": [phishing_url, legit_url], "label": [0, 1]})
        metrics, y_pred, y_prob_phish, _ = evaluate_dataset(df, self.model, threshold=0.5)
        self.assertEqual(list(y_pred), [0, 1])
        self.assertEqual(y_prob_phish[0] >= 0.5, True)
        self.assertEqual(metrics["confusion_matrix"]["true_phishing_tp"], 1)
        self.assertEqual(metrics["confusion_matrix"]["verified_legit_tn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["missed_phishing_fn"], 0)
        self.assertEqual(metrics["confusion_matrix"]["falsely_flagged_legit_fp"], 0)

    def test_confusion_matrix_zero_denominator_free_metrics(self):
        # Single legitimate record predicted legitimate: phishing precision is a
        # ZeroDivision case -> must be 0.0 via zero_division, not crash.
        df = pd.DataFrame({"url": ["https://ab.example.org"], "label": [1]})
        metrics, _, _, _ = evaluate_dataset(df, self.model)
        self.assertEqual(metrics["phishing_precision"], 0.0)
        self.assertEqual(metrics["phishing_recall"], 0.0)
        self.assertEqual(metrics["confusion_matrix"]["verified_legit_tn"], 1)


# ---------------------------------------------------------------------------
# 8. Benchmark Reconstruction Tests (audit additions)
# ---------------------------------------------------------------------------

class TestBuildExternalBenchmark(unittest.TestCase):
    """Deterministic consolidation of raw feeds with first-source deduplication."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_snapshot_files(self):
        openphish_path = os.path.join(self.temp_dir.name, "openphish.txt")
        with open(openphish_path, "w", encoding="utf-8") as f:
            f.write("https://shareddupe.com/full\n")
            f.write("https://only-openphish.com/a\n")

        phishtank_path = os.path.join(self.temp_dir.name, "phishtank.csv")
        pd.DataFrame({
            "phish_id": ["9001", "9002"],
            "url": ["https://shareddupe.com/full", "https://only-phishtank.com/b"],
            "verified": ["yes", "yes"]
        }).to_csv(phishtank_path, index=False)

        legit_path = os.path.join(self.temp_dir.name, "legit.csv")
        pd.DataFrame({
            "url": ["https://www.gov.example/portal", "https://www.edu.example/about?x=1"],
            "label": [1, 1],
            "source": ["Curated-Legit"] * 2,
            "source_id": ["LEGIT-1", "LEGIT-2"],
            "collection_timestamp": ["T1", "T2"],
            "verification_method": ["institutional"] * 2,
            "notes": ["gov portal", "edu page"]
        }).to_csv(legit_path, index=False)
        return openphish_path, phishtank_path, legit_path

    def test_consolidation_dedup_and_labels(self):
        openphish_path, phishtank_path, legit_path = self._write_snapshot_files()
        rebuilt = build_external_benchmark(openphish_path, phishtank_path, legit_path)

        # 2 OP + 2 PT + 2 legit = 6 raw records, 1 cross-feed duplicate -> 5 unique URLs
        self.assertEqual(len(rebuilt), 5)
        # Duplicate attributed to first-priority feed (OpenPhish)
        shared = rebuilt[rebuilt["url"] == "https://shareddupe.com/full"]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared.iloc[0]["source"], "OpenPhish-Community")
        # Labels preserved
        labels = dict(zip(rebuilt["url"], rebuilt["label"]))
        self.assertEqual(labels["https://only-phishtank.com/b"], 0)
        self.assertEqual(labels["https://www.gov.example/portal"], 1)
        self.assertEqual(labels["https://www.edu.example/about?x=1"], 1)
        # Provenance columns fully populated
        self.assertFalse(rebuilt[["source_id", "collection_timestamp"]].isna().any().any())

    def test_refuses_to_overwrite_existing_output(self):
        openphish_path, phishtank_path, legit_path = self._write_snapshot_files()
        out_path = os.path.join(self.temp_dir.name, "rebuilt.csv")
        build_external_benchmark(openphish_path, phishtank_path, legit_path, output_path=out_path)
        first_content = open(out_path).read()

        with self.assertRaises(ValueError):
            build_external_benchmark(openphish_path, phishtank_path, legit_path, output_path=out_path)

        # Existing output untouched
        self.assertEqual(open(out_path).read(), first_content)

        # Explicit overwrite permitted
        build_external_benchmark(openphish_path, phishtank_path, legit_path, output_path=out_path, overwrite=True)


# ---------------------------------------------------------------------------
# 9. Orchestration Output-Protection Tests (audit additions)
# ---------------------------------------------------------------------------

class TestRunExternalValidationOutputProtection(unittest.TestCase):
    """run_external_validation must never silently overwrite a prior artifact."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model = _StubModel()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _prepare_inputs(self):
        benchmark_path = os.path.join(self.temp_dir.name, "mini_benchmark.csv")
        phi_path = os.path.join(self.temp_dir.name, "mini_phi.csv")
        # Long phishing URLs -> stub model catches; short legit URLs -> clean TNs.
        pd.DataFrame({
            "url": [
                "https://" + "a" * 60 + ".verify-login.example.org/step",
                "https://" + "b" * 60 + ".confirm-id.example.org/wait",
                "https://ab.example.org",
                "https://cd.example.org"
            ],
            "label": [0, 0, 1, 1],
            "source": ["Feed", "Feed", "Curated", "Curated"],
            "source_id": ["F-1", "F-2", "C-1", "C-2"],
            "collection_timestamp": ["T"] * 4,
            "verification_method": ["V"] * 4,
            "notes": ["N"] * 4
        }).to_csv(benchmark_path, index=False)
        pd.DataFrame({
            "URL": ["https://ab.example.org", "https://cd.example.org"]
        }).to_csv(phi_path, index=False)
        return benchmark_path, phi_path

    def _run(self, output_json, overwrite=False):
        benchmark_path, phi_path = self._prepare_inputs()
        with patch("src.external_validation.load_model_and_explainer", return_value=(self.model, None)):
            return run_external_validation(
                benchmark_path=benchmark_path,
                model_path="ignored.joblib",
                phi_path=phi_path,
                output_json=output_json,
                run_phase_3_repro=False,
                overwrite=overwrite
            )

    def test_writes_requested_path_when_missing(self):
        output_json = os.path.join(self.temp_dir.name, "results.json")
        results = self._run(output_json)
        self.assertTrue(os.path.exists(output_json))
        with open(output_json) as f:
            stored = json.load(f)
        self.assertEqual(stored["combined_benchmark_metrics"]["sample_size"], 4)
        self.assertEqual(results["combined_benchmark_metrics"]["accuracy"], 1.0)
        # Phishing correctly caught, zero FPs by stub design
        cm = stored["combined_benchmark_metrics"]["confusion_matrix"]
        self.assertEqual(cm["true_phishing_tp"], 2)
        self.assertEqual(cm["falsely_flagged_legit_fp"], 0)

    def test_existing_output_preserved_with_timestamped_fallback(self):
        output_json = os.path.join(self.temp_dir.name, "results.json")
        self._run(output_json)
        original_content = open(output_json).read()

        # Second run WITHOUT overwrite must not replace the original
        self._run(output_json)
        self.assertEqual(open(output_json).read(), original_content)

        stem, ext = os.path.splitext(output_json)
        fallback_files = [
            fname for fname in os.listdir(self.temp_dir.name)
            if re.match(r"^results_\d{8}T\d{6}Z\.json$", fname)
        ]
        self.assertEqual(len(fallback_files), 1)

    def test_overwrite_flag_replaces_output(self):
        output_json = os.path.join(self.temp_dir.name, "results.json")
        self._run(output_json)
        self._run(output_json, overwrite=True)
        with open(output_json) as f:
            stored = json.load(f)
        self.assertEqual(stored["dataset_summary"]["total_validated_records"], 4)


if __name__ == "__main__":
    unittest.main()
