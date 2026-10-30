"""
Regression and consistency tests for the 18 extracted URL features (Step 2).

Verifies:
- Feature names, count, order, and types (14 int, 4 float).
- Model artifact alignment with FEATURE_NAMES.
- Domain-based URLs vs. IP-address URLs (TLDLength, NoOfSubDomain).
- Trailing slash preservation in NoOfOtherSpecialCharsInURL.
- Inclusion of query delimiters (=, ?, &) in SpacialCharRatioInURL.
- 3-decimal rounding for all 4 ratio features.
- Representative dataset verification on both Legitimate and Phishing samples.
"""

import unittest
import os
import re
import joblib
import pandas as pd
from src.features import (
    extract_url_features,
    extract_features_dataframe,
    FEATURE_NAMES
)

class TestFeatureConsistency(unittest.TestCase):
    """Test suite ensuring feature extraction strictly adheres to audited conventions."""

    # --- 1. Schema, Names, Order, and Data Types ---
    def test_feature_vector_schema(self):
        url = "https://www.example.com/login?user=admin&token=xyz"
        feat_dict = extract_url_features(url)

        # 18 features exactly
        self.assertEqual(len(feat_dict), 18)
        self.assertEqual(list(feat_dict.keys()), FEATURE_NAMES)

        # Expected integer features
        int_features = [
            "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
            "HasObfuscation", "NoOfObfuscatedChar", "NoOfLettersInURL", "NoOfDegitsInURL",
            "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
            "NoOfOtherSpecialCharsInURL", "IsHTTPS"
        ]
        for f in int_features:
            self.assertIsInstance(feat_dict[f], int, f"Feature {f} should be int")

        # Expected float ratio features
        float_features = [
            "ObfuscationRatio", "LetterRatioInURL", "DegitRatioInURL", "SpacialCharRatioInURL"
        ]
        for f in float_features:
            self.assertIsInstance(feat_dict[f], float, f"Feature {f} should be float")

    def test_model_artifact_feature_order_alignment(self):
        model_path = "models/random_forest_model.joblib"
        if os.path.exists(model_path):
            artifact = joblib.load(model_path)
            model_features = artifact["feature_names"]
            self.assertEqual(model_features, FEATURE_NAMES,
                             "Trained model features must match FEATURE_NAMES in exact order")

    # --- 2. IP Address vs Domain Features ---
    def test_ip_address_tld_and_subdomains(self):
        # In PhiUSIIL, IP addresses are dot-split consistently
        url = "http://198.98.58.123/"
        feats = extract_url_features(url)

        self.assertEqual(feats["IsDomainIP"], 1)
        self.assertEqual(feats["DomainLength"], 13)
        # 198.98.58.123 -> parts=['198', '98', '58', '123']
        # TLD is '123' -> len 3
        self.assertEqual(feats["TLDLength"], 3)
        # Subdomains: len(parts) - 2 = 4 - 2 = 2
        self.assertEqual(feats["NoOfSubDomain"], 2)

    def test_standard_domain_tld_and_subdomains(self):
        url = "https://www.google.com"
        feats = extract_url_features(url)

        self.assertEqual(feats["IsDomainIP"], 0)
        self.assertEqual(feats["DomainLength"], 14)  # www.google.com
        self.assertEqual(feats["TLDLength"], 3)       # com
        self.assertEqual(feats["NoOfSubDomain"], 1)   # www (parts=3, sub=1)

    def test_multi_level_subdomain(self):
        url = "https://sub.portal.domain.co.uk/page"
        feats = extract_url_features(url)

        # parts: ['sub', 'portal', 'domain', 'co', 'uk'] -> 5 parts
        # subdomains: 5 - 2 = 3
        self.assertEqual(feats["TLDLength"], 2)       # uk
        self.assertEqual(feats["NoOfSubDomain"], 3)

    # --- 3. URL Length and Trailing Slash Handling ---
    def test_url_length_no_artificial_https_deduction(self):
        http_url = "http://example.com/test"
        https_url = "https://example.com/test"

        http_feats = extract_url_features(http_url)
        https_feats = extract_url_features(https_url)

        # Length must be the actual length of the input URL
        self.assertEqual(http_feats["URLLength"], len(http_url))
        self.assertEqual(https_feats["URLLength"], len(https_url))

    def test_trailing_slash_counted_in_special_chars(self):
        url_with_slash = "https://sabadonoappmagazine.com/"
        feats = extract_url_features(url_with_slash)

        # s_no_www = "sabadonoappmagazine.com/"
        # Non-alphanumeric special characters in s_no_www: '.' and '/' -> 2
        self.assertEqual(feats["NoOfOtherSpecialCharsInURL"], 2)

    # --- 4. Special Characters and Ratio Delimiters ---
    def test_spacial_char_ratio_includes_query_delimiters(self):
        # URL has '=', '?', '&' and other special characters
        url = "https://example.com/search?q=test&lang=en"
        feats = extract_url_features(url)

        # u_full = "https://example.com/search?q=test&lang=en" (len = 41)
        # s_no_www = "example.com/search?q=test&lang=en"
        # other_special: '.' and '/' -> 2
        # equals: 2 ('q=', 'lang=')
        # qmark: 1 ('?')
        # ampersand: 1 ('&')
        # total_special = 2 + 2 + 1 + 1 = 6
        # ratio = round(6 / 41, 3) = 0.146
        self.assertEqual(feats["NoOfOtherSpecialCharsInURL"], 2)
        self.assertEqual(feats["NoOfEqualsInURL"], 2)
        self.assertEqual(feats["NoOfQMarkInURL"], 1)
        self.assertEqual(feats["NoOfAmpersandInURL"], 1)

        expected_ratio = round((2 + 2 + 1 + 1) / len(url), 3)
        self.assertEqual(feats["SpacialCharRatioInURL"], expected_ratio)

    # --- 5. Ratio Rounding Precision (3 Decimal Places) ---
    def test_all_ratios_bounded_and_rounded_to_three_decimals(self):
        url = "https://www.sample-test-domain.org/path/resource?param1=value1&param2=value2"
        feats = extract_url_features(url)

        for ratio_key in ["ObfuscationRatio", "LetterRatioInURL", "DegitRatioInURL", "SpacialCharRatioInURL"]:
            val = feats[ratio_key]
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)
            # Check maximum 3 decimal places
            str_val = f"{val:.6f}".rstrip("0")
            if "." in str_val:
                decimals = len(str_val.split(".")[1])
                self.assertLessEqual(decimals, 3, f"{ratio_key} exceeded 3 decimal places: {val}")

    # --- 6. Obfuscation (%XX) Escape Patterns ---
    def test_obfuscation_hex_patterns(self):
        url = "http://example.com/path%20with%2Fescapes"
        feats = extract_url_features(url)

        # %20 and %2F -> 2 obfuscated occurrences -> 6 characters
        self.assertEqual(feats["HasObfuscation"], 1)
        self.assertEqual(feats["NoOfObfuscatedChar"], 6)
        expected_ratio = round(6 / len(url), 3)
        self.assertEqual(feats["ObfuscationRatio"], expected_ratio)

    def test_no_obfuscation(self):
        url = "https://example.com/clean/path"
        feats = extract_url_features(url)

        self.assertEqual(feats["HasObfuscation"], 0)
        self.assertEqual(feats["NoOfObfuscatedChar"], 0)
        self.assertEqual(feats["ObfuscationRatio"], 0.0)

    # --- 7. Representative Dataset Checks (PhiUSIIL Verification) ---
    def test_dataset_ip_row_representation(self):
        # Row 125 from PhiUSIIL dataset: http://198.98.58.123/ (Phishing, label=0)
        url = "http://198.98.58.123/"
        feats = extract_url_features(url)

        self.assertEqual(feats["IsDomainIP"], 1)
        self.assertEqual(feats["DomainLength"], 13)
        self.assertEqual(feats["TLDLength"], 3)
        self.assertEqual(feats["NoOfSubDomain"], 2)
        self.assertEqual(feats["NoOfDegitsInURL"], 10)
        self.assertEqual(feats["IsHTTPS"], 0)

    def test_dataset_phishing_row_representation(self):
        # Row 34 from PhiUSIIL dataset: https://hidok4f8zl.firebaseapp.com/ (Phishing, label=0)
        url = "https://hidok4f8zl.firebaseapp.com/"
        feats = extract_url_features(url)

        self.assertEqual(feats["URLLength"], len(url))
        self.assertEqual(feats["IsDomainIP"], 0)
        self.assertEqual(feats["TLDLength"], 3)
        self.assertEqual(feats["NoOfSubDomain"], 1)
        self.assertEqual(feats["NoOfDegitsInURL"], 2)  # 4 and 8
        self.assertEqual(feats["IsHTTPS"], 1)

    def test_dataset_legitimate_row_representation(self):
        # Row 0 from PhiUSIIL dataset: https://www.southbankmosaics.com
        url = "https://www.southbankmosaics.com"
        feats = extract_url_features(url)

        self.assertEqual(feats["DomainLength"], 24)
        self.assertEqual(feats["TLDLength"], 3)
        self.assertEqual(feats["NoOfSubDomain"], 1)
        self.assertEqual(feats["NoOfDegitsInURL"], 0)
        self.assertEqual(feats["IsHTTPS"], 1)

    # --- 8. Feature Input Validation (Missing, Extra, Non-finite, Scrambled) ---
    def test_feature_input_missing_columns_rejected(self):
        from src.predict import get_class_probabilities, load_model_and_explainer
        model, _ = load_model_and_explainer()
        # 17 features (missing IsHTTPS)
        cols_17 = FEATURE_NAMES[:-1]
        df_17 = pd.DataFrame([{c: 1.0 for c in cols_17}], columns=cols_17)
        with self.assertRaises(ValueError) as ctx:
            get_class_probabilities(model, df_17)
        self.assertIn("Feature columns do not match expected FEATURE_NAMES", str(ctx.exception))

    def test_feature_input_extra_columns_rejected(self):
        from src.predict import get_class_probabilities, load_model_and_explainer
        model, _ = load_model_and_explainer()
        cols_19 = FEATURE_NAMES + ["ExtraUnexpectedFeature"]
        df_19 = pd.DataFrame([{c: 1.0 for c in cols_19}], columns=cols_19)
        with self.assertRaises(ValueError) as ctx:
            get_class_probabilities(model, df_19)
        self.assertIn("Feature columns do not match expected FEATURE_NAMES", str(ctx.exception))

    def test_feature_input_nan_values_rejected(self):
        from src.predict import get_class_probabilities, load_model_and_explainer
        import numpy as np
        model, _ = load_model_and_explainer()
        df_nan = pd.DataFrame([{c: 1.0 for c in FEATURE_NAMES}], columns=FEATURE_NAMES)
        df_nan.iloc[0, 2] = np.nan
        with self.assertRaises(ValueError) as ctx:
            get_class_probabilities(model, df_nan)
        self.assertIn("contains missing (NaN) values", str(ctx.exception))

    def test_feature_input_inf_values_rejected(self):
        from src.predict import get_class_probabilities, load_model_and_explainer
        import numpy as np
        model, _ = load_model_and_explainer()
        df_inf = pd.DataFrame([{c: 1.0 for c in FEATURE_NAMES}], columns=FEATURE_NAMES)
        df_inf.iloc[0, 3] = np.inf
        with self.assertRaises(ValueError) as ctx:
            get_class_probabilities(model, df_inf)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_shap_input_non_finite_rejected(self):
        from src.predict import compute_shap_explanation, load_model_and_explainer
        import numpy as np
        _, explainer = load_model_and_explainer()
        df_nan = pd.DataFrame([{c: 1.0 for c in FEATURE_NAMES}], columns=FEATURE_NAMES)
        df_nan.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError) as ctx:
            compute_shap_explanation(explainer, df_nan, phishing_index=0)
        self.assertIn("missing (NaN) values", str(ctx.exception))

    # --- 9. Unusual and Valid URL Feature Extraction ---
    def test_unusual_urls_feature_extraction(self):
        # IPv6 URL
        url_v6 = "http://[2001:db8::1]:8080/path"
        f_v6 = extract_url_features(url_v6)
        self.assertEqual(f_v6["IsDomainIP"], 1)
        self.assertEqual(f_v6["IsHTTPS"], 0)

        # Credentials authority URL
        url_cred = "https://user:password@target.org:8443/login"
        f_cred = extract_url_features(url_cred)
        self.assertEqual(f_cred["IsHTTPS"], 1)
        self.assertEqual(f_cred["IsDomainIP"], 0)

        # Internationalized domain (IDNA Unicode)
        url_idna = "https://münchen.de/path"
        f_idna = extract_url_features(url_idna)
        self.assertEqual(f_idna["IsHTTPS"], 1)
        self.assertGreater(f_idna["DomainLength"], 0)

        # Heavy percent-encoding
        url_pct = "http://example.com/%2e%2e/%2f%20%7e%21"
        f_pct = extract_url_features(url_pct)
        self.assertEqual(f_pct["HasObfuscation"], 1)
        self.assertGreater(f_pct["NoOfObfuscatedChar"], 0)
        self.assertGreater(f_pct["ObfuscationRatio"], 0.0)

    # --- 10. Malformed URLs Rejected Safely ---
    def test_malformed_urls_raise_clear_error(self):
        malformed = [
            "ftp://files.example.com",
            "javascript:alert(1)",
            "mailto:admin@site.com",
            "data:text/html,test",
            "http://example.com:99999",
            "http://example.com:-5",
            "http://..invalid..domain",
            "http://invalid-.com",
            "http://-invalid.com",
            "http://256.100.1.1/",
            "",
            "   "
        ]
        for bad_url in malformed:
            with self.assertRaises(ValueError):
                extract_url_features(bad_url)

if __name__ == "__main__":
    unittest.main()
