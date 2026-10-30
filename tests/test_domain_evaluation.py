"""
Unit tests for domain-separated evaluation logic (Phase 3 Task 3.3).

Verifies:
- Public-suffix-aware registered domain extraction across standard, multi-part ccTLDs,
  IPv4/IPv6, userinfo credentials, and custom ports.
- Strict zero-domain-overlap enforcement (AssertionError on violation).
- Expected Calibration Error (ECE) computation.
"""

import unittest
import numpy as np
import pandas as pd

from src.evaluate_domain_split import (
    extract_registered_domain,
    verify_zero_domain_overlap,
    compute_ece,
    find_balanced_group_split
)

class TestDomainSeparatedEvaluation(unittest.TestCase):
    """Test suite for domain extraction, overlap enforcement, and calibration."""

    # --- 1. Registered Domain Extraction Tests ---
    def test_standard_domains_and_subdomains(self):
        cases = [
            ("https://login.example.com/account", "example.com"),
            ("https://shop.example.com/products", "example.com"),
            ("https://example.com/about", "example.com"),
            ("https://sub.domain.deep.example.com/path", "example.com"),
            ("http://www.google.com/", "google.com"),
            ("https://amazon.com", "amazon.com")
        ]
        for url, expected in cases:
            self.assertEqual(extract_registered_domain(url), expected, f"Failed on {url}")

    def test_multi_part_cctld_domains(self):
        cases = [
            ("https://www.voicefmradio.co.uk/stream", "voicefmradio.co.uk"),
            ("https://sub.ooty.ind.in/path", "ooty.ind.in"),
            ("https://portal.gov.uk/login", "portal.gov.uk"),
            ("https://shop.co.jp/item", "shop.co.jp"),
            ("https://university.ac.nz/info", "university.ac.nz"),
            ("https://bank.com.au/netbank", "bank.com.au"),
            ("https://company.co.za/home", "company.co.za"),
            ("https://empresa.com.br/contato", "empresa.com.br")
        ]
        for url, expected in cases:
            self.assertEqual(extract_registered_domain(url), expected, f"Failed on {url}")

    def test_ip_address_hostnames(self):
        cases = [
            ("http://198.98.58.123/login", "198.98.58.123"),
            ("http://198.98.58.123:8080/path", "198.98.58.123"),
            ("https://127.0.0.1/", "127.0.0.1"),
            ("http://[::1]:8080/test", "::1"),
            ("http://[2001:db8::1]:8443/api", "2001:db8::1")
        ]
        for url, expected in cases:
            self.assertEqual(extract_registered_domain(url), expected, f"Failed on {url}")

    def test_credentials_and_ports(self):
        cases = [
            ("https://user:password@target.org:8443/path", "target.org"),
            ("https://admin@sub.secure-portal.net/", "secure-portal.net"),
            ("http://target.com:8080", "target.com")
        ]
        for url, expected in cases:
            self.assertEqual(extract_registered_domain(url), expected, f"Failed on {url}")

    def test_schemeless_and_single_label_hosts(self):
        self.assertEqual(extract_registered_domain("www.example.com"), "example.com")
        self.assertEqual(extract_registered_domain("http://localhost:8000"), "localhost")

    def test_invalid_url_raises_value_error(self):
        with self.assertRaises(ValueError):
            extract_registered_domain("ftp://unsupported.com")
        with self.assertRaises(ValueError):
            extract_registered_domain("")

    # --- 2. Programmatic Zero Domain Overlap Enforcement ---
    def test_zero_domain_overlap_passes_when_disjoint(self):
        train_domains = {"example.com", "google.com", "github.com"}
        test_domains = {"apple.com", "microsoft.com", "cloudflare.com"}
        # Must not raise
        verify_zero_domain_overlap(train_domains, test_domains)

    def test_zero_domain_overlap_fails_when_overlap_exists(self):
        train_domains = {"example.com", "shared-domain.com", "github.com"}
        test_domains = {"apple.com", "shared-domain.com", "cloudflare.com"}
        with self.assertRaises(AssertionError) as ctx:
            verify_zero_domain_overlap(train_domains, test_domains)
        self.assertIn("Domain overlap violation", str(ctx.exception))
        self.assertIn("shared-domain.com", str(ctx.exception))

    # --- 3. Calibration ECE Computation ---
    def test_compute_ece_perfect_calibration(self):
        # 100 predictions with 1.0 confidence for true positives, 0.0 for true negatives
        probs = np.array([1.0] * 50 + [0.0] * 50)
        y_true = np.array([1] * 50 + [0] * 50)
        ece = compute_ece(probs, y_true, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=5)

    def test_compute_ece_imperfect_calibration(self):
        # Predicted 0.9 for all 100 samples, but only 50 are positive
        probs = np.array([0.9] * 100)
        y_true = np.array([1] * 50 + [0] * 50)
        # Expected accuracy = 0.5, mean confidence = 0.9 -> gap = 0.4
        ece = compute_ece(probs, y_true, n_bins=10)
        self.assertAlmostEqual(ece, 0.4, places=5)

    # --- 4. Balanced Group Split Candidate Selection ---
    def test_find_balanced_group_split(self):
        # Dummy dataset with 10 domains and 50 URLs
        data = []
        for i in range(10):
            dom = f"domain{i}.com"
            label = 0 if i < 4 else 1  # 40% phishing
            for j in range(5):
                data.append({"URL": f"https://{dom}/page{j}", "reg_domain": dom, "label": label})
        dummy_df = pd.DataFrame(data)

        seed, train_idx, test_idx, ratio, phish_pct = find_balanced_group_split(
            dummy_df, target_test_ratio=0.20, candidate_seeds=5
        )

        train_doms = set(dummy_df.iloc[train_idx]["reg_domain"])
        test_doms = set(dummy_df.iloc[test_idx]["reg_domain"])

        # Enforce zero overlap on dummy split
        verify_zero_domain_overlap(train_doms, test_doms)
        self.assertAlmostEqual(ratio, 0.20, delta=0.1)

if __name__ == "__main__":
    unittest.main()
