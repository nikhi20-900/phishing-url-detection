"""
Unit tests for URL parsing, validation, and feature compatibility.
Verifies Step 1 requirements:
- Valid URL schemes (HTTP/HTTPS) and defaults
- Hostname, port, IPv4, IPv6 handling
- Credential detection and authority masking protection
- Explicit rejection of malformed IP addresses and domain labels
- IDNA / Punycode normalization
- Unsupported scheme rejection (including javascript:alert(1))
- Feature extraction compatibility and order preservation
"""

import unittest
from src.features import (
    parse_url,
    ParsedURL,
    is_ip_address,
    extract_url_features,
    extract_features_dataframe,
    FEATURE_NAMES
)

class TestURLParsing(unittest.TestCase):
    """Test suite for URL parsing and validation (Step 1)."""

    # --- 1. Valid URLs and Default Schemes ---
    def test_valid_https_url(self):
        res = parse_url("https://example.com")
        self.assertIsInstance(res, ParsedURL)
        self.assertEqual(res.scheme, "https")
        self.assertEqual(res.is_https, 1)
        self.assertEqual(res.hostname, "example.com")
        self.assertIsNone(res.port)
        self.assertEqual(res.is_ip, 0)
        self.assertEqual(res.has_credentials, 0)

    def test_default_http_scheme(self):
        res = parse_url("example.com")
        self.assertEqual(res.scheme, "http")
        self.assertEqual(res.is_https, 0)
        self.assertEqual(res.hostname, "example.com")

    def test_default_scheme_with_path(self):
        res = parse_url("example.com/test/page.html")
        self.assertEqual(res.scheme, "http")
        self.assertEqual(res.hostname, "example.com")
        self.assertEqual(res.path, "/test/page.html")

    def test_protocol_relative_url(self):
        res = parse_url("//example.com/api")
        self.assertEqual(res.scheme, "http")
        self.assertEqual(res.hostname, "example.com")

    # --- 2. Port Validation ---
    def test_valid_port(self):
        res = parse_url("http://example.com:8080/path")
        self.assertEqual(res.hostname, "example.com")
        self.assertEqual(res.port, 8080)

    def test_standard_https_port(self):
        res = parse_url("https://example.com:443/")
        self.assertEqual(res.port, 443)

    def test_non_numeric_port_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://example.com:abc")
        self.assertIn("Invalid URL port", str(ctx.exception))

    def test_out_of_range_port_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://example.com:70000")
        self.assertIn("Invalid URL port", str(ctx.exception))

    def test_negative_port_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://example.com:-1")
        self.assertIn("Invalid URL port", str(ctx.exception))

    # --- 3. Credentials & Authority Masking ---
    def test_credentials_detection_user_pass(self):
        res = parse_url("http://user:pass@example.com")
        self.assertEqual(res.hostname, "example.com")
        self.assertEqual(res.has_credentials, 1)
        self.assertEqual(res.username, "user")
        self.assertEqual(res.password, "pass")

    def test_credentials_detection_user_only(self):
        res = parse_url("https://user@example.com")
        self.assertEqual(res.hostname, "example.com")
        self.assertEqual(res.has_credentials, 1)

    def test_credentials_detection_empty_userinfo(self):
        res = parse_url("https://@example.com")
        self.assertEqual(res.hostname, "example.com")
        self.assertEqual(res.has_credentials, 1)

    def test_authority_masking_destination_host(self):
        # Critical security test: for trusted.com@attacker.com, hostname MUST be attacker.com
        res = parse_url("http://trusted.com@attacker.com/login")
        self.assertEqual(res.hostname, "attacker.com")
        self.assertEqual(res.has_credentials, 1)
        self.assertEqual(res.username, "trusted.com")

    def test_no_credentials(self):
        res = parse_url("https://example.com")
        self.assertEqual(res.has_credentials, 0)

    # --- 4. IPv4 and IPv6 Handling ---
    def test_valid_ipv4(self):
        res = parse_url("http://192.168.1.1")
        self.assertEqual(res.hostname, "192.168.1.1")
        self.assertEqual(res.is_ip, 1)
        self.assertEqual(res.ip_version, 4)

    def test_valid_ipv4_with_port(self):
        res = parse_url("http://192.168.1.1:8080")
        self.assertEqual(res.hostname, "192.168.1.1")
        self.assertEqual(res.port, 8080)
        self.assertEqual(res.is_ip, 1)

    def test_valid_ipv6_with_port(self):
        res = parse_url("http://[2001:db8::1]:8080/")
        self.assertEqual(res.hostname, "2001:db8::1")
        self.assertEqual(res.port, 8080)
        self.assertEqual(res.is_ip, 1)
        self.assertEqual(res.ip_version, 6)

    def test_valid_ipv6_loopback(self):
        res = parse_url("http://[::1]/")
        self.assertEqual(res.hostname, "::1")
        self.assertEqual(res.is_ip, 1)
        self.assertEqual(res.ip_version, 6)

    def test_malformed_ipv4_octets_out_of_range(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://256.256.256.256")
        self.assertIn("Malformed IPv4 address", str(ctx.exception))

    def test_malformed_ipv4_high_value(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://192.168.1.999")
        self.assertIn("Malformed IPv4 address", str(ctx.exception))

    def test_malformed_ipv4_five_octets(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://1.2.3.4.5")
        self.assertIn("Malformed IPv4 address", str(ctx.exception))

    def test_invalid_ipv6_syntax(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://[2001:db8:::1]")
        self.assertIn("Malformed URL", str(ctx.exception))

    # --- 5. Malformed Hostnames and Domain Labels ---
    def test_consecutive_dots_rejected(self):
        with self.assertRaises(ValueError):
            parse_url("http://example..com")

    def test_starting_hyphen_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://-example.com")
        self.assertIn("cannot start or end with a hyphen", str(ctx.exception))

    def test_trailing_hyphen_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://example-.com")
        self.assertIn("cannot start or end with a hyphen", str(ctx.exception))

    def test_empty_hostname_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://")
        self.assertIn("URL must contain a valid hostname", str(ctx.exception))

    def test_slash_only_hostname_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http:///path")
        self.assertIn("URL must contain a valid hostname", str(ctx.exception))

    def test_space_in_hostname_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("http://exam ple.com")
        self.assertIn("Invalid domain label", str(ctx.exception))

    # --- 6. Unsupported Schemes ---
    def test_ftp_scheme_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("ftp://example.com")
        self.assertIn("Unsupported scheme 'ftp'", str(ctx.exception))

    def test_javascript_scheme_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("javascript:alert(1)")
        self.assertIn("Unsupported scheme 'javascript'", str(ctx.exception))

    def test_data_scheme_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("data:text/html;base64,PHNjcmlwdD4=")
        self.assertIn("Unsupported scheme 'data'", str(ctx.exception))

    def test_mailto_scheme_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("mailto:phish@attacker.com")
        self.assertIn("Unsupported scheme 'mailto'", str(ctx.exception))

    # --- 7. Input Type and Empty Checks ---
    def test_empty_string_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("")
        self.assertEqual(str(ctx.exception), "URL cannot be empty")

    def test_whitespace_string_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url("   ")
        self.assertEqual(str(ctx.exception), "URL cannot be empty")

    def test_non_string_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url(12345)
        self.assertEqual(str(ctx.exception), "URL must be a string")

    def test_none_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_url(None)
        self.assertEqual(str(ctx.exception), "URL must be a string")

    # --- 8. Unicode and Punycode Normalization ---
    def test_unicode_domain_punycode_normalization(self):
        res = parse_url("https://münchen.de")
        self.assertEqual(res.hostname, "xn--mnchen-3ya.de")
        self.assertEqual(res.raw_hostname, "münchen.de")

    def test_punycode_input_preserved(self):
        res = parse_url("https://xn--mnchen-3ya.de")
        self.assertEqual(res.hostname, "xn--mnchen-3ya.de")

    # --- 9. Feature Compatibility and Consistency ---
    def test_extract_url_features_keys_and_count(self):
        features = extract_url_features("https://www.google.com")
        self.assertEqual(len(features), 18)
        self.assertEqual(list(features.keys()), FEATURE_NAMES)

    def test_extract_features_dataframe_shape_and_columns(self):
        df = extract_features_dataframe("https://www.google.com")
        self.assertEqual(df.shape, (1, 18))
        self.assertEqual(df.columns.tolist(), FEATURE_NAMES)

    def test_is_ip_address_helper(self):
        self.assertEqual(is_ip_address("192.168.1.1"), 1)
        self.assertEqual(is_ip_address("2001:db8::1"), 1)
        self.assertEqual(is_ip_address("example.com"), 0)

if __name__ == "__main__":
    unittest.main()
