"""
Unit and integration tests for CLI error handling and execution (Task 2).

Verifies:
- Clean error messages printed to stderr on input-validation errors.
- Non-zero exit code (exit 1) on invalid inputs.
- No unhandled Python tracebacks leaked on expected validation errors.
- Successful execution, exit code 0, and correct output format on valid URLs.
- Tests malformed URLs, unsupported schemes, invalid ports, and empty strings.
"""

import unittest
import subprocess
import sys
import os

class TestCLIExecution(unittest.TestCase):
    """Test suite for CLI execution and error handling."""

    def run_cli(self, url: str):
        """Helper to invoke src/cli.py in a subprocess."""
        cmd = [sys.executable, os.path.join("src", "cli.py"), url]
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

    # --- 1. Successful CLI Executions ---
    def test_cli_legitimate_url(self):
        res = self.run_cli("https://www.google.com")
        self.assertEqual(res.returncode, 0)
        self.assertIn("VERDICT     : ✅ LEGITIMATE", res.stdout)
        self.assertIn("RISK SCORES :", res.stdout)
        self.assertIn("TOP 5 INFLUENCING FEATURES", res.stdout)
        self.assertEqual(res.stderr.strip(), "")

    def test_cli_phishing_url(self):
        res = self.run_cli("http://198.98.58.123/")
        self.assertEqual(res.returncode, 0)
        self.assertIn("VERDICT     : 🚨 PHISHING", res.stdout)
        self.assertIn("RISK SCORES :", res.stdout)
        self.assertEqual(res.stderr.strip(), "")

    # --- 2. Unsupported Schemes ---
    def test_cli_unsupported_ftp_scheme(self):
        res = self.run_cli("ftp://example.com/files")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error:", res.stderr)
        self.assertIn("Unsupported scheme 'ftp'", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_cli_unsupported_javascript_scheme(self):
        res = self.run_cli("javascript:alert(1)")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error:", res.stderr)
        self.assertIn("Unsupported scheme 'javascript'", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    # --- 3. Malformed URLs & Ports ---
    def test_cli_empty_url(self):
        res = self.run_cli("")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error:", res.stderr)
        self.assertIn("URL cannot be empty", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_cli_invalid_port_out_of_range(self):
        res = self.run_cli("http://example.com:99999")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error:", res.stderr)
        self.assertIn("Port out of range", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_cli_consecutive_dots(self):
        res = self.run_cli("http://bad..domain.com")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error:", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

    def test_cli_malformed_ipv4(self):
        res = self.run_cli("http://999.999.999.999/")
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error:", res.stderr)
        self.assertNotIn("Traceback", res.stderr)

if __name__ == "__main__":
    unittest.main()
