"""
Adversarial and Unit Tests for Phase 7 Milestone 1.2.1 (M1.2.1):
Retrieval Security, DNS Rebinding Prevention, Mixed DNS Record Rejection,
Redirect Attack Mitigation, TLS Failure Rejection, and Review Ledger Tamper-Evidence.
"""

import json
import socket
import ssl
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experiments.phase7.validation.audit_corpus_m1_2_1 import (
    REPO_ROOT,
    FROZEN_MODEL_PATH,
    EXPECTED_FROZEN_MODEL_HASH,
    PHASE4_BENCHMARK_PATH,
    compute_sha256,
    compute_gini,
)
from experiments.phase7.validation.secure_http_client import (
    CrossDomainRedirectBlockedError,
    DirectIPHTTPConnection,
    DirectIPHTTPSConnection,
    SecurityViolationError,
    execute_secure_retrieval,
    is_disallowed_ip,
    resolve_and_validate_hostname,
)
from experiments.phase7.validation.review_workflow_m1_2_1 import (
    compute_record_hash,
    validate_review_ledger,
)


# --- 1. Invariant & Milestone Preservation Tests ---

def test_frozen_production_model_remains_untouched():
    """Verify that models/random_forest_model.joblib is completely untouched and frozen."""
    actual_hash = compute_sha256(FROZEN_MODEL_PATH)
    assert actual_hash == EXPECTED_FROZEN_MODEL_HASH, (
        f"Production model was modified! Expected {EXPECTED_FROZEN_MODEL_HASH}, got {actual_hash}"
    )


def test_no_candidate_model_training_in_m1_2_1():
    """Verify strict milestone boundaries: M1.2.1 must NOT train or output candidate models."""
    phase7_model_dir = REPO_ROOT / "models" / "phase7_candidates"
    if phase7_model_dir.exists():
        model_files = list(phase7_model_dir.glob("*.joblib"))
        allowed = {"rf_augmented_e4.joblib"}
        unexpected = [f.name for f in model_files if f.name not in allowed]
        assert len(unexpected) == 0, f"Unauthorized candidate models found: {unexpected}"


def test_historical_milestones_preserved_unmodified():
    """Verify that prior M1, M1.1, and M1.2 deliverables were preserved untouched."""
    m1_corpus = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus.csv"
    m1_1_corpus = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"
    m1_2_corpus = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2.csv"

    assert m1_corpus.exists()
    assert m1_1_corpus.exists()
    assert m1_2_corpus.exists()

    assert len(pd.read_csv(m1_corpus)) == 1182
    assert len(pd.read_csv(m1_1_corpus)) == 1176
    assert len(pd.read_csv(m1_2_corpus)) == 334


# --- 2. Adversarial Threat Model: DNS Rebinding & Direct Socket Connection ---

def test_anti_dns_rebinding_direct_ip_connection():
    """Verify DirectIPHTTPSConnection connects socket directly to validated IP without DNS resolution."""
    conn = DirectIPHTTPSConnection("example.org", "93.184.216.34", port=443)
    with patch("socket.create_connection") as mock_create_conn:
        mock_sock = MagicMock()
        mock_create_conn.return_value = mock_sock

        with patch.object(conn, "_context") as mock_ctx:
            mock_ssock = MagicMock()
            mock_ctx.wrap_socket.return_value = mock_ssock
            mock_ssock.version.return_value = "TLSv1.3"
            mock_ssock.cipher.return_value = ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)
            mock_ssock.getpeercert.return_value = {"subject": [[("commonName", "example.org")]]}

            conn.connect()

            # Must connect directly to the validated IP tuple
            mock_create_conn.assert_called_once_with(("93.184.216.34", 443), conn.timeout)
            # Must wrap socket with the SNI host for certificate verification
            mock_ctx.wrap_socket.assert_called_once_with(mock_sock, server_hostname="example.org")
            assert conn.tls_verified is True
            assert conn.tls_version == "TLSv1.3"


def test_mixed_public_private_dns_records_rejected():
    """Adversarial Test: Hostname resolving to mixed public and private IP records must be rejected."""
    # Mock getaddrinfo returning both a public IP and an RFC 1918 private IP
    fake_addrinfo = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
    ]

    with patch("socket.getaddrinfo", return_value=fake_addrinfo):
        with pytest.raises(SecurityViolationError, match="DNS rebinding / SSRF hazard"):
            resolve_and_validate_hostname("attacker-controlled-rebinding.com")


def test_private_and_loopback_ips_blocked():
    """Verify that private, loopback, link-local, and carrier-grade NAT IPs are blocked."""
    assert is_disallowed_ip("127.0.0.1") is True
    assert is_disallowed_ip("10.0.0.1") is True
    assert is_disallowed_ip("172.16.0.5") is True
    assert is_disallowed_ip("192.168.1.1") is True
    assert is_disallowed_ip("169.254.169.254") is True
    assert is_disallowed_ip("100.64.0.1") is True
    assert is_disallowed_ip("::1") is True
    assert is_disallowed_ip("fe80::1") is True
    assert is_disallowed_ip("8.8.8.8") is False
    assert is_disallowed_ip("1.1.1.1") is False


# --- 3. Adversarial Threat Model: Redirect Attacks ---

def test_cross_domain_redirect_blocked_before_connection():
    """Adversarial Test: Cross-domain redirect is detected and blocked before connecting to destination."""
    with patch("experiments.phase7.validation.secure_http_client.resolve_and_validate_hostname") as mock_resolve:
        mock_resolve.return_value = (["93.184.216.34"], 1.5)

        with patch("experiments.phase7.validation.secure_http_client.DirectIPHTTPSConnection") as mock_conn_cls:
            mock_conn = MagicMock()
            mock_conn_cls.return_value = mock_conn
            mock_conn.tls_verified = True
            mock_conn.tls_version = "TLSv1.3"
            mock_conn.tls_cipher = "TLS_AES_256_GCM_SHA384"
            mock_conn.tls_cert = {"subject": [[("commonName", "example.com")]]}

            mock_resp = MagicMock()
            mock_resp.status = 302
            mock_resp.getheader.side_effect = lambda h: "https://evil-attacker.example.org/tracker" if h == "Location" else None
            mock_conn.getresponse.return_value = mock_resp

            obs = execute_secure_retrieval(
                candidate_id="cand_redirect_test",
                target_url="https://example.com/login?redirect=out",
            )

            # In M1.2.1, this redirect must be flagged and connection to evil-attacker must NOT occur
            assert obs["is_cross_domain_redirect"] is True
            assert "CROSS_DOMAIN_REDIRECT_BLOCKED" in (obs["observation_error"] or "")
            assert obs["is_accessible"] is False
            assert obs["final_url"] == "https://example.com/login?redirect=out"

            # Connection should only have been established ONCE (to example.com, NEVER to evil-attacker)
            assert mock_conn_cls.call_count == 1


def test_redirect_to_private_ip_aborted():
    """Adversarial Test: Redirect targeting an RFC 1918 IP literal must be aborted."""
    obs = execute_secure_retrieval(
        candidate_id="cand_priv_test",
        target_url="http://192.168.1.1/admin/network/settings",
    )
    assert obs["is_private_ip"] is True
    assert obs["is_accessible"] is False
    assert "SECURITY_VIOLATION" in obs["observation_error"]


# --- 4. Adversarial Threat Model: TLS Failures ---

def test_tls_failure_prevents_corpus_admission():
    """Adversarial Test: HTTPS candidate with TLS certificate failure cannot be admitted to corpus."""
    from experiments.phase7.validation.revalidate_corpus_m1_2_1 import revalidate_corpus_m1_2_1
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        fake_obs = tmp_dir / "obs.jsonl"
        fake_prim = tmp_dir / "prim.jsonl"
        fake_sen = tmp_dir / "sen.jsonl"
        fake_rec = tmp_dir / "rec.jsonl"
        fake_corp = tmp_dir / "corp.csv"
        stub_q = tmp_dir / "q.csv"

        # Observation with tls_verified = False
        with open(fake_obs, "w") as f:
            f.write(json.dumps({
                "candidate_id": "cand_bad_tls",
                "target_url": "https://self-signed.invalid/test",
                "retrieval_timestamp_utc": "2026-11-13T00:00:00Z",
                "resolved_ip": "198.51.100.1",
                "connected_ip": "198.51.100.1",
                "http_status_code": 200,
                "tls_verified": False,  # FAILED TLS!
                "tls_protocol": None,
                "tls_cipher": None,
                "tls_cert_subject_cn": None,
                "redirect_count": 0,
                "final_url": "https://self-signed.invalid/test",
                "is_cross_domain_redirect": False,
                "content_length_bytes": 100,
                "response_sha256": "abcdef1234567890",
                "html_title": "Test",
                "is_accessible": False,
                "is_private_ip": False,
                "observation_error": "TLS_VERIFICATION_FAILED: Self-signed certificate",
                "is_synthetic": False,
            }) + "\n")

        with open(fake_prim, "w") as f:
            f.write(json.dumps({
                "candidate_id": "cand_bad_tls",
                "url": "https://self-signed.invalid/test",
                "registered_domain": "self-signed.invalid",
                "reviewer_id": "auditor-systems-01",
                "review_timestamp_utc": "2026-11-13T00:00:00Z",
                "decision": "approved",
                "rationale": "Mistaken approval of failed TLS",
            }) + "\n")

        with open(fake_sen, "w") as f:
            pass

        pd.DataFrame([{
            "candidate_id": "cand_bad_tls",
            "url": "https://self-signed.invalid/test",
            "url_exact": "https://self-signed.invalid/test",
            "registered_domain": "self-signed.invalid",
            "source_name": "Test",
            "collection_timestamp": "2026-11-13T00:00:00Z",
            "has_risk_flags": False,
            "risk_flags": None,
        }]).to_csv(stub_q, index=False)

        corp_df = revalidate_corpus_m1_2_1(stub_q, fake_obs, fake_prim, fake_sen, fake_rec, fake_corp)
        # Must NOT be admitted!
        assert len(corp_df) == 0

        # Check rejection reason in validation record
        with open(fake_rec, "r") as f:
            rec = json.loads(f.readline())
            assert rec["admissible_for_training"] is False
            assert "UNVERIFIED_TLS" in rec["rejection_reason"]


# --- 5. Adversarial Threat Model: Fabricated Reviewer Records ---

def test_fabricated_reviewer_record_rejected():
    """Adversarial Test: Fabricated or tampered reviewer record must be rejected by review ledger validator."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        prim_file = tmp_dir / "prim.jsonl"
        sen_file = tmp_dir / "sen.jsonl"
        q_file = tmp_dir / "q.csv"

        # Valid candidate queue
        pd.DataFrame([{
            "candidate_id": "cand_1",
            "url": "https://example.com/test",
            "registered_domain": "example.com",
            "has_risk_flags": False,
        }]).to_csv(q_file, index=False)

        # 1. Unknown reviewer ID
        with open(prim_file, "w") as f:
            f.write(json.dumps({
                "candidate_id": "cand_1",
                "url": "https://example.com/test",
                "reviewer_id": "rogue-agent-99",  # UNKNOWN REVIEWER!
                "review_timestamp_utc": "2026-11-13T00:00:00Z",
                "decision": "approved",
                "rationale": "Valid looking documentation page",
                "record_hash": "dummy",
            }) + "\n")

        with open(sen_file, "w") as f:
            pass

        with pytest.raises(ValueError, match="Unauthenticated reviewer_id"):
            validate_review_ledger(prim_file, sen_file, q_file)

        # 2. Tampered record hash
        with open(prim_file, "w") as f:
            f.write(json.dumps({
                "candidate_id": "cand_1",
                "url": "https://example.com/test",
                "reviewer_id": "auditor-systems-01",
                "review_timestamp_utc": "2026-11-13T00:00:00Z",
                "decision": "approved",
                "rationale": "Valid looking documentation page",
                "record_hash": "tampered_hash_value_1234567890",  # TAMPERED HASH!
            }) + "\n")

        with pytest.raises(ValueError, match="Tamper detected"):
            validate_review_ledger(prim_file, sen_file, q_file)


# --- 6. Substantiated Corpus v1.2.1 Audit & Governance Tests ---

def test_substantiated_corpus_v1_2_1_invariants():
    """Verify substantiated corpus v1.2.1 properties and exclusion of all non-viable URLs."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2_1.csv"
    assert corpus_path.exists()
    df = pd.read_csv(corpus_path)

    # 347 verified URLs across 90 domains
    assert len(df) == 347
    assert df["registered_domain"].nunique() == 90

    # 100% verified TLS for corpus admissions
    assert (df["tls_verified"] == True).all()

    # Zero exact collisions with Phase 4 external benchmark
    p4_df = pd.read_csv(PHASE4_BENCHMARK_PATH)
    p4_urls = set(p4_df["url"].dropna().str.strip())
    v1_2_1_urls = set(df["url"].dropna().str.strip())
    assert len(v1_2_1_urls.intersection(p4_urls)) == 0

    # Dynamic/interactive wiki endpoints strictly excluded
    excluded_patterns = [
        "Special:RandomInCategory",
        "Special:Search?search=cryptographic",
        "Special:NewItem",
        "Special:RecentChanges",
        "veaction=edit",
        "ubuntu.com/blog/tag/open-source/?page=1&sort=date",
    ]
    for pattern in excluded_patterns:
        matches = df[df["url"].str.contains(pattern, regex=False)]
        assert len(matches) == 0, f"Excluded pattern '{pattern}' leaked into corpus v1.2.1"


def test_domain_concentration_limits_in_v1_2_1():
    """Verify anti-concentration limits in corpus v1.2.1."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2_1.csv"
    df = pd.read_csv(corpus_path)

    counts = df["registered_domain"].value_counts()
    total = len(df)

    top1_share = counts.iloc[0] / total
    top5_share = counts.iloc[:5].sum() / total
    top10_share = counts.iloc[:10].sum() / total

    assert top1_share <= 0.05, f"Top 1 domain share exceeded 5%: {top1_share:.2%}"
    assert top5_share <= 0.15, f"Top 5 domain share exceeded 15%: {top5_share:.2%}"
    assert top10_share <= 0.27, f"Top 10 domain share exceeded 27%: {top10_share:.2%}"

    gini = compute_gini(counts.values.tolist())
    assert gini < 0.40, f"Gini coefficient exceeded 0.40: {gini:.4f}"

    shares = counts.values / total
    eff_domains = 1.0 / (shares ** 2).sum()
    assert eff_domains >= 50.0, f"Effective domain count below 50: {eff_domains:.1f}"


def test_priority_strata_coverage_in_v1_2_1():
    """Verify priority strata coverage in substantiated corpus v1.2.1."""
    corpus_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2_1.csv"
    df = pd.read_csv(corpus_path)

    query_count = df["has_query"].sum() if "has_query" in df.columns else df["url"].str.contains(r"\?", regex=True).sum()
    assert query_count >= 150, f"Expected >= 150 query URLs, got {query_count}"

    multilevel_count = df["is_multilevel_path"].sum() if "is_multilevel_path" in df.columns else (df["url"].str.count("/") >= 4).sum()
    assert multilevel_count >= 250, f"Expected >= 250 multi-level paths, got {multilevel_count}"

    trailing_slash_count = df["has_trailing_slash"].sum() if "has_trailing_slash" in df.columns else df["url"].str.contains(r"/(?:\?|$)", regex=True).sum()
    assert trailing_slash_count >= 100, f"Expected >= 100 trailing-slash URLs, got {trailing_slash_count}"


def test_metric_exactness_across_report_and_manifest():
    """Verify exact numerical agreement between audit report, manifest, and datasets."""
    report_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_2_1_corpus_audit.md"
    manifest_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m1_2_1_validation_manifest.json"
    ds_manifest_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "manifests" / "validation_manifest_v1_2_1.json"

    assert report_path.exists()
    assert manifest_path.exists()
    assert ds_manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["milestone"] == "M1.2.1"
    assert manifest["status"] == "APPROVED"
    assert manifest["exit_criteria_met"] is True
    assert manifest["security_controls"]["anti_dns_rebinding_direct_ip_connection"] is True
    assert manifest["summary"]["substantiated_training_eligible_corpus_v1_2_1"] == 347
    assert manifest["summary"]["total_registered_domains"] == 90
    assert manifest["summary"]["exact_overlap_with_phase4_benchmark"] == 0

    # Verify all artifact hashes
    for key, item in manifest["artifact_hashes"].items():
        artifact_path = REPO_ROOT / item["path"]
        assert artifact_path.exists(), f"Artifact {item['path']} missing"
        actual_hash = compute_sha256(artifact_path)
        assert actual_hash == item["sha256"], f"Hash mismatch for {key}: expected {item['sha256']}, got {actual_hash}"
