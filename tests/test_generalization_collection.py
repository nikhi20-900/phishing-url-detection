"""Tests for the Phase 5 controlled, provenance-first legitimate URL collector.

Validates:
1. Input CSV schema parsing and required provenance enforcement.
2. Syntax validation and rejection of malformed or non-HTTP/HTTPS URLs.
3. Verbatim exact URL preservation for downstream model evaluation.
4. Canonical deduplication policy (candidate-normalization-v1).
5. Exact vs normalized duplicate detection.
6. Multi-dimensional syntax dimension auditing (independent dimensions, not mutually exclusive).
7. Risk flag heuristics (open redirects, user-generated content, insecure HTTP).
8. Human-review gating and candidate promotion without model label assignment.
9. End-to-end CLI workflow execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from experiments.generalization.collection.collect_legitimate_corpus import (
    BASE_REQUIRED_COLUMNS,
    NORMALIZATION_VERSION,
    extract_syntax_dimensions,
    load_candidates,
    main,
    normalized_dedupe_key,
    prepare_review_queue,
    promote_approved,
    validate_candidate,
)


@pytest.fixture
def sample_candidate_records() -> list[dict[str, Any]]:
    return [
        {
            "url": "https://www.w3.org/TR/REC-xml/",
            "source_name": "W3C Standards",
            "source_url": "https://www.w3.org/standards/",
            "collection_timestamp": "2026-11-09T10:00:00Z",
            "verification_method": "manual_standards_registry_audit",
            "review_status": "approved",
            "reviewer": "auditor-01",
            "notes": "Canonical XML 1.0 recommendation root",
            "legitimacy_reason": "W3C formal technical recommendation specification",
        },
        {
            "url": "https://docs.python.org/3/library/urllib.parse.html?highlight=urlsplit#urlsplit",
            "source_name": "Python Software Foundation",
            "source_url": "https://docs.python.org/3/",
            "collection_timestamp": "2026-11-09T10:05:00Z",
            "verification_method": "canonical_documentation_audit",
            "review_status": "approved",
            "reviewer": "auditor-01",
            "notes": "Standard library documentation deep link",
            "legitimacy_reason": "Official language reference documentation",
        },
        {
            "url": "https://www.nasa.gov/missions/artemis/?category=overview",
            "source_name": "NASA",
            "source_url": "https://www.nasa.gov/",
            "collection_timestamp": "2026-11-09T10:10:00Z",
            "verification_method": "us_gov_institutional_registry",
            "review_status": "pending",
            "reviewer": "",
            "notes": "Artemis mission overview page",
            "legitimacy_reason": "Federal government scientific agency portal",
        },
        {
            "url": "https://example.edu/~student/index.html",
            "source_name": "University Example",
            "source_url": "https://example.edu/",
            "collection_timestamp": "2026-11-09T10:15:00Z",
            "verification_method": "institutional_seed",
            "review_status": "rejected",
            "reviewer": "auditor-02",
            "notes": "Student home directory detected",
            "legitimacy_reason": "Unvetted user-generated personal page",
        },
    ]


@pytest.fixture
def sample_candidate_csv(tmp_path: Path, sample_candidate_records: list[dict[str, Any]]) -> Path:
    csv_path = tmp_path / "candidates.csv"
    pd.DataFrame(sample_candidate_records).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# 1. Schema and Ingestion Tests
# ---------------------------------------------------------------------------

def test_load_candidates_success(sample_candidate_csv: Path) -> None:
    df = load_candidates(sample_candidate_csv)
    assert len(df) == 4
    for col in BASE_REQUIRED_COLUMNS:
        assert col in df.columns
    assert "legitimacy_reason" in df.columns


def test_load_candidates_missing_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad_candidates.csv"
    pd.DataFrame([{"url": "https://example.com", "source_name": "Test"}]).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_candidates(bad_csv)


def test_load_candidates_without_legitimacy_reason_defaults_gracefully(tmp_path: Path) -> None:
    records = [{col: "test" for col in BASE_REQUIRED_COLUMNS}]
    csv_path = tmp_path / "legacy_candidates.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    df = load_candidates(csv_path)
    assert "legitimacy_reason" in df.columns
    assert df["legitimacy_reason"].iloc[0] == ""


# ---------------------------------------------------------------------------
# 2. Syntax Validation Tests
# ---------------------------------------------------------------------------

def test_validate_candidate_valid() -> None:
    valid, err, parsed, dims = validate_candidate("https://www.rfc-editor.org/rfc/rfc3986.txt")
    assert valid is True
    assert err == ""
    assert parsed is not None
    assert parsed.scheme == "https"
    assert dims["is_multilevel_path"] is True
    assert dims["has_path"] is True


@pytest.mark.parametrize(
    "invalid_url,expected_substr",
    [
        ("", "empty_or_non_string_url"),
        ("   ", "empty_or_non_string_url"),
        ("ftp://files.example.org/download", "invalid_url"),
        ("javascript:alert(1)", "invalid_url"),
        ("not_a_valid_url", "invalid_url"),
    ],
)
def test_validate_candidate_invalid(invalid_url: str, expected_substr: str) -> None:
    valid, err, parsed, dims = validate_candidate(invalid_url)
    assert valid is False
    assert expected_substr in err
    assert parsed is None
    assert dims == {}


# ---------------------------------------------------------------------------
# 3. Exact URL Preservation Tests
# ---------------------------------------------------------------------------

def test_exact_url_preservation_unmodified() -> None:
    # URL with mixed case path, preserved query ordering, and trailing slash
    raw_url = "https://www.example.org/API/v1/Resource/?Sort=DESC&Filter=Active/"
    valid, err, parsed, dims = validate_candidate(raw_url)
    assert valid is True

    input_df = pd.DataFrame([{
        "url": raw_url,
        "source_name": "Test",
        "source_url": "https://www.example.org/",
        "collection_timestamp": "2026-11-09T10:00:00Z",
        "verification_method": "test_method",
        "review_status": "approved",
        "reviewer": "auditor",
        "notes": "Exact path test",
        "legitimacy_reason": "Verbatim preservation check",
    }])
    queue, summary = prepare_review_queue(input_df)

    # url_exact must be character-for-character identical to input
    assert queue["url_exact"].iloc[0] == raw_url
    assert queue["url"].iloc[0] == raw_url
    assert summary["exact_model_input_preserved"] is True


# ---------------------------------------------------------------------------
# 4. Normalization and Deduplication Policy Tests
# ---------------------------------------------------------------------------

def test_normalized_dedupe_key_policy() -> None:
    # 1. Scheme and host lowercased
    assert normalized_dedupe_key("HTTPS://WWW.EXAMPLE.COM/Path") == "https://www.example.com/Path"

    # 2. Trailing dot in host stripped
    assert normalized_dedupe_key("https://www.example.com./path") == "https://www.example.com/path"

    # 3. Default port removed (443 for https, 80 for http)
    assert normalized_dedupe_key("https://www.example.com:443/path") == "https://www.example.com/path"
    assert normalized_dedupe_key("http://www.example.com:80/path") == "http://www.example.com/path"

    # 4. Non-default port preserved
    assert normalized_dedupe_key("https://www.example.com:8443/path") == "https://www.example.com:8443/path"

    # 5. Fragment stripped
    assert normalized_dedupe_key("https://www.example.com/path#section-1") == "https://www.example.com/path"

    # 6. Empty path becomes '/'
    assert normalized_dedupe_key("https://www.example.com") == "https://www.example.com/"

    # 7. Path case and query parameter order preserved verbatim
    assert normalized_dedupe_key("https://www.example.com/Doc?b=2&a=1") == "https://www.example.com/Doc?b=2&a=1"


def test_exact_vs_normalized_duplicate_detection() -> None:
    input_df = pd.DataFrame([
        # Row 1: First occurrence
        {
            "url": "https://www.example.org:443/Docs/API/#v1",
            "source_name": "Test",
            "source_url": "https://www.example.org/",
            "collection_timestamp": "2026-11-09T10:00:00Z",
            "verification_method": "test",
            "review_status": "approved",
            "reviewer": "auditor",
            "notes": "First occurrence",
            "legitimacy_reason": "Reason 1",
        },
        # Row 2: Verbatim exact duplicate
        {
            "url": "https://www.example.org:443/Docs/API/#v1",
            "source_name": "Test",
            "source_url": "https://www.example.org/",
            "collection_timestamp": "2026-11-09T10:01:00Z",
            "verification_method": "test",
            "review_status": "approved",
            "reviewer": "auditor",
            "notes": "Exact duplicate",
            "legitimacy_reason": "Reason 2",
        },
        # Row 3: Normalized duplicate (different port representation, different fragment, uppercase host)
        {
            "url": "HTTPS://WWW.EXAMPLE.ORG/Docs/API/#v2",
            "source_name": "Test",
            "source_url": "https://www.example.org/",
            "collection_timestamp": "2026-11-09T10:02:00Z",
            "verification_method": "test",
            "review_status": "approved",
            "reviewer": "auditor",
            "notes": "Canonical duplicate",
            "legitimacy_reason": "Reason 3",
        },
    ])

    queue, summary = prepare_review_queue(input_df)

    # Row 1: First occurrence
    assert queue["duplicate_exact"].iloc[0] is False or queue["duplicate_exact"].iloc[0] == False
    assert queue["duplicate_normalized"].iloc[0] is False or queue["duplicate_normalized"].iloc[0] == False

    # Row 2: Exact duplicate
    assert bool(queue["duplicate_exact"].iloc[1]) is True
    assert bool(queue["duplicate_normalized"].iloc[1]) is True

    # Row 3: Normalized duplicate only (not exact duplicate)
    assert bool(queue["duplicate_exact"].iloc[2]) is False
    assert bool(queue["duplicate_normalized"].iloc[2]) is True

    assert summary["duplicate_exact_after_first_occurrence"] == 1
    assert summary["duplicate_normalized_after_first_occurrence"] == 2


# ---------------------------------------------------------------------------
# 5. Multi-Dimensional Syntax Auditing Tests
# ---------------------------------------------------------------------------

def test_multidimensional_syntax_overlap_not_exclusive() -> None:
    # URL that has multi-level path, query parameter, and trailing slash simultaneously
    url = "https://www.w3.org/TR/2026/REC-xml-names/?edition=second&format=html/"
    dims = extract_syntax_dimensions(url)

    # Must be flagged across all separate dimensions simultaneously
    assert dims["is_multilevel_path"] is True
    assert dims["path_depth"] >= 3
    assert dims["has_trailing_slash"] is True
    assert dims["has_query"] is True
    assert dims["query_param_count"] == 2
    assert dims["has_special_chars"] is True
    assert dims["is_https"] is True
    assert dims["registered_domain"] == "w3.org"
    assert dims["tld"] == "org"


def test_prepare_review_queue_dimensional_audit_summary() -> None:
    records = [
        # Candidate 1: Multilevel path + query + trailing slash
        {
            "url": "https://www.gov.uk/guidance/digital-standards/?topic=security/",
            "source_name": "GovUK",
            "source_url": "https://www.gov.uk/",
            "collection_timestamp": "2026-11-09T10:00:00Z",
            "verification_method": "gov_portal",
            "review_status": "approved",
            "reviewer": "auditor",
            "notes": "All three attributes",
            "legitimacy_reason": "UK official guidance document",
        },
        # Candidate 2: Multilevel path only (no query, no trailing slash)
        {
            "url": "https://docs.kernel.org/kernel-hacking/hacking.html",
            "source_name": "LinuxKernel",
            "source_url": "https://docs.kernel.org/",
            "collection_timestamp": "2026-11-09T10:01:00Z",
            "verification_method": "kernel_docs",
            "review_status": "approved",
            "reviewer": "auditor",
            "notes": "Multilevel path only",
            "legitimacy_reason": "Linux kernel documentation",
        },
    ]
    queue, summary = prepare_review_queue(pd.DataFrame(records))
    dims = summary["independent_syntax_dimensions"]
    co = summary["dimensional_co_occurrences"]

    assert dims["multilevel_path_count"] == 2
    assert dims["trailing_slash_count"] == 1
    assert dims["query_string_count"] == 1

    # Co-occurrence reflects overlapping dimensions
    assert co["multilevel_path_and_query"] == 1
    assert co["multilevel_path_and_trailing_slash"] == 1
    assert co["multilevel_path_query_and_trailing_slash"] == 1


# ---------------------------------------------------------------------------
# 6. Risk Flags Heuristics Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url,expected_flag",
    [
        ("https://example.edu/~student2026/project/", "possible_user_generated_content"),
        ("https://wiki.example.org/Special:RecentChanges", "possible_user_generated_content"),
        ("https://portal.example.gov/redirect.php?url=https://other.com", "possible_open_redirect"),
        ("https://service.example.org/auth?next=/dashboard", "possible_open_redirect"),
        ("http://insecure-reference.org/specs/index.html", "insecure_http"),
    ],
)
def test_risk_flags_detection(url: str, expected_flag: str) -> None:
    dims = extract_syntax_dimensions(url)
    assert expected_flag in dims["risk_flags"]
    assert dims["has_risk_flags"] is True


# ---------------------------------------------------------------------------
# 7. Human Review Gating and Promotion Tests
# ---------------------------------------------------------------------------

def test_promote_approved_gating_behavior() -> None:
    records = [
        # Record 0: Fully reviewed and approved -> should promote
        {
            "url": "https://www.w3.org/TR/REC-xml/",
            "source_name": "W3C",
            "source_url": "https://www.w3.org/",
            "collection_timestamp": "2026-11-09T10:00:00Z",
            "verification_method": "manual_audit",
            "review_status": "approved",
            "reviewer": "auditor-01",
            "notes": "W3C Recommendation",
            "legitimacy_reason": "Official recommendation standard",
        },
        # Record 1: Pending review -> must NOT promote
        {
            "url": "https://www.ietf.org/rfc/rfc2616.txt",
            "source_name": "IETF",
            "source_url": "https://www.ietf.org/",
            "collection_timestamp": "2026-11-09T10:01:00Z",
            "verification_method": "manual_audit",
            "review_status": "pending",
            "reviewer": "",
            "notes": "HTTP RFC",
            "legitimacy_reason": "IETF RFC specification",
        },
        # Record 2: Rejected -> must NOT promote
        {
            "url": "https://example.edu/~user/phish.html",
            "source_name": "University",
            "source_url": "https://example.edu/",
            "collection_timestamp": "2026-11-09T10:02:00Z",
            "verification_method": "manual_audit",
            "review_status": "rejected",
            "reviewer": "auditor-01",
            "notes": "Suspicious user folder",
            "legitimacy_reason": "Unvetted personal page",
        },
        # Record 3: Approved status but missing reviewer -> must NOT promote
        {
            "url": "https://docs.python.org/3/whatsnew/3.14.html",
            "source_name": "PSF",
            "source_url": "https://docs.python.org/",
            "collection_timestamp": "2026-11-09T10:03:00Z",
            "verification_method": "manual_audit",
            "review_status": "approved",
            "reviewer": "",  # missing reviewer!
            "notes": "Python release notes",
            "legitimacy_reason": "Official docs",
        },
        # Record 4: Approved status but missing verification_method -> must NOT promote
        {
            "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP",
            "source_name": "MDN",
            "source_url": "https://developer.mozilla.org/",
            "collection_timestamp": "2026-11-09T10:04:00Z",
            "verification_method": "",  # missing verification method!
            "review_status": "approved",
            "reviewer": "auditor-01",
            "notes": "MDN HTTP guide",
            "legitimacy_reason": "Mozilla documentation",
        },
    ]

    queue, summary = prepare_review_queue(pd.DataFrame(records))
    approved = promote_approved(queue)

    assert len(approved) == 1
    assert approved["url_exact"].iloc[0] == "https://www.w3.org/TR/REC-xml/"


def test_no_model_labels_assigned_in_queue_or_approved() -> None:
    records = [{
        "url": "https://www.w3.org/TR/REC-xml/",
        "source_name": "W3C",
        "source_url": "https://www.w3.org/",
        "collection_timestamp": "2026-11-09T10:00:00Z",
        "verification_method": "manual_audit",
        "review_status": "approved",
        "reviewer": "auditor-01",
        "notes": "W3C Recommendation",
        "legitimacy_reason": "Official standard",
    }]
    queue, _ = prepare_review_queue(pd.DataFrame(records))
    approved = promote_approved(queue)

    # Neither review queue nor approved manifest must contain model labels
    assert "label" not in queue.columns
    assert "prediction" not in queue.columns
    assert "label" not in approved.columns
    assert "prediction" not in approved.columns


# ---------------------------------------------------------------------------
# 8. End-to-End CLI Tests
# ---------------------------------------------------------------------------

def test_cli_execution_with_approved_output(
    sample_candidate_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_queue = tmp_path / "review_queue.csv"
    out_summary = tmp_path / "summary.json"
    out_approved = tmp_path / "approved.csv"

    test_args = [
        "collect_legitimate_corpus.py",
        "--input",
        str(sample_candidate_csv),
        "--output",
        str(out_queue),
        "--summary",
        str(out_summary),
        "--approved-output",
        str(out_approved),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    exit_code = main()
    assert exit_code == 0
    assert out_queue.exists()
    assert out_summary.exists()
    assert out_approved.exists()

    queue_df = pd.read_csv(out_queue)
    assert len(queue_df) == 4
    assert "url_exact" in queue_df.columns
    assert "normalized_dedupe_key" in queue_df.columns
    assert "is_multilevel_path" in queue_df.columns

    approved_df = pd.read_csv(out_approved)
    # Exactly 2 of the 4 sample records were approved with complete reviewer & rationale
    assert len(approved_df) == 2
    assert "label" not in approved_df.columns

    summary_data = json.loads(out_summary.read_text(encoding="utf-8"))
    assert summary_data["input_rows"] == 4
    assert summary_data["valid_candidates"] == 4
    assert summary_data["normalization_version"] == NORMALIZATION_VERSION
    assert summary_data["exact_model_input_preserved"] is True
