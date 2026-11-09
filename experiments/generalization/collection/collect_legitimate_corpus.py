"""Collect candidate legitimate URLs from an explicitly supplied URL list.

This module is deliberately offline: it does not crawl, resolve DNS, issue HTTP
requests, or decide that a URL is legitimate. It validates URL syntax using the
project's existing parser, attaches operator-supplied provenance, extracts
multi-dimensional syntax characteristics, and emits a review queue.
Only a separately reviewed manifest may promote records into a verified
augmentation corpus. Candidate rows are never assigned a model label by this script.

Input CSV columns:
    url,source_name,source_url,collection_timestamp,verification_method,review_status,reviewer,notes[,legitimacy_reason]

Core Architectural Principles:
1. Source reputation is a collection criterion, not proof of legitimacy. Institutional
   domains can host compromised content, open redirects, or user-generated material.
2. Exact URL preservation: verbatim URL strings are stored in `url_exact` for model
   evaluation. Normalized strings (`normalized_dedupe_key`) are computed strictly for
   duplicate detection and never replace the raw model input.
3. Separation of collection from verification: candidates enter a review queue and
   remain unpromoted until explicit human review and verification metadata are supplied.
   No model labels (0 or 1) are assigned by this module.
4. Multi-dimensional syntax auditing: path depth, trailing slashes, query parameters,
   and special characters overlap and are audited as independent dimensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

# Ensure repository root is in sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import ParsedURL, parse_url  # noqa: E402
from src.evaluate_domain_split import extract_registered_domain  # noqa: E402


BASE_REQUIRED_COLUMNS = (
    "url",
    "source_name",
    "source_url",
    "collection_timestamp",
    "verification_method",
    "review_status",
    "reviewer",
    "notes",
)

APPROVED_REVIEW_STATUSES = {"approved", "rejected", "pending"}
NORMALIZATION_VERSION = "candidate-normalization-v1"


def normalized_dedupe_key(url: str) -> str:
    """Build a conservative dedupe key without changing the model input URL.

    Normalization policy (candidate-normalization-v1):
    1. Scheme and hostname are converted to lowercase.
    2. Trailing dots in hostname are removed.
    3. Default ports (80 for http, 443 for https) are omitted.
    4. IPv6 addresses are enclosed in brackets.
    5. Userinfo (username:password@) is preserved.
    6. Empty path is normalized to '/'.
    7. Fragments are discarded (they do not reach web servers).
    8. Path case, query parameters, query order, and percent-encoding are preserved verbatim.
    """
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    port = parsed.port
    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += ":" + parsed.password
        userinfo += "@"
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        hostport = f"{hostname}:{port}"
    else:
        hostport = hostname
    if ":" in hostname and not hostname.startswith("["):
        hostport = f"[{hostname}]" + (f":{port}" if port is not None else "")
    netloc = userinfo + hostport
    path = parsed.path or "/"
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def extract_syntax_dimensions(url: str, parsed: ParsedURL | None = None) -> dict[str, Any]:
    """Extract independent multi-dimensional syntax characteristics for auditing."""
    if parsed is None:
        try:
            parsed = parse_url(url)
        except Exception:
            return {}

    path = parsed.path or ""
    segments = [s for s in path.split("/") if s]
    path_depth = len(segments)

    has_path = path_depth > 0 or (path != "" and path != "/")
    is_multilevel_path = path_depth >= 2
    has_trailing_slash = path.endswith("/") if path else False
    has_query = bool(parsed.query)
    query_param_count = len([p for p in parsed.query.split("&") if p]) if parsed.query else 0
    has_fragment = bool(parsed.fragment)

    raw_path_query = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")
    special_chars = [c for c in raw_path_query if not c.isalnum() and c not in ("/", "?", "&", "=")]

    try:
        reg_domain = extract_registered_domain(url)
    except Exception:
        reg_domain = parsed.hostname.lower().rstrip(".") if parsed.hostname else ""

    tld = reg_domain.split(".")[-1] if "." in reg_domain else ""

    # Heuristic risk flags: user-generated paths/subdomains or potential open redirects
    risk_flags: list[str] = []
    lower_path = path.lower()
    lower_host = (parsed.hostname or "").lower()
    lower_query = (parsed.query or "").lower()
    if (
        "~" in path
        or "/users/" in lower_path
        or "/wiki/" in lower_path
        or "/blog/" in lower_path
        or lower_host.startswith("wiki.")
        or ".wiki." in lower_host
        or lower_host.startswith("blog.")
    ):
        risk_flags.append("possible_user_generated_content")
    if any(param in lower_query for param in ["redirect=", "url=", "next=", "target=", "dest="]):
        risk_flags.append("possible_open_redirect")
    if parsed.scheme == "http":
        risk_flags.append("insecure_http")

    return {
        "scheme": parsed.scheme,
        "is_https": bool(parsed.is_https),
        "hostname": parsed.hostname,
        "registered_domain": reg_domain,
        "tld": tld,
        "path": path,
        "path_depth": path_depth,
        "has_path": has_path,
        "is_multilevel_path": is_multilevel_path,
        "has_trailing_slash": has_trailing_slash,
        "has_query": has_query,
        "query_param_count": query_param_count,
        "has_fragment": has_fragment,
        "has_special_chars": len(special_chars) > 0,
        "special_char_count": len(special_chars),
        "risk_flags": risk_flags,
        "has_risk_flags": len(risk_flags) > 0,
        "is_ip": bool(parsed.is_ip),
    }


def validate_candidate(url: Any) -> tuple[bool, str, ParsedURL | None, dict[str, Any]]:
    """Validate candidate URL syntax using the project's existing parser."""
    if not isinstance(url, str) or not url.strip():
        return False, "empty_or_non_string_url", None, {}
    try:
        parsed = parse_url(url.strip())
    except (TypeError, ValueError) as exc:
        return False, f"invalid_url:{exc}", None, {}

    dimensions = extract_syntax_dimensions(url.strip(), parsed)
    return True, "", parsed, dimensions


def load_candidates(input_path: Path) -> pd.DataFrame:
    """Load operator-supplied candidate CSV and verify required provenance columns."""
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    missing = sorted(set(BASE_REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    columns = list(BASE_REQUIRED_COLUMNS)
    if "legitimacy_reason" in df.columns:
        columns.append("legitimacy_reason")
    out = df.loc[:, columns].copy()
    if "legitimacy_reason" not in out.columns:
        out["legitimacy_reason"] = ""
    return out


def prepare_review_queue(input_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest candidate URLs into an audited review queue with provenance and syntax dimensions."""
    rows: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    duplicate_exact = 0
    duplicate_normalized = 0
    invalid = 0

    dim_multilevel_path = 0
    dim_trailing_slash = 0
    dim_query = 0
    dim_fragment = 0
    dim_special_chars = 0
    dim_https = 0
    co_path_and_query = 0
    co_path_and_trailing_slash = 0
    co_query_and_trailing_slash = 0
    co_path_query_trailing_slash = 0
    candidates_with_risks = 0
    risk_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    tld_counter: Counter[str] = Counter()

    for index, row in input_df.iterrows():
        original_url = str(row["url"])
        valid, error, parsed, dims = validate_candidate(original_url)
        record = row.to_dict()

        legitimacy_reason = str(record.get("legitimacy_reason", "")).strip()
        if not legitimacy_reason and record.get("notes"):
            legitimacy_reason = str(record.get("notes")).strip()

        record.update({
            "input_row": int(index) + 2,
            "candidate_id": hashlib.sha256(original_url.strip().encode("utf-8")).hexdigest(),
            "url_exact": original_url,
            "legitimacy_reason": legitimacy_reason,
            "normalization_version": NORMALIZATION_VERSION,
            "normalized_dedupe_key": "",
            "syntax_valid": valid,
            "validation_error": error,
            "scheme": dims.get("scheme", ""),
            "hostname": dims.get("hostname", ""),
            "registered_domain": dims.get("registered_domain", ""),
            "tld": dims.get("tld", ""),
            "path": dims.get("path", ""),
            "path_depth": dims.get("path_depth", 0),
            "is_multilevel_path": dims.get("is_multilevel_path", False),
            "has_trailing_slash": dims.get("has_trailing_slash", False),
            "has_query": dims.get("has_query", False),
            "query_param_count": dims.get("query_param_count", 0),
            "has_fragment": dims.get("has_fragment", False),
            "has_special_chars": dims.get("has_special_chars", False),
            "risk_flags": ";".join(dims.get("risk_flags", [])),
            "has_risk_flags": dims.get("has_risk_flags", False),
            "review_status": str(row["review_status"]).strip().lower(),
            "duplicate_exact": False,
            "duplicate_normalized": False,
        })

        status = record["review_status"]
        if status not in APPROVED_REVIEW_STATUSES:
            record["syntax_valid"] = False
            record["validation_error"] = (
                (error + "; " if error else "") + f"invalid_review_status:{status}"
            )

        if valid:
            exact_key = original_url.strip()
            norm_key = normalized_dedupe_key(exact_key)
            record["normalized_dedupe_key"] = norm_key

            if exact_key in seen_exact:
                record["duplicate_exact"] = True
                duplicate_exact += 1
            else:
                seen_exact.add(exact_key)

            if norm_key in seen_normalized:
                record["duplicate_normalized"] = True
                duplicate_normalized += 1
            else:
                seen_normalized.add(norm_key)

            # Accumulate multi-dimensional audit statistics
            if dims.get("is_multilevel_path"):
                dim_multilevel_path += 1
            if dims.get("has_trailing_slash"):
                dim_trailing_slash += 1
            if dims.get("has_query"):
                dim_query += 1
            if dims.get("has_fragment"):
                dim_fragment += 1
            if dims.get("has_special_chars"):
                dim_special_chars += 1
            if dims.get("is_https"):
                dim_https += 1

            # Multi-dimensional overlap counts
            if dims.get("is_multilevel_path") and dims.get("has_query"):
                co_path_and_query += 1
            if dims.get("is_multilevel_path") and dims.get("has_trailing_slash"):
                co_path_and_trailing_slash += 1
            if dims.get("has_query") and dims.get("has_trailing_slash"):
                co_query_and_trailing_slash += 1
            if (
                dims.get("is_multilevel_path")
                and dims.get("has_query")
                and dims.get("has_trailing_slash")
            ):
                co_path_query_trailing_slash += 1

            if dims.get("has_risk_flags"):
                candidates_with_risks += 1
                for rf in dims.get("risk_flags", []):
                    risk_counter[rf] += 1

            domain_counter[dims.get("registered_domain", "")] += 1
            tld_counter[dims.get("tld", "")] += 1
        else:
            invalid += 1

        rows.append(record)

    output = pd.DataFrame(rows)
    status_counts = Counter(output["review_status"].tolist()) if not output.empty else Counter()
    total_valid = len(input_df) - invalid

    summary = {
        "input_rows": int(len(input_df)),
        "syntax_invalid_or_bad_status": invalid,
        "valid_candidates": total_valid,
        "duplicate_exact_after_first_occurrence": duplicate_exact,
        "duplicate_normalized_after_first_occurrence": duplicate_normalized,
        "unique_exact_urls": len(seen_exact),
        "unique_normalized_keys": len(seen_normalized),
        "review_status_counts": dict(status_counts),
        "independent_syntax_dimensions": {
            "multilevel_path_count": dim_multilevel_path,
            "multilevel_path_rate": round(dim_multilevel_path / total_valid, 4) if total_valid else 0.0,
            "trailing_slash_count": dim_trailing_slash,
            "trailing_slash_rate": round(dim_trailing_slash / total_valid, 4) if total_valid else 0.0,
            "query_string_count": dim_query,
            "query_string_rate": round(dim_query / total_valid, 4) if total_valid else 0.0,
            "fragment_count": dim_fragment,
            "fragment_rate": round(dim_fragment / total_valid, 4) if total_valid else 0.0,
            "special_chars_count": dim_special_chars,
            "special_chars_rate": round(dim_special_chars / total_valid, 4) if total_valid else 0.0,
            "https_count": dim_https,
            "https_rate": round(dim_https / total_valid, 4) if total_valid else 0.0,
        },
        "dimensional_co_occurrences": {
            "multilevel_path_and_query": co_path_and_query,
            "multilevel_path_and_trailing_slash": co_path_and_trailing_slash,
            "query_and_trailing_slash": co_query_and_trailing_slash,
            "multilevel_path_query_and_trailing_slash": co_path_query_trailing_slash,
        },
        "risk_heuristics": {
            "candidates_with_risk_flags": candidates_with_risks,
            "flag_breakdown": dict(risk_counter),
        },
        "domain_diversity": {
            "unique_registered_domains": len([d for d in domain_counter if d]),
            "top_registered_domains": dict(domain_counter.most_common(10)),
            "top_tlds": dict(tld_counter.most_common(5)),
        },
        "normalization_version": NORMALIZATION_VERSION,
        "label_assignment": "none; candidate review queue only",
        "exact_model_input_preserved": True,
    }
    return output, summary


def promote_approved(review_queue: pd.DataFrame) -> pd.DataFrame:
    """Return reviewed, approved, valid, unique records without assigning labels.

    Promotion requires:
    1. review_status == 'approved'
    2. syntax_valid == True
    3. duplicate_exact == False
    4. duplicate_normalized == False
    5. Non-empty reviewer
    6. Non-empty verification_method
    7. Non-empty source_name and source_url
    8. Non-empty legitimacy_reason or notes documenting reason for authenticity

    This output remains a vetted candidate dataset, NOT a labeled training set.
    No model labels (0 or 1) are assigned.
    """
    has_legitimacy_rationale = (
        review_queue["legitimacy_reason"].astype(str).str.strip().ne("")
        | review_queue["notes"].astype(str).str.strip().ne("")
    )
    required_reviewed = (
        review_queue["review_status"].eq("approved")
        & review_queue["syntax_valid"].astype(bool)
        & ~review_queue["duplicate_exact"].astype(bool)
        & ~review_queue["duplicate_normalized"].astype(bool)
        & review_queue["reviewer"].astype(str).str.strip().ne("")
        & review_queue["verification_method"].astype(str).str.strip().ne("")
        & review_queue["source_name"].astype(str).str.strip().ne("")
        & review_queue["source_url"].astype(str).str.strip().ne("")
        & has_legitimacy_rationale
    )
    approved = review_queue.loc[required_reviewed].copy()
    # Explicitly ensure no model label column exists in the output
    if "label" in approved.columns:
        approved = approved.drop(columns=["label"])
    if "prediction" in approved.columns:
        approved = approved.drop(columns=["prediction"])
    return approved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Operator-supplied candidate CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generalization/collection/review_queue.csv"),
        help="Destination path for candidate review queue CSV",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/generalization/collection/collection_summary.json"),
        help="Destination path for audit summary JSON",
    )
    parser.add_argument(
        "--approved-output",
        type=Path,
        help="Optional destination path for vetted approved candidate export (still no model labels)",
    )
    args = parser.parse_args()

    candidates = load_candidates(args.input)
    queue, summary = prepare_review_queue(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.approved_output:
        approved = promote_approved(queue)
        args.approved_output.parent.mkdir(parents=True, exist_ok=True)
        approved.to_csv(args.approved_output, index=False)
        print(f"Approved candidates ({len(approved)} rows) written to: {args.approved_output}")

    print(json.dumps(summary, indent=2))
    print(f"Review queue ({len(queue)} rows) written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
