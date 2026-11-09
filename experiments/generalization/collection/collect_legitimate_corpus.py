"""Collect candidate legitimate URLs from an explicitly supplied URL list.

This module is deliberately offline: it does not crawl, resolve DNS, issue HTTP
requests, or decide that a URL is legitimate. It validates URL syntax using the
project's existing parser, attaches operator-supplied provenance, and emits a
review queue. Only a separately reviewed manifest may promote records into a
verified augmentation corpus.

Input CSV columns:
    url,source_name,source_url,collection_timestamp,verification_method,review_status,reviewer,notes

Required source metadata is preserved verbatim. Candidate rows are never assigned
a legitimate model label by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

# Run as a script from any working directory.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import parse_url  # noqa: E402


REQUIRED_COLUMNS = (
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

    Hostname and scheme are case-insensitive. Default ports are removed,
    fragments are discarded, and an empty path is treated as '/'.
    Path and query case/order are retained; percent-encoding is not decoded.
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


def registered_domain(hostname: str) -> str:
    """Return a conservative last-two-label grouping key.

    This is NOT a public-suffix implementation. For authoritative domain
    grouping, the audit stage must use the project's public-suffix helper.
    """
    host = hostname.lower().rstrip(".")
    labels = host.split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def validate_candidate(url: Any) -> tuple[bool, str, dict[str, Any]]:
    if not isinstance(url, str) or not url.strip():
        return False, "empty_or_non_string_url", {}
    try:
        parsed = parse_url(url)
    except (TypeError, ValueError) as exc:
        return False, f"invalid_url:{exc}", {}
    # Keep raw input intact for model input; parser supplies normalized
    # parse information for audits only.
    return True, "", {
        "scheme": parsed.scheme,
        "hostname": parsed.hostname,
        "path": parsed.path,
        "has_query": bool(parsed.query),
        "has_fragment": bool(parsed.fragment),
        "is_ip": bool(parsed.is_ip),
    }


def load_candidates(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    return df.loc[:, REQUIRED_COLUMNS].copy()


def prepare_review_queue(input_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    duplicate_exact = 0
    duplicate_normalized = 0
    invalid = 0

    for index, row in input_df.iterrows():
        original_url = str(row["url"])
        valid, error, parsed = validate_candidate(original_url)
        record = row.to_dict()
        record.update({
            "input_row": int(index) + 2,
            "candidate_id": hashlib.sha256(original_url.encode("utf-8")).hexdigest(),
            "url_exact": original_url,
            "normalization_version": NORMALIZATION_VERSION,
            "normalized_dedupe_key": "",
            "syntax_valid": valid,
            "validation_error": error,
            "hostname": parsed.get("hostname", ""),
            "scheme": parsed.get("scheme", ""),
            "path": parsed.get("path", ""),
            "has_query": parsed.get("has_query", False),
            "has_fragment": parsed.get("has_fragment", False),
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
        else:
            invalid += 1

        rows.append(record)

    output = pd.DataFrame(rows)
    status_counts = Counter(output["review_status"].tolist()) if not output.empty else Counter()
    summary = {
        "input_rows": int(len(input_df)),
        "syntax_invalid_or_bad_status": invalid,
        "duplicate_exact_after_first_occurrence": duplicate_exact,
        "duplicate_normalized_after_first_occurrence": duplicate_normalized,
        "review_status_counts": dict(status_counts),
        "normalization_version": NORMALIZATION_VERSION,
        "label_assignment": "none; candidate queue only",
    }
    return output, summary


def promote_approved(review_queue: pd.DataFrame) -> pd.DataFrame:
    """Return reviewed approved, valid, unique records without assigning labels.

    Promotion is disabled until an explicit reviewer and verification method
    are present. This output is still a candidate dataset, not a training set.
    """
    required_reviewed = (
        review_queue["review_status"].eq("approved")
        & review_queue["syntax_valid"].astype(bool)
        & ~review_queue["duplicate_exact"].astype(bool)
        & ~review_queue["duplicate_normalized"].astype(bool)
        & review_queue["reviewer"].astype(str).str.strip().ne("")
        & review_queue["verification_method"].astype(str).str.strip().ne("")
        & review_queue["source_name"].astype(str).str.strip().ne("")
        & review_queue["source_url"].astype(str).str.strip().ne("")
    )
    return review_queue.loc[required_reviewed].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Operator-supplied candidate CSV")
    parser.add_argument("--output", type=Path, default=Path("experiments/generalization/collection/review_queue.csv"))
    parser.add_argument("--summary", type=Path, default=Path("experiments/generalization/collection/collection_summary.json"))
    parser.add_argument("--approved-output", type=Path, help="Optional reviewed-candidate export; still no labels")
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

    print(json.dumps(summary, indent=2))
    print(f"Review queue written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
