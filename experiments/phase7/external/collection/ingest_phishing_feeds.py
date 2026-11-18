"""
Ingest and Filter Live Phishing Threat Feeds for Phase 7 Milestone 5.

Features:
- Ingests verified community-reported phishing URLs from PhishTank.
- Audits each candidate against the dynamic master exclusion set D_historical.
- Enforces strict zero-domain-overlap invariance: D_phish ∩ D_historical = ∅.
- Enforces domain clustering caps: max 3 URLs per registered domain.
- Selects 650 high-diversity phishing URLs with verified metadata.
"""

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate_domain_split import extract_registered_domain
from src.features import extract_url_features, parse_url
from experiments.phase7.external.evaluation.audit_domain_independence import (
    build_dynamic_historical_domain_inventory,
)

RAW_PHISHTANK_CACHE = REPO_ROOT / "experiments" / "phase7" / "external" / "evidence" / "phishtank_raw.csv"
EXPECTED_RAW_PHISHTANK_SHA256 = "4885e37276d8a33b3da045d9fca0888c152e4d20c91e7a69f39eb032893a54ba"


def extract_structural_stratum(url: str) -> Tuple[str, bool, bool, bool]:
    """
    Determine the structural stratum and marginal flags for a URL.
    Returns: (stratum_name, is_query_bearing, is_multi_level_path, has_special_syntax)
    """
    try:
        parsed = parse_url(url)
    except Exception:
        return "unparsable", False, False, False

    path = parsed.path or ""
    query = parsed.query or ""
    raw_url = parsed.raw_url

    has_query = bool(query)
    clean_path = path.strip("/")
    segments = [s for s in clean_path.split("/") if s]
    depth = len(segments)

    is_multi_level = depth >= 2
    has_special_syntax = bool("%" in raw_url or parsed.port or "#" in raw_url)

    if has_special_syntax:
        stratum = "special_syntax_variants"
    elif depth == 0 and not has_query:
        stratum = "homepage_root"
    elif depth == 1 and not has_query:
        stratum = "single_level_path_no_query"
    elif depth >= 2 and not has_query:
        stratum = "multi_level_path_no_query"
    elif depth == 0 and has_query:
        stratum = "root_with_query"
    elif depth >= 1 and has_query:
        stratum = "path_and_query"
    else:
        stratum = "multi_level_path_no_query"

    return stratum, has_query, is_multi_level, has_special_syntax


def ingest_clean_phishing_candidates(
    master_domains: Set[str],
    target_count: int = 650,
    max_urls_per_domain: int = 3,
    raw_cache_path: Path = RAW_PHISHTANK_CACHE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Ingest verified phishing URLs from PhishTank cache, filtering against master_domains.
    Returns: (clean_phish_records, quarantined_records)
    """
    if not raw_cache_path.exists():
        raise FileNotFoundError(f"PhishTank cache not found at {raw_cache_path}")

    EXPECTED_RAW_SNAPSHOT_SHA256 = "4885e37276d8a33b3da045d9fca0888c152e4d20c91e7a69f39eb032893a54ba"
    actual_hash = hashlib.sha256(raw_cache_path.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_RAW_SNAPSHOT_SHA256:
        raise ValueError(
            f"PhishTank upstream snapshot hash mismatch! Expected {EXPECTED_RAW_SNAPSHOT_SHA256}, got {actual_hash}"
        )

    with open(raw_cache_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    clean_records: List[Dict[str, Any]] = []
    quarantined_records: List[Dict[str, Any]] = []
    domain_counts: Dict[str, int] = {}

    for r in raw_rows:
        # Defensively enforce that the threat feed entry is explicitly verified and online
        if str(r.get("verified", "")).strip().lower() != "yes":
            continue
        if str(r.get("online", "")).strip().lower() != "yes":
            continue

        url = r.get("url", "").strip()
        if not url:
            continue

        try:
            dom = extract_registered_domain(url)
            # Ensure features can be extracted cleanly
            _ = extract_url_features(url)
        except Exception:
            continue

        if not dom:
            continue

        if dom in master_domains:
            quarantined_records.append({
                "url": url,
                "registered_domain": dom,
                "source": "PhishTank-Verified",
                "source_id": str(r.get("phish_id", "")),
                "quarantine_reason": "HISTORICAL_DOMAIN_OVERLAP",
            })
            continue

        current_dom_count = domain_counts.get(dom, 0)
        if current_dom_count >= max_urls_per_domain:
            continue

        stratum, has_query, is_multi_level, has_special = extract_structural_stratum(url)

        domain_counts[dom] = current_dom_count + 1
        clean_records.append({
            "url": url,
            "label": 0,  # 0 = Phishing
            "registered_domain": dom,
            "sector": "phishing_threat_feed",
            "structural_stratum": stratum,
            "source": "PhishTank-Verified",
            "source_id": str(r.get("phish_id", f"PT-{len(clean_records)+1:06d}")),
            "verified": r.get("verified", "yes"),
            "online": r.get("online", "yes"),
            "verification_time": r.get("verification_time", ""),
            "phish_detail_url": r.get("phish_detail_url", ""),
            "submission_time": r.get("submission_time", ""),
            "target": r.get("target", "Unknown"),
            "query_bearing": has_query,
            "multi_level_path": is_multi_level,
            "special_syntax": has_special,
        })

        if len(clean_records) >= target_count * 2:
            # Gather extra pool for balanced stratification
            break

    # Subsample / select exactly target_count to maximize domain representation
    # If we have more than target_count, sort/order to prioritize distinct domains first
    seen_domains: Set[str] = set()
    pass1: List[Dict[str, Any]] = []
    pass2: List[Dict[str, Any]] = []

    for rec in clean_records:
        d = rec["registered_domain"]
        if d not in seen_domains:
            seen_domains.add(d)
            pass1.append(rec)
        else:
            pass2.append(rec)

    final_phish = (pass1 + pass2)[:target_count]

    # Verify zero overlap invariant
    for rec in final_phish:
        if rec["registered_domain"] in master_domains:
            raise ValueError(f"INVARIANT VIOLATION: {rec['registered_domain']} in master_domains")

    return final_phish, quarantined_records


if __name__ == "__main__":
    print("Building historical domain inventory...")
    master_domains, _ = build_dynamic_historical_domain_inventory(REPO_ROOT)
    print(f"Master exclusion domains: {len(master_domains)}")
    clean_phish, quar = ingest_clean_phishing_candidates(master_domains, target_count=650)
    print(f"Selected clean phishing candidates: {len(clean_phish)}")
    unique_doms = len(set(r["registered_domain"] for r in clean_phish))
    print(f"Unique registered domains in selected phishing set: {unique_doms}")
    print(f"Quarantined collision records: {len(quar)}")
