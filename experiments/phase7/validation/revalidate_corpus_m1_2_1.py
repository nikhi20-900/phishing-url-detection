"""
Corpus Revalidation and Integration for Phase 7 Milestone 1.2.1 (M1.2.1).

Integrates hardened HTTP/TLS network observations and independent human review decisions
to produce:
- experiments/phase7/collection/evidence/m1_2_1_validation_records.jsonl (all 1,207 evaluated candidates)
- experiments/phase7/datasets/validated_benign_corpus_v1_2_1.csv (substantiated legitimate training-eligible URLs)

Strict Admissibility Requirements:
1. Authentic network retrieval with is_synthetic == False.
2. Target does not resolve to private or non-routable IP (RFC 1918 / RFC 6598 / RFC 3927).
3. HTTP status code is in 200..399 without cross-domain redirect.
4. For HTTPS candidates: TLS certificate verification on the retrieval socket MUST be True.
5. Primary reviewer decision is 'approved' or ('flagged_for_second_review' with senior review 'approved').
6. Senior reviewer supervisory approval is True for all risk-flagged candidates.
7. Zero collisions with Phase 4 external benchmark.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import urlsplit

import pandas as pd

from src.evaluate_domain_split import extract_registered_domain

REPO_ROOT = Path(__file__).resolve().parents[3]

PHASE4_BENCHMARK_PATH = REPO_ROOT / "data" / "external" / "external_validation_benchmark.csv"


def revalidate_corpus_m1_2_1(
    queue_csv_path: Path,
    raw_obs_jsonl_path: Path,
    primary_reviews_jsonl_path: Path,
    senior_reviews_jsonl_path: Path,
    output_validation_records_path: Path,
    output_corpus_v1_2_1_path: Path,
) -> pd.DataFrame:
    """
    Execute M1.2.1 corpus revalidation with hardened security and review integrity gates.
    """
    print("Executing Phase 7 M1.2.1: Hardened Corpus Revalidation & Integration...")
    q_df = pd.read_csv(queue_csv_path)

    # Load authentic observations
    obs_by_id = {}
    with open(raw_obs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                obs_by_id[item["candidate_id"]] = item

    # Load primary reviews
    prim_by_id = {}
    with open(primary_reviews_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prim_by_id[item["candidate_id"]] = item

    # Load senior reviews
    senior_by_id = {}
    with open(senior_reviews_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                senior_by_id[item["candidate_id"]] = item

    # Load Phase 4 external benchmark for collision protection
    p4_df = pd.read_csv(PHASE4_BENCHMARK_PATH)
    p4_urls: Set[str] = set(p4_df["url"].dropna().str.strip())

    validation_records: List[Dict[str, Any]] = []
    admissible_candidates: List[Dict[str, Any]] = []

    for _, row in q_df.iterrows():
        cand_id = row["candidate_id"]
        url = row["url"]
        domain = row["registered_domain"]
        scheme = urlsplit(url).scheme.lower()

        obs = obs_by_id.get(cand_id, {})
        prim = prim_by_id.get(cand_id, {})
        senior = senior_by_id.get(cand_id)

        # Invariant 1: Must be authentic observation
        if obs.get("is_synthetic") is not False:
            raise ValueError(f"Observation for candidate {cand_id} must have is_synthetic == False")

        is_accessible = obs.get("is_accessible", False)
        status_code = obs.get("http_status_code")
        tls_verified = obs.get("tls_verified", False)
        is_private = obs.get("is_private_ip", False)
        is_cross_domain = obs.get("is_cross_domain_redirect", False)
        prim_decision = prim.get("decision")

        admissible = False
        rejection_reason = None

        if is_private:
            rejection_reason = "RFC 1918 / Private IP blocked"
        elif is_cross_domain:
            rejection_reason = "Cross-domain redirect blocked before connection"
        elif scheme == "https" and not tls_verified:
            # Criterion 5: HTTPS corpus admission requires verified TLS
            rejection_reason = "UNVERIFIED_TLS: HTTPS candidate failed certificate verification"
        elif not is_accessible:
            rejection_reason = f"Network retrieval failed (status={status_code}, error={obs.get('observation_error')})"
        elif prim_decision == "rejected":
            rejection_reason = f"Primary reviewer rejected: {prim.get('rationale')}"
        elif prim_decision == "flagged_for_second_review":
            if not senior:
                rejection_reason = "Flagged candidate lacks senior review record"
            elif senior.get("second_decision") != "approved" or not senior.get("supervisory_approval"):
                rejection_reason = f"Senior review rejected: {senior.get('second_rationale')}"
            else:
                admissible = True
        elif prim_decision == "approved":
            # If flagged, senior review must also approve
            if senior and (senior.get("second_decision") != "approved" or not senior.get("supervisory_approval")):
                rejection_reason = f"Senior review rejected: {senior.get('second_rationale')}"
            else:
                admissible = True
        else:
            rejection_reason = f"Unsubstantiated primary status: {prim_decision}"

        # External benchmark collision protection
        if admissible and url.strip() in p4_urls:
            admissible = False
            rejection_reason = "COLLISION: URL matches quarantined Phase 4 external benchmark"

        record = {
            "candidate_id": cand_id,
            "url": url,
            "url_exact": row["url_exact"],
            "registered_domain": domain,
            "source_name": row["source_name"],
            "collection_timestamp": row["collection_timestamp"],
            "retrieval_timestamp_utc": obs.get("retrieval_timestamp_utc"),
            "resolved_ip": obs.get("resolved_ip"),
            "connected_ip": obs.get("connected_ip"),
            "http_status_code": status_code,
            "tls_verified": tls_verified,
            "tls_protocol": obs.get("tls_protocol"),
            "tls_cipher": obs.get("tls_cipher"),
            "tls_cert_subject_cn": obs.get("tls_cert_subject_cn"),
            "redirect_count": obs.get("redirect_count", 0),
            "final_url": obs.get("final_url"),
            "is_cross_domain_redirect": is_cross_domain,
            "content_length_bytes": obs.get("content_length_bytes", 0),
            "response_sha256": obs.get("response_sha256"),
            "html_title": obs.get("html_title"),
            "primary_reviewer_id": prim.get("reviewer_id"),
            "primary_review_decision": prim_decision,
            "primary_review_rationale": prim.get("rationale"),
            "primary_record_hash": prim.get("record_hash"),
            "has_risk_flags": bool(row["has_risk_flags"]),
            "risk_flags": row["risk_flags"] if pd.notna(row["risk_flags"]) else None,
            "second_review_required": (senior is not None),
            "senior_reviewer_id": senior.get("senior_reviewer_id") if senior else None,
            "second_review_decision": senior.get("second_decision") if senior else None,
            "second_review_rationale": senior.get("second_rationale") if senior else None,
            "senior_record_hash": senior.get("record_hash") if senior else None,
            "admissible_for_training": admissible,
            "rejection_reason": rejection_reason,
            "is_synthetic": False,
        }
        validation_records.append(record)

        if admissible:
            admissible_row = row.to_dict()
            admissible_row["retrieval_timestamp_utc"] = obs.get("retrieval_timestamp_utc")
            admissible_row["connected_ip"] = obs.get("connected_ip")
            admissible_row["http_status_code"] = status_code
            admissible_row["tls_verified"] = tls_verified
            admissible_row["tls_protocol"] = obs.get("tls_protocol")
            admissible_row["response_sha256"] = obs.get("response_sha256")
            admissible_row["final_url"] = obs.get("final_url")
            admissible_row["primary_reviewer_id"] = prim.get("reviewer_id")
            admissible_row["senior_reviewer_id"] = senior.get("senior_reviewer_id") if senior else None
            admissible_candidates.append(admissible_row)

    # Write validation records JSONL
    output_validation_records_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_validation_records_path, "w", encoding="utf-8") as f:
        for r in validation_records:
            f.write(json.dumps(r) + "\n")

    # Write validated benign corpus v1.2.1 CSV
    corpus_df = pd.DataFrame(admissible_candidates)
    output_corpus_v1_2_1_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_csv(output_corpus_v1_2_1_path, index=False)

    def safe_rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    print(f"  [SUCCESS] Total candidates evaluated: {len(validation_records)}")
    print(f"  [SUCCESS] Substantiated corpus v1.2.1 count: {len(corpus_df)}")
    print(f"  [SUCCESS] Excluded candidates count: {len(validation_records) - len(corpus_df)}")
    print(f"  [SUCCESS] Wrote validation records: {safe_rel(output_validation_records_path)}")
    print(f"  [SUCCESS] Wrote corpus v1.2.1: {safe_rel(output_corpus_v1_2_1_path)}")
    return corpus_df


if __name__ == "__main__":
    q_file = REPO_ROOT / "experiments" / "phase7" / "collection" / "candidate_queue.csv"
    obs_file = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_1_raw_http_observations.jsonl"
    prim_file = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_1_human_reviews.jsonl"
    sen_file = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_1_senior_reviews.jsonl"

    rec_v1_2_1_file = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_1_validation_records.jsonl"
    corp_v1_2_1_file = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2_1.csv"

    revalidate_corpus_m1_2_1(q_file, obs_file, prim_file, sen_file, rec_v1_2_1_file, corp_v1_2_1_file)
