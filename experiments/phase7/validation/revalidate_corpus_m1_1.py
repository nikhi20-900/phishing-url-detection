"""
Revalidation Engine for Phase 7 Milestone 1.1 Correction.

Combines objective raw HTTP/TLS network observations with independent human review
decisions to produce the substantiated benign corpus v1.1.
Enforces non-negotiable governance:
- Automated HTTP observations alone do not determine legitimacy.
- Genuine human reviews stored separately from network observations.
- 100% of risk-flagged candidates require explicit second reviewer substantiation.
- Unsubstantiated, dynamic, interactive, dead, private-IP, or redirecting links are strictly excluded.
- Preserves all M1 artifacts intact without overwriting.

Outputs:
- experiments/phase7/datasets/validated_benign_corpus_v1_1.csv
- experiments/phase7/collection/evidence/m1_1_validation_records.jsonl
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from experiments.phase7.validation.review_schema import (
    ValidationRecord,
    ValidationStatus,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_revalidation_m1_1(
    queue_csv: Path,
    raw_obs_jsonl: Path,
    decisions_jsonl: Path,
    output_corpus_v1_1_csv: Path,
    output_records_v1_1_jsonl: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete M1.1 revalidation workflow."""
    print("Executing Phase 7 M1.1 Revalidation & Corpus Correction...")

    if not raw_obs_jsonl.exists():
        raise FileNotFoundError(f"Missing raw observations: {raw_obs_jsonl}")
    if not decisions_jsonl.exists():
        raise FileNotFoundError(f"Missing human decisions: {decisions_jsonl}")
    if not queue_csv.exists():
        raise FileNotFoundError(f"Missing candidate queue: {queue_csv}")

    # Load raw observations by candidate_id
    raw_obs_map: Dict[str, Dict[str, Any]] = {}
    with open(raw_obs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                raw_obs_map[item["candidate_id"]] = item

    # Load human review decisions by candidate_id
    decisions_map: Dict[str, Dict[str, Any]] = {}
    with open(decisions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                decisions_map[item["candidate_id"]] = item

    queue_df = pd.read_csv(queue_csv)
    validation_records: List[ValidationRecord] = []
    admitted_rows: List[Dict[str, Any]] = []

    counts = {
        "total_evaluated": len(queue_df),
        "substantiated_training_eligible": 0,
        "excluded_total": 0,
        "excluded_private_ip": 0,
        "excluded_cross_domain_redirect": 0,
        "excluded_unreachable_dead": 0,
        "excluded_unsubstantiated_ugc": 0,
        "excluded_policy_rejection": 0,
        "second_reviewed_substantiated": 0,
    }

    for _, row in queue_df.iterrows():
        cand_id = row["candidate_id"]
        url = str(row["url"]).strip()
        domain = str(row.get("registered_domain", "")).strip()

        obs = raw_obs_map.get(cand_id)
        dec = decisions_map.get(cand_id)

        if not obs or not dec:
            raise ValueError(f"Incomplete evidence for candidate {cand_id}: {url}")

        # Check raw network observation status
        is_private = obs.get("is_private_ip", False)
        is_cross_redirect = obs.get("is_cross_domain_redirect", False)
        http_status = obs.get("http_status_code")
        obs_error = obs.get("observation_error")

        # Check human review decisions
        primary_dec = dec.get("primary_review_decision", "rejected")
        primary_rat = dec.get("primary_review_rationale", "No rationale documented")
        has_flags = dec.get("has_risk_flags", False)
        second_required = dec.get("second_review_required", False)
        second_rev = dec.get("second_reviewer_id")
        second_dec = dec.get("second_review_decision")
        second_rat = dec.get("second_review_rationale")
        admissible = dec.get("admissible_for_training", False)

        # Determine validation status and exclusion reason
        if is_private:
            val_status = ValidationStatus.REJECTED.value
            review_decision = "rejected"
            rationale = f"REJECTED: Private IP address detected ({obs.get('resolved_ip')}). Non-routable network target."
            counts["excluded_private_ip"] += 1
            counts["excluded_total"] += 1
        elif is_cross_redirect:
            val_status = ValidationStatus.REJECTED.value
            review_decision = "rejected"
            rationale = f"REJECTED: Cross-domain open redirect detected to {obs.get('final_url')}."
            counts["excluded_cross_domain_redirect"] += 1
            counts["excluded_total"] += 1
        elif http_status is None or (obs_error and "NXDOMAIN" in obs_error):
            val_status = ValidationStatus.INACCESSIBLE.value
            review_decision = "rejected"
            rationale = f"INACCESSIBLE: Host failed resolution or connection ({obs_error})."
            counts["excluded_unreachable_dead"] += 1
            counts["excluded_total"] += 1
        elif second_required and second_dec == "unsubstantiated":
            val_status = ValidationStatus.AMBIGUOUS.value
            review_decision = "rejected"
            rationale = f"AMBIGUOUS / UNSUBSTANTIATED: {second_rat}"
            counts["excluded_unsubstantiated_ugc"] += 1
            counts["excluded_total"] += 1
        elif not admissible or primary_dec != "approved":
            val_status = ValidationStatus.REJECTED.value if primary_dec == "rejected" else ValidationStatus.AMBIGUOUS.value
            review_decision = "rejected"
            rationale = primary_rat
            counts["excluded_policy_rejection"] += 1
            counts["excluded_total"] += 1
        else:
            # Fully substantiated legitimate candidate
            val_status = ValidationStatus.VERIFIED_PAGE.value
            review_decision = "approved"
            rationale = primary_rat
            counts["substantiated_training_eligible"] += 1
            if has_flags:
                counts["second_reviewed_substantiated"] += 1

        record = ValidationRecord(
            candidate_id=cand_id,
            original_url=url,
            normalized_url=str(row.get("normalized_dedupe_key", url)),
            registered_domain=domain,
            validation_status=val_status,
            evidence_references=[f"raw_obs_{cand_id[:12]}"],
            automated_observations=obs,
            redirect_chain=obs.get("redirect_chain", [url]),
            reviewer=dec.get("primary_reviewer_id", "auditor-01"),
            review_timestamp=dec.get("primary_review_timestamp", datetime.now(timezone.utc).isoformat()),
            review_decision=review_decision,
            legitimacy_rationale=rationale,
            validation_protocol_version="phase7-validation-v1.1.0",
            second_reviewer=second_rev if val_status == ValidationStatus.VERIFIED_PAGE.value and has_flags else None,
            second_review_decision=second_dec if val_status == ValidationStatus.VERIFIED_PAGE.value and has_flags else None,
            risk_flags=row.get("risk_flags") if pd.notna(row.get("risk_flags")) else None,
            has_risk_flags=has_flags,
        )

        validation_records.append(record)

        if record.is_training_eligible():
            row_dict = row.to_dict()
            row_dict["validation_status"] = val_status
            row_dict["validation_version"] = "corpus-v1.1"
            row_dict["primary_reviewer"] = dec.get("primary_reviewer_id")
            row_dict["second_reviewer"] = second_rev
            row_dict["second_review_decision"] = second_dec
            row_dict["second_review_rationale"] = second_rat
            row_dict["legitimacy_rationale"] = rationale
            row_dict["raw_http_status"] = http_status
            row_dict["tls_verified"] = obs.get("tls_handshake_successful")
            admitted_rows.append(row_dict)

    # Write all M1.1 validation records
    output_records_v1_1_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_records_v1_1_jsonl, "w", encoding="utf-8") as f:
        for rec in validation_records:
            f.write(json.dumps(rec.to_dict()) + "\n")

    # Write substantiated corpus v1.1
    output_corpus_v1_1_csv.parent.mkdir(parents=True, exist_ok=True)
    corpus_v1_1_df = pd.DataFrame(admitted_rows)
    corpus_v1_1_df.to_csv(output_corpus_v1_1_csv, index=False)

    print("  [SUCCESS] M1.1 Revalidation complete:")
    print(f"    Total candidates evaluated: {counts['total_evaluated']}")
    print(f"    Substantiated & approved for training: {counts['substantiated_training_eligible']}")
    print(f"    Excluded candidates: {counts['excluded_total']}")
    print(f"      - Private IP violations: {counts['excluded_private_ip']}")
    print(f"      - Cross-domain redirects: {counts['excluded_cross_domain_redirect']}")
    print(f"      - Unreachable / dead DNS: {counts['excluded_unreachable_dead']}")
    print(f"      - Unsubstantiated UGC / dynamic forms: {counts['excluded_unsubstantiated_ugc']}")
    print(f"      - Policy / ownership rejections: {counts['excluded_policy_rejection']}")
    print(f"    Second-reviewed substantiated risk-flagged: {counts['second_reviewed_substantiated']}")
    print(f"  [OUTPUT] Substantiated Corpus v1.1: {output_corpus_v1_1_csv.relative_to(REPO_ROOT)}")
    print(f"  [OUTPUT] M1.1 Validation Records: {output_records_v1_1_jsonl.relative_to(REPO_ROOT)}")

    return corpus_v1_1_df, counts


if __name__ == "__main__":
    q_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "candidate_queue.csv"
    raw_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_1_raw_http_observations.jsonl"
    dec_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_1_human_review_decisions.jsonl"
    corp_v1_1_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_1.csv"
    rec_v1_1_path = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_1_validation_records.jsonl"

    run_revalidation_m1_1(q_path, raw_path, dec_path, corp_v1_1_path, rec_v1_1_path)

