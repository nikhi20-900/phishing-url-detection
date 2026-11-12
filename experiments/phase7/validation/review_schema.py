"""
Review Schema and Validation Model for Phase 7 Page-Level Authenticity.

Defines the validation statuses, evidence record schema, and eligibility rules
governing whether a candidate URL can be included in training or internal validation.
Enforces non-negotiable governance:
- Automated HTTP observations alone do not establish legitimacy.
- Every training candidate requires source provenance, human review, and explicit rationale.
- Risk-flagged or anomalous candidates require independent second review approval.
- Inaccessible or ambiguous candidates can never be approved.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ValidationStatus(str, Enum):
    PENDING = "pending"
    AUTOMATED_EVIDENCE_COLLECTED = "automated_evidence_collected"
    VERIFIED_PAGE = "verified_page"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    INACCESSIBLE = "inaccessible"


VALID_STATUSES = {s.value for s in ValidationStatus}
ELIGIBLE_STATUSES = {ValidationStatus.VERIFIED_PAGE.value}


@dataclass
class AutomatedObservations:
    http_status: Optional[int] = None
    redirect_chain: List[str] = field(default_factory=list)
    final_destination_url: Optional[str] = None
    tls_valid: Optional[bool] = None
    tls_cert_hostname: Optional[str] = None
    page_title: Optional[str] = None
    page_content_metadata: Dict[str, Any] = field(default_factory=dict)
    canonical_url: Optional[str] = None
    organization_relationship: Optional[str] = None
    collected_timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ValidationRecord:
    candidate_id: str
    original_url: str
    normalized_url: str
    registered_domain: str
    validation_status: str
    evidence_references: List[str]
    automated_observations: Dict[str, Any]
    redirect_chain: List[str]
    reviewer: str
    review_timestamp: str
    review_decision: str
    legitimacy_rationale: str
    validation_protocol_version: str = "phase7-validation-v1.0.0"
    second_reviewer: Optional[str] = None
    second_review_decision: Optional[str] = None
    risk_flags: Optional[str] = None
    has_risk_flags: bool = False

    def __post_init__(self):
        if self.validation_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid validation_status '{self.validation_status}'. "
                f"Must be one of: {sorted(VALID_STATUSES)}"
            )

        # Invariant: Inaccessible, rejected, or ambiguous records must NEVER have review_decision == 'approved'
        if self.validation_status in {
            ValidationStatus.INACCESSIBLE.value,
            ValidationStatus.AMBIGUOUS.value,
            ValidationStatus.REJECTED.value,
        }:
            if self.review_decision == "approved":
                raise ValueError(
                    f"Invalid record {self.candidate_id}: Cannot approve an "
                    f"'{self.validation_status}' record."
                )

        # Invariant: verified_page requires review_decision == 'approved' and documented rationale
        if self.validation_status == ValidationStatus.VERIFIED_PAGE.value:
            if self.review_decision != "approved":
                raise ValueError(
                    f"Record {self.candidate_id} has status 'verified_page' "
                    f"but review_decision is '{self.review_decision}' (expected 'approved')."
                )
            if not self.legitimacy_rationale or len(self.legitimacy_rationale.strip()) < 10:
                raise ValueError(
                    f"Record {self.candidate_id} verified without documented legitimacy_rationale."
                )

        # Invariant: Risk flagged or UGC records require second review approval
        if self.has_risk_flags and self.validation_status == ValidationStatus.VERIFIED_PAGE.value:
            if not self.second_reviewer or self.second_review_decision != "approved":
                raise ValueError(
                    f"Risk-flagged record {self.candidate_id} requires independent "
                    f"second reviewer approval before 'verified_page' status."
                )

    def is_training_eligible(self) -> bool:
        """Return True if and only if candidate meets all page-level validation criteria."""
        if self.validation_status != ValidationStatus.VERIFIED_PAGE.value:
            return False
        if self.review_decision != "approved":
            return False
        if not self.legitimacy_rationale or len(self.legitimacy_rationale.strip()) < 10:
            return False
        if self.has_risk_flags:
            if not self.second_reviewer or self.second_review_decision != "approved":
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
