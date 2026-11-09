"""Tests for the Phase 5 legitimate pilot corpus audit module.

Validates:
1. Registered-domain diversity and concentration calculation (HHI, entropy, Gini).
2. Risk-flag auditing and review rationale enforcement.
3. Structural feature extraction and comparison against PhiUSIIL baseline.
4. Zero-collision benchmark quarantine enforcement.
5. Provenance completeness and deterministic re-ingestion checks.
6. Full end-to-end execution of audit_pilot_corpus.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.generalization.audit.audit_pilot_corpus import (
    audit_domain_diversity,
    audit_provenance_and_reproducibility,
    audit_quarantine_overlap,
    audit_risk_flags_and_verification,
    audit_structural_feature_coverage,
    compute_concentration_metrics,
    main,
)


@pytest.fixture
def pilot_datasets_paths() -> dict[str, Path]:
    base = Path("experiments/generalization/datasets")
    return {
        "raw": base / "pilot_candidates_raw.csv",
        "queue": base / "pilot_review_queue.csv",
        "approved": base / "approved_pilot_corpus.csv",
    }


def test_concentration_metrics_calculation() -> None:
    from collections import Counter
    counts = Counter({"domain-a.org": 50, "domain-b.org": 30, "domain-c.org": 20})
    metrics = compute_concentration_metrics(counts, 100)
    assert metrics["unique_registered_domains"] == 3
    assert metrics["total_urls"] == 100
    assert metrics["top1_share"] == 0.50
    assert metrics["top5_share"] == 1.00
    # HHI = 50^2 + 30^2 + 20^2 = 2500 + 900 + 400 = 3800
    assert metrics["hhi_index"] == 3800.0


def test_audit_domain_diversity_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    result = audit_domain_diversity(approved_df)
    assert result["status"] == "PASS"
    assert result["metrics"]["unique_registered_domains"] >= 50
    assert result["metrics"]["top1_share"] < 0.10
    assert result["metrics"]["hhi_index"] < 1500


def test_audit_risk_flags_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    result = audit_risk_flags_and_verification(queue_df)
    assert result["status"] == "PASS"
    assert result["flagged_candidates_count"] > 0
    assert result["approved_rationale_coverage"] == 1.0
    assert len(result["rejected_cases_audited"]) > 0


def test_audit_structural_coverage_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    result = audit_structural_feature_coverage(approved_df)
    assert result["status"] == "PASS"
    dist = result["pilot_corpus_distribution"]
    assert dist["path_depth_ge_2_rate"] > 0.80
    assert dist["NoOfOtherSpecialCharsInURL_pct_ge_2"] > 0.80
    assert dist["SpacialCharRatioInURL_pct_ge_0_0785"] > 0.60


def test_audit_quarantine_overlap_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    approved_df = pd.read_csv(pilot_datasets_paths["approved"])
    result = audit_quarantine_overlap(approved_df)
    assert result["status"] == "PASS"
    assert result["exact_collisions_with_benchmark"] == 0
    assert result["normalized_collisions_with_benchmark"] == 0
    assert result["quarantine_guarantee_maintained"] is True


def test_audit_quarantine_detects_collision(tmp_path: Path) -> None:
    # Intentionally insert a known URL from external benchmark
    colliding_df = pd.DataFrame([{
        "url_exact": "https://www.mit.edu/",
        "registered_domain": "mit.edu",
    }])
    result = audit_quarantine_overlap(colliding_df)
    assert result["status"] == "FAIL"
    assert result["exact_collisions_with_benchmark"] > 0


def test_audit_provenance_reproducibility_live_pilot(pilot_datasets_paths: dict[str, Path]) -> None:
    queue_df = pd.read_csv(pilot_datasets_paths["queue"])
    result = audit_provenance_and_reproducibility(pilot_datasets_paths["raw"], queue_df)
    assert result["status"] == "PASS"
    assert len(result["missing_required_fields"]) == 0
    assert result["deterministic_reproducibility"]["deterministic_match"] is True


def test_audit_pilot_corpus_cli_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"

    test_args = [
        "audit_pilot_corpus.py",
        "--json-output",
        str(json_out),
        "--md-output",
        str(md_out),
    ]
    monkeypatch.setattr("sys.argv", test_args)
    exit_code = main()
    assert exit_code == 0
    assert json_out.exists()
    assert md_out.exists()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["overall_summary"]["overall_status"] == "PASS"
    assert data["overall_summary"]["approved_candidates"] == 602
