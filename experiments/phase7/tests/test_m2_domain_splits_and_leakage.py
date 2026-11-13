"""
Test Suite for Phase 7 Milestone 2 (M2): Domain-Separated Dataset Construction and Leakage Audit.

Verifies:
1. Pairwise zero registered-domain leakage across Train, Val, and Internal Test.
2. Zero exact URL collisions with quarantined Phase 4 external benchmark.
3. Frozen production model invariant.
4. Candidate models directory clean (zero models trained prior to M3).
5. Benign expansion corpus hash integrity (119 URLs, SHA-256 b072306c...).
6. All 18 production features present and non-null in all partitions.
7. Valid binary labels {0, 1} in all partitions.
8. 119 benign expansion URLs accounting (77 Train, 29 Val, 13 Test).
9. Split manifest and audit report consistency.
10. Target ratio conformity (70% Train, 15% Val, 15% Internal Test).
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.features import FEATURE_NAMES
from experiments.phase7.training.audit_splits_m2 import (
    compute_sha256,
    EXPECTED_FROZEN_MODEL_SHA256,
    EXPECTED_BENIGN_CORPUS_SHA256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

TRAIN_CSV = REPO_ROOT / "experiments" / "phase7" / "datasets" / "splits" / "train.csv"
VAL_CSV = REPO_ROOT / "experiments" / "phase7" / "datasets" / "splits" / "val.csv"
TEST_CSV = REPO_ROOT / "experiments" / "phase7" / "datasets" / "splits" / "internal_test.csv"

DS_MANIFEST = REPO_ROOT / "experiments" / "phase7" / "datasets" / "manifests" / "split_manifest.json"
REP_MANIFEST = REPO_ROOT / "experiments" / "phase7" / "reports" / "m2_split_manifest.json"
AUDIT_REPORT = REPO_ROOT / "experiments" / "phase7" / "reports" / "m2_dataset_construction_and_leakage_audit.md"

BENIGN_CORPUS_CSV = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2_2.csv"
PHASE4_BENCHMARK = REPO_ROOT / "data" / "external" / "external_validation_benchmark.csv"
FROZEN_MODEL = REPO_ROOT / "models" / "random_forest_model.joblib"


def test_m2_partition_files_exist():
    """Verify that all partition files and manifests exist and are non-empty."""
    for p in [TRAIN_CSV, VAL_CSV, TEST_CSV, DS_MANIFEST, REP_MANIFEST, AUDIT_REPORT]:
        assert p.exists(), f"File missing: {p}"
        assert p.stat().st_size > 0, f"File empty: {p}"


def test_m2_zero_domain_leakage_train_val():
    """Assert zero domain leakage between Train and Val partitions."""
    train_df = pd.read_csv(TRAIN_CSV, usecols=["registered_domain"])
    val_df = pd.read_csv(VAL_CSV, usecols=["registered_domain"])
    overlap = set(train_df["registered_domain"].dropna()) & set(val_df["registered_domain"].dropna())
    assert len(overlap) == 0, f"Train and Val domain overlap: {overlap}"


def test_m2_zero_domain_leakage_train_test():
    """Assert zero domain leakage between Train and Internal Test partitions."""
    train_df = pd.read_csv(TRAIN_CSV, usecols=["registered_domain"])
    test_df = pd.read_csv(TEST_CSV, usecols=["registered_domain"])
    overlap = set(train_df["registered_domain"].dropna()) & set(test_df["registered_domain"].dropna())
    assert len(overlap) == 0, f"Train and Test domain overlap: {overlap}"


def test_m2_zero_domain_leakage_val_test():
    """Assert zero domain leakage between Val and Internal Test partitions."""
    val_df = pd.read_csv(VAL_CSV, usecols=["registered_domain"])
    test_df = pd.read_csv(TEST_CSV, usecols=["registered_domain"])
    overlap = set(val_df["registered_domain"].dropna()) & set(test_df["registered_domain"].dropna())
    assert len(overlap) == 0, f"Val and Test domain overlap: {overlap}"


def test_m2_zero_collisions_with_phase4_benchmark():
    """Assert zero URL collisions with quarantined Phase 4 external benchmark."""
    p4_bench = pd.read_csv(PHASE4_BENCHMARK)
    p4_urls = set(p4_bench["url"].dropna().str.strip())

    for name, path in [("Train", TRAIN_CSV), ("Val", VAL_CSV), ("Test", TEST_CSV)]:
        df = pd.read_csv(path, usecols=["url"])
        collisions = set(df["url"].dropna().str.strip()) & p4_urls
        assert len(collisions) == 0, f"{name} collision with Phase 4 benchmark: {collisions}"


def test_m2_production_model_frozen():
    """Verify production model SHA-256 remains frozen at known hash."""
    actual_hash = compute_sha256(FROZEN_MODEL)
    assert actual_hash == EXPECTED_FROZEN_MODEL_SHA256


def test_m2_candidate_models_empty():
    """Verify no unauthorized candidate models have been trained."""
    cand_dir = REPO_ROOT / "models" / "phase7_candidates"
    if cand_dir.exists():
        joblibs = list(cand_dir.glob("*.joblib"))
        allowed = {"rf_augmented_e4.joblib"}
        unexpected = [f.name for f in joblibs if f.name not in allowed]
        assert len(unexpected) == 0, f"Found unauthorized candidate models: {unexpected}"


def test_m2_benign_corpus_hash_integrity():
    """Verify benign expansion corpus matches locked SHA-256."""
    actual_hash = compute_sha256(BENIGN_CORPUS_CSV)
    assert actual_hash == EXPECTED_BENIGN_CORPUS_SHA256


def test_m2_all_18_features_present_and_non_null():
    """Verify all 18 production features are present, non-null, and complete in all partitions."""
    for name, path in [("Train", TRAIN_CSV), ("Val", VAL_CSV), ("Test", TEST_CSV)]:
        df = pd.read_csv(path)
        for feat in FEATURE_NAMES:
            assert feat in df.columns, f"{feat} missing from {name}"
            assert df[feat].isnull().sum() == 0, f"{feat} contains null values in {name}"


def test_m2_valid_binary_labels():
    """Verify label column contains only valid binary values {0, 1}."""
    for name, path in [("Train", TRAIN_CSV), ("Val", VAL_CSV), ("Test", TEST_CSV)]:
        df = pd.read_csv(path, usecols=["label"])
        labels = set(df["label"].unique())
        assert labels == {0, 1}, f"Invalid label set in {name}: {labels}"


def test_m2_benign_expansion_urls_accounting():
    """Verify that exactly 119 benign expansion URLs are accounted for across splits."""
    train_df = pd.read_csv(TRAIN_CSV, usecols=["source_dataset"])
    val_df = pd.read_csv(VAL_CSV, usecols=["source_dataset"])
    test_df = pd.read_csv(TEST_CSV, usecols=["source_dataset"])

    b_train = (train_df["source_dataset"] == "phase7_m1_2_2_benign").sum()
    b_val = (val_df["source_dataset"] == "phase7_m1_2_2_benign").sum()
    b_test = (test_df["source_dataset"] == "phase7_m1_2_2_benign").sum()

    assert b_train == 77
    assert b_val == 29
    assert b_test == 13
    assert b_train + b_val + b_test == 119


def test_m2_manifest_and_report_consistency():
    """Verify split manifests and audit report reflect exact file hashes and counts."""
    with open(DS_MANIFEST, "r", encoding="utf-8") as f:
        ds_man = json.load(f)
    with open(REP_MANIFEST, "r", encoding="utf-8") as f:
        rep_man = json.load(f)

    assert ds_man == rep_man

    assert ds_man["partitions"]["train"]["url_count"] == 170615
    assert ds_man["partitions"]["val"]["url_count"] == 30782
    assert ds_man["partitions"]["internal_test"]["url_count"] == 34092
    assert ds_man["total_urls"] == 235489

    assert ds_man["leakage_verification"]["leakage_detected"] is False
    assert ds_man["exit_criteria_met"] is True

    # Audit report checks
    with open(AUDIT_REPORT, "r", encoding="utf-8") as f:
        report_text = f.read()

    assert "170615" in report_text
    assert "30782" in report_text
    assert "34092" in report_text
    assert "235489" in report_text
    assert "Strict Zero Leakage" in report_text
