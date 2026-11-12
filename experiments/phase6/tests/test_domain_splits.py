"""
Tests for Phase 6 Milestone 2: Dataset Construction and Domain Separation.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from experiments.phase6.training.build_dataset import (
    REPO_ROOT,
    compute_file_sha256,
)


@pytest.fixture(scope="module")
def split_manifest():
    manifest_path = REPO_ROOT / "experiments" / "phase6" / "datasets" / "manifests" / "split_manifest.json"
    assert manifest_path.exists(), "Split manifest missing"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_zero_domain_leakage(split_manifest):
    """Verify that train, validation, and test partitions share zero registered domains."""
    train_path = REPO_ROOT / split_manifest["partitions"]["train"]["path"]
    val_path = REPO_ROOT / split_manifest["partitions"]["val"]["path"]
    test_path = REPO_ROOT / split_manifest["partitions"]["internal_test"]["path"]

    train_df = pd.read_csv(train_path, usecols=["registered_domain"])
    val_df = pd.read_csv(val_path, usecols=["registered_domain"])
    test_df = pd.read_csv(test_path, usecols=["registered_domain"])

    train_doms = set(train_df["registered_domain"])
    val_doms = set(val_df["registered_domain"])
    test_doms = set(test_df["registered_domain"])

    assert len(train_doms & val_doms) == 0, f"Domain leakage between Train and Val: {len(train_doms & val_doms)}"
    assert len(train_doms & test_doms) == 0, f"Domain leakage between Train and Test: {len(train_doms & test_doms)}"
    assert len(val_doms & test_doms) == 0, f"Domain leakage between Val and Test: {len(val_doms & test_doms)}"


def test_split_hashes_match_manifest(split_manifest):
    """Verify that disk CSV hashes match the SHA-256 recorded in the manifest."""
    for pname in ["train", "val", "internal_test"]:
        part_info = split_manifest["partitions"][pname]
        fpath = REPO_ROOT / part_info["path"]
        assert fpath.exists()
        assert compute_file_sha256(fpath) == part_info["sha256"]


def test_label_mapping_in_partitions(split_manifest):
    """Verify that label column contains only 0 and 1 across all partitions."""
    for pname in ["train", "val", "internal_test"]:
        fpath = REPO_ROOT / split_manifest["partitions"][pname]["path"]
        df = pd.read_csv(fpath, usecols=["label"])
        labels = set(df["label"].unique())
        assert labels.issubset({0, 1}), f"Invalid labels in {pname}: {labels}"


def test_structural_strata_presence(split_manifest):
    """Verify that structural strata fields are computed and present in manifest and datasets."""
    strata_keys = [
        "homepage",
        "single_level_path",
        "multi_level_path",
        "trailing_slash",
        "query_bearing",
        "fragment_bearing",
        "subdomain_bearing",
        "high_special_char",
        "short_url",
        "long_url",
    ]
    for pname in ["train", "val", "internal_test"]:
        fpath = REPO_ROOT / split_manifest["partitions"][pname]["path"]
        df = pd.read_csv(fpath, nrows=5)
        for k in strata_keys:
            assert k in df.columns, f"Strata column {k} missing from {pname}"

        strata_manifest = split_manifest["partitions"][pname]["strata_breakdown"]
        for k in strata_keys:
            assert k in strata_manifest, f"Strata {k} missing from {pname} manifest breakdown"
            assert "count" in strata_manifest[k]
            assert "percent" in strata_manifest[k]
