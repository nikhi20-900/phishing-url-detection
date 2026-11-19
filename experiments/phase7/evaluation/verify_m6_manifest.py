"""
Stage 0 Pre-Flight Manifest Verification Script for Phase 7 Milestone 6.

Validates all cryptographic hashes, model invariants, dataset partitions,
feature pipeline definitions, environment dependencies, and TreeSHAP additivity.
Produces a comprehensive Stage 0 Manifest Verification Report.
"""

import hashlib
import json
import os
import sys

# Ensure MPLCONFIGDIR is set to a writable directory before any third-party imports
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m6_operational_readiness_manifest.json"


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def run_stage_0_verification() -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 7 MILESTONE 6: STAGE 0 PRE-FLIGHT MANIFEST VERIFICATION")
    print("=" * 80)

    assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    results: Dict[str, Any] = {
        "manifest_version": manifest["manifest_version"],
        "status": "PASS",
        "checks": [],
    }

    # 1. Models Verification
    for model_key, model_meta in manifest["models"].items():
        model_path = REPO_ROOT / model_meta["path"]
        assert model_path.exists(), f"Model file missing: {model_path}"
        actual_sha = compute_sha256(model_path)
        expected_sha = model_meta["sha256"]
        match = (actual_sha == expected_sha)
        print(f"[{'PASS' if match else 'FAIL'}] Model {model_key}: {actual_sha[:16]}... (Expected: {expected_sha[:16]}...)")
        assert match, f"SHA-256 mismatch for {model_key}!"
        results["checks"].append({
            "check": f"Model SHA-256: {model_key}",
            "expected": expected_sha,
            "actual": actual_sha,
            "passed": match,
        })

    # 2. Benchmarks Verification
    import pandas as pd
    for bench_key, bench_meta in manifest["benchmarks"].items():
        bench_path = REPO_ROOT / bench_meta["path"]
        assert bench_path.exists(), f"Benchmark file missing: {bench_path}"
        actual_sha = compute_sha256(bench_path)
        expected_sha = bench_meta["sha256"]
        match_sha = (actual_sha == expected_sha)
        print(f"[{'PASS' if match_sha else 'FAIL'}] Benchmark {bench_key} SHA: {actual_sha[:16]}...")
        assert match_sha, f"SHA-256 mismatch for benchmark {bench_key}!"

        df = pd.read_csv(bench_path)
        actual_rows = len(df)
        expected_rows = bench_meta["sample_count"]
        assert actual_rows == expected_rows, f"Row count mismatch for {bench_key}: {actual_rows} vs {expected_rows}"
        print(f"       Rows: {actual_rows} (Expected: {expected_rows})")

        results["checks"].append({
            "check": f"Benchmark: {bench_key}",
            "sha256": actual_sha,
            "sample_count": actual_rows,
            "passed": True,
        })

    # 3. Training Partition Verification
    train_meta = manifest["training_data"]["phase7_train_split"]
    train_path = REPO_ROOT / train_meta["path"]
    assert train_path.exists(), f"Train file missing: {train_path}"
    actual_train_sha = compute_sha256(train_path)
    assert actual_train_sha == train_meta["sha256"]
    train_df = pd.read_csv(train_path)
    assert len(train_df) == train_meta["total_samples"]
    assert int((train_df["label"] == 1).sum()) == train_meta["legitimate_count"]
    assert int((train_df["label"] == 0).sum()) == train_meta["phishing_count"]
    print(f"[PASS] Training Partition: {len(train_df)} rows (Legit: {train_meta['legitimate_count']}, Phish: {train_meta['phishing_count']})")
    results["checks"].append({
        "check": "Training Partition Integrity",
        "sha256": actual_train_sha,
        "total_samples": len(train_df),
        "passed": True,
    })

    # 4. Feature Pipeline Code & Schema
    feat_src_meta = manifest["feature_pipeline"]["source_code"]
    feat_src_path = REPO_ROOT / feat_src_meta["path"]
    actual_feat_src_sha = compute_sha256(feat_src_path)
    assert actual_feat_src_sha == feat_src_meta["sha256"]
    from src.features import FEATURE_NAMES
    assert FEATURE_NAMES == manifest["feature_pipeline"]["ordered_features"]
    print(f"[PASS] Feature Pipeline: 18 features locked, source SHA {actual_feat_src_sha[:16]}...")
    results["checks"].append({
        "check": "Feature Pipeline Schema & Source Code Lock",
        "sha256": actual_feat_src_sha,
        "feature_count": len(FEATURE_NAMES),
        "passed": True,
    })

    # 5. Environment & Dependencies Verification
    env_meta = manifest["environment_metadata"]
    import sklearn, joblib, numpy as np, scipy, shap
    deps = env_meta["dependencies"]
    assert sklearn.__version__ == deps["scikit-learn"]
    assert joblib.__version__ == deps["joblib"]
    assert np.__version__ == deps["numpy"]
    assert pd.__version__ == deps["pandas"]
    assert scipy.__version__ == deps["scipy"]
    assert shap.__version__ == deps["shap"]
    print(f"[PASS] Environment Dependencies: All 6 core libraries match exact manifest versions.")
    results["checks"].append({
        "check": "Environment Dependencies Exact Lock",
        "passed": True,
    })

    # 6. SHAP Additivity Verification
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    from experiments.phase7.evaluation.m6_diagnostics_utils import verify_shap_probability_additivity
    from src.features import extract_url_features

    bench_m5 = pd.read_csv(REPO_ROOT / manifest["benchmarks"]["m5_1_qualified_external"]["path"])
    sample_urls = bench_m5["url"].iloc[:25]
    X_sample = pd.DataFrame([extract_url_features(u) for u in sample_urls])[FEATURE_NAMES].values

    for model_key, model_meta in manifest["models"].items():
        model_dict = joblib.load(REPO_ROOT / model_meta["path"])
        clf = model_dict["model"] if isinstance(model_dict, dict) else model_dict
        res = verify_shap_probability_additivity(clf, X_sample, tolerance=manifest["shap_configuration"]["numerical_additivity_tolerance"])
        print(f"[PASS] TreeSHAP Additivity for {model_key}: Max Error = {res['max_absolute_error']:.2e} <= {res['tolerance']}")
        results["checks"].append({
            "check": f"TreeSHAP Additivity: {model_key}",
            "max_absolute_error": res["max_absolute_error"],
            "passed": True,
        })

    print("=" * 80)
    print("ALL STAGE 0 PRE-FLIGHT CHECKS PASSED SUCCESSFULLY.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_stage_0_verification()
