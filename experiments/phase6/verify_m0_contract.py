"""
M0 Research Freeze and Protocol Verification for Phase 6.

Verifies non-negotiable research invariants:
- Frozen Random Forest reference model SHA-256
- Quarantined Phase 4 external benchmark SHA-256
- Immutable PhiUSIIL source dataset SHA-256
- Phase 5 approved pilot corpus & review queue SHA-256
- Model prediction reproducibility
- Historical evaluation artifacts integrity
- Machine-readable experiment contract creation
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_HASHES = {
    "frozen_model": {
        "path": REPO_ROOT / "models" / "random_forest_model.joblib",
        "expected_sha256": "95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb",
    },
    "phase4_benchmark": {
        "path": REPO_ROOT / "data" / "external" / "external_validation_benchmark.csv",
        "expected_sha256": "88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3",
    },
    "phiusiil_source": {
        "path": REPO_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv",
        "expected_sha256": "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6",
    },
    "phase5_approved_pilot": {
        "path": REPO_ROOT / "experiments" / "generalization" / "datasets" / "approved_pilot_corpus.csv",
        "expected_sha256": "cca9ef3f8b265c4c3b0395c94fc1c4c40442f90a5381713e33511ccc30fe89cb",
    },
    "phase5_review_queue": {
        "path": REPO_ROOT / "experiments" / "generalization" / "datasets" / "pilot_review_queue.csv",
        "expected_sha256": "e4ea2a206f477eb9902378ef599d80024395a09bd9cc2dd6dc527d14b05da360",
    },
}

REQUIRED_HISTORICAL_REPORTS = [
    REPO_ROOT / "reports" / "phase_3_evaluation_and_leakage_audit.md",
    REPO_ROOT / "reports" / "domain_separated_evaluation_results.json",
    REPO_ROOT / "reports" / "phase_4_external_validation_and_fn_analysis.md",
    REPO_ROOT / "reports" / "external_validation_results.json",
    REPO_ROOT / "reports" / "phase_4_audit_addendum.md",
    REPO_ROOT / "experiments" / "reports" / "issue_4_validation_report.md",
    REPO_ROOT / "experiments" / "generalization" / "reports" / "pilot_corpus_audit.md",
    REPO_ROOT / "experiments" / "generalization" / "reports" / "audit_manifest.json",
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def get_git_info() -> Dict[str, Any]:
    """Retrieve current Git metadata safely."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        branch = "unknown"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
        is_clean = len(status_out) == 0
    except Exception:
        status_out = "error"
        is_clean = False

    return {
        "commit": commit,
        "branch": branch,
        "clean": is_clean,
        "status_summary": "clean" if is_clean else f"{len(status_out.splitlines())} modified/untracked files",
    }


def verify_frozen_model_prediction() -> Dict[str, Any]:
    """Verify that the frozen model loads and produces expected deterministic predictions."""
    from src.features import extract_url_features
    from src.predict import analyze_url

    test_urls = [
        ("https://www.google.com", 1),  # Legitimate
        ("http://paypal-verification-account-security-alert.com/login", 0),  # Obvious Phishing
    ]

    results = []
    for url, expected_class in test_urls:
        analysis = analyze_url(url)
        predicted_class = 0 if analysis["prediction"] == "Phishing" else 1
        results.append({
            "url": url,
            "prediction": analysis["prediction"],
            "phishing_probability": analysis["prob_phishing"],
            "matches_expected": predicted_class == expected_class,
        })
        if predicted_class != expected_class:
            raise ValueError(f"Frozen model prediction mismatch for {url}: got {analysis['prediction']}")

    return {"status": "PASSED", "verifications": results}


def run_m0_verification() -> Dict[str, Any]:
    """Execute all M0 verifications and compile manifest."""
    print("Executing Phase 6 M0: Research Freeze & Protocol Verification...")

    # 1. Verify frozen hashes
    hash_results = {}
    for name, spec in EXPECTED_HASHES.items():
        actual_hash = compute_sha256(spec["path"])
        matches = actual_hash == spec["expected_sha256"]
        try:
            rel_path = str(spec["path"].relative_to(REPO_ROOT))
        except ValueError:
            rel_path = str(spec["path"])
        hash_results[name] = {
            "path": rel_path,
            "expected_sha256": spec["expected_sha256"],
            "actual_sha256": actual_hash,
            "matches": matches,
        }
        if not matches:
            raise ValueError(
                f"FATAL INVARIANT VIOLATION: {name} hash mismatch!\n"
                f"Expected: {spec['expected_sha256']}\n"
                f"Actual:   {actual_hash}"
            )
        print(f"  [PASS] {name} hash verified: {actual_hash[:16]}...")

    # 2. Verify historical reports exist
    report_checks = {}
    for rpath in REQUIRED_HISTORICAL_REPORTS:
        rel = str(rpath.relative_to(REPO_ROOT))
        exists = rpath.exists()
        report_checks[rel] = {"exists": exists, "size_bytes": rpath.stat().st_size if exists else 0}
        if not exists:
            raise FileNotFoundError(f"Missing required historical report: {rel}")
        print(f"  [PASS] Historical artifact verified: {rel}")

    # 3. Verify configs
    exp_config_path = REPO_ROOT / "experiments" / "phase6" / "configs" / "experiment.yaml"
    split_config_path = REPO_ROOT / "experiments" / "phase6" / "configs" / "split_config.yaml"
    if not exp_config_path.exists():
        raise FileNotFoundError(f"Missing experiment.yaml at {exp_config_path}")
    if not split_config_path.exists():
        raise FileNotFoundError(f"Missing split_config.yaml at {split_config_path}")

    with open(exp_config_path, "r", encoding="utf-8") as f:
        exp_config = yaml.safe_load(f)
    with open(split_config_path, "r", encoding="utf-8") as f:
        split_config = yaml.safe_load(f)

    # 4. Verify model prediction reproducibility
    pred_verification = verify_frozen_model_prediction()
    print("  [PASS] Frozen model prediction reproducibility confirmed.")

    # 5. Environment metadata
    git_info = get_git_info()
    import sklearn
    import xgboost

    manifest = {
        "contract_version": "phase6-m0-contract-v1",
        "phase": 6,
        "milestone": "M0",
        "status": "APPROVED",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_metadata": git_info,
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "joblib_version": joblib.__version__,
            "pyyaml_version": yaml.__version__,
        },
        "artifact_hashes": hash_results,
        "historical_artifacts": report_checks,
        "frozen_model_verification": pred_verification,
        "experiment_configuration": exp_config,
        "split_configuration": split_config,
        "exit_criteria_met": True,
    }

    output_manifest_path = REPO_ROOT / "experiments" / "phase6" / "reports" / "experiment_manifest.json"
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[M0 EXIT CRITERION MET] Experiment manifest generated at {output_manifest_path.relative_to(REPO_ROOT)}")
    return manifest


if __name__ == "__main__":
    run_m0_verification()
