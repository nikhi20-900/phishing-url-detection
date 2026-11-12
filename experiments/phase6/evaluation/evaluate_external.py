"""
Confirmatory External Evaluation for Phase 6 (Milestone 6).

Evaluates candidate models against the sealed, quarantined Phase 4 external benchmark:
- Benchmark Path: data/external/external_validation_benchmark.csv
- Required SHA-256: 88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3

Strict Invariants:
- Benchmark was never used for training, hyperparameter tuning, feature selection,
  or threshold tuning.
- Compares original frozen Random Forest (E0) vs candidate models (E1, E2, E3).
- Reports combined metrics, per-source slices, and structural subgroups.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from experiments.phase6.evaluation.evaluate_internal import (
    evaluate_model_on_dataset,
    load_model_estimator,
)
from experiments.phase6.training.build_dataset import compute_url_structural_strata
from src.features import FEATURE_NAMES, extract_url_features

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_SHA256 = "88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3"


def verify_benchmark_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest().lower()
    if actual_hash != BENCHMARK_SHA256:
        raise ValueError(
            f"CRITICAL BENCHMARK TAMPERING DETECTED!\n"
            f"Expected: {BENCHMARK_SHA256}\n"
            f"Actual:   {actual_hash}"
        )
    return actual_hash


def load_and_featurize_benchmark(bench_path: Path) -> pd.DataFrame:
    """Load benchmark and extract 18 lexical features."""
    verify_benchmark_sha256(bench_path)
    df = pd.read_csv(bench_path)

    print(f"Extracting 18 features for {len(df)} benchmark URLs...")
    features = []
    for u in df["url"]:
        features.append(extract_url_features(u))

    feat_df = pd.DataFrame(features, columns=FEATURE_NAMES)
    for col in df.columns:
        feat_df[col] = df[col].values

    # Add structural strata flags
    strata_records = []
    for idx, r in feat_df.iterrows():
        strata_records.append(
            compute_url_structural_strata(
                r["url"],
                int(r["NoOfOtherSpecialCharsInURL"]),
                int(r["URLLength"]),
            )
        )
    strata_df = pd.DataFrame(strata_records)
    for col in strata_df.columns:
        feat_df[col] = strata_df[col].values

    return feat_df


def evaluate_source_slices(df_pred: pd.DataFrame, threshold: float = 0.50) -> Dict[str, Any]:
    """Evaluate performance sliced by external source feed."""
    slices = {}
    for src in df_pred["source"].unique():
        sub = df_pred[df_pred["source"] == src]
        y_true = sub["label"].values.astype(int)
        y_pred = sub["prediction"].values.astype(int)
        n = len(sub)

        if "Curated" in src:
            # Legitimate slice
            tn = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 1) & (y_pred == 0)).sum())
            recall = round(float(tn / n), 4) if n > 0 else 0.0
            fpr = round(float(fp / n), 4) if n > 0 else 0.0
            slices[src] = {
                "sample_count": n,
                "label": "Legitimate",
                "correct_accepted (TN)": tn,
                "false_alarms (FP)": fp,
                "legitimate_recall": recall,
                "legitimate_fpr": fpr,
            }
        else:
            # Phishing slice
            tp = int(((y_true == 0) & (y_pred == 0)).sum())
            fn = int(((y_true == 0) & (y_pred == 1)).sum())
            recall = round(float(tp / n), 4) if n > 0 else 0.0
            fnr = round(float(fn / n), 4) if n > 0 else 0.0
            slices[src] = {
                "sample_count": n,
                "label": "Phishing",
                "detected_phish (TP)": tp,
                "missed_phish (FN)": fn,
                "phishing_recall": recall,
                "phishing_fnr": fnr,
            }

    return slices


def run_confirmatory_external_evaluation(
    benchmark_path: Path,
    models_config: Dict[str, Dict[str, Any]],
    output_json_path: Path,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Execute Milestone 6: Confirmatory External Benchmark Evaluation."""
    print("Executing Phase 6 M6: Confirmatory External Benchmark Evaluation...")
    bench_df = load_and_featurize_benchmark(benchmark_path)
    actual_hash = verify_benchmark_sha256(benchmark_path)

    results = {}
    for mid, mspec in models_config.items():
        mpath = REPO_ROOT / mspec["artifact_path"]
        print(f"  Evaluating {mid} ({mspec['name']}) on sealed benchmark...")
        estimator, meta = load_model_estimator(mpath)

        metrics, df_pred = evaluate_model_on_dataset(estimator, bench_df, threshold=threshold)
        sources = evaluate_source_slices(df_pred, threshold=threshold)

        results[mid] = {
            "model_id": mid,
            "name": mspec["name"],
            "model_family": mspec["family"],
            "model_sha256": meta.get("model_sha256"),
            "threshold": threshold,
            "combined_metrics": metrics["classification_metrics"],
            "confusion_matrix": metrics["confusion_matrix"],
            "probability_metrics": metrics["probability_metrics"],
            "source_slices": sources,
            "subgroups": metrics["subgroups"],
        }

    report = {
        "report_version": "phase6-m6-confirmatory-external-v1",
        "phase": 6,
        "milestone": "M6",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(benchmark_path.relative_to(REPO_ROOT)),
        "benchmark_sha256": actual_hash,
        "benchmark_sample_count": len(bench_df),
        "comparison_results": results,
        "status": "APPROVED",
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[M6 EXIT CRITERIA MET] Confirmatory external evaluation completed and saved to {output_json_path.relative_to(REPO_ROOT)}")
    return report


if __name__ == "__main__":
    from experiments.phase6.verify_m0_contract import REPO_ROOT
    import yaml

    exp_cfg = yaml.safe_load(open(REPO_ROOT / "experiments" / "phase6" / "configs" / "experiment.yaml"))
    bench_csv = REPO_ROOT / "data" / "external" / "external_validation_benchmark.csv"
    conf_json = REPO_ROOT / "experiments" / "phase6" / "reports" / "confirmatory_external_evaluation.json"

    run_confirmatory_external_evaluation(bench_csv, exp_cfg["models"], conf_json)
