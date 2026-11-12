"""
Subgroup Analysis Module for Phase 6.

Evaluates model performance across structural strata:
- Homepage URLs
- Single-level paths
- Multi-level paths
- Trailing-slash URLs
- Query-bearing URLs
- Fragment-bearing URLs
- Subdomain-bearing URLs
- High special characters (>= 2)
- Short URLs (<= 35)
- Long URLs (> 75)
- Unseen registered domains
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, accuracy_score

STRATA_KEYS = [
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


def evaluate_stratum_metrics(sub_df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate detailed metrics for a subset of data."""
    n = len(sub_df)
    if n == 0:
        return {
            "sample_count": 0,
            "phishing_count": 0,
            "legitimate_count": 0,
            "legitimate_recall": None,
            "legitimate_fpr": None,
            "phishing_recall": None,
            "phishing_precision": None,
            "accuracy": None,
        }

    y_true = sub_df["label"].values
    prob_phish = sub_df["prob_phishing"].values
    y_pred = (prob_phish < threshold).astype(int)  # 0 if prob_phish >= threshold else 1

    n_phish = int((y_true == 0).sum())
    n_legit = int((y_true == 1).sum())

    acc = float(accuracy_score(y_true, y_pred))

    # Phishing metrics (class 0 treated as positive for detection)
    # y_pred_phish is 1 where predicted 0
    y_true_phish = (y_true == 0).astype(int)
    y_pred_phish = (y_pred == 0).astype(int)

    if n_phish > 0:
        phish_recall = float(recall_score(y_true_phish, y_pred_phish, zero_division=0))
    else:
        phish_recall = None

    if y_pred_phish.sum() > 0:
        phish_prec = float(precision_score(y_true_phish, y_pred_phish, zero_division=0))
    else:
        phish_prec = None

    # Legitimate metrics (class 1)
    y_true_legit = (y_true == 1).astype(int)
    y_pred_legit = (y_pred == 1).astype(int)

    if n_legit > 0:
        legit_recall = float(recall_score(y_true_legit, y_pred_legit, zero_division=0))
        # FPR on legitimate URLs: false alarms = legit URLs flagged as phish / total legit URLs
        fp_legit_flagged_as_phish = int(((y_true == 1) & (y_pred == 0)).sum())
        legit_fpr = float(fp_legit_flagged_as_phish / n_legit)
    else:
        legit_recall = None
        legit_fpr = None

    return {
        "sample_count": n,
        "phishing_count": n_phish,
        "legitimate_count": n_legit,
        "legitimate_recall": round(legit_recall, 4) if legit_recall is not None else None,
        "legitimate_fpr": round(legit_fpr, 4) if legit_fpr is not None else None,
        "phishing_recall": round(phish_recall, 4) if phish_recall is not None else None,
        "phishing_precision": round(phish_prec, 4) if phish_prec is not None else None,
        "accuracy": round(acc, 4),
    }


def compute_all_subgroup_metrics(df_with_predictions: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute metrics across all structural strata."""
    results = {}
    for stratum in STRATA_KEYS:
        if stratum in df_with_predictions.columns:
            mask = df_with_predictions[stratum] == True
            sub = df_with_predictions[mask]
            results[stratum] = evaluate_stratum_metrics(sub, threshold=threshold)

    # Pilot vs PhiUSIIL source stratum
    if "source_dataset" in df_with_predictions.columns:
        for src in ["phiusiil", "phase6_validated_pilot"]:
            sub = df_with_predictions[df_with_predictions["source_dataset"] == src]
            results[f"source_{src}"] = evaluate_stratum_metrics(sub, threshold=threshold)

    return results
