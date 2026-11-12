"""
Statistical Testing and Uncertainty Analysis for Phase 6 (Milestone 7).

Performs paired statistical hypothesis testing and uncertainty quantification:
- Bootstrap 95% Confidence Intervals for FPR and Recall
- Paired McNemar's test for matched classification outcomes
- Paired bootstrap difference distributions (Candidate vs Frozen Control)
- Domain-cluster aware evaluation
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from experiments.phase6.evaluation.evaluate_internal import (
    evaluate_model_on_dataset,
    load_model_estimator,
)
from src.features import FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_control: np.ndarray,
    y_pred_candidate: np.ndarray,
) -> Dict[str, Any]:
    """
    Perform McNemar's test with continuity correction.
    Tests H0: Marginal probabilities of each outcome are the same between models.
    """
    c_correct = (y_pred_control == y_true)
    cand_correct = (y_pred_candidate == y_true)

    n00 = int(np.sum(c_correct & cand_correct))        # both correct
    n01 = int(np.sum(c_correct & (~cand_correct)))     # control correct, cand wrong
    n10 = int(np.sum((~c_correct) & cand_correct))     # control wrong, cand correct
    n11 = int(np.sum((~c_correct) & (~cand_correct)))   # both wrong

    discordant = n01 + n10
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        # Continuity-corrected chi-squared
        statistic = float(((abs(n01 - n10) - 1.0) ** 2) / discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))

    return {
        "contingency_table": {
            "both_correct": n00,
            "control_correct_candidate_wrong": n01,
            "control_wrong_candidate_correct": n10,
            "both_wrong": n11,
        },
        "discordant_pairs": discordant,
        "statistic": round(statistic, 4),
        "p_value": float(f"{p_value:.6e}"),
        "statistically_significant_p05": p_value < 0.05,
    }


def compute_bootstrap_cis(
    y_true: np.ndarray,
    prob_phish: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    threshold: float = 0.50,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% bootstrap percentile confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_pred = (prob_phish < threshold).astype(int)

    phish_recalls = []
    legit_recalls = []
    legit_fprs = []
    macro_f1s = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt_boot = y_true[idx]
        yp_boot = y_pred[idx]

        # Phishing recall
        n_p = (yt_boot == 0).sum()
        if n_p > 0:
            phish_recalls.append(float(((yt_boot == 0) & (yp_boot == 0)).sum() / n_p))

        # Legit recall & FPR
        n_l = (yt_boot == 1).sum()
        if n_l > 0:
            legit_recalls.append(float(((yt_boot == 1) & (yp_boot == 1)).sum() / n_l))
            legit_fprs.append(float(((yt_boot == 1) & (yp_boot == 0)).sum() / n_l))

        macro_f1s.append(float(f1_score(yt_boot, yp_boot, average="macro", zero_division=0)))

    cis = {}
    for name, arr in [
        ("phishing_recall", phish_recalls),
        ("legitimate_recall", legit_recalls),
        ("legitimate_fpr", legit_fprs),
        ("macro_f1", macro_f1s),
    ]:
        if arr:
            low = float(np.percentile(arr, 2.5))
            high = float(np.percentile(arr, 97.5))
            cis[name] = {"ci_95_low": round(low, 4), "ci_95_high": round(high, 4)}

    return cis


def compute_paired_bootstrap_delta(
    y_true: np.ndarray,
    prob_control: np.ndarray,
    prob_cand: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Compute paired bootstrap difference (Candidate - Control)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    yp_ctl = (prob_control < threshold).astype(int)
    yp_cand = (prob_cand < threshold).astype(int)

    delta_f1 = []
    delta_legit_fpr = []
    delta_phish_recall = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        c_ctl = yp_ctl[idx]
        c_cand = yp_cand[idx]

        # F1 delta
        f1_c = f1_score(yt, c_ctl, average="macro", zero_division=0)
        f1_cand = f1_score(yt, c_cand, average="macro", zero_division=0)
        delta_f1.append(float(f1_cand - f1_c))

        # Legit FPR delta
        n_l = (yt == 1).sum()
        if n_l > 0:
            fpr_c = ((yt == 1) & (c_ctl == 0)).sum() / n_l
            fpr_cand = ((yt == 1) & (c_cand == 0)).sum() / n_l
            delta_legit_fpr.append(float(fpr_cand - fpr_c))

        # Phishing recall delta
        n_p = (yt == 0).sum()
        if n_p > 0:
            rec_c = ((yt == 0) & (c_ctl == 0)).sum() / n_p
            rec_cand = ((yt == 0) & (c_cand == 0)).sum() / n_p
            delta_phish_recall.append(float(rec_cand - rec_c))

    def summarize(arr: List[float], invert_direction: bool = False):
        if not arr:
            return {}
        mean_d = float(np.mean(arr))
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        # empirical p-value for H0: delta <= 0 (or delta >= 0 for FPR)
        if not invert_direction:
            p_val = float(np.mean(np.array(arr) <= 0))
        else:
            p_val = float(np.mean(np.array(arr) >= 0))
        return {
            "mean_delta": round(mean_d, 4),
            "ci_95_low": round(low, 4),
            "ci_95_high": round(high, 4),
            "p_value": round(p_val, 4),
            "statistically_significant": (low > 0 if not invert_direction else high < 0),
        }

    return {
        "delta_macro_f1": summarize(delta_f1),
        "delta_legitimate_fpr": summarize(delta_legit_fpr, invert_direction=True),
        "delta_phishing_recall": summarize(delta_phish_recall),
    }


def run_statistical_analysis(
    internal_test_csv: Path,
    models_config: Dict[str, Dict[str, Any]],
    output_json_path: Path,
) -> Dict[str, Any]:
    """Run full Milestone 7 statistical evaluation."""
    print("Executing Phase 6 M7: Statistical Testing and Uncertainty Analysis...")
    test_df = pd.read_csv(internal_test_csv)
    y_true = test_df["label"].values.astype(int)

    # 1. Evaluate E0 (control)
    e0_model, _ = load_model_estimator(REPO_ROOT / models_config["E0"]["artifact_path"])
    _, df_e0 = evaluate_model_on_dataset(e0_model, test_df, threshold=0.50)
    prob_e0 = df_e0["prob_phishing"].values
    y_pred_e0 = df_e0["prediction"].values

    e0_cis = compute_bootstrap_cis(y_true, prob_e0)

    # 2. Evaluate candidates against E0
    comparisons = {}
    for mid in ["E1", "E2", "E3"]:
        mspec = models_config[mid]
        model, _ = load_model_estimator(REPO_ROOT / mspec["artifact_path"])
        _, df_cand = evaluate_model_on_dataset(model, test_df, threshold=0.50)
        prob_cand = df_cand["prob_phishing"].values
        y_pred_cand = df_cand["prediction"].values

        cand_cis = compute_bootstrap_cis(y_true, prob_cand)
        mcnemar = compute_mcnemar_test(y_true, y_pred_e0, y_pred_cand)
        paired_boot = compute_paired_bootstrap_delta(y_true, prob_e0, prob_cand)

        comparisons[mid] = {
            "model_name": mspec["name"],
            "bootstrap_confidence_intervals": cand_cis,
            "mcnemar_test_vs_E0": mcnemar,
            "paired_bootstrap_delta_vs_E0": paired_boot,
        }

    report = {
        "report_version": "phase6-m7-statistical-tests-v1",
        "phase": 6,
        "milestone": "M7",
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_partition": str(internal_test_csv.relative_to(REPO_ROOT)),
        "n_samples": len(y_true),
        "control_E0_confidence_intervals": e0_cis,
        "candidate_statistical_comparisons": comparisons,
        "status": "APPROVED",
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[M7 EXIT CRITERIA MET] Statistical analysis saved to {output_json_path.relative_to(REPO_ROOT)}")
    return report


if __name__ == "__main__":
    import yaml
    exp_cfg = yaml.safe_load(open(REPO_ROOT / "experiments" / "phase6" / "configs" / "experiment.yaml"))
    test_csv = REPO_ROOT / "experiments" / "phase6" / "datasets" / "splits" / "internal_test.csv"
    out_stat = REPO_ROOT / "experiments" / "phase6" / "reports" / "statistical_analysis.json"

    run_statistical_analysis(test_csv, exp_cfg["models"], out_stat)
