"""
M6 Diagnostic & Operational Readiness Utilities Module.

Provides deterministic mathematical and statistical functions for:
1. Structural strata assignment with robust parsing and anomaly tracking.
2. Exact Clopper-Pearson binomial confidence intervals with boundary handling.
3. Population Stability Index (PSI) with epsilon zero-smoothing and constant handling.
4. Comprehensive training bounding-box and multivariate coverage distinction.
5. TreeSHAP probability-space additivity verification against predict_proba.
6. Out-of-fold domain-grouped lexical probe evaluation (diagnostic only).
"""

import ipaddress
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def is_ip_host(host: str) -> bool:
    """Determine whether a host string is a valid IPv4 or IPv6 address."""
    if not host:
        return False
    clean_host = host.strip("[]")
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False


def extract_path_segments(path: str) -> List[str]:
    """Normalize and split path into non-empty segments, collapsing repeated slashes."""
    clean = re.sub(r"/+", "/", path.strip())
    return [s for s in clean.split("/") if s]


def classify_structural_stratum(url: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Deterministically classify a URL into one of the 6 frozen structural strata.
    
    Strata Priority:
    1. special_syntax_variants:
       - IP literal (IPv4 or IPv6 in netloc/host)
       - Userinfo / Credentials (@ in netloc)
       - Non-standard port (present and not in (80, 443))
       - Malformed / non-numeric port or unparseable host
       - Hex / percent-encoding (% in URL, including valid and malformed)
    2. root_with_query:
       - 0 path segments and non-empty query string
    3. path_and_query:
       - >= 1 path segments and non-empty query string
    4. homepage_root:
       - 0 path segments and no query
    5. single_level_path_no_query:
       - Exactly 1 path segment and no query
    6. multi_level_path_no_query:
       - >= 2 path segments and no query

    Handles parsing exceptions, bare query delimiters, and trailing slashes.
    Returns: (stratum_name, parsing_anomaly_or_none)
    """
    parsing_anomaly: Optional[Dict[str, str]] = None
    try:
        parsed = urlsplit(url)
    except Exception as exc:
        return "special_syntax_variants", {
            "error": str(exc),
            "exception_type": exc.__class__.__name__,
            "url": url,
        }

    netloc = parsed.netloc or ""
    has_credentials = "@" in netloc

    has_special_port = False
    try:
        port = parsed.port
        if port is not None and port not in (80, 443):
            has_special_port = True
    except ValueError as val_err:
        has_special_port = True
        parsing_anomaly = {
            "error": str(val_err),
            "exception_type": "ValueError",
            "url": url,
        }

    try:
        host = parsed.hostname or ""
    except ValueError as host_err:
        host = ""
        has_special_port = True
        parsing_anomaly = {
            "error": str(host_err),
            "exception_type": "ValueError",
            "url": url,
        }

    is_ip = is_ip_host(host)
    has_obfuscation = "%" in url

    if has_credentials or is_ip or has_special_port or has_obfuscation:
        return "special_syntax_variants", parsing_anomaly

    segments = extract_path_segments(parsed.path)
    # Bare query delimiter (e.g., '?' without query content) yields parsed.query == ''
    has_query = bool(parsed.query and parsed.query.strip())

    if len(segments) == 0 and has_query:
        return "root_with_query", parsing_anomaly
    if len(segments) >= 1 and has_query:
        return "path_and_query", parsing_anomaly
    if len(segments) == 0:
        return "homepage_root", parsing_anomaly
    if len(segments) == 1:
        return "single_level_path_no_query", parsing_anomaly
    return "multi_level_path_no_query", parsing_anomaly


def compute_clopper_pearson(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute exact Clopper-Pearson two-sided (1 - alpha) binomial confidence interval.
    Explicitly handles boundary cases x = 0 and x = n.
    """
    if n <= 0:
        return 0.0, 1.0
    if x <= 0:
        lower = 0.0
        upper = 1.0 - (alpha / 2.0) ** (1.0 / n)
        return float(lower), float(upper)
    if x >= n:
        lower = (alpha / 2.0) ** (1.0 / n)
        upper = 1.0
        return float(lower), float(upper)

    lower = float(stats.beta.ppf(alpha / 2.0, x, n - x + 1))
    upper = float(stats.beta.ppf(1.0 - alpha / 2.0, x + 1, n - x))
    return lower, upper


def compute_psi(
    ref: np.ndarray,
    eval_data: np.ndarray,
    num_bins: int = 10,
    feature_name: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Compute Population Stability Index (PSI) with epsilon zero-count smoothing.
    
    Formula:
        PSI = sum_{b=1}^B (p_b - q_b) * ln(p_b / q_b)
    where Q is reference (training) and P is evaluation (external).
    
    Epsilon smoothing:
        epsilon = 1.0 / (2.0 * N_eval)
        Added to zero-count bins and normalized so sum(p) == sum(q) == 1.0.
        
    Handles constant features:
        - If Q has zero variance and P is identical: returns (0.0, 'CONSTANT_IDENTICAL')
        - If Q has zero variance and P differs: returns (inf, 'CONSTANT_DEGENERATE_SHIFT')

    Discrete and binary features:
        - Exact categorical bins [0, 1] for binary features: IsDomainIP, HasObfuscation, IsHTTPS.
        - Tail-separated quantile bins for sparse discrete count features when quantiles collapse.
    """
    ref = np.asarray(ref, dtype=float)
    eval_data = np.asarray(eval_data, dtype=float)

    # Constant feature check
    if np.all(ref == ref[0]):
        if np.all(eval_data == ref[0]):
            return 0.0, "CONSTANT_IDENTICAL"
        else:
            return float("inf"), "CONSTANT_DEGENERATE_SHIFT"

    binary_features = {"IsDomainIP", "HasObfuscation", "IsHTTPS"}
    if feature_name in binary_features:
        bin_edges = np.array([-np.inf, 0.5, np.inf])
    else:
        quantiles = np.linspace(0, 100, num_bins + 1)
        raw_edges = np.unique(np.percentile(ref, quantiles))
        if len(raw_edges) >= 3:
            bin_edges = raw_edges.copy()
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
        else:
            u_vals = np.unique(ref)
            if len(u_vals) <= 1:
                if np.all(eval_data == ref[0]):
                    return 0.0, "CONSTANT_IDENTICAL"
                return float("inf"), "CONSTANT_DEGENERATE_SHIFT"
            elif len(u_vals) <= num_bins + 1:
                midpoints = (u_vals[:-1] + u_vals[1:]) / 2.0
                bin_edges = np.concatenate([[-np.inf], midpoints, [np.inf]])
            else:
                pos = ref[ref > 0]
                pos_edges = np.unique(np.percentile(pos, np.linspace(0, 100, num_bins)))
                mid_zero = 0.5 if np.min(pos) >= 1.0 else float(np.min(pos) / 2.0)
                tail = pos_edges[pos_edges > mid_zero]
                bin_edges = np.concatenate([[-np.inf, mid_zero], tail, [np.inf]])

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    eval_counts, _ = np.histogram(eval_data, bins=bin_edges)

    n_eval = len(eval_data)
    eps = 1.0 / (2.0 * n_eval)

    # Apply epsilon smoothing to zero-count bins
    ref_smoothed = np.where(ref_counts == 0, eps, ref_counts.astype(float))
    eval_smoothed = np.where(eval_counts == 0, eps, eval_counts.astype(float))

    # Normalize to proportions summing to 1.0
    q = ref_smoothed / np.sum(ref_smoothed)
    p = eval_smoothed / np.sum(eval_smoothed)

    psi_val = float(np.sum((p - q) * np.log(p / q)))
    return psi_val, "OK"


def compute_bounding_box_analysis(
    train_features: pd.DataFrame,
    eval_features: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Conduct training bounding-box analysis and report:
    1. Per-feature out-of-range rates (under, over, total).
    2. Training quantiles (P0/Min, P1, P5, P25, P50, P75, P95, P99, P100/Max).
    3. Record-level union out-of-range rate (>= 1 feature OOR).
    4. Record-level multi-feature out-of-range rates (>= 2, >= 3 features OOR).
    5. Explicit documentation of univariate vs joint multivariate coverage.
    """
    train_df = train_features[feature_names]
    eval_df = eval_features[feature_names]

    train_min = train_df.min()
    train_max = train_df.max()

    quantiles = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    train_quantiles = train_df.quantile(quantiles).to_dict()

    under_mask = eval_df < train_min
    over_mask = eval_df > train_max
    total_oor_mask = under_mask | over_mask

    per_feature_oor = {}
    for f in feature_names:
        per_feature_oor[f] = {
            "under_min_rate": float(under_mask[f].mean()),
            "over_max_rate": float(over_mask[f].mean()),
            "total_oor_rate": float(total_oor_mask[f].mean()),
            "train_min": float(train_min[f]),
            "train_max": float(train_max[f]),
            "eval_min": float(eval_df[f].min()),
            "eval_max": float(eval_df[f].max()),
        }

    record_oor_counts = total_oor_mask.sum(axis=1)
    n_eval = len(eval_df)

    union_oor_rate = float((record_oor_counts >= 1).mean())
    multi_oor_2_rate = float((record_oor_counts >= 2).mean())
    multi_oor_3_rate = float((record_oor_counts >= 3).mean())

    oor_histogram = {
        int(k): int((record_oor_counts == k).sum())
        for k in range(len(feature_names) + 1)
        if (record_oor_counts == k).sum() > 0
    }

    return {
        "per_feature_out_of_range": per_feature_oor,
        "training_quantiles": train_quantiles,
        "record_level_union_oor_rate": union_oor_rate,
        "record_level_multi_oor_2_rate": multi_oor_2_rate,
        "record_level_multi_oor_3_rate": multi_oor_3_rate,
        "oor_feature_count_distribution": oor_histogram,
        "theoretical_distinction": (
            "Univariate bounding box analysis checks whether individual feature values fall outside "
            "1D training extremes [min_f, max_f]. A sample can satisfy all 18 univariate bounds "
            "while residing in an empty or unpopulated region of the joint 18-dimensional feature space. "
            "Univariate range satisfaction is a necessary but not sufficient condition for in-distribution coverage."
        ),
    }


def verify_shap_probability_additivity(
    clf,
    X_sample: np.ndarray,
    tolerance: float = 1e-05,
) -> Dict[str, Any]:
    """
    Verify TreeSHAP probability-space additivity for Random Forest models:
        expected_value[0] + sum(shap_values[:, :, 0], axis=1) == predict_proba[:, 0]
    
    Locks:
        - Class index: 0 (Phishing)
        - Explainer: TreeExplainer(feature_perturbation='tree_path_dependent')
        - Numerical tolerance: <= tolerance (default 1e-5)
    
    Fails explicitly with AssertionError if violated.
    """
    import shap

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_sample)
    expected_val = explainer.expected_value

    prob_true = clf.predict_proba(X_sample)[:, 0]

    if isinstance(shap_vals, list):
        shap_0 = shap_vals[0]
        base_0 = expected_val[0]
    elif len(np.shape(shap_vals)) == 3:
        shap_0 = shap_vals[:, :, 0]
        base_0 = expected_val[0]
    else:
        shap_0 = shap_vals
        base_0 = expected_val

    reconstructed_prob = base_0 + np.sum(shap_0, axis=1)
    diff = np.abs(reconstructed_prob - prob_true)
    max_error = float(np.max(diff))
    mean_error = float(np.mean(diff))

    assert max_error <= tolerance, (
        f"TreeSHAP probability-space additivity violated: max error {max_error:.2e} > tolerance {tolerance:.2e}"
    )

    return {
        "status": "VERIFIED_EXACT_ADDITIVITY",
        "sample_size": len(X_sample),
        "class_index": 0,
        "expected_value_base": float(base_0),
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
        "tolerance": tolerance,
    }


def evaluate_lexical_probe_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate an observational linear probe using registered-domain-grouped cross-validation
    and out-of-fold predictions on the external benchmark.
    
    Classification: DIAGNOSTIC_ONLY_STATISTICAL_FIT
    Prohibition: Zero model retraining. Not a mathematical upper bound, but an empirical out-of-fold baseline.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y))

    for train_idx, val_idx in sgkf.split(X, y, groups):
        # Assert domain disjointness between train and validation folds
        train_domains = set(groups[train_idx])
        val_domains = set(groups[val_idx])
        assert train_domains.isdisjoint(val_domains), "Domain leakage detected between CV folds!"

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        # Diagnostic fit: L1 logistic regression (solver='saga' with l1_ratio=1.0)
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            random_state=random_state,
        )
        clf.fit(X_tr, y[train_idx])

        # Predict probability for class 0 (Phishing)
        # In sklearn, clf.classes_ are sorted [0, 1] so index 0 is Phishing
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 0]

    # Evaluate out-of-fold metrics
    actual_phish = (y == 0).astype(int)
    oof_auc = float(roc_auc_score(actual_phish, oof_probs))

    pred_050 = (oof_probs >= 0.50).astype(int)
    pred_075 = (oof_probs >= 0.75).astype(int)

    f1_050 = float(f1_score(actual_phish, pred_050))
    f1_075 = float(f1_score(actual_phish, pred_075))

    n_legit = np.sum(y == 1)
    fpr_050 = float(np.sum((oof_probs >= 0.50) & (y == 1)) / n_legit)
    fpr_075 = float(np.sum((oof_probs >= 0.75) & (y == 1)) / n_legit)

    n_phish = np.sum(y == 0)
    recall_050 = float(np.sum((oof_probs >= 0.50) & (y == 0)) / n_phish)
    recall_075 = float(np.sum((oof_probs >= 0.75) & (y == 0)) / n_phish)

    return {
        "probe_type": "LogisticRegression(penalty='l1', solver='liblinear', C=1.0)",
        "classification": "DIAGNOSTIC_ONLY_STATISTICAL_FIT",
        "description": (
            "Empirical out-of-fold baseline measuring linear separability under domain-disjoint grouping. "
            "This probe is strictly an observational diagnostic and is NOT an E0/E4 candidate or retraining."
        ),
        "validation_scheme": f"StratifiedGroupKFold(n_splits={n_splits}, grouped by registered_domain)",
        "out_of_fold_metrics": {
            "roc_auc": oof_auc,
            "threshold_0_50": {
                "phishing_recall": recall_050,
                "legitimate_fpr": fpr_050,
                "phishing_f1": f1_050,
            },
            "threshold_0_75": {
                "phishing_recall": recall_075,
                "legitimate_fpr": fpr_075,
                "phishing_f1": f1_075,
            },
        },
    }
