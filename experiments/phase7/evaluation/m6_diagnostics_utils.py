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
    epsilon: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Compute Population Stability Index (PSI) with epsilon zero-count smoothing.
    
    Formula:
        PSI = sum_{b=1}^B (p_b - q_b) * ln(p_b / q_b)
    where Q is reference (training) and P is evaluation (external).
    
    Epsilon smoothing:
        epsilon = epsilon if provided else 1.0 / (2.0 * N_eval)
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
    eps = float(epsilon) if epsilon is not None else (1.0 / (2.0 * n_eval))

    # Apply epsilon smoothing to zero-count bins
    ref_smoothed = np.where(ref_counts == 0, eps, ref_counts.astype(float))
    eval_smoothed = np.where(eval_counts == 0, eps, eval_counts.astype(float))

    # Normalize to proportions summing to 1.0
    q = ref_smoothed / np.sum(ref_smoothed)
    p = eval_smoothed / np.sum(eval_smoothed)

    psi_val = float(np.sum((p - q) * np.log(p / q)))
    return psi_val, "OK"


def compute_bh_fdr(raw_pvalues: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Benjamini-Hochberg (BH) False Discovery Rate (FDR) adjusted p-values (q-values).

    Parameters:
        raw_pvalues: 1D array of unadjusted p-values.
        alpha: False discovery rate threshold (default: 0.05).

    Returns:
        q_values: 1D array of adjusted p-values matching input order.
        significant: Boolean array indicating which hypotheses reject null under FDR control.
    """
    p = np.asarray(raw_pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.zeros(m)
    adjusted[-1] = sorted_p[-1]

    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * m / (i + 1))
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_values = np.zeros(m)
    q_values[order] = adjusted
    significant = q_values < alpha
    return q_values, significant


def compute_two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> Dict[str, Any]:
    """
    Perform a two-sample pooled z-test and Fisher's exact test for binary proportions.
    Uses stats.norm.sf(abs(z)) for numerical stability to prevent catastrophic cancellation.
    Avoids serializing underflowed p-values as true zero.

    Parameters:
        k1: Positive count in sample 1 (e.g., reference training)
        n1: Total size of sample 1
        k2: Positive count in sample 2 (e.g., evaluation external)
        n2: Total size of sample 2

    Returns:
        Dict with proportions, z-statistic, two-tailed p-value, Fisher exact p-value, and method metadata.
    """
    p1 = k1 / n1 if n1 > 0 else 0.0
    p2 = k2 / n2 if n2 > 0 else 0.0
    pooled_p = (k1 + k2) / (n1 + n2) if (n1 + n2) > 0 else 0.0
    se = np.sqrt(pooled_p * (1.0 - pooled_p) * (1.0 / n1 + 1.0 / n2)) if n1 > 0 and n2 > 0 else 0.0

    if se > 1e-12:
        z = (p2 - p1) / se
        # Numerically stable survival function: 2 * norm.sf(|z|) avoids cancellation in 1 - cdf
        sf_val = stats.norm.sf(abs(z))
        logsf_val = stats.norm.logsf(abs(z))
        z_p_val = 2.0 * sf_val
        # Prevent float underflow to true zero
        if z_p_val == 0.0 or z_p_val < 1e-300:
            z_p_val = 1e-300
    else:
        z = 0.0
        z_p_val = 1.0
        logsf_val = 0.0

    # Fisher's exact test for 2x2 contingency table: [[k1, n1 - k1], [k2, n2 - k2]]
    table = [[int(k1), int(n1 - k1)], [int(k2), int(n2 - k2)]]
    try:
        fisher_res = stats.fisher_exact(table, alternative="two-sided")
        fisher_stat = float(fisher_res.statistic)
        fisher_p = float(fisher_res.pvalue)
        if fisher_p == 0.0 or fisher_p < 1e-300:
            fisher_p = 1e-300
    except Exception:
        fisher_stat = float("nan")
        fisher_p = 1.0

    # For sparse binary events (counts < 10), Fisher's exact test is the authoritative primary test
    is_sparse = (k1 < 10 or k2 < 10 or (n1 - k1) < 10 or (n2 - k2) < 10)
    primary_p = fisher_p if is_sparse else z_p_val
    primary_method = "fisher_exact" if is_sparse else "two_sample_z_test_survival_function"

    return {
        "p_ref": float(p1),
        "p_eval": float(p2),
        "difference": float(p2 - p1),
        "z_statistic": float(z),
        "z_p_value": float(z_p_val),
        "z_log_p_value": float(logsf_val + np.log(2.0)),
        "fisher_exact_statistic": float(fisher_stat),
        "fisher_exact_p_value": float(fisher_p),
        "p_value": float(primary_p),
        "method_used": primary_method,
        "is_sparse_event": bool(is_sparse),
        "significant_005": bool(primary_p < 0.05),
    }


def compute_discrete_association_test(
    ref_series: np.ndarray,
    eval_series: np.ndarray,
    feature_name: str,
) -> Dict[str, Any]:
    """
    Perform discrete-appropriate statistical tests for features with heavy ties.
    For binary features: invokes compute_two_proportion_test (Fisher exact + z-test).
    For discrete count features: performs Mann-Whitney U test with exact tie handling.
    """
    ref = np.asarray(ref_series, dtype=float)
    eval_d = np.asarray(eval_series, dtype=float)

    binary_features = {"IsDomainIP", "HasObfuscation", "IsHTTPS"}
    if feature_name in binary_features:
        k1 = int(np.sum(ref >= 0.5))
        n1 = len(ref)
        k2 = int(np.sum(eval_d >= 0.5))
        n2 = len(eval_d)
        res = compute_two_proportion_test(k1, n1, k2, n2)
        return {
            "feature_name": feature_name,
            "feature_type": "BINARY",
            "test_name": res["method_used"],
            "statistic": res["z_statistic"] if "z_test" in res["method_used"] else res["fisher_exact_statistic"],
            "p_value": res["p_value"],
            "is_exact_or_tie_corrected": True,
            "details": res,
        }
    else:
        # Discrete count features with heavy ties (e.g. NoOfQMarkInURL, NoOfEqualsInURL, NoOfSubDomain)
        mwu_res = stats.mannwhitneyu(ref, eval_d, alternative="two-sided")
        p_val = float(mwu_res.pvalue)
        if p_val == 0.0 or p_val < 1e-300:
            p_val = 1e-300
        return {
            "feature_name": feature_name,
            "feature_type": "DISCRETE_COUNT",
            "test_name": "mann_whitney_u_tie_corrected",
            "statistic": float(mwu_res.statistic),
            "p_value": p_val,
            "is_exact_or_tie_corrected": True,
            "details": {
                "u_statistic": float(mwu_res.statistic),
                "tie_corrected_p_value": p_val,
            },
        }


def compute_domain_cluster_bootstrap(
    ref_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cluster_series: pd.Series,
    feature_names: List[str],
    num_iterations: int = 2000,
    random_seed: int = 42,
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Perform domain-cluster-aware bootstrap uncertainty analysis.
    Resamples unique registered domains with replacement and computes 95% CIs.

    Methodological Specifications:
    - Conditional on Training Reference: The training reference distribution D_train
      (N=94,523 legit / 76,092 phish) is held fixed as an asymptotic baseline because its
      sample size is ~145x larger than the external benchmark. Confidence intervals are
      strictly conditional on D_train.
    - Zero-Count Smoothing: Epsilon smoothing is fixed at nominal eps = 1.0 / (2.0 * N_eval)
      where N_eval is nominal evaluation size, ensuring stable logarithmic ratios across varying
      bootstrap cluster sizes.

    Parameters:
        ref_df: Reference training DataFrame (rows = samples, cols = features)
        eval_df: Evaluation benchmark DataFrame (rows = samples, cols = features)
        cluster_series: Series containing registered domain identifiers for each row in eval_df
        feature_names: List of feature names to evaluate
        num_iterations: Number of bootstrap iterations (>= 2,000)
        random_seed: Fixed seed for exact reproducibility
        num_bins: Number of quantile bins for PSI

    Returns:
        Dict containing 95% confidence intervals, bootstrap means, stds, and quantiles.
    """
    unique_domains = np.unique(cluster_series)
    n_clusters = len(unique_domains)
    domain_to_indices = {d: np.where(cluster_series == d)[0] for d in unique_domains}

    rng = np.random.default_rng(random_seed)

    # Pre-extract numpy matrices and pre-sort reference arrays for fast KS computation
    ref_mat = {f: ref_df[f].values.astype(float) for f in feature_names}
    ref_sorted = {f: np.sort(ref_mat[f]) for f in feature_names}
    n1_ref = {f: len(ref_mat[f]) for f in feature_names}
    eval_mat = {f: eval_df[f].values.astype(float) for f in feature_names}

    # Precompute bin edges for reference distributions for high performance
    bin_edges_map = {}
    binary_features = {"IsDomainIP", "HasObfuscation", "IsHTTPS"}
    for f in feature_names:
        ref = ref_mat[f]
        if f in binary_features:
            edges = np.array([-np.inf, 0.5, np.inf])
        else:
            quantiles = np.linspace(0, 100, num_bins + 1)
            raw_edges = np.unique(np.percentile(ref, quantiles))
            if len(raw_edges) >= 3:
                edges = raw_edges.copy()
                edges[0] = -np.inf
                edges[-1] = np.inf
            else:
                u_vals = np.unique(ref)
                if len(u_vals) <= num_bins + 1:
                    midpoints = (u_vals[:-1] + u_vals[1:]) / 2.0
                    edges = np.concatenate([[-np.inf], midpoints, [np.inf]])
                else:
                    pos = ref[ref > 0]
                    pos_edges = np.unique(np.percentile(pos, np.linspace(0, 100, num_bins)))
                    mid_zero = 0.5 if np.min(pos) >= 1.0 else float(np.min(pos) / 2.0)
                    tail = pos_edges[pos_edges > mid_zero]
                    edges = np.concatenate([[-np.inf, mid_zero], tail, [np.inf]])
        bin_edges_map[f] = edges

    # Precompute reference histograms
    ref_counts_map = {}
    for f in feature_names:
        counts, _ = np.histogram(ref_mat[f], bins=bin_edges_map[f])
        ref_counts_map[f] = counts

    # Storage for bootstrap distributions
    boot_mean_psi = np.zeros(num_iterations)
    boot_feature_psi = {f: np.zeros(num_iterations) for f in feature_names}
    boot_feature_ks = {f: np.zeros(num_iterations) for f in feature_names}

    n_eval_nominal = len(eval_df)
    eps = 1.0 / (2.0 * n_eval_nominal)

    for it in range(num_iterations):
        sampled_doms = rng.choice(unique_domains, size=n_clusters, replace=True)
        sample_idx = np.concatenate([domain_to_indices[d] for d in sampled_doms])
        n2_sample = len(sample_idx)

        iter_psis = []
        for f in feature_names:
            eval_sample = eval_mat[f][sample_idx]
            edges = bin_edges_map[f]
            ref_counts = ref_counts_map[f]
            eval_counts, _ = np.histogram(eval_sample, bins=edges)

            # Smoothed proportions
            ref_s = np.where(ref_counts == 0, eps, ref_counts.astype(float))
            eval_s = np.where(eval_counts == 0, eps, eval_counts.astype(float))
            q = ref_s / np.sum(ref_s)
            p = eval_s / np.sum(eval_s)

            psi_val = float(np.sum((p - q) * np.log(p / q)))
            boot_feature_psi[f][it] = psi_val
            iter_psis.append(psi_val)

            # Fast exact 2-sample KS D-statistic using pre-sorted reference
            y_sorted = np.sort(eval_sample)
            n1 = n1_ref[f]
            c1_right = np.searchsorted(ref_sorted[f], y_sorted, side="right") / n1
            c1_left = np.searchsorted(ref_sorted[f], y_sorted, side="left") / n1
            c2_right = np.arange(1, n2_sample + 1) / n2_sample
            c2_left = np.arange(0, n2_sample) / n2_sample
            d_stat = float(max(np.max(c2_right - c1_right), np.max(c1_left - c2_left)))
            boot_feature_ks[f][it] = d_stat

        boot_mean_psi[it] = float(np.mean(iter_psis))

    # Calculate summaries
    feature_summaries = {}
    for f in feature_names:
        psis = boot_feature_psi[f]
        ks_ds = boot_feature_ks[f]
        feature_summaries[f] = {
            "psi": {
                "bootstrap_mean": float(np.mean(psis)),
                "bootstrap_std": float(np.std(psis)),
                "ci_95": [float(np.percentile(psis, 2.5)), float(np.percentile(psis, 97.5))],
                "median": float(np.median(psis)),
            },
            "ks_d": {
                "bootstrap_mean": float(np.mean(ks_ds)),
                "bootstrap_std": float(np.std(ks_ds)),
                "ci_95": [float(np.percentile(ks_ds, 2.5)), float(np.percentile(ks_ds, 97.5))],
                "median": float(np.median(ks_ds)),
            },
        }

    return {
        "num_iterations": num_iterations,
        "random_seed": random_seed,
        "cluster_count": n_clusters,
        "confidence_interval_type": "CONDITIONAL_ON_TRAINING_REFERENCE",
        "reference_specification": {
            "reference_partition": "frozen_training_split",
            "reference_sample_size": len(ref_df),
            "reference_condition": "fixed_asymptotic_baseline",
            "smoothing_epsilon_policy": "fixed_nominal_epsilon",
            "smoothing_epsilon_value": float(eps),
        },
        "mean_psi": {
            "bootstrap_mean": float(np.mean(boot_mean_psi)),
            "bootstrap_std": float(np.std(boot_mean_psi)),
            "ci_95": [float(np.percentile(boot_mean_psi, 2.5)), float(np.percentile(boot_mean_psi, 97.5))],
            "median": float(np.median(boot_mean_psi)),
        },
        "features": feature_summaries,
    }


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
