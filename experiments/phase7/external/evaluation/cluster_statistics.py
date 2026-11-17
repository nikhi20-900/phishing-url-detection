"""
Domain-Aware Clustered Statistical Inference Engine for Phase 7 Milestone 5.

Implements rigorous domain-aware statistical methods:
1. Monte Carlo Cluster Permutation Test (M = 10,000 iterations, Phipson & Smyth 2010 formula).
2. Domain-Clustered Bootstrap Percentile Confidence Intervals (B = 1,000 resamples).
3. Domain-Aggregated Wilcoxon Signed-Rank Test (unweighted, size-weighted, Pratt check, rank-biserial).
4. Domain-Thinning Sensitivity Analysis (100 iterations of 1 URL/domain).
5. Population-Weighted Bayesian PPV/NPV Curves (prevalence priors pi in [0.0001, 0.01]).
"""

import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


def compute_monte_carlo_cluster_permutation(
    domains: np.ndarray,
    pred_e0: np.ndarray,
    pred_e4: np.ndarray,
    y_true: np.ndarray,
    m_iterations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Monte Carlo Cluster Permutation Test at the registered domain level.
    Under H0, model predictions (y_d(E0), y_d(E4)) are exchangeable within each registered domain d.
    Uses corrected Monte Carlo p-value formula: p = (1 + k) / (M + 1).
    """
    rng = np.random.default_rng(seed)
    unique_domains, domain_inverse = np.unique(domains, return_inverse=True)
    n_domains = len(unique_domains)

    is_phish = (y_true == 1)
    is_legit = (y_true == 0)

    total_p = int(is_phish.sum())
    total_l = int(is_legit.sum())

    # Pre-aggregate domain sums for fast evaluation
    tp0_by_domain = np.bincount(domain_inverse, weights=(pred_e0 == 1) & is_phish, minlength=n_domains)
    tp4_by_domain = np.bincount(domain_inverse, weights=(pred_e4 == 1) & is_phish, minlength=n_domains)

    fp0_by_domain = np.bincount(domain_inverse, weights=(pred_e0 == 1) & is_legit, minlength=n_domains)
    fp4_by_domain = np.bincount(domain_inverse, weights=(pred_e4 == 1) & is_legit, minlength=n_domains)

    # Observed statistics
    obs_tp0 = tp0_by_domain.sum()
    obs_tp4 = tp4_by_domain.sum()
    obs_fp0 = fp0_by_domain.sum()
    obs_fp4 = fp4_by_domain.sum()

    obs_delta_recall = (obs_tp4 - obs_tp0) / total_p if total_p > 0 else 0.0
    obs_delta_fpr = (obs_fp4 - obs_fp0) / total_l if total_l > 0 else 0.0

    k_recall = 0
    k_fpr = 0

    # Differences if domain predictions are swapped
    diff_tp = tp4_by_domain - tp0_by_domain
    diff_fp = fp4_by_domain - fp0_by_domain

    for _ in range(m_iterations):
        # Draw Bernoulli(0.5) for each domain cluster
        # swap = 1 means swap E0 and E4 for that domain
        swap = rng.integers(0, 2, size=n_domains)

        # Under permutation, swapped domains contribute -diff instead of +diff
        # Multiplier: 1 if not swapped, -1 if swapped
        sign = 1 - 2 * swap

        perm_delta_tp = np.dot(sign, diff_tp)
        perm_delta_fp = np.dot(sign, diff_fp)

        perm_delta_recall = perm_delta_tp / total_p if total_p > 0 else 0.0
        perm_delta_fpr = perm_delta_fp / total_l if total_l > 0 else 0.0

        if abs(perm_delta_recall) >= abs(obs_delta_recall):
            k_recall += 1
        if abs(perm_delta_fpr) >= abs(obs_delta_fpr):
            k_fpr += 1

    p_val_recall = float((1 + k_recall) / (m_iterations + 1))
    p_val_fpr = float((1 + k_fpr) / (m_iterations + 1))

    return {
        "m_iterations": m_iterations,
        "n_domain_clusters": n_domains,
        "phishing_recall": {
            "obs_delta": float(obs_delta_recall),
            "k_extreme": k_recall,
            "monte_carlo_p_value": p_val_recall,
            "formula": "(1 + k) / (M + 1)",
        },
        "legitimate_fpr": {
            "obs_delta": float(obs_delta_fpr),
            "k_extreme": k_fpr,
            "monte_carlo_p_value": p_val_fpr,
            "formula": "(1 + k) / (M + 1)",
        },
    }


def compute_domain_clustered_bootstrap(
    domains: np.ndarray,
    pred_e0: np.ndarray,
    pred_e4: np.ndarray,
    y_true: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Domain-Clustered Bootstrap Percentile Confidence Intervals.
    Resamples unique registered domains with replacement.
    """
    rng = np.random.default_rng(seed)
    unique_domains, domain_inverse = np.unique(domains, return_inverse=True)
    n_domains = len(unique_domains)

    is_phish = (y_true == 1)
    is_legit = (y_true == 0)

    n_phish_by_domain = np.bincount(domain_inverse, weights=is_phish, minlength=n_domains)
    n_legit_by_domain = np.bincount(domain_inverse, weights=is_legit, minlength=n_domains)

    tp_e0_by_domain = np.bincount(domain_inverse, weights=(pred_e0 == 1) & is_phish, minlength=n_domains)
    fp_e0_by_domain = np.bincount(domain_inverse, weights=(pred_e0 == 1) & is_legit, minlength=n_domains)

    tp_e4_by_domain = np.bincount(domain_inverse, weights=(pred_e4 == 1) & is_phish, minlength=n_domains)
    fp_e4_by_domain = np.bincount(domain_inverse, weights=(pred_e4 == 1) & is_legit, minlength=n_domains)

    boot_rec_e0, boot_rec_e4, boot_delta_rec = [], [], []
    boot_fpr_e0, boot_fpr_e4, boot_delta_fpr = [], [], []
    boot_mf1_e0, boot_mf1_e4, boot_delta_mf1 = [], [], []

    for _ in range(n_bootstrap):
        sampled_counts = np.bincount(rng.choice(n_domains, size=n_domains, replace=True), minlength=n_domains)

        total_p = np.dot(sampled_counts, n_phish_by_domain)
        total_l = np.dot(sampled_counts, n_legit_by_domain)

        tp0 = np.dot(sampled_counts, tp_e0_by_domain)
        fp0 = np.dot(sampled_counts, fp_e0_by_domain)
        tn0 = total_l - fp0
        fn0 = total_p - tp0

        tp4 = np.dot(sampled_counts, tp_e4_by_domain)
        fp4 = np.dot(sampled_counts, fp_e4_by_domain)
        tn4 = total_l - fp4
        fn4 = total_p - tp4

        if total_p > 0 and total_l > 0:
            rec0 = tp0 / total_p
            fpr0 = fp0 / total_l
            prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
            f1_p0 = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
            rec_l0 = tn0 / total_l
            prec_l0 = tn0 / (tn0 + fn0) if (tn0 + fn0) > 0 else 0.0
            f1_l0 = (2 * prec_l0 * rec_l0) / (prec_l0 + rec_l0) if (prec_l0 + rec_l0) > 0 else 0.0
            mf1_0 = (f1_p0 + f1_l0) / 2.0

            rec4 = tp4 / total_p
            fpr4 = fp4 / total_l
            prec4 = tp4 / (tp4 + fp4) if (tp4 + fp4) > 0 else 0.0
            f1_p4 = (2 * prec4 * rec4) / (prec4 + rec4) if (prec4 + rec4) > 0 else 0.0
            rec_l4 = tn4 / total_l
            prec_l4 = tn4 / (tn4 + fn4) if (tn4 + fn4) > 0 else 0.0
            f1_l4 = (2 * prec_l4 * rec_l4) / (prec_l4 + rec_l4) if (prec_l4 + rec_l4) > 0 else 0.0
            mf1_4 = (f1_p4 + f1_l4) / 2.0

            boot_rec_e0.append(rec0)
            boot_rec_e4.append(rec4)
            boot_delta_rec.append(rec4 - rec0)

            boot_fpr_e0.append(fpr0)
            boot_fpr_e4.append(fpr4)
            boot_delta_fpr.append(fpr4 - fpr0)

            boot_mf1_e0.append(mf1_0)
            boot_mf1_e4.append(mf1_4)
            boot_delta_mf1.append(mf1_4 - mf1_0)

    def ci95(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"point_estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "n_bootstrap": n_bootstrap,
        "n_domain_clusters": n_domains,
        "e0": {
            "phishing_recall": ci95(boot_rec_e0),
            "legitimate_fpr": ci95(boot_fpr_e0),
            "macro_f1": ci95(boot_mf1_e0),
        },
        "e4": {
            "phishing_recall": ci95(boot_rec_e4),
            "legitimate_fpr": ci95(boot_fpr_e4),
            "macro_f1": ci95(boot_mf1_e4),
        },
        "paired_delta_e4_minus_e0": {
            "phishing_recall": ci95(boot_delta_rec),
            "legitimate_fpr": ci95(boot_delta_fpr),
            "macro_f1": ci95(boot_delta_mf1),
        },
    }


def compute_domain_wilcoxon_analysis(
    domains: np.ndarray,
    pred_e0: np.ndarray,
    pred_e4: np.ndarray,
    y_true: np.ndarray,
    target_class: int = 0,  # 0 = Legitimate (FPR analysis)
) -> Dict[str, Any]:
    """
    Domain-Aggregated Wilcoxon Signed-Rank Test.
    Evaluates paired domain differences f_d(E4) - f_d(E0) across distinct registered domains.
    Provides standard Wilcoxon, Pratt zero-difference check, and size-weighted analysis.
    """
    unique_domains, domain_inverse = np.unique(domains, return_inverse=True)
    n_domains = len(unique_domains)

    mask = (y_true == target_class)
    n_per_domain = np.bincount(domain_inverse, weights=mask, minlength=n_domains)

    # Domain-level error rates
    err_e0_by_domain = np.bincount(domain_inverse, weights=(pred_e0 != target_class) & mask, minlength=n_domains)
    err_e4_by_domain = np.bincount(domain_inverse, weights=(pred_e4 != target_class) & mask, minlength=n_domains)

    # Filter to domains containing at least 1 URL of target class
    valid_dom_indices = np.where(n_per_domain > 0)[0]
    K_total = len(valid_dom_indices)

    rates_e0 = err_e0_by_domain[valid_dom_indices] / n_per_domain[valid_dom_indices]
    rates_e4 = err_e4_by_domain[valid_dom_indices] / n_per_domain[valid_dom_indices]
    domain_sizes = n_per_domain[valid_dom_indices]

    deltas = rates_e4 - rates_e0
    K_zero = int((deltas == 0).sum())
    K_effective = K_total - K_zero

    # Primary Wilcoxon (discard zeros)
    if K_effective > 0:
        res_wilcoxon = stats.wilcoxon(rates_e4, rates_e0, zero_method="wilcox", alternative="two-sided")
        p_val_primary = float(res_wilcoxon.pvalue)
        stat_primary = float(res_wilcoxon.statistic)

        # Matched-pairs rank-biserial correlation
        # r_rb = (W+ - W-) / (W+ + W-)
        non_zero_deltas = deltas[deltas != 0]
        abs_ranks = stats.rankdata(np.abs(non_zero_deltas))
        w_plus = float(np.sum(abs_ranks[non_zero_deltas > 0]))
        w_minus = float(np.sum(abs_ranks[non_zero_deltas < 0]))
        total_w = w_plus + w_minus
        r_rb = (w_plus - w_minus) / total_w if total_w > 0 else 0.0
    else:
        p_val_primary = 1.0
        stat_primary = 0.0
        r_rb = 0.0
        w_plus = 0.0
        w_minus = 0.0

    # Pratt modification (retain zeros during ranking)
    try:
        res_pratt = stats.wilcoxon(rates_e4, rates_e0, zero_method="pratt", alternative="two-sided")
        p_val_pratt = float(res_pratt.pvalue)
    except Exception:
        p_val_pratt = float("nan")

    # Sample-size weighted Wilcoxon (weight = sqrt(n_d))
    # Approximation via weighted rank sign test
    weighted_deltas = deltas * np.sqrt(domain_sizes)
    if np.any(weighted_deltas != 0):
        try:
            res_weighted = stats.wilcoxon(weighted_deltas, zero_method="wilcox", alternative="two-sided")
            p_val_weighted = float(res_weighted.pvalue)
        except Exception:
            p_val_weighted = float("nan")
    else:
        p_val_weighted = 1.0

    return {
        "target_class": "legitimate_fpr" if target_class == 0 else "phishing_error",
        "K_total_domains": K_total,
        "K_zero_difference_domains": K_zero,
        "K_effective_domains": K_effective,
        "median_domain_delta": float(np.median(deltas)),
        "mean_domain_delta": float(np.mean(deltas)),
        "primary_wilcoxon": {
            "statistic": stat_primary,
            "p_value": p_val_primary,
            "w_plus": w_plus,
            "w_minus": w_minus,
            "rank_biserial_correlation": float(r_rb),
        },
        "robustness_checks": {
            "pratt_p_value": p_val_pratt,
            "size_weighted_p_value": p_val_weighted,
        },
    }


def compute_domain_thinning_sensitivity(
    domains: np.ndarray,
    p_phish_e0: np.ndarray,
    p_phish_e4: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.50,
    n_iterations: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Domain-Thinning Sensitivity Analysis (Designated strictly as sensitivity analysis).
    Samples exactly 1 URL per registered domain across 100 iterations.
    """
    rng = np.random.default_rng(seed)
    unique_domains = np.unique(domains)
    domain_to_indices = {d: np.where(domains == d)[0] for d in unique_domains}

    thinned_rec_e0, thinned_rec_e4, thinned_delta_rec = [], [], []
    thinned_fpr_e0, thinned_fpr_e4, thinned_delta_fpr = [], [], []
    thinned_mf1_e0, thinned_mf1_e4, thinned_delta_mf1 = [], [], []

    for _ in range(n_iterations):
        selected_indices = [
            rng.choice(domain_to_indices[d]) for d in unique_domains
        ]
        sub_y = y_true[selected_indices]
        sub_p0 = p_phish_e0[selected_indices]
        sub_p4 = p_phish_e4[selected_indices]

        pred0 = (sub_p0 >= threshold).astype(int)
        pred4 = (sub_p4 >= threshold).astype(int)

        is_p = (sub_y == 1)
        is_l = (sub_y == 0)

        tp0 = int(((pred0 == 1) & is_p).sum())
        fp0 = int(((pred0 == 1) & is_l).sum())
        tn0 = int(((pred0 == 0) & is_l).sum())
        fn0 = int(((pred0 == 0) & is_p).sum())

        tp4 = int(((pred4 == 1) & is_p).sum())
        fp4 = int(((pred4 == 1) & is_l).sum())
        tn4 = int(((pred4 == 0) & is_l).sum())
        fn4 = int(((pred4 == 0) & is_p).sum())

        n_p = int(is_p.sum())
        n_l = int(is_l.sum())

        if n_p > 0 and n_l > 0:
            rec0 = tp0 / n_p
            fpr0 = fp0 / n_l
            rec4 = tp4 / n_p
            fpr4 = fp4 / n_l

            prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
            prec4 = tp4 / (tp4 + fp4) if (tp4 + fp4) > 0 else 0.0

            f1_0 = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
            f1_4 = (2 * prec4 * rec4) / (prec4 + rec4) if (prec4 + rec4) > 0 else 0.0

            thinned_rec_e0.append(rec0)
            thinned_rec_e4.append(rec4)
            thinned_delta_rec.append(rec4 - rec0)

            thinned_fpr_e0.append(fpr0)
            thinned_fpr_e4.append(fpr4)
            thinned_delta_fpr.append(fpr4 - fpr0)

            thinned_mf1_e0.append(f1_0)
            thinned_mf1_e4.append(f1_4)
            thinned_delta_mf1.append(f1_4 - f1_0)

    def stats_summary(arr: List[float]) -> Dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p025": float(np.percentile(arr, 2.5)),
            "p975": float(np.percentile(arr, 97.5)),
        }

    return {
        "n_iterations": n_iterations,
        "sample_size_per_iteration": len(unique_domains),
        "e0": {
            "phishing_recall": stats_summary(thinned_rec_e0),
            "legitimate_fpr": stats_summary(thinned_fpr_e0),
        },
        "e4": {
            "phishing_recall": stats_summary(thinned_rec_e4),
            "legitimate_fpr": stats_summary(thinned_fpr_e4),
        },
        "paired_delta": {
            "phishing_recall": stats_summary(thinned_delta_rec),
            "legitimate_fpr": stats_summary(thinned_delta_fpr),
        },
    }


def compute_bayesian_precision_curves(
    recall: float,
    fpr: float,
    prevalence_priors: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute population-weighted Bayesian Positive Predictive Value (PPV) and
    Negative Predictive Value (NPV) across operational prevalence priors.
    """
    if prevalence_priors is None:
        prevalence_priors = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.50]

    results = []
    for pi in prevalence_priors:
        # PPV = (Recall * pi) / (Recall * pi + FPR * (1 - pi))
        denom_ppv = (recall * pi) + (fpr * (1.0 - pi))
        ppv = (recall * pi) / denom_ppv if denom_ppv > 0 else 0.0

        # NPV = ((1 - FPR) * (1 - pi)) / ((1 - FPR) * (1 - pi) + (1 - Recall) * pi)
        denom_npv = ((1.0 - fpr) * (1.0 - pi)) + ((1.0 - recall) * pi)
        npv = ((1.0 - fpr) * (1.0 - pi)) / denom_npv if denom_npv > 0 else 0.0

        results.append({
            "prevalence_pi": float(pi),
            "prevalence_percent": f"{pi * 100:.2f}%",
            "ppv_precision": float(ppv),
            "npv_specificity": float(npv),
        })

    return results
