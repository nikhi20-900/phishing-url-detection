"""
Domain-Separated Model Evaluation (Phase 3 Task 3.3).

Implements rigorous out-of-domain evaluation using public-suffix-aware registered domain
grouping (GroupShuffleSplit) with strict zero-domain-overlap enforcement:
    D_train ∩ D_test = ∅

Evaluates a newly trained Random Forest exclusively on unseen domains to measure
the empirical generalization gap against the random-split baseline.
"""

import os
import sys
import json
import time
from typing import Tuple, Dict, Any, List, Set, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_url_features, FEATURE_NAMES, parse_url

# Comprehensive list of two-part ccTLD public suffixes to avoid incorrect splitting (e.g. co.uk)
TWO_PART_SUFFIXES: Set[str] = {
    # United Kingdom
    "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "ac.uk", "sch.uk", "gov.uk",
    # Australia
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au",
    # New Zealand
    "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz", "geek.nz", "school.nz",
    # Japan
    "co.jp", "ne.jp", "or.jp", "ac.jp", "ed.jp", "go.jp", "gr.jp", "lg.jp",
    # India
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in", "nic.in", "ac.in", "edu.in", "res.in", "gov.in",
    # Brazil
    "com.br", "net.br", "org.br", "gov.br", "edu.br", "art.br", "eco.br",
    # South Africa
    "co.za", "org.za", "net.za", "ac.za", "gov.za", "edu.za",
    # Mexico
    "com.mx", "org.mx", "net.mx", "edu.mx", "gob.mx",
    # Turkey
    "com.tr", "org.tr", "net.tr", "edu.tr", "gov.tr", "gen.tr", "biz.tr", "info.tr",
    # Argentina
    "com.ar", "org.ar", "net.ar", "gov.ar", "edu.ar",
    # Singapore & Hong Kong & Taiwan
    "com.sg", "net.sg", "org.sg", "edu.sg", "gov.sg",
    "com.hk", "net.hk", "org.hk", "edu.hk", "gov.hk", "idv.hk",
    "com.tw", "org.tw", "net.tw", "idv.tw", "gov.tw", "edu.tw",
    # Poland, Russia, Ukraine
    "com.pl", "org.pl", "net.pl", "biz.pl", "info.pl",
    "com.ru", "net.ru", "org.ru", "pp.ru",
    "com.ua", "net.ua", "org.ua", "edu.ua", "gov.ua",
    # China, Korea, Malaysia, Philippines, Pakistan, Nigeria, Indonesia, Israel, Thailand
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "co.kr", "ne.kr", "or.kr", "re.kr", "pe.kr", "go.kr", "edu.kr",
    "com.my", "net.my", "org.my", "edu.my", "gov.my",
    "com.ph", "net.ph", "org.ph", "edu.ph", "gov.ph",
    "com.pk", "net.pk", "org.pk", "edu.pk", "gov.pk",
    "com.ng", "org.ng", "net.ng", "edu.ng", "gov.ng",
    "co.id", "net.id", "or.id", "ac.id", "sch.id", "go.id", "web.id",
    "co.il", "org.il", "net.il", "ac.il", "k12.il", "gov.il", "muni.il",
    "co.th", "ac.th", "go.th", "net.th", "or.th"
}

def extract_registered_domain(url: str) -> str:
    """
    Extract the registered domain (public-suffix-aware) from an arbitrary URL.

    Correctly resolves:
    - Multi-level subdomains: 'https://login.example.com/account' -> 'example.com'
    - Multi-part ccTLDs: 'https://www.voicefmradio.co.uk' -> 'voicefmradio.co.uk'
    - Special ccTLDs: 'https://sub.ooty.ind.in/path' -> 'ooty.ind.in'
    - IPv4 literals: 'http://198.98.58.123:8080/path' -> '198.98.58.123'
    - IPv6 literals: 'http://[2001:db8::1]:8080/path' -> '2001:db8::1'
    - Authority credentials: 'https://user:pass@target.org/login' -> 'target.org'

    Returns:
        str: Normalized lowercase registered domain or IP address literal.
    """
    parsed = parse_url(url)
    host = parsed.hostname.lower().strip(".")
    if parsed.is_ip:
        return host

    parts = host.split(".")
    if len(parts) <= 2:
        return host

    last_two = ".".join(parts[-2:])
    if last_two in TWO_PART_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def verify_zero_domain_overlap(train_domains: Set[str], test_domains: Set[str]) -> None:
    """
    Programmatically enforce that D_train ∩ D_test = ∅.

    Raises:
        AssertionError: If any registered domain appears in both partitions.
    """
    overlap = train_domains.intersection(test_domains)
    if overlap:
        sample_overlap = list(overlap)[:5]
        raise AssertionError(
            f"Domain overlap violation! {len(overlap)} registered domains found in both train and test partitions: "
            f"{sample_overlap}..."
        )

def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) with uniform probability bins.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        if i < n_bins - 1:
            bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        else:
            bin_mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        bin_size = int(np.sum(bin_mask))
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[bin_mask]))
            bin_conf = float(np.mean(probs[bin_mask]))
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)

def find_balanced_group_split(
    df: pd.DataFrame,
    target_test_ratio: float = 0.20,
    candidate_seeds: int = 20,
    group_col: str = "reg_domain",
    label_col: str = "label"
) -> Tuple[int, np.ndarray, np.ndarray, float, float]:
    """
    Find candidate GroupShuffleSplit that best preserves target size and class balance
    while strictly keeping all registered domains grouped.

    Selection Criteria:
    - Test-set ratio close to target (20%).
    - Class distribution close to population prevalence.
    - Zero overlap between registered domains in training and testing.
    - Fixed, documented candidate-seed range (0..candidate_seeds-1) with deterministic scoring.

    IMPORTANT: Selection is strictly based on dataset partitioning characteristics.
    No model training, accuracy, precision, or recall metrics are used during split
    selection, eliminating any form of model-driven selection bias.
    """
    pop_phish_pct = float((df[label_col] == 0).mean())
    best_split = None
    best_score = float("inf")

    for seed in range(candidate_seeds):
        gss = GroupShuffleSplit(n_splits=1, test_size=target_test_ratio, random_state=seed)
        train_idx, test_idx = next(gss.split(df, df[label_col], groups=df[group_col]))

        # Enforce zero domain overlap for candidate
        train_doms = set(df.iloc[train_idx][group_col])
        test_doms = set(df.iloc[test_idx][group_col])
        verify_zero_domain_overlap(train_doms, test_doms)

        test_ratio = len(test_idx) / len(df)
        test_phish_pct = float((df.iloc[test_idx][label_col] == 0).mean())

        score = abs(test_ratio - target_test_ratio) * 2.0 + abs(test_phish_pct - pop_phish_pct)

        if score < best_score:
            best_score = score
            best_split = (seed, train_idx, test_idx, test_ratio, test_phish_pct)

    assert best_split is not None
    return best_split

def run_domain_separated_evaluation(
    data_path: str = "data/PhiUSIIL_Phishing_URL_Dataset.csv",
    split_seed: Optional[int] = None,
    candidate_seeds: int = 20,
    model_random_state: int = 42,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Execute full domain-separated evaluation:
    1. Deduplicate dataset by URL.
    2. Extract registered domains using public-suffix parser.
    3. Partition into train and held-out test domains using find_balanced_group_split()
       (or an explicit seed if specified).
    4. Verify zero domain overlap (D_train ∩ D_test = ∅).
    5. Train fresh Random Forest on domain-separated training partition.
    6. Evaluate on held-out unseen domains.
    7. Record exact metrics, calibration, and confusion matrix.
    """
    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    df_clean = df.drop_duplicates(subset="URL").copy()
    print(f"Clean unique URLs: {len(df_clean)}")

    print("Extracting registered domains with public-suffix awareness...")
    t0 = time.time()
    df_clean["reg_domain"] = df_clean["URL"].apply(extract_registered_domain)
    t1 = time.time()
    unique_domains = df_clean["reg_domain"].nunique()
    print(f"Extracted {unique_domains} registered domains in {t1 - t0:.2f}s")

    # Perform group split
    if split_seed is None:
        print(f"Searching {candidate_seeds} candidate seeds for balanced group split (target test ratio=20%)...")
        split_seed, train_idx, test_idx, test_ratio, test_phish_pct = find_balanced_group_split(
            df_clean, target_test_ratio=0.20, candidate_seeds=candidate_seeds,
            group_col="reg_domain", label_col="label"
        )
        selection_method = f"balanced_group_search(candidates={candidate_seeds})"
        print(f"Selected optimal Seed {split_seed}: Test size = {len(test_idx)} ({test_ratio:.2%}), Phish% = {test_phish_pct:.2%}")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=split_seed)
        train_idx, test_idx = next(gss.split(df_clean, df_clean["label"], groups=df_clean["reg_domain"]))
        selection_method = f"explicit_seed({split_seed})"
        test_ratio = len(test_idx) / len(df_clean)
        test_phish_pct = float((df_clean.iloc[test_idx]["label"] == 0).mean())

    train_data = df_clean.iloc[train_idx].copy()
    test_data = df_clean.iloc[test_idx].copy()

    # Enforce zero domain overlap
    train_domains = set(train_data["reg_domain"])
    test_domains = set(test_data["reg_domain"])
    verify_zero_domain_overlap(train_domains, test_domains)

    print("\n--- Domain Separation Verification ---")
    print(f"  Selection Method     : {selection_method}")
    print(f"  Split Seed           : {split_seed}")
    print(f"  Train Partition URLs : {len(train_data)} ({len(train_data)/len(df_clean):.2%}) across {len(train_domains)} domains")
    print(f"  Test Partition URLs  : {len(test_data)} ({len(test_data)/len(df_clean):.2%}) across {len(test_domains)} domains")
    print(f"  Overlap Check        : D_train ∩ D_test = ∅ (PASS, {len(train_domains & test_domains)} overlapping domains)")

    train_phish = int(np.sum(train_data["label"] == 0))
    train_legit = int(np.sum(train_data["label"] == 1))
    test_phish = int(np.sum(test_data["label"] == 0))
    test_legit = int(np.sum(test_data["label"] == 1))

    print(f"  Train Class Counts   : Phishing={train_phish} ({train_phish/len(train_data):.2%}), Legitimate={train_legit}")
    print(f"  Test Class Counts    : Phishing={test_phish} ({test_phish/len(test_data):.2%}), Legitimate={test_legit}")

    print("\nExtracting 18 lexical and structural features...")
    X_train = pd.DataFrame([extract_url_features(u) for u in train_data["URL"]], columns=FEATURE_NAMES)
    y_train = train_data["label"].values

    X_test = pd.DataFrame([extract_url_features(u) for u in test_data["URL"]], columns=FEATURE_NAMES)
    y_test = test_data["label"].values

    print(f"Training Random Forest (100 estimators, random_state={model_random_state}, n_jobs=-1)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=model_random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    classes = list(rf.classes_)
    phish_idx = classes.index(0)

    y_pred = rf.predict(X_test)
    y_prob_phish = rf.predict_proba(X_test)[:, phish_idx]

    # Metrics
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    acc = float(accuracy_score(y_test, y_pred))
    prec_p = float(precision_score(y_test, y_pred, pos_label=0))
    rec_p = float(recall_score(y_test, y_pred, pos_label=0))
    f1_p = float(f1_score(y_test, y_pred, pos_label=0))
    prec_l = float(precision_score(y_test, y_pred, pos_label=1))
    rec_l = float(recall_score(y_test, y_pred, pos_label=1))
    f1_l = float(f1_score(y_test, y_pred, pos_label=1))

    roc_auc = float(roc_auc_score((y_test == 0).astype(int), y_prob_phish))
    pr_auc = float(average_precision_score((y_test == 0).astype(int), y_prob_phish))
    brier = float(brier_score_loss((y_test == 0).astype(int), y_prob_phish))
    ll = float(log_loss((y_test == 0).astype(int), y_prob_phish))
    ece = float(compute_ece(y_prob_phish, (y_test == 0).astype(int), n_bins=10))

    results = {
        "evaluation_type": "domain_separated_holdout",
        "split_selection_method": selection_method,
        "split_seed": split_seed,
        "candidate_seeds_searched": candidate_seeds if selection_method.startswith("balanced_group_search") else None,
        "model_random_state": model_random_state,
        "total_dataset_urls": len(df_clean),
        "total_registered_domains": unique_domains,
        "train_urls": len(train_data),
        "train_domains": len(train_domains),
        "train_phishing": train_phish,
        "train_legitimate": train_legit,
        "test_urls": len(test_data),
        "test_domains": len(test_domains),
        "test_phishing": test_phish,
        "test_legitimate": test_legit,
        "domain_overlap": 0,
        "metrics": {
            "accuracy": acc,
            "phishing_precision": prec_p,
            "phishing_recall": rec_p,
            "phishing_f1": f1_p,
            "legitimate_precision": prec_l,
            "legitimate_recall": rec_l,
            "legitimate_f1": f1_l,
            "macro_f1": float((f1_p + f1_l) / 2.0),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "log_loss": ll,
            "ece_10_bins": ece
        },
        "confusion_matrix": {
            "true_phishing_tp": int(cm[0, 0]),
            "missed_phishing_fn": int(cm[0, 1]),
            "falsely_flagged_legit_fp": int(cm[1, 0]),
            "verified_legit_tn": int(cm[1, 1])
        }
    }

    print("\n============================================================")
    print("DOMAIN-SEPARATED EVALUATION RESULTS (Zero Domain Overlap)")
    print("============================================================")
    print(f"Accuracy         : {acc:.5f} ({acc*100:.3f}%)")
    print(f"Phishing Prec (0): {prec_p:.5f} ({prec_p*100:.3f}%)")
    print(f"Phishing Rec  (0): {rec_p:.5f} ({rec_p*100:.3f}%)")
    print(f"Phishing F1   (0): {f1_p:.5f} ({f1_p*100:.3f}%)")
    print(f"Legit Prec    (1): {prec_l:.5f} ({prec_l*100:.3f}%)")
    print(f"Legit Rec     (1): {rec_l:.5f} ({rec_l*100:.3f}%)")
    print(f"Legit F1      (1): {f1_l:.5f} ({f1_l*100:.3f}%)")
    print(f"ROC-AUC          : {roc_auc:.5f}")
    print(f"PR-AUC           : {pr_auc:.5f}")
    print(f"Log Loss         : {ll:.5f}")
    print(f"Brier Score      : {brier:.5f}")
    print(f"ECE (10 bins)    : {ece:.6f}")
    print("\nConfusion Matrix:")
    print(f"  True Positives (Phish Caught): {cm[0, 0]}")
    print(f"  False Negatives (Missed)     : {cm[0, 1]}")
    print(f"  False Positives (False Alarm): {cm[1, 0]}")
    print(f"  True Negatives (Legit Conf)  : {cm[1, 1]}")
    print("============================================================")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "domain_separated_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to '{out_path}'.")

    return results

if __name__ == "__main__":
    run_domain_separated_evaluation()
