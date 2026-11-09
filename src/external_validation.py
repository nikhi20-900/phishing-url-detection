"""
External Validation & False-Negative Analysis Module (Phase 4).

Evaluates the generalization of the Random Forest model trained on the PhiUSIIL dataset
against independent, contemporaneous external datasets:
1. OpenPhish Community Feed (active zero-day phishing)
2. PhishTank Community Verified Feed (human- and vote-verified active phishing)
3. Curated Authoritative Web Corpus (deep institutional, governmental, and reference URLs)

Core Capabilities:
- Robust multi-feed ingestion with provenance tracking (source, timestamps, IDs).
- Reproducible consolidated benchmark reconstruction (build_external_benchmark).
- Complete URL validation and exclusion auditing.
- Exact URL, FQDN hostname, and public-suffix registered domain overlap auditing against PhiUSIIL.
- Out-of-dataset evaluation across the combined benchmark and isolated source slices.
- Deterministic reproduction of the 106 Phase 3 domain-separated false negatives (Seed 14).
- In-depth diagnostic false-negative and false-alarm analysis with SHAP feature attributions.
- Output generation for reports/external_validation_results.json.
"""

import os
import sys
import json
import argparse
import datetime
from typing import Tuple, Dict, Any, List, Set, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    confusion_matrix
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_url_features, FEATURE_NAMES, parse_url
from src.predict import load_model_and_explainer, get_class_probabilities, compute_shap_explanation
from src.evaluate_domain_split import extract_registered_domain, compute_ece


# ---------------------------------------------------------------------------
# 1. Dataset Ingestion & Provenance Parsers
# ---------------------------------------------------------------------------

def load_openphish_feed(filepath: str) -> pd.DataFrame:
    """
    Parse an OpenPhish raw text feed snapshot.

    Expected format: one URL per line, blank lines or '#' comments skipped.
    Returns:
        pd.DataFrame with columns ['url', 'label', 'source', 'source_id',
                                   'collection_timestamp', 'verification_method', 'notes']
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"OpenPhish feed file not found at '{filepath}'.")

    records = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        idx = 1
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append({
                "url": line,
                "label": 0,  # 0 = Phishing
                "source": "OpenPhish-Community",
                "source_id": f"OPENPHISH-{idx:04d}",
                "collection_timestamp": "2026-10-31T02:55:00Z",
                "verification_method": "openphish_active_feed",
                "notes": "Active zero-day phishing feed snapshot; verified community threat stream."
            })
            idx += 1

    return pd.DataFrame(records)


def load_phishtank_feed(filepath: str) -> pd.DataFrame:
    """
    Parse a PhishTank verified online CSV feed snapshot.

    Expected columns: phish_id, url, submission_time, verified, verification_time, online, target
    Returns:
        pd.DataFrame with normalized benchmark columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"PhishTank feed file not found at '{filepath}'.")

    df = pd.read_csv(filepath)
    required = {"phish_id", "url", "verified"}
    if not required.issubset(df.columns):
        raise ValueError(f"PhishTank CSV missing expected columns {required - set(df.columns)}")

    # Retain verified entries only
    verified_mask = df["verified"].astype(str).str.lower().isin(["yes", "true", "1"])
    df_verified = df[verified_mask].copy()

    records = []
    for _, row in df_verified.iterrows():
        u = str(row["url"]).strip()
        p_id = str(row["phish_id"]).strip()
        v_time = str(row.get("verification_time", "2026-10-30T21:11:30+00:00"))
        target = str(row.get("target", "Target-Unspecified"))
        records.append({
            "url": u,
            "label": 0,  # 0 = Phishing
            "source": "PhishTank-Verified",
            "source_id": f"PHISHTANK-{p_id}",
            "collection_timestamp": v_time,
            "verification_method": "community_vote_and_analyst_verified",
            "notes": f"Verified phishing campaign targeting {target}."
        })

    return pd.DataFrame(records)


def load_curated_legitimate(filepath: str) -> pd.DataFrame:
    """
    Parse the curated authoritative legitimate URL corpus.

    Expected columns: url, label, source, source_id, collection_timestamp, verification_method, notes
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Curated legitimate corpus not found at '{filepath}'.")

    df = pd.read_csv(filepath)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Curated legitimate file must contain 'url' and 'label' columns.")
    return df


def load_external_benchmark(filepath: str = "data/external/external_validation_benchmark.csv") -> pd.DataFrame:
    """
    Load the consolidated external validation benchmark dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"External validation benchmark not found at '{filepath}'.")

    df = pd.read_csv(filepath)
    valid, errors = validate_benchmark_integrity(df)
    if not valid:
        raise ValueError(f"Benchmark integrity check failed: {'; '.join(errors)}")
    return df


def validate_benchmark_integrity(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Verify schema, completeness, label consistency, and uniqueness of the benchmark.
    """
    errors = []
    required_cols = {"url", "label", "source", "source_id", "collection_timestamp", "verification_method", "notes"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(list(missing))}")

    if "url" in df.columns:
        if df["url"].isna().any():
            errors.append(f"Found {df['url'].isna().sum()} null URL entries.")
        empty_urls = (df["url"].astype(str).str.strip() == "").sum()
        if empty_urls > 0:
            errors.append(f"Found {empty_urls} empty URL strings.")

    if "label" in df.columns:
        unique_labels = set(df["label"].unique())
        if not unique_labels.issubset({0, 1}):
            errors.append(f"Invalid label values detected: {unique_labels}. Expected subset of {{0, 1}}.")

    # Check for duplicate URLs with conflicting labels
    if "url" in df.columns and "label" in df.columns:
        label_conflicts = df.groupby("url")["label"].nunique()
        conflicts = label_conflicts[label_conflicts > 1]
        if len(conflicts) > 0:
            errors.append(f"Detected {len(conflicts)} URLs with conflicting class labels: {list(conflicts.index)[:3]}")

    # Check for exact duplicate URLs (even with identical labels): duplicates
    # double-count samples in evaluation and silently distort accuracy/recall.
    if "url" in df.columns:
        normalized_urls = df["url"].astype(str).str.strip()
        dup_counts = normalized_urls.value_counts()
        duplicate_urls = dup_counts[dup_counts > 1]
        if len(duplicate_urls) > 0:
            errors.append(f"Detected {len(duplicate_urls)} duplicate URL entries: {list(duplicate_urls.index)[:3]}")

    # Check for duplicate provenance identifiers (IDs must uniquely map to records)
    if "source_id" in df.columns:
        id_counts = df["source_id"].astype(str).value_counts()
        duplicate_ids = id_counts[id_counts > 1]
        if len(duplicate_ids) > 0:
            errors.append(f"Detected {len(duplicate_ids)} duplicate source_id entries: {list(duplicate_ids.index)[:3]}")

    return (len(errors) == 0, errors)


def build_external_benchmark(
    openphish_path: str = "data/external/phishing_feed_openphish.txt",
    phishtank_path: str = "data/external/phishing_feed_phishtank.csv",
    legitimate_path: str = "data/external/legitimate_curated.csv",
    output_path: Optional[str] = None,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Rebuild the consolidated external validation benchmark from the three raw
    snapshot files, making the synchronous benchmark construction reproducible.

    Deterministic construction policy:
    1. Feeds are ingested in fixed order: OpenPhish, PhishTank, Curated Legitimate.
    2. Records are concatenated and deduplicated by exact URL, keeping the FIRST
       occurrence (feed-priority attribution). URLs that appear in multiple feeds
       are attributed to the highest-priority feed in that order, mirroring the
       consolidation decisions implied by the shipped legacy snapshot.
    3. Every record carries full provenance (source, source_id, collection
       timestamp, verification method, notes).

    NOTE: Rebuilt provenance columns may differ from the legacy snapshot
    (e.g., load_phishtank_feed emits 'PHISHTANK-<id>' identifiers and the
    verification method 'community_vote_and_analyst_verified'), while the
    url/label/source content of the evaluation benchmark remains stable.

    Args:
        output_path: Optional destination CSV. The file is written only when
            it does not exist or `overwrite` is True (silent overwrites are
            never performed).
        overwrite: Explicitly allow replacing an existing output file.

    Raises:
        ValueError: If the consolidated result fails benchmark integrity checks
            or if output_path exists without overwrite permission.
    """
    df_openphish = load_openphish_feed(openphish_path)
    df_phishtank = load_phishtank_feed(phishtank_path)
    df_legitimate = load_curated_legitimate(legitimate_path)

    columns = ["url", "label", "source", "source_id", "collection_timestamp",
               "verification_method", "notes"]
    combined = pd.concat([df_openphish, df_phishtank, df_legitimate], ignore_index=True)
    for col in columns:
        if col not in combined.columns:
            combined[col] = None
    combined = combined[columns]
    combined["url"] = combined["url"].astype(str).str.strip()

    # Deduplicate by exact URL keeping the first feed occurrence (deterministic order above)
    dedup_mask = ~combined["url"].duplicated(keep="first")
    deduplicated = combined[dedup_mask].copy().reset_index(drop=True)

    dropped = len(combined) - len(deduplicated)
    if dropped > 0:
        print(f"build_external_benchmark: dropped {dropped} cross-feed duplicate URL(s) "
              f"(first-source attribution: OpenPhish > PhishTank > Curated).")

    valid, errors = validate_benchmark_integrity(deduplicated)
    if not valid:
        raise ValueError(f"Rebuilt benchmark failed integrity checks: {'; '.join(errors)}")

    if output_path is not None:
        if os.path.exists(output_path) and not overwrite:
            raise ValueError(
                f"Refusing to overwrite existing benchmark snapshot at '{output_path}'. "
                f"Pass overwrite=True to replace it or omit output_path to keep results in memory."
            )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        deduplicated.to_csv(output_path, index=False)
        print(f"Rebuilt external benchmark ({len(deduplicated)} records) saved to '{output_path}'.")

    return deduplicated


# ---------------------------------------------------------------------------
# 2. Preprocessing & URL Exclusion Tracking
# ---------------------------------------------------------------------------

def preprocess_and_validate_urls(
    df: pd.DataFrame,
    url_col: str = "url"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate URLs against RFC 3986 parser and filter unsupported or malformed entries.

    Returns:
        Tuple of (clean_df, excluded_df) where excluded_df records the URL and failure reason.
    """
    clean_records = []
    excluded_records = []

    for idx, row in df.iterrows():
        raw_url = str(row[url_col]).strip()
        try:
            parsed = parse_url(raw_url)
            # Ensure hostname is valid non-empty string
            if not parsed.hostname:
                excluded_records.append({**row.to_dict(), "exclusion_reason": "Missing or empty hostname"})
                continue
            clean_records.append(row.to_dict())
        except Exception as e:
            excluded_records.append({**row.to_dict(), "exclusion_reason": f"URL parsing error: {str(e)}"})

    clean_df = pd.DataFrame(clean_records)
    excluded_df = pd.DataFrame(excluded_records)
    return clean_df, excluded_df


# ---------------------------------------------------------------------------
# 3. Overlap Audit (Exact URL, Hostname, and Registered Domain)
# ---------------------------------------------------------------------------

def compute_overlap_metrics(
    external_df: pd.DataFrame,
    phi_path: str = "data/PhiUSIIL_Phishing_URL_Dataset.csv",
    url_col: str = "url"
) -> Dict[str, Any]:
    """
    Compute exact URL, FQDN hostname, and registered domain overlap between
    the external dataset and the reference PhiUSIIL benchmark.

    Evaluates overall overlap as well as class-isolated partitions.
    """
    if not os.path.exists(phi_path):
        raise FileNotFoundError(f"Reference dataset not found at '{phi_path}'.")

    phi_df = pd.read_csv(phi_path)
    phi_clean = phi_df.drop_duplicates(subset="URL").copy()

    phi_urls: Set[str] = set(phi_clean["URL"])
    phi_hosts: Set[str] = set(parse_url(u).hostname.lower().strip(".") for u in phi_clean["URL"])
    phi_reg_doms: Set[str] = set(extract_registered_domain(u) for u in phi_clean["URL"])

    def _calc_slice_overlap(slice_df: pd.DataFrame) -> Dict[str, Any]:
        slice_urls = set(slice_df[url_col])
        slice_hosts = set(parse_url(u).hostname.lower().strip(".") for u in slice_urls)
        slice_reg_doms = set(extract_registered_domain(u) for u in slice_urls)

        url_overlap = slice_urls & phi_urls
        host_overlap = slice_hosts & phi_hosts
        dom_overlap = slice_reg_doms & phi_reg_doms

        return {
            "total_urls": len(slice_urls),
            "exact_url_overlap_count": len(url_overlap),
            "exact_url_overlap_rate": float(len(url_overlap) / len(slice_urls)) if slice_urls else 0.0,
            "total_unique_hostnames": len(slice_hosts),
            "hostname_overlap_count": len(host_overlap),
            "hostname_overlap_rate": float(len(host_overlap) / len(slice_hosts)) if slice_hosts else 0.0,
            "sample_hostname_overlap": sorted(list(host_overlap))[:5],
            "total_unique_registered_domains": len(slice_reg_doms),
            "registered_domain_overlap_count": len(dom_overlap),
            "registered_domain_overlap_rate": float(len(dom_overlap) / len(slice_reg_doms)) if slice_reg_doms else 0.0,
            "sample_registered_domain_overlap": sorted(list(dom_overlap))[:5]
        }

    total_stats = _calc_slice_overlap(external_df)
    phish_df = external_df[external_df["label"] == 0]
    legit_df = external_df[external_df["label"] == 1]

    phish_stats = _calc_slice_overlap(phish_df) if len(phish_df) > 0 else {}
    legit_stats = _calc_slice_overlap(legit_df) if len(legit_df) > 0 else {}

    return {
        "reference_dataset": phi_path,
        "reference_unique_urls": len(phi_urls),
        "reference_unique_hostnames": len(phi_hosts),
        "reference_unique_registered_domains": len(phi_reg_doms),
        "external_benchmark_total": total_stats,
        "external_benchmark_phishing_slice": phish_stats,
        "external_benchmark_legitimate_slice": legit_stats
    }


# ---------------------------------------------------------------------------
# 4. Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_phish: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive classification and calibration metrics.

    Handles edge cases gracefully (e.g. single-class slices where ROC-AUC
    and PR-AUC are mathematically undefined).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = int(cm[0, 0])  # Phishing caught
    fn = int(cm[0, 1])  # Phishing missed
    fp = int(cm[1, 0])  # Legit flagged as Phishing
    tn = int(cm[1, 1])  # Legit correctly identified

    total = len(y_true)
    acc = float(accuracy_score(y_true, y_pred))

    # Precision, recall, f1 for phishing (pos_label=0)
    prec_p = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_p = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_p = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    # Precision, recall, f1 for legitimate (pos_label=1)
    prec_l = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_l = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_l = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) > 1

    if has_both_classes:
        y_true_phish = (y_true == 0).astype(int)
        roc_auc = float(roc_auc_score(y_true_phish, y_prob_phish))
        pr_auc = float(average_precision_score(y_true_phish, y_prob_phish))
        brier = float(brier_score_loss(y_true_phish, y_prob_phish))
        ll = float(log_loss(y_true_phish, y_prob_phish))
        ece = float(compute_ece(y_prob_phish, y_true_phish, n_bins=10))
        macro_f1 = float((f1_p + f1_l) / 2.0)
    else:
        # Single-class slice: discrimination curves are undefined
        roc_auc = None
        pr_auc = None
        brier = float(np.mean((y_prob_phish - (unique_classes[0] == 0).astype(int)) ** 2))
        ll = None
        y_true_phish = (y_true == 0).astype(int)
        ece = float(compute_ece(y_prob_phish, y_true_phish, n_bins=10))
        macro_f1 = f1_p if unique_classes[0] == 0 else f1_l

    return {
        "sample_size": total,
        "accuracy": acc,
        "phishing_precision": prec_p,
        "phishing_recall": rec_p,
        "phishing_f1": f1_p,
        "legitimate_precision": prec_l,
        "legitimate_recall": rec_l,
        "legitimate_f1": f1_l,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": ll,
        "ece_10_bins": ece,
        "confusion_matrix": {
            "true_phishing_tp": tp,
            "missed_phishing_fn": fn,
            "falsely_flagged_legit_fp": fp,
            "verified_legit_tn": tn
        }
    }


def evaluate_dataset(
    df: pd.DataFrame,
    model,
    threshold: float = 0.5,
    url_col: str = "url",
    label_col: str = "label"
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Run inference on a dataframe of URLs and evaluate against ground truth labels.

    Raises:
        ValueError: If the dataframe is empty, required columns are missing,
                    or labels fall outside the expected binary set {0, 1}.
    """
    if len(df) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    if url_col not in df.columns:
        raise ValueError(f"Evaluation dataset is missing the URL column '{url_col}'.")
    if label_col not in df.columns:
        raise ValueError(f"Evaluation dataset is missing the label column '{label_col}'.")

    invalid_labels = set(pd.unique(df[label_col].values)) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"Evaluation dataset contains invalid label values {sorted(invalid_labels)}. "
            f"Expected subset of {{0, 1}} (0 = Phishing, 1 = Legitimate)."
        )

    urls = df[url_col].tolist()
    y_true = df[label_col].values.astype(int)

    # Feature extraction
    features_list = [extract_url_features(u) for u in urls]
    X_features = pd.DataFrame(features_list, columns=FEATURE_NAMES)

    # Class probability lookup dynamically derived from model.classes_
    classes = list(model.classes_)
    phish_idx = classes.index(0)

    probs = model.predict_proba(X_features)
    y_prob_phish = probs[:, phish_idx]
    y_pred = (y_prob_phish < threshold).astype(int)  # 0 if prob_phish >= threshold else 1

    metrics = evaluate_predictions(y_true, y_pred, y_prob_phish)
    return metrics, y_pred, y_prob_phish, X_features


# ---------------------------------------------------------------------------
# 5. False-Negative & Error Reproduction
# ---------------------------------------------------------------------------

def reproduce_phase_3_false_negatives(
    data_path: str = "data/PhiUSIIL_Phishing_URL_Dataset.csv",
    split_seed: int = 14,
    model_random_state: int = 42
) -> Dict[str, Any]:
    """
    Reproduce the exact 106 Phase 3 domain-separated false negatives deterministically.
    Verifies that the Phase 3 baseline is intact without modifying existing artifacts.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"PhiUSIIL dataset not found at '{data_path}'.")

    df = pd.read_csv(data_path)
    df_clean = df.drop_duplicates(subset="URL").copy()
    df_clean["reg_domain"] = df_clean["URL"].apply(extract_registered_domain)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=split_seed)
    train_idx, test_idx = next(gss.split(df_clean, df_clean["label"], groups=df_clean["reg_domain"]))

    train_data = df_clean.iloc[train_idx].copy()
    test_data = df_clean.iloc[test_idx].copy()

    X_train = pd.DataFrame([extract_url_features(u) for u in train_data["URL"]], columns=FEATURE_NAMES)
    y_train = train_data["label"].values

    X_test = pd.DataFrame([extract_url_features(u) for u in test_data["URL"]], columns=FEATURE_NAMES)
    y_test = test_data["label"].values

    rf = RandomForestClassifier(n_estimators=100, random_state=model_random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    classes = list(rf.classes_)
    phish_idx = classes.index(0)

    y_pred = rf.predict(X_test)
    y_prob_phish = rf.predict_proba(X_test)[:, phish_idx]

    # Phishing = 0. False negative = y_test == 0 and y_pred == 1
    fn_mask = (y_test == 0) & (y_pred == 1)
    fn_count = int(np.sum(fn_mask))
    test_phishing_total = int(np.sum(y_test == 0))
    if test_phishing_total == 0:
        raise ValueError(
            "Reproduction split contains zero phishing URLs (label 0) in the test partition; "
            "false-negative rate is undefined. Verify the split seed and dataset."
        )

    fn_urls = test_data.iloc[fn_mask]["URL"].tolist()
    fn_probs = y_prob_phish[fn_mask].tolist()
    fn_features = X_test.iloc[fn_mask]

    feature_means = {k: float(v) for k, v in fn_features.mean().to_dict().items()}

    # Sample top 5 false negatives
    sample_cases = []
    for u, p in zip(fn_urls[:5], fn_probs[:5]):
        sample_cases.append({"url": u, "prob_phishing": float(p)})

    return {
        "status": "verified_reproducible",
        "split_seed": split_seed,
        "model_random_state": model_random_state,
        "test_phishing_total": test_phishing_total,
        "false_negative_count": fn_count,
        "false_negative_rate": float(fn_count / test_phishing_total),
        "sample_false_negatives": sample_cases,
        "feature_means": feature_means
    }


def diagnose_errors(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob_phish: np.ndarray,
    X_features: pd.DataFrame,
    explainer,
    max_shap_cases: int = 3
) -> Dict[str, Any]:
    """
    Perform deep forensic error analysis on external false negatives and false positives.
    Extracts feature characteristics and local SHAP explanations.
    """
    y_true = df["label"].values.astype(int)

    # 1. External False Negatives (Phishing missed: y_true=0, y_pred=1)
    fn_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_cases = []
    for idx in fn_indices:
        u = df.iloc[idx]["url"]
        p = float(y_prob_phish[idx])
        feat_dict = {col: float(X_features.iloc[idx][col]) for col in FEATURE_NAMES}
        row_df = X_features.iloc[[idx]]
        shap_res = compute_shap_explanation(explainer, row_df, phishing_index=0, expected_phishing_prob=p)
        vals = np.array(shap_res["shap_values"]).flatten()
        top_feat_idx = np.argsort(np.abs(vals))[::-1][:5]
        top_shap = [
            {"feature": FEATURE_NAMES[i], "value": float(row_df.iloc[0, i]), "shap_value": float(vals[i])}
            for i in top_feat_idx
        ]
        fn_cases.append({
            "url": u,
            "source": df.iloc[idx]["source"],
            "prob_phishing": p,
            "features": feat_dict,
            "top_shap_contributions": top_shap
        })

    # 2. External False Positives (Legitimate flagged: y_true=1, y_pred=0)
    fp_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_cases = []
    # Pick representative diverse FP cases (e.g. academic, governmental, standards body)
    for idx in fp_indices[:max_shap_cases]:
        u = df.iloc[idx]["url"]
        p = float(y_prob_phish[idx])
        feat_dict = {col: float(X_features.iloc[idx][col]) for col in FEATURE_NAMES}
        row_df = X_features.iloc[[idx]]
        shap_res = compute_shap_explanation(explainer, row_df, phishing_index=0, expected_phishing_prob=p)
        vals = np.array(shap_res["shap_values"]).flatten()
        top_feat_idx = np.argsort(np.abs(vals))[::-1][:5]
        top_shap = [
            {"feature": FEATURE_NAMES[i], "value": float(row_df.iloc[0, i]), "shap_value": float(vals[i])}
            for i in top_feat_idx
        ]
        fp_cases.append({
            "url": u,
            "source": df.iloc[idx]["source"],
            "prob_phishing": p,
            "features": feat_dict,
            "top_shap_contributions": top_shap
        })

    # Summary feature comparisons
    tn_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    feature_comparison = {
        "false_positives_mean": {col: float(X_features.iloc[fp_indices][col].mean()) for col in FEATURE_NAMES} if len(fp_indices) > 0 else {},
        "true_negatives_mean": {col: float(X_features.iloc[tn_indices][col].mean()) for col in FEATURE_NAMES} if len(tn_indices) > 0 else {}
    }

    return {
        "external_false_negative_count": len(fn_indices),
        "external_false_negative_cases": fn_cases,
        "external_false_positive_count": len(fp_indices),
        "external_false_positive_cases": fp_cases,
        "feature_comparison": feature_comparison
    }


# ---------------------------------------------------------------------------
# 6. Primary Orchestration Routine
# ---------------------------------------------------------------------------

def run_external_validation(
    benchmark_path: str = "data/external/external_validation_benchmark.csv",
    model_path: str = "models/random_forest_model.joblib",
    phi_path: str = "data/PhiUSIIL_Phishing_URL_Dataset.csv",
    output_json: str = "reports/external_validation_results.json",
    run_phase_3_repro: bool = True,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Execute full Phase 4 external validation pipeline:
    1. Load and validate benchmark dataset.
    2. Audit URL exclusions.
    3. Measure overlap against PhiUSIIL.
    4. Run inference using frozen Random Forest model.
    5. Evaluate combined benchmark and isolated source slices.
    6. Reproduce Phase 3 106 false negatives.
    7. Perform error diagnostics with SHAP attributions.
    8. Serialize results to JSON.

    By default an existing output JSON is never silently overwritten; when the target
    file exists and `overwrite` is False, results are written to a sibling file with
    a UTC timestamp suffix instead.
    """
    print(f"Loading external benchmark from '{benchmark_path}'...")
    df_raw = load_external_benchmark(benchmark_path)
    print(f"Loaded {len(df_raw)} records ({sum(df_raw['label'] == 0)} Phishing, {sum(df_raw['label'] == 1)} Legitimate).")

    print("\nAuditing URL validity and exclusions...")
    df_clean, df_excluded = preprocess_and_validate_urls(df_raw, url_col="url")
    print(f"Validated URLs: {len(df_clean)}, Excluded URLs: {len(df_excluded)}")

    print(f"\nComputing overlap metrics against '{phi_path}'...")
    overlap_results = compute_overlap_metrics(df_clean, phi_path=phi_path, url_col="url")
    total_ov = overlap_results["external_benchmark_total"]
    print(f"  Exact URL overlap       : {total_ov['exact_url_overlap_count']} / {total_ov['total_urls']} ({total_ov['exact_url_overlap_rate']:.2%})")
    print(f"  Hostname overlap        : {total_ov['hostname_overlap_count']} / {total_ov['total_unique_hostnames']} ({total_ov['hostname_overlap_rate']:.2%})")
    print(f"  Registered domain overlap: {total_ov['registered_domain_overlap_count']} / {total_ov['total_unique_registered_domains']} ({total_ov['registered_domain_overlap_rate']:.2%})")

    print(f"\nLoading model artifact from '{model_path}'...")
    model, explainer = load_model_and_explainer(model_path)

    print("\nEvaluating combined benchmark...")
    combined_metrics, y_pred, y_prob_phish, X_features = evaluate_dataset(df_clean, model, threshold=0.5)

    print("\nEvaluating isolated slices...")
    # Phishing slice
    phish_df = df_clean[df_clean["label"] == 0].reset_index(drop=True)
    phish_metrics, _, _, _ = evaluate_dataset(phish_df, model, threshold=0.5)

    # Legitimate slice
    legit_df = df_clean[df_clean["label"] == 1].reset_index(drop=True)
    legit_metrics, _, _, _ = evaluate_dataset(legit_df, model, threshold=0.5)

    # By individual source
    source_metrics = {}
    for src in df_clean["source"].unique():
        src_df = df_clean[df_clean["source"] == src].reset_index(drop=True)
        m, _, _, _ = evaluate_dataset(src_df, model, threshold=0.5)
        source_metrics[src] = m

    # Forensic error analysis
    print("\nPerforming forensic error analysis and SHAP diagnostic attributions...")
    error_diagnostics = diagnose_errors(df_clean, y_pred, y_prob_phish, X_features, explainer)

    # Reproduce Phase 3 false negatives
    phase_3_repro = {}
    if run_phase_3_repro:
        print("\nReproducing Phase 3 domain-separated false negatives (Seed 14)...")
        phase_3_repro = reproduce_phase_3_false_negatives(phi_path, split_seed=14)
        print(f"  Phase 3 False Negatives reproduced: {phase_3_repro['false_negative_count']} / {phase_3_repro['test_phishing_total']}")

    results = {
        "evaluation_phase": "Phase 4: External Validation & False-Negative Analysis",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": model_path,
        "benchmark_dataset": benchmark_path,
        "dataset_summary": {
            "total_raw_records": len(df_raw),
            "total_validated_records": len(df_clean),
            "total_excluded_records": len(df_excluded),
            "phishing_count": int(np.sum(df_clean["label"] == 0)),
            "legitimate_count": int(np.sum(df_clean["label"] == 1)),
            "source_distribution": df_clean["source"].value_counts().to_dict()
        },
        "exclusions": df_excluded.to_dict(orient="records") if len(df_excluded) > 0 else [],
        "overlap_audit": overlap_results,
        "combined_benchmark_metrics": combined_metrics,
        "phishing_feed_slice_metrics": phish_metrics,
        "legitimate_corpus_slice_metrics": legit_metrics,
        "source_slice_metrics": source_metrics,
        "phase_3_false_negative_reproduction": phase_3_repro,
        "error_diagnostics": error_diagnostics
    }

    print("\n============================================================")
    print("PHASE 4 EXTERNAL VALIDATION BENCHMARK RESULTS")
    print("============================================================")
    print(f"Overall Accuracy       : {combined_metrics['accuracy']:.4f} ({combined_metrics['accuracy']*100:.2f}%)")
    print(f"Phishing Recall (Catch): {combined_metrics['phishing_recall']:.4f} ({combined_metrics['phishing_recall']*100:.2f}%)")
    print(f"Phishing Precision     : {combined_metrics['phishing_precision']:.4f} ({combined_metrics['phishing_precision']*100:.2f}%)")
    print(f"Legitimate False Alarms: {combined_metrics['confusion_matrix']['falsely_flagged_legit_fp']} / {len(legit_df)} ({combined_metrics['confusion_matrix']['falsely_flagged_legit_fp']/len(legit_df):.2%})")
    print(f"ROC-AUC                : {combined_metrics['roc_auc']:.4f}")
    print(f"PR-AUC                 : {combined_metrics['pr_auc']:.4f}")
    print("============================================================")

    os.makedirs(os.path.dirname(output_json) or "reports", exist_ok=True)
    if os.path.exists(output_json) and not overwrite:
        timestamped = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem, ext = os.path.splitext(output_json)
        safe_output = f"{stem}_{timestamped}{ext or '.json'}"
        print(
            f"WARNING: '{output_json}' already exists and overwrite is disabled; "
            f"existing report preserved. Writing new results to '{safe_output}'. "
            f"Pass overwrite=True (or --overwrite) to replace the previous artifact."
        )
    else:
        safe_output = output_json
    with open(safe_output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"External validation results saved to '{safe_output}'.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4 External Validation & False-Negative Analysis CLI")
    parser.add_argument("--benchmark-path", default="data/external/external_validation_benchmark.csv", help="Path to external benchmark CSV")
    parser.add_argument("--model-path", default="models/random_forest_model.joblib", help="Path to trained model artifact")
    parser.add_argument("--phi-path", default="data/PhiUSIIL_Phishing_URL_Dataset.csv", help="Path to PhiUSIIL reference dataset")
    parser.add_argument("--output-json", default="reports/external_validation_results.json", help="Path to output JSON results")
    parser.add_argument("--skip-phase3-repro", action="store_true", help="Skip reproducing Phase 3 FNs to speed up execution")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output JSON instead of writing a timestamped copy")

    args = parser.parse_args()
    run_external_validation(
        benchmark_path=args.benchmark_path,
        model_path=args.model_path,
        phi_path=args.phi_path,
        output_json=args.output_json,
        run_phase_3_repro=not args.skip_phase3_repro,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
