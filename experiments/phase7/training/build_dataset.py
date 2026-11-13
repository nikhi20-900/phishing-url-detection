"""
Dataset Construction and Leakage-Free Domain Splitting for Phase 7 (Milestone 2).

Constructs the combined research dataset and partitions it using strict
registered-domain group separation:
- Train: 70% of registered domain groups
- Validation: 15% of registered domain groups
- Internal Test: 15% of registered domain groups

Enforces non-negotiable invariants:
- Zero domain overlap: D_train ∩ D_val = ∅, D_train ∩ D_test = ∅, D_val ∩ D_test = ∅
- Label mapping: 0 = Phishing, 1 = Legitimate
- Exact URL deduplication
- Structural strata tracking across all partitions and classes
- Complete reproducible split manifest with SHA-256 hashes
- Frozen model and benchmark quarantine controls

Outputs:
- experiments/phase7/datasets/splits/train.csv
- experiments/phase7/datasets/splits/val.csv
- experiments/phase7/datasets/splits/internal_test.csv
- experiments/phase7/datasets/manifests/split_manifest.json
- experiments/phase7/reports/m2_split_manifest.json
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.evaluate_domain_split import extract_registered_domain
from src.features import FEATURE_NAMES, extract_url_features

REPO_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_BENIGN_CORPUS_SHA256 = "b072306cbbf2a9aae19d79cc8af9ef6ddbb3417ab05aeb105b7d4f91cfbcf2d1"
EXPECTED_FROZEN_MODEL_SHA256 = "95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def compute_url_structural_strata(url: str, special_char_count: int, url_length: int) -> Dict[str, bool]:
    """Compute structural stratum flags for a URL."""
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        host = parsed.hostname or ""
        query = parsed.query or ""
        frag = parsed.fragment or ""
    except Exception:
        path, host, query, frag = "", "", "", ""

    path_segments = [s for s in path.strip("/").split("/") if s]

    is_homepage = (path in ("", "/") and not query and not frag)
    single_level_path = (len(path_segments) == 1)
    multi_level_path = (len(path_segments) >= 2)
    trailing_slash = url.endswith("/")
    query_bearing = bool(query or "?" in url)
    path_and_query = bool((single_level_path or multi_level_path) and query_bearing)
    fragment_bearing = bool(frag or "#" in url)
    percent_encoded_path = bool(re.search(r"%[0-9a-fA-F]{2}", path))

    # Subdomain (excluding standard www prefix)
    host_no_www = re.sub(r"^www\.", "", host.lower())
    subdomain_bearing = (host_no_www.count(".") >= 2) or (host.count(".") >= 2 and not host.startswith("www."))

    high_special_char = (special_char_count >= 2)
    short_url = (url_length <= 35)
    long_url = (url_length > 75)

    return {
        "homepage": is_homepage,
        "single_level_path": single_level_path,
        "multi_level_path": multi_level_path,
        "trailing_slash": trailing_slash,
        "query_bearing": query_bearing,
        "path_and_query": path_and_query,
        "fragment_bearing": fragment_bearing,
        "percent_encoded_path": percent_encoded_path,
        "subdomain_bearing": subdomain_bearing,
        "high_special_char": high_special_char,
        "short_url": short_url,
        "long_url": long_url,
    }


def compute_strata_breakdown(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate strata counts and proportions overall and by class."""
    total = len(df)
    if total == 0:
        return {}

    strata_keys = [
        "homepage",
        "single_level_path",
        "multi_level_path",
        "trailing_slash",
        "query_bearing",
        "path_and_query",
        "fragment_bearing",
        "percent_encoded_path",
        "subdomain_bearing",
        "high_special_char",
        "short_url",
        "long_url",
    ]

    breakdown: Dict[str, Any] = {"total_urls": total}

    for key in strata_keys:
        count = int(df[key].sum())
        pct = round(float(count / total * 100), 2)
        breakdown[key] = {"count": count, "percent": pct}

    by_class: Dict[str, Any] = {}
    for cls_val, cls_name in [(0, "phishing"), (1, "legitimate")]:
        sub_df = df[df["label"] == cls_val]
        c_tot = len(sub_df)
        c_stats: Dict[str, Any] = {"count": c_tot}
        if c_tot > 0:
            for key in strata_keys:
                c_cnt = int(sub_df[key].sum())
                c_stats[key] = {
                    "count": c_cnt,
                    "percent": round(float(c_cnt / c_tot * 100), 2),
                }
        by_class[cls_name] = c_stats
    breakdown["by_class"] = by_class

    return breakdown


def build_phase7_datasets(
    phiusiil_csv_path: Path,
    benign_corpus_csv_path: Path,
    splits_dir: Path,
    manifest_paths: List[Path],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build Phase 7 research dataset and perform leakage-free domain-separated split."""
    print("=" * 70)
    print("Executing Phase 7 M2: Dataset Construction and Domain Separation...")
    print("=" * 70)

    # 0. Verify locked protocol invariants before building dataset
    print("Verifying locked protocol invariants...")
    actual_benign_sha = compute_file_sha256(benign_corpus_csv_path)
    if actual_benign_sha != EXPECTED_BENIGN_CORPUS_SHA256:
        raise ValueError(f"Corpus hash mismatch! Expected {EXPECTED_BENIGN_CORPUS_SHA256}, got {actual_benign_sha}")
    print(f"  [PASS] Locked benign corpus hash verified ({actual_benign_sha[:16]}...).")

    frozen_model_path = REPO_ROOT / "models" / "random_forest_model.joblib"
    actual_model_sha = compute_file_sha256(frozen_model_path)
    if actual_model_sha != EXPECTED_FROZEN_MODEL_SHA256:
        raise ValueError(f"Production model hash mismatch! Expected {EXPECTED_FROZEN_MODEL_SHA256}, got {actual_model_sha}")
    print(f"  [PASS] Frozen production model verified ({actual_model_sha[:16]}...).")

    # 1. Load PhiUSIIL dataset (phishing source data + baseline legitimate data)
    print(f"Loading PhiUSIIL dataset from {phiusiil_csv_path}...")
    phi_cols = ["URL", "label"] + FEATURE_NAMES
    phi_df = pd.read_csv(phiusiil_csv_path, usecols=phi_cols)
    phi_df = phi_df.rename(columns={"URL": "url"})
    phi_df["source_dataset"] = "phiusiil"

    # Deduplicate exact URLs in PhiUSIIL
    phi_pre_dedup = len(phi_df)
    phi_df = phi_df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    phi_dedup_dropped = phi_pre_dedup - len(phi_df)
    print(f"PhiUSIIL: {len(phi_df)} unique URLs ({phi_dedup_dropped} duplicate rows dropped)")

    # 2. Load Validated Benign Expansion Corpus (119 URLs from M1.2.2)
    # Keeping benign expansion separate from phishing source data until combined split
    print(f"Loading validated benign expansion corpus from {benign_corpus_csv_path}...")
    benign_raw = pd.read_csv(benign_corpus_csv_path)
    if len(benign_raw) != 119:
        raise ValueError(f"Expected 119 benign URLs, found {len(benign_raw)}")

    # Extract 18 production features for benign expansion records
    print("Extracting 18 production features for benign expansion records...")
    benign_features = []
    for u in benign_raw["url"]:
        benign_features.append(extract_url_features(u))
    benign_feat_df = pd.DataFrame(benign_features, columns=FEATURE_NAMES)
    benign_feat_df["url"] = benign_raw["url"].values
    benign_feat_df["label"] = 1  # Legitimate
    benign_feat_df["source_dataset"] = "phase7_m1_2_2_benign"

    # Deduplicate exact URLs against PhiUSIIL and within benign corpus
    benign_feat_df = benign_feat_df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    overlap_with_phi = set(benign_feat_df["url"]) & set(phi_df["url"])
    if overlap_with_phi:
        raise ValueError(f"FATAL: Exact URL overlap with PhiUSIIL found: {overlap_with_phi}")
    print(f"Validated benign expansion corpus: {len(benign_feat_df)} unique records across {benign_raw['registered_domain'].nunique()} unseen domains.")

    # 3. Combine datasets
    combined_df = pd.concat([phi_df, benign_feat_df], ignore_index=True)
    print(f"Combined research dataset: {len(combined_df)} URLs total")

    # 4. Extract registered domains
    print("Resolving registered domains (public-suffix aware)...")
    url_domain_map = {}
    for u in combined_df["url"]:
        if u not in url_domain_map:
            url_domain_map[u] = extract_registered_domain(u)

    combined_df["registered_domain"] = combined_df["url"].map(url_domain_map)
    total_domains = combined_df["registered_domain"].nunique()
    print(f"Resolved {total_domains} unique registered domains across combined corpus.")

    # 5. Group splitting by registered domain
    print(f"Performing stratified registered-domain group partition (seed={seed}, {int(train_ratio*100)}% Train, {int(val_ratio*100)}% Val, {int(test_ratio*100)}% Test)...")
    domain_stats = combined_df.groupby("registered_domain")["label"].agg(["count", "mean"])

    # Stratified binning of domains by mean label (0.0 pure phish, 1.0 pure legit, mixed)
    domain_bins: Dict[str, List[str]] = {"pure_phish": [], "pure_legit": [], "mixed": []}
    for dom, row in domain_stats.iterrows():
        dom_str = str(dom)
        mean_val = row["mean"]
        if mean_val == 0.0:
            domain_bins["pure_phish"].append(dom_str)
        elif mean_val == 1.0:
            domain_bins["pure_legit"].append(dom_str)
        else:
            domain_bins["mixed"].append(dom_str)

    rng = np.random.RandomState(seed)
    train_doms: Set[str] = set()
    val_doms: Set[str] = set()
    test_doms: Set[str] = set()

    for bname, dom_list in domain_bins.items():
        arr = list(dom_list)
        rng.shuffle(arr)
        nb = len(arr)
        nb_train = int(nb * train_ratio)
        nb_val = int(nb * val_ratio)

        train_doms.update(arr[:nb_train])
        val_doms.update(arr[nb_train:nb_train + nb_val])
        test_doms.update(arr[nb_train + nb_val:])

    # 6. Verify mathematical zero domain leakage over combined dataset
    assert len(train_doms & val_doms) == 0, "Domain leakage between Train and Val!"
    assert len(train_doms & test_doms) == 0, "Domain leakage between Train and Test!"
    assert len(val_doms & test_doms) == 0, "Domain leakage between Val and Test!"
    print("  [PASS] Zero domain leakage verified mathematically: D_train ∩ D_val = ∅, D_train ∩ D_test = ∅, D_val ∩ D_test = ∅")

    # 7. Partition rows
    train_df = combined_df[combined_df["registered_domain"].isin(train_doms)].copy().reset_index(drop=True)
    val_df = combined_df[combined_df["registered_domain"].isin(val_doms)].copy().reset_index(drop=True)
    test_df = combined_df[combined_df["registered_domain"].isin(test_doms)].copy().reset_index(drop=True)

    # 8. Compute structural strata for every row
    print("Computing structural strata across all partitions...")
    for split_df in [train_df, val_df, test_df]:
        strata_records = []
        for idx, r in split_df.iterrows():
            strata_records.append(
                compute_url_structural_strata(
                    r["url"],
                    int(r["NoOfOtherSpecialCharsInURL"]),
                    int(r["URLLength"]),
                )
            )
        strata_df = pd.DataFrame(strata_records)
        for col in strata_df.columns:
            split_df[col] = strata_df[col].values

    # 9. Save partition CSVs
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "internal_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_sha = compute_file_sha256(train_path)
    val_sha = compute_file_sha256(val_path)
    test_sha = compute_file_sha256(test_path)

    # 10. Calculate strata breakdowns
    train_strata = compute_strata_breakdown(train_df)
    val_strata = compute_strata_breakdown(val_df)
    test_strata = compute_strata_breakdown(test_df)

    # Track benign expansion distribution across partitions
    b_train = int((train_df["source_dataset"] == "phase7_m1_2_2_benign").sum())
    b_val = int((val_df["source_dataset"] == "phase7_m1_2_2_benign").sum())
    b_test = int((test_df["source_dataset"] == "phase7_m1_2_2_benign").sum())

    manifest = {
        "manifest_version": "phase7-split-manifest-v1",
        "phase": 7,
        "milestone": "M2",
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "target_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_urls": len(combined_df),
        "total_unique_domains": total_domains,
        "feature_schema": {
            "name": "existing_18_production_features",
            "feature_count": len(FEATURE_NAMES),
            "features": FEATURE_NAMES,
        },
        "labels": {
            "0": "Phishing",
            "1": "Legitimate",
        },
        "evaluation_protocol": {
            "primary_threshold": 0.50,
            "threshold_tuning_partition": "validation_only",
            "confirmatory_test_partition": "internal_test",
        },
        "benign_corpus_provenance": {
            "corpus_path": "experiments/phase7/datasets/validated_benign_corpus_v1_2_2.csv",
            "sha256": actual_benign_sha,
            "total_urls": len(benign_raw),
            "unique_domains": int(benign_raw["registered_domain"].nunique()),
            "partition_distribution": {
                "train_count": b_train,
                "val_count": b_val,
                "test_count": b_test,
                "total": b_train + b_val + b_test,
            },
        },
        "partitions": {
            "train": {
                "path": str(train_path.relative_to(REPO_ROOT)),
                "sha256": train_sha,
                "url_count": len(train_df),
                "domain_count": len(train_doms),
                "phishing_count": int((train_df["label"] == 0).sum()),
                "legitimate_count": int((train_df["label"] == 1).sum()),
                "benign_expansion_count": b_train,
                "strata_breakdown": train_strata,
            },
            "val": {
                "path": str(val_path.relative_to(REPO_ROOT)),
                "sha256": val_sha,
                "url_count": len(val_df),
                "domain_count": len(val_doms),
                "phishing_count": int((val_df["label"] == 0).sum()),
                "legitimate_count": int((val_df["label"] == 1).sum()),
                "benign_expansion_count": b_val,
                "strata_breakdown": val_strata,
            },
            "internal_test": {
                "path": str(test_path.relative_to(REPO_ROOT)),
                "sha256": test_sha,
                "url_count": len(test_df),
                "domain_count": len(test_doms),
                "phishing_count": int((test_df["label"] == 0).sum()),
                "legitimate_count": int((test_df["label"] == 1).sum()),
                "benign_expansion_count": b_test,
                "strata_breakdown": test_strata,
            },
        },
        "leakage_verification": {
            "domain_overlap_train_val": len(train_doms & val_doms),
            "domain_overlap_train_test": len(train_doms & test_doms),
            "domain_overlap_val_test": len(val_doms & test_doms),
            "leakage_detected": False,
        },
        "exit_criteria_met": True,
    }

    for m_path in manifest_paths:
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"  [PASS] Saved split manifest to {m_path.relative_to(REPO_ROOT)}")

    print(
        f"\n[M2 EXIT CRITERIA MET] Leakage-free dataset partitions created:\n"
        f"  Train: {len(train_df)} URLs across {len(train_doms)} domains (SHA-256: {train_sha[:16]}...)\n"
        f"  Val:   {len(val_df)} URLs across {len(val_doms)} domains (SHA-256: {val_sha[:16]}...)\n"
        f"  Test:  {len(test_df)} URLs across {len(test_doms)} domains (SHA-256: {test_sha[:16]}...)\n"
    )

    return train_df, val_df, test_df, manifest


if __name__ == "__main__":
    phi_csv = REPO_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    val_csv = REPO_ROOT / "experiments" / "phase7" / "datasets" / "validated_benign_corpus_v1_2_2.csv"
    splits_d = REPO_ROOT / "experiments" / "phase7" / "datasets" / "splits"
    ds_man_path = REPO_ROOT / "experiments" / "phase7" / "datasets" / "manifests" / "split_manifest.json"
    rep_man_path = REPO_ROOT / "experiments" / "phase7" / "reports" / "m2_split_manifest.json"

    build_phase7_datasets(phi_csv, val_csv, splits_d, [ds_man_path, rep_man_path])
