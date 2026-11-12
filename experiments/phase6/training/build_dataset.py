"""
Dataset Construction and Leakage-Resistant Domain Splitting for Phase 6.

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


def compute_file_sha256(path: Path) -> str:
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
    fragment_bearing = bool(frag or "#" in url)

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
        "fragment_bearing": fragment_bearing,
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
        "fragment_bearing",
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


def build_phase6_datasets(
    phiusiil_csv_path: Path,
    validated_pilot_csv_path: Path,
    splits_dir: Path,
    manifest_path: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build research dataset and perform leakage-free domain-separated split."""
    print("Executing Phase 6 M2: Dataset Construction and Domain Separation...")

    # 1. Load PhiUSIIL dataset
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

    # 2. Load Validated Benign Pilot Corpus
    print(f"Loading validated benign corpus from {validated_pilot_csv_path}...")
    pilot_raw = pd.read_csv(validated_pilot_csv_path)
    
    # Extract 18 features for pilot records
    pilot_features = []
    for u in pilot_raw["url"]:
        pilot_features.append(extract_url_features(u))
    pilot_feat_df = pd.DataFrame(pilot_features, columns=FEATURE_NAMES)
    pilot_feat_df["url"] = pilot_raw["url"].values
    pilot_feat_df["label"] = 1  # Legitimate
    pilot_feat_df["source_dataset"] = "phase6_validated_pilot"

    # Deduplicate exact URLs against PhiUSIIL and within pilot
    pilot_feat_df = pilot_feat_df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    overlap_with_phi = set(pilot_feat_df["url"]) & set(phi_df["url"])
    if overlap_with_phi:
        print(f"Warning: Dropping {len(overlap_with_phi)} pilot URLs matching PhiUSIIL exact strings")
        pilot_feat_df = pilot_feat_df[~pilot_feat_df["url"].isin(overlap_with_phi)].reset_index(drop=True)

    print(f"Validated pilot corpus: {len(pilot_feat_df)} unique records")

    # Combine datasets
    combined_df = pd.concat([phi_df, pilot_feat_df], ignore_index=True)
    print(f"Combined research dataset: {len(combined_df)} URLs total")

    # 3. Extract registered domains
    print("Resolving registered domains (public-suffix aware)...")
    url_domain_map = {}
    for u in combined_df["url"]:
        if u not in url_domain_map:
            url_domain_map[u] = extract_registered_domain(u)

    combined_df["registered_domain"] = combined_df["url"].map(url_domain_map)
    total_domains = combined_df["registered_domain"].nunique()
    print(f"Resolved {total_domains} unique registered domains across combined corpus.")

    # 4. Group splitting by registered domain
    print("Performing stratified registered-domain group partition (70% Train, 15% Val, 15% Test)...")
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

    # Verify zero domain leakage
    assert len(train_doms & val_doms) == 0, "Domain leakage between Train and Val!"
    assert len(train_doms & test_doms) == 0, "Domain leakage between Train and Test!"
    assert len(val_doms & test_doms) == 0, "Domain leakage between Val and Test!"
    print("Zero domain leakage verified mathematically: D_train ∩ D_val = ∅, D_train ∩ D_test = ∅, D_val ∩ D_test = ∅")

    # 5. Partition rows
    train_df = combined_df[combined_df["registered_domain"].isin(train_doms)].copy().reset_index(drop=True)
    val_df = combined_df[combined_df["registered_domain"].isin(val_doms)].copy().reset_index(drop=True)
    test_df = combined_df[combined_df["registered_domain"].isin(test_doms)].copy().reset_index(drop=True)

    # 6. Compute structural strata for every row
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

    # Save partition CSVs
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

    # Calculate strata breakdowns
    train_strata = compute_strata_breakdown(train_df)
    val_strata = compute_strata_breakdown(val_df)
    test_strata = compute_strata_breakdown(test_df)

    manifest = {
        "manifest_version": "phase6-split-manifest-v1",
        "phase": 6,
        "milestone": "M2",
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "target_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_urls": len(combined_df),
        "total_unique_domains": total_domains,
        "partitions": {
            "train": {
                "path": str(train_path.relative_to(REPO_ROOT)),
                "sha256": train_sha,
                "url_count": len(train_df),
                "domain_count": len(train_doms),
                "phishing_count": int((train_df["label"] == 0).sum()),
                "legitimate_count": int((train_df["label"] == 1).sum()),
                "validated_pilot_count": int((train_df["source_dataset"] == "phase6_validated_pilot").sum()),
                "strata_breakdown": train_strata,
            },
            "val": {
                "path": str(val_path.relative_to(REPO_ROOT)),
                "sha256": val_sha,
                "url_count": len(val_df),
                "domain_count": len(val_doms),
                "phishing_count": int((val_df["label"] == 0).sum()),
                "legitimate_count": int((val_df["label"] == 1).sum()),
                "validated_pilot_count": int((val_df["source_dataset"] == "phase6_validated_pilot").sum()),
                "strata_breakdown": val_strata,
            },
            "internal_test": {
                "path": str(test_path.relative_to(REPO_ROOT)),
                "sha256": test_sha,
                "url_count": len(test_df),
                "domain_count": len(test_doms),
                "phishing_count": int((test_df["label"] == 0).sum()),
                "legitimate_count": int((test_df["label"] == 1).sum()),
                "validated_pilot_count": int((test_df["source_dataset"] == "phase6_validated_pilot").sum()),
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

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"\n[M2 EXIT CRITERIA MET] Leakage-free dataset partitions created:\n"
        f"  Train: {len(train_df)} URLs across {len(train_doms)} domains (SHA-256: {train_sha[:16]}...)\n"
        f"  Val:   {len(val_df)} URLs across {len(val_doms)} domains (SHA-256: {val_sha[:16]}...)\n"
        f"  Test:  {len(test_df)} URLs across {len(test_doms)} domains (SHA-256: {test_sha[:16]}...)\n"
        f"  Manifest: {manifest_path.relative_to(REPO_ROOT)}"
    )

    return train_df, val_df, test_df, manifest


if __name__ == "__main__":
    phi_csv = REPO_ROOT / "data" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    val_csv = REPO_ROOT / "experiments" / "phase6" / "datasets" / "validated_benign_corpus.csv"
    splits_d = REPO_ROOT / "experiments" / "phase6" / "datasets" / "splits"
    man_path = REPO_ROOT / "experiments" / "phase6" / "datasets" / "manifests" / "split_manifest.json"

    build_phase6_datasets(phi_csv, val_csv, splits_d, man_path)
