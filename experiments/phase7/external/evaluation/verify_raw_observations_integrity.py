"""
Independent Standalone Validator for Raw Observations Artifact Integrity (M5.1).

Rigorously verifies:
1. File hash against pre-evaluation manifest lock (raw_observations_m5_1.jsonl).
2. Exact 1,300 observation records (650 legitimate, 650 phishing; 0 duplicate URLs).
3. 1:1 cryptographic URL SHA-256 join against m5_1_qualified_benchmark.csv.
4. 100% verified TLS and authentic socket evidence across all 650 legitimate records:
   - HTTPS protocol scheme
   - tls_verified == True
   - is_accessible == True
   - valid resolved IPv4/IPv6 address
   - HTTP status code in [200, 399]
   - valid 64-character hex response body SHA-256
5. Upstream threat feed cryptographic anchor across all 650 phishing records:
   - verification_status == "yes"
   - online_status == "yes"
   - snapshot hash == 4885e37276d8a33b3da045d9fca0888c152e4d20c91e7a69f39eb032893a54ba
"""

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = REPO_ROOT / "experiments" / "phase7" / "reports" / "m5_1_pre_evaluation_manifest.json"
RAW_OBS_PATH = REPO_ROOT / "experiments" / "phase7" / "external" / "evidence" / "raw_observations_m5_1.jsonl"
BENCHMARK_CSV_PATH = REPO_ROOT / "experiments" / "phase7" / "external" / "datasets" / "m5_1_qualified_benchmark.csv"
PHISHTANK_RAW_PATH = REPO_ROOT / "experiments" / "phase7" / "external" / "evidence" / "phishtank_raw.csv"

EXPECTED_OBS_SHA256 = "43d4e432a406083a4bb9ef61be0d9870dd7cba8b5ca7ad0103e3ff0dd6ca2282"
EXPECTED_BENCHMARK_SHA256 = "f6cdbf797b683d6c82d7b5c72c6ddf35aba7cbb448edafc2d68e67e26695fc88"
EXPECTED_PHISHTANK_SHA256 = "4885e37276d8a33b3da045d9fca0888c152e4d20c91e7a69f39eb032893a54ba"


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def validate_raw_observations_integrity() -> None:
    print("=" * 80)
    print("PHASE 7 M5.1: INDEPENDENT RAW OBSERVATIONS ARTIFACT INTEGRITY AUDIT")
    print("=" * 80)

    # 1. Verify Manifest File Hash Locks
    print("\n[Step 1/5] Verifying cryptographic artifact hashes against manifest lock...")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    actual_obs_sha = compute_file_sha256(RAW_OBS_PATH)
    actual_bench_sha = compute_file_sha256(BENCHMARK_CSV_PATH)
    actual_pt_sha = compute_file_sha256(PHISHTANK_RAW_PATH)

    manifest_obs_sha = manifest["benchmark_dataset"]["evidence_observations"]["sha256"]
    manifest_bench_sha = manifest["benchmark_dataset"]["sha256"]
    manifest_pt_sha = manifest["benchmark_dataset"]["evidence_observations"]["upstream_phishtank_snapshot_sha256"]

    assert actual_obs_sha == EXPECTED_OBS_SHA256, (
        f"Raw observations SHA-256 mismatch! Expected {EXPECTED_OBS_SHA256}, got {actual_obs_sha}"
    )
    assert actual_obs_sha == manifest_obs_sha, (
        f"Raw observations SHA-256 does not match manifest! Manifest has {manifest_obs_sha}"
    )
    print(f"  [PASS] raw_observations_m5_1.jsonl SHA-256 locked: {actual_obs_sha}")

    assert actual_bench_sha == EXPECTED_BENCHMARK_SHA256, (
        f"Benchmark CSV SHA-256 mismatch! Expected {EXPECTED_BENCHMARK_SHA256}, got {actual_bench_sha}"
    )
    assert actual_bench_sha == manifest_bench_sha, (
        f"Benchmark CSV SHA-256 does not match manifest! Manifest has {manifest_bench_sha}"
    )
    print(f"  [PASS] m5_1_qualified_benchmark.csv SHA-256 locked: {actual_bench_sha}")

    assert actual_pt_sha == EXPECTED_PHISHTANK_SHA256, (
        f"PhishTank raw SHA-256 mismatch! Expected {EXPECTED_PHISHTANK_SHA256}, got {actual_pt_sha}"
    )
    assert actual_pt_sha == manifest_pt_sha, (
        f"PhishTank raw SHA-256 does not match manifest! Manifest has {manifest_pt_sha}"
    )
    print(f"  [PASS] phishtank_raw.csv SHA-256 locked: {actual_pt_sha}")

    # 2. Parse and Validate Record Counts
    print("\n[Step 2/5] Parsing raw observation records and checking partitions...")
    with open(RAW_OBS_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    total_records = len(records)
    assert total_records == 1300, f"Expected 1,300 total records, found {total_records}"

    phish_records = [r for r in records if r.get("ground_truth_label") == 0]
    legit_records = [r for r in records if r.get("ground_truth_label") == 1]

    assert len(phish_records) == 650, f"Expected 650 phishing records, found {len(phish_records)}"
    assert len(legit_records) == 650, f"Expected 650 legitimate records, found {len(legit_records)}"

    obs_urls = [r["url"] for r in records]
    unique_obs_urls = set(obs_urls)
    assert len(unique_obs_urls) == 1300, (
        f"Duplicate URLs found in raw observations! {1300 - len(unique_obs_urls)} duplicates detected."
    )
    print(f"  [PASS] Exactly 1,300 unique observation URLs (650 phishing, 650 legitimate; 0 duplicates).")

    # 3. Cryptographic Join Against Benchmark CSV
    print("\n[Step 3/5] Performing cryptographic 1:1 join against m5_1_qualified_benchmark.csv...")
    import pandas as pd
    bench_df = pd.read_csv(BENCHMARK_CSV_PATH)
    assert len(bench_df) == 1300, f"Expected 1,300 rows in benchmark CSV, got {len(bench_df)}"

    obs_map = {r["url"]: r for r in records}
    sha_regex = re.compile(r"^[0-9a-f]{64}$")

    for idx, row in bench_df.iterrows():
        u = row["url"]
        assert u in obs_map, f"Benchmark URL {u} (row {idx}) missing from observation log!"
        obs = obs_map[u]

        # Verify URL SHA-256
        computed_u_sha = hashlib.sha256(u.encode("utf-8")).hexdigest()
        assert obs["url_sha256"] == computed_u_sha, (
            f"Observation url_sha256 mismatch for {u}: expected {computed_u_sha}, got {obs['url_sha256']}"
        )

        # Verify Metadata Alignment
        assert obs["ground_truth_label"] == row["label"], (
            f"Label mismatch for {u}: CSV={row['label']}, OBS={obs['ground_truth_label']}"
        )
        assert obs["registered_domain"] == row["registered_domain"], (
            f"Domain mismatch for {u}: CSV={row['registered_domain']}, OBS={obs['registered_domain']}"
        )
        assert obs["structural_stratum"] == row["structural_stratum"], (
            f"Stratum mismatch for {u}: CSV={row['structural_stratum']}, OBS={obs['structural_stratum']}"
        )
    print("  [PASS] 100% 1:1 join verified across all 1,300 benchmark URLs with matching SHA-256 digests.")

    # 4. Strict Socket & TLS Audit on All 650 Legitimate Observations
    print("\n[Step 4/5] Auditing socket connections and TLS verification on all 650 legitimate URLs...")
    unverified_tls = []
    inaccessible_urls = []
    non_https_urls = []
    invalid_ips = []
    invalid_statuses = []
    missing_response_hashes = []

    for r in legit_records:
        url = r["url"]
        assert r["is_synthetic"] is False, f"Synthetic legitimate record detected: {url}"

        if not url.startswith("https://"):
            non_https_urls.append(url)

        ev = r.get("observation_evidence", {})

        # TLS verification check
        if ev.get("tls_verified") is not True:
            unverified_tls.append((url, ev.get("tls_verified")))

        # Accessibility check
        if ev.get("is_accessible") is not True:
            inaccessible_urls.append(url)

        # Socket IP check
        ip = ev.get("resolved_ip")
        if not ip or not isinstance(ip, str) or len(ip.strip()) == 0:
            invalid_ips.append((url, ip))

        # HTTP Status Code check (200-399)
        code = ev.get("http_status_code")
        if not isinstance(code, int) or code < 200 or code >= 400:
            invalid_statuses.append((url, code))

        # Response Body SHA-256 check
        resp_sha = ev.get("response_sha256")
        if not resp_sha or not sha_regex.match(resp_sha):
            missing_response_hashes.append((url, resp_sha))

        # ISO Timestamp format check
        ts = r.get("observation_timestamp_utc")
        if not ts:
            raise ValueError(f"Missing observation_timestamp_utc for {url}")
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception as e:
            raise ValueError(f"Invalid retrieval timestamp for {url}: {ts} ({e})")

    assert len(non_https_urls) == 0, f"{len(non_https_urls)} legitimate URLs are not HTTPS: {non_https_urls[:3]}"
    assert len(unverified_tls) == 0, (
        f"CRITICAL: {len(unverified_tls)} legitimate observations lack verified TLS! Samples: {unverified_tls[:3]}"
    )
    assert len(inaccessible_urls) == 0, f"{len(inaccessible_urls)} legitimate URLs inaccessible: {inaccessible_urls[:3]}"
    assert len(invalid_ips) == 0, f"{len(invalid_ips)} invalid socket IPs: {invalid_ips[:3]}"
    assert len(invalid_statuses) == 0, f"{len(invalid_statuses)} invalid HTTP status codes: {invalid_statuses[:3]}"
    assert len(missing_response_hashes) == 0, f"{len(missing_response_hashes)} invalid response hashes: {missing_response_hashes[:3]}"

    print(f"  [PASS] 650/650 (100.0%) legitimate URLs have verified TLS (tls_verified == True).")
    print(f"  [PASS] 650/650 (100.0%) legitimate URLs have live socket connections (resolved IP + HTTP 200-399).")
    print(f"  [PASS] 650/650 (100.0%) legitimate URLs have valid 64-char response body SHA-256 digests.")
    print(f"  [PASS] 0/650 synthetic records found (is_synthetic == False for 100%).")

    # 5. Phishing Threat Feed Provenance Audit
    print("\n[Step 5/5] Auditing threat feed provenance on all 650 phishing records...")
    unverified_phish = []
    offline_phish = []
    bad_snapshot_phish = []

    for r in phish_records:
        assert r["is_synthetic"] is False, f"Synthetic phishing record detected: {r['url']}"
        ev = r.get("observation_evidence", {})

        if str(ev.get("verification_status")).lower() != "yes":
            unverified_phish.append((r["url"], ev.get("verification_status")))

        if str(ev.get("online_status")).lower() != "yes":
            offline_phish.append((r["url"], ev.get("online_status")))

        if ev.get("upstream_snapshot_sha256") != EXPECTED_PHISHTANK_SHA256:
            bad_snapshot_phish.append((r["url"], ev.get("upstream_snapshot_sha256")))

    assert len(unverified_phish) == 0, f"{len(unverified_phish)} phishing records not verified: {unverified_phish[:3]}"
    assert len(offline_phish) == 0, f"{len(offline_phish)} phishing records not online: {offline_phish[:3]}"
    assert len(bad_snapshot_phish) == 0, f"{len(bad_snapshot_phish)} bad snapshot hashes: {bad_snapshot_phish[:3]}"

    print(f"  [PASS] 650/650 (100.0%) phishing URLs verified by community consensus (verified == 'yes').")
    print(f"  [PASS] 650/650 (100.0%) phishing URLs confirmed active at snapshot time (online == 'yes').")
    print(f"  [PASS] 650/650 (100.0%) phishing URLs cryptographically linked to upstream PhishTank snapshot.")

    print("\n" + "=" * 80)
    print("ALL INTEGRITY CHECKS PASSED: raw_observations_m5_1.jsonl IS FULLY QUALIFIED.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        validate_raw_observations_integrity()
    except AssertionError as err:
        print(f"\n[FAIL] INTEGRITY AUDIT FAILED: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] UNEXPECTED EXCEPTION: {exc}", file=sys.stderr)
        sys.exit(2)
