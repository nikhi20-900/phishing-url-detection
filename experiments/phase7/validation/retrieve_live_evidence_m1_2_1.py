"""
Batch Secure Retrieval Runner for Phase 7 Milestone 1.2.1 (M1.2.1).

Retrieves live HTTP/TLS evidence for all 1,207 candidates using the hardened SecureHTTPClient:
- Validates all DNS records before connecting; blocks mixed public/private IPs.
- Connects directly to the validated IP to prevent DNS rebinding.
- Verifies TLS and extracts certificate/cipher on the exact socket used for data transfer.
- Blocks cross-domain redirects before connecting to destination.
- Enforces 5.0s timeout and 512 KB response-size cap.

Outputs:
- experiments/phase7/collection/evidence/m1_2_1_raw_http_observations.jsonl
"""

import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from experiments.phase7.validation.secure_http_client import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_REDIRECTS,
    MAX_RESPONSE_BYTES,
    execute_secure_retrieval,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def retrieve_all_candidates_m1_2_1(
    queue_csv_path: Path,
    output_observations_path: Path,
    max_workers: int = 20,
) -> List[Dict[str, Any]]:
    """
    Retrieve live HTTP/TLS evidence for all candidate URLs in parallel with progress reporting.
    """
    df = pd.read_csv(queue_csv_path)
    print(f"Starting M1.2.1 secure live HTTP retrieval for {len(df)} candidates (max_workers={max_workers})...")

    candidates = [
        (row["candidate_id"], row["url"])
        for _, row in df.iterrows()
    ]

    observations = []
    output_observations_path.parent.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cand = {
            executor.submit(execute_secure_retrieval, cand_id, url): (cand_id, url)
            for cand_id, url in candidates
        }

        completed = 0
        total = len(candidates)
        t_start = time.perf_counter()

        for future in concurrent.futures.as_completed(future_to_cand):
            obs = future.result()
            observations.append(obs)
            completed += 1
            if completed % 100 == 0 or completed == total:
                elapsed = time.perf_counter() - t_start
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed}/{total}] securely retrieved ({rate:.1f} URLs/sec)...")

    # Sort observations by candidate_id order matching the queue
    cand_order = {cid: idx for idx, (cid, _) in enumerate(candidates)}
    observations.sort(key=lambda x: cand_order.get(x["candidate_id"], 999999))

    # Write JSONL
    with open(output_observations_path, "w", encoding="utf-8") as f:
        for obs in observations:
            f.write(json.dumps(obs) + "\n")

    print(f"[SUCCESS] Wrote {len(observations)} authentic observation records to {output_observations_path.relative_to(REPO_ROOT)}")
    return observations


if __name__ == "__main__":
    q_file = REPO_ROOT / "experiments" / "phase7" / "collection" / "candidate_queue.csv"
    out_file = REPO_ROOT / "experiments" / "phase7" / "collection" / "evidence" / "m1_2_1_raw_http_observations.jsonl"
    retrieve_all_candidates_m1_2_1(q_file, out_file, max_workers=20)

