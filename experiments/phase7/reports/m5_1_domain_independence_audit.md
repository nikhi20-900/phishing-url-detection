# Phase 7 Milestone 5.1 — Domain Independence and Benchmark Qualification Audit

## Executive Summary
This audit certifies the complete qualification, evidence integrity, and strict domain independence of the **M5.1 Qualified External Benchmark** (`m5_1_qualified_benchmark.csv`).

- **Total URLs Qualified:** 1,300 (650 Phishing, 650 Legitimate)
- **Total Unique Registered Domains:** 810 (650 Phishing, 160 Legitimate)
- **Dynamic Historical Domain Exclusions:** 173,421 unique domains across 11 registered partitions
- **Historical Domain Collisions in Qualified Benchmark:** **0 (Certified 100% Zero-Overlap)**
- **Phishing-to-Legitimate Domain Collisions:** **0 (Certified 100% Zero-Overlap)**
- **Quarantined Historical Phishing Collisions:** 613 URLs
- **Socket Network Retrieval Evidence:** 100% authentic live observations; zero synthetic URL generation or synthetic observation fallbacks.
- **Upstream Threat Intelligence Feed:** PhishTank verified-online snapshot (SHA-256 `4885e37276d8a33b3da045d9fca0888c152e4d20c91e7a69f39eb032893a54ba`).
- **Cluster Domain Caps:** Phishing max 1 <= 3; Legitimate max 5 <= 5.

## Dynamic Historical Inventory
| Dataset Partition | Registered Domains | Collision Status |
|:-------------------|-------------------:|:-----------------|
| Phase 0 PhiUSIIL Raw | 134,850 | 0 Overlap |
| Phase 2 Cleaned Splits | 134,850 | 0 Overlap |
| Phase 3 Validation Split | 25,655 | 0 Overlap |
| Phase 4 External Benchmark | 884 | 0 Overlap |
| Phase 6 Augmented Candidates | 119 | 0 Overlap |
| Phase 7 Master Partitions (Train/Val/Test) | 170,615 | 0 Overlap |
| **Combined Master Historical Inventory** | **173,421** | **0 Overlap** |

## Cryptographic Artifact Hashes
| Artifact | Path | SHA-256 Hash |
|:---------|:-----|:-------------|
| Qualified Benchmark CSV | `experiments/phase7/external/datasets/m5_1_qualified_benchmark.csv` | `8e524bb539371c3e129037cf9a28d22b2f73a99f6201995fcc90951586c6110b` |
| Raw Observations JSONL | `experiments/phase7/external/evidence/raw_observations_m5_1.jsonl` | `8dfc8bd6f79b1a78434afc652b591d94cd78efda8282ddce45730c7df4ef6f06` |
| Quarantined Overlap CSV | `experiments/phase7/external/datasets/m5_1_quarantined_overlap.csv` | `038317d8bd61fb2c580328a9d167174fc8ca72302482d63050ddd657898ac935` |
| Pre-Evaluation Manifest | `experiments/phase7/reports/m5_1_pre_evaluation_manifest.json` | `d08fdaa6ad6862c4dc0689b29493374656790212a37991974e93dfc81a396279` |
| Frozen Production Control E0 | `models/random_forest_model.joblib` | `95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb` |
| Frozen Candidate E4 | `models/phase7_candidates/rf_augmented_e4.joblib` | `0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b` |
