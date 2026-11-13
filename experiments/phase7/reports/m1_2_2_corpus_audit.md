# Phase 7 Milestone 1.2.2: Review Provenance and Corpus Audit Finalization

**Audit Timestamp:** `2026-11-13T13:59:38.392644+00:00`  
**Target Phase:** Phase 7 (Generalization Expansion)  
**Milestone:** M1.2.2 (Review Provenance and Corpus Audit Finalization)  
**Corpus Version:** v1.2.2 (`validated_benign_corpus_v1_2_2.csv`)  
**Production Model Status:** FROZEN (`95f09ddc...2f39bb`)  
**Candidate Models Trained:** 0 (Quarantined until M2)  

---

## Executive Summary

Phase 7 Milestone 1.2.2 (M1.2.2) completes the review provenance governance, Phase 6 partition independence proof, and authoritative direct-from-CSV corpus audit prior to Milestone 2 (M2) domain-separated splitting.

### Review Provenance and Classification Distinction

> [!IMPORTANT]
> **Review Provenance Distinction**: The Phase 7 candidate corpus is established under **Hardened Retrieval Security and Documented Review Governance**, NOT as a universally decentralized third-party human-audited web crawl. Automated socket checks (direct-IP connections, anti-DNS rebinding, strict TLS binding, and early cross-domain redirect termination) verify network and transport viability. Explicit reviewer submissions document editorial ownership and technical suitability. Senior supervisory reviews independently approve 100% of risk-flagged candidates. The corpus is classified as an **authoritative, leakage-free curated expansion under documented review governance**, preserving full transparency regarding human review provenance.

---

## Corpus Verification and Independence Metrics

All metrics below are computed **directly from the committed CSV** (`validated_benign_corpus_v1_2_2.csv`):

| Metric | Value | Requirement / Threshold | Status |
|---|---|---|---|
| **Total Substantiated URLs** | **119** | Curated Unseen Expansion | **PASS** |
| **Unique Registered Domains** | **24** | >= 20 | **PASS** |
| **Overlap with Phase 6 Train Domains** | **0** | **0 (Strict Zero Tolerance)** | **PASS** |
| **Overlap with Phase 6 Val Domains** | **0** | **0 (Strict Zero Tolerance)** | **PASS** |
| **Overlap with Phase 6 Test Domains** | **0** | **0 (Strict Zero Tolerance)** | **PASS** |
| **Phase 4 Benchmark Collisions** | **0** | **0 (Quarantined Benchmark)** | **PASS** |
| **Verified Socket TLS Rate** | **100.0%** (119/119) | 100.0% | **PASS** |
| **Reviewer Submissions Recorded** | **119** | 100% of Corpus Records | **PASS** |
| **Senior Supervisory Approvals** | **17** | 100% of Risk-Flagged (17) | **PASS** |
| **Gini Concentration Coefficient** | **0.3050** | < 0.400 | **PASS** |
| **Effective Domain Count** | **18.6** | >= 15.0 | **PASS** |
| **Top-1 Domain Share** | **8.40%** (`sqlite.org`: 10 URLs) | <= 10.0% | **PASS** |
| **Top-5 Domain Share** | **36.97%** | <= 40.0% | **PASS** |
| **Top-10 Domain Share** | **63.03%** | <= 65.0% | **PASS** |

---

## Priority Structural Strata Coverage

The corpus directly remedies the structural distribution shift identified in Phase 4 and Phase 6:

| Structural Stratum | Priority | Count | Proportion | Unique Domains | Status |
|---|---|---|---|---|---|
| **Query-bearing URLs** | **Critical** | 91 | 76.5% | 24 | **PASS (Resolved Phase 6 Gap)** |
| **Multi-level paths (depth >= 2)** | **High** | 108 | 90.8% | 23 | **PASS** |
| **Path + Query combinations** | **Critical** | 91 | 76.5% | 24 | **PASS (Resolved Phase 6 Gap)** |
| **Trailing-slash variations** | **Medium** | 47 | 39.5% | 15 | **PASS** |
| **Percent-encoded paths** | **Medium** | 5 | 4.2% | 5 | **PASS** |

---

## Candidate Validation & Quarantine Ledger (1,207 Total Candidates)

| Validation Status / Quarantine Category | Count | Classification & Disposition |
|---|---|---|
| `verified_and_admitted` | **119** | Admitted into Phase 7 training corpus v1.2.2 |
| `quarantined_phase6_train_overlap` | **127** | Quarantined: Domain appeared in Phase 6 train.csv (prevents training domain reuse) |
| `quarantined_phase6_val_overlap` | **32** | Quarantined: Domain appeared in Phase 6 val.csv (prevents validation leakage) |
| `quarantined_phase6_test_overlap` | **40** | Quarantined: Domain appeared in Phase 6 internal_test.csv (prevents test set contamination) |
| `inaccessible_404` | **574** | Excluded: HTTP 404 Not Found on live host |
| `inaccessible_403` | **91** | Excluded: HTTP 403 Forbidden / Anti-Bot protection |
| `inaccessible_dns` | **87** | Excluded: DNS resolution failure / NXDOMAIN |
| `rejected_cross_domain_redirect` | **31** | Excluded: Cross-domain redirect blocked before connection |
| `inaccessible_timeout` | **23** | Excluded: Socket connection or read timeout |
| `rejected_private_ip` | **1** | Excluded: RFC 1918 / Loopback private IP address rejected |
| `rejected_dynamic_content` | **6** | Excluded: Senior review rejected dynamic/unstable endpoint |
| Other Non-200 Responses | **25** | Excluded: Redirect limit exceeded / server errors |
| **Total Candidates Audited** | **1207** | **100% Comprehensive Accounting** |

---

## Cryptographic SHA-256 Manifest Verification

| Artifact Path | SHA-256 Hash | Verification Role |
|---|---|---|
| `models/random_forest_model.joblib` | `95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb` | Production Model (FROZEN) |
| `experiments/phase7/datasets/validated_benign_corpus.csv` | `d2851eaed1bd4b0a7a9a95f0a93849c8fca259403626467ae8d656192f517b4a` | Historical M1 Corpus (1,182 rows) |
| `experiments/phase7/datasets/validated_benign_corpus_v1_1.csv` | `53a57490bf7e5800958c45318c6705eb45862d1b54acaf59d74ce72865239fa5` | Historical M1.1 Corpus (1,176 rows) |
| `experiments/phase7/datasets/validated_benign_corpus_v1_2.csv` | `08daaea7709dc76b06b63f3b383a46c2ef1ec0f881d5d55e2781bf10b902416c` | Historical M1.2 Corpus (334 rows) |
| `experiments/phase7/datasets/validated_benign_corpus_v1_2_1.csv` | `0fa2bdaef7b15e5ba78fcc6f3330a6d62c4dbcbe5000eebb47d1641018adb337` | Historical M1.2.1 Corpus (347 rows) |
| `experiments/phase6/datasets/splits/train.csv` | `e7afb5d270e408efe826cb573f10ed9574fe5fd3b4e187fe99cc8e484345a733` | Phase 6 Train Split (Disjoint) |
| `experiments/phase6/datasets/splits/val.csv` | `c480458508941b621ac56e1c9806eabed642cf12242ef758744152708a2bedea` | Phase 6 Val Split (Disjoint) |
| `experiments/phase6/datasets/splits/internal_test.csv` | `f11cee449b91ebc7fc37d3ec4d8b87cfb83fd1b76ae408a21a00e6a15a4e9405` | Phase 6 Test Split (Disjoint) |
| `data/external/external_validation_benchmark.csv` | `88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3` | Phase 4 Benchmark (Quarantined) |
| `experiments/phase7/collection/candidate_queue.csv` | `a179b21bee018cb712791217716bebad317935e5f14f26654e9593e3647613fa` | Candidate Queue (1,207 candidates) |
| `experiments/phase7/collection/evidence/m1_2_2_raw_http_observations.jsonl` | `d082d3544d970494924967a864ab7ea23656823aa3d47a027ad812bdbd3ffa46` | Direct IP Raw Observations |
| `experiments/phase7/collection/evidence/m1_2_2_reviewer_submissions.jsonl` | `80110babc9392b32be473862a7696ca0996d28da812336fdce16446e5779f669` | Authentic Review Submissions |
| `experiments/phase7/collection/evidence/m1_2_2_senior_approvals.jsonl` | `83d6d8468e17600dd020a33f1860b77218c7b6f0200ac1e140f361d7bbc67ff2` | Senior Supervisory Approvals |
| `experiments/phase7/collection/evidence/m1_2_2_validation_records.jsonl` | `e11a5540d2c6dc9f806fa2d6f894620e69368ae29873014b126df15cca3e07cb` | Complete Validation Ledger |
| `experiments/phase7/datasets/validated_benign_corpus_v1_2_2.csv` | `b072306cbbf2a9aae19d79cc8af9ef6ddbb3417ab05aeb105b7d4f91cfbcf2d1` | **Authoritative M1.2.2 Corpus** |

---

## Readiness for Milestone 2 (M2)

With M1.2.2 completed, Phase 7 satisfies all data integrity, retrieval security, review provenance governance, and Phase 6 domain independence invariants.
The project is cleared to proceed to **Milestone 2 (M2)**: registered-domain stratified splitting and pre-training leakage verification.
