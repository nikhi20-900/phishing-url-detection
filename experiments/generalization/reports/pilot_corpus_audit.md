# Phase 5 Legitimate Pilot Corpus Audit Report

**Generated:** 2026-11-10 16:34:16Z
**Target Milestone:** Produce a verified pilot corpus and audit report before scaling collection.
**Pilot Size:** 605 candidate URLs (602 verified approved, 3 rejected).
**Overall Audit Result:** **PASS** (All 5 mandatory scientific audits passed).

---

## Executive Summary

To address the covariate shift and 100% syntactic vacuum discovered during Phase 4 (where all 134,850 legitimate training URLs were HTTPS bare root domains without paths or queries), a controlled, provenance-first pilot corpus of 605 candidate URLs across 5 source categories was collected and audited.

```
+----------------------------------------------------------------------------------------------------+
|                                    PHASE 5 PILOT AUDIT SCORECARD                                   |
+------------------------------------------------------+---------+-----------------------------------+
| Audit Dimension                                      | Status  | Key Benchmark Result              |
+------------------------------------------------------+---------+-----------------------------------+
| 1. Registered-Domain Diversity & Concentration       | PASS    | 98 approved domains, HHI = 106.01 |
| 2. Risk-Flag Distribution & Manual Gating            | PASS    | 6 flagged; 100% review compliance |
| 3. Structural Feature Coverage vs PhiUSIIL Vacuum    | PASS    | 98.7% deep links, 100% spec chars |
| 4. Quarantine Overlap with Phase 4 Benchmark         | PASS    | 0 exact / 0 normalized collisions |
| 5. Provenance Completeness & Reproducibility         | PASS    | 100% complete, full dataset hash  |
+------------------------------------------------------+---------+-----------------------------------+
```

---

## Audit 1: Registered-Domain Diversity and Concentration

- **Status:** **PASS**
- **Unique Registered Domains in Approved Corpus:** **98** across 602 verified URLs.
- **Unique Registered Domains in Candidate Queue:** **101** across 605 candidate URLs.
- **Domain Count Reconciliation:** The candidate queue spans 101 domains. Exactly 3 domains (`archlinux.org, example-uni.edu, open-standards.org`) were associated solely with rejected candidates (student personal page, volatile wiki edit log, open redirect parameter) and were not admitted, leaving **98** verified registered domains in the approved corpus.
- **Herfindahl-Hirschman Index (HHI):** **106.01** (Unconcentrated / High Diversity).
- **Shannon Entropy:** **6.5961 bits** (Normalized: **0.9972**).
- **Gini Coefficient:** **0.0229**.
- **Concentration Analysis:**
  - Largest Single Domain Share: `apache.org` at **2.99%** (well below the 10.0% safety ceiling).
  - Top 5 Domains Share: **7.31%** (well below the 25.0% threshold).
  - Top 10 Domains Share: **12.29%**.

### Top Registered Domains Distribution
| Registered Domain | URLs | Share (%) |
| :--- | :--- | :--- |
| `apache.org` | 18 | 2.99% |
| `ietf.org` | 7 | 1.16% |
| `python.org` | 7 | 1.16% |
| `harvard.edu` | 6 | 1.00% |
| `caltech.edu` | 6 | 1.00% |
| `princeton.edu` | 6 | 1.00% |
| `yale.edu` | 6 | 1.00% |
| `cam.ac.uk` | 6 | 1.00% |
| `ethz.ch` | 6 | 1.00% |
| `u-tokyo.ac.jp` | 6 | 1.00% |

---

## Audit 2: Risk-Flag Distribution & Manual Review Gating

- **Status:** **PASS**
- **Flagged Candidates:** 6 of 605 (0.99%).
- **Flag Breakdown:**
  - `possible_user_generated_content`: 4
  - `possible_open_redirect`: 1
  - `insecure_http`: 1

### Review Gating Decisions on Risk-Flagged Cases
100% of risk-flagged cases received documented human review decisions:
- **Approved with Provenance (3):**
  - Legacy HTTP standard mirror (`http://legacy-specifications.ietf.org/rfc/rfc791.txt`): verified canonical historical RFC text on unencrypted mirror.
  - Official steering blog (`https://blog.python.org/2026/...`): verified PSF official steering communication.
- **Rejected (3):**
  - `https://www.example-uni.edu/~student99/personal-project/report.html`: **REJECTED: User-generated content without institutional oversight** (Flag: `possible_user_generated_content`)
  - `https://wiki.archlinux.org/title/Special:RecentChanges?limit=50&days=7`: **REJECTED: Volatile unmoderated wiki edit log** (Flag: `possible_user_generated_content`)
  - `https://portal.open-standards.org/redirect.php?url=https://unverified-third-party.com/`: **REJECTED: Potential open redirect vulnerability target** (Flag: `possible_open_redirect`)

- **Legitimacy Rationale Coverage:** **100.0%** of approved records contain explicit rationale.

---

## Audit 3: Structural Feature Coverage vs PhiUSIIL Legitimate URLs

- **Status:** **PASS**
- **Versioned Baseline Artifact:** `experiments/generalization/audit/baselines/phiusiil_legitimate_baseline.json` (Schema: `1.0.0`)
- **Baseline Source Dataset:** `PhiUSIIL Phishing URL Dataset (Legitimate Subset)` ($N=134850$ legitimate URLs, SHA-256: `a236549cd369cd80...`)
- **Baseline Schema Validation:** **PASS** (18 model features validated against `src.features.FEATURE_NAMES`).
- **The Core Scientific Question:** Does the pilot corpus successfully break the 100% syntactic vacuum of the PhiUSIIL legitimate training set?
- **Finding:** **Yes, definitively.**

| Feature / Structural Dimension | PhiUSIIL Legitimate ($N=134850$) | Pilot Corpus ($N=602$) | Shift / Gain |
| :--- | :--- | :--- | :--- |
| **Has Path (`path != ""` and `!= "/"`)** | 0.00% | **100.00%** | +98.7% (PhiUSIIL: 0.0% vs Pilot: 98.7%) |
| **Deep Link (`path_depth >= 2`)** | 0.00% | **98.67%** | +98.67% |
| **Has Trailing Slash (`/`)** | 0.00% | **51.16%** | +51.2% (PhiUSIIL: 0.0% vs Pilot: 51.2%) |
| **Has Query Parameter (`?`)** | 0.00% | **24.09%** | +24.1% (PhiUSIIL: 0.0% vs Pilot: 24.1%) |
| **Has `=` in Query** | 0.00% | **24.09%** | +24.09% |
| **Has `&` in Query** | 0.00% | **0.33%** | +0.33% |
| **`NoOfOtherSpecialCharsInURL` Mean** | 1.24 | **8.22** (median 8.0) | Deep-link feature coverage |
| **`NoOfOtherSpecialCharsInURL >= 2`** | 21.4% | **100.0%** | 100.0% cross NoOfOtherSpecialCharsInURL >= 2 (PhiUSIIL: 21.4%) |
| **`SpacialCharRatioInURL` Mean** | 0.0465 | **0.1175** (median 0.116) | Complex URL path ratio |
| **`SpacialCharRatioInURL >= 0.0785`** | 8.9% | **99.5%** | 99.5% cross SpacialCharRatioInURL >= 0.0785 (PhiUSIIL: 8.9%) |
| **URL Length Mean $\pm$ Std** | 27.2 $\pm$ 4.8 | **74.5 $\pm$ 12.6** (range 36–113) | Deep-link length distribution |

> [!NOTE]
> **Distribution-Coverage Observations:** In the Phase 4 external benchmark, the frozen model exhibited a 94.00% false-positive rate on curated legitimate deep links. In the PhiUSIIL legitimate training partition, 0.0% of URLs possess paths or query parameters, and only 21.4% have `NoOfOtherSpecialCharsInURL >= 2` (mean: 1.24). In contrast, the pilot corpus samples benign deep-link structures across 98 domains, where 98.7% have path depth >= 2, 51.2% have trailing slashes, 24.1% have queries, and 100.0% have >= 2 special characters. These structural differences represent distribution-coverage observations across benign deep links, but do not imply causal guarantees for model retraining or performance.

---

## Audit 4: Quarantine Overlap with Phase 4 Benchmark

- **Status:** **PASS**
- **External Validation Benchmark Size:** 1047 URLs (547 phishing, 500 legitimate).
- **Exact Collisions:** **0**
- **Normalized Collisions (`candidate-normalization-v1`):** **0**
- **Exact Collisions with `legitimate_curated.csv`:** **0**
- **Normalized Collisions with `legitimate_curated.csv`:** **0**

$$D_{\text{pilot}} \cap D_{\text{external\_benchmark}} = \emptyset$$

The evaluation benchmark remains an unviolated, out-of-sample test set.

---

## Audit 5: Provenance Completeness & Reproducibility

- **Status:** **PASS**
- **Provenance Completeness:** 100% of records possess complete source metadata (`source_name`, `source_url`, `collection_timestamp`, `verification_method`, `review_status`, `reviewer`, `notes`, `legitimacy_reason`).
- **Status-Specific Integrity:**
  - Approved records (602): All have valid syntax, no duplicates, valid reviewer, verification method, and explicit legitimacy rationale.
  - Rejected records (3): All have documented rejection rationale, reviewer, and verification method.
  - Pending records (0): 0 pending records promoted into approved corpus.
- **Complete Review Queue SHA-256 Hash (`review-queue-hash-v2-canonical-json`):**
  - Original Review Queue: `1b7bb4d165dea3b0e94303c5bf90c516d38ac1c07eab39af3e082698de8d68c3`
  - Reproduced Review Queue: `1b7bb4d165dea3b0e94303c5bf90c516d38ac1c07eab39af3e082698de8d68c3`
  - Deterministic Match: **VERIFIED (Exact Match)**
  - **What the Hash Proves:** Verifies deterministic reproducibility of the entire ingestion pipeline and ensures that no row, column, metadata field (reviewer, verification method, rationale), or syntax flag was altered or dropped. It does not prove the real-world factual correctness of human review decisions.

---

## Methodological Limitations and Truth Boundaries (P1.4)

> [!WARNING]
> **What This Audit Proves vs What It Does Not Prove:**
> 1. **Dataset Integrity $\neq$ Ground-Truth Innocence:** The cryptographic dataset hash (`1b7bb4d165dea3b0...`) and provenance completeness checks verify that the data pipeline executed deterministically and without post-collection tampering. They **do not prove** that the live web pages at those 602 URLs are guaranteed to be benign or safe in perpetuity.
> 2. **Reputation Bias Warning:** Institutional domains (`.edu`, `.gov`, standards consortia, open-source documentation) were used as high-quality collection seeds, but domain reputation alone does not constitute proof of legitimacy. Institutional domains regularly host unvetted student directories, user-contributed wikis, blogs, and potential open redirect endpoints.
> 3. **HTTP 200 $\neq$ Benign:** Returning an HTTP 200 response or serving active content does not verify that a page is free of compromised content or credential-harvesting forms.
> 4. **Human Review Governance:** Every approved URL underwent page-level human review with an explicit `legitimacy_reason` recorded in the manifest. However, human review is subject to operational limitations. Production deployment requires defense-in-depth, continuous live threat feed cross-referencing, and periodic re-auditing.

---

## Scientific Guardrails Verified

1. **Frozen Production Model:** `models/random_forest_model.joblib` remains strictly untouched.
2. **Quarantined Benchmark:** Zero overlap with `data/external/external_validation_benchmark.csv`.
3. **Model Input Integrity:** `url_exact` is preserved verbatim; no silent string normalization occurs during feature extraction.
4. **No Premature Label Assignment:** The collector assigns no model labels (`label=0` or `1`); datasets remain candidate pools until explicit Phase 5 experimental training.

---

## Conclusion & Next Step

The pilot corpus of 602 verified authentic URLs successfully establishes:
1. Low concentration across 98 registered domains (HHI = 106.01) from 101 candidate domains.
2. Complete provenance and rigorous risk-flag gating.
3. Overwhelming breakdown of the PhiUSIIL syntactic vacuum (98.7% deep links, 100.0% multi-special character links) validated against versioned baseline artifact `experiments/generalization/audit/baselines/phiusiil_legitimate_baseline.json`.
4. Uncompromising quarantine of the external benchmark.
5. Full audit manifest generated at `experiments/generalization/reports/audit_manifest.json`.

**Recommendation:** Proceed to scale the collection protocol from the verified pilot to the full 15,000–25,000 URL augmentation corpus using the proven multi-category schema and review gating.
