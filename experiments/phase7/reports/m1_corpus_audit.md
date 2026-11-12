# Phase 7 Milestone 1 (M1): Legitimate URL Corpus Expansion & Audit Report

- **Date (UTC):** 2026-11-12T17:03:30.967394+00:00
- **Contract Version:** `phase7-m1-corpus-v1.0.0`
- **Audit Status:** **APPROVED** (`exit_criteria_met: True`)
- **Production Model Governance:** **FROZEN** (`95f09ddc7bda88ba...`)

---

## 1. Executive Summary & Objective

Phase 7 Milestone 1 delivers an independently validated legitimate URL expansion corpus engineered to resolve the structural distribution shifts observed during Phase 4 external validation and Phase 6 internal testing.

In Phase 6, although benign deep links improved path-bearing accuracy, query-bearing legitimate URLs remained a critical failure mode (FPR of 100% on the internal test query subgroup). Furthermore, model robustness on previously unseen registered domains required broader representation.

### Key Audit Findings
- **Corpus Size:** **1182** human-approved, verified legitimate URLs (expanded from 602 in Phase 6).
- **Domain Diversity:** **164** unique registered domains, of which **66** are brand-new and previously unseen in Phase 6.
- **Query-Bearing Coverage (Critical):** **513** query-bearing URLs (**43.4%**) across **148** domains (vs 145 in Phase 6).
- **Path + Query Combinations (Critical):** **513** URLs (**43.4%**).
- **Multi-Level Paths (High):** **1157** URLs (**97.9%**).
- **Percent-Encoded Paths (Medium):** **32** URLs across **29** domains.
- **Trailing-Slash Variations (Medium):** **361** URLs (**30.5%**).
- **Discriminative Review Pipeline:** **25** invalid/dead/ambiguous queue candidates were actively rejected and excluded from the training corpus.
- **Second-Review Compliance:** **100%** (34/34) of risk-flagged URLs have independent senior reviewer approvals.
- **Confirmatory Independence:** **0 exact URL collisions** with the quarantined Phase 4 external benchmark.
- **Model Freeze:** `models/random_forest_model.joblib` SHA-256 verified identical to Phase 3 control.

---

## 2. Priority Strata Coverage Matrix

| URL Category / Stratum | Priority | Count | Corpus Share | Domain Count | Audit Status |
|---|---|---|---|---|---|
| **Query-bearing URLs** | **Critical** | 513 | 43.4% | 148 | **PASS (Substantial Expansion)** |
| **Multi-level paths** | **High** | 1157 | 97.9% | 164 | **PASS (Broad Depth)** |
| **Path + Query combinations** | **Critical** | 513 | 43.4% | 148 | **PASS (Resolved Phase 6 Gap)** |
| **Percent-encoded paths** | **Medium** | 32 | 2.7% | 29 | **PASS (Adequate Representation)** |
| **Trailing-slash variations** | **Medium** | 361 | 30.5% | 107 | **PASS (Robust Coverage)** |
| **Unseen registered domains** | **Critical** | 66 | 40.2% of domains | 66 | **PASS (66 New Authoritative Domains)** |

---

## 3. Registered-Domain Diversity & Concentration Audit

To ensure that model training will not memorize a single dominant institutional domain, domain concentration was audited:

| Concentration Metric | Observed Value | Threshold Target | Status |
|---|---|---|---|
| **Total Unique Domains** | 164 | >= 120 | **PASS** |
| **Unseen Domains (vs Phase 6)** | 66 | >= 50 | **PASS** |
| **Gini Coefficient** | 0.1132 | < 0.450 | **PASS (Balanced Spread)** |
| **Top-1 Domain Share** | 1.52% | <= 5.0% | **PASS** |
| **Top-5 Domain Share** | 5.25% | <= 15.0% | **PASS** |
| **Top-10 Domain Share** | 9.48% | <= 25.0% | **PASS** |
| **Effective Domain Count ($1/\sum p_i^2$)** | 154.9 | >= 100.0 | **PASS** |

### Top 10 Registered Domains in Verified Corpus

| Rank | Registered Domain | URL Count | Proportion |
|---|---|---|---|
| 1 | `apache.org` | 18 | 1.52% |
| 2 | `nature.com` | 12 | 1.02% |
| 3 | `wikipedia.org` | 12 | 1.02% |
| 4 | `stanford.edu` | 10 | 0.85% |
| 5 | `berkeley.edu` | 10 | 0.85% |
| 6 | `ox.ac.uk` | 10 | 0.85% |
| 7 | `mit.edu` | 10 | 0.85% |
| 8 | `sciencemag.org` | 10 | 0.85% |
| 9 | `jstor.org` | 10 | 0.85% |
| 10 | `iana.org` | 10 | 0.85% |

---

## 4. Page-Level Authenticity & Review Governance Audit

> [!IMPORTANT]
> **Governance Invariant:** Automated HTTP status codes (such as HTTP 200 or HTTPS) do NOT establish legitimacy.
> Every candidate URL in the verified corpus required explicit source provenance, human reviewer assignment, timestamped review, and documented legitimacy rationale.

### Candidate Evaluation Breakdown
- **Total Candidates Evaluated:** 1207
- **Verified Pages Admitted (`verified_page`):** 1182 (97.9%)
- **Candidates Excluded / Filtered Out:** 25
  - **Rejected (Policy Breaches / Open Redirects / Private IPs):** 12
  - **Ambiguous (Unverified User-Generated Content):** 11
  - **Inaccessible (Dead Links / NXDOMAIN / Timeouts):** 2

### Second Review Compliance for Risk-Flagged Candidates
Under the Phase 7 protocol, any URL bearing risk flags (e.g. unencrypted HTTP, project engineering blogs, open knowledge platforms) must undergo independent second review.

- **Risk-Flagged Verified URLs:** 34
- **Second-Review Approvals:** 34 (100.0%)
- **Compliance Status:** **100% COMPLIANT**

---

## 5. Confirmatory Independence & Overlap Audit

- **Phase 4 External Benchmark Overlap:** **0 exact URLs** (quarantine preserved).
- **PhiUSIIL Legitimate Benchmark Overlap:** **0 exact URLs**.
- **Phase 6 Pilot Foundation:** 602 base pilot records re-verified and included, providing continuous structural continuity while adding 580 fresh expansion URLs.

---

## 6. Cryptographic Manifest of Generated Artifacts

| Artifact | Relative Path | Records | SHA-256 |
|---|---|---|---|
| `candidate_queue` | `experiments/phase7/collection/candidate_queue.csv` | 1207 | `a179b21bee018cb7...` |
| `automated_evidence` | `experiments/phase7/collection/evidence/automated_evidence.jsonl` | 1207 | `0e2bf3399cc68743...` |
| `validation_records` | `experiments/phase7/collection/evidence/validation_records.jsonl` | 1207 | `1d3c34a6a604427f...` |
| `validated_benign_corpus` | `experiments/phase7/datasets/validated_benign_corpus.csv` | 1182 | `d2851eaed1bd4b0a...` |
| `frozen_production_model` | `models/random_forest_model.joblib` | N/A | `95f09ddc7bda88ba...` |

---

## 7. Milestone 1 Exit Decision

- [x] Controlled collection pipeline implemented (`experiments/phase7/collection/collect_candidates.py`).
- [x] Candidate queue established (`experiments/phase7/collection/candidate_queue.csv`).
- [x] Automated evidence gathered under observational constraints (`experiments/phase7/collection/evidence/automated_evidence.jsonl`).
- [x] Human review schema and validation engine enforced (`experiments/phase7/validation/validate_candidates.py`).
- [x] 100% provenance and second-review compliance verified (`experiments/phase7/datasets/validated_benign_corpus.csv`).
- [x] Comprehensive corpus audit completed (`experiments/phase7/reports/m1_corpus_audit.md`).
- [x] Production model confirmed frozen (`models/random_forest_model.joblib`).
- [x] **No model training was performed in M1.**

**Milestone 1 Status:** **COMPLETE & APPROVED**
