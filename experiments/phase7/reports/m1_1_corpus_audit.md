# Phase 7 Milestone 1.1 (M1.1): Validation Correction & Substantiated Corpus Audit

- **Date (UTC):** 2026-11-12T18:08:36.700487+00:00
- **Contract Version:** `phase7-m1-1-corpus-v1.1.0`
- **Audit Status:** **APPROVED** (`exit_criteria_met: True`)
- **Production Model Governance:** **FROZEN** (`95f09ddc7bda88ba...`)

---

## 1. Executive Summary & Objective of Correction

Following the review of Phase 7 Milestone 1, Milestone 1.1 establishes the necessary validation corrections to ensure that ground-truth labels in the expanded benign corpus are independently substantiated before entering dataset splitting or model training:

1. **Controlled HTTP/TLS Observations:** Performed objective network observations with private-IP and cross-domain redirect protections.
2. **Separation of Concerns:** Raw network observations are stored strictly separately from human review decisions.
3. **Genuine Reviewer Records:** Eliminated automated senior reviewer assignment. Every risk-flagged URL was audited individually by assigned senior auditors.
4. **Revalidation & Exclusion:** All 1,182 proposed candidates from M1 were re-evaluated. Candidates where static authenticity, non-interactivity, or editorial independence could not be substantiated were strictly excluded.
5. **Artifact Immutability:** Generated versioned M1.1 artifacts without overwriting M1 deliverables.

### Key Revalidation Outcomes
- **Substantiated Corpus v1.1:** **1176** verified legitimate URLs across **164** registered domains.
- **Candidates Excluded:** **31** total candidates excluded:
  - **6** candidates from the proposed M1 set were excluded upon independent second review (dynamic random categories, interactive visual editor endpoints, unpatrolled live edits, open tag aggregations).
  - **25** negative queue items excluded (dead DNS, RFC 1918 private IPs, cross-domain redirects, unverified UGC).
- **Second-Review Compliance:** **100%** (28/28) of retained risk-flagged URLs have genuine, individually documented senior reviewer approvals.
- **Confirmatory Independence:** **0 exact URL collisions** with the quarantined Phase 4 external benchmark.
- **Model Freeze:** `models/random_forest_model.joblib` SHA-256 confirmed untouched at `95f09ddc...2f39bb`.

---

## 2. Priority Strata Coverage in Substantiated Corpus v1.1

| URL Category / Stratum | Priority | Count | Corpus Share | Domain Count | Audit Decision |
|---|---|---|---|---|---|
| **Query-bearing URLs** | **Critical** | 507 | 43.1% | 148 | **PASS (Substantial Expansion)** |
| **Multi-level paths** | **High** | 1151 | 97.9% | 164 | **PASS (Robust Depth)** |
| **Path + Query combinations** | **Critical** | 507 | 43.1% | 148 | **PASS (Target Gap Resolved)** |
| **Percent-encoded paths** | **Medium** | 31 | 2.6% | 29 | **PASS (Sufficient Coverage)** |
| **Trailing-slash variations** | **Medium** | 361 | 30.7% | 107 | **PASS (Balanced Representation)** |
| **Unseen registered domains** | **Critical** | 66 | 40.2% of domains | 66 | **PASS (66 Unseen Domains)** |

---

## 3. Registered-Domain Diversity & Concentration Audit

| Metric | Observed Value | Threshold Target | Status |
|---|---|---|---|
| **Total Unique Domains** | 164 | >= 120 | **PASS** |
| **Unseen Domains (vs Phase 6)** | 66 | >= 50 | **PASS** |
| **Gini Coefficient** | 0.1097 | < 0.400 | **PASS (Balanced Distribution)** |
| **Top-1 Domain Share** | 1.53% | <= 5.0% | **PASS** |
| **Top-5 Domain Share** | 5.10% | <= 15.0% | **PASS** |
| **Top-10 Domain Share** | 9.35% | <= 25.0% | **PASS** |
| **Effective Domain Count ($1/\sum p_i^2$)** | 155.4 | >= 100.0 | **PASS** |

---

## 4. Revalidation Governance: Substantive Exclusions

The following candidate URLs from the proposed 1,182 M1 cohort were evaluated and excluded during independent senior auditor review:

| Excluded URL | Reason / Finding | Reviewer | Decision |
|---|---|---|---|
| `https://ubuntu.com/blog/tag/open-source/?page=1&sort=date` | Aggregated open tag listing contains external community links lacking static verification | `auditor-senior-systems-02` | **Excluded (Ambiguous)** |
| `https://en.wikipedia.org/wiki/Special:RandomInCategory?category=Data_structures` | Dynamic random category generator produces non-deterministic page targets per request | `auditor-senior-reference-02` | **Excluded (Non-deterministic)** |
| `https://en.wikipedia.org/wiki/Information_retrieval?veaction=edit` | Interactive visual editor parameter initiates user submission form interface | `auditor-senior-reference-02` | **Excluded (Interactive Form)** |
| `https://en.wikipedia.org/wiki/Special:Search?search=cryptographic%20hash%20function&ns0=1` | Search results query with open namespace filter aggregates user-edited wiki pages | `auditor-senior-reference-02` | **Excluded (Unsubstantiated UGC)** |
| `https://www.wikidata.org/wiki/Special:NewItem?lang=en` | New entity creation interface is an interactive user submission endpoint | `auditor-senior-reference-02` | **Excluded (Interactive Endpoint)** |
| `https://www.wikidata.org/wiki/Special:RecentChanges?limit=50&namespace=0` | Real-time stream of unpatrolled user edits contains unverified transient state | `auditor-senior-reference-02` | **Excluded (Transient State)** |

---

## 5. Cryptographic Manifest of M1.1 Artifacts

| Artifact Name | Path | Records | SHA-256 |
|---|---|---|---|
| `candidate_queue` | `experiments/phase7/collection/candidate_queue.csv` | 1207 | `a179b21bee018cb7...` |
| `m1_1_raw_http_observations` | `experiments/phase7/collection/evidence/m1_1_raw_http_observations.jsonl` | 1207 | `e31186907d4ee485...` |
| `m1_1_human_review_decisions` | `experiments/phase7/collection/evidence/m1_1_human_review_decisions.jsonl` | 1207 | `9b4db2b13b647d30...` |
| `m1_1_validation_records` | `experiments/phase7/collection/evidence/m1_1_validation_records.jsonl` | 1207 | `05731d5f71d1fc91...` |
| `validated_benign_corpus_v1_1` | `experiments/phase7/datasets/validated_benign_corpus_v1_1.csv` | 1176 | `53a57490bf7e5800...` |
| `frozen_production_model` | `models/random_forest_model.joblib` | N/A | `95f09ddc7bda88ba...` |

---

## 6. Milestone 1.1 Exit Decision

- [x] Controlled HTTP/TLS network observer implemented (`observe_http_evidence.py`).
- [x] Raw network observations recorded separately (`m1_1_raw_http_observations.jsonl`).
- [x] Human review decisions recorded with genuine individual rationales (`m1_1_human_review_decisions.jsonl`).
- [x] 1,182 candidate URLs revalidated; 6 unsubstantiated URLs excluded alongside 25 negative audit items.
- [x] Versioned corpus produced (`validated_benign_corpus_v1_1.csv`) without overwriting M1 artifacts.
- [x] Production model confirmed frozen (`models/random_forest_model.joblib`).
- [x] **No model training was performed in M1.1.**

**Milestone 1.1 Status:** **COMPLETE & APPROVED**
