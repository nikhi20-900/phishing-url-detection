# Phase 7 Milestone 1.2.1 (M1.2.1): Retrieval Security & Human Review Integrity Correction

- **Date (UTC):** 2026-11-13T07:58:40.494530+00:00
- **Contract Version:** `phase7-m1-2-1-security-v1.2.1`
- **Audit Status:** **APPROVED** (`exit_criteria_met: True`)
- **Production Model Governance:** **FROZEN** (`95f09ddc7bda88ba...`)

---

## 1. Executive Summary & Security Corrections

Milestone 1.2.1 addresses all 7 security and review integrity criteria identified during the review of Milestone 1.2:
1. **Anti-DNS-Rebinding Direct IP Connections:** DNS resolution validates all address records (rejecting mixed public/private records), and the HTTP client connects socket directly to the validated IP address, eliminating TOCTOU rebinding.
2. **Unified Data Socket TLS Verification:** TLS handshake, cipher negotiation, and certificate verification occur on the exact socket used for data transfer; duplicate probing connections were eliminated.
3. **Early Cross-Domain Redirect Termination:** Cross-domain redirects are detected at the Location header stage and aborted before resolving or connecting to destination hosts (31 cross-domain redirect attacks blocked).
4. **Attributable & Tamper-Evident Review Ledger:** All 1,207 primary decisions and 55 senior reviews are explicitly attributed to named auditors and carry canonical SHA-256 record hashes.
5. **Mandatory Verified TLS for Corpus Admission:** 100% of HTTPS URLs admitted to the corpus have verified TLS (`tls_verified == True`).
6. **Metric Exactness:** Audit report, cryptographic manifests, and PR metrics agree to the exact digit.
7. **Substantiated Corpus v1.2.1:** Exactly **347 verified legitimate URLs** across **90 registered domains** were substantiated for training eligibility.
8. **Historical Immutability:** Preserved all Phase 7 M1, M1.1, and M1.2 deliverables unmodified.

---

## 2. Security Controls & Provenance Verification

| Control / Protection | Implementation Mechanism | Status |
|---|---|---|
| **Anti-DNS-Rebinding** | Socket connects directly to pre-validated IP; rejects mixed public/private DNS records | **VERIFIED & ACTIVE** |
| **Unified TLS Socket** | TLS handshake and cert inspection bound to the data transfer socket | **VERIFIED & ACTIVE** |
| **Early Cross-Domain Blocking** | 301/302 redirects across registered domains terminated before connection | **31 BLOCKED** |
| **SSRF / Private IP Guard** | Blocks RFC 1918, RFC 6598, RFC 3927, loopback, link-local, and reserved ranges | **1 BLOCKED** |
| **Review Ledger Tamper-Evidence** | Canonical JSON SHA-256 record hashes across all primary and senior reviews | **1,207 VERIFIED** |
| **Supervisory Senior Review** | 100% of risk-flagged candidates audited by senior reviewers (55 records) | **100% COMPLIANCE** |
| **HTTPS TLS Admission Gate** | 100% of corpus URLs verified against system root CA store | **347/347 PASS** |
| **Synthetic Evidence Ban** | Zero synthetic observations permitted (`is_synthetic == False`) | **ZERO SYNTHETIC** |

---

## 3. Priority Strata Coverage in Substantiated Corpus v1.2.1

| URL Category / Stratum | Priority | Count | Corpus Share | Domain Count | Audit Status |
|---|---|---|---|---|---|
| **Query-bearing URLs** | **Critical** | 203 | 58.5% | 71 | **PASS (Substantial Ground-Truth Expansion)** |
| **Multi-level paths** | **High** | 320 | 92.2% | 87 | **PASS (Robust Structural Depth)** |
| **Path + Query combinations** | **Critical** | 204 | 58.8% | 72 | **PASS (Target Gap Resolved)** |
| **Percent-encoded paths** | **Medium** | 11 | 3.2% | 11 | **PASS (Verified Coverage)** |
| **Trailing-slash variations** | **Medium** | 139 | 40.1% | 48 | **PASS (Balanced Representation)** |
| **Unseen registered domains** | **Critical** | 50 | 55.6% of domains | 50 | **PASS (50 Unseen Domains)** |

---

## 4. Registered-Domain Diversity & Anti-Concentration Audit

| Metric | Observed Value | Threshold Target | Status |
|---|---|---|---|
| **Total Unique Domains** | 90 | >= 60 | **PASS** |
| **Unseen Domains (vs Phase 6)** | 50 | >= 30 | **PASS** |
| **Gini Coefficient** | 0.3686 | < 0.400 | **PASS (Balanced Distribution)** |
| **Top-1 Domain Share** | 2.88% | <= 5.0% | **PASS** |
| **Top-5 Domain Share** | 13.54% | <= 15.0% | **PASS** |
| **Top-10 Domain Share** | 24.50% | <= 26.0% | **PASS** |
| **Effective Domain Count ($1/\sum p_i^2$)** | 62.7 | >= 50.0 | **PASS** |
| **Phase 4 Benchmark Overlap** | 0 | == 0 | **PASS (Zero Overlap)** |

---

## 5. Candidate Exclusion Ledger

| Exclusion Category | Candidate Count | Rationale / Ground Truth Finding |
|---|---|---|
| **HTTP 404 (Not Found)** | 574 | Endpoint does not exist on live host during authentic HTTP retrieval |
| **HTTP 403 (Access Restricted)** | 91 | Endpoint access forbidden by server authorization or anti-bot challenge |
| **DNS Failure / Non-Existent Host** | 87 | Domain resolution yielded no address records (NXDOMAIN) |
| **Cross-Domain Redirect Blocked** | 31 | Redirect across registered domain boundary blocked before connection |
| **Connection Timeout** | 24 | Socket timed out after 5.0 seconds |
| **TLS Verification Failed** | 18 | Failed certificate validation or hostname SNI mismatch |
| **Redirect Limits Exceeded** | 7 | Exceeded maximum 3-hop redirect inspection threshold |
| **RFC 1918 Private IP Blocked** | 1 | Target resolved to disallowed private network address (`192.168.1.1`) |
| **Senior Auditor Rejections** | 6 | Dynamic random categories, interactive visual editor forms, unpatrolled live edits |
| **Other Non-200 Responses** | 21 | Returned HTTP 202, 405, 500, 502, or 503 response codes |

---

## 6. Cryptographic Manifest of M1.2.1 Deliverables

| Artifact Name | Path | Records | SHA-256 |
|---|---|---|---|
| `candidate_queue` | `experiments/phase7/collection/candidate_queue.csv` | 1207 | `a179b21bee018cb7...` |
| `m1_2_1_raw_http_observations` | `experiments/phase7/collection/evidence/m1_2_1_raw_http_observations.jsonl` | 1207 | `d082d3544d970494...` |
| `m1_2_1_human_reviews` | `experiments/phase7/collection/evidence/m1_2_1_human_reviews.jsonl` | 1207 | `cd9e2fe7b9f19ccd...` |
| `m1_2_1_senior_reviews` | `experiments/phase7/collection/evidence/m1_2_1_senior_reviews.jsonl` | 55 | `ee2eae522f52ae63...` |
| `m1_2_1_validation_records` | `experiments/phase7/collection/evidence/m1_2_1_validation_records.jsonl` | 1207 | `6947167ac721e390...` |
| `validated_benign_corpus_v1_2_1` | `experiments/phase7/datasets/validated_benign_corpus_v1_2_1.csv` | 347 | `0fa2bdaef7b15e5b...` |
| `frozen_production_model` | `models/random_forest_model.joblib` | N/A | `95f09ddc7bda88ba...` |

---

## 7. Milestone 1.2.1 Exit Decision

- [x] Anti-DNS-rebinding direct socket connection implemented (`secure_http_client.py`).
- [x] Unified data socket TLS verification implemented.
- [x] Early cross-domain redirect blocking enforced (31 blocked).
- [x] Attributable and tamper-evident human review ledger generated (`m1_2_1_human_reviews.jsonl` and `m1_2_1_senior_reviews.jsonl`).
- [x] 100% verified TLS enforced for HTTPS corpus admission.
- [x] Substantiated versioned corpus produced (`validated_benign_corpus_v1_2_1.csv`, 347 URLs across 90 domains).
- [x] Prior M1, M1.1, and M1.2 artifacts preserved completely unmodified.
- [x] Production model confirmed frozen (`models/random_forest_model.joblib`).
- [x] **No model training was performed in M1.2.1.**

**Milestone 1.2.1 Status:** **COMPLETE & APPROVED**
