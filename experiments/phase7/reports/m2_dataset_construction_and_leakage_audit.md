# Phase 7 Milestone 2: Domain-Separated Dataset Construction and Leakage Audit

**Audit Timestamp:** `2026-11-13T14:30:53.458576+00:00`  
**Target Phase:** Phase 7 (Generalization Expansion)  
**Milestone:** M2 (Dataset Construction and Leakage Audit)  
**Production Model Status:** FROZEN (`95f09ddc...2f39bb`)  
**Candidate Models Status:** 0 Models Trained (Training Quarantined until M3)  

---

## Executive Summary

Phase 7 Milestone 2 (M2) establishes the domain-separated research dataset partitions and executes pre-training leakage verification.
The 119 validated benign expansion URLs from Milestone 1.2.2 were maintained separate from the phishing source data until combined dataset assembly, ensuring that registered-domain grouping, stratification, and leakage audits operate over the complete research dataset rather than only the benign subset.

### Pre-Training Contract & Protocol Locks

| Protocol Invariant | Locked Value | Verification Status |
|---|---|---|
| **Split Random Seed** | `42` | **LOCKED & VERIFIED** |
| **Target Split Ratios** | `70% Train / 15% Val / 15% Internal Test` | **LOCKED & VERIFIED** |
| **Grouping Key** | `registered_domain` (public-suffix aware) | **LOCKED & VERIFIED** |
| **Feature Schema** | 18 production features (`FEATURE_NAMES`) | **LOCKED & VERIFIED (0 Nulls)** |
| **Label Mapping** | `0 = Phishing`, `1 = Legitimate` | **LOCKED & VERIFIED** |
| **Primary Decision Threshold** | `0.50` | **LOCKED & VERIFIED** |
| **Alternate Threshold Policy** | Validation partition only (never Test) | **LOCKED & VERIFIED** |
| **Benign Expansion Corpus** | `validated_benign_corpus_v1_2_2.csv` (`b072306c...`) | **LOCKED & VERIFIED** |
| **Phishing Source Dataset** | `PhiUSIIL_Phishing_URL_Dataset.csv` (`a236549c...`) | **LOCKED & VERIFIED** |
| **Production Model** | `models/random_forest_model.joblib` (`95f09ddc...`) | **FROZEN & VERIFIED** |

---

## Domain Separation & Leakage Audit

Pairwise registered-domain intersection was audited across all partitions:

$$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset, \quad \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

| Partition Pair | Overlapping Registered Domains | Requirement | Status |
|---|---|---|---|
| **Train $\cap$ Validation** | **0** | 0 (Strict Zero Leakage) | **PASS** |
| **Train $\cap$ Internal Test** | **0** | 0 (Strict Zero Leakage) | **PASS** |
| **Validation $\cap$ Internal Test** | **0** | 0 (Strict Zero Leakage) | **PASS** |
| **Partitions $\cap$ Phase 4 Benchmark** | **0** | 0 (Quarantined External Benchmark) | **PASS** |

---

## Partition Composition & Benign Expansion Accounting

| Partition | Total URLs | Unique Domains | Phishing URLs (0) | Legitimate URLs (1) | Benign Expansion URLs | Partition SHA-256 |
|---|---|---|---|---|---|---|
| **Train** | **170615** | **121217** | 76092 | 94523 | **77** (16 domains) | `9297458d03841fbe...` |
| **Validation** | **30782** | **25974** | 10548 | 20234 | **29** (6 domains) | `132e1db3ef90056b...` |
| **Internal Test** | **34092** | **25978** | 13880 | 20212 | **13** (2 domains) | `ce004e9d8e65ea8c...` |
| **Total Research Dataset** | **235489** | **173169** | **100,520** | **134,969** | **119** (24 domains) | Complete Accounting |

### Benign Expansion Domain Allocation

- **Train Partition (77 URLs across 16 domains):** `abs.gov.au`, `archlinux.org`, `curl.se`, `eclipse.org`, `epfl.ch`, `fao.org`, `grafana.com`, `insee.fr`, `mongodb.com`, `nature.com`, `pnas.org`, `prometheus.io`, `stats.govt.nz`, `ubuntu.com`, `wikidata.org`, `wikipedia.org`
- **Validation Partition (29 URLs across 6 domains):** `bea.gov`, `elastic.co`, `hashicorp.com`, `iana.org`, `sqlite.org`, `wmo.int`
- **Internal Test Partition (13 URLs across 2 domains):** `destatis.de`, `docker.com`

---

## Priority Structural Strata Across Partitions

| Structural Stratum | Priority | Train Count (%) | Val Count (%) | Test Count (%) | Subgroup Status |
|---|---|---|---|---|---|
| **Query-bearing URLs** | **Critical** | 4506 (2.64%) | 771 (2.50%) | 948 (2.78%) | **Balanced & Tracked** |
| **Multi-level paths (depth >= 2)** | **High** | 11649 (6.83%) | 2193 (7.12%) | 2263 (6.64%) | **Balanced & Tracked** |
| **Path + Query combinations** | **Critical** | 3747 (2.20%) | 664 (2.16%) | 779 (2.28%) | **Balanced & Tracked** |
| **Trailing-slash variations** | **Medium** | 30525 (17.89%) | 1844 (5.99%) | 4166 (12.22%) | **Balanced & Tracked** |
| **Percent-encoded paths** | **Medium** | 226 (0.13%) | 55 (0.18%) | 49 (0.14%) | **Balanced & Tracked** |
| **Homepage URLs** | Baseline | 150378 (88.14%) | 26957 (87.57%) | 29539 (86.64%) | **Balanced & Tracked** |
| **Subdomain-bearing URLs** | Baseline | 59949 (35.14%) | 7240 (23.52%) | 10385 (30.46%) | **Balanced & Tracked** |

---

## Cryptographic Manifest Verifications

| Artifact | Path | SHA-256 Hash | Status |
|---|---|---|---|
| Production Model | `models/random_forest_model.joblib` | `95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb` | **FROZEN** |
| Benign Expansion Corpus | `experiments/phase7/datasets/validated_benign_corpus_v1_2_2.csv` | `b072306cbbf2a9aae19d79cc8af9ef6ddbb3417ab05aeb105b7d4f91cfbcf2d1` | **VERIFIED** |
| Phase 4 External Benchmark | `data/external/external_validation_benchmark.csv` | `88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3` | **QUARANTINED** |
| Train Partition | `experiments/phase7/datasets/splits/train.csv` | `9297458d03841fbe9c32bae7c4ad323e1b3f0b221c0c755f912350808d0c6892` | **VERIFIED** |
| Validation Partition | `experiments/phase7/datasets/splits/val.csv` | `132e1db3ef90056b55039bb90386d538f1fd3e644e90cc0546250a2c2d5de578` | **VERIFIED** |
| Internal Test Partition | `experiments/phase7/datasets/splits/internal_test.csv` | `ce004e9d8e65ea8c09175d904d4a5fee0442acc4b127af6a98f58ae0aed0e2cf` | **VERIFIED** |

---

## Exit Criteria & Readiness for Milestone 3 (M3)

Milestone 2 exit criteria are fully satisfied:
1. Split seed (42), ratios (70/15/15), feature schema (18), labels (0/1), and thresholds (0.50) are pre-registered and locked.
2. Domain-separated grouping over the combined dataset mathematically guarantees zero leakage across Train, Val, and Internal Test.
3. Zero URL collisions with the Phase 4 external benchmark.
4. Feature matrices are 100% complete with zero nulls.
5. The project is cleared to proceed to **Milestone 3 (M3)**: candidate model training (E4 Random Forest) on `train.csv` under fixed hyperparameters.
