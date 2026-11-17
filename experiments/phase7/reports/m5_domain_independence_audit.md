# Phase 7 Milestone 5: Domain-Independence Audit Report

**Audit Execution Timestamp:** `2026-11-17T18:59:33.034301+00:00`  
**Audit Protocol:** Dynamic Master Historical Exclusion Universe  
**Total Historical Files Audited:** 11 files  
**Master Exclusion Domain Universe ($|D_{\text{historical}}|$):** 173421 unique registered domains  
**Audit Status:** PASSED (Zero Domain Overlap Certified)  

---

## 1. Dynamic Historical File Inventory & Registered Domain Census

The master exclusion universe was constructed dynamically by ingesting all historical source files across baseline, Phase 4, Phase 6, Phase 7, and pilot datasets:

| Phase | Dataset Description | File Path | SHA-256 Checksum | Rows | Unique Domains |
|---|---|---|---|---|---|
| **Baseline** | Original PhiUSIIL Training Corpus | `data/PhiUSIIL_Phishing_URL_Dataset.csv` | `a236549cd369cd80...` | 235795 | 173145 |
| **Phase 4** | Phase 4 Historical External Benchmark | `data/external/external_validation_benchmark.csv` | `88525183abd402e3...` | 1047 | 316 |
| **Phase 6** | Phase 6 Domain Train Split | `experiments/phase6/datasets/splits/train.csv` | `e7afb5d270e408ef...` | 170852 | 121225 |
| **Phase 6** | Phase 6 Domain Validation Split | `experiments/phase6/datasets/splits/val.csv` | `c480458508941b62...` | 30977 | 25976 |
| **Phase 6** | Phase 6 Domain Internal Test Split | `experiments/phase6/datasets/splits/internal_test.csv` | `f11cee449b91ebc7...` | 34143 | 25980 |
| **Phase 6** | Phase 6 Historical Benign Corpus | `experiments/phase6/datasets/validated_benign_corpus.csv` | `cca9ef3f8b265c4c...` | 602 | 98 |
| **Phase 7** | Phase 7 Domain Train Split | `experiments/phase7/datasets/splits/train.csv` | `9297458d03841fbe...` | 170615 | 121217 |
| **Phase 7** | Phase 7 Domain Validation Split | `experiments/phase7/datasets/splits/val.csv` | `132e1db3ef90056b...` | 30782 | 25974 |
| **Phase 7** | Phase 7 Domain Internal Test Split | `experiments/phase7/datasets/splits/internal_test.csv` | `ce004e9d8e65ea8c...` | 34092 | 25978 |
| **Phase 7** | Phase 7 M1.2.2 Benign Corpus (Programmatic Review Governance) | `experiments/phase7/datasets/validated_benign_corpus_v1_2_2.csv` | `b072306cbbf2a9aa...` | 119 | 24 |
| **Phase 5 / Generalization** | Pilot Generalization Raw Candidates | `experiments/generalization/datasets/pilot_candidates_raw.csv` | `1797c62100464ba3...` | 605 | 101 |

**Total Master Exclusion Universe:** **173421 unique registered domains**.

---

## 2. Prospective Benchmark Candidate Ingestion & Collision Audit

- **Benchmark Dataset File Path:** `experiments/phase7/external/datasets/m5_independent_benchmark.csv`
- **Benchmark Dataset SHA-256:** `388e4eb58b05256030df494aa546384b69740d6b2bfe40734ebbb6ab5a1dfb7b`
- **Total Candidates Audited:** 1300 URLs
- **Certified Independent Benchmark URLs:** **1300 URLs** across **857 unique registered domains**
- **Quarantined Historical Collisions:** **0 URLs** across **0 domains**

### Collision Reasons Breakdown

| Quarantine Reason | Candidate URL Count | Action Taken |
|---|---|---|

---

## 3. Class and Structural Composition of Independent Benchmark

- **Phishing URLs ($Y = 0$):** 650 URLs (50.00%)
- **Legitimate URLs ($Y = 1$):** 650 URLs (50.00%)
- **Total Independent Sample ($N$):** 1300 URLs across 857 registered domains

### Mutually Exclusive Structural Breakdown (Legitimate Partition)

| Mutually Exclusive Stratum | Count ($n_l$) | Share of Legit (%) | Quota Met? |
|---|---|---|---|
| `multi_level_path_no_query` | 165 | 25.38% | **YES** |
| `homepage_root` | 130 | 20.00% | **YES** |
| `path_and_query` | 130 | 20.00% | **YES** |
| `single_level_path_no_query` | 100 | 15.38% | **YES** |
| `root_with_query` | 65 | 10.00% | **YES** |
| `special_syntax_variants` | 60 | 9.23% | **YES** |

---

## 4. Certification of Independence Invariants

1. **Mathematical Isolation Certified:**
   $$D_{\text{M5\_independent}} \cap D_{\text{historical}} = \emptyset$$
2. **No Soft Independence:** Candidate URLs differing only in path or query from previously observed domains were strictly excluded.
3. **Zero Synthetic Records:** All admitted URLs correspond to traceable upstream source feed records.
4. **Production Model Immutability:** Production Model E0 (`models/random_forest_model.joblib`) remains strictly frozen.
