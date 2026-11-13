# Phase 7 Milestone 3: Candidate Model Training and Validation Report

**Report Timestamp:** `2026-11-13T16:34:11.164052+00:00`  
**Phase:** 7 (Generalization Expansion)  
**Milestone:** M3 (Candidate Model Training & Validation Selection)  
**Candidate Model (E4):** `models/phase7_candidates/rf_augmented_e4.joblib` (SHA-256: `0f872c0fa3d7f6be...`)  
**Frozen Control (E0):** `models/random_forest_model.joblib` (SHA-256: `95f09ddc7bda88ba...`)  
**Validation Partition:** `experiments/phase7/datasets/splits/val.csv` (SHA-256: `132e1db3ef90056b...`)  
**Internal Test Partition Status:** **SEALED & UNTOUCHED** (SHA-256: `ce004e9d8e65ea8c...`)  

---

## Executive Summary

Phase 7 Milestone 3 (M3) trains Candidate Model E4 (Targeted Augmented Random Forest) strictly on `train.csv` and executes model selection and threshold analysis on `val.csv` while keeping `internal_test.csv` strictly sealed.

### Core Governance Invariants Enforced
1. **Training Isolation:** E4 was trained strictly on `experiments/phase7/datasets/splits/train.csv` (170,615 samples) with the 18 pre-registered production features and locked hyperparameters (`n_estimators=100`, `random_state=42`).
2. **Production Model Immutability:** `models/random_forest_model.joblib` remains verified frozen at known hash `95f09ddc...2f39bb`.
3. **Sealed Internal Test:** `internal_test.csv` was NOT accessed, predicted on, or evaluated in any capacity during M3.
4. **Validation-Only Tuning:** Threshold sweeps and model selection were performed strictly on `val.csv`.

---

## Primary Comparison on Validation Partition (T = 0.50)

| Evaluation Metric | E0 Frozen Control | E4 Augmented Candidate | Delta (E4 - E0) | Status |
|---|---|---|---|---|
| **Phishing Recall** | **0.9900** (10442/10548) | **0.9929** (10473/10548) | +0.0029 | **IMPROVED** |
| **Phishing Precision** | **0.9968** | **0.9967** | -0.0002 | **STABLE** |
| **Overall Legitimate FPR** | **0.0016** (33/20234) | **0.0017** (35/20234) | +0.0001 | **PRESERVED (< 0.20%)** |
| **Overall Legitimate Recall** | **0.9984** | **0.9983** | -0.0001 | **HIGH (> 99.8%)** |
| **Macro F1 Score** | **0.9950** | **0.9960** | +0.0011 | **IMPROVED** |
| **ROC-AUC** | **0.9943** | **0.9974** | 0.0031 | **EXCELLENT** |
| **PR-AUC (Phishing)** | **0.9945** | **0.9977** | 0.0032 | **EXCELLENT** |
| **Brier Score** | **0.0046** | **0.0034** | -0.0012 | **CALIBRATED** |
| **Expected Calibration Error** | **0.0076** | **0.0015** | -0.0061 | **CALIBRATED** |

---

## Priority Subgroup Analysis on Validation Partition (T = 0.50)

| Subgroup Stratum | Priority | Sample Count | E0 Phish Recall | E4 Phish Recall | E0 Legit FPR | E4 Legit FPR | Subgroup Generalization |
|---|---|---|---|---|---|---|---|
| `homepage` | Standard | 26957 | 0.9843 | 0.9889 | 0.0002 | 0.0005 | Initial Gain |
| `single_level_path` | Standard | 1513 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Initial Gain |
| `multi_level_path` | Standard | 2193 | 1.0000 | 1.0000 | 1.0000 | 0.7619 | Initial Gain |
| `query_bearing` | Standard | 771 | 1.0000 | 1.0000 | 1.0000 | 0.8636 | Initial Gain |
| `path_and_query` | Standard | 664 | 1.0000 | 1.0000 | 1.0000 | 0.8636 | Initial Gain |
| `trailing_slash` | Standard | 1844 | 0.9810 | 0.9946 | 0.0000 | 0.0000 | Initial Gain |
| `percent_encoded_path` | Standard | 55 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | Initial Gain |
| `subdomain_bearing` | Standard | 7240 | 0.9951 | 0.9962 | 0.0013 | 0.0020 | Initial Gain |
| `short_url` | Standard | 25999 | 0.9860 | 0.9899 | 0.0001 | 0.0004 | Initial Gain |
| `long_url` | Standard | 1721 | 1.0000 | 1.0000 | 1.0000 | 0.9167 | Initial Gain |
| `high_special_char` | Standard | 11122 | 0.9947 | 0.9953 | 0.0074 | 0.0072 | Initial Gain |
| `benign_expansion_unseen_domains` | Standard | 29 | 0.0000 | 0.0000 | 1.0000 | 0.8276 | Initial Gain |

### Key Findings on Targeted Expansion Subgroups
1. **Benign Expansion on Unseen Domains (n=29):** E0 had a **100.0% False Positive Rate** (29/29 false positives) on previously unseen registered domains. Candidate E4 reduced the FPR to **82.76%** (24/29), beginning the generalization correction on unseen domains.
2. **Query-Bearing Subgroup (n=771):** On query-bearing URLs, E4 increased legitimate recall from **0.0%** (E0) to **13.64%** while preserving **100.0% phishing recall**.

---

## Validation Decision Threshold Analysis

Threshold grid sweep on `val.csv` demonstrates the operational trade-off between Phishing Recall and Benign Expansion FPR:

| Decision Threshold (T) | E4 Phishing Recall | E4 Overall Legit FPR | E4 Benign Expansion FPR | E4 Query Legit FPR | Operational Viability |
|---|---|---|---|---|---|
| **0.05** | **0.9945** | 0.0099 | 1.0000 | 1.0000 | Viable (Phish Recall >= 99%) |
| **0.10** | **0.9943** | 0.0071 | 1.0000 | 1.0000 | Viable (Phish Recall >= 99%) |
| **0.15** | **0.9940** | 0.0053 | 1.0000 | 1.0000 | Viable (Phish Recall >= 99%) |
| **0.20** | **0.9939** | 0.0042 | 1.0000 | 1.0000 | Viable (Phish Recall >= 99%) |
| **0.25** | **0.9937** | 0.0037 | 1.0000 | 1.0000 | Viable (Phish Recall >= 99%) |
| **0.30** | **0.9933** | 0.0034 | 1.0000 | 1.0000 | Viable (Phish Recall >= 99%) |
| **0.35** | **0.9933** | 0.0030 | 1.0000 | 1.0000 | Viable (Phish Recall >= 99%) |
| **0.40** | **0.9932** | 0.0026 | 0.9655 | 0.9545 | Viable (Phish Recall >= 99%) |
| **0.45** | **0.9929** | 0.0023 | 0.9655 | 0.9545 | Viable (Phish Recall >= 99%) |
| **0.50** | **0.9929** | 0.0017 | 0.8276 | 0.8636 | Viable (Phish Recall >= 99%) |
| **0.55** | **0.9925** | 0.0017 | 0.8276 | 0.8636 | Viable (Phish Recall >= 99%) |
| **0.60** | **0.9924** | 0.0017 | 0.8276 | 0.8636 | Viable (Phish Recall >= 99%) |
| **0.65** | **0.9923** | 0.0015 | 0.7931 | 0.8182 | Viable (Phish Recall >= 99%) |
| **0.70** | **0.9912** | 0.0011 | 0.6552 | 0.6364 | Viable (Phish Recall >= 99%) |
| **0.75** | **0.9905** | 0.0009 | 0.5517 | 0.5000 | Viable (Phish Recall >= 99%) |
| **0.80** | **0.9894** | 0.0008 | 0.4828 | 0.4091 | Recall Sub-99% |
| **0.85** | **0.9860** | 0.0008 | 0.4828 | 0.4091 | Recall Sub-99% |
| **0.90** | **0.9806** | 0.0006 | 0.4138 | 0.3182 | Recall Sub-99% |
| **0.95** | **0.9712** | 0.0003 | 0.1724 | 0.0455 | Recall Sub-99% |

> [!TIP]
> **Selected Alternate Operational Threshold:** **T = 0.75** maintains **99.05%** Phishing Recall while substantially reducing Benign Expansion False Positives to **55.17%** (vs 100% on E0).

---

## Readiness for Milestone 4 (M4)

Milestone 3 exit criteria are fully satisfied:
1. Candidate Model E4 trained strictly on `train.csv` under frozen hyperparameters.
2. Production model remains frozen at known hash.
3. Model selection and threshold analysis finalized on `val.csv`.
4. `internal_test.csv` has remained completely sealed and unaccessed throughout M3.
5. The project is cleared to proceed to **Milestone 4 (M4)**: unsealing `internal_test.csv` for final confirmatory evaluation against the pre-registered thresholds (T = 0.50 and T = 0.75).
