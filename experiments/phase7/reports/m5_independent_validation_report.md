# Phase 7 Milestone 5: Independent External Validation Report

**Evaluation Timestamp:** `2026-11-17T18:57:24.672314+00:00`  
**Milestone Goal:** Independent Out-of-Distribution Confirmatory Evaluation of Candidate E4 vs Frozen Production E0  
**Benchmark Dataset:** `experiments/phase7/external/datasets/m5_independent_benchmark.csv`  
**Benchmark SHA-256:** `388e4eb58b05256030df494aa546384b69740d6b2bfe40734ebbb6ab5a1dfb7b`  
**Total Samples ($N$):** 1,300 URLs (650 Phishing, 650 Legitimate)  
**Unique Registered Domains:** 857 domains ($D_{\text{benchmark}} \cap D_{\text{historical}} = \emptyset$)  

---

## Executive Summary

Phase 7 Milestone 5 evaluates the empirical generalization capability of Candidate Model E4 (`rf_augmented_e4.joblib`) against the Frozen Production Baseline E0 (`random_forest_model.joblib`) on a freshly constructed, cryptographically audited external benchmark. All 1,300 URLs were collected from upstream verified feeds (PhishTank live threat consensus and 5 institutional sectors) with certified zero domain overlap against all 173,421 historical registered domains.

### Key Findings at Primary Operating Point ($T = 0.50$):

1. **Phishing Threat Detection Sensitivity:**
   - Candidate E4 achieved **99.69%** phishing recall (648/650) compared to **99.85%** for E0 (649/650).
   - Paired difference: $\Delta_{\text{Recall}} = -0.0015$ (95% Clustered CI: `[-0.0073, +0.0031]`).
   - Monte Carlo cluster permutation test ($M = 10,000$): **$p_{\text{MC}} = 1.0000$**.
   - **Conclusion:** Candidate E4 satisfies the pre-specified non-inferiority margin ($\delta_{\text{NI}} = -0.010$, lower bound $-0.0073 \ge -0.010$).

2. **Legitimate False Alarm Generalization:**
   - Candidate E4 achieved a Legitimate False Positive Rate of **85.38%** (555/650) compared to **100.00%** for E0 (650/650).
   - Paired difference: $\Delta_{\text{FPR}} = -0.1462$ (95% Clustered CI: `[-0.1687, -0.1237]`).
   - Monte Carlo cluster permutation test ($M = 10,000$): **$p_{\text{MC}} = 0.0001$**.
   - Domain-level Wilcoxon signed-rank test: **$p = 0.0000$**, rank-biserial correlation $r_{\text{rb}} = -1.0000$.

3. **Decision Matrix Evaluation at $T = 0.75$:**
   - **Determination:** **`State 2 (Empirical Boundary): EMPIRICAL DETERMINATION`**
   - Details: Measured Recall=99.23%, FPR=78.4615%. Evaluated under domain-clustered confidence bounds.

---

## 1. Primary Confirmatory Results ($T = 0.50$)

| Metric | Frozen Production E0 | Candidate E4 | Paired Difference ($\Delta$) | 95% Clustered Bootstrap CI | Monte Carlo Cluster $p$-value |
|---|---|---|---|---|---|
| **Phishing Recall** | **0.9985** (649/650) | **0.9969** (648/650) | **-0.0015** | `[-0.0073, +0.0031]` | $p = 1.0000$ |
| **Legitimate FPR** | **1.0000** (650/650) | **0.8538** (555/650) | **-0.1462** | `[-0.1687, -0.1237]` | $p = 0.0001$ |
| **Macro F1** | **0.3330** | **0.4769** | **+0.1439** | `[+0.1239, +0.1638]` | — |
| **ROC-AUC** | **0.4618** | **0.7333** | **+0.2714** | — | — |
| **PR-AUC (Phishing)** | **0.6976** | **0.8024** | **+0.1048** | — | — |
| **Brier Score** | **0.4963** | **0.3943** | **-0.1020** | — | — |
| **ECE Calibration** | **0.4956** | **0.4230** | **-0.0726** | — | — |

---

## 2. Pre-Specified Operating Point ($T = 0.75$) & Decision Matrix

| Metric | Frozen Production E0 ($T = 0.75$) | Candidate E4 ($T = 0.75$) | Paired Difference ($\Delta$) | 95% Clustered Bootstrap CI |
|---|---|---|---|---|
| **Phishing Recall** | **0.9708** (631/650) | **0.9923** (645/650) | **+0.0215** | `[+0.0089, +0.0334]` |
| **Legitimate FPR** | **0.9954** (647/650) | **0.7846** (510/650) | **-0.2108** | `[-0.2311, -0.1891]` |
| **Macro F1** | **0.3317** | **0.5334** | **+0.2017** | `[+0.1843, +0.2179]` |

**Decision Framework Evaluation:** `State 2 (Empirical Boundary)` — **EMPIRICAL DETERMINATION**  
*Measured Recall=99.23%, FPR=78.4615%. Evaluated under domain-clustered confidence bounds.*

---

## 3. Subgroup Stratification Analysis

### 3.1 Mutually Exclusive Structural Strata ($T = 0.50$)

| Stratum | $N$ | Phish ($n_p$) | Legit ($n_l$) | E0 FPR ($n_l$) | E4 FPR ($n_l$) | $\Delta_{{\text{{FPR}}}}$ | E0 Recall ($n_p$) | E4 Recall ($n_p$) | Denominator Quality |
|---|---|---|---|---|---|---|---|---|---|
| `homepage_root` | 497 | 367 | 130 | 100.00% | 100.00% | +0.00% | 99.73% | 100.00% | Adequate (n_l >= 30) |
| `single_level_path_no_query` | 210 | 110 | 100 | 100.00% | 100.00% | +0.00% | 100.00% | 99.09% | Adequate (n_l >= 30) |
| `multi_level_path_no_query` | 273 | 108 | 165 | 100.00% | 100.00% | +0.00% | 100.00% | 100.00% | Adequate (n_l >= 30) |
| `root_with_query` | 74 | 9 | 65 | 100.00% | 100.00% | +0.00% | 100.00% | 100.00% | Adequate (n_l >= 30) |
| `path_and_query` | 154 | 24 | 130 | 100.00% | 30.77% | -69.23% | 100.00% | 95.83% | Adequate (n_l >= 30) |
| `special_syntax_variants` | 92 | 32 | 60 | 100.00% | 91.67% | -8.33% | 100.00% | 100.00% | Adequate (n_l >= 30) |

### 3.2 Marginal Structural Flags Breakdown ($T = 0.50$)

| Marginal Structural Feature | $N$ | Phish ($n_p$) | Legit ($n_l$) | E0 FPR ($n_l$) | E4 FPR ($n_l$) | $\Delta_{{\text{{FPR}}}}$ |
|---|---|---|---|---|---|---|
| `query_bearing==True` | 257 | 39 | 218 | 100.00% | 56.42% | -43.58% |
| `query_bearing==False` | 1043 | 611 | 432 | 100.00% | 100.00% | +0.00% |
| `multi_level_path==True` | 476 | 142 | 334 | 100.00% | 74.55% | -25.45% |
| `multi_level_path==False` | 824 | 508 | 316 | 100.00% | 96.84% | -3.16% |
| `special_syntax==True` | 92 | 32 | 60 | 100.00% | 91.67% | -8.33% |
| `special_syntax==False` | 1208 | 618 | 590 | 100.00% | 84.75% | -15.25% |

---

## 4. Sensitivity Analyses & Epidemiological Context

### 4.1 Domain-Thinning Sensitivity Analysis (100 Iterations of 1 URL/Domain)

- **Unique Registered Domains Sampled:** 857 domains per run
- **E0 Mean Phishing Recall:** 99.85% (95% range: `[99.85%, 99.85%]`)
- **E4 Mean Phishing Recall:** 99.69% (95% range: `[99.69%, 99.69%]`)
- **Mean $\Delta_{\text{Recall}}$:** -0.0015
- **E0 Mean Legitimate FPR:** 100.00%
- **E4 Mean Legitimate FPR:** 84.84%
- **Mean $\Delta_{\text{FPR}}$:** -0.1516

### 4.2 Population-Weighted Bayesian Precision (PPV) Across Realistic Prevalence Priors

In production enterprise traffic, phishing prevalence is rare ($\pi \in [0.0001, 0.01]$). Unadjusted precision from a balanced 50/50 benchmark is a stress-test metric. Below are population-weighted Bayesian PPV values modeled across realistic priors:

| Operational Scenario | Prevalence Prior ($\pi$) | E0 Bayesian PPV | E4 Bayesian PPV | E0 NPV | E4 NPV |
|---|---|---|---|---|---|
| **Ultra-Low Threat (1 in 10,000 visits)** | 0.01% | **0.01%** | **0.01%** | 0.0000% | 99.9998% |
| **Standard Enterprise Web Filter** | 0.05% | **0.05%** | **0.06%** | 0.0000% | 99.9989% |
| **Typical Enterprise Traffic (1 in 1,000 visits)** | 0.10% | **0.10%** | **0.12%** | 0.0000% | 99.9979% |
| **Elevated Risk Corporate Network** | 0.50% | **0.50%** | **0.58%** | 0.0000% | 99.9894% |
| **Targeted Email / Messaging Stream (1%)** | 1.00% | **1.00%** | **1.17%** | 0.0000% | 99.9787% |
| **High-Risk Threat Stream (5%)** | 5.00% | **4.99%** | **5.79%** | 0.0000% | 99.8893% |
| **Aggressive Quarantine Stream (10%)** | 10.00% | **9.99%** | **11.48%** | 0.0000% | 99.7666% |
| **Balanced Research Benchmark (50%)** | 50.00% | **49.96%** | **53.87%** | 0.0000% | 97.9381% |

---

## 5. Architectural & Governance Safeguards

1. **Production Baseline Immutability:** Production Model E0 (`models/random_forest_model.joblib`) remains strictly frozen and unmodified.
2. **Candidate Status Preserved:** Candidate E4 remains an experimental artifact in `models/phase7_candidates/`. No promotion was executed.
3. **Zero Retraining or Model Tuning:** No model was retrained, fine-tuned, re-weighted, or altered during M5.
4. **Cryptographic Attribution:** All benchmark observations and review submissions carry verified HMAC-SHA256 signatures in `experiments/phase7/external/evidence/ledger/`.
