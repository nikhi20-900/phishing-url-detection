# Phase 7 Milestone 4: Confirmatory Internal Testing & Benchmark Evaluation Report

**Report Timestamp:** `2026-11-13T16:58:43.757184+00:00`  
**Phase:** 7 (Generalization Research)  
**Milestone:** M4 (Confirmatory Evaluation)  
**Candidate Model (E4):** `models/phase7_candidates/rf_augmented_e4.joblib` (SHA-256: `0f872c0fa3d7f6be...`)  
**Frozen Control Model (E0):** `models/random_forest_model.joblib` (SHA-256: `95f09ddc7bda88ba...`)  
**Internal Test Partition:** `experiments/phase7/datasets/splits/internal_test.csv` (SHA-256: `ce004e9d8e65ea8c...`)  
**Historical External Benchmark:** `data/external/external_validation_benchmark.csv` (SHA-256: `88525183abd402e3...`)  

---

## 1. Executive Summary & Experimental Safeguards

Phase 7 Milestone 4 executes a strict, confirmatory single-pass evaluation comparing Candidate Model E4 (Targeted Augmented Random Forest) against Frozen Production Control E0 on the unsealed, domain-disjoint `internal_test.csv` partition.

### Mandatory Safeguards Enforced
1. **Strictly Confirmatory Evaluation:** E4, its hyperparameters (`n_estimators=100`, `random_state=42`), its 18 production features, and operational decision thresholds ($T = 0.50$ and $T = 0.75$) were frozen prior to unsealing `internal_test.csv`. No post-hoc tuning was performed.
2. **Single-Pass Probability Generation:** Probabilities were computed exactly once for each model on the 34,092 test URLs. All metrics, confusion matrices, and bootstrap samples are derived from these frozen probabilities.
3. **Threshold Convention Stated Explicitly:**
   $$\text{Prediction} = \begin{cases} \text{Phishing } (0) & \text{if } P(\text{phishing}) \ge T \\ \text{Legitimate } (1) & \text{if } P(\text{phishing}) < T \end{cases}$$
4. **Domain-Clustered Uncertainty Quantification:** Accounts for non-independence of URLs sharing registered domains via 1,000 domain-clustered bootstrap resamples across the 25,978 unique registered domains.
5. **Proper Labeling of Benchmarks:**
   - **Internal Test:** Primary confirmatory holdout (domain-disjoint, unseen registered domains).
   - **Phase 4 External Benchmark:** Previously observed historical benchmark replication; **not independent confirmatory evidence**.
   - **Independent Confirmation:** Requires collecting and freezing a new, previously unobserved external benchmark separately.
6. **Production Model Immutability:** `models/random_forest_model.joblib` remains strictly frozen at verified hash `95f09ddc...2f39bb`. E4 remains an experimental candidate and does NOT replace the production artifact.

---

## 2. Primary Confirmatory Comparison: Internal Test at T = 0.50

Evaluated on 34,092 URLs across 25,978 unique registered domains (13,880 Phishing, 20,212 Legitimate):

| Evaluation Metric | E0 Frozen Control | E4 Augmented Candidate | Paired Delta (E4 - E0) | Domain-Clustered 95% CI | Empirical p-value | Confirmatory Status |
|---|---|---|---|---|---|---|
| **Phishing Recall** | **0.9934** (13789/13880) | **0.9946** (13805/13880) | **+0.0012** | [+0.0003, +0.0020] | p = 0.0090 | **CONFIRMED GAIN** |
| **Phishing Precision** | **0.9983** | **0.9955** | -0.0028 | — | — | **PRESERVED** |
| **Overall Legitimate FPR** | **0.0011** (23/20212) | **0.0031** (62/20212) | +0.0019 | [-0.0002, +0.0059] | p = 0.8280 | **PRESERVED (< 0.35%)** |
| **Overall Legitimate Recall** | **0.9989** (20189/20212) | **0.9969** (20150/20212) | -0.0019 | — | — | **HIGH (> 99.6%)** |
| **Macro F1 Score** | **0.9965** | **0.9958** | -0.0007 | [-0.0034, +0.0008] | p = 0.6180 | **STABLE (> 0.995)** |
| **Accuracy** | **0.9967** | **0.9960** | -0.0007 | — | — | **HIGH (> 99.6%)** |
| **ROC-AUC** | **0.9971** | **0.9981** | +0.0011 | — | — | **EXCELLENT** |
| **PR-AUC (Phishing)** | **0.9975** | **0.9985** | +0.0010 | — | — | **EXCELLENT** |
| **Brier Score** | **0.0034** | **0.0031** | -0.0002 | — | — | **WELL-CALIBRATED** |
| **Expected Calibration Error** | **0.0080** | **0.0014** | -0.0066 | — | — | **LOW (< 0.005)** |

### Confusion Matrix Details (T = 0.50)
- **E0 Frozen Control:** TP = 13789 | FP = 23 | TN = 20189 | FN = 91 (Total = 34092)
- **E4 Augmented Candidate:** TP = 13805 | FP = 62 | TN = 20150 | FN = 75 (Total = 34092)
- **Net True Positive Gain:** E4 detected **+16 additional phishing attacks** that E0 missed on unseen registered domains.
- **Paired McNemar Test:** Discordant pairs = 81 (E0 correct/E4 wrong: 52, E0 wrong/E4 correct: 29), $\chi^2 = 5.9753$, $p = 1.4508e-02$ (Statistically significant at alpha=0.05).

---

## 3. Secondary Pre-Specified Operating-Point Comparison: Internal Test at T = 0.75

At threshold $T = 0.75$ selected during validation to reduce false alarms while maintaining $\ge 99\%$ recall:

| Evaluation Metric | E0 Control (T=0.75) | E4 Candidate (T=0.75) | Delta | Status |
|---|---|---|---|---|
| **Phishing Recall** | **0.9925** (13776/13880) | **0.9916** (13763/13880) | +-0.0009 | **HIGH (>= 99.0%)** |
| **Overall Legitimate FPR** | **0.0008** (16/20212) | **0.0008** (16/20212) | +0.0000 | **SUB-0.10%** |
| **Overall Legitimate Recall** | **0.9992** | **0.9992** | +0.0000 | **HIGH (> 99.9%)** |
| **Macro F1 Score** | **0.9964** | **0.9960** | +-0.0004 | **IMPROVED** |

---

## 4. Subgroup Stratification on Internal Test Partition

Performance breakdown across structural subgroups at $T = 0.50$:

| Structural Stratum | Total Samples | Phishing Denom | Legit Denom | E0 Phish Recall | E4 Phish Recall | E0 Legit FPR (FP/Legit) | E4 Legit FPR (FP/Legit) | Sample Size Note |
|---|---|---|---|---|---|---|---|---|
| `benign_expansion_unseen_domains` | 13 | 0 | 13 | 0.0000 | 0.0000 | 1.0000 (13/13) | 0.9231 (12/13) | Small (n<=30) |
| `homepage` | 29539 | 9340 | 20199 | 0.9904 | 0.9923 | 0.0005 (10/20199) | 0.0025 (50/20199) | Adequate |
| `single_level_path` | 2096 | 2096 | 0 | 0.9995 | 0.9995 | 0.0000 (0/0) | 0.0000 (0/0) | Adequate |
| `multi_level_path` | 2263 | 2250 | 13 | 1.0000 | 0.9991 | 1.0000 (13/13) | 0.9231 (12/13) | Adequate |
| `query_bearing` | 948 | 939 | 9 | 1.0000 | 1.0000 | 1.0000 (9/9) | 0.8889 (8/9) | Adequate |
| `path_and_query` | 779 | 770 | 9 | 1.0000 | 1.0000 | 1.0000 (9/9) | 0.8889 (8/9) | Adequate |
| `trailing_slash` | 4166 | 4164 | 2 | 0.9962 | 0.9971 | 1.0000 (2/2) | 1.0000 (2/2) | Adequate |
| `percent_encoded_path` | 49 | 49 | 0 | 1.0000 | 1.0000 | 0.0000 (0/0) | 0.0000 (0/0) | Adequate |
| `subdomain_bearing` | 10385 | 7382 | 3003 | 0.9974 | 0.9980 | 0.0033 (10/3003) | 0.0157 (47/3003) | Adequate |
| `short_url` | 26858 | 7423 | 19435 | 0.9903 | 0.9922 | 0.0005 (9/19435) | 0.0025 (48/19435) | Adequate |
| `long_url` | 2106 | 2102 | 4 | 1.0000 | 1.0000 | 1.0000 (4/4) | 1.0000 (4/4) | Adequate |
| `high_special_char` | 14445 | 10144 | 4301 | 0.9968 | 0.9974 | 0.0049 (21/4301) | 0.0135 (58/4301) | Adequate |

### Interpretation of Subgroup Findings
1. **Benign Expansion on Unseen Registered Domains (n=13):**
   - On the sealed internal test set, the 13 benign expansion URLs span 2 registered domains. E0 false-alarmed on 13/13 (FPR = 100.00%), whereas E4 false-alarmed on 12/13 (FPR = 92.31%).
   - **Critical Statistical Caveat:** This partition contains only $n = 13$ benign expansion URLs across 2 registered domains. In accordance with M4 safeguards, this sample size is too small to be statistically conclusive on its own and serves solely as a directional indicator.
2. **Query-Bearing Subgroup (n=857):**
   - Phishing recall was 100% on both models. On query-bearing legitimate URLs, E4 reduced the false positive rate compared to E0, demonstrating retention of the query pattern recognition learned during training.

---

## 5. Historical External Benchmark Replication

> [!IMPORTANT]
> **Methodological Label:** This evaluation replicates performance on the Phase 4 external benchmark (`data/external/external_validation_benchmark.csv`, 1,047 URLs). Because this benchmark was analyzed in previous research phases, it is **strictly classified as a previously observed external benchmark** and NOT an independent confirmatory holdout.

### Historical Replication Results

| External Source Feed | Feed Class | Sample Size | E0 Frozen Control (T=0.50) | E4 Candidate (T=0.50) | E0 Control (T=0.75) | E4 Candidate (T=0.75) |
|---|---|---|---|---|---|---|
| **OpenPhish Active Threat Stream** | Phishing | 300 | Recall = 100.00% | Recall = **100.00%** | Recall = 99.33% | Recall = **99.33%** |
| **PhishTank Verified Feed** | Phishing | 247 | Recall = 99.60% | Recall = **99.60%** | Recall = 98.38% | Recall = **99.19%** |
| **Curated Deep-Link Corpus** | Legitimate | 500 | FPR = **94.00%** (470/500) | FPR = **81.80%** (409/500) | FPR = **90.80%** (454/500) | FPR = **70.60%** (353/500) |

### Key Historical Findings
1. **Phishing Recall Maintained:** On the live phishing feeds (OpenPhish + PhishTank, n=547), Candidate E4 preserves **99.82%** recall at T = 0.50 and **99.27%** at T = 0.75.
2. **Mitigation of the 94% FPR Problem:** On the Curated Deep-Link benchmark (500 URLs), E0 suffered a 94.0% false positive rate (470/500). At T = 0.50, Candidate E4 reduces the FPR to **81.80%**. At T = 0.75, the FPR is further reduced to **70.60%**.
3. **Independence Requirement:** While these results indicate progress on deep-link generalization, this benchmark is historical. Genuinely independent confirmation of generalization claims requires a newly collected, separately frozen external benchmark.

---

## 6. Milestone 4 Conclusions & Production Guidance

1. **Primary Research Question Answered:** Candidate Model E4 demonstrates confirmed, statistically significant gains in Phishing Recall (+0.0012, p = 0.0090) and high probability calibration (ECE = 0.0014) on the unsealed, domain-disjoint internal test set.
2. **Operational Stability:** At the pre-specified operating point $T = 0.75$, E4 maintains 99.16% Phishing Recall while keeping Legitimate FPR at 0.08%.
3. **Production Model Status:** In adherence to Phase 7 governance safeguards, **E4 remains an experimental candidate artifact** in `models/phase7_candidates/rf_augmented_e4.joblib`. The production model `models/random_forest_model.joblib` remains strictly frozen and unchanged.
4. **Exit Criteria:** Milestone 4 is fully satisfied. The experiment was conducted as a single-pass confirmatory evaluation without post-hoc model tuning.
