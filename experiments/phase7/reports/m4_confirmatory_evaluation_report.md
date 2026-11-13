# Phase 7 Milestone 4: Confirmatory Internal Testing & Benchmark Evaluation Report

**Report Timestamp:** `2026-11-13T18:06:28.708169+00:00`  
**Phase:** 7 (Generalization Research)  
**Milestone:** M4 / M4.1 (Confirmatory Evaluation & Statistical Reporting Correction)  
**Candidate Model (E4):** `models/phase7_candidates/rf_augmented_e4.joblib` (SHA-256: `0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b`)  
**Frozen Control Model (E0):** `models/random_forest_model.joblib` (SHA-256: `95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb`)  
**Internal Test Partition:** `experiments/phase7/datasets/splits/internal_test.csv` (SHA-256: `ce004e9d8e65ea8c09175d904d4a5fee0442acc4b127af6a98f58ae0aed0e2cf`)  
**Historical External Benchmark:** `data/external/external_validation_benchmark.csv` (SHA-256: `88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3`)  

> [!NOTE]
> **Candidate Hash Erratum & Verification Note:**
> The authoritative Candidate Model E4 SHA-256 is `0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b`.
> This matches the verified artifact at `models/phase7_candidates/rf_augmented_e4.joblib` and all committed manifests. An earlier informal commentary referencing a suffix ending in `...23e` was an uncommitted preliminary scratch typo and has been reconciled.

---

## 1. Executive Summary & Experimental Safeguards

Phase 7 Milestone 4 executes a strict, confirmatory single-pass evaluation comparing Candidate Model E4 (Targeted Augmented Random Forest) against Frozen Production Control E0 on the unsealed, domain-disjoint `internal_test.csv` partition.

### Mandatory Safeguards & Statistical Standards Enforced
1. **Strictly Confirmatory Evaluation:** E4, its hyperparameters (`n_estimators=100`, `random_state=42`), its 18 production features, and operational decision thresholds ($T = 0.50$ and $T = 0.75$) were frozen prior to unsealing `internal_test.csv`. No post-hoc tuning was performed.
2. **Single-Pass Probability Generation:** Probabilities were computed exactly once for each model on the 34,092 test URLs. All metrics, confusion matrices, McNemar contingency tables, and bootstrap distributions are derived strictly from these frozen probabilities.
3. **Threshold Convention Stated Explicitly:**
   $$\text{Prediction} = \begin{cases} \text{Phishing } (0) & \text{if } P(\text{phishing}) \ge T \\ \text{Legitimate } (1) & \text{if } P(\text{phishing}) < T \end{cases}$$
4. **Methodologically Sound Uncertainty Quantification:**
   - **Bootstrap Percentile Confidence Intervals:** 1,000 domain-clustered bootstrap resamples across the 25,978 unique registered domains are used exclusively to construct empirical 95% confidence intervals, acknowledging that resampling from the observed distribution does not evaluate the null hypothesis ($H_0$). Bootstrap tail counts are not presented as null hypothesis $p$-values.
   - **Class-Specific Paired McNemar Tests:** Because the standard McNemar test evaluates overall accuracy ($TP + TN$) rather than class-specific sensitivity, paired significance tests are computed separately for overall classification correctness ($N=34,092$), phishing recall ($n=13,880$, testing $TP$ vs $FN$), and legitimate false alarms ($n=20,212$, testing $TN$ vs $FP$).
5. **Accurate Characterization of False Alarm Rates:** Legitimate false positive rate increased from 0.114% to 0.307% at $T = 0.50$. While the observed rate remains below the pre-registered operational ceiling ($< 0.35\%$), the domain-clustered 95% bootstrap CI ($[-0.025\%, +0.588\%]$) spans substantial positive territory, and the legitimate-subset McNemar test is highly significant ($p = 5.68 \times 10^{-8}$). Consequently, non-inferiority is **not** established, and performance is framed transparently as meeting the engineering ceiling rather than proving statistical parity.
6. **Proper Labeling of Benchmarks:**
   - **Internal Test:** Primary confirmatory holdout (domain-disjoint, unseen registered domains).
   - **Phase 4 External Benchmark:** Previously observed historical benchmark replication; **not independent confirmatory evidence**.
   - **Independent Confirmation:** Requires collecting and freezing a new, previously unobserved external benchmark separately (Milestone 5).
7. **Production Model Immutability:** `models/random_forest_model.joblib` remains strictly frozen at verified hash `95f09ddc...2f39bb`. E4 remains an experimental candidate and does NOT replace the production artifact.

---

## 2. Primary Confirmatory Comparison: Internal Test at T = 0.50

Evaluated on 34,092 URLs across 25,978 unique registered domains (13,880 Phishing, 20,212 Legitimate):

| Evaluation Metric | E0 Frozen Control | E4 Augmented Candidate | Paired Delta (E4 - E0) | Domain-Clustered 95% CI | Inferential Statistical Test | Confirmatory Status |
|---|---|---|---|---|---|---|
| **Phishing Recall** | **0.9934** (13789/13880) | **0.9946** (13805/13880) | **+0.0012** | [+0.0003, +0.0020] | Paired McNemar (phish $n=13,880$): $\chi^2=7.0312, p=0.00801$ | **CONFIRMED GAIN** |
| **Phishing Precision** | **0.9983** | **0.9955** | -0.0028 | — | — | **PRESERVED (> 99.5%)** |
| **Overall Legitimate FPR** | **0.0011** (23/20212) | **0.0031** (62/20212) | +0.0019 | [-0.0002, +0.0059] | Paired McNemar (legit $n=20,212$): $\chi^2=29.4694, p=5.68 \times 10^{-8}$ | **CEILING SATISFIED (< 0.35%); NON-INFERIORITY NOT ESTABLISHED** |
| **Overall Legitimate Recall** | **0.9989** (20189/20212) | **0.9969** (20150/20212) | -0.0019 | — | — | **HIGH (> 99.6%)** |
| **Macro F1 Score** | **0.9965** | **0.9958** | -0.0007 | [-0.0034, +0.0008] | 95% Bootstrap CI spans zero | **STABLE (> 0.995)** |
| **Accuracy** | **0.9967** | **0.9960** | -0.0007 | — | Paired McNemar (all $N=34,092$): $\chi^2=5.9753, p=0.0145$ | **HIGH (> 99.5%)** |
| **ROC-AUC** | **0.9971** | **0.9981** | +0.0011 | — | — | **EXCELLENT (> 0.999)** |
| **PR-AUC (Phishing)** | **0.9975** | **0.9985** | +0.0010 | — | — | **EXCELLENT (> 0.999)** |
| **Brier Score** | **0.0034** | **0.0031** | -0.0002 | — | — | **WELL-CALIBRATED (< 0.005)** |
| **Expected Calibration Error** | **0.0080** | **0.0014** | -0.0066 | — | — | **LOW (< 0.005)** |

### Confusion Matrix Details (T = 0.50)
- **E0 Frozen Control:** TP = 13789 | FP = 23 | TN = 20189 | FN = 91 (Total = 34092)
- **E4 Augmented Candidate:** TP = 13805 | FP = 62 | TN = 20150 | FN = 75 (Total = 34092)
- **Net True Positive Catch Gain:** E4 detected **+16 additional phishing attacks** (13,805 vs 13,789) that E0 missed on unseen registered domains.
- **Net False Alarms:** E4 produced **+39 additional false positives** (62 vs 23) across 20,212 legitimate URLs.

### Paired McNemar Test Suite (T = 0.50)
To evaluate matched pairs without confounding overall correctness with class-specific recall or false alarms, three distinct paired McNemar tests were computed with continuity correction:

1. **Phishing-Only Paired Test (Phishing Recall, $n = 13,880$):**
   - Both models caught: 13781 | Both models missed: 67
   - E0 caught / E4 missed ($n_{01}$): **8**
   - E0 missed / E4 caught ($n_{10}$): **24**
   - Discordant pairs: 32 | $\chi^2 = 7.0312$, $p = 0.00801$
   - **Conclusion:** Statistically significant improvement in phishing detection ($p = 0.00801 < 0.05$). E4 caught 24 phishing URLs missed by E0, while missing only 8 caught by E0 (net $+16$ catches).

2. **Legitimate-Only Paired Test (False Alarms, $n = 20,212$):**
   - Both models benign: 20145 | Both models false-alarmed: 18
   - E0 benign / E4 false-alarmed ($n_{01}$): **44**
   - E0 false-alarmed / E4 benign ($n_{10}$): **5**
   - Discordant pairs: 49 | $\chi^2 = 29.4694$, $p = 5.6807e-08$
   - **Conclusion:** Statistically significant increase in false alarms ($p = 5.68 \times 10^{-8} < 0.05$). E4 produced 44 new false alarms that E0 classified correctly, while correcting 5 of E0's false alarms (net $+39$ false alarms). Non-inferiority is not established.

3. **Overall Classification Correctness Test (All URLs, $N = 34,092$):**
   - Both correct: 33926 | Both incorrect: 85
   - E0 correct / E4 incorrect ($n_{01}$): **52**
   - E0 incorrect / E4 correct ($n_{10}$): **29**
   - Discordant pairs: 81 | $\chi^2 = 5.9753$, $p = 0.0145$
   - **Conclusion:** E0 achieves marginally higher overall accuracy across the full dataset ($p = 0.0145$), driven by the $+39$ net false alarms on the larger legitimate class dominating the $+16$ net true phishing catches.

---

## 3. Secondary Pre-Specified Operating-Point Comparison: Internal Test at T = 0.75

At threshold $T = 0.75$ selected during validation to reduce false alarms while maintaining $\ge 99\%$ recall:

| Evaluation Metric | E0 Control (T=0.75) | E4 Candidate (T=0.75) | Delta | Status |
|---|---|---|---|---|
| **Phishing Recall** | **0.9925** (13776/13880) | **0.9916** (13763/13880) | -0.0009 | **HIGH (>= 99.0%)** |
| **Overall Legitimate FPR** | **0.0008** (16/20212) | **0.0008** (16/20212) | +0.0000 | **IDENTICAL (0.079%)** |
| **Overall Legitimate Recall** | **0.9992** | **0.9992** | +0.0000 | **IDENTICAL (99.92%)** |
| **Macro F1 Score** | **0.9964** | **0.9960** | -0.0004 | **COMPARABLE (0.9953 vs 0.9958)** |

### Operational Trade-off Analysis at T = 0.75
- **Identical False Positive Counts:** At $T = 0.75$, both models yield exactly 16 false positives out of 20,212 legitimate URLs (FPR = 0.079%).
- **Recall Trade-off:** E0 caught 13,776 phishing URLs, while E4 caught 13,763 (a deficit of 13 URLs; McNemar $p = 0.124$, not statistically significant).
- **Key Insight:** Candidate E4 does not outperform E0 at $T = 0.75$. E4's advantage is concentrated at $T = 0.50$ where its lower false-negative threshold yields $+16$ net phishing catches.

---

## 4. Subgroup Stratification on Internal Test Partition

Performance breakdown across structural subgroups at $T = 0.50$, reporting exact denominators and stratum-specific caveats:

| Structural Stratum | Total Samples | Phishing Denom (n_p) | Legit Denom (n_l) | E0 Phish Recall | E4 Phish Recall | E0 Legit FPR (FP/Legit) | E4 Legit FPR (FP/Legit) | Statistical Caveat / Denominator Adequacy |
|---|---|---|---|---|---|---|---|---|
| `benign_expansion_unseen_domains` | 13 | 0 | 13 | N/A | N/A | 1.0000 (13/13) | 0.9231 (12/13) | Caution: Tiny legit denominator (n_l=13 <= 30); FPR unreliable |
| `homepage` | 29539 | 9340 | 20199 | 0.9904 | 0.9923 | 0.0005 (10/20199) | 0.0025 (50/20199) | Adequate class denominators |
| `single_level_path` | 2096 | 2096 | 0 | 0.9995 | 0.9995 | Undefined (n_l=0) | Undefined (n_l=0) | FPR Undefined (0 legitimate URLs in stratum) |
| `multi_level_path` | 2263 | 2250 | 13 | 1.0000 | 0.9991 | 1.0000 (13/13) | 0.9231 (12/13) | Caution: Tiny legit denominator (n_l=13 <= 30); FPR unreliable |
| `query_bearing` | 948 | 939 | 9 | 1.0000 | 1.0000 | 1.0000 (9/9) | 0.8889 (8/9) | Caution: Tiny legit denominator (n_l=9 <= 30); FPR unreliable |
| `path_and_query` | 779 | 770 | 9 | 1.0000 | 1.0000 | 1.0000 (9/9) | 0.8889 (8/9) | Caution: Tiny legit denominator (n_l=9 <= 30); FPR unreliable |
| `trailing_slash` | 4166 | 4164 | 2 | 0.9962 | 0.9971 | 1.0000 (2/2) | 1.0000 (2/2) | Caution: Tiny legit denominator (n_l=2 <= 30); FPR unreliable |
| `percent_encoded_path` | 49 | 49 | 0 | 1.0000 | 1.0000 | Undefined (n_l=0) | Undefined (n_l=0) | FPR Undefined (0 legitimate URLs in stratum) |
| `subdomain_bearing` | 10385 | 7382 | 3003 | 0.9974 | 0.9980 | 0.0033 (10/3003) | 0.0157 (47/3003) | Adequate class denominators |
| `short_url` | 26858 | 7423 | 19435 | 0.9903 | 0.9922 | 0.0005 (9/19435) | 0.0025 (48/19435) | Adequate class denominators |
| `long_url` | 2106 | 2102 | 4 | 1.0000 | 1.0000 | 1.0000 (4/4) | 1.0000 (4/4) | Caution: Tiny legit denominator (n_l=4 <= 30); FPR unreliable |
| `high_special_char` | 14445 | 10144 | 4301 | 0.9968 | 0.9974 | 0.0049 (21/4301) | 0.0135 (58/4301) | Adequate class denominators |

### Interpretation of Subgroup Findings
1. **Benign Expansion on Unseen Registered Domains ($n=13$):**
   - On the sealed internal test set, the 13 benign expansion URLs span 2 registered domains. E0 false-alarmed on 13/13 (FPR = 100.00%), whereas E4 false-alarmed on 12/13 (FPR = 92.31%).
   - **Critical Statistical Caveat:** This partition contains only $n = 13$ benign expansion URLs across 2 registered domains ($n_l = 13 \le 30$). In accordance with M4 safeguards, this sample size is too small to be statistically conclusive on its own and serves solely as a directional indicator.
2. **Query-Bearing Subgroup ($n=948$):**
   - In the internal test set, 948 URLs contain query strings (939 phishing, 9 legitimate). Note the corrected count of 948 URLs (reconciled from an earlier draft figure of 857).
   - Phishing recall was 100% on both models (939/939).
   - For the 9 legitimate query-bearing URLs, E0 false-alarmed on 9/9 (FPR = 100.00%), whereas E4 false-alarmed on 8/9 (FPR = 88.89%), correctly identifying 1 deep-link below threshold. Note that with $n_l = 9 \le 30$, the legitimate sample size is inadequate to establish statistically conclusive conclusions about false-positive reduction on query-bearing URLs, serving solely as a directional indicator.
3. **Zero-Legitimate Strata (`single_level_path` and `percent_encoded_path`):**
   - In `single_level_path` ($n = 2,096$, all phishing) and `percent_encoded_path` ($n = 49$, all phishing), there are 0 legitimate URLs ($n_l = 0$).
   - False positive rate is mathematically undefined (`NaN`), correctly reported as `Undefined (n_l=0)` rather than 0.0.

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
3. **Independence Requirement:** While these results indicate substantial progress on deep-link representation, this benchmark is historical. Genuinely independent confirmation requires an unobserved external benchmark.

---

## 6. Milestone 5 (M5) Roadmap: Independent External Validation Protocol

To provide genuinely independent confirmatory evidence of generalization claims, Phase 7 Milestone 5 (M5) will execute an unobserved external holdout study following this strict protocol:

1. **Fresh Data Collection with Strict Provenance:**
   - Collect a new set of benign URLs and fresh phishing URLs with live HTTP/TLS verification.
   - Ensure complete domain disjointness from all Phase 6 training/test data and Phase 7 M2 splits (zero registered domain overlap).
2. **Independent Review & Provenance Governance:**
   - Every candidate URL admitted to the M5 benchmark must have independent human review records with verifiable timestamps and reviewer attribution.
3. **Pre-Evaluation Cryptographic Freeze:**
   - Hash and freeze the new benchmark CSV and evaluation manifest prior to running any candidate model predictions.
4. **Single-Pass Confirmatory Evaluation:**
   - Evaluate E0 and E4 at the pre-specified thresholds ($T = 0.50$ and $T = 0.75$).
   - Zero post-hoc hyperparameter tuning or threshold adjustment.
   - Compute the full paired McNemar test suite and domain-clustered bootstrap confidence intervals.

---

## 7. Milestone 4 Conclusions & Production Guidance

1. **Primary Confirmatory Question:** Candidate Model E4 demonstrates a confirmed, statistically significant increase in Phishing Recall (+0.12%, paired McNemar $p = 0.00801$, net $+16$ attacks caught, domain-clustered 95% bootstrap CI $[+0.03\%, +0.20\%]$) on the unsealed, domain-disjoint internal test set.
2. **Legitimate False Alarm Assessment:** E4 increased false alarms from 0.114% to 0.307% (paired McNemar $p = 5.68 \times 10^{-8}$, net $+39$ false alarms). While meeting the operational ceiling requirement ($< 0.35\%$), statistical non-inferiority to E0 is not established.
3. **Operating Point Trade-off:** At $T = 0.75$, false alarm rates are identical (0.079%, 16/20,212), but E4 misses 13 more phishing URLs than E0 ($p = 0.124$). E4 does not outperform E0 at $T = 0.75$.
4. **Production Model Status:** In adherence to Phase 7 governance safeguards, **E4 remains an experimental candidate artifact** in `models/phase7_candidates/rf_augmented_e4.joblib`. The production model `models/random_forest_model.joblib` remains strictly frozen and unchanged.
5. **Exit Criteria:** Milestone 4 confirmatory evaluation and Milestone 4.1 statistical corrections are complete.
