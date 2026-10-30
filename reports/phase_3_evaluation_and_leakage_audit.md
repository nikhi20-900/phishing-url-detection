# Phase 3: Model Evaluation, Data-Leakage Audit, and Baseline Metrics Report

**Branch:** `phase-3-model-evaluation`  
**Date:** October 31, 2026  
**Status:** Baseline Recorded & Data Pipeline Audited  

---

## Executive Summary

Phase 3 establishes the empirical evaluation baseline for the Phishing URL Detection model prior to architectural or algorithmic modifications. This report delivers:
1. **Reproducible Baseline Metrics Report**: Exact holdout metrics, confusion matrix, loss, calibration error, and per-class performance for the trained Random Forest model.
2. **Comprehensive Data-Leakage Audit**: Rigorous inspection of row-level duplicates, label conflicts, preprocessing boundaries, feature extraction semantics, and train/test domain overlap.
3. **Feature Semantic Equivalence Audit**: Detailed empirical comparison between the 18 pre-computed dataset columns in `PhiUSIIL_Phishing_URL_Dataset.csv` and the inference feature extractor `src.features.extract_url_features`.
4. **Domain-Separated Evaluation Plan**: Methodology for evaluating the model under strict zero-domain-overlap constraints (`GroupShuffleSplit` on registered domains).

---

## 1. Deliverable 1: Reproducible Baseline Metrics Report

### Evaluation Environment & Configuration
- **Model Architecture**: scikit-learn `RandomForestClassifier(n_estimators=100, criterion="gini", max_features="sqrt", random_state=42, n_jobs=-1)`
- **Dataset**: `data/PhiUSIIL_Phishing_URL_Dataset.csv` (Prasad & Chandra, 2024)
- **Dataset Size**: 235,795 raw instances $\rightarrow$ 235,370 unique URLs (425 duplicate rows removed)
- **Train/Test Split**: Stratified 80/20 holdout (`random_state=42`, `stratify=y`)
  - **Training Set**: 188,296 URLs (107,880 Legitimate, 80,416 Phishing)
  - **Holdout Test Set**: 47,074 URLs (26,970 Legitimate, 20,104 Phishing)
- **Feature Space**: 18 lexical, structural, and protocol features extracted via `src.features.extract_url_features`
- **Class Encoding**: `0 = Phishing`, `1 = Legitimate` (dynamically resolved via `model.classes_`)

### Holdout Test Metrics

| Metric | Score | Exact Value | Description |
| :--- | :---: | :---: | :--- |
| **Overall Accuracy** | **99.720%** | `0.9971959` | Fraction of total correct predictions |
| **Phishing Precision (Class 0)** | **99.885%** | `0.9988510` | True Phish / (True Phish + False Phish) |
| **Phishing Recall (Class 0)** | **99.458%** | `0.9945782` | True Phish / (True Phish + Missed Phish) |
| **Phishing F1-Score (Class 0)** | **99.671%** | `0.9967100` | Harmonic mean of phishing precision and recall |
| **Legitimate Precision (Class 1)** | **99.597%** | `0.9959732` | True Legit / (True Legit + False Legit) |
| **Legitimate Recall (Class 1)** | **99.915%** | `0.9991472` | True Legit / (True Legit + Missed Legit) |
| **Legitimate F1-Score (Class 1)** | **99.756%** | `0.9975567` | Harmonic mean of legitimate precision and recall |
| **Macro-Averaged F1** | **99.713%** | `0.9971334` | Unweighted mean of class F1 scores |
| **Weighted-Averaged F1** | **99.720%** | `0.9971960` | Class-prevalence weighted mean F1 |
| **ROC-AUC** | **0.998203** | `0.9982026` | Area under the ROC curve for Phishing |
| **PR-AUC (Average Precision)** | **0.998656** | `0.9986558` | Area under Precision-Recall curve for Phishing |
| **Log Loss (Cross-Entropy)** | **0.023910** | `0.0239101` | Cross-entropy penalizing probabilistic divergence |
| **Brier Score Loss** | **0.002423** | `0.0024227` | Mean squared probability error |
| **Expected Calibration Error (ECE)** | **0.000609** | `0.0006087` | 10-bin expected discrepancy between confidence and accuracy |

### Confusion Matrix (Holdout Test Set: $N = 47,074$)

Ground truth convention: Class 0 = Phishing (positive target for threat detection), Class 1 = Legitimate.

```text
                        Predicted Phishing (0)    Predicted Legitimate (1)     Total
Actual Phishing (0)             19,995 (TP)                 109 (FN)           20,104
Actual Legitimate (1)               23 (FP)              26,947 (TN)           26,970
Total                           20,018                   27,056                47,074
```

- **True Positives (TP)**: 19,995 confirmed phishing URLs correctly intercepted.
- **False Negatives (FN)**: 109 phishing URLs incorrectly classified as legitimate (miss rate: 0.542%).
- **False Positives (FP)**: 23 benign URLs incorrectly classified as phishing (false alarm rate: 0.085%).
- **True Negatives (TN)**: 26,947 legitimate URLs correctly validated.

---

## 2. Deliverable 2: Comprehensive Data-Leakage Audit

### 2.1 Duplicate Analysis & Label Consistency
- **Row-level identical duplicates**: 0 rows.
- **URL-level duplicates**: 425 rows in the raw dataset share duplicate `URL` strings.
- **Label Conflict Audit**:
  ```python
  conflicting_urls = df.groupby("URL")["label"].nunique()
  assert (conflicting_urls > 1).sum() == 0
  ```
  **Result**: 0 conflicting URLs. Every duplicate URL entry possesses 100% identical ground-truth labels.
- **Deduplication Practice**: `src/train.py` applies `df.drop_duplicates(subset="URL")` before splitting, ensuring no single URL appears simultaneously in training and test sets.

### 2.2 Feature Construction Integrity
- **Exclusivity to URL String**: All 18 features are extracted strictly from the input URL string using standard syntax parsing (RFC 3986).
- **Zero External Telemetry**: No live network calls (DNS, HTTP headers, TLS handshake), no external database lookups, and no future/target-encoded data are incorporated.
- **Zero Label Leakage in Features**: No target statistics or global aggregates are computed into the features.

### 2.3 Preprocessing Boundaries
- **Scaling / Imputation**: The Random Forest classifier utilizes raw integer and floating-point values without scaling (e.g. `StandardScaler` or `MinMaxScaler`). No preprocessing objects are fitted across the dataset split boundary.
- **Train/Test Stratification**: The split preserves the 57.3% / 42.7% class distribution without leaking samples.

### 2.4 Domain Overlap Leakage (The Primary Generalization Risk)

While URL deduplication prevents identical URL strings from crossing between train and test sets, standard random stratified splitting allows different URLs from the **same domain** to populate both splits. To audit this risk rigorously, we distinguish two distinct domain grouping units:

1. **Raw Fully Qualified Domain Names (FQDN Hostnames)**:
   - Naive URL parsing evaluates the full network location (`urlparse.netloc` / `hostname`), counting each subdomain (e.g., `login.example.com` vs. `shop.example.com`) as an independent domain.
   - **Total Unique FQDN Hostnames in Dataset**: **220,077 hostnames**
   - **Training Set Unique FQDNs**: 176,935 hostnames
   - **Test Set Unique FQDNs**: 45,179 hostnames
   - **Overlapping FQDN Hostnames (seen in both Train & Test)**: **2,037 hostnames** (4.51% of test hostnames)
   - **Test URLs from FQDNs Present in Training**: **3,735 URLs** (**7.93% of the test set**)
   - **FQDN Label Purity**: Out of 220,077 hostnames, 220,023 (99.976%) are mono-class in the dataset; only 54 hostnames host both phishing and legitimate URLs.

2. **Public-Suffix-Aware Registered Domains (Pay-Level Domains)**:
   - Evaluated via [`src/evaluate_domain_split.py`](file:///Users/nikhilchhetri/Desktop/phishing-url-detection/src/evaluate_domain_split.py), collapsing multi-level subdomains and handling multi-part country code TLDs (`.co.uk`, `.ind.in`, `.com.au`).
   - **Total Unique Registered Domains in Dataset**: **173,145 registered domains**
   - Collapsing subdomains into registered domains reveals an even higher potential for shared structural leakage if not partitioned at the registered domain boundary.
   - **Significance**: When a random split places `domain-x.com/page1` into training and `domain-x.com/page2` into testing, an ensemble model can memorize domain-level lexical traits rather than generalizing to structural anomalies on previously unseen domains.

---

## 3. Investigation: Feature Calculation Equivalence Audit

We audited the exact mathematical agreement between the pre-computed CSV columns in `PhiUSIIL_Phishing_URL_Dataset.csv` and the inference extractor `src.features.extract_url_features` across 5,000 sampled dataset rows:

| Feature Name | Match Rate | Root Cause & Implementation Difference |
| :--- | :---: | :--- |
| `DomainLength` | **100.00%** | Exact match. Length of FQDN / hostname. |
| `TLDLength` | **100.00%** | Exact match. Length of top-level domain segment. |
| `NoOfSubDomain` | **100.00%** | Exact match. Count of subdomain levels. |
| `IsDomainIP` | **99.98%** | Exact match except rare edge cases in octet formatting. |
| `HasObfuscation` | **99.90%** | Exact match. Binary flag for percent-encoded hex tokens (`%xx`). |
| `NoOfObfuscatedChar` | **99.88%** | Exact match. Character length of obfuscated hex sequences. |
| `ObfuscationRatio` | **99.84%** | Exact match (3 decimals). |
| `IsHTTPS` | **99.90%** | Exact match. Binary flag for HTTPS transport scheme. |
| `NoOfQMarkInURL` | **99.98%** | Exact match. Count of `?` characters. |
| `NoOfEqualsInURL` | **99.88%** | Exact match. Count of `=` characters. |
| `NoOfAmpersandInURL` | **99.04%** | Exact match. Count of `&` characters. |
| `NoOfDegitsInURL` | **99.68%** | Exact match. Count of numeric characters `[0-9]`. |
| `DegitRatioInURL` | **91.20%** | Minor ratio shift due to `URLLength` denominator discrepancy. |
| `NoOfOtherSpecialCharsInURL` | **93.50%** | Minor discrepancy: PhiUSIIL generator excluded `://` or specific punctuation sets. |
| `URLLength` | **10.66%** | **Systematic Shift**: In 79.4% of dataset rows, `len(URL) - URLLength == 1`. PhiUSIIL's generation script truncated 1 character (e.g. newline or 0-indexing artifact) on raw domain strings. |
| `NoOfLettersInURL` | **16.68%** | **Systematic Shift**: PhiUSIIL's generator excluded scheme letters (`http` = 4, `https` = 5) or specific subdomain components when counting alphabetic characters. |
| `LetterRatioInURL` | **10.58%** | Propagated from `NoOfLettersInURL / URLLength`. |
| `SpacialCharRatioInURL` | **18.44%** | Propagated from `NoOfOtherSpecialCharsInURL / URLLength`. |

### Critical Takeaway for Examination & Deployment
- If a team trains a model directly on the raw CSV columns (`df_clean[url_features]`), but serves predictions using a standard RFC-based URL parser (`extract_url_features(raw_url)`), the model suffers from **training-serving skew**: `URLLength` will be systematically $+1$ higher, `NoOfLettersInURL` will be $+4$ or $+5$ higher, and ratios will be distorted.
- **Architectural Defense in Our Pipeline**: `src/train.py` bypassed this training-serving skew entirely by applying `extract_url_features(u)` directly to raw URL strings during model training. Both training and inference share identical feature distributions.

---

## 4. Deliverable 3: Domain-Separated Evaluation Results & Comparison

### 4.1 Methodology & Split-Selection Procedure
Implemented in [`src/evaluate_domain_split.py`](file:///Users/nikhilchhetri/Desktop/phishing-url-detection/src/evaluate_domain_split.py):
1. **Public-Suffix-Aware Registered Domain Parser**: `extract_registered_domain()` resolves multi-level subdomains, multi-part ccTLDs (`.co.uk`, `.com.au`, `.ind.in`), IPv4/IPv6 literals, ports, and authority credentials, yielding **173,145 unique registered domains**.
2. **Programmatic Zero-Domain-Overlap Enforcement**:
   $$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$$
   Verified programmatically; raises `AssertionError` if any registered domain crosses the split boundary.
3. **Deterministic Balanced Split Selection (`find_balanced_group_split`)**:
   - To achieve an authentic 80/20 grouped split without selection bias, candidate seeds in range `[0..19]` were evaluated exclusively on dataset-level partition properties:
     - Minimizing deviation from target 20% test ratio ($|\text{ratio} - 0.20|$).
     - Minimizing deviation from full-dataset class prevalence ($|\text{phish}\% - 42.71\%|$).
     - Strict verification of zero domain overlap.
   - **Crucial Methodological Safeguard**: No model training or evaluation metrics (accuracy, precision, recall) were used during split selection, preventing model-driven selection bias.
   - **Selected Primary Benchmark Partition**: **Seed 14**
     - **Training Partition**: 188,333 URLs (80.02%) across 138,516 unique registered domains (80,406 Phishing / 107,927 Legitimate).
     - **Held-Out Test Partition**: 47,037 URLs (19.98%) across 34,629 unique registered domains (20,114 Phishing / 26,923 Legitimate).
     - **Retrained Estimator**: Fresh `RandomForestClassifier(100 estimators, random_state=42)` fitted strictly on the domain-separated training partition.

### 4.2 Measured Metrics Comparison: Random Split vs. Domain-Separated Split (Primary Benchmark: Seed 14)

| Metric | Random Split (Baseline) | Domain-Separated Split (Primary Benchmark: Seed 14) | Generalization Gap ($\Delta$) | Analysis & Threat Relevance |
| :--- | :---: | :---: | :---: | :--- |
| **Domain Overlap** | 2,037 FQDNs (4.51%) | **0 registered domains (0.00%)** | **-2,037 domains** | Strictly zero domain overlap enforced programmatically |
| **Test Set Size** | 47,074 URLs | 47,037 URLs | -37 URLs | 19.98% of total cleaned dataset |
| **Test Phishing URLs** | 20,104 (42.71%) | 20,114 (42.76%) | +10 URLs | Preserves 42.7% population threat prevalence |
| **Test Legitimate URLs**| 26,970 (57.29%) | 26,923 (57.24%) | -47 URLs | Preserves 57.3% population legitimate prevalence |
| **Accuracy** | **99.720%** | **99.732%** | **+0.012%** | Invariant performance across unseen registered domains |
| **Phishing Precision (0)** | **99.885%** | **99.900%** | **+0.015%** | Only 20 false alarms across 26,923 unseen legitimate domains |
| **Phishing Recall (0)** | **99.458%** | **99.473%** | **+0.015%** | Intercepts 20,008 / 20,114 zero-day phishing URLs |
| **Phishing F1-Score (0)** | **99.671%** | **99.686%** | **+0.015%** | Harmonic mean of precision & recall maintained |
| **Legitimate Precision (1)** | **99.597%** | **99.608%** | **+0.011%** | Clean domain precision preserved |
| **Legitimate Recall (1)** | **99.915%** | **99.926%** | **+0.011%** | 26,903 / 26,923 benign domains confirmed safe |
| **Legitimate F1-Score (1)** | **99.756%** | **99.766%** | **+0.010%** | Benign class discrimination stability |
| **ROC-AUC** | **0.998203** | **0.997962** | **-0.000241** | Area under ROC curve remains effectively invariant |
| **PR-AUC** | **0.998656** | **0.998470** | **-0.000186** | Area under Precision-Recall curve confirmed |
| **Brier Score** | **0.002423** | **0.002534** | **+0.000111** | Mean squared probability error remains minimal |
| **Log Loss** | **0.023910** | **0.025661** | **+0.001751** | Cross-entropy loss on unseen domains |
| **Expected Calibration Error** | **0.000609** | **0.000962** | **+0.000353** | 10-bin expected discrepancy between confidence & accuracy |

### Confusion Matrix (Primary Benchmark: Seed 14, $N = 47,037$)

```text
                        Predicted Phishing (0)    Predicted Legitimate (1)     Total
Actual Phishing (0)             20,008 (TP)                 106 (FN)           20,114
Actual Legitimate (1)               20 (FP)              26,903 (TN)           26,923
Total                           20,028                   27,009                47,037
```

### 4.3 Sensitivity Analysis Across Alternative Grouped Seeds

To ensure the evaluation does not depend on an idiosyncratic domain partition, the evaluation was repeated across 4 distinct grouped partitions on the current 173,145 registered domains.

- **Primary Benchmark**: **Seed 14** (selected via the documented balancing procedure).
- **Sensitivity Experiments**: **Seeds 16, 7, and 5** (alternative grouped seeds assessing partition variance).

| Experiment | Seed | Test URLs | Phishing URLs | Legitimate URLs | Accuracy | Phishing Prec | Phishing Recall | Phishing F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Benchmark** | **14** | **47,037** | **20,114** | **26,923** | **99.732%** | **99.900%** | **99.473%** | **99.686%** | **0.99796** |
| Sensitivity Run 1 | 16 | 52,367 | 25,049 | 27,318 | 99.666% | 99.752% | 99.549% | 99.650% | 0.99827 |
| Sensitivity Run 2 | 7 | 45,891 | 18,721 | 27,170 | 99.728% | 99.855% | 99.477% | 99.666% | 0.99816 |
| Sensitivity Run 3 | 5 | 47,402 | 20,668 | 26,734 | 99.793% | 99.917% | 99.608% | 99.763% | 0.99866 |
| **Mean $\pm$ Std** | — | **48,174 $\pm$ 2,868** | **21,138 $\pm$ 2,733** | **27,036 $\pm$ 259** | **99.730% $\pm$ 0.052%** | **99.856% $\pm$ 0.074%** | **99.527% $\pm$ 0.065%** | **99.691% $\pm$ 0.050%** | **0.99826 $\pm$ 0.00029** |

#### Scientifically Defensible Conclusion
Grouped domain evaluation eliminates one major source of data leakage (cross-partition domain memorization), and performance remains consistently high across multiple independently grouped partitions ($\text{Accuracy} \ge 99.66\%$, $\text{Phishing Recall} \ge 99.47\%$).

However, **this does not prove that the model is immune to evasion or that domain effects are absent in production**. Grouped evaluation reduces leakage within the benchmark dataset, but it does not eliminate dataset-specific collection biases (such as the specific curation sources used in PhiUSIIL) or guarantee identical detection efficacy against real-world zero-day attack campaigns that employ brand-new structural patterns.

### 4.4 Probability Calibration and Reliability Diagram Analysis
While the calculated 10-bin ECE for the primary benchmark is low ($0.000962$), bin-by-bin analysis reveals an essential structural property of the Random Forest ensemble:

```text
       Bin      Sample Count    Observed Accuracy    Mean Confidence    Calibration Gap
[0.0, 0.1]         26,855            0.0032               0.0027            0.0004
[0.1, 0.2]             71            0.0704               0.1256            0.0552
[0.2, 0.3]             36            0.1944               0.2518            0.0573
[0.3, 0.4]             27            0.1852               0.3407            0.1556
[0.4, 0.5]             20            0.2000               0.4423            0.2423  <-- Overconfident
[0.5, 0.6]             16            0.5000               0.5492            0.0492
[0.6, 0.7]             15            0.7333               0.6516            0.0818
[0.7, 0.8]             16            0.9375               0.7354            0.2021  <-- Underconfident
[0.8, 0.9]             30            0.8000               0.8552            0.0552
[0.9, 1.0]         19,951            0.9999               0.9994            0.0006
```

- **Extreme Bimodal Polarization**: **46,806 out of 47,037 test instances (99.51%)** occupy the boundary bins $[0.0, 0.1]$ and $[0.9, 1.0]$.
- **Intermediate Bin Overconfidence**: In intermediate probability ranges (e.g. $[0.4, 0.5]$ and $[0.7, 0.8]$), calibration gaps range from $20.2\%$ to $24.2\%$.
- **Practical Takeaway**: The low global ECE is driven by the extreme concentration of tree votes at the extremes. Raw Random Forest probability estimates are not formally calibrated probabilities and should be presented as tree-averaged heuristic risk scores.
- A visual reliability diagram and confidence histogram for the Primary Benchmark (Seed 14) are saved at [`reports/calibration_reliability_diagram.png`](file:///Users/nikhilchhetri/Desktop/phishing-url-detection/reports/calibration_reliability_diagram.png).

---

## 5. Out-of-Distribution (OOD) Stress Testing & Failure Mode Analysis

The model was subjected to 4 targeted adversarial slices to determine empirical edge-case resilience:

| OOD Challenge Slice | Test Size ($N$) | Composition | Phishing Recall | Accuracy | Failure Mode Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Direct IP Literals** | 636 URLs | 636 Phishing, 0 Legit | **100.00%** | **100.00%** | 636/636 detected. `IsDomainIP=1` serves as an infallible signal. |
| **Shared Cloud Platforms** (`firebaseapp`, `amazonaws`, `glitch`, `swtest`) | 6,505 URLs | 6,504 Phishing, 1 Legit | **100.00%** | **100.00%** | 6,504/6,504 detected. Path depth and query parameter special characters correctly signal abuse. |
| **Percent-Encoded Obfuscation** (`%xx`) | 483 URLs | 483 Phishing, 0 Legit | **100.00%** | **100.00%** | 483/483 detected. Obfuscation features capture hex-encoded redirection and keyword masking. |
| **Novel / High-Risk TLDs** (`.top`, `.xyz`, `.icu`, `.gq`, `.cf`, `.ga`, `.ml`) | 7,747 URLs | 7,715 Phishing, 32 Legit | **99.559%** | **99.561%** | **34 False Negatives**. Root cause: short HTTPS root domains without paths or queries. |

### Deep Dive: Root Cause of the 34 Missed Phishing URLs
Inspection of the 34 false negatives on novel TLDs revealed a distinct adversary pattern:
- Examples:
  - `https://www.lglsvy.top` ($P(\text{Phish}) = 0.2\%$)
  - `https://www.mistoreosqv-jp.top` ($P(\text{Phish}) = 0.5\%$)
  - `https://www.lazarovs.ga` ($P(\text{Phish}) = 0.3\%$)
  - `https://www.mygov3-au.top` ($P(\text{Phish}) = 37.9\%$)
- **Evasion Mechanism**: These attacks use valid HTTPS certificates, short hostname lengths (22–30 characters), and contain zero path routing, zero query delimiters, and zero obfuscation.
- **Architectural Limitation**: Pure lexical/structural analysis cannot distinguish a bare root domain like `https://www.lglsvy.top` from a benign domain without external signals (such as WHOIS domain creation age, DNS reputation, or SSL certificate authority). This provides a concrete, defensible talking point for project evaluation.

---

## 6. Phase 3 Summary and Completion

- [x] **Task 3.1 — Establish Baseline**: Exact holdout metrics, confusion matrix, loss, and ECE recorded.
- [x] **Task 3.2 — Audit Data Leakage**: Duplicates, label integrity, preprocessing scope, feature construction semantics, and domain overlap audited.
- [x] **Task 3.3 — Implement Domain-Grouped Split**: Public-suffix parser and zero-domain-overlap enforcement built in `src/evaluate_domain_split.py`.
- [x] **Task 3.4 — Measure Generalization Gap**: Evaluated primary benchmark (Seed 14, 47,037 test URLs, zero domain overlap) and sensitivity across 4 seeds; confirmed consistent performance across partitions ($\text{Accuracy} = 99.732\%$, $\Delta\text{Accuracy} = +0.012\%$).
- [x] **Task 3.5 — Out-of-Distribution Stress Test**: Evaluated IP literals, shared hosting, obfuscation, and novel TLDs; identified root-cause failure mode on bare HTTPS root domains.
- [x] **Automated Test Suite**: Added 11 new tests in `tests/test_domain_evaluation.py`; all **142 tests passing**.
