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
While URL deduplication prevents identical URL strings from crossing between train and test sets, standard random stratified splitting allows different URLs from the **same domain** to populate both splits:

- **Total Unique Domains in Dataset**: 220,077 domains
- **Training Set Unique Domains**: 176,935 domains
- **Test Set Unique Domains**: 45,179 domains
- **Overlapping Domains (seen in both Train & Test)**: **2,037 domains** (4.51% of all test domains)
- **Test URLs from Domains Present in Training**: **3,735 URLs** (**7.93% of the test set**)
- **Domain Label Purity in Dataset**:
  - Out of 220,077 domains, **220,023 domains (99.976%)** are mono-class (100% Phishing or 100% Legitimate).
  - Only **54 domains (0.024%)** host both phishing and legitimate URLs in the dataset.
- **Significance**: When a random split places paths from `domain-x.com/page1` into the training set and `domain-x.com/page2` into the test set, the ensemble classifier can memorize domain-level lexical traits rather than generalizing to structural anomalies on previously unseen domains.

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

## 4. Deliverable 3: Domain-Separated Evaluation Plan

### Problem Statement
Standard random holdout evaluation yields an optimistic accuracy of **99.72%** because 7.93% of test URLs belong to domains seen during training, and 99.97% of domains in the dataset are mono-class. To establish realistic generalization bounds against zero-day phishing attacks, we must evaluate the model against **previously unseen domains**.

### Evaluation Methodology: Domain-Grouped Split
We specify a domain-separated evaluation protocol:
1. **Domain Normalization**: Extract the registered domain / second-level domain (e.g. `target.com` from `https://sub.target.com/path`).
2. **Grouping Rule**: Use `sklearn.model_selection.GroupShuffleSplit` (or `GroupKFold`) grouped by `registered_domain`:
   $$\text{Train Domains} \cap \text{Test Domains} = \emptyset$$
   Zero domain overlap between training and testing.
3. **Stratification Balancing**: Target an 80/20 train/test distribution while maintaining class balance (~57% Legitimate / ~43% Phishing) across groups.

### Metrics to Track and Compare

```text
+----------------------------+-----------------------+-----------------------+--------------------+
| Metric                     | Random 80/20 Split    | Domain-Grouped Split  | Generalization Gap |
|                            | (Baseline Reported)   | (Zero Domain Overlap) | (Delta)            |
+----------------------------+-----------------------+-----------------------+--------------------+
| Accuracy                   | 99.720%               | [To be measured]      | -Delta             |
| Phishing Precision (0)     | 99.885%               | [To be measured]      | -Delta             |
| Phishing Recall (0)        | 99.458%               | [To be measured]      | -Delta             |
| Phishing F1-Score (0)      | 99.671%               | [To be measured]      | -Delta             |
| ROC-AUC                    | 0.998203              | [To be measured]      | -Delta             |
| PR-AUC                     | 0.998656              | [To be measured]      | -Delta             |
+----------------------------+-----------------------+-----------------------+--------------------+
```

### Out-of-Distribution (OOD) Stress Testing Scenarios
1. **Shared Infrastructure / Cloud Worker URLs**: Evaluate URLs hosted on platforms where legitimate and phishing URLs coexist (e.g., `firebaseapp.com`, `ipfs.io`, `glitch.me`, `amazonaws.com`).
2. **Novel TLD Profiles**: Test on new top-level domains unseen or rarely seen in training (e.g., `.top`, `.xyz`, `.club`, `.icu`).
3. **IDNA / Punycode Attacks**: Test on internationalized homograph domains.
4. **Direct IP Literal Attacks**: Compare detection rates on IPv4/IPv6 direct destinations vs hostname destinations.

---

## 5. Phase 3 Action Plan & Execution Roadmap

- [x] **Task 3.1 — Establish Baseline**: Record exact test set metrics, confusion matrix, loss, and ECE without altering model weights.
- [x] **Task 3.2 — Audit Data Leakage**: Audit duplicates, label conflicts, preprocessing scope, feature extraction semantics, and train/test domain overlap.
- [ ] **Task 3.3 — Implement Domain-Grouped Split Evaluation**: Build `src/evaluate_domain_split.py` to run reproducible `GroupShuffleSplit` experiments.
- [ ] **Task 3.4 — Measure Generalization Gap**: Quantify performance drops on unseen domains and document failure modes.
- [ ] **Task 3.5 — Out-of-Distribution Stress Test**: Evaluate resilience against shared-worker infrastructure, novel TLDs, and adversarial homoglyphs.
