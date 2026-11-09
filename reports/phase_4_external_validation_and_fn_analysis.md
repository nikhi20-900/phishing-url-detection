# Phase 4: External Validation & False-Negative Analysis Report

**Date:** October 31, 2026
**Branch:** `phase-4-external-validation`
**Baseline Commit:** `ee0202b`
**Evaluation Script:** `src/external_validation.py`
**Serialized Machine-Readable Artifact:** `reports/external_validation_results.json`

---

## 1. Executive Summary & Core Scientific Findings

Phase 4 investigates the out-of-dataset generalization capabilities and vulnerability surface of the Random Forest classifier trained on the PhiUSIIL phishing URL dataset ($N=235,370$). In Phase 3, evaluating against unseen domains via balanced grouped splitting (`Seed 14`) established a domain-separated benchmark of **99.73% accuracy** and **99.47% phishing recall**, confirming robust cross-domain performance *within the distribution of the PhiUSIIL dataset*.

Phase 4 confronts this model with an independent, contemporaneous external validation benchmark ($N=1,047$) comprising:
- **547 active zero-day phishing URLs** from OpenPhish and PhishTank.
- **500 authentic authoritative URLs** spanning academic, governmental, standards bodies, open-source repositories, and scientific reference domains.

### Key Empirical Findings

1. **Near-Perfect External Phishing Catch Rate (99.82% Recall):**
   The model caught **546 out of 547** active external phishing URLs ($99.82\%$ recall), with only **1 missed phishing URL** ($0.18\%$ false-negative rate). On the OpenPhish zero-day stream, recall was **100.0% (300/300)**; on the verified PhishTank feed, recall was **99.60% (246/247)**.
2. **Deterministic Reproduction of Phase 3 False Negatives (106 Missed URLs):**
   The Seed 14 domain-separated evaluation partition was reproduced deterministically, identifying the exact **106 false negatives** reported in Phase 3.
3. **The Common "Root Domain" False-Negative Archetype:**
   The sole external false negative (`https://www.gonoconlatarjetaverde.com.mx/`, $P(\text{Phish})=0.15$) and the 106 Phase 3 false negatives share an identical morphological archetype: **bare HTTPS root domains with zero query parameters, zero digits, zero obfuscated characters, and zero path routing**. Purely lexical models cannot distinguish freshly registered deceptive roots from benign corporate homepages without domain age, TLS reputation, or page content signals.
4. **Catastrophic Benign False Alarm Rate (94.00% FPR):**
   When evaluated against realistic authoritative URLs with deep subpaths and query strings (e.g., `https://www.mit.edu/research`), the model suffered a **94.00% false-positive rate** (470 false alarms out of 500 legitimate URLs), achieving only 6.00% recall on benign traffic.
5. **Root Cause Diagnosis — The Benign Covariate Shift:**
   Forensic analysis and SHAP feature attributions demonstrate that in the PhiUSIIL dataset, legitimate URLs were almost exclusively **homepage root domains** (mean URL length: $26.22$, mean domain length: $19.22$, diff: $\approx 7$ chars). Any URL with subpaths or query strings in PhiUSIIL was disproportionately phishing. Consequently, the Random Forest trees learned that path depth, slashes, and parameter separators are strong indicators of phishing, producing false alarms on benign deep links.

> [!IMPORTANT]
> **Core Scientific Conclusion:** While the model exhibits exceptional sensitivity to active deceptive campaigns (catching 99.82% of external phishing links), **high in-dataset domain-separated accuracy does not imply safe out-of-dataset generalization to realistic web traffic**. Purely lexical URL models trained on homepage-biased benign datasets require architectural expansion—specifically incorporating deep-path benign data, content inspection, and domain reputation—before deployment in production environments.

---

## 2. External Validation Benchmark: Provenance & Composition

To ensure reproducible, scientifically defensible evaluation without data contamination, all external validation data was captured as fixed offline snapshots with complete audit trails.

### 2.1 Data Sources & Collection Specifications

| Source Name | Category | Record Count | Collection Timestamp | Verification Method & Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **OpenPhish Community Feed** | Phishing | 300 | 2026-10-31T02:55:00Z | Active zero-day threat stream; live phishing attacks targeted at global banking, webmail, and crypto assets. |
| **PhishTank Verified Feed** | Phishing | 247 | 2026-10-30T21:11:30Z | Human analyst and community vote-verified; filtered for active `online == yes` and `verified == yes`. |
| **Curated Authoritative Corpus** | Legitimate | 500 | 2026-10-31T03:00:00Z | Authoritative educational (`.edu`, `.ac.uk`, `.ac.in`), government (`.gov`, `.gov.uk`, `.gov.in`), standards bodies (W3C, IETF), open-source (Python, Linux, Apache, GitHub), and science/reference (Wikipedia, arXiv). |
| **Total Benchmark** | Mixed | **1,047** | — | **547 Phishing (52.24%), 500 Legitimate (47.76%)** |

### 2.2 Schema & Integrity Validation

The benchmark file `data/external/external_validation_benchmark.csv` adheres to the following audit schema:
- `url`: Full target URL string.
- `label`: Binary ground truth (`0` = Phishing, `1` = Legitimate).
- `source`: Distinct originating data provider.
- `source_id`: Canonical record identifier (`OPENPHISH-xxxx`, `PHISHTANK-xxxx`, `LEGIT-xxxx`).
- `collection_timestamp`: ISO 8601 UTC timestamp.
- `verification_method`: Verification standard applied.
- `notes`: Target brand or institutional categorization.

**Pre-flight Integrity Audit (`validate_benchmark_integrity`):**
- Null or empty URLs: **0**
- Invalid label values: **0** (strictly $\{0, 1\}$)
- Conflicting labels for identical URLs: **0**
- RFC 3986 parser exclusions: **0** (all 1,047 URLs parsed successfully with valid hostnames)

---

## 3. Overlap Audit Against PhiUSIIL Reference Dataset

We conducted an overlap audit comparing the 1,047 external benchmark URLs against the 235,370 unique URLs in the PhiUSIIL reference dataset across three distinct grouping granularities:
1. **Exact URL Overlap:** Identical full URL string matching.
2. **FQDN Hostname Overlap:** Identical fully qualified domain name (including subdomains).
3. **Public-Suffix Registered Domain Overlap:** Identical registrable domain extracted via `extract_registered_domain`.

### 3.1 Empirical Overlap Summary

| Granularity | External Benchmark Total ($N=1,047$) | External Phishing Slice ($N=547$) | External Legitimate Slice ($N=500$) |
| :--- | :---: | :---: | :---: |
| **Exact URLs** | **0 / 1,047 (0.00%)** | **0 / 547 (0.00%)** | **0 / 500 (0.00%)** |
| **Unique Hostnames** | 40 / 593 (6.75%) | 4 / 509 (0.79%) | 36 / 84 (42.86%) |
| **Registered Domains** | 89 / 316 (28.16%) | 36 / 241 (14.94%) | 53 / 75 (70.67%) |

### 3.2 Methodological Discussion on Independence

1. **Zero Exact URL Overlap (100% URL Novelty):**
   Every URL evaluated in Phase 4 is completely novel and never appeared in the PhiUSIIL dataset.
2. **Phishing Hostname Overlap (0.79%):**
   Only 4 hostnames overlapped in the phishing slice: `e.vg`, `s.id` (URL shorteners), `storage.googleapis.com` (public cloud storage), and `www.dropbox.com` (file sharing). Because threat actors continuously abuse reputable platforms, these overlaps reflect shared hosting vectors rather than memorized malicious infrastructure.
3. **Phishing Registered Domain Overlap (14.94%):**
   The 36 overlapping registered domains are modern free-tier hosting platforms (`pages.dev`, `vercel.app`, `github.io`, `gitbook.io`, `4everland.app`). Because multiple attackers host ephemeral phishing sites on shared platforms, grouping by registrable public suffix demonstrates that phishing attacks frequently share hosting platforms without sharing campaign infrastructure.
4. **Legitimate Hostname and Domain Overlap (42.86% Hostname, 70.67% Domain):**
   Authoritative legitimate entities (e.g., `mit.edu`, `ox.ac.uk`, `python.org`, `w3.org`) are widely documented across institutional registries. The overlap here is expected because the curated corpus intentionally benchmarks established institutions to evaluate deep-path navigation.

---

## 4. External Validation Performance & Confusion Matrix

Evaluation was performed using the frozen model artifact `models/random_forest_model.joblib` without retraining, parameter updates, or threshold adjustments (default decision threshold: $0.5$).

### 4.1 Benchmark Metrics Summary

| Metric | Combined Benchmark ($N=1,047$) | Phishing Feeds Slice ($N=547$) | Legitimate Corpus Slice ($N=500$) | Phase 3 Domain Split ($N=47,037$) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **55.01%** | **99.82%** | **6.00%** | 99.73% |
| **Phishing Recall** | **99.82%** | **99.82%** | N/A | 99.47% |
| **Phishing Precision** | **53.74%** | **100.00%** | N/A | 99.90% |
| **Phishing F1-Score** | **69.87%** | **99.91%** | N/A | 99.69% |
| **Legitimate Recall** | **6.00%** | N/A | **6.00%** | 99.93% |
| **Legitimate Precision**| **96.77%** | N/A | **100.00%** | 99.61% |
| **Macro F1-Score** | **40.58%** | **99.91%** | **11.32%** | 99.73% |
| **ROC-AUC** | **0.6827** | *Undefined\** | *Undefined\** | 0.9980 |
| **PR-AUC** | **0.6381** | *Undefined\** | *Undefined\** | 0.9985 |
| **Brier Score** | **0.4220** | **0.0031** | **0.8802** | 0.0025 |
| **Log Loss** | **10.2685** | *Undefined\** | *Undefined\** | 0.0257 |
| **ECE (10 Bins)** | **0.4330** | **0.0094** | **0.9143** | 0.0010 |

*\*Note: ROC-AUC, PR-AUC, and Log Loss are mathematically undefined when only one class is present in a test slice.*

### 4.2 Confusion Matrices

#### Combined External Benchmark ($N=1,047$)
```
                     Predicted Phishing (0)    Predicted Legitimate (1)
Actual Phishing (0)          546 (TP)                     1 (FN)
Actual Legitimate (1)        470 (FP)                    30 (TN)
```

#### Individual Source Breakdown

| Source Partition | Sample Size | True Positive (TP) | False Negative (FN) | False Positive (FP) | True Negative (TN) | Phishing Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OpenPhish Community** | 300 | 300 | 0 | 0 | 0 | **100.00%** |
| **PhishTank Verified** | 247 | 246 | 1 | 0 | 0 | **99.60%** |
| **Curated Legitimate** | 500 | 0 | 0 | 470 | 30 | N/A (FPR: 94.0%) |

---

## 5. False-Negative & Error Diagnostic Analysis

### 5.1 Phase 3 False Negatives Reproduction ($N=106$)

Running the balanced group split (`Seed 14`) against PhiUSIIL reproduced the exact **106 false negatives** from Phase 3:
- **Test Phishing Population:** 20,114 URLs
- **Phishing Caught (TP):** 20,008
- **Phishing Missed (FN):** 106 ($FNR = 0.527\%$)

#### Common Feature Profile of the 106 Phase 3 False Negatives
```
Feature Means (106 Phase 3 False Negatives):
  - URLLength                  : 29.28
  - DomainLength               : 21.08
  - IsHTTPS                    : 1.0000 (100% encrypted TLS)
  - IsDomainIP                 : 0.0000 (0% IP addresses)
  - HasObfuscation             : 0.0000 (0% hex/URL encoding)
  - NoOfEqualsInURL            : 0.0000 (0 query parameters)
  - NoOfQMarkInURL             : 0.0000 (0 question marks)
  - NoOfAmpersandInURL         : 0.0000 (0 ampersands)
  - NoOfDegitsInURL            : 0.3868 (near-zero digits)
  - SpacialCharRatioInURL      : 0.0545
  - LetterRatioInURL           : 0.4979
```

**Representative Phase 3 FN Samples:**
- `https://www.omahanewsbulletin.com` ($P(\text{Phish}) = 0.0015$)
- `https://www.sharestion.com` ($P(\text{Phish}) = 0.0022$)
- `https://www.mistoreosqv-jp.top` ($P(\text{Phish}) = 0.0028$)
- `https://www.mygov3-au.top` ($P(\text{Phish}) = 0.2468$)
- `https://www.xn--ingenieurbro-brasnic-yec.de` ($P(\text{Phish}) = 0.3000$)

### 5.2 External False Negative Case Study ($N=1$)

Across all 547 external phishing URLs, exactly one URL bypassed the classifier:
- **URL:** `https://www.gonoconlatarjetaverde.com.mx/`
- **Source:** PhishTank (`PHISHTANK-9552109`, verified online attack)
- **Model Output:** $P(\text{Phish}) = 0.1500$, $P(\text{Legit}) = 0.8500 \implies$ Predicted **Legitimate**
- **Decision Delta:** $-0.3500$ below default classification threshold ($0.5$).

#### Feature Vector & Local SHAP Attribution

| Feature Name | Value | SHAP Impact (Toward Phishing) | Cybersecurity Interpretation |
| :--- | :---: | :---: | :--- |
| `URLLength` | 41.0 | $-0.1342$ | Below average length for malicious links in training data. |
| `IsHTTPS` | 1.0 | $-0.1287$ | Presence of TLS transport pushes prediction toward legitimate. |
| `NoOfEqualsInURL` | 0.0 | $-0.0984$ | Absence of query key-value parameters. |
| `NoOfQMarkInURL` | 0.0 | $-0.0751$ | Absence of dynamic query delimiter. |
| `DomainLength` | 32.0 | $+0.1620$ | Elevated domain character length moderately increases risk. |
| **Base Value** | — | **0.4270** | Model prior expectation across training distribution. |
| **Sum of Attributions** | — | **$-0.2770$** | **Net model output: $0.4270 - 0.2770 = 0.1500$** |

### 5.3 The Root Domain Vulnerability Archetype

The empirical convergence between the Phase 3 false negatives and the external false negative reveals a systemic vulnerability in lexical URL detection:
- **Structural Anonymity:** When an attacker hosts a phishing lure directly at the domain root over HTTPS (e.g., `https://www.gonoconlatarjetaverde.com.mx/` or `https://www.mistoreosqv-jp.top/`), the URL string exhibits no structural anomalies: no query strings, no IP addresses, no hex encoding, and standard letter ratios.
- **The Limits of Lexical Inspection:** To a static lexical parser, this URL is indistinguishable from a newly launched legitimate business website. Resolving this blind spot requires **external contextual signals**: WHOIS domain registration date (<30 days old), certificate issuer telemetry, DNS fast-flux history, or live HTML DOM analysis.

---

## 6. False-Positive Analysis: The Benign Covariate Shift

While the model demonstrated near-flawless phishing detection, it classified **470 out of 500 legitimate URLs** as phishing (94.00% False Positive Rate).

### 6.1 Feature Distribution Comparison: True Negatives vs. False Positives

| Feature Name | Verified True Negatives ($N=30$) | Falsely Flagged Benign ($N=470$) | PhiUSIIL Benign Baseline ($N=134,850$) |
| :--- | :---: | :---: | :---: |
| `URLLength` | **22.80** | **40.16** | **26.23** |
| `DomainLength` | **12.53** | **14.50** | **19.23** |
| `Path / Parameter Delta` | **$\approx 3$ chars** | **$\approx 23$ chars** | **$\approx 7$ chars** |
| `NoOfLettersInURL` | **8.20** | **25.13** | **19.14** |
| `LetterRatioInURL` | **0.3485** | **0.5969** | **0.7135** |
| `NoOfEqualsInURL` | **0.0000** | **0.4213** | **0.0001** |
| `NoOfQMarkInURL` | **0.0000** | **0.2106** | **0.0000** |
| `NoOfOtherSpecialCharsInURL` | **2.57** | **3.44** | **1.85** |
| `SpacialCharRatioInURL` | **0.1119** | **0.1031** | **0.0484** |

### 6.2 SHAP Deep Dive: Root Domain vs. Deep Subpath

Comparing an authoritative root domain (`https://www.mit.edu/`) with its subpath (`https://www.mit.edu/research`):

```
Case A: https://www.mit.edu/ (Correctly Verified Legitimate, P(Phish) = 0.0400)
  Base Value: 0.4270
  Top Attributions Pushing Toward Legitimate:
    - NoOfLettersInURL = 6         (SHAP = -0.1472)
    - IsHTTPS = 1                  (SHAP = -0.1448)
    - LetterRatioInURL = 0.3000    (SHAP = -0.1348)
    - URLLength = 20               (SHAP = -0.0770)
  Net Probability: 0.0400 (Confidence: 96.0% Legitimate)

Case B: https://www.mit.edu/research (Falsely Flagged as Phishing, P(Phish) = 0.9600)
  Base Value: 0.4270
  Top Attributions Pushing Toward Phishing:
    - DomainLength = 11            (SHAP = +0.3584)
    - SpacialCharRatioInURL = 0.071 (SHAP = +0.1141)
    - NoOfOtherSpecialChars = 2    (SHAP = +0.0948)
    - NoOfSubDomain = 1            (SHAP = +0.0652)
  Net Probability: 0.9600 (Confidence: 96.0% Phishing False Alarm)
```

### 6.3 Diagnosis of the Training Set Flaw

In the PhiUSIIL dataset:
- Benign URLs have a mean URL length of $26.23$ and a mean domain length of $19.23$.
- The query parameter features (`NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`) are virtually zero across the entire legitimate class ($mean \le 0.0001$).
- In contrast, phishing URLs in PhiUSIIL have deep directory trees, tokens, and query strings ($mean = 45.72$).

**The Consequence:**
The decision trees split on path presence, slash counts, and URL length as shortcuts for phishing detection. When applied to real-world web traffic where legitimate sites naturally use paths (`/research`, `/academics`) and query parameters (`?view=standard`), the model triggers false alarms.

---

## 7. Threat Modeling, Sampling Biases, & Evaluation Limitations

To adhere to rigorous scientific standards, we document the key limitations and boundary conditions of this evaluation:

1. **Homepage / Root Domain Bias in Public Baselines:**
   Most public academic benchmark datasets (including PhiUSIIL) sample benign URLs from top-level domain rankings (such as Tranco or Alexa), capturing primarily bare root domains (`https://example.com/`). Real web traffic consists heavily of deep links. Benchmarking on bare roots masks the model's sensitivity to path complexity.
2. **Label Uncertainty in Community Threat Feeds:**
   Community feeds (PhishTank, OpenPhish) contain active URLs submitted during live attack campaigns. While high-confidence for positive malicious indicators, threat feeds suffer from rapid infrastructure churn: a phishing URL active at 02:00 UTC may be taken down by 04:00 UTC, returning an HTTP 404 or hosting provider landing page.
3. **Temporal Drift and Domain Age Blindness:**
   Attackers increasingly register deceptive domains that mimic legitimate brands without using complex URLs. Because lexical models do not observe domain age, DNS registration records, or TLS transparency logs, they remain blind to root-domain impersonations.
4. **Targeted Benign Corpus Construction:**
   The 500 curated legitimate URLs were intentionally selected to include deep paths and query parameters to test the hypothesis of path-complexity sensitivity. This provides rigorous stress testing, though the resulting 94.00% FPR reflects performance on structured subpaths rather than an unweighted sample of all internet requests.

---

## 8. Reproducibility & Verification Instructions

All evaluation results and diagnostic analyses are deterministically reproducible using the provided codebase:

### Run External Validation Benchmark
```bash
export MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=. python3 src/external_validation.py
```

### Run Phase 4 Unit Test Suite
```bash
PYTHONPATH=. pytest tests/test_external_validation.py -v
```

### Run Full Test Suite (155 Tests)
```bash
PYTHONPATH=. pytest
```

### Outputs Generated
- Machine-readable evaluation metrics: `reports/external_validation_results.json`
- Comprehensive Markdown audit report: `reports/phase_4_external_validation_and_fn_analysis.md`
