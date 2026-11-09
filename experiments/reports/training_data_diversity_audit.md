# Training Data Structural Diversity & Covariate Shift Audit

**Execution Timestamp (UTC)**: `2026-11-08T11:56:45Z`
**Dataset Analyzed**: `data/PhiUSIIL_Phishing_URL_Dataset.csv` ($N=235,370$)
**Comparison Corpora**: Curated External Legitimate ($N=500$) and External Phishing Threat Feeds ($N=547$)

---

## 1. Executive Summary: The Syntactic Vacuum

This empirical audit investigates the exact structural characteristics of the training dataset to uncover why the production Random Forest classifier exhibits an extreme **94.00% False-Positive Rate** on external legitimate deep links while achieving **99.73% accuracy** in-distribution.

The investigation reveals a total **syntactic vacuum** in the legitimate partition of PhiUSIIL ($N=134,850$):

1. **100.0000% Scheme Invariance**: Every single legitimate URL begins with `https://www.`. There is not a single `http://` URL, bare root URL (`https://domain.com`), or departmental subdomain (`https://cs.stanford.edu`) in the entire legitimate training set.
2. **100.0000% Path Vacuum**: Exactly 0 out of 134,850 legitimate URLs contain a directory path segment, file extension, or trailing slash. Every legitimate URL has exactly two slashes (`https://`).
3. **100.0000% Delimiter Vacuum**: Exactly 0 legitimate URLs contain query parameters (`?`, `=`, `&`), fragments (`#`), or explicit port specifications.
4. **Artificially Constrained Feature Space**: In the legitimate training set, `NoOfOtherSpecialCharsInURL` has a median of `1.0` (range: 1–4), and `SpacialCharRatioInURL` has a median of `0.040`. In real-world authentic traffic, the median is `3.0` and `0.103` respectively.

Because the model was trained on an artificially restricted subset of legitimate URLs (root homepages only), the Random Forest learned decision tree split rules that associate any directory path, query string, or trailing slash directly with phishing.

---

## 2. Structural Divergence Matrix

| Structural Characteristic | PhiUSIIL Legitimate ($N=134,850$) | External Curated Legitimate ($N=500$) | PhiUSIIL Phishing ($N=100,520$) | External Phishing Feeds ($N=547$) | Covariate Shift Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`https://www.` Prefix** | **100.00%** | **65.60%** | 2.43% | 0.91% | **-34.40 pp** |
| **`https://` (All)** | 100.00% | 99.80% | 48.63% | 76.78% | -0.20 pp |
| **`http://` (Cleartext)** | **0.00%** | **0.20%** | 51.37% | 23.22% | +0.20 pp |
| **Exactly 2 Slashes (`//`)** | **100.00%** | **16.80%** | 72.86% | 67.28% | **-83.20 pp** |
| **Mean Slash Count** | **2.000** | **3.608** | 3.016 | 2.687 | **+1.608 slashes** |
| **Trailing Slash (`/`)** | **0.00%** | **30.60%** | 23.89% | 27.61% | **+30.60 pp** |
| **Non-Empty Directory Path** | **0.00%** | **83.20%** | 27.14% | 32.72% | **+83.20 pp** |
| **Query Parameters (`?`)** | **0.00%** | **19.80%** | 6.10% | 12.07% | **+19.80 pp** |
| **Equals Delimiters (`=`)** | **0.00%** | **19.40%** | 12.30% | 13.53% | **+19.40 pp** |
| **Ampersand Delimiters (`&`)** | **0.00%** | **12.60%** | 7.70% | 7.31% | **+12.60 pp** |
| **Fragment Delimiters (`#`)** | **0.00%** | **3.00%** | 0.08% | 0.37% | **+3.00 pp** |
| **Port Specification (`:port`)** | **0.00%** | **0.00%** | 0.42% | 0.00% | 0.00 pp |

---

## 3. Lexical and Feature Distribution Analysis

Descriptive statistics across the 18 model features demonstrate that authentic external deep links reside entirely in regions of feature space that the model exclusively observed in phishing samples during training:

| Feature Name | PhiUSIIL Legit Median [P10, P90] | External Legit Median [P10, P90] | PhiUSIIL Phish Median [P10, P90] | Tree Split Boundary | Forensic Implication |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`URLLength`** | 27.0 [21.0, 34.0] | **34.0 [24.0, 61.0]** | 35.0 [23.0, 71.0] | $\ge 33.5$ | Real deep links are longer than synthetic homepages |
| **`NoOfOtherSpecialCharsInURL`** | **1.0 [1.0, 2.0]** | **3.0 [2.0, 5.0]** | 3.0 [1.0, 7.0] | **$\ge 1.5$** | Slashes in paths push authentic URLs over the tree cutoff |
| **`SpacialCharRatioInURL`** | **0.040 [0.031, 0.077]** | **0.103 [0.067, 0.138]** | 0.086 [0.038, 0.136] | **$\ge 0.0785$** | Real legitimate median (0.103) exceeds 99th pct of training legit |
| **`NoOfLettersInURL`** | 14.0 [8.0, 21.0] | 20.0 [11.0, 41.0] | 21.0 [10.0, 48.0] | $\ge 17.5$ | Path text increases letter count |
| **`NoOfEqualsInURL`** | **0.0 [0.0, 0.0]** | **0.0 [0.0, 1.0]** | 0.0 [0.0, 1.0] | **$\ge 0.5$** | Any legitimate URL with a query parameter is flagged |
| **`NoOfQMarkInURL`** | **0.0 [0.0, 0.0]** | **0.0 [0.0, 1.0]** | 0.0 [0.0, 0.0] | **$\ge 0.5$** | Any legitimate URL with `?` is flagged |
| **`NoOfAmpersandInURL`** | **0.0 [0.0, 0.0]** | **0.0 [0.0, 1.0]** | 0.0 [0.0, 0.0] | **$\ge 0.5$** | Multi-parameter queries are flagged |

---

## 4. Path Depth Distribution Breakdown

The distribution of path segments highlights the architectural flaw of using homepage-only legitimate data:

```
PhiUSIIL Legitimate (N=134,850):
  0 segments (root homepage):  [========================================] 100.00%
  1 segment  (/dir):           [                                        ]   0.00%
  2 segments (/dir/sub):       [                                        ]   0.00%
  3+ segments:                 [                                        ]   0.00%

External Curated Legitimate (N=500):
  0 segments (root homepage):  [======                                  ]  16.80%
  1 segment  (/dir):           [===================                     ]  47.00%
  2 segments (/dir/sub):       [============                            ]  31.00%
  3+ segments:                 [==                                      ]   5.20%
```

---

## 5. Specification for Issue #5 Data Augmentation ($D_{\text{aug}}$)

To resolve the 94% false-positive rate without sacrificing the **99.82% phishing recall**, future training pipelines must incorporate a structurally diverse legitimate augmentation corpus that bridges this covariate gap.

### 5.1 Dataset Scale and Provenance
- **Recommended Size**: `10,000 to 25,000 authentic deep links`
- **Authoritative Provenance**:
  - Top authoritative academic institutions (.edu, .ac.uk)
  - Governmental, international, and standards bodies (.gov, .org, w3c, ietf)
  - Established open source and documentation repositories (python.org, apache.org, kernel.org)
  - Tranco / Cisco Umbrella high-reputation domain subpaths

### 5.2 Required Structural Composition Targets
- **Path Depth Allocation**:
  - Root homepages (0 segments): `15% - 20%`
  - Single-level subpaths (1 segment): `35% - 40%`
  - Multi-level subpaths (2 segments): `25% - 30%`
  - Deep subpaths (3+ segments): `10% - 15%`
  - Static documents (.html, .pdf, .php): `10% - 15% (.html, .pdf, .php, .jsp)`
- **Query Parameter Prevalence**: `15% - 25% containing ?, =, & (search, paging, filters)`
- **Trailing Slash Prevalence**: `30% - 50% across root and directory subpaths`
- **Scheme Diversity**: `90% - 95% HTTPS, 5% - 10% HTTP (legacy legitimate endpoints)`
- **Subdomain Diversity**: `40% www., 35% bare root, 25% departmental/service subdomains (docs., api., dept.)`

### 5.3 Target Feature Envelope
- **`NoOfOtherSpecialCharsInURL`**: `Range 1 to 8 (median target: 3.0)`
- **`SpacialCharRatioInURL`**: `Range 0.030 to 0.160 (median target: 0.095)`
- **`NoOfEqualsInURL` / `NoOfQMarkInURL` / `NoOfAmpersandInURL`**: Supported across full legitimate ranges
- **`URLLength`**: `Range 20 to 120 (median target: 38)`
