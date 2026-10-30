# Phishing URL Detection: Machine Learning and Explainable AI

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![CI](https://github.com/nikhi20-900/phishing-url-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/nikhi20-900/phishing-url-detection/actions)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.9.0-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.62-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-142%20Passed-success.svg)]()
[![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-99.72%25-blue.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9982-blueviolet.svg)]()

An end-to-end cybersecurity system designed to identify phishing URLs using structural, lexical, and transport signals. Built with a Random Forest classifier that outputs model-estimated probabilities and local Explainable AI (SHAP) attributions, this project provides transparent, interpretable risk assessments rather than opaque black-box verdicts.

---

## Table of Contents

1. [Project Overview and Problem Statement](#1-project-overview-and-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack and Repository Structure](#3-technology-stack-and-repository-structure)
4. [Dataset Description and Model Performance](#4-dataset-description-and-model-performance)
5. [The 18 Extracted URL Features](#5-the-18-extracted-url-features)
6. [URL Parsing and Feature Extraction Pipeline](#6-url-parsing-and-feature-extraction-pipeline)
7. [Random Forest Training and Inference Workflow](#7-random-forest-training-and-inference-workflow)
8. [Explainable AI with SHAP](#8-explainable-ai-with-shap)
9. [Installation and Environment Setup](#9-installation-and-environment-setup)
10. [Command-Line Interface (CLI)](#10-command-line-interface-cli)
11. [Streamlit Web Application](#11-streamlit-web-application)
12. [Automated Testing and Verification](#12-automated-testing-and-verification)
13. [Limitations and Responsible Use](#13-limitations-and-responsible-use)
14. [Reproducibility, Future Work, and References](#14-reproducibility-future-work-and-references)

---

## 1. Project Overview and Problem Statement

Phishing attacks remain one of the most widespread initial access vectors in cybersecurity. Adversaries frequently engineer deceptive web links designed to harvest credentials, deliver malware, or initiate fraud. Traditional defense mechanisms rely heavily on static blocklists and reputation feeds. While necessary, blocklists fail against zero-day campaigns, disposable ephemeral hostnames, domain generation algorithms (DGAs), and homograph attacks.

This project implements a machine learning system that evaluates URLs at the structural and lexical levels without requiring live HTTP requests to untrusted targets. The system extracts 18 domain, path, character distribution, and protocol features, classifies the target link using a trained Random Forest ensemble, and generates local feature attributions using SHAP (Shapley Additive Explanations).

### Primary Engineering Objectives

- **Safe Pre-Flight Analysis**: Analyze URLs purely through static lexical and structural properties, avoiding execution of malicious payloads or active network queries to attack infrastructure.
- **Explainability Over Black-Box Decisions**: Deliver quantifiable feature contributions (SHAP values) that explain why a URL was flagged as phishing or legitimate.
- **Robust URL Parsing**: Correctly handle edge cases including IPv4/IPv6 literals, custom ports, userinfo authority credentials, IDNA Punycode internationalized domains, and malformed structures.
- **Interface Parity**: Maintain identical feature extraction, probability alignment, and decision thresholds across both command-line (CLI) and web (Streamlit) interfaces.

---

## 2. System Architecture

The analysis pipeline processes arbitrary URL strings through strict syntactic validation, feature computation, ensemble classification, and Shapley attribution:

```mermaid
flowchart TD
    A["Raw URL Input"] --> B["parse_url: RFC Validation & Normalization"]
    B -->|"Syntactically Valid"| C["extract_url_features: 18 Numerical Features"]
    B -->|"Invalid Scheme / Malformed"| X["ValueError / Stderr Error Output"]
    C --> D["RandomForestClassifier: 100 Decision Trees"]
    D --> E["Dynamic Class Probability Extraction: model.classes_"]
    E --> F["Decision Rule: prob_phishing >= threshold"]
    D --> G["SHAP TreeExplainer: Probability Output Space"]
    G --> H["Class 0 Phishing Attribution Extraction"]
    H --> I["Cybersecurity Domain Interpretation Mapping"]
    F --> J["Unified Presentation Layer: CLI & Streamlit UI"]
    I --> J
```

---

## 3. Technology Stack and Repository Structure

### Technology Stack

| Layer | Component | Version / Library | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10+ (tested through 3.14) | Core application and testing runtime |
| **Machine Learning** | scikit-learn | 1.9.0 | Random Forest ensemble classifier and evaluation metrics |
| **Explainable AI** | SHAP | 0.52.0 | TreeExplainer local attribution calculation |
| **Data Processing** | pandas, numpy | Modern releases | Feature representation and tabular structures |
| **Visualization** | matplotlib | Modern release | Rendered feature attribution horizontal bar plots |
| **Web UI** | Streamlit | 1.62.0 | Interactive cybersecurity analysis dashboard |
| **Model Storage** | joblib | 1.5.0 | Compressed model serialization and metadata caching |
| **Testing** | unittest | Python Standard Library | Unit, integration, and regression test suites |

### Repository Structure

```text
phishing-url-detection/
|-- .github/
|   `-- workflows/
|       `-- ci.yml                  # GitHub Actions CI matrix (Python 3.10, 3.11, 3.12)
|-- app/
|   |-- __init__.py
|   `-- main.py                     # Streamlit web application
|-- data/
|   `-- PhiUSIIL_Phishing_URL_Dataset.csv   # Benchmark training dataset
|-- models/
|   |-- .gitkeep
|   `-- random_forest_model.joblib  # Trained model artifact with feature metadata
|-- notebook/
|   `-- 01_dataset_analysis.ipynb   # Exploratory analysis and model benchmarks
|-- reports/
|   |-- calibration_reliability_diagram.png # 10-bin calibration curve and distribution
|   |-- domain_separated_evaluation_results.json # Serialized evaluation and OOD metrics
|   `-- phase_3_evaluation_and_leakage_audit.md # Complete Phase 3 audit and evaluation report
|-- src/
|   |-- __init__.py
|   |-- cli.py                      # Command-line interface with structured error handling
|   |-- evaluate_domain_split.py    # Public-suffix domain splitting & zero-overlap evaluation
|   |-- features.py                 # RFC-compliant URL parser and 18-feature extraction
|   |-- predict.py                  # Dynamic probability mapping and SHAP local attribution
|   `-- train.py                    # Random Forest training pipeline
|-- tests/
|   |-- __init__.py
|   |-- test_cli.py                 # CLI execution and error exit tests (8 tests)
|   |-- test_domain_evaluation.py   # Registered domain extraction & zero-overlap checks (11 tests)
|   |-- test_error_handling.py      # Artifact metadata validation, failure isolation, and CLI suppression (16 tests)
|   |-- test_feature_consistency.py # Feature extraction, bounds, and input validation (21 tests)
|   |-- test_prediction.py          # Dynamic probability mapping and artifact compatibility (21 tests)
|   |-- test_regression_protection.py # Invariant protections for classes, features, and SHAP (5 tests)
|   |-- test_shap_explanations.py   # SHAP attribution, class alignment, and additive consistency (19 tests)
|   `-- test_url_parsing.py         # URL parsing and RFC validation tests (41 tests)
|-- .gitignore
|-- .python-version                 # Target Python runtime specification (3.11)
|-- constraints.txt                 # Exact pinned dependency lockfile for deterministic builds
|-- requirements.txt                # Bounded semantic requirements specification
`-- README.md
```

---

## 4. Dataset Description and Model Performance

### Dataset Overview

The model is trained on the [PhiUSIIL Phishing URL Dataset](https://www.sciencedirect.com/science/article/pii/S0167404824000266) (Prasad & Chandra, 2024), a comprehensive benchmark dataset compiled from verified benign sources and confirmed phishing feeds.

| Metric | Value | Details |
| :--- | :--- | :--- |
| **Raw Instances** | 235,795 | Total rows in dataset |
| **Cleaned Instances** | 235,370 | Unique URLs after deduplication |
| **Legitimate Instances** | 134,850 (57.3%) | Class label `1`: authentic web domains |
| **Phishing Instances** | 100,520 (42.7%) | Class label `0`: confirmed malicious destinations |
| **Train / Test Split** | 80% / 20% | Stratified holdout (188,296 train / 47,074 test) |
| **Random State** | 42 | Deterministic seed for reproducible evaluation |

### Ground Truth Class Label Mapping

The dataset encodes class labels as integers:

- **`0 = Phishing`** (Malicious / Deceptive)
- **`1 = Legitimate`** (Authentic / Safe)

Inference logic verifies this mapping dynamically via `model.classes_` rather than assuming index order.

### Evaluated Model Test Metrics (Random Stratified Split)

Metrics evaluated on the holdout test set (47,074 URLs; 20,104 Phishing / 26,970 Legitimate) using features extracted by `src.features.extract_url_features`:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Overall Accuracy** | **99.720%** | Ratio of correctly classified URLs across both classes |
| **Phishing Precision** | **99.885%** | True phish divided by all instances flagged as phishing (23 false alarms) |
| **Phishing Recall** | **99.458%** | True phish detected divided by all actual phishing URLs (109 missed phish) |
| **Phishing F1-Score** | **99.671%** | Harmonic mean of phishing precision and recall |
| **ROC-AUC** | **0.9982** | Area under the Receiver Operating Characteristic curve |
| **PR-AUC** | **0.9987** | Area under the Precision-Recall curve |
| **Log Loss** | **0.0239** | Multi-class cross-entropy loss |
| **Brier Score** | **0.0024** | Mean squared probability error |
| **Expected Calibration Error (ECE)** | **0.0006** | 10-bin expected discrepancy between confidence and empirical accuracy |

### Domain-Separated Evaluation ($\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$)

A critical threat in URL classification is **domain memorization**: if subdomains or paths from the same registered domain appear in both training and test partitions, evaluation metrics can reflect memorized domain characteristics rather than generalizable structural patterns.

#### Domain Terminology & Grouping Units
To audit domain overlap rigorously, we distinguish two grouping units:
- **Raw FQDN Hostnames (220,077 in dataset)**: Naive parsing (`urlparse.netloc`) counts subdomains separately (e.g. `mail.example.com` vs `shop.example.com`). In the standard random 80/20 split, **2,037 FQDN hostnames overlap**, representing 3,735 test URLs (7.93% of the test set).
- **Public-Suffix Registered Domains (173,145 in dataset)**: Extracted via [`src/evaluate_domain_split.py`](file:///Users/nikhilchhetri/Desktop/phishing-url-detection/src/evaluate_domain_split.py), collapsing subdomains to the pay-level domain (`example.com`) with awareness of multi-part ccTLDs (`.co.uk`, `.ind.in`, `.com.au`).

#### Primary Benchmark (Seed 14: Balanced Group Split)
To evaluate genuine zero-day generalization, `find_balanced_group_split()` deterministically searches candidate seeds `[0..19]` strictly based on dataset-level criteria (target 20% test ratio, population class balance, zero domain overlap) **without evaluating model performance metrics**, eliminating model-driven selection bias.

The primary benchmark partition (**Seed 14**) trained a fresh Random Forest strictly on 188,333 URLs (138,516 domains) and evaluated on 47,037 held-out URLs across **34,629 completely unseen domains** with zero domain overlap:

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
| **ROC-AUC** | **0.998203** | **0.997962** | **-0.000241** | Area under ROC curve remains effectively invariant |
| **PR-AUC** | **0.998656** | **0.998470** | **-0.000186** | Area under Precision-Recall curve confirmed |
| **Expected Calibration Error** | **0.000609** | **0.000962** | **+0.000353** | Minimal calibration divergence on unseen domains |

#### Multi-Split Sensitivity Analysis
To confirm results do not depend on an idiosyncratic domain partition, sensitivity evaluations were performed across 4 distinct grouped partitions on the current 173,145 registered domains:

| Experiment | Seed | Test URLs | Phishing URLs | Legitimate URLs | Accuracy | Phishing Prec | Phishing Recall | Phishing F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary Benchmark** | **14** | **47,037** | **20,114** | **26,923** | **99.732%** | **99.900%** | **99.473%** | **99.686%** | **0.99796** |
| Sensitivity Run 1 | 16 | 52,367 | 25,049 | 27,318 | 99.666% | 99.752% | 99.549% | 99.650% | 0.99827 |
| Sensitivity Run 2 | 7 | 45,891 | 18,721 | 27,170 | 99.728% | 99.855% | 99.477% | 99.666% | 0.99816 |
| Sensitivity Run 3 | 5 | 47,402 | 20,668 | 26,734 | 99.793% | 99.917% | 99.608% | 99.763% | 0.99866 |
| **Mean $\pm$ Std** | — | **48,174 $\pm$ 2,868** | **21,138 $\pm$ 2,733** | **27,036 $\pm$ 259** | **99.730% $\pm$ 0.052%** | **99.856% $\pm$ 0.074%** | **99.527% $\pm$ 0.065%** | **99.691% $\pm$ 0.050%** | **0.99826 $\pm$ 0.00029** |

**Scientifically Defensible Finding**: Grouped domain evaluation reduces one major source of data leakage (cross-partition domain memorization), and performance remains consistently high across multiple independently grouped partitions ($\text{Accuracy} \ge 99.66\%$, $\text{Phishing Recall} \ge 99.47\%$). However, this does not prove immunity to evasion or that domain effects are absent in real-world deployment; it addresses benchmark leakage, but cannot eliminate dataset-specific collection biases or guarantee identical efficacy against novel adversarial techniques.

#### Probability Calibration & Reliability Analysis
While the global ECE is very low ($0.000962$), bin-by-bin reliability analysis reveals a crucial nuance:
- **Extreme Bins**: $99.51\%$ of predictions concentrate in the extreme boundary bins ($[0.0, 0.1]$ and $[0.9, 1.0]$), where empirical accuracy matches confidence within $0.06\%$.
- **Intermediate Bins**: In ambiguous intermediate ranges (e.g., $[0.4, 0.5]$), the uncalibrated Random Forest exhibits overconfidence (mean confidence of $44.2\%$ vs. empirical accuracy of $20.0\%$, a $24.2\%$ gap).
- See [`reports/calibration_reliability_diagram.png`](file:///Users/nikhilchhetri/Desktop/phishing-url-detection/reports/calibration_reliability_diagram.png) and [`reports/phase_3_evaluation_and_leakage_audit.md`](file:///Users/nikhilchhetri/Desktop/phishing-url-detection/reports/phase_3_evaluation_and_leakage_audit.md) for full details.

---

## 5. The 18 Extracted URL Features

The feature extractor extracts 18 lexical and structural features for every input link:

| Index | Feature Name | Category | Calculation Method | Security Relevance |
| :-: | :--- | :--- | :--- | :--- |
| 1 | `URLLength` | Structural | Total character count of full URL string | Phishing URLs often use lengthy paths to obscure destination hosts |
| 2 | `DomainLength` | Structural | Character length of the hostname / FQDN | Extended domains or typo-squatted names inflate domain length |
| 3 | `IsDomainIP` | Domain Security | Binary flag: `1` if host is IPv4 or IPv6 literal, `0` otherwise | Direct IP addresses bypass domain reputation feeds and TLS SNI |
| 4 | `TLDLength` | Domain Structure | Character length of the top-level domain segment | Unconventional generic TLDs (e.g. `.online`, `.accountant`) exhibit distinct length profiles |
| 5 | `NoOfSubDomain` | Domain Structure | Count of subdomain levels separated by dots | Adversaries chain multi-level subdomains to simulate legitimate brand names |
| 6 | `HasObfuscation` | Obfuscation | Binary flag: `1` if percent-encoded hex sequences (`%xx`) are present | Hex encoding is frequently used to mask keywords or executable extensions |
| 7 | `NoOfObfuscatedChar` | Obfuscation | Total character count within valid hex-encoded sequences | Quantifies the volume of encoded content in the URL string |
| 8 | `ObfuscationRatio` | Obfuscation | Ratio of obfuscated characters to `URLLength` (3 decimals) | Relative density of encoded tokens within the full target string |
| 9 | `NoOfLettersInURL` | Lexical | Total count of alphabetic characters `[a-zA-Z]` | Core baseline for measuring lexical naturalness |
| 10 | `LetterRatioInURL` | Lexical | Ratio of alphabetic characters to `URLLength` (3 decimals) | Benign URLs typically exhibit high alphabetic ratios |
| 11 | `NoOfDegitsInURL` | Lexical | Total count of numeric digits `[0-9]` | High digit counts frequently appear in tracking hashes, IP hosts, or session IDs |
| 12 | `DegitRatioInURL` | Lexical | Ratio of numeric digits to `URLLength` (3 decimals) | High digit ratios correlate with machine-generated phishing links |
| 13 | `NoOfEqualsInURL` | Query Parameters | Total occurrences of the `=` character | Indicates active query parameters used for session forwarding or tracking |
| 14 | `NoOfQMarkInURL` | Query Parameters | Total occurrences of the `?` character | Flags query string delimiters; multiple `?` characters indicate malformed routing |
| 15 | `NoOfAmpersandInURL` | Query Parameters | Total occurrences of the `&` character | Delimits multiple parameter pairs in complex query strings |
| 16 | `NoOfOtherSpecialCharsInURL` | Punctuation | Count of special characters excluding `=`, `?`, `&`, `%` | Punctuation characters (`/`, `-`, `_`, `@`, `.`) used in path routing |
| 17 | `SpacialCharRatioInURL` | Punctuation | Ratio of all special characters (including query delimiters) to length | High special character density often indicates multi-level directory traversal |
| 18 | `IsHTTPS` | Protocol Transport | Binary flag: `1` if scheme is HTTPS, `0` if HTTP | While phishing increasingly adopts HTTPS, plain HTTP remains a strong threat signal |

---

## 6. URL Parsing and Feature Extraction Pipeline

URL parsing in `src/features.py` implements strict syntactic validation before feature computation:

### Validation and Normalization Rules

- **Protocol Enforcement**: Restricts schemes to `http` and `https`. Unsupported protocols (`ftp:`, `javascript:`, `mailto:`, `data:`) raise a structured `ValueError`. Schemeless URLs (e.g. `www.google.com`) safely default to `http://`.
- **Authority and Credential Parsing**: Detects userinfo masking attacks. For deceptive links such as `https://trusted-bank.com@attacker.com`, the authority parser separates credentials from the destination host, correctly identifying `attacker.com` as the target domain while recording `has_credentials = 1`.
- **Port Syntax and Range**: Validates explicit ports against the RFC integer range ($1 \le \text{port} \le 65535$). Non-numeric or out-of-range ports are rejected.
- **IPv4 and IPv6 Detection**: Identifies 4-octet dotted IPv4 addresses and bracketed IPv6 literals (e.g. `[::1]`, `[2001:db8::1]:8080`). Rejects malformed structures such as octets $> 255$ or non-4-octet dotted strings.
- **Domain Label Syntax**: Rejects empty labels, consecutive dots (`..`), invalid hyphens at label boundaries, and labels exceeding 63 characters (RFC 1035).
- **Internationalized Domain Names (IDNA)**: Normalizes non-ASCII Unicode domain names (e.g. `bücher.de`) to ASCII Punycode (`xn--bcher-kva.de`) via the IDNA standard.

---

## 7. Random Forest Training and Inference Workflow

### Model Hyperparameters

The classification model is a scikit-learn `RandomForestClassifier` configured for deterministic reproducibility:

```python
RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

### Feature Consistency

To prevent distribution drift and training-serving skew, `src/train.py` applies the exact same `extract_url_features()` function directly to raw URLs during training:

```python
# From src/train.py
feats = [extract_url_features(u) for u in df_clean["URL"]]
X = pd.DataFrame(feats, columns=FEATURE_NAMES)
y = df_clean["label"]
```

### Dynamic Class Probability Alignment

Because scikit-learn stores class identifiers in `model.classes_`, prediction code must never assume that index `0` corresponds to label `0`. In `src/predict.py`, `get_class_probabilities()` dynamically maps probabilities:

```python
classes = list(model.classes_)
phishing_index = classes.index(0)
legitimate_index = classes.index(1)

prob_phishing = float(probabilities[phishing_index])
prob_legitimate = float(probabilities[legitimate_index])
```

Similarly, `src/train.py` dynamically resolves `phishing_index` from `rf.classes_` for evaluation metrics and test set probability calculations.

### Model Probabilities and Probability Calibration

The classifier outputs **model-estimated probabilities** computed via scikit-learn's `predict_proba()`, which averages the class probability distributions predicted by individual decision trees (derived from the class proportions in the leaf node reached by the sample across all trees in the ensemble).

> [!NOTE]
> The training pipeline does not explicitly perform probability calibration (e.g., via Platt scaling or isotonic regression). Outputs represent tree-averaged leaf probability distributions (model-estimated probabilities), not automatically calibrated probabilities. If formal probability calibration is evaluated in the future, it should be assessed on held-out validation data using calibration curves, expected calibration error (ECE), and Brier score metrics before deploying calibrated probability models (such as `CalibratedClassifierCV`).

### Decision Threshold

The decision boundary is governed by a configurable `threshold` parameter (default: `0.5`):

```text
is_phishing = prob_phishing >= threshold
```

In high-risk operational environments where minimizing false negatives is critical, the threshold can be lowered (e.g. `threshold=0.35`) to flag borderline URLs for secondary inspection.

---

## 8. Explainable AI with SHAP

### TreeExplainer Output Space

Explainability is implemented via `shap.TreeExplainer`. For a scikit-learn `RandomForestClassifier`, each leaf node represents the proportion of training instances per class, meaning `TreeExplainer` operates in **probability space**:

$$\text{Predicted Probability} = \text{Base Value} + \sum_{i=1}^{18} \phi_i$$

where $\text{Base Value}$ is the prior probability of phishing across the training data ($\approx 42.7\%$), and $\phi_i$ represents the local additive contribution of feature $i$ for that specific URL.

### Phishing Class Orientation

To prevent ambiguity, all SHAP explanations specifically target **Class 0 (Phishing)**:

- **Positive SHAP Value ($\phi_i > 0$)**: The feature increased the phishing probability (pushed the prediction toward Phishing).
- **Negative SHAP Value ($\phi_i < 0$)**: The feature reduced the phishing probability (supported a Legitimate classification).

When a URL receives a **Legitimate** verdict, the explanations explicitly state which features reduced phishing risk (e.g., active HTTPS encryption, standard domain length, zero hex obfuscation).

---

## 9. Installation and Environment Setup

### Prerequisites & Supported Python Versions

- **Supported Runtime**: **Python 3.10, 3.11, or 3.12** (Python 3.11 is recommended for optimal stability across all C-extensions).
- **Experimental Support**: Python 3.13 and 3.14 (all tests pass, but upstream scientific packages like NumPy 2.5 and Matplotlib emit internal CPython deprecation warnings).
- **System Tools**: Git

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/nikhi20-900/phishing-url-detection.git
cd phishing-url-detection

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# Option A: Standard installation with semantic version bounds
pip install -r requirements.txt

# Option B: Deterministic installation with tested exact constraints (recommended for reproducibility)
pip install -r requirements.txt -c constraints.txt
```

### Verification and Reproducibility Quickstart

After installing dependencies, verify the environment and run all components with these standard commands:

```bash
# 1. Run the complete automated test suite (131 tests)
export MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=. pytest -v

# 2. Retrain the Random Forest model artifact (optional; pre-trained artifact included)
PYTHONPATH=. python3 src/train.py

# 3. Run threat scan via CLI
PYTHONPATH=. python3 src/cli.py "https://www.google.com"

# 4. Launch the interactive Streamlit web dashboard
streamlit run app/main.py
```

---

## 10. Command-Line Interface (CLI)

The command-line interface provides fast, scriptable threat scanning.

### Basic Usage

```bash
python3 src/cli.py "<URL>"
```

### Examples

**Analyzing a legitimate domain:**

```bash
python3 src/cli.py "https://www.google.com"
```

Output:
```text
Analyzing URL: https://www.google.com...

============================================================
VERDICT     : LEGITIMATE
PROBABILITY : 99.8% (Legitimate)
RISK SCORES : Phishing: 0.2% | Legitimate: 99.8%
============================================================

TOP 5 INFLUENCING FEATURES (SHAP):
  1. IsHTTPS = 1.0
     Impact: Supports Legitimate (-0.1651)
     Note  : HTTPS is active, establishing encrypted transport standard for genuine domains.

  2. NoOfOtherSpecialCharsInURL = 1.0
     Impact: Supports Legitimate (-0.1178)
     Note  : Minimal special characters (1) aligns with clean site architecture.

  3. DomainLength = 14.0
     Impact: Increases Phishing Risk (+0.0550)
     Note  : Domain length (14 chars) contributed toward the phishing classification based on patterns learned from the training data.

  4. SpacialCharRatioInURL = 0.045
     Impact: Supports Legitimate (-0.0534)
     Note  : Special character ratio (4.5%) within expected baseline bounds.

  5. LetterRatioInURL = 0.409
     Impact: Supports Legitimate (-0.0420)
     Note  : Alphabetic composition (40.9%) contributed toward the classification based on lexical distributions in the training data.
```

**Analyzing a raw IP-based attack URL:**

```bash
python3 src/cli.py "http://198.98.58.123/"
```

Output:
```text
Analyzing URL: http://198.98.58.123/...

============================================================
VERDICT     : PHISHING
PROBABILITY : 100.0% (Phishing)
RISK SCORES : Phishing: 100.0% | Legitimate: 0.0%
============================================================

TOP 5 INFLUENCING FEATURES (SHAP):
  1. IsHTTPS = 0.0
     Impact: Increases Phishing Risk (+0.2399)
     Note  : Unencrypted HTTP transport contributed toward the phishing classification, as absence of TLS is prevalent among deceptive links in the training data.

  2. NoOfOtherSpecialCharsInURL = 4.0
     Impact: Increases Phishing Risk (+0.1161)
     Note  : Special character count (4) contributed toward the phishing classification based on patterns learned from the training data.

  3. NoOfDegitsInURL = 10.0
     Impact: Increases Phishing Risk (+0.1022)
     Note  : Digit composition (10 digits) contributed toward the phishing classification based on patterns learned from the training data.
```

### CLI Error Handling

Invalid URLs produce clean messages on `stderr` and terminate with a non-zero exit status (`1`):

```bash
python3 src/cli.py "ftp://malicious.site"
# Stderr: Error: Unsupported scheme 'ftp'. Only HTTP and HTTPS URLs are supported.
# Exit code: 1

python3 src/cli.py "http://example.com:99999"
# Stderr: Error: Invalid URL port: Port out of range 0-65535
# Exit code: 1
```

---

## 11. Streamlit Web Application

The interactive web dashboard provides an interface for security analysts and end users.

### Launching the Dashboard

```bash
streamlit run app/main.py
```

The application will be accessible at `http://localhost:8501`.

### Key Interface Components

- **Unified Threat Banner**: Displays classification verdict (Phishing vs. Legitimate) alongside exact model confidence percentages.
- **Probability Meter**: Dual progress bars contrasting Phishing Score against Legitimate Score.
- **Top 5 Decision Drivers**: Ranked cards detailing the strongest local SHAP signals, their direction, and cybersecurity context.
- **Visual Attribution Plot**: A horizontal bar plot illustrating how each signal moved the probability relative to baseline prior.
- **Collapsible Feature Table**: Expandable view listing all 18 extracted values, feature categories, raw values, and individual SHAP impact scores.
- **Attack Samples**: Pre-configured selector for testing credential harvesting, worker scripts, and IP attacks.

---

## 12. Automated Testing and Verification

The repository includes an automated test suite comprising **142 unit, integration, and regression tests**.

### Running the Test Suite

Execute the full suite using either `pytest` or Python's built-in `unittest` runner:

```bash
# Option 1: pytest (recommended)
export MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=. pytest -v

# Option 2: Python standard library unittest
export MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=. python3 -m unittest discover tests -v
```

### Reproducing the Domain-Separated Evaluation

To execute the public-suffix-aware domain grouping, retrain on the domain-separated partition, verify zero domain overlap, and evaluate on unseen domains:

```bash
export MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=. python3 src/evaluate_domain_split.py
```

### Test Coverage by Module

| Test Module | Test Count | Scope of Verification |
| :--- | :-: | :--- |
| `tests/test_url_parsing.py` | 41 | Schemes, ports, credentials in authority, IPv4 octet bounds, bracketed IPv6, IDNA Punycode, malformed hosts |
| `tests/test_feature_consistency.py` | 21 | 18-feature schema, ratio bounds [0, 1], 3-decimal rounding, missing/extra/reordered columns, NaN/Inf non-finite validation, unusual URLs |
| `tests/test_prediction.py` | 21 | Dynamic probability extraction, reversed `model.classes_`, threshold boundary conditions, interface parity, backward compatibility with older artifacts |
| `tests/test_shap_explanations.py` | 19 | Phishing class alignment, 3D multi-class & 2D formats, model output verification, ambiguity rejection, additive consistency with floating-point tolerance on legitimate & phishing URLs |
| `tests/test_error_handling.py` | 16 | Missing/truncated/corrupt artifacts, stored metadata validation vs estimator, unsupported schemes, clean CLI stderr reporting, SHAP failure isolation |
| `tests/test_domain_evaluation.py` | 11 | Registered domain extraction (ccTLDs, IPv4/IPv6, userinfo, ports, localhost), zero-domain-overlap enforcement assertion checks, Expected Calibration Error (ECE) calculation |
| `tests/test_cli.py` | 8 | Subprocess invocation, exit codes (0 vs. 1), clean stderr error formatting, validation rejection |
| `tests/test_regression_protection.py` | 5 | Invariant protection: 0=Phishing/1=Legitimate mapping, exact 18-feature order, dynamic class index selection, SHAP class alignment, artifact schema contract |
| **Total Test Count** | **142** | **All 142 tests passing** |

### Whitespace and Git Integrity Verification

```bash
git diff --check
```

---

## 13. Limitations and Responsible Use

### Probabilistic Nature

Model outputs are statistical model-estimated probabilities based on patterns learned from training data, not calibrated probabilities or absolute guarantees of safety or malice.

- A URL flagged as **Legitimate (99.8%)** indicates strong structural resemblance to authentic web architecture; it does not guarantee that the remote server has not been compromised or host malicious content.
- A URL flagged as **Phishing (100.0%)** reflects severe malicious structural indicators (e.g. unencrypted IP address with high special-character ratios); organizations should treat this as a high-risk indicator for automated isolation or analyst review.

### Evasion Techniques and Threat Evolution

- **Adversarial Adaptation & Bare Root Domain Limitation**: In our out-of-distribution stress testing across 7,747 novel TLD URLs, the model achieved $99.56\%$ accuracy but produced 34 false negatives. In-depth inspection revealed that adversaries engineered minimal URLs (e.g. `https://www.lglsvy.top`) using valid HTTPS encryption, short hostnames (22–30 chars), and zero path/query delimiters. Pure lexical features cannot distinguish these bare root domains without external signals (such as WHOIS domain creation age, passive DNS reputation, or SSL certificate authority).
- **Defense-in-Depth**: This tool should be deployed as one layer of a multi-tiered security architecture alongside DNS filtering, SSL inspection, behavioral sandboxing, and endpoint protection.
- **No Payload Execution**: This system evaluates URLs pre-flight and does not inspect DOM structure, JavaScript execution, or SSL certificate chains.

---

## 14. Reproducibility, Future Work, and References

### Model Retraining

To retrain the Random Forest model artifact from the raw dataset:

```bash
python3 src/train.py
```

### Future Enhancements

- **Ensemble Comparison**: Integrate gradient-boosted trees (XGBoost, LightGBM) as comparative models (preliminary exploration documented in `notebook/01_dataset_analysis.ipynb`).
- **External Telemetry**: Incorporate WHOIS domain age and passive DNS signals to augment lexical indicators.
- **FastAPI Endpoint**: Package the inference pipeline as a containerized REST API for integration into security orchestration platforms (SOAR).

### References

1. **Dataset Publication**: Prasad, A., & Chandra, S. (2024). *PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning*. Computers & Security, 138, 103545. [DOI: 10.1016/j.cose.2024.103545](https://doi.org/10.1016/j.cose.2024.103545).
2. **Explainable AI (SHAP)**: Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems (NeurIPS 30).
3. **URL Standards**: Berners-Lee, T., Fielding, R., & Masinter, L. (2005). *Uniform Resource Identifier (URI): Generic Syntax*. RFC 3986, Internet Engineering Task Force (IETF).
