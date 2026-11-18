# Phase 7 Milestone 6: Operational Readiness & Generalisation Assessment
## Engineering & Diagnostic Specification (Revision 2 — Post-Review)

- **Milestone Code:** `Phase 7 — M6`
- **Milestone Name:** Operational Readiness & Generalisation Assessment
- **Target Branch:** `phase-7-m6-operational-readiness-and-diagnostics`
- **Evaluation Status:** Diagnostic & Investigative Protocol (**Zero Retraining / Zero Production Promotion**)
- **Production Baseline Model (E0):** `models/random_forest_model.joblib`  
  - SHA-256: `95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb` (Immutable Control)
- **Experimental Candidate Model (E4):** `models/phase7_candidates/rf_augmented_e4.joblib`  
  - SHA-256: `0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b` (Frozen Candidate)

---

## 1. Executive Summary & Objective

In Phase 7 Milestone 5.1, confirmatory external evaluation against an independently crawled, socket-verified benchmark (`m5_1_qualified_benchmark.csv`, $N=1,300$, 810 unique registered domains) confirmed that Candidate Model E4 achieves high threat recall ($99.23\%$ at $T=0.75$, $+2.15\%$ gain over control E0, $p = 0.0010$). However, both models exhibited catastrophic false positive rates (FPR) on authentic legitimate URLs:
- **Frozen Control E0:** $\text{FPR} = 94.1538\%$ ($612/650$ false alarms at $T=0.75$)
- **Candidate Model E4:** $\text{FPR} = 92.6154\%$ ($602/650$ false alarms at $T=0.75$)

Because the operational deployment ceiling mandates $\text{FPR} \le 0.10\%$, Candidate E4 was determined as **State 3 — FAIL: False Alarm Deficit Exceeds Operational Ceiling**.

### Core Objective of Milestone 6
Milestone 6 is strictly a **diagnostic and operational readiness assessment milestone**, NOT a model-training or model-improvement milestone. Its sole purpose is to establish, with reproducible observational evidence, **why** the current phishing detection architecture fails on authentic legitimate URLs.

> [!IMPORTANT]
> **Strict Governance Invariants:**
> 1. **Zero Retraining:** Under no circumstances will any model be retrained, fine-tuned, re-weighted, threshold-tuned, modified, or promoted.
> 2. **Model Immutability:** E0 remains the frozen production baseline control (`95f09ddc...`). E4 remains an experimental candidate (`0f872c0f...`). Both model artifacts are loaded read-only and cryptographically verified before and after every diagnostic run.
> 3. **Observational Discipline:** M6 evaluates observational associations, distributions, and structural properties. Correlational observations must not be reported as proven causal conclusions.

---

## 2. Research Hypotheses & Diagnostic Boundaries

To avoid premature causal conclusions, M6 evaluates five distinct, non-exclusive hypotheses using observational data:

```mermaid
flowchart TD
    Obs["Observed Failure: ~92-94% FPR on External Legitimate URLs"] --> H1["H1: Training Representation Bias<br>(PhiUSIIL Benign URLs = Truncated Roots)"]
    Obs --> H2["H2: Covariate Shift<br>(External Web Distribution Diverges from 2021 Corpus)"]
    Obs --> H3["H3: Lexical Feature Ceiling<br>(18 String Metrics Inadequate Without Reputation/Content)"]
    Obs --> H4["H4: Augmentation Scale Deficit<br>(119 Augmented Benign Rows Drowned by 170,496 Base Rows)"]
    Obs --> H5["H5: Multi-Factor Interaction<br>(Combined Structural Collinearity & Sample Imbalance)"]
```

- **Hypothesis $H_1$ (Training Representation Bias):** Benign URLs in the `PhiUSIIL` training set are structurally unrepresentative of the live web (e.g., in the training partition, $100\%$ lack query parameters and max length is $\le 57$), causing tree splits to associate URL complexity with phishing.
- **Hypothesis $H_2$ (Covariate Shift):** Natural temporal and architectural evolution of institutional web routing since 2021 has introduced feature distributions unseen during model development.
- **Hypothesis $H_3$ (Lexical Feature Ceiling):** Purely lexical string properties (character ratios, length, delimiters) are mathematically insufficient to differentiate complex legitimate routes (e.g., query filters, deep REST routes) from phishing lures without domain reputation, WHOIS age, or content signals.
- **Hypothesis $H_4$ (Augmentation Scale Deficit):** The candidate augmentation in M3 added only 119 benign URLs ($0.07\%$ of the 170,615 training rows), which was numerically diluted across the 100 bootstrap estimators.
- **Hypothesis $H_5$ (Interaction Effect):** The failure is driven by joint interactions between training representation bias and feature collinearity.

---

## 3. Required Specification Structure & Deliverables

### Section M6.1: Root-Cause and Tree-Path Investigation
**Objective:** Dissect the internal voting mechanism of both Random Forest ensembles (E0 and E4) to identify the exact feature thresholds and split sequences responsible for misclassifying legitimate URLs.

1. **Estimator Correspondence Methodology:**
   - Because Random Forest decision trees are trained via bootstrap subsampling and random feature selection, individual estimators between independently trained forests (E0 vs E4) **do not have a 1:1 index correspondence** (i.e. Tree $k$ in E0 does not correspond to Tree $k$ in E4).
   - Therefore, tree comparisons between E0 and E4 will be reported as **ensemble-wide structural distributions** and **aggregate node statistics**, not as individual tree-to-tree differences.
2. **Decision Path Extraction Protocol:**
   - For every false-positive URL $x \in \mathcal{D}_{\text{External, Legit}}^{\text{FP}}$ ($n = 612$ for E0, $n = 602$ for E4 at $T=0.75$), extract the full decision path across all 100 decision trees:
     $$\mathcal{P}_t(x) = \left[ (f_1, \theta_1, \text{dir}_1), (f_2, \theta_2, \text{dir}_2), \dots, (f_d, \theta_d, \text{leaf}) \right] \quad \text{for } t \in \{1, \dots, 100\}$$
   - Record the leaf node class distributions $\left( \frac{n_{\text{phish}}}{n_{\text{total}}}, \frac{n_{\text{legit}}}{n_{\text{total}}} \right)$ where samples land.
3. **Quantitative Metrics:**
   - **Root & Depth-1 Split Feature Dominance:** Percentage of trees where root or depth-1 splits use `IsHTTPS`, `NoOfOtherSpecialCharsInURL`, `URLLength`, or `LetterRatioInURL`.
   - **Critical Threshold Crossing Frequency:** For each feature $f \in \mathcal{F}$, compute the proportion of trees where $x[f] > \theta_{t, i}$ directly routes traversal into a predominantly phishing subtree.
   - **Ensemble Agreement (Vote Entropy):** Compute the distribution of tree votes $V(x) = \frac{1}{100} \sum_{t=1}^{100} \mathbb{I}(\text{Tree}_t(x) = \text{Phishing})$ to assess whether false positives are borderline ($V \approx 0.51$) or unanimous ($V > 0.90$).
   - **Ensemble Tree Purity Distribution:** Compare tree leaf impurity distributions (Gini index at termination) between E0 and E4 across all 100 estimators.

---

### Section M6.2: SHAP Global and Local Attribution
**Objective:** Quantify additive feature contributions to model predictions using TreeSHAP.

1. **Output Space & Formulation:**
   - **Single Output Space:** All SHAP values are computed strictly in **probability space for the Phishing class ($P(\text{phishing}) \in [0, 1]$)**.
   - **Class Index Lock:** Class index 0 is Phishing (`clf.classes_ == [0, 1]`, `phishing_index == 0`).
   - **Additive Property:**
     $$P(\text{phishing} \mid x) = \phi_0 + \sum_{i=1}^{18} \phi_i(x)$$
     where $\phi_0 = \mathbb{E}[P(\text{phishing})]$ is the expected tree leaf probability of phishing across the background dataset, and $\phi_i(x)$ is the marginal additive probability contribution of feature $i$.
   - **Interpretation:** $\phi_i(x) > 0$ increases the predicted phishing probability, while $\phi_i(x) < 0$ decreases it.
2. **Deliverables:**
   - **Global Summary Beeswarm Plot:** Global feature impact on the external benchmark, showing directionality and magnitude.
   - **Class-Separated SHAP Distributions:** Compare mean absolute SHAP values $|\phi_i|$ between:
     - True Positives (genuine phishing detected)
     - False Positives (legitimate URLs false-alarmed)
     - True Negatives (legitimate URLs correctly passed)
   - **Local Force Plots for Archetypal Errors:** Individual breakdown of 5 representative institutional URLs from diverse sectors (`.gov`, `.edu`, `.be`, e-commerce).
   - **SHAP Interaction Values:** Compute second-order interaction matrices $\phi_{i, j}(x)$ to evaluate non-linear compounding effects (e.g., `URLLength` $\times$ `NoOfOtherSpecialCharsInURL`).

---

### Section M6.3: Class-Separated Covariate Shift & Feature Drift
**Objective:** Measure distribution shift between training and external operational environments without confounding class-composition differences with feature drift.

> [!IMPORTANT]
> **Mandatory Methodological Rule:**
> Covariate shift MUST be evaluated separately for legitimate and phishing partitions:
> 1. $\mathcal{D}_{\text{Train, Legit}}$ vs $\mathcal{D}_{\text{External, Legit}}$ ($n = 134,850$ vs $n = 650$)
> 2. $\mathcal{D}_{\text{Train, Phish}}$ vs $\mathcal{D}_{\text{External, Phish}}$ ($n = 100,945$ vs $n = 650$)
> A pooled comparison ($X_{\text{train}}$ vs $X_{\text{external}}$) is strictly prohibited as it would conflate class prevalence with genuine feature drift.

1. **Statistical Drift Quantifiers:**
   - **Population Stability Index (PSI):**
     $$\text{PSI}_f = \sum_{b=1}^{B} \left( P_b - Q_b \right) \ln\left(\frac{P_b}{Q_b}\right)$$
     - **Reference Population ($Q$):** Training split ($\mathcal{D}_{\text{train}}$).
     - **Evaluation Population ($P$):** External benchmark ($\mathcal{D}_{\text{external}}$).
     - **Bin Edges:** 10 quantile bins derived from the reference population. Discrete/binary features (`IsDomainIP`, `HasObfuscation`, `IsHTTPS`) use exact categorical bins $[0, 1]$.
     - **Zero-Count Smoothing:** Epsilon $\epsilon = \frac{1}{2 \cdot N_{\text{ext}}}$ added to any empty bin count before computing $\ln(P_b / Q_b)$.
     - **Constant Features:** If a feature has zero variance in the reference population, report $\text{PSI} = 0.0$ if identically constant in the evaluation set, or flag as degenerate shift ($\infty$).
     - **Thresholds:** $\text{PSI} < 0.10$ (No significant shift), $0.10 \le \text{PSI} < 0.25$ (Moderate shift), $\text{PSI} \ge 0.25$ (Severe shift).
   - **Two-Sample Kolmogorov-Smirnov (KS) Test:**
     - Report the **effect size $D_{KS} = \sup_v |F_{\text{train}}(v) - F_{\text{ext}}(v)|$** alongside the asymptotic $p$-value. Because $N_{\text{train}} > 100,000$, virtually all $p$-values reach significance ($p < 10^{-15}$); operational drift must be judged by effect size:
       - $D_{KS} < 0.10$: Negligible divergence.
       - $0.10 \le D_{KS} < 0.25$: Moderate divergence.
       - $D_{KS} \ge 0.25$: Substantial / severe divergence.
   - **Wasserstein Distance:** $\mathcal{W}_1(P, Q) = \int_{-\infty}^{\infty} |F_P(v) - F_Q(v)| dv$.
2. **Tabular Deliverable:** Complete 18-feature drift scorecard reporting $\text{PSI}$, $D_{KS}$, $p_{KS}$, mean $\mu$, standard deviation $\sigma$, and 95th percentiles across both partitions.

---

### Section M6.4: Structural and Sector-Level Error Analysis
**Objective:** Decompose false positive errors across frozen architectural URL categories and institutional sectors with explicit statistical denominators.

1. **Frozen Structural Strata Definitions (Mutually Exclusive):**
   Parsing follows unambiguous priority logic to prevent edge-case misclassification:
   ```python
   def extract_path_segments(path: str) -> List[str]:
       """Normalize and split path into non-empty segments, collapsing repeated slashes."""
       clean = re.sub(r"/+", "/", path.strip())
       return [s for s in clean.split("/") if s]

   def assign_structural_stratum(parsed_url) -> str:
       # Priority 1: Special syntax (IP literal, @ credentials, non-standard port, %xx encoding)
       if parsed_url.is_ip or parsed_url.has_credentials or parsed_url.has_obfuscation:
           return "special_syntax_variants"
       if parsed_url.port and parsed_url.port not in (80, 443):
           return "special_syntax_variants"

       segments = extract_path_segments(parsed_url.path)
       has_query = bool(parsed_url.query and parsed_url.query.strip())

       # Priority 2: Root with query
       if len(segments) == 0 and has_query:
           return "root_with_query"

       # Priority 3: Path and query
       if len(segments) >= 1 and has_query:
           return "path_and_query"

       # Priority 4: Homepage root
       if len(segments) == 0:
           return "homepage_root"

       # Priority 5: Single-level path
       if len(segments) == 1:
           return "single_level_path_no_query"

       # Priority 6: Multi-level path
       return "multi_level_path_no_query"
   ```

2. **Institutional Sector Decomposition:**
   - **Scope:** The five institutional sectors apply specifically to the 650 legitimate URLs in the M5.1 qualified benchmark ($n_{\text{legit}} = 130$ per sector):
     1. E-Commerce & Retail ($n_{\text{legit}} = 130$)
     2. Higher Education & Research ($n_{\text{legit}} = 130$)
     3. Banking & Financial Regulators ($n_{\text{legit}} = 130$)
     4. National Government & Ministries ($n_{\text{legit}} = 130$)
     5. General Institutional & Open Infrastructure ($n_{\text{legit}} = 130$)
   - **Sector Provenance:** Sector labels were assigned during external catalog curation based on official institutional registries, independently of model predictions.
   - **Phishing Denominators:** The 650 phishing URLs in M5.1 were sourced from PhishTank community consensus and reside in the separate `phishing_threat_feed` category ($n_{\text{phish}} = 650$). For the 5 institutional sectors, $n_{\text{phish}} = 0$; therefore, **Phishing Threat Recall is mathematically undefined for individual institutional sectors** (reported as `Undefined (n_p = 0)`), NOT zero.
3. **Statistical Reporting Standard:**
   - Denominator strictly defined: $\text{FPR} = \frac{\text{FP}}{n_{\text{legit}}}$ where $n_{\text{legit}}$ is the number of legitimate observations in that stratum/sector.
   - Exact Clopper-Pearson 95% binomial confidence intervals:
     $$\text{CI}_{0.95} = \left[ B\left(\frac{\alpha}{2}; \text{FP}, n_l - \text{FP} + 1\right), B\left(1 - \frac{\alpha}{2}; \text{FP} + 1, n_l - \text{FP}\right) \right]$$
   - Explicit sample size caveat: `Small (n_l < 30)` or `Adequate (n_l >= 30)`.

---

### Section M6.5: Attribution Disentanglement (Strictly Observational)
**Objective:** Empirically evaluate the five research hypotheses through observational analysis of existing frozen models and datasets without retraining.

1. **Observational Analyses (Zero Retraining):**
   - **Factor A (Representation Disparity):** Compare feature distribution overlap between training legitimate URLs ($N=134,850$) and external legitimate URLs ($N=650$). Measure what percentage of external legitimate URLs fall outside the minimum-maximum bounding boxes of the training set across each feature.
   - **Factor B (Lexical Separability Bounds):** Fit an observational linear probe (Logistic Regression with $L_1$ penalty) strictly on the frozen external benchmark features to compute the empirical upper bound of linear separability under the 18 lexical features. (This evaluates whether the features contain sufficient discriminatory information, without retraining E0 or E4).
   - **Factor C (Class-Conditional Feature Overlap):** Quantify intra-class and inter-class Wasserstein distance and feature correlations between legitimate and phishing URLs within the external benchmark. Determine whether lexical features (such as `URLLength` or `NoOfOtherSpecialCharsInURL`) exhibit structural collinearity with the phishing label in real-world traffic.
   - **Factor D (Augmentation Representation in Frozen E4):** Inspect the exact composition of E4's training data ($170,496$ PhiUSIIL rows vs $119$ augmented benign rows; $0.07\%$ ratio). Quantify the aggregate structural differences between E0 and E4 (distribution of node counts, leaf counts, and threshold shifts) to assess whether 119 rows caused measurable structural changes across the 100 trees.
2. **Scientific Reporting Language:**
   - Findings must be framed as observed empirical associations, bounds of separability, and distribution divergence. Do not claim causal mechanisms from observational comparisons alone.

---

### Section M6.6: Multi-Benchmark Cross-Generalisation
**Objective:** Cross-evaluate frozen models E0 and E4 across all three distinct historical and qualified evaluation benchmarks with full cryptographic provenance.

1. **Exact Holdout Inventory (Three Distinct Benchmarks):**
   | Benchmark Dataset | Artifact Path | Expected SHA-256 | Sample Count | Registered Domains | Provenance & Role |
   |:---|:---|:---|:---:|:---:|:---|
   | **1. Internal Test Split** | `experiments/phase7/datasets/splits/internal_test.csv` | `ce004e9d8e65ea8c09175d904d4a5fee0442acc4b127af6a98f58ae0aed0e2cf` | 34,092 | 25,978 | Domain-disjoint holdout from PhiUSIIL (Internal Baseline) |
   | **2. Historical External Benchmark** | `data/external/external_validation_benchmark.csv` | `88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3` | 1,047 | 892 | Historical external benchmark used in Phase 4 & Phase 6 (Replication Holdout) |
   | **3. M5.1 Qualified External Benchmark** | `experiments/phase7/external/datasets/m5_1_qualified_benchmark.csv` | `f6cdbf797b683d6c82d7b5c72c6ddf35aba7cbb448edafc2d68e67e26695fc88` | 1,300 | 810 | Socket-verified authentic crawl with 100% verified TLS (Independent Operational Holdout) |

   > [!NOTE]
   > **Holdout Resolution Note:** The earlier informal reference to a "1,200-sample Phase 6 benchmark" was legacy planning nomenclature from Phase 4/5. The actual committed historical external artifact is the 1,047-sample `data/external/external_validation_benchmark.csv`. The evaluation inventory is strictly frozen at these three distinct datasets.

2. **Pre-Flight Invariant Checks:**
   - Verify SHA-256 digests against pre-evaluation manifest before running comparisons.
   - Audit registered-domain overlap across all 3 datasets to document historical leakage and independence.
3. **Paired Statistical Comparison Standards:**
   - Thresholds: $T = 0.50$ (Primary) and $T = 0.75$ (Secondary).
   - Metrics: Phishing Recall, Legitimate FPR, Macro F1, ROC-AUC, PR-AUC, Brier Score.
   - **Paired McNemar Tests:** Computed with Edward's continuity correction on discordant pairs:
     $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$
     where $n_{01}$ represents samples where E0 is correct and E4 is incorrect, and $n_{10}$ represents samples where E0 is incorrect and E4 is correct. McNemar tests evaluate discordant pairs between models, not general metric uncertainty.

---

### Section M6.7: Bayesian Operational Reliability
**Objective:** Translate raw TPR and FPR into actionable real-world security metrics across varying operational threat priors.

1. **Prevalence Priors $\pi$:**
   - Gateway / Proxy Inspection: $\pi = 0.0001$ ($0.01\%$, 1 phishing URL per 10,000 URLs).
   - Targeted Email Filter: $\pi = 0.001$ ($0.10\%$, 1 phishing URL per 1,000 URLs).
   - High-Risk Influx / Incident Response: $\pi = 0.01$ ($1.00\%$, 1 phishing URL per 100 URLs).
2. **Formulations:**
   $$\text{PPV}(\pi) = \frac{\pi \cdot \text{Recall}}{\pi \cdot \text{Recall} + (1 - \pi) \cdot \text{FPR}}$$
   $$\text{False Alarm Multiplier (FAM)} = \frac{(1 - \pi) \cdot \text{FPR}}{\pi \cdot \text{Recall}} = \frac{1}{\text{PPV}} - 1$$
3. **Reporting Standard:**
   - Tabulate PPV and False Alarm Multipliers across both operating thresholds ($T = 0.50, T = 0.75$) for all three benchmarks.
   - State the False Alarm Multiplier explicitly: how many benign URLs are falsely blocked for every genuine phishing URL detected.

---

### Section M6.8: Production Governance & Evidence Integrity
**Objective:** Maintain cryptographic reproducibility, model immutability, and compliance with project invariants.

1. **Model Governance Invariants:**
   - Control E0 (`models/random_forest_model.joblib`) must match `95f09ddc...2f39bb`.
   - Candidate E4 (`models/phase7_candidates/rf_augmented_e4.joblib`) must match `0f872c0f...c6f1b`.
   - Zero model files may be created or overwritten in `models/`. Models must be loaded read-only and verified before and after execution.
2. **Manifest Architecture:**
   - Pre-evaluation lock: `experiments/phase7/reports/m6_operational_readiness_manifest.json` (Stage 0).
   - Diagnostic scorecard: `experiments/phase7/reports/m6_diagnostic_scorecard.json`.
   - Comprehensive audit report: `experiments/phase7/reports/m6_operational_readiness_report.md`.
3. **Automated Verification:**
   - All diagnostic outputs, decision-path extractors, drift calculators, and parsing rules must be tested in `experiments/phase7/tests/test_m6_operational_readiness.py`.

---

## 4. Formal Acceptance Criteria (AC-1 through AC-8)

| Criteria | Title | Verification Target |
|:---|:---|:---|
| **AC-1** | **Tree Decision Path Extraction** | Decision paths extracted across 100 trees for all false-positive URLs; root split dominance, threshold crossing frequency, and leaf class purity quantified without asserting unproven tree-to-tree correspondence. |
| **AC-2** | **SHAP Global & Local Attribution** | TreeSHAP computed strictly in Phishing probability space ($P(\text{phishing}) \in [0, 1]$); beeswarm, class-separated $|\phi_i|$ values, and 5 local force plots generated and saved. |
| **AC-3** | **Class-Separated Covariate Shift** | PSI and KS statistics ($D_{KS}$ effect size and $p$-value) computed strictly within legitimate ($\mathcal{D}_{\text{train, legit}}$ vs $\mathcal{D}_{\text{ext, legit}}$) and phishing ($\mathcal{D}_{\text{train, phish}}$ vs $\mathcal{D}_{\text{ext, phish}}$) partitions independently across all 18 features using 10 quantile bins and epsilon zero-smoothing. |
| **AC-4** | **Structural Strata & Sector Error Matrix** | Error rates evaluated across the 6 frozen strata and 5 sectors ($n_{\text{legit}} = 130$ each) with explicit denominators ($N, n_p, n_l$), Clopper-Pearson 95% CIs, and undefined phishing recall explicitly noted for institutional sectors. |
| **AC-5** | **Attribution Disentanglement** | Strictly observational analyses completed: training bounding box overlap, linear probe lexical separability, class-conditional Wasserstein distances, and E0 vs E4 structural ensemble metrics without training new models. |
| **AC-6** | **Multi-Benchmark Generalisation** | Comparative metrics and paired McNemar tests computed across exactly three holdouts: Internal Test ($N=34,092$), Historical External ($N=1,047$), and M5.1 Qualified ($N=1,300$) with verified SHA-256 hashes. |
| **AC-7** | **Bayesian PPV & False Alarm Multipliers** | PPV and False Alarm Multipliers calculated and tabulated across $\pi \in \{10^{-4}, 10^{-3}, 10^{-2}\}$ at $T=0.50$ and $T=0.75$. |
| **AC-8** | **Production Governance & Model Immutability** | Cryptographic assertion that E0 and E4 model files are byte-for-byte unchanged; models loaded read-only; zero model promotion or retraining permitted. |

---

## 5. Execution Roadmap & Staged Implementation Plan

Implementation proceeds in strictly staged phases:

```mermaid
flowchart LR
    S0["Stage 0: Pre-Evaluation Manifest Lock"] --> S1["Stage 1: Tree-Path & SHAP Diagnostics"]
    S1 --> S2["Stage 2: Class-Separated Covariate Shift"]
    S2 --> S3["Stage 3: Stratified & Multi-Benchmark Inference"]
    S3 --> S4["Stage 4: Automated Verification Suite (AC-1..AC-8)"]
    S4 --> S5["Stage 5: Final Report & Pull Request"]
```

1. **Stage 0 (Pre-Flight Lock):** Generate `m6_operational_readiness_manifest.json` locking all dataset and model hashes, feature schema, label conventions, and environment metadata.
2. **Stage 1 (Ensemble Dissection):** Author `extract_tree_decision_paths.py` and `compute_shap_attributions.py`.
3. **Stage 2 (Distribution Drift):** Author `compute_covariate_shift.py` enforcing class-separated PSI and KS tests.
4. **Stage 3 (Multi-Holdout Inference):** Author `evaluate_multi_benchmark_generalisation.py` covering all three benchmarks.
5. **Stage 4 (Verification Suite):** Implement `experiments/phase7/tests/test_m6_operational_readiness.py` validating AC-1 through AC-8.
6. **Stage 5 (Report Compilation):** Emit `m6_diagnostic_scorecard.json` and `m6_operational_readiness_report.md`, and open PR for review.
