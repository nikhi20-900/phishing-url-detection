# Phase 7 Milestone 5.1 — Evidence Integrity Audit and Benchmark Qualification Report

**Execution Timestamp (UTC):** 2026-11-18T18:57:22.455041+00:00  
**Evaluation Protocol:** Confirmatory Single-Pass Evaluation (Frozen Production Control E0 vs Experimental Candidate E4)  
**Qualified External Benchmark:** `m5_1_qualified_benchmark.csv` (SHA-256: `8e524bb539371c3e129037cf9a28d22b2f73a99f6201995fcc90951586c6110b`)  
**Raw Observations Log:** `raw_observations_m5_1.jsonl` (Authentic socket-level retrieval; zero synthetic fallback)  

---

## 1. Executive Summary & Regulatory Determination

### 1.1 Remediation of Critical M5 Evidence-Integrity Defects
In Milestone 5.1, the 6 critical evidence-integrity defects identified in the legacy M5 benchmark were rigorously remediated:
1. **Security Quarantine:** The leaked HMAC secret key (`m5_audit_signing_secret.key`) was untracked from Git, placed in `.gitignore`, and permanently decommissioned. Reviewer logs were transparently relabeled as automated research execution records (`AUTOMATED_RESEARCH_LOG`).
2. **Fail-Closed Domain Auditing:** `audit_domain_independence.py` enforces a fail-closed invariant, dynamically verifying zero domain collision across 173,421 historical domains across 11 registered partitions.
3. **Authentic Web Crawling:** Synthetic path templates were eliminated. 650 authentic legitimate URLs were crawled live from verified institutional domains across 5 sectors, satisfying exact structural strata quotas via Mixed-Integer Linear Programming (MILP).
4. **Authentic Socket-Level Network Evidence:** Every admitted URL is supported by live socket connections, DNS resolutions, IP pinning, TLS handshake logs, and body SHA-256 digests in `raw_observations_m5_1.jsonl` with zero synthetic fallback.
5. **Defensive Threat Feed Verification:** Phishing candidates strictly require `verified == 'yes'` and `online == 'yes'`, cryptographically anchored to upstream snapshot SHA-256 `4885e372...`.
6. **Benchmark Legacy Reclassification:** Legacy M5 was reclassified as `M5-Synthetic-Lexical-Stress-Test-v1`, preserving historical integrity while establishing `m5_1_qualified_benchmark.csv` as the genuine independent benchmark.

### 1.2 Confirmatory Evaluation Verdict ($T = 0.75$)
- **Decision Matrix State:** **State 2 (Empirical Boundary)**
- **Determination:** **EMPIRICAL DETERMINATION**
- **Evaluation Details:** Measured Recall=99.23%, FPR=92.1538%. Evaluated under domain-clustered confidence bounds.
- **Governance Mandate:** **Zero Retraining and Zero Production Promotion.** E0 remains the frozen production baseline. E4 remains an experimental candidate.

---

## 2. Model Performance Summary

### 2.1 Primary Operating Point ($T = 0.50$)
| Metric | Frozen Control E0 | Candidate E4 | Delta (E4 − E0) | 95% Clustered Bootstrap CI | Monte Carlo Permutation $p$ |
|:-------|:-----------------:|:------------:|:---------------:|:--------------------------:|:---------------------------:|
| **Phishing Recall** | 99.8462% | 99.6923% | -0.1538% | [-0.7454%, +0.3125%] | $p = 1.0000$ |
| **Legitimate FPR** | 97.5385% | 96.9231% | -0.6154% | [-2.0447%, +0.7251%] | $p = 0.5487$ |
| **Macro F1** | 0.3597 | 0.3659 | +0.0061 | [-0.0078, +0.0212] | — |
| **Phishing Precision** | 50.5846% | 50.7042% | — | — | — |
| **ROC-AUC** | 0.6093 | 0.7485 | — | — | — |
| **PR-AUC** | 0.7456 | 0.8113 | — | — | — |
| **Brier Score** | 0.4655 | 0.4398 | — | — | — |
| **Expected Calibration Error (ECE)** | 0.4695 | 0.4589 | — | — | — |

### 2.2 Pre-Specified Operational Operating Point ($T = 0.75$)
| Metric | Frozen Control E0 | Candidate E4 | Delta (E4 − E0) | 95% Clustered Bootstrap CI | Monte Carlo Permutation $p$ |
|:-------|:-----------------:|:------------:|:---------------:|:--------------------------:|:---------------------------:|
| **Phishing Recall** | 97.0769% | 99.2308% | +2.1538% | [+0.9433%, +3.3691%] | $p = 0.0012$ |
| **Legitimate FPR** | 95.5385% | 92.1538% | -3.3846% | [-6.0326%, -0.9598%] | $p = 0.0154$ |
| **Macro F1** | 0.3733 | 0.4128 | +0.0395 | [+0.0157, +0.0653] | — |

---

## 3. Clustered Inferential Statistics & Uncertainty Bounds

### 3.1 Domain-Clustered Bootstrap Percentile Intervals (B = 1,000)
- **Δ Phishing Recall (T = 0.50):** -0.1538% (95% CI: [-0.7454%, +0.3125%])
- **Δ Legitimate FPR (T = 0.50):** -0.6154% (95% CI: [-2.0447%, +0.7251%])
- **Δ Phishing Recall (T = 0.75):** +2.1538% (95% CI: [+0.9433%, +3.3691%])
- **Δ Legitimate FPR (T = 0.75):** -3.3846% (95% CI: [-6.0326%, -0.9598%])

### 3.2 Domain-Level Wilcoxon Signed-Rank Test (Legitimate FPR)
- **W-statistic:** 49.5
- **Asymptotic p-value:** 0.8457
- **Rank-Biserial Effect Size (r_rb):** -0.0571 (negligible effect)
- **Domain Breakdown:** 160 evaluable domains (14 effective non-zero differences, 146 zero differences).

---

## 4. Stratified Subgroup Performance (Explicit Denominators)

| Subgroup / Stratum | Total N | Phish n_p | Legit n_l | E0 Legit FPR | E4 Legit FPR | Δ Legit FPR | E0 Phish Recall | E4 Phish Recall | Status / Caveat |
|:-------------------|--------:|----------:|----------:|:------------:|:------------:|:-----------:|:---------------:|:---------------:|:----------------|
| `homepage_root` | 497 | 367 | 130 | 90.0000% | 92.3077% | +2.3077% | 99.7275% | 100.0000% | Adequate (n_l >= 30) |
| `single_level_path_no_query` | 210 | 110 | 100 | 97.0000% | 100.0000% | +3.0000% | 100.0000% | 99.0909% | Adequate (n_l >= 30) |
| `multi_level_path_no_query` | 273 | 108 | 165 | 100.0000% | 99.3939% | -0.6061% | 100.0000% | 100.0000% | Adequate (n_l >= 30) |
| `root_with_query` | 74 | 9 | 65 | 100.0000% | 93.8462% | -6.1538% | 100.0000% | 100.0000% | Adequate (n_l >= 30) |
| `path_and_query` | 154 | 24 | 130 | 100.0000% | 96.9231% | -3.0769% | 100.0000% | 95.8333% | Adequate (n_l >= 30) |
| `special_syntax_variants` | 92 | 32 | 60 | 100.0000% | 98.3333% | -1.6667% | 100.0000% | 100.0000% | Adequate (n_l >= 30) |

---

## 5. Marginal Structural Feature Flags
| Feature Condition | Total N | Phish n_p | Legit n_l | E0 Legit FPR | E4 Legit FPR | Δ Legit FPR |
|:------------------|--------:|----------:|----------:|:------------:|:------------:|:-----------:|
| `query_bearing==True` | 240 | 39 | 201 | 100.0000% | 95.5224% | -4.4776% |
| `query_bearing==False` | 1060 | 611 | 449 | 96.4365% | 97.5501% | +1.1136% |
| `multi_level_path==True` | 404 | 142 | 262 | 100.0000% | 98.0916% | -1.9084% |
| `multi_level_path==False` | 896 | 508 | 388 | 95.8763% | 96.1340% | +0.2577% |
| `special_syntax==True` | 92 | 32 | 60 | 100.0000% | 98.3333% | -1.6667% |
| `special_syntax==False` | 1208 | 618 | 590 | 97.2881% | 96.7797% | -0.5085% |

---

## 6. Sensitivity Analysis: Domain Thinning (100 Iterations of 1 URL / Domain)
To evaluate the impact of within-domain correlation, domain-thinning was performed across 100 randomized draws (selecting exactly 1 URL per registered domain):
- **Thinning Sample Size:** 810 samples per draw
- **E0 Legitimate FPR:** Mean 97.3750% (SD: 1.0789%)
- **E4 Legitimate FPR:** Mean 96.9688% (SD: 1.0982%)
- **E0 Phishing Recall:** Mean 99.8462% (SD: 0.0000%)
- **E4 Phishing Recall:** Mean 99.6923% (SD: 0.0000%)

---

## 7. Bayesian Population-Weighted Precision Curves
Across three operational phishing prevalence priors $\pi \in \{10^{-4}, 10^{-3}, 10^{-2}\}:

| Prevalence Prior $\pi$ | E0 Bayesian PPV ($T=0.50$) | E4 Bayesian PPV ($T=0.50$) | E0 Bayesian PPV ($T=0.75$) | E4 Bayesian PPV ($T=0.75$) |
|:-----------------------|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|
| **0.01% (Enterprise Gateway)** | 0.0102% | 0.0103% | 0.0102% | 0.0108% |
| **0.10% (Targeted Email)** | 0.1024% | 0.1029% | 0.1016% | 0.1077% |
| **1.00% (High-Risk Influx)** | 1.0234% | 1.0283% | 1.0159% | 1.0760% |

---

## 8. Cryptographic Artifact Attestations
All hashes verified against `experiments/phase7/reports/m5_1_pre_evaluation_manifest.json`:
- **Frozen Production Control E0:** `95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb` (Unmodified)
- **Frozen Candidate Model E4:** `0f872c0fa3d7f6bedfb5ed922355bb121f9c80d9bd2f991ad1cbb46be08c6f1b` (Unmodified)
- **Qualified Benchmark Dataset:** `8e524bb539371c3e129037cf9a28d22b2f73a99f6201995fcc90951586c6110b`
- **Upstream Threat Snapshot:** `4885e37276d8a33b3da045d9fca0888c152e4d20c91e7a69f39eb032893a54ba`
