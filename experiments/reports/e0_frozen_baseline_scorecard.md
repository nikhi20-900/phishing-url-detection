# E0 Frozen-Baseline Evaluation Scorecard

**Execution Timestamp (UTC)**: `2026-11-08T11:58:08Z`
**Model Artifact**: `models/random_forest_model.joblib` (Strictly Frozen)
**Evaluation Scope**: In-Distribution ($N=235,370$), External Validation ($N=1,047$), Controlled Perturbations ($N=726$)

---

## 1. Executive Summary: The Generalization Dichotomy

The frozen production Random Forest model achieves outstanding performance on in-distribution test sets (both stratified and domain-separated), but undergoes catastrophic performance degradation when confronted with out-of-distribution authentic deep-link URLs.

| Evaluation Dimension | Benchmark Dataset | Sample Size ($N$) | Phishing Recall | Legitimate FPR | ROC-AUC | ECE (10 Bins) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **In-Dist Stratified** | PhiUSIIL 80/20 Holdout | 47,074 | 99.46% | 0.09% | 0.9982 | 0.0006 |
| **In-Dist Out-of-Domain** | PhiUSIIL Seed 14 Split | 47,037 | 99.54% | 0.01% | 0.9987 | 0.0005 |
| **Out-of-Dist External** | External Benchmark (Raw) | 1,047 | **99.82%** | **94.00%** | 0.6827 | 0.4330 |
| **Out-of-Dist External** | Curated Legit (Stripped /) | 500 | N/A | **86.80%** | N/A | N/A |

---

## 2. Suite 1: In-Distribution Benchmark (PhiUSIIL)

### 2.1 Stratified Test Holdout ($N=47,074$, Seed 42)
- **Accuracy**: `99.720%`
- **Phishing Precision / Recall / F1**: `99.89%` / `99.46%` / `99.67%`
- **Legitimate Precision / Recall / F1**: `99.60%` / `99.91%` / `99.76%`
- **Confusion Matrix**: TP = `19995`, FN = `109`, FP = `23`, TN = `26947`
- **Brier Score / ECE**: `0.002423` / `0.000609`

### 2.2 Domain-Separated Test Holdout ($N=47,037$, Seed 14)
- **Domain Overlap**: Strict $D_{train} \cap D_{test} = \emptyset$ across 34,629 test domains.
- **Accuracy**: `99.794%`
- **Phishing Precision / Recall / F1**: `99.98%` / `99.54%` / `99.76%`
- **Confusion Matrix**: TP = `20021`, FN = `93`, FP = `4`, TN = `26919`
- **Phase 3 Split Model Benchmark Reference**: Accuracy = `99.73%`, FN = `106`, FP = `20`, ECE = `0.0010`.

---

## 3. Suite 2: External Validation Benchmark ($N=1,047$)

### 3.1 Threat Feed Detection (Phishing Catch Rate)
- **Overall Phishing Catch Rate**: **`99.82%`** (546 / 547 detected)
- **OpenPhish Community Feed**: **100.00%** (300 / 300 detected)
- **PhishTank Verified Feed**: **99.60%** (246 / 247 detected)
- **Sole Escaped Phishing URL**: `https://www.gonoconlatarjetaverde.com.mx/` ($P(\text{Phish}) = 0.1500$)
  - *Archetype*: Bare institutional root HTTPS domain matching the exact structural profile of Phase 3 false negatives.

### 3.2 False Positive Investigation on Legitimate URLs ($N=500$)
- **Raw False-Positive Rate**: **`94.00%`** (470 / 500 legitimate URLs incorrectly classified as phishing)
- **Trailing-Slash Stripped FPR**: **`86.80%`** (434 / 500 legitimate URLs still misclassified)
- **Trailing-Slash Mitigation Effect**: Recovers only 36 root domains (7.2% relative improvement), proving that trailing slash is merely a symptom of a deeper path/query syntactic covariate shift.

---

## 4. Suite 3: Controlled Perturbation Matrix ($N=726$)

Controlled perturbation testing across 22 representative seed domains isolates the exact structural tipping points:

| Structural Mutation Variant | Total URLs Tested | Phishing Classification Rate | Vulnerability Impact |
| :--- | :--- | :--- | :--- |
| `www_root` (`https://www.domain.tld`) | 22 | **`0.0%`** | Safe baseline (0 false positives) |
| `www_root_slash` (`https://www.domain.tld/`) | 22 | **`68.2%`** | +68.2% false alarms from a single trailing slash |
| `subdomain_root` (`https://sub.domain.tld`) | 22 | **`59.1%`** | +59.1% false alarms from subdomains |
| `subdomain_root_slash` (`https://sub.domain.tld/`) | 22 | **`86.4%`** | +86.4% false alarms |
| `path_single_short_slash` (`/dir/`) | 22 | **`95.5%`** | +95.5% false alarms on short path |
| `path_deep_multi_slash` (`/path/to/page/`) | 22 | **`100.0%`** | **100.0% false alarms** on deep links |
| `query_single_param` (`?q=query`) | 22 | **`100.0%`** | **100.0% false alarms** on query strings |
| `path_query_complex` (`/path?q=1&v=2`) | 22 | **`100.0%`** | **100.0% false alarms** on complex deep links |

---

## 5. Frozen Baseline Target Criteria for Future Iterations

Any proposed intervention (e.g. data augmentation or retraining in subsequent phases) will be measured against this frozen baseline scorecard:

1. **Phishing Recall Retention**: Must maintain external threat feed recall $\ge 99.00\%$ (current: 99.82%).
2. **In-Distribution Retention**: Must maintain in-distribution domain-separated accuracy $\ge 99.00\%$ (current: 99.79%).
3. **External Legitimate FPR Reduction**: Must reduce external legitimate FPR from **`94.00%`** to **$< 5.00\%$**.
4. **Deep-Link Stability**: Must achieve $< 5.0\%$ false-positive rate on deep paths and query strings in the perturbation matrix (current: 100.0%).
