# Deep-Link & URL Perturbation Forensic Report

**Investigation:** Issue #4 — Forensic Root-Cause Analysis of the 94% False-Positive Rate
**Date:** November 8, 2026
**Status:** Diagnostic Complete (Model Frozen, Unmodified Baseline)
**Artifacts Generated:**
- Raw Perturbation Data: `experiments/reports/url_perturbation_results.csv` ($N=726$ URL tests)
- Aggregated Metrics: `experiments/reports/url_perturbation_results.json`
- Feature Sweeps & Thresholds: `experiments/reports/feature_sensitivity_results.json`
- Diagnostic Visualizations:
  - `experiments/reports/probability_shifts_by_perturbation.png`
  - `experiments/reports/special_char_sensitivity_curve.png`
  - `experiments/reports/shap_attribution_comparison.png`

---

## 1. Executive Summary & Core Discovery

During Phase 4 external validation, evaluating the frozen Random Forest classifier against an independent corpus of 500 legitimate URLs revealed a **94.00% false-positive rate** (470 false alarms out of 500 benign links), despite achieving a **99.82% catch rate** on external zero-day phishing feeds.

This experiment performed systematic, controlled perturbation testing across **22 seed domains** (spanning top universities, governmental portals, standards bodies, open-source repositories, and reference sites) through **33 distinct structural variations** ($N=726$ URL evaluations).

### Core Forensic Finding: The PhiUSIIL Benign Template Invariant
The extreme false-positive rate is **not** an unpredictable stochastic failure or a feature extraction bug. It is the deterministic mathematical consequence of an extreme **covariate shift** in the underlying PhiUSIIL training dataset ($N=235,370$):

> [!CRITICAL]
> **The 100% PhiUSIIL Benign Training Invariant:**
> Across all **134,850 legitimate URLs** in the PhiUSIIL dataset, every single entry adhered strictly to the following rigid syntax:
> 1. **Scheme:** `https://` on **134,850 / 134,850 (100.000%)** of entries.
> 2. **Hostname Prefix:** `www.` present on **134,850 / 134,850 (100.000%)** of entries.
> 3. **Total Slashes:** Exactly two slashes (`//` in `https://`) on **134,850 / 134,850 (100.000%)** of entries.
> 4. **Trailing Slash:** Exactly **0 / 134,850 (0.000%)** of legitimate URLs had a trailing slash.
> 5. **Subpaths / Directories:** Exactly **0 / 134,850 (0.000%)** of legitimate URLs contained any path segment.
> 6. **Query Strings:** Exactly **0 / 134,850 (0.000%)** of legitimate URLs contained query parameters (`?`, `=`, `&`).
> 7. **Subdomain Count:** Exactly **0 / 134,850 (0.000%)** had `NoOfSubDomain == 0`.
>
> In contrast, in the phishing training partition ($N=100,520$), 60.1% contained paths, 36.5% had trailing slashes, 13.9% had `NoOfSubDomain == 0`, and 58.6% lacked `www.`.

Because the training dataset exhibited 100% separation on path presence, the Random Forest decision trees learned that **any deviation from `https://www.<domain>` (such as adding a trailing slash, stripping `www.`, or navigating to `/research`) is an almost certain indicator of phishing**.

---

## 2. Controlled URL Perturbation Matrix & Empirical Results

We tested 33 structural variations across 22 representative domains (including Stanford, Harvard, MIT, Berkeley, CMU, Oxford, Cambridge, GOV.UK, CISA, NASA, Python.org, Apache.org, Kernel.org, GitHub, Wikipedia).

### 2.1 Variant Progression Table (Sorted by Mean $P(\text{Phish})$)

| Variant Key | Structural Pattern | Mean $P(\text{Phish})$ | Phishing Flag Rate | Mean Special Chars | Mean URL Length | Diagnostic Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `www_root` | `https://www.{domain}` | **0.0024** | **0.0%** | 1.00 | 22.95 | **Canonical Benign Template**: 100% verified legitimate. Matches PhiUSIIL training distribution perfectly. |
| `subdomain_root` | `https://{sub}.{domain}` | **0.5239** | **59.1%** | 1.00 | 23.27 | Ambiguous boundary: replacing `www.` with `cs.` or `dept.` pushes 59% of domains across the threshold. |
| `www_root_slash` | `https://www.{domain}/` | **0.6914** | **68.2%** | **2.00** | 23.95 | **Trailing Slash Cliff**: Adding `/` increases special chars to 2, causing 68.2% of domains to flip to Phishing! |
| `subdomain_root_slash` | `https://{sub}.{domain}/` | **0.8314** | **86.4%** | 2.00 | 24.27 | Subdomain + trailing slash produces 86.4% false alarms. |
| `path_single_short` | `https://www.{domain}/info` | **0.8816** | **86.4%** | 2.00 | 27.95 | Single short path segment flips 86.4% of domains to Phishing. |
| `path_deep_five_levels` | `https://www.{domain}/a/b/c/d/e` | **0.8773** | **100.0%** | 6.00 | 32.95 | Multi-directory traversal triggers 100% phishing classification. |
| `path_single_medium` | `https://www.{domain}/research` | **0.9573** | **100.0%** | 2.00 | 31.95 | Typical institutional subpath triggers 100% phishing classification ($P \ge 0.95$). |
| `bare_root` | `https://{domain}` | **0.9333** | **100.0%** | 1.00 | 18.95 | **Stripping `www.`**: In PhiUSIIL, 0% of legitimate URLs had `NoOfSubDomain == 0`. Flips 100% of domains to Phishing! |
| `bare_root_slash` | `https://{domain}/` | **0.9559** | **100.0%** | 2.00 | 19.95 | Bare domain + slash triggers 100% phishing classification. |
| `path_two_levels` | `https://www.{domain}/research/projects` | **0.9714** | **100.0%** | 3.00 | 40.95 | Real-world departmental paths trigger near-certain malicious score ($P = 0.97$). |
| `query_single_param` | `https://www.{domain}/search?q=cybersecurity` | **0.9674** | **100.0%** | 2.00 | 45.95 | Search queries trigger near-certain malicious score ($P = 0.97$). |
| `query_complex` | `https://www.{domain}/view?doc=annual&format=pdf` | **0.9664** | **100.0%** | 2.00 | 68.95 | Key-value parameters trigger near-certain malicious score. |
| `http_www_root` | `http://www.{domain}` | **1.0000** | **100.0%** | 1.00 | 21.95 | Plain HTTP transport triggers 100% phishing classification across all domains. |

---

## 3. Exact 1D Feature Thresholds & Split Discoveries

Using synthetic sweeps across `https://www.stanford.edu` holding all other features at baseline values, `experiments/feature_sensitivity.py` isolated the exact decision boundaries within the 100 decision trees:

| Feature Name | Baseline Value (`www.stanford.edu`) | Critical Threshold Crossing $P \ge 0.5$ | Underlying Tree Logic |
| :--- | :---: | :---: | :--- |
| `NoOfOtherSpecialCharsInURL` | **1** | **2** | **The Trailing Slash Cliff**: A single dot (`stanford.edu`) is safe. The moment a slash or second delimiter appears, count reaches 2 and trees branch to Phishing. |
| `NoOfSubDomain` | **1** | **0** | **The www Requirement**: In PhiUSIIL, 100% of legitimate URLs had `www.` (`NoOfSubDomain >= 1`). If `www.` is omitted (`https://stanford.edu`), `NoOfSubDomain = 0`, immediately triggering Phishing branches. |
| `SpacialCharRatioInURL` | **0.042** | **0.100** | Special character density above 10% indicates phishing in the training set. |
| `LetterRatioInURL` | **0.458** | **0.700** | Long paths with high alphabetic ratios (`/research-and-development`) inflate the letter ratio toward phishing thresholds. |
| `IsHTTPS` | **1** | **0** | Unencrypted HTTP triggers an immediate jump to $P(\text{Phish}) = 1.000$. |

---

## 4. SHAP Attribution Deconstruction

Forensic local Shapley explanations (`experiments/deep_link_report.py`) reveal the exact feature mechanisms:

### Case 1: Trailing Slash Impact (`stanford.edu` vs `stanford.edu/`)
- `https://www.stanford.edu`:
  - $P(\text{Phish}) = 0.0021$ (**99.79% Legitimate**)
  - `NoOfOtherSpecialCharsInURL = 1` pushes toward Legitimate ($\text{SHAP} = -0.142$)
  - `SpacialCharRatioInURL = 0.042` pushes toward Legitimate ($\text{SHAP} = -0.118$)
- `https://www.stanford.edu/`:
  - $P(\text{Phish}) = 1.0000$ (**100.0% Phishing**)
  - Adding `/` increments `NoOfOtherSpecialCharsInURL` to **2** ($\text{SHAP} = +0.330$)
  - `SpacialCharRatioInURL` doubles to **0.080** ($\text{SHAP} = +0.127$)
  - **Net Flip:** $+0.9979$ probability shift from a single character!

### Case 2: Deep Link Path Impact (`mit.edu/` vs `mit.edu/research`)
- `https://www.mit.edu/`:
  - $P(\text{Phish}) = 0.0400$ (**96.0% Legitimate**)
  - Short domain length ($11$) and low letter count ($6$) keep probability low.
- `https://www.mit.edu/research`:
  - $P(\text{Phish}) = 0.9600$ (**96.0% Phishing False Alarm**)
  - `DomainLength = 11` combined with path creates a high ratio mismatch ($\text{SHAP} = +0.358$).
  - Slashes in path increase `SpacialCharRatioInURL` to $0.071$ ($\text{SHAP} = +0.114$).

---

## 5. Why "Quick Fixes" Are Counterproductive

The user's cautionary guidance was validated empirically:

1. **Stripping Trailing Slashes Solves Only a Minor Fraction:**
   - In our 500-URL legitimate test set, stripping trailing slashes from root URLs increased true negatives from $30$ to $66$.
   - However, **434 out of 500 URLs (86.8%) remained false alarms** because the remaining 416 URLs contained real subpaths (`/research`, `/academics`) and query strings (`?lang=en`).
2. **Ignoring Path Features Cripples Phishing Detection:**
   - Attackers frequently hide credential lures in deep directories (e.g., `/secupdf/10625.html` or `/account/login/verify.php`). Blindly ignoring paths or slashes would create massive false negatives on genuine attacks.
3. **Domain Whitelisting Violates Zero-Trust Cybersecurity:**
   - Whitelisting domains like `mit.edu`, `stanford.edu`, or `github.com` is dangerous: adversaries frequently exploit open redirects, compromised subdomains, and public user hosting (e.g. `github.io`, Cloudflare Pages) to deliver malware.

---

## 6. Strategic Recommendations

To resolve this limitation scientifically without degrading phishing detection:

1. **Training Dataset Rebalancing (Primary Fix):**
   - Augment the training set with authentic legitimate URLs containing diverse path structures, query strings, and fragments (e.g. from Common Crawl or verified benign web crawls).
   - The Random Forest must observe that legitimate web architecture routinely includes directory hierarchies (`/docs/guide`), parameter delimiters (`?page=2`), and trailing slashes.
2. **Feature Engineering Refinement:**
   - Decouple **domain structural features** from **path lexical features**:
     - `NoOfSubDomain` should be computed from FQDN without depending on `www.` presence.
     - `PathDepth` and `PathSpecialChars` should be separated from root domain delimiters.
     - A trailing slash on a bare root (`/`) should be treated as a path-root canonicalizer rather than an anomalous special character delimiter.
3. **Multi-Modal Signals:**
   - Complement static lexical URL analysis with external non-intrusive telemetry: WHOIS domain age (<30 days), TLS certificate issuer transparency, and DNS fast-flux history.
