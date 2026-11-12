# Project Capabilities & Research Status

**Repository:** https://github.com/nikhi20-900/phishing-url-detection  
**Status snapshot:** 12 November 2026  
**Scope:** Implemented capabilities and completed research/audit work through Phase 5. Phase 6 is planned, not yet implemented.

## What the project can do today

### 1. Analyze URLs without visiting them

- Accept an HTTP or HTTPS URL through the CLI or Streamlit application.
- Parse and validate URL syntax, including IPv4/IPv6 literals, ports, user information in the authority, IDNA/Punycode hostnames, and malformed inputs.
- Extract the project's fixed set of 18 numerical URL features covering host/domain, path, character composition, and protocol structure.
- Perform static pre-flight analysis. The inference pipeline does not need to request the target URL, execute page scripts, or download its content.

**Important:** This is URL-string analysis, not a browser sandbox, website-content scanner, DNS reputation service, or certificate-chain inspection service.

### 2. Classify and explain a URL

- Load the existing Random Forest model artifact (100 trees) and align class probabilities using the estimator class metadata.
- Return the predicted class and model-estimated phishing/legitimate scores.
- Generate local SHAP TreeExplainer attributions for the phishing class.
- Present ranked feature contributions and contextual explanations.
- Keep feature extraction, class mapping, and prediction behavior consistent between CLI and Streamlit.

Label convention used throughout the project:

- **0 = Phishing**
- **1 = Legitimate**

The scores are model estimates, not guarantees that a destination is safe or malicious.

### 3. Provide two usable interfaces

**CLI**
- Scan a URL from the terminal.
- Print the verdict, class scores, and leading SHAP feature contributions.
- Reject malformed or unsupported URLs with structured error output and non-zero exit codes.

**Streamlit web application**
- Submit a URL through an interactive dashboard.
- View the verdict and phishing/legitimate score comparison.
- Inspect top SHAP decision drivers, an attribution plot, and the extracted 18-feature table.
- Try the application's sample URLs.

Run locally:

~~~bash
PYTHONPATH=. python3 src/cli.py "https://www.example.com"
streamlit run app/main.py
~~~

### 4. Reproduce and audit model evaluation

The repository contains research code and reports for:

- Dataset inspection and the original stratified holdout evaluation.
- Registered-domain-separated evaluation to reduce domain memorization leakage.
- Multiple grouped-split sensitivity runs.
- Probability calibration and reliability analysis.
- Phase 4 external benchmark ingestion, integrity checks, overlap analysis, inference, confusion matrices, and false-negative/SHAP diagnostics.
- Structural perturbation and feature-sensitivity investigations into deep-link false positives.

The original PhiUSIIL dataset contains 235,370 unique URLs after deduplication, with 18 model features. The reported random-split accuracy is 99.72%; this is a benchmark result, not a promise of deployment performance.

### 5. Stress-test the frozen model on external URLs

Phase 4 evaluates the frozen Random Forest on a 1,047-URL benchmark:

- 547 phishing URLs from external threat-feed snapshots.
- 500 curated legitimate URLs, intentionally including deep-link structures.
- At the locked threshold of 0.5: TP=546, FN=1, FP=470, TN=30.
- Phishing recall on this benchmark: 99.82%.
- False-positive rate on the 500-URL curated legitimate slice: 94%.

The 94% figure applies only to this targeted, curated stress-test slice. It is not an estimate of false-positive rate across all web traffic. The benchmark is not a population sample, and the audit documents URL/domain overlap and snapshot provenance limitations.

### 6. Collect and audit legitimate URL candidates (Phase 5)

The Phase 5 tooling supports an offline, provenance-first workflow:

- Ingest operator-supplied URL candidates into a review queue.
- Preserve the exact URL used for downstream feature extraction while maintaining a separate normalization key for deduplication.
- Record source, collection time, reviewer, verification method, rationale, syntax, registered domain, structural properties, and heuristic risk flags.
- Detect exact and normalized duplicates.
- Require human review; institutional domain reputation and heuristic checks are not proof of legitimacy.
- Audit domain diversity, structural coverage, provenance completeness, and overlap with the quarantined Phase 4 benchmark.

The committed pilot contains 605 candidates: 602 approved under the documented review protocol and 3 rejected, spanning 98 approved registered domains. Its structural profile includes deep paths and query strings that are absent from the legitimate PhiUSIIL training subset.

**Critical limitation:** “Approved” in this Phase 5 pilot means reviewed under the documented collection protocol. The pilot audit does not establish page-level authenticity for every URL. The 602 records must not be represented as independently page-verified training labels. Phase 6 must add hybrid page-level evidence collection and human validation before any candidate is admitted to model training.

### 7. Protect research integrity

The repository includes integrity checks and documentation to preserve:

- The original PhiUSIIL source dataset and its recorded hash.
- The frozen production Random Forest artifact.
- The Phase 4 external benchmark as a quarantined confirmatory benchmark.
- Candidate collection and audit artifacts separately from model training.
- Explicit provenance and reproducibility limitations rather than silently filling missing evidence.

The frozen model SHA-256 recorded for the current research baseline is:

~~~text
95f09ddc7bda88bad29cf8329a5ee83f74079da24d4e916427e95524d02f39bb
~~~

The Phase 4 benchmark SHA-256 recorded for the current confirmatory benchmark is:

~~~text
88525183abd402e385c6329c49b8d551db86b22bcf03d8429de37d477c8312c3
~~~

Do not retrain over, replace, or silently modify these frozen artifacts as part of Phase 6 development.

## Research findings so far

1. The initial high random-split metrics were not sufficient evidence of generalization; domain-separated evaluation was added to reduce cross-partition domain memorization.
2. The Phase 4 stress test exposed a severe failure on the selected curated legitimate deep-link slice while phishing detection remained high on the selected threat-feed URLs.
3. Structural analysis found that the legitimate training subset had almost no path/query coverage, while the curated external benign URLs contained substantial deep-link structure. This is evidence of distribution mismatch, not proof that one feature alone caused the errors.
4. Phase 5 built and audited a diverse candidate corpus, but page-level authenticity remains a required next validation step before training use.
5. The correct research question is whether independently validated benign deep-link data improves generalization on unseen domains without unacceptable loss of phishing detection.

## Phase 6 — completed: Generalization Research

Phase 6 completed milestones M0–M7 and compared the frozen production Random Forest (E0) with augmented candidate models E1 (Random Forest), E2 (XGBoost), and E3 (Logistic Regression).

- Hybrid page-level evidence collection and human review produced a validated benign corpus of 602 records across 98 registered domains.
- A new research corpus of 235,972 unique URLs was split by registered domain into train (170,852), validation (30,977), and internal test (34,143), with zero registered-domain overlap.
- On the internal domain-separated test set, E1 reduced the pilot deep-link subgroup FPR from 100.00% to 31.58% at threshold 0.50; phishing recall was 99.40% versus 99.34% for E0.
- On the Phase 4 external benchmark, E1 matched E0 phishing recall at 99.82%; curated-benign FPR changed from 94.00% to 92.20%. This remains a severe limitation and applies only to that targeted curated slice.
- Query-bearing URLs remained a major weakness: E1's legitimate FPR was 100.00% on the reported internal query-bearing subgroup.
- Paired statistical analysis reported a significant improvement in Macro F1 and legitimate FPR; the phishing-recall difference was not statistically significant.

**Governance decision:** The original production model remains frozen. E1/E2/E3 are research candidates stored separately and are not deployed replacements. See [Phase 6 final research report](../experiments/phase6/reports/phase6_final_report.md) for methodology, artifact hashes, subgroup metrics, and limitations.

The Phase 4 benchmark was evaluated during Phase 6. It must therefore not be described as an untouched confirmatory benchmark for later phases.

## Phase 7 — protocol initiated, experiments not yet run

Phase 7 targets legitimate deep-link and query-string generalization on unseen registered domains. The protocol requires a new domain-separated internal split and a newly sourced external confirmatory benchmark. The Phase 4 benchmark and Phase 6 internal test partition are historical comparators, not fresh Phase 7 holdouts.

The Phase 7 protocol and configuration are being developed on the `phase-7-generalization-expansion` branch. No Phase 7 dataset collection, model training, or evaluation is claimed yet.

## Repository guide

| Area | Purpose |
|---|---|
| src/features.py | URL parsing and 18-feature extraction |
| src/predict.py | Model inference, class probability mapping, SHAP explanations |
| src/cli.py | Command-line scanner |
| app/main.py | Streamlit application |
| src/evaluate_domain_split.py | Domain-separated evaluation |
| src/external_validation.py | Phase 4 external benchmark validation |
| experiments/generalization/collection/ | Phase 5 candidate collection workflow |
| experiments/generalization/audit/ | Pilot corpus integrity and diversity audits |
| experiments/generalization/reports/ | Phase 5 pilot audit artifacts |
| experiments/reports/ and reports/ | Evaluation, sensitivity, and research reports |
| tests/ | Parsing, feature, prediction, SHAP, audit, and regression tests |

## Verification snapshot

The Phase 5 final verification recorded 229 passing local pytest tests, 169 passing unittest tests, and successful GitHub Actions jobs for Python 3.11 and 3.12. These are historical verification results for the Phase 5 final commit; rerun the test suite on the current checkout before relying on them after later changes.

~~~bash
export MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=. pytest -v
PYTHONPATH=. python3 -m unittest discover tests -v
git diff --check
~~~

## Responsible-use limitations

- A URL classified as legitimate is not guaranteed safe; legitimate sites can be compromised or host malicious content.
- A URL classified as phishing is a model assessment and should be handled according to an organization's review and response process.
- The classifier does not fetch pages, inspect DOM/JavaScript, execute payloads, or validate live TLS certificates.
- The curated Phase 4 benchmark is a targeted stress test, not a representative sample of the public web.
- Collection provenance and review records do not replace page-level verification.
