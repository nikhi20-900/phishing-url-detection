# Phase 7 Research Protocol: Generalization Expansion

**Project:** Phishing URL Detection  
**Phase:** 7 — Generalization Expansion  
**Status:** M0 protocol draft / research freeze pending artifact audit  
**Branch:** `phase-7-generalization-expansion`  
**Date:** 12 November 2026

## 1. Research question

Can additional independently validated legitimate URLs containing paths, nested paths, and query strings reduce false-positive classifications on previously unseen registered domains, while preserving phishing detection?

Phase 6 established that benign deep-link augmentation materially improved the internal deep-link subgroup for candidate E1, but did not adequately resolve query-bearing URLs. In the Phase 6 internal test report, the query-bearing subgroup contained 976 URLs and E1's legitimate FPR remained 100%. Phase 7 therefore prioritizes query-bearing legitimate URLs while continuing to test deep-link generalization.

## 2. Research hypotheses

- **H1 — Deep-link generalization:** Adding validated legitimate path-bearing URLs reduces legitimate FPR on unseen-domain deep-link test data.
- **H2 — Query generalization:** Adding validated legitimate query-bearing URLs reduces legitimate FPR on unseen-domain query-bearing test data.
- **H3 — Phishing preservation:** The augmentation does not cause an unacceptable reduction in phishing recall on unseen domains.
- **H4 — Structural coverage:** Improvements are measurable across predeclared URL strata, not just one narrow URL pattern.

These are hypotheses, not assumed outcomes.

## 3. Non-negotiable governance

1. Do not modify or overwrite `models/random_forest_model.joblib`.
2. Do not modify the original PhiUSIIL source dataset.
3. Do not modify Phase 4 or Phase 6 reports, manifests, datasets, or model artifacts.
4. Keep all Phase 7 datasets, manifests, candidate models, predictions, and reports under `experiments/phase7/` and `models/phase7_candidates/`.
5. Preserve the Phase 6 E1/E2/E3 artifacts as historical comparators; do not overwrite them.
6. Do not use any confirmatory test partition for training, feature selection, threshold tuning, hyperparameter selection, or model selection.
7. Do not silently assign labels. Only records with documented page-level evidence and human approval may enter the legitimate training pool.
8. Keep the production feature schema fixed at the existing 18 features for the primary experiment. Any feature engineering change must be a separate, explicitly labelled experiment.
9. Record source provenance, collection time, review decisions, exact URL, normalization key, registered domain, evidence references, and artifact hashes.
10. Never claim that a model is production-ready solely from research metrics.

### Confirmatory benchmark rule

The Phase 4 benchmark was evaluated during Phase 6. It is therefore **not an untouched benchmark for Phase 7** and must not be described as a fresh, sealed confirmatory test. It may be cited only as a historical Phase 4/6 comparison under its documented limitations.

Phase 7 must assemble a **new, independently sourced confirmatory benchmark**. Freeze its contents, labels, provenance manifest, and SHA-256 before any candidate is evaluated against it. If a new independent benchmark cannot be assembled, report that confirmatory external validation is pending; do not substitute Phase 4.

The Phase 6 internal test partition has also already been used for model comparison. Phase 7 must create a new registered-domain-separated internal split for its confirmatory internal analysis.

## 4. Scope and dataset strategy

### 4.1 Collection priorities

Prioritize independently validated legitimate examples from:
- Single-level and multi-level paths.
- Query strings with common parameter patterns (search, pagination, filters, product/resource IDs, language/locale, and tracking parameters).
- URLs containing both nested paths and query strings.
- Trailing-slash and percent-encoded path variants.
- Short and long URLs, subdomains, and relevant special-character strata.

Also retain phishing examples containing paths and query strings. Do not create a dataset where query syntax itself becomes a proxy for the legitimate class.

### 4.2 Authenticity validation

Use the Phase 6 hybrid approach as the starting protocol:
- Automated evidence collection is observational only; it must not execute scripts, download binaries, submit forms, or authenticate.
- Enforce strict timeouts, response-size caps, redirect limits, and no automatic cross-domain redirect following.
- Record HTTP status, redirect chain, final URL, TLS hostname evidence where available, page title, content metadata, canonical URL, collection timestamp, and evidence references.
- HTTP 200, HTTPS, a familiar domain, or a reputable TLD is not sufficient proof of legitimacy.
- Human reviewer records the legitimacy rationale and evidence.
- Require independent second review for risk flags, user-generated content, ambiguous ownership, cross-domain redirects, or reviewer disagreement.
- Use explicit states: pending, evidence_collected, verified_page, rejected, ambiguous, inaccessible.
- Only `verified_page` records with reviewer approval are eligible for the legitimate training pool.

### 4.3 Deduplication and grouping

- Preserve original URL strings; maintain normalization keys separately.
- Deduplicate exact URLs and audit normalized collisions.
- Group all partitions by registered domain, not URL row.
- Report unique URL counts and unique registered-domain counts for every stratum and partition.
- No registered domain may appear in more than one partition.

## 5. Experimental design

### E0 — Frozen production control
Use the existing production Random Forest as a non-retrained control. Verify its known SHA-256 before and after experiments.

### E1 — Phase 6 augmented Random Forest comparator
Use the existing Phase 6 E1 artifact read-only, if its hash and loadability are verified. It is a historical comparator, not a new Phase 7 model.

### E4 — Phase 7 targeted augmented Random Forest
Train a new Random Forest candidate using the original permitted training data plus Phase 7 verified benign examples. Keep the production 18-feature schema and pre-register all hyperparameters.

### Optional E5 — Controlled XGBoost comparator
Only run if E4 completes and the same data, splits, and evaluation protocol can be maintained. Keep it separate from the primary E4 hypothesis test.

Do not add a new model family or change the 18-feature schema in the primary experiment. This phase tests data coverage first, not algorithm novelty.

## 6. Splits and leakage controls

- Construct a new Phase 7 dataset snapshot and manifest.
- Deduplicate before splitting.
- Group by registered domain.
- Use a deterministic, recorded seed and a stratified 70/15/15 train/validation/internal-test split where feasible.
- Record class counts, domain counts, structural strata, and all deviations from target ratios.
- Assert pairwise zero registered-domain overlap across train, validation, and internal test.
- Fit any preprocessing using training data only.
- Freeze the internal test partition before model selection begins.

A domain with multiple URLs must remain entirely within one partition. If class/stratum balance cannot be achieved under domain grouping, document the limitation rather than breaking domain isolation.

## 7. Evaluation plan

Primary metrics:
- Legitimate false-positive rate and legitimate recall.
- Phishing recall and phishing precision.
- Macro F1.
- PR-AUC and ROC-AUC.
- Brier score and expected calibration error.

Predeclared subgroups:
- Homepage.
- Single-level path.
- Multi-level path.
- Query-bearing.
- Path + query.
- Trailing slash.
- Percent-encoded path.
- Subdomain-bearing.
- Short and long URL.
- High special-character count.

Report denominators and confidence intervals for every subgroup. Do not report a percentage without its sample count.

### Threshold policy

- `0.50` is the primary fixed comparison threshold.
- Any alternate threshold must be selected using validation data only and reported separately.
- Do not select a threshold using internal-test or external-confirmatory outcomes.
- Report the phishing-recall / legitimate-FPR trade-off rather than presenting one metric alone.

### Statistical analysis

- Use paired comparisons on identical test URLs.
- Use McNemar's test for paired binary outcomes.
- Use paired bootstrap confidence intervals.
- Account for clustering by registered domain; URLs from one domain are not independent observations.
- Report effect sizes and confidence intervals, not p-values alone.
- A non-significant difference is not proof of equivalence or non-inferiority.

## 8. Provisional research success criteria

Freeze or amend these criteria **before inspecting Phase 7 test outcomes**:

1. E4 reduces legitimate FPR by at least 50% relative to E0 on the predeclared unseen-domain deep-link subgroup.
2. E4 achieves legitimate FPR of at most 30% on the predeclared unseen-domain query-bearing subgroup, with subgroup denominator and confidence interval reported.
3. The lower bound of the paired 95% confidence interval for the E4-minus-E0 phishing-recall difference is no worse than -1.0 percentage point on the internal test.
4. No material regression is hidden in homepage, path, query, or path+query subgroups.
5. All split, provenance, artifact-integrity, and reproducibility checks pass.

These are research gates, not deployment certification. If sample sizes are too small for meaningful inference, mark the criterion inconclusive rather than pass.

## 9. New external confirmatory evaluation

- Source a new benchmark independently of the Phase 4 benchmark and Phase 7 training/validation data.
- Include both phishing and legitimate URLs, including deep links and query-bearing examples.
- Record source, capture date, label rationale, registered domain, exact URL, and deduplication decisions.
- Audit URL-level and registered-domain overlap with all training and development data.
- Freeze benchmark CSV, labels, provenance manifest, and SHA-256 before evaluation.
- Evaluate only after E4, its configuration, feature extraction, and threshold are frozen.
- Run the confirmatory evaluation once for the primary claim. Any repeated exploration invalidates the untouched-confirmatory designation.

## 10. Milestones

| Milestone | Deliverable | Exit condition |
|---|---|---|
| M0 — Protocol freeze | Protocol, config, invariant manifest | Hashes verified; hypotheses, splits, metrics, and decision criteria frozen |
| M1 — Collection and validation | Candidate queue, evidence records, reviewed corpus | Only human-approved verified pages eligible for training |
| M2 — Dataset and split | Dataset snapshot, domain split manifest | Deduplication and zero domain leakage verified |
| M3 — Baseline reproduction | E0/E1 baseline report | Reproducible predictions on the new internal split |
| M4 — Controlled training | E4 candidate (optional E5) | Candidate artifacts isolated and hashed |
| M5 — Internal evaluation | Locked-test metrics and subgroup report | All predeclared metrics, CIs, and paired tests reported |
| M6 — External confirmation | New benchmark and one-shot report | Benchmark was frozen before candidate evaluation |
| M7 — Research decision | Final report and governance decision | Explicit pass/fail/inconclusive decision; production model unchanged |

## 11. Proposed repository layout

```text
experiments/phase7/
  configs/
    experiment.yaml
    split_config.yaml
  collection/
    candidate_queue.csv
    evidence/
  datasets/
    manifests/
    train.csv
    validation.csv
    internal_test.csv
    new_external_confirmatory.csv
  evaluation/
    baseline/
    candidates/
    confirmatory/
  reports/
    phase7_final_report.md
models/
  phase7_candidates/
```

Do not commit sensitive collection metadata, credentials, private browsing data, or unsafe raw payloads. Follow the repository's existing data-governance and size policies.

## 12. M0 checklist

- [ ] Confirm current `main` head and Phase 6 merge state.
- [ ] Verify frozen production model SHA-256.
- [ ] Verify original PhiUSIIL dataset SHA-256.
- [ ] Verify Phase 4 benchmark SHA-256 and mark it historical/reused for Phase 6.
- [ ] Verify Phase 6 candidate artifacts and manifests; record hashes without modifying them.
- [ ] Verify Phase 6 report and reconcile its reported dataset counts and subgroup definitions.
- [ ] Confirm Phase 6 internal test and Phase 4 benchmark will not be reused as fresh Phase 7 confirmatory data.
- [ ] Freeze this protocol and experiment configuration before collection or training.
- [ ] Record environment, dependency versions, random seeds, Git commit, and working-tree state.

**Production governance decision:** The production model remains frozen throughout Phase 7. Any future replacement requires a separate, explicit release decision after research gates and operational validation are satisfied.
