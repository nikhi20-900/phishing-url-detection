# Phase 7 Milestone 0 (M0): Research Freeze & Protocol Verification Report

- **Date (UTC):** 2026-11-12T16:43:54.600647+00:00
- **Contract Version:** `phase7-m0-contract-v1`
- **Git Commit:** `9c946e54c9363d8ef40c7176419c30a0bfacd59c`
- **Git Branch:** `phase-7-m0-integrity-verification`
- **Audit Status:** **APPROVED** (`exit_criteria_met: True`)

---

## 1. Executive Summary

Milestone 0 (M0) establishes the pre-experimental scientific freeze and governance contract for **Phase 7: Generalization Expansion**.
Before any new URLs are collected, processed, split, or used for model training, all non-negotiable research invariants were audited and cryptographically verified against their immutable references.

### Key Audit Outcomes
- **Production Model Frozen:** `models/random_forest_model.joblib` SHA-256 verified matches Phase 3 reference.
- **PhiUSIIL Dataset Frozen:** `data/PhiUSIIL_Phishing_URL_Dataset.csv` SHA-256 verified matches original benchmark.
- **Phase 4 Benchmark Quarantine:** `data/external/external_validation_benchmark.csv` SHA-256 verified; quarantined as historical comparison only, NOT an untouched confirmatory benchmark for Phase 7.
- **Phase 6 Internal Test Quarantine:** Formally quarantined from Phase 7 model selection and evaluation.
- **Phase 6 Candidate Artifacts Audited:** E1 (RF), E2 (LR), and E3 (XGBoost) artifacts hashed and preserved read-only.
- **Deterministic Invariant Test:** Frozen control model (E0) verified producing exact baseline predictions.
- **Leakage & Domain Governance:** Registered-domain isolation, 70/15/15 target split, and 18-feature schema locked.

---

## 2. Cryptographic Hash Audit of Frozen Artifacts

| Artifact Name | Path | Role | Expected SHA-256 | Actual SHA-256 | Status |
|---|---|---|---|---|---|
| `frozen_production_model` | `models/random_forest_model.joblib` | `frozen_control_e0` | `95f09ddc7bda88ba...` | `95f09ddc7bda88ba...` | **PASS** |
| `phiusiil_source` | `data/PhiUSIIL_Phishing_URL_Dataset.csv` | `immutable_source_dataset` | `a236549cd369cd80...` | `a236549cd369cd80...` | **PASS** |
| `phase4_benchmark` | `data/external/external_validation_benchmark.csv` | `historical_comparison_only_not_untouched_confirmatory` | `88525183abd402e3...` | `88525183abd402e3...` | **PASS** |
| `phase5_approved_pilot` | `experiments/generalization/datasets/approved_pilot_corpus.csv` | `phase5_historical_pilot` | `cca9ef3f8b265c4c...` | `cca9ef3f8b265c4c...` | **PASS** |
| `phase5_review_queue` | `experiments/generalization/datasets/pilot_review_queue.csv` | `phase5_audit_queue` | `e4ea2a206f477eb9...` | `e4ea2a206f477eb9...` | **PASS** |
| `phase6_validated_benign` | `experiments/phase6/datasets/validated_benign_corpus.csv` | `phase6_validated_benign_corpus` | `cca9ef3f8b265c4c...` | `cca9ef3f8b265c4c...` | **PASS** |
| `phase6_rf_augmented` | `models/phase6_candidates/rf_augmented.joblib` | `phase6_candidate_e1_historical_comparator` | `072e1b74fe69b406...` | `072e1b74fe69b406...` | **PASS** |
| `phase6_lr_augmented` | `models/phase6_candidates/logistic_regression_augmented.joblib` | `phase6_candidate_e2_linear_baseline` | `63113186f6730392...` | `63113186f6730392...` | **PASS** |
| `phase6_xgb_augmented` | `models/phase6_candidates/xgboost_augmented.joblib` | `phase6_candidate_e3_gradient_boosted` | `2dbabdc62e2cd7a5...` | `2dbabdc62e2cd7a5...` | **PASS** |

---

## 3. Confirmatory Benchmark Quarantine & Governance Rules

> [!IMPORTANT]
> **Phase 4 External Benchmark Status:** The Phase 4 benchmark was evaluated during Phase 6. Under scientific generalization rules, it is **no longer an untouched benchmark for Phase 7**.
> It is designated `historical_comparison_only_not_untouched_confirmatory`. Phase 7 must assemble a **new, independently sourced confirmatory benchmark** before evaluating primary candidate E4.

> [!IMPORTANT]
> **Phase 6 Internal Test Status:** The Phase 6 internal test set was used for comparative candidate selection in Phase 6. Phase 7 requires a newly generated, domain-separated 70/15/15 split for all internal evaluation.

| Governance Invariant | Policy Value | Verification Status |
|---|---|---|
| `production_model_immutable` | `True` | **PASS** |
| `phase4_benchmark_immutable` | `True` | **PASS** |
| `phase4_benchmark_not_untouched` | `True` | **PASS** |
| `phase4_benchmark_historical_role` | `True` | **PASS** |
| `requires_new_independent_benchmark` | `True` | **PASS** |
| `phase6_internal_test_quarantined` | `True` | **PASS** |
| `feature_schema_locked` | `True` | **PASS** |
| `zero_registered_domain_overlap_required` | `True` | **PASS** |
| `candidate_models_isolated` | `True` | **PASS** |

---

## 4. Historical Artifact Verification (Phases 3–6)

All historical evaluation reports, numerical JSON results, and audit addenda from prior phases were verified present and non-empty:

| Historical Artifact Path | Size (bytes) | Status |
|---|---|---|
| `reports/phase_3_evaluation_and_leakage_audit.md` | 22726 | **PASS** |
| `reports/domain_separated_evaluation_results.json` | 1204 | **PASS** |
| `reports/phase_4_external_validation_and_fn_analysis.md` | 18963 | **PASS** |
| `reports/external_validation_results.json` | 16584 | **PASS** |
| `reports/phase_4_audit_addendum.md` | 10752 | **PASS** |
| `experiments/reports/issue_4_validation_report.md` | 14884 | **PASS** |
| `experiments/generalization/reports/pilot_corpus_audit.md` | 12459 | **PASS** |
| `experiments/generalization/reports/audit_manifest.json` | 5695 | **PASS** |
| `experiments/phase6/reports/phase6_final_report.md` | 21838 | **PASS** |
| `experiments/phase6/reports/experiment_manifest.json` | 10185 | **PASS** |
| `experiments/phase6/reports/baseline_results.json` | 5119 | **PASS** |
| `experiments/phase6/reports/candidate_comparison.json` | 41727 | **PASS** |
| `experiments/phase6/reports/confirmatory_external_evaluation.json` | 19780 | **PASS** |
| `experiments/phase6/reports/statistical_analysis.json` | 5526 | **PASS** |

---

## 5. Frozen Control (E0) Prediction Invariant Verification

The production model was loaded and evaluated on deterministic sanity URLs:

| Target URL | Expected Class | Predicted Class | Phishing Probability | Verification Status |
|---|---|---|---|---|
| `https://www.google.com` | Legitimate (1) | Legitimate | 0.0018 | **PASS** |
| `http://paypal-verification-account-security-alert.com/login` | Phishing (0) | Phishing | 1.0000 | **PASS** |

---

## 6. Pre-Registered Hypotheses & Success Criteria

### Hypotheses
- **H1 (Deep-Link Generalization):** Validated legitimate path-bearing URLs reduce legitimate FPR on unseen-domain deep-link test data.
- **H2 (Query Generalization):** Validated legitimate query-bearing URLs reduce legitimate FPR on unseen-domain query-bearing test data.
- **H3 (Phishing Preservation):** Augmentation does not cause unacceptable reduction in phishing recall on unseen domains.
- **H4 (Structural Coverage):** Improvements are measurable across predeclared URL strata, not just one narrow URL pattern.

### Provisional Research Success Criteria (Pre-Registered)
1. **Deep-Link FPR Reduction:** E4 reduces legitimate FPR by at least 50% relative to E0 on unseen-domain deep links.
2. **Query-Bearing FPR Bound:** E4 achieves legitimate FPR <= 30% on unseen-domain query-bearing URLs with reported sample denominator and confidence interval.
3. **Phishing Recall Non-Inferiority:** Lower bound of paired 95% CI for (E4 - E0) phishing recall difference is >= -1.0 percentage point on internal test.
4. **No Hidden Subgroup Regressions:** Subgroups (homepage, path, query, path+query, trailing slash, percent-encoded) monitored for regressions.
5. **Domain Leakage & Provenance Zero-Tolerance:** Strict disjoint registered-domain groups across all splits.

---

## 7. Execution Environment & Dependencies

- **Python Version:** `3.14.7`
- **Platform:** `macOS-27.0.1-arm64-arm-64bit-Mach-O`
- **scikit-learn:** `1.9.0`
- **xgboost:** `3.4.1`
- **joblib:** `1.5.3`
- **pyyaml:** `6.0.3`
- **scipy:** `1.18.0`

---

## 8. M0 Milestone Exit Sign-Off

- [x] Head commit & Phase 6 merge confirmed.
- [x] Production model SHA-256 audited and frozen (`95f09ddc...2f39bb`).
- [x] PhiUSIIL dataset SHA-256 audited and frozen (`a236549c...6d1c6`).
- [x] Phase 4 benchmark SHA-256 audited (`88525183...312c3`) and quarantined as historical only.
- [x] Phase 6 internal test quarantined from Phase 7 internal evaluation.
- [x] Phase 6 candidate models hashed and preserved read-only.
- [x] Phase 7 protocol (`PHASE7_RESEARCH_PROTOCOL.md`) and configurations frozen.
- [x] Machine-readable manifest (`experiment_manifest.json`) generated.
- [x] Production model confirmed immutable throughout Phase 7.

**Milestone 0 Status:** **COMPLETE & APPROVED**
