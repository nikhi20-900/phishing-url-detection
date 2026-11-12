# Generalization Corpus Collection: Controlled Legitimate URL Collector

This directory contains the provenance-first legitimate URL collection module for Phase 5 (`experiments/generalization/collection/collect_legitimate_corpus.py`).

The collector is designed to address the deep-link covariate shift discovered during Phase 4 without compromising the scientific integrity of the evaluation pipeline.

---

## 1. Architectural Principles

### Principle 1: Source Reputation is a Collection Criterion, Not Proof of Legitimacy
An institutional domain (`.gov`, `.edu`, `.org`, or official documentation domains such as `w3.org` or `ietf.org`) provides a high-quality collection seed, but **domain reputation alone does not prove a URL is legitimate**.
- Institutional domains frequently host unvetted user-generated content (e.g. `~student/`, wiki edits, comment sections, or user uploads).
- Institutional domains may host vulnerable endpoints (e.g. open redirects like `?redirect=`, `?url=`, or unmaintained legacy paths).
- The collector requires recording:
  - `source_name`: The institution, organization, or curated catalog.
  - `source_url`: The exact canonical origin or reference document.
  - `verification_method`: The specific method used to verify legitimacy (e.g. manual audit, DNSSEC/TLS validation, static docs manifest).
  - `legitimacy_reason`: Explicit rationale explaining why the resource is verified as authentic.
  - Heuristic risk screening: Flags potential user-generated paths (`~`, `/wiki/`, `/blog/`) and open-redirect query parameters for mandatory human auditor review.

### Principle 2: Exact URL Preservation for Model Evaluation
Model feature extraction depends on character counts, special characters, and structural tokens. Modifying URL strings during collection alters the feature representation and compromises evaluation fidelity.
- **Model input rule**: Downstream feature extraction and model inference MUST consume `url_exact` (the verbatim URL string), NEVER a normalized representation.
- **Deduplication key**: A separate string `normalized_dedupe_key` is calculated solely to detect duplicates across representations.

#### Normalization Policy (`candidate-normalization-v1`)
1. Convert scheme and hostname to lowercase.
2. Remove trailing dot from hostname.
3. Remove standard default ports (port `80` for `http`, port `443` for `https`). Non-default ports are retained.
4. Enclose raw IPv6 hostnames in brackets (`[...]`).
5. Retain user authentication components if present (`username:password@`).
6. Treat empty paths as `/`.
7. Strip URL fragments (`#...`), as fragments are processed client-side and not sent to HTTP servers.
8. Retain path case, query parameter case, parameter ordering, and percent-encoding verbatim.

### Principle 3: Separation of Collection from Verification
- Ingestion creates a candidate review queue (`review_queue.csv`).
- The collector **does not assign model labels** (`label=1` or `0`, `"Legitimate"` or `"Phishing"`).
- Candidate records default to `review_status="pending"` until an auditor has inspected them.
- Only vetted records with `review_status="approved"`, an explicit `reviewer`, documented `verification_method`, documented `legitimacy_reason`, valid syntax, and zero exact/normalized duplicates can be promoted via `promote_approved()`.
- Promoted datasets remain candidate feature pools; model labeling occurs downstream during controlled dataset synthesis.

### Principle 4: Multi-Dimensional Syntax Auditing (No Mutually Exclusive Quotas)
Real-world URLs possess overlapping syntactic attributes:
- A URL can simultaneously have a multi-level path (`/docs/v2/`), contain query parameters (`?format=json`), and end with a trailing slash (`/`).
- Treating these attributes as mutually exclusive buckets distorts coverage statistics.
- The collector audits syntax across **independent dimensions**:
  - `is_multilevel_path`: `path_depth >= 2`
  - `has_trailing_slash`: path ends with `/`
  - `has_query`: non-empty query string
  - `has_fragment`: non-empty fragment
  - `has_special_chars`: non-alphanumeric characters in path/query
  - `is_https`: transport security status
- Co-occurrences are explicitly tracked and reported in `collection_summary.json`.

---

## 2. Schemas

### Input CSV Schema (`--input`)
The collector ingests an operator-supplied CSV with the following columns:

| Column | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `url` | string | Yes | Exact URL string to inspect | `https://www.w3.org/TR/REC-xml/` |
| `source_name` | string | Yes | Institutional or dataset origin | `W3C Standards` |
| `source_url` | string | Yes | Provenance origin page | `https://www.w3.org/standards/` |
| `collection_timestamp` | ISO-8601 | Yes | Timestamp of collection | `2026-11-09T12:00:00Z` |
| `verification_method` | string | Yes | Verification protocol used | `manual_standards_registry_audit` |
| `review_status` | string | Yes | Status (`pending`, `approved`, `rejected`) | `approved` |
| `reviewer` | string | Yes | Reviewer identity / process ID | `auditor-01` |
| `notes` | string | Yes | Context notes or observations | `Canonical XML 1.0 recommendation` |
| `legitimacy_reason` | string | Optional | Detailed legitimacy rationale | `W3C formal technical recommendation specification` |

### Review Queue Output Schema (`review_queue.csv`)
In addition to the input metadata, each record contains:
- `candidate_id`: SHA-256 hash of the exact URL string.
- `url_exact`: Verbatim URL string (authoritative model input).
- `normalized_dedupe_key`: Canonical key under `candidate-normalization-v1`.
- `normalization_version`: Policy version string.
- `syntax_valid`: Boolean indicating valid URL parsing under `src/features.py`.
- `validation_error`: Reason for rejection if invalid.
- `hostname`, `registered_domain`, `tld`: Public-suffix-aware domain components.
- `path`, `path_depth`, `is_multilevel_path`: Path structure metrics.
- `has_trailing_slash`, `has_query`, `query_param_count`, `has_fragment`: Syntax indicators.
- `has_special_chars`: Special character indicators.
- `risk_flags`, `has_risk_flags`: Heuristic indicators for open redirects or user-generated paths.
- `duplicate_exact`: Boolean indicating duplicate of a previously seen exact URL string.
- `duplicate_normalized`: Boolean indicating duplicate under canonical normalization.

---

## 3. CLI Usage

### Ingest Candidates into Review Queue
```bash
python experiments/generalization/collection/collect_legitimate_corpus.py \
  --input path/to/candidates.csv \
  --output experiments/generalization/collection/review_queue.csv \
  --summary experiments/generalization/collection/collection_summary.json
```

### Ingest and Export Approved Subset
```bash
python experiments/generalization/collection/collect_legitimate_corpus.py \
  --input path/to/candidates.csv \
  --output experiments/generalization/collection/review_queue.csv \
  --summary experiments/generalization/collection/collection_summary.json \
  --approved-output experiments/generalization/collection/approved_candidates.csv
```

---

## 4. Scientific Guardrails

1. **Quarantine Benchmark**: Candidate URLs must NEVER overlap with `data/external/external_validation_benchmark.csv` (1,047 URLs). Downstream auditing enforces zero intersection:
   $$D_{\text{candidate}} \cap D_{\text{external\_benchmark}} = \emptyset$$
2. **Frozen Production Model**: The baseline model `models/random_forest_model.joblib` remains strictly frozen.
3. **No Automated Labeling**: Every URL must pass human review gating before downstream usage.
