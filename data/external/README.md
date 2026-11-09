# External Validation Datasets — Provenance, Schema & Usage Notes

This directory contains the Phase 4 external validation benchmark and its raw
snapshot inputs. All files are **static offline snapshots**; the codebase
contains **no network code** that fetches or refreshes them.

## Files

| File | Status | Role | Rows |
| :--- | :--- | :--- | ---: |
| `external_validation_benchmark.csv` | **Tracked** | Consolidated benchmark used for evaluation | 1,047 |
| `legitimate_curated.csv` | **Tracked** | Curated authoritative legitimate URL corpus | 500 |
| `phishing_feed_openphish.txt` | *Gitignored* | Raw upstream OpenPhish feed snapshot (redistribution-restricted) | 300 |
| `phishing_feed_phishtank.csv` | *Gitignored* | Raw upstream PhishTank feed snapshot (redistribution-restricted) | 249 |

## Label convention

`label` is shared with the PhiUSIIL training dataset and is consistent across
all pipeline modules:

- `0` = **Phishing**
- `1` = **Legitimate**

## Snapshot provenance

| Source | Collected (snapshot timestamp) | Captured as | Verification basis |
| :--- | :--- | :--- | :--- |
| OpenPhish Community Feed | `2026-10-31T02:55:00Z` (recorded in `collection_timestamp`) | plain-text feed snapshot | Active feed entries |
| PhishTank Community Feed | verification timestamps per row (`verification_time`), all `verified=yes`, `online=yes` | CSV feed snapshot | Community vote + analyst verification |
| Curated authoritative corpus | `2026-10-31T03:00:00Z` | manually curated list | Well-known institutional origins (`.edu`, `.gov`, standards bodies, open-source ecosystems) |

The collection scripts that captured these snapshots are **not committed to
this repository** — the ingestion code (`load_openphish_feed`,
`load_phishtank_feed`, `load_curated_legitimate`, `build_external_benchmark`
in `src/external_validation.py`) parses and consolidates the snapshots only.
This is a documented provenance limitation.

## Known snapshot irregularities (documented for audit honesty)

1. **Cross-feed duplication:** 2 URLs appear in both the OpenPhish and
   PhishTank snapshots. The consolidated benchmark attributes them to
   OpenPhish (first-source), so the benchmark contains 247 PhishTank-sourced
   rows although the raw PhishTank CSV holds 249.
2. **`source_id` format:** PhishTank rows in the consolidated benchmark use
   bare numeric identifiers (`9552311`) rather than the documented
   `PHISHTANK-<id>` format that `load_phishtank_feed` produces. The
   `verification_method` value (`phishtank_community_verified`) also differs
   from the loader's output (`community_vote_and_analyst_verified`).
   Rebuilding via `build_external_benchmark()` yields a schema-consistent file
   whose `url`/`label`/`source` sequences match the shipped benchmark exactly;
   only the provenance columns differ.

## Safe handling

- The phishing snapshots contain **live hostile URLs** (PhishTank ROCs,
  active attacker infrastructure). Do not visit, resolve, or request them
  outside a controlled sandbox. All analysis is purely lexical: the
  evaluation pipeline performs **no HTTP requests**.
- The children of this directory are untracked research snapshots. Do not
  redistribute the feeds without consulting their providers' current terms
  (OpenPhish and PhishTank each publish their own data-usage policies; no
  license file for these snapshots is included here, and none is implied).
