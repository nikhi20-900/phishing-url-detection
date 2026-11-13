# Phase 7 Milestone 2: Audit Reconciliation and Class Distribution Addendum

**Document Timestamp:** `2026-11-13T16:00:00Z`  
**Milestone:** Phase 7 Milestone 2 (M2) Reconciliation Addendum  
**Scope:** Formal verification and reconciliation of partition class distributions across committed CSVs, manifests, and audit reports  
**Status:** RECONCILED & LOCKED  

---

## 1. Context and Objective

This addendum provides the formal, independent reconciliation of class distributions across all Phase 7 Milestone 2 (M2) deliverables prior to model training in Milestone 3 (M3).

In accordance with scientific data governance:
1. All merged M2 partition files (`train.csv`, `val.csv`, and `internal_test.csv`) remain **strictly immutable**.
2. All manifest records (`split_manifest.json` and `m2_split_manifest.json`) are confirmed to match the exact row-level contents of the committed CSVs.
3. Class counts across all three partitions are verified to sum precisely to the full research dataset.

---

## 2. Reconciled Partition Class Distributions

Direct programmatic inspection of the committed CSV files confirms the exact class breakdowns:

| Partition | Total URLs | Registered Domains | Phishing URLs (`label = 0`) | Legitimate URLs (`label = 1`) | Benign Expansion URLs | Partition SHA-256 |
|---|---|---|---|---|---|---|
| **Train** | **170,615** (72.45%) | **121,217** (70.00%) | **76,092** (44.60%) | **94,523** (55.40%) | **77** across 16 domains | `9297458d03841fbe9c32bae7c4ad323e1b3f0b221c0c755f912350808d0c6892` |
| **Validation** | **30,782** (13.07%) | **25,974** (15.00%) | **10,548** (34.27%) | **20,234** (65.73%) | **29** across 6 domains | `132e1db3ef90056b55039bb90386d538f1fd3e644e90cc0546250a2c2d5de578` |
| **Internal Test** | **34,092** (14.48%) | **25,978** (15.00%) | **13,880** (40.71%) | **20,212** (59.29%) | **13** across 2 domains | `ce004e9d8e65ea8c09175d904d4a5fee0442acc4b127af6a98f58ae0aed0e2cf` |
| **Combined Research Dataset** | **235,489** (100.0%) | **173,169** (100.0%) | **100,520** (42.69%) | **134,969** (57.31%) | **119** across 24 domains | **Complete Independent Reconciliation** |

### Mathematical Invariants Confirmed

1. **URL Conservation:**
   $$\sum \text{URLs} = 170{,}615 + 30{,}782 + 34{,}092 = 235{,}489$$
2. **Phishing Class Conservation:**
   $$\sum \text{Phishing} = 76{,}092 + 10{,}548 + 13{,}880 = 100{,}520$$
3. **Legitimate Class Conservation:**
   $$\sum \text{Legitimate} = 94{,}523 + 20{,}234 + 20{,}212 = 134{,}969$$
4. **Benign Expansion Conservation:**
   $$\sum \text{Benign Expansion} = 77 + 29 + 13 = 119$$
5. **Pairwise Registered-Domain Disjointness:**
   $$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset, \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset, \quad \mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

---

## 3. Reconciliation with Manifests

The committed manifest files:
- `experiments/phase7/datasets/manifests/split_manifest.json`
- `experiments/phase7/reports/m2_split_manifest.json`

both explicitly record these exact counts under `partitions.train`, `partitions.val`, and `partitions.internal_test`. This addendum supersedes any preliminary draft notes in pull request commentary, establishing this table as the authoritative reference for Phase 7 experimental milestones.

With class distributions, domain boundaries, and partition integrity fully reconciled and locked, Milestone 2 exit criteria are officially reaffirmed and Milestone 3 may proceed.
