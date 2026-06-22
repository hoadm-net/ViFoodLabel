# Dataset Statistics

> **Feasibility study** (v1 — 134 annotated images). Raw image collection is complete at **550 images** (target reached, see `plan/phases.md`); annotation of the remaining ~416 images is in progress. Statistics below will be updated as annotation progresses toward the full dataset.

## Split Summary

| Split | Images | Tokens | Relations |
|---|---|---|---|
| Train | 107 | 27,715 | 627 |
| Validation | 27 | 7,532 | 190 |
| Test | — | — | — |
| **Total (so far)** | **134** | **35,247** | **817** |

**Split ratio**: 80% train / 20% validation (feasibility phase). Final dataset will use an 80/10/10 split.

**Token length**: min 48, max 671, mean ~265 tokens/image.

## Entity Distribution (train + validation)

| Entity Type | Spans | % of total |
|---|---|---|
| INGREDIENT | 1,829 | 25.6% |
| ADDITIVE | 1,469 | 20.6% |
| WARNING | 1,054 | 14.8% |
| NUTRITION_VALUE | 991 | 13.9% |
| NUTRITION_NAME | 988 | 13.8% |
| MANUFACTURER | 364 | 5.1% |
| ORIGIN | 136 | 1.9% |
| PRODUCT_NAME | 189 | 2.6% |
| NET_WEIGHT | 68 | 1.0% |
| EXPIRY_DATE | 33 | 0.5% |
| MFG_DATE | 22 | 0.3% |
| **Total** | **7,143** | 100% |

⚠️ `MFG_DATE` and `EXPIRY_DATE` are intentionally under-represented in this batch — many labels were photographed at angles that cropped the date area. Labels will be targeted for these entities in future collection rounds.

## Inter-Annotator Agreement (IAA)

| Metric | Value |
|---|---|
| Cohen's κ (token-level) | — |
| Entity-level F1 (pairwise) | — |
| Relation F1 (pairwise) | — |

IAA measurement pending double-annotation pass (planned for Phase 3 batch 2).

## Product Category Distribution

| Category | Images | % |
|---|---|---|
| Confectionery | — | — |
| Beverages | — | — |
| Instant foods | — | — |
| Snacks | — | — |
| Condiments | — | — |
| Dairy | — | — |
| Other | — | — |
