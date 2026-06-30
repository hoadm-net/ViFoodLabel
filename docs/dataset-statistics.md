# Dataset Statistics

> **Current annotated set: 416 images** (raw collection complete at 560; annotation
> of the remaining images is in progress). All numbers below are produced by
> `scripts/dataset/compute_statistics.py` from `data/processed/dataset_meta.json`
> and will be refreshed when annotation reaches the full target. Splits come from
> `scripts/dataset/split_dataset.py` (`data/processed/splits.json`).

## Split Summary

Stratified **80/10/10** train/dev/test (seed 42), stratified jointly by
`product_category`, `background_color`, and `material`.

| Split | Images | Tokens | Relations |
|---|---|---|---|
| Train | 333 | 89,949 | 2,389 |
| Dev | 42 | 12,454 | 314 |
| Test | 41 | 11,276 | 295 |
| **Total** | **416** | **113,679** | **2,998** |

**Total entities (spans)**: 17,961.

**Token length per image**: min 41, median 265, mean 273.3, max 760 (p90 407,
std 106.3).

## Entity Distribution

| Entity Type | Spans | % of total |
|---|---|---|
| INGREDIENT | 4,717 | 26.3% |
| NUTRITION_VALUE | 3,269 | 18.2% |
| ADDITIVE | 3,221 | 17.9% |
| NUTRITION_NAME | 3,199 | 17.8% |
| WARNING | 1,813 | 10.1% |
| MANUFACTURER | 690 | 3.8% |
| PRODUCT_NAME | 350 | 1.9% |
| ORIGIN | 328 | 1.8% |
| NET_WEIGHT | 192 | 1.1% |
| EXPIRY_DATE | 102 | 0.6% |
| MFG_DATE | 80 | 0.4% |
| **Total** | **17,961** | 100% |

The distribution is strongly imbalanced: the four ingredient/nutrition types plus
`WARNING` account for ~90% of all spans, while the date and net-weight fields form
a long tail (each < 2%). `MFG_DATE` and `EXPIRY_DATE` are the rarest — many labels
were photographed at angles that cropped or obscured the date area. This imbalance
is a property of real food labels and is noted as a usage caveat rather than
corrected.

## Relation Distribution (`HAS_VALUE`)

| Metric | Value |
|---|---|
| Total relations | 2,998 |
| Per image (median / mean / max) | 7 / 7.2 / 59 |
| Images with no nutrition table (0 relations) | 74 |

Relations link `NUTRITION_NAME` → `NUTRITION_VALUE`, one per nutrition row. Images
with zero relations are mostly products without a nutrition facts table.

## Product-Category Distribution

| Category | Images | % |
|---|---|---|
| beverage | 132 | 31.7% |
| cake | 93 | 22.4% |
| dried_food | 68 | 16.3% |
| snack | 42 | 10.1% |
| seasoning | 40 | 9.6% |
| candy | 30 | 7.2% |
| other | 11 | 2.6% |

## Packaging Metadata

Auxiliary descriptive attributes (proposed by a vision model, corrected by an
annotator); used for stratified splitting, **not** extraction targets.

| `background_color` | Images | % | | `material` | Images | % |
|---|---|---|---|---|---|---|
| white | 125 | 30.0% | | paper | 173 | 41.6% |
| yellow | 99 | 23.8% | | foil | 109 | 26.2% |
| red | 66 | 15.9% | | plastic | 97 | 23.3% |
| green | 48 | 11.5% | | can_bottle | 29 | 7.0% |
| blue | 43 | 10.3% | | other | 8 | 1.9% |
| other | 35 | 8.4% | | | | |

## Inter-Annotator Agreement (IAA)

A subset is double-annotated by two independent annotators and three agreement
levels are reported: token-level Cohen's κ (BIO), entity-level F1 (exact span +
type), and relation-level F1. IAA is measured separately and will be filled in
here once available.

| Metric | Value |
|---|---|
| Cohen's κ (token-level BIO) | — |
| Entity-level F1 (pairwise) | — |
| Relation-level F1 (pairwise) | — |

## Figures

`scripts/dataset/compute_statistics.py` also writes histograms/bar charts to
`data/processed/figures/`: token-length distribution, entity distribution,
relation-count distribution, and product-category distribution.
