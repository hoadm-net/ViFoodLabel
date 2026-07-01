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

### Primary vs. auxiliary entities

The dataset is **panel-centric by design**: images are framed on the food-label
information panel, so five **primary** entity types account for **90.3%** of all
spans:

| Group | Entity types | Spans | % |
|---|---|---|---|
| **Primary** | INGREDIENT, ADDITIVE, NUTRITION_NAME, NUTRITION_VALUE, WARNING | 16,219 | 90.3% |
| **Auxiliary** | PRODUCT_NAME, MANUFACTURER, ORIGIN, NET_WEIGHT, MFG_DATE, EXPIRY_DATE | 1,742 | 9.7% |

The six auxiliary types (typically front-of-pack or in separate spots) are
captured **when they appear in frame**, not targeted by the collection — hence
their long tail. This is a deliberate scope decision, not an annotation gap (see
*Field coverage* below).

## Field Coverage

Fraction of images in which each KIE field is present. Primary (information-panel)
fields are present in most images; auxiliary fields are opportunistic.

| Field | Group | Images present | % |
|---|---|---|---|
| ingredients | primary | 386 | 92.8% |
| warnings | primary | 370 | 88.9% |
| additives | primary | 355 | 85.3% |
| nutrition_value | primary | 342 | 82.2% |
| origin | auxiliary | 255 | 61.3% |
| manufacturer | auxiliary | 236 | 56.7% |
| product_name | auxiliary | 219 | 52.6% |
| net_weight | auxiliary | 136 | 32.7% |
| expiry_date | auxiliary | 73 | 17.5% |
| mfg_date | auxiliary | 65 | 15.6% |

Many images are single-panel photographs of the back/side of the package, where
the dense regulated information lives; the front-of-pack name and the net-weight /
date markings are therefore often out of frame. This is consistent with the
dataset's focus on ingredient, additive, nutrition, and warning extraction.

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

`scripts/dataset/compute_statistics.py` writes the following to
`data/processed/figures/`:

| File | Chart |
|---|---|
| `entity_distribution.png` | Entity spans per type, coloured by primary/auxiliary group |
| `token_length_hist.png` | Distribution of tokens per image |
| `relation_per_image_hist.png` | Distribution of `HAS_VALUE` relations per image |
| `field_coverage.png` | % of images in which each KIE field is present (primary vs. auxiliary) |
| `metadata_distributions.png` | Product-category, background-colour, and material distributions |
| `layout_heatmap.png` | Spatial density of token-box centroids per entity type on the normalized `[0,1000]` canvas |
