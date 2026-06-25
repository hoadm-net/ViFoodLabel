# Dataset Overview

## At a Glance

| Property | Value |
|---|---|
| Language | Vietnamese |
| Domain | Food product packaging labels |
| Annotation format | BIO NER + Bounding Box + Relations |
| Coordinate system | Normalized [0, 1000] (LayoutLM-compatible) |
| Annotation tool | [Label Studio](https://labelstud.io/) |
| Entity types | 11 |
| Relation types | 1 (`HAS_VALUE`) |
| Image sources | Self-photographed from Vietnamese supermarkets and convenience stores |

## Image Sources

All images are self-photographed directly from physical products in Vietnamese supermarkets and convenience stores. No e-commerce images, stock photos, or third-party packaging archives are used.

## Product Categories

The dataset spans a range of food product types to ensure schema coverage:

| Category | Examples |
|---|---|
| Confectionery | Candy, gummies, chocolate, cookies |
| Beverages | Juice, milk, energy drinks, tea |
| Instant foods | Noodles, porridge, soups |
| Snacks | Chips, crackers, dried fruits |
| Condiments | Sauces, seasonings, oils |
| Dairy | Yogurt, cheese, condensed milk |

## Data Format

Each annotated sample is exported from Label Studio as a JSON record containing:

```json
{
  "id": "sample_001",
  "image": "images/sample_001.jpg",
  "image_width": 1200,
  "image_height": 1600,
  "tokens": [
    {
      "text": "KẸO",
      "bbox": [x0, y0, x1, y1],
      "label": "B-PRODUCT_NAME"
    },
    ...
  ],
  "relations": [
    {
      "from_id": "token_045",
      "to_id": "token_046",
      "type": "HAS_VALUE"
    }
  ]
}
```

Bounding box coordinates are normalized to **[0, 1000]** on both axes for compatibility with LayoutLM and its variants.

## Per-Image Dataset Record

In addition to the token-level annotation above, each image has a derived
per-image record (`data/processed/dataset/<id>.json`) that bundles the
end-to-end KIE ground truth with descriptive metadata. This is the unit used for
Task 3 evaluation and for stratified train/dev/test splitting.

```json
{
  "id": "0001",
  "image": "0001.jpeg",
  "meta": {
    "product_category": "candy",
    "background_color": "green",
    "material": "plastic",
    "token_count": 287,
    "entity_counts": {"INGREDIENT": 4, "ADDITIVE": 7, "NUTRITION_NAME": 8, "...": 0},
    "num_relations": 7
  },
  "kie": { "product_name": "...", "ingredients": ["..."], "nutrition_value": {"...": "..."} }
}
```

- **`meta` (auto-derived)** — `token_count`, `entity_counts` (per entity type),
  `num_relations`, and image dimensions are computed from the token annotation.
- **`meta` (visual attributes)** — `product_category`, `background_color`, and
  `material` are drawn from a small closed vocabulary, proposed by a vision model
  and corrected by an annotator. They describe the dataset and enable stratified
  splitting; they are **not** Task 3 evaluation targets.

| Field | Closed vocabulary |
|---|---|
| `product_category` | `cake`, `candy`, `snack`, `beverage`, `seasoning`, `dried_food`, `other` |
| `background_color` | `red`, `yellow`, `green`, `blue`, `white`, `other` |
| `material` | `paper`, `plastic`, `foil`, `can_bottle`, `other` |

- **`kie`** — the structured Task 3 record. Its construction from the token
  annotation (reading-order sort, BIO span merging, relation-based nutrition
  pairing, bilingual handling) and the evaluation protocol are documented in
  [task3-kie-record.md](task3-kie-record.md).

## Related Work

For comparison with other visually-rich document understanding datasets, see [benchmark-tasks.md](benchmark-tasks.md).
