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

## Related Work

For comparison with other visually-rich document understanding datasets, see [benchmark-tasks.md](benchmark-tasks.md).
